# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end runner: pick N puzzles, run mini-swe-agent on each, print table.

For each picked task, builds a hardened DockerEnvironment + a
SamplerChatCompletionModel, wraps the puzzle with the system/instance
templates, awaits ``agent.run``, scores the submission, saves the
trajectory, and prints a per-task summary.

Usage:
  cd <repo_root>
  PYTHONPATH=src uv run python -m example_recipe.word_puzzle.agent_solver.minisweagent.run_agent_solver \\
      --num-tasks 10 --seed 42 \\
      --project <your-gcp-project> --thinking-level medium
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

from minisweagent.environments.docker import DockerEnvironment
from minisweagent.exceptions import LimitsExceeded, TimeExceeded

from backends.gemini_api_endpoint import GeminiApiSampler
from example_recipe.word_puzzle.agent_solver.minisweagent.async_default_agent import (
    AsyncDefaultAgent,
)
from example_recipe.word_puzzle.agent_solver.minisweagent.docker_run_args import (
    SANDBOX_IMAGE,
    SANDBOX_RUN_ARGS,
)
from example_recipe.word_puzzle.agent_solver.minisweagent.extract_answer import (
    TABLE_HEADER,
    TaskResult,
    build_task_result,
)
from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
    SamplerChatCompletionModel,
)
from example_recipe.word_puzzle.agent_solver.minisweagent.word_puzzle_task import (
    INSTANCE_TEMPLATE,
    SYSTEM_TEMPLATE,
)

logger = logging.getLogger(__name__)

# Agent-mode withhold OOD eval set. Same puzzles/references as the
# direct-answer baseline JSONL but with the "Put your final answer in
# the following format: $\\text{...}$" trailer stripped: the agent
# submits via a structured tool call, so the trailer is unnecessary
# and actively harmful (the model may LaTeX-wrap the submit argument
# and fail exact-match).
DEFAULT_JSONL = (
    Path(__file__).resolve().parents[2]
    / "dataset"
    / "combined_eval_word_transformation_withhold_agent_v1.jsonl"
)

DEFAULT_TRACE_DIR = Path("/tmp/agent_traces")


# ----------------------------------------------------------------------
# Task selection
# ----------------------------------------------------------------------


def _validate_retry_metadata(retry_meta: dict | None) -> None:
    """Fail loud on partial ``retry_metadata`` blocks.

    A well-formed block emits all three fields together (``rule``,
    ``extra_word_pool``, ``reverse_task``). Silently defaulting a
    missing field is unsafe: a missing ``reverse_task`` would generate
    injected pairs in the wrong direction, a missing ``rule`` would
    make input==output nonsense, and a missing ``extra_word_pool``
    would silently downgrade to single-attempt.

    Empty dict and None both mean "no retry metadata" (accepted).
    """
    if not retry_meta:
        return
    required = {"rule", "extra_word_pool", "reverse_task"}
    missing = required - set(retry_meta.keys())
    if missing:
        raise ValueError(
            f"row has partial retry_metadata block: missing "
            f"{sorted(missing)!r}. A well-formed block must contain "
            f"all of {sorted(required)!r}. Regenerate the JSONL with "
            f"``puzzle_dataset_gen --include-retry-metadata`` or "
            f"remove the retry_metadata key from the row."
        )


def load_and_select_tasks(
    jsonl_path: Path,
    *,
    per_tier: dict[str, int],
    seed: int,
) -> list[dict]:
    """Load rows, filter by tier, pick per-tier counts deterministically.

    Returns a flat list in tier order (A rows, then B, then C).
    """
    rng = random.Random(seed)
    all_rows = [
        json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()
    ]

    picked: list[dict] = []
    for tier in ("A", "B", "C"):
        n_wanted = per_tier.get(tier, 0)
        if n_wanted <= 0:
            continue
        tier_rows = [r for r in all_rows if r.get("tags", {}).get("tier") == tier]
        if len(tier_rows) < n_wanted:
            raise ValueError(
                f"Requested {n_wanted} tier-{tier} tasks but only "
                f"{len(tier_rows)} available in {jsonl_path.name}"
            )
        picked.extend(rng.sample(tier_rows, n_wanted))
    return picked


# ----------------------------------------------------------------------
# Live prober
# ----------------------------------------------------------------------


class _EvalProber:
    """Shared tracker for live progress reporting during eval.

    Each task calls ``update()`` after every agent step; a background
    asyncio task calls ``print_summary()`` on a timer.
    """

    def __init__(self, total_tasks: int) -> None:
        self.total = total_tasks
        # task_index -> {steps, status, tier, t_start}
        self._tasks: dict[int, dict[str, Any]] = {}
        self._completed = 0
        self._t0 = time.perf_counter()

    def register(self, idx: int, tier: str) -> None:
        self._tasks[idx] = {
            "steps": 0,
            "status": "running",
            "tier": tier,
            "t_start": time.perf_counter(),
        }

    def step(self, idx: int) -> None:
        if idx in self._tasks:
            self._tasks[idx]["steps"] += 1

    def done(self, idx: int, status: str) -> None:
        if idx in self._tasks:
            self._tasks[idx]["status"] = status
            self._completed += 1

    def print_summary(self) -> None:
        elapsed = time.perf_counter() - self._t0
        active = [v for v in self._tasks.values() if v["status"] == "running"]
        if not active and self._completed == 0:
            return
        # Step-count buckets for active tasks
        step_counts = [v["steps"] for v in active]
        buckets: dict[str, int] = {"0": 0, "1-5": 0, "6-15": 0, "16-30": 0, "31+": 0}
        for s in step_counts:
            if s == 0:
                buckets["0"] += 1
            elif s <= 5:
                buckets["1-5"] += 1
            elif s <= 15:
                buckets["6-15"] += 1
            elif s <= 30:
                buckets["16-30"] += 1
            else:
                buckets["31+"] += 1
        bucket_str = " ".join(f"{k}:{v}" for k, v in buckets.items() if v)
        # Completed status breakdown
        done_statuses: dict[str, int] = {}
        for v in self._tasks.values():
            if v["status"] != "running":
                done_statuses[v["status"]] = done_statuses.get(v["status"], 0) + 1
        done_str = " ".join(f"{k}:{v}" for k, v in done_statuses.items())
        # NOTE: keep as print (NOT logger) \u2014 the \\r + end="" pattern
        # OVERWRITES the previous prober line in-place on a TTY. Routing
        # through logger would break the single-line-refresh UX and
        # flood the log with one entry every 10s per task.
        print(
            f"\r[prober {elapsed:5.0f}s] "
            f"active={len(active)} done={self._completed}/{self.total} "
            f"steps=[{bucket_str}] "
            f"exits=[{done_str or '-'}]",
            end="",
            flush=True,
        )


async def _run_prober(prober: _EvalProber, interval: float = 10.0) -> None:
    """Background task: print prober summary every ``interval`` seconds."""
    try:
        while True:
            await asyncio.sleep(interval)
            prober.print_summary()
    except asyncio.CancelledError:
        pass


# ----------------------------------------------------------------------
# One-task runner
# ----------------------------------------------------------------------


async def run_one_task_async(
    *,
    task_index: int,
    row: dict,
    sampler: Any,
    model_cfg: dict[str, Any],
    trace_dir: Path,
    step_limit: int,
    wall_time_limit_seconds: int,
    cost_limit: float,
    submit_retries: int,
    semaphore: asyncio.Semaphore,
    prober: _EvalProber | None = None,
    sample_suffix: str = "",
    rollout_id: int | None = None,
    capture_ctx: Any | None = None,
    atif_writer: Any | None = None,
) -> TaskResult:
    """Run the agent on one puzzle. Returns a compact TaskResult.

    ``sampler`` is shared across all concurrent tasks (one
    httpx.AsyncClient, shared ADC token refresh). ``semaphore`` caps
    parallel rollouts (docker load + upstream QPS burst).
    """
    async with semaphore:
        tier = row["tags"]["tier"]
        puzzle_prompt = row["contents"][0]["parts"][0]["text"]
        expected_answer = row["references"]["reference"]
        trace_filename = f"task_{task_index:03d}_tier{tier}{sample_suffix}.json"
        # When an ATIF writer is active, write trajectory JSONs into its
        # job_dir so they're under the same blob_root as raw capture
        # blobs and get clean GCS paths. Otherwise use trace_dir.
        if atif_writer is not None:
            trajectory_path = atif_writer.job_dir / trace_filename
        else:
            trajectory_path = trace_dir / trace_filename
        # Use rollout_id (globally unique) for the prober key so
        # samples_per_task > 1 doesn't overwrite the same entry.
        prober_key = rollout_id if rollout_id is not None else task_index
        if prober is not None:
            prober.register(prober_key, tier)

        logger.info(
            "[task %d] tier=%s expected=%r prompt_chars=%d",
            task_index,
            tier,
            expected_answer,
            len(puzzle_prompt),
        )

        env = DockerEnvironment(
            image=SANDBOX_IMAGE,
            cwd="/workspace",
            # Per-action shell command timeout (seconds).
            timeout=30,
            # Max container lifetime (belt-and-suspenders against
            # container leaks even if cleanup() fails).
            container_timeout=f"{wall_time_limit_seconds + 60}s",
            # forward_env stays default [] -- no host env leaks in.
            run_args=SANDBOX_RUN_ARGS,
        )
        model = SamplerChatCompletionModel(
            sampler=sampler, capture_ctx=capture_ctx, **model_cfg
        )
        if prober is not None:
            _orig_query = model.query

            async def _tracked_query(messages: list, **kw: Any) -> dict:
                result = await _orig_query(messages, **kw)
                prober.step(prober_key)
                return result

            model.query = _tracked_query  # type: ignore[assignment]
        # ``_validate_retry_metadata`` rejects PARTIAL blocks that
        # would otherwise silently take unsafe defaults. Absent block
        # or empty dict are both fine.
        retry_meta_raw = row.get("retry_metadata")
        _validate_retry_metadata(retry_meta_raw)
        retry_meta = retry_meta_raw or {}
        agent = AsyncDefaultAgent(
            model=model,
            env=env,
            system_template=SYSTEM_TEMPLATE,
            instance_template=INSTANCE_TEMPLATE,
            step_limit=step_limit,
            wall_time_limit_seconds=wall_time_limit_seconds,
            cost_limit=cost_limit,
            output_path=trajectory_path,
            # Function-calling makes format errors rare (only "no tool
            # call at all" can fail). Two in a row => stuck; fail fast
            # rather than burn more tokens on retries.
            max_consecutive_format_errors=2,
            submit_retries=submit_retries,
            expected_answer=expected_answer,
            rule=retry_meta.get("rule") or [],
            extra_word_pool=retry_meta.get("extra_word_pool") or [],
            reverse_task=retry_meta.get("reverse_task", False),
            # Deterministic per-task shuffle at a fixed seed.
            task_seed=task_index,
        )

        try:
            result = await agent.run(task=puzzle_prompt)
        except (LimitsExceeded, TimeExceeded):
            # Agent should catch these itself; belt-and-suspenders.
            result = {"exit_status": "LimitsExceededEscaped", "submission": ""}
        except Exception as e:  # noqa: BLE001 - we WANT to keep going
            logging.exception("task %d crashed: %s", task_index, e)
            result = {"exit_status": f"Crash:{type(e).__name__}", "submission": ""}
        finally:
            # env.cleanup is sync docker rm; off-load to avoid
            # serializing the next task.
            await asyncio.to_thread(env.cleanup)

        task_result = build_task_result(
            tier=tier,
            expected_answer=expected_answer,
            run_return_value=result,
            trajectory_path=trajectory_path,
        )
        # Enqueue trajectory JSON for background GCS upload.
        # Raw LLM blobs are already uploaded continuously via
        # CaptureContext.on_written (wired by AtifDiskWriter).
        if atif_writer is not None and trajectory_path.exists():
            atif_writer.enqueue_for_upload(trajectory_path)
        if prober is not None:
            prober.done(prober_key, task_result.exit_status)
        logger.info(
            "[task %d] %s: submitted=%r exit=%s n_llm=%d n_bash=%d t_llm=%.1fs",
            task_index,
            "CORRECT" if task_result.is_correct else "WRONG",
            task_result.submitted_answer,
            task_result.exit_status,
            task_result.n_model_calls,
            task_result.n_bash_calls,
            task_result.total_model_seconds,
        )
        return task_result


# ----------------------------------------------------------------------
# CLI + main
# ----------------------------------------------------------------------


def _positive_int(raw: str) -> int:
    """argparse ``type=`` validator: reject non-positive integers.

    ``asyncio.gather(*[])`` returns ``[]``, which would leave the
    summary table dividing zero results into zero tasks — a nonsense
    input we'd rather reject at the CLI boundary than silently print
    "0/0 correct" for.
    """
    try:
        n = int(raw)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"expected an integer, got {raw!r}") from e
    if n < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {n}")
    return n


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0] if __doc__ else "",
    )
    p.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    p.add_argument("--num-tasks", type=_positive_int, default=10)
    p.add_argument(
        "--tier-a",
        type=int,
        default=None,
        help="Override count of tier-A tasks (default: 3 out of --num-tasks=10)",
    )
    p.add_argument("--tier-b", type=int, default=None)
    p.add_argument("--tier-c", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    # Model config forwarded to SamplerChatCompletionModel.
    p.add_argument("--project", required=True, help="Your GCP project.")
    p.add_argument("--env", default="prod")
    # One flag, two defaults: the publisher-model path wants "global",
    # the --tuning-job-id path wants a real region. Resolved in main().
    p.add_argument(
        "--location",
        default=None,
        help=(
            "Vertex location. Defaults to 'global' for the publisher "
            "model, or 'us-central1' when --tuning-job-id is set."
        ),
    )
    p.add_argument("--model", default="gemini-3.5-flash")
    p.add_argument(
        "--endpoint-id",
        default=None,
        help=(
            "Optional deployed-endpoint resource id (numeric or full "
            "'projects/.../locations/.../endpoints/<id>' path). When "
            "set, requests target the endpoint's :generateContent RPC "
            "instead of the publisher model. Use to evaluate a tuned "
            "model checkpoint (set --env to match the region/tier the "
            "endpoint is deployed in)."
        ),
    )
    p.add_argument(
        "--thinking-level",
        default="medium",
        choices=["minimal", "low", "medium", "high", ""],
        help="'' = off (server default)",
    )
    p.add_argument("--max-generation-tokens", type=int, default=65536)
    # Agent config. step_limit=50 covers the hard algorithm-fitting
    # tasks that need many hypothesis-test rounds; easy ones still
    # submit fast. wall_time_limit_seconds=1800 (30 min) gives high-
    # thinking-level runs enough headroom to finish without timing out.
    p.add_argument("--step-limit", type=int, default=50)
    p.add_argument("--wall-time-limit-seconds", type=int, default=1800)
    p.add_argument("--cost-limit", type=float, default=0.50)
    # Values > 1 require the JSONL to have been generated with
    # ``--include-retry-metadata`` (rule + extra_word_pool +
    # reverse_task); rows without it degrade to single-attempt.
    p.add_argument(
        "--submit-retries",
        type=int,
        default=1,
        help=(
            "Total submit attempts per task (1 = off; N > 1 injects 2 "
            "fresh worked examples after each wrong submit until the "
            "budget is exhausted). Requires JSONL generated with "
            "gen_withhold_eval.py --include-retry-metadata."
        ),
    )
    # MSRL tuning-job sampling: evaluate a tuned checkpoint by sampling
    # through the tuning-scope endpoint of a RUNNING tuning job.
    p.add_argument(
        "--tuning-job-id",
        default=None,
        help=(
            "Vertex MSRL tuning job id (numeric). When set, samples via "
            "the tuning-scope :generateContentTuningScope endpoint "
            "instead of the publisher model. The job must be in "
            "JOB_STATE_RUNNING. Mutually exclusive with --endpoint-id."
        ),
    )
    # Each rollout gets its own docker container + AsyncDefaultAgent
    # but shares ONE sampler (httpx client + ADC token).
    p.add_argument("--concurrency", type=int, default=1)
    # --- Vertex Experiment + raw capture ---
    p.add_argument(
        "--vertex-experiment",
        default=None,
        help=(
            "Vertex Experiment name. When set, logs run params + metrics "
            "via aiplatform.log_params/log_metrics after the sweep."
        ),
    )
    p.add_argument(
        "--vertex-run-name",
        default=None,
        help="Vertex Experiment run name (auto-derived if not set).",
    )
    p.add_argument(
        "--atif-output",
        default=None,
        help=(
            "ATIF output destination: 'gs://bucket' for local writes + "
            "background GCS upload, 'local' for local-only, 'off' to "
            "disable. Uses the same AtifDiskWriter as the training "
            "pipeline. Raw LLM blobs land at "
            "<trace-dir>/atif/agent-solver/<run>/task-NNN-0/artifacts/. "
            "GCS paths mirror the local tree under the bucket."
        ),
    )
    p.add_argument(
        "--samples-per-task",
        type=int,
        default=1,
        help=(
            "Number of independent rollouts per task. Each sample gets "
            "its own Docker container and trajectory file. Traces are "
            "named task_NNN_tierX_sY.json (Y=0..N-1). Useful for "
            "measuring solve-rate variance."
        ),
    )
    return p.parse_args(argv)


async def async_main(args: argparse.Namespace) -> int:
    # Determine per-tier picks.
    if args.tier_a is None and args.tier_b is None and args.tier_c is None:
        # Default split for N=10: 3/4/3. For other N: even spread.
        n = args.num_tasks
        a = n * 3 // 10
        b = n * 4 // 10
        c = n - a - b
        per_tier = {"A": a, "B": b, "C": c}
    else:
        per_tier = {
            "A": args.tier_a or 0,
            "B": args.tier_b or 0,
            "C": args.tier_c or 0,
        }
    total = sum(per_tier.values())
    if total == 0:
        logger.error("0 tasks requested (check --tier-a/b/c)")
        return 2
    logger.info(
        "Loading %s; picking %s = %d tasks (seed=%d, concurrency=%d)",
        args.jsonl,
        per_tier,
        total,
        args.seed,
        args.concurrency,
    )
    tasks = load_and_select_tasks(args.jsonl, per_tier=per_tier, seed=args.seed)

    args.trace_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Traces will be written to %s/", args.trace_dir)

    # --- ATIF writer: raw LLM capture + GCS upload ---
    #
    # Uses the same build_atif_writer / capture_ctx_for infrastructure as the
    # training and eval pipelines. The writer handles:
    #   - local directory layout (<log_dir>/atif/<recipe>/<job_id>/...)
    #   - per-task CaptureContext with on_written wired to _GcsUploader
    #   - background GCS upload with retry, backoff, failure ledger
    #   - lifecycle (start workers on construction, drain at end)
    atif_writer: Any | None = None
    if args.atif_output and args.atif_output.lower() not in ("", "off", "none"):
        from utils.atif_disk_writer import build_atif_writer

        run_name = args.vertex_run_name or (
            f"sl{args.step_limit}-sr{args.submit_retries}-s{args.seed}"
        )
        atif_writer = build_atif_writer(
            atif_output=args.atif_output,
            log_dir=str(args.trace_dir),
            recipe="agent-solver",
            tuning_job_id=run_name,
        )
        if atif_writer is not None:
            logger.info(
                "ATIF writer: atif_output=%r log_dir=%s/ recipe=agent-solver job_id=%s",
                args.atif_output,
                args.trace_dir,
                run_name,
            )

    # Per-agent knobs; sampler config lives on the sampler itself below.
    model_cfg = {
        "model_name": args.model,
        "env": args.env,
        "thinking_level": args.thinking_level or None,
        "max_generation_tokens": args.max_generation_tokens,
    }

    # ONE shared sampler for ALL rollouts (connection pooling + ADC
    # token reused). The generation cap is a per-call knob passed on
    # each sample() (see model_cfg["max_generation_tokens"] above and
    # SamplerChatCompletionModel), not a constructor arg.
    msrl_client = None  # set when using tuning-job sampling; closed in finally
    if args.tuning_job_id:
        if args.endpoint_id:
            logger.error("--tuning-job-id and --endpoint-id are mutually exclusive")
            return 2
        from backends.vertex_msrl_rest import (
            VertexMsrlRestClient,
            VertexMsrlRestSampler,
        )

        msrl_client = VertexMsrlRestClient(
            project=args.project,
            location=args.location,
            tuning_job_id=args.tuning_job_id,
        )
        logger.info("Waiting for tuning job %s to be RUNNING...", args.tuning_job_id)
        await msrl_client.wait_for_job_running(timeout=120)
        sampler = VertexMsrlRestSampler(msrl_client)
        logger.info(
            "VertexMsrlRestSampler: job=%s project=%r location=%r\n"
            "  tuning_job_name=%s",
            args.tuning_job_id,
            args.project,
            args.location,
            msrl_client.tuning_job_name,
        )
    else:
        sampler = GeminiApiSampler(
            model=args.model,
            project=args.project,
            env=args.env,
            location=args.location,
            endpoint_id=args.endpoint_id,
        )
        # Log the resolved sampler URL up-front so misrouted calls
        # (wrong --endpoint-id / --env combination) surface before the
        # first LLM call rather than as opaque per-task 404s minutes in.
        logger.info(
            "GeminiApiSampler URL: %s\n  (model=%r env=%r project=%r endpoint_id=%r)",
            sampler.url,
            sampler.model,
            sampler.env,
            sampler.project,
            sampler.endpoint_id,
        )
    # Resolve per-task CaptureContexts up front (async, needs writer).
    capture_ctxs_by_task: dict[int, Any] = {}
    if atif_writer is not None:
        for i in range(len(tasks)):
            ctx = await atif_writer.capture_ctx_for(
                task_id=f"task-{i:03d}",
                rollout_idx=0,
                session_id=f"task-{i:03d}-seed{args.seed}",
            )
            capture_ctxs_by_task[i] = ctx
        logger.info("Raw LLM capture: %d contexts", len(capture_ctxs_by_task))

    try:
        semaphore = asyncio.Semaphore(args.concurrency)
        spt = args.samples_per_task
        total_rollouts = len(tasks) * spt
        prober = _EvalProber(total_tasks=total_rollouts)
        prober_task = asyncio.create_task(_run_prober(prober, interval=10.0))
        wall_start = time.perf_counter()
        results: list[TaskResult] = await asyncio.gather(
            *(
                run_one_task_async(
                    task_index=i,
                    row=row,
                    sampler=sampler,
                    model_cfg=model_cfg,
                    trace_dir=args.trace_dir,
                    step_limit=args.step_limit,
                    wall_time_limit_seconds=args.wall_time_limit_seconds,
                    cost_limit=args.cost_limit,
                    submit_retries=args.submit_retries,
                    semaphore=semaphore,
                    prober=prober,
                    sample_suffix=f"_s{s}" if spt > 1 else "",
                    rollout_id=i * spt + s,
                    capture_ctx=capture_ctxs_by_task.get(i),
                    atif_writer=atif_writer,
                )
                for i, row in enumerate(tasks)
                for s in range(spt)
            )
        )
        wall_elapsed = time.perf_counter() - wall_start
        prober_task.cancel()
        print("", flush=True)  # newline after last \r prober line
    finally:
        if msrl_client is not None:
            await msrl_client.aclose()
        elif hasattr(sampler, "aclose"):
            await sampler.aclose()

    _print_summary_table(results, model_cfg, wall_elapsed=wall_elapsed)

    # --- Post-run: Vertex Experiment logging ---
    if args.vertex_experiment:
        _log_to_vertex_experiment(args, results, wall_elapsed)

    # Drain the ATIF writer's background GCS uploader.
    if atif_writer is not None:
        report = await atif_writer.drain()
        if report.not_uploaded:
            logger.warning("%d files not uploaded to GCS.", report.not_uploaded)

    return 0


def _log_to_vertex_experiment(
    args: argparse.Namespace,
    results: list[TaskResult],
    wall_elapsed: float,
) -> None:
    """Log params + metrics to a Vertex AI Experiment run."""
    try:
        from google.cloud import aiplatform
    except ImportError:
        logger.warning(
            "google-cloud-aiplatform not installed; skipping Vertex Experiment logging."
        )
        return

    run_name = args.vertex_run_name or (
        f"sl{args.step_limit}-sr{args.submit_retries}-s{args.seed}"
    )
    # Vertex run names: lowercase, alphanumeric + hyphens, max 128.
    run_name = run_name.lower().replace("_", "-")[:128]

    location = getattr(args, "location", "us-central1")
    logger.info(
        "[vertex] Logging to experiment=%r run=%r project=%r",
        args.vertex_experiment,
        run_name,
        args.project,
    )
    try:
        aiplatform.init(
            project=args.project,
            location=location,
            experiment=args.vertex_experiment,
        )
        aiplatform.start_run(run_name, resume=True)
    except Exception:
        try:
            aiplatform.start_run(run_name, resume=False)
        except Exception as e:
            logger.warning("Vertex start_run failed: %s", e)
            return

    # Log hyperparams
    params = {
        "model": args.model,
        "thinking_level": args.thinking_level or "off",
        "max_generation_tokens": args.max_generation_tokens,
        "step_limit": args.step_limit,
        "submit_retries": args.submit_retries,
        "wall_time_limit_seconds": args.wall_time_limit_seconds,
        "cost_limit": args.cost_limit,
        "seed": args.seed,
        "num_tasks": len(results),
        "concurrency": args.concurrency,
    }
    try:
        aiplatform.log_params(params)
    except Exception as e:
        logger.warning("log_params failed: %s", e)

    # Compute metrics
    n_correct = sum(1 for r in results if r.is_correct)
    denom = len(results) or 1
    exits: dict[str, int] = {}
    for r in results:
        exits[r.exit_status] = exits.get(r.exit_status, 0) + 1

    metrics: dict[str, float] = {
        "overall_accuracy": n_correct / denom,
        "avg_model_time_seconds": (sum(r.total_model_seconds for r in results) / denom),
        "avg_bash_calls_per_task": (sum(r.n_bash_calls for r in results) / denom),
        "wall_elapsed_seconds": wall_elapsed,
        "submitted_count": float(exits.get("Submitted", 0)),
        "limits_exceeded_count": float(exits.get("LimitsExceeded", 0)),
    }
    # Per-tier accuracy
    for tier in ("A", "B", "C"):
        tier_rs = [r for r in results if r.tier == tier]
        if tier_rs:
            tier_ok = sum(1 for r in tier_rs if r.is_correct)
            metrics[f"tier_{tier.lower()}_accuracy"] = tier_ok / len(tier_rs)

    try:
        aiplatform.log_metrics(metrics)
        logger.info("[vertex] Logged %d metrics.", len(metrics))
    except Exception as e:
        logger.warning("log_metrics failed: %s", e)

    try:
        aiplatform.end_run()
    except Exception as e:
        logger.warning("end_run failed: %s", e)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.location is None:
        args.location = "us-central1" if args.tuning_job_id else "global"
    return asyncio.run(async_main(args))


# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------


def _print_summary_table(
    results: list[TaskResult],
    model_cfg: dict,
    *,
    wall_elapsed: float | None = None,
) -> None:
    # NOTE: this whole function uses print (NOT logger) on purpose \u2014 it
    # emits a multi-line right-aligned results table meant for direct
    # human viewing at the end of a run. Routing through logger would
    # prefix a timestamp+level to every row and destroy the column
    # alignment (`_fmt` widths get shifted per row).
    print("")
    print("=" * 100)
    header = (
        f"AGENT SOLVER SUMMARY -- model={model_cfg['model_name']}, "
        f"thinking={model_cfg.get('thinking_level') or 'off'}, "
        f"n_tasks={len(results)}"
    )
    if wall_elapsed is not None:
        header += f", wall={wall_elapsed:.1f}s"
    print(header)
    print("=" * 100)

    # Compute column widths.
    rows = [TABLE_HEADER] + [r.to_row() for r in results]
    str_rows = [[str(c) for c in row] for row in rows]
    widths = [max(len(row[i]) for row in str_rows) for i in range(len(str_rows[0]))]

    def _fmt(row: list[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(row, widths))

    print(_fmt(str_rows[0]))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for r in str_rows[1:]:
        print(_fmt(r))

    n_correct = sum(1 for r in results if r.is_correct)
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    # ``len(results)`` is >= 1 by construction: main() gates
    # ``--num-tasks`` through ``_positive_int`` (rejects <1 at
    # argparse time), and ``asyncio.gather`` returns one entry per
    # awaited coroutine. The ``or 1`` guard is defense-in-depth for
    # a future caller that constructs this function directly (i.e.
    # bypassing main()'s argparse) with an empty list — cheaper than
    # a divide-by-zero crash for what would be a nonsense input.
    denom = len(results) or 1
    print(
        f"OVERALL: {n_correct}/{len(results)} correct "
        f"= {100.0 * n_correct / denom:.1f}%"
    )
    for tier in ("A", "B", "C"):
        tier_rs = [r for r in results if r.tier == tier]
        if not tier_rs:
            continue
        tier_ok = sum(1 for r in tier_rs if r.is_correct)
        print(
            f"  tier {tier}: {tier_ok}/{len(tier_rs)} "
            f"({100.0 * tier_ok / len(tier_rs):.0f}%)"
        )
    avg_llm = sum(r.total_model_seconds for r in results) / max(1, len(results))
    avg_bash = sum(r.n_bash_calls for r in results) / max(1, len(results))
    print(f"  avg model time / task: {avg_llm:.1f}s")
    print(f"  avg bash calls / task: {avg_bash:.1f}")
    exits: dict[str, int] = {}
    for r in results:
        exits[r.exit_status] = exits.get(r.exit_status, 0) + 1
    print(f"  exit statuses: {exits}")


if __name__ == "__main__":
    sys.exit(main())
