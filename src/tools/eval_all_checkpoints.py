#!/usr/bin/env python3
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
"""Evaluate all checkpoints of a finished Vertex MSRL tuning job.

Recipe-agnostic core engine.  Fetches the checkpoint list from
``GetTuningJob``, builds shared infrastructure (reward fn, eval dataset,
ATIF writer) once, then runs one ``EvalPipeline`` sweep per checkpoint
(sequentially) -- only the backend/sampler/agent are rebuilt for each
checkpoint's tuned-model endpoint via ``GeminiApiBackendConfig``.  All
per-checkpoint metrics are logged to a single Vertex Experiment run with
``step=checkpoint_step`` so the experiment UI shows an eval curve across
checkpoints.

This module exposes :func:`run_checkpoint_eval` as the public entry
point.  Recipe-specific CLI wrappers (which resolve config names,
apply internal defaults, and set recipe-specific experiment names)
live outside ``src/``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import pathlib
import re
import sys
import time
import urllib.request
from typing import Any, Callable, Dict

from backends.backend_config import GeminiApiBackendConfig
from core.composite_callback import CompositeTrajectoryCallback
from core.run_config import RunConfig
from core.task_data import TaskData
from pipeline.eval_only import EvalPipeline
from pipeline.eval_serializer import write_eval_jsonl
from utils.atif_disk_writer import build_atif_writer
from utils.experiment_fields import build_experiment_fields
from utils.logger import Logger, resolve_vertex_tensorboard_config, setup_logging

logger = logging.getLogger(__name__)


class _CampaignProgressAdapter:
    """Re-base per-sweep eval-item counts onto campaign-wide totals.

    ``EvalPipeline`` reports ``on_eval_item_completed`` with counts local
    to ONE sweep (``1..items_per_sweep``, then resets for the next
    checkpoint). A progress observer's experiment record models a single
    ``completed_eval_items`` / ``total_eval_items`` pair, so without
    re-basing, a campaign's progress would restart on every checkpoint.

    This adapter adds the completed-sweep offset and reports the
    campaign-wide total. Every other observer hook (``on_eval_started``,
    ``on_eval_completed``, ``on_experiment_*``) passes straight through
    via ``__getattr__``, so the observer's "current eval step" still
    tracks the checkpoint step being evaluated.
    """

    def __init__(self, inner: Any, *, items_per_sweep: int, total_sweeps: int):
        self._inner = inner
        self._items_per_sweep = max(items_per_sweep, 0)
        self._campaign_total = max(items_per_sweep * total_sweeps, 0)
        self._offset = 0

    def sweep_finished(self) -> None:
        """Advance the offset by one full sweep's worth of items."""
        self._offset += self._items_per_sweep

    async def on_eval_item_completed(
        self, *, completed_eval_items: int, total_eval_items: int
    ) -> None:
        await self._inner.on_eval_item_completed(
            completed_eval_items=self._offset + completed_eval_items,
            total_eval_items=self._campaign_total or total_eval_items,
        )

    def __getattr__(self, name: str) -> Any:
        # Pass through every other ProgressObserver hook unchanged.
        return getattr(self._inner, name)


class _NullLogger(Logger):
    """No-op logger for per-sweep EvalPipeline (metrics logged centrally)."""

    def log_hparams(self, config: Any) -> None:
        pass

    def log_metrics(self, metrics: Dict[str, Any], step: int | None = None) -> None:
        pass


# ---------------------------------------------------------------------------
# Tuning-job checkpoint discovery
# ---------------------------------------------------------------------------

_TUNING_JOB_RE = re.compile(
    r"^projects/(?P<project>[^/]+)/locations/(?P<location>[^/]+)"
    r"/tuningJobs/(?P<job_id>[^/]+)$"
)


def _parse_tuning_job_resource(resource: str) -> dict[str, str]:
    """Parse ``projects/P/locations/L/tuningJobs/ID`` into components."""
    m = _TUNING_JOB_RE.match(resource)
    if not m:
        raise ValueError(
            f"Invalid tuning job resource name: {resource!r}.  "
            "Expected projects/P/locations/L/tuningJobs/ID."
        )
    return m.groupdict()


def _get_access_token() -> str:
    """Get an access token via ADC."""
    import google.auth
    import google.auth.transport.requests

    creds, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    creds.refresh(google.auth.transport.requests.Request())
    return creds.token


def _get_tuning_job(resource: str) -> dict[str, Any]:
    """Call GetTuningJob REST API and return the full response dict."""
    parts = _parse_tuning_job_resource(resource)
    location = parts["location"]
    host = f"https://{location}-aiplatform.googleapis.com"
    url = f"{host}/v1beta1/{resource}"

    token = _get_access_token()
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


@dataclasses.dataclass
class Checkpoint:
    """A tuning-job checkpoint with its step and tuned-model endpoint."""

    step: int  # checkpoint index from the API (1-based sequence number)
    epoch: int
    tuned_model_endpoint: str  # projects/P/locations/L/endpoints/E


def _extract_checkpoints(job: dict[str, Any]) -> list[Checkpoint]:
    """Extract checkpoints from a GetTuningJob response.

    Checkpoints live under ``tunedModel.checkpoints`` -- a list of dicts
    with ``checkpointId``, ``step`` (string), and ``endpoint`` (the
    deployed tuned-model endpoint resource for that checkpoint).
    """
    tuned_model = job.get("tunedModel", {})
    raw = tuned_model.get("checkpoints", [])
    if not raw:
        raise ValueError(
            "Tuning job has no checkpoints under tunedModel.checkpoints.  "
            "Ensure checkpointInterval was set and the job completed.  "
            f"tunedModel keys: {list(tuned_model.keys())}"
        )
    checkpoints = []
    for ckpt in raw:
        step = ckpt.get("step")
        if step is None:
            logger.warning("Skipping checkpoint with no step: %s", ckpt)
            continue
        endpoint = ckpt.get("endpoint", "")
        if not endpoint:
            logger.warning("Skipping checkpoint step %s: no endpoint", step)
            continue
        checkpoints.append(
            Checkpoint(
                step=int(step),
                epoch=int(ckpt.get("epoch", 0)),
                tuned_model_endpoint=endpoint,
            )
        )
    checkpoints.sort(key=lambda c: c.step)
    return checkpoints


_ENDPOINT_RESOURCE_RE = re.compile(
    r"^projects/(?P<project>[^/]+)/locations/(?P<location>[^/]+)"
    r"/endpoints/(?P<endpoint_id>[^/]+)$"
)


def _parse_endpoint_resource(resource: str) -> dict[str, str]:
    """Parse ``projects/P/locations/L/endpoints/E`` into components."""
    m = _ENDPOINT_RESOURCE_RE.match(resource)
    if not m:
        raise ValueError(
            f"Invalid endpoint resource name: {resource!r}.  "
            "Expected projects/P/locations/L/endpoints/E."
        )
    return m.groupdict()


def _resolve_project_number(project: str) -> str:
    """Resolve a project ID to its project number.

    Returns the project number string if resolution succeeds, otherwise
    returns the input ``project`` unchanged.

    This is needed to work around a Vertex SDK bug where
    ``assign_backing_tensorboard`` compares the TB resource's project
    NUMBER against the ``project`` arg as raw strings.
    """
    try:
        from google.cloud import resourcemanager_v3  # type: ignore[import-untyped]

        rm_client = resourcemanager_v3.ProjectsClient()
        proj_info = rm_client.get_project(name=f"projects/{project}")
        project_number = proj_info.name.split("/")[-1]
        if project_number != project:
            logger.info(
                "Resolved project for Vertex Experiment: %s -> %s",
                project,
                project_number,
            )
            return project_number
    except Exception:
        logger.debug("Could not resolve project number; using %s as-is", project)
    return project


def _resolve_tensorboard(
    tb_config: Any,
    project: str,
    location: str,
    tb_display_name: str = "gemini-rl-exp",
) -> Any:
    """Pre-lookup TensorBoard in the correct project.

    When ``tb_config`` is ``None`` (auto-provision requested), the
    ``_resolve_tensorboard`` inside ``VertexExperimentLogger`` would
    search the ADC quota project instead of the target project.  This
    function pre-inits aiplatform with the correct project number and
    looks up the TB ourselves, returning the full resource name.
    """
    if tb_config is not None:
        return tb_config
    try:
        from google.cloud import aiplatform as _aip  # type: ignore[import-untyped]

        _aip.init(project=project, location=location)
        existing = list(
            _aip.Tensorboard.list(filter=f'display_name="{tb_display_name}"')
        )
        if existing:
            resource = existing[0].resource_name
            logger.info("Using existing TensorBoard: %s", resource)
            return resource
        logger.info(
            "No %r TensorBoard found in %s/%s; disabling TB time-series.",
            tb_display_name,
            project,
            location,
        )
        return False
    except Exception:
        logger.debug("TB lookup failed; disabling TB time-series")
        return False


def build_agent_trajectory_callback(atif_writer: Any, observer: Any) -> Any:
    """Fan the agent's trajectory hooks out to the ATIF writer AND the observer.

    Mirrors the standalone eval path (``run_eval_with_progress``). The progress
    observer also implements the ``TrajectoryCallback`` hooks; wiring it ONLY as
    a ``progress_observer`` yields experiment-level stats but leaves the
    per-trajectory records empty -- stats render, individual rollouts do
    not. Also connects GCS upload status so the observer can report it.
    """
    if observer is not None and atif_writer is not None:
        status = getattr(atif_writer, "gcs_upload_status", None)
        setter = getattr(observer, "set_gcs_status_source", None)
        if status is not None and setter is not None:
            setter(status)

    callbacks = [cb for cb in (atif_writer, observer) if cb is not None]
    if not callbacks:
        return None
    if len(callbacks) == 1:
        return callbacks[0]
    return CompositeTrajectoryCallback(callbacks)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_checkpoint_eval(
    config: RunConfig,
    tuning_job: str,
    *,
    recipe: str,
    vertex_experiment: str,
    vertex_experiment_run_name: str = "",
    vertex_env: str = "prod",
    log_dir: str = "logs",
    jsonl_output_dir: str = "",
    vertex_experiment_tensorboard: str = "",
    include_base_model: bool = True,
    max_checkpoints: int = 0,
    summary_metric_keys: list[str] | None = None,
    progress_observer_factory: Callable[[], Any] | None = None,
    config_name: str = "",
    atif_viewer_base_url: str = "",
) -> list[dict[str, Any]]:
    """Evaluate all checkpoints of a finished tuning job.

    This is the recipe-agnostic core.  Callers are responsible for:

    * Resolving ``config_name`` into a fully-configured ``RunConfig``
      (with project defaults, launch overrides, etc. already applied).
    * Setting ``vertex_experiment`` and ``vertex_experiment_run_name``
      to recipe-appropriate values.
    * Providing ``recipe`` (the ATIF recipe label).
    * Optionally providing ``summary_metric_keys`` -- a list of
      ``eval/*`` keys to include in the final summary table (beyond
      the default accuracy/reward).

    Args:
        config: Fully resolved RunConfig (backend, agent, reward, task,
            eval knobs).  The backend in this config is used for the
            base-model sweep; checkpoint sweeps build their own backend.
        tuning_job: Full resource name
            ``projects/P/locations/L/tuningJobs/ID``.
        recipe: ATIF recipe label (e.g. ``"word_puzzle"``).
        vertex_experiment: Vertex Experiment name for cross-checkpoint
            metric logging.
        vertex_experiment_run_name: Run name within the experiment.
            Defaults to ``eval-{job_id}-{YYYYMMDD-HHMMSS}`` when empty.
        vertex_env: Default sampler env/realm name (e.g. ``"prod"``).
            Passed through to each checkpoint sweep unchanged.
        log_dir: Root log directory.
        jsonl_output_dir: Per-checkpoint JSONL output dir.  Defaults to
            ``{log_dir}/eval-checkpoints/{job_id}/``.
        vertex_experiment_tensorboard: TB config string (``""`` =
            auto-provision, ``"disable"`` = no TB).
        include_base_model: Whether to evaluate the base model at step 0.
        max_checkpoints: If > 0, evaluate only the first N checkpoints.
        summary_metric_keys: Extra ``eval/*`` keys to show in the
            summary table.  ``sampling_accuracy`` and ``reward/mean``
            are always shown.
        progress_observer_factory: Zero-arg callable returning a
            ``ProgressObserver``.  A FACTORY rather than an instance
            because an observer may own async resources bound to the
            running loop, and this function owns its own ``asyncio.run``
            -- the observer is therefore constructed inside the loop
            (same reason ``build_atif_writer`` is called there).  The
            whole campaign is reported as ONE experiment: per-sweep item
            counts are re-based onto campaign-wide totals via
            ``_CampaignProgressAdapter`` so progress advances
            monotonically across checkpoints instead of resetting.  If
            the observer exposes ``close()``, it is awaited on teardown.
        config_name: Registered config name, forwarded to the observer's
            experiment record (informational).

    Returns:
        List of per-checkpoint result dicts with ``step``, ``label``,
        ``metrics``, and ``job_id`` (or ``error=True`` on failure).
    """
    # --- Resolve project/location from config ---
    project = config.backend.project
    location = getattr(config.backend, "location", "us-central1")
    # Sampling location and Experiment location are DIFFERENT axes.
    # Gemini 3.x base models on prod are served only from ``global``, but
    # Vertex Experiments are regional -- a MetadataStore cannot be created
    # in ``global`` (the API returns 400), which would abort the whole
    # sweep before any eval runs. Source the Experiment region from the
    # task config (default ``us-central1``), mirroring how
    # ``eval_config_entry.run_eval`` wires ``vertex_experiment_location``.
    exp_location = (
        getattr(config.task, "vertex_experiment_location", "") or "us-central1"
    )
    # Project is the same kind of axis: the experiment can live outside the
    # project that owns the tuning job.
    exp_project = getattr(config.task, "vertex_experiment_project", "") or project

    # --- Fetch tuning job and extract checkpoints ---
    logger.info("Fetching tuning job: %s", tuning_job)
    job_response = _get_tuning_job(tuning_job)
    job_state = job_response.get("state", "UNKNOWN")
    base_model = job_response.get("baseModel", "unknown")
    logger.info("Job state: %s, base model: %s", job_state, base_model)
    checkpoints = _extract_checkpoints(job_response)
    if max_checkpoints > 0:
        checkpoints = checkpoints[:max_checkpoints]
    logger.info(
        "Found %d checkpoint(s): steps %s",
        len(checkpoints),
        [c.step for c in checkpoints],
    )

    job_parts = _parse_tuning_job_resource(tuning_job)
    job_id = job_parts["job_id"]

    if not vertex_experiment_run_name:
        # UTC (gmtime), not local: VMs run in different timezones, so a local
        # stamp makes run names from different hosts non-comparable and sort
        # wrongly against each other.
        vertex_experiment_run_name = (
            f"eval-{job_id}-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"
        )

    if not jsonl_output_dir:
        jsonl_output_dir = f"{log_dir}/eval-checkpoints/{job_id}"

    logger.info(
        "Vertex Experiment: %s, run: %s",
        vertex_experiment,
        vertex_experiment_run_name,
    )
    logger.info("JSONL output dir: %s", jsonl_output_dir)

    # --- Set up cross-checkpoint Vertex Experiment logger ---
    exp_log_dir = f"{log_dir}/eval-checkpoints/{job_id}/experiment"
    tb_config = resolve_vertex_tensorboard_config(vertex_experiment_tensorboard)

    vertex_project_for_exp = _resolve_project_number(exp_project)
    tb_config = _resolve_tensorboard(tb_config, vertex_project_for_exp, exp_location)

    exp_logger = setup_logging(
        log_dir=exp_log_dir,
        do_configure_logging_module=False,
        vertex_experiment=vertex_experiment,
        vertex_project=vertex_project_for_exp,
        vertex_location=exp_location,
        vertex_tensorboard=tb_config,
        vertex_run_name=vertex_experiment_run_name,
    )

    exp_logger.log_hparams(
        {
            "tuning_job": tuning_job,
            "base_model": base_model,
            "config_name": recipe,
            "num_checkpoints": len(checkpoints),
            "atif_output": config.task.atif_output,
        }
    )

    # --- Build eval targets ---
    eval_targets: list[tuple[int, str, str | None, str | None]] = []

    if include_base_model:
        eval_targets.append((0, f"base-model ({base_model})", None, None))

    for ckpt in checkpoints:
        ep_parts = _parse_endpoint_resource(ckpt.tuned_model_endpoint)
        ep_id = ep_parts["endpoint_id"]
        ep_location = ep_parts["location"]
        eval_targets.append(
            (
                ckpt.step,
                f"checkpoint step {ckpt.step} (endpoint {ep_id})",
                ep_id,
                ep_location,
            )
        )

    # --- Build shared infrastructure (once) ---
    reward_fn = config.reward.make()
    eval_dataset = config.task.make_eval()
    service_account = getattr(config.backend, "service_account", "")

    agent_config = config.agent
    if (
        hasattr(agent_config, "service_account")
        and not agent_config.service_account
        and service_account
    ):
        agent_config = dataclasses.replace(
            agent_config, service_account=service_account
        )

    # Placeholders — built inside async context (see _run_all_sweeps).
    atif_writer = None
    progress_adapter: _CampaignProgressAdapter | None = None
    agent_trajectory_callback = None

    sweep_task_config = dataclasses.replace(config.task, vertex_experiment="")

    pathlib.Path(jsonl_output_dir).mkdir(parents=True, exist_ok=True)

    # --- Sequential eval sweeps ---
    all_results: list[dict[str, Any]] = []
    null_logger = _NullLogger()
    _campaign_t0 = time.monotonic()
    # Tasks per sweep = dataset size capped by ``max_eval_items`` (-1 = all).
    # Used to re-base per-sweep progress onto a campaign-wide total.
    _ds_len = len(eval_dataset) if eval_dataset is not None else 0
    _max_items = config.eval.max_eval_items
    items_per_sweep = _ds_len if _max_items < 0 else min(_max_items, _ds_len)

    async def _run_one_sweep(
        step: int,
        label: str,
        endpoint_id: str | None,
        endpoint_location: str | None,
    ) -> dict[str, Any]:
        """Run one eval sweep for a single checkpoint (or base model)."""
        sweep_env = vertex_env
        sweep_location = endpoint_location or location

        sweep_backend = GeminiApiBackendConfig(
            model=base_model,
            project=project,
            env=sweep_env,
            location=sweep_location,
            endpoint_id=endpoint_id or "",
            service_account=service_account,
        )

        sweep_config = dataclasses.replace(
            config,
            backend=sweep_backend,
            agent=agent_config,
            train=dataclasses.replace(config.train, log_dir=log_dir),
            task=sweep_task_config,
        )

        bundle = await sweep_backend.make()
        try:
            agent = agent_config.make(
                sampler=bundle.sampler,
                job_id=bundle.job_id,
                reward_fn=reward_fn,
                atif_writer=agent_trajectory_callback,
            )

            collected_groups: list[tuple[Any, dict[str, Any]]] = []

            def _collect_group(
                task_data: TaskData, group_metrics: dict[str, Any]
            ) -> None:
                collected_groups.append((task_data, group_metrics))

            eval_pipeline = EvalPipeline(
                agent=agent,
                config=sweep_config,
                ml_logger=null_logger,
                eval_dataset=eval_dataset,
                progress_observer=progress_adapter,
            )
            # Pass the checkpoint step explicitly so the observer reports
            # WHICH checkpoint is running. Metrics still land centrally
            # via ``exp_logger`` below; ``null_logger`` discards the
            # per-sweep copy, so this only affects observer/log step.
            metrics = await eval_pipeline.run(step, on_group=_collect_group)

            jsonl_path = pathlib.Path(f"{jsonl_output_dir}/eval_step_{step}.jsonl")
            if collected_groups:
                write_eval_jsonl(jsonl_path, collected_groups)

            logger.info("Eval complete for step %d. Metrics: %s", step, metrics)
            return {"metrics": metrics, "job_id": bundle.job_id}
        finally:
            await bundle.aclose()

    async def _run_all_sweeps() -> None:
        """Run all sweeps in a single event loop.

        The ATIF writer is built here (not in the outer sync caller)
        because ``build_atif_writer`` calls ``_GcsUploader.start()``
        which spawns asyncio tasks via ``ensure_future`` -- that
        requires a running event loop.
        """
        nonlocal atif_writer, progress_adapter, agent_trajectory_callback
        # Campaign-unique id: it is BOTH the observer's experiment-record
        # key and the ATIF path segment (``{atif_output}/{recipe}/{id}``).
        # ``job_id`` alone collides when two campaigns evaluate the same
        # tuning job concurrently (e.g. an agent sweep and a single-step
        # sweep), so scope by recipe and stamp with a timestamp -- the
        # same anti-clobber convention the standalone eval path uses.
        # UTC (gmtime), not local: this id is the experiment-record key
        # AND the ATIF path segment, so a local stamp makes records from
        # differently-zoned VMs sort and read wrongly.
        effective_job_id = (
            f"{recipe}-{job_id}-{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}"
        )
        atif_writer = build_atif_writer(
            atif_output=config.task.atif_output,
            log_dir=log_dir,
            recipe=recipe,
            tuning_job_id=effective_job_id,
            max_concurrent_uploads=config.task.atif_upload_concurrency,
        )

        observer = progress_observer_factory() if progress_observer_factory else None
        agent_trajectory_callback = build_agent_trajectory_callback(
            atif_writer, observer
        )
        if observer is not None:
            progress_adapter = _CampaignProgressAdapter(
                observer,
                items_per_sweep=items_per_sweep,
                total_sweeps=len(eval_targets),
            )
            try:
                await observer.on_experiment_started(
                    **build_experiment_fields(
                        config,
                        effective_job_id=effective_job_id,
                        recipe=recipe,
                        bundle_job_id=job_id,
                        total_train_steps=0,  # eval-only campaign
                        total_eval_items=items_per_sweep * len(eval_targets),
                        config_name=config_name,
                        experiment_group=vertex_experiment,
                        atif_viewer_base_url=atif_viewer_base_url or "",
                        # This campaign registers its own run name; without
                        # passing it the experiment deep link 404s.
                        vertex_run_name=vertex_experiment_run_name,
                    )
                )
            except Exception:
                logger.warning(
                    "Progress observer on_experiment_started failed", exc_info=True
                )

        campaign_failed = False
        for step, label, endpoint_id, endpoint_location in eval_targets:
            logger.info("%s\nEvaluating: %s\n%s", "=" * 60, label, "=" * 60)

            try:
                result = await _run_one_sweep(
                    step, label, endpoint_id, endpoint_location
                )
                metrics = result["metrics"]

                scalar_metrics = {}
                for k, v in metrics.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        scalar_metrics[k] = float(v)

                if scalar_metrics:
                    exp_logger.log_metrics(scalar_metrics, step=step)
                    logger.info(
                        "Logged %d metrics at step %d", len(scalar_metrics), step
                    )
                else:
                    logger.info("No scalar metrics to log for step %d", step)

                all_results.append(
                    {
                        "step": step,
                        "label": label,
                        "metrics": metrics,
                        "job_id": result.get("job_id"),
                    }
                )

            except Exception:
                campaign_failed = True
                logger.exception("Eval sweep failed for %s", label)
                logger.error("FAILED: %s", label)
                all_results.append(
                    {
                        "step": step,
                        "label": label,
                        "error": True,
                    }
                )
            finally:
                # Advance the campaign progress offset whether the sweep
                # succeeded or failed, so a failed checkpoint doesn't
                # stall the bar for the rest of the campaign.
                if progress_adapter is not None:
                    progress_adapter.sweep_finished()

        if observer is not None:
            try:
                await observer.on_experiment_completed(
                    status="failed" if campaign_failed else "completed",
                    wall_time_seconds=time.monotonic() - _campaign_t0,
                )
            except Exception:
                logger.warning(
                    "Progress observer on_experiment_completed failed", exc_info=True
                )
            close = getattr(observer, "close", None)
            if close is not None:
                try:
                    await close()
                except Exception:
                    logger.warning("Progress observer close failed", exc_info=True)

        if atif_writer is not None:
            await atif_writer.on_job_completed(
                {"recipe": recipe, "tuning_job_id": job_id}
            )
            if atif_writer.gcs_uri:
                logger.info("ATIF trajectories uploaded to: %s", atif_writer.gcs_uri)
            logger.info("ATIF local dir: %s", atif_writer.job_dir)

    asyncio.run(_run_all_sweeps())

    # --- Summary ---
    exp_logger.close()

    _extra_keys = summary_metric_keys or []
    # Build the whole summary table as a single log record so rows stay
    # visually aligned (per-row logger calls would each get their own
    # timestamp prefix, breaking the table).
    lines = ["=" * 60, "All checkpoint evaluations complete.", "=" * 60]
    for r in all_results:
        status = "FAIL" if r.get("error") else "OK"
        step = r["step"]
        label = r["label"]
        metrics = r.get("metrics", {})
        accuracy = metrics.get("eval/learning/sampling_accuracy", "N/A")
        reward = metrics.get("eval/learning/reward/mean", "N/A")
        parts = [
            f"  Step {step:>4d} [{status}] {label}",
            f"accuracy={accuracy}  reward={reward}",
        ]
        for key in _extra_keys:
            parts.append(f"{key.split('/')[-1]}={metrics.get(key, 'N/A')}")
        lines.append("  ".join(parts))
    lines.append("")
    lines.append(
        f"Vertex Experiment: {vertex_experiment} / run: {vertex_experiment_run_name}"
    )
    logger.info("\n%s", "\n".join(lines))

    return all_results
