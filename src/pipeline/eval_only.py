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

"""Eval pipeline — schedules eval rollouts with concurrency control.

``EvalPipeline`` is the single eval scheduling primitive. Both the
standalone eval entry point (``eval_config_entry.run_eval``) and the
training-inline eval path (``BaseRolloutPipeline.schedule_eval``)
construct an ``EvalPipeline`` and call ``run()``.

Two construction patterns:

1. **Standalone eval** — caller provides ``eval_dataset``; the pipeline
   creates its own semaphore from ``config.rollout.max_concurrent_samples``.
2. **Training-inline** — caller provides a pre-built ``eval_iter`` and
   the training pipeline's shared ``semaphore`` (training + eval compete
   for one concurrency budget).

``EvalPipeline.run()`` drains eval tasks, runs concurrent rollouts with
semaphore backpressure, spec-driven metric aggregation, auto-derives the
x-axis step from ``sampler/policy_step`` when ``step_num=0``, and logs
aggregated ``eval/*`` metrics.
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from collections.abc import Callable
from typing import Any

from core.agent import Agent
from core.dataset import JsonlDataset
from core.metric_spec import aggregate_by_spec
from core.protocols import ProgressObserver
from core.task_data import TaskData
from utils.logger import Logger

logger = logging.getLogger(__name__)

# How often the eval prober logs a progress line (seconds).
_EVAL_PROBER_INTERVAL_SECONDS = 60

__all__ = ["EvalPipeline"]


class EvalPipeline:
    """Eval scheduling pipeline: drains tasks, runs rollouts, aggregates.

    Constructed with EITHER ``eval_dataset`` (standalone — builds its own
    semaphore + iter) OR ``eval_iter`` + ``semaphore`` (training-inline —
    reuses the training pipeline's shared concurrency budget).

    ``eval_source`` names which model this sweep samples: ``"sampler"``
    for the live training sampler, ``"checkpoint"`` for a frozen deployed
    endpoint. It is reported once per sweep via ``on_eval_started`` so an
    observer can attribute the sweep's trajectories; it does not affect
    sampling. A run with both sweeps enabled emits both values at the same
    step, which is the only way to tell the two apart downstream — they
    share the ``eval`` phase and the step number.
    """

    def __init__(
        self,
        *,
        agent: Agent,
        config: Any,
        ml_logger: Logger,
        # Standalone construction: provide eval_dataset, pipeline builds
        # its own semaphore and iter.
        eval_dataset: JsonlDataset | None = None,
        # Training-inline construction: provide pre-built iter + shared
        # semaphore from the training pipeline.
        eval_iter: Any | None = None,
        semaphore: asyncio.Semaphore | None = None,
        progress_observer: ProgressObserver | None = None,
        eval_source: str = "sampler",
    ):
        self._agent = agent
        self._config = config
        self._ml_logger = ml_logger
        self._progress_observer = progress_observer
        self._eval_source = eval_source

        if eval_dataset is not None:
            # Standalone mode: build iter + semaphore from config.
            self._eval_iter = eval_dataset.build_iter(
                max_items=config.eval.max_eval_items, repeat=False
            )
            self._semaphore = asyncio.Semaphore(config.rollout.max_concurrent_samples)
            self._dataset_len: int | None = len(eval_dataset)
        elif eval_iter is not None and semaphore is not None:
            # Training-inline mode: reuse shared resources.
            self._eval_iter = eval_iter
            self._semaphore = semaphore
            self._dataset_len = None
        else:
            raise ValueError(
                "EvalPipeline requires either eval_dataset (standalone)"
                " or both eval_iter + semaphore (training-inline)."
            )

    async def _run_prober(
        self,
        total_tasks: int,
        active_counter: list[int],
        completed_counter: list[int],
        t0: float,
    ) -> None:
        """Periodically log eval progress: active/completed, step-count
        buckets (duck-typed from agent.active_step_counts), and wall time."""
        while True:
            await asyncio.sleep(_EVAL_PROBER_INTERVAL_SECONDS)
            elapsed = time.monotonic() - t0
            active = active_counter[0]
            completed = completed_counter[0]

            # Step-count buckets (duck-typed: only agents that expose
            # active_step_counts will show step distribution).
            step_bucket_str = "-"
            step_counts = getattr(self._agent, "active_step_counts", None)
            if step_counts:
                step_buckets: dict[str, int] = {
                    "0": 0,
                    "1-5": 0,
                    "6-15": 0,
                    "16-30": 0,
                    "31+": 0,
                }
                for sc in step_counts.values():
                    if sc == 0:
                        step_buckets["0"] += 1
                    elif sc <= 5:
                        step_buckets["1-5"] += 1
                    elif sc <= 15:
                        step_buckets["6-15"] += 1
                    elif sc <= 30:
                        step_buckets["16-30"] += 1
                    else:
                        step_buckets["31+"] += 1
                step_bucket_str = (
                    " ".join(f"{k}:{v}" for k, v in step_buckets.items() if v) or "-"
                )

            logger.info(
                f"[Eval prober {elapsed:5.0f}s]"
                f" active={active}"
                f" done={completed}/{total_tasks}"
                f" | Active steps: [{step_bucket_str}]"
            )

    def _collect_eval_tasks(self) -> list[TaskData]:
        """Drain the eval iterator into a flat task list, capped by
        ``max_eval_items``.

        Resets the iterator first (training-inline calls ``run()``
        multiple times on the same iter). Returns ``[]`` if the iter
        is ``None`` (no eval dataset configured).

        ``max_eval_items`` caps the total number of tasks returned
        (``-1`` = no cap, use entire dataset).
        """
        if self._eval_iter is None:
            return []
        max_items = self._config.eval.max_eval_items
        eval_tasks: list[TaskData] = []
        self._eval_iter.reset()
        try:
            for batch in self._eval_iter:
                eval_tasks.extend(batch)
                if max_items > 0 and len(eval_tasks) >= max_items:
                    eval_tasks = eval_tasks[:max_items]
                    break
        except StopIteration:
            pass
        return eval_tasks

    async def run(
        self,
        step_num: int = 0,
        *,
        on_group: Callable[[TaskData, dict[str, Any]], None] | None = None,
        metric_prefix: str = "eval",
    ) -> dict[str, Any]:
        """Run one eval sweep and log aggregated ``{metric_prefix}/*`` metrics.

        Args:
            step_num: x-axis step for logged metrics. When 0 (default,
                standalone eval), auto-derives the step from the
                ``sampler/policy_step`` metric (the weighted-mean
                ``trainStepId`` across all sampled responses).
            on_group: optional callback invoked with ``(task_data, metrics)``
                for each completed eval group (used for JSONL serialization).
            metric_prefix: namespace prefix for the logged metrics. Defaults
                to ``"eval"``; the checkpoint-endpoint sweep passes
                ``"eval_ckpt"`` so both sweeps can log at the same step
                without overwriting each other.

        Returns:
            The aggregated ``{metric_prefix}/*`` dict (also logged via
            ml_logger).
        """
        eval_tasks = self._collect_eval_tasks()

        if not eval_tasks:
            if self._dataset_len is not None:
                logger.warning(
                    "No eval tasks found (max_eval_items=%d, dataset len=%d)."
                    " Returning empty metrics.",
                    self._config.eval.max_eval_items,
                    self._dataset_len,
                )
            return {}

        if self._progress_observer is not None:
            try:
                await self._progress_observer.on_eval_started(
                    step_num, eval_source=self._eval_source
                )
            except Exception:
                pass  # best-effort

        eval_samples_per_task = self._config.eval.eval_samples_per_task
        thinking_level = self._config.agent.thinking_level
        max_generation_tokens = self._config.agent.max_generation_tokens
        semaphore = self._semaphore

        logger.info(
            f"\n========== Starting evaluation after step {step_num} =========="
        )
        # Mutable counters for the prober (list[int] so the closure mutates
        # the shared object, not a local copy).
        active_counter = [0]
        completed_counter = [0]
        # Per-eval-step monotonic dispatch counter (mirrors the training
        # dispatcher's rollout_index; assigned in evaluate_single below).
        eval_dispatch_counter = [0]
        t0 = time.monotonic()
        prober_task = asyncio.create_task(
            self._run_prober(
                total_tasks=len(eval_tasks),
                active_counter=active_counter,
                completed_counter=completed_counter,
                t0=t0,
            )
        )
        try:

            async def evaluate_single(
                task_data: TaskData,
            ) -> dict[str, Any]:
                logger.info(
                    f"Starting single task rollout at eval step {step_num} for"
                    f" {eval_samples_per_task} samples"
                )
                # Per-trajectory permit release (mirrors the training
                # dispatcher): each completed eval trajectory frees one
                # permit immediately instead of waiting for the group's
                # slowest straggler. ``pending`` tracks how many of the N
                # permits acquired up front are still un-released so the
                # ``finally`` drains whatever the agent didn't fire --
                # guaranteeing exactly N net acquires no matter what.
                pending = eval_samples_per_task

                def _on_trajectory_done() -> None:
                    nonlocal pending
                    if pending <= 0:
                        # Defensive: a buggy agent firing more than N times
                        # would otherwise over-release and inflate budget.
                        return
                    semaphore.release()
                    pending -= 1

                active_counter[0] += 1
                # Reserve dispatch index before the await to avoid
                # concurrent-task collisions.
                dispatch_index = eval_dispatch_counter[0]
                eval_dispatch_counter[0] += 1
                try:
                    _, metrics = await self._agent.group_rollout(
                        task_data,
                        eval_samples_per_task,
                        thinking_level=thinking_level or None,
                        max_generation_tokens=max_generation_tokens,
                        on_trajectory_done=_on_trajectory_done,
                        phase="eval",
                        dispatch_index=dispatch_index,
                        # Scopes this sweep's artifact dirs to the step, so
                        # re-running the same eval tasks at a later step (or
                        # the checkpoint sweep running at the SAME step) can't
                        # overwrite them.
                        step=step_num,
                    )
                    logger.info(
                        f"Finished single task rollout at eval step {step_num}"
                        f" for {eval_samples_per_task} samples"
                    )
                    if on_group is not None:
                        on_group(task_data, metrics)
                    return metrics
                except Exception as e:
                    traceback.print_exc()
                    logger.error(f"Evaluation error: {e}")
                    return {}
                finally:
                    active_counter[0] -= 1
                    completed_counter[0] += 1
                    # Notify progress observer of eval item completion.
                    if self._progress_observer is not None:
                        try:
                            await self._progress_observer.on_eval_item_completed(
                                completed_eval_items=completed_counter[0],
                                total_eval_items=len(eval_tasks),
                            )
                        except Exception:
                            pass  # best-effort, never block eval
                    # Drain any unfired callbacks so this eval nets exactly
                    # ``eval_samples_per_task`` releases against the N
                    # acquired up front (agent raised pre-fanout, some
                    # trajectories cancelled, etc.).
                    while pending > 0:
                        semaphore.release()
                        pending -= 1

            pending = []
            for task_data in eval_tasks:
                for _ in range(eval_samples_per_task):
                    await semaphore.acquire()
                pending.append(asyncio.create_task(evaluate_single(task_data)))

            results = await asyncio.gather(*pending)

            # Per-metric reduction strategy comes from its registered
            # MetricSpec.
            aggregated = aggregate_by_spec(results)

            # Auto-derive x-axis from the response's trainStepId when
            # the caller didn't specify one (standalone eval, step_num=0).
            effective_step = step_num
            if step_num == 0:
                policy_step = aggregated.get("sampler/policy_step")
                if policy_step is not None:
                    effective_step = int(policy_step)
                    logger.info(
                        "Standalone eval: derived log step %d from response "
                        "trainStepId (sampler/policy_step).",
                        effective_step,
                    )

            eval_metrics = {f"{metric_prefix}/{k}": v for k, v in aggregated.items()}
            self._ml_logger.log_metrics(eval_metrics, step=effective_step)
            logger.info(f"Evaluation completed for step {effective_step}.")
            if self._progress_observer is not None:
                await self._progress_observer.on_eval_completed(effective_step)
            return eval_metrics

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Evaluation loop error for step {step_num} : {e}")
            return {}
        finally:
            prober_task.cancel()
            try:
                await prober_task
            except asyncio.CancelledError:
                pass
