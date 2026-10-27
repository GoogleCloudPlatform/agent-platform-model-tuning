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

"""Inline dynamic-sampling sync rollout pipeline.

- ``SyncRolloutPipeline`` — inline dynamic-sampling accumulate loop with
  no concurrent producer. Implements the ``RolloutPipeline`` protocol; trainer
  sees always-fresh (on-policy) rollouts.

The shared rollout machinery (``InOrderBuffer``, ``DispatchGate``,
``RolloutDispatcher``, ``MetricsAggregator``, and the eval-scheduling
helpers) still lives in ``pipeline.rollout_pipeline`` and is imported
here. ``AsyncRolloutPipeline`` lives in
``pipeline.async_rollout_pipeline``; sync and async do NOT import each
other.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections import Counter
from typing import Any

from dataclasses import dataclass

from core.agent import Agent
from core.atif_trajectory import GroupFilterOutcome
from core.rollout_protocol import BatchItem
from core.dataset import JsonlDataset
from core.metric_spec import aggregate_by_spec
from core.run_config import RolloutConfig, RunConfig
from core.task_data import TaskData
from pipeline.rollout_pipeline import BaseRolloutPipeline, group_filter_metrics
from utils.logger import Logger

logger = logging.getLogger(__name__)

__all__ = [
    "SyncRolloutConfig",
    "SyncRolloutPipeline",
]


@dataclass(frozen=True)
class SyncRolloutConfig(RolloutConfig):
    """Sync rollout config. Selects ``SyncRolloutPipeline`` via ``make()``."""

    def make(
        self,
        *,
        run_config,
        agent,
        train_dataset,
        eval_dataset,
        eval_agent,
        ml_logger,
        resume_train_iter_position=None,
    ) -> SyncRolloutPipeline:
        return SyncRolloutPipeline(
            agent=agent,
            config=run_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            eval_agent=eval_agent,
            ml_logger=ml_logger,
            resume_train_iter_position=resume_train_iter_position,
        )


class SyncRolloutPipeline(BaseRolloutPipeline):
    """Synchronous rollout pipeline: inline dynamic-sampling accumulate loop.

    Implements the ``RolloutPipeline`` protocol. No concurrent producer;
    ``next_batch`` runs the whole rollout loop inline, over-sampling
    until ``n`` valid trajectory groups accumulate. ``run()`` is a thin
    awaiter that blocks on ``stop()`` and drains pending eval tasks —
    it exists so ``asyncio.gather(pipeline.run(),
    train_and_eval_loop(...))`` works identically for both pipelines.

    Off-policy staleness is zero by construction (every batch is fresh
    right before its train step). Dynamic sampling drops zero-variance
    and (optionally) transient-failure-heavy groups; the
    ``running_pass_rate`` EMA adapts how much we over-fetch per round.

    ``next_batch()`` awaits any pending eval tasks at the START of its
    call so "eval blocks next training step" holds without making
    ``schedule_eval`` async.
    """

    def __init__(
        self,
        *,
        agent: Agent,
        config: RunConfig,
        train_dataset: JsonlDataset,
        eval_dataset: JsonlDataset | None,
        eval_agent: Agent | None,
        ml_logger: Logger,
        resume_train_iter_position: int | None = None,
    ):
        super().__init__(
            agent=agent,
            config=config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            eval_agent=eval_agent,
            ml_logger=ml_logger,
            resume_train_iter_position=resume_train_iter_position,
        )

        # Total in-flight LLM samples allowed at once; ``next_batch``
        # derives its per-round chunk size as
        # ``max_concurrent_samples // samples_per_task``.
        self._max_concurrent_samples = config.rollout.max_concurrent_samples

        # batch_size=1 because dynamic sampling pulls tasks one at a
        # time (over-fetching by ``over_sample_factor / running_pass_rate``).
        self._train_iter = train_dataset.build_iter(
            batch_size=1,
            resume_from_position=resume_train_iter_position,
        )

        # Dynamic-sampling state persists across next_batch calls so the
        # EMA and total-tasks counter don't reset every step.
        self._running_pass_rate: float = 0.8
        self._tasks_processed_total: int = 0

    # ---- public methods (run + next_batch are abstract on the base) ------

    async def run(self) -> None:
        """Wait for ``stop()``, then drain any pending eval tasks.

        Minimal by design — exists so ``asyncio.gather(pipeline.run(),
        train_and_eval_loop(...))`` works identically for both pipelines.
        """
        await self._stop_event.wait()
        await self._drain_pending_evals()

    async def next_batch(
        self, n: int, *, step_num: int | None = None
    ) -> tuple[list[BatchItem], int]:
        """Run the dynamic-sampling accumulate loop until ``n`` valid
        trajectory groups collected; return ``(batch, train_iter_position)``.

        - Awaits any pending eval tasks BEFORE starting the loop so
          "eval blocks next training step" holds.
        - Over-fetches by ``over_sample_factor / running_pass_rate``
          each round.
        - Filters via ``TrajectoryGroupFilter``.
        - Updates the ``running_pass_rate`` EMA per group inspected
          (first-dropped-task is skipped so it doesn't drag the EMA to 0).
        - Caps at ``max(n * 10, 2 * len(dataset))`` tasks inspected to
          prevent infinite loops when drop rate approaches 100%.
        - Logs per-step pipeline-side metrics (``train/running_pass_rate``,
          ``train/rollout_rounds``, ``train/rollout_tasks_total``,
          ``train/first_round_rollout_tasks``,
          ``train/pipeline/iterator_current``, the ``pipeline/group_drop*``
          filter gauges, and the ``rollout/*`` env metrics aggregated over
          ALL groups inspected — kept AND dropped) via ``ml_logger`` when
          ``step_num`` is provided. ``step_num=None`` disables logging.
        """
        # Drain any pending eval scheduled since the last call so
        # "eval blocks the NEXT training step" holds.
        if self._evaluation_tasks:
            pending = list(self._evaluation_tasks)
            await asyncio.gather(*pending, return_exceptions=True)

        cfg = self._config
        # Fixed over-fetch cushion: aim for slightly more than n valid
        # groups per step so a single sampling round usually suffices
        # (surplus beyond n is discarded). Constant, not a config field.
        over_sample_factor = 1.1
        samples_per_task = cfg.rollout.samples_per_task
        # Resolved once in __init__.
        max_concurrent_samples = self._max_concurrent_samples
        # Reasoning-budget hint forwarded to agent.group_rollout; empty
        # string / None disables.
        thinking_level = getattr(cfg.agent, "thinking_level", None) or None
        max_generation_tokens = getattr(cfg.agent, "max_generation_tokens", None)

        accumulated: list[BatchItem] = []
        all_rollout_metrics: list[dict[str, Any]] = []
        tasks_processed_in_step = 0
        outcome_counts: Counter[GroupFilterOutcome] = Counter()
        num_rollout_rounds = 0
        num_rollout_tasks_total = 0
        first_round_rollout_tasks = 0
        saved_current = self._train_iter.current

        # Infinite-loop guard for near-100% drop rate.
        max_tasks_per_step = max(n * 10, 2 * len(self._train_iter.dataset))
        num_target_samples = int(n * over_sample_factor)

        while len(accumulated) < n:
            if tasks_processed_in_step >= max_tasks_per_step:
                logger.warning(
                    "Dynamic sampling cap reached"
                    f" (processed={tasks_processed_in_step},"
                    f" cap={max_tasks_per_step},"
                    f" accumulated={len(accumulated)})."
                    " Proceeding with what we have."
                )
                break

            target_needed = num_target_samples - len(accumulated)
            est_to_read = math.ceil(target_needed / self._running_pass_rate)
            logger.info(
                f"Dynamic Sampling: Needed={n - len(accumulated)},"
                f" Est. Read={est_to_read},"
                f" Pass Rate={self._running_pass_rate:.3f}"
            )

            tasks = self._read_next_tasks(est_to_read)
            if not tasks:
                logger.warning("No tasks read from dataset.")
                break

            num_rollout_rounds += 1
            num_rollout_tasks_total += len(tasks)
            if num_rollout_rounds == 1:
                first_round_rollout_tasks = len(tasks)

            results = await self._rollout_tasks(
                tasks,
                samples_per_task=samples_per_task,
                thinking_level=thinking_level,
                max_generation_tokens=max_generation_tokens,
                max_concurrent_samples=max_concurrent_samples,
            )

            for group, metrics in results:
                if len(accumulated) >= n:
                    break
                tasks_processed_in_step += 1
                # Aggregate rollout/* env metrics over EVERY inspected
                # group (kept AND dropped) so counters reflect true
                # sampling volume of the step.
                all_rollout_metrics.append(metrics)

                should_keep, outcome = self._classify_group(group)
                outcome_counts[outcome] += 1
                if should_keep:
                    accumulated.append(BatchItem(group=group, metrics=metrics))

                self._tasks_processed_total += 1
                self._update_running_pass_rate(should_keep)

        batch = accumulated[:n]
        # Advance iterator by tasks actually inspected this step (may
        # exceed n due to over-sampling + drops). Wraps at dataset length;
        # this is the resume marker callers log per step.
        self._train_iter.current = (saved_current + tasks_processed_in_step) % len(
            self._train_iter.dataset
        )

        # step_num=None disables per-step logging (test convenience).
        if step_num is not None:
            self._log_step_sampling_metrics(
                step_num=step_num,
                all_rollout_metrics=all_rollout_metrics,
                outcome_counts=outcome_counts,
                tasks_processed_in_step=tasks_processed_in_step,
                num_rollout_rounds=num_rollout_rounds,
                num_rollout_tasks_total=num_rollout_tasks_total,
                first_round_rollout_tasks=first_round_rollout_tasks,
            )

        return batch, self._train_iter.current

    # ---- next_batch sampling-loop helpers ---------------------------------

    def _read_next_tasks(self, est_to_read: int) -> list[TaskData]:
        """Pull up to ``est_to_read`` task_data items from the train iterator.

        Each ``next(self._train_iter)`` yields a (batch-size-1) list; we take
        its single element. Fewer than ``est_to_read`` may be returned if the
        iterator yields empty batches.
        """
        tasks: list[TaskData] = []
        for _ in range(est_to_read):
            batch = next(self._train_iter)
            if batch:
                tasks.append(batch[0])
        return tasks

    async def _rollout_tasks(
        self,
        tasks: list[TaskData],
        *,
        samples_per_task: int,
        thinking_level: str | None,
        max_generation_tokens: int | None,
        max_concurrent_samples: int,
    ) -> list[tuple[Any, dict[str, Any]]]:
        """Run ``group_rollout`` for every task, chunked so at most
        ``max_concurrent_samples // samples_per_task`` groups are in flight at
        once (keeping total in-flight samples ~= ``max_concurrent_samples``).

        Returns the ``(group, metrics)`` tuples in task order.
        """
        results: list[tuple[Any, dict[str, Any]]] = []
        chunk_size = max(1, max_concurrent_samples // samples_per_task)
        # Global dispatch counter across the batch (matches async pipeline's
        # rollout_index semantics for dashboard re-dispatch keying).
        for i in range(0, len(tasks), chunk_size):
            chunk = tasks[i : i + chunk_size]
            chunk_results = await asyncio.gather(
                *[
                    self._agent.group_rollout(
                        p,
                        num_samples=samples_per_task,
                        thinking_level=thinking_level,
                        max_generation_tokens=max_generation_tokens,
                        dispatch_index=i + j,
                    )
                    for j, p in enumerate(chunk)
                ]
            )
            results.extend(chunk_results)
        return results

    def _update_running_pass_rate(self, should_keep: bool) -> None:
        """Fold one group's keep/drop outcome into the ``running_pass_rate`` EMA.

        Assumes ``self._tasks_processed_total`` has already been incremented
        for this group. The very first task of a run, if dropped, is skipped so
        the EMA isn't seeded at 0.0 (which would take many samples to recover
        from). The rate is clamped to ``[0.05, 1.0]``.
        """
        if self._tasks_processed_total == 1 and not should_keep:
            logger.info(
                "Skipping running_pass_rate update for the first"
                " task as it was dropped."
            )
            return
        decay = min(
            0.999,
            (self._tasks_processed_total - 1) / self._tasks_processed_total,
        )
        self._running_pass_rate = self._running_pass_rate * decay + (1 - decay) * float(
            should_keep
        )
        self._running_pass_rate = max(min(self._running_pass_rate, 1.0), 0.05)

    # ---- internal helpers -------------------------------------------------

    def _log_step_sampling_metrics(
        self,
        *,
        step_num: int,
        all_rollout_metrics: list[dict[str, Any]],
        outcome_counts: Counter[GroupFilterOutcome],
        tasks_processed_in_step: int,
        num_rollout_rounds: int,
        num_rollout_tasks_total: int,
        first_round_rollout_tasks: int,
    ) -> None:
        """Write the pipeline-side per-step sampling metrics.

        Emits aggregated env/* metrics + pipeline running counters at
        their natural prefix. Trainer-owned metrics (``train/loss`` etc.)
        are logged separately by ``train_and_eval_loop``.
        """
        # Strip pipeline bookkeeping keys before spec-driven aggregation.
        filtered_metrics = [
            {k: v for k, v in m.items() if k not in self._NON_METRIC_KEYS}
            for m in all_rollout_metrics
        ]
        step_metrics: dict[str, Any] = {}
        step_metrics.update(aggregate_by_spec(filtered_metrics))

        n_dropped = sum(n for o, n in outcome_counts.items() if o.is_drop)
        step_metrics["pipeline/group_drop_ratio"] = (
            n_dropped / tasks_processed_in_step if tasks_processed_in_step else 0.0
        )
        step_metrics.update(
            group_filter_metrics(outcome_counts, tasks_processed_in_step)
        )
        step_metrics["train/running_pass_rate"] = self._running_pass_rate
        step_metrics["train/rollout_rounds"] = num_rollout_rounds
        step_metrics["train/rollout_tasks_total"] = num_rollout_tasks_total
        step_metrics["train/first_round_rollout_tasks"] = first_round_rollout_tasks
        step_metrics["train/pipeline/iterator_current"] = self._train_iter.current

        self._ml_logger.log_metrics(step_metrics, step=step_num)
