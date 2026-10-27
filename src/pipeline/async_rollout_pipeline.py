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

"""Concurrent async rollout pipeline + its config.

- ``AsyncRolloutConfig`` — frozen dataclass that selects the async
  pipeline; ``make()`` constructs an ``AsyncRolloutPipeline``.
- ``AsyncRolloutPipeline`` — concurrent producer + bounded buffer +
  backpressure gate + periodic progress prober + eval scheduling. Implements the
  ``RolloutPipeline`` protocol; off-policy staleness bounded by
  ``config.rollout.max_pending_rollout_tasks``.

The shared rollout machinery (``InOrderBuffer``, ``DispatchGate``,
``RolloutDispatcher``, ``MetricsAggregator``) still lives in
``pipeline.rollout_pipeline`` and is imported here.
``SyncRolloutPipeline`` lives in ``pipeline.sync_rollout_pipeline``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Any

from core.agent import Agent
from core.atif_trajectory import GroupFilterOutcome
from core.rollout_protocol import BatchItem
from core.dataset import JsonlDataset
from core.run_config import RolloutConfig, RunConfig
from core.task_data import TaskData
from pipeline.rollout_pipeline import (
    BaseRolloutPipeline,
    DispatchGate,
    InOrderBuffer,
    MetricsAggregator,
    RolloutDispatcher,
    group_filter_metrics,
)
from utils.logger import Logger

logger = logging.getLogger(__name__)

# How often the background prober logs a rollout-progress line. 120s is
# frequent enough to watch a run come alive / spot a stall, without spamming.
_PROBER_INTERVAL_SECONDS = 120

__all__ = [
    "AsyncRolloutConfig",
    "AsyncRolloutPipeline",
]


@dataclass(frozen=True)
class AsyncRolloutConfig(RolloutConfig):
    """Async rollout config. Selects ``AsyncRolloutPipeline`` via ``make()``.

    ``max_pending_rollout_tasks`` is the backpressure / off-policy-staleness
    bound, counted in TASKS (one per ``rollout_index`` /
    ``TrajectoryGroup``), NOT in rollout batches: the producer refuses to
    start a task whose ``rollout_index`` would sit >= this many ahead of
    the trainer's in-order release cursor, and the ``InOrderBuffer`` holds
    at most this many pending task groups. Must be
    >= ``train.tasks_per_train_step`` (else ``InOrderBuffer.get`` deadlocks).
    """

    max_pending_rollout_tasks: int = 256

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
    ) -> AsyncRolloutPipeline:
        return AsyncRolloutPipeline(
            agent=agent,
            config=run_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            eval_agent=eval_agent,
            ml_logger=ml_logger,
            resume_train_iter_position=resume_train_iter_position,
        )


class AsyncRolloutPipeline(BaseRolloutPipeline):
    """Async rollout pipeline: concurrent producer + bounded buffer +
    backpressure gate + per-batch metrics aggregator + progress prober + eval
    scheduling. Implements the ``RolloutPipeline`` protocol.

    Off-policy staleness bounded by ``config.rollout.max_pending_rollout_tasks``.
    Producer runs concurrently with trainer inside
    ``asyncio.gather(pipeline.run(), train_and_eval_loop(pipeline, ...))``.
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
        assert (
            config.train.tasks_per_train_step
            <= config.rollout.max_pending_rollout_tasks
        ), (
            "tasks_per_train_step must be <= max_pending_rollout_tasks"
            " to prevent InOrderBuffer.get() deadlock"
        )

        self._buffer = InOrderBuffer(
            maxsize=config.rollout.max_pending_rollout_tasks,
        )
        self._gate = DispatchGate(
            get_cursor=lambda: self._buffer.cursor,
            cursor_event=self._buffer.cursor_event,
            maxsize=config.rollout.max_pending_rollout_tasks,
            # Exclude dropped groups from the look-ahead window: they hold
            # no buffer slot and add no staleness, so counting them would
            # let a drop burst starve a same-size consumer batch (mid-run
            # freeze). See DispatchGate / InOrderBuffer.dropped_ahead_count.
            get_dropped_ahead=lambda: self._buffer.dropped_ahead_count,
        )
        self._dispatcher = RolloutDispatcher(
            agent=agent,
            buffer=self._buffer,
            semaphore=self._semaphore,
            config=config,
            trajectory_callback=getattr(agent, "_trajectory_callback", None),
        )
        self._aggregator = MetricsAggregator(
            buffer=self._buffer,
            ml_logger=ml_logger,
            config=config,
        )

        # Producer state (mutated only by _run_producer).
        self._train_iter = train_dataset.build_iter(
            batch_size=config.train.tasks_per_train_step,
            resume_from_position=resume_train_iter_position,
        )
        # Bookkeeping counters — always start fresh across restarts.
        self._rollout_batch = 1
        self._next_rollout_index = 0
        self._active_dispatches = 0
        # Cumulative rollouts that have finished a dispatch (success OR
        # failure); paired with _next_rollout_index (cumulative dispatched)
        # and _active_dispatches (in-flight) for the prober's progress line.
        self._completed_dispatches = 0
        self._failed_dispatches = 0
        self._dropped_dispatches = 0
        # Counter values at the previous ``sample_pipeline_gauges`` call;
        # the delta between them is one train step's worth of finished
        # rollouts (see that method).
        self._gauged_completed_dispatches = 0
        self._gauged_dropped_dispatches = 0
        # Same idea for the dispatcher's per-outcome tally. Counter
        # subtraction drops non-positive entries, which is safe because
        # these only ever grow.
        self._gauged_outcome_counts: Counter[GroupFilterOutcome] = Counter()
        # Per-dispatch start times for wall-time bucketing in the prober.
        # rollout_index -> monotonic start time. Inserted on dispatch entry,
        # removed on exit (including exception paths).
        self._dispatch_start_times: dict[int, float] = {}

    # ---- public methods (run + next_batch are abstract on the base) ------

    async def run(self) -> None:
        """Own the pipeline's runtime lifecycle: spawn producer + prober,
        run until ``stop()`` is called, drain pending evals, return."""
        producer_task = asyncio.create_task(self._run_producer())
        prober_task = asyncio.create_task(self._run_prober())
        try:
            await producer_task
        finally:
            # Prober is blocked on asyncio.sleep(_PROBER_INTERVAL_SECONDS);
            # explicit cancel so shutdown doesn't wait up to that long.
            prober_task.cancel()
            try:
                await prober_task
            except asyncio.CancelledError:
                pass
            await self._drain_pending_evals()

    async def next_batch(
        self, n: int, *, step_num: int | None = None
    ) -> tuple[list[BatchItem], int]:
        """Return ``(batch, train_iter_position)``.

        Blocks until N items are available; returns them in strict
        rollout_index order. Advances the buffer's release cursor,
        unblocking the producer's ``wait_until_schedulable`` checks.

        ``step_num`` is ignored by async (accepted only for ``RolloutPipeline``
        protocol conformance).
        """
        del step_num  # unused; see docstring
        batch = await self._buffer.get(n)
        return batch, self._train_iter.current

    def sample_pipeline_gauges(self) -> dict[str, float]:
        """``pipeline/buffer_utilization`` + the group-filter gauges.

        Utilization is read at the caller's instant — the trainer's, just
        before it blocks on ``next_batch``. Reading it anywhere inside the
        producer instead (as this used to, right after a rollout batch was
        put) samples a local maximum and answers a question nobody asked.

        The drop ratio is a DELTA over the rollouts that finished since the
        previous call, not a cumulative rate: cumulative would flatten out
        and hide a late drop burst. It cannot be recomputed from the
        trainer's batch, because a dropped group is by construction absent
        from it. Numerator/denominator keep ``MetricsAggregator``'s old
        semantics — a rollout whose dispatch raised counts in the
        denominator only, since it was never filter-rejected. The key is
        omitted when nothing finished in the window, so a quiet step reads
        as "no data" rather than as "no drops".

        ``group_drop_reason/*`` and ``group_zero_variance_ratio`` delta the
        dispatcher's per-outcome tally over that same window. They are
        denominated in classified groups rather than finished dispatches,
        so a window containing a raised dispatch makes the three reason
        shares sum to slightly less than ``group_drop_ratio`` — the gap is
        exactly the dispatches that never reached the filter.
        """
        gauges: dict[str, float] = {}
        maxsize = self._buffer.maxsize
        if maxsize > 0:
            gauges["pipeline/buffer_utilization"] = self._buffer.qsize() / maxsize

        completed = self._completed_dispatches - self._gauged_completed_dispatches
        dropped = self._dropped_dispatches - self._gauged_dropped_dispatches
        self._gauged_completed_dispatches = self._completed_dispatches
        self._gauged_dropped_dispatches = self._dropped_dispatches
        if completed > 0:
            gauges["pipeline/group_drop_ratio"] = dropped / completed

        outcomes = self._dispatcher.outcome_counts
        window = outcomes - self._gauged_outcome_counts
        self._gauged_outcome_counts = outcomes.copy()
        gauges.update(group_filter_metrics(window, sum(window.values())))
        return gauges

    # ---- internal coroutines ---------------------------------------------

    async def _dispatch(
        self,
        task_data: TaskData,
        rollout_batch: int,
        task_idx: int,
        rollout_index: int,
    ) -> dict[str, Any]:
        """Wrap ``_dispatcher.dispatch`` in a try/finally that bumps the
        in-flight counter on entry and decrements it on exit (including
        on exception). Called only by ``_run_producer``."""
        self._active_dispatches += 1
        self._dispatch_start_times[rollout_index] = time.monotonic()
        failed = False
        try:
            result = await self._dispatcher.dispatch(
                task_data, rollout_batch, task_idx, rollout_index
            )
            if result.get("dropped"):
                self._dropped_dispatches += 1
            return result
        except BaseException:
            failed = True
            raise
        finally:
            self._active_dispatches -= 1
            self._completed_dispatches += 1
            self._dispatch_start_times.pop(rollout_index, None)
            if failed:
                self._failed_dispatches += 1

    async def _await_or_stop(self, coro: Coroutine[Any, Any, Any]) -> bool:
        """Race ``coro`` against ``self._stop_event.wait()``.

        Returns ``True`` if ``coro`` completed normally; ``False`` if
        ``_stop_event`` fired first (in which case ``coro`` was
        cancelled). Callers use the ``False`` return to break out of
        the producer loop promptly instead of blocking indefinitely on
        a condition (buffer-cursor / semaphore-permit) that will never
        be signalled once the trainer has stopped consuming.

        The bug this fixes: without this race, ``_run_producer``'s two
        awaits (``wait_until_schedulable`` and ``semaphore.acquire``)
        wait on primitives that only advance when ``next_batch`` is
        called or a dispatched rollout completes. After ``stop()`` is
        called from ``train_and_eval_loop``'s finally, no more
        ``next_batch`` calls arrive, so a producer parked inside
        either await never wakes; ``run()``'s ``await producer_task``
        then hangs, and ``asyncio.gather(pipeline.run(), ...)``
        deadlocks -- with the event loop watching only its own
        self-pipe wakeup fd (i.e. nothing external will ever kick it).
        """
        work = asyncio.ensure_future(coro)
        stop = asyncio.ensure_future(self._stop_event.wait())
        try:
            done, _ = await asyncio.wait(
                {work, stop}, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            # Whichever finished second (if any) is still pending; make
            # sure neither future leaks. asyncio.wait doesn't cancel
            # pending futures on FIRST_COMPLETED.
            if not work.done():
                work.cancel()
            if not stop.done():
                stop.cancel()
        return work in done and work.exception() is None

    async def _run_producer(self) -> None:
        """Outer sampling loop.

        Per iteration: pull the next task_batch from train_iter
        (wrap+reset on StopIteration); for each task acquire
        backpressure + semaphore, spawn _dispatch, and register with
        MetricsAggregator.log_batch. Exits when stop() is called.

        The two per-task awaits (``wait_until_schedulable`` and
        ``semaphore.acquire``) are wrapped in ``_await_or_stop`` so
        that ``stop()`` unblocks them promptly even when the trainer
        has stopped calling ``next_batch`` (which would otherwise
        prevent the cursor from advancing and the semaphore permits
        from being released, leaving the producer parked forever). On
        stop-triggered exit we return early WITHOUT dispatching any
        further tasks so half-scheduled batches don't leak in-flight
        RPCs past shutdown.
        """
        while not self._stop_event.is_set():
            start_idx, task_batch = self._next_task_batch()

            logger.info(
                "\n========== Sampling Iterator Loop"
                f" (rollout_batch={self._rollout_batch},"
                f" tasks={len(task_batch)},"
                f" samples={len(task_batch) * self._config.rollout.samples_per_task})"
                " =========="
            )

            pending = []
            dataset_len = len(self._train_iter.dataset)
            for i, task_data in enumerate(task_batch):
                task_idx = (start_idx + i) % dataset_len
                rollout_index = self._next_rollout_index
                self._next_rollout_index += 1
                # Backpressure: don't START a group whose rollout_index
                # would sit >= maxsize ahead of the release cursor.
                if not await self._await_or_stop(
                    self._gate.wait_until_schedulable(rollout_index)
                ):
                    return
                # Acquire samples_per_task semaphore permits, one at a
                # time; each acquire also races _stop_event so shutdown
                # is prompt even mid-acquire. If we bail out after
                # having acquired some permits, release them so eval
                # tasks scheduled during shutdown-drain aren't starved.
                acquired = 0
                for _ in range(self._config.rollout.samples_per_task):
                    if not await self._await_or_stop(self._semaphore.acquire()):
                        for _ in range(acquired):
                            self._semaphore.release()
                        return
                    acquired += 1

                task = asyncio.create_task(
                    self._dispatch(
                        task_data,
                        self._rollout_batch,
                        task_idx,
                        rollout_index,
                    )
                )
                pending.append(task)

            # TRACKED, not fire-and-forget: this task owns the only copy of
            # the batch's metrics while it waits on every rollout in it, and
            # the loop holds a bare `create_task` only weakly.
            self.track_metrics_task(
                asyncio.create_task(
                    self._aggregator.log_batch(pending, self._rollout_batch)
                )
            )
            self._rollout_batch += 1

    def _next_task_batch(self) -> tuple[int, list[TaskData]]:
        """Return (start_idx, task_batch); wrap+reset on StopIteration."""
        start_idx = self._train_iter.current
        try:
            task_batch = next(self._train_iter)
        except StopIteration:
            self._train_iter.reset()
            start_idx = self._train_iter.current
            task_batch = next(self._train_iter)
        return start_idx, task_batch

    async def _run_prober(self) -> None:
        """Periodically log rollout progress (dispatched/active/completed/
        failed) + buffer utilization + active wall-time buckets, every
        ``_PROBER_INTERVAL_SECONDS``. Exits when stop() is called."""
        while not self._stop_event.is_set():
            await asyncio.sleep(_PROBER_INTERVAL_SECONDS)
            qsize = self._buffer.qsize()
            maxsize = self._buffer.maxsize
            utilization = qsize / maxsize if maxsize > 0 else 0

            # Wall-time buckets for active dispatches.
            now = time.monotonic()
            buckets: dict[str, int] = {
                "<1m": 0,
                "1-5m": 0,
                "5-15m": 0,
                "15-30m": 0,
                "30m+": 0,
            }
            for t_start in self._dispatch_start_times.values():
                elapsed_m = (now - t_start) / 60.0
                if elapsed_m < 1:
                    buckets["<1m"] += 1
                elif elapsed_m < 5:
                    buckets["1-5m"] += 1
                elif elapsed_m < 15:
                    buckets["5-15m"] += 1
                elif elapsed_m < 30:
                    buckets["15-30m"] += 1
                else:
                    buckets["30m+"] += 1
            bucket_str = " ".join(f"{k}:{v}" for k, v in buckets.items() if v)

            # Per-rollout step counts (duck-typed: only AgentSolverAgent
            # exposes active_step_counts; other agents silently skip).
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
                f"[Prober] Rollouts: dispatched={self._next_rollout_index}"
                f" active={self._active_dispatches}"
                f" completed={self._completed_dispatches}"
                f" (failed={self._failed_dispatches})"
                f" | Buffer: {qsize}/{maxsize}"
                f" ({utilization:.2%} full)"
                f" | Active wall-time: [{bucket_str or '-'}]"
                f" | Active steps: [{step_bucket_str}]"
            )
