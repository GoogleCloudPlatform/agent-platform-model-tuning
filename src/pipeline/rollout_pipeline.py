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

"""Shared rollout machinery: base class + buffer + backpressure + dispatch + eval.

This module provides:

- ``BaseRolloutPipeline`` — ABC that both concrete pipelines inherit.
  Owns the shared constructor, ``stop()``, ``schedule_eval()``, and the
  ``_classify_group()`` helper. Declares ``run()`` and ``next_batch()``
  as abstract methods.
- ``InOrderBuffer``, ``DispatchGate``, ``RolloutDispatcher``,
  ``MetricsAggregator`` — async-pipeline-specific machinery (not used
  by sync).
- Eval helpers (``_collect_eval_tasks``, ``_drain_pending_evals``,
  and the eval-scheduling logic) are methods on ``BaseRolloutPipeline``.

The pipeline abstractions (``BatchItem``, ``RolloutPipeline`` Protocol)
live in ``core.rollout_protocol``. The composed run config is
``core.run_config.RunConfig``.

Both concrete pipelines live in their own modules:

- ``AsyncRolloutPipeline`` (``pipeline.async_rollout_pipeline``): concurrent
  producer + bounded buffer + backpressure gate + 120s prober.
- ``SyncRolloutPipeline`` (``pipeline.sync_rollout_pipeline``): inline
  dynamic-sampling accumulate loop, no concurrent producer.
"""

from __future__ import annotations

import abc
import asyncio
import dataclasses
import logging
from collections import Counter
from collections.abc import Mapping
from typing import Any, Callable

from core.agent import Agent
from core.atif_trajectory import GROUP_DROP_OUTCOMES, GroupFilterOutcome
from core.rollout_protocol import BatchItem
from core.dataset import JsonlDataset
from core.metric_spec import aggregate_by_spec
from core.protocols import ProgressObserver
from core.run_config import RunConfig
from core.task_data import TaskData
from pipeline.eval_only import EvalPipeline
from utils.logger import Logger

logger = logging.getLogger(__name__)

__all__: list[str] = []


def group_filter_metrics(
    outcome_counts: Mapping[GroupFilterOutcome, int], total: int
) -> dict[str, float]:
    """Per-reason breakdown of the trajectory-group filter for one step.

    ``total`` is how many groups the filter classified in the window;
    every value is a fraction of it, so the three
    ``group_drop_reason/*`` shares sum to ``group_drop_ratio``. Both
    pipelines route through here so async and sync emit the same key set
    over the same denominator convention — they differ only in what the
    window is (rollouts finished since the last gauge sample vs tasks
    inspected by this step's dynamic-sampling loop).

    ``group_zero_variance_ratio`` counts zero-variance groups whether or
    not they were then dropped. It is the one to read when asking whether
    zero-variance filtering earns its keep on a recipe: at
    ``zero_variance_keep_prob=1.0`` nothing is ever dropped for it, so the
    drop share sits at 0 and says nothing, while this still shows how
    often groups tie.

    Every key is emitted on every step, zeros included — an absent series
    cannot distinguish "that reason never fires" from "not measured",
    which is the confusion this exists to end. Returns ``{}`` for an empty
    window, since a ratio over nothing is not zero (matching
    ``group_drop_ratio``'s own quiet-step semantics).
    """
    if total <= 0:
        return {}
    n_zero_variance = sum(
        n for o, n in outcome_counts.items() if GroupFilterOutcome(o).is_zero_variance
    )
    metrics = {"pipeline/group_zero_variance_ratio": n_zero_variance / total}
    for outcome in GROUP_DROP_OUTCOMES:
        metrics[f"pipeline/group_drop_reason/{outcome.value}"] = (
            outcome_counts.get(outcome, 0) / total
        )
    return metrics


# ---------------------------------------------------------------------------
# Base class — shared init / stop / schedule_eval / classify_group
# ---------------------------------------------------------------------------


class BaseRolloutPipeline(abc.ABC):
    """Implementation base for both rollout pipelines.

    Owns the shared constructor (agent, config, iterators, semaphore,
    stop-event, eval state), ``stop()``, ``schedule_eval()``, and the
    trajectory-group filter helper ``_classify_group()``.

    ``run()`` and ``next_batch()`` are abstract — their implementations
    differ fundamentally between async (concurrent producer) and sync
    (inline accumulate loop).

    This is NOT the ``RolloutPipeline`` Protocol the trainer talks to
    (that lives in ``core.rollout_protocol``). Both concrete
    subclasses duck-type that Protocol; this ABC provides the shared
    implementation they inherit.
    """

    #: Keys stripped from per-rollout metrics before spec-driven
    #: aggregation. Both pipelines filter the same bookkeeping keys.
    _NON_METRIC_KEYS: frozenset[str] = frozenset({"dropped", "task_idx"})

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
        self._agent = agent
        self._config = config
        self._ml_logger = ml_logger

        # Single semaphore shared by training + eval rollouts.
        self._semaphore = asyncio.Semaphore(config.rollout.max_concurrent_samples)
        self._stop_event = asyncio.Event()

        # Eval pipeline — pre-wired with the shared semaphore so
        # training + eval compete for one concurrency budget.
        eval_iter = (
            eval_dataset.build_iter(max_items=config.eval.max_eval_items, repeat=False)
            if eval_dataset is not None
            else None
        )
        self._eval_pipeline = (
            EvalPipeline(
                agent=eval_agent or agent,
                config=config,
                ml_logger=ml_logger,
                eval_iter=eval_iter,
                semaphore=self._semaphore,
            )
            if eval_iter is not None
            else None
        )
        self._evaluation_lock = asyncio.Lock()
        self._evaluation_tasks: set[asyncio.Task[None]] = set()
        # Background metrics aggregation, tracked so it is not collected:
        # the event loop keeps only a WEAK reference to a bare
        # `create_task`, so an unreferenced one can be garbage-collected
        # mid-flight. Unlike the eval tasks, the shutdown drain does NOT
        # await these (see `track_metrics_task`).
        # `log_batch` parks on `gather(*rollout_tasks)` for a whole batch,
        # which is a long window in which to be collectable, and it holds
        # the only copy of that batch's metrics while it waits.
        self._metrics_tasks: set[asyncio.Task[None]] = set()
        # Set by ``set_progress_observer``; forwarded to the one-off
        # EvalPipelines the checkpoint sweeps build.
        self._progress_observer: ProgressObserver | None = None

        # Trajectory-group filter (built once from the config).
        self._trajectory_group_filter = config.rollout.trajectory_group_filter.make()

    # ---- concrete shared methods ------------------------------------------

    def stop(self) -> None:
        """Signal the pipeline to shut down. Idempotent."""
        self._stop_event.set()

    def sample_pipeline_gauges(self) -> dict[str, float]:
        """No pipeline-internal gauges by default (see the protocol).

        Sync has no inter-step buffer to report occupancy for, and
        already logs its own ``pipeline/group_drop_ratio`` from inside
        its accumulate loop, where the groups it dropped are still in
        scope. Async overrides this.
        """
        return {}

    def schedule_eval(self, step_val: int) -> None:
        """Spawn a background eval sweep. Non-blocking.

        Delegates to the pre-wired ``EvalPipeline`` under
        ``_evaluation_lock``, and registers the task so the pipeline's
        shutdown path can drain it.
        """
        if self._eval_pipeline is None:
            return

        async def _locked_eval():
            async with self._evaluation_lock:
                await self._eval_pipeline.run(step_val)

        task = asyncio.create_task(_locked_eval())
        self._evaluation_tasks.add(task)
        task.add_done_callback(self._evaluation_tasks.discard)

    def schedule_checkpoint_eval(self, step_val: int, endpoint: str) -> None:
        """Spawn a background eval sweep against a deployed checkpoint endpoint.

        Non-blocking, fire-and-forget (mirrors ``schedule_eval``). Builds a
        one-off ``EvalPipeline`` bound to a ``GeminiApiBackendConfig`` whose
        project/location come from the ENDPOINT resource path (not the
        training backend -- the checkpoint endpoint may live in a different
        location, e.g. ``locations/us``). The sweep's agent uses a raised
        generation budget (``eval.checkpoint_eval_max_generation_tokens``)
        and logs under the ``eval_ckpt/*`` prefix so it never collides with
        the live-sampler ``eval/*`` at the same step.

        A zero-result sweep (e.g. an unreachable endpoint) logs a WARN and
        writes NO metrics, so a failed sweep can't masquerade as
        ``eval_ckpt/accuracy=0``.
        """
        if self._eval_pipeline is None:
            # No eval dataset configured -> nothing to sweep.
            return

        from backends.backend_config import GeminiApiBackendConfig
        from tools.eval_all_checkpoints import _parse_endpoint_resource

        try:
            parts = _parse_endpoint_resource(endpoint)
        except ValueError:
            logger.warning(
                "Checkpoint eval skipped at step %d: unparseable endpoint %r.",
                step_val,
                endpoint,
            )
            return

        sweep_backend = GeminiApiBackendConfig(
            project=parts["project"],
            location=parts["location"],
            endpoint_id=parts["endpoint_id"],
            service_account=getattr(self._config.backend, "service_account", ""),
        )
        self._schedule_sweep(step_val, sweep_backend, f"endpoint {endpoint}")

    def schedule_base_model_eval(self, step_val: int = 0) -> None:
        """Spawn a background ``eval_ckpt/*`` sweep against the BASE model.

        The step-0 twin of ``schedule_checkpoint_eval``: before any
        gradient update there is no checkpoint endpoint, so the sweep
        targets the publisher (untuned) model instead — same dataset,
        same generation budget, same ``eval_ckpt/*`` prefix, so step 0
        is directly comparable with the later checkpoint sweeps.

        No-op (with a WARN) when the training backend does not name a
        base model, e.g. an eval-only backend.

        Unlike the checkpoint sweep, this one does NOT inherit the
        training backend's region: a deployed checkpoint endpoint is a
        regional resource, but a publisher model is not served from
        every region. Inheriting ``us-central1`` produced

            HTTP 404: Publisher model `projects/P/locations/us-central1/
            publishers/google/models/gemini-3.5-flash` was not found or
            your project does not have access to it.

        so the sweep uses the publisher realm
        (``GeminiApiBackendConfig.location`` default, ``global``) --
        the same target the standalone gemini_api eval configs use.
        """
        if self._eval_pipeline is None:
            return

        from backends.backend_config import GeminiApiBackendConfig

        base_model = getattr(self._config.backend, "vertex_base_model", "")
        if not base_model:
            logger.warning(
                "Base-model eval skipped at step %d: backend %r names no "
                "base model.",
                step_val,
                type(self._config.backend).__name__,
            )
            return

        sweep_backend = GeminiApiBackendConfig(
            model=base_model,
            project=self._config.backend.project,
            # location intentionally left at its default (``global``) --
            # see the docstring; the training backend's region is wrong
            # for a publisher model.
            # "" -> target the publisher (base) model, not an endpoint.
            endpoint_id="",
            service_account=getattr(self._config.backend, "service_account", ""),
        )
        self._schedule_sweep(step_val, sweep_backend, f"base model {base_model}")

    def _schedule_sweep(
        self, step_val: int, sweep_backend: Any, target_label: str
    ) -> None:
        """Fire-and-forget one ``eval_ckpt/*`` sweep against ``sweep_backend``.

        Shared body of ``schedule_checkpoint_eval`` (a deployed checkpoint
        endpoint) and ``schedule_base_model_eval`` (the step-0 base model).
        """
        sweep_agent_config = dataclasses.replace(
            self._config.agent,
            max_generation_tokens=(
                self._config.eval.checkpoint_eval_max_generation_tokens
            ),
        )
        # Reuse the training agent's trajectory callback (ATIF disk writer
        # + progress sink) so the sweep's trajectories are persisted and
        # surface in experiment tracking exactly like the eval/* ones.
        # A fresh agent with no callback runs them into the void.
        # Same getattr pattern the async dispatcher uses.
        trajectory_callback = getattr(self._agent, "_trajectory_callback", None)
        # ...and the training job id, NOT the sweep bundle's. The sweep
        # backend is an eval-only GeminiApiBackendConfig whose ``job_id``
        # is the endpoint id (checkpoint sweep) or the model name (base
        # model), so taking it from the bundle made one run's artifacts
        # self-describe under three different ids -- e.g. trial dirs
        # named ``<task>-gemini-3.5-flash-s0000-...`` next to
        # ``<task>-<tuning_job>-...`` from the training rollouts. The
        # job id identifies the RUN; the sweep's target is already
        # identified by the step segment.
        training_job_id = getattr(self._agent, "tuning_job_id", "")

        async def _run_checkpoint_sweep() -> None:
            async with self._evaluation_lock:
                bundle = await sweep_backend.make()
                try:
                    agent = sweep_agent_config.make(
                        sampler=bundle.sampler,
                        job_id=training_job_id or bundle.job_id,
                        reward_fn=self._config.reward.make(),
                        atif_writer=trajectory_callback,
                    )
                    eval_dataset = self._config.task.make_eval()
                    sweep_config = dataclasses.replace(
                        self._config,
                        agent=sweep_agent_config,
                        backend=sweep_backend,
                    )
                    sweep = EvalPipeline(
                        agent=agent,
                        config=sweep_config,
                        ml_logger=self._ml_logger,
                        eval_dataset=eval_dataset,
                        eval_source="checkpoint",
                    )
                    # The observer stamps eval trajectories with the step
                    # from ``on_eval_started``; without it the
                    # sweep's trajectories inherit a stale eval step (or
                    # none at all when the live-sampler eval is off).
                    sweep._progress_observer = self._progress_observer
                    metrics = await sweep.run(step_val, metric_prefix="eval_ckpt")
                    if not metrics:
                        logger.warning(
                            "Checkpoint eval at step %d produced no results"
                            " (%s may be unavailable).",
                            step_val,
                            target_label,
                        )
                finally:
                    await bundle.aclose()

        task = asyncio.create_task(_run_checkpoint_sweep())
        self._evaluation_tasks.add(task)
        task.add_done_callback(self._evaluation_tasks.discard)

    def set_progress_observer(self, observer: ProgressObserver | None) -> None:
        """Attach a progress observer to the eval sub-pipeline.

        Also retained on the pipeline so the one-off checkpoint-eval
        sweeps (which build their own ``EvalPipeline``) can attach it.
        """
        self._progress_observer = observer
        if self._eval_pipeline is not None:
            self._eval_pipeline._progress_observer = observer

    def track_metrics_task(self, task: asyncio.Task[None]) -> None:
        """Keep a strong reference to a background metrics task.

        An untracked ``create_task`` running ``log_batch`` is destroyed
        when the loop closes -- observed live as ``Task was destroyed
        but it is pending! ... MetricsAggregator.log_batch()`` -- and
        that batch's metrics are simply lost, with the run reporting
        no error. Holding the reference (plus the done-callback that
        drops it once the task finishes cleanly) is enough to stop
        the GC-drop without blocking shutdown.

        Deliberately NOT drained on shutdown. ``log_batch`` parks on
        ``gather(*rollout_tasks)`` for a whole batch, so draining it
        would make ``run_training``'s ``finally: aclose()`` (which
        cancels the tuning job) wait for the longest in-flight rollout
        -- up to ``poll_deadline_s`` (10 h) for a hung LRO. One rollout
        stall would then keep a tuning job billing after the run was
        supposed to end. If shutdown starts with a metrics task
        in-flight, its rollouts are cancelled by pipeline teardown as
        before, its gather raises inside, that batch's metrics are lost
        -- same as main -- but shutdown is bounded again.
        """
        self._metrics_tasks.add(task)
        task.add_done_callback(self._metrics_tasks.discard)

    async def _drain_pending_evals(self) -> None:
        """Await still-running EVAL tasks on pipeline shutdown.

        Eval tasks are bounded (a single eval batch runs to completion
        against a frozen sampler and returns), so draining them does not
        risk the shutdown-hang the metrics-task drain would -- see
        ``track_metrics_task`` for why metrics tasks are NOT drained.

        ``return_exceptions=True`` -- a failing eval shouldn't mask shutdown.
        """
        if self._evaluation_tasks:
            logger.info(
                "Waiting for %d pending evaluation task(s) to complete...",
                len(self._evaluation_tasks),
            )
            await asyncio.gather(
                *self._evaluation_tasks, return_exceptions=True
            )

    def _classify_group(self, group: Any) -> tuple[bool, GroupFilterOutcome]:
        """Return ``(passes_filter, outcome)`` for ``group``.

        On a drop, logs the reason: at WARNING for the two structural
        failures that usually indicate a real problem, else at DEBUG.
        """
        should_keep, outcome = self._trajectory_group_filter(group)
        if not should_keep:
            log_fn = logger.warning if outcome.is_structural_failure else logger.debug
            log_fn(f"Dropping group: {outcome.value}.")
        return should_keep, outcome

    # ---- abstract methods -------------------------------------------------

    @abc.abstractmethod
    async def run(self) -> None:
        """Pipeline runtime lifecycle. Concrete shape differs per pipeline."""
        ...

    @abc.abstractmethod
    async def next_batch(
        self, n: int, *, step_num: int | None = None
    ) -> tuple[list[BatchItem], int]:
        """Return ``(batch, train_iter_position)``."""
        ...


# ---------------------------------------------------------------------------
# Async-only machinery (not used by SyncRolloutPipeline)
# ---------------------------------------------------------------------------


class InOrderBuffer:
    """Bounded async storage that releases items in strict rollout_index order.

    Producers `put(item, rollout_index=N)` with the index they were
    dispatched at. Consumers `get(n)` receive items strictly in
    rollout_index order (skipping indices that were `mark_dropped`).

    Exposes `cursor` and `cursor_event` as read-only observation points
    so flow-control layers (see `DispatchGate`) can react to release
    progress without reaching into internal state.
    """

    def __init__(
        self,
        maxsize: int = 100,
        start_rollout_index: int = 0,
    ):
        self._maxsize = maxsize
        # rollout_index -> item, awaiting in-order release.
        self._store: dict[int, BatchItem] = {}
        # rollout indices filtered out before reaching the buffer.
        self._dropped: set[int] = set()
        # ``start_rollout_index`` is kept as a knob for tests exercising
        # resume-alignment invariants; production callers pass 0.
        self._next_rollout_index = start_rollout_index
        self._cond = asyncio.Condition()
        # Edge-triggered notification of cursor advances (set + immediately
        # cleared) so external observers wake and re-check the cursor
        # value rather than seeing a stale True.
        self._cursor_event = asyncio.Event()

    @property
    def maxsize(self) -> int:
        return self._maxsize

    @property
    def cursor(self) -> int:
        """Read-only view of the next rollout index to release."""
        return self._next_rollout_index

    @property
    def cursor_event(self) -> asyncio.Event:
        """Fires (set + immediately cleared) whenever the cursor advances."""
        return self._cursor_event

    @property
    def dropped_ahead_count(self) -> int:
        """Number of dropped indices at or above the release cursor.

        All live ``_dropped`` entries satisfy ``index >= cursor`` (leading
        drops are folded into the cursor and out-of-window drops are pruned
        on every advance), and only already-dispatched indices are ever
        dropped. So for the producer's next-to-dispatch index ``k`` this is
        exactly the count of drops in ``[cursor, k)``.

        ``DispatchGate`` subtracts this from the raw dispatch-index
        distance so that dropped groups -- which occupy no real buffer slot
        and add no off-policy staleness -- do not consume the look-ahead
        window. Without this correction a burst of drops (e.g. a 503 storm
        tripping the transient-failure gate) fills the window with
        tombstones and starves a same-size consumer batch: the buffer
        freezes with only a handful of valid items and no in-flight
        rollouts. See ``DispatchGate.wait_until_schedulable``.
        """
        return len(self._dropped)

    def _size(self) -> int:
        return len(self._store)

    def qsize(self) -> int:
        """Return the current number of items in the buffer.

        Note: In an async context, this is an approximation.
        """
        return self._size()

    async def put(self, item: BatchItem, rollout_index: int) -> int:
        """Put a single item into the buffer, indexed by rollout_index.

        Blocks if the buffer is full. Returns the current number of items.

        Raises:
          ValueError: if `rollout_index` is already stored, is already
            marked dropped, or has already been released. Fail fast —
            each of these would silently lose data.
        """
        async with self._cond:
            self._check_rollout_index_unused(rollout_index, op="put")
            while self._size() >= self._maxsize:
                await self._cond.wait()
                # Re-check after waking: another producer may have claimed
                # this index while we slept.
                self._check_rollout_index_unused(rollout_index, op="put")

            self._store[rollout_index] = item
            self._cond.notify_all()
            return self._size()

    async def mark_dropped(self, rollout_index: int) -> None:
        """Record that a dispatched group was filtered out, so ordered
        release can skip past its rollout index instead of blocking forever.

        Idempotent and last-writer-wins: evicts any item already `put` at
        this index (else it would orphan in `_store`, since ordered
        release skips dropped indices). Repeated drops and drops of
        already-released indices are no-ops.

        Never blocks on `maxsize` — blocking a drop notification while the
        consumer waits on this index would deadlock. `_dropped` is pruned
        on every cursor advance so it stays bounded.
        """
        async with self._cond:
            if rollout_index < self._next_rollout_index:
                return
            old_cursor = self._next_rollout_index
            self._store.pop(rollout_index, None)  # evict to avoid orphan
            self._dropped.add(rollout_index)
            self._advance_cursor_past_leading_drops()
            self._notify_cursor_advance_if_moved(old_cursor)
            self._cond.notify_all()

    async def get(self, n: int) -> list[BatchItem]:
        """Get n items from the buffer.

        Blocks until n items are available. Returns items strictly in
        rollout-dispatch order (waits for the next contiguous block of
        rollout indices, skipping dropped ones).
        """
        async with self._cond:
            old_cursor = self._next_rollout_index
            while True:
                indices = self._ready_ordered(n)
                if indices is not None:
                    break
                await self._cond.wait()
            batch = [self._store.pop(i) for i in indices]
            self._next_rollout_index = indices[-1] + 1
            self._advance_cursor_past_leading_drops()
            self._notify_cursor_advance_if_moved(old_cursor)
            self._cond.notify_all()
            return batch

    def _advance_cursor_past_leading_drops(self) -> None:
        """Advance the release cursor past any leading contiguous dropped
        indices, then prune out-of-window drops.

        Called at the tail of `get()` and inside `mark_dropped()`.
        Without this advance the producer's scheduling window stays
        anchored to a permanently-dropped low index — if the first
        `maxsize` slots all get dropped, dispatch stalls and the
        consumer never accumulates enough to advance the cursor itself:
        classic deadlock.

        Does NOT fire `cursor_event`; callers snapshot the cursor
        BEFORE any cursor-mutating step and call
        `_notify_cursor_advance_if_moved(old_cursor)` after so that the
        aggregate move (batch tail + drop-fold) fires exactly one event.

        Callers must hold `_cond`.
        """
        while self._next_rollout_index in self._dropped:
            self._next_rollout_index += 1
        self._dropped = {i for i in self._dropped if i >= self._next_rollout_index}

    def _notify_cursor_advance_if_moved(self, old_cursor: int) -> None:
        """Fire `cursor_event` if the cursor moved since `old_cursor`.

        Edge-triggered: `set()` immediately followed by `clear()` wakes any
        waiter that was blocked at the moment of the advance, but leaves
        the event cleared so waiters that start after re-check the cursor
        value directly rather than seeing a stale True.

        Callers must hold `_cond`.
        """
        if self._next_rollout_index != old_cursor:
            self._cursor_event.set()
            self._cursor_event.clear()

    def _check_rollout_index_unused(self, rollout_index: int, *, op: str) -> None:
        """Raise if `rollout_index` is already stored, already dropped, or
        already released. Callers must hold `_cond`."""
        if rollout_index in self._store:
            raise ValueError(
                f"{op}(rollout_index={rollout_index}): index is already in"
                " the buffer; duplicate put would overwrite the stored item."
            )
        if rollout_index in self._dropped:
            raise ValueError(
                f"{op}(rollout_index={rollout_index}): index was already"
                " marked dropped; putting it now would silently orphan the"
                " item (ordered release skips dropped indices)."
            )
        if rollout_index < self._next_rollout_index:
            raise ValueError(
                f"{op}(rollout_index={rollout_index}): index is below"
                f" _next_rollout_index={self._next_rollout_index};"
                " ordered release has already advanced past it."
            )

    def _ready_ordered(self, n: int) -> list[int] | None:
        """Return the next n rollout indices to release (in order) if all
        are present, else None. Skips indices that were dropped."""
        indices: list[int] = []
        cur = self._next_rollout_index
        while len(indices) < n:
            if cur in self._dropped:
                cur += 1
                continue
            if cur in self._store:
                indices.append(cur)
                cur += 1
                continue
            return None  # next required index not ready yet -> caller waits
        return indices


class DispatchGate:
    """Producer-side backpressure. Refuses to dispatch a rollout index whose
    count of *valid* (non-dropped) predecessors still awaiting release would
    reach ``maxsize``.

    Prevents the bounded-buffer + in-order-release deadlock (a slow low
    index gets locked out of a full buffer by faster later-dispatched
    groups, so strict in-order release can never advance) AND bounds
    off-policy staleness between sampler and trainer.

    Backpressure is measured against *valid outstanding* indices, not the
    raw dispatch-index distance ``rollout_index - cursor``. Dropped groups
    occupy no buffer slot and add no staleness, so counting them against the
    window would let a drop burst (e.g. a 503 storm tripping the
    transient-failure gate) fill the window with tombstones and starve a
    same-size consumer batch -- a mid-run freeze distinct from the reorder
    deadlock this gate also guards. The correction subtracts the buffer's
    ``dropped_ahead_count`` (drops in ``[cursor, rollout_index)``) from the
    distance.
    """

    def __init__(
        self,
        get_cursor: Callable[[], int],
        cursor_event: asyncio.Event,
        maxsize: int,
        get_dropped_ahead: Callable[[], int] | None = None,
    ):
        self._get_cursor = get_cursor
        self._cursor_event = cursor_event
        self._maxsize = maxsize
        # Defaults to "no drops" so callers/tests that only exercise the
        # staleness-window behavior (no drops in play) need not wire it.
        self._get_dropped_ahead = get_dropped_ahead or (lambda: 0)

    async def wait_until_schedulable(self, rollout_index: int) -> None:
        """Block until fewer than ``maxsize`` *valid* (non-dropped) indices in
        ``[cursor, rollout_index)`` remain unreleased, so ``rollout_index``
        can be produced without overflowing ordered release.

        The gate condition is ``valid_outstanding >= maxsize`` where
        ``valid_outstanding = (rollout_index - cursor) - dropped_ahead``.
        Dropped indices are excluded because they hold no buffer slot and
        the consumer skips them during ordered release; counting them would
        starve batches of the same size as the window (see class docstring).
        """
        while (
            rollout_index - self._get_cursor() - self._get_dropped_ahead()
            >= self._maxsize
        ):
            await self._cursor_event.wait()


class RolloutDispatcher:
    """Runs one rollout end-to-end: group_rollout, filter, put/mark_dropped.

    Caller must have acquired samples_per_task semaphore permits before
    calling dispatch(); they are released in finally.

    Its only state is the cumulative filter-outcome tally that
    ``AsyncRolloutPipeline.sample_pipeline_gauges`` deltas once per train
    step, alongside the dispatch counters the pipeline itself keeps.
    """

    def __init__(
        self,
        agent: Agent,
        buffer: InOrderBuffer,
        semaphore: asyncio.Semaphore,
        config: RunConfig,
        trajectory_callback: Any = None,
    ):
        self._agent = agent
        self._buffer = buffer
        self._semaphore = semaphore
        self._config = config
        self._trajectory_group_filter = config.rollout.trajectory_group_filter.make()
        self._trajectory_callback = trajectory_callback
        # Cumulative per-outcome tally, incremented once a dispatch has
        # been CLASSIFIED. A dispatch that raised never reaches the filter
        # and so appears in none of these buckets -- matching
        # ``_dropped_dispatches``, whose denominator counts it but whose
        # numerator does not.
        self.outcome_counts: Counter[GroupFilterOutcome] = Counter()

    async def dispatch(
        self,
        task_data: TaskData,
        rollout_batch: int,
        task_idx: int,
        rollout_index: int,
    ) -> dict[str, Any]:
        """Run one rollout; return its per-rollout metrics dict.

        The three ids: `rollout_batch` = sampling-loop iteration (metrics
        x-axis); `task_idx` = dataset index (stamped on metrics);
        `rollout_index` = monotonic dispatch order (used by buffer/gate
        for in-order release + backpressure).
        """
        # Per-trajectory permit release: the producer acquired
        # ``samples_per_task`` permits up front. ``pending`` tracks how many
        # are still un-released; the agent fires ``_on_trajectory_done`` once
        # per trajectory as it finishes, freeing one permit immediately
        # instead of holding all N until the group's slowest straggler
        # completes. The ``finally`` drains whatever the agent didn't fire
        # (pre-fanout exceptions, cancellation, partial completion) so this
        # dispatch always nets exactly N releases -- no leak, no over-release.
        pending = self._config.rollout.samples_per_task

        def _on_trajectory_done() -> None:
            nonlocal pending
            if pending <= 0:
                # Defensive: a buggy agent that fires more than N times
                # would otherwise over-release and inflate the budget.
                return
            self._semaphore.release()
            pending -= 1

        try:
            # dispatch_index threads through the RolloutContext to every
            # trajectory callback so the value is stable per dispatch
            # even under concurrent async dispatches.
            logger.info(
                f"[Rollout batch {rollout_batch} (task {task_idx})] Starting rollout for {self._config.rollout.samples_per_task} samples."
            )
            trajectory_group, rollout_metrics = await self._agent.group_rollout(
                task_data,
                self._config.rollout.samples_per_task,
                thinking_level=self._config.agent.thinking_level or None,
                max_generation_tokens=self._config.agent.max_generation_tokens,
                on_trajectory_done=_on_trajectory_done,
                dispatch_index=rollout_index,
            )
            rollout_metrics["task_idx"] = task_idx
            logger.info(
                f"[Rollout batch {rollout_batch} (task {task_idx})] Finished rollout. Got {len(trajectory_group.scored_trajectories)} samples."
            )
            is_valid, outcome = self._trajectory_group_filter(trajectory_group)
            self.outcome_counts[outcome] += 1
            if not is_valid:
                # Upstream/infra failures (NaN reward or transient failure
                # gate) get a WARN so operators notice. Routine RL variance
                # (zero-variance drop) stays at INFO.
                log_fn = (
                    logger.warning if outcome.is_structural_failure else logger.info
                )
                log_fn(
                    f"[Rollout batch {rollout_batch}] Dropping group:"
                    f" {outcome.value}."
                )

            if is_valid:
                await self._buffer.put(
                    BatchItem(group=trajectory_group, metrics=rollout_metrics),
                    rollout_index,
                )
            else:
                rollout_metrics["dropped"] = True
                await self._buffer.mark_dropped(rollout_index)
                if self._trajectory_callback is not None:
                    for _gt in trajectory_group.scored_trajectories:
                        # Pass the live Trajectory; observers read
                        # rollout_idx / phase / dispatch_index / session_id
                        # off traj.extra so the dropped-status update lands
                        # on the same doc on_trajectory_started created.
                        try:
                            await self._trajectory_callback.on_trajectory_dropped(
                                task_data,
                                getattr(_gt, "trajectory", None),
                                outcome.value,
                            )
                        except Exception:
                            logger.warning(
                                "on_trajectory_dropped callback failed",
                                exc_info=True,
                            )
                logger.info(f"Discarded task {task_idx} in sampling loop.")

            return rollout_metrics
        except Exception as e:
            logger.error(
                f"Error in rollout at rollout batch {rollout_batch}: {e}",
                exc_info=True,
            )
            # Mark dropped so ordered release can advance past this index;
            # otherwise the trainer deadlocks waiting on it. Safe to call
            # unconditionally (idempotent, evicts already-put items). Any
            # failure here is a real buffer bug — let it propagate.
            await self._buffer.mark_dropped(rollout_index)
            return {}
        finally:
            # Drain any unfired callbacks -- agent raised before firing for
            # some trajectories, rollout was cancelled, etc. -- so this
            # dispatch nets exactly ``samples_per_task`` releases against
            # the ``samples_per_task`` acquired up front.
            while pending > 0:
                self._semaphore.release()
                pending -= 1


class MetricsAggregator:
    """Aggregates one batch's per-rollout metrics (spec-driven reduction).
    Stateless.

    Delegates per-key reduction to ``aggregate_by_spec`` in
    ``core.metric_spec`` so the reduction convention lives in one place.
    """

    def __init__(
        self,
        buffer: InOrderBuffer,
        ml_logger: Logger,
        config: RunConfig,
    ):
        self._buffer = buffer
        # Currently unused; ml_logger emission moved to the train-step
        # boundary. Retained for the sink-layer refactor.
        self._ml_logger = ml_logger
        self._config = config

    async def log_batch(
        self, rollout_tasks: list[asyncio.Task[dict[str, Any]]], rollout_batch: int
    ) -> None:
        """Await all rollout tasks and aggregate their metrics.

        No ml_logger emission — Layer-2 aggregation happens at the
        train-step boundary (see ``_aggregate_step_metrics`` in
        ``train_and_eval.py``), and the two gauges that cannot ride on a
        ``BatchItem`` (``pipeline/buffer_utilization`` /
        ``pipeline/group_drop_ratio``) are sampled there too, via
        ``AsyncRolloutPipeline.sample_pipeline_gauges``.
        ``rollout_batch`` is unused; kept for call-signature
        compatibility.
        """
        results = await asyncio.gather(*rollout_tasks)
        # Strip pipeline bookkeeping keys (per-rollout ints/bools, not
        # spec-registered metrics) before spec-driven aggregation.
        _NON_METRIC_KEYS = frozenset({"dropped", "task_idx"})
        filtered_results = [
            {k: v for k, v in r.items() if k not in _NON_METRIC_KEYS} for r in results
        ]
        aggregate_by_spec(filtered_results)
