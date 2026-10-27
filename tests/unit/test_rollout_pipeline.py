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

"""Tests for pipeline.rollout_pipeline.

Consolidates:
  - InOrderBuffer tests (was tests/unit/test_in_order_buffer.py, 21 tests)
  - DispatchGate tests (was tests/unit/test_dispatch_gate.py, 8 tests)
  - AsyncRolloutPipeline facade / producer-loop / eval-scheduling tests (new, 7 tests)

The private classes InOrderBuffer, DispatchGate, RolloutDispatcher,
MetricsAggregator are not in the module's ``__all__`` but the test file
imports them directly since that's the point of unit-testing them.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.atif_trajectory import Agent as AtifAgent, Trajectory as AtifTrajectory

from core.atif_trajectory import (
    ScoredTrajectory,
    Reward,
    StepStats,
    TrajectoryGroup,
    Step,
    TrajectoryStep as AtifTrajectoryStep,
)
from core.atif_trajectory import GroupFilterOutcome
from core.rollout_protocol import BatchItem
from pipeline.async_rollout_pipeline import AsyncRolloutPipeline
from pipeline.sync_rollout_pipeline import SyncRolloutPipeline
from pipeline.rollout_pipeline import (
    # Module-private for testing:
    InOrderBuffer,
    DispatchGate,
    RolloutDispatcher,
    MetricsAggregator,
    group_filter_metrics,
)


def _make_test_scored(reward: float) -> ScoredTrajectory:
    us = Step(step_id=1, source="user", message="p")
    a = AtifTrajectoryStep.from_wire(step_id=2, generation="g", stats=StepStats())
    traj = AtifTrajectory(
        schema_version="ATIF-v1.7",
        session_id="s",
        agent=AtifAgent(name="t", version="0", model_name="m"),
        steps=[us, a],
    )
    return ScoredTrajectory(trajectory=traj, reward=Reward(reward=reward))


def _make_test_group(rewards) -> TrajectoryGroup:
    return TrajectoryGroup(
        task_data=None,
        scored_trajectories=[_make_test_scored(r) for r in rewards],
    )


# ---------------------------------------------------------------------------
# Shared test fixtures for the AsyncRolloutPipeline facade tests below.
# ---------------------------------------------------------------------------


def _make_test_config(
    *,
    max_pending=8,
    samples_per_task=2,
    tasks_per_train_step=2,
    max_eval_items=-1,
    eval_samples_per_task=1,
    zero_variance_keep_prob=1.0,
):
    """Minimal composed ``RunConfig``-shaped SimpleNamespace for pipeline
    construction.

    The pipelines read the composed ``RunConfig`` shape: training-cadence
    knobs live under ``config.train``, rollout knobs under
    ``config.rollout``, eval knobs under ``config.eval``, and
    ``thinking_level`` under ``config.agent``. The async pipeline reads:
    train.tasks_per_train_step; rollout.{max_pending_rollout_tasks,
    max_concurrent_samples, samples_per_task,
    trajectory_group_filter}; eval.max_eval_items; agent.thinking_level.
    """
    from core.atif_trajectory import TrajectoryGroupFilterConfig

    return SimpleNamespace(
        train=SimpleNamespace(
            tasks_per_train_step=tasks_per_train_step,
            resume_from_train_step=0,
            sync_every_n_train_steps=1,
        ),
        rollout=SimpleNamespace(
            max_pending_rollout_tasks=max_pending,
            max_concurrent_samples=max_pending * samples_per_task,
            samples_per_task=samples_per_task,
            trajectory_group_filter=TrajectoryGroupFilterConfig(
                zero_variance_keep_prob=zero_variance_keep_prob,
                max_transient_failure_fraction=None,
            ),
        ),
        eval=SimpleNamespace(
            eval_samples_per_task=eval_samples_per_task,
            max_eval_items=max_eval_items,
        ),
        agent=SimpleNamespace(thinking_level=None, max_generation_tokens=None),
    )


class _FakeIter:
    """Minimal BatchIterator-shaped stand-in.

    `batches` is a list of lists (each inner list is one task_batch).
    Iterator raises StopIteration when exhausted; supports .reset(),
    .current attribute, .dataset attribute.
    """

    def __init__(self, batches: list[list], dataset_size: int | None = None):
        self._batches = batches
        self._i = 0
        self.current = 0
        self.dataset = list(range(dataset_size or sum(len(b) for b in batches)))

    def __iter__(self):
        return self

    def __next__(self):
        if self._i >= len(self._batches):
            raise StopIteration
        b = self._batches[self._i]
        self._i += 1
        self.current += len(b)
        return b

    def reset(self):
        self._i = 0
        self.current = 0


class _FakeDataset:
    """Duck-typed dataset stand-in exposing ``build_iter``.

    ``batches`` = pre-computed batch sequence returned by the built iterator.
    ``dataset_size`` = value for ``len(dataset)`` / ``iter.dataset`` (for
    task_idx modulo math inside the producer loop).

    Honors ``resume_from_position`` (sets built iterator's ``current`` field)
    for regression tests; ignores ``batch_size`` / ``max_items`` / ``repeat``
    since the fixture is scripted by ``batches``.
    """

    def __init__(self, batches: list[list], dataset_size: int | None = None):
        self._batches = batches
        self._dataset_size = dataset_size or sum(len(b) for b in batches)

    def __len__(self) -> int:
        return self._dataset_size

    def build_iter(self, **kwargs) -> "_FakeIter | None":
        # Zero-sized eval dataset → mimic real dataset's None return.
        if self._dataset_size == 0 or kwargs.get("max_items") == 0:
            return None
        it = _FakeIter(self._batches, dataset_size=self._dataset_size)
        position = kwargs.get("resume_from_position")
        if position is not None:
            it.current = position
        return it


def _make_mock_agent(*, group_rollout_returns=None):
    """Mock agent whose group_rollout returns a synthetic (TrajectoryGroup, metrics) pair.

    Default: 2 trivial trajectories with reward 1.0 each so the default
    TrajectoryGroupFilter (zero_variance_keep_prob=1.0) keeps the group.
    """
    if group_rollout_returns is None:
        group_rollout_returns = (
            _make_test_group([0.0, 1.0]),
            {"env/average_reward": 0.5},
        )
    agent = MagicMock()
    agent.group_rollout = AsyncMock(return_value=group_rollout_returns)
    return agent


def _make_pipeline(
    *,
    train_batches=None,
    agent=None,
    max_pending=8,
    samples_per_task=2,
    tasks_per_train_step=2,
    eval_dataset=None,
    zero_variance_keep_prob=1.0,
):
    """Construct a AsyncRolloutPipeline with sensible test defaults."""
    train_batches = train_batches if train_batches is not None else [[0, 1]]
    config = _make_test_config(
        max_pending=max_pending,
        samples_per_task=samples_per_task,
        tasks_per_train_step=tasks_per_train_step,
        zero_variance_keep_prob=zero_variance_keep_prob,
    )
    agent = agent or _make_mock_agent()
    return AsyncRolloutPipeline(
        agent=agent,
        config=config,
        train_dataset=_FakeDataset(train_batches),
        eval_dataset=eval_dataset,
        eval_agent=None,
        ml_logger=MagicMock(),
    )


# ---------------------------------------------------------------------------
# InOrderBuffer tests (moved from tests/unit/test_in_order_buffer.py)
# ---------------------------------------------------------------------------


class TestInOrderBuffer:
    """Tests for the buffer's rollout-dispatch-ordered release contract."""

    async def test_releases_in_dispatch_order_regardless_of_put_order(self):
        # Core contract: get_config() returns items strictly in dispatch order even
        # if they are put() out of order (the "fast prompt finishes first"
        # scenario the buffer is designed to handle).
        buf = InOrderBuffer(maxsize=10, start_rollout_index=0)
        await buf.put("c", rollout_index=2)
        await buf.put("a", rollout_index=0)
        await buf.put("b", rollout_index=1)
        assert await buf.get(3) == ["a", "b", "c"]

    async def test_get_blocks_for_missing_intermediate_index(self):
        # If indices 0 and 2 arrive but 1 has not, get_config(3) must wait for 1.
        # This is the head-of-line block by design.
        buf = InOrderBuffer(maxsize=10, start_rollout_index=0)
        await buf.put("a", rollout_index=0)
        await buf.put("c", rollout_index=2)
        consumer = asyncio.create_task(buf.get(3))
        await asyncio.sleep(0)
        assert not consumer.done(), "consumer should block waiting for index=1"
        await buf.put("b", rollout_index=1)
        out = await asyncio.wait_for(consumer, timeout=1.0)
        assert out == ["a", "b", "c"]

    async def test_mark_dropped_unblocks_consumer(self):
        # mark_dropped is the recovery path: if index=1 is filtered out (zero
        # variance / invalid group / rollout exception), the consumer must
        # skip past it instead of blocking forever.
        buf = InOrderBuffer(maxsize=10, start_rollout_index=0)
        await buf.put("a", rollout_index=0)
        await buf.put("c", rollout_index=2)
        consumer = asyncio.create_task(buf.get(2))
        await asyncio.sleep(0)
        assert not consumer.done()
        await buf.mark_dropped(1)
        out = await asyncio.wait_for(consumer, timeout=1.0)
        assert out == ["a", "c"]

    async def test_mark_dropped_before_put_also_unblocks(self):
        # The drop can also arrive before the surrounding indices are put.
        buf = InOrderBuffer(maxsize=10, start_rollout_index=0)
        await buf.mark_dropped(1)
        await buf.put("a", rollout_index=0)
        await buf.put("c", rollout_index=2)
        out = await asyncio.wait_for(buf.get(2), timeout=1.0)
        assert out == ["a", "c"]

    async def test_get_advances_past_trailing_dropped_indices(self):
        # After a batch is returned, _next_rollout_index should skip past
        # dropped indices so the next get_config() doesn't have to wait on them.
        buf = InOrderBuffer(maxsize=10, start_rollout_index=0)
        await buf.put("a", rollout_index=0)
        await buf.mark_dropped(1)
        await buf.mark_dropped(2)
        await buf.put("d", rollout_index=3)
        # First batch consumes index=0; _next_rollout_index should advance
        # to 3 (skipping 1 and 2). Subsequent get_config(1) returns "d" without
        # waiting.
        assert await buf.get(1) == ["a"]
        assert await buf.get(1) == ["d"]

    async def test_qsize_excludes_dropped_indices(self):
        # buffer_utilization is computed from qsize(); dropped indices are
        # not buffered items and should not inflate utilization.
        buf = InOrderBuffer(maxsize=10, start_rollout_index=0)
        await buf.put("a", rollout_index=0)
        await buf.put("b", rollout_index=2)
        await buf.mark_dropped(1)
        assert buf.qsize() == 2  # only the 2 stored items, not the dropped one

    async def test_dropped_ahead_count_tracks_window_drops(self):
        # dropped_ahead_count is what DispatchGate subtracts from the
        # dispatch-index distance. It counts live drops at/above the cursor
        # and shrinks as the cursor advances past them.
        buf = InOrderBuffer(maxsize=10, start_rollout_index=0)
        assert buf.dropped_ahead_count == 0
        await buf.put("a", rollout_index=0)  # cursor stays 0 (0 is stored)
        await buf.mark_dropped(1)
        await buf.mark_dropped(2)
        assert buf.dropped_ahead_count == 2  # drops 1, 2 in [cursor=0, ...)
        # Releasing index 0 folds the trailing drops 1, 2 into the cursor
        # (cursor -> 3) and prunes them, so the count drops to 0.
        assert await buf.get(1) == ["a"]
        assert buf.cursor == 3
        assert buf.dropped_ahead_count == 0

    async def test_start_rollout_index_respected(self):
        # When resuming from a checkpoint, start_rollout_index must align
        # with the producer's first dispatch index. Items below it should
        # never be released.
        buf = InOrderBuffer(maxsize=10, start_rollout_index=5)
        await buf.put("a", rollout_index=5)
        await buf.put("b", rollout_index=6)
        assert await buf.get(2) == ["a", "b"]

    async def test_full_buffer_blocks_producer(self):
        # The maxsize backpressure gate.
        buf = InOrderBuffer(maxsize=2, start_rollout_index=0)
        await buf.put("a", rollout_index=0)
        await buf.put("b", rollout_index=1)
        producer = asyncio.create_task(buf.put("c", rollout_index=2))
        await asyncio.sleep(0)
        assert not producer.done(), "producer should be blocked while full"
        await buf.get(2)
        await asyncio.wait_for(producer, timeout=1.0)
        assert buf.qsize() == 1

    async def test_concurrent_out_of_order_producers(self):
        # End-to-end stress: 20 items dispatched, produced in random order,
        # plus a few drops. Consumer must still see them in dispatch order.
        import random

        rng = random.Random(0)
        buf = InOrderBuffer(maxsize=20, start_rollout_index=0)
        dropped_indices = {3, 11, 17}
        all_indices = list(range(20))
        kept_indices = [i for i in all_indices if i not in dropped_indices]
        # Shuffle to simulate completion order != dispatch order.
        produce_order = list(all_indices)
        rng.shuffle(produce_order)

        async def producer():
            for idx in produce_order:
                # Small yield so other tasks can interleave.
                await asyncio.sleep(0)
                if idx in dropped_indices:
                    await buf.mark_dropped(idx)
                else:
                    await buf.put(f"item-{idx}", rollout_index=idx)

        async def consumer(out):
            # Pull in two batches to also exercise resumption.
            out.extend(await buf.get(len(kept_indices) // 2))
            out.extend(await buf.get(len(kept_indices) - len(kept_indices) // 2))

        out: list = []
        await asyncio.wait_for(asyncio.gather(producer(), consumer(out)), timeout=5.0)
        assert out == [f"item-{i}" for i in kept_indices]


class TestInOrderBufferDefensiveChecks:
    """Rollout-index reuse would silently lose data. The buffer raises
    fast in those cases instead. mark_dropped is idempotent for repeated
    drops / already-released indices so it's safe to call defensively."""

    async def test_put_rejects_duplicate_index_in_store(self):
        # Putting the same index twice would overwrite the first item.
        buf = InOrderBuffer(maxsize=10, start_rollout_index=0)
        await buf.put("a", rollout_index=0)
        with pytest.raises(ValueError, match="already in the buffer"):
            await buf.put("a2", rollout_index=0)
        # Original item is intact.
        assert await buf.get(1) == ["a"]

    async def test_put_rejects_index_already_dropped(self):
        # Putting an index that was already marked dropped would orphan
        # the item: ordered release skips dropped indices before checking
        # the store.
        #
        # We use a non-leading dropped index (5) so mark_dropped does not
        # immediately advance _next_rollout_index past it. A *leading*
        # dropped index would be pruned by mark_dropped's cursor-advance
        # step, and a subsequent put at that index would then hit the
        # "below _next_rollout_index" check instead -- also correct, but
        # not the code path this test is exercising.
        buf = InOrderBuffer(maxsize=10, start_rollout_index=0)
        await buf.mark_dropped(5)
        with pytest.raises(ValueError, match="already marked dropped"):
            await buf.put("a", rollout_index=5)

    async def test_put_rejects_index_below_next_rollout_index(self):
        # An index that's already been released sits in _store forever
        # because _ready_ordered scans starting from _next_rollout_index.
        buf = InOrderBuffer(maxsize=10, start_rollout_index=0)
        await buf.put("a", rollout_index=0)
        await buf.get(1)  # advances _next_rollout_index to 1
        with pytest.raises(ValueError, match="below _next_rollout_index"):
            await buf.put("stale", rollout_index=0)

    async def test_put_rejects_index_below_start_after_resume(self):
        # Resume scenario: buffer is constructed with start_rollout_index=10
        # (mid-training resume from checkpoint). Putting any index below 10
        # must raise -- if a producer is mistakenly initialized from 0 while
        # the buffer expects 10, all early puts would be silently lost.
        buf = InOrderBuffer(maxsize=10, start_rollout_index=10)
        with pytest.raises(ValueError, match="below _next_rollout_index=10"):
            await buf.put("stale", rollout_index=0)
        with pytest.raises(ValueError, match="below _next_rollout_index=10"):
            await buf.put("stale", rollout_index=9)
        # The expected resumed index works.
        await buf.put("ok", rollout_index=10)
        assert await buf.get(1) == ["ok"]

    async def test_mark_dropped_evicts_index_in_store(self):
        # Last-writer-wins: mark_dropped after a successful put evicts the
        # stored item rather than orphaning it. Without eviction the item
        # would sit in _store forever (ordered release skips dropped indices
        # before checking _store) and eventually deadlock producers on the
        # maxsize cap.
        buf = InOrderBuffer(maxsize=10, start_rollout_index=0)
        await buf.put("a", rollout_index=0)
        assert buf.qsize() == 1
        await buf.mark_dropped(0)  # evicts "a"
        assert buf.qsize() == 0
        # The next put at a later index releases past the dropped 0.
        await buf.put("b", rollout_index=1)
        assert await buf.get(1) == ["b"]

    async def test_mark_dropped_eviction_unblocks_full_buffer(self):
        # An evicted item frees a maxsize slot; a producer waiting on
        # buffer-full must be woken so it can put.
        buf = InOrderBuffer(maxsize=1, start_rollout_index=0)
        await buf.put("a", rollout_index=0)
        producer_done = asyncio.Event()

        async def slow_producer():
            await buf.put("b", rollout_index=1)
            producer_done.set()

        task = asyncio.create_task(slow_producer())
        # Give producer a chance to start and block on buffer-full.
        await asyncio.sleep(0.05)
        assert not producer_done.is_set()
        # Evict "a" -> frees the slot -> producer wakes and puts "b".
        await buf.mark_dropped(0)
        await asyncio.wait_for(producer_done.wait(), timeout=1.0)
        assert await buf.get(1) == ["b"]
        await task

    async def test_mark_dropped_is_idempotent_for_repeated_drops(self):
        # The defensive `mark_dropped` call in run_single_rollout's exception
        # handler may run after `mark_dropped` already succeeded earlier
        # (e.g. an exception raised post-disposal). Repeat drops are no-ops.
        buf = InOrderBuffer(maxsize=10, start_rollout_index=0)
        await buf.mark_dropped(5)
        await buf.mark_dropped(5)  # must not raise
        await buf.put("a", rollout_index=0)
        # Buffer still works normally.
        assert await buf.get(1) == ["a"]

    async def test_mark_dropped_is_idempotent_for_already_released_index(self):
        # After get_config() advances past an index, re-dropping it is a no-op AND
        # leaves `_dropped` empty (otherwise late drops between consumer
        # batches would accumulate unboundedly when the trainer stalls).
        buf = InOrderBuffer(maxsize=10, start_rollout_index=0)
        await buf.put("a", rollout_index=0)
        await buf.get(1)  # _next_rollout_index is now 1
        await buf.mark_dropped(0)  # must not raise
        assert 0 not in buf._dropped, (
            "mark_dropped for an already-released index must early-return"
            " rather than accumulate stale entries in _dropped."
        )


class TestInOrderBufferCursorEvent:
    """Pin down the cursor_event contract that DispatchGate relies on.
    If a future edit to InOrderBuffer forgets to fire the event on a
    cursor-advancing operation, DispatchGate silently starts hanging.
    These tests fail fast in that case."""

    async def test_cursor_event_fires_on_get(self):
        buf = InOrderBuffer(maxsize=10, start_rollout_index=0)
        await buf.put("a", rollout_index=0)
        # Waiter armed BEFORE the cursor-advancing get_config().
        waiter = asyncio.create_task(buf.cursor_event.wait())
        await asyncio.sleep(0)
        assert not waiter.done(), "waiter should be blocked before get_config()"
        assert await buf.get(1) == ["a"]
        # get_config() advanced cursor 0 -> 1; the edge-triggered event should
        # have woken the waiter.
        await asyncio.wait_for(waiter, timeout=1.0)
        assert buf.cursor == 1

    async def test_cursor_event_fires_on_mark_dropped_that_advances_cursor(self):
        buf = InOrderBuffer(maxsize=10, start_rollout_index=0)
        waiter = asyncio.create_task(buf.cursor_event.wait())
        await asyncio.sleep(0)
        assert not waiter.done(), "waiter should be blocked before mark_dropped()"
        # Dropping the leading index advances the cursor past it.
        await buf.mark_dropped(0)
        await asyncio.wait_for(waiter, timeout=1.0)
        assert buf.cursor == 1

    async def test_cursor_event_does_not_fire_on_put(self):
        # put() doesn't advance the cursor, so it should NOT fire the event.
        # (Waiter blocked before AND after put() -- verified via timeout.)
        buf = InOrderBuffer(maxsize=10, start_rollout_index=0)
        waiter = asyncio.create_task(buf.cursor_event.wait())
        await asyncio.sleep(0)
        assert not waiter.done()
        # put an out-of-order index (index 5 with cursor at 0) so the cursor
        # can't advance from this operation.
        await buf.put("x", rollout_index=5)
        await asyncio.sleep(0)
        assert not waiter.done(), (
            "cursor_event should not fire on a put that doesn't advance the cursor"
        )
        waiter.cancel()

    async def test_cursor_event_does_not_fire_on_mark_dropped_of_high_index(self):
        # mark_dropped on a NON-leading index (cursor stays at 0) should
        # not fire the event.
        buf = InOrderBuffer(maxsize=10, start_rollout_index=0)
        waiter = asyncio.create_task(buf.cursor_event.wait())
        await asyncio.sleep(0)
        assert not waiter.done()
        await buf.mark_dropped(5)  # cursor still 0
        await asyncio.sleep(0)
        assert not waiter.done(), (
            "cursor_event should not fire when the cursor didn't move"
        )
        assert buf.cursor == 0
        waiter.cancel()


# ---------------------------------------------------------------------------
# DispatchGate tests (moved from tests/unit/test_dispatch_gate.py)
# ---------------------------------------------------------------------------


class TestDispatchGate:
    """Test DispatchGate in isolation via a mock cursor source. Every test
    here would previously have had to construct a full TrajectoryBuffer
    and shuffle puts/gets just to move the cursor."""

    @pytest.mark.asyncio
    async def test_in_window_indices_pass_immediately(self):
        cursor = [0]
        event = asyncio.Event()
        gate = DispatchGate(get_cursor=lambda: cursor[0], cursor_event=event, maxsize=3)
        # Window [0, 3): indices 0, 1, 2 should not block.
        await asyncio.wait_for(gate.wait_until_schedulable(0), timeout=0.5)
        await asyncio.wait_for(gate.wait_until_schedulable(1), timeout=0.5)
        await asyncio.wait_for(gate.wait_until_schedulable(2), timeout=0.5)

    @pytest.mark.asyncio
    async def test_out_of_window_index_blocks(self):
        cursor = [0]
        event = asyncio.Event()
        gate = DispatchGate(get_cursor=lambda: cursor[0], cursor_event=event, maxsize=3)
        # Index 3 is exactly at cursor + maxsize (out of window: strict <).
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(gate.wait_until_schedulable(3), timeout=0.2)

    @pytest.mark.asyncio
    async def test_cursor_advance_unblocks_waiter(self):
        cursor = [0]
        event = asyncio.Event()
        gate = DispatchGate(get_cursor=lambda: cursor[0], cursor_event=event, maxsize=3)
        # Arm a waiter on the out-of-window index 3.
        waiter = asyncio.create_task(gate.wait_until_schedulable(3))
        await asyncio.sleep(0)
        assert not waiter.done()
        # Advance the mock cursor and fire the edge-triggered event.
        cursor[0] = 1
        event.set()
        event.clear()
        # Now window is [1, 4); index 3 is inside -> waiter unblocks.
        await asyncio.wait_for(waiter, timeout=1.0)

    @pytest.mark.asyncio
    async def test_gate_wakes_multiple_concurrent_waiters(self):
        # With cursor=0 and maxsize=2, window is [0, 2). Indices 2, 3, 4
        # are all out of window (2 >= 0+2, 3 >= 0+2, 4 >= 0+2) -> all block.
        # After cursor advances to 3, window becomes [3, 5). Predicate is
        # `rollout_index >= cursor + maxsize` = `rollout_index >= 5` for
        # all three indices -> FALSE -> all three unblock.
        cursor = [0]
        event = asyncio.Event()
        gate = DispatchGate(get_cursor=lambda: cursor[0], cursor_event=event, maxsize=2)
        waiters = [
            asyncio.create_task(gate.wait_until_schedulable(i)) for i in (2, 3, 4)
        ]
        await asyncio.sleep(0)
        assert all(not w.done() for w in waiters)
        cursor[0] = 3
        event.set()
        event.clear()
        await asyncio.wait_for(asyncio.gather(*waiters), timeout=1.0)

    @pytest.mark.asyncio
    async def test_dropped_ahead_excluded_from_window(self):
        # The drop-adjustment: with maxsize=2 and cursor=0, index 4 would
        # normally be out of window (4 >= 0+2). But if 3 indices in
        # [0, 4) were dropped, valid_outstanding = 4 - 0 - 3 = 1 < 2, so
        # index 4 is schedulable. This is the fix for the drop-storm
        # window-starvation freeze.
        cursor = [0]
        dropped = [0]
        event = asyncio.Event()
        gate = DispatchGate(
            get_cursor=lambda: cursor[0],
            cursor_event=event,
            maxsize=2,
            get_dropped_ahead=lambda: dropped[0],
        )
        # No drops yet: index 2 is out of window (2 - 0 - 0 >= 2) -> blocks.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(gate.wait_until_schedulable(2), timeout=0.2)
        # Now 3 drops in [0, 4): index 4 fits (4 - 0 - 3 = 1 < 2).
        dropped[0] = 3
        await asyncio.wait_for(gate.wait_until_schedulable(4), timeout=0.5)

    @pytest.mark.asyncio
    async def test_dropped_ahead_advance_unblocks_waiter(self):
        # A waiter parked out-of-window unblocks when a cursor advance both
        # raises the cursor and prunes drops (the get_config()-release path fires
        # cursor_event, which is what wakes the gate).
        cursor = [0]
        dropped = [0]
        event = asyncio.Event()
        gate = DispatchGate(
            get_cursor=lambda: cursor[0],
            cursor_event=event,
            maxsize=2,
            get_dropped_ahead=lambda: dropped[0],
        )
        waiter = asyncio.create_task(gate.wait_until_schedulable(3))
        await asyncio.sleep(0)
        assert not waiter.done()  # 3 - 0 - 0 >= 2 -> blocked
        # Release advances cursor to 2 (drops in [0,2) pruned): 3 - 2 - 0 < 2.
        cursor[0] = 2
        event.set()
        event.clear()
        await asyncio.wait_for(waiter, timeout=1.0)

    @pytest.mark.asyncio
    async def test_below_cursor_index_passes_immediately(self):
        # The gate's predicate is `rollout_index >= cursor + maxsize`.
        # An index below the cursor is trivially not >= cursor+maxsize
        # (since maxsize > 0), so it passes without blocking. This is
        # documenting-behavior, not a use case: `wait_until_schedulable`
        # is only called with the current index being dispatched.
        cursor = [5]
        event = asyncio.Event()
        gate = DispatchGate(get_cursor=lambda: cursor[0], cursor_event=event, maxsize=3)
        # Index 3 < cursor 5, so `3 >= 5 + 3 = 8` is FALSE -> passes.
        await asyncio.wait_for(gate.wait_until_schedulable(3), timeout=0.5)


def _make_buffer_and_gate(maxsize: int, start_rollout_index: int = 0):
    """Construct the pair the way train_and_eval_loop_async does."""
    buf = InOrderBuffer(maxsize=maxsize, start_rollout_index=start_rollout_index)
    gate = DispatchGate(
        get_cursor=lambda: buf.cursor,
        cursor_event=buf.cursor_event,
        maxsize=maxsize,
        get_dropped_ahead=lambda: buf.dropped_ahead_count,
    )
    return buf, gate


class TestBufferAndGateIntegration:
    """End-to-end tests wiring the two classes together as the training
    loop does. Regression coverage for the deadlock that the gate exists
    to prevent."""

    @pytest.mark.asyncio
    async def test_gate_blocks_beyond_window_then_releases_after_get(self):
        buf, gate = _make_buffer_and_gate(maxsize=2)  # window [0, 2)
        # In-window indices immediately schedulable.
        await asyncio.wait_for(gate.wait_until_schedulable(0), timeout=0.5)
        await asyncio.wait_for(gate.wait_until_schedulable(1), timeout=0.5)
        # Index 2 out of window -> blocks.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(gate.wait_until_schedulable(2), timeout=0.2)
        # Produce 0, 1 and consume index 0 -> cursor 0 -> 1, window -> [1, 3).
        await buf.put("g0", 0)
        await buf.put("g1", 1)
        assert await buf.get(1) == ["g0"]
        # Index 2 now fits.
        await asyncio.wait_for(gate.wait_until_schedulable(2), timeout=0.5)

    @pytest.mark.asyncio
    async def test_reorder_deadlocks_without_gate(self):
        """Documents the failure mode the gate fixes: with a full buffer of
        high indices, a late low index can neither be put (full) nor
        released (missing). Constructed with buffer alone (no gate)."""
        buf = InOrderBuffer(maxsize=4)
        for i in (1, 2, 3, 4):  # fills the buffer; index 0 arrives late
            await buf.put(f"g{i}", i)

        async def late_put0():
            await buf.put("g0", 0)  # blocks: buffer full

        async def consume():
            return await buf.get(1)  # blocks: index 0 not present

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.gather(late_put0(), consume()), timeout=0.5)

    @pytest.mark.asyncio
    async def test_drops_filling_window_do_not_starve_batch(self):
        """Regression for the mid-run window-vs-batch starvation deadlock.

        Distinct from ``test_reorder_deadlocks_without_gate`` (which the
        gate fixes): here the gate IS in place, yet a burst of drops that
        fills the look-ahead window starves a same-size consumer batch.

        Scenario (scaled-down mirror of the production freeze, where
        ``max_pending_rollout_batches == tasks_per_train_step == 32`` and a
        503 storm tripped the transient-failure gate on most groups):

          - ``maxsize == n == 4`` (the allowed-by-assertion equality case).
          - Producer fills the window ``[0, 4)`` with indices 0..3, then
            blocks at ``wait_until_schedulable(4)`` (``4 >= cursor+4``).
          - Index 0 completes *valid* -> stored at the cursor; indices
            1, 2, 3 complete *dropped*. Those drops are non-leading
            (cursor sits on stored index 0), so they never advance the
            cursor.
          - Consumer ``get_config(4)`` can never assemble 4 contiguous stored
            items: the window holds at most 4 indices and 3 of them are
            drops, so at most 1 valid item is ever simultaneously present.

        The cursor can only advance via ``get`` (blocked, needs 4) or a
        *leading* drop (index 0 is stored, not dropped) -> circular wait.
        Production prober signature: buffer stuck (``7/32``) with
        ``Active Sampling Tasks: 0`` (all dispatched rollouts finished).

        The whole ``gather`` must complete within the timeout; a hang is
        the bug.
        """
        # maxsize == batch (the allowed-by-assertion equality case, and
        # the production default). Drop 3 of every 4 dispatched indices, so
        # to assemble one batch of `batch` valid items the producer must be
        # able to run well beyond `maxsize` indices ahead of the release
        # cursor -- which the current window accounting forbids because
        # drops occupy window budget they never release.
        batch = maxsize = 4
        drop_stride = 4  # keep 1 in 4; need 4*batch dispatches per batch
        # Enough valid items for two full consumer batches.
        kept = [i for i in range(drop_stride * batch * 2) if i % drop_stride == 0]
        N = kept[-1] + 1
        dropped = {i for i in range(N) if i % drop_stride != 0}

        buf, gate = _make_buffer_and_gate(maxsize=maxsize)
        consumed: list = []

        async def consumer():
            while len(consumed) < len(kept):
                # Ask for `batch` items -- exactly what the trainer does
                # with tasks_per_train_step == max_pending.
                consumed.extend(await buf.get(batch))

        async def producer():
            for idx in range(N):
                await gate.wait_until_schedulable(idx)
                if idx in dropped:
                    await buf.mark_dropped(idx)
                else:
                    await buf.put(f"g{idx}", idx)

        await asyncio.wait_for(asyncio.gather(consumer(), producer()), timeout=5.0)
        assert consumed == [f"g{i}" for i in kept]

    @pytest.mark.asyncio
    async def test_gate_prevents_reorder_deadlock_end_to_end(self):
        """With the gate, in-order release drains fully even when the
        window-leading index is always the slowest to complete (the
        prompt-7 straggler pattern)."""
        N, batch, maxsize = 16, 4, 4
        buf, gate = _make_buffer_and_gate(maxsize=maxsize)
        consumed: list = []

        async def consumer():
            while len(consumed) < N:
                consumed.extend(await buf.get(batch))

        async def producer():
            tasks = []
            for idx in range(N):
                # Gate: never dispatch more than `maxsize` ahead of the cursor.
                await gate.wait_until_schedulable(idx)
                # Make the leading index of each window the slowest (worst
                # case for in-order release); others finish immediately.
                delay = 0.03 if idx % batch == 0 else 0.0

                async def rollout(i: int, d: float):
                    await asyncio.sleep(d)
                    await buf.put(f"g{i}", i)

                tasks.append(asyncio.create_task(rollout(idx, delay)))
            await asyncio.gather(*tasks)

        await asyncio.wait_for(asyncio.gather(consumer(), producer()), timeout=10)
        assert consumed == [f"g{i}" for i in range(N)]


# ---------------------------------------------------------------------------
# AsyncRolloutPipeline facade tests.
# ---------------------------------------------------------------------------


class TestAsyncRolloutPipelineConstruction:
    def test_construct_wires_all_internals(self):
        """Verify pipeline __init__ wires buffer, gate, dispatcher,
        aggregator and initializes producer + eval state."""
        pipeline = _make_pipeline(max_pending=4)
        # Buffer/gate/dispatcher/aggregator constructed.
        assert isinstance(pipeline._buffer, InOrderBuffer)
        assert isinstance(pipeline._gate, DispatchGate)
        assert isinstance(pipeline._dispatcher, RolloutDispatcher)
        assert isinstance(pipeline._aggregator, MetricsAggregator)
        # Buffer is empty, cursor at start_rollout_index=0.
        assert pipeline._buffer.qsize() == 0
        assert pipeline._buffer.maxsize == 4
        assert pipeline._buffer.cursor == 0
        # Producer state at initial values.
        assert pipeline._active_dispatches == 0
        assert pipeline._rollout_batch == 1  # always starts at 1; no resume knob
        assert pipeline._next_rollout_index == 0
        # Eval state clean.
        assert pipeline._evaluation_tasks == set()
        # Aggregator wired to the same buffer.
        assert pipeline._aggregator._buffer is pipeline._buffer

    def test_construct_applies_resume_position(self):
        """resume_train_iter_position should reach dataset.build_iter.

        Note: this is the ONLY resume knob the pipeline exposes now.
        The old ``resume_train_iter_advance`` kwarg + ``resume_from_rollout_batch``
        config field were dropped because strict-ordered pipeline release
        makes the trainer's dataset consumption pointer the single
        source of truth for resume correctness — see
        ``resolve_resume_position`` docstring in
        ``pipeline.train_and_eval``.
        """
        config = _make_test_config()
        pipeline = AsyncRolloutPipeline(
            agent=_make_mock_agent(),
            config=config,
            train_dataset=_FakeDataset([[0, 1], [2, 3]], dataset_size=4),
            eval_dataset=None,
            eval_agent=None,
            ml_logger=MagicMock(),
            resume_train_iter_position=7,
        )
        assert pipeline._train_iter.current == 7

    def test_init_rejects_train_step_exceeding_pending_batches(self):
        """The InOrderBuffer sizing invariant is enforced in __init__ (the
        single construction point). If tasks_per_train_step >
        max_pending_rollout_tasks, get_config(n) would block forever (the
        producer can never run n batches ahead of the release cursor), so
        construction must fail fast rather than deadlock at runtime.
        """
        config = _make_test_config(max_pending=2, tasks_per_train_step=4)
        with pytest.raises(AssertionError, match="tasks_per_train_step must be <="):
            AsyncRolloutPipeline(
                agent=_make_mock_agent(),
                config=config,
                train_dataset=_FakeDataset([[0, 1]]),
                eval_dataset=None,
                eval_agent=None,
                ml_logger=MagicMock(),
            )


class TestMetricsAggregatorLogBatch:
    @pytest.mark.asyncio
    async def test_log_batch_does_not_emit_to_ml_logger(self):
        """MetricsAggregator.log_batch aggregates but no longer writes to
        ml_logger -- the rollout_batch step axis is gone. Rollout metrics
        are emitted only at train-step boundary by train_and_eval_loop."""
        ml_logger = MagicMock()
        buffer = InOrderBuffer(maxsize=8)
        config = _make_test_config()
        aggregator = MetricsAggregator(
            buffer=buffer, ml_logger=ml_logger, config=config
        )

        from core.metric_spec import MetricValue

        async def _stub_task(result: dict):
            return result

        tasks = [
            asyncio.create_task(
                _stub_task(
                    {
                        "learning/sampling_accuracy": MetricValue.of_scalar(0.5),
                        "dropped": False,
                    }
                )
            ),
            asyncio.create_task(
                _stub_task(
                    {
                        "learning/sampling_accuracy": MetricValue.of_scalar(0.7),
                        "dropped": False,
                    }
                )
            ),
        ]
        await aggregator.log_batch(tasks, rollout_batch=1)

        ml_logger.log_metrics.assert_not_called()


class TestGroupFilterMetrics:
    """The counts -> per-reason gauges mapping both pipelines share."""

    def test_empty_window_emits_nothing(self):
        # A ratio over zero classified groups is undefined, not 0.0 --
        # and group_drop_ratio already omits itself on a quiet step.
        assert group_filter_metrics({}, 0) == {}

    def test_reason_shares_sum_to_the_drop_ratio(self):
        counts = {
            GroupFilterOutcome.KEPT: 5,
            GroupFilterOutcome.ZERO_VARIANCE: 2,
            GroupFilterOutcome.TOO_FEW_VALID_TRAJECTORIES: 2,
            GroupFilterOutcome.TOO_MANY_TRANSIENT_FAILURES: 1,
        }
        m = group_filter_metrics(counts, 10)
        assert sum(v for k, v in m.items() if "group_drop_reason/" in k) == 0.5
        assert m["pipeline/group_drop_reason/zero_variance"] == pytest.approx(0.2)

    def test_every_reason_key_present_even_at_zero(self):
        # The whole point: a flat-zero series must be distinguishable from
        # an absent one, so all three are emitted whenever the window is
        # non-empty.
        m = group_filter_metrics({GroupFilterOutcome.KEPT: 4}, 4)
        assert set(k for k in m if "group_drop_reason/" in k) == {
            "pipeline/group_drop_reason/zero_variance",
            "pipeline/group_drop_reason/too_few_valid_trajectories",
            "pipeline/group_drop_reason/too_many_transient_failures",
        }
        assert all(v == 0.0 for k, v in m.items() if "group_drop_reason/" in k)

    def test_zero_variance_ratio_counts_ties_that_were_kept(self):
        """At keep_prob=1.0 no group is ever dropped for zero variance, so
        the drop share is 0 while the rate is not. Without this metric the
        filter reads as inert on exactly the configs that disable it."""
        counts = {
            GroupFilterOutcome.KEPT: 6,
            GroupFilterOutcome.KEPT_ZERO_VARIANCE: 4,
        }
        m = group_filter_metrics(counts, 10)
        assert m["pipeline/group_zero_variance_ratio"] == pytest.approx(0.4)
        assert m["pipeline/group_drop_reason/zero_variance"] == 0.0

    def test_zero_variance_ratio_spans_kept_and_dropped_ties(self):
        counts = {
            GroupFilterOutcome.KEPT: 2,
            GroupFilterOutcome.KEPT_ZERO_VARIANCE: 3,
            GroupFilterOutcome.ZERO_VARIANCE: 5,
        }
        m = group_filter_metrics(counts, 10)
        assert m["pipeline/group_zero_variance_ratio"] == pytest.approx(0.8)
        assert m["pipeline/group_drop_reason/zero_variance"] == pytest.approx(0.5)


class TestAsyncPipelineGauges:
    """``AsyncRolloutPipeline.sample_pipeline_gauges`` — the train-step-
    boundary sampling point for the two gauges that cannot ride on a
    ``BatchItem``. Both were computed and discarded before this landed."""

    @pytest.mark.asyncio
    async def test_buffer_utilization_is_qsize_over_maxsize(self):
        pipeline = _make_pipeline(max_pending=8)
        await pipeline._buffer.put("a", rollout_index=0)
        await pipeline._buffer.put("b", rollout_index=1)

        gauges = pipeline.sample_pipeline_gauges()

        assert gauges["pipeline/buffer_utilization"] == 2 / 8

    @pytest.mark.asyncio
    async def test_group_drop_ratio_is_a_per_step_delta_not_cumulative(self):
        # Window 1: 4 rollouts finished, 1 of them dropped.
        pipeline = _make_pipeline()
        pipeline._completed_dispatches = 4
        pipeline._dropped_dispatches = 1
        assert pipeline.sample_pipeline_gauges()["pipeline/group_drop_ratio"] == 0.25

        # Window 2: 4 more finished, all 4 dropped. A cumulative rate would
        # read 5/8 = 0.625 and smear the burst; the delta reports it as 1.0.
        pipeline._completed_dispatches = 8
        pipeline._dropped_dispatches = 5
        assert pipeline.sample_pipeline_gauges()["pipeline/group_drop_ratio"] == 1.0

    @pytest.mark.asyncio
    async def test_group_drop_ratio_omitted_when_nothing_finished(self):
        # A quiet step must read as "no data", not as "no drops" — a
        # fabricated 0.0 would dilute the series it exists to warn on.
        pipeline = _make_pipeline()
        pipeline._completed_dispatches = 4
        pipeline._dropped_dispatches = 4
        pipeline.sample_pipeline_gauges()

        assert "pipeline/group_drop_ratio" not in pipeline.sample_pipeline_gauges()

    @pytest.mark.asyncio
    async def test_reason_shares_are_a_per_step_delta_too(self):
        pipeline = _make_pipeline()
        pipeline._dispatcher.outcome_counts.update(
            {GroupFilterOutcome.KEPT: 3, GroupFilterOutcome.ZERO_VARIANCE: 1}
        )
        g = pipeline.sample_pipeline_gauges()
        assert g["pipeline/group_drop_reason/zero_variance"] == 0.25

        # Second window: only structural drops. The zero-variance share must
        # fall back to 0 rather than carry the first window's tally forward.
        pipeline._dispatcher.outcome_counts.update(
            {GroupFilterOutcome.TOO_FEW_VALID_TRAJECTORIES: 2}
        )
        g = pipeline.sample_pipeline_gauges()
        assert g["pipeline/group_drop_reason/too_few_valid_trajectories"] == 1.0
        assert g["pipeline/group_drop_reason/zero_variance"] == 0.0

    @pytest.mark.asyncio
    async def test_dispatcher_tallies_outcomes_so_the_rate_survives_keep_prob_one(self):
        """The wiring test: run real dispatches whose rewards all tie, at
        the default keep_prob=1.0. Nothing is dropped, so group_drop_ratio
        stays 0 and only the zero-variance RATE reports anything."""
        pipeline = _make_pipeline(samples_per_task=2, max_pending=8)
        for i, rewards in enumerate([[1.0, 1.0], [1.0, 1.0], [0.0, 1.0]]):
            pipeline._agent.group_rollout = AsyncMock(
                return_value=(_make_test_group(rewards), {})
            )
            for _ in range(2):
                await pipeline._semaphore.acquire()
            await pipeline._dispatcher.dispatch(
                task_data=None, rollout_batch=0, task_idx=i, rollout_index=i
            )

        g = pipeline.sample_pipeline_gauges()
        assert g["pipeline/group_zero_variance_ratio"] == pytest.approx(2 / 3)
        assert g["pipeline/group_drop_reason/zero_variance"] == 0.0

    @pytest.mark.asyncio
    async def test_sync_pipeline_reports_no_gauges(self):
        # Sync has no inter-step buffer and already logs its own drop ratio
        # from inside its accumulate loop; reporting here would double-count.
        pipeline = SyncRolloutPipeline(
            agent=_sync_fake_agent(),
            config=_make_sync_config(),
            train_dataset=_SyncFakeDataset(list(range(20))),
            eval_dataset=None,
            eval_agent=None,
            ml_logger=MagicMock(),
        )

        assert pipeline.sample_pipeline_gauges() == {}


class TestAsyncRolloutPipelineProducerLoop:
    @pytest.mark.asyncio
    async def test_run_dispatches_and_buffers_batches(self):
        """Spawn run(); consume 3 batches via next_batch; verify all 6
        rollouts arrive in order and aggregator.log_batch was invoked
        3 times."""
        pipeline = _make_pipeline(
            train_batches=[[0, 1], [2, 3], [4, 5]],
            max_pending=8,
            samples_per_task=2,
        )
        with patch.object(
            pipeline._aggregator, "log_batch", new=AsyncMock()
        ) as log_batch_mock:
            run_task = asyncio.create_task(pipeline.run())
            # Consume the 3 batches (2 rollouts each). next_batch returns
            # (batch_items, train_iter_position); pos monotonically advances.
            batch_0, pos_0 = await pipeline.next_batch(2)
            batch_1, pos_1 = await pipeline.next_batch(2)
            batch_2, pos_2 = await pipeline.next_batch(2)
            # Trigger shutdown.
            pipeline.stop()
            try:
                await asyncio.wait_for(run_task, timeout=5.0)
            except asyncio.TimeoutError:
                run_task.cancel()
                raise
            # All 6 rollouts arrived in order.
            all_batches = batch_0 + batch_1 + batch_2
            assert len(all_batches) == 6
            # aggregator.log_batch was called at least 3 times (once per
            # batch we consumed). Producer wraps train_iter and may have
            # started additional batches before stop() fired.
            assert log_batch_mock.await_count >= 3
            # Positions are valid resume markers (non-negative ints).
            # NOT asserting monotonic order: train_iter wraps on
            # StopIteration, so under a fast producer pos can reset to
            # a lower value between successive next_batch calls -- that
            # was always the semantic; the resume marker is the position
            # AT batch-emit time, not a cumulative counter.
            assert pos_0 >= 0 and pos_1 >= 0 and pos_2 >= 0

    @pytest.mark.asyncio
    async def test_active_dispatches_tracks_in_flight_count(self):
        """Block the mock agent's group_rollout on a controllable event;
        verify _active_dispatches climbs to 2 (samples_per_task * 1 task)
        then drains to 0 when the block releases."""
        gate_event = asyncio.Event()

        async def blocking_group_rollout(*args, **kwargs):
            await gate_event.wait()
            return _make_test_group([0.0, 1.0]), {}

        agent = MagicMock()
        agent.group_rollout = blocking_group_rollout
        pipeline = _make_pipeline(
            train_batches=[[0]],  # 1 task
            agent=agent,
            max_pending=4,
            samples_per_task=2,
        )
        run_task = asyncio.create_task(pipeline.run())
        # Give producer a chance to acquire semaphore + create_task.
        await asyncio.sleep(0.1)
        # At least 1 in-flight dispatch (the producer wraps train_iter
        # and dispatches until the gate/semaphore blocks it).
        assert pipeline._active_dispatches >= 1
        # Release the block.
        gate_event.set()
        # Drain a batch.
        await pipeline.next_batch(1)
        pipeline.stop()
        try:
            await asyncio.wait_for(run_task, timeout=5.0)
        except asyncio.TimeoutError:
            run_task.cancel()
            raise
        # Counter drained.
        assert pipeline._active_dispatches == 0

    @pytest.mark.asyncio
    async def test_run_exits_cleanly_on_stop(self):
        """Spawn run(); call stop() immediately; verify run() returns
        within a bounded wait (asyncio.sleep patched to no-op so prober's
        120s doesn't gate shutdown)."""
        pipeline = _make_pipeline(train_batches=[[0]])

        # Patch asyncio.sleep inside the pipeline module so the prober's
        # 120s sleep returns immediately.
        original_sleep = asyncio.sleep

        async def fast_sleep(delay, *args, **kwargs):
            # Yield to let other tasks run, but skip the delay.
            await original_sleep(0)

        with patch("pipeline.async_rollout_pipeline.asyncio.sleep", new=fast_sleep):
            pipeline.stop()
            run_task = asyncio.create_task(pipeline.run())
            # Should exit within 2s even without any next_batch call.
            await asyncio.wait_for(run_task, timeout=2.0)

    @pytest.mark.asyncio
    async def test_run_exits_when_stopped_while_producer_blocked_on_backpressure(
        self,
    ):
        """Regression: after the trainer stops consuming, the producer
        can be parked inside ``wait_until_schedulable`` on a cursor
        that will never advance. Verify ``stop()`` unblocks it and
        ``run()`` returns within a bounded wait instead of deadlocking.

        Repro shape mirrors the on-VM smoke that surfaced this bug:
        tight buffer (max_pending=1) + samples_per_task large enough
        that a single batch fills the buffer, so the producer parks on
        backpressure the moment we stop consuming.
        """
        pipeline = _make_pipeline(
            train_batches=[[0], [1], [2], [3]],  # more than buffer capacity
            max_pending=1,
            samples_per_task=1,
            # Keep tasks_per_train_step <= max_pending so the pipeline's
            # deadlock-guard assertion (tasks_per_train_step <=
            # max_pending_rollout_tasks) holds; this test only exercises
            # backpressure-unblock-on-stop, not the guard.
            tasks_per_train_step=1,
        )
        original_sleep = asyncio.sleep

        async def fast_sleep(delay, *args, **kwargs):
            await original_sleep(0)

        with patch("pipeline.async_rollout_pipeline.asyncio.sleep", new=fast_sleep):
            run_task = asyncio.create_task(pipeline.run())
            # Consume one batch so the producer starts wrapping and
            # eventually parks on wait_until_schedulable for the next
            # rollout_index.
            await pipeline.next_batch(1)
            # Give producer time to park on backpressure.
            await asyncio.sleep(0.2)
            # Trigger shutdown. With the fix, the producer's blocked
            # await races _stop_event and unwinds promptly. Without the
            # fix, the producer stays parked forever and this wait
            # times out.
            pipeline.stop()
            await asyncio.wait_for(run_task, timeout=2.0)

    @pytest.mark.asyncio
    async def test_run_exits_when_stopped_while_producer_blocked_on_semaphore(
        self,
    ):
        """Companion to the backpressure regression: the producer can
        also park on ``semaphore.acquire()`` when all permits are held
        by in-flight dispatches. Verify ``stop()`` unblocks that too.

        Repro shape: monkey-patch the shared semaphore to a fully-drained
        one BEFORE spawning run(). The producer then parks on the very
        first ``semaphore.acquire()`` because no permits are available.
        Without the fix, ``stop()`` doesn't wake the acquire; with it,
        the acquire races _stop_event and unwinds.
        """
        pipeline = _make_pipeline(
            train_batches=[[0]],
            max_pending=8,
            samples_per_task=1,
        )
        # Drain every permit so producer parks on the first acquire.
        drained = asyncio.Semaphore(value=0)
        pipeline._semaphore = drained
        # DispatchGate still holds the old semaphore reference on the
        # dispatcher, but we don't await any dispatches here -- the
        # producer never gets past the acquire, which is what we're
        # testing.
        original_sleep = asyncio.sleep

        async def fast_sleep(delay, *args, **kwargs):
            await original_sleep(0)

        with patch("pipeline.async_rollout_pipeline.asyncio.sleep", new=fast_sleep):
            run_task = asyncio.create_task(pipeline.run())
            # Give producer time to park on acquire.
            await asyncio.sleep(0.1)
            pipeline.stop()
            await asyncio.wait_for(run_task, timeout=2.0)


class TestAsyncRolloutPipelineEval:
    @pytest.mark.asyncio
    async def test_schedule_eval_spawns_locked_task(self):
        """schedule_eval creates a task registered in _evaluation_tasks
        that holds _evaluation_lock while running."""
        # eval_dataset yields one batch of 2 tasks.
        eval_dataset = _FakeDataset([[10, 11]], dataset_size=2)
        pipeline = _make_pipeline(eval_dataset=eval_dataset)

        # Replace EvalPipeline.run to observe that the lock is held during eval.
        lock_held_during_eval = {"held": False}

        async def fake_run(step_num=0, *, on_group=None):
            lock_held_during_eval["held"] = pipeline._evaluation_lock.locked()
            return {}

        pipeline._eval_pipeline.run = fake_run

        pipeline.schedule_eval(0)
        # Task registered.
        assert len(pipeline._evaluation_tasks) == 1
        # Wait for eval to finish.
        eval_task = next(iter(pipeline._evaluation_tasks))
        await eval_task
        # Lock was held during the eval.
        assert lock_held_during_eval["held"] is True
        # After completion, task is discarded.
        assert len(pipeline._evaluation_tasks) == 0

    @pytest.mark.asyncio
    async def test_multiple_schedule_evals_serialize(self):
        """Two consecutive schedule_eval calls: the second waits for the
        first to finish (via _evaluation_lock)."""
        eval_dataset = _FakeDataset([[10]], dataset_size=1)
        pipeline = _make_pipeline(eval_dataset=eval_dataset)

        call_order = []
        eval1_gate = asyncio.Event()

        async def fake_run(step_num=0, *, on_group=None):
            call_order.append(("start", step_num))
            if step_num == 1:
                await eval1_gate.wait()
            call_order.append(("end", step_num))
            return {}

        pipeline._eval_pipeline.run = fake_run

        # eval 1 will block on eval1_gate.
        pipeline.schedule_eval(1)
        pipeline.schedule_eval(2)
        # Give both tasks a chance to start.
        await asyncio.sleep(0.05)
        # Only eval 1 has started; eval 2 is waiting on the lock.
        assert call_order == [("start", 1)]
        # Release eval 1.
        eval1_gate.set()
        # Both tasks finish.
        await asyncio.gather(*pipeline._evaluation_tasks)
        assert call_order == [
            ("start", 1),
            ("end", 1),
            ("start", 2),
            ("end", 2),
        ]

    @pytest.mark.asyncio
    async def test_run_drains_pending_eval_on_exit(self):
        """schedule_eval at pipeline start; call stop(); verify
        run() awaits the pending eval before returning."""
        eval_dataset = _FakeDataset([[10]], dataset_size=1)
        pipeline = _make_pipeline(eval_dataset=eval_dataset)

        eval_started = asyncio.Event()
        eval_release = asyncio.Event()

        async def fake_run(step_num=0, *, on_group=None):
            eval_started.set()
            await eval_release.wait()
            return {}

        pipeline._eval_pipeline.run = fake_run

        original_sleep = asyncio.sleep

        async def fast_sleep(delay, *args, **kwargs):
            await original_sleep(0)

        with patch("pipeline.async_rollout_pipeline.asyncio.sleep", new=fast_sleep):
            pipeline.schedule_eval(0)
            await eval_started.wait()  # eval is running
            pipeline.stop()
            run_task = asyncio.create_task(pipeline.run())
            # run() should be blocked waiting on the eval to finish.
            await asyncio.sleep(0.1)
            assert not run_task.done()
            # Release the eval.
            eval_release.set()
            await asyncio.wait_for(run_task, timeout=2.0)

    @pytest.mark.asyncio
    async def test_schedule_checkpoint_eval_builds_backend_from_endpoint_path(
        self, monkeypatch
    ):
        """The checkpoint sweep's backend takes project+location from the
        endpoint resource path (NOT the training backend), overrides the
        generation budget, and logs under eval_ckpt/*."""
        from backends.backend_config import GeminiApiBackendConfig
        import pipeline.rollout_pipeline as rp

        eval_dataset = _FakeDataset([[10, 11]], dataset_size=2)
        pipeline = _make_pipeline(eval_dataset=eval_dataset)

        # Enrich the config with the fields schedule_checkpoint_eval reads.
        # (The shared _make_test_config fixture omits task/reward/backend.)
        # The whole config MUST be a real dataclass because the
        # implementation uses dataclasses.replace() to build the sweep config
        # (overriding agent + backend) and the agent config.
        import dataclasses as _dc

        @_dc.dataclass(frozen=True)
        class _AgentCfg:
            thinking_level: str | None = None
            max_generation_tokens: int | None = None

            def make(self, **kw):
                return _make_mock_agent()

        eval_sweep_dataset = _FakeDataset([[10, 11]], dataset_size=2)

        @_dc.dataclass(frozen=True)
        class _EvalCfg:
            eval_samples_per_task: int = 1
            max_eval_items: int = -1
            checkpoint_eval_max_generation_tokens: int = 65535

        @_dc.dataclass(frozen=True)
        class _RunCfg:
            eval: object
            agent: object
            reward: object
            task: object
            backend: object
            rollout: object

        pipeline._config = _RunCfg(
            eval=_EvalCfg(),
            agent=_AgentCfg(),
            reward=SimpleNamespace(make=lambda: MagicMock()),
            task=SimpleNamespace(make_eval=lambda: eval_sweep_dataset),
            backend=SimpleNamespace(service_account="sa@x.iam"),
            rollout=SimpleNamespace(max_concurrent_samples=4),
        )

        captured: dict = {}

        class _FakeEvalPipeline:
            def __init__(self, *, agent, config, ml_logger, eval_dataset, **kw):
                captured["config"] = config
                captured["eval_source"] = kw.get("eval_source")

            async def run(self, step_num=0, *, metric_prefix="eval", **kw):
                captured["step_num"] = step_num
                captured["metric_prefix"] = metric_prefix
                return {"learning/sampling_accuracy": 0.5}

        # Fake bundle returned by GeminiApiBackendConfig.make().
        fake_bundle = SimpleNamespace(
            sampler=MagicMock(),
            job_id="ENDID",
            aclose=AsyncMock(),
        )

        async def _fake_make(self):
            captured["backend"] = self
            return fake_bundle

        monkeypatch.setattr(rp, "EvalPipeline", _FakeEvalPipeline)
        monkeypatch.setattr(GeminiApiBackendConfig, "make", _fake_make)

        endpoint = "projects/PNUM/locations/us/endpoints/ENDID"
        pipeline.schedule_checkpoint_eval(30, endpoint)
        assert len(pipeline._evaluation_tasks) == 1
        await next(iter(pipeline._evaluation_tasks))

        built = captured["backend"]
        assert isinstance(built, GeminiApiBackendConfig)
        assert built.project == "PNUM"
        assert built.location == "us"
        assert built.endpoint_id == "ENDID"
        assert built.service_account == "sa@x.iam"
        assert captured["metric_prefix"] == "eval_ckpt"
        assert captured["step_num"] == 30
        # Generation budget overridden on the sweep's agent config.
        assert captured["config"].agent.max_generation_tokens == 65535
        # Labels its trajectories so consumers can tell this sweep from
        # the live-sampler one running at the same step.
        assert captured["eval_source"] == "checkpoint"
        # Bundle closed after the sweep.
        fake_bundle.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_schedule_checkpoint_eval_skips_unparseable_endpoint(self):
        """A malformed endpoint path is a no-op (no task spawned)."""
        eval_dataset = _FakeDataset([[10]], dataset_size=1)
        pipeline = _make_pipeline(eval_dataset=eval_dataset)
        pipeline.schedule_checkpoint_eval(30, "not-a-resource-path")
        assert len(pipeline._evaluation_tasks) == 0

    @pytest.mark.asyncio
    async def test_schedule_base_model_eval_targets_publisher_model(
        self, monkeypatch
    ):
        """The step-0 sweep targets the BASE model: no endpoint_id, model and
        project/location taken from the training backend, still eval_ckpt/*."""
        from backends.backend_config import GeminiApiBackendConfig
        import pipeline.rollout_pipeline as rp
        import dataclasses as _dc

        eval_dataset = _FakeDataset([[10, 11]], dataset_size=2)
        pipeline = _make_pipeline(eval_dataset=eval_dataset)

        @_dc.dataclass(frozen=True)
        class _AgentCfg:
            thinking_level: str | None = None
            max_generation_tokens: int | None = None

            def make(self, **kw):
                return _make_mock_agent()

        @_dc.dataclass(frozen=True)
        class _EvalCfg:
            eval_samples_per_task: int = 1
            max_eval_items: int = -1
            checkpoint_eval_max_generation_tokens: int = 65535

        @_dc.dataclass(frozen=True)
        class _RunCfg:
            eval: object
            agent: object
            reward: object
            task: object
            backend: object
            rollout: object

        pipeline._config = _RunCfg(
            eval=_EvalCfg(),
            agent=_AgentCfg(),
            reward=SimpleNamespace(make=lambda: MagicMock()),
            task=SimpleNamespace(
                make_eval=lambda: _FakeDataset([[10, 11]], dataset_size=2)
            ),
            backend=SimpleNamespace(
                service_account="sa@x.iam",
                project="my-proj",
                location="us-central1",
                vertex_base_model="gemini-3.5-flash",
            ),
            rollout=SimpleNamespace(max_concurrent_samples=4),
        )

        captured: dict = {}

        class _FakeEvalPipeline:
            def __init__(self, *, agent, config, ml_logger, eval_dataset, **kw):
                captured["config"] = config

            async def run(self, step_num=0, *, metric_prefix="eval", **kw):
                captured["step_num"] = step_num
                captured["metric_prefix"] = metric_prefix
                return {"learning/sampling_accuracy": 0.5}

        fake_bundle = SimpleNamespace(
            sampler=MagicMock(), job_id="BASE", aclose=AsyncMock()
        )

        async def _fake_make(self):
            captured["backend"] = self
            return fake_bundle

        monkeypatch.setattr(rp, "EvalPipeline", _FakeEvalPipeline)
        monkeypatch.setattr(GeminiApiBackendConfig, "make", _fake_make)

        pipeline.schedule_base_model_eval(0)
        assert len(pipeline._evaluation_tasks) == 1
        await next(iter(pipeline._evaluation_tasks))

        built = captured["backend"]
        assert isinstance(built, GeminiApiBackendConfig)
        assert built.model == "gemini-3.5-flash"
        assert built.endpoint_id == ""  # "" -> publisher (base) model
        assert built.project == "my-proj"
        # Regression: the training backend's region (us-central1 here) must
        # NOT be inherited. Publisher models are not served from every
        # region -- doing so returned HTTP 404 "Publisher model
        # projects/P/locations/us-central1/publishers/google/models/
        # gemini-3.5-flash was not found". The publisher realm is `global`.
        assert built.location == "global", (
            "base-model sweep must target the publisher realm, not the "
            f"training backend's region (got {built.location!r})"
        )
        assert captured["metric_prefix"] == "eval_ckpt"
        assert captured["step_num"] == 0
        assert captured["config"].agent.max_generation_tokens == 65535
        fake_bundle.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_schedule_base_model_eval_noop_without_base_model(self):
        """A backend that names no base model (e.g. eval-only) is a no-op."""
        eval_dataset = _FakeDataset([[10]], dataset_size=1)
        pipeline = _make_pipeline(eval_dataset=eval_dataset)
        pipeline._config = SimpleNamespace(backend=SimpleNamespace(project="p"))
        pipeline.schedule_base_model_eval(0)
        assert len(pipeline._evaluation_tasks) == 0

    @pytest.mark.asyncio
    async def test_checkpoint_sweep_reuses_trajectory_callback_and_observer(
        self, monkeypatch
    ):
        """The sweep's one-off agent must inherit, from the training side:
        the trajectory callback (ATIF writer + progress sink), the progress
        observer, and the training job id.

        Without the first two its trajectories are never persisted and never
        reach the observer. Without the third, its session_ids -- and hence
        its artifact dirs -- carry the sweep bundle's job_id (the endpoint id,
        or the model name for the base-model sweep) instead of the run's, so
        one run's artifacts self-describe under several different ids.
        """
        from backends.backend_config import GeminiApiBackendConfig
        import pipeline.rollout_pipeline as rp
        import dataclasses as _dc

        sentinel_callback = MagicMock(name="trajectory_callback")
        sentinel_observer = MagicMock(name="progress_observer")

        agent = _make_mock_agent()
        agent._trajectory_callback = sentinel_callback
        agent.tuning_job_id = "TRAINING-JOB-42"
        pipeline = _make_pipeline(
            agent=agent, eval_dataset=_FakeDataset([[10, 11]], dataset_size=2)
        )
        pipeline.set_progress_observer(sentinel_observer)

        captured: dict = {}

        @_dc.dataclass(frozen=True)
        class _AgentCfg:
            thinking_level: str | None = None
            max_generation_tokens: int | None = None

            def make(self, **kw):
                captured["atif_writer"] = kw.get("atif_writer")
                captured["job_id"] = kw.get("job_id")
                return _make_mock_agent()

        @_dc.dataclass(frozen=True)
        class _EvalCfg:
            eval_samples_per_task: int = 1
            max_eval_items: int = -1
            checkpoint_eval_max_generation_tokens: int = 65535

        @_dc.dataclass(frozen=True)
        class _RunCfg:
            eval: object
            agent: object
            reward: object
            task: object
            backend: object
            rollout: object

        pipeline._config = _RunCfg(
            eval=_EvalCfg(),
            agent=_AgentCfg(),
            reward=SimpleNamespace(make=lambda: MagicMock()),
            task=SimpleNamespace(
                make_eval=lambda: _FakeDataset([[10, 11]], dataset_size=2)
            ),
            backend=SimpleNamespace(service_account="sa@x.iam"),
            rollout=SimpleNamespace(max_concurrent_samples=4),
        )

        class _FakeEvalPipeline:
            def __init__(self, *, agent, config, ml_logger, eval_dataset, **kw):
                self._progress_observer = None

            async def run(self, step_num=0, *, metric_prefix="eval", **kw):
                captured["observer"] = self._progress_observer
                return {"learning/sampling_accuracy": 0.5}

        async def _fake_make(self):
            return SimpleNamespace(
                sampler=MagicMock(), job_id="ENDID", aclose=AsyncMock()
            )

        monkeypatch.setattr(rp, "EvalPipeline", _FakeEvalPipeline)
        monkeypatch.setattr(GeminiApiBackendConfig, "make", _fake_make)

        pipeline.schedule_checkpoint_eval(30, "projects/P/locations/us/endpoints/E")
        await next(iter(pipeline._evaluation_tasks))

        assert captured["atif_writer"] is sentinel_callback
        assert captured["observer"] is sentinel_observer
        # Not the sweep bundle's job_id ("ENDID"), which would put this
        # run's sweep artifacts under a different id than its training ones.
        assert captured["job_id"] == "TRAINING-JOB-42"


# ---------------------------------------------------------------------------
# SyncRolloutPipeline tests
# ---------------------------------------------------------------------------


def _make_sync_config(
    *,
    samples_per_task=2,
    max_concurrent_samples=8,
    zero_variance_keep_prob=1.0,
    max_transient_failure_fraction=None,
    max_eval_items=-1,
    eval_samples_per_task=1,
):
    """Minimal composed ``RunConfig``-shaped SyncRolloutPipeline config.

    The sync pipeline reads the composed ``RunConfig`` shape: rollout
    knobs (trajectory_group_filter, max_concurrent_samples,
    samples_per_task) under ``config.rollout``, eval.max_eval_items under
    ``config.eval``, and ``thinking_level`` under ``config.agent``.
    """
    from core.atif_trajectory import TrajectoryGroupFilterConfig

    return SimpleNamespace(
        rollout=SimpleNamespace(
            samples_per_task=samples_per_task,
            trajectory_group_filter=TrajectoryGroupFilterConfig(
                zero_variance_keep_prob=zero_variance_keep_prob,
                max_transient_failure_fraction=max_transient_failure_fraction,
            ),
            max_concurrent_samples=max_concurrent_samples,
        ),
        eval=SimpleNamespace(
            eval_samples_per_task=eval_samples_per_task,
            max_eval_items=max_eval_items,
        ),
        agent=SimpleNamespace(thinking_level=None, max_generation_tokens=None),
    )


class _SyncFakeIter:
    """Duck-typed BatchIterator for SyncRolloutPipeline.

    Sync uses ``batch_size=1`` for the train iter (wraps forever) and
    ``max_items=N, repeat=False`` for the eval iter (StopIteration
    after one pass). ``current`` mimics JsonlDataset's per-item
    position marker that sync mutates directly at end-of-step.
    """

    def __init__(self, items, repeat=True, batch_size=1):
        self._items = list(items)
        self._i = 0
        self._batch_size = batch_size
        self._repeat = repeat
        self.current = 0
        self.dataset = self._items

    def __iter__(self):
        return self

    def __next__(self):
        if not self._items:
            raise StopIteration
        if self._i >= len(self._items):
            if not self._repeat:
                raise StopIteration
            self._i = 0
        end = min(self._i + self._batch_size, len(self._items))
        batch = self._items[self._i : end]
        self._i = end
        self.current = self._i % len(self._items)
        return batch

    def reset(self):
        self._i = 0
        self.current = 0


class _SyncFakeDataset:
    """Duck-typed dataset for SyncRolloutPipeline: exposes ``build_iter``.

    Sync's build_iter is called with ``batch_size=1`` for the train
    iterator and ``max_items=cfg.max_eval_items, repeat=False`` for the
    eval iterator.
    """

    def __init__(self, items):
        self._items = list(items)

    def __len__(self):
        return len(self._items)

    def build_iter(self, **kwargs):
        max_items = kwargs.get("max_items")
        batch_size = kwargs.get("batch_size")
        repeat = kwargs.get("repeat", True)
        if max_items == 0 or len(self._items) == 0:
            return None
        # Eval-shaped call: yields ALL items in one big list of length max_items.
        if max_items is not None:
            batch_size = (
                len(self._items) if max_items < 0 else min(max_items, len(self._items))
            )
        it = _SyncFakeIter(self._items, repeat=repeat, batch_size=batch_size or 1)
        pos = kwargs.get("resume_from_position")
        if pos is not None:
            it.current = pos
            it._i = pos
        return it


def _sync_fake_agent(*, per_group_reward_variance=True):
    """Sync group_rollout returns (TrajectoryGroup, metrics).

    ``per_group_reward_variance=True`` (default) yields non-zero variance
    so the TrajectoryGroupFilter keeps the group at
    ``zero_variance_keep_prob=1.0``. ``False`` yields all-equal rewards
    that get dropped under variance filtering.
    """
    from core.metric_spec import MetricValue

    rewards = [0.0, 1.0] if per_group_reward_variance else [0.0, 0.0]
    group = _make_test_group(rewards)
    # Use a spec-registered metric key so aggregate_by_spec can reduce it.
    metrics = {
        "learning/sampling_accuracy": MetricValue.of_scalar(sum(rewards) / len(rewards))
    }
    agent = MagicMock()
    agent.group_rollout = AsyncMock(return_value=(group, metrics))
    return agent


class TestSyncRolloutPipelineNextBatch:
    """Contract tests for SyncRolloutPipeline.next_batch."""

    @pytest.mark.asyncio
    async def test_accumulates_requested_n(self):
        """next_batch(n) returns exactly n valid groups when the agent
        yields non-zero-variance rewards on every call."""
        agent = _sync_fake_agent(per_group_reward_variance=True)
        pipeline = SyncRolloutPipeline(
            agent=agent,
            config=_make_sync_config(),
            train_dataset=_SyncFakeDataset(list(range(20))),
            eval_dataset=None,
            eval_agent=None,
            ml_logger=MagicMock(),
        )
        batch, position = await pipeline.next_batch(4)
        assert len(batch) == 4
        assert all(isinstance(b, BatchItem) for b in batch)
        assert position >= 0

    @pytest.mark.asyncio
    async def test_returns_train_iter_position(self):
        """next_batch returns (batch, position) where position matches
        self._train_iter.current after the step."""
        agent = _sync_fake_agent()
        pipeline = SyncRolloutPipeline(
            agent=agent,
            config=_make_sync_config(),
            train_dataset=_SyncFakeDataset(list(range(20))),
            eval_dataset=None,
            eval_agent=None,
            ml_logger=MagicMock(),
        )
        _, position = await pipeline.next_batch(2)
        assert position == pipeline._train_iter.current

    @pytest.mark.asyncio
    async def test_dynamic_sampling_cap_prevents_infinite_loop(self):
        """When every group is dropped (variance filter kills them all),
        next_batch bails out at the max_tasks_per_step cap rather than
        looping forever."""
        agent = _sync_fake_agent(per_group_reward_variance=False)
        pipeline = SyncRolloutPipeline(
            agent=agent,
            config=_make_sync_config(zero_variance_keep_prob=0.0),
            train_dataset=_SyncFakeDataset(list(range(8))),
            eval_dataset=None,
            eval_agent=None,
            ml_logger=MagicMock(),
        )
        # asyncio.wait_for catches a runaway loop; the cap should trip well
        # before the timeout.
        batch, _ = await asyncio.wait_for(pipeline.next_batch(4), timeout=5.0)
        # Batch may be short (0..n) because everything got dropped.
        assert len(batch) <= 4

    @pytest.mark.asyncio
    async def test_step_num_logs_pipeline_metrics(self):
        """When step_num is provided, ml_logger.log_metrics is called
        once with the pipeline-side per-step metrics keyed at that step."""
        agent = _sync_fake_agent()
        ml_logger = MagicMock()
        pipeline = SyncRolloutPipeline(
            agent=agent,
            config=_make_sync_config(),
            train_dataset=_SyncFakeDataset(list(range(20))),
            eval_dataset=None,
            eval_agent=None,
            ml_logger=ml_logger,
        )
        await pipeline.next_batch(2, step_num=7)
        assert ml_logger.log_metrics.called
        # At least one call should be keyed at step=7 with train/* keys.
        step_7_calls = [
            c for c in ml_logger.log_metrics.call_args_list if c.kwargs.get("step") == 7
        ]
        assert step_7_calls, "expected log_metrics(step=7) call"
        merged = {}
        for c in step_7_calls:
            merged.update(c.args[0] if c.args else c.kwargs.get("metrics", {}))
        assert "train/running_pass_rate" in merged
        assert "train/rollout_rounds" in merged
        # Iterator position is emitted under the train/pipeline/* prefix.
        assert "train/pipeline/iterator_current" in merged
        assert "train/iterator_current" not in merged
        # Aggregated env/learning metrics are emitted at their natural
        # prefix, NOT wrapped in rollout/* (sync mode never had a
        # rollout_batch axis).
        assert "learning/sampling_accuracy" in merged
        assert not any(k.startswith("rollout/") for k in merged), (
            f"unexpected rollout/* keys: {[k for k in merged if k.startswith('rollout/')]}"
        )

    @pytest.mark.asyncio
    async def test_logs_the_same_filter_gauges_as_async(self):
        """Sync emits the group-filter breakdown itself (it reports no
        pipeline gauges), so its key set must match async's -- otherwise
        the two modes answer the same question on different dashboards."""
        # per_group_reward_variance=False -> every group ties; keep_prob=0
        # -> every tie is dropped.
        agent = _sync_fake_agent(per_group_reward_variance=False)
        ml_logger = MagicMock()
        pipeline = SyncRolloutPipeline(
            agent=agent,
            config=_make_sync_config(zero_variance_keep_prob=0.0),
            train_dataset=_SyncFakeDataset(list(range(20))),
            eval_dataset=None,
            eval_agent=None,
            ml_logger=ml_logger,
        )
        await pipeline.next_batch(2, step_num=3)
        merged = {}
        for c in ml_logger.log_metrics.call_args_list:
            if c.kwargs.get("step") == 3:
                merged.update(c.args[0] if c.args else c.kwargs.get("metrics", {}))

        assert merged["pipeline/group_drop_ratio"] == pytest.approx(1.0)
        assert merged["pipeline/group_drop_reason/zero_variance"] == pytest.approx(1.0)
        assert merged["pipeline/group_zero_variance_ratio"] == pytest.approx(1.0)
        assert merged["pipeline/group_drop_reason/too_few_valid_trajectories"] == 0.0

    @pytest.mark.asyncio
    async def test_step_num_none_skips_metric_logging(self):
        """step_num=None (test convenience) suppresses pipeline-side
        metric emission."""
        agent = _sync_fake_agent()
        ml_logger = MagicMock()
        pipeline = SyncRolloutPipeline(
            agent=agent,
            config=_make_sync_config(),
            train_dataset=_SyncFakeDataset(list(range(20))),
            eval_dataset=None,
            eval_agent=None,
            ml_logger=ml_logger,
        )
        await pipeline.next_batch(2, step_num=None)
        assert not ml_logger.log_metrics.called

    @pytest.mark.asyncio
    async def test_forwards_max_generation_tokens_to_group_rollout(self):
        """next_batch forwards config.agent.max_generation_tokens to agent.group_rollout."""
        agent = _sync_fake_agent()
        config = _make_sync_config()
        config.agent.max_generation_tokens = 8192
        pipeline = SyncRolloutPipeline(
            agent=agent,
            config=config,
            train_dataset=_SyncFakeDataset(list(range(20))),
            eval_dataset=None,
            eval_agent=None,
            ml_logger=MagicMock(),
        )
        await pipeline.next_batch(1)
        assert agent.group_rollout.called
        assert (
            agent.group_rollout.call_args.kwargs.get("max_generation_tokens")
            == 8192
        )


class TestSyncRolloutPipelineRun:
    """Lifecycle tests for run()/stop()/eval draining."""

    @pytest.mark.asyncio
    async def test_run_blocks_until_stop(self):
        """run() awaits stop() and returns promptly after."""
        pipeline = SyncRolloutPipeline(
            agent=_sync_fake_agent(),
            config=_make_sync_config(),
            train_dataset=_SyncFakeDataset([0, 1]),
            eval_dataset=None,
            eval_agent=None,
            ml_logger=MagicMock(),
        )
        run_task = asyncio.create_task(pipeline.run())
        await asyncio.sleep(0)  # let run() actually park on the wait
        assert not run_task.done()
        pipeline.stop()
        await asyncio.wait_for(run_task, timeout=1.0)

    @pytest.mark.asyncio
    async def test_run_drains_pending_evals(self):
        """schedule_eval + stop before eval finishes; run() waits for
        the eval task to complete before returning."""
        eval_dataset = _SyncFakeDataset([100])
        pipeline = SyncRolloutPipeline(
            agent=_sync_fake_agent(),
            config=_make_sync_config(max_eval_items=1),
            train_dataset=_SyncFakeDataset([0, 1, 2]),
            eval_dataset=eval_dataset,
            eval_agent=None,
            ml_logger=MagicMock(),
        )

        eval_started = asyncio.Event()
        eval_release = asyncio.Event()

        async def fake_run(step_num=0, *, on_group=None):
            eval_started.set()
            await eval_release.wait()
            return {}

        pipeline._eval_pipeline.run = fake_run

        pipeline.schedule_eval(0)
        await eval_started.wait()
        pipeline.stop()
        run_task = asyncio.create_task(pipeline.run())
        await asyncio.sleep(0.05)
        assert not run_task.done()
        eval_release.set()
        await asyncio.wait_for(run_task, timeout=2.0)

    @pytest.mark.asyncio
    async def test_schedule_eval_deferred_await_blocks_next_batch(self):
        """schedule_eval spawns a background task; the NEXT next_batch
        call awaits it before starting its accumulate loop. Preserves
        the pre-unification 'eval blocks next training step' semantics
        of the sync loop."""
        eval_dataset = _SyncFakeDataset([100])
        pipeline = SyncRolloutPipeline(
            agent=_sync_fake_agent(),
            config=_make_sync_config(max_eval_items=1),
            train_dataset=_SyncFakeDataset(list(range(20))),
            eval_dataset=eval_dataset,
            eval_agent=None,
            ml_logger=MagicMock(),
        )

        eval_release = asyncio.Event()

        async def slow_run(step_num=0, *, on_group=None):
            await eval_release.wait()
            return {}

        pipeline._eval_pipeline.run = slow_run

        pipeline.schedule_eval(0)
        # Kick off next_batch: it should park at the deferred await.
        batch_task = asyncio.create_task(pipeline.next_batch(2))
        await asyncio.sleep(0.05)
        assert not batch_task.done(), (
            "next_batch should be blocked on the pending eval task"
        )
        eval_release.set()
        batch, _ = await asyncio.wait_for(batch_task, timeout=2.0)
        assert len(batch) == 2


def test_pipeline_step_aggregation_uses_spec_strategies():
    """The pipeline's per-step aggregation reduces finish_reason via ratio and
    sample_time via pooled p95, not the legacy sum/mean convention. This is
    the regression guard against reverting to convention-based
    aggregation."""
    import numpy as np

    from core.metric_spec import MetricValue, aggregate_by_spec

    # Two groups' metric dicts as _compute_group_metrics would produce them
    # (MetricValue-carrying).
    group_metrics = [
        {
            "sampler/finish_reason/STOP": MetricValue.of_ratio(1, 2),
            "pipeline/sample_time_p95": MetricValue.of_samples([1.0, 2.0]),
        },
        {
            "sampler/finish_reason/STOP": MetricValue.of_ratio(2, 2),
            "pipeline/sample_time_p95": MetricValue.of_samples([3.0, 4.0]),
        },
    ]
    result = aggregate_by_spec(group_metrics)
    # ratio: (1+2)/(2+2) = 0.75 (not mean-of-per-group-ratios = 0.75 by luck
    # here, but the mechanics are different -- see test_metric_spec's ratio
    # tests)
    assert abs(result["sampler/finish_reason/STOP"] - 0.75) < 1e-9
    # pooled p95 over [1,2,3,4], NOT mean-of-per-group-p95
    expected_p95 = float(np.percentile([1.0, 2.0, 3.0, 4.0], 95))
    assert abs(result["pipeline/sample_time_p95"] - expected_p95) < 1e-6


def test_pipeline_step_aggregation_accepts_bare_float_dicts():
    """Recipe agents that OVERRIDE group_rollout can return
    dict[str, float] directly rather than routing through
    ``Agent._compute_group_metrics``'s MetricValue wrap step. All three
    pipeline entry points (``log_batch``, ``_log_step_sampling_metrics``,
    ``eval_only.EvalPipeline.run``) hand those dicts straight to
    ``aggregate_by_spec``, so the batch reducer must accept the raw shape
    without crashing.

    This is a regression guard for a BLOCKER: the initial
    ``aggregate_by_spec`` assumed all values were already ``MetricValue``s,
    and such a recipe would have crashed at the first log_batch call
    ("AttributeError: 'float' object has no attribute 'value'").
    """
    from core.metric_spec import aggregate_by_spec

    # Mixed shape (a recipe may emit some MetricValue from wrapping helpers,
    # some bare floats from its custom rollup). Also exercises int-valued
    # keys (a status_counts dict may emit raw ints).
    group_metrics = [
        {
            "env/average_reward": 0.5,  # bare float, unregistered -> MEAN
            "learning/sampling_accuracy": 0.3,  # bare float, registered as MEAN
            "sampler/trajectory_status/OK": 3,  # bare int, RATIO family; wrapped bare gets MEAN via fallback? No -- ratio strategy requires ratio pair.
        },
        {
            "env/average_reward": 1.5,
            "learning/sampling_accuracy": 0.7,
            "sampler/trajectory_status/OK": 1,
        },
    ]
    # The bare-int RATIO/family cases are caller error (the recipe owes an
    # of_ratio wrap for those); drop them from this smoke and only cover
    # the pure scalar-friendly shapes here. The important assertion is that
    # bare floats + ints for MEAN-friendly keys survive end-to-end.
    scalar_only = [
        {k: v for k, v in g.items() if not k.startswith("sampler/trajectory_status")}
        for g in group_metrics
    ]
    result = aggregate_by_spec(scalar_only)
    assert abs(result["env/average_reward"] - 1.0) < 1e-9
    assert abs(result["learning/sampling_accuracy"] - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# D3 acceptance: a REAL composed RunConfig constructs a pipeline via
# ``config.rollout.make(...)`` without the A-era ``config.loop`` AttributeError.
#
# This is the regression guard for the half-migration this task closes:
# the pipelines used to read ``config.loop.*`` (an A-era nested shape that
# nothing populated), so passing a real RunConfig would AttributeError.
# After the read-repoint, a registered ``word_puzzle_async`` /
# ``word_puzzle_sync`` RunConfig flows straight through ``rollout.make()``
# to the right concrete pipeline.
# ---------------------------------------------------------------------------


def _acceptance_run_config(name: str):
    """Load a registered RunConfig and make it construction-ready.

    The checked-in operating points leave ``rollout.trajectory_group_filter``
    as ``None`` (the recipe wires a real filter at launch); the pipeline
    calls ``.make()`` on it during construction, so we substitute a
    concrete filter. We also narrow the async look-ahead window so the
    ``tasks_per_train_step <= max_pending_rollout_tasks`` sizing
    invariant holds at test scale. Nothing else about the real config is
    changed — the point is that its SHAPE (train/rollout/eval/agent)
    is what the pipeline reads.
    """
    import dataclasses

    from core.atif_trajectory import TrajectoryGroupFilterConfig
    from config_registry import get_config

    config = get_config(name)
    rollout_over = dict(
        trajectory_group_filter=TrajectoryGroupFilterConfig(
            zero_variance_keep_prob=1.0,
        ),
    )
    if hasattr(config.rollout, "max_pending_rollout_tasks"):
        # Async: keep the sizing invariant satisfied at test scale.
        rollout_over["max_pending_rollout_tasks"] = config.train.tasks_per_train_step
    return dataclasses.replace(
        config, rollout=dataclasses.replace(config.rollout, **rollout_over)
    )


def test_word_puzzle_async_run_config_constructs_pipeline_via_make():
    """A real ``word_puzzle_async`` RunConfig -> ``rollout.make()`` yields an
    AsyncRolloutPipeline with no ``config.loop`` AttributeError."""
    config = _acceptance_run_config("word_puzzle/async")
    pipeline = config.rollout.make(
        run_config=config,
        agent=_make_mock_agent(),
        train_dataset=_FakeDataset([[0, 1]], dataset_size=4),
        eval_dataset=None,
        eval_agent=None,
        ml_logger=MagicMock(),
    )
    assert isinstance(pipeline, AsyncRolloutPipeline)


def test_word_puzzle_sync_run_config_constructs_pipeline_via_make():
    """A real ``word_puzzle_sync`` RunConfig -> ``rollout.make()`` yields a
    SyncRolloutPipeline with no ``config.loop`` AttributeError."""
    config = _acceptance_run_config("word_puzzle/sync")
    pipeline = config.rollout.make(
        run_config=config,
        agent=_make_mock_agent(),
        train_dataset=_FakeDataset([[0, 1]], dataset_size=4),
        eval_dataset=None,
        eval_agent=None,
        ml_logger=MagicMock(),
    )
    assert isinstance(pipeline, SyncRolloutPipeline)


# ---------------------------------------------------------------------------
# RolloutDispatcher per-trajectory permit release
# ---------------------------------------------------------------------------


def _make_dispatcher(*, samples_per_task, semaphore, agent, maxsize=100):
    """Build a RolloutDispatcher with a real InOrderBuffer + semaphore."""
    config = _make_test_config(samples_per_task=samples_per_task)
    buf = InOrderBuffer(maxsize=maxsize, start_rollout_index=0)
    return RolloutDispatcher(
        agent=agent, buffer=buf, semaphore=semaphore, config=config
    )


class _CallbackAgent:
    """Agent double whose group_rollout fires on_trajectory_done a
    controllable number of times, then optionally raises. Lets the test
    assert the dispatcher's permit-release + drain accounting directly."""

    def __init__(self, *, fire_count, rewards=(0.0, 1.0), raise_after=False):
        self._fire_count = fire_count
        self._rewards = rewards
        self._raise_after = raise_after

    async def group_rollout(
        self,
        task_data,
        num_samples,
        thinking_level=None,
        max_generation_tokens=None,
        on_trajectory_done=None,
    ):
        if on_trajectory_done is not None:
            for _ in range(self._fire_count):
                on_trajectory_done()
        if self._raise_after:
            raise RuntimeError("agent boom after firing some callbacks")
        return _make_test_group(self._rewards), {"env/average_reward": 0.5}


class TestRolloutDispatcherPermitRelease:
    """dispatch() must release permits per-trajectory (via on_trajectory_done)
    and drain any unfired permits in finally, always netting exactly
    samples_per_task releases against the N acquired up front."""

    async def test_dispatch_nets_exactly_samples_per_task_releases_on_success(self):
        # Producer acquires N up front -> sem starts at 0; dispatch must
        # bring it back to N (all released) after a successful group.
        n = 4
        sem = asyncio.Semaphore(0)
        agent = _CallbackAgent(fire_count=n)
        d = _make_dispatcher(samples_per_task=n, semaphore=sem, agent=agent)
        await d.dispatch(task_data=None, rollout_batch=0, task_idx=0, rollout_index=0)
        assert sem._value == n

    async def test_dispatch_passes_on_trajectory_done_that_releases_per_call(self):
        # The dispatcher must hand group_rollout a real callback; firing it
        # once releases one permit immediately (not batched at the end).
        n = 4
        sem = asyncio.Semaphore(0)
        seen = {}

        class _Probe(_CallbackAgent):
            async def group_rollout(self, task_data, num_samples, **kw):
                cb = kw.get("on_trajectory_done")
                assert cb is not None, "dispatcher must pass on_trajectory_done"
                cb()
                seen["after_one"] = sem._value  # one permit freed already
                cb()
                cb()
                cb()
                return _make_test_group(self._rewards), {}

        agent = _Probe(fire_count=0)
        d = _make_dispatcher(samples_per_task=n, semaphore=sem, agent=agent)
        await d.dispatch(task_data=None, rollout_batch=0, task_idx=0, rollout_index=0)
        assert seen["after_one"] == 1  # released mid-group, not all-at-end
        assert sem._value == n

    async def test_dispatch_drains_unfired_permits_when_agent_underfires(self):
        # Agent fires only 1 of N callbacks then raises: the finally drain
        # must release the remaining N-1 so the pool never leaks.
        n = 4
        sem = asyncio.Semaphore(0)
        agent = _CallbackAgent(fire_count=1, raise_after=True)
        d = _make_dispatcher(samples_per_task=n, semaphore=sem, agent=agent)
        # dispatch swallows the exception (logs + mark_dropped) and returns {}.
        await d.dispatch(task_data=None, rollout_batch=0, task_idx=0, rollout_index=0)
        assert sem._value == n  # 1 from callback + 3 from drain

    async def test_dispatch_does_not_over_release_if_agent_overfires(self):
        # Defensive: a buggy agent firing MORE than N callbacks must not
        # inflate the budget past N.
        n = 4
        sem = asyncio.Semaphore(0)
        agent = _CallbackAgent(fire_count=n + 3)
        d = _make_dispatcher(samples_per_task=n, semaphore=sem, agent=agent)
        await d.dispatch(task_data=None, rollout_batch=0, task_idx=0, rollout_index=0)
        assert sem._value == n


class TestBackgroundMetricsTasksSurviveShutdown:
    """A metrics task the loop closes under loses that batch, silently.

    Observed live 2026-09-29 as ``Task was destroyed but it is pending!
    ... MetricsAggregator.log_batch()``: the run reported no error, exited
    cleanly, and published no metrics for the batch.
    """

    @staticmethod
    def _bare_pipeline():
        """A pipeline with ONLY the task bookkeeping initialised.

        `BaseRolloutPipeline` is abstract and its `__init__` wants a full
        config; the drain touches nothing else, so a concrete subclass
        bypassing `__init__` keeps the test about the drain.
        """
        from pipeline.rollout_pipeline import BaseRolloutPipeline

        class _Pipe(BaseRolloutPipeline):
            async def next_batch(self, n, *, step_num=None):  # pragma: no cover
                raise NotImplementedError

            async def run(self, *a, **k):  # pragma: no cover
                raise NotImplementedError

        pipe = _Pipe.__new__(_Pipe)
        pipe._evaluation_tasks = set()
        pipe._metrics_tasks = set()
        return pipe

    @pytest.mark.asyncio
    async def test_the_drain_does_NOT_wait_on_metrics_tasks(self):
        """Shutdown must not block on `log_batch`'s inner rollouts.

        The metrics task's body is `gather(*rollout_tasks)`, which can
        take up to `poll_deadline_s` (10 h) if a rollout is hung on an
        unresponsive LRO. Draining it here would delay `run_training`'s
        `finally: aclose()` (which cancels the tuning job) for the same
        window, keeping the job billing after the run was supposed to
        end. `track_metrics_task` holds a strong reference so the task
        is not GC-dropped, and that is ALL it does.
        """
        import asyncio
        import time

        finished: list[str] = []

        async def _slow_metrics():
            await asyncio.sleep(5.0)  # would blow the test's wall clock
            finished.append("log_batch")

        pipe = self._bare_pipeline()
        pipe.track_metrics_task(asyncio.create_task(_slow_metrics()))
        started = time.monotonic()

        await pipe._drain_pending_evals()

        elapsed = time.monotonic() - started
        assert elapsed < 0.5, (
            f"the drain waited {elapsed:.2f}s on a metrics task; it must "
            "return within a bounded window regardless of in-flight rollouts"
        )
        assert finished == [], "the metrics task should not have finished"

    @pytest.mark.asyncio
    async def test_the_drain_still_waits_on_eval_tasks(self):
        """Eval tasks ARE drained -- they are bounded and their metrics
        must reach ATIF before shutdown closes the loop.
        """
        import asyncio

        finished: list[str] = []

        async def _eval():
            await asyncio.sleep(0.05)
            finished.append("eval")

        pipe = self._bare_pipeline()
        pipe._evaluation_tasks.add(asyncio.create_task(_eval()))

        await pipe._drain_pending_evals()
        assert finished == ["eval"]

    @pytest.mark.asyncio
    async def test_a_failing_eval_does_not_mask_shutdown(self):
        """`return_exceptions=True` -- a broken eval must not wedge exit."""
        import asyncio

        async def _boom():
            raise RuntimeError("eval exploded")

        pipe = self._bare_pipeline()
        pipe._evaluation_tasks.add(asyncio.create_task(_boom()))

        await pipe._drain_pending_evals()  # must not raise

    @pytest.mark.asyncio
    async def test_tracking_releases_the_task_when_it_completes(self):
        """The set must not grow without bound over a long run."""
        import asyncio

        async def _noop():
            return None

        pipe = self._bare_pipeline()
        pipe.track_metrics_task(asyncio.create_task(_noop()))
        assert len(pipe._metrics_tasks) == 1
        # Give the task a chance to finish naturally, then its
        # done-callback discards it from the tracking set.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not pipe._metrics_tasks, "done-callback never discarded the task"
