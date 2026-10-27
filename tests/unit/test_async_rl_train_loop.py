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

"""End-to-end tests for the async training + eval loop.

Drives the top-level ``train(config=async_config, ...)`` entry point (which
composes ``AsyncRolloutPipeline`` + ``train_and_eval_loop`` in
``pipeline.train_and_eval``). Tests use a small local ``_run_async_loop``
adapter that mirrors the recipe caller pattern — this keeps the test
cases focused on scenarios (concurrency, exception paths, resume) rather
than boilerplate.
"""

import asyncio
import collections.abc
import dataclasses
import math
from typing import Any, Dict, List, Tuple

import pytest

from core.atif_trajectory import Agent as AtifAgent, Trajectory as AtifTrajectory

from pipeline.rollout_pipeline import InOrderBuffer
from pipeline.train_and_eval import train
from config_registry import get_config
from core.run_config import RunConfig
from core.atif_trajectory import (
    ScoredTrajectory,
    Reward,
    StepStats,
    TrajectoryGroup,
    TrajectoryGroupFilterConfig,
    Step,
    TrajectoryStep as AtifTrajectoryStep,
)
from core.agent import Agent
from core.protocols import Trainer
from core.task_data import TaskData


@dataclasses.dataclass
class _TestTaskData(TaskData):
    """Lightweight TaskData subclass for tests that don't care about payload."""

    pass


async def _run_async_loop(
    *,
    train_dataset,
    eval_dataset,
    agent,
    trainer,
    ml_logger,
    config: RunConfig,
    num_train_steps: int,
    eval_every_n_train_steps: int,
) -> None:
    """Adapter around ``train(config=run_config, ...)`` that lets each test
    override ``num_train_steps`` / ``eval_every_n_train_steps`` per
    call. ``train()`` reads ``num_train_steps`` off ``config.train`` and
    ``eval_every_n_train_steps`` off ``config.eval``, so we clone the
    composed (frozen dataclass) ``RunConfig``, routing each override into
    its sub-config via nested ``dataclasses.replace``.
    """
    config = dataclasses.replace(
        config,
        train=dataclasses.replace(config.train, num_train_steps=num_train_steps),
        eval=dataclasses.replace(
            config.eval, eval_every_n_train_steps=eval_every_n_train_steps
        ),
    )
    await train(
        agent=agent,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        trainer=trainer,
        ml_logger=ml_logger,
    )


class _ListDataset(collections.abc.Sequence):
    """Duck-typed dataset stand-in for pipeline tests.

    Exposes ``build_iter`` matching the JsonlDataset shape (accepts the
    kwargs the pipeline passes: batch_size + resume_*, or max_items +
    repeat), returning a lightweight ``_SimpleIter`` over a slice.
    """

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    def build_iter(
        self,
        *,
        batch_size: int | None = None,
        max_items: int | None = None,
        repeat: bool = True,
        resume_from_position: int | None = None,
    ) -> "_SimpleIter | None":
        # Same XOR gate as JsonlDataset.build_iter.
        if (batch_size is None) == (max_items is None):
            raise ValueError("Pass exactly one of batch_size or max_items")
        if max_items is not None:
            if max_items == 0 or len(self) == 0:
                return None
            batch_size = len(self) if max_items < 0 else min(max_items, len(self))
        assert batch_size is not None
        it = _SimpleIter(self, batch_size=batch_size, repeat=repeat)
        if resume_from_position is not None:
            it.current = resume_from_position
        return it


class _SimpleIter(collections.abc.Iterator):
    """Wraps around when ``repeat=True``; raises StopIteration otherwise."""

    def __init__(self, dataset, batch_size, repeat=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.current = 0
        self.repeat = repeat

    def __next__(self):
        if self.current >= len(self.dataset):
            if not self.repeat:
                raise StopIteration
            self.current = 0
        end = min(self.current + self.batch_size, len(self.dataset))
        batch = [self.dataset[i] for i in range(self.current, end)]
        self.current = end
        return batch

    def reset(self) -> None:
        self.current = 0

    def advance(self, num_items: int) -> None:
        self.current = (self.current + num_items) % max(1, len(self.dataset))


def _traj(reward: float) -> ScoredTrajectory:
    us = Step(step_id=1, source="user", message="p")
    a = AtifTrajectoryStep.from_wire(step_id=2, generation="g", stats=StepStats())
    traj = AtifTrajectory(
        schema_version="ATIF-v1.7",
        session_id="s",
        agent=AtifAgent(name="t", version="0", model_name="m"),
        steps=[us, a],
    )
    return ScoredTrajectory(trajectory=traj, reward=Reward(reward=reward))


class _FakeAgent(Agent):
    """Agent that produces a TrajectoryGroup of N varied-reward trajectories."""

    def __init__(self):
        super().__init__(
            sampler=None,
            env_factory=None,
            recipe_name="test",
            tuning_job_id="tj-test",
        )
        self.group_calls = 0

    async def _rollout_internal(self, task_data, ctx):  # pragma: no cover
        raise NotImplementedError

    async def group_rollout(
        self,
        task_data: TaskData,
        num_samples: int,
        thinking_level: str | None = None,
        max_generation_tokens: int | None = None,
        on_trajectory_done=None,
        phase: str | None = None,
        dispatch_index: int | None = None,
        step: int | None = None,
    ) -> Tuple[TrajectoryGroup, Dict[str, Any]]:
        from core.metric_spec import MetricValue

        self.group_calls += 1
        # Fire the per-trajectory done signal once per sample (success path),
        # matching a real agent so the dispatcher releases permits per-sample.
        if on_trajectory_done is not None:
            for _ in range(num_samples):
                on_trajectory_done()
        # Produce non-zero variance rewards so the group isn't dropped.
        rewards = [1.0 if i % 2 == 0 else 0.0 for i in range(num_samples)]
        group = TrajectoryGroup(
            task_data=task_data,
            scored_trajectories=[_traj(r) for r in rewards],
        )
        # Use spec-registered metric keys so aggregate_by_spec can reduce them.
        metrics = {
            "learning/sampling_accuracy": MetricValue.of_scalar(
                sum(1 for r in rewards if r > 0) / max(1, len(rewards))
            ),
            "learning/reward/mean": MetricValue.of_weighted(
                sum(rewards) / max(1, len(rewards)), max(1, len(rewards))
            ),
            "learning/pass_at_k": MetricValue.of_scalar(
                1.0 if any(r > 0 for r in rewards) else 0.0
            ),
        }
        return group, metrics


class _FlakyAgent(Agent):
    """Agent that raises on a configurable subset of calls.

    Use to exercise the exception path in run_single_rollout.
    """

    def __init__(self, raise_on_call_indices=None):
        super().__init__(
            sampler=None,
            env_factory=None,
            recipe_name="test",
            tuning_job_id="tj-test",
        )
        self.group_calls = 0
        self.raise_on_call_indices: set[int] = set(raise_on_call_indices or ())

    async def _rollout_internal(self, task_data, ctx):  # pragma: no cover
        raise NotImplementedError

    async def group_rollout(
        self,
        task_data: TaskData,
        num_samples: int,
        thinking_level: str | None = None,
        max_generation_tokens: int | None = None,
        on_trajectory_done=None,
        phase: str | None = None,
        dispatch_index: int | None = None,
        step: int | None = None,
    ) -> Tuple[TrajectoryGroup, Dict[str, Any]]:
        from core.metric_spec import MetricValue

        call_idx = self.group_calls
        self.group_calls += 1
        if call_idx in self.raise_on_call_indices:
            # Deliberately raise BEFORE firing any callback -- a pre-fanout
            # failure. The dispatcher's finally-drain must release all N
            # permits itself; a leak here would deadlock the producer.
            raise RuntimeError(f"simulated rollout failure on call {call_idx}")
        # Success path: fire once per sample so the dispatcher releases
        # permits per-trajectory.
        if on_trajectory_done is not None:
            for _ in range(num_samples):
                on_trajectory_done()
        rewards = [1.0 if i % 2 == 0 else 0.0 for i in range(num_samples)]
        group = TrajectoryGroup(
            task_data=task_data,
            scored_trajectories=[_traj(r) for r in rewards],
        )
        # Use spec-registered metric keys so aggregate_by_spec can reduce them.
        metrics = {
            "learning/sampling_accuracy": MetricValue.of_scalar(
                sum(1 for r in rewards if r > 0) / max(1, len(rewards))
            ),
            "learning/reward/mean": MetricValue.of_weighted(
                sum(rewards) / max(1, len(rewards)), max(1, len(rewards))
            ),
            "learning/pass_at_k": MetricValue.of_scalar(
                1.0 if any(r > 0 for r in rewards) else 0.0
            ),
        }
        return group, metrics


class _FakeTrainer(Trainer):
    def __init__(self):
        self.calls: List[Tuple[int, bool]] = []

    async def train_step(
        self,
        trajectories: List[ScoredTrajectory],
        sync_trainer_and_sampler: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        self.calls.append((len(trajectories), sync_trainer_and_sampler))
        return {"loss": 0.123}


class _CapturingLogger:
    def __init__(self):
        self.metrics: List[Tuple[Dict[str, Any], int | None]] = []

    def log_metrics(self, metrics, step=None):
        self.metrics.append((dict(metrics), step))

    def log_hparams(self, *_, **__):
        pass

    def close(self):
        pass


# Small test-scale values (NOT the recipe production defaults) that keep
# the async-loop assertions valid. Each dict maps to one composed
# ``RunConfig`` sub-module; overrides route into the matching sub-config
# in ``_make_config`` below. ``trajectory_group_filter`` is set here
# because the registered word_puzzle_async config leaves it ``None`` (the
# recipe wires a real filter at launch); the pipeline calls
# ``.make()`` on it, so tests must supply a concrete filter.
_TRAIN_TEST_DEFAULTS = dict(
    num_train_steps=1,
    tasks_per_train_step=2,
    sync_every_n_train_steps=8,
)
_ROLLOUT_TEST_DEFAULTS = dict(
    samples_per_task=2,
    max_concurrent_samples=8,
    max_pending_rollout_tasks=4,
    trajectory_group_filter=TrajectoryGroupFilterConfig(
        zero_variance_keep_prob=1.0,
    ),
)
_EVAL_TEST_DEFAULTS = dict(
    eval_samples_per_task=1,
    eval_every_n_train_steps=1,
    max_eval_items=1,
)
_TRAIN_FIELDS = frozenset(_TRAIN_TEST_DEFAULTS)
_ROLLOUT_FIELDS = frozenset(_ROLLOUT_TEST_DEFAULTS)
_EVAL_FIELDS = frozenset(_EVAL_TEST_DEFAULTS)


def _make_config(**overrides) -> RunConfig:
    """Build a composed ``RunConfig`` (the registered word_puzzle_async
    operating point) with test-scale sub-config values, routing each flat
    override into its ``train`` / ``rollout`` / ``eval`` sub-config.

    Accepts the same flat kwarg names the tests historically passed
    (e.g. ``num_train_steps=1``); unknown names are rejected so a typo
    doesn't silently no-op.
    """
    train_over = {k: v for k, v in overrides.items() if k in _TRAIN_FIELDS}
    rollout_over = {k: v for k, v in overrides.items() if k in _ROLLOUT_FIELDS}
    eval_over = {k: v for k, v in overrides.items() if k in _EVAL_FIELDS}
    unknown = set(overrides) - _TRAIN_FIELDS - _ROLLOUT_FIELDS - _EVAL_FIELDS
    assert not unknown, f"unknown override(s): {unknown}"
    base = get_config("word_puzzle/async")
    return dataclasses.replace(
        base,
        train=dataclasses.replace(base.train, **{**_TRAIN_TEST_DEFAULTS, **train_over}),
        rollout=dataclasses.replace(
            base.rollout, **{**_ROLLOUT_TEST_DEFAULTS, **rollout_over}
        ),
        eval=dataclasses.replace(base.eval, **{**_EVAL_TEST_DEFAULTS, **eval_over}),
    )


class TestTrainAndEvalLoopAsync:
    async def test_runs_one_iteration_and_calls_train(self):
        agent = _FakeAgent()
        trainer = _FakeTrainer()
        logger = _CapturingLogger()
        config = _make_config(num_train_steps=1)

        await asyncio.wait_for(
            _run_async_loop(
                train_dataset=_ListDataset(
                    [_TestTaskData(task_id=str(i)) for i in range(8)]
                ),
                eval_dataset=_ListDataset([_TestTaskData(task_id="v")]),
                agent=agent,
                trainer=trainer,
                num_train_steps=1,
                config=config,
                ml_logger=logger,
                eval_every_n_train_steps=1,
            ),
            timeout=10.0,
        )

        assert trainer.calls, "trainer.train_step was never called"
        # Training got at least tasks_per_train_step samples flattened.
        n_traj, _sync = trainer.calls[0]
        # Each group has samples_per_task=2 trajectories, train_step gets all groups.
        assert n_traj >= config.rollout.samples_per_task
        # Eval ran at least once: at least one logged metrics dict carries
        # an eval/ key (initial evaluation + post-iteration evaluation).
        assert any(any(k.startswith("eval/") for k in m) for m, _ in logger.metrics), (
            "expected eval-prefixed metrics to be logged"
        )

    async def test_group_filter_gauges_reach_the_ml_platform_sink(self):
        """Fence on the sink hop, which a capturing-logger assertion cannot
        see: an unregistered key still REACHES the logger and is only then
        silently dropped by ``MLPlatformSink`` (the spec=None path). Drive
        the real ``SinkDispatcher`` and check what the ml_platform backend
        received, since that is what the dashboard shows.
        """
        from core.sinks import MLPlatformSink, SinkDispatcher

        class _Backend:
            def __init__(self):
                self.keys: set[str] = set()

            def log_metrics(self, metrics, step=None):
                self.keys.update(metrics)

            def log_hparams(self, *_, **__):
                pass

        ml_platform = _Backend()
        logger = SinkDispatcher([MLPlatformSink(ml_platform)])
        # keep_prob=1.0 (the suite default) plus _FakeAgent's alternating
        # rewards means nothing is dropped and no group ties, so every
        # reason share must be present AND zero. A metric that appears only
        # once it fires cannot answer "is this filter inert".
        #
        # Two steps, not one: the gauges are sampled BEFORE next_batch, so
        # step 1's window opens before any dispatch could have finished and
        # is legitimately empty. Step 2 reports step 1's rollouts.
        config = _make_config(num_train_steps=2)

        await asyncio.wait_for(
            _run_async_loop(
                train_dataset=_ListDataset(
                    [_TestTaskData(task_id=str(i)) for i in range(16)]
                ),
                eval_dataset=None,
                agent=_FakeAgent(),
                trainer=_FakeTrainer(),
                num_train_steps=2,
                config=config,
                ml_logger=logger,
                eval_every_n_train_steps=1,
            ),
            timeout=10.0,
        )

        assert {
            "train/pipeline/group_zero_variance_ratio",
            "train/pipeline/group_drop_reason/zero_variance",
            "train/pipeline/group_drop_reason/too_few_valid_trajectories",
            "train/pipeline/group_drop_reason/too_many_transient_failures",
        } <= ml_platform.keys

    async def test_eval_metrics_logged_with_eval_prefix(self):
        agent = _FakeAgent()
        trainer = _FakeTrainer()
        logger = _CapturingLogger()
        config = _make_config(num_train_steps=1)

        await asyncio.wait_for(
            _run_async_loop(
                train_dataset=_ListDataset(
                    [_TestTaskData(task_id=str(i)) for i in range(4)]
                ),
                eval_dataset=_ListDataset([_TestTaskData(task_id="v")]),
                agent=agent,
                trainer=trainer,
                num_train_steps=1,
                config=config,
                ml_logger=logger,
                eval_every_n_train_steps=1,
            ),
            timeout=10.0,
        )
        eval_metric_keys = set()
        for metrics, _step in logger.metrics:
            eval_metric_keys.update(k for k in metrics if k.startswith("eval/"))
        assert eval_metric_keys, "expected at least one eval/* metric logged"

    async def test_no_eval_dataset_skips_eval(self):
        agent = _FakeAgent()
        trainer = _FakeTrainer()
        logger = _CapturingLogger()
        config = _make_config(num_train_steps=1)

        await asyncio.wait_for(
            _run_async_loop(
                train_dataset=_ListDataset(
                    [_TestTaskData(task_id=str(i)) for i in range(4)]
                ),
                eval_dataset=None,  # no evaluation dataset
                agent=agent,
                trainer=trainer,
                num_train_steps=1,
                config=config,
                ml_logger=logger,
                eval_every_n_train_steps=1,
            ),
            timeout=10.0,
        )
        # With eval_dataset=None, run_eval is still triggered but with empty
        # tasks -> just a "no eval data" scenario; main thing is the loop
        # completes and trainer ran.
        assert trainer.calls

    # NOTE: the old `test_resume_from_step_advances_iterator` (which
    # drove resume via `resume_from_rollout_batch`) was deleted along
    # with that config field. Resume is now single-sourced through
    # `resume_from_train_step` + a metrics-file lookup; construction-
    # level coverage lives in
    # ``TestAsyncRolloutPipelineConstruction.test_construct_applies_resume_position``.


class TestRunSingleRolloutErrorHandling:
    """Regression tests for the run_single_rollout exception path.

    Before the fix, a raised exception inside `run_single_rollout` left the
    sampling index orphaned: it was never put() and never mark_dropped(), so
    in ordered mode the buffer's _next_rollout_index would stall on it
    forever, deadlocking the trainer.
    """

    async def test_rollout_exception_does_not_deadlock(self):
        # Make the *first* rollout call fail. rollout_index=0 is never put or
        # marked dropped by the successful path -- the ordered buffer must
        # skip past it via mark_dropped(). The trainer needs
        # tasks_per_train_step=2 groups, so it must succeed using
        # rollout_index=1 and 2 (skipping the failed 0).
        agent = _FlakyAgent(raise_on_call_indices={0})
        trainer = _FakeTrainer()
        logger = _CapturingLogger()
        config = _make_config(num_train_steps=1)

        # If the bug regresses, wait_for will time out and the test fails.
        await asyncio.wait_for(
            _run_async_loop(
                train_dataset=_ListDataset(
                    [_TestTaskData(task_id=str(i)) for i in range(8)]
                ),
                eval_dataset=None,
                agent=agent,
                trainer=trainer,
                num_train_steps=1,
                config=config,
                ml_logger=logger,
                eval_every_n_train_steps=1,
            ),
            timeout=10.0,
        )

        # Trainer must have made progress despite the early failure.
        assert trainer.calls, "trainer never ran -- ordered buffer likely deadlocked"

    # NOTE: the old `test_resume_with_ordered_buffer_aligns_indices`
    # was a regression test for a bug where the buffer's
    # `start_rollout_index` and the producer's `_next_rollout_index`
    # could diverge across a resume driven by `resume_from_rollout_batch`.
    # Both counters now always start at 0 (rollout-side state is not
    # persisted across restarts; see
    # `AsyncRolloutPipeline.__init__` docstring on
    # `resume_train_iter_position`), so the divergence the test was
    # catching is architecturally impossible — deleted with the code
    # path that made it possible.

    async def test_multiple_rollout_exceptions_do_not_deadlock_ordered_mode(self):
        # Several consecutive failures at the start: sampling indices 0, 1, 2
        # all raise; trainer should still pick up the next 2 successful
        # indices (3, 4).
        agent = _FlakyAgent(raise_on_call_indices={0, 1, 2})
        trainer = _FakeTrainer()
        logger = _CapturingLogger()
        config = _make_config(num_train_steps=1)

        await asyncio.wait_for(
            _run_async_loop(
                train_dataset=_ListDataset(
                    [_TestTaskData(task_id=str(i)) for i in range(16)]
                ),
                eval_dataset=None,
                agent=agent,
                trainer=trainer,
                num_train_steps=1,
                config=config,
                ml_logger=logger,
                eval_every_n_train_steps=1,
            ),
            timeout=10.0,
        )

        assert trainer.calls, (
            "trainer never ran -- ordered buffer likely deadlocked on early failures"
        )


class TestRolloutTrainStepIdSummary:
    """After a train-step emission, the logged metrics dict must contain
    sampler/policy_step (weighted mean over per-step stats.train_step_id)
    emitted directly by ``SamplerPolicyStepMetricAggregator``. Async-RL
    staleness signal."""

    async def test_rollout_metrics_get_train_step_id_summary(self):
        """Feed the async loop trajectories whose extras carry two distinct
        trainStepId values; verify the train-step emission contains
        sampler/policy_step (weighted mean)."""
        pytest.skip(
            "TODO: reuse existing async-loop test rig with "
            "trainStepId=5 on N trajectories + trainStepId=6 on M "
            "trajectories, assert train-step emission carries "
            "sampler/policy_step == (5*N + 6*M) / (N+M)."
        )


class TestEvalTrainStepIdSummary:
    """After an async eval emission, the logged metrics dict must contain
    sampler/policy_step (weighted mean over per-step stats.train_step_id)
    emitted via ``SamplerPolicyStepMetricAggregator``."""

    async def test_eval_metrics_get_train_step_id_summary(self):
        """Feed the async loop trajectories at two distinct trainStepId
        values during eval; verify the eval emission contains
        sampler/policy_step (weighted mean)."""
        pytest.skip(
            "TODO: reuse the same test rig described in "
            "TestRolloutTrainStepIdSummary TODO, but drive an eval "
            "emission instead of a rollout emission; assert "
            "sampler/policy_step == weighted mean of the trainStepIds "
            "present in the trajectories eval consumed."
        )
