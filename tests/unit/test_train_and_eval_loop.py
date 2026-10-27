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

"""Tests for the shared pipeline-agnostic ``train_and_eval_loop``.

Both AsyncRolloutPipeline and SyncRolloutPipeline are driven by the
same trainer function in ``pipeline.train_and_eval``. These tests
exercise that function against a hand-rolled fake pipeline that
implements the ``RolloutPipeline`` protocol — so the tests verify the trainer
side of the contract independently of any concrete pipeline.

Concrete-pipeline behavior is covered by ``test_rollout_pipeline.py``
(sync + async) and end-to-end wiring by ``test_async_rl_train_loop.py``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, List
from unittest.mock import MagicMock

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
from core.rollout_protocol import BatchItem
from pipeline.train_and_eval import train_and_eval_loop


def _item(
    group: TrajectoryGroup | None = None, metrics: dict | None = None
) -> BatchItem:
    """Terse BatchItem factory for scripted next_batch fixtures."""
    return BatchItem(
        group=group if group is not None else _group(), metrics=metrics or {}
    )


def _scored(reward: float = 1.0) -> ScoredTrajectory:
    us = Step(step_id=1, source="user", message="p")
    a = AtifTrajectoryStep.from_wire(step_id=2, generation="g", stats=StepStats())
    traj = AtifTrajectory(
        schema_version="ATIF-v1.7",
        session_id="s",
        agent=AtifAgent(name="t", version="0", model_name="m"),
        steps=[us, a],
    )
    return ScoredTrajectory(trajectory=traj, reward=Reward(reward=reward))


def _group(n_traj: int = 2) -> TrajectoryGroup:
    """Non-zero-variance group so filter_trajectory_group keeps it."""
    rewards = [1.0 if i % 2 == 0 else 0.0 for i in range(n_traj)]
    return TrajectoryGroup(
        task_data=None, scored_trajectories=[_scored(r) for r in rewards]
    )


class _FakePipeline:
    """Implements the ``RolloutPipeline`` protocol with scripted next_batch results.

    ``next_batch_returns``: list of ``[BatchItem, ...]`` batches yielded on
    successive calls. ``next_batch`` records ``step_num`` on each call for
    assertion.

    ``pipeline_gauges`` is what ``sample_pipeline_gauges`` returns; the
    call order relative to ``next_batch`` is recorded in ``call_order``
    so a test can pin that the gauges are sampled BEFORE the block.
    """

    def __init__(self, next_batch_returns: list, pipeline_gauges: dict | None = None):
        self._returns = list(next_batch_returns)
        self._call_idx = 0
        self.pipeline_gauges = pipeline_gauges or {}
        self.gauge_calls = 0
        self.call_order: list[str] = []
        self.step_num_calls: list[int | None] = []
        self.eval_calls: list[int] = []
        self.checkpoint_eval_calls: list[tuple[int, str]] = []
        self.base_model_eval_calls: list[int] = []
        self.stop_called = False
        self._stop_event = asyncio.Event()

    async def run(self) -> None:
        await self._stop_event.wait()

    def stop(self) -> None:
        self.stop_called = True
        self._stop_event.set()

    def sample_pipeline_gauges(self) -> dict:
        self.gauge_calls += 1
        self.call_order.append("sample_pipeline_gauges")
        return dict(self.pipeline_gauges)

    async def next_batch(self, n: int, *, step_num: int | None = None):
        self.call_order.append("next_batch")
        self.step_num_calls.append(step_num)
        if self._call_idx >= len(self._returns):
            # Emergency exit: return empty so trainer sees no trajectories
            # and skips, rather than hanging.
            return [], 0
        items = self._returns[self._call_idx]
        self._call_idx += 1
        # Position mirrors call count for assertion visibility.
        return items, self._call_idx * 10

    def schedule_eval(self, step_val: int) -> None:
        self.eval_calls.append(step_val)

    def schedule_checkpoint_eval(self, step_val: int, endpoint: str) -> None:
        self.checkpoint_eval_calls.append((step_val, endpoint))

    def schedule_base_model_eval(self, step_val: int = 0) -> None:
        self.base_model_eval_calls.append(step_val)


class _FakeTrainer:
    """Records train_step calls; ``metrics_fn`` lets a test control what
    train_step returns per call (list-indexed by call count)."""

    def __init__(self, metrics_fn=None):
        self.calls: list[tuple[int, bool]] = []
        self._metrics_fn = metrics_fn or (lambda i: {"train_step_seconds": 0.1})

    async def train_step(
        self, trajectories: list, sync_trainer_and_sampler: bool = True, **kwargs
    ) -> dict:
        idx = len(self.calls)
        self.calls.append((len(trajectories), sync_trainer_and_sampler))
        return self._metrics_fn(idx)


class _CapturingLogger:
    def __init__(self):
        self.log_calls: list[tuple[dict, int | None]] = []

    def log_metrics(self, metrics, step=None):
        self.log_calls.append((dict(metrics), step))


class TestTrainAndEvalLoop:
    """Contract tests for the shared trainer loop against a fake pipeline."""

    @pytest.mark.asyncio
    async def test_calls_next_batch_with_step_num(self):
        """next_batch is called once per training step with the CURRENT
        train_step index (1-indexed). Sync pipelines rely on this to
        key their per-step sampling metrics; async ignores it."""
        pipe = _FakePipeline(next_batch_returns=[[_item(), _item()]] * 2)
        trainer = _FakeTrainer()
        logger = _CapturingLogger()

        # gather the fake pipeline (blocks on stop_event) and the trainer.
        await asyncio.gather(
            pipe.run(),
            train_and_eval_loop(
                pipeline=pipe,
                trainer=trainer,
                ml_logger=logger,
                num_train_steps=2,
                tasks_per_train_step=2,
                eval_every_n_train_steps=1,
            ),
        )
        assert pipe.step_num_calls == [1, 2], (
            "next_batch should be called with step_num=_train_step at each iter"
        )
        assert len(trainer.calls) == 2
        assert pipe.stop_called

    @pytest.mark.asyncio
    async def test_schedule_eval_on_cadence_and_at_final_step(self):
        """schedule_eval fires at initial step 0, then whenever
        global_step % eval_every_n_train_steps == 0, and always at the
        final step."""
        pipe = _FakePipeline(next_batch_returns=[[_item(), _item()]] * 4)
        trainer = _FakeTrainer()
        logger = _CapturingLogger()

        await asyncio.gather(
            pipe.run(),
            train_and_eval_loop(
                pipeline=pipe,
                trainer=trainer,
                ml_logger=logger,
                num_train_steps=4,
                tasks_per_train_step=2,
                eval_every_n_train_steps=2,
            ),
        )
        # Initial eval at step 0, then at steps 2 and 4.
        assert pipe.eval_calls == [0, 2, 4], f"got {pipe.eval_calls}"

    @pytest.mark.asyncio
    async def test_skips_when_trainer_returns_empty(self):
        """When trainer.train_step returns an empty dict (failure),
        the loop does NOT advance global_step and does NOT log
        train/* metrics for that step."""
        pipe = _FakePipeline(next_batch_returns=[[_item(), _item()]] * 3)
        # Fail on call 0, succeed on 1+.
        trainer = _FakeTrainer(
            metrics_fn=lambda i: {} if i == 0 else {"train_step_seconds": 0.05}
        )
        logger = _CapturingLogger()

        await asyncio.gather(
            pipe.run(),
            train_and_eval_loop(
                pipeline=pipe,
                trainer=trainer,
                ml_logger=logger,
                num_train_steps=2,
                tasks_per_train_step=2,
                eval_every_n_train_steps=1,
            ),
        )
        # train_step called 3 times: step 1 fails, step 1 retries (call 2),
        # step 2 succeeds (call 3). Because global_step doesn't advance
        # after failure, the loop DOES try the same step index again on
        # the next range() iteration -- but range is fixed. Actually
        # range advances regardless; global_step just doesn't tick.
        # So: 2 train_step calls (one per _train_step in range), one
        # returning {} (skipped) and one returning metrics.
        assert len(trainer.calls) == 2
        # Only one metrics log with train/train_step_seconds (the successful one).
        step_time_logs = [
            (m, s) for m, s in logger.log_calls if "train/train_step_seconds" in m
        ]
        assert len(step_time_logs) == 1, (
            f"expected 1 train/train_step_seconds log, got {step_time_logs}"
        )

    @pytest.mark.asyncio
    async def test_skips_when_batch_has_no_trajectories(self):
        """When next_batch returns items but every group has zero
        trajectories, the trainer is NOT called at all and the step
        is skipped without advancing global_step."""
        # Group with zero trajectories -> flattened list is empty ->
        # continue path fires before trainer.train_step.
        empty_group = TrajectoryGroup(task_data=None, scored_trajectories=[])
        pipe = _FakePipeline(
            next_batch_returns=[
                [_item(group=empty_group)],  # first step: empty
                [_item()],  # second step: real
            ]
        )
        trainer = _FakeTrainer()
        logger = _CapturingLogger()

        await asyncio.gather(
            pipe.run(),
            train_and_eval_loop(
                pipeline=pipe,
                trainer=trainer,
                ml_logger=logger,
                num_train_steps=2,
                tasks_per_train_step=1,
                eval_every_n_train_steps=1,
            ),
        )
        # Trainer called only for the second step (first was skipped).
        assert len(trainer.calls) == 1

    @pytest.mark.asyncio
    async def test_sync_every_n_train_steps_controls_sync_flag(self):
        """sync_trainer_and_sampler passed to trainer.train_step is
        True whenever (global_step + 1) % sync_every_n_train_steps == 0."""
        pipe = _FakePipeline(next_batch_returns=[[_item()]] * 4)
        trainer = _FakeTrainer()
        logger = _CapturingLogger()

        await asyncio.gather(
            pipe.run(),
            train_and_eval_loop(
                pipeline=pipe,
                trainer=trainer,
                ml_logger=logger,
                num_train_steps=4,
                tasks_per_train_step=1,
                eval_every_n_train_steps=99,  # eval doesn't fire
                sync_every_n_train_steps=2,
            ),
        )
        # sync flag: step 0 -> (0+1)%2 = 1 -> False; step 1 -> (1+1)%2 = 0
        # -> True; step 2 -> False; step 3 -> True.
        sync_flags = [c[1] for c in trainer.calls]
        assert sync_flags == [False, True, False, True], (
            f"expected [F,T,F,T], got {sync_flags}"
        )

    @pytest.mark.asyncio
    async def test_emits_staleness_steps_metric(self):
        """After one train step, the logged metrics carry
        train/staleness_steps = global_step (1) - mean(policy_step) from the
        batch (0.5 when the batch's two items used policy_step 0 and 1), plus
        the pass-through train/sampler/policy_step."""
        pipe = _FakePipeline(
            next_batch_returns=[
                [
                    _item(metrics={"sampler/policy_step": 0}),
                    _item(metrics={"sampler/policy_step": 1}),
                ]
            ]
        )
        trainer = (
            _FakeTrainer()
        )  # returns {"train_step_seconds": 0.1} -> step advances + logs
        logger = _CapturingLogger()

        await asyncio.gather(
            pipe.run(),
            train_and_eval_loop(
                pipeline=pipe,
                trainer=trainer,
                ml_logger=logger,
                num_train_steps=1,
                tasks_per_train_step=2,
                eval_every_n_train_steps=1,
            ),
        )

        calls_at_step_1 = [m for m, s in logger.log_calls if s == 1]
        assert len(calls_at_step_1) == 1
        metrics = calls_at_step_1[0]
        assert metrics["train/staleness_steps"] == 0.5
        assert metrics["train/sampler/policy_step"] == 0.5  # pass-through

    @pytest.mark.asyncio
    async def test_omits_staleness_when_policy_step_absent(self):
        """When no batch item carries a numeric policy_step (e.g. a
        gemini_api backend with train_step_id=None), staleness_steps is NOT
        emitted (no fake zero, no NaN), and policy_step is absent too."""
        pipe = _FakePipeline(
            next_batch_returns=[[_item(metrics={"sampler/policy_step": None})]]
        )
        trainer = _FakeTrainer()
        logger = _CapturingLogger()

        await asyncio.gather(
            pipe.run(),
            train_and_eval_loop(
                pipeline=pipe,
                trainer=trainer,
                ml_logger=logger,
                num_train_steps=1,
                tasks_per_train_step=1,
                eval_every_n_train_steps=1,
            ),
        )

        calls_at_step_1 = [m for m, s in logger.log_calls if s == 1]
        assert len(calls_at_step_1) == 1
        metrics = calls_at_step_1[0]
        assert "train/staleness_steps" not in metrics
        assert "train/sampler/policy_step" not in metrics

    @pytest.mark.asyncio
    async def test_sources_global_step_from_backend(self):
        """global_step follows the backend's completed_train_step_id when
        present. If the backend jumps (2 -> 4, skipping 3 after a retry
        whose response was lost), the client counter follows -- no silent
        drift."""
        step_id_sequence = [1, 2, 4, 5]  # jumps 2 -> 4 (skip 3)
        pipe = _FakePipeline(next_batch_returns=[[_item()] for _ in range(4)])
        trainer = _FakeTrainer(
            metrics_fn=lambda i: (
                {
                    "train_step_seconds": 0.5,
                    "completed_train_step_id": step_id_sequence[i],
                }
                if i < len(step_id_sequence)
                else {}
            )
        )
        logger = _CapturingLogger()

        await asyncio.gather(
            pipe.run(),
            train_and_eval_loop(
                pipeline=pipe,
                trainer=trainer,
                ml_logger=logger,
                num_train_steps=4,
                tasks_per_train_step=1,
                eval_every_n_train_steps=10,  # disable eval cadence noise
            ),
        )

        steps_emitted = [s for _m, s in logger.log_calls]
        assert steps_emitted == [1, 2, 4, 5], (
            f"expected backend-sourced steps [1, 2, 4, 5]; got {steps_emitted}"
        )

    @pytest.mark.asyncio
    async def test_falls_back_to_client_counter_when_backend_omits_step_id(self):
        """When train_metrics lacks completed_train_step_id (e.g. gemini_api
        backend), global_step falls back to the client counter
        (initial_global_step + N)."""
        pipe = _FakePipeline(next_batch_returns=[[_item()] for _ in range(3)])
        trainer = (
            _FakeTrainer()
        )  # default metrics_fn -> {"train_step_seconds": 0.1}, no step id
        logger = _CapturingLogger()

        await asyncio.gather(
            pipe.run(),
            train_and_eval_loop(
                pipeline=pipe,
                trainer=trainer,
                ml_logger=logger,
                num_train_steps=3,
                tasks_per_train_step=1,
                eval_every_n_train_steps=10,
            ),
        )

        steps_emitted = [s for _m, s in logger.log_calls]
        assert steps_emitted == [1, 2, 3], (
            f"expected client counter [1, 2, 3]; got {steps_emitted}"
        )

    @pytest.mark.asyncio
    async def test_staleness_reflects_backend_step_jumps(self):
        """When the backend jumps global_step 2 -> 4, staleness computed
        against the batch's mean policy_step (still 2) reflects the jump:
        staleness = 4 - 2 = 2 (was 0 at the pre-jump step)."""
        step_sequence = [2, 4]  # jump 2 -> 4
        # Batch mean policy_step = 2 (both items report step 2).
        pipe = _FakePipeline(
            next_batch_returns=[
                [
                    _item(metrics={"sampler/policy_step": 2}),
                    _item(metrics={"sampler/policy_step": 2}),
                ]
                for _ in range(2)
            ]
        )
        trainer = _FakeTrainer(
            metrics_fn=lambda i: {
                "train_step_seconds": 0.5,
                "completed_train_step_id": step_sequence[i],
            }
        )
        logger = _CapturingLogger()

        await asyncio.gather(
            pipe.run(),
            train_and_eval_loop(
                pipeline=pipe,
                trainer=trainer,
                ml_logger=logger,
                num_train_steps=2,
                tasks_per_train_step=2,
                eval_every_n_train_steps=10,
            ),
        )

        step2 = [m for m, s in logger.log_calls if s == 2][0]
        step4 = [m for m, s in logger.log_calls if s == 4][0]
        assert step2["train/staleness_steps"] == 0  # global 2, mean policy 2
        assert step4["train/staleness_steps"] == 2  # global 4, mean policy 2

    @pytest.mark.asyncio
    async def test_aggregates_learning_metrics_under_train_prefix(self):
        """Per-group ``learning/reward/mean`` and ``learning/sampling_accuracy``
        from ``CoreRewardMetricAggregator`` are aggregated via
        ``aggregate_by_spec`` and logged under the ``train/`` prefix so
        the Vertex Experiment UI shows a train-time learning curve."""
        pipe = _FakePipeline(
            next_batch_returns=[
                [
                    _item(
                        metrics={
                            "learning/reward/mean": 0.5,
                            "learning/sampling_accuracy": 0.5,
                        }
                    ),
                    _item(
                        metrics={
                            "learning/reward/mean": 0.7,
                            "learning/sampling_accuracy": 1.0,
                        }
                    ),
                ]
            ]
        )
        trainer = _FakeTrainer()
        logger = _CapturingLogger()

        await asyncio.gather(
            pipe.run(),
            train_and_eval_loop(
                pipeline=pipe,
                trainer=trainer,
                ml_logger=logger,
                num_train_steps=1,
                tasks_per_train_step=2,
                eval_every_n_train_steps=1,
            ),
        )

        metrics = [m for m, s in logger.log_calls if s == 1][0]
        assert "train/learning/reward/mean" in metrics
        assert "train/learning/sampling_accuracy" in metrics
        # aggregate_by_spec uses MEAN for sampling_accuracy
        assert metrics["train/learning/sampling_accuracy"] == pytest.approx(0.75)

    @pytest.mark.asyncio
    async def test_emits_pipeline_timing_keys(self):
        """Timing metrics use flat train/pipeline/*_time leaf names, NOT
        the old train/time/* nested prefix."""
        pipe = _FakePipeline(next_batch_returns=[[_item()]])
        trainer = _FakeTrainer()
        logger = _CapturingLogger()

        await asyncio.gather(
            pipe.run(),
            train_and_eval_loop(
                pipeline=pipe,
                trainer=trainer,
                ml_logger=logger,
                num_train_steps=1,
                tasks_per_train_step=1,
                eval_every_n_train_steps=1,
            ),
        )

        metrics = [m for m, s in logger.log_calls if s == 1][0]
        # New names present.
        assert "train/pipeline/train_step_time" in metrics
        assert "train/pipeline/wait_for_rollout_time" in metrics
        # Old names absent.
        assert "train/time/train_step" not in metrics
        assert "train/time/wait_for_rollout" not in metrics

    @pytest.mark.asyncio
    async def test_emits_pipeline_gauges_sampled_before_the_block(self):
        """The two gauges that cannot ride on a BatchItem
        (buffer_utilization / group_drop_ratio) reach the sink, under the
        train/ prefix, sampled BEFORE next_batch blocks.

        Regression fence: they used to be computed into a local dict at the
        end of MetricsAggregator.log_batch and thrown away, so both were
        registered-but-dead for the whole async pipeline.
        """
        pipe = _FakePipeline(
            next_batch_returns=[[_item()]],
            pipeline_gauges={
                "pipeline/buffer_utilization": 0.75,
                "pipeline/group_drop_ratio": 0.25,
            },
        )
        trainer = _FakeTrainer()
        logger = _CapturingLogger()

        await asyncio.gather(
            pipe.run(),
            train_and_eval_loop(
                pipeline=pipe,
                trainer=trainer,
                ml_logger=logger,
                num_train_steps=1,
                tasks_per_train_step=1,
                eval_every_n_train_steps=1,
            ),
        )

        metrics = [m for m, s in logger.log_calls if s == 1][0]
        assert metrics.get("train/pipeline/buffer_utilization") == 0.75, (
            "buffer_utilization never reached the sink: "
            f"pipeline/* keys logged = "
            f"{[k for k in metrics if 'pipeline/' in k]}"
        )
        assert metrics.get("train/pipeline/group_drop_ratio") == 0.25
        # Sampling point: occupancy must be read while the trainer is still
        # asking, not after next_batch has already drained the buffer.
        assert pipe.call_order[:2] == ["sample_pipeline_gauges", "next_batch"], (
            f"gauges sampled at the wrong point: {pipe.call_order}"
        )
        assert pipe.gauge_calls == 1, (
            "sample_pipeline_gauges must be called exactly once per train "
            f"step (it advances a delta window); got {pipe.gauge_calls}"
        )

    @pytest.mark.asyncio
    async def test_omits_pipeline_gauges_the_pipeline_does_not_report(self):
        """A pipeline with nothing to report (sync) emits no gauge keys —
        rather than a fabricated 0.0 that reads as 'buffer empty'."""
        pipe = _FakePipeline(next_batch_returns=[[_item()]])
        trainer = _FakeTrainer()
        logger = _CapturingLogger()

        await asyncio.gather(
            pipe.run(),
            train_and_eval_loop(
                pipeline=pipe,
                trainer=trainer,
                ml_logger=logger,
                num_train_steps=1,
                tasks_per_train_step=1,
                eval_every_n_train_steps=1,
            ),
        )

        metrics = [m for m, s in logger.log_calls if s == 1][0]
        assert "train/pipeline/buffer_utilization" not in metrics
        assert "train/pipeline/group_drop_ratio" not in metrics

    @pytest.mark.asyncio
    async def test_emits_pipeline_iterator_current(self):
        """Iterator position is emitted as
        train/pipeline/iterator_current, NOT train/iterator_current."""
        # _FakePipeline.next_batch returns position = call_idx * 10, so the
        # first (and only) batch reports train_iter_position = 10.
        pipe = _FakePipeline(next_batch_returns=[[_item()]])
        trainer = _FakeTrainer()
        logger = _CapturingLogger()

        await asyncio.gather(
            pipe.run(),
            train_and_eval_loop(
                pipeline=pipe,
                trainer=trainer,
                ml_logger=logger,
                num_train_steps=1,
                tasks_per_train_step=1,
                eval_every_n_train_steps=1,
            ),
        )

        metrics = [m for m, s in logger.log_calls if s == 1][0]
        assert metrics["train/pipeline/iterator_current"] == 10
        assert "train/iterator_current" not in metrics

    @pytest.mark.asyncio
    async def test_schedules_checkpoint_eval_when_endpoint_present_and_enabled(self):
        """With run_checkpoint_endpoint_eval=True, a step whose trainer
        returns a _checkpoint_endpoint triggers schedule_checkpoint_eval at
        that step; steps without an endpoint do not."""
        pipe = _FakePipeline(next_batch_returns=[[_item(), _item()]] * 2)

        def _metrics(i):
            # completed_train_step_id follows the loop step (1, 2).
            base = {"train_step_seconds": 0.1, "completed_train_step_id": i + 1}
            if i == 1:  # second step surfaces a checkpoint endpoint
                base["_checkpoint_endpoint"] = "projects/P/locations/us/endpoints/E"
            return base

        trainer = _FakeTrainer(metrics_fn=_metrics)

        await asyncio.gather(
            pipe.run(),
            train_and_eval_loop(
                pipeline=pipe,
                trainer=trainer,
                ml_logger=_CapturingLogger(),
                num_train_steps=2,
                tasks_per_train_step=2,
                eval_every_n_train_steps=1,
                run_checkpoint_endpoint_eval=True,
            ),
        )

        assert pipe.checkpoint_eval_calls == [
            (2, "projects/P/locations/us/endpoints/E")
        ]

    @pytest.mark.asyncio
    async def test_checkpoint_eval_off_by_default(self):
        """Default (feature disabled): schedule_checkpoint_eval is never
        called even when an endpoint is present."""
        pipe = _FakePipeline(next_batch_returns=[[_item(), _item()]])
        trainer = _FakeTrainer(
            metrics_fn=lambda i: {
                "train_step_seconds": 0.1,
                "completed_train_step_id": 1,
                "_checkpoint_endpoint": "projects/P/locations/us/endpoints/E",
            }
        )

        await asyncio.gather(
            pipe.run(),
            train_and_eval_loop(
                pipeline=pipe,
                trainer=trainer,
                ml_logger=_CapturingLogger(),
                num_train_steps=1,
                tasks_per_train_step=2,
                eval_every_n_train_steps=1,
                # run_checkpoint_endpoint_eval omitted -> defaults False
            ),
        )

        assert pipe.checkpoint_eval_calls == []

    @pytest.mark.asyncio
    async def test_base_model_eval_fires_at_step_zero_when_enabled(self):
        """With the checkpoint sweep enabled and a fresh run, the eval_ckpt/*
        series must start from the base model at step 0 — there is no
        checkpoint endpoint before the first gradient update."""
        pipe = _FakePipeline(next_batch_returns=[[_item(), _item()]])

        await asyncio.gather(
            pipe.run(),
            train_and_eval_loop(
                pipeline=pipe,
                trainer=_FakeTrainer(),
                ml_logger=_CapturingLogger(),
                num_train_steps=1,
                tasks_per_train_step=2,
                eval_every_n_train_steps=1,
                run_checkpoint_endpoint_eval=True,
            ),
        )

        assert pipe.base_model_eval_calls == [0]

    @pytest.mark.asyncio
    async def test_base_model_eval_skipped_on_resume_and_when_disabled(self):
        """No base-model sweep when the feature is off, and none when
        resuming mid-run (a base-model score at step N would misattribute)."""
        off = _FakePipeline(next_batch_returns=[[_item(), _item()]])
        await asyncio.gather(
            off.run(),
            train_and_eval_loop(
                pipeline=off,
                trainer=_FakeTrainer(),
                ml_logger=_CapturingLogger(),
                num_train_steps=1,
                tasks_per_train_step=2,
                eval_every_n_train_steps=1,
            ),
        )
        assert off.base_model_eval_calls == []

        resumed = _FakePipeline(next_batch_returns=[[_item(), _item()]])
        await asyncio.gather(
            resumed.run(),
            train_and_eval_loop(
                pipeline=resumed,
                trainer=_FakeTrainer(),
                ml_logger=_CapturingLogger(),
                num_train_steps=6,
                tasks_per_train_step=2,
                eval_every_n_train_steps=1,
                initial_global_step=5,
                run_checkpoint_endpoint_eval=True,
            ),
        )
        assert resumed.base_model_eval_calls == []


class _RecordingObserver:
    """Records the train-step lifecycle hooks of the ProgressObserver protocol."""

    def __init__(self):
        self.started: list[int] = []
        self.completed: list[tuple[int, int, dict]] = []

    async def on_train_step_started(self, step: int, total_steps: int) -> None:
        self.started.append(step)

    async def on_train_step_completed(
        self, step: int, global_step: int, train_metrics: dict | None = None
    ) -> None:
        self.completed.append((step, global_step, dict(train_metrics or {})))

    async def on_batch_assembled(
        self, step: int, trajectory_ids: list, mean_reward: float | None = None
    ) -> None:
        pass


class TestTrainStepFailureReporting:
    """A failed train step must be reported as failed, not silently skipped.

    Without this the step doc is stranded at "training" forever, its error
    is discarded, and the batch's trajectories never leave TRAINING.
    """

    @staticmethod
    def _failures(obs: _RecordingObserver) -> list[tuple[int, int, dict]]:
        return [c for c in obs.completed if c[2].get("_train_step_error")]

    @pytest.mark.asyncio
    async def test_reports_error_to_observer_when_trainer_fails(self):
        """train_step returning _train_step_error reaches
        on_train_step_completed with the error and an unadvanced global_step."""
        pipe = _FakePipeline(next_batch_returns=[[_item()], [_item()]])
        trainer = _FakeTrainer(
            metrics_fn=lambda i: (
                {"_train_step_error": "TrainStep LRO failed: code=13"}
                if i == 0
                else {"train_step_seconds": 0.1, "completed_train_step_id": 1}
            )
        )
        obs = _RecordingObserver()

        await asyncio.gather(
            pipe.run(),
            train_and_eval_loop(
                pipeline=pipe,
                trainer=trainer,
                ml_logger=_CapturingLogger(),
                num_train_steps=2,
                tasks_per_train_step=1,
                eval_every_n_train_steps=10,
                progress_observer=obs,
            ),
        )

        failures = self._failures(obs)
        assert len(failures) == 1, f"expected 1 reported failure, got {obs.completed}"
        step, global_step, metrics = failures[0]
        assert step == 1
        assert global_step == 0, "a failed step must not advance global_step"
        assert "code=13" in metrics["_train_step_error"]

    @pytest.mark.asyncio
    async def test_reports_failure_when_trainer_returns_bare_empty_dict(self):
        """A backend that signals failure with a bare {} still gets a
        synthesized error, so the step doc is never stranded."""
        pipe = _FakePipeline(next_batch_returns=[[_item()], [_item()]])
        trainer = _FakeTrainer(
            metrics_fn=lambda i: ({} if i == 0 else {"train_step_seconds": 0.1})
        )
        obs = _RecordingObserver()

        await asyncio.gather(
            pipe.run(),
            train_and_eval_loop(
                pipeline=pipe,
                trainer=trainer,
                ml_logger=_CapturingLogger(),
                num_train_steps=2,
                tasks_per_train_step=1,
                eval_every_n_train_steps=10,
                progress_observer=obs,
            ),
        )

        assert len(self._failures(obs)) == 1, (
            "expected the empty-metrics step to be reported as failed,"
            f" got {obs.completed}"
        )

    @pytest.mark.asyncio
    async def test_aborts_run_after_three_consecutive_failures(self):
        """Three consecutive trainer failures stop the run instead of
        burning the remaining budget against a dead backend."""
        pipe = _FakePipeline(next_batch_returns=[[_item()] for _ in range(10)])
        trainer = _FakeTrainer(metrics_fn=lambda i: {"_train_step_error": "boom"})

        with pytest.raises(RuntimeError, match="consecutive train step failures"):
            await asyncio.gather(
                pipe.run(),
                train_and_eval_loop(
                    pipeline=pipe,
                    trainer=trainer,
                    ml_logger=_CapturingLogger(),
                    num_train_steps=10,
                    tasks_per_train_step=1,
                    eval_every_n_train_steps=10,
                ),
            )

        assert len(trainer.calls) == 3, (
            f"expected abort on the 3rd failure, got {len(trainer.calls)} attempts"
        )

    @pytest.mark.asyncio
    async def test_failed_step_does_not_emit_the_completed_log_line(self, caplog):
        """The integration smokes count finished steps by parsing
        ``Train step (\\d+)/(\\d+) completed in``. A failed step emitting
        that line makes them over-count -- and reads as a flat lie next to
        the error."""
        pipe = _FakePipeline(next_batch_returns=[[_item()], [_item()]])
        trainer = _FakeTrainer(
            metrics_fn=lambda i: (
                {"_train_step_error": "boom"}
                if i == 0
                else {"train_step_seconds": 0.1, "completed_train_step_id": 1}
            )
        )

        with caplog.at_level(logging.INFO, logger="pipeline.train_and_eval"):
            await asyncio.gather(
                pipe.run(),
                train_and_eval_loop(
                    pipeline=pipe,
                    trainer=trainer,
                    ml_logger=_CapturingLogger(),
                    num_train_steps=2,
                    tasks_per_train_step=1,
                    eval_every_n_train_steps=10,
                ),
            )

        completed = [
            r.getMessage() for r in caplog.records if "completed in" in r.getMessage()
        ]
        assert len(completed) == 1, (
            f"only the successful step may log 'completed in'; got {completed}"
        )
        assert "2/2" in completed[0]

    @pytest.mark.asyncio
    async def test_successful_step_resets_the_failure_counter(self):
        """Failures must be *consecutive* to abort: fail, fail, succeed,
        fail, fail runs to completion."""
        outcomes: list[dict] = [
            {"_train_step_error": "boom"},
            {"_train_step_error": "boom"},
            {"train_step_seconds": 0.1, "completed_train_step_id": 1},
            {"_train_step_error": "boom"},
            {"_train_step_error": "boom"},
        ]
        pipe = _FakePipeline(next_batch_returns=[[_item()] for _ in outcomes])
        trainer = _FakeTrainer(metrics_fn=lambda i: outcomes[i])

        await asyncio.gather(
            pipe.run(),
            train_and_eval_loop(
                pipeline=pipe,
                trainer=trainer,
                ml_logger=_CapturingLogger(),
                num_train_steps=len(outcomes),
                tasks_per_train_step=1,
                eval_every_n_train_steps=10,
            ),
        )

        assert len(trainer.calls) == len(outcomes)


class TestSamplerEvalToggle:
    """``run_sampler_eval=False`` suppresses every live-sampler eval/* sweep
    while leaving the checkpoint-endpoint sweep untouched."""

    @pytest.mark.asyncio
    async def test_no_sampler_eval_when_disabled(self):
        pipe = _FakePipeline(next_batch_returns=[[_item(), _item()]] * 2)

        def _metrics(i):
            base = {"train_step_seconds": 0.1, "completed_train_step_id": i + 1}
            if i == 1:
                base["_checkpoint_endpoint"] = "projects/P/locations/us/endpoints/E"
            return base

        await asyncio.gather(
            pipe.run(),
            train_and_eval_loop(
                pipeline=pipe,
                trainer=_FakeTrainer(metrics_fn=_metrics),
                ml_logger=_CapturingLogger(),
                num_train_steps=2,
                tasks_per_train_step=2,
                eval_every_n_train_steps=1,
                run_sampler_eval=False,
                run_checkpoint_endpoint_eval=True,
            ),
        )

        # No step-0 baseline, no cadence sweep, no final-step sweep...
        assert pipe.eval_calls == []
        # ...but the checkpoint sweeps (base model + endpoint) still fire.
        assert pipe.base_model_eval_calls == [0]
        assert pipe.checkpoint_eval_calls == [
            (2, "projects/P/locations/us/endpoints/E")
        ]

    @pytest.mark.asyncio
    async def test_sampler_eval_on_by_default(self):
        pipe = _FakePipeline(next_batch_returns=[[_item(), _item()]])

        await asyncio.gather(
            pipe.run(),
            train_and_eval_loop(
                pipeline=pipe,
                trainer=_FakeTrainer(),
                ml_logger=_CapturingLogger(),
                num_train_steps=1,
                tasks_per_train_step=2,
                eval_every_n_train_steps=1,
            ),
        )

        assert pipe.eval_calls == [0, 1]


class TestResolveResumePosition:
    """resolve_resume_position reads the iterator position back from a
    metrics.jsonl file. The emission key moved to
    train/pipeline/iterator_current, so the consumer must look that up --
    atomically with the emission rename, or resume silently breaks."""

    def _write_metrics(self, tmp_path, rows: list[dict]) -> str:
        import json

        path = tmp_path / "metrics.jsonl"
        with open(path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return str(path)

    def test_reads_pipeline_iterator_current(self, tmp_path):
        from pipeline.train_and_eval import resolve_resume_position

        path = self._write_metrics(
            tmp_path, [{"step": 5, "train/pipeline/iterator_current": 100}]
        )
        assert (
            resolve_resume_position(resume_from_step=5, resume_metrics_path=path) == 100
        )

    def test_old_key_only_fails_loud(self, tmp_path):
        """A legacy metrics file (old key only) must fail loud rather
        than silently resume from zero."""
        from pipeline.train_and_eval import resolve_resume_position

        path = self._write_metrics(
            tmp_path, [{"step": 5, "train/iterator_current": 100}]
        )
        with pytest.raises(RuntimeError):
            resolve_resume_position(resume_from_step=5, resume_metrics_path=path)
