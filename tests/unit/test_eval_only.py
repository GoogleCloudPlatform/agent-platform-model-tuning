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

"""Tests for the eval pipeline ``pipeline.eval_only.EvalPipeline``.

Covers the ``EvalPipeline.run()`` step-key contract and the auto-derivation
of the x-axis step from the response ``trainStepId`` (via
``sampler/policy_step``) when ``step_num=0`` (standalone eval).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from pipeline.eval_only import EvalPipeline


class _FakeEvalSub:
    def __init__(self, eval_samples_per_task: int = 1, max_eval_items: int = -1):
        self.eval_samples_per_task = eval_samples_per_task
        self.max_eval_items = max_eval_items


class _FakeAgentSub:
    def __init__(
        self, thinking_level: str = "", max_generation_tokens: int | None = None
    ):
        self.thinking_level = thinking_level
        self.max_generation_tokens = max_generation_tokens


class _FakeRolloutSub:
    def __init__(self, max_concurrent_samples: int = 4):
        self.max_concurrent_samples = max_concurrent_samples


class _FakeConfig:
    """Minimal nested config matching the surface EvalPipeline reads."""

    def __init__(
        self,
        *,
        eval_samples_per_task: int = 1,
        thinking_level: str = "",
        max_eval_items: int = -1,
        max_concurrent_samples: int = 4,
        max_generation_tokens: int | None = None,
    ):
        self.eval = _FakeEvalSub(eval_samples_per_task, max_eval_items)
        self.agent = _FakeAgentSub(thinking_level, max_generation_tokens)
        self.rollout = _FakeRolloutSub(max_concurrent_samples)


class _ResettableIter:
    """Minimal iterator with ``reset()`` matching ``BatchIterator``."""

    def __init__(self, batches: list[list[Any]]):
        self._batches = batches
        self._pos = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._pos >= len(self._batches):
            raise StopIteration
        batch = self._batches[self._pos]
        self._pos += 1
        return batch

    def reset(self) -> None:
        self._pos = 0


class _FakeDataset:
    """Tiny in-memory eval dataset with ``build_iter`` + ``__len__``."""

    def __init__(self, tasks: list[Any]):
        self._tasks = tasks

    def __len__(self) -> int:
        return len(self._tasks)

    def build_iter(self, *, max_items=None, repeat=False):
        tasks = self._tasks if max_items in (None, -1) else self._tasks[:max_items]
        return _ResettableIter([tasks]) if tasks else _ResettableIter([])


class _FakeAgent:
    """Stub agent whose ``group_rollout`` returns bare-float metrics."""

    def __init__(self):
        self.step_args: list[int | None] = []

    async def group_rollout(
        self,
        task_data,
        num_samples,
        *,
        thinking_level=None,
        max_generation_tokens=None,
        on_trajectory_done=None,
        phase=None,
        dispatch_index=None,
        step=None,
    ):
        self.step_args.append(step)
        if on_trajectory_done is not None:
            for _ in range(num_samples):
                on_trajectory_done()
        return object(), {"acc": 1.0}


class _CapturingLogger:
    """Records (metrics, step) pairs passed to ``log_metrics``."""

    def __init__(self):
        self.log_calls: list[tuple[dict, int | None]] = []

    def log_metrics(self, metrics, step=None):
        self.log_calls.append((dict(metrics), step))


def _run_eval(log_step: int = 0, *, metric_prefix: str = "eval") -> _CapturingLogger:
    ml_logger = _CapturingLogger()
    config = _FakeConfig()
    dataset = _FakeDataset(tasks=["task-a"])
    pipeline = EvalPipeline(
        agent=_FakeAgent(),
        config=config,
        ml_logger=ml_logger,
        eval_dataset=dataset,
    )
    asyncio.run(pipeline.run(log_step, metric_prefix=metric_prefix))
    return ml_logger


class TestEvalPipelineStepScoping:
    """Every eval sweep stamps its step onto the rollouts it dispatches, so
    the same eval task run at step 3 and step 6 (or by the live sampler and
    the checkpoint sweep at the SAME step) writes to distinct artifact dirs
    instead of overwriting."""

    def test_run_forwards_step_num_to_group_rollout(self):
        agent = _FakeAgent()
        pipeline = EvalPipeline(
            agent=agent,
            config=_FakeConfig(),
            ml_logger=_CapturingLogger(),
            eval_dataset=_FakeDataset(tasks=["task-a", "task-b"]),
        )
        asyncio.run(pipeline.run(7))
        assert agent.step_args == [7, 7], agent.step_args

    def test_step_zero_is_forwarded_not_dropped(self):
        """Step 0 is the base-model sweep, not 'no step'."""
        agent = _FakeAgent()
        pipeline = EvalPipeline(
            agent=agent,
            config=_FakeConfig(),
            ml_logger=_CapturingLogger(),
            eval_dataset=_FakeDataset(tasks=["task-a"]),
        )
        asyncio.run(pipeline.run(0))
        assert agent.step_args == [0]


class TestEvalPipelineMetricPrefix:
    def test_run_uses_custom_metric_prefix(self):
        """metric_prefix relabels every aggregated key; default stays 'eval'."""
        ml_logger = _run_eval(log_step=30, metric_prefix="eval_ckpt")
        prefixed = [
            (m, s)
            for m, s in ml_logger.log_calls
            if any(k.startswith("eval_ckpt/") for k in m)
        ]
        assert len(prefixed) == 1, f"expected one eval_ckpt/* log, got {ml_logger.log_calls}"
        metrics, step = prefixed[0]
        assert step == 30
        assert metrics  # non-empty
        assert all(k.startswith("eval_ckpt/") for k in metrics)
        assert not any(k.startswith("eval/") for k in metrics)


class TestEvalPipelineLogStep:
    def test_logs_eval_metrics_at_explicit_log_step(self):
        ml_logger = _run_eval(log_step=42)
        eval_logs = [
            (m, s)
            for m, s in ml_logger.log_calls
            if any(k.startswith("eval/") for k in m)
        ]
        assert len(eval_logs) == 1, (
            f"expected one eval/* log, got {ml_logger.log_calls}"
        )
        _metrics, step = eval_logs[0]
        assert step == 42

    def test_defaults_to_step_zero_when_log_step_omitted(self):
        ml_logger = _run_eval()
        eval_logs = [
            (m, s)
            for m, s in ml_logger.log_calls
            if any(k.startswith("eval/") for k in m)
        ]
        assert len(eval_logs) == 1, (
            f"expected one eval/* log, got {ml_logger.log_calls}"
        )
        _metrics, step = eval_logs[0]
        assert step == 0


class _PolicyStepAgent(_FakeAgent):
    """Stub agent that returns ``sampler/policy_step`` in its metrics."""

    def __init__(self, policy_step: float):
        self._policy_step = policy_step

    async def group_rollout(
        self,
        task_data,
        num_samples,
        *,
        thinking_level=None,
        max_generation_tokens=None,
        on_trajectory_done=None,
        phase=None,
        dispatch_index=None,
        step=None,
    ):
        if on_trajectory_done is not None:
            for _ in range(num_samples):
                on_trajectory_done()
        return object(), {"acc": 1.0, "sampler/policy_step": self._policy_step}


class TestAutoDerivePolicyStep:
    """``EvalPipeline.run()`` auto-derives the x-axis from
    ``sampler/policy_step`` when ``step_num=0`` (standalone eval). When a
    training caller passes a nonzero ``step_num``, the provided value is
    used verbatim."""

    @pytest.mark.asyncio
    async def test_step_derived_from_policy_step_when_zero(self):
        ml_logger = _CapturingLogger()
        pipeline = EvalPipeline(
            agent=_PolicyStepAgent(42.0),
            config=_FakeConfig(),
            ml_logger=ml_logger,
            eval_iter=_ResettableIter([["task1"]]),
            semaphore=asyncio.Semaphore(10),
        )
        await pipeline.run(0)
        eval_logs = [
            (m, s)
            for m, s in ml_logger.log_calls
            if any(k.startswith("eval/") for k in m)
        ]
        assert len(eval_logs) == 1
        _metrics, step = eval_logs[0]
        assert step == 42

    @pytest.mark.asyncio
    async def test_explicit_step_num_not_overridden(self):
        ml_logger = _CapturingLogger()
        pipeline = EvalPipeline(
            agent=_PolicyStepAgent(42.0),
            config=_FakeConfig(),
            ml_logger=ml_logger,
            eval_iter=_ResettableIter([["task1"]]),
            semaphore=asyncio.Semaphore(10),
        )
        await pipeline.run(100)
        eval_logs = [
            (m, s)
            for m, s in ml_logger.log_calls
            if any(k.startswith("eval/") for k in m)
        ]
        assert len(eval_logs) == 1
        _metrics, step = eval_logs[0]
        assert step == 100


class TestEvalSourceLabelling:
    """Each sweep reports WHICH model it sampled via ``on_eval_started``.

    Both the live-sampler sweep and the checkpoint-endpoint sweep run under
    ``phase="eval"`` at the same step, so an observer has nothing else to
    attribute their trajectories with.
    """

    class _RecordingObserver:
        def __init__(self):
            self.started: list[tuple[int, str]] = []

        async def on_eval_started(self, eval_step, eval_source="sampler"):
            self.started.append((eval_step, eval_source))

        async def on_eval_item_completed(self, **kw):
            pass

        async def on_eval_completed(self, eval_step):
            pass

    def _run_with_source(self, **kwargs) -> list[tuple[int, str]]:
        observer = self._RecordingObserver()
        pipeline = EvalPipeline(
            agent=_FakeAgent(),
            config=_FakeConfig(),
            ml_logger=_CapturingLogger(),
            eval_dataset=_FakeDataset(tasks=["task-a"]),
            progress_observer=observer,
            **kwargs,
        )
        asyncio.run(pipeline.run(7))
        return observer.started

    def test_defaults_to_sampler(self):
        """The training-inline and standalone sweeps are the live sampler,
        so the default must not require every call site to say so."""
        assert self._run_with_source() == [(7, "sampler")]

    def test_explicit_source_is_forwarded(self):
        """The checkpoint sweep passes ``eval_source="checkpoint"``; it must
        reach the observer unchanged, at the same step as the sampler sweep."""
        assert self._run_with_source(eval_source="checkpoint") == [(7, "checkpoint")]
