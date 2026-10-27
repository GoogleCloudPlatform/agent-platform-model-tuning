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

"""_run_one_rollout binds a per-rollout log and uploads it unconditionally."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

import pytest

from core.protocols import Action, Env, EnvStepResult, Sampler
from core.task_data import TaskData
from google.cloud.aiplatform_v1.types import Content, GenerateContentRequest
from utils.rollout_log import RolloutLogHandler, current_rollout_log_path


def _prompt_gcr(text: str) -> GenerateContentRequest:
    return GenerateContentRequest(
        contents=[Content(role="user", parts=[{"text": text}])]
    )


@dataclasses.dataclass
class _TestTaskData(TaskData):
    """Lightweight TaskData subclass for rollout-log tests."""

    pass


class _FakeEnv(Env):
    def get_prompt(self) -> GenerateContentRequest:
        return _prompt_gcr("p")

    async def step(self, action: Action, *, step_count: int = 0) -> EnvStepResult:  # type: ignore[override]
        return EnvStepResult(
            function_responses=[],
            reward=0.5,
            done=True,
            is_correct=True,
            task_stats=None,
        )


class _FakeSampler(Sampler):
    model_name = "test-model"

    async def sample_n(
        self,
        prompt,
        num_samples: int = 1,
        thinking_level=None,
        max_generation_tokens=None,
        capture_ctxs=None,
    ):
        from tests.unit._sample_result_factories import make_sample_result

        return [
            make_sample_result(candidate_id=f"c{i}", text=f"a{i}")
            for i in range(num_samples)
        ]


@dataclasses.dataclass
class _FakeCaptureCtx:
    """Stand-in for ``utils.raw_capture.CaptureContext``.

    Only ``trial_dir`` and ``on_written`` are read by the code under test, so
    a real CaptureContext (which drags in a Counter and a trajectory id) buys
    nothing here.
    """

    trial_dir: Path
    enqueued: list[Path] = dataclasses.field(default_factory=list)

    def __post_init__(self) -> None:
        self.on_written = self.enqueued.append


@pytest.fixture
def handler():
    """Install a RolloutLogHandler on root; remove it so tests don't leak state."""
    h = RolloutLogHandler()
    root = logging.getLogger()
    root.addHandler(h)
    old_level = root.level
    root.setLevel(logging.INFO)
    yield h
    root.removeHandler(h)
    root.setLevel(old_level)
    h.close()


def _make_agent(rollout_impl, capture_ctx):
    """Build a SingleStepAgent whose rollout() and capture ctx are stubbed."""
    from core.single_step_agent import SingleStepAgent

    class _Agent(SingleStepAgent):
        async def _resolve_capture_ctx(self, **kwargs):  # type: ignore[override]
            return capture_ctx

        async def rollout(self, task_data, *args, **kwargs):  # type: ignore[override]
            return await rollout_impl(task_data)

    return _Agent(
        _FakeSampler(),
        lambda td: _FakeEnv(),
        recipe_name="rcp",
        tuning_job_id="tj-1",
        extract_prompt_text=lambda td: "x",
    )


async def _run(agent, task_data):
    return await agent._run_one_rollout(
        task_data,
        0,
        thinking_level=None,
        max_generation_tokens=None,
        phase="train",
        dispatch_index=None,
        on_trajectory_done=None,
    )


@pytest.mark.asyncio
async def test_binds_log_path_under_the_trial_dir(tmp_path, handler):
    ctx = _FakeCaptureCtx(trial_dir=tmp_path / "task-0")
    seen: list[Path | None] = []

    async def rollout_impl(task_data):
        seen.append(current_rollout_log_path())
        raise RuntimeError("stop-after-observing")

    agent = _make_agent(rollout_impl, ctx)
    with pytest.raises(RuntimeError):
        await _run(agent, _TestTaskData(task_id="t1"))

    assert seen == [tmp_path / "task-0" / "agent" / "agent.log"]


@pytest.mark.asyncio
async def test_uploads_the_log_even_when_the_rollout_raises(tmp_path, handler):
    """THE acceptance test: a crashed rollout still ships its log."""
    ctx = _FakeCaptureCtx(trial_dir=tmp_path / "task-0")

    async def rollout_impl(task_data):
        logging.getLogger("t.rollout").error("about to explode")
        raise RuntimeError("boom")

    agent = _make_agent(rollout_impl, ctx)
    with pytest.raises(RuntimeError):
        await _run(agent, _TestTaskData(task_id="t1"))

    log_path = tmp_path / "task-0" / "agent" / "agent.log"
    assert ctx.enqueued == [log_path]
    assert "about to explode" in log_path.read_text()
