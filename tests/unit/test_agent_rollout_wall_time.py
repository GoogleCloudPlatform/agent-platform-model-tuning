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

"""``rollout_wall_time_s``: the true end-to-end clock for one rollout.

The metric exists because nothing else in the framework measures it. The
per-step ``sample_time_s`` / ``env_time_s`` pair starts after the env
exists and stops when the step returns, so a consumer that sums them to
get a per-trajectory duration silently drops env construction,
tool/sandbox bring-up, retries and every other wait.
These tests pin that gap, at both seams:

* ``Agent._run_one_rollout`` — the per-rollout fan-out path, one
  measurement per trajectory.
* ``Agent.group_rollout`` — the fallback for agents that override
  ``_group_rollout_internal`` with a single batched call serving all N
  samples, where the group's interval genuinely IS each sample's.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from typing import Any

import pytest

from core import atif_extras, protocols
from core.atif_trajectory import ScoredTrajectory
from core.multi_step_agent import MultiStepAgent
from core.single_step_agent import SingleStepAgent
from core.task_data import TaskData
from google.cloud.aiplatform_v1.types import Content, GenerateContentRequest
from tests.unit._sample_result_factories import make_sample_result


def _prompt_gcr(text: str) -> GenerateContentRequest:
    return GenerateContentRequest(
        contents=[Content(role="user", parts=[{"text": text}])]
    )


@dataclasses.dataclass
class _TaskData(TaskData):
    pass


class _FakeEnv(protocols.Env):
    """Env that finishes on the first step."""

    def get_prompt(self) -> GenerateContentRequest:
        return _prompt_gcr("p")

    async def step(
        self, action: protocols.Action, *, step_count: int = 0
    ) -> protocols.EnvStepResult:
        return protocols.EnvStepResult(
            function_responses=[],
            reward=1.0,
            done=True,
            is_correct=True,
        )


class _FakeSampler(protocols.Sampler):
    model_name = "test-model"

    def __init__(self, delays: list[float] | None = None):
        # Popped in call order, so concurrent rollouts in one group can be
        # given deliberately different durations.
        self._delays = list(delays or [])

    async def sample_n(self, prompt: Any, num_samples: int = 1, **kwargs: Any):
        if self._delays:
            await asyncio.sleep(self._delays.pop(0))
        return [
            make_sample_result(candidate_id=f"c{i}", text=f"a{i}")
            for i in range(num_samples)
        ]


_AGENT_KWARGS = dict(
    recipe_name="rcp",
    tuning_job_id="tj-test",
    extract_prompt_text=lambda td: "x",
)

# Long enough to dwarf scheduling jitter, short enough to keep the suite fast.
_ENV_BUILD_DELAY_S = 0.15


def _step_timing_sum(scored: ScoredTrajectory) -> float:
    """What a Sigma(sample_time_s + env_time_s) consumer would report."""
    return sum(
        (atif_extras.get_sample_time_s(s) or 0.0)
        + (atif_extras.get_env_time_s(s) or 0.0)
        for s in scored.trajectory.steps
    )


def _slow_env_factory(task_data: Any) -> _FakeEnv:
    """Env whose CONSTRUCTION is slow.

    The delay lands before the first agent step, so no per-step timer can
    observe it — which is exactly the class of time this metric adds.
    """
    time.sleep(_ENV_BUILD_DELAY_S)
    return _FakeEnv()


@pytest.mark.asyncio
async def test_fanout_wall_time_covers_what_step_timers_never_see():
    agent = MultiStepAgent(
        sampler=_FakeSampler(),
        env_factory=_slow_env_factory,
        max_steps=4,
        **_AGENT_KWARGS,
    )

    group, _ = await agent.group_rollout(_TaskData(task_id="t1"), num_samples=1)
    scored = group.scored_trajectories[0]

    wall = atif_extras.get_rollout_wall_time_s(scored.trajectory)
    assert wall is not None, (
        "rollout_wall_time_s was never stamped; Agent._run_one_rollout is "
        "not measuring the rollout"
    )
    assert wall >= _ENV_BUILD_DELAY_S, (
        f"wall time {wall:.3f}s < the {_ENV_BUILD_DELAY_S}s spent building "
        "the env, so the measurement does not span the whole rollout"
    )
    step_sum = _step_timing_sum(scored)
    assert wall > step_sum, (
        f"wall time {wall:.3f}s is not greater than the per-step timing sum "
        f"{step_sum:.3f}s — the metric adds nothing over Sigma(steps)"
    )


@pytest.mark.asyncio
async def test_each_rollout_in_a_fanout_group_is_measured_separately():
    """One straggler among three concurrent rollouts must show up as ONE
    long sample, not as three.

    This is what separates the per-rollout seam from the group-wide
    fallback: the fallback would stamp every trajectory with the group's
    (i.e. the straggler's) duration, flattening the spread the p95 exists
    to expose.
    """
    straggler_s, fast_s = 0.30, 0.01
    agent = MultiStepAgent(
        sampler=_FakeSampler(delays=[straggler_s, fast_s, fast_s]),
        env_factory=lambda td: _FakeEnv(),
        max_steps=4,
        **_AGENT_KWARGS,
    )

    group, _ = await agent.group_rollout(_TaskData(task_id="t1"), num_samples=3)

    walls = [
        atif_extras.get_rollout_wall_time_s(s.trajectory)
        for s in group.scored_trajectories
    ]
    assert len(walls) == 3
    assert all(w is not None and w > 0.0 for w in walls), walls
    spread = max(walls) - min(walls)  # type: ignore[type-var]
    assert spread > (straggler_s - fast_s) / 2, (
        f"per-rollout wall times {walls} are nearly identical (spread "
        f"{spread:.3f}s) even though one rollout ran {straggler_s}s and the "
        f"others {fast_s}s — the group-wide value is being stamped on every "
        "trajectory instead of each rollout's own"
    )


@pytest.mark.asyncio
async def test_batched_agent_gets_the_group_fallback_measurement():
    # SingleStepAgent overrides _group_rollout_internal and never calls
    # _run_one_rollout, so without the group_rollout fallback every
    # single-step recipe would emit no wall time at all.
    agent = SingleStepAgent(
        _FakeSampler(),
        _slow_env_factory,
        **_AGENT_KWARGS,
    )

    group, _ = await agent.group_rollout(_TaskData(task_id="t1"), num_samples=2)

    walls = [
        atif_extras.get_rollout_wall_time_s(s.trajectory)
        for s in group.scored_trajectories
    ]
    assert len(walls) == 2
    assert all(w is not None and w >= _ENV_BUILD_DELAY_S for w in walls), walls
