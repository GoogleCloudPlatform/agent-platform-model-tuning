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

"""Regression: Trajectory.agent.model_name comes from the sampler,
not the hardcoded '' sentinel.

Design: sampler.model_name (natural) -> "unknown" (fallback).

Covers the ``core.single_step_agent.SingleStepAgent`` path. Other
recipe agents' coverage lives alongside their own trees.
"""

from __future__ import annotations

import dataclasses
from typing import Any, List

import pytest

from core.protocols import Action, Env, EnvStepResult, Sampler
from core.task_data import TaskData
from google.cloud.aiplatform_v1.types import Content, GenerateContentRequest


def _prompt_gcr(text: str) -> GenerateContentRequest:
    """Wrap a plain string prompt into a minimal GenerateContentRequest."""
    return GenerateContentRequest(
        contents=[Content(role="user", parts=[{"text": text}])]
    )


@dataclasses.dataclass
class _TestTaskData(TaskData):
    """Lightweight TaskData subclass for tests that don't care about payload."""

    pass


class _FakeEnv(Env):
    """Minimal Env that returns a done step immediately."""

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


class _SamplerWithModelName(Sampler):
    """Sampler test-double that exposes ``.model_name`` (natural path)."""

    def __init__(self, model_name: str, responses: List[Any] | None = None):
        self.model_name = model_name
        self._responses = responses or []

    async def sample_n(
        self,
        prompt,
        num_samples: int = 1,
        **kwargs,
    ):
        # Empty response -> SingleStepAgent's fast path (no agent step),
        # which still builds the AtifAgent from the adapter/sampler.
        return list(self._responses)


class _SamplerWithoutModelName(Sampler):
    """Sampler test-double that does NOT expose ``.model_name``.

    Used to verify the "unknown" fallback fires only when neither the
    adapter nor the sampler carries a name.
    """

    async def sample_n(
        self,
        prompt,
        num_samples: int = 1,
        **kwargs,
    ):
        return []


_AGENT_KWARGS = dict(
    recipe_name="test",
    tuning_job_id="tj-test",
    extract_prompt_text=lambda td: "x",
)


@pytest.mark.asyncio
async def test_single_step_agent_reads_model_name_from_sampler():
    """``SingleStepAgent`` populates ``Trajectory.agent.model_name`` from
    ``sampler.model_name``."""
    from core.single_step_agent import SingleStepAgent

    sampler = _SamplerWithModelName("gemini-3.1-flash-lite")
    agent = SingleStepAgent(sampler, lambda td: _FakeEnv(), **_AGENT_KWARGS)
    group, _ = await agent.group_rollout(_TestTaskData(task_id="t1"), num_samples=1)
    assert (
        group.scored_trajectories[0].trajectory.agent.model_name
        == "gemini-3.1-flash-lite"
    )


@pytest.mark.asyncio
async def test_single_step_agent_falls_back_to_unknown():
    """Sampler has no ``model_name`` -> ``"unknown"`` sentinel."""
    from core.single_step_agent import SingleStepAgent

    sampler = _SamplerWithoutModelName()
    agent = SingleStepAgent(sampler, lambda td: _FakeEnv(), **_AGENT_KWARGS)
    group, _ = await agent.group_rollout(_TestTaskData(task_id="t1"), num_samples=1)
    assert group.scored_trajectories[0].trajectory.agent.model_name == "unknown"
