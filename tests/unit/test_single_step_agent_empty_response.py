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

"""Regression: SingleStepAgent's empty-sampler-response path emits
SAMPLING_FAILURE_UNHANDLED, not OK.

Empty response means the sampler returned zero candidates. This is
infra weirdness, not a legitimate low-reward completion, so it must
be classified as a sampling failure (excluded from transient-retry
gate, drops via the standard <2 valid check downstream).
"""

from __future__ import annotations

import dataclasses
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.protocols import Action
from core.single_step_agent import SingleStepAgent
from core.atif_extras import get_rollout_status
from core.rollout_status import RolloutStatus
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


class _FakeEnv:
    def __init__(self):
        self.done = False

    def get_prompt(self):
        return _prompt_gcr("prompt")

    async def step(self, action: Action, *, step_count: int = 0):
        result = MagicMock()
        result.done = True
        result.reward = 0.5
        result.function_responses = []
        result.task_stats = None
        result.is_correct = True
        return result


_AGENT_KWARGS = dict(
    recipe_name="test_recipe",
    tuning_job_id="tj-test",
    extract_prompt_text=lambda td: "test prompt",
)


@pytest.mark.asyncio
async def test_empty_response_is_not_transient_failure():
    """SAMPLING_FAILURE_UNHANDLED must NOT be classified as transient
    (so the transient-retry gate doesn't spam retries on deterministic
    input failures)."""
    assert not RolloutStatus.SAMPLING_FAILURE_UNHANDLED.is_transient_failure()
