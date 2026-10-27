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

"""Tests for core.multi_step_agent.MultiStepAgent."""

import asyncio
import math
from typing import Any

import pytest

from core import atif_extras, protocols
from core import context_compaction as _cc
from core.atif_trajectory import ScoredTrajectory
from core.multi_step_agent import MultiStepAgent
from core.rollout_status import RolloutStatus
from core.task_data import TaskData
from google.cloud.aiplatform_v1.types import (
    Content as _Content,
    GenerateContentRequest as _GenerateContentRequest,
)
from tests.unit._sample_result_factories import make_sample_result


def _prompt_gcr(text: str) -> _GenerateContentRequest:
    """Wrap a plain string prompt into a minimal GenerateContentRequest."""
    return _GenerateContentRequest(
        contents=[_Content(role="user", parts=[{"text": text}])]
    )


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeEnv(protocols.Env):
    """Multi-step env that returns done after ``done_at_step`` calls."""

    def __init__(
        self,
        prompt: str = "test prompt",
        done_at_step: int = 1,
        reward: float = 1.0,
        function_responses: list[dict[str, Any]] | None = None,
    ):
        self._prompt = prompt
        self._done_at_step = done_at_step
        self._reward = reward
        self._function_responses = function_responses or []
        self._step_count = 0

    def get_prompt(self) -> _GenerateContentRequest:
        return _prompt_gcr(self._prompt)

    async def step(
        self, action: protocols.Action, *, step_count: int = 0
    ) -> protocols.EnvStepResult:
        self._step_count += 1
        done = self._step_count >= self._done_at_step
        reward = self._reward if done else None
        return protocols.EnvStepResult(
            function_responses=list(self._function_responses),
            reward=reward,
            done=done,
            is_correct=True if done else None,
        )


class _FakeSampler(protocols.Sampler):
    """Returns a fixed SampleResult for every sample() call."""

    def __init__(
        self,
        result: Any = None,
        results: list[Any] | None = None,
    ):
        # If results is given, pop from it per call. Otherwise use result forever.
        self._fixed = result
        self._results_queue = list(results) if results is not None else None
        self.model_name = "test-model"

    async def sample_n(
        self,
        prompt,
        num_samples=1,
        **kwargs,
    ):
        if self._results_queue is not None:
            if not self._results_queue:
                return []
            return [self._results_queue.pop(0)]
        if self._fixed is None:
            return []
        return [self._fixed]


class _SlowSampler(protocols.Sampler):
    """Sampler that sleeps, used to test timeout."""

    def __init__(self, delay: float):
        self._delay = delay
        self.model_name = "slow-model"

    async def sample_n(
        self,
        prompt,
        num_samples=1,
        **kwargs,
    ):
        await asyncio.sleep(self._delay)
        return [make_sample_result(text="slow", finish_reason="STOP")]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task_data(task_id: str = "task-1") -> TaskData:
    return TaskData(task_id=task_id)


def _default_kwargs() -> dict:
    return dict(
        recipe_name="test_recipe",
        tuning_job_id="tj-test",
        extract_prompt_text=lambda td: "test prompt text",
    )


def _make_agent(
    sampler: protocols.Sampler,
    env_factory=None,
    max_steps: int = 10,
    **kwargs,
) -> MultiStepAgent:
    merged = {**_default_kwargs(), **kwargs}
    if env_factory is None:
        env_factory = lambda td: _FakeEnv(done_at_step=2, reward=1.0)
    return MultiStepAgent(
        sampler=sampler,
        env_factory=env_factory,
        max_steps=max_steps,
        **merged,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMultiStepAgentHappyPath:
    """3-step env that returns done on step 2 → 2 agent steps, reward=1.0."""

    @pytest.mark.asyncio
    async def test_happy_path_returns_scored_trajectory(self):
        result = make_sample_result(text="hello", finish_reason="STOP")
        sampler = _FakeSampler(result=result)
        env_factory = lambda td: _FakeEnv(done_at_step=2, reward=5.0)
        agent = _make_agent(sampler, env_factory=env_factory, max_steps=10)

        scored = await agent.rollout(_make_task_data())
        assert scored is not None
        assert isinstance(scored, ScoredTrajectory)

    @pytest.mark.asyncio
    async def test_happy_path_reward(self):
        result = make_sample_result(text="hello", finish_reason="STOP")
        sampler = _FakeSampler(result=result)
        env_factory = lambda td: _FakeEnv(done_at_step=2, reward=5.0)
        agent = _make_agent(sampler, env_factory=env_factory, max_steps=10)

        scored = await agent.rollout(_make_task_data())
        assert scored.reward.reward == pytest.approx(5.0)

    @pytest.mark.asyncio
    async def test_happy_path_status_ok(self):
        result = make_sample_result(text="hello", finish_reason="STOP")
        sampler = _FakeSampler(result=result)
        env_factory = lambda td: _FakeEnv(done_at_step=2, reward=5.0)
        agent = _make_agent(sampler, env_factory=env_factory, max_steps=10)

        scored = await agent.rollout(_make_task_data())
        status = atif_extras.get_rollout_status(scored.trajectory)
        assert status == RolloutStatus.OK

    @pytest.mark.asyncio
    async def test_happy_path_step_count(self):
        """2-step env: step 1 not done, step 2 done → 2 agent steps."""
        result = make_sample_result(text="hello", finish_reason="STOP")
        sampler = _FakeSampler(result=result)
        env_factory = lambda td: _FakeEnv(done_at_step=2, reward=1.0)
        agent = _make_agent(sampler, env_factory=env_factory, max_steps=10)

        scored = await agent.rollout(_make_task_data())
        agent_steps = [s for s in scored.trajectory.steps if s.source == "agent"]
        assert len(agent_steps) == 2

    @pytest.mark.asyncio
    async def test_happy_path_leading_user_step(self):
        result = make_sample_result(text="hello", finish_reason="STOP")
        sampler = _FakeSampler(result=result)
        agent = _make_agent(sampler, max_steps=10)

        scored = await agent.rollout(_make_task_data())
        user_steps = [s for s in scored.trajectory.steps if s.source == "user"]
        assert len(user_steps) == 1
        assert user_steps[0].message == "test prompt"

    @pytest.mark.asyncio
    async def test_happy_path_1_step_done_immediately(self):
        """Env done on step 1 → 1 agent step."""
        result = make_sample_result(text="hello", finish_reason="STOP")
        sampler = _FakeSampler(result=result)
        env_factory = lambda td: _FakeEnv(done_at_step=1, reward=3.0)
        agent = _make_agent(sampler, env_factory=env_factory, max_steps=10)

        scored = await agent.rollout(_make_task_data())
        agent_steps = [s for s in scored.trajectory.steps if s.source == "agent"]
        assert len(agent_steps) == 1
        assert scored.reward.reward == pytest.approx(3.0)

    @pytest.mark.asyncio
    async def test_happy_path_is_correct_set(self):
        result = make_sample_result(text="hello", finish_reason="STOP")
        sampler = _FakeSampler(result=result)
        env_factory = lambda td: _FakeEnv(done_at_step=1, reward=1.0)
        agent = _make_agent(sampler, env_factory=env_factory, max_steps=10)

        scored = await agent.rollout(_make_task_data())
        # _FakeEnv returns is_correct=True on done
        assert atif_extras.get_is_correct(scored.trajectory) is True

    @pytest.mark.asyncio
    async def test_agent_step_carries_the_prompt_token_count(self):
        """A backend-reported prompt count must reach ``step.extra``.

        ``SampleResult.prompt_token_count`` feeds ``to_step_stats()``,
        which ``TrajectoryStep.from_wire`` stamps onto the step; every
        context-size metric reads it back from there.
        """
        result = make_sample_result(
            text="hello",
            finish_reason="STOP",
            prompt_token_count=4321,
            generation_token_count=7,
        )
        sampler = _FakeSampler(result=result)
        env_factory = lambda td: _FakeEnv(done_at_step=1, reward=1.0)
        agent = _make_agent(sampler, env_factory=env_factory, max_steps=10)

        scored = await agent.rollout(_make_task_data())
        agent_steps = [s for s in scored.trajectory.steps if s.source == "agent"]
        assert agent_steps
        assert atif_extras.get_prompt_token_count(agent_steps[-1]) == 4321


class TestMultiStepAgentMaxStepsExhausted:
    """Agent hits max_steps without env returning done=True."""

    @pytest.mark.asyncio
    async def test_max_steps_reward_is_lowest(self):
        result = make_sample_result(text="loop", finish_reason="STOP")
        sampler = _FakeSampler(result=result)
        # Env never returns done — done_at_step=999 vs max_steps=3
        env_factory = lambda td: _FakeEnv(done_at_step=999, reward=10.0)
        agent = _make_agent(sampler, env_factory=env_factory, max_steps=3)

        scored = await agent.rollout(_make_task_data())
        # Default lowest_reward is -1.0
        assert scored.reward.reward == pytest.approx(-1.0)

    @pytest.mark.asyncio
    async def test_max_steps_status(self):
        result = make_sample_result(text="loop", finish_reason="STOP")
        sampler = _FakeSampler(result=result)
        env_factory = lambda td: _FakeEnv(done_at_step=999, reward=10.0)
        agent = _make_agent(sampler, env_factory=env_factory, max_steps=3)

        scored = await agent.rollout(_make_task_data())
        status = atif_extras.get_rollout_status(scored.trajectory)
        assert status == RolloutStatus.MAX_STEP_LIMIT

    @pytest.mark.asyncio
    async def test_max_steps_step_count(self):
        result = make_sample_result(text="loop", finish_reason="STOP")
        sampler = _FakeSampler(result=result)
        env_factory = lambda td: _FakeEnv(done_at_step=999, reward=10.0)
        agent = _make_agent(sampler, env_factory=env_factory, max_steps=3)

        scored = await agent.rollout(_make_task_data())
        agent_steps = [s for s in scored.trajectory.steps if s.source == "agent"]
        assert len(agent_steps) == 3


class TestMultiStepAgentEmptySamplerResponse:
    """Sampler returns None → sampling failure."""

    @pytest.mark.asyncio
    async def test_none_result_gives_nan_reward(self):
        sampler = _FakeSampler(result=None)  # returns [] → sample() returns None
        agent = _make_agent(sampler, max_steps=5)

        scored = await agent.rollout(_make_task_data())
        assert scored is not None
        assert math.isnan(scored.reward.reward)

    @pytest.mark.asyncio
    async def test_none_result_status_is_sampling_failure(self):
        sampler = _FakeSampler(result=None)
        agent = _make_agent(sampler, max_steps=5)

        scored = await agent.rollout(_make_task_data())
        status = atif_extras.get_rollout_status(scored.trajectory)
        assert status == RolloutStatus.SAMPLING_FAILURE_UNHANDLED

    @pytest.mark.asyncio
    async def test_none_result_has_agent_step(self):
        """Even on sampling failure, we emit at least one agent step."""
        sampler = _FakeSampler(result=None)
        agent = _make_agent(sampler, max_steps=5)

        scored = await agent.rollout(_make_task_data())
        agent_steps = [s for s in scored.trajectory.steps if s.source == "agent"]
        assert len(agent_steps) >= 1


class TestMultiStepAgentTimeout:
    """Timeout wraps the loop in asyncio.wait_for."""

    @pytest.mark.asyncio
    async def test_timeout_gives_nan_reward(self):
        sampler = _SlowSampler(delay=5.0)
        agent = _make_agent(
            sampler,
            max_steps=10,
            agent_rollout_timeout_seconds=0.05,
        )

        scored = await agent.rollout(_make_task_data())
        assert scored is not None
        assert math.isnan(scored.reward.reward)

    @pytest.mark.asyncio
    async def test_timeout_status(self):
        sampler = _SlowSampler(delay=5.0)
        agent = _make_agent(
            sampler,
            max_steps=10,
            agent_rollout_timeout_seconds=0.05,
        )

        scored = await agent.rollout(_make_task_data())
        status = atif_extras.get_rollout_status(scored.trajectory)
        assert status == RolloutStatus.SAMPLING_FAILURE_TIMEOUT


class TestMultiStepAgentNonStopFinishReason:
    """Non-STOP finish reason → triage as sampling error, episode ends."""

    @pytest.mark.asyncio
    async def test_max_tokens_ends_episode(self):
        result = make_sample_result(text="partial", finish_reason="MAX_TOKENS")
        sampler = _FakeSampler(result=result)
        env_factory = lambda td: _FakeEnv(done_at_step=5, reward=1.0)
        agent = _make_agent(sampler, env_factory=env_factory, max_steps=10)

        scored = await agent.rollout(_make_task_data())
        # Default triage returns NaN + SAMPLING_FAILURE_UNHANDLED
        assert math.isnan(scored.reward.reward)
        agent_steps = [s for s in scored.trajectory.steps if s.source == "agent"]
        # Should stop after first step (non-STOP finish reason)
        assert len(agent_steps) == 1


class TestMultiStepAgentGroupRollout:
    """group_rollout fans out N independent rollout() calls via base Agent."""

    @pytest.mark.asyncio
    async def test_group_rollout_produces_n_trajectories(self):
        result = make_sample_result(text="hello", finish_reason="STOP")
        sampler = _FakeSampler(result=result)
        env_factory = lambda td: _FakeEnv(done_at_step=1, reward=1.0)
        agent = _make_agent(sampler, env_factory=env_factory, max_steps=5)

        group, metrics = await agent.group_rollout(_make_task_data(), num_samples=3)
        assert len(group.scored_trajectories) == 3


class TestMultiStepAgentCustomLowestReward:
    """Custom lowest_reward is used on max_steps exhaustion."""

    @pytest.mark.asyncio
    async def test_custom_lowest_reward(self):
        result = make_sample_result(text="loop", finish_reason="STOP")
        sampler = _FakeSampler(result=result)
        env_factory = lambda td: _FakeEnv(done_at_step=999, reward=10.0)
        agent = _make_agent(
            sampler, env_factory=env_factory, max_steps=2, lowest_reward=-50.0
        )

        scored = await agent.rollout(_make_task_data())
        assert scored.reward.reward == pytest.approx(-50.0)


class TestMultiStepAgentAtifShape:
    """Verify ATIF trajectory shape details."""

    @pytest.mark.asyncio
    async def test_agent_step_has_candidate_id(self):
        result = make_sample_result(
            text="hello", finish_reason="STOP", candidate_id="abc123"
        )
        sampler = _FakeSampler(result=result)
        env_factory = lambda td: _FakeEnv(done_at_step=1, reward=1.0)
        agent = _make_agent(sampler, env_factory=env_factory, max_steps=5)

        scored = await agent.rollout(_make_task_data())
        agent_steps = [s for s in scored.trajectory.steps if s.source == "agent"]
        assert len(agent_steps) == 1
        assert atif_extras.get_candidate_id(agent_steps[0]) == "abc123"

    @pytest.mark.asyncio
    async def test_session_id_set(self):
        result = make_sample_result(text="hello", finish_reason="STOP")
        sampler = _FakeSampler(result=result)
        agent = _make_agent(sampler, max_steps=5)

        scored = await agent.rollout(_make_task_data())
        assert scored.trajectory.session_id is not None
        assert "test_recipe" in scored.trajectory.session_id

    @pytest.mark.asyncio
    async def test_step_ids_sequential(self):
        """Leading user step_id=1, agent steps start at 2."""
        result = make_sample_result(text="hello", finish_reason="STOP")
        sampler = _FakeSampler(result=result)
        env_factory = lambda td: _FakeEnv(done_at_step=3, reward=1.0)
        agent = _make_agent(sampler, env_factory=env_factory, max_steps=10)

        scored = await agent.rollout(_make_task_data())
        ids = [s.step_id for s in scored.trajectory.steps]
        # user(1), agent(2), agent(3), agent(4)
        assert ids == list(range(1, len(ids) + 1))


class TestMultiStepAgentTriageHookOverride:
    """Subclass can override _triage_sampling_error to handle non-STOP finish reasons."""

    @pytest.mark.asyncio
    async def test_override_triage_hook_returns_custom_reward(self):
        """Subclass returns custom reward for MAX_TOKENS finish reason."""

        class _CustomTriageAgent(MultiStepAgent):
            def _triage_sampling_error(self, result, finish_reason):
                if finish_reason == "MAX_TOKENS":
                    return -99.0, RolloutStatus.SAMPLING_FAILURE_UNHANDLED
                return super()._triage_sampling_error(result, finish_reason)

        result = make_sample_result(text="partial", finish_reason="MAX_TOKENS")
        sampler = _FakeSampler(result=result)
        env_factory = lambda td: _FakeEnv(done_at_step=5, reward=1.0)
        agent = _CustomTriageAgent(
            sampler=sampler,
            env_factory=env_factory,
            max_steps=10,
            **_default_kwargs(),
        )

        scored = await agent.rollout(_make_task_data())
        assert scored.reward.reward == pytest.approx(-99.0)
        status = atif_extras.get_rollout_status(scored.trajectory)
        assert status == RolloutStatus.SAMPLING_FAILURE_UNHANDLED


class TestMultiStepAgentFunctionResponses:
    """_append_turn appends function_responses to observation across turns."""

    @pytest.mark.asyncio
    async def test_function_responses_grow_observation(self):
        """Non-empty function_responses are appended to conversation each turn."""
        fr = {
            "functionResponse": {
                "name": "bash",
                "response": {"output": "hello"},
            }
        }
        result = make_sample_result(text="action", finish_reason="STOP")
        sampler = _FakeSampler(result=result)
        # 3-step env: steps 1-2 not done (function_responses appended),
        # step 3 done (reward returned).
        env_factory = lambda td: _FakeEnv(
            done_at_step=3, reward=2.0, function_responses=[fr]
        )
        agent = _make_agent(sampler, env_factory=env_factory, max_steps=10)

        scored = await agent.rollout(_make_task_data())
        agent_steps = [s for s in scored.trajectory.steps if s.source == "agent"]
        # 3-step env → 3 agent steps
        assert len(agent_steps) == 3
        assert scored.reward.reward == pytest.approx(2.0)
        status = atif_extras.get_rollout_status(scored.trajectory)
        assert status == RolloutStatus.OK


class TestMultiStepAgentParallelFunctionResponses:
    """Parallel function responses go back as ONE Content with N parts."""

    @staticmethod
    def _two_frs() -> list[dict[str, Any]]:
        return [
            {
                "functionResponse": {
                    "name": "bash",
                    "response": {"output": "one"},
                }
            },
            {
                "functionResponse": {
                    "name": "bash",
                    "response": {"output": "two"},
                }
            },
        ]

    @pytest.mark.asyncio
    async def test_parallel_responses_share_one_content(self):
        """Two function responses -> one user Content with two parts."""
        sampler = _RecordingSampler(
            [make_sample_result(text="action", finish_reason="STOP")] * 2
        )
        env_factory = lambda td: _FakeEnv(
            done_at_step=2, reward=1.0, function_responses=self._two_frs()
        )
        agent = _make_agent(sampler, env_factory=env_factory, max_steps=10)

        await agent.rollout(_make_task_data())

        assert len(sampler.prompts) == 2, (
            f"expected 2 sampler calls, got {len(sampler.prompts)}"
        )

        # Turn 2's prompt = [initial user, model turn, ONE tool Content].
        turn2 = sampler.prompts[1]
        assert len(turn2.contents) == 3, (
            f"expected 3 contents (user, model, grouped tool), got "
            f"{[c.role for c in turn2.contents]}"
        )
        tool_content = turn2.contents[2]
        assert tool_content.role == "user"
        assert len(tool_content.parts) == 2

    @pytest.mark.asyncio
    async def test_parallel_responses_preserve_order(self):
        """Part i of the grouped Content is env response i."""
        sampler = _RecordingSampler(
            [make_sample_result(text="action", finish_reason="STOP")] * 2
        )
        env_factory = lambda td: _FakeEnv(
            done_at_step=2, reward=1.0, function_responses=self._two_frs()
        )
        agent = _make_agent(sampler, env_factory=env_factory, max_steps=10)

        await agent.rollout(_make_task_data())

        assert len(sampler.prompts) == 2, (
            f"expected 2 sampler calls, got {len(sampler.prompts)}"
        )

        parts = sampler.prompts[1].contents[2].parts
        outputs = [dict(p.function_response.response).get("output") for p in parts]
        assert outputs == ["one", "two"]

    @pytest.mark.asyncio
    async def test_no_empty_content_when_no_function_responses(self):
        """Zero responses -> no tool Content appended at all."""
        sampler = _RecordingSampler(
            [make_sample_result(text="action", finish_reason="STOP")] * 2
        )
        env_factory = lambda td: _FakeEnv(
            done_at_step=2, reward=1.0, function_responses=[]
        )
        agent = _make_agent(sampler, env_factory=env_factory, max_steps=10)

        await agent.rollout(_make_task_data())

        assert len(sampler.prompts) == 2, (
            f"expected 2 sampler calls, got {len(sampler.prompts)}"
        )

        # [initial user, model turn] only.
        assert len(sampler.prompts[1].contents) == 2


class TestTriagePromptTooLong:
    """Base triage maps the backend's overflow flag to a real status."""

    def test_prompt_too_long_maps_to_status_and_lowest_reward(self):
        # candidate_id="" is the backend's empty-result shape for an
        # overflow (finish_reason then resolves to "OTHER").
        result = make_sample_result(candidate_id="", is_prompt_too_long=True)
        agent = _make_agent(_FakeSampler(), lowest_reward=-1.0)
        reward, status = agent._triage_sampling_error(result, result.finish_reason)
        assert status == RolloutStatus.SAMPLING_FAILURE_PROMPT_TOO_LONG
        assert reward == -1.0

    def test_other_failures_still_unhandled_with_nan(self):
        result = make_sample_result(candidate_id="", is_prompt_too_long=False)
        agent = _make_agent(_FakeSampler())
        reward, status = agent._triage_sampling_error(result, result.finish_reason)
        assert status == RolloutStatus.SAMPLING_FAILURE_UNHANDLED
        assert math.isnan(reward)

    @pytest.mark.asyncio
    async def test_overflow_rollout_scores_lowest_reward_not_nan(self):
        """The point of the change: an overflowed rollout must stay in
        the reward mean as a loss, not vanish as missing data."""
        result = make_sample_result(candidate_id="", is_prompt_too_long=True)
        agent = _make_agent(_FakeSampler(result=result), lowest_reward=-1.0)

        scored = await agent.rollout(_make_task_data())

        assert scored.reward.reward == pytest.approx(-1.0)
        assert not math.isnan(scored.reward.reward)
        assert atif_extras.get_rollout_status(scored.trajectory) == (
            RolloutStatus.SAMPLING_FAILURE_PROMPT_TOO_LONG
        )


class TestTriageMalformedUsage:
    """A success response missing its token counts is a client error.

    The backend returns its failsafe result rather than raising (raising
    loses the whole task), so triage is what makes the failure countable
    under a status of its own instead of folding into UNHANDLED.
    """

    def test_malformed_usage_maps_to_client_error(self):
        result = make_sample_result(candidate_id="", is_malformed_usage=True)
        agent = _make_agent(_FakeSampler(), lowest_reward=-1.0)
        reward, status = agent._triage_sampling_error(result, result.finish_reason)
        assert status == RolloutStatus.SAMPLING_FAILURE_SAMPLING_CLIENT_ERROR
        assert math.isnan(reward)

    def test_prompt_too_long_wins_over_malformed_usage(self):
        """Overflow is the more specific diagnosis and is checked first."""
        result = make_sample_result(
            candidate_id="", is_prompt_too_long=True, is_malformed_usage=True
        )
        agent = _make_agent(_FakeSampler(), lowest_reward=-1.0)
        _, status = agent._triage_sampling_error(result, result.finish_reason)
        assert status == RolloutStatus.SAMPLING_FAILURE_PROMPT_TOO_LONG


# ---------------------------------------------------------------------------
# core.context_compaction (pure helpers, exercised here to keep one test
# module per component family)
# ---------------------------------------------------------------------------


def _model_content(payload: str) -> _Content:
    return _Content(role="model", parts=[{"text": payload}])


def _tool_content(payload: str) -> _Content:
    return _Content(role="user", parts=[{"text": payload}])


class TestShouldCompact:
    """Trigger rule: fire from the backend's own reported token counts."""

    def test_disabled_when_cap_is_none(self):
        assert not _cc.should_compact(
            prompt_tokens=999_999,
            generation_tokens=999_999,
            max_context_tokens=None,
            trigger_ratio=0.7,
        )

    def test_false_below_threshold(self):
        # 0.7 * 122880 = 86016; 50000 + 10000 is well under.
        assert not _cc.should_compact(
            prompt_tokens=50_000,
            generation_tokens=10_000,
            max_context_tokens=122_880,
            trigger_ratio=0.7,
        )

    def test_true_above_threshold(self):
        """The real failure: 116,962 + 14,065 projected past the cap."""
        assert _cc.should_compact(
            prompt_tokens=116_962,
            generation_tokens=14_065,
            max_context_tokens=122_880,
            trigger_ratio=0.7,
        )

    def test_false_when_counts_unknown(self):
        assert not _cc.should_compact(
            prompt_tokens=None,
            generation_tokens=None,
            max_context_tokens=122_880,
            trigger_ratio=0.7,
        )

    def test_exact_threshold_does_not_fire(self):
        """Pins ``>`` (not ``>=``): landing exactly on the point is fine."""
        assert not _cc.should_compact(
            prompt_tokens=80_000,
            generation_tokens=6_016,  # 80000 + 6016 == 0.7 * 122880
            max_context_tokens=122_880,
            trigger_ratio=0.7,
        )
        assert _cc.should_compact(
            prompt_tokens=80_000,
            generation_tokens=6_017,
            max_context_tokens=122_880,
            trigger_ratio=0.7,
        )


def test_tail_budget_is_a_fifth_of_the_cap():
    assert _cc.tail_token_budget(122_880) == 24_576  # 0.2 * cap, floored


def test_estimate_scales_with_serialized_length():
    small, big = _model_content("x" * 100), _model_content("x" * 10_000)
    assert _cc.estimate_content_tokens(big) > 30 * _cc.estimate_content_tokens(small)
    assert _cc.estimate_content_tokens(big) == pytest.approx(10_000 / 3.1, rel=0.05)


class TestSelectTail:
    """Tail selection: whole exchange pairs, newest-first, under budget."""

    def _contents(self, n_pairs: int, payload: str = "x" * 300) -> list[_Content]:
        out = [_tool_content("the original task")]
        for i in range(n_pairs):
            out.append(_model_content(f"call-{i}-{payload}"))
            out.append(_tool_content(f"resp-{i}"))
        return out

    def test_retains_two_pairs_when_both_fit(self):
        contents = self._contents(5)
        tail, pairs = _cc.select_tail(contents, budget_tokens=24_576)
        assert pairs == 2
        assert len(tail) == 4
        assert tail == contents[-4:]

    def test_never_includes_the_original_task_content(self):
        contents = self._contents(1)
        tail, pairs = _cc.select_tail(contents, budget_tokens=24_576)
        assert pairs == 1
        assert contents[0] not in tail

    def test_drops_second_pair_when_budget_is_tight(self):
        contents = self._contents(3, payload="y" * 30_000)
        tail, pairs = _cc.select_tail(contents, budget_tokens=12_000)
        assert pairs == 1
        assert tail == contents[-2:]

    def test_floor_retains_one_pair_even_when_it_alone_exceeds(self):
        contents = self._contents(3, payload="z" * 200_000)
        tail, pairs = _cc.select_tail(contents, budget_tokens=100)
        assert pairs == 1
        assert tail == contents[-2:]

    def test_never_splits_a_call_from_its_response(self):
        tail, pairs = _cc.select_tail(self._contents(4), budget_tokens=24_576)
        assert pairs == 2
        assert len(tail) % 2 == 0
        assert all(tail[i].role == "model" for i in range(0, len(tail), 2))
        assert all(tail[i].role == "user" for i in range(1, len(tail), 2))

    def test_max_pairs_override_caps_retention(self):
        contents = self._contents(4)
        tail, pairs = _cc.select_tail(contents, budget_tokens=24_576, max_pairs=1)
        assert pairs == 1
        assert tail == contents[-2:]

    def test_empty_body_yields_no_tail(self):
        tail, pairs = _cc.select_tail([_tool_content("task only")], budget_tokens=1000)
        assert tail == []
        assert pairs == 0

    def test_dangling_model_turn_still_yields_the_pairs_behind_it(self):
        """An env step with no function responses leaves a bare model turn.

        Pairing by index parity from the end then misaligns every pair
        behind it and silently returns an empty tail -- for the rest of
        the rollout, since the parity never recovers.
        """
        contents = self._contents(2) + [_model_content("dangling call")]
        assert len(contents[1:]) == 5
        tail, pairs = _cc.select_tail(contents, budget_tokens=24_576)
        assert pairs == 2
        assert tail == contents[1:5]
        assert all(tail[i].role == "model" for i in range(0, len(tail), 2))
        assert all(tail[i].role == "user" for i in range(1, len(tail), 2))

    def test_lone_dangling_model_turn_yields_no_tail(self):
        contents = [_tool_content("the original task"), _model_content("call-0")]
        assert _cc.select_tail(contents, budget_tokens=24_576) == ([], 0)

    def test_odd_body_of_three_yields_the_one_whole_pair(self):
        contents = self._contents(1) + [_model_content("dangling call")]
        assert len(contents[1:]) == 3
        tail, pairs = _cc.select_tail(contents, budget_tokens=24_576)
        assert pairs == 1
        assert tail == contents[1:3]

    def test_mid_history_double_model_stops_rather_than_mis_rolls(self):
        # [m0, r0, m1, m2, r2]: the m1/m2 adjacency is malformed history,
        # not a dangling tail, so the walk stops at the last whole pair.
        contents = [
            _tool_content("the original task"),
            _model_content("call-0"),
            _tool_content("resp-0"),
            _model_content("call-1"),
            _model_content("call-2"),
            _tool_content("resp-2"),
        ]
        tail, pairs = _cc.select_tail(contents, budget_tokens=24_576)
        assert pairs == 1
        assert tail == contents[-2:]


# ---------------------------------------------------------------------------
# In-loop context compaction
# ---------------------------------------------------------------------------


_FR = {"functionResponse": {"name": "bash", "response": {"output": "ok"}}}


def _big_turn():
    """A sample whose counts project past a 1000-token cap at ratio 0.7."""
    return make_sample_result(
        text="turn",
        finish_reason="STOP",
        prompt_token_count=800,
        generation_token_count=100,
    )


def _long_turn(idx: int, *, triggers: bool = False):
    """A bulky, individually identifiable agent turn.

    The text is long enough that eliding pairs really shrinks the
    char-estimated context (the re-entry preamble is itself ~100
    estimated tokens, so toy turns would make the rebuild *bigger*).
    ``triggers`` stamps the reported counts that cross the cap; without
    it the counts are unreported and compaction stays off.
    """
    return make_sample_result(
        text=f"TURN-{idx} " + "x" * 4000,
        finish_reason="STOP",
        prompt_token_count=800 if triggers else None,
        generation_token_count=100 if triggers else 0,
    )


class _ConfiguredEnv(_FakeEnv):
    """Env whose prompt carries the fields a real request carries.

    The summarization call has to preserve these; a bare ``_FakeEnv``
    prompt has none of them, so it cannot catch dropping them.
    """

    def get_prompt(self) -> _GenerateContentRequest:
        req = _prompt_gcr(self._prompt)
        req.model = "models/test-model"
        req.system_instruction = _Content(parts=[{"text": "you are a test agent"}])
        req.tools = [{"function_declarations": [{"name": "bash"}]}]
        req.generation_config = {"temperature": 0.25}
        return req


# Recovery is configuration, not a subclass hook: name the reasons and
# the base loop does the rest, exactly as the recipe agents do.
_RECOVER_ANY = frozenset({"MAX_TOKENS", "MALFORMED_FUNCTION_CALL", "OTHER"})


def _is_compaction(step) -> bool:
    """A compaction step is agent-sourced; its extras are the only tell."""
    return (step.extra or {}).get("compaction_pre_tokens") is not None


class _TokenChargeEnv(_FakeEnv):
    """Env that records the rollout token total handed to it before each step."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.charged: list[int | None] = []

    def set_rollout_generation_tokens(self, total: int | None) -> None:
        self.charged.append(total)


class _RecordingSampler(_FakeSampler):
    """``_FakeSampler`` that snapshots every prompt it is handed.

    The agent mutates the observation in place, so each prompt is copied
    at call time.
    """

    def __init__(self, results):
        super().__init__(results=results)
        self.prompts: list[_GenerateContentRequest] = []

    async def sample_n(self, prompt, num_samples=1, **kwargs):
        self.prompts.append(_GenerateContentRequest(prompt))
        return await super().sample_n(prompt, num_samples=num_samples, **kwargs)


def _long_run_agent(sampler, *, max_steps=8):
    """Agent over an env that never finishes, so the budget bounds it."""
    return _make_agent(
        sampler,
        env_factory=lambda td: _FakeEnv(done_at_step=99, function_responses=[_FR]),
        max_steps=max_steps,
        lowest_reward=-1.0,
        max_context_tokens=1000,
        compaction_trigger_ratio=0.7,
        max_compactions=3,
    )


def _long_run_results():
    """Four bulky agent turns (4 exchange pairs), then the summary."""
    return (
        [_long_turn(0), _long_turn(1), _long_turn(2), _long_turn(3, triggers=True)]
        + [make_sample_result(text="SUMMARY", finish_reason="STOP")]
        + [_long_turn(i) for i in range(4, 8)]
    )


class TestCompactionLoop:
    """End-to-end loop behavior with compaction enabled."""

    def _agent(
        self,
        results,
        *,
        max_steps=6,
        max_compactions=3,
        cap=1000,
        done_at_step=2,
    ):
        return _make_agent(
            _FakeSampler(results=results),
            env_factory=lambda td: _FakeEnv(
                done_at_step=done_at_step, function_responses=[_FR]
            ),
            max_steps=max_steps,
            lowest_reward=-1.0,
            max_context_tokens=cap,
            compaction_trigger_ratio=0.7,
            max_compactions=max_compactions,
        )

    @pytest.mark.asyncio
    async def test_compaction_emits_a_system_step_with_the_summary(self):
        """Turn 0 samples; its 800+100 projection exceeds 0.7*1000, so
        turn 1 is spent summarizing instead of acting."""
        summary = make_sample_result(
            text="SUMMARY: ruled out A and B", finish_reason="STOP"
        )
        agent = self._agent([_big_turn(), summary] + [_big_turn() for _ in range(6)])

        scored = await agent.rollout(_make_task_data())

        # Agent-sourced, NOT system: the model authored the summary and it
        # enters the context, so its candidate must reach the trainer.
        # The compaction_* extras are what identify it.
        events = [s for s in scored.trajectory.steps if _is_compaction(s)]
        assert len(events) == 1
        assert events[0].source == "agent"
        assert "SUMMARY" in events[0].message

    @pytest.mark.asyncio
    async def test_compaction_pre_tokens_is_the_backend_projection(self):
        """``pre`` is the backend's own numbers, not a char estimate.

        ``_compact`` is unreachable unless ``_should_compact`` fired, which
        requires both reported counts, so the projection that tripped the
        trigger (prompt + generation of the last call) is always available.
        """
        agent = _long_run_agent(_FakeSampler(results=_long_run_results()))

        scored = await agent.rollout(_make_task_data())

        step = [s for s in scored.trajectory.steps if _is_compaction(s)][0]
        # _long_turn(3, triggers=True) reports 800 prompt + 100 generation.
        assert step.extra["compaction_pre_tokens"] == 900
        assert step.extra["compaction_pairs_retained"] >= 1

    @pytest.mark.asyncio
    async def test_compaction_post_tokens_backfilled_from_the_next_sample(self):
        """The rebuilt context has never been sampled when the step is
        written, so ``post`` is stamped by the next call that measures it."""
        after = make_sample_result(
            text="after compaction",
            finish_reason="STOP",
            prompt_token_count=250,
            generation_token_count=10,
        )
        results = (
            [_long_turn(0), _long_turn(1), _long_turn(2), _long_turn(3, triggers=True)]
            + [make_sample_result(text="SUMMARY", finish_reason="STOP")]
            + [after]
            + [_long_turn(i) for i in range(5, 9)]
        )
        agent = _long_run_agent(_FakeSampler(results=results))

        scored = await agent.rollout(_make_task_data())

        step = [s for s in scored.trajectory.steps if _is_compaction(s)][0]
        assert step.extra["compaction_post_tokens"] == 250
        assert step.extra["compaction_pre_tokens"] == 900

    @pytest.mark.asyncio
    async def test_compaction_post_tokens_absent_until_backfilled(self):
        """Missing, not zero: a bogus 0 would read as a total reclaim."""
        summary = make_sample_result(text="SUMMARY", finish_reason="STOP")
        # Nothing samples successfully after the summary, so no count lands.
        results = [
            _long_turn(0),
            _long_turn(1),
            _long_turn(2),
            _long_turn(3, triggers=True),
            summary,
            make_sample_result(candidate_id=""),
        ]
        agent = _long_run_agent(_FakeSampler(results=results))

        scored = await agent.rollout(_make_task_data())

        step = [s for s in scored.trajectory.steps if _is_compaction(s)][0]
        assert "compaction_post_tokens" not in (step.extra or {})

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tail_pairs", [None, 1, 3])
    async def test_compaction_tail_pairs_caps_the_retained_pairs(self, tail_pairs):
        """``compaction_tail_pairs`` sets how many recent exchange pairs
        survive verbatim (None = the default, 2). Small turns keep the token
        budget from binding, so the count alone decides."""
        small = [make_sample_result(text="turn", finish_reason="STOP") for _ in range(3)]
        # 8000+100 > 0.7*10000 trips the trigger; the 10k cap gives a 2k-token
        # tail budget, far above 3 tiny pairs.
        trigger = make_sample_result(
            text="turn", finish_reason="STOP",
            prompt_token_count=8000, generation_token_count=100)
        results = (
            small + [trigger]
            + [make_sample_result(text="SUMMARY", finish_reason="STOP")]
            + [make_sample_result(text="turn", finish_reason="STOP") for _ in range(4)]
        )
        kwargs = {} if tail_pairs is None else {"compaction_tail_pairs": tail_pairs}
        agent = _make_agent(
            _FakeSampler(results=results),
            env_factory=lambda td: _FakeEnv(done_at_step=99, function_responses=[_FR]),
            max_steps=6,
            lowest_reward=-1.0,
            max_context_tokens=10_000,
            compaction_trigger_ratio=0.7,
            max_compactions=1,
            **kwargs,
        )

        scored = await agent.rollout(_make_task_data())

        step = [s for s in scored.trajectory.steps if _is_compaction(s)][0]
        assert step.extra["compaction_pairs_retained"] == (tail_pairs or 2)
        assert step.extra["compaction_pairs_elided"] == 4 - (tail_pairs or 2)

    @pytest.mark.asyncio
    async def test_compaction_pairs_elided_accounts_for_every_pair(self):
        """Four exchange pairs exist; every one is either retained or
        elided."""
        agent = _long_run_agent(_FakeSampler(results=_long_run_results()))

        scored = await agent.rollout(_make_task_data())

        step = [s for s in scored.trajectory.steps if _is_compaction(s)][0]
        pairs_total = 4
        assert step.extra["compaction_pairs_retained"] >= 1
        assert step.extra["compaction_pairs_elided"] >= 1
        assert (
            step.extra["compaction_pairs_elided"]
            == pairs_total - step.extra["compaction_pairs_retained"]
        )

    @pytest.mark.asyncio
    async def test_compaction_rebuilds_the_observation(self):
        """The prompt sampled AFTER compaction keeps the original task,
        carries the re-entry text, and has dropped the middle turns."""
        sampler = _RecordingSampler(_long_run_results())
        agent = _long_run_agent(sampler)

        await agent.rollout(_make_task_data())

        # 0-3 agent turns, 4 the summarization call, 5 the next agent turn.
        first, rebuilt = sampler.prompts[0], sampler.prompts[5]
        assert rebuilt.contents[0] == first.contents[0]
        text = _GenerateContentRequest.to_json(rebuilt)
        assert "Continue from here." in text
        assert "SUMMARY" in text
        # The summary is delimited, so its boundaries are stated rather
        # than inferred from the markdown.
        assert "--- BEGIN SUMMARY OF REMOVED TURNS ---" in text
        assert "--- END SUMMARY ---" in text
        assert "TURN-0" not in text
        assert "TURN-1" not in text
        assert "TURN-3" in text

    @pytest.mark.asyncio
    async def test_summarization_request_omits_the_retained_tail(self):
        """The tail is kept verbatim in the rebuilt context, so sending it
        to the summarizer too would duplicate those tokens."""
        sampler = _RecordingSampler(_long_run_results())
        agent = _long_run_agent(sampler)

        await agent.rollout(_make_task_data())

        # 0-3 agent turns, 4 the summarization call, 5 the next agent turn.
        summarization = _GenerateContentRequest.to_json(sampler.prompts[4])
        following = _GenerateContentRequest.to_json(sampler.prompts[5])
        # The dropped turns ARE summarized...
        assert "TURN-0" in summarization
        assert "TURN-1" in summarization
        # ...the retained tail is not, because it survives verbatim.
        assert "TURN-3" not in summarization
        assert "TURN-3" in following
        # The compaction instruction is the last turn of that request.
        assert "Reply with the summary only." in summarization

    @pytest.mark.asyncio
    async def test_summarization_request_keeps_system_instruction_and_tools(self):
        """The summarization call must otherwise look like a normal call:
        dropping the tools or the system prompt changes model behavior."""
        sampler = _RecordingSampler(_long_run_results())
        agent = _make_agent(
            sampler,
            env_factory=lambda td: _ConfiguredEnv(
                done_at_step=99, function_responses=[_FR]
            ),
            max_steps=8,
            lowest_reward=-1.0,
            max_context_tokens=1000,
            compaction_trigger_ratio=0.7,
            max_compactions=3,
        )

        await agent.rollout(_make_task_data())

        normal, summarization = sampler.prompts[0], sampler.prompts[4]
        assert summarization.system_instruction == normal.system_instruction
        assert summarization.tools == normal.tools
        assert summarization.generation_config == normal.generation_config
        assert summarization.model == normal.model
        # Sanity: the fixture really does set them.
        assert normal.tools

    @pytest.mark.asyncio
    async def test_rollout_token_total_charges_the_summarization_call(self):
        """The summarization call is generation the rollout paid for, so a
        cost-shaped reward must see it.

        Turn 0 (100) triggers compaction; the summary call generates 500;
        turn 1 (100) ends the episode. The env is handed 700, not 200.
        """
        env = _TokenChargeEnv(done_at_step=2, function_responses=[_FR])
        summary = make_sample_result(
            text="SUMMARY", finish_reason="STOP", generation_token_count=500
        )
        agent = _make_agent(
            _FakeSampler(results=[_big_turn(), summary, _big_turn()]),
            env_factory=lambda td: env,
            max_steps=6,
            lowest_reward=-1.0,
            max_context_tokens=1000,
            compaction_trigger_ratio=0.7,
            max_compactions=3,
        )

        await agent.rollout(_make_task_data())

        assert env.charged == [100, 700]

    @pytest.mark.asyncio
    async def test_rollout_token_total_excludes_recovered_samples(self):
        """A recovered non-STOP sample took no action, so it is not billed
        to the rollout -- only the actions and the compaction summaries are.
        """
        env = _TokenChargeEnv(done_at_step=1)
        truncated = make_sample_result(
            text="cut off", finish_reason="MAX_TOKENS", generation_token_count=9999
        )
        good = make_sample_result(
            text="hi", finish_reason="STOP", generation_token_count=100
        )
        agent = MultiStepAgent(
            sampler=_FakeSampler(results=[truncated, good]),
            env_factory=lambda td: env,
            max_steps=4,
            recoverable_finish_reasons=_RECOVER_ANY,
            **_default_kwargs(),
        )

        await agent.rollout(_make_task_data())

        assert env.charged == [100]

    @pytest.mark.asyncio
    async def test_summarization_failure_terminates_compaction_failed(self):
        empty = make_sample_result(candidate_id="")
        agent = self._agent([_big_turn(), empty])

        scored = await agent.rollout(_make_task_data())

        assert (
            atif_extras.get_rollout_status(scored.trajectory)
            == RolloutStatus.SAMPLING_FAILURE_COMPACTION_FAILED
        )
        assert scored.reward.reward == pytest.approx(-1.0)

    @pytest.mark.asyncio
    async def test_compaction_failure_step_keeps_the_finish_reason(self):
        """A truncated summary fails the compaction, but the recorded
        step must still say WHY -- that is how we learn to strip the tool
        declarations from the summarization call."""
        truncated = make_sample_result(text="", finish_reason="MAX_TOKENS")
        agent = self._agent([_big_turn(), truncated])

        scored = await agent.rollout(_make_task_data())

        assert (
            atif_extras.get_rollout_status(scored.trajectory)
            == RolloutStatus.SAMPLING_FAILURE_COMPACTION_FAILED
        )
        failed = scored.trajectory.steps[-1]
        assert failed.extra["finish_reason"] == "MAX_TOKENS"

    @pytest.mark.asyncio
    async def test_budget_exhaustion_terminates_prompt_too_long(self):
        agent = self._agent([_big_turn() for _ in range(4)], max_compactions=0)

        scored = await agent.rollout(_make_task_data())

        assert (
            atif_extras.get_rollout_status(scored.trajectory)
            == RolloutStatus.SAMPLING_FAILURE_PROMPT_TOO_LONG
        )
        assert scored.reward.reward == pytest.approx(-1.0)
        assert [s for s in scored.trajectory.steps if s.source == "system"] == []

    @pytest.mark.asyncio
    async def test_no_compaction_with_fewer_than_two_steps_left(self):
        """max_steps=2: at step_idx 1 only one step remains, so compacting
        would buy a summary the agent never gets to use."""
        agent = self._agent([_big_turn() for _ in range(4)], max_steps=2)

        scored = await agent.rollout(_make_task_data())

        assert [s for s in scored.trajectory.steps if s.source == "system"] == []
        assert (
            atif_extras.get_rollout_status(scored.trajectory)
            == RolloutStatus.SAMPLING_FAILURE_PROMPT_TOO_LONG
        )

    @pytest.mark.asyncio
    async def test_disabled_when_cap_is_none(self):
        """No cap configured -> compaction never runs, however big the counts."""
        huge = make_sample_result(
            text="turn",
            finish_reason="STOP",
            prompt_token_count=10**9,
            generation_token_count=10**9,
        )
        agent = _make_agent(
            _FakeSampler(results=[huge for _ in range(6)]),
            env_factory=lambda td: _FakeEnv(done_at_step=3),
            max_steps=6,
        )

        scored = await agent.rollout(_make_task_data())

        assert [s for s in scored.trajectory.steps if s.source == "system"] == []


class TestCompactionChargesAStep:
    """The summarization call really consumes one step of ``max_steps``.

    Same env (needs 3 agent turns to finish), same max_steps=3, same
    number of sampler calls: with compaction on, one of those three
    generations is the summary, so the agent only reaches 2 env steps and
    the rollout runs out of budget. That displacement IS the charge.
    """

    def _run(self, **compaction_kwargs):
        summary = make_sample_result(text="SUMMARY", finish_reason="STOP")
        sampler = _FakeSampler(results=[_big_turn(), summary, _big_turn(), _big_turn()])
        agent = _make_agent(
            sampler,
            env_factory=lambda td: _FakeEnv(
                done_at_step=3, reward=1.0, function_responses=[_FR]
            ),
            max_steps=3,
            lowest_reward=-1.0,
            **compaction_kwargs,
        )
        return sampler, agent

    @pytest.mark.asyncio
    async def test_compaction_displaces_an_agent_turn_from_the_budget(self):
        sampler, agent = self._run(
            max_context_tokens=1000,
            compaction_trigger_ratio=0.7,
            max_compactions=3,
        )

        scored = await agent.rollout(_make_task_data())

        steps = scored.trajectory.steps
        assert len([s for s in steps if _is_compaction(s)]) == 1
        # Three sampler calls were made, but only two were solving turns.
        # All three are agent-sourced (all three train); the compaction
        # one is excluded by its extras, not by its source.
        assert len(sampler._results_queue) == 1
        assert len([s for s in steps if s.source == "agent"]) == 3
        assert (
            len([s for s in steps if s.source == "agent" and not _is_compaction(s)])
            == 2
        )
        assert (
            atif_extras.get_rollout_status(scored.trajectory)
            == RolloutStatus.MAX_STEP_LIMIT
        )

    @pytest.mark.asyncio
    async def test_same_budget_without_compaction_finishes(self):
        """Baseline: the identical run with compaction off spends all
        three steps on agent turns and reaches done."""
        sampler, agent = self._run()

        scored = await agent.rollout(_make_task_data())

        steps = scored.trajectory.steps
        assert [s for s in steps if s.source == "system"] == []
        assert len(sampler._results_queue) == 1
        assert len([s for s in steps if s.source == "agent"]) == 3
        assert atif_extras.get_rollout_status(scored.trajectory) == RolloutStatus.OK


class _StepCountEnv(_FakeEnv):
    """Env that records the ``step_count`` it is handed on every step."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_counts: list[int] = []

    async def step(self, action, *, step_count: int = 0):
        self.step_counts.append(step_count)
        return await super().step(action, step_count=step_count)


class TestSampleRecoveryIsNotAStep:
    """A retry makes ONE step succeed; it is not a step of its own.

    The old loop charged one from ``max_steps`` per retry, which silently
    shrank the budget and the ``steps_remaining`` the env echoes for
    something the model neither did nor was told about.
    """

    @staticmethod
    def _truncated(tokens: int = 9999):
        return make_sample_result(
            text="cut off", finish_reason="MAX_TOKENS", generation_token_count=tokens
        )

    @staticmethod
    def _good(tokens: int = 100):
        return make_sample_result(
            text="hi", finish_reason="STOP", generation_token_count=tokens
        )

    @pytest.mark.asyncio
    async def test_retry_does_not_advance_the_step_count_the_env_sees(self):
        """Two retries before a good sample: the env still sees step 1."""
        env = _StepCountEnv(done_at_step=1)
        agent = MultiStepAgent(
            sampler=_FakeSampler(
                results=[self._truncated(), self._truncated(), self._good()]
            ),
            env_factory=lambda td: env,
            max_steps=4,
            recoverable_finish_reasons=_RECOVER_ANY,
            **_default_kwargs(),
        )

        await agent.rollout(_make_task_data())

        assert env.step_counts == [1]

    @pytest.mark.asyncio
    async def test_retries_do_not_consume_the_step_budget(self):
        """max_steps=2 with a retry before each turn still completes both
        turns -- under the old loop the retries ate the whole budget."""
        env = _StepCountEnv(done_at_step=2)
        agent = MultiStepAgent(
            sampler=_FakeSampler(
                results=[
                    self._truncated(),
                    self._good(),
                    self._truncated(),
                    self._good(),
                ]
            ),
            env_factory=lambda td: env,
            max_steps=2,
            recoverable_finish_reasons=_RECOVER_ANY,
            **_default_kwargs(),
        )

        scored = await agent.rollout(_make_task_data())

        assert env.step_counts == [1, 2]
        assert atif_extras.get_rollout_status(scored.trajectory) == RolloutStatus.OK

    @pytest.mark.asyncio
    async def test_recovery_step_is_system_sourced_so_it_never_trains(self):
        """The discarded candidate must not reach the trainer, which
        selects on ``source == "agent"``."""
        env = _FakeEnv(done_at_step=1)
        agent = MultiStepAgent(
            sampler=_FakeSampler(results=[self._truncated(), self._good()]),
            env_factory=lambda td: env,
            max_steps=4,
            recoverable_finish_reasons=_RECOVER_ANY,
            **_default_kwargs(),
        )

        scored = await agent.rollout(_make_task_data())

        recovery = [
            s
            for s in scored.trajectory.steps
            if atif_extras.get_finish_reason(s) == "MAX_TOKENS"
        ]
        assert len(recovery) == 1
        assert recovery[0].source == "system"
        # It IS retained for debugging, with its cost intact.
        assert atif_extras.get_generation_token_count(recovery[0]) == 9999

    @pytest.mark.asyncio
    async def test_recovery_step_with_thoughts_round_trips_through_disk(self):
        """Harbor rejects ``reasoning_content`` on a non-agent step, so a
        recovery step carrying thought parts must still reload via
        ``atif_codec.from_disk`` -- with the discarded thoughts kept."""
        from core import atif_codec

        thinking = make_sample_result(
            content={
                "role": "model",
                "parts": [
                    {"text": "pondering ", "thought": True},
                    {"text": "cut off"},
                ],
            },
            finish_reason="MAX_TOKENS",
            generation_token_count=9999,
        )
        agent = MultiStepAgent(
            sampler=_FakeSampler(results=[thinking, self._good()]),
            env_factory=lambda td: _FakeEnv(done_at_step=1),
            max_steps=4,
            recoverable_finish_reasons=_RECOVER_ANY,
            **_default_kwargs(),
        )

        scored = await agent.rollout(_make_task_data())
        assert scored is not None
        reloaded = atif_codec.from_disk(*atif_codec.to_disk(scored))

        (recovery,) = [s for s in reloaded.trajectory.steps if s.source == "system"]
        assert recovery.reasoning_content is None
        assert recovery.message == "pondering cut off"

    @pytest.mark.asyncio
    async def test_recovery_step_with_function_call_survives_and_round_trips(self):
        """Harbor rejects ``tool_calls`` (and every other agent-only field)
        on a non-agent step. A discarded sample that emitted a function_call
        must not crash the rollout; the call is kept readable in
        ``message`` and the step reloads from disk."""
        from core import atif_codec

        trunc = make_sample_result(
            content={
                "role": "model",
                "parts": [
                    {"text": "plan ", "thought": True},
                    {"function_call": {"name": "submit", "args": {"a": "x"}}},
                    {"text": "more"},
                ],
            },
            finish_reason="MAX_TOKENS",
            generation_token_count=10,
        )
        agent = MultiStepAgent(
            sampler=_FakeSampler(results=[trunc, self._good()]),
            env_factory=lambda td: _FakeEnv(done_at_step=1),
            max_steps=4,
            recoverable_finish_reasons=_RECOVER_ANY,
            **_default_kwargs(),
        )

        scored = await agent.rollout(_make_task_data())
        assert scored is not None
        reloaded = atif_codec.from_disk(*atif_codec.to_disk(scored))

        for traj in (scored.trajectory, reloaded.trajectory):
            (recovery,) = [s for s in traj.steps if s.source == "system"]
            for field in (
                "model_name",
                "reasoning_effort",
                "reasoning_content",
                "tool_calls",
                "metrics",
            ):
                assert getattr(recovery, field) is None, field
            assert recovery.message == (
                'plan more\n[discarded function_call] submit({"a": "x"})'
            )

    @pytest.mark.asyncio
    async def test_per_turn_budget_caps_a_streak(self):
        """max_sample_retries=2 gives 2 retries, then the rollout fails
        rather than looping until the wallclock timeout."""
        env = _FakeEnv(done_at_step=1)
        sampler = _FakeSampler(results=[self._truncated() for _ in range(10)])
        agent = MultiStepAgent(
            sampler=sampler,
            env_factory=lambda td: env,
            max_steps=8,
            recoverable_finish_reasons=_RECOVER_ANY,
            max_sample_retries=2,
            **_default_kwargs(),
        )

        scored = await agent.rollout(_make_task_data())

        # 2 recovered + 1 terminal failure = 3 sampler calls, not 8.
        assert len(sampler._results_queue) == 7
        assert (
            atif_extras.get_rollout_status(scored.trajectory)
            != RolloutStatus.MAX_STEP_LIMIT
        )

    @pytest.mark.asyncio
    async def test_per_turn_budget_resets_after_a_success(self):
        """The per-turn counter is consecutive: a good sample clears it,
        so turn 2 gets a fresh allowance."""
        env = _StepCountEnv(done_at_step=2)
        agent = MultiStepAgent(
            sampler=_FakeSampler(
                results=[
                    self._truncated(),
                    self._good(),  # turn 1 after 1 retry
                    self._truncated(),
                    self._good(),  # turn 2 after 1 more retry
                ]
            ),
            env_factory=lambda td: env,
            max_steps=4,
            recoverable_finish_reasons=_RECOVER_ANY,
            max_sample_retries=1,
            **_default_kwargs(),
        )

        scored = await agent.rollout(_make_task_data())

        assert env.step_counts == [1, 2]
        assert atif_extras.get_rollout_status(scored.trajectory) == RolloutStatus.OK

    @pytest.mark.asyncio
    async def test_rollout_budget_caps_the_whole_run(self):
        """Per-turn resets would otherwise allow max_steps * retries
        calls; the rollout-wide ceiling bounds the total."""
        env = _StepCountEnv(done_at_step=5)
        # Every turn: one truncation then a good sample.
        results = []
        for _ in range(5):
            results += [self._truncated(), self._good()]
        sampler = _FakeSampler(results=results)
        agent = MultiStepAgent(
            sampler=sampler,
            env_factory=lambda td: env,
            max_steps=6,
            recoverable_finish_reasons=_RECOVER_ANY,
            max_sample_retries=3,
            max_sample_retries_per_rollout=2,
            **_default_kwargs(),
        )

        scored = await agent.rollout(_make_task_data())

        # Turns 1-2 recover; turn 3's truncation exhausts the rollout
        # ceiling and terminates the run.
        assert env.step_counts == [1, 2]
        assert (
            atif_extras.get_rollout_status(scored.trajectory)
            == RolloutStatus.SAMPLING_FAILURE_UNHANDLED
        )

    @pytest.mark.asyncio
    async def test_unlisted_finish_reason_is_not_recovered(self):
        """Recovery is opt-in per reason: an unlisted one aborts."""
        env = _FakeEnv(done_at_step=1)
        agent = MultiStepAgent(
            sampler=_FakeSampler(results=[self._truncated(), self._good()]),
            env_factory=lambda td: env,
            max_steps=4,
            recoverable_finish_reasons=frozenset({"MALFORMED_FUNCTION_CALL"}),
            **_default_kwargs(),
        )

        scored = await agent.rollout(_make_task_data())

        assert (
            atif_extras.get_rollout_status(scored.trajectory)
            == RolloutStatus.SAMPLING_FAILURE_UNHANDLED
        )

    @pytest.mark.asyncio
    async def test_trainer_selection_takes_compaction_and_drops_recovery(self):
        """The contract this whole split exists for.

        ``VertexMsrlTrainer.train_step`` builds one training example per
        ``source == "agent"`` step, keyed by candidate_id. Applying that
        exact predicate must yield the solving turns AND the compaction
        summary (the model authored it and it entered the context), but
        NOT the discarded MAX_TOKENS draw.
        """
        env = _StepCountEnv(done_at_step=2)
        agent = MultiStepAgent(
            sampler=_FakeSampler(
                results=[
                    self._good(),  # turn 1
                    self._truncated(),  # discarded -> must NOT train
                    self._good(),  # turn 2
                ]
            ),
            env_factory=lambda td: env,
            max_steps=4,
            recoverable_finish_reasons=_RECOVER_ANY,
            **_default_kwargs(),
        )

        scored = await agent.rollout(_make_task_data())

        trained = [s for s in scored.trajectory.steps if s.source == "agent"]
        assert [atif_extras.get_finish_reason(s) for s in trained] == ["STOP", "STOP"]
        # The discarded draw is retained for debugging, just not trained.
        assert any(
            atif_extras.get_finish_reason(s) == "MAX_TOKENS" and s.source == "system"
            for s in scored.trajectory.steps
        )

    @pytest.mark.asyncio
    async def test_default_config_does_not_recover(self):
        """Empty ``recoverable_finish_reasons`` (the base default) keeps
        the abort-on-first-failure behaviour."""
        env = _FakeEnv(done_at_step=1)
        agent = MultiStepAgent(
            sampler=_FakeSampler(results=[self._truncated(), self._good()]),
            env_factory=lambda td: env,
            max_steps=4,
            **_default_kwargs(),
        )

        scored = await agent.rollout(_make_task_data())

        assert (
            atif_extras.get_rollout_status(scored.trajectory)
            == RolloutStatus.SAMPLING_FAILURE_UNHANDLED
        )
