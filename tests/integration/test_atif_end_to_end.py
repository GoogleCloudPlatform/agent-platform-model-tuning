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

"""End-to-end integration test for the ATIF-native trajectory shape.

Builds a synthetic 3-turn rollout through the migrated code paths,
validates the resulting Trajectory against harbor's pydantic schema,
round-trips through the codec (to_disk / from_disk / to_wire / from_wire),
and asserts that reward-in-trajectory is rejected by typed extras validation.

This test is deliberately hermetic — no sampler, no env, no live
network. The unit tests cover component-level correctness;
this integration test pins the composition invariants that only
surface when a full trajectory flows through all the migrated layers
in one shot.
"""

from __future__ import annotations

import dataclasses
import json
import pytest
from core.atif_trajectory import Agent, FinalMetrics, Trajectory

from core import atif_extras
from core.atif_codec import from_disk, from_wire, to_disk, to_wire
from core.protocols import Action
from core.atif_trajectory import (
    ScoredTrajectory,
    Reward,
    StepStats,
    StepTimings,
    Step,
    TrajectoryStep,
)
from core.rollout_status import RolloutStatus
from core.session import build_session_id, sanitize_path_segment
from core.task_data import TaskData


@dataclasses.dataclass
class _FakeTaskData(TaskData):
    """Stand-in for a recipe task_data with a task_id attribute."""

    instance_id: str = "django__django-99999"
    issue_description: str = "The test task"


def _build_three_turn_scored() -> ScoredTrajectory:
    """Construct a realistic 3-agent-turn ATIF trajectory + Reward wrapper."""
    us = Step(step_id=1, source="user", message="Solve the puzzle")
    turns = []
    for step_id in (2, 3, 4):
        step = TrajectoryStep.from_wire(
            step_id=step_id,
            generation=f"generation-{step_id}",
            stats=StepStats(
                candidate_id=f"c-{step_id}",
                train_step_id=step_id,
                finish_reason="stop",
                operation_id=f"op-{step_id}",
                prompt_token_count=100 + step_id,
                generation_token_count=20 + step_id,
                status=RolloutStatus.OK,
            ),
            timings=StepTimings(sample_time_s=0.1 * step_id, env_time_s=0.05 * step_id),
        )
        step.advantage = 0.5
        turns.append(step)

    steps = [us, *turns]

    traj = Trajectory(
        schema_version="ATIF-v1.7",
        session_id=build_session_id(
            recipe="test-recipe",
            tuning_job_id="tj-integration",
            task_data=_FakeTaskData(task_id="django__django-99999"),
            rollout_idx=0,
        ),
        agent=Agent(name="test-agent", version="1.0.0", model_name="test-model"),
        steps=steps,
        final_metrics=FinalMetrics(
            total_steps=len(steps),
            extra={"total_sample_time_s": 0.6, "total_env_time_s": 0.3},
        ),
        extra={
            "rollout_status": "OK",
            "is_correct": True,
            "task_stats": {"score": 1.0},
            "task_step_stats": [{"n": 1}, {"n": 2}, {"n": 3}],
        },
    )
    return ScoredTrajectory(
        trajectory=traj, reward=Reward(reward=0.9, normalized_reward=0.4)
    )


def test_end_to_end_construction_and_reads():
    """Verify every migrated read path returns the expected value."""
    scored = _build_three_turn_scored()

    # Trajectory-level extras via typed accessors.
    assert atif_extras.get_is_correct(scored.trajectory) is True
    assert atif_extras.get_rollout_status(scored.trajectory) == RolloutStatus.OK
    assert scored.reward.reward == 0.9
    assert scored.reward.normalized_reward == 0.4

    # Session id was built from the recipe/tuning_job/task_id/rollout_idx
    # + process-start ISO stamp.
    assert scored.trajectory.session_id.startswith(
        "test-recipe-tj-integration-django__django-99999-0-"
    )

    # Step-level: filter agent-source and verify each turn's extras.
    agent_steps = [s for s in scored.trajectory.steps if s.source == "agent"]
    assert len(agent_steps) == 3
    for i, s in enumerate(agent_steps, start=2):
        assert atif_extras.get_candidate_id(s) == f"c-{i}"
        assert atif_extras.get_train_step_id(s) == i
        assert atif_extras.get_finish_reason(s) == "stop"
        assert atif_extras.get_operation_id(s) == f"op-{i}"
        assert atif_extras.get_advantage(s) == 0.5
        assert atif_extras.get_sample_time_s(s) == 0.1 * i
        assert atif_extras.get_env_time_s(s) == 0.05 * i
        # Token counts flow through (regression guard).
        assert s.extra["prompt_token_count"] == 100 + i
        assert s.extra["generation_token_count"] == 20 + i


def test_end_to_end_disk_roundtrip():
    """to_disk → from_disk preserves every field."""
    orig = _build_three_turn_scored()
    agent_json, reward_json = to_disk(orig)
    back = from_disk(agent_json, reward_json)

    assert back.trajectory.session_id == orig.trajectory.session_id
    assert back.reward.reward == orig.reward.reward
    assert back.reward.normalized_reward == orig.reward.normalized_reward
    assert len(back.trajectory.steps) == len(orig.trajectory.steps)
    assert atif_extras.get_is_correct(back.trajectory) is True
    # Advantage on each agent step survives.
    for s in back.trajectory.steps:
        if s.source == "agent":
            assert s.extra["advantage"] == 0.5


def test_end_to_end_wire_roundtrip():
    """to_wire → from_wire preserves the training-request-body shape."""
    orig = _build_three_turn_scored()
    body = to_wire(orig)
    # The wire body must be JSON-serializable (regression guard).
    payload = json.dumps(body)
    assert payload

    back = from_wire(body)
    assert back.reward.reward == orig.reward.reward
    assert len(back.trajectory.steps) == len(orig.trajectory.steps)
    # Typed extras survive the wire round-trip.
    step = back.trajectory.steps[1]
    assert step.extra["candidate_id"] == "c-2"
    assert step.extra["train_step_id"] == "2"
    assert step.extra["advantage"] == 0.5


def test_end_to_end_typed_extras_reject_reward_in_traj_extra():
    """Reward MUST NOT appear inside Trajectory.extra (ATIF invariant).

    Enforced by CloudsideTrajectoryExtras(extra="forbid") via the codec's
    _validate_typed_extras — pydantic rejects undeclared keys.
    """
    from pydantic import ValidationError

    us = Step(step_id=1, source="user", message="x")
    a = TrajectoryStep.from_wire(step_id=2, generation="g", stats=StepStats())
    bad_traj = Trajectory(
        schema_version="ATIF-v1.7",
        session_id="s",
        agent=Agent(name="t", version="0", model_name="m"),
        steps=[us, a],
        extra={"reward": 0.99},  # smuggled-in reward
    )
    bad_scored = ScoredTrajectory(trajectory=bad_traj, reward=Reward(reward=0.0))
    with pytest.raises(ValidationError, match="extra_forbidden"):
        to_disk(bad_scored)
    with pytest.raises(ValidationError, match="extra_forbidden"):
        to_wire(bad_scored)


def test_end_to_end_typed_extras_reject_reward_in_step_extra():
    """Reward at the step level is also rejected by typed extras validation."""
    from pydantic import ValidationError

    us = Step(step_id=1, source="user", message="x")
    a = TrajectoryStep.from_wire(step_id=2, generation="g", stats=StepStats())
    a.extra["reward"] = 0.5  # per-step reward is also verboten
    traj = Trajectory(
        schema_version="ATIF-v1.7",
        session_id="s",
        agent=Agent(name="t", version="0", model_name="m"),
        steps=[us, a],
    )
    scored = ScoredTrajectory(trajectory=traj, reward=Reward(reward=0.0))
    with pytest.raises(ValidationError, match="extra_forbidden"):
        to_disk(scored)


# --------------------------------------------------------------------------
# AtifDiskWriter end-to-end via a real group_rollout (spec §11.3)
# --------------------------------------------------------------------------


@dataclasses.dataclass
class _AtifTaskData(TaskData):
    pass


class _AtifFakeSampler:
    """Sampler returning N canned SampleResults (single-step happy path)."""

    model_name = "test-model"

    async def sample_n(
        self,
        prompt,
        num_samples: int = 1,
        **kwargs,
    ):
        from tests.unit._sample_result_factories import make_sample_result

        return [
            make_sample_result(candidate_id=f"c{i}", text=f"answer-{i}")
            for i in range(num_samples)
        ]


class _AtifFakeEnv:
    def get_prompt(self):
        from google.cloud.aiplatform_v1.types import Content, GenerateContentRequest

        return GenerateContentRequest(
            contents=[Content(role="user", parts=[{"text": "solve it"}])]
        )

    async def step(self, action: Action, *, step_count: int = 0):
        from core.protocols import EnvStepResult

        return EnvStepResult(
            function_responses=[], reward=0.75, done=True, is_correct=True
        )


_E2E_AGENT_KWARGS = dict(
    recipe_name="word_puzzle",
    tuning_job_id="9999",
    extract_prompt_text=lambda td: "solve it",
)


@pytest.mark.asyncio
async def test_group_rollout_writes_atif_tree_to_disk(tmp_path):
    """A real SingleStepAgent.group_rollout with AtifDiskWriter hooked in
    produces the 3-file trial dir per rollout, with valid round-tripping
    content (spec §11.3)."""
    from core.single_step_agent import SingleStepAgent
    from utils.atif_disk_writer import build_atif_writer

    writer = build_atif_writer(
        atif_output="local",
        log_dir=str(tmp_path),
        recipe="word_puzzle",
        tuning_job_id="9999",
    )
    assert writer is not None

    agent = SingleStepAgent(
        _AtifFakeSampler(),
        lambda _td: _AtifFakeEnv(),
        **_E2E_AGENT_KWARGS,
        trajectory_callback=writer,
    )

    await writer.on_job_started({"recipe": "word_puzzle", "lr": 0.001})
    group, _ = await agent.group_rollout(
        task_data=_AtifTaskData(task_id="task-e2e"), num_samples=3
    )
    await writer.on_job_completed({"total_rollouts": 3})

    assert len(group.scored_trajectories) == 3

    job_dir = tmp_path / "atif" / "word_puzzle" / "9999"
    # Job-level files.
    assert (job_dir / "config.json").is_file()
    assert (job_dir / "result.json").is_file()
    assert json.loads((job_dir / "config.json").read_text())["lr"] == 0.001

    # One trial dir per rollout, each with the 3 contract files. The trial
    # dir name IS the trajectory's session_id (see _trial_name / #284), so
    # derive it from each scored trajectory rather than hardcoding a layout.
    for idx, scored in enumerate(group.scored_trajectories):
        trial = job_dir / sanitize_path_segment(scored.trajectory.session_id)
        agent_json = (trial / "agent" / "trajectory.json").read_text()
        reward_json = (trial / "verifier" / "reward.json").read_text()
        result = json.loads((trial / "result.json").read_text())

        # Content round-trips through the codec.
        back = from_disk(agent_json, reward_json)
        assert back.reward.reward == 0.75
        assert back.trajectory.session_id  # a session id was written
        assert result["rollout_idx"] == idx
        assert result["reward"] == 0.75
        assert result["status"] == "OK"

        # Contract: no leaked reward key inside the trajectory JSON.
        assert '"reward"' not in agent_json
        # Contract: verifier/reward.json is an object carrying reward.
        assert json.loads(reward_json)["reward"] == 0.75
