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

"""Tests for src/core/atif_codec.py to_disk/from_disk/to_wire/from_wire."""

from __future__ import annotations

import math as _math
import json

from core.atif_trajectory import Agent, Trajectory

from core.atif_codec import from_disk, from_wire, to_disk, to_wire
from core.atif_trajectory import (
    ScoredTrajectory,
    Reward,
    StepStats,
    Step,
    TrajectoryStep,
)


def _make_scored() -> ScoredTrajectory:
    us = Step(step_id=1, source="user", message="do X")
    a = TrajectoryStep.from_wire(
        step_id=2,
        generation="g",
        stats=StepStats(
            candidate_id="c1",
            train_step_id=7,
            finish_reason="stop",
            prompt_token_count=100,
            generation_token_count=25,
        ),
    )
    a.advantage = 0.5
    traj = Trajectory(
        schema_version="ATIF-v1.7",
        session_id="s-1",
        agent=Agent(name="t", version="0", model_name="m"),
        steps=[us, a],
    )
    return ScoredTrajectory(
        trajectory=traj, reward=Reward(reward=0.7, normalized_reward=0.3)
    )


def test_to_disk_returns_two_json_bodies():
    agent_json, reward_json = to_disk(_make_scored())
    assert "session_id" in agent_json
    assert "reward" in reward_json


# --------------------------------------------------------------------------
# Recipe-owned typed stats in Trajectory.extra (task_stats / _task_step_stats)
# --------------------------------------------------------------------------


import dataclasses  # noqa: E402
import json as _json  # noqa: E402


@dataclasses.dataclass
class _FakeStepStats:
    """Stand-in for a recipe's typed per-step stats dataclass attached to
    Trajectory.extra by real agents."""

    f05_files: float = 0.0
    shell_executions: int = 0


def _scored_with_typed_stats() -> ScoredTrajectory:
    us = Step(step_id=1, source="user", message="do X")
    a = TrajectoryStep.from_wire(
        step_id=2, generation="g", stats=StepStats(candidate_id="c1")
    )
    traj = Trajectory(
        schema_version="ATIF-v1.7",
        session_id="s-stats",
        agent=Agent(name="t", version="0", model_name="m"),
        steps=[us, a],
        # Exactly what a recipe agent's _build_scored / SingleStepAgent writes:
        # a typed dataclass under task_stats + the underscore per-step list.
        extra={
            "task_stats": _FakeStepStats(f05_files=0.9, shell_executions=3),
            "_task_step_stats": [_FakeStepStats(f05_files=0.9, shell_executions=3)],
        },
    )
    return ScoredTrajectory(trajectory=traj, reward=Reward(reward=1.0))


def test_to_disk_accepts_dataclass_task_stats():
    """Regression: real agents put a typed dataclass in extra['task_stats'];
    to_disk must not reject it (was a ValidationError -> swallowed -> no
    trial dir written)."""
    agent_json, _ = to_disk(_scored_with_typed_stats())
    obj = _json.loads(agent_json)
    # Serialized as a plain dict, schema-conformant.
    assert obj["extra"]["task_stats"]["f05_files"] == 0.9
    assert obj["extra"]["task_stats"]["shell_executions"] == 3


def test_to_disk_renames_underscore_task_step_stats():
    """The in-memory '_task_step_stats' key becomes the schema's real
    'task_step_stats' on disk (dicts, no underscore)."""
    agent_json, _ = to_disk(_scored_with_typed_stats())
    obj = _json.loads(agent_json)
    assert "_task_step_stats" not in obj["extra"]
    assert obj["extra"]["task_step_stats"][0]["f05_files"] == 0.9


def test_disk_roundtrip_with_typed_stats():
    """from_disk of a typed-stats trajectory validates + round-trips."""
    orig = _scored_with_typed_stats()
    agent_json, reward_json = to_disk(orig)
    back = from_disk(agent_json, reward_json)
    assert back.trajectory.extra["task_stats"]["f05_files"] == 0.9
    assert back.trajectory.extra["task_step_stats"][0]["shell_executions"] == 3
    assert back.reward.reward == 1.0


def test_to_disk_handles_none_per_step_task_stat():
    """Regression (live sampling smoke): a sampling-error step appends
    None to _task_step_stats (no env call -> no per-step stats). The
    schema's task_step_stats is list[dict], so None must normalize to {}
    (position-preserving) rather than crashing to_disk -> swallowed -> no
    trial dir written."""
    us = Step(step_id=1, source="user", message="do X")
    a = TrajectoryStep.from_wire(
        step_id=2, generation="g", stats=StepStats(candidate_id="c1")
    )
    traj = Trajectory(
        schema_version="ATIF-v1.7",
        session_id="s-none-stats",
        agent=Agent(name="t", version="0", model_name="m"),
        steps=[us, a],
        extra={
            # None for the sampling-error step, then a real dataclass step.
            "_task_step_stats": [
                None,
                _FakeStepStats(f05_files=0.5, shell_executions=2),
            ],
        },
    )
    scored = ScoredTrajectory(trajectory=traj, reward=Reward(reward=0.5))

    agent_json, _ = to_disk(scored)  # must NOT raise
    obj = _json.loads(agent_json)
    stats = obj["extra"]["task_step_stats"]
    assert len(stats) == 2  # positions preserved (not filtered)
    assert stats[0] == {}  # None -> {} (schema-valid empty stats)
    assert stats[1]["f05_files"] == 0.5
    assert stats[1]["shell_executions"] == 2


def test_disk_roundtrip_with_none_per_step_task_stat():
    """from_disk of a trajectory whose _task_step_stats had a None slot
    validates + reads the None slot back as {} (position-preserving)."""
    us = Step(step_id=1, source="user", message="do X")
    a = TrajectoryStep.from_wire(
        step_id=2, generation="g", stats=StepStats(candidate_id="c1")
    )
    traj = Trajectory(
        schema_version="ATIF-v1.7",
        session_id="s-none-stats-rt",
        agent=Agent(name="t", version="0", model_name="m"),
        steps=[us, a],
        extra={
            "_task_step_stats": [
                None,
                _FakeStepStats(f05_files=0.5, shell_executions=2),
            ],
        },
    )
    orig = ScoredTrajectory(trajectory=traj, reward=Reward(reward=0.5))
    agent_json, reward_json = to_disk(orig)
    back = from_disk(agent_json, reward_json)
    assert back.trajectory.extra["task_step_stats"][0] == {}
    assert back.trajectory.extra["task_step_stats"][1]["shell_executions"] == 2


def test_to_wire_accepts_dataclass_task_stats():
    """to_wire must apply the same normalization as to_disk."""
    body = to_wire(_scored_with_typed_stats())
    payload = _json.loads(_json.dumps(body))  # must be JSON-serializable
    extra = payload["trajectory"]["extra"]
    assert extra["task_stats"]["f05_files"] == 0.9
    assert "_task_step_stats" not in extra
    assert extra["task_step_stats"][0]["f05_files"] == 0.9


def test_disk_roundtrip_identity():
    orig = _make_scored()
    agent_json, reward_json = to_disk(orig)
    back = from_disk(agent_json, reward_json)
    assert back.reward.reward == orig.reward.reward
    assert back.reward.normalized_reward == orig.reward.normalized_reward
    assert back.trajectory.session_id == orig.trajectory.session_id
    assert len(back.trajectory.steps) == len(orig.trajectory.steps)
    assert back.trajectory.steps[1].extra["advantage"] == 0.5
    assert back.trajectory.steps[1].extra["prompt_token_count"] == 100
    assert back.trajectory.steps[1].extra["generation_token_count"] == 25


def test_wire_roundtrip_identity():
    orig = _make_scored()
    body = to_wire(orig)
    back = from_wire(body)
    assert back.reward.reward == orig.reward.reward
    assert back.trajectory.steps[1].extra["candidate_id"] == "c1"
    assert back.trajectory.steps[1].extra["train_step_id"] == "7"
    assert back.trajectory.steps[1].extra["advantage"] == 0.5
    assert back.trajectory.steps[1].extra["prompt_token_count"] == 100
    assert back.trajectory.steps[1].extra["generation_token_count"] == 25


# --------------------------------------------------------------------------
# Context-compaction extras on a source="system" step
# --------------------------------------------------------------------------


_COMPACTION_EXTRA = {
    "compaction_pre_tokens": 116_962,
    "compaction_post_tokens": 19_500,
    "compaction_pairs_elided": 19,
    "compaction_pairs_retained": 2,
}


def _scored_with_compaction_step() -> ScoredTrajectory:
    """_make_scored() plus the source="system" step a compaction emits."""
    scored = _make_scored()
    scored.trajectory.steps.append(
        Step(
            step_id=3,
            source="system",
            message="[conversation compacted]",
            extra=dict(_COMPACTION_EXTRA),
        )
    )
    return scored


def test_compaction_step_survives_wire_and_disk_roundtrip():
    """The risk this schema change mitigates: the codec validates EVERY
    step's .extra against one extra="forbid" model, source-blind
    (atif_codec._validate_typed_extras), so an undeclared compaction key
    would raise at serialization time and the artifact would never land.
    Both transports must accept the step and preserve all four values.
    """
    orig = _scored_with_compaction_step()

    wire_back = from_wire(to_wire(orig))
    agent_json, reward_json = to_disk(orig)
    disk_back = from_disk(agent_json, reward_json)

    for back in (wire_back, disk_back):
        step = back.trajectory.steps[-1]
        assert step.source == "system"
        for key, value in _COMPACTION_EXTRA.items():
            assert step.extra[key] == value


# --------------------------------------------------------------------------
# NaN reward serialization (timeout trajectories carry reward=NaN)
# --------------------------------------------------------------------------


def test_to_disk_reward_json_is_strict_valid_on_nan_reward():
    """Timeout trajectories carry reward=NaN; reward.json must be RFC 8259
    valid (no bare NaN token) so orjson / BigQuery can parse it."""
    g = _make_scored()
    g = ScoredTrajectory(
        trajectory=g.trajectory,
        reward=Reward(reward=float("nan"), normalized_reward=None),
    )
    _agent_json, reward_json = to_disk(g)
    # Bare NaN is the invalid-JSON smoking gun.
    assert "NaN" not in reward_json
    parsed = _json.loads(reward_json)
    assert parsed["reward"] is None


def test_from_disk_restores_nan_reward_written_as_null():
    """The inverse: a null reward on disk reads back as NaN, not None.

    ``to_disk`` writes NaN as null (strict JSON). Offline consumers then
    ran ``reward > 0`` on a None and crashed with TypeError, while the
    same aggregator inside the pipeline saw NaN and coped. The round trip
    has to preserve the NaN for the two paths to agree.
    """
    g = _make_scored()
    g = ScoredTrajectory(
        trajectory=g.trajectory,
        reward=Reward(reward=float("nan"), normalized_reward=None),
    )
    agent_json, reward_json = to_disk(g)
    back = from_disk(agent_json, reward_json)
    assert isinstance(back.reward.reward, float)
    assert _math.isnan(back.reward.reward)
    # The pipeline's two NaN-tolerant idioms must not raise.
    assert (back.reward.reward > 0) is False


# -- Pain 9: raw_reward / details are decoupled from the training signal ----
#
# Before this, ``Reward`` was ``(reward, normalized_reward)`` and
# ``final_reward()`` returned a bare float, so a grader that measured 0.85
# and a fault policy that floored it to 0.0 were indistinguishable on disk,
# and the grader's named dimensions were dropped entirely.


def _scored_with(**reward_kwargs):
    base = _make_scored()
    return ScoredTrajectory(
        trajectory=base.trajectory, reward=Reward(**reward_kwargs)
    )


def test_raw_reward_and_details_reach_verifier_reward_json():
    _, reward_json = to_disk(
        _scored_with(
            reward=0.0,
            raw_reward=0.85,
            details={"partial_credit": 0.85, "completed": 17},
        )
    )
    body = json.loads(reward_json)
    # The floored training signal AND the true benchmark score, side by side.
    assert body["reward"] == 0.0
    assert body["raw_reward"] == 0.85
    assert body["details"] == {"partial_credit": 0.85, "completed": 17}


def test_a_reward_with_neither_omits_the_keys_entirely():
    # Every trajectory written before this change has no raw_reward and no
    # details. The keys must be OMITTED, not written as null, or every
    # existing file's bytes move and the viewer renders an empty "raw
    # reward" row on a million old trials.
    _, reward_json = to_disk(_scored_with(reward=1.0, normalized_reward=0.5))
    assert json.loads(reward_json) == {"reward": 1.0, "normalized_reward": 0.5}


def test_round_trip_through_disk_preserves_all_four_quantities():
    agent_json, reward_json = to_disk(
        _scored_with(
            reward=0.0, normalized_reward=-1.2, raw_reward=0.85, details={"d": 1}
        )
    )
    back = from_disk(agent_json, reward_json)
    assert back.reward.reward == 0.0
    assert back.reward.normalized_reward == -1.2
    assert back.reward.raw_reward == 0.85
    assert back.reward.details == {"d": 1}


def test_round_trip_through_the_wire_preserves_them_too():
    back = from_wire(to_wire(_scored_with(reward=0.0, raw_reward=0.85, details={"d": 1})))
    assert back.reward.raw_reward == 0.85
    assert back.reward.details == {"d": 1}


def test_a_grader_dataclass_in_details_does_not_kill_the_whole_write():
    """`Reward.details` is typed `dict[str, Any]` and recipes put their own
    stats dataclass in it. Once details reached verifier/reward.json that
    became a TypeError inside json.dumps -- which the callback fan-out
    swallows as a WARNING, so the symptom was not a crash but NO ATIF
    OUTPUT AT ALL, while the eval still reported a clean score. Found on
    the first VM smoke of the migrated agent, not by any unit test.
    """
    import dataclasses as _dc

    @_dc.dataclass
    class RecipeStats:
        tool_errors: int = 3
        completed: float = 1.0

    _, reward_json = to_disk(
        _scored_with(reward=1.0, raw_reward=1.0, details={"stats": RecipeStats()})
    )
    assert json.loads(reward_json)["details"]["stats"] == {
        "tool_errors": 3,
        "completed": 1.0,
    }


def test_an_unserializable_detail_is_stringified_rather_than_fatal():
    _, reward_json = to_disk(
        _scored_with(reward=1.0, raw_reward=1.0, details={"weird": object()})
    )
    assert "object at" in json.loads(reward_json)["details"]["weird"]


def test_agent_extra_generation_params_roundtrip_through_disk():
    scored = _make_scored()
    scored.trajectory.agent.extra = {"thinking_level": "HIGH", "include_thoughts": True}
    restored = from_disk(*to_disk(scored))
    assert restored.trajectory.agent.extra == {
        "thinking_level": "HIGH",
        "include_thoughts": True,
    }


def test_agent_extra_rejects_undeclared_key():
    import pydantic
    import pytest

    scored = _make_scored()
    scored.trajectory.agent.extra = {"temprature": 0.5}
    with pytest.raises(pydantic.ValidationError):
        to_disk(scored)
