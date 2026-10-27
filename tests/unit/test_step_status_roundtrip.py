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

"""W1.3 — ``StepStats.status`` survives to disk as ``step.extra["step_status"]``.

Before this landed, ``TrajectoryStep.from_wire`` copied every other
``StepStats`` field into ``extra`` and dropped ``status`` on the floor, so
the per-step status was computed and then lost at the ATIF boundary.
"""

from __future__ import annotations

import json

from core.atif_codec import from_disk, to_disk, to_wire
from core.atif_extras import get_step_status
from core.atif_trajectory import (
    Agent,
    Reward,
    ScoredTrajectory,
    Step,
    StepStats,
    Trajectory,
    TrajectoryStep,
)
from core.rollout_status import RolloutStatus


def _traj(*steps: Step) -> Trajectory:
    return Trajectory(
        schema_version="ATIF-v1.7",
        session_id="s-1",
        agent=Agent(name="t", version="0", model_name="m"),
        steps=[Step(step_id=1, source="user", message="do X"), *steps],
    )


def test_ok_status_is_omitted():
    step = TrajectoryStep.from_wire(
        step_id=2, generation="g", stats=StepStats(candidate_id="c1")
    )
    assert "step_status" not in (step.extra or {})
    # Absent means OK for readers.
    assert get_step_status(step) is RolloutStatus.OK


def test_non_ok_status_is_stamped_as_the_enum_value():
    step = TrajectoryStep.from_wire(
        step_id=2,
        generation="",
        stats=StepStats(status=RolloutStatus.SAMPLING_FAILURE_MAX_TOKENS),
    )
    # NOT "RolloutStatus.SAMPLING_FAILURE_MAX_TOKENS": RolloutStatus is a
    # (str, Enum), not a StrEnum, so str(status) would write the repr.
    assert step.extra["step_status"] == "sampling_failure/MAX_TOKENS"
    assert get_step_status(step) is RolloutStatus.SAMPLING_FAILURE_MAX_TOKENS


def test_round_trips_through_disk():
    step = TrajectoryStep.from_wire(
        step_id=2,
        generation="g",
        stats=StepStats(
            candidate_id="c1", status=RolloutStatus.SAMPLING_FAILURE_TIMEOUT
        ),
    )
    scored = ScoredTrajectory(trajectory=_traj(step), reward=Reward(reward=0.0))

    agent_json, reward_json = to_disk(scored)
    assert json.loads(agent_json)["steps"][1]["extra"]["step_status"] == (
        "sampling_failure/TIMEOUT"
    )

    restored = from_disk(agent_json, reward_json)
    assert (
        get_step_status(restored.trajectory.steps[1])
        is RolloutStatus.SAMPLING_FAILURE_TIMEOUT
    )


def test_passes_typed_extras_validation_on_the_wire_path():
    # CloudsideStepExtras has extra="forbid": an undeclared key would make
    # to_wire raise for every trajectory carrying a non-OK step.
    step = TrajectoryStep.from_wire(
        step_id=2,
        generation="g",
        stats=StepStats(status=RolloutStatus.SAMPLING_FAILURE_UNHANDLED),
    )
    payload = to_wire(ScoredTrajectory(trajectory=_traj(step), reward=Reward(reward=0.0)))
    assert payload is not None
