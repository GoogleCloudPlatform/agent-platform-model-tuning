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

"""Smoke test proving the Harbor ATIF schema dependency is wired correctly.

This test does not exercise any of our own code — it only proves that after
`uv add harbor`, the `harbor.models.trajectories` module imports cleanly on
Python 3.12 and that its pydantic validators fire as advertised. If this
file starts failing, either the `harbor` dep is not installed, the ATIF
schema version shipped in the wheel changed shape, or Python was
downgraded below 3.12.

Reference — ATIF (Agent Trajectory Interchange Format) RFC:
https://github.com/harbor-framework/harbor/blob/main/rfcs/0001-trajectory-format.md

The actual replacement of our home-grown Trajectory dataclass with
`harbor.models.trajectories.Trajectory` happens in a follow-up PR; this
file is intentionally the ONLY consumer of `harbor` in the tree until
that lands.
"""

import pydantic
import pytest

from core.atif_trajectory import Agent, Step, Trajectory


def test_import_and_construct_minimal_valid_trajectory():
    """A 1-user + 1-agent trajectory is the minimum valid ATIF payload."""
    traj = Trajectory(
        agent=Agent(name="test-agent", version="0.1"),
        steps=[
            Step(step_id=1, source="user", message="hi"),
            Step(step_id=2, source="agent", message="ok", model_name="test-model"),
        ],
    )
    assert traj.schema_version.startswith("ATIF-v"), (
        f"expected schema_version 'ATIF-v*'; got {traj.schema_version!r}"
    )
    assert len(traj.steps) == 2
    assert traj.steps[0].source == "user"
    assert traj.steps[1].model_name == "test-model"


def test_construction_rejects_non_sequential_step_ids():
    """Harbor's Trajectory validator requires step_ids to be sequential from 1."""
    with pytest.raises(pydantic.ValidationError):
        Trajectory(
            agent=Agent(name="test-agent", version="0.1"),
            steps=[
                Step(step_id=1, source="user", message="hi"),
                # Jumps from 1 -> 5; should be rejected.
                Step(step_id=5, source="agent", message="ok", model_name="test-model"),
            ],
        )


def test_step_agent_only_field_rejected_on_user_step():
    """`model_name` on a user Step is nonsensical; Harbor's Step validator rejects it."""
    with pytest.raises(pydantic.ValidationError):
        Step(
            step_id=1,
            source="user",
            message="hi",
            # model_name is an agent-only field per the ATIF spec.
            model_name="should-not-be-allowed-on-user-step",
        )
