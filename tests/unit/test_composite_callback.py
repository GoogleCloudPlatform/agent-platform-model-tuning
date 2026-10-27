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

"""Tests for CompositeTrajectoryCallback."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.atif_trajectory import Trajectory

import pytest
from core.atif_trajectory import Agent as AtifAgent
from core.atif_trajectory import Trajectory

from core.atif_trajectory import Step
from core.composite_callback import CompositeTrajectoryCallback


def _new_traj(
    *,
    session_id: str = "sid",
    dispatch_index: int | None = None,
    phase: str = "train",
    rollout_idx: int = 0,
    task_id: str = "task",
) -> Trajectory:
    """Build a minimal live Trajectory for callback tests."""
    return Trajectory(
        schema_version="ATIF-v1.7",
        session_id=session_id,
        agent=AtifAgent(name="t", version="1", model_name="m"),
        steps=[Step(step_id=1, source="user", message="hi")],
        extra={
            "dispatch_index": dispatch_index,
            "phase": phase,
            "rollout_idx": rollout_idx,
            "task_id": task_id,
        },
    )


@dataclass
class _RecordingCallback:
    """Minimal TrajectoryCallback that records calls."""

    calls: list[tuple[str, tuple[Any, ...]]] = field(default_factory=list)
    _capture_ctx: Any = None

    async def on_job_started(self, config: Any) -> None:
        self.calls.append(("on_job_started", (config,)))

    async def on_trajectory_started(
        self,
        task_data: Any,
        traj: Any = None,
    ) -> None:
        e = (traj.extra or {}) if traj is not None else {}
        self.calls.append(
            ("on_trajectory_started", (task_data, e.get("rollout_idx"), e.get("phase")))
        )

    async def on_trajectory_step_completed(
        self,
        task_data: Any,
        traj: Any = None,
    ) -> None:
        e = (traj.extra or {}) if traj is not None else {}
        step = max(0, len(traj.steps) - 1) if traj is not None else None
        self.calls.append(
            (
                "on_trajectory_step_completed",
                (task_data, e.get("rollout_idx"), step, e.get("phase")),
            )
        )

    async def on_trajectory_produced(
        self,
        scored: Any,
        task_data: Any,
        traj: Any = None,
    ) -> None:
        e = (traj.extra or {}) if traj is not None else {}
        self.calls.append(
            (
                "on_trajectory_produced",
                (scored, task_data, e.get("rollout_idx"), traj),
            )
        )

    async def on_trajectory_dropped(
        self,
        task_data: Any,
        traj: Any = None,
        reason: str = "",
    ) -> None:
        e = (traj.extra or {}) if traj is not None else {}
        self.calls.append(
            ("on_trajectory_dropped", (task_data, e.get("rollout_idx"), reason))
        )

    async def on_batch_assembled(
        self, step: int, trajectory_ids: list[str], mean_reward: Any = None
    ) -> None:
        self.calls.append(("on_batch_assembled", (step, trajectory_ids, mean_reward)))

    async def on_job_completed(self, aggregate_stats: dict[str, Any]) -> None:
        self.calls.append(("on_job_completed", (aggregate_stats,)))

    async def capture_ctx_for(
        self, *, task_id: str, rollout_idx: int, session_id: str
    ) -> Any:
        self.calls.append(("capture_ctx_for", (task_id, rollout_idx, session_id)))
        return self._capture_ctx


class TestCompositeTrajectoryCallback:
    """Fan-out and error-isolation tests."""

    @pytest.mark.asyncio
    async def test_fan_out_to_all_callbacks(self):
        cb1, cb2 = _RecordingCallback(), _RecordingCallback()
        composite = CompositeTrajectoryCallback([cb1, cb2])
        await composite.on_job_started({"key": "val"})
        assert len(cb1.calls) == 1
        assert len(cb2.calls) == 1
        assert cb1.calls[0] == ("on_job_started", ({"key": "val"},))

    @pytest.mark.asyncio
    async def test_trajectory_produced_fans_out(self):
        cb1, cb2 = _RecordingCallback(), _RecordingCallback()
        composite = CompositeTrajectoryCallback([cb1, cb2])
        traj = _new_traj(rollout_idx=3)
        await composite.on_trajectory_produced("graded_obj", "task_data_obj", traj)
        assert cb1.calls[0][0] == "on_trajectory_produced"
        assert cb2.calls[0][0] == "on_trajectory_produced"
        # rollout_idx (read off traj.extra) forwarded to each callback
        assert cb1.calls[0][1][2] == 3
        assert cb2.calls[0][1][2] == 3

    @pytest.mark.asyncio
    async def test_error_isolation_continues_to_next(self):
        """If cb1 raises, cb2 still gets called."""
        cb1 = _RecordingCallback()
        cb2 = _RecordingCallback()

        async def _boom(*args):
            raise RuntimeError("boom")

        cb1.on_trajectory_produced = _boom
        composite = CompositeTrajectoryCallback([cb1, cb2])
        # Should NOT raise
        await composite.on_trajectory_produced("g", "t", _new_traj())
        assert len(cb2.calls) == 1  # cb2 still called

    @pytest.mark.asyncio
    async def test_capture_ctx_returns_first_non_none(self):
        cb1 = _RecordingCallback(_capture_ctx=None)
        cb2 = _RecordingCallback(_capture_ctx="ctx_from_cb2")
        composite = CompositeTrajectoryCallback([cb1, cb2])
        result = await composite.capture_ctx_for(
            task_id="t1", rollout_idx=0, session_id="s1"
        )
        assert result == "ctx_from_cb2"

    @pytest.mark.asyncio
    async def test_capture_ctx_returns_none_when_all_none(self):
        cb1 = _RecordingCallback(_capture_ctx=None)
        composite = CompositeTrajectoryCallback([cb1])
        result = await composite.capture_ctx_for(
            task_id="t1", rollout_idx=0, session_id="s1"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_job_completed_fans_out(self):
        cb1, cb2 = _RecordingCallback(), _RecordingCallback()
        composite = CompositeTrajectoryCallback([cb1, cb2])
        await composite.on_job_completed({"recipe": "wp"})
        assert cb1.calls[0] == ("on_job_completed", ({"recipe": "wp"},))
        assert cb2.calls[0] == ("on_job_completed", ({"recipe": "wp"},))

    @pytest.mark.asyncio
    async def test_trajectory_started_fans_out(self):
        cb1, cb2 = _RecordingCallback(), _RecordingCallback()
        composite = CompositeTrajectoryCallback([cb1, cb2])
        traj = _new_traj(rollout_idx=0, phase="eval")
        await composite.on_trajectory_started("task", traj)
        assert cb1.calls[0] == ("on_trajectory_started", ("task", 0, "eval"))
        assert cb2.calls[0] == ("on_trajectory_started", ("task", 0, "eval"))

    @pytest.mark.asyncio
    async def test_trajectory_step_completed_fans_out(self):
        cb1, cb2 = _RecordingCallback(), _RecordingCallback()
        composite = CompositeTrajectoryCallback([cb1, cb2])
        # Fake 3 agent steps completed by appending to the shell (leading
        # user step already occupies index 0, so 3 completed => 4 total).
        traj = _new_traj(rollout_idx=0)
        for _ in range(3):
            traj.steps.append(traj.steps[0])  # dummy step; only count matters
        await composite.on_trajectory_step_completed("task", traj)
        assert cb1.calls[0][0] == "on_trajectory_step_completed"
        assert cb2.calls[0][0] == "on_trajectory_step_completed"
        assert cb1.calls[0][1][2] == 3  # step count derived from len(traj.steps) - 1
