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

"""The checkpoint-eval campaign must fan trajectory hooks to BOTH sinks.

Regression guard for a silent observability bug: the campaign wired the
progress observer only as ``progress_observer`` and handed the agent the ATIF
writer alone. Experiment-level stats therefore rendered while the
per-trajectory records stayed empty -- no error, no log line, just missing
rollouts downstream. These tests pin the fan-out so that regresses
loudly instead of silently.
"""

from __future__ import annotations

import asyncio

from core.composite_callback import CompositeTrajectoryCallback
from tools.eval_all_checkpoints import build_agent_trajectory_callback


class _Writer:
    """Stands in for AtifDiskWriter (has gcs_upload_status, no dropped hook)."""

    def __init__(self) -> None:
        self.gcs_upload_status = "STATUS_SRC"
        self.produced: list[str] = []

    async def on_trajectory_produced(self, scored, task_data, traj=None) -> None:
        self.produced.append("writer")


class _Observer:
    """Stands in for FirestoreProgressSink (full TrajectoryCallback surface)."""

    def __init__(self) -> None:
        self.produced: list[str] = []
        self.dropped: list[str] = []
        self.gcs_source = None

    def set_gcs_status_source(self, src) -> None:
        self.gcs_source = src

    async def on_trajectory_produced(self, scored, task_data, traj=None) -> None:
        self.produced.append("observer")

    async def on_trajectory_dropped(self, task_data, traj=None, reason="") -> None:
        self.dropped.append(reason)


def test_both_sinks_receive_trajectory_produced() -> None:
    """The regression: the observer MUST get on_trajectory_produced."""
    writer, observer = _Writer(), _Observer()
    cb = build_agent_trajectory_callback(writer, observer)

    assert isinstance(cb, CompositeTrajectoryCallback)
    asyncio.run(cb.on_trajectory_produced(None, None, None))

    assert writer.produced == ["writer"]
    assert observer.produced == ["observer"], (
        "observer got no trajectory hook -> per-trajectory records would be empty"
    )


def test_hook_missing_on_one_sink_does_not_break_the_other() -> None:
    """AtifDiskWriter has no on_trajectory_dropped; the observer still gets it."""
    writer, observer = _Writer(), _Observer()
    cb = build_agent_trajectory_callback(writer, observer)

    asyncio.run(cb.on_trajectory_dropped(None, None, "zero_variance"))

    assert observer.dropped == ["zero_variance"]


def test_gcs_status_source_is_connected() -> None:
    """Upload status is piped to the observer so the UI can render it."""
    writer, observer = _Writer(), _Observer()
    build_agent_trajectory_callback(writer, observer)
    assert observer.gcs_source == "STATUS_SRC"


def test_degenerate_cases_return_the_lone_callback_or_none() -> None:
    writer = _Writer()
    observer = _Observer()

    assert build_agent_trajectory_callback(writer, None) is writer
    assert build_agent_trajectory_callback(None, observer) is observer
    assert build_agent_trajectory_callback(None, None) is None
