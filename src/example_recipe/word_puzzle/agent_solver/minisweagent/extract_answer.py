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

"""Extract the agent's submitted answer + per-task summary stats.

Collapses the extra dict returned by ``DefaultAgent.run`` and the saved
trajectory into a compact ``TaskResult`` for the summary table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TaskResult:
    """Compact summary of one puzzle attempt."""

    tier: str
    expected_answer: str
    submitted_answer: str  # raw submission from the agent (may be "")
    is_correct: bool
    exit_status: str  # "Submitted" | "LimitsExceeded" | ...
    n_model_calls: int
    n_bash_calls: int
    total_cost: float
    total_model_seconds: float
    trajectory_path: Path

    def to_row(self) -> tuple:
        """One row for the summary table."""
        ok = "OK " if self.is_correct else "  "
        return (
            ok,
            self.tier,
            _clip(self.expected_answer, 12),
            _clip(self.submitted_answer, 12),
            self.exit_status[:16],
            self.n_model_calls,
            self.n_bash_calls,
            f"{self.total_model_seconds:.1f}s",
            self.trajectory_path.name,
        )


TABLE_HEADER = (
    "OK ",
    "tier",
    "expected",
    "submitted",
    "exit_status",
    "n_llm",
    "n_bash",
    "t_llm",
    "trace",
)


def build_task_result(
    *,
    tier: str,
    expected_answer: str,
    run_return_value: dict,
    trajectory_path: Path,
) -> TaskResult:
    """Build a TaskResult from one ``agent.run(task=...)`` call.

    Args:
      tier: from the JSONL row's tags['tier'].
      expected_answer: ground truth from references.reference.
      run_return_value: the extra dict of the final "exit" message
        returned by ``agent.run()``.
      trajectory_path: where mini saved the trajectory.
    """
    submitted = (run_return_value.get("submission") or "").strip()
    exit_status = run_return_value.get("exit_status", "Unknown")

    data = json.loads(trajectory_path.read_text())
    stats = data.get("info", {}).get("model_stats", {})
    n_model_calls = int(stats.get("api_calls", 0))
    total_cost = float(stats.get("instance_cost", 0.0))
    # bash calls = tool observations = messages with a returncode in
    # their extra. Mini's ``format_toolcall_observation_messages``
    # emits ``role="tool"`` when the action has a ``tool_call_id``
    # (which every bash action from
    # ``SamplerChatCompletionModel._parse_tool_calls`` does), so the
    # live path only produces ``role="tool"`` today. The ``"user"``
    # branch is kept as a defensive fallback for replaying older
    # trajectory JSON that was written under a text-based observation
    # shape (no ``tool_call_id`` -> mini emits ``role="user"``); those
    # trajectories still exist on disk from pre-migration runs.
    messages = data.get("messages", [])
    n_bash_calls = sum(
        1
        for m in messages
        if m.get("role") in ("tool", "user") and "returncode" in m.get("extra", {})
    )
    total_model_seconds = float(
        data.get("info", {}).get("config", {}).get("total_sample_seconds", 0.0)
    )
    is_correct = submitted == expected_answer

    return TaskResult(
        tier=tier,
        expected_answer=expected_answer,
        submitted_answer=submitted,
        is_correct=is_correct,
        exit_status=exit_status,
        n_model_calls=n_model_calls,
        n_bash_calls=n_bash_calls,
        total_cost=total_cost,
        total_model_seconds=total_model_seconds,
        trajectory_path=trajectory_path,
    )


def _clip(s: str, n: int) -> str:
    """Truncate string s to n chars, adding ellipsis if needed."""
    if len(s) <= n:
        return s
    return s[: n - 1] + "\u2026"
