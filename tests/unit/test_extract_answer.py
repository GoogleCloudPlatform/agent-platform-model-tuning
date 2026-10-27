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

"""Tests for ``extract_answer.build_task_result``.

Motivation: the ``n_bash_calls`` counter was filtered by
``role == "user"``, but bash observations produced by the toolcall-based
path (``SamplerChatCompletionModel`` +
mini's ``format_toolcall_observation_messages``) set ``role="tool"``
whenever the action carries a ``tool_call_id`` — which every bash
action from ``_parse_tool_calls`` does. So the filter never matched
and every task reported ``n_bash_calls=0`` on the summary table.

The stale ``role="user"`` filter was correct for the OLD text-based
model (``LitellmTextbasedModel``, referenced in the pre-fix comment)
but wrong for the toolcall path this PR uses. Fix: match on
``role in {"tool", "user"}`` so both message shapes are counted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_trajectory(
    path: Path,
    *,
    observation_role: str,
    n_observations: int,
    api_calls: int = 3,
    instance_cost: float = 0.0,
    total_sample_seconds: float = 12.5,
) -> None:
    """Write a synthetic trajectory JSON to ``path`` shaped like what
    ``AsyncDefaultAgent.save`` would produce, with N observation
    messages carrying ``returncode`` in ``extra``.
    """
    messages: list[dict] = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "task"},
    ]
    for i in range(n_observations):
        msg: dict = {
            "role": observation_role,
            "content": f"<returncode>0</returncode>\n<output>\nline {i}</output>",
            "extra": {
                "raw_output": f"line {i}",
                "returncode": 0,
                "timestamp": 1_780_000_000.0 + i,
                "exception_info": None,
            },
        }
        if observation_role == "tool":
            msg["tool_call_id"] = f"tc-{i}"
        messages.append(msg)
    messages.append(
        {
            "role": "exit",
            "content": "42",
            "extra": {"exit_status": "Submitted", "submission": "42"},
        }
    )
    trajectory = {
        "info": {
            "model_stats": {
                "api_calls": api_calls,
                "instance_cost": instance_cost,
            },
            "config": {
                "total_sample_seconds": total_sample_seconds,
            },
        },
        "messages": messages,
    }
    path.write_text(json.dumps(trajectory))


def _build(traj_path: Path, expected: str = "42") -> "TaskResult":
    from example_recipe.word_puzzle.agent_solver.minisweagent.extract_answer import (
        build_task_result,
    )

    return build_task_result(
        tier="tierA",
        expected_answer=expected,
        run_return_value={"exit_status": "Submitted", "submission": expected},
        trajectory_path=traj_path,
    )


# --------------------------------------------------------------------------
# Regression: role="tool" observations must be counted
# --------------------------------------------------------------------------


def test_n_bash_calls_counts_tool_role_observations(tmp_path: Path):
    """Post-fix: toolcall-path observations (role="tool", with
    tool_call_id) are counted. Pre-fix, this returned 0 for every
    task using SamplerChatCompletionModel (which is every task in
    this PR).
    """
    traj = tmp_path / "trajectory.json"
    _write_trajectory(traj, observation_role="tool", n_observations=3)
    result = _build(traj)
    assert result.n_bash_calls == 3, (
        f"Expected 3 bash observations (role='tool'), got "
        f"{result.n_bash_calls}. The extract_answer filter must "
        f"match role='tool' — that's what mini's "
        f"format_toolcall_observation_messages emits for bash "
        f"observations."
    )


def test_n_bash_calls_still_counts_user_role_observations(tmp_path: Path):
    """Backward compat: text-based-model path (role="user") is still
    counted. Not currently exercised by any caller in this repo, but
    keeps the counter honest if a future model implementation goes
    back to the text-based observation shape.
    """
    traj = tmp_path / "trajectory.json"
    _write_trajectory(traj, observation_role="user", n_observations=2)
    result = _build(traj)
    assert result.n_bash_calls == 2


def test_n_bash_calls_zero_when_no_observations(tmp_path: Path):
    """Baseline: trajectory with no bash observations (agent went
    straight to submit) yields n_bash_calls=0.
    """
    traj = tmp_path / "trajectory.json"
    _write_trajectory(traj, observation_role="tool", n_observations=0)
    result = _build(traj)
    assert result.n_bash_calls == 0


def test_n_bash_calls_ignores_non_observation_user_messages(tmp_path: Path):
    """A user message WITHOUT returncode in extra (e.g. the initial
    task message, or a follow-up user turn) must not be counted.
    Post-fix filter still requires returncode-in-extra, not just
    role-matching.
    """
    from example_recipe.word_puzzle.agent_solver.minisweagent.extract_answer import (
        build_task_result,
    )

    traj_path = tmp_path / "trajectory.json"
    trajectory = {
        "info": {
            "model_stats": {"api_calls": 1, "instance_cost": 0.0},
            "config": {"total_sample_seconds": 5.0},
        },
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},  # NO extra.returncode
            # An assistant message that happens to have "returncode" in
            # extra (contrived case) must ALSO not be counted -- filter
            # requires role in {tool, user}, not "any role".
            {
                "role": "assistant",
                "content": "reasoning",
                "extra": {"returncode": 0, "raw_output": "trap"},
            },
            {
                "role": "exit",
                "content": "answer",
                "extra": {"exit_status": "Submitted", "submission": "answer"},
            },
        ],
    }
    traj_path.write_text(json.dumps(trajectory))
    result = build_task_result(
        tier="tierA",
        expected_answer="answer",
        run_return_value={"exit_status": "Submitted", "submission": "answer"},
        trajectory_path=traj_path,
    )
    assert result.n_bash_calls == 0


# --------------------------------------------------------------------------
# Other build_task_result plumbing (not touched by this fix, but worth
# pinning while we have a fixture)
# --------------------------------------------------------------------------


def test_build_task_result_plumbs_all_fields(tmp_path: Path):
    """One end-to-end check that every TaskResult field is populated
    from the expected source, so future refactors of extract_answer
    can't silently swap fields around."""
    traj = tmp_path / "trajectory.json"
    _write_trajectory(
        traj,
        observation_role="tool",
        n_observations=2,
        api_calls=5,
        instance_cost=0.0125,
        total_sample_seconds=42.3,
    )
    result = _build(traj, expected="42")
    assert result.tier == "tierA"
    assert result.expected_answer == "42"
    assert result.submitted_answer == "42"
    assert result.is_correct is True
    assert result.exit_status == "Submitted"
    assert result.n_model_calls == 5
    assert result.n_bash_calls == 2
    assert result.total_cost == pytest.approx(0.0125)
    assert result.total_model_seconds == pytest.approx(42.3)
    assert result.trajectory_path == traj
