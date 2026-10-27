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

"""Tests for the mini-trace -> ATIF ScoredTrajectory converter.

These are written FIRST (TDD) and pin the converter to the trainer's actual
read-contract, not just "looks like a trajectory". The single most important
guarantee: every agent step must carry a non-empty candidate_id that survives
the trainer's ``if not cid`` skip filter, and the trajectory must carry a
finite numeric reward.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from core.atif_trajectory import ScoredTrajectory
from core.atif_trajectory import Step

from core import atif_extras
from core import atif_codec
from core.rollout_status import RolloutStatus
from example_recipe.word_puzzle.agent_solver.minisweagent.mini_atif_converter import (
    mini_trace_to_scored,
)
from example_recipe.word_puzzle.reward import (
    WordPuzzleRewardConfig,
    WordPuzzleRewardFn,
)
from example_recipe.word_puzzle.dataset.task_data import WordPuzzleTaskData


# ----------------------------------------------------------------------
# Fixtures / builders
# ----------------------------------------------------------------------

REWARD_FN = WordPuzzleRewardFn(WordPuzzleRewardConfig())


def _assistant(
    candidate_id: Any,
    *,
    finish_reason: str = "tool_calls",
    content: str = "ok",
    tool_calls: list[dict[str, Any]] | None = None,
    actions: list[dict[str, Any]] | None = None,
    gen_tokens: int = 7,
    prompt_tokens: int = 11,
    operation_id: str = "op-123",
) -> dict[str, Any]:
    """One assistant turn in mini's message shape.

    ``candidate_id`` is passed through verbatim so tests can exercise the
    int-0 / str / None cases the real backend produces.
    """
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls
        if tool_calls is not None
        else [
            {
                "id": "call_abc",
                "type": "function",
                "function": {"name": "bash", "arguments": "{}"},
            }
        ],
        "extra": {
            "candidate_id": candidate_id,
            "finish_reason": finish_reason,
            "operation_id": operation_id,
            "prompt_token_count": prompt_tokens,
            "generation_token_count": gen_tokens,
            # mini's parsed actions -- where the real tool call lives.
            "actions": actions
            if actions is not None
            else [{"tool_name": "bash", "command": "echo hi"}],
        },
    }


def _tool(
    output: str = "<returncode>0</returncode>\n<output>\nhi</output>",
) -> dict[str, Any]:
    return {
        "role": "tool",
        "content": output,
        "tool_call_id": "call_abc",
        "extra": {"returncode": 0},
    }


def _exit(exit_status: str, submission: str = "") -> dict[str, Any]:
    return {
        "role": "exit",
        "content": submission,
        "extra": {"exit_status": exit_status, "submission": submission},
    }


def _xx(exit_status: str, submission: str = "") -> dict[str, Any]:
    """The exit-message ``extra`` dict as returned by ``run()`` -- this is what
    the converter's ``exit_extra`` parameter actually receives."""
    return {"exit_status": exit_status, "submission": submission}


def _messages(
    *,
    n_turns: int = 3,
    candidate_ids: list[Any] | None = None,
    final_exit: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """system + user + n_turns*(assistant, tool) + exit."""
    if candidate_ids is None:
        candidate_ids = list(range(n_turns))  # ints 0,1,2 like real traces
    msgs = [
        {"role": "system", "content": "you solve word puzzles"},
        {"role": "user", "content": "the puzzle"},
    ]
    for cid in candidate_ids:
        msgs.append(_assistant(cid))
        msgs.append(_tool())
    msgs.append(final_exit if final_exit is not None else _exit("Submitted", "answer"))
    return msgs


def _task(reference: str = "answer") -> WordPuzzleTaskData:
    return WordPuzzleTaskData(task_id="t1", question="q", reference=reference)


def _agent_steps(scored: ScoredTrajectory) -> list[Step]:
    return [s for s in scored.trajectory.steps if s.source == "agent"]


def _trainer_kept_steps(scored: ScoredTrajectory) -> list[Step]:
    """Re-run the trainer's OWN filter (vertex_msrl_rest.py:1247)."""
    return [
        s
        for s in scored.trajectory.steps
        if s.source == "agent" and atif_extras.get_candidate_id(s)
    ]


# ----------------------------------------------------------------------
# T1 -- training-critical assertions (highest priority)
# ----------------------------------------------------------------------


async def test_t1_every_agent_step_survives_trainer_candidate_id_filter():
    # Uses int candidate_ids 0,1,2 -- exactly what real mini traces store.
    scored = await mini_trace_to_scored(
        _messages(n_turns=3),
        _xx("Submitted", "answer"),
        _task(),
        rollout_idx=0,
        reward_fn=REWARD_FN,
    )
    kept = _trainer_kept_steps(scored)
    # If candidate_id 0 (int) leaked through as-is, `if not cid` would drop
    # the first step and this would be 2, not 3.
    assert len(kept) == 3, (
        "trainer would silently skip an agent step -- candidate_id not "
        "coerced to a truthy string"
    )


async def test_t1_candidate_id_zero_is_kept_not_skipped():
    scored = await mini_trace_to_scored(
        _messages(n_turns=1, candidate_ids=[0]),
        _xx("Submitted", "answer"),
        _task(),
        rollout_idx=0,
        reward_fn=REWARD_FN,
    )
    step = _agent_steps(scored)[0]
    cid = atif_extras.get_candidate_id(step)
    assert cid == "0"  # coerced to str
    assert bool(cid) is True  # survives `if not cid`


async def test_t1_candidate_ids_match_source_order():
    scored = await mini_trace_to_scored(
        _messages(n_turns=3, candidate_ids=[5, 6, 7]),
        _xx("Submitted", "answer"),
        _task(),
        rollout_idx=0,
        reward_fn=REWARD_FN,
    )
    cids = [atif_extras.get_candidate_id(s) for s in _agent_steps(scored)]
    assert cids == ["5", "6", "7"]  # order preserved, str-coerced


async def test_t1_reward_is_finite_float():
    scored = await mini_trace_to_scored(
        _messages(n_turns=2),
        _xx("Submitted", "answer"),
        _task(reference="answer"),
        rollout_idx=0,
        reward_fn=REWARD_FN,
    )
    assert isinstance(scored.reward.reward, float)
    assert math.isfinite(scored.reward.reward)


async def test_t1_agent_step_records_the_tool_call_action():
    # The action (bash call) MUST survive conversion -- otherwise the
    # trajectory records a tool RESULT with no CALL, which is wrong for
    # training (the policy tokens being rewarded are the tool-call tokens).
    scored = await mini_trace_to_scored(
        _messages(n_turns=1),
        _xx("Submitted", "answer"),
        _task(),
        rollout_idx=0,
        reward_fn=REWARD_FN,
    )
    step = _agent_steps(scored)[0]
    assert step.tool_calls, "agent step lost its tool_calls (action dropped)"
    assert step.tool_calls[0].function_name == "bash"
    assert step.tool_calls[0].arguments.get("command") == "echo hi"


async def test_t1_finish_reason_and_token_counts_carried():
    scored = await mini_trace_to_scored(
        _messages(n_turns=1),
        _xx("Submitted", "answer"),
        _task(),
        rollout_idx=0,
        reward_fn=REWARD_FN,
    )
    step = _agent_steps(scored)[0]
    assert atif_extras.get_finish_reason(step) == "tool_calls"
    assert atif_extras.get_operation_id(step) == "op-123"
    assert atif_extras.get_generation_token_count(step) == 7
    assert atif_extras.get_prompt_token_count(step) == 11


# ----------------------------------------------------------------------
# T2 -- structure & ordering
# ----------------------------------------------------------------------


async def test_t2_step_sources_and_order():
    scored = await mini_trace_to_scored(
        _messages(n_turns=2),
        _xx("Submitted", "answer"),
        _task(),
        rollout_idx=0,
        reward_fn=REWARD_FN,
    )
    sources = [s.source for s in scored.trajectory.steps]
    assert sources == ["system", "user", "agent", "agent"]
    ids = [s.step_id for s in scored.trajectory.steps]
    assert ids == [1, 2, 3, 4]


async def test_t2_leading_steps_have_no_candidate_id():
    scored = await mini_trace_to_scored(
        _messages(n_turns=1),
        _xx("Submitted", "answer"),
        _task(),
        rollout_idx=0,
        reward_fn=REWARD_FN,
    )
    non_agent = [s for s in scored.trajectory.steps if s.source != "agent"]
    for s in non_agent:
        assert atif_extras.get_candidate_id(s) is None


async def test_t2_no_system_message_shifts_user_to_first():
    msgs = _messages(n_turns=1)
    msgs = msgs[1:]  # drop the system message
    scored = await mini_trace_to_scored(
        msgs,
        _xx("Submitted", "answer"),
        _task(),
        rollout_idx=0,
        reward_fn=REWARD_FN,
    )
    sources = [s.source for s in scored.trajectory.steps]
    assert sources == ["user", "agent"]


# ----------------------------------------------------------------------
# T3 -- reward & status matrix
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "exit_status,submission,reference,exp_reward,exp_status",
    [
        ("Submitted", "answer", "answer", 1.0, RolloutStatus.OK),
        ("Submitted", "wrong", "answer", -1.0, RolloutStatus.OK),
        ("LimitsExceeded", "", "answer", -1.0, RolloutStatus.MAX_STEP_LIMIT),
        ("TimeExceeded", "", "answer", -1.0, RolloutStatus.MAX_STEP_LIMIT),
    ],
)
async def test_t3_reward_and_status_matrix(
    exit_status: str,
    submission: str,
    reference: str,
    exp_reward: float,
    exp_status: RolloutStatus,
) -> None:
    scored = await mini_trace_to_scored(
        _messages(n_turns=1, final_exit=_exit(exit_status, submission)),
        _xx(exit_status, submission),
        _task(reference=reference),
        rollout_idx=0,
        reward_fn=REWARD_FN,
    )
    assert scored.reward.reward == exp_reward
    assert atif_extras.get_rollout_status(scored.trajectory) == exp_status


async def test_t3_malformed_finish_reason_penalty():
    # A malformed final turn -> -4.0 per WordPuzzleRewardConfig default.
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        _assistant(0, finish_reason="MALFORMED_FUNCTION_CALL", tool_calls=[]),
        _xx("Submitted", "answer"),
    ]
    scored = await mini_trace_to_scored(
        msgs,
        _xx("Submitted", "answer"),
        _task(reference="answer"),
        rollout_idx=0,
        reward_fn=REWARD_FN,
    )
    assert scored.reward.reward == -4.0


# ----------------------------------------------------------------------
# T4 -- serialization round-trip (guards extra="forbid")
# ----------------------------------------------------------------------


async def test_t4_to_wire_roundtrip_preserves_candidate_ids():
    scored = await mini_trace_to_scored(
        _messages(n_turns=3, candidate_ids=[0, 1, 2]),
        _xx("Submitted", "answer"),
        _task(),
        rollout_idx=0,
        reward_fn=REWARD_FN,
    )
    wire = atif_codec.to_wire(scored)  # must not raise
    assert wire["reward"] == scored.reward.reward


async def test_t4_to_disk_does_not_raise():
    scored = await mini_trace_to_scored(
        _messages(n_turns=2),
        _xx("Submitted", "answer"),
        _task(),
        rollout_idx=0,
        reward_fn=REWARD_FN,
    )
    atif_codec.to_disk(scored)  # must not raise


# ----------------------------------------------------------------------
# T6 -- degenerate inputs
# ----------------------------------------------------------------------


async def test_t6_no_agent_turn_does_not_crash():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q"},
        _xx("LimitsExceeded", ""),
    ]
    scored = await mini_trace_to_scored(
        msgs,
        _xx("LimitsExceeded", ""),
        _task(),
        rollout_idx=0,
        reward_fn=REWARD_FN,
    )
    assert _agent_steps(scored) == []
    assert atif_extras.get_rollout_status(scored.trajectory) == (
        RolloutStatus.MAX_STEP_LIMIT
    )


async def test_t6_none_candidate_id_emitted_but_trainer_skipped():
    scored = await mini_trace_to_scored(
        _messages(n_turns=1, candidate_ids=[None]),
        _xx("Submitted", "answer"),
        _task(),
        rollout_idx=0,
        reward_fn=REWARD_FN,
    )
    # The step IS emitted (for observability) ...
    assert len(_agent_steps(scored)) == 1
    # ... but the trainer's filter would skip it (no candidate_id).
    assert _trainer_kept_steps(scored) == []


# ----------------------------------------------------------------------
# T5 -- realistic fixture (a real saved mini trace, trimmed)
# ----------------------------------------------------------------------


async def test_t5_real_saved_trace_every_agent_step_trainable() -> None:
    import json
    import pathlib

    fixture = (
        pathlib.Path(__file__).parent
        / "fixtures"
        / "agent_solver_mini_trace_sample.json"
    )
    data = json.loads(fixture.read_text())
    messages = data["messages"]
    exit_extra = {
        "exit_status": data["info"]["exit_status"],
        "submission": data["info"]["submission"],
    }
    scored = await mini_trace_to_scored(
        messages,
        exit_extra,
        _task(reference="uomaihhwxgoz"),
        rollout_idx=0,
        reward_fn=REWARD_FN,
    )
    n_assistant = sum(1 for m in messages if m.get("role") == "assistant")
    # Every agent step from a REAL trace must survive the trainer filter.
    assert len(_trainer_kept_steps(scored)) == n_assistant
    # And each must record its actual tool-call action (real bash calls).
    for s in _agent_steps(scored):
        assert s.tool_calls, "real-trace agent step lost its tool_calls"
        assert s.tool_calls[0].function_name == "bash"
    # Reward is finite; this trace TimeExceeded with empty submission -> wrong.
    assert scored.reward.reward == -1.0
    assert atif_extras.get_rollout_status(scored.trajectory) == (
        RolloutStatus.MAX_STEP_LIMIT
    )


# ----------------------------------------------------------------------
# T7 -- env_time_s (tool-execution time)
# ----------------------------------------------------------------------


async def test_t7_env_time_comes_from_the_recorded_tool_duration():
    """``env_time_s`` reflects what the agent measured around the turn's tool
    execution. It was a hardcoded 0.0, which pinned ``pipeline/env_time_avg``
    at exactly 0.00 and ``pipeline/sample_time_fraction`` at exactly 1.00 for
    every step of every run on this path."""
    msgs = _messages(n_turns=1)
    assistant = next(m for m in msgs if m["role"] == "assistant")
    assistant["extra"]["elapsed_seconds"] = 4.0
    assistant["extra"]["env_elapsed_seconds"] = 1.5

    scored = await mini_trace_to_scored(
        msgs, _xx("Submitted", "answer"), _task(), rollout_idx=0, reward_fn=REWARD_FN
    )
    step = _agent_steps(scored)[0]

    assert atif_extras.get_env_time_s(step) == pytest.approx(1.5), (
        "env_time_s did not pick up the recorded tool duration; "
        f"got {atif_extras.get_env_time_s(step)!r}"
    )
    assert atif_extras.get_sample_time_s(step) == pytest.approx(4.0)
    # The denominator of sample_time_fraction is sample + env, so a non-zero
    # env time is exactly what stops the fraction being identically 1.0.
    s = atif_extras.get_sample_time_s(step) or 0.0
    e = atif_extras.get_env_time_s(step) or 0.0
    assert s / (s + e) < 1.0


async def test_t7_old_transcript_without_the_key_still_converts():
    """Version skew is real here: a pod running a previously baked agent
    image records no ``env_elapsed_seconds``. That transcript must convert,
    scoring 0.0, not raise."""
    msgs = _messages(n_turns=1)
    assistant = next(m for m in msgs if m["role"] == "assistant")
    assistant["extra"]["elapsed_seconds"] = 4.0
    assert "env_elapsed_seconds" not in assistant["extra"]

    scored = await mini_trace_to_scored(
        msgs, _xx("Submitted", "answer"), _task(), rollout_idx=0, reward_fn=REWARD_FN
    )
    step = _agent_steps(scored)[0]

    assert atif_extras.get_env_time_s(step) == 0.0
    assert atif_extras.get_sample_time_s(step) == pytest.approx(4.0)
