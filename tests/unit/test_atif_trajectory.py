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

"""Tests for src/core/atif_trajectory.py (new types)."""

from __future__ import annotations

import pytest
from core.atif_trajectory import Agent, Trajectory

from core.atif_trajectory import (
    ScoredTrajectory,
    Reward,
    Step,
    StepStats,
    StepTimings,
    TrajectoryGroup,
    TrajectoryStep,
    build_leading_steps,
)
from core.rollout_status import RolloutStatus


def test_from_wire_populates_message_from_string_generation():
    stats = StepStats(
        finish_reason="stop", candidate_id="c0", train_step_id=7, operation_id="op-1"
    )
    step = TrajectoryStep.from_wire(
        step_id=2,
        generation="the reply",
        stats=stats,
        timings=StepTimings(sample_time_s=0.5, env_time_s=0.1),
    )
    assert step.step_id == 2
    assert step.source == "agent"
    assert step.message == "the reply"
    assert step.extra["finish_reason"] == "stop"
    assert step.extra["candidate_id"] == "c0"
    assert step.extra["train_step_id"] == "7"  # ATIF: str form
    assert step.extra["operation_id"] == "op-1"
    assert step.extra["sample_time_s"] == 0.5
    assert step.extra["env_time_s"] == 0.1


def test_from_wire_typed_accessors():
    stats = StepStats(candidate_id="c1", train_step_id=42, finish_reason="length")
    step = TrajectoryStep.from_wire(step_id=2, generation="g", stats=stats)
    assert step.candidate_id == "c1"
    assert step.train_step_id == 42  # int coerce from str extra
    assert step.finish_reason == "length"


def test_advantage_setter_writes_to_extra():
    step = TrajectoryStep.from_wire(step_id=2, generation="g", stats=StepStats())
    assert step.advantage is None
    step.advantage = 0.75
    assert step.advantage == 0.75
    assert step.extra["advantage"] == 0.75


def test_system_user_agent_steps_form_valid_trajectory():
    ss = Step(step_id=1, source="system", message="system prompt")
    us = Step(step_id=2, source="user", message="task instruction")
    a1 = TrajectoryStep.from_wire(step_id=3, generation="g1", stats=StepStats())
    traj = Trajectory(
        schema_version="ATIF-v1.7",
        session_id="test-session",
        agent=Agent(name="t", version="0", model_name="m"),
        steps=[ss, us, a1],
    )
    assert [s.source for s in traj.steps] == ["system", "user", "agent"]
    assert [s.step_id for s in traj.steps] == [1, 2, 3]


def test_leading_user_step_plus_agent_steps_form_valid_trajectory():
    us = Step(step_id=1, source="user", message="task instruction")
    a1 = TrajectoryStep.from_wire(step_id=2, generation="g1", stats=StepStats())
    a2 = TrajectoryStep.from_wire(step_id=3, generation="g2", stats=StepStats())
    traj = Trajectory(
        schema_version="ATIF-v1.7",
        session_id="test-session",
        agent=Agent(name="t", version="0", model_name="m"),
        steps=[us, a1, a2],
    )
    assert len(traj.steps) == 3
    assert traj.steps[0].source == "user"
    assert traj.steps[1].source == "agent"


def test_scored_trajectory_holds_trajectory_and_reward():
    us = Step(step_id=1, source="user", message="x")
    a = TrajectoryStep.from_wire(step_id=2, generation="g", stats=StepStats())
    traj = Trajectory(
        schema_version="ATIF-v1.7",
        session_id="s",
        agent=Agent(name="t", version="0", model_name="m"),
        steps=[us, a],
    )
    scored = ScoredTrajectory(trajectory=traj, reward=Reward(reward=0.7))
    assert scored.reward.reward == 0.7
    assert scored.reward.normalized_reward is None
    assert scored.trajectory is traj


def test_trajectory_group_holds_scored_trajectories():
    us = Step(step_id=1, source="user", message="x")
    a = TrajectoryStep.from_wire(step_id=2, generation="g", stats=StepStats())
    traj = Trajectory(
        schema_version="ATIF-v1.7",
        session_id="s",
        agent=Agent(name="t", version="0", model_name="m"),
        steps=[us, a],
    )
    scored = ScoredTrajectory(trajectory=traj, reward=Reward(reward=0.5))
    group = TrajectoryGroup(task_data=None, scored_trajectories=[scored])
    assert group.scored_trajectories[0].reward.reward == 0.5


def test_rollout_status_enum_present():
    assert RolloutStatus.OK.value == "OK"


from core import atif_extras


def test_atif_extras_accessors_roundtrip():
    us = Step(step_id=1, source="user", message="x")
    a = TrajectoryStep.from_wire(
        step_id=2,
        generation="g",
        stats=StepStats(train_step_id=5, candidate_id="c", finish_reason="stop"),
        timings=StepTimings(sample_time_s=0.4, env_time_s=0.2),
    )
    assert atif_extras.get_train_step_id(a) == 5
    assert atif_extras.get_candidate_id(a) == "c"
    assert atif_extras.get_finish_reason(a) == "stop"
    assert atif_extras.get_sample_time_s(a) == 0.4
    assert atif_extras.get_env_time_s(a) == 0.2

    atif_extras.set_advantage(a, 0.9)
    assert atif_extras.get_advantage(a) == 0.9

    traj = Trajectory(
        schema_version="ATIF-v1.7",
        session_id="s",
        agent=Agent(name="t", version="0", model_name="m"),
        steps=[us, a],
    )
    atif_extras.set_rollout_status(traj, RolloutStatus.OK)
    atif_extras.set_is_correct(traj, True)
    assert atif_extras.get_rollout_status(traj) == RolloutStatus.OK
    assert atif_extras.get_is_correct(traj) is True


def test_from_wire_carries_token_counts():
    """B1 regression: token counts must flow from StepStats into step.extra."""
    stats = StepStats(prompt_token_count=123, generation_token_count=45)
    step = TrajectoryStep.from_wire(
        step_id=2,
        generation="g",
        stats=stats,
    )
    assert step.extra["prompt_token_count"] == 123
    assert step.extra["generation_token_count"] == 45
    assert atif_extras.get_prompt_token_count(step) == 123
    assert atif_extras.get_generation_token_count(step) == 45


def test_from_wire_omits_token_counts_when_none():
    """None values should not appear as keys in extra (keeps extra tidy)."""
    stats = StepStats()  # both token counts default to None
    step = TrajectoryStep.from_wire(
        step_id=2,
        generation="g",
        stats=stats,
    )
    assert "prompt_token_count" not in step.extra
    assert "generation_token_count" not in step.extra


def test_from_wire_stamps_llm_call_uri_when_set() -> None:
    """raw_call_uri kwarg lands in step.extra['llm_call_uri']."""
    step = TrajectoryStep.from_wire(
        step_id=2,
        generation="g",
        stats=StepStats(),
        raw_call_uri="artifacts/raw_llm_calls/call_000003.json",
    )
    assert step.extra is not None
    assert step.extra.get("llm_call_uri") == "artifacts/raw_llm_calls/call_000003.json"


def test_from_wire_omits_llm_call_uri_when_none() -> None:
    """None should keep extra tidy (same convention as token counts)."""
    step = TrajectoryStep.from_wire(
        step_id=2,
        generation="g",
        stats=StepStats(),
        raw_call_uri=None,
    )
    assert "llm_call_uri" not in step.extra


def test_from_wire_llm_call_uri_default_is_none() -> None:
    """Backward-compat: existing callers that don't pass raw_call_uri
    must not get llm_call_uri stamped."""
    step = TrajectoryStep.from_wire(
        step_id=2,
        generation="g",
        stats=StepStats(),
    )
    assert "llm_call_uri" not in step.extra


def test_from_wire_llm_call_uri_round_trips_through_codec() -> None:
    """Codec accepts a step carrying llm_call_uri (declared field, no forbid).

    to_disk() invokes _validate_typed_extras which runs step.extra through
    CloudsideStepExtras (model_config extra='forbid'). If llm_call_uri were
    ever removed from that schema, this test would raise ValidationError.
    """
    import json as _json

    from core.atif_codec import to_disk

    step = TrajectoryStep.from_wire(
        step_id=2,
        generation="g",
        stats=StepStats(),
        raw_call_uri="artifacts/raw_llm_calls/call_000007.json",
    )
    user = Step(step_id=1, source="user", message="hi")
    traj = Trajectory(
        agent=Agent(name="test", version="0", model_name="m"),
        steps=[user, step],
        extra={"is_correct": True, "rollout_status": "OK"},
    )
    scored = ScoredTrajectory(trajectory=traj, reward=Reward(reward=1.0))
    # Must not raise (validates CloudsideStepExtras).
    agent_json, _reward_json = to_disk(scored)
    body = _json.loads(agent_json)
    agent_step = next(s for s in body["steps"] if s.get("source") == "agent")
    assert agent_step["extra"]["llm_call_uri"] == (
        "artifacts/raw_llm_calls/call_000007.json"
    )


def test_train_step_id_returns_none_on_garbage_extra():
    """I3 regression: don't crash if some producer wrote a non-numeric value."""
    step = TrajectoryStep.from_wire(step_id=2, generation="g", stats=StepStats())
    step.extra["train_step_id"] = "not-a-number"
    assert step.train_step_id is None
    assert atif_extras.get_train_step_id(step) is None


def test_rollout_status_is_canonical_not_duplicate():
    """Regression: prevent re-introducing a duplicate RolloutStatus enum.

    core.atif_trajectory re-exports the canonical RolloutStatus from
    core.rollout_status. If a future refactor accidentally re-creates
    a local enum, this test catches it.
    """
    from core import atif_trajectory as at
    from core import rollout_status as rs

    assert at.RolloutStatus is rs.RolloutStatus, (
        "atif_trajectory.RolloutStatus must BE core.rollout_status.RolloutStatus, "
        "not a distinct class. Delete any local `class RolloutStatus(...)` in "
        "atif_trajectory.py and re-export the canonical one instead."
    )
    # Canonical has richer members than the placeholder had:
    assert hasattr(rs.RolloutStatus, "MAX_STEP_LIMIT")
    assert hasattr(rs.RolloutStatus, "SAMPLING_FAILURE_TIMEOUT")


def test_from_wire_dict_generation_no_longer_stringifies_into_message():
    """Regression (superseded behavior): a dict generation used to be
    json.dumps'd into message. It now splits into a text message + ATIF
    tool_calls."""
    dict_gen = {"role": "model", "parts": [{"text": "hello"}]}
    step = TrajectoryStep.from_wire(
        step_id=2,
        generation=dict_gen,
        stats=StepStats(finish_reason="stop"),
    )
    assert step.message == "hello"
    assert step.tool_calls is None


def test_from_wire_still_accepts_str_generation():
    """Regression: str path unchanged (already covered but pin explicitly)."""
    step = TrajectoryStep.from_wire(
        step_id=2,
        generation="text reply",
        stats=StepStats(),
    )
    assert step.message == "text reply"


def test_from_wire_rejects_non_str_non_dict_generation():
    """Anything else (bytes, int, list, None) should raise TypeError
    with a clear message so future callers don't silently pass junk."""
    import pytest

    with pytest.raises(TypeError, match="generation must be str or dict"):
        TrajectoryStep.from_wire(
            step_id=2,
            generation=b"bytes",
            stats=StepStats(),
        )


def test_parse_generation_dict_splits_text_and_function_calls():
    """A Gemini generation dict must split into a text `message` and
    structured ATIF tool_calls (one per function_call part)."""
    from core.atif_trajectory import _parse_generation_dict

    gen = {
        "role": "model",
        "parts": [
            {"text": "I will search the repo."},
            {"function_call": {"name": "exec", "args": {"command1": "ls"}}},
            {"function_call": {"name": "exec", "args": {"command1": "grep x"}}},
            {"thought_signature": "opaque"},
        ],
    }
    message, tool_calls, _tool_responses = _parse_generation_dict(gen)
    assert message == "I will search the repo."
    assert len(tool_calls) == 2
    assert tool_calls[0].function_name == "exec"
    assert tool_calls[0].arguments == {"command1": "ls"}
    assert tool_calls[1].arguments == {"command1": "grep x"}
    assert tool_calls[0].tool_call_id and tool_calls[1].tool_call_id
    assert tool_calls[0].tool_call_id != tool_calls[1].tool_call_id


def test_parse_generation_dict_no_function_calls_returns_none_tool_calls():
    """Text-only generation -> message set, tool_calls None (not [])."""
    from core.atif_trajectory import _parse_generation_dict

    message, tool_calls, _tool_responses = _parse_generation_dict(
        {"role": "model", "parts": [{"text": "just text"}]}
    )
    assert message == "just text"
    assert tool_calls is None


def test_parse_generation_dict_only_function_calls_empty_message():
    """Function-call-only generation -> empty message, tool_calls populated."""
    from core.atif_trajectory import _parse_generation_dict

    message, tool_calls, _tool_responses = _parse_generation_dict(
        {"role": "model", "parts": [{"function_call": {"name": "exec", "args": {}}}]}
    )
    assert message == ""
    assert len(tool_calls) == 1
    assert tool_calls[0].arguments == {}


def test_parse_generation_dict_missing_parts_is_safe():
    """No 'parts' key -> empty message, None tool_calls (no crash)."""
    from core.atif_trajectory import _parse_generation_dict

    message, tool_calls, _tool_responses = _parse_generation_dict({"role": "model"})
    assert message == ""
    assert tool_calls is None


def test_parse_generation_dict_non_dict_args_defaults_to_empty():
    """A malformed function_call whose 'args' isn't a dict must not crash
    ToolCall validation; it defaults to {} (symmetric with the other guards)."""
    from core.atif_trajectory import _parse_generation_dict

    message, tool_calls, _tool_responses = _parse_generation_dict(
        {"parts": [{"function_call": {"name": "exec", "args": "not-a-dict"}}]}
    )
    assert message == ""
    assert len(tool_calls) == 1
    assert tool_calls[0].function_name == "exec"
    assert tool_calls[0].arguments == {}


def test_parse_generation_dict_excludes_thought_parts_from_message():
    """``thought=True`` parts are reasoning, not the reply: they must not be
    joined into ``message`` (a thinking model's chain-of-thought used to
    appear glued to the front of its answer)."""
    from core.atif_trajectory import _parse_generation_dict

    message, _, _ = _parse_generation_dict(
        {"parts": [{"text": "let me think", "thought": True}, {"text": "answer"}]}
    )
    assert message == "answer"


def test_from_wire_routes_thought_parts_to_reasoning_content():
    step = TrajectoryStep.from_wire(
        step_id=2,
        generation={
            "role": "model",
            "parts": [
                {"text": "step 1. ", "thought": True},
                {"text": "step 2.", "thought": True},
                {"text": "final"},
                {"function_call": {"name": "submit", "args": {"answer": "x"}}},
            ],
        },
        stats=StepStats(),
    )
    assert step.message == "final"
    assert step.reasoning_content == "step 1. step 2."
    assert step.tool_calls is not None and len(step.tool_calls) == 1


def test_from_wire_system_source_keeps_thoughts_in_message_and_revalidates():
    """Harbor allows ``reasoning_content`` only on ``source="agent"``; a
    system step (a discarded recovery sample) must fold its thoughts back
    into ``message`` (in part order, as before thought-splitting) so it
    survives a dump -> validate round-trip without losing them."""
    step = TrajectoryStep.from_wire(
        step_id=1,
        generation={
            "parts": [{"text": "hmm ", "thought": True}, {"text": "cut off"}]
        },
        stats=StepStats(),
        source="system",
    )
    assert step.source == "system"
    assert step.reasoning_content is None
    assert step.message == "hmm cut off"
    again = Step.model_validate_json(step.model_dump_json())
    assert again.message == "hmm cut off"


def test_from_wire_system_source_renders_function_calls_into_message():
    """Harbor forbids ``tool_calls`` (and the other agent-only fields) on
    non-agent steps, so a system step renders each discarded call as one
    ``[discarded function_call] name(<json args>)`` line after the text."""
    step = TrajectoryStep.from_wire(
        step_id=1,
        generation={
            "parts": [
                {"function_call": {"name": "f", "args": {"x": "é"}}},
                {"text": "t"},
                {"function_call": {"name": "g"}},
            ]
        },
        stats=StepStats(),
        source="system",
    )
    for field in (
        "model_name",
        "reasoning_effort",
        "reasoning_content",
        "tool_calls",
        "metrics",
    ):
        assert getattr(step, field) is None, field
    assert step.message == (
        't\n[discarded function_call] f({"x": "é"})\n[discarded function_call] g({})'
    )
    again = Step.model_validate_json(step.model_dump_json())
    assert again.message == step.message

    only_call = TrajectoryStep.from_wire(
        step_id=1,
        generation={"parts": [{"function_call": {"name": "g", "args": {}}}]},
        stats=StepStats(),
        source="system",
    )
    assert only_call.message == "[discarded function_call] g({})"


def test_from_wire_without_thoughts_leaves_reasoning_content_none():
    step = TrajectoryStep.from_wire(
        step_id=2,
        generation={"role": "model", "parts": [{"text": "final"}]},
        stats=StepStats(),
    )
    assert step.reasoning_content is None
    assert TrajectoryStep.from_wire(
        step_id=2, generation="g", stats=StepStats()
    ).reasoning_content is None


def test_from_wire_dict_generation_populates_tool_calls_and_clean_message():
    """from_wire on a tool-using generation dict: message = text parts,
    tool_calls = structured ATIF ToolCalls."""
    dict_gen = {
        "role": "model",
        "parts": [
            {"text": "searching"},
            {"function_call": {"name": "exec", "args": {"command1": "ls"}}},
        ],
    }
    step = TrajectoryStep.from_wire(
        step_id=2,
        generation=dict_gen,
        stats=StepStats(finish_reason="stop"),
    )
    assert step.message == "searching"
    assert step.tool_calls is not None and len(step.tool_calls) == 1
    assert step.tool_calls[0].function_name == "exec"
    assert step.tool_calls[0].arguments == {"command1": "ls"}
    s2 = TrajectoryStep.from_wire(
        step_id=3, generation="plain reply", stats=StepStats()
    )
    assert s2.message == "plain reply"
    assert s2.tool_calls is None


def test_from_wire_populates_observation_from_function_responses():
    """A tool-using step whose tool call produced an env result must surface
    that result in the ATIF `observation` field, linked to the tool_call_id
    via source_call_id (ATIF contract: observations reference tool_calls)."""
    dict_gen = {
        "role": "model",
        "parts": [
            {"function_call": {"name": "exec", "args": {"command1": "ls"}}},
        ],
    }
    function_responses = [
        {
            "function_response": {
                "name": "exec",
                "response": {"response": {"command1": "file_a.py\nfile_b.py"}},
            }
        }
    ]
    step = TrajectoryStep.from_wire(
        step_id=2,
        generation=dict_gen,
        stats=StepStats(),
        tool_responses=function_responses,
    )
    assert step.tool_calls is not None and len(step.tool_calls) == 1
    call_id = step.tool_calls[0].tool_call_id
    assert step.observation is not None
    assert len(step.observation.results) == 1
    result = step.observation.results[0]
    # The result is linked back to the tool call it answers.
    assert result.source_call_id == call_id
    # The result content carries the env's tool output.
    assert "file_a.py" in result.content


def test_from_wire_populates_observation_from_proto_json_function_responses():
    """Same as above but using the proto-JSON ``functionResponse`` (camelCase)
    wire shape emitted by envs that build parts via ``Part.from_json``.

    Both wire shapes must resolve to the same stringified ``response`` payload
    on the ATIF ``observation`` \u2014 otherwise proto-JSON callers (multi-step
    tool envs, the simple word-puzzle env) silently store the full wrapping dict.
    """
    dict_gen = {
        "role": "model",
        "parts": [
            {"function_call": {"name": "exec", "args": {"command1": "ls"}}},
        ],
    }
    function_responses = [
        {
            "functionResponse": {
                "name": "exec",
                "response": {"response": {"command1": "file_a.py\nfile_b.py"}},
            }
        }
    ]
    step = TrajectoryStep.from_wire(
        step_id=2,
        generation=dict_gen,
        stats=StepStats(),
        tool_responses=function_responses,
    )
    assert step.observation is not None
    result = step.observation.results[0]
    assert "file_a.py" in result.content
    # The clean response payload IS surfaced (not the whole wrapping dict).
    assert "functionResponse" not in result.content


def test_from_wire_no_observation_when_no_function_responses():
    """A step with tool_calls but no env feedback (or a text-only step) has
    observation=None (ATIF convention: absent, not empty)."""
    step = TrajectoryStep.from_wire(
        step_id=2,
        generation={
            "role": "model",
            "parts": [{"function_call": {"name": "exec", "args": {}}}],
        },
        stats=StepStats(),
    )
    assert step.observation is None
    text_step = TrajectoryStep.from_wire(
        step_id=3, generation="just text", stats=StepStats()
    )
    assert text_step.observation is None


def test_from_wire_observation_pairs_multiple_responses_to_calls():
    """Multiple function_responses pair to multiple tool_calls by position,
    each linked to its own tool_call_id."""
    dict_gen = {
        "role": "model",
        "parts": [
            {"function_call": {"name": "exec", "args": {"command1": "ls"}}},
            {"function_call": {"name": "exec", "args": {"command1": "pwd"}}},
        ],
    }
    function_responses = [
        {"function_response": {"name": "exec", "response": {"response": "listing"}}},
        {"function_response": {"name": "exec", "response": {"response": "/repo"}}},
    ]
    step = TrajectoryStep.from_wire(
        step_id=2,
        generation=dict_gen,
        stats=StepStats(),
        tool_responses=function_responses,
    )
    assert step.observation is not None
    assert len(step.observation.results) == 2
    ids = {r.source_call_id for r in step.observation.results}
    call_ids = {tc.tool_call_id for tc in step.tool_calls}
    assert ids == call_ids
    assert "listing" in step.observation.results[0].content
    assert "/repo" in step.observation.results[1].content


# ---- build_leading_steps: reconstruct leading + agent history from a
# (possibly partial) GenerateContentRequest --------------------------------


def _make_gcr(system_text: str | None, contents_dicts: list[dict[str, Any]]):
    """Small helper: build a GenerateContentRequest from wire-shape dicts."""
    from google.cloud.aiplatform_v1.types import Content, GenerateContentRequest

    req = GenerateContentRequest()
    if system_text:
        req.system_instruction = Content(role="user", parts=[{"text": system_text}])
    for cd in contents_dicts:
        req.contents.append(Content(**cd))
    return req


def test_build_leading_steps_user_only_request():
    """The minimum-viable request: no system instruction, one user turn.
    Produces a single ``user`` leading step (no agent history)."""
    req = _make_gcr(None, [{"role": "user", "parts": [{"text": "solve X"}]}])
    steps = build_leading_steps(req)
    assert [s.source for s in steps] == ["user"]
    assert [s.step_id for s in steps] == [1]
    assert steps[0].message == "solve X"


def test_build_leading_steps_carries_thought_parts_as_reasoning_content():
    """Reconstructed agent turns route thought parts to reasoning_content,
    same as live steps (from_wire) -- neither dropped nor left in message."""
    req = _make_gcr(
        None,
        [
            {"role": "user", "parts": [{"text": "q"}]},
            {
                "role": "model",
                "parts": [
                    {"text": "think A", "thought": True},
                    {"function_call": {"name": "bash", "args": {"command": "ls"}}},
                ],
            },
            {
                "role": "user",
                "parts": [{"function_response": {"name": "bash", "response": {"o": "x"}}}],
            },
            {
                "role": "model",
                "parts": [{"text": "think B", "thought": True}, {"text": "done"}],
            },
        ],
    )
    steps = build_leading_steps(req)
    agent = [s for s in steps if s.source == "agent"]
    assert [(s.message, s.reasoning_content) for s in agent] == [
        ("", "think A"),
        ("done", "think B"),
    ]


def test_build_leading_steps_system_plus_user_request():
    """system_instruction present → prepended as leading system step;
    step_ids stay contiguous starting at 1."""
    req = _make_gcr(
        "you are a helpful agent",
        [{"role": "user", "parts": [{"text": "do the task"}]}],
    )
    steps = build_leading_steps(req)
    assert [s.source for s in steps] == ["system", "user"]
    assert [s.step_id for s in steps] == [1, 2]
    assert steps[0].message == "you are a helpful agent"
    assert steps[1].message == "do the task"


def test_build_leading_steps_reconstructs_multi_turn_agent_history():
    """A partial GCR carrying prior agent turns (model + tool_response +
    another user prompt) reconstructs those turns as ATIF ``agent`` steps
    with observations linked to the corresponding tool_calls."""
    req = _make_gcr(
        "sys",
        [
            {"role": "user", "parts": [{"text": "start"}]},
            {
                "role": "model",
                "parts": [
                    {"text": "let me search"},
                    {"function_call": {"name": "grep", "args": {"q": "foo"}}},
                ],
            },
            {
                "role": "user",
                "parts": [
                    {
                        "function_response": {
                            "name": "grep",
                            "response": {"response": "foo/bar.py"},
                        }
                    }
                ],
            },
        ],
    )
    steps = build_leading_steps(req)
    # system + leading user + reconstructed agent (with observation).
    assert [s.source for s in steps] == ["system", "user", "agent"]
    assert [s.step_id for s in steps] == [1, 2, 3]
    agent = steps[2]
    assert agent.message == "let me search"
    assert agent.tool_calls is not None and len(agent.tool_calls) == 1
    assert agent.tool_calls[0].function_name == "grep"
    assert agent.observation is not None
    assert (
        agent.observation.results[0].source_call_id == agent.tool_calls[0].tool_call_id
    )
    assert "foo/bar.py" in agent.observation.results[0].content


def test_build_leading_steps_trailing_agent_turn_without_response():
    """A tail agent turn with a plain text message (no tool_calls) is emitted
    as a text-only agent step — this is the resumability case where the
    request stops at the last agent output waiting for a new user turn."""
    req = _make_gcr(
        None,
        [
            {"role": "user", "parts": [{"text": "hi"}]},
            {"role": "model", "parts": [{"text": "hello back"}]},
        ],
    )
    steps = build_leading_steps(req)
    assert [s.source for s in steps] == ["user", "agent"]
    assert steps[1].message == "hello back"
    assert steps[1].tool_calls is None
    assert steps[1].observation is None


def test_build_leading_steps_rejects_non_user_first_turn():
    """The first content MUST be user-role; anything else is malformed."""
    req = _make_gcr(None, [{"role": "model", "parts": [{"text": "oops"}]}])
    with pytest.raises(ValueError, match="first role is not user"):
        build_leading_steps(req)


def test_build_leading_steps_rejects_tail_tool_call_without_response():
    """A tail agent turn that emitted tool_calls but has no matching
    tool_response is a malformed request (the tool call was never resolved)."""
    req = _make_gcr(
        None,
        [
            {"role": "user", "parts": [{"text": "start"}]},
            {
                "role": "model",
                "parts": [{"function_call": {"name": "grep", "args": {}}}],
            },
        ],
    )
    with pytest.raises(ValueError, match="Tail tool calls"):
        build_leading_steps(req)
