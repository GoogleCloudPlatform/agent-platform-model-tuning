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

"""Parallel-tool-calls dispatch tests for ``AsyncDefaultAgent.execute_actions``.

Motivation: the model MAY emit multiple tool_calls per turn. An earlier
implementation raised ``Submitted`` immediately on the first submit
action, which:

  * Dropped any bash actions appearing AFTER a submit in the same turn.
  * Silently lost the observation message for bash actions appearing
    BEFORE the submit — the ``add_messages`` call at the end of the
    loop was unreachable once ``Submitted`` fired.

Fixed policy: run ALL non-submit actions first (appending their outputs
as one observation message), then handle a deferred submit if any. This
file pins the four permutations that matter:

  * single ``[submit]`` — no bash, submit fires (baseline).
  * single ``[bash]`` — no submit, observation appended, no raise.
  * ``[bash, submit]`` — bash runs + output visible + submit fires.
  * ``[submit, bash]`` — bash STILL runs + output visible + submit fires.
  * Two submits — first wins, deterministic.

Round-4 addendum: the round-3 fix passed the UNFILTERED ``message`` to
``format_observation_messages``, which zips ``extra.actions`` against
outputs — so a mixed ``[bash, submit]`` or ``[submit, bash]`` turn
would produce misaligned / spurious tool-observation messages on the
trajectory. Fix: pass a shim message with
``extra.actions`` filtered to only executed (non-submit) actions.
The ``TestObservationMessageAlignment`` class pins that the emitted
observation messages match the executed actions 1:1 by using the REAL
``format_toolcall_observation_messages`` (via a real
``SamplerChatCompletionModel`` instance) rather than a stub model.
"""

from __future__ import annotations

import pytest

# execute_actions lives on AsyncDefaultAgent (imports minisweagent).
# Skip cleanly if the agent_solver extra isn't installed (parity with
# test_async_default_agent_bash_cap.py).
pytest.importorskip("minisweagent")

from minisweagent.exceptions import Submitted  # noqa: E402

from example_recipe.word_puzzle.agent_solver.minisweagent.async_default_agent import (  # noqa: E402
    AsyncDefaultAgent,
)


# --------------------------------------------------------------------------
# Fake env + model wired into a real AsyncDefaultAgent
# --------------------------------------------------------------------------


class _FakeEnv:
    """Records every ``execute(action)`` call and returns a canned output.

    Matches the two attributes execute_actions reads: ``execute()``
    (called with the bash action dict, must return a dict with an
    ``output`` key so ``_cap_bash_output`` is a no-op passthrough).
    """

    def __init__(self):
        self.calls: list[dict] = []
        self.config = _NullEnvConfig()

    def execute(self, action: dict) -> dict:
        self.calls.append(action)
        return {"output": f"stdout of {action.get('command', '?')}", "returncode": 0}

    def get_template_vars(self) -> dict:
        return {}


class _NullEnvConfig:
    """Placeholder for env.config which mini's DefaultAgent references
    during template rendering. We never render templates in these
    tests (we bypass ``format_observation_messages`` via a fake model),
    but the AsyncDefaultAgent constructor path may still touch it. Any
    attribute lookup returns None so nothing blows up."""

    def __getattr__(self, name: str):
        return None


class _FakeModel:
    """Records every ``format_observation_messages`` call and returns
    a single stub message that ``execute_actions`` appends via
    ``self.add_messages(*...)``.

    Matches the two attributes execute_actions reads at the tail:
    ``format_observation_messages(message, outputs, template_vars)``
    → iterable of message dicts.
    """

    def __init__(self):
        self.format_calls: list[tuple[dict, list[dict]]] = []
        self.config = _NullModelConfig()

    def format_observation_messages(self, message, outputs, template_vars):
        self.format_calls.append((message, list(outputs)))
        # Return one observation-message per call. Content includes the
        # count so assertions can distinguish "was format called at all"
        # from "was format called with N outputs".
        return [
            {
                "role": "user",
                "content": f"OBS: {len(outputs)} bash output(s)",
            }
        ]

    async def query(self, messages, **kwargs):
        # Not exercised by these tests (we call execute_actions directly).
        raise NotImplementedError

    def serialize(self):
        return {}

    def get_template_vars(self) -> dict:
        return {}


class _NullModelConfig:
    def __getattr__(self, name: str):
        return None


def _make_agent() -> tuple[AsyncDefaultAgent, _FakeEnv, _FakeModel]:
    """Instantiate an AsyncDefaultAgent with fake env + model, bypassing
    template config (DefaultAgent's __init__ requires system_template /
    instance_template, so pass empty strings; execute_actions doesn't
    touch them)."""
    env = _FakeEnv()
    model = _FakeModel()
    agent = AsyncDefaultAgent(
        model=model,
        env=env,
        system_template="",
        instance_template="",
    )
    return agent, env, model


def _msg(actions: list[dict]) -> dict:
    """Build the assistant message shape execute_actions expects."""
    return {
        "role": "assistant",
        "content": "",
        "extra": {"actions": actions},
    }


def _bash(cmd: str, tc_id: str = "call-b") -> dict:
    return {"tool_name": "bash", "command": cmd, "tool_call_id": tc_id}


def _submit(answer: str, tc_id: str = "call-s") -> dict:
    return {"tool_name": "submit", "answer": answer, "tool_call_id": tc_id}


# --------------------------------------------------------------------------
# Baseline: single-action turns still work
# --------------------------------------------------------------------------


async def test_single_bash_action_appends_observation_no_raise():
    agent, env, model = _make_agent()
    await agent.execute_actions(_msg([_bash("echo hello")]))
    assert len(env.calls) == 1
    assert env.calls[0]["command"] == "echo hello"
    assert len(model.format_calls) == 1
    _, outputs = model.format_calls[0]
    assert len(outputs) == 1
    assert "hello" in outputs[0]["output"]


async def test_single_submit_action_raises_submitted():
    agent, env, model = _make_agent()
    with pytest.raises(Submitted) as excinfo:
        await agent.execute_actions(_msg([_submit("42", tc_id="tc-42")]))
    # Env was never called (submit bypasses the sandbox entirely).
    assert env.calls == []
    # format_observation_messages was not called (no outputs to format).
    assert model.format_calls == []
    # Exit message payload matches the terminal contract.
    exit_msg = excinfo.value.messages[0]
    assert exit_msg["role"] == "exit"
    assert exit_msg["content"] == "42"
    assert exit_msg["extra"]["exit_status"] == "Submitted"
    assert exit_msg["extra"]["submission"] == "42"
    assert exit_msg["extra"]["tool_call_id"] == "tc-42"


# --------------------------------------------------------------------------
# Regression: parallel tool_calls dispatch order
# --------------------------------------------------------------------------


async def test_bash_then_submit_runs_bash_before_submitting():
    """``[bash, submit]``: bash runs, its output is appended to the
    trajectory (via format_observation_messages), THEN submit fires.

    Pre-fix behavior: submit's raise happened mid-loop and the
    ``self.add_messages(*format_observation_messages(...))`` call at
    the tail was unreachable, so the bash's stdout was silently lost
    from the trajectory even though env.execute ran it.
    """
    agent, env, model = _make_agent()
    with pytest.raises(Submitted) as excinfo:
        await agent.execute_actions(
            _msg([_bash("echo pre-submit", tc_id="tc-b"), _submit("done")])
        )
    # Bash ran.
    assert len(env.calls) == 1
    assert env.calls[0]["command"] == "echo pre-submit"
    # Observation was formatted + added BEFORE the submit raised.
    assert len(model.format_calls) == 1
    _, outputs = model.format_calls[0]
    assert len(outputs) == 1
    assert "pre-submit" in outputs[0]["output"]
    # Submit's payload survived the deferred dispatch.
    assert excinfo.value.messages[0]["content"] == "done"


async def test_submit_then_bash_still_runs_bash_before_submitting():
    """``[submit, bash]``: even though submit appears first in the
    action list, we defer it until AFTER all non-submit actions run.
    The trailing bash's output is still captured (matching the
    "trajectory shows every bash the model asked for" invariant).

    Pre-fix behavior: submit raised immediately at index 0; the
    trailing bash was silently dropped (env.execute never called).
    """
    agent, env, model = _make_agent()
    with pytest.raises(Submitted) as excinfo:
        await agent.execute_actions(
            _msg([_submit("early"), _bash("echo post-submit", tc_id="tc-b")])
        )
    # Bash ran despite appearing AFTER the submit in the list.
    assert len(env.calls) == 1
    assert env.calls[0]["command"] == "echo post-submit"
    # Observation was formatted with the bash's output.
    assert len(model.format_calls) == 1
    _, outputs = model.format_calls[0]
    assert len(outputs) == 1
    assert "post-submit" in outputs[0]["output"]
    # Submit fires last with its own payload.
    assert excinfo.value.messages[0]["content"] == "early"


async def test_multiple_bash_calls_all_run_in_one_observation():
    """``[bash1, bash2, bash3]``: all three bashes run in order and
    the three outputs are batched into ONE observation message (one
    format_observation_messages call, three outputs).
    """
    agent, env, model = _make_agent()
    await agent.execute_actions(
        _msg(
            [
                _bash("echo one", tc_id="tc-1"),
                _bash("echo two", tc_id="tc-2"),
                _bash("echo three", tc_id="tc-3"),
            ]
        )
    )
    assert [c["command"] for c in env.calls] == ["echo one", "echo two", "echo three"]
    assert len(model.format_calls) == 1
    _, outputs = model.format_calls[0]
    assert len(outputs) == 3


async def test_bash_bash_submit_runs_both_bashes_before_submitting():
    """``[bash1, bash2, submit]``: both bashes run, both outputs batched
    into one observation, then submit fires.
    """
    agent, env, model = _make_agent()
    with pytest.raises(Submitted) as excinfo:
        await agent.execute_actions(
            _msg(
                [
                    _bash("echo a", tc_id="tc-a"),
                    _bash("echo b", tc_id="tc-b"),
                    _submit("answer"),
                ]
            )
        )
    assert [c["command"] for c in env.calls] == ["echo a", "echo b"]
    assert len(model.format_calls) == 1
    _, outputs = model.format_calls[0]
    assert len(outputs) == 2
    assert excinfo.value.messages[0]["content"] == "answer"


async def test_two_submits_first_wins_second_ignored():
    """``[submit-a, submit-b]``: first submit's answer is used, second
    is dropped. Deterministic choice (parallel submits are model
    confusion, not a legitimate use case).
    """
    agent, env, model = _make_agent()
    with pytest.raises(Submitted) as excinfo:
        await agent.execute_actions(
            _msg(
                [
                    _submit("first-answer", tc_id="tc-a"),
                    _submit("second-answer", tc_id="tc-b"),
                ]
            )
        )
    # No bash, no env calls, no observation.
    assert env.calls == []
    assert model.format_calls == []
    # First submit's payload wins.
    assert excinfo.value.messages[0]["content"] == "first-answer"
    assert excinfo.value.messages[0]["extra"]["tool_call_id"] == "tc-a"


# --------------------------------------------------------------------------
# Unknown tool_name still hard-raises (preserves existing contract)
# --------------------------------------------------------------------------


async def test_unknown_tool_name_raises_value_error():
    agent, env, model = _make_agent()
    with pytest.raises(ValueError, match="unknown tool_name"):
        await agent.execute_actions(
            _msg([{"tool_name": "curl", "url": "http://evil.example/"}])
        )
    # Env was not touched (curl isn't an allowed tool).
    assert env.calls == []


# --------------------------------------------------------------------------
# Round-4 Bug 1: observation-message alignment with the REAL formatter
# --------------------------------------------------------------------------
# _FakeModel above stubs format_observation_messages so it always returns
# one message regardless of how many actions/outputs come in. That's fine
# for the round-3 dispatch-order tests, but it hides the round-4 bug:
# the REAL mini helper zips actions against outputs, so passing the
# UNFILTERED action list produces misaligned tool_call_id -> output
# pairings on the trajectory. These tests wire a REAL
# SamplerChatCompletionModel (whose format_observation_messages calls
# mini's real format_toolcall_observation_messages) into execute_actions
# to catch that class of bug.


def _make_agent_with_real_model():
    """Instantiate an AsyncDefaultAgent with a real
    SamplerChatCompletionModel (so format_observation_messages invokes
    the real mini helper) but a stub sampler + fake env.
    """
    from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
        SamplerChatCompletionModel,
    )

    class _UnusedSampler:
        """Never invoked; execute_actions doesn't call the sampler."""

        async def sample_chat_completion(self, *a, **kw):
            raise AssertionError("sampler should not be invoked from execute_actions")

    env = _FakeEnv()
    model = SamplerChatCompletionModel(sampler=_UnusedSampler())
    agent = AsyncDefaultAgent(
        model=model,
        env=env,
        system_template="",
        instance_template="",
    )
    return agent, env, model


class TestObservationMessageAlignment:
    """Round-4 Bug 1: with the REAL format_toolcall_observation_messages,
    ensure the tool_call_id -> output pairing in emitted observation
    messages matches the executed (non-submit) actions 1:1, regardless
    of where the submit action appears in the original action list.
    """

    async def test_bash_only_pairing_correct(self):
        """Baseline: pure bash turn produces one observation per bash
        with matching tool_call_id."""
        agent, env, _ = _make_agent_with_real_model()
        # Snapshot msg count before execute_actions.
        before = len(agent.messages)
        await agent.execute_actions(
            _msg(
                [
                    _bash("echo a", tc_id="tc-a"),
                    _bash("echo b", tc_id="tc-b"),
                ]
            )
        )
        added = agent.messages[before:]
        assert len(added) == 2
        assert added[0]["role"] == "tool"
        assert added[0]["tool_call_id"] == "tc-a"
        assert "echo a" in added[0]["content"]
        assert added[1]["role"] == "tool"
        assert added[1]["tool_call_id"] == "tc-b"
        assert "echo b" in added[1]["content"]

    async def test_bash_then_submit_pairing_ignores_submit(self):
        """``[bash, submit]``: the ONE emitted observation message must
        carry the bash's tool_call_id and content — NOT a spurious
        second message keyed off submit's tool_call_id with "action was
        not executed" filler.

        Pre-fix: since actions=[bash, submit] and outputs=[bash_out],
        the mini helper padded to [bash_out, not_executed] and zipped:
          (bash, bash_out) -> OK
          (submit, not_executed) -> spurious role=tool msg with
              tool_call_id=<submit-id>, content="action was not executed"
        This test proves that second message no longer exists.
        """
        agent, env, _ = _make_agent_with_real_model()
        before = len(agent.messages)
        with pytest.raises(Submitted):
            await agent.execute_actions(
                _msg(
                    [
                        _bash("echo pre-submit", tc_id="tc-b"),
                        _submit("done", tc_id="tc-s"),
                    ]
                )
            )
        added = agent.messages[before:]
        assert len(added) == 1, (
            f"Expected exactly one observation message (for the bash), got "
            f"{len(added)}: {[m.get('tool_call_id') for m in added]}"
        )
        assert added[0]["role"] == "tool"
        assert added[0]["tool_call_id"] == "tc-b"
        assert "pre-submit" in added[0]["content"]
        # Explicitly verify the submit's tool_call_id did NOT leak into
        # a spurious observation message.
        assert not any(m.get("tool_call_id") == "tc-s" for m in added), (
            "Submit's tool_call_id leaked into an observation message — "
            "the format_observation_messages call was passed unfiltered "
            "actions."
        )

    async def test_submit_then_bash_pairing_not_swapped(self):
        """``[submit, bash]``: the emitted observation must carry
        bash's tool_call_id + bash's output. Pre-fix, the zip pairing
        was SWAPPED: submit's tool_call_id ended up on disk with
        bash's stdout, and bash's tool_call_id got "not executed".
        That was a data corruption bug in the trajectory JSON.
        """
        agent, env, _ = _make_agent_with_real_model()
        before = len(agent.messages)
        with pytest.raises(Submitted):
            await agent.execute_actions(
                _msg(
                    [
                        _submit("early", tc_id="tc-s"),
                        _bash("echo post-submit", tc_id="tc-b"),
                    ]
                )
            )
        added = agent.messages[before:]
        assert len(added) == 1, (
            f"Expected exactly one observation message (for the bash), got "
            f"{len(added)}: {[m.get('tool_call_id') for m in added]}"
        )
        assert added[0]["role"] == "tool"
        assert added[0]["tool_call_id"] == "tc-b", (
            f"Observation's tool_call_id should be bash's ({'tc-b'!r}), "
            f"got {added[0]['tool_call_id']!r}. Pre-fix, the zip swap "
            f"would put submit's id ({'tc-s'!r}) here."
        )
        assert "post-submit" in added[0]["content"], (
            f"Observation content should be bash's stdout; got "
            f"{added[0]['content']!r}. Pre-fix, bash's id would get "
            f"'action was not executed' content."
        )

    async def test_mixed_bash_bash_submit_pairing_correct(self):
        """``[bash1, bash2, submit]``: two observation messages, one
        per bash, each with correct tool_call_id + content. Zero
        spurious submit-keyed observations.
        """
        agent, env, _ = _make_agent_with_real_model()
        before = len(agent.messages)
        with pytest.raises(Submitted):
            await agent.execute_actions(
                _msg(
                    [
                        _bash("echo one", tc_id="tc-1"),
                        _bash("echo two", tc_id="tc-2"),
                        _submit("answer", tc_id="tc-s"),
                    ]
                )
            )
        added = agent.messages[before:]
        assert len(added) == 2
        assert [m["tool_call_id"] for m in added] == ["tc-1", "tc-2"]
        assert "one" in added[0]["content"]
        assert "two" in added[1]["content"]
        assert not any(m.get("tool_call_id") == "tc-s" for m in added)


# --------------------------------------------------------------------------
# Env-time measurement (feeds StepTimings.env_time_s via the ATIF converter)
# --------------------------------------------------------------------------


class _SlowEnv(_FakeEnv):
    """Env whose ``execute`` blocks, so the turn has real tool time."""

    DELAY_S = 0.10

    def execute(self, action: dict) -> dict:
        import time as _time

        _time.sleep(self.DELAY_S)
        return super().execute(action)


async def test_bash_turn_records_env_elapsed_seconds_on_the_assistant_message():
    """The converter reads ``env_elapsed_seconds`` off the assistant message
    to fill ``StepTimings.env_time_s``, which was a hardcoded 0.0 before
    anything measured tool execution at all."""
    env = _SlowEnv()
    model = _FakeModel()
    agent = AsyncDefaultAgent(
        model=model, env=env, system_template="", instance_template=""
    )
    message = _msg([_bash("sleep")])

    await agent.execute_actions(message)

    elapsed = message["extra"].get("env_elapsed_seconds")
    assert elapsed is not None, (
        "no env_elapsed_seconds on the assistant message; the ATIF converter "
        "has nothing to read and env_time_s stays 0.0"
    )
    assert elapsed >= _SlowEnv.DELAY_S, (
        f"measured {elapsed:.3f}s for a tool call that blocked for "
        f"{_SlowEnv.DELAY_S}s — the timer is not around the execution"
    )
    # Sanity bound: a per-turn value in the tens of seconds means the timer
    # is wrapped around the wrong thing (the rollout, say).
    assert elapsed < 5.0, f"implausible per-turn env time {elapsed:.3f}s"


async def test_parallel_bash_actions_are_not_double_counted():
    """Two bash calls in ONE turn run concurrently, so the turn's env time is
    the wall clock of the gather (~1x the delay), not their sum (~2x)."""
    env = _SlowEnv()
    model = _FakeModel()
    agent = AsyncDefaultAgent(
        model=model, env=env, system_template="", instance_template=""
    )
    message = _msg([_bash("a", "id-a"), _bash("b", "id-b")])

    await agent.execute_actions(message)

    elapsed = message["extra"]["env_elapsed_seconds"]
    assert len(env.calls) == 2
    assert elapsed < 2 * _SlowEnv.DELAY_S, (
        f"{elapsed:.3f}s for two parallel {_SlowEnv.DELAY_S}s calls looks "
        "like a sum of per-action durations rather than the gather's wall clock"
    )


async def test_submit_only_turn_records_no_env_time():
    """No tool ran, so zero env time is the truth; leave the key unset
    rather than stamping a meaningless duration."""
    agent, env, _ = _make_agent()
    message = _msg([_submit("answer")])

    with pytest.raises(Submitted):
        await agent.execute_actions(message)

    assert env.calls == []
    assert "env_elapsed_seconds" not in message["extra"]
