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

"""Retry-observation wire-shape tests through the OpenAI->Gemini converter.

Replaces the coverage lost when ``test_gemini_sampler_build_request_retry.py``
was deleted along with ``GeminiSamplerModel``. That test pinned the
retry-observation to ``functionResponse`` conversion inside the (now-
gone) ``_build_request`` method. The migration to
``SamplerChatCompletionModel`` routes through the generic
``convert_openai_request_to_gemini`` converter, which only handles
``role in {"tool", "function"}`` as ``functionResponse`` — hence the
retry observation MUST be emitted as ``role="tool"`` (not
``role="user"``, which would become a plain text Part and drift the
model out of tool-calling mode).

Bugs caught by these tests, if regressed:

  1. Retry observation is ``role="user"`` (pre-fix
     state): converter emits a plain text Part; multi-turn tool-call
     history breaks.
  2. Prior assistant message's ``tool_calls`` list is empty for the
     submit tool_call (submit-filter runs but the retry-mode fixup
     doesn't restore it): converter's ``tool_call_id_to_name`` map
     lacks the entry; the retry-obs functionResponse Part has
     ``name=""``; Vertex rejects with ``REQUIRED_FIELD_MISSING``.
  3. ``tool_call_id`` at top level of the retry-obs is missing:
     converter defaults it to ``""`` and the id/name map lookup
     misses; same ``name=""`` bug.

These tests exercise the full pipeline end-to-end: build the
assistant message via the same path the adapter uses on wrong-submit
turns (submit filtered, then restored via ``_restore_submit_tool_call``),
then convert through ``convert_openai_request_to_gemini`` and inspect
the resulting Vertex request dict.
"""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("minisweagent")

from backends.chat_completion import (  # noqa: E402
    convert_openai_request_to_gemini,
)

# Reuse the existing retry-suite's agent fixture + submit-message
# helper so this file only owns the wire-shape-specific assertions.
from tests.unit.test_async_default_agent_retry import (  # noqa: E402
    _make_agent,
    _submit_message,
)


def _assistant_message_after_submit_filter(
    *,
    reasoning: str = "The rule is Caesar shift by 2.",
    tool_call_id: str = "call_submit_1",
) -> dict:
    """Emulate ``SamplerChatCompletionModel._make_message`` output for a
    turn that consisted of exactly one submit tool_call (which the
    submit-filter strips from ``tool_calls`` — that's the point of the
    round-3 fixup restoring it).

    Returns a dict shaped the way ``self.messages[-1]`` looks right
    before the retry-mode fixup runs.
    """
    return {
        "role": "assistant",
        "content": reasoning,
        # tool_calls list is EMPTY — the submit was filtered.
        "extra": {
            "actions": [
                {
                    "tool_name": "submit",
                    "answer": "goodbye",
                    "tool_call_id": tool_call_id,
                }
            ],
        },
    }


# --------------------------------------------------------------------------
# _restore_submit_tool_call: reconstruction is lossless + idempotent
# --------------------------------------------------------------------------


def test_restore_submit_tool_call_reconstructs_openai_shape():
    """Fixup adds an OpenAI-format tool_call {id, type, function{name,arguments}}."""
    agent = _make_agent(submit_retries=3, expected_answer="hello")
    msg = _assistant_message_after_submit_filter()
    submit_action = {
        "tool_name": "submit",
        "answer": "goodbye",
        "tool_call_id": "call_submit_1",
    }
    agent._restore_submit_tool_call(msg, submit_action)
    assert msg["tool_calls"] == [
        {
            "id": "call_submit_1",
            "type": "function",
            "function": {
                "name": "submit",
                "arguments": json.dumps({"answer": "goodbye"}),
            },
        }
    ]


def test_restore_submit_tool_call_preserves_existing_bash_tool_calls():
    """If the assistant turn was [bash, submit], bash tool_call stays
    at index 0; submit gets appended at index 1. Order matters:
    Vertex's convert-back needs the ORIGINAL emission order for
    functionCall Part ordering."""
    agent = _make_agent(submit_retries=3, expected_answer="hello")
    msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_bash_1",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"command": "ls"}'},
            }
        ],
        "extra": {"actions": []},
    }
    submit_action = {
        "tool_name": "submit",
        "answer": "goodbye",
        "tool_call_id": "call_submit_1",
    }
    agent._restore_submit_tool_call(msg, submit_action)
    ids = [tc["id"] for tc in msg["tool_calls"]]
    assert ids == ["call_bash_1", "call_submit_1"]


def test_restore_submit_tool_call_is_idempotent():
    """Defensive: calling twice must NOT double-insert."""
    agent = _make_agent(submit_retries=3, expected_answer="hello")
    msg = _assistant_message_after_submit_filter()
    submit_action = {
        "tool_name": "submit",
        "answer": "x",
        "tool_call_id": "call_submit_1",
    }
    agent._restore_submit_tool_call(msg, submit_action)
    agent._restore_submit_tool_call(msg, submit_action)
    assert len(msg["tool_calls"]) == 1


# --------------------------------------------------------------------------
# End-to-end: assistant + retry-obs -> convert_openai_request_to_gemini
# --------------------------------------------------------------------------


def test_retry_observation_converts_to_function_response_part():
    """After execute_actions runs, the messages list has the correct
    wire shape for the OpenAI->Gemini converter:

      * prior assistant message has ``tool_calls=[submit]`` (fixup
        restored it).
      * retry-obs message has ``role="tool"`` and a top-level
        ``tool_call_id``.

    Converting the whole message list through
    ``convert_openai_request_to_gemini`` produces a Vertex request
    where the last Content is a ``functionResponse(name="submit")``
    Part — NOT a plain text Part.
    """
    agent = _make_agent(submit_retries=3, expected_answer="hello")
    # ``_make_agent`` already seeded messages with system+user and
    # called _init_retry_state; now append the assistant message
    # we're about to execute_actions on.
    prior_assistant = _submit_message("goodbye", tool_call_id="call_submit_1")
    agent.messages.append(prior_assistant)

    # execute_actions mutates messages: restores submit tool_call on
    # the prior assistant, appends the retry observation.
    asyncio.run(agent.execute_actions(prior_assistant))

    # Convert to Vertex wire format.
    openai_request = {
        "model": "gemini-3.5-flash",
        "messages": agent.messages,
    }
    vertex_request = convert_openai_request_to_gemini(None, openai_request)

    # Find the tool-role Content (Vertex convention: no role field on
    # tool-role Contents, they just have functionResponse Parts).
    contents = vertex_request["contents"]
    tool_contents = [
        c
        for c in contents
        if "functionResponse" in (c["parts"][0] if c["parts"] else {})
    ]
    assert len(tool_contents) == 1, (
        f"expected exactly one functionResponse Content, got {len(tool_contents)}"
    )

    fr_part = tool_contents[0]["parts"][0]["functionResponse"]
    # ``name`` MUST be "submit" — the whole point of the fixup +
    # role="tool" change. Empty name = Vertex REQUIRED_FIELD_MISSING.
    assert fr_part["name"] == "submit", (
        f"functionResponse.name must be 'submit', got {fr_part['name']!r} — "
        "the id->name map lookup failed, meaning either the retry-obs is "
        "missing top-level tool_call_id or the prior assistant message "
        "is missing the paired submit tool_call."
    )
    # tool_call_id round-trips.
    assert fr_part["id"] == "call_submit_1"
    # Payload arrives as a Struct-like dict.
    response_payload = fr_part["response"]
    assert response_payload["status"] == "wrong_answer"
    assert response_payload["wrong_answer"] == "goodbye"


def test_prior_assistant_functionCall_pairs_with_retry_functionResponse():
    """Wire-history invariant: on the retry turn, the prior assistant
    message MUST carry a ``functionCall(submit)`` Part so it pairs
    1:1 with the retry-obs ``functionResponse(submit)`` Part. Vertex
    rejects unpaired functionResponse Parts."""
    agent = _make_agent(submit_retries=3, expected_answer="hello")
    prior_assistant = _submit_message("goodbye", tool_call_id="call_submit_1")
    agent.messages.append(prior_assistant)
    asyncio.run(agent.execute_actions(prior_assistant))

    openai_request = {"model": "gemini-3.5-flash", "messages": agent.messages}
    vertex_request = convert_openai_request_to_gemini(None, openai_request)
    contents = vertex_request["contents"]

    # Walk the Contents: the one before the tool-role Content must be
    # a model-role Content containing a functionCall(submit) Part.
    tool_idx = None
    for i, c in enumerate(contents):
        if c["parts"] and "functionResponse" in c["parts"][0]:
            tool_idx = i
            break
    assert tool_idx is not None and tool_idx > 0, (
        "expected a functionResponse Content preceded by an assistant Content"
    )
    prior = contents[tool_idx - 1]
    # role field: converter emits "model" for assistant (ConversionContext
    # .use_model_role defaults to True; the tuning-scope endpoint rejects
    # "assistant").
    assert prior.get("role") == "model", (
        f"prior Content role must be 'model', got {prior.get('role')!r}"
    )
    fc_parts = [p for p in prior["parts"] if "functionCall" in p]
    assert len(fc_parts) == 1, (
        f"expected exactly one functionCall Part in the prior assistant "
        f"Content, got {len(fc_parts)}: {prior['parts']}"
    )
    assert fc_parts[0]["functionCall"]["name"] == "submit"


def test_retry_obs_never_becomes_plain_text_part():
    """Regression: pre-fix, role='user' retry-obs became a plain text
    Part on the wire. Even if the retry-obs happens to convert to
    something with functionResponse elsewhere, there must NEVER be a
    text Part containing the retry payload JSON — that would mean the
    conversion is doubling up and the model sees the retry payload
    both as tool result AND as user text."""
    agent = _make_agent(submit_retries=3, expected_answer="hello")
    prior_assistant = _submit_message("goodbye", tool_call_id="call_submit_1")
    agent.messages.append(prior_assistant)
    asyncio.run(agent.execute_actions(prior_assistant))

    openai_request = {"model": "gemini-3.5-flash", "messages": agent.messages}
    vertex_request = convert_openai_request_to_gemini(None, openai_request)
    for content in vertex_request["contents"]:
        for part in content["parts"]:
            text_val = part.get("text", "")
            assert '"status": "wrong_answer"' not in text_val, (
                f"retry-obs payload leaked into a text Part on the wire "
                f"(should be a functionResponse instead): {text_val!r}"
            )


# --------------------------------------------------------------------------
# _restore_submit_tool_call ALSO keeps provider_specific_fields aligned
# --------------------------------------------------------------------------
#
# Regression guard: ``_restore_submit_tool_call`` mutates ``message["tool_calls"]`` from
# length N to length N+1 to add the submit back. If
# ``message["provider_specific_fields"]["thought_signatures"]`` is also
# present (emitted by ``_make_message`` when the model returned per-
# tool_call signatures), it MUST also grow from N to N+1 to stay
# length-aligned. Otherwise the next Vertex turn crashes at the
# strict length check in ``convert_openai_request_to_gemini`` line
# 1223-1232 with ``ValueError: provider_specific_fields.
# thought_signatures length (N) does not match tool_calls length (N+1)``.
#
# Only bites under ``--submit-retries >= 2`` AND the model returned
# thought signatures. Latent under the default ``--submit-retries=1``
# path.


def test_restore_submit_tool_call_keeps_signatures_aligned():
    """Fixup grows thought_signatures list by 1; uses submit's stashed sig.

    Signature source: ``SamplerChatCompletionModel._make_message`` stashes
    the submit's original psf.thought_signatures slot onto
    ``action["thought_signature"]`` at emit time, specifically so we
    can echo it back here on the retry path.

    Falls back to ``""`` (empty string) if not stashed (e.g. thinking
    off, or a failsafe zero-actions message that carries no signatures).
    """
    agent = _make_agent(submit_retries=3, expected_answer="hello")
    msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_bash_1",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"command": "ls"}'},
            }
        ],
        "provider_specific_fields": {"thought_signatures": ["bash_sig_b64"]},
        "extra": {"actions": []},
    }
    # Submit action carries its OWN stashed thought_signature
    # (put there by _make_message when it filtered the submit
    # from tool_calls at emit time).
    submit_action = {
        "tool_name": "submit",
        "answer": "goodbye",
        "tool_call_id": "call_submit_1",
        "thought_signature": "submit_sig_b64",
    }
    agent._restore_submit_tool_call(msg, submit_action)
    # tool_calls grew by 1.
    assert len(msg["tool_calls"]) == 2
    # thought_signatures grew by 1 to match, with the SUBMIT'S OWN
    # stashed signature at the same tail index — not empty string.
    # This is the fix for the round-6 HTTP 400 crash where Vertex
    # rejected requests with unsigned submit functionCalls when
    # other functionCalls in the same conversation had signatures.
    sigs = msg["provider_specific_fields"]["thought_signatures"]
    assert sigs == ["bash_sig_b64", "submit_sig_b64"], (
        f"expected [bash_sig, submit_sig], got {sigs}"
    )


def test_restore_submit_tool_call_falls_back_to_empty_when_no_stashed_sig():
    """When submit_action has no stashed thought_signature (e.g.
    thinking off, or failsafe path), fall back to empty-string fill.

    This preserves the pre-fix behavior for cases where the model
    truly returned no signature — the empty-string fill was the
    least-bad option there (Vertex tolerates missing signatures IF
    no other functionCall in the conversation has one)."""
    agent = _make_agent(submit_retries=3, expected_answer="hello")
    msg = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_bash_1",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"command": "ls"}'},
            }
        ],
        "provider_specific_fields": {"thought_signatures": [""]},
        "extra": {"actions": []},
    }
    # No thought_signature on submit_action.
    submit_action = {
        "tool_name": "submit",
        "answer": "goodbye",
        "tool_call_id": "call_submit_1",
    }
    agent._restore_submit_tool_call(msg, submit_action)
    sigs = msg["provider_specific_fields"]["thought_signatures"]
    assert sigs == ["", ""], f"expected fallback ['', ''], got {sigs}"


def test_restore_submit_tool_call_creates_psf_when_absent_and_sig_stashed():
    """Solo-submit case: assistant emitted [submit] only, no bash.

    ``_make_message`` filtered submit from tool_calls (leaving empty),
    so the ``if filtered_tool_calls:`` gate didn't attach a psf envelope
    on the message. The submit's signature was still stashed onto
    ``action["thought_signature"]`` though.

    When _restore_submit_tool_call runs, ``psf`` is absent — but we
    must reconstruct it so the restored submit's signature reaches the
    wire. Without this fix, Vertex rejects with the ``missing
    thought_signature`` HTTP 400 (observed empirically in the first
    3-retry eval run WITH the sig-stash but BEFORE this reconstruction
    case was handled: 6 crashes out of 31 tasks).
    """
    agent = _make_agent(submit_retries=3, expected_answer="hello")
    # No tool_calls, no psf on the assistant message (matches what
    # _make_message emits for a solo-submit turn after the filter).
    msg = {
        "role": "assistant",
        "content": "I'll submit my guess.",
        "extra": {
            "actions": [
                {
                    "tool_name": "submit",
                    "answer": "goodbye",
                    "tool_call_id": "call_submit_1",
                    "thought_signature": "submit_sig_b64",
                }
            ]
        },
    }
    submit_action = msg["extra"]["actions"][0]
    agent._restore_submit_tool_call(msg, submit_action)
    # tool_calls now has the restored submit.
    assert len(msg["tool_calls"]) == 1
    # psf envelope was CREATED with the stashed signature aligned to
    # the (now non-empty) tool_calls list.
    psf = msg.get("provider_specific_fields")
    assert psf == {"thought_signatures": ["submit_sig_b64"]}, (
        f"expected psf reconstructed with submit sig, got {psf}"
    )


def test_restore_submit_tool_call_no_psf_reconstruction_when_no_sig():
    """Solo-submit case with NO stashed sig (thinking off): psf stays
    absent. Vertex tolerates all-signatures-absent when nothing else
    in the conversation was signed."""
    agent = _make_agent(submit_retries=3, expected_answer="hello")
    msg = {
        "role": "assistant",
        "content": "",
        "extra": {
            "actions": [
                {
                    "tool_name": "submit",
                    "answer": "x",
                    "tool_call_id": "call_submit_1",
                    # No thought_signature stashed.
                }
            ]
        },
    }
    submit_action = msg["extra"]["actions"][0]
    agent._restore_submit_tool_call(msg, submit_action)
    assert "provider_specific_fields" not in msg
