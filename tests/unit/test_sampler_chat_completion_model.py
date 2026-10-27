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

"""Contract tests for SamplerChatCompletionModel.

Uses a _FakeSampler subclass whose sample_chat_completion returns
caller-supplied ChatCompletionSampleResults, so tests drive the
model's parsing + extras plumbing without any HTTP.
"""

from __future__ import annotations

import json
import time

import pytest

# The sampler_chat_completion_model adapter (and every test below) imports
# minisweagent (mini-swe-agent). It's an optional extra (see the
# ``agent_solver`` extra in pyproject.toml) so callers who only care about
# RL training don't pay the litellm/textual/hf-datasets transitive cost.
# When it's not installed, skip the module cleanly instead of failing at
# collection time with an obscure ImportError.
pytest.importorskip("minisweagent")

from google.cloud.aiplatform_v1.types import Candidate, Content, Part

from core.protocols import Sampler
from core.sample_result import (
    ChatCompletionSampleResult,
    SampleResult,
    SamplingContext,
    TuningCandidate,
)


def _make_candidate_proto(text: str = "hi") -> Candidate:
    return Candidate(
        content=Content(role="model", parts=[Part(text=text)]),
        finish_reason=Candidate.FinishReason.STOP,
    )


def _make_sample_result(
    *,
    candidate_id: str = "cand-uuid-0",
    text: str = "hi",
    gen_tokens: int = 5,
    prompt_tokens: int | None = 17,
    response_id: str | None = "resp-abc",
    operation_id: str | None = None,
    model_version: str | None = "gemini-3.5-flash",
    thoughts_tokens: int = 0,
    thinking_blocks: tuple | None = None,
) -> SampleResult:
    if candidate_id == "":
        candidate = Candidate()  # empty; failsafe sentinel
    else:
        candidate = _make_candidate_proto(text)
    return SampleResult(
        tuning_candidate=TuningCandidate(
            candidate_id=candidate_id,
            candidate=candidate,
            generation_token_count=gen_tokens,
            thoughts_token_count=thoughts_tokens,
        ),
        context=SamplingContext(
            prompt="hello",
            response_id=response_id,
            operation_id=operation_id,
            model_version=model_version,
            prompt_token_count=prompt_tokens,
            thinking_blocks=thinking_blocks,
        ),
        train_step_id=None,
    )


def _make_chat_completion_ret(
    *,
    tool_calls: list[dict] | None = None,
    content: str | None = None,
    finish_reason: str = "stop",
    sample_result: SampleResult | None = None,
    thought_signatures: list[str] | None = None,
) -> ChatCompletionSampleResult:
    """Build a ChatCompletionSampleResult with caller-controlled shape.

    Uses backends.chat_completion dataclasses directly rather than a
    real converter roundtrip so tests are hermetic and fast.

    ``thought_signatures``: if provided, attach a
    ``ProviderSpecificFields(thought_signatures=[...])`` on the
    Message (aligned 1:1 with ``tool_calls``). Used for tests
    that need to exercise the psf-preservation / signature-
    stashing logic in ``_make_message``.
    """
    from backends.chat_completion import (
        ChatCompletionResponse,
        Choice,
        Message,
        ProviderSpecificFields,
        Usage,
    )

    psf = (
        ProviderSpecificFields(thought_signatures=list(thought_signatures))
        if thought_signatures is not None
        else None
    )
    msg = Message(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        provider_specific_fields=psf,
    )
    choice = Choice(index=0, message=msg, finish_reason=finish_reason)
    usage = Usage(prompt_tokens=17, completion_tokens=5, total_tokens=22)
    response = ChatCompletionResponse(
        id="resp-abc",
        object="chat.completion",
        created=int(time.time()),
        model="gemini-3.5-flash",
        choices=[choice],
        usage=usage,
    )
    if sample_result is None:
        sample_result = _make_sample_result()
    return ChatCompletionSampleResult(response=response, results=[sample_result])


class _FakeSampler(Sampler):
    """Records the last sample_chat_completion call and returns
    caller-supplied ChatCompletionSampleResults."""

    def __init__(self, returns: ChatCompletionSampleResult | Exception):
        self._returns = returns
        self.last_request: dict | None = None
        self.last_num_samples: int | None = None
        self.last_thinking_level: str | None = None
        # Everything absorbed by **kwargs, so forwarding of ABC kwargs this
        # fake doesn't name explicitly can still be asserted (see
        # test_capture_ctx_forwarded_*). Without this, **kwargs would make
        # a dropped kwarg silently untestable.
        self.last_kwargs: dict = {}

    async def sample_n(self, prompt, num_samples=1, **kwargs):
        raise NotImplementedError("this fake exposes sample_chat_completion only")

    async def sample_chat_completion(
        self, openai_request, num_samples=1, thinking_level=None, **kwargs
    ):
        self.last_request = openai_request
        self.last_num_samples = num_samples
        self.last_thinking_level = thinking_level
        self.last_kwargs = kwargs
        if isinstance(self._returns, Exception):
            raise self._returns
        return self._returns


OPENAI_MESSAGES = [
    {"role": "system", "content": "You are a puzzle solver."},
    {"role": "user", "content": "Solve: 2+2=?"},
]


def _bash_tool_call(command: str, tc_id: str = "call_abc") -> dict:
    return {
        "id": tc_id,
        "type": "function",
        "function": {"name": "bash", "arguments": json.dumps({"command": command})},
    }


def _submit_tool_call(answer: str, tc_id: str = "call_sub") -> dict:
    return {
        "id": tc_id,
        "type": "function",
        "function": {"name": "submit", "arguments": json.dumps({"answer": answer})},
    }


async def test_query_happy_path_bash_tool_call():
    from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
        SamplerChatCompletionModel,
    )

    sampler = _FakeSampler(
        _make_chat_completion_ret(
            tool_calls=[_bash_tool_call("ls -la")],
            finish_reason="tool_calls",
        )
    )
    model = SamplerChatCompletionModel(sampler=sampler)
    message = await model.query(OPENAI_MESSAGES)
    assert message["role"] == "assistant"
    actions = message["extra"]["actions"]
    assert len(actions) == 1
    assert actions[0]["tool_name"] == "bash"
    assert actions[0]["command"] == "ls -la"


async def test_query_carries_thinking_blocks_on_assistant_message():
    # Interleaved thinking: a sample result carrying opaque thinking_blocks must
    # surface them on the assistant message (top-level, not in `extra`) so a
    # backend that replays them across the tool loop can find them on the next
    # request. Must be present on tool-call turns too (thinking precedes
    # tool_use for Anthropic).
    from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
        SamplerChatCompletionModel,
    )

    tblocks = (
        {"type": "thinking", "thinking": "let me compute", "signature": "sig-xyz"},
    )
    sr = _make_sample_result(thinking_blocks=tblocks)
    sampler = _FakeSampler(
        _make_chat_completion_ret(
            tool_calls=[_bash_tool_call("ls")],
            finish_reason="tool_calls",
            sample_result=sr,
        )
    )
    model = SamplerChatCompletionModel(sampler=sampler)
    message = await model.query(OPENAI_MESSAGES)
    assert message.get("thinking_blocks") == [
        {"type": "thinking", "thinking": "let me compute", "signature": "sig-xyz"}
    ]
    # and it survives the request-shaping (only `extra` is stripped).
    stripped = SamplerChatCompletionModel._strip_extras([message])[0]
    assert "thinking_blocks" in stripped and "extra" not in stripped


async def test_query_no_thinking_blocks_absent_by_default():
    # Providers that don't emit thinking blocks (e.g. Gemini) leave the key
    # absent -> no-op, no empty key polluting the message.
    from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
        SamplerChatCompletionModel,
    )

    sampler = _FakeSampler(
        _make_chat_completion_ret(
            tool_calls=[_bash_tool_call("ls")],
            finish_reason="tool_calls",
            sample_result=_make_sample_result(),
        )
    )
    model = SamplerChatCompletionModel(sampler=sampler)
    message = await model.query(OPENAI_MESSAGES)
    assert "thinking_blocks" not in message


async def test_query_happy_path_submit_tool_call():
    from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
        SamplerChatCompletionModel,
    )

    sampler = _FakeSampler(
        _make_chat_completion_ret(
            tool_calls=[_submit_tool_call("42")],
            finish_reason="tool_calls",
        )
    )
    model = SamplerChatCompletionModel(sampler=sampler)
    message = await model.query(OPENAI_MESSAGES)
    actions = message["extra"]["actions"]
    assert len(actions) == 1
    assert actions[0]["tool_name"] == "submit"
    assert actions[0]["answer"] == "42"


async def test_query_no_tool_call_raises_format_error():
    """Response has text but no tool_calls -> FormatError."""
    from minisweagent.exceptions import FormatError
    from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
        SamplerChatCompletionModel,
    )

    sampler = _FakeSampler(
        _make_chat_completion_ret(
            tool_calls=None,
            content="I refuse to call a tool. Here is my reasoning: ...",
            finish_reason="stop",
        )
    )
    model = SamplerChatCompletionModel(sampler=sampler)
    with pytest.raises(FormatError) as excinfo:
        await model.query(OPENAI_MESSAGES)
    err_msg = excinfo.value.messages[0]
    assert "extra" in err_msg
    assert err_msg["extra"].get("candidate_id") == "cand-uuid-0"
    assert err_msg["extra"].get("response_id") == "resp-abc"


async def test_query_unknown_tool_raises_format_error():
    """Response has a tool_call for a tool we didn't declare -> FormatError."""
    from minisweagent.exceptions import FormatError
    from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
        SamplerChatCompletionModel,
    )

    unknown = {
        "id": "call_x",
        "type": "function",
        "function": {"name": "not_a_real_tool", "arguments": "{}"},
    }
    sampler = _FakeSampler(
        _make_chat_completion_ret(
            tool_calls=[unknown],
            finish_reason="tool_calls",
        )
    )
    model = SamplerChatCompletionModel(sampler=sampler)
    with pytest.raises(FormatError):
        await model.query(OPENAI_MESSAGES)


async def test_query_swallows_empty_parts_error_into_failsafe_message():
    """``EmptyCandidatePartsError`` from the converter (Vertex returned
    a candidate with ``parts=[]`` and non-content-filter finish_reason)
    is caught by ``query()`` and converted to a zero-actions message.

    Rationale: this is an observed Vertex-side flake (~13% of long-
    running rollouts against real Vertex during a baseline eval).
    The pre-migration ``GeminiSamplerModel._parse_response`` had an
    equivalent failsafe: turned the failure into a zero-actions
    message so the rollout ``LimitsExceeded``-out with a partial
    trajectory instead of ``Crash``-ing outright. This test pins the
    failsafe behavior after migration.

    Contract: catches the SPECIFIC ``EmptyCandidatePartsError``
    subclass; does NOT catch plain ``ValueError`` (which the sampler
    layer uses to signal malformed-input bugs the operator needs to
    see loudly).
    """
    from backends.chat_completion import EmptyCandidatePartsError
    from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
        SamplerChatCompletionModel,
    )

    exc = EmptyCandidatePartsError("Vertex candidate 0 has empty parts")
    sampler = _FakeSampler(exc)
    model = SamplerChatCompletionModel(sampler=sampler)
    message = await model.query(OPENAI_MESSAGES)

    # Zero-actions failsafe shape.
    assert message["role"] == "assistant"
    assert message["extra"]["actions"] == []
    # Content mentions the underlying error for observability.
    assert "empty-parts" in message["content"]
    assert "Vertex candidate 0 has empty parts" in message["content"]
    # finish_reason is "OTHER" (matches old GeminiSamplerModel's failsafe).
    assert message["extra"]["finish_reason"] == "OTHER"
    # elapsed is populated (t0 accumulator still ran).
    assert message["extra"]["elapsed_seconds"] >= 0.0
    # sample_result is None on the failsafe path (no candidate to
    # extract candidate_id / response_id from) — _sample_result_extras
    # returns the None-fill dict for downstream trace-viewer readers.
    assert message["extra"].get("candidate_id") is None
    assert message["extra"].get("response_id") is None


async def test_query_swallows_empty_parts_error_accumulates_elapsed():
    """The failsafe branch still updates ``_total_sample_seconds`` so
    per-task latency reporting stays accurate."""
    from backends.chat_completion import EmptyCandidatePartsError
    from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
        SamplerChatCompletionModel,
    )

    sampler = _FakeSampler(
        EmptyCandidatePartsError("Vertex candidate 0 has empty parts")
    )
    model = SamplerChatCompletionModel(sampler=sampler)
    t_before = model._total_sample_seconds
    await model.query(OPENAI_MESSAGES)
    assert model._total_sample_seconds > t_before, (
        "elapsed was not accumulated on the failsafe path — "
        "extract_answer's per-task latency will be under-counted."
    )


async def test_query_does_not_swallow_plain_valueerror():
    """Plain ``ValueError`` (e.g. from converter input validation)
    MUST still surface — only the ``EmptyCandidatePartsError``
    subclass is caught.

    Sanity: catching plain ``ValueError`` would swallow malformed-input
    bugs the sampler layer uses to signal contract violations
    (documented on ``Sampler.sample_chat_completion``). Verifies the
    subclass filter is narrow enough.
    """
    from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
        SamplerChatCompletionModel,
    )

    sampler = _FakeSampler(ValueError("openai_request is missing 'messages'"))
    model = SamplerChatCompletionModel(sampler=sampler)
    with pytest.raises(ValueError, match="missing 'messages'"):
        await model.query(OPENAI_MESSAGES)


async def test_query_propagates_runtime_error_from_sampler_failsafe():
    """Backend failsafe (empty candidates, ``candidate_id == ""``, or
    empty results list) MUST surface as an exception, not a synthetic
    "no candidates" message.

    Contract per ``core/protocols.py:Sampler.sample_chat_completion``:
    the base Template Method raises ``RuntimeError`` on
    ``candidate_id == ""`` and ``ValueError`` (from
    ``reconstitute_generate_content_response``) on ``results == []``.
    The adapter deliberately does NOT catch either -- suppressing them
    with a synthetic message would mask genuine HTTP / LRO errors that
    operators need to see in sampler logs.

    This test verifies the adapter's ``query()`` does not intercept the
    exception. It uses a concrete Sampler subclass (not the _FakeSampler
    that overrides sample_chat_completion) so the base class's failsafe
    guard actually runs.
    """
    from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
        SamplerChatCompletionModel,
    )

    class _FailsafeSampler(Sampler):
        async def sample_n(self, prompt, num_samples=1, **kwargs):
            # Return a failsafe empty result -- the base class Template
            # Method (Sampler.sample_chat_completion) is expected to
            # detect this and raise RuntimeError before the adapter
            # sees the wrapped result.
            return [_make_sample_result(candidate_id="")]

    sampler = _FailsafeSampler()
    model = SamplerChatCompletionModel(sampler=sampler)
    with pytest.raises(RuntimeError, match="failsafe empty"):
        await model.query(OPENAI_MESSAGES)


async def test_extras_populated_from_sample_result():
    from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
        SamplerChatCompletionModel,
    )

    sampler = _FakeSampler(
        _make_chat_completion_ret(
            tool_calls=[_bash_tool_call("echo hi")],
            finish_reason="tool_calls",
            sample_result=_make_sample_result(
                candidate_id="c-42",
                gen_tokens=7,
                prompt_tokens=99,
                response_id="resp-xyz",
                operation_id="op-123",
                model_version="gemini-3.5-flash-thinking",
                thoughts_tokens=15,
            ),
        )
    )
    model = SamplerChatCompletionModel(sampler=sampler)
    message = await model.query(OPENAI_MESSAGES)
    extras = message["extra"]
    assert extras["candidate_id"] == "c-42"
    assert extras["response_id"] == "resp-xyz"
    assert extras["operation_id"] == "op-123"
    assert extras["model_version"] == "gemini-3.5-flash-thinking"
    assert extras["prompt_token_count"] == 99
    assert extras["generation_token_count"] == 7
    assert extras["thoughts_token_count"] == 15
    # The candidate built by _make_candidate_proto has no Part.thought_signature,
    # so the adapter emits [] -- proves the field is plumbed through, while the
    # actual b64 encoding is exercised by test_extract_thought_signatures_b64_with_signature.
    assert extras["thought_signatures_b64"] == []
    assert extras["finish_reason"] == "tool_calls"
    assert "timestamp" in extras
    assert "elapsed_seconds" in extras
    assert extras.get("cost", 0.0) == 0.0


async def test_thinking_level_forwarded():
    from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
        SamplerChatCompletionModel,
    )

    sampler = _FakeSampler(
        _make_chat_completion_ret(
            tool_calls=[_bash_tool_call("ls")],
            finish_reason="tool_calls",
        )
    )
    model = SamplerChatCompletionModel(sampler=sampler, thinking_level="high")
    await model.query(OPENAI_MESSAGES)
    assert sampler.last_thinking_level == "high"


async def test_capture_ctx_forwarded_as_single_element_list():
    """A configured capture_ctx must reach sample_chat_completion.

    This assertion is the reason ``_FakeSampler`` records ``last_kwargs``.
    ``capture_ctxs`` was added to the Sampler ABC and to the adapter's
    call site, but the fake here still declared the old signature — every
    test in this file died with TypeError, and nothing covered the
    forwarding itself. **kwargs stops the TypeError; this stops the
    forwarding from silently disappearing.
    """
    from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
        SamplerChatCompletionModel,
    )

    ctx = object()
    sampler = _FakeSampler(
        _make_chat_completion_ret(
            tool_calls=[_bash_tool_call("ls")],
            finish_reason="tool_calls",
        )
    )
    model = SamplerChatCompletionModel(sampler=sampler, capture_ctx=ctx)
    await model.query(OPENAI_MESSAGES)
    assert sampler.last_kwargs["capture_ctxs"] == [ctx]


async def test_capture_ctxs_none_when_unset():
    """No capture_ctx configured -> None, not [None] (a present-but-empty
    list would read as 'capture requested' downstream)."""
    from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
        SamplerChatCompletionModel,
    )

    sampler = _FakeSampler(
        _make_chat_completion_ret(
            tool_calls=[_bash_tool_call("ls")],
            finish_reason="tool_calls",
        )
    )
    model = SamplerChatCompletionModel(sampler=sampler)
    await model.query(OPENAI_MESSAGES)
    assert sampler.last_kwargs["capture_ctxs"] is None


async def test_message_history_passed_through_verbatim():
    """The mini messages list must reach the sampler unchanged (modulo
    stripped 'extra' keys). Proves we're NOT rebuilding history proto-side."""
    from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
        SamplerChatCompletionModel,
    )

    sampler = _FakeSampler(
        _make_chat_completion_ret(
            tool_calls=[_bash_tool_call("ls")],
            finish_reason="tool_calls",
        )
    )
    model = SamplerChatCompletionModel(sampler=sampler)
    multi_turn = [
        {"role": "user", "content": "start"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [_bash_tool_call("pwd", tc_id="call_1")],
            "extra": {"actions": [{"tool_name": "bash", "command": "pwd"}]},
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "/tmp\n"},
        {"role": "user", "content": "now what"},
    ]
    await model.query(multi_turn)
    sent = sampler.last_request["messages"]
    assert len(sent) == 4
    assert sent[0] == {"role": "user", "content": "start"}
    # 'extra' key must be stripped (mini's own convention).
    for m in sent:
        assert "extra" not in m


def test_serialize_shape():
    """mini v2 .serialize() contract check."""
    from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
        SamplerChatCompletionModel,
    )

    sampler = _FakeSampler(_make_chat_completion_ret())
    model = SamplerChatCompletionModel(sampler=sampler, model_name="gemini-3.5-flash")
    serialized = model.serialize()
    assert "info" in serialized
    assert "config" in serialized["info"]
    assert serialized["info"]["config"]["model"]["model_name"] == "gemini-3.5-flash"
    assert "SamplerChatCompletionModel" in serialized["info"]["config"]["model_type"]
    # total_sample_seconds is present on a fresh model (0.0). Accumulation
    # behavior is pinned by test_query_accumulates_total_sample_seconds.
    assert serialized["info"]["config"]["total_sample_seconds"] == 0.0


async def test_query_accumulates_total_sample_seconds():
    """query() must add its sampler-call elapsed time onto
    self._total_sample_seconds on every call, and serialize() must emit
    the running total under info.config.total_sample_seconds.

    Pins the contract that extract_answer.build_task_result relies on
    (extract_answer.py:99 reads
    data['info']['config']['total_sample_seconds']). If someone drops the
    accumulator or the serialize key in the future, this test fails and
    the summary-CSV t_llm regression is caught before it ships.
    """
    from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
        SamplerChatCompletionModel,
    )

    sampler = _FakeSampler(
        _make_chat_completion_ret(
            tool_calls=[_bash_tool_call("ls")],
            finish_reason="tool_calls",
        )
    )
    model = SamplerChatCompletionModel(sampler=sampler)
    # Fresh model: counter starts at 0.0.
    assert model._total_sample_seconds == 0.0
    assert model.serialize()["info"]["config"]["total_sample_seconds"] == 0.0

    await model.query(OPENAI_MESSAGES)
    after_one = model._total_sample_seconds
    # perf_counter delta over a real await is strictly positive, even
    # for a trivial fake -- the coroutine switch alone takes >0 seconds.
    assert after_one > 0.0

    await model.query(OPENAI_MESSAGES)
    after_two = model._total_sample_seconds
    # Second call must ADD to the running total, not replace it.
    assert after_two > after_one

    # serialize() reports the current cumulative value under the exact
    # nesting extract_answer.py:99 reads (info.config.total_sample_seconds).
    serialized = model.serialize()
    assert serialized["info"]["config"]["total_sample_seconds"] == after_two


def test_format_message_pass_through():
    from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
        SamplerChatCompletionModel,
    )

    sampler = _FakeSampler(_make_chat_completion_ret())
    model = SamplerChatCompletionModel(sampler=sampler)
    m = model.format_message(role="user", content="hello")
    assert m == {"role": "user", "content": "hello"}


# ------------------------------------------------------------------
# Direct tests for the custom parser + thought_signature extractor.
# These live at module level so we can exercise the parser's error
# branches without going through _FakeSampler.
# ------------------------------------------------------------------


def test_parse_tool_calls_dict_shaped_tc():
    """The custom parser must handle dict-shaped tool_calls (the shape
    the Fake sampler emits and the shape convert_gemini_response_to_openai
    produces after .to_dict())."""
    from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
        _parse_tool_calls,
    )

    tcs = [_bash_tool_call("echo hi", tc_id="call_1")]
    actions = _parse_tool_calls(
        tcs, format_error_template="{{ error }}", finish_reason="tool_calls"
    )
    assert actions == [
        {"tool_name": "bash", "command": "echo hi", "tool_call_id": "call_1"}
    ]


def test_parse_tool_calls_attr_shaped_tc():
    """The custom parser must ALSO handle attribute-shaped tool_calls
    (some real backends' ChatCompletionResponse.choices[0].message.tool_calls
    returns dataclass instances rather than dicts)."""
    from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
        _parse_tool_calls,
    )

    class _AttrTc:
        def __init__(self, tc_id, name, args_json):
            self.id = tc_id
            self.type = "function"
            self.function = type("F", (), {"name": name, "arguments": args_json})()

    tc = _AttrTc("call_a", "bash", json.dumps({"command": "ls"}))
    actions = _parse_tool_calls(
        [tc], format_error_template="{{ error }}", finish_reason="tool_calls"
    )
    assert actions == [{"tool_name": "bash", "command": "ls", "tool_call_id": "call_a"}]


def test_parse_tool_calls_malformed_json_args_raises():
    from minisweagent.exceptions import FormatError
    from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
        _parse_tool_calls,
    )

    bad = {
        "id": "call_x",
        "type": "function",
        "function": {"name": "bash", "arguments": "{not valid json"},
    }
    with pytest.raises(FormatError) as excinfo:
        _parse_tool_calls(
            [bad], format_error_template="{{ error }}", finish_reason="tool_calls"
        )
    # Error message mentions the parse failure so the model can course-correct.
    assert "json" in excinfo.value.messages[0]["content"].lower() or (
        "argument" in excinfo.value.messages[0]["content"].lower()
    )


def test_parse_tool_calls_non_dict_args_raises():
    """Args JSON parses but isn't an object (e.g. a list) -> FormatError."""
    from minisweagent.exceptions import FormatError
    from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
        _parse_tool_calls,
    )

    bad = {
        "id": "call_x",
        "type": "function",
        "function": {"name": "bash", "arguments": "[1, 2, 3]"},
    }
    with pytest.raises(FormatError):
        _parse_tool_calls(
            [bad], format_error_template="{{ error }}", finish_reason="tool_calls"
        )


def test_parse_tool_calls_bash_missing_command_raises():
    from minisweagent.exceptions import FormatError
    from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
        _parse_tool_calls,
    )

    bad = {
        "id": "call_x",
        "type": "function",
        "function": {"name": "bash", "arguments": json.dumps({"wrong_key": "ls"})},
    }
    with pytest.raises(FormatError) as excinfo:
        _parse_tool_calls(
            [bad], format_error_template="{{ error }}", finish_reason="tool_calls"
        )
    assert "command" in excinfo.value.messages[0]["content"].lower()


def test_parse_tool_calls_submit_missing_answer_raises():
    from minisweagent.exceptions import FormatError
    from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
        _parse_tool_calls,
    )

    bad = {
        "id": "call_x",
        "type": "function",
        "function": {"name": "submit", "arguments": json.dumps({"wrong_key": "42"})},
    }
    with pytest.raises(FormatError) as excinfo:
        _parse_tool_calls(
            [bad], format_error_template="{{ error }}", finish_reason="tool_calls"
        )
    assert "answer" in excinfo.value.messages[0]["content"].lower()


def test_parse_tool_calls_empty_id_raises():
    """Missing / empty tool_call id must raise FormatError.

    Regression: a passing tool_call with id == '' would produce an
    assistant message with tool_calls=[{"id": "", ...}], and the
    next-turn tool observation would key tool_call_id_to_name with '',
    resulting in Vertex REQUIRED_FIELD_MISSING on functionResponse.name.
    Better to fail fast so the model can retry via FormatError feedback.
    """
    from minisweagent.exceptions import FormatError
    from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
        _parse_tool_calls,
    )

    for bad_id in ("", None):
        bad = {
            "id": bad_id,
            "type": "function",
            "function": {"name": "bash", "arguments": json.dumps({"command": "ls"})},
        }
        with pytest.raises(FormatError) as excinfo:
            _parse_tool_calls(
                [bad], format_error_template="{{ error }}", finish_reason="tool_calls"
            )
        assert "id" in excinfo.value.messages[0]["content"].lower()


def test_extract_thought_signatures_b64_empty_candidate():
    from google.cloud.aiplatform_v1.types import Candidate
    from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
        _extract_thought_signatures_b64,
    )

    assert _extract_thought_signatures_b64(Candidate()) == []


def test_extract_thought_signatures_b64_no_signature():
    from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
        _extract_thought_signatures_b64,
    )

    candidate = _make_candidate_proto(text="just text, no signature")
    assert _extract_thought_signatures_b64(candidate) == []


def test_extract_thought_signatures_b64_with_signature():
    import base64 as _b64
    from google.cloud.aiplatform_v1.types import Candidate, Content, FunctionCall, Part
    from google.protobuf.struct_pb2 import Struct
    from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
        _extract_thought_signatures_b64,
    )

    raw = b"\x00\x01thought-sig-bytes\xff"
    args = Struct()
    args.update({"command": "ls"})
    part = Part(
        function_call=FunctionCall(name="bash", args=args),
        thought_signature=raw,
    )
    candidate = Candidate(
        content=Content(role="model", parts=[part]),
        finish_reason=Candidate.FinishReason.STOP,
    )
    assert _extract_thought_signatures_b64(candidate) == [
        _b64.b64encode(raw).decode("ascii")
    ]


def test_extract_thought_signatures_b64_multiple_signatures_preserved():
    """Multi-Part candidate: preserve every non-empty signature in parts-order.

    Regression against the earlier first-only behavior that silently
    dropped later signatures. Analytics carrier (extras); the
    wire-roundtrip carrier remains ``provider_specific_fields.thought_signatures``
    on the assistant message.
    """
    import base64 as _b64
    from google.cloud.aiplatform_v1.types import Candidate, Content, FunctionCall, Part
    from google.protobuf.struct_pb2 import Struct
    from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
        _extract_thought_signatures_b64,
    )

    args = Struct()
    args.update({"command": "ls"})
    sig1 = b"first-part-sig"
    sig2 = b"second-part-sig"
    # Middle part has no signature - proves the collector skips empties
    # rather than short-circuiting at the first miss.
    candidate = Candidate(
        content=Content(
            role="model",
            parts=[
                Part(text="thinking...", thought=True, thought_signature=sig1),
                Part(text="more thinking, no sig", thought=True),
                Part(
                    function_call=FunctionCall(name="bash", args=args),
                    thought_signature=sig2,
                ),
            ],
        ),
        finish_reason=Candidate.FinishReason.STOP,
    )
    assert _extract_thought_signatures_b64(candidate) == [
        _b64.b64encode(sig1).decode("ascii"),
        _b64.b64encode(sig2).decode("ascii"),
    ]


def test_provider_specific_fields_to_dict_none_returns_empty():
    """None input -> {} (baseline case)."""
    from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
        _provider_specific_fields_to_dict,
    )

    assert _provider_specific_fields_to_dict(None) == {}


def test_provider_specific_fields_to_dict_empty_list_returns_empty():
    """Empty thought_signatures list must collapse to the same {} result
    as None. Pre-fix, an empty list fell through the ``is None`` guard
    and emitted ``{"thought_signatures": []}``, which the caller (guarded
    by ``if psf_dict:``, truthy on any non-empty dict) would then attach
    to the wire message. Wire boundary is benign — the request-side
    reader ignores empty sig lists — but the emitted shape is
    inconsistent with the None case.

    Covers both a dataclass-shaped input (attribute access via
    ``_tc_field``) and a dict-shaped input (dict access via
    ``_tc_field``).
    """
    from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
        _provider_specific_fields_to_dict,
    )

    # Dict-shaped input.
    assert _provider_specific_fields_to_dict({"thought_signatures": []}) == {}

    # Attribute-shaped (dataclass-style) input.
    class _Psf:
        def __init__(self):
            self.thought_signatures: list[str] = []

    assert _provider_specific_fields_to_dict(_Psf()) == {}


# ----------------------------------------------------------------------
# Regression: assistant message MUST carry OpenAI-format ``tool_calls``
# so the next-turn request converter can populate its tool_call_id ->
# tool_name map. Without it, chat_completion.py:1173 falls back to
# fn_name="" and Vertex rejects with:
#   contents[...].parts[0].function_response.name: REQUIRED_FIELD_MISSING
# Caught by the first real gemini_api integration smoke run.
# ----------------------------------------------------------------------


async def test_query_assistant_message_carries_openai_tool_calls_for_next_turn():
    """The assistant message returned by query() MUST include a top-level
    ``tool_calls`` list in OpenAI shape (id + type + function.{name,
    arguments}) so a downstream ChatCompletion request converter can
    map tool_call_id -> tool_name on the follow-up turn."""
    from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
        SamplerChatCompletionModel,
    )

    sampler = _FakeSampler(
        _make_chat_completion_ret(
            tool_calls=[_bash_tool_call("echo hi", tc_id="call_abc123")],
            finish_reason="tool_calls",
        )
    )
    model = SamplerChatCompletionModel(sampler=sampler)
    message = await model.query(OPENAI_MESSAGES)

    assert "tool_calls" in message, (
        "assistant message MUST include top-level 'tool_calls' or the "
        "next-turn function_response conversion will produce empty name"
    )
    tcs = message["tool_calls"]
    assert isinstance(tcs, list) and len(tcs) == 1
    tc = tcs[0]
    assert tc["id"] == "call_abc123"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "bash"
    # arguments is a JSON-encoded string per OpenAI spec, not a dict.
    parsed_args = json.loads(tc["function"]["arguments"])
    assert parsed_args == {"command": "echo hi"}


async def test_query_assistant_message_multiturn_roundtrip_through_converter():
    """End-to-end regression for the Vertex REQUIRED_FIELD_MISSING bug:
    the assistant tool_calls emitted by query() must survive a wire
    conversion through chat_completion._convert_openai_request_to_gemini
    such that the follow-up tool-role message's functionResponse.name is
    populated (non-empty).
    """
    from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
        SamplerChatCompletionModel,
    )
    from backends.chat_completion import (
        convert_openai_request_to_gemini,
    )

    sampler = _FakeSampler(
        _make_chat_completion_ret(
            tool_calls=[_bash_tool_call("ls", tc_id="call_multi_xyz")],
            finish_reason="tool_calls",
        )
    )
    model = SamplerChatCompletionModel(sampler=sampler)
    assistant_msg = await model.query(OPENAI_MESSAGES)

    # Build the follow-up request just like mini would on the next turn:
    # system + user + assistant(with tool_calls) + tool-role observation.
    tool_observation = {
        "role": "tool",
        "tool_call_id": "call_multi_xyz",
        "content": "hello\nworld\n",
    }
    # Strip extras just like the adapter does before sending.
    next_turn_messages = (
        OPENAI_MESSAGES
        + [{k: v for k, v in assistant_msg.items() if k != "extra"}]
        + [tool_observation]
    )
    request = {
        "model": "gemini-3.5-flash",
        "messages": next_turn_messages,
        "tools": [],
    }
    gemini_request = convert_openai_request_to_gemini(None, request)
    # Locate the functionResponse Part in the converted contents.
    fr_part = None
    for content in gemini_request.get("contents", []):
        for part in content.get("parts", []):
            if "functionResponse" in part:
                fr_part = part["functionResponse"]
                break
        if fr_part is not None:
            break
    assert fr_part is not None, "no functionResponse Part in converted request"
    assert fr_part["name"] == "bash", (
        f"functionResponse.name must be 'bash', got {fr_part['name']!r}. "
        f"Empty name causes Vertex REQUIRED_FIELD_MISSING."
    )
    assert fr_part.get("id") == "call_multi_xyz"


async def test_query_assistant_message_omits_submit_from_tool_calls():
    """When the model emits a submit tool_call, the assistant message's
    ``tool_calls`` list must NOT contain the submit entry. Submit is
    terminal (agent raises ``Submitted``, no next turn, no matching
    ``role="tool"`` observation on the trajectory), so keeping it in
    ``tool_calls`` produces an unpaired tool_call that violates the
    strict-OpenAI-format invariant "every non-terminal tool_call has a
    tool_result".

    Pre-fix behavior kept ALL tool_calls on the assistant message
    verbatim from ``raw_message.tool_calls``. That was fine for the
    live path (no next turn happens after submit) but broke any
    downstream code that (a) replays the trajectory through
    ``convert_openai_request_to_gemini`` for resume, or (b) checks
    tool_call -> tool_result pairing invariants during analytics.

    Post-fix: submit is filtered from ``tool_calls`` (its terminal
    intent is captured in the exit message's ``submission`` field
    instead). The full pre-filter action list is still preserved on
    ``extra.actions`` for observability.
    """
    from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
        SamplerChatCompletionModel,
    )

    sampler = _FakeSampler(
        _make_chat_completion_ret(
            tool_calls=[
                _bash_tool_call("echo pre", tc_id="call-bash"),
                _submit_tool_call("42", tc_id="call-submit"),
            ],
            finish_reason="tool_calls",
        )
    )
    model = SamplerChatCompletionModel(sampler=sampler)
    message = await model.query(OPENAI_MESSAGES)

    # The assistant message keeps only the bash tool_call. Submit is
    # dropped so downstream conversion doesn't see an unpaired call.
    assert "tool_calls" in message
    tool_call_ids = [tc["id"] for tc in message["tool_calls"]]
    tool_call_names = [tc["function"]["name"] for tc in message["tool_calls"]]
    assert "call-submit" not in tool_call_ids
    assert "submit" not in tool_call_names
    assert tool_call_ids == ["call-bash"]

    # Preservation invariant: the full pre-filter action list is
    # still on extra.actions for observability / analytics.
    action_tool_names = [a.get("tool_name") for a in message["extra"]["actions"]]
    assert action_tool_names == ["bash", "submit"]


async def test_query_bash_only_tool_calls_unchanged():
    """Regression: bash-only turn keeps its bash tool_call. The submit-
    filter must be scoped to submit specifically, not to "the last
    tool_call" or any other broader rule.
    """
    from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
        SamplerChatCompletionModel,
    )

    sampler = _FakeSampler(
        _make_chat_completion_ret(
            tool_calls=[
                _bash_tool_call("echo a", tc_id="call-a"),
                _bash_tool_call("echo b", tc_id="call-b"),
            ],
            finish_reason="tool_calls",
        )
    )
    model = SamplerChatCompletionModel(sampler=sampler)
    message = await model.query(OPENAI_MESSAGES)
    tool_call_ids = [tc["id"] for tc in message["tool_calls"]]
    assert tool_call_ids == ["call-a", "call-b"]


async def test_query_submit_only_tool_calls_omits_the_submit():
    """A submit-only turn: the assistant message has no bash to
    report, so ``tool_calls`` should be absent (or empty) rather than
    ``tool_calls=[submit_tc]`` (which would again be an unpaired
    call). The exit message's submission field carries the terminal
    intent.
    """
    from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
        SamplerChatCompletionModel,
    )

    sampler = _FakeSampler(
        _make_chat_completion_ret(
            tool_calls=[_submit_tool_call("done", tc_id="tc-sub-only")],
            finish_reason="tool_calls",
        )
    )
    model = SamplerChatCompletionModel(sampler=sampler)
    message = await model.query(OPENAI_MESSAGES)
    # Either absent or empty list is acceptable; both mean "no
    # non-terminal tool calls the next turn would need to respond to".
    assert not message.get("tool_calls")
    # Full action list still preserved for observability.
    action_tool_names = [a.get("tool_name") for a in message["extra"]["actions"]]
    assert action_tool_names == ["submit"]


async def test_query_multiturn_roundtrip_after_bash_and_submit_no_unpaired_tc():
    """End-to-end regression: after a ``[bash, submit]`` turn, feeding
    the assistant message + a tool-role observation for the bash into
    ``convert_openai_request_to_gemini`` must NOT trip any "tool_call
    without matching tool_result" invariant. Pre-fix, the extra submit
    tool_call would either produce a REQUIRED_FIELD_MISSING on the
    wire OR silently create a mismatched conversion.

    Verifies:
      1. The converted request contains exactly ONE functionCall Part
         on the assistant turn (bash), not two.
      2. The follow-up functionResponse Part's name is 'bash'.
    """
    from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
        SamplerChatCompletionModel,
    )
    from backends.chat_completion import (
        convert_openai_request_to_gemini,
    )

    sampler = _FakeSampler(
        _make_chat_completion_ret(
            tool_calls=[
                _bash_tool_call("ls", tc_id="call-b-rt"),
                _submit_tool_call("42", tc_id="call-s-rt"),
            ],
            finish_reason="tool_calls",
        )
    )
    model = SamplerChatCompletionModel(sampler=sampler)
    assistant_msg = await model.query(OPENAI_MESSAGES)

    tool_observation = {
        "role": "tool",
        "tool_call_id": "call-b-rt",
        "content": "file1 file2\n",
    }
    next_turn_messages = (
        OPENAI_MESSAGES
        + [{k: v for k, v in assistant_msg.items() if k != "extra"}]
        + [tool_observation]
    )
    request = {
        "model": "gemini-3.5-flash",
        "messages": next_turn_messages,
        "tools": [],
    }
    gemini_request = convert_openai_request_to_gemini(None, request)

    # Find the assistant turn's Content in the converted request.
    assistant_content = None
    for content in gemini_request.get("contents", []):
        if content.get("role") in ("assistant", "model"):
            assistant_content = content
            break
    assert assistant_content is not None
    fc_parts = [p for p in assistant_content["parts"] if "functionCall" in p]
    assert len(fc_parts) == 1, (
        f"Expected exactly ONE functionCall Part on the assistant turn "
        f"(the bash), got {len(fc_parts)}. Pre-fix, submit would have "
        f"produced a second functionCall Part with no matching "
        f"functionResponse Part on the next turn."
    )
    assert fc_parts[0]["functionCall"]["name"] == "bash"


# --------------------------------------------------------------------------
# Signature-stashing on submit actions (for retry-mode restoration)
# --------------------------------------------------------------------------
#
# ``_make_message``'s submit-filter strips the submit tool_call from the
# assistant message, which is correct for the common terminal-submit
# case. In retry mode (``--submit-retries > 1``) that submit may turn
# out to be non-terminal — ``AsyncDefaultAgent._restore_submit_tool_call``
# re-inserts it before appending the retry-obs.
#
# Vertex requires functionCall Parts to carry a ``thoughtSignature`` when
# other functionCalls in the same conversation have one — a restored
# submit without a signature triggers HTTP 400:
#
#   "Function call is missing a thought_signature in functionCall parts.
#    This is required for tools to work correctly ..."
#
# Fix: ``_make_message`` stashes each submit's aligned signature onto
# ``action["thought_signature"]`` at emit time, so the restore path can
# echo it back into the re-inserted psf.thought_signatures slot.
#
# Empirically observed: 14/150 crashes in the first ``--submit-retries 3
# --thinking-level high`` eval run (2026-07-15), all HTTP 400 on
# ``default_api:submit``. Fix verified by these tests + a follow-up
# eval run.


async def test_query_stashes_submit_thought_signature_on_action():
    """Submit's aligned psf.thought_signature is stashed onto the
    submit action so retry-mode restoration can echo it back."""
    from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
        SamplerChatCompletionModel,
    )

    sampler = _FakeSampler(
        _make_chat_completion_ret(
            tool_calls=[
                _bash_tool_call("ls", tc_id="call_bash"),
                _submit_tool_call("hello", tc_id="call_submit"),
            ],
            finish_reason="tool_calls",
            thought_signatures=["bash_sig_b64", "submit_sig_b64"],
        )
    )
    model = SamplerChatCompletionModel(sampler=sampler)
    msg = await model.query(OPENAI_MESSAGES)

    actions = msg["extra"]["actions"]
    # Both actions present pre-filter (extra.actions is the full list).
    assert [a["tool_name"] for a in actions] == ["bash", "submit"]
    submit_action = actions[1]
    # The stashed signature is the CORRECT one from the psf.thought_signatures
    # list (aligned index=1 matches submit's tool_calls index=1).
    assert submit_action.get("thought_signature") == "submit_sig_b64", (
        f"submit's thought_signature was not stashed onto the action; "
        f"retry-mode restore would fill an empty string and trigger "
        f"HTTP 400 on the next turn. Got: {submit_action!r}"
    )
    # tool_calls list has the submit FILTERED OUT (the round-6 fix); psf
    # is preserved for the remaining tool_calls (the bash).
    tc = msg.get("tool_calls") or []
    assert len(tc) == 1 and tc[0]["function"]["name"] == "bash"
    psf = msg.get("provider_specific_fields")
    assert psf == {"thought_signatures": ["bash_sig_b64"]}, (
        f"psf.thought_signatures should be filtered to just bash's sig, got {psf}"
    )


async def test_query_stashes_submit_signature_when_only_submit_returned():
    """Even a solo-submit response (no bash) has its signature stashed."""
    from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
        SamplerChatCompletionModel,
    )

    sampler = _FakeSampler(
        _make_chat_completion_ret(
            tool_calls=[_submit_tool_call("hi", tc_id="call_submit")],
            finish_reason="tool_calls",
            thought_signatures=["submit_sig_b64"],
        )
    )
    model = SamplerChatCompletionModel(sampler=sampler)
    msg = await model.query(OPENAI_MESSAGES)

    actions = msg["extra"]["actions"]
    assert actions[0]["thought_signature"] == "submit_sig_b64"
    # No tool_calls survive filter (submit was the only one); psf isn't
    # attached because there's nothing to attach it to.
    assert "tool_calls" not in msg or not msg["tool_calls"]


async def test_query_stashes_empty_string_when_submit_signature_absent():
    """If the response omits the submit's signature (e.g. thinking off),
    the stashed value is empty-string — not None, not missing."""
    from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
        SamplerChatCompletionModel,
    )

    sampler = _FakeSampler(
        _make_chat_completion_ret(
            tool_calls=[
                _bash_tool_call("ls", tc_id="call_bash"),
                _submit_tool_call("hi", tc_id="call_submit"),
            ],
            finish_reason="tool_calls",
            thought_signatures=["bash_sig_b64", ""],  # submit sig empty
        )
    )
    model = SamplerChatCompletionModel(sampler=sampler)
    msg = await model.query(OPENAI_MESSAGES)

    actions = msg["extra"]["actions"]
    assert actions[1]["thought_signature"] == ""


async def test_query_does_not_stash_when_no_psf_at_all():
    """No psf envelope on the response -> no stashing (no key added)."""
    from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
        SamplerChatCompletionModel,
    )

    sampler = _FakeSampler(
        _make_chat_completion_ret(
            tool_calls=[_submit_tool_call("hi", tc_id="call_submit")],
            finish_reason="tool_calls",
            # thought_signatures=None -> no psf on Message.
        )
    )
    model = SamplerChatCompletionModel(sampler=sampler)
    msg = await model.query(OPENAI_MESSAGES)

    actions = msg["extra"]["actions"]
    # No stash when the model didn't return any sigs.
    assert "thought_signature" not in actions[0]
