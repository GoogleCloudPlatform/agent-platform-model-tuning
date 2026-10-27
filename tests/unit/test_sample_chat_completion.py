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

"""Unit tests for Sampler.sample_chat_completion + reconstitution helper.

Uses a fake Sampler subclass whose sample() returns caller-supplied
SampleResults, so we can drive the reconstitution + converter path
without any HTTP. Covers the fail-fast contract + failsafe detection +
num_samples authority + thinking_level passthrough.
"""

from __future__ import annotations

import pytest
from google.cloud.aiplatform_v1.types import Candidate, Content, Part

from core.protocols import Sampler
from core.sample_result import SampleResult, SamplingContext, TuningCandidate


def _make_candidate_proto(text: str = "hi", finish: str = "STOP") -> Candidate:
    """Small helper to build a minimal Candidate proto (proto-plus API)."""
    return Candidate(
        content=Content(role="model", parts=[Part(text=text)]),
        finish_reason=Candidate.FinishReason[finish],
    )


def _make_result(
    *,
    candidate_id: str = "cand-uuid-0",
    text: str = "hi",
    gen_tokens: int = 5,
    prompt_tokens: int | None = 17,
    response_id: str | None = "resp-abc",
    operation_id: str | None = None,
    model_version: str | None = "gemini-3.5-flash",
    is_prompt_too_long: bool = False,
    is_malformed_usage: bool = False,
) -> SampleResult:
    # For the failsafe sentinel (candidate_id == ""), the Candidate is
    # empty (matches _empty_result shape in all three real samplers).
    if candidate_id == "":
        candidate = Candidate()
    else:
        candidate = _make_candidate_proto(text)
    return SampleResult(
        tuning_candidate=TuningCandidate(
            candidate_id=candidate_id,
            candidate=candidate,
            generation_token_count=gen_tokens,
            thoughts_token_count=0,
        ),
        context=SamplingContext(
            prompt="hello",
            response_id=response_id,
            operation_id=operation_id,
            model_version=model_version,
            prompt_token_count=prompt_tokens,
            is_prompt_too_long=is_prompt_too_long,
            is_malformed_usage=is_malformed_usage,
        ),
        train_step_id=None,
    )


class _FakeSampler(Sampler):
    """Sampler subclass that returns caller-supplied SampleResults."""

    def __init__(self, results: list[SampleResult]):
        self._results = results
        self.last_prompt: object = None
        self.last_num_samples: int | None = None
        self.last_thinking_level: str | None = None

    async def sample_n(
        self, prompt, num_samples=1, thinking_level=None, **kwargs
    ):
        self.last_prompt = prompt
        self.last_num_samples = num_samples
        self.last_thinking_level = thinking_level
        return self._results


OPENAI_REQUEST = {
    "model": "gemini-3.5-flash",
    "messages": [{"role": "user", "content": "hi"}],
    "max_tokens": 16,
}


async def test_happy_path_returns_valid_chat_completion_response():
    sampler = _FakeSampler([_make_result()])
    result = await sampler.sample_chat_completion(OPENAI_REQUEST, num_samples=1)
    # result is a ChatCompletionSampleResult wrapper: .response is the
    # OpenAI-shape ChatCompletionResponse, .results is the underlying
    # list[SampleResult] preserving training-relevant metadata.
    assert result.response.id == "resp-abc"
    assert result.response.model == "gemini-3.5-flash"
    assert len(result.response.choices) == 1
    assert result.response.usage.prompt_tokens == 17
    assert result.response.usage.completion_tokens == 5
    # The .results side-channel round-trips the underlying SampleResults.
    assert len(result.results) == 1
    assert result.results[0].tuning_candidate.candidate_id == "cand-uuid-0"


async def test_assistant_role_rewritten_to_model_on_wire():
    """Regression (prod HTTP 400): the tuning-scope endpoint rejects
    role='assistant' -- only 'user'/'model'/empty are allowed. The
    sample_chat_completion Template Method converts the OpenAI request via
    a DEFAULT ConversionContext, so that default MUST rewrite assistant ->
    model. Assert the GenerateContentRequest handed to sample() carries
    'model', never 'assistant', for a prior-assistant turn."""
    sampler = _FakeSampler([_make_result()])
    multi_turn = {
        "model": "gemini-3.5-flash",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "again"},
        ],
        "max_tokens": 16,
    }
    await sampler.sample_chat_completion(multi_turn, num_samples=1)

    prompt = sampler.last_prompt  # the converted GenerateContentRequest proto
    roles = [c.role for c in prompt.contents]
    assert "assistant" not in roles, (
        f"assistant role leaked to the wire (tuning-scope rejects it): {roles}"
    )
    assert "model" in roles, f"expected a 'model' turn, got roles={roles}"


async def test_num_samples_overrides_openai_n():
    """When openai_request has "n": 5 but arg num_samples=2, sample()
    receives num_samples=2."""
    sampler = _FakeSampler([_make_result(), _make_result(candidate_id="c1")])
    req = {**OPENAI_REQUEST, "n": 5}
    await sampler.sample_chat_completion(req, num_samples=2)
    assert sampler.last_num_samples == 2


async def test_thinking_level_forwarded():
    sampler = _FakeSampler([_make_result()])
    await sampler.sample_chat_completion(
        OPENAI_REQUEST, num_samples=1, thinking_level="MEDIUM"
    )
    assert sampler.last_thinking_level == "MEDIUM"


async def test_fail_fast_missing_prompt_token_count():
    sampler = _FakeSampler([_make_result(prompt_tokens=None)])
    with pytest.raises(ValueError, match="prompt_token_count"):
        await sampler.sample_chat_completion(OPENAI_REQUEST, num_samples=1)


async def test_fail_fast_missing_id_both_none():
    """response_id AND operation_id both None -> raise."""
    sampler = _FakeSampler([_make_result(response_id=None, operation_id=None)])
    with pytest.raises(ValueError, match="response_id.*operation_id"):
        await sampler.sample_chat_completion(OPENAI_REQUEST, num_samples=1)


async def test_operation_id_satisfies_id_requirement():
    """response_id None but operation_id set -> passes."""
    sampler = _FakeSampler([_make_result(response_id=None, operation_id="op-xyz")])
    result = await sampler.sample_chat_completion(OPENAI_REQUEST, num_samples=1)
    assert result.response.id == "op-xyz"


async def test_fail_fast_missing_model_version():
    sampler = _FakeSampler([_make_result(model_version=None)])
    with pytest.raises(ValueError, match="model_version"):
        await sampler.sample_chat_completion(OPENAI_REQUEST, num_samples=1)


async def test_failsafe_empty_result_raises_runtime_error():
    """SampleResult with candidate_id == "" is the failsafe sentinel.
    sample_chat_completion refuses to convert it to a bogus response."""
    sampler = _FakeSampler([_make_result(candidate_id="")])
    with pytest.raises(RuntimeError, match="failsafe empty result"):
        await sampler.sample_chat_completion(OPENAI_REQUEST, num_samples=1)


async def test_the_failsafe_names_why_it_is_empty():
    """The raise is the ONLY thing the caller sees, so it must carry the reason.

    The failsafe knows on its context WHY it is empty, but a caller only
    gets the exception. Anything not repeated in the message is lost, and a
    caller classifying the exception falls back to "unknown transport
    error" -- which is not cosmetic: an over-long prompt is an INPUT fault
    (deterministic, re-sending cannot help, and it is evidence about the
    rollout) while an unknown transport error is INFRA (retriable, NaN
    instead of a score). Dropping the reason silently converts the first
    into the second.
    """
    sampler = _FakeSampler(
        [_make_result(candidate_id="", is_prompt_too_long=True)]
    )
    with pytest.raises(RuntimeError, match="prompt is too long"):
        await sampler.sample_chat_completion(OPENAI_REQUEST, num_samples=1)

    sampler = _FakeSampler(
        [_make_result(candidate_id="", is_malformed_usage=True)]
    )
    with pytest.raises(RuntimeError, match="unusable usageMetadata"):
        await sampler.sample_chat_completion(OPENAI_REQUEST, num_samples=1)

    # And the unremarkable case still says the generic thing.
    sampler = _FakeSampler([_make_result(candidate_id="")])
    with pytest.raises(RuntimeError, match="upstream sampling failed"):
        await sampler.sample_chat_completion(OPENAI_REQUEST, num_samples=1)


async def test_empty_results_list_raises():
    """sample() returning [] -> fail-fast (nothing to reconstitute)."""
    sampler = _FakeSampler([])
    with pytest.raises(ValueError, match="empty"):
        await sampler.sample_chat_completion(OPENAI_REQUEST, num_samples=1)


async def test_multi_candidate_usage_aggregates_correctly():
    sampler = _FakeSampler(
        [
            _make_result(candidate_id="c0", gen_tokens=5),
            _make_result(candidate_id="c1", gen_tokens=7),
        ]
    )
    result = await sampler.sample_chat_completion(OPENAI_REQUEST, num_samples=2)
    assert result.response.usage.completion_tokens == 12  # 5 + 7
    assert result.response.usage.prompt_tokens == 17  # request-level, shared
    assert len(result.response.choices) == 2
    assert len(result.results) == 2
