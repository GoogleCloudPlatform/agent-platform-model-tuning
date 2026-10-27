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

"""Tests for the proto-aligned SampleResult wire boundary type.

Covers:
  * Frozenness of all three dataclasses.
  * completion_text accessor (empty / single-part / multi-part / thought parts).
  * finish_reason accessor (proto enum name + FINISH_REASON_UNSPECIFIED -> "OTHER").
  * SamplingContext defaults.
  * Per-backend field usage (operation-id/prompt-length vs gemini_api).
  * to_dict() round-trip against pinned real-shape fixtures for both backends.

The to_dict() tests are DELETED IN COMMIT 3 when the bridge method is
removed. All other tests survive.
"""

from __future__ import annotations

import dataclasses

import pytest
from google.cloud.aiplatform_v1.types import Candidate, Content, Part

from core.sample_result import (
    SampleResult,
    SamplingContext,
    TuningCandidate,
)


def _make_candidate(
    *, text: str = "", thought: bool = False, finish: str = "STOP"
) -> Candidate:
    """Construct a Candidate proto with a single Part."""
    parts = []
    if text or thought:
        parts.append(Part(text=text, thought=thought))
    return Candidate(
        content=Content(role="model", parts=parts) if parts else Content(),
        finish_reason=Candidate.FinishReason[finish],
    )


class TestFrozenness:
    def test_sample_result_frozen(self):
        r = SampleResult(
            tuning_candidate=TuningCandidate("", Candidate(), 0, 0),
            context=SamplingContext(),
            train_step_id=None,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.train_step_id = 5  # type: ignore[misc]

    def test_tuning_candidate_frozen(self):
        tc = TuningCandidate("id", Candidate(), 0, 0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            tc.candidate_id = "other"  # type: ignore[misc]

    def test_sampling_context_frozen(self):
        ctx = SamplingContext()
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.operation_id = "op"  # type: ignore[misc]


class TestCompletionText:
    def test_empty_candidate(self):
        r = SampleResult(
            tuning_candidate=TuningCandidate("", Candidate(), 0, 0),
            context=SamplingContext(),
            train_step_id=None,
        )
        assert r.completion_text == ""

    def test_single_part(self):
        r = SampleResult(
            tuning_candidate=TuningCandidate(
                "cid", _make_candidate(text="hello"), 0, 0
            ),
            context=SamplingContext(),
            train_step_id=None,
        )
        assert r.completion_text == "hello"

    def test_multi_part_concatenates(self):
        cand = Candidate(
            content=Content(role="model", parts=[Part(text="foo"), Part(text="bar")]),
            finish_reason=Candidate.FinishReason.STOP,
        )
        r = SampleResult(
            tuning_candidate=TuningCandidate("cid", cand, 0, 0),
            context=SamplingContext(),
            train_step_id=None,
        )
        assert r.completion_text == "foobar"

    def test_thought_parts_skipped(self):
        """Thought parts must NOT leak into the answer text."""
        cand = Candidate(
            content=Content(
                role="model",
                parts=[
                    Part(text="thinking...", thought=True),
                    Part(text="answer"),
                ],
            ),
            finish_reason=Candidate.FinishReason.STOP,
        )
        r = SampleResult(
            tuning_candidate=TuningCandidate("cid", cand, 0, 0),
            context=SamplingContext(),
            train_step_id=None,
        )
        assert r.completion_text == "answer"

    def test_content_unset(self):
        cand = Candidate(finish_reason=Candidate.FinishReason.STOP)
        r = SampleResult(
            tuning_candidate=TuningCandidate("cid", cand, 0, 0),
            context=SamplingContext(),
            train_step_id=None,
        )
        assert r.completion_text == ""


class TestFinishReason:
    @pytest.mark.parametrize(
        "reason_enum,expected",
        [
            (Candidate.FinishReason.STOP, "STOP"),
            (Candidate.FinishReason.MAX_TOKENS, "MAX_TOKENS"),
            (Candidate.FinishReason.MALFORMED_FUNCTION_CALL, "MALFORMED_FUNCTION_CALL"),
        ],
    )
    def test_populated_returns_enum_name(self, reason_enum, expected):
        cand = Candidate(finish_reason=reason_enum)
        r = SampleResult(
            tuning_candidate=TuningCandidate("cid", cand, 0, 0),
            context=SamplingContext(),
            train_step_id=None,
        )
        assert r.finish_reason == expected

    def test_unspecified_returns_other(self):
        """FINISH_REASON_UNSPECIFIED (empty candidate on failure paths) -> 'OTHER'."""
        r = SampleResult(
            tuning_candidate=TuningCandidate("", Candidate(), 0, 0),
            context=SamplingContext(),
            train_step_id=None,
        )
        assert r.finish_reason == "OTHER"


class TestSamplingContextDefaults:
    def test_all_defaults(self):
        ctx = SamplingContext()
        assert ctx.prompt is None
        assert ctx.operation_id is None
        assert ctx.response_id is None
        assert ctx.model_version is None
        assert ctx.is_prompt_too_long is False
        assert ctx.prompt_token_count is None

    def test_operation_id_backend_fields(self):
        ctx = SamplingContext(operation_id="op-1", is_prompt_too_long=True)
        assert ctx.operation_id == "op-1"
        assert ctx.is_prompt_too_long is True
        # gemini_api fields default
        assert ctx.response_id is None
        assert ctx.model_version is None

    def test_gemini_api_backend_fields(self):
        ctx = SamplingContext(response_id="r-1", model_version="gemini-2.5-flash")
        assert ctx.response_id == "r-1"
        assert ctx.model_version == "gemini-2.5-flash"
        # operation-id / prompt-length fields default
        assert ctx.operation_id is None
        assert ctx.is_prompt_too_long is False

    def test_prompt_token_count_field(self):
        """SamplingContext.prompt_token_count is a new optional int field.

        Populated by samplers from the raw usageMetadata.promptTokenCount
        they parse from the backend response; used by
        Sampler.sample_chat_completion() to reconstitute the response proto
        with a real usage.prompt_tokens value (fail-fast contract forbids
        silent 0).
        """
        ctx = SamplingContext(prompt_token_count=42)
        assert ctx.prompt_token_count == 42
        # Default: None (unset).
        assert SamplingContext().prompt_token_count is None


class TestSamplingContextRawCallUri:
    def test_raw_call_uri_defaults_to_none(self) -> None:
        ctx = SamplingContext(prompt="hi")
        assert ctx.raw_call_uri is None

    def test_raw_call_uri_can_be_set(self) -> None:
        ctx = SamplingContext(
            prompt="hi",
            raw_call_uri="artifacts/raw_llm_calls/call_000003.json",
        )
        assert ctx.raw_call_uri == "artifacts/raw_llm_calls/call_000003.json"

    def test_sampling_context_still_frozen(self) -> None:
        """Adding a field must not break the frozen dataclass contract."""
        ctx = SamplingContext(prompt="hi")
        assert dataclasses.is_dataclass(ctx)
        try:
            ctx.raw_call_uri = "x"  # type: ignore[misc]
        except dataclasses.FrozenInstanceError:
            pass
        else:
            raise AssertionError("SamplingContext should still be frozen")


class TestPromptTokenCountAccessor:
    """``SampleResult.prompt_token_count`` must surface the context's count.

    The property used to hardcode ``None``, so ``to_step_stats()`` never
    carried a prompt count and ``TrajectoryStep.from_wire`` never stamped
    ``extra["prompt_token_count"]`` — silently nulling every downstream
    context-size metric.
    """

    def test_property_reads_the_context_field(self) -> None:
        from tests.unit._sample_result_factories import make_sample_result

        result = make_sample_result(text="hi", prompt_token_count=1234)
        assert result.prompt_token_count == 1234

    def test_property_is_none_when_the_backend_did_not_report(self) -> None:
        from tests.unit._sample_result_factories import make_sample_result

        result = make_sample_result(text="hi")
        assert result.prompt_token_count is None

    def test_to_step_stats_carries_the_prompt_token_count(self) -> None:
        from tests.unit._sample_result_factories import make_sample_result

        result = make_sample_result(
            text="hi", prompt_token_count=777, generation_token_count=11
        )
        stats = result.to_step_stats()
        assert stats.prompt_token_count == 777
        assert stats.generation_token_count == 11

    def test_step_built_from_stats_stamps_the_extra(self) -> None:
        from core.atif_trajectory import TrajectoryStep
        from tests.unit._sample_result_factories import make_sample_result

        result = make_sample_result(
            text="hi", prompt_token_count=777, generation_token_count=11
        )
        step = TrajectoryStep.from_wire(
            step_id=2, generation="hi", stats=result.to_step_stats()
        )
        assert step.extra["prompt_token_count"] == 777
