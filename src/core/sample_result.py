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

"""Typed Sampler response object, proto-aligned with the backend.

``Sampler.sample()`` returns ``SampleResult | None``; ``sample_n()``
returns ``list[SampleResult]``. Each bundles:

  * ``tuning_candidate`` — mirrors the backend's ``TuningCandidate``
    proto 1:1; ``candidate`` is the real public ``Candidate`` proto.
  * ``context`` — ``SamplingContext`` sidecar for client-side metadata
    not in the backend proto (request echoes, backend-specific
    identifiers, producer-set triage flags).
  * ``train_step_id`` — the response-level train step id, duplicated
    onto each candidate.

Derived accessors: ``completion_text`` / ``finish_reason`` properties,
``prompt_token_count`` / ``generation_token_count`` best-effort
``int | None``, and ``to_step_stats()`` for trajectory storage.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from google.cloud.aiplatform_v1.types import Candidate, GenerateContentRequest

from core.atif_trajectory import StepStats

if TYPE_CHECKING:
    # Kept behind TYPE_CHECKING so importing ``core.sample_result`` (which
    # every core consumer -- trainer, agent, sinks, dispatcher -- pulls at
    # startup) does not transitively load ``backends.chat_completion`` and
    # its heavy proto / converter dependency stack. Only the annotation on
    # ``ChatCompletionSampleResult.response`` uses this symbol, and
    # ``from __future__ import annotations`` (line 18) makes annotations
    # lazy strings that never need the runtime symbol.
    from backends.chat_completion import ChatCompletionResponse


@dataclasses.dataclass(frozen=True)
class TuningCandidate:
    """Python mirror of the backend proto ``TuningCandidate``.

    Token-count contract (all backends MUST honor):

      * ``generation_token_count`` — TOTAL tokens for this candidate,
        INCLUDING thinking. The full "generation cost" for length-based
        reward shaping, latency accounting, quota tracking.
      * ``thoughts_token_count`` — thinking-only SUBSET.
        ``0 <= thoughts_token_count <= generation_token_count``; 0 when
        thinking is disabled or not reported.

    Backends whose wire reports output-only and thinking-only counts as
    disjoint fields MUST sum them into ``generation_token_count`` (so
    an all-thinking, no-output response still yields a defined total).
    """

    candidate_id: str
    candidate: Candidate
    generation_token_count: int
    thoughts_token_count: int


@dataclasses.dataclass(frozen=True)
class SamplingContext:
    """Client-side sampling metadata NOT in the backend proto.

    Different backends populate different subsets: LRO-based backends
    fill ``operation_id`` / ``is_prompt_too_long``; single-request
    backends fill ``response_id`` / ``model_version``; all populate
    ``prompt`` (request echo) and ``prompt_token_count``. Unpopulated
    fields stay at their defaults.
    """

    prompt: GenerateContentRequest | str | None = None
    operation_id: str | None = None
    response_id: str | None = None
    model_version: str | None = None
    is_prompt_too_long: bool = False
    # Set by a backend that parsed a SUCCESSFUL response whose
    # ``usageMetadata`` was unusable (today: no prompt-token count). The
    # backend returns its empty/failsafe result instead of raising --
    # raising loses the whole task -- and this flag is what lets the
    # agent's triage give it a status of its own rather than folding it
    # into SAMPLING_FAILURE_UNHANDLED.
    is_malformed_usage: bool = False
    # Consumed by ``Sampler.sample_chat_completion`` to fill
    # ``ChatCompletionResponse.usage.prompt_tokens``; fail-fast if None.
    prompt_token_count: int | None = None
    # Relative URI of the archived raw request/response for this call,
    # written by ``utils.raw_capture.dump_raw`` when RAW_CAPTURE=1.
    # Populated by each concrete Sampler.sample() implementation; None
    # when capture is disabled. Flows down to
    # ``TrajectoryStep.from_wire`` which stamps
    # ``step.extra.llm_call_uri`` from this value.
    raw_call_uri: str | None = None
    # Extended-thinking blocks from this turn, as loose provider block dicts
    # (LiteLLM's cross-provider ``thinking_blocks`` shape: normal
    # ``{type, thinking, signature}`` and ``{type:redacted_thinking, data}``). A
    # multi-turn agent replays them verbatim on the next request; the model
    # requires the prior turn's thinking during function calling. The agent
    # treats them as opaque; only the producing backend knows the block shapes.
    # None when the backend produced no thinking.
    thinking_blocks: tuple[dict[str, Any], ...] | None = None


@dataclasses.dataclass(frozen=True)
class SampleResult:
    """One flattened candidate at the ``Sampler.sample()`` boundary.

    Consumers detect sampling failure via
    ``result.tuning_candidate.candidate_id == ""``.
    """

    tuning_candidate: TuningCandidate
    context: SamplingContext
    train_step_id: int | None

    @property
    def completion_text(self) -> str:
        """Flattened text of every non-thought ``Part``.

        Returns ``""`` when there are no text parts. Skips
        ``part.thought is True`` so thought-summary parts never leak
        into the model's answer text.
        """
        content = self.tuning_candidate.candidate.content
        if not content or not content.parts:
            return ""
        return "".join(p.text for p in content.parts if p.text and not p.thought)

    @property
    def finish_reason(self) -> str:
        """String name of the finish reason. Returns ``"OTHER"`` for
        ``FINISH_REASON_UNSPECIFIED`` (empty candidate on failure paths).
        """
        reason = self.tuning_candidate.candidate.finish_reason
        if reason == Candidate.FinishReason.FINISH_REASON_UNSPECIFIED:
            return "OTHER"
        return Candidate.FinishReason(reason).name

    @property
    def prompt_token_count(self) -> int | None:
        """Best-effort prompt-token count for this call.

        Surfaces ``context.prompt_token_count``, which every backend
        populates from the response's ``usageMetadata``. ``None`` when the
        backend did not report one. Flows through ``to_step_stats()`` into
        ``step.extra["prompt_token_count"]``, which is what the
        context-size metrics read.
        """
        return self.context.prompt_token_count

    @property
    def generation_token_count(self) -> int | None:
        """TOTAL generation tokens (thinking + non-thinking).

        The proto's int32 with 0-as-unset semantics is impossible to
        distinguish from a genuine zero-generation candidate, so 0 is
        mapped to ``None`` to force explicit handling by consumers.
        Use ``tuning_candidate.thoughts_token_count`` for the
        thinking-only subset.
        """
        if self.tuning_candidate.generation_token_count > 0:
            return self.tuning_candidate.generation_token_count
        return None

    def to_step_stats(self) -> StepStats:
        """Normalize this ``SampleResult`` into a typed ``StepStats``.

        ``candidate_id`` is ``None`` on empty (failure) so downstream
        stays back-compat with the older ``extras.get("candidateId")``
        semantics. ``operation_id`` is ``None`` for non-LRO backends.
        """
        return StepStats(
            finish_reason=self.finish_reason,
            candidate_id=self.tuning_candidate.candidate_id or None,
            operation_id=self.context.operation_id,
            prompt_token_count=self.prompt_token_count,
            generation_token_count=self.generation_token_count,
            train_step_id=self.train_step_id,
        )


@dataclasses.dataclass(frozen=True)
class ChatCompletionSampleResult:
    """Combined output of ``Sampler.sample_chat_completion``.

    Holds both the OpenAI-shape wire response (for callers speaking
    OpenAI Chat Completions) and the underlying list of ``SampleResult``s
    (for callers that need training-relevant metadata like
    ``candidate_id`` / ``response_id`` / ``thoughts_token_count`` /
    ``thought_signature`` that the OpenAI wire shape doesn't preserve).

    Attributes:
        response: The OpenAI-shape ``ChatCompletionResponse`` returned by
            ``convert_gemini_response_to_openai``. Use ``.to_dict()`` for
            the JSON-serializable wire dict.
        results: The list of ``SampleResult`` produced by the underlying
            ``self.sample()`` call, one per candidate. Preserves
            training-relevant metadata that the OpenAI wire shape drops.
    """

    response: ChatCompletionResponse
    results: list[SampleResult]


def with_raw_call_uris(
    results: list[SampleResult], uris: "list[str | None]"
) -> list[SampleResult]:
    """Stamp each result with the raw-ledger URI of its own call.

    A blob nothing points at is a blob nobody finds: the URI is what the
    journal turns into ``step.extra.llm_call_uri``, and the only way back
    from a trajectory step to the bytes that produced it. Every sampler
    was doing this replace-inside-a-replace by hand at eleven sites.

    Positional, one URI per result. A short ``uris`` leaves the remaining
    results untouched rather than raising -- losing the pointer to a blob
    must not lose the sample it belongs to.
    """
    return [
        (
            dataclasses.replace(
                r, context=dataclasses.replace(r.context, raw_call_uri=uris[i])
            )
            if i < len(uris) and uris[i] is not None
            else r
        )
        for i, r in enumerate(results)
    ]
