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

"""Contract tests for GeminiApiSampler token-count mapping semantics.

The `generateContent` API reports token usage in three separate fields
under `usageMetadata`:

  * ``candidatesTokenCount`` -- decoded OUTPUT tokens ONLY (excludes
    thinking); omitted from the response when the model produced zero
    output (e.g. all budget went to thinking).
  * ``thoughtsTokenCount``   -- decoded THINKING tokens; present when
    thinking is enabled, omitted otherwise.
  * ``totalTokenCount``      -- prompt + candidates + thoughts.

This repo's convention (per user, 2026-07-12):

  * ``TuningCandidate.generation_token_count`` = TOTAL tokens the model
    generated (thinking + non-thinking). Consumers reading this field
    (e.g. length-based reward shaping) want the model's full generation
    cost, not the visible-output slice.
  * ``TuningCandidate.thoughts_token_count``   = THINKING-only subset.

So the mapping from the Vertex wire is:
    generation_token_count = candidatesTokenCount + thoughtsTokenCount
    thoughts_token_count   = thoughtsTokenCount

This matters especially for "all-thinking, no-output" responses (real
observation: `usageMetadata={thoughtsTokenCount: 1846, totalTokenCount:
7627, promptTokenCount: 5781}` with `candidates=[{finishReason: STOP}]`
and no content). Before this contract:
  * The sampler raised ValueError on the missing candidatesTokenCount,
    crashing every rollout that hit this state (11/150 at thinking=medium
    in the withhold word_puzzle eval, 2026-07-12).
After:
  * `generation_token_count = 0 + 1846 = 1846` (correctly reflects the
    model's generation cost -- all thinking).
  * `thoughts_token_count = 1846`.
  * The rollout continues; the empty candidate is a legitimate outcome
    the agent handles via its existing empty-parts path.

These tests pin the mapping so a future refactor can't silently break
either the semantic OR the empty-candidate path.
"""

from __future__ import annotations

from unittest import mock

import pytest

from backends.gemini_api_endpoint import GeminiApiSampler


def _make_sampler() -> GeminiApiSampler:
    """Return a sampler wired to a real (unstarted) client; no network."""
    # We short-circuit `_post_with_retries` in each test, so auth /
    # endpoint construction only needs to not crash. `project` /
    # `env` values are inert placeholders.
    return GeminiApiSampler(
        model="gemini-3.5-flash",
        project="test-project",
        env="prod",
    )


async def _sample_with_response(sampler: GeminiApiSampler, response: dict) -> list:
    """Invoke `sampler.sample(prompt="hi")` with `_post_with_retries`
    monkey-patched to return `response`. Returns the list[SampleResult]."""

    async def fake_post(*_args, **_kwargs):
        return response

    with mock.patch.object(sampler, "_post_with_retries", side_effect=fake_post):
        return await sampler.sample_n(prompt="hi", num_samples=1)


# --------------------------------------------------------------------------
# Semantic: generation_token_count = candidatesTokenCount + thoughtsTokenCount
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gen_count_is_output_plus_thoughts():
    """The 'normal' case: response has both candidate output AND thinking.

    Wire: candidatesTokenCount=100, thoughtsTokenCount=50
    Expect: generation_token_count=150 (total), thoughts_token_count=50
    """
    sampler = _make_sampler()
    response = {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "hello"}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 100,
            "thoughtsTokenCount": 50,
            "totalTokenCount": 160,
        },
        "responseId": "resp-1",
        "modelVersion": "gemini-3.5-flash-001",
    }
    results = await _sample_with_response(sampler, response)
    assert len(results) == 1
    tc = results[0].tuning_candidate
    assert tc.generation_token_count == 150, (
        "generation_token_count should be candidates+thoughts (total generated)"
    )
    assert tc.thoughts_token_count == 50


@pytest.mark.asyncio
async def test_gen_count_when_no_thoughts_reported():
    """Thinking disabled: only candidatesTokenCount is present.

    Wire: candidatesTokenCount=42, thoughtsTokenCount absent
    Expect: generation_token_count=42, thoughts_token_count=0
    """
    sampler = _make_sampler()
    response = {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "hi"}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 5,
            "candidatesTokenCount": 42,
            "totalTokenCount": 47,
        },
    }
    results = await _sample_with_response(sampler, response)
    tc = results[0].tuning_candidate
    assert tc.generation_token_count == 42
    assert tc.thoughts_token_count == 0


# --------------------------------------------------------------------------
# Empty-candidate case (the 2026-07-12 bug fix)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_thinking_no_output_does_not_raise():
    """Real observation: model burns thinking budget then produces zero
    output. Response omits candidatesTokenCount entirely (not an API shape
    violation -- there's no candidate content to count).

    Wire:
      candidates=[{finishReason: STOP}]  (no content field!)
      usageMetadata={thoughtsTokenCount: 1846, totalTokenCount: 7627,
                     promptTokenCount: 5781}
    Expect:
      * No ValueError (empty-output is a legitimate state)
      * generation_token_count = 1846 (all thinking, no visible output)
      * thoughts_token_count = 1846
    """
    sampler = _make_sampler()
    response = {
        "candidates": [{"finishReason": "STOP"}],
        "usageMetadata": {
            "promptTokenCount": 5781,
            "thoughtsTokenCount": 1846,
            "totalTokenCount": 7627,
            "trafficType": "ON_DEMAND",
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 5781}],
        },
        "responseId": "resp-empty",
        "modelVersion": "gemini-3.5-flash-001",
    }
    # Must not raise.
    results = await _sample_with_response(sampler, response)
    assert len(results) == 1
    tc = results[0].tuning_candidate
    assert tc.generation_token_count == 1846, (
        "when candidatesTokenCount omitted (empty output), gen count "
        "should still reflect all-thinking generation cost"
    )
    assert tc.thoughts_token_count == 1846


@pytest.mark.asyncio
async def test_all_thinking_no_output_no_thoughts_field_either():
    """Pathological edge: response has neither candidatesTokenCount NOR
    thoughtsTokenCount, and the single candidate has no content.

    This should NOT raise (we can't distinguish 'truly empty' from
    'thinking not enabled') -- treat as (0, 0). The rollout will handle
    the empty-content candidate via the empty-parts path.
    """
    sampler = _make_sampler()
    response = {
        "candidates": [{"finishReason": "STOP"}],
        "usageMetadata": {
            "promptTokenCount": 100,
            "totalTokenCount": 100,
        },
    }
    results = await _sample_with_response(sampler, response)
    tc = results[0].tuning_candidate
    assert tc.generation_token_count == 0
    assert tc.thoughts_token_count == 0


# --------------------------------------------------------------------------
# Real API-shape violation still raises
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_candidate_has_content_but_no_token_counts_raises():
    """A candidate WITH content but no token counts anywhere IS a real
    API shape violation -- we can't attribute the visible generation to
    any count. Fail fast so the caller sees the malformed response.
    """
    sampler = _make_sampler()
    response = {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "hello"}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "totalTokenCount": 10,
            # Deliberately no candidatesTokenCount, no thoughtsTokenCount,
            # but the candidate DID produce content -- contradiction.
        },
    }
    with pytest.raises(ValueError, match="candidatesTokenCount"):
        await _sample_with_response(sampler, response)


# --------------------------------------------------------------------------
# Missing promptTokenCount on the success path
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_prompt_token_count_returns_failsafe():
    """A response with candidates but no ``promptTokenCount`` is malformed.

    Every request has a prompt, so unlike the generation counts there is
    no legitimate response shape that omits this. Accepting it silently
    as ``None`` disables in-loop context compaction (which refuses to
    guess) and leaves the context-size metrics blank. We surface it as a
    sampling failure rather than raising: raising loses the whole task.
    """
    sampler = _make_sampler()
    response = {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "hello"}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "candidatesTokenCount": 3,
            "totalTokenCount": 13,
            # No promptTokenCount.
        },
    }
    results = await _sample_with_response(sampler, response)
    assert len(results) == 1
    assert results[0].tuning_candidate.candidate_id == ""
    assert results[0].context.is_malformed_usage is True
    assert results[0].context.prompt_token_count is None
