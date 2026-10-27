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

"""Contract tests for VertexMsrlRestSampler.sample().

Two test-suite themes:

  1. thinking_level must reach the request body (regression guard;
     see class TestSampleWiresThinkingLevel below).
  2. SamplingContext population — the sampler must plumb request-scoped
     metadata (prompt_token_count, model_version) onto every
     SampleResult so ``Sampler.sample_chat_completion`` can reconstitute
     an OpenAI-shape response without falling back to silent defaults.

Historical: the original file was single-theme (thinking-level only);
the SamplingContext theme was added when sample_chat_completion landed.


Regression guard for a bug where
``src/backends/vertex_msrl_rest.py:sample()`` silently discarded the
``thinking_level`` kwarg via ``del thinking_level``. That meant a
recipe declaring ``config.thinking_level=MINIMAL`` on the vertex_msrl
backend was a no-op -- the tuning server used its default (higher)
reasoning budget instead of the requested one, inflating per-sample
latency and burning steps on thinking rather than tool calls.

The fix mirrors ``backends.gemini_api_endpoint.GeminiApiSampler.sample``
which stamps ``generation_config.thinking_config.thinking_level`` onto
the prompt via ``backends.vertex_thinking.apply_thinking_level`` before
serializing. ``_build_body`` already forwards ``generationConfig``
through the wire dict verbatim, so the fix is a one-line addition on
top of the existing sampler machinery.
"""

from __future__ import annotations

from unittest import mock

import pytest
from google.cloud.aiplatform_v1.types import GenerateContentRequest

from backends.vertex_msrl_rest import VertexMsrlRestClient, VertexMsrlRestSampler


def _make_sampler():
    """Return a sampler wired to a real (unstarted) client instance."""
    client = VertexMsrlRestClient(
        project="test-project",
        location="us-central1",
        endpoint="test.example.com",
        tuning_job_id="12345",
    )
    return VertexMsrlRestSampler(client)


async def _run_sample_capturing_body(sampler, prompt, thinking_level):
    """Invoke ``sampler.sample(prompt, thinking_level=...)``, patching
    the client's ``_request`` to capture the JSON body without hitting
    the network. Returns the captured body dict.

    The sampler is expected to (a) build the body via ``_build_body``,
    (b) POST it via ``client._request(..., json=body)``. We short-
    circuit that POST by raising an exception; the sampler's ``except``
    branch returns empty results and we're done -- the body has already
    been captured. This exercises the exact code path we care about
    without any LRO / polling machinery.
    """
    captured = {}

    async def fake_request(*args, **kwargs):
        captured["body"] = kwargs.get("json")
        # Raise to short-circuit the LRO polling path. The sampler's
        # try/except returns empty_results without inspecting further.
        raise RuntimeError("short-circuit for test capture")

    with mock.patch.object(sampler.client, "_request", side_effect=fake_request):
        results = await sampler.sample_n(
            prompt, num_samples=1, thinking_level=thinking_level
        )
    assert results  # empty_result placeholders should be returned
    assert "body" in captured, "sample() should have called _request(json=body)"
    return captured["body"]


class TestSampleWiresThinkingLevel:
    """The regression: sample(prompt, thinking_level='MINIMAL') MUST
    produce a request body whose generationConfig.thinkingConfig
    carries thinkingLevel='MINIMAL'.
    """

    @pytest.mark.asyncio
    async def test_minimal_thinking_level_reaches_request_body(self):
        sampler = _make_sampler()
        prompt = GenerateContentRequest(
            contents=[{"role": "user", "parts": [{"text": "hi"}]}]
        )
        body = await _run_sample_capturing_body(sampler, prompt, "MINIMAL")

        # The tuning-scope RPC nests everything under
        # contentGenerationParameters.
        params = body["contentGenerationParameters"]
        gen_cfg = params.get("generationConfig") or {}
        thinking = gen_cfg.get("thinkingConfig") or {}
        assert thinking.get("thinkingLevel") == "MINIMAL", (
            "sample(thinking_level='MINIMAL') must set "
            "generationConfig.thinkingConfig.thinkingLevel='MINIMAL' in the "
            f"request body; got body={body!r}"
        )

    @pytest.mark.asyncio
    async def test_high_thinking_level_reaches_request_body(self):
        """Non-MINIMAL levels also propagate (validates the level is
        passed through verbatim, not hardcoded)."""
        sampler = _make_sampler()
        prompt = GenerateContentRequest(
            contents=[{"role": "user", "parts": [{"text": "hi"}]}]
        )
        body = await _run_sample_capturing_body(sampler, prompt, "HIGH")

        params = body["contentGenerationParameters"]
        thinking = (params.get("generationConfig") or {}).get("thinkingConfig") or {}
        assert thinking.get("thinkingLevel") == "HIGH"

    @pytest.mark.asyncio
    async def test_none_thinking_level_omits_thinking_config(self):
        """No thinking_level -> no thinkingConfig in the body. The
        server default (whatever that is) then applies, which is what
        the sampler contract has always promised for a bare sample()
        call."""
        sampler = _make_sampler()
        prompt = GenerateContentRequest(
            contents=[{"role": "user", "parts": [{"text": "hi"}]}]
        )
        body = await _run_sample_capturing_body(sampler, prompt, None)

        gen_cfg = body["contentGenerationParameters"].get("generationConfig") or {}
        assert "thinkingConfig" not in gen_cfg, (
            "sample(thinking_level=None) must NOT populate thinkingConfig; "
            f"got generationConfig={gen_cfg!r}"
        )

    @pytest.mark.asyncio
    async def test_empty_string_thinking_level_omits_thinking_config(self):
        """The recipe surface uses '' as the 'server default' sentinel
        for the ``thinking_level`` config default. It MUST be treated the
        same as None -- no thinkingConfig."""
        sampler = _make_sampler()
        prompt = GenerateContentRequest(
            contents=[{"role": "user", "parts": [{"text": "hi"}]}]
        )
        body = await _run_sample_capturing_body(sampler, prompt, "")

        gen_cfg = body["contentGenerationParameters"].get("generationConfig") or {}
        assert "thinkingConfig" not in gen_cfg


async def _run_sample_with_fake_lro(sampler, prompt, lro_response):
    """Invoke ``sampler.sample(prompt)`` with the full LRO happy path
    stubbed: ``_request`` returns a fake op-start, ``wait_for_lro``
    returns ``{"response": lro_response}``. Returns the list of
    SampleResult objects the sampler produced."""

    async def fake_request(*args, **kwargs):
        return {"name": "projects/p/locations/l/tuningJobs/j/operations/op-1"}

    async def fake_wait(*args, **kwargs):
        return {"response": lro_response}

    with mock.patch.object(sampler.client, "_request", side_effect=fake_request):
        with mock.patch.object(sampler.client, "wait_for_lro", side_effect=fake_wait):
            return await sampler.sample_n(prompt, num_samples=1)


# A minimal wire-shape tuningCandidate that _json_dict_to_candidate
# (proto-plus Candidate.from_json) accepts. Kept as a fixture rather
# than embedded so the train-step-id tests below stay focused.
_MINIMAL_TUNING_CANDIDATE = {
    "candidateId": "cand-0",
    "candidate": {
        "content": {"role": "model", "parts": [{"text": "ok"}]},
        "finishReason": "STOP",
    },
    "generationTokenCount": 3,
    "thoughtsTokenCount": 0,
    "promptTokenCount": 5,
}


class TestSampleCoercesTrainStepIdToInt:
    """Regression guard: Vertex tuning-scope responses marshal the
    ``trainStepId`` field (proto int64) as a JSON string per the
    standard protobuf-JSON encoding for 64-bit ints (precision-safe).
    The sampler must coerce it to Python ``int`` before stamping onto
    ``SampleResult.train_step_id`` -- otherwise the string propagates
    into ``step.stats.train_step_id`` (annotated ``int | None``) and
    downstream aggregators like
    ``SamplerPolicyStepMetricAggregator`` blow up on ``int += str``,
    silently killing every trajectory group without a visible error
    to the operator.

    Another backend already handles this coercion; vertex_msrl_rest was
    written without it and needs to match.
    """

    @pytest.mark.asyncio
    async def test_train_step_id_as_wire_string_becomes_int(self):
        sampler = _make_sampler()
        prompt = GenerateContentRequest(
            contents=[{"role": "user", "parts": [{"text": "hi"}]}]
        )
        results = await _run_sample_with_fake_lro(
            sampler,
            prompt,
            lro_response={
                # The Vertex tuning server sends int64 as a JSON string
                # to preserve 53-bit-plus precision.
                "trainStepId": "42",
                "tuningCandidates": [_MINIMAL_TUNING_CANDIDATE],
            },
        )
        assert len(results) == 1
        ts_id = results[0].train_step_id
        assert isinstance(ts_id, int) and not isinstance(ts_id, bool), (
            "SampleResult.train_step_id must be Python int (per "
            "sample_result.py:88 annotation `int | None`); got "
            f"{type(ts_id).__name__}={ts_id!r}. Downstream "
            "SamplerPolicyStepMetricAggregator does `int += ts_id` and "
            "will TypeError on str."
        )
        assert ts_id == 42

    @pytest.mark.asyncio
    async def test_train_step_id_as_wire_int_stays_int(self):
        """Some Vertex responses (int32-fits paths, older servers) may
        send trainStepId already-numeric. Coercion must be idempotent."""
        sampler = _make_sampler()
        prompt = GenerateContentRequest(
            contents=[{"role": "user", "parts": [{"text": "hi"}]}]
        )
        results = await _run_sample_with_fake_lro(
            sampler,
            prompt,
            lro_response={
                "trainStepId": 7,  # numeric on the wire
                "tuningCandidates": [_MINIMAL_TUNING_CANDIDATE],
            },
        )
        assert results[0].train_step_id == 7
        assert isinstance(results[0].train_step_id, int)

    @pytest.mark.asyncio
    async def test_train_step_id_absent_stays_none(self):
        """If the server omits trainStepId (e.g. pre-training-start
        eval, or a backend that never sets it), the sampler must produce
        None -- NOT 0, which would corrupt staleness metrics."""
        sampler = _make_sampler()
        prompt = GenerateContentRequest(
            contents=[{"role": "user", "parts": [{"text": "hi"}]}]
        )
        results = await _run_sample_with_fake_lro(
            sampler,
            prompt,
            lro_response={
                # No trainStepId field at all.
                "tuningCandidates": [_MINIMAL_TUNING_CANDIDATE],
            },
        )
        assert results[0].train_step_id is None

    @pytest.mark.asyncio
    async def test_train_step_id_empty_string_becomes_none(self):
        """The current sampler already treats ``"" or None`` as None
        (via ``response.get("trainStepId") or None``); we must
        preserve that behavior after coercion. Empty string coerced to
        int would ValueError, so the fix must special-case falsy."""
        sampler = _make_sampler()
        prompt = GenerateContentRequest(
            contents=[{"role": "user", "parts": [{"text": "hi"}]}]
        )
        results = await _run_sample_with_fake_lro(
            sampler,
            prompt,
            lro_response={
                "trainStepId": "",
                "tuningCandidates": [_MINIMAL_TUNING_CANDIDATE],
            },
        )
        assert results[0].train_step_id is None

    @pytest.mark.asyncio
    async def test_train_step_id_zero_string_becomes_int_zero_not_none(self):
        """0 is a valid train step id -- baseline-eval / step-0
        rollouts legitimately carry trainStepId=0. The wire-side
        ``"0"`` must coerce to Python ``int(0)``, NOT collapse to
        None. The pre-fix ``response.get(...) or None`` idiom would
        have wrongly returned None for numeric 0 (falsy); the fix must
        distinguish empty-string / missing (both -> None) from
        stringified-zero (-> int 0)."""
        sampler = _make_sampler()
        prompt = GenerateContentRequest(
            contents=[{"role": "user", "parts": [{"text": "hi"}]}]
        )
        results = await _run_sample_with_fake_lro(
            sampler,
            prompt,
            lro_response={
                "trainStepId": "0",
                "tuningCandidates": [_MINIMAL_TUNING_CANDIDATE],
            },
        )
        assert results[0].train_step_id == 0
        assert isinstance(results[0].train_step_id, int)


class TestSamplingContextPopulation:
    """VertexMsrlRestSampler must populate SamplingContext with the
    request-scoped metadata Sampler.sample_chat_completion needs to
    reconstitute a valid ChatCompletionResponse:

      * prompt_token_count from tuningCandidates[i].promptTokenCount
      * model_version from client.base_model (the tuning job's base)

    Fail-fast contract: both fields None would cause
    sample_chat_completion to raise ValueError. This test guards the
    populator so future refactors don't silently drop the plumbing.
    """

    @pytest.mark.asyncio
    async def test_prompt_token_count_and_model_version_populated(self):
        sampler = _make_sampler()
        # Simulate the client having a base_model (as if create_tuning_job
        # was called or wait_for_job_running refreshed it).
        sampler.client.base_model = "gemini-3.5-flash"
        prompt = GenerateContentRequest(
            contents=[{"role": "user", "parts": [{"text": "hi"}]}]
        )
        results = await _run_sample_with_fake_lro(
            sampler,
            prompt,
            lro_response={
                "trainStepId": "42",
                "tuningCandidates": [
                    {**_MINIMAL_TUNING_CANDIDATE, "promptTokenCount": 17}
                ],
            },
        )
        assert len(results) == 1
        assert results[0].context.prompt_token_count == 17
        assert results[0].context.model_version == "gemini-3.5-flash"

    @pytest.mark.asyncio
    async def test_prompt_token_count_none_when_absent(self):
        """No per-candidate promptTokenCount -> prompt_token_count is None
        (not silent 0). This surfaces a real backend bug rather than masking
        it."""
        sampler = _make_sampler()
        sampler.client.base_model = "gemini-3.5-flash"
        prompt = GenerateContentRequest(
            contents=[{"role": "user", "parts": [{"text": "hi"}]}]
        )
        candidate_without_prompt = {
            k: v
            for k, v in _MINIMAL_TUNING_CANDIDATE.items()
            if k != "promptTokenCount"
        }
        results = await _run_sample_with_fake_lro(
            sampler,
            prompt,
            lro_response={
                "trainStepId": "1",
                "tuningCandidates": [candidate_without_prompt],
            },
        )
        assert results[0].context.prompt_token_count is None

    @pytest.mark.asyncio
    async def test_model_version_none_when_base_model_unset(self):
        """Client with base_model=None -> context.model_version is None.
        Happens for the pre-existing-job code path if get_tuning_job
        never populated it (shouldn't in practice but guarded here)."""
        sampler = _make_sampler()
        # Default: base_model is None on a fresh client.
        assert sampler.client.base_model is None
        prompt = GenerateContentRequest(
            contents=[{"role": "user", "parts": [{"text": "hi"}]}]
        )
        results = await _run_sample_with_fake_lro(
            sampler,
            prompt,
            lro_response={
                "trainStepId": "1",
                "tuningCandidates": [_MINIMAL_TUNING_CANDIDATE],
            },
        )
        assert results[0].context.model_version is None
