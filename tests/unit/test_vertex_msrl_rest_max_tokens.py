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

"""Unit tests for per-call ``max_generation_tokens`` on VertexMsrlRestSampler.

``max_generation_tokens`` is a single-tier per-call ``sample()`` kwarg
(mirroring ``thinking_level``): the sampler has NO constructor override, so a
per-call value wins and ``None`` falls back to the module constant
``_DEFAULT_MAX_OUTPUT_TOKENS``. These tests exercise the pure body-builder
``_build_body`` directly (no network / LRO machinery) to assert that behavior
on the request body's ``generationConfig.maxOutputTokens``.

Contract:
  * ``max_generation_tokens=None`` (or omitted) -> ``_DEFAULT_MAX_OUTPUT_TOKENS``.
  * ``max_generation_tokens=<int>`` -> that per-call value wins.

The value lands at
``body["contentGenerationParameters"]["generationConfig"]["maxOutputTokens"]``
(the tuning-scope RPC nests generationConfig under
contentGenerationParameters).
"""

from __future__ import annotations

from google.cloud.aiplatform_v1.types import GenerateContentRequest

from backends.vertex_msrl_rest import (
    _DEFAULT_MAX_OUTPUT_TOKENS,
    VertexMsrlRestClient,
    VertexMsrlRestSampler,
)


def _make_sampler():
    """Return a sampler wired to a real (unstarted) client instance.

    ``_build_body`` reads only the per-call arg (falling back to
    ``_DEFAULT_MAX_OUTPUT_TOKENS``) and ``self.client.tuning_job_name``; no
    network is touched.
    """
    client = VertexMsrlRestClient(
        project="test-project",
        location="us-central1",
        endpoint="test.example.com",
        tuning_job_id="12345",
    )
    return VertexMsrlRestSampler(client)


def _max_output_tokens(body: dict) -> int:
    return body["contentGenerationParameters"]["generationConfig"]["maxOutputTokens"]


def _prompt() -> GenerateContentRequest:
    return GenerateContentRequest(
        contents=[{"role": "user", "parts": [{"text": "hi"}]}]
    )


def test_sampler_has_no_max_generation_tokens_ctor_param():
    """The generation cap is per-call only -- no constructor override."""
    import inspect

    params = inspect.signature(VertexMsrlRestSampler.__init__).parameters
    assert "max_generation_tokens" not in params


def test_none_falls_back_to_module_default():
    sampler = _make_sampler()
    body = sampler._build_body(_prompt(), 1, max_generation_tokens=None)
    assert _max_output_tokens(body) == _DEFAULT_MAX_OUTPUT_TOKENS


def test_omitted_falls_back_to_module_default():
    """Backward-compat: not passing the kwarg reproduces the 32768 default."""
    sampler = _make_sampler()
    body = sampler._build_body(_prompt(), 1)
    assert _max_output_tokens(body) == _DEFAULT_MAX_OUTPUT_TOKENS


def test_per_call_value_overrides_default():
    sampler = _make_sampler()
    body = sampler._build_body(_prompt(), 1, max_generation_tokens=99)
    assert _max_output_tokens(body) == 99
