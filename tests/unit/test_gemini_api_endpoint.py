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

"""Unit tests for `backends.gemini_api_endpoint.GeminiApiSampler`.

Focused on the SampleResult / SamplingContext fields the sampler
populates from a mocked generateContent response. Uses a fake httpx
client so no network I/O.

Also covers the src/-side ENDPOINTS registry: it holds the production
entries and exposes ``register_endpoint`` so callers can inject
additional hosts at import time without modifying ``src/``.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from backends.gemini_api_endpoint import (
    _DEFAULT_MAX_OUTPUT_TOKENS,
    ENDPOINT_HOSTS,
    KNOWN_ENVS,
    GeminiApiSampler,
    register_endpoint,
)


@pytest.fixture(autouse=True)
def _stub_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep this file hermetic, as its docstring claims.

    Construction resolves call credentials eagerly, so on a host with no
    ADC and no metadata server every sampler here fell back to the retry
    loop and only failed once the retry budget was spent -- an ambient
    dependency masquerading as an assertion failure.
    """
    monkeypatch.setattr(
        "backends.gemini_api_endpoint.get_call_credentials",
        lambda **_kw: MagicMock(valid=True, token="fake"),
    )


class _FakeResponse:
    def __init__(self, status_code: int, json_body: Dict[str, Any]):
        self.status_code = status_code
        self._json = json_body
        self.request: Any = None

    def json(self) -> Dict[str, Any]:
        return self._json

    @property
    def text(self) -> str:
        import json as _j

        return _j.dumps(self._json)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError(
                "boom",
                request=self.request,
                response=self,  # type: ignore[arg-type]
            )


class _FakeClient:
    """Minimal httpx.AsyncClient stand-in for GeminiApiSampler."""

    def __init__(self, response_json: Dict[str, Any]):
        self._response_json = response_json

    async def post(self, *args: Any, **kwargs: Any) -> _FakeResponse:
        return _FakeResponse(200, self._response_json)

    async def aclose(self) -> None:  # matches the sampler's cleanup path
        pass


def _make_sampler(response_json: Dict[str, Any]) -> GeminiApiSampler:
    return GeminiApiSampler(
        model="gemini-3.5-flash",
        project="test-project",
        env="prod",
        http_client=_FakeClient(response_json),  # type: ignore[arg-type]
    )


async def test_prompt_token_count_populated_on_sampling_context():
    """GeminiApiSampler must set SamplingContext.prompt_token_count from
    the raw usageMetadata.promptTokenCount returned by generateContent.

    Fail-fast contract: sample_chat_completion refuses to reconstitute
    without this field, so any sampler that dropped it would break
    OpenAI-shape output for that backend.
    """
    fake_response = {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "hi"}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 42,
            "candidatesTokenCount": 5,
            "totalTokenCount": 47,
        },
        "modelVersion": "gemini-3.5-flash",
        "responseId": "resp-123",
    }
    sampler = _make_sampler(fake_response)
    results = await sampler.sample_n("hello", num_samples=1)
    assert len(results) == 1
    assert results[0].context.prompt_token_count == 42


# ---------------------------------------------------------------------------
# Per-call max_generation_tokens override (Phase B)
# ---------------------------------------------------------------------------


def _make_max_tokens_sampler() -> GeminiApiSampler:
    return GeminiApiSampler(
        model="gemini-3.5-flash",
        project="test-project",
        env="prod",
        http_client=_FakeClient({}),  # type: ignore[arg-type]
    )


def test_sampler_has_no_max_generation_tokens_ctor_param():
    """The generation cap is per-call only -- no constructor override."""
    import inspect

    params = inspect.signature(GeminiApiSampler.__init__).parameters
    assert "max_generation_tokens" not in params


def test_build_body_uses_per_call_max_generation_tokens():
    """A per-call ``max_generation_tokens`` sets the request body's
    ``generationConfig.maxOutputTokens``."""
    sampler = _make_max_tokens_sampler()
    body = sampler._build_body("hi", num_samples=1, max_generation_tokens=99)
    assert body["generationConfig"]["maxOutputTokens"] == 99


def test_build_body_falls_back_to_module_default_when_none():
    """Omitting (``None``) the per-call value falls back to the module
    constant ``_DEFAULT_MAX_OUTPUT_TOKENS`` (no ctor override)."""
    sampler = _make_max_tokens_sampler()
    body_none = sampler._build_body("hi", num_samples=1, max_generation_tokens=None)
    assert (
        body_none["generationConfig"]["maxOutputTokens"] == _DEFAULT_MAX_OUTPUT_TOKENS
    )
    # Default argument (kwarg omitted entirely) matches the None case.
    body_default = sampler._build_body("hi", num_samples=1)
    assert (
        body_default["generationConfig"]["maxOutputTokens"]
        == _DEFAULT_MAX_OUTPUT_TOKENS
    )


async def test_sample_threads_per_call_max_generation_tokens():
    """``sample()`` threads a per-call value into the wire body."""
    captured: Dict[str, Any] = {}

    class _CapturingClient(_FakeClient):
        async def post(self, *args: Any, **kwargs: Any) -> _FakeResponse:
            captured["body"] = kwargs.get("json")
            return await super().post(*args, **kwargs)

    fake_response = {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "hi"}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
    }
    sampler = GeminiApiSampler(
        model="gemini-3.5-flash",
        project="test-project",
        env="prod",
        http_client=_CapturingClient(fake_response),  # type: ignore[arg-type]
    )
    await sampler.sample_n("hello", num_samples=1, max_generation_tokens=77)
    assert captured["body"]["generationConfig"]["maxOutputTokens"] == 77


# ---------------------------------------------------------------------------
# ENDPOINTS registry: prod-only default + register_endpoint extension hook
# ---------------------------------------------------------------------------


@pytest.fixture
def endpoints_snapshot():
    """Snapshot / restore ENDPOINT_HOSTS and KNOWN_ENVS so per-test
    registrations don't leak.

    ``register_endpoint`` mutates the module-level dicts in place;
    tests that add entries would otherwise contaminate other tests
    (and the src-only default assertion) via import-order coupling.
    """
    original_hosts = dict(ENDPOINT_HOSTS)
    original_envs = set(KNOWN_ENVS)
    try:
        yield
    finally:
        ENDPOINT_HOSTS.clear()
        ENDPOINT_HOSTS.update(original_hosts)
        KNOWN_ENVS.clear()
        KNOWN_ENVS.update(original_envs)


def test_default_endpoints_are_prod_only(endpoints_snapshot):
    """Only production entries are baked into ``src/``.

    Any other entry is supplied at import time via ``register_endpoint``
    and must NOT be hardcoded here.
    """
    assert KNOWN_ENVS == {"prod"}
    assert ENDPOINT_HOSTS[("prod", "global")] == "aiplatform.googleapis.com"
    assert ENDPOINT_HOSTS[("prod", "us-central1")] == (
        "us-central1-aiplatform.googleapis.com"
    )
    assert ENDPOINT_HOSTS[("prod", "us")] == "aiplatform.us.rep.googleapis.com"


def test_sampler_rejects_unregistered_env(endpoints_snapshot):
    """Constructing with an env not in ENDPOINTS raises ValueError with
    a message listing the registered set. This is the error a customer
    would hit if they passed a non-prod env name without an alternate
    distribution's register shim installed."""
    with pytest.raises(ValueError, match="Unknown env 'staging'"):
        GeminiApiSampler(
            model="gemini-3.5-flash",
            project="p",
            env="staging",
        )


def test_register_endpoint_extends_the_map(endpoints_snapshot):
    """After ``register_endpoint``, the new env resolves like a built-in."""
    register_endpoint(
        "sandbox-x",
        "sandbox-x-aiplatform.example.googleapis.com",
        "us-central1",
    )
    assert ("sandbox-x", "us-central1") in ENDPOINT_HOSTS
    # Sampler now accepts the new env and builds a URL against its host.
    sampler = GeminiApiSampler(
        model="gemini-3.5-flash",
        project="p",
        env="sandbox-x",
        location="us-central1",
    )
    assert sampler.host == "sandbox-x-aiplatform.example.googleapis.com"
    assert sampler.location == "us-central1"


def test_register_endpoint_is_idempotent(endpoints_snapshot):
    """Re-registering an env with the SAME ``(host, location)`` pair is
    a no-op. This lets multiple internal shims safely import each other
    without racing on registration order."""
    host = "sandbox-y-aiplatform.example.googleapis.com"
    location = "us-central1"
    register_endpoint("sandbox-y", host, location)
    # No exception on identical re-register.
    register_endpoint("sandbox-y", host, location)
    assert ENDPOINT_HOSTS[("sandbox-y", location)] == host


def test_register_endpoint_conflict_raises(endpoints_snapshot):
    """Re-registering an env with a DIFFERENT host for the same
    ``(env, location)`` pair raises ValueError. Env identities must not
    silently drift because two shims disagreed about the canonical hostname."""
    register_endpoint(
        "sandbox-z",
        "sandbox-z-aiplatform.example.googleapis.com",
        "us-central1",
    )
    with pytest.raises(ValueError, match="already registered"):
        register_endpoint(
            "sandbox-z",
            "different-host.example.googleapis.com",
            "us-central1",
        )
    # Original binding survived the failed re-register.
    assert ENDPOINT_HOSTS[("sandbox-z", "us-central1")] == (
        "sandbox-z-aiplatform.example.googleapis.com"
    )


@pytest.mark.parametrize(
    "env,host,location,match",
    [
        ("", "h", "l", "env must be non-empty"),
        ("e", "", "l", "host must be non-empty"),
        ("e", "h", "", "location must be non-empty"),
    ],
)
def test_register_endpoint_rejects_empty_fields(
    endpoints_snapshot, env, host, location, match
):
    """All three fields are required to build a working URL; empty
    values are unrecoverable misconfigurations, not runtime conditions."""
    with pytest.raises(ValueError, match=match):
        register_endpoint(env, host, location)
