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

"""Sampler for the standard Vertex Gemini API (`generateContent`).

Unlike RL tuning-scope backends (which drive generation via a long-running
operation that must be polled), this sampler calls the ordinary
`generateContent` endpoint, which returns candidates synchronously.

Preserves ``thought`` / ``thoughtSignature`` parts in the completion so
multi-step function-calling rollouts round-trip signatures correctly
across turns.

Endpoint/location:
  * Gemini 3.x models on prod are served ONLY from the GLOBAL endpoint
    (``aiplatform.googleapis.com`` + ``locations/global``); the regional
    prod endpoint returns 404. Use ``env='prod', location='global'``
    for those.
  * ``env`` selects the data realm;
    ``location`` selects the region (global / us-central1 / us). The
    two axes together determine the API hostname via
    :data:`ENDPOINT_HOSTS`.
  * Additional (env, location) -> host entries can be injected via
    :func:`register_endpoint`; the registry holds the production
    entries by default.
"""

import asyncio
import json
import logging
import os
import random
import time
from collections.abc import Sequence
from typing import Any, Dict, List, Optional

import google.auth  # pytype: disable=import-error
import google.auth.transport.requests  # pytype: disable=import-error
import httpx
from google.cloud.aiplatform_v1.types import Candidate, GenerateContentRequest

from backends._proto_utils import _json_dict_to_candidate
from backends.prompt_coerce import to_wire_dict
from backends.retry_profile import DEFAULT_SYNC_RETRY, SyncRetryProfile
from core.protocols import Sampler
from core.sample_result import (
    SampleResult,
    SamplingContext,
    TuningCandidate,
    with_raw_call_uris,
)

from utils.auth_utils import get_call_credentials
from utils.http_utils import build_http_client
from utils.raw_capture import CaptureContext, dump_raw_fanout

logger = logging.getLogger(__name__)

# (env, location) -> host.  ``env`` selects the data realm; ``location``
# selects the region (global / us-central1 / us). The host depends on
# BOTH axes -- prod-global is a different host from prod-us-central1.
# Alternate-realm entries can be injected via ``register_endpoint``.
ENDPOINT_HOSTS: Dict["tuple[str, str]", str] = {
    ("prod", "global"): "aiplatform.googleapis.com",
    ("prod", "us-central1"): "us-central1-aiplatform.googleapis.com",
    ("prod", "us"): "aiplatform.us.rep.googleapis.com",
}

# Set of registered env (realm) names — for error messages.
KNOWN_ENVS: set[str] = {k[0] for k in ENDPOINT_HOSTS}

# Fallback maxOutputTokens when a caller does not pass a per-call
# ``max_generation_tokens`` to ``sample()``. The generation cap is a
# single-tier per-call knob (mirroring ``thinking_level``): there is no
# sampler-constructor override, so this module constant is the sole
# fallback and preserves the historical 32768 default on the wire.
_DEFAULT_MAX_OUTPUT_TOKENS = 32768


def register_endpoint(env: str, host: str, location: str) -> None:
    """Register a ``host`` for the ``(env, location)`` pair.

    Idempotent for the same triple; raises ``ValueError`` when a
    different host is already registered for the pair, and on empty inputs.
    """
    if not env:
        raise ValueError("env must be non-empty")
    if not host:
        raise ValueError("host must be non-empty")
    if not location:
        raise ValueError("location must be non-empty")
    key = (env, location)
    existing = ENDPOINT_HOSTS.get(key)
    if existing is not None and existing != host:
        raise ValueError(
            f"(env={env!r}, location={location!r}) already registered "
            f"as host={existing!r}; cannot re-register to host={host!r}."
        )
    ENDPOINT_HOSTS[key] = host
    KNOWN_ENVS.add(env)


# Transient server / rate-limit statuses.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

# Retry-log throttle: emit the full line at most once per interval (per
# sampler); suppressed retries are counted into the next one. Mirrors
# vertex_msrl_rest's _RETRY_LOG_INTERVAL -- at a 48-wide fan-out an
# unthrottled INFO line per retry turns a quota blip into a log flood.
_RETRY_LOG_INTERVAL = 30.0


def _extract_endpoint_id(spec: str) -> str:
    """Return the trailing numeric id from an endpoint spec.

    Accepts either a bare id or a full ``projects/.../endpoints/<id>``
    resource name. Raises ``ValueError`` otherwise.
    """
    spec = spec.strip()
    if "/" not in spec:
        if not spec.isdigit():
            raise ValueError(f"endpoint_id {spec!r} is not a numeric id")
        return spec
    marker = "/endpoints/"
    idx = spec.rfind(marker)
    if idx < 0:
        raise ValueError(
            f"endpoint_id {spec!r} looks like a resource name but has no "
            "'/endpoints/' segment"
        )
    tail = spec[idx + len(marker) :].split("/")[0].split(":")[0]
    if not tail.isdigit():
        raise ValueError(
            f"endpoint_id {spec!r} resource name trailing segment "
            f"{tail!r} is not numeric"
        )
    return tail


class GeminiApiSampler(Sampler):
    """Samples completions from the standard Vertex `generateContent` API."""

    def __init__(
        self,
        model: str,
        project: str,
        env: str = "prod",
        location: str = "global",
        include_thoughts: Optional[bool] = None,
        timeout: float = 600.0,
        service_account: Optional[str] = None,
        use_user_credentials: bool = False,
        api_version: str = "v1beta1",
        http_client: Optional[httpx.AsyncClient] = None,
        retry: SyncRetryProfile = DEFAULT_SYNC_RETRY,
        endpoint_id: Optional[str] = None,
    ):
        key = (env, location)
        if key not in ENDPOINT_HOSTS:
            known_for_env = sorted(loc for (e, loc) in ENDPOINT_HOSTS if e == env)
            if known_for_env:
                raise ValueError(
                    f"Unknown location {location!r} for env {env!r}; "
                    f"registered locations: {known_for_env}"
                )
            raise ValueError(
                f"Unknown env {env!r}; registered envs: {sorted(KNOWN_ENVS)}"
            )
        self.env = env
        self.host = ENDPOINT_HOSTS[key]
        self.location = location
        self.model = model
        # Public alias read by concrete Agent subclasses to populate
        # ATIF ``Trajectory.agent.model_name``. Kept as a plain attribute
        # so samplers without a meaningful "model name" can omit it and
        # let agents fall back to the ``"unknown"`` sentinel.
        self.model_name = model
        self.project = project
        # None -> omit thinkingConfig entirely; True/False -> set includeThoughts.
        self.include_thoughts = include_thoughts
        # Retry policy for this synchronous backend: a wall-clock deadline
        # with an attempt count as the secondary cap. Set at construction
        # from the backend config.
        self.retry = retry
        # Retry-log throttle state; see _should_log_retry.
        self._retry_log_last = 0.0
        self._retry_log_suppressed = 0
        # When set, target a deployed endpoint (same request body,
        # different URL path). Accepts a bare numeric id or a full
        # resource name.
        self.endpoint_id = _extract_endpoint_id(endpoint_id) if endpoint_id else None
        if self.endpoint_id:
            self.url = (
                f"https://{self.host}/{api_version}/projects/{self.project}"
                f"/locations/{self.location}/endpoints/{self.endpoint_id}"
                f":generateContent"
            )
        else:
            self.url = (
                f"https://{self.host}/{api_version}/projects/{self.project}"
                f"/locations/{self.location}/publishers/google/models"
                f"/{self.model}:generateContent"
            )

        self._service_account = service_account
        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        if use_user_credentials:
            # Use end-user ADC directly (no SA impersonation) so the
            # request is attributed to the developer's account. Only
            # works against production endpoints; other realms reject
            # end-user credentials.
            self._creds, _ = google.auth.default(scopes=scopes)
        else:
            self._creds = get_call_credentials(
                target_service_account=service_account, scopes=scopes
            )
        self._auth_lock = asyncio.Lock()
        self._client = (
            http_client
            if http_client is not None
            else build_http_client(read_timeout=timeout)
        )

    async def _get_headers(self) -> Dict[str, str]:
        async with self._auth_lock:
            if not self._creds.valid:
                await asyncio.to_thread(
                    self._creds.refresh, google.auth.transport.requests.Request()
                )
            token = self._creds.token
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            # Quota/billing project (the impersonated SA's home project
            # differs from the serving project).
            "x-goog-user-project": self.project,
        }

    def _build_body(
        self,
        prompt: GenerateContentRequest | str,
        num_samples: int,
        max_generation_tokens: int | None = None,
    ) -> Dict[str, Any]:
        # Marshal to camelCase JSON; str prompts get canonical single-turn wrap.
        wire = to_wire_dict(prompt)

        gen_cfg = dict(wire.get("generationConfig") or {})
        # Single-tier per-call cap: the passed value wins; None falls back to
        # the module constant (no sampler-level override; mirrors
        # thinking_level). This sampler always sets maxOutputTokens
        # (assignment, not setdefault).
        gen_cfg["maxOutputTokens"] = (
            max_generation_tokens
            if max_generation_tokens is not None
            else _DEFAULT_MAX_OUTPUT_TOKENS
        )
        gen_cfg["candidateCount"] = num_samples
        if self.include_thoughts is not None:
            gen_cfg["thinkingConfig"] = {"includeThoughts": bool(self.include_thoughts)}

        body: Dict[str, Any] = {
            "contents": wire["contents"],
            "generationConfig": gen_cfg,
        }
        if wire.get("tools"):
            body["tools"] = wire["tools"]
        if wire.get("toolConfig"):
            body["toolConfig"] = wire["toolConfig"]
        if wire.get("systemInstruction"):
            body["systemInstruction"] = wire["systemInstruction"]
        return body

    async def sample_n(
        self,
        prompt: GenerateContentRequest | str,
        num_samples: int = 1,
        thinking_level: str | None = None,
        max_generation_tokens: int | None = None,
        capture_ctxs: Sequence[CaptureContext | None] | None = None,
    ) -> List[SampleResult]:
        """Sample completions; returns one :class:`SampleResult` per candidate.

        On unrecoverable failure returns failsafe empty results so the
        rollout terminates gracefully instead of crashing.

        ``thinking_level`` (when set) is stamped onto the prompt's
        ``generationConfig.thinkingConfig.thinkingLevel`` via
        ``backends.vertex_thinking.apply_thinking_level`` before building the
        request body. Vertex generateContent honors this field on Gemini 2.5+
        models. Empty / None -> omit thinkingConfig entirely (server default).

        ``max_generation_tokens`` (when set) overrides the constructor
        default for this call's ``generationConfig.maxOutputTokens``;
        ``None`` (default) falls back to ``_DEFAULT_MAX_OUTPUT_TOKENS``.

        ``capture_ctxs`` (if provided) is a per-candidate list of length
        ``num_samples``; each candidate's raw wire I/O is dumped under
        its ctx's ``trial_dir`` (or skipped if ctx is None). This enables
        the raw-LLM ledger to carry per-rollout provenance even when
        ``num_samples > 1`` fans out over one wire call. See the ABC.
        """
        if thinking_level:
            from backends.vertex_thinking import apply_thinking_level

            prompt = apply_thinking_level(prompt, thinking_level)
        body = self._build_body(prompt, num_samples, max_generation_tokens)
        if os.getenv("MSRL_DEBUG_TRAJECTORY") == "1":
            logger.info(
                "generateContent request:\n%s",
                json.dumps(body.get("generationConfig", {}), indent=2),
            )

        # Normalize the per-candidate capture-ctx list: None -> list of
        # Nones of length num_samples (skips capture for every candidate);
        # otherwise assert length matches num_samples. Fails loudly on
        # mismatch instead of silently truncating/padding — the caller
        # (SingleStepAgent.group_rollout) builds the list from
        # rollout_idx=0..num_samples-1 and any drift there is a real bug.
        ctxs: list[CaptureContext | None] = (
            list(capture_ctxs) if capture_ctxs is not None else [None] * num_samples
        )
        # ValueError, not assert: assert is stripped under `python -O`
        # and this is a contract check the caller must not silently
        # violate.
        if len(ctxs) != num_samples:
            raise ValueError(
                f"capture_ctxs length {len(ctxs)} does not match num_samples {num_samples}"
            )

        t0 = time.monotonic()
        data = await self._post_with_retries(body)
        duration_s = time.monotonic() - t0

        if data is None:
            # POST failed after retries. Each rollout ctx gets its own
            # error-status ledger blob recording the failed wire call.
            error_payload = {"__error__": "post_failed_after_retries"}
            raw_uris = await dump_raw_fanout(
                ctxs,
                count=num_samples,
                request=body,
                response=error_payload,
                status="error",
                duration_s=duration_s,
            )
            failsafes = [self._empty_result(prompt) for _ in range(num_samples)]
            return with_raw_call_uris(failsafes, raw_uris)

        candidates = data.get("candidates") or []
        if not candidates:
            logger.warning("generateContent returned no candidates: %s", data)
            raw_uris = await dump_raw_fanout(
                ctxs,
                count=num_samples,
                request=body,
                response=data,
                status="empty",
                duration_s=duration_s,
            )
            failsafes = [self._empty_result(prompt) for _ in range(num_samples)]
            return with_raw_call_uris(failsafes, raw_uris)

        # generateContent reports token counts only at the request level
        # via ``usageMetadata`` (no per-candidate breakdown in the public
        # API). We divide equally; deterministic and preserves the
        # aggregate exactly, but approximate when num_samples > 1.
        # Consumers needing per-candidate accuracy must use num_samples=1.
        #
        # Wire semantics:
        #   * ``candidatesTokenCount`` = decoded OUTPUT tokens only
        #     (excludes thinking); omitted when the model produced zero
        #     output (e.g. all budget went to thinking).
        #   * ``thoughtsTokenCount``   = decoded thinking tokens;
        #     omitted when thinking is disabled.
        #
        # ``TuningCandidate.generation_token_count`` is TOTAL generated
        # tokens (thinking + non-thinking):
        #     generation_token_count = candidatesTokenCount + thoughtsTokenCount
        # An "all-thinking, no-output" response is legitimate; the sum
        # correctly reflects the model's generation cost.
        #
        # We fail-fast only when a candidate HAS content but usage
        # reports neither count — that combination cannot be legitimate.
        usage = data.get("usageMetadata") or {}
        agg_out_tokens = usage.get("candidatesTokenCount")
        agg_thoughts_tokens = usage.get("thoughtsTokenCount")
        if agg_out_tokens is None and agg_thoughts_tokens is None:
            # Any visible content -> unattributable -> raise.
            # All-empty -> legitimate (0, 0).
            def _has_content(c: Dict[str, Any]) -> bool:
                content = (c or {}).get("content") or {}
                parts = content.get("parts") or []
                return any(
                    (p or {}).get("text") or (p or {}).get("functionCall")
                    for p in parts
                )

            if any(_has_content(c) for c in candidates):
                raise ValueError(
                    "generateContent response has candidate content but is "
                    "missing BOTH 'usageMetadata.candidatesTokenCount' AND "
                    "'usageMetadata.thoughtsTokenCount'; cannot attribute "
                    f"per-candidate generation_token_count. Response keys: "
                    f"{sorted(data.keys())}, usageMetadata keys: "
                    f"{sorted(usage.keys())}"
                )
        agg_out_tokens = int(agg_out_tokens or 0)
        agg_thoughts_tokens = int(agg_thoughts_tokens or 0)
        agg_gen_tokens = agg_out_tokens + agg_thoughts_tokens

        # Integer division; distribute remainder to first candidates so
        # the aggregate is preserved exactly.
        n_cands = len(candidates)
        base_gen, rem_gen = divmod(agg_gen_tokens, n_cands)
        base_thoughts, rem_thoughts = divmod(agg_thoughts_tokens, n_cands)

        # Request-level prompt tokens shared across all candidates.
        # Surfaced onto SamplingContext for sample_chat_completion.
        _raw_prompt_tokens = usage.get("promptTokenCount")
        _prompt_tokens: Optional[int] = (
            int(_raw_prompt_tokens) if _raw_prompt_tokens is not None else None
        )
        if _prompt_tokens is None:
            # Unlike the generation counts, there is no legitimate
            # response shape with candidates and no prompt: every request
            # has a prompt. Accepting None silently disables in-loop
            # compaction (which refuses to guess) and blanks the
            # context-size metrics, so surface it as a sampling failure
            # -- but return the failsafe rather than raising, which would
            # lose the whole task.
            logger.warning(
                "generateContent response has candidates but no "
                "'usageMetadata.promptTokenCount'; usageMetadata keys: %s",
                sorted(usage.keys()),
            )
            raw_uris = await dump_raw_fanout(
                ctxs,
                count=num_samples,
                request=body,
                response=data,
                status="error",
                duration_s=duration_s,
            )
            failsafes = [
                self._empty_result(prompt, extras={"is_malformed_usage": True})
                for _ in range(num_samples)
            ]
            return with_raw_call_uris(failsafes, raw_uris)

        # Per-candidate ledger: write ONE blob per candidate, each
        # containing the wire envelope (usage / modelVersion / responseId /
        # etc.) alongside JUST that one candidate. Slicing keeps blob size
        # bounded (~1x per rollout instead of Nx), and preserves enough
        # context that a replay tool can reconstruct the original wire
        # response by re-assembling the N slices for a group.
        envelope = {k: v for k, v in data.items() if k != "candidates"}

        # Pre-compute per-candidate response slices (synchronous, no I/O)
        # so the fan-out below writes them without an inline loop body.
        # Slice ordering matches candidate ordering; both are indexed by i.
        cand_dicts: List[Dict[str, Any]] = [cand or {} for cand in candidates]
        cand_response_slices: List[Dict[str, Any]] = [
            {**envelope, "candidates": [cand_dicts[i]]} for i in range(len(cand_dicts))
        ]

        # Fan out one per-candidate ledger write per candidate. ``ctxs``
        # is guaranteed length == num_samples but candidates may be
        # shorter/longer in pathological server responses; the fan-out
        # pairs each slice with the same-index ctx (or no ctx on
        # overrun), which keeps the mapping consistent with the prior
        # serial loop.
        raw_uris: List[str | None] = await dump_raw_fanout(
            ctxs,
            count=len(cand_dicts),
            request=body,
            response=cand_response_slices,
            status="ok",
            duration_s=duration_s,
        )

        results: List[SampleResult] = []
        for i, cand in enumerate(cand_dicts):
            candidate_proto = _json_dict_to_candidate(cand)
            per_gen = base_gen + (1 if i < rem_gen else 0)
            per_thoughts = base_thoughts + (1 if i < rem_thoughts else 0)
            results.append(
                SampleResult(
                    tuning_candidate=TuningCandidate(
                        candidate_id=str(cand.get("index", i)),
                        candidate=candidate_proto,
                        generation_token_count=per_gen,
                        thoughts_token_count=per_thoughts,
                    ),
                    context=SamplingContext(
                        # Typed prompt (not the wire body); callers can
                        # re-marshal via ``to_wire_dict`` if needed.
                        prompt=prompt,
                        response_id=data.get("responseId"),
                        model_version=data.get("modelVersion"),
                        prompt_token_count=_prompt_tokens,
                        raw_call_uri=raw_uris[i],
                        # No operation_id, no is_prompt_too_long for gemini_api.
                    ),
                    train_step_id=None,  # no trainStepId concept here
                )
            )
        return results

    def _should_log_retry(self) -> tuple[bool, int]:
        """Throttle retry logging to one full line per ``_RETRY_LOG_INTERVAL``."""
        now = time.monotonic()
        if now - self._retry_log_last >= _RETRY_LOG_INTERVAL:
            suppressed = self._retry_log_suppressed
            self._retry_log_last = now
            self._retry_log_suppressed = 0
            return True, suppressed
        self._retry_log_suppressed += 1
        return False, 0

    def _log_retry(
        self,
        attempt: int,
        reason: object,
        backoff: float,
        elapsed: float,
    ) -> None:
        """Announce a retry at INFO, throttled.

        INFO rather than DEBUG because the two failure shapes are otherwise
        indistinguishable in a run log: the only line a retried-then-exhausted
        call emits is the same terminal error a never-retried call emits.
        """
        emit, suppressed = self._should_log_retry()
        if not emit:
            return
        logger.info(
            "generateContent attempt %d/%d failed after %.0fs%s (%s); "
            "retrying in ~%.0fs%s",
            attempt,
            self.retry.max_attempts,
            elapsed,
            (
                ""
                if self.retry.deadline_s is None
                else f"/{self.retry.deadline_s:.0f}s"
            ),
            reason,
            backoff,
            (
                f" (+{suppressed} more retries suppressed in the last "
                f"~{_RETRY_LOG_INTERVAL:.0f}s)"
                if suppressed
                else ""
            ),
        )

    async def _post_with_retries(
        self, body: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """POST until it succeeds, attempts run out, or the deadline passes.

        The attempt count and the backoff schedule come from
        ``self.retry``; with the default profile this is exactly the
        previous loop (10 attempts, ``2 → 30 s`` jittered +/-20%).

        ``deadline_s`` is opt-in and ``None`` by default, so nothing stops
        on wall-clock unless a config asks for it. When it is set, the wait
        before an attempt is clipped to what remains -- sleeping past the
        stop only adds latency to a call already given up on.
        """
        start = time.monotonic()
        retry = self.retry
        for attempt in range(1, retry.max_attempts + 1):
            try:
                resp = await self._client.post(
                    self.url, json=body, headers=await self._get_headers()
                )
                if resp.status_code in _RETRYABLE_STATUS:
                    raise httpx.HTTPStatusError(
                        f"retryable status {resp.status_code}",
                        request=resp.request,
                        response=resp,
                    )
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as e:
                status = e.response.status_code if e.response is not None else None
                if status not in _RETRYABLE_STATUS:
                    body_text = e.response.text if e.response is not None else ""
                    logger.error(
                        "generateContent failed (HTTP %s): %s", status, body_text[:500]
                    )
                    return None
                reason: object = f"HTTP {status}"
                detail = e
            except Exception as e:  # noqa: BLE001 - transport hiccups, retry
                reason = e
                detail = e

            elapsed = time.monotonic() - start
            deadline = retry.deadline_s
            remaining = None if deadline is None else deadline - elapsed
            if attempt >= retry.max_attempts or (
                remaining is not None and remaining <= 0
            ):
                logger.error(
                    "generateContent failed after %d attempt(s) over %.0fs "
                    "(caps: %d attempts, %s): %s",
                    attempt,
                    elapsed,
                    retry.max_attempts,
                    "no deadline" if deadline is None else f"{deadline:.0f}s",
                    detail,
                )
                return None
            backoff = retry.backoff.delay_for(attempt)
            if remaining is not None:
                backoff = min(backoff, remaining)
            self._log_retry(attempt, reason, backoff, elapsed)
            # Sleep the interval that was reported, so the log does not promise
            # one wait and take another.
            await asyncio.sleep(backoff)
        return None

    def _empty_result(
        self,
        prompt: GenerateContentRequest | str,
        extras: Optional[Dict[str, Any]] = None,
    ) -> SampleResult:
        """Failsafe SampleResult; consumers detect this via
        ``result.tuning_candidate.candidate_id == ""``.

        ``extras`` currently only forwards ``is_malformed_usage`` onto
        ``SamplingContext``.
        """
        return SampleResult(
            tuning_candidate=TuningCandidate(
                candidate_id="",
                candidate=Candidate(),
                generation_token_count=0,
                thoughts_token_count=0,
            ),
            context=SamplingContext(
                prompt=prompt,
                is_malformed_usage=bool(
                    (extras or {}).get("is_malformed_usage", False)
                ),
            ),
            train_step_id=None,
        )

    async def aclose(self) -> None:
        await self._client.aclose()
