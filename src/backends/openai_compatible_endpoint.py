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

"""Sampler for any server exposing OpenAI ``/v1/chat/completions``.

For self-hosted OpenAI-compatible stacks (SGLang, vLLM, ...). The framework's
``Sampler`` contract has two entry points, and this sampler serves both:

* ``sample_n`` (Gemini-shaped callers: ``GenerateContentRequest`` in,
  ``Candidate`` out). ``OpenAICompatibleSampler`` subclasses
  ``GeminiApiSampler`` and converts only at the wire layer with
  ``backends.chat_completion.convert_gemini_request_to_openai`` /
  ``convert_openai_response_to_gemini``, so the parent's ``sample_n`` body
  (candidate fan-out, token attribution, raw capture, failsafe results) runs
  unchanged. ``thinking_level`` is ignored (chat completions has no such
  knob), with a one-time warning. Tool calls replayed in the history are sent
  with the argument strings the server returned (``_remember_tool_call_args``).
* ``sample_chat_completion`` (OpenAI-shaped callers). Sends the caller's body
  straight to the server instead of the ABC default, which would convert
  OpenAI -> Gemini -> OpenAI and back (lossy: history ``reasoning_content``
  is dropped and ``response_format`` is rejected). It does not go through
  ``sample_n``: it posts directly (same auth and retries), captures the body
  actually sent and the raw OpenAI response, and builds its ``SampleResult``s
  from the response's choices. ``thinking_blocks`` on assistant history turns
  are sent as ``reasoning_content``; the reply's ``reasoning_content`` comes
  back as ``thinking_blocks`` (and in ``SamplingContext.thinking_blocks``,
  which multi-turn callers replay).

Auth is optional: with ``token_command`` set, the stdout of that command is
sent as a Bearer token (for example ``gcloud auth print-identity-token`` for
a Cloud Run service behind IAM). The token is cached until shortly before its
JWT ``exp``, minted outside the parent's retry loop (so an auth failure raises
instead of turning into empty results), retried on transient failures,
remembered briefly on persistent ones, and dropped on a 401 (the call is then
retried once with a fresh token) unless it was minted less than
``_TOKEN_MIN_REUSE_S`` ago, so a persistently rejected token (e.g. wrong
audience) does not spawn the token command on every call. All attempts of one call's retry loop reuse
the headers minted for that call, so a refresh falling due mid-retry cannot
fail inside the loop either.
"""

from __future__ import annotations

import asyncio
import base64
import contextvars
import dataclasses
import errno
import hashlib
import json
import logging
import shlex
import time
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any, Dict, List, Optional

import httpx
from google.cloud.aiplatform_v1.types import GenerateContentRequest

from backends.chat_completion import (
    ChatCompletionResponse,
    convert_gemini_request_to_openai,
    convert_openai_response_to_gemini,
    reasoning_content_to_thinking_blocks,
    thinking_blocks_to_reasoning_content,
)
from backends.gemini_api_endpoint import (
    GeminiApiSampler,
    _DEFAULT_MAX_OUTPUT_TOKENS,
    _json_dict_to_candidate,
)
from backends.prompt_coerce import to_wire_dict
from backends.retry_profile import DEFAULT_SYNC_RETRY, SyncRetryProfile
from core.sample_result import (
    ChatCompletionSampleResult,
    SampleResult,
    SamplingContext,
    TuningCandidate,
    with_raw_call_uris,
)
from utils.http_utils import build_http_client
from utils.raw_capture import CaptureContext, client_request_scope, dump_raw_fanout

logger = logging.getLogger(__name__)

# Refresh the cached token this long before its ``exp`` claim.
_TOKEN_REFRESH_MARGIN_S = 300.0
# Used when the token's ``exp`` claim cannot be decoded.
_TOKEN_FALLBACK_TTL_S = 45 * 60.0
# Never re-mint more often than this, even for a token already inside the
# refresh margin.
_TOKEN_MIN_REUSE_S = 30.0
# Upper bound on the token command (e.g. gcloud blocking on a reauth prompt).
_TOKEN_COMMAND_TIMEOUT_S = 60.0
# Transient (non-timeout) token-command failures: attempts and linear backoff.
_TOKEN_MINT_ATTEMPTS = 3
_TOKEN_MINT_BACKOFF_S = 2.0
# After a token-command timeout or missing binary, fail fast for this long.
_TOKEN_TIMEOUT_BLOCK_S = 60.0
# After the last retry of any other token-command failure, fail fast this long.
_TOKEN_FAILURE_BLOCK_S = 15.0
# Consecutive copies of ``degenerate_marker`` that make a reply degenerate.
# Observed degenerate replies hold tens of thousands; a model merely
# mentioning the marker in text does not repeat it this often.
_DEGENERATE_RUN = 16
# Entries in the tool-call argument memory (LRU; two per remembered call). A
# live rollout re-sends, and so refreshes, all its calls every turn, so only
# finished rollouts' entries age out and the cap only bounds a run's total.
# Measured: <=158 calls per rollout (~1.6 KB of arguments each), so this covers
# ~800 concurrent rollouts, in ~250 MB if ever full.
_TOOL_ARGS_MEMORY_MAX = 262_144


def _hash_message(hasher: Any, message: Dict[str, Any]) -> None:
    hasher.update(json.dumps(message, sort_keys=True).encode())
    hasher.update(b"\0")


def _digest(tag: bytes, *parts: bytes | str) -> bytes:
    h = hashlib.blake2b(tag, digest_size=16)
    for part in parts:
        h.update(part if isinstance(part, bytes) else part.encode())
    return h.digest()


def _tool_call_key(name: Any, arguments: Any) -> Optional[str]:
    """A tool call's identity, blind to key order and ``3000`` vs ``3000.0``.

    Those are what the history's protobuf ``Struct`` changes, so a reply's call
    and its replay get the same key. None unless ``arguments`` is a JSON object
    string.
    """
    if not isinstance(arguments, str):
        return None
    try:
        args = json.loads(arguments, parse_int=float)
    except json.JSONDecodeError:
        return None
    return json.dumps([name, args], sort_keys=True) if isinstance(args, dict) else None


@dataclasses.dataclass
class _CallAuth:
    """Auth state of one ``_post_authed`` call."""

    headers: Dict[str, str]
    # Set by the response hook when the server rejects ``headers`` with 401.
    rejected: bool = False


# Per-call auth: ``_get_headers`` inside the parent's retry loop returns the
# headers minted for this call, and the response hook flags a 401 on them.
# A ContextVar so concurrent calls on one sampler don't share it; httpx runs
# response hooks in the task that sent the request.
_CALL_AUTH: contextvars.ContextVar[Optional[_CallAuth]] = contextvars.ContextVar(
    "openai_compatible_call_auth", default=None
)


class _TokenCommandTimeout(RuntimeError):
    """The token command did not finish within ``_TOKEN_COMMAND_TIMEOUT_S``."""


class _TokenCommandMissing(RuntimeError):
    """The token command's executable is missing or cannot run (permanent)."""


def _jwt_exp(token: str) -> Optional[float]:
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return float(json.loads(base64.urlsafe_b64decode(payload))["exp"])
    except Exception:  # noqa: BLE001 - opaque token -> fallback TTL
        return None


class OpenAICompatibleSampler(GeminiApiSampler):
    """Samples a model served behind an OpenAI-compatible chat-completions API."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 600.0,
        token_command: Optional[str] = None,
        http_client: Optional[httpx.AsyncClient] = None,
        retry: SyncRetryProfile = DEFAULT_SYNC_RETRY,
        degenerate_marker: Optional[str] = None,
        degenerate_retries: int = 1,
    ):
        """``degenerate_marker`` (opt-in, off by default): a token some servers
        emit in a run when generation degenerates (e.g. ``"[PAD]"``). A reply
        whose reasoning or content holds ``_DEGENERATE_RUN`` consecutive
        copies is re-requested up to ``degenerate_retries`` times, then the
        call fails like a transport failure instead of returning the run as
        output."""
        # Deliberately NOT calling GeminiApiSampler.__init__: it resolves a
        # Vertex host and builds Vertex credentials, neither of which apply.
        # Set exactly the attributes the inherited sample_n /
        # _post_with_retries / retry-log helpers read. If the parent grows a
        # new one, the fake-transport sampler tests fail.
        self.base_url = base_url.rstrip("/")
        self.url = f"{self.base_url}/v1/chat/completions"
        self.model = model
        self.model_name = model
        self.include_thoughts = None
        self.retry = retry
        if token_command is not None and not shlex.split(token_command):
            raise ValueError(f"token_command {token_command!r} is empty.")
        if degenerate_retries < 0:
            raise ValueError(f"degenerate_retries must be >= 0, got {degenerate_retries}")
        self._degenerate_marker = degenerate_marker or None
        self._degenerate_retries = degenerate_retries
        self._retry_log_last = 0.0
        self._retry_log_suppressed = 0
        self._token_command = token_command
        self._token: Optional[str] = None
        self._token_refresh_at = 0.0
        # When the current token was minted; a 401 re-mints only once the
        # token is older than _TOKEN_MIN_REUSE_S.
        self._last_mint_at = 0.0
        # _last_mint_at of the token whose throttled 401 was last logged.
        self._throttle_logged_mint_at: Optional[float] = None
        # After a final mint failure (timeout / missing binary: 60s; other
        # failures after retries: 15s), fail fast until this time instead of
        # letting every caller queued on the lock repeat the attempt.
        self._mint_blocked_until = 0.0
        self._mint_block_reason = ""
        self._warned_thinking_level = False
        self._warned_unreported_reasoning_tokens = False
        # Digest of (conversation, call) or of the call -> argument string as
        # the server returned it. See _remember_tool_call_args.
        self._tool_args: OrderedDict[bytes, str] = OrderedDict()
        self._tool_args_misses = 0
        self._auth_lock = asyncio.Lock()
        self._client = (
            http_client if http_client is not None else build_http_client(read_timeout=timeout)
        )
        self._client.event_hooks.setdefault("response", []).append(self._on_response)

    # ---- auth ----------------------------------------------------------------

    async def _on_response(self, response: httpx.Response) -> None:
        # A 401 means our cached token is no longer accepted (revoked, or the
        # clock and ``exp`` disagree). Drop it so the next header fetch mints
        # a fresh one instead of reusing it until its nominal expiry. Only if
        # the rejected token is still the current one: late 401s from calls
        # sent before a re-mint must not force another re-mint. The calling
        # ``_post_authed`` is flagged separately (per call), so it
        # retries once without affecting concurrent calls.
        # A token minted less than _TOKEN_MIN_REUSE_S ago is kept until then:
        # a fresh token rejected again will not be fixed by another mint.
        if response.status_code != 401 or response.request.url != httpx.URL(self.url):
            return
        if self._token_command is None:
            return  # no auth sent; nothing a retry could change
        sent = response.request.headers.get("authorization")
        if sent is None:
            return
        call = _CALL_AUTH.get()
        if call is not None and sent == call.headers.get("Authorization"):
            call.rejected = True
        if self._token is not None and sent == f"Bearer {self._token}":
            remint_at = self._last_mint_at + _TOKEN_MIN_REUSE_S
            self._token_refresh_at = min(self._token_refresh_at, remint_at)
            if time.time() < remint_at and self._throttle_logged_mint_at != self._last_mint_at:
                # Once per minted token, i.e. once per throttle window.
                self._throttle_logged_mint_at = self._last_mint_at
                logger.error(
                    "token from %r rejected (401) %.0fs after mint; not re-minting "
                    "until %s (min reuse %.0fs). Calls fail until then; check "
                    "the token's audience / permissions.",
                    self._token_command,
                    time.time() - self._last_mint_at,
                    time.strftime("%H:%M:%S", time.localtime(remint_at)),
                    _TOKEN_MIN_REUSE_S,
                )

    async def _mint_token(self) -> str:
        assert self._token_command is not None
        argv = shlex.split(self._token_command)
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as e:
            raise _TokenCommandMissing(
                f"{argv[0]!r} not found on PATH; it is needed to mint a token "
                f"({self._token_command!r})."
            ) from e
        except OSError as e:
            if isinstance(e, PermissionError) or e.errno == errno.ENOEXEC:
                # Not executable / not a valid executable: permanent.
                raise _TokenCommandMissing(
                    f"cannot run token command {self._token_command!r}: {e}"
                ) from e
            # EMFILE, ENOMEM, EAGAIN, ...: resource pressure, worth retrying.
            raise RuntimeError(
                f"cannot spawn token command {self._token_command!r}: {e}"
            ) from e
        try:
            out, err = await asyncio.wait_for(
                proc.communicate(), timeout=_TOKEN_COMMAND_TIMEOUT_S
            )
        except BaseException as e:
            # Timeout (e.g. waiting on a reauth prompt) or cancellation: never
            # leave the child running while callers hold the auth lock.
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
            if isinstance(e, asyncio.TimeoutError):
                raise _TokenCommandTimeout(
                    f"token command {self._token_command!r} did not finish in "
                    f"{_TOKEN_COMMAND_TIMEOUT_S:.0f}s (needs an interactive login?)."
                ) from e
            raise
        token = out.decode().strip()
        if proc.returncode != 0 or not token:
            raise RuntimeError(
                f"token command {self._token_command!r} failed "
                f"(exit {proc.returncode}): {err.decode().strip()[:500]}"
            )
        return token

    async def _mint_with_retry(self) -> str:
        """Mint, retrying transient failures; caller holds ``_auth_lock``.

        A token command that refreshes over the network must not fail every
        task minting at the moment of one blip. A timeout (usually an
        interactive login) and a missing / non-executable binary are not
        retried. Any final failure is remembered briefly so the callers
        queued on the lock fail fast instead of each repeating the whole
        retry sequence.
        """
        if time.time() < self._mint_blocked_until:
            raise RuntimeError(self._mint_block_reason)
        for attempt in range(1, _TOKEN_MINT_ATTEMPTS + 1):
            try:
                return await self._mint_token()
            except (_TokenCommandTimeout, _TokenCommandMissing) as e:
                self._block_minting(e, _TOKEN_TIMEOUT_BLOCK_S)
                raise
            except RuntimeError as e:
                if attempt == _TOKEN_MINT_ATTEMPTS:
                    self._block_minting(e, _TOKEN_FAILURE_BLOCK_S)
                    raise
                logger.warning(
                    "token mint attempt %d/%d failed: %s",
                    attempt, _TOKEN_MINT_ATTEMPTS, e,
                )
                await asyncio.sleep(_TOKEN_MINT_BACKOFF_S * attempt)
        raise AssertionError("unreachable")

    def _block_minting(self, error: Exception, seconds: float) -> None:
        self._mint_blocked_until = time.time() + seconds
        self._mint_block_reason = f"{error} (not retrying for {seconds:.0f}s)"

    async def _get_headers(self) -> Dict[str, str]:
        # Inside ``_post_authed`` (i.e. the parent's retry loop, which
        # calls this per attempt): reuse the headers minted for this call.
        call = _CALL_AUTH.get()
        if call is not None:
            return dict(call.headers)
        return await self._mint_headers()

    async def _mint_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._token_command is None:
            return headers
        async with self._auth_lock:
            now = time.time()
            if self._token is None or now >= self._token_refresh_at:
                self._token = await self._mint_with_retry()
                self._last_mint_at = now
                exp = _jwt_exp(self._token)
                refresh_at = (
                    exp - _TOKEN_REFRESH_MARGIN_S
                    if exp is not None
                    else now + _TOKEN_FALLBACK_TTL_S
                )
                # A token already inside the margin would otherwise be
                # re-minted on every request (serialized under the lock).
                self._token_refresh_at = max(refresh_at, now + _TOKEN_MIN_REUSE_S)
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    # ---- Sampler entry points ----------------------------------------------------

    def _warn_thinking_level_ignored(self, thinking_level: str | None) -> None:
        # chat.completions has no thinking-level knob; say so once instead of
        # silently dropping the request.
        if thinking_level and not self._warned_thinking_level:
            self._warned_thinking_level = True
            logger.warning(
                "%s ignores thinking_level=%r: chat completions has no "
                "thinking-level control.",
                type(self).__name__,
                thinking_level,
            )

    async def sample_n(
        self,
        prompt: GenerateContentRequest | str,
        num_samples: int = 1,
        thinking_level: str | None = None,
        max_generation_tokens: int | None = None,
        capture_ctxs: Sequence[CaptureContext | None] | None = None,
    ) -> List[SampleResult]:
        self._warn_thinking_level_ignored(thinking_level)
        return await super().sample_n(
            prompt,
            num_samples,
            thinking_level=None,
            max_generation_tokens=max_generation_tokens,
            capture_ctxs=capture_ctxs,
        )

    async def sample_chat_completion(
        self,
        openai_request: Dict[str, Any],
        num_samples: int = 1,
        thinking_level: str | None = None,
        capture_ctxs: Sequence[CaptureContext | None] | None = None,
    ) -> ChatCompletionSampleResult:
        """Send an OpenAI request straight to the server (no Gemini detour).

        Does NOT go through ``sample_n``: posts via ``_post_raw`` (same auth
        and retries), writes one raw-capture blob per choice holding the body
        actually sent and that choice's slice of the raw OpenAI response
        (an error blob per sample on failure), all under
        ``client_request_scope`` so each blob also records the caller's
        request. ``SampleResult``s are built from the choices;
        ``context.prompt`` is ``""`` (there is no ``GenerateContentRequest``
        for an OpenAI-native call). ``model`` and ``n`` are pinned to this
        sampler; ``max_tokens`` is defaulted when the caller sets neither it
        nor ``max_completion_tokens``. Raises ``RuntimeError`` naming the
        cause when sampling failed or the response is unusable, and
        ``ValueError`` for ``stream: true`` (not supported).
        """
        name = f"{type(self).__name__}.sample_chat_completion"
        if openai_request.get("stream"):
            raise ValueError(
                f"{name}: streaming is not supported; drop 'stream' from the request."
            )
        self._warn_thinking_level_ignored(thinking_level)
        body = thinking_blocks_to_reasoning_content(openai_request)
        body["model"] = self.model
        body["n"] = num_samples
        if "max_tokens" not in body and "max_completion_tokens" not in body:
            body["max_tokens"] = _DEFAULT_MAX_OUTPUT_TOKENS

        # Same contract as the parent's sample_n: one context per sample.
        if capture_ctxs is not None and len(capture_ctxs) != num_samples:
            raise ValueError(
                f"capture_ctxs length {len(capture_ctxs)} does not match "
                f"num_samples {num_samples}"
            )
        started = time.monotonic()
        raw: Optional[Dict[str, Any]] = None
        results: Optional[List[SampleResult]] = None
        error: Optional[str] = None
        with client_request_scope(openai_request):
            try:
                raw = await self._post_raw(body)
                if raw is None:
                    error = "post_failed_after_retries"
                else:
                    results = self._results_from_openai(raw, name)
            except BaseException as e:
                error = f"{type(e).__name__}: {str(e)[:300]}"
                raise
            finally:
                # Capture whatever happened, including a raise above.
                raw_uris = await self._capture(
                    capture_ctxs, num_samples, body, raw, error, time.monotonic() - started
                )
        if results is None:
            raise RuntimeError(
                f"{name}: upstream sampling failed; check sampler logs for the "
                "underlying HTTP error."
            )
        response = ChatCompletionResponse.from_dict(reasoning_content_to_thinking_blocks(raw or {}))
        return ChatCompletionSampleResult(
            response=response, results=with_raw_call_uris(results, raw_uris)
        )

    @staticmethod
    async def _capture(
        capture_ctxs: Sequence[CaptureContext | None] | None,
        num_samples: int,
        body: Dict[str, Any],
        raw: Optional[Dict[str, Any]],
        error: Optional[str],
        duration_s: float,
    ) -> List[str | None]:
        """Raw-capture blobs for one passthrough call: the body actually sent
        and, per choice, that choice's slice of the raw OpenAI response (so
        each sample's blob holds its own wire reply); one blob per sample
        otherwise. ``status`` is ``error`` whenever the call did not produce
        results, even if a response arrived (e.g. unusable ``usage``)."""
        choices = (raw or {}).get("choices") or []
        if raw is not None and choices:
            envelope = {k: v for k, v in raw.items() if k != "choices"}
            return await dump_raw_fanout(
                capture_ctxs,
                count=len(choices),
                request=body,
                response=[{**envelope, "choices": [c]} for c in choices],
                status="error" if error else "ok",
                duration_s=duration_s,
            )
        return await dump_raw_fanout(
            capture_ctxs,
            count=num_samples,
            request=body,
            response=raw if raw is not None else {"__error__": error},
            status="error",
            duration_s=duration_s,
        )

    def _results_from_openai(self, raw: Dict[str, Any], name: str) -> List[SampleResult]:
        """One ``SampleResult`` per choice of a raw OpenAI response.

        ``tuning_candidate`` holds a Gemini ``Candidate`` (that is the
        ``SampleResult`` type), made from each choice by the response
        converter; token counts come straight from ``usage`` and are split
        across choices like ``sample_n`` does.
        """
        choices = raw.get("choices") or []
        if not choices:
            raise RuntimeError(f"{name}: the server returned no choices.")
        usage = raw.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        if usage.get("prompt_tokens") is None:
            raise RuntimeError(f"{name}: the response has no usage.prompt_tokens.")
        # Like the parent: missing counts are only unattributable when some
        # choice actually produced output.
        if usage.get("completion_tokens") is None and any(
            (c.get("message") or {}).get(k)
            for c in choices
            for k in ("content", "tool_calls", "reasoning_content", "reasoning")
        ):
            raise RuntimeError(f"{name}: the response has no usage.completion_tokens.")
        details = usage.get("completion_tokens_details")
        reasoning = usage.get("reasoning_tokens")
        if reasoning is None and isinstance(details, dict):
            reasoning = details.get("reasoning_tokens")
        thoughts_total = int(reasoning or 0)
        # generation_token_count is output + thoughts (the parent's sample_n
        # convention), i.e. the server's completion_tokens; never below the
        # thoughts it includes, even if a server over-reports reasoning.
        gen_total = max(int(usage.get("completion_tokens") or 0), thoughts_total)
        n = len(choices)
        candidates = convert_openai_response_to_gemini(raw)["candidates"]
        thinking = reasoning_content_to_thinking_blocks(raw).get("choices") or []
        results: List[SampleResult] = []
        for i, cand in enumerate(candidates):
            message = (thinking[i].get("message") or {}) if i < len(thinking) else {}
            blocks = tuple(message.get("thinking_blocks") or ()) or None
            results.append(
                SampleResult(
                    tuning_candidate=TuningCandidate(
                        candidate_id=str(cand.get("index", i)),
                        candidate=_json_dict_to_candidate(cand),
                        generation_token_count=gen_total // n + (1 if i < gen_total % n else 0),
                        thoughts_token_count=thoughts_total // n
                        + (1 if i < thoughts_total % n else 0),
                    ),
                    context=SamplingContext(
                        prompt="",
                        response_id=raw.get("id"),
                        model_version=raw.get("model"),
                        prompt_token_count=int(usage["prompt_tokens"]),
                        thinking_blocks=blocks,
                    ),
                    train_step_id=None,
                )
            )
        return results

    # ---- wire-layer hooks into the inherited sample_n ------------------------

    def _build_body(
        self,
        prompt: GenerateContentRequest | str,
        num_samples: int,
        max_generation_tokens: int | None = None,
    ) -> Dict[str, Any]:
        body = convert_gemini_request_to_openai(
            to_wire_dict(prompt),
            model=self.model,
            n=num_samples,
            max_tokens=(
                max_generation_tokens
                if max_generation_tokens is not None
                else _DEFAULT_MAX_OUTPUT_TOKENS
            ),
        )
        self._restore_tool_call_args(body)
        return body

    async def _post_with_retries(self, body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # The sample_n path: raw OpenAI response -> generateContent shape.
        data = await self._post_raw(body)
        if data is None:
            return None
        self._remember_tool_call_args(body, data)
        return convert_openai_response_to_gemini(data)

    def _remember_tool_call_args(self, body: Dict[str, Any], data: Dict[str, Any]) -> None:
        """Keep each reply tool call's argument string as the server sent it.

        The history ``sample_n`` callers re-send goes through Gemini protos,
        whose ``Struct`` loses the key order and the int-vs-float spelling of
        the arguments. A server whose chat template renders a replayed call's
        arguments as given (e.g. Kimi-K3's) would then show the model a call it
        did not write, so ``_restore_tool_call_args`` puts the string back.

        Each call is filed twice: under the conversation that produced it
        (``body``'s messages, which later requests repeat as the history before
        this turn) plus the call, which no other rollout or turn shares; and
        under the call alone, for a history rewritten since (e.g. compacted).
        """
        hasher = hashlib.blake2b(digest_size=16)
        for msg in body["messages"]:
            _hash_message(hasher, msg)
        conversation = hasher.digest()
        for choice in data.get("choices") or []:
            for tc in (choice.get("message") or {}).get("tool_calls") or []:
                fn = tc.get("function") or {}
                call = _tool_call_key(fn.get("name"), fn.get("arguments"))
                if call is None:
                    continue
                for key in (_digest(b"turn", conversation, call), _digest(b"call", call)):
                    self._tool_args[key] = fn["arguments"]
                    self._tool_args.move_to_end(key)
        while len(self._tool_args) > _TOOL_ARGS_MEMORY_MAX:
            self._tool_args.popitem(last=False)

    def _restore_tool_call_args(self, body: Dict[str, Any]) -> None:
        """Send each history tool call with the argument string its reply had.

        Looks the call up under the conversation before its turn first, then
        under the call alone. A call found under neither (a turn this sampler
        did not return, or one the memory dropped) keeps the converter's JSON:
        the ``Struct``'s key order, int literals restored.
        """
        hasher = hashlib.blake2b(digest_size=16)
        for msg in body["messages"]:
            conversation = hasher.digest()
            for tc in msg.get("tool_calls") or []:
                fn = tc["function"]
                call = _tool_call_key(fn["name"], fn["arguments"])
                if call is None:
                    continue
                keys = [_digest(b"turn", conversation, call), _digest(b"call", call)]
                found = [k for k in keys if k in self._tool_args]
                if not found:
                    self._tool_args_misses += 1
                    if self._tool_args_misses & (self._tool_args_misses - 1) == 0:
                        logger.warning(
                            "%s: %d history tool call(s) so far had no remembered argument "
                            "string; sent with the converter's key order instead.",
                            type(self).__name__,
                            self._tool_args_misses,
                        )
                    continue
                fn["arguments"] = self._tool_args[found[0]]
                for key in found:
                    self._tool_args.move_to_end(key)
            # After restoring, so later turns see the history as it was sent.
            _hash_message(hasher, msg)

    async def _post_raw(self, body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """POST ``body``; the raw OpenAI response dict, or None on failure.

        With ``degenerate_marker`` set, a degenerate reply is re-requested up
        to ``degenerate_retries`` times and, if every attempt degenerates,
        the call fails (None) like a transport failure.
        """
        data = await self._post_authed(body)
        if self._degenerate_marker is None:
            return data
        for attempt in range(self._degenerate_retries + 1):
            if data is None or not self._is_degenerate(data):
                return data
            final = attempt == self._degenerate_retries
            logger.warning(
                "%s: degenerate reply (id=%s, a run of %r in the output, "
                "finish_reason=%s); %s",
                type(self).__name__,
                data.get("id"),
                self._degenerate_marker,
                [c.get("finish_reason") for c in data.get("choices") or [] if isinstance(c, dict)],
                "failing the call" if final else
                f"re-requesting ({attempt + 1}/{self._degenerate_retries})",
            )
            if final:
                return None
            data = await self._post_authed(body)
        return data

    def _is_degenerate(self, data: Dict[str, Any]) -> bool:
        assert self._degenerate_marker is not None
        run = self._degenerate_marker * _DEGENERATE_RUN
        for choice in data.get("choices") or []:
            message = (choice.get("message") or {}) if isinstance(choice, dict) else {}
            for key in ("reasoning_content", "reasoning", "content"):
                text = message.get(key) if isinstance(message, dict) else None
                if isinstance(text, str) and run in text:
                    return True
        return False

    async def _post_authed(self, body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """One call through the parent's HTTP retry loop, with auth around it."""
        # Mint (or reuse) the token OUTSIDE the parent's retry loop: inside
        # it, a failing token command is treated as a transport hiccup,
        # retried with backoff, and finally turned into empty results. Raised
        # here, an auth failure reaches the caller instead. Every attempt of
        # the loop reuses these headers (see ``_get_headers``).
        call = _CallAuth(headers=await self._mint_headers())
        reset_token = _CALL_AUTH.set(call)
        try:
            data = await super()._post_with_retries(body)
            if data is None and call.rejected:
                # This call was rejected with 401; retry once on a fresh
                # token, minted here (outside the loop) for the same reason.
                # If no new token was minted (the rejected one is younger
                # than _TOKEN_MIN_REUSE_S), resending it is pointless.
                rejected_headers = call.headers
                call.headers = await self._mint_headers()
                call.rejected = False
                if call.headers != rejected_headers:
                    data = await super()._post_with_retries(body)
        finally:
            _CALL_AUTH.reset(reset_token)
        if data is None:
            return None
        if not isinstance(data, dict):
            logger.error("chat.completions returned a non-object body: %.300r", data)
            return None
        self._warn_if_reasoning_tokens_unreported(data)
        return data

    def _warn_if_reasoning_tokens_unreported(self, data: Dict[str, Any]) -> None:
        # Without a reasoning-token count, the converter books all completion
        # tokens as output (thoughtsTokenCount absent); say so once.
        if self._warned_unreported_reasoning_tokens:
            return
        usage = data.get("usage") or {}
        if not isinstance(usage, dict):
            return
        details = usage.get("completion_tokens_details") or {}
        if usage.get("reasoning_tokens") is not None or (
            isinstance(details, dict) and details.get("reasoning_tokens") is not None
        ):
            return
        for choice in data.get("choices") or []:
            message = (choice.get("message") or {}) if isinstance(choice, dict) else {}
            if isinstance(message, dict) and (
                message.get("reasoning_content") or message.get("reasoning")
            ):
                self._warned_unreported_reasoning_tokens = True
                logger.warning(
                    "%s: server returned reasoning text but no reasoning_tokens "
                    "in usage; all completion tokens are counted as output "
                    "(candidatesTokenCount) and thoughts as 0.",
                    type(self).__name__,
                )
                return

    async def aclose(self) -> None:
        # super().aclose() closes the client, injected or not (the parent's
        # semantics). Detach our hook anyway so anything still holding the
        # client does not keep this closed sampler reachable through it.
        hooks = self._client.event_hooks.get("response") or []
        if self._on_response in hooks:
            hooks.remove(self._on_response)
        await super().aclose()
