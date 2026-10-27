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

"""REST/HTTPS Vertex MSRL sampler + trainer.

Public classes: :class:`VertexMsrlRestClient`,
:class:`VertexMsrlRestSampler`, :class:`VertexMsrlRestTrainer`. Same
LRO-poll semantics and result shape as any Vertex MSRL sampler, over
`httpx.AsyncClient`.

Built by ``backends.backend_config.VertexMsrlBackendConfig.make()``
(the name-keyed ``backends.factory`` registry calls it "vertex_msrl").
The tuning-scope
custom-method verbs are exposed as class-level constants
(:data:`VertexMsrlRestClient._URL_GENERATE_CONTENT_TUNING_SCOPE`,
:data:`_URL_TRAIN_STEP`) for easy override when a sandbox uses different
names.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import threading
import time
from collections.abc import Sequence
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import google.auth  # pytype: disable=import-error
import google.auth.transport.requests  # pytype: disable=import-error
import httpx

from google.cloud.aiplatform_v1.types import Candidate, GenerateContentRequest

from backends._proto_utils import _json_dict_to_candidate
from backends.prompt_coerce import to_wire_dict
from backends.retry_profile import DEFAULT_LRO_RETRY, LroRetryProfile
from utils.auth_utils import get_call_credentials
from utils.http_utils import build_http_client
from utils.logging_utils import simplify_trajectory_step_for_logging
from utils.raw_capture import CaptureContext, dump_raw_fanout

from core.protocols import Sampler, Trainer
from core.trainer_audit import parse_candidate_audit
from core.trainer_metrics import map_trainer_wire_metrics
from core.atif_trajectory import ScoredTrajectory
from core import atif_extras
from core.sample_result import (
    SampleResult,
    SamplingContext,
    TuningCandidate,
    with_raw_call_uris,
)

# ---- Vertex environment -> endpoint hostname mapping -------------------------
DEFAULT_LOCATION = "us-central1"

# Production endpoint template; ``{location}`` is substituted at build
# time. Other backend implementations may resolve a different endpoint
# from the environment name.
PROD_ENDPOINT_TEMPLATE = "{location}-aiplatform.googleapis.com"


def default_endpoint(location: str = DEFAULT_LOCATION) -> str:
    """Return the production Vertex endpoint hostname for ``location``."""
    if not location:
        raise ValueError("location is required to resolve a Vertex endpoint")
    return PROD_ENDPOINT_TEMPLATE.format(location=location)


# Pre-formatted constant retained for callers that import it directly.
DEFAULT_ENDPOINT = default_endpoint(DEFAULT_LOCATION)

# ---- Tuning-job state / retry / poll operational constants ------------------

# Terminal job states used to bail out of wait loops.
_JOB_STATE_RUNNING = "JOB_STATE_RUNNING"
_JOB_TERMINAL_BAD_STATES = frozenset(
    {
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_CANCELLING",
        "JOB_STATE_EXPIRED",
        "JOB_STATE_PAUSED",
    }
)
_JOB_TERMINAL_GOOD_STATES = frozenset({"JOB_STATE_SUCCEEDED"})

# ---- Unified retry / poll backoff model -------------------------------------
#
# All retrying is deadline-based with a shared annealed backoff.
# Per-category deadlines:
#   * Tuning-job CRUD (Create/Get/Cancel): 10 min. Owned here -- it is
#     lifecycle, not sampling, so it is not part of the retry profile.
#   * Sample-op create: `LroRetryProfile.create_deadline_s` (2h).
#   * GetOperation poll: single-attempt RPC, loop retries unboundedly
#     within `LroRetryProfile.poll_deadline_s` (10h).
#
# The sampling-shaped numbers moved to `backends.retry_profile` so a config
# can state them; the module constants below are DERIVED from its default,
# so there is one source of truth and the values are unchanged.

# Backoff schedule for RPC retries.
_RETRY_BACKOFF_INITIAL = 1.0
_RETRY_BACKOFF_MAX = 30.0
# Backoff schedule for LRO polling, from the default profile. Ops run
# seconds-to-minutes and there can be thousands in flight; the slow
# schedule keeps steady-state GetOperation QPS under the per-project
# `resource_management_requests` quota.
_POLL_BACKOFF_INITIAL = DEFAULT_LRO_RETRY.poll_backoff.initial_s
_POLL_BACKOFF_MAX = DEFAULT_LRO_RETRY.poll_backoff.max_s
_POLL_BACKOFF_FACTOR = DEFAULT_LRO_RETRY.poll_backoff.factor
_BACKOFF_FACTOR = 1.5  # RPC-retry backoff
_BACKOFF_JITTER = 0.5  # +/-50%

# Retry deadlines (seconds).
_CRUD_RETRY_DEADLINE = 10 * 60.0
_OP_START_RETRY_DEADLINE = DEFAULT_LRO_RETRY.create_deadline_s
# Total budget for polling a single LRO. 10h is generous for slow ops
# but bounded so a wedged op can't poll forever.
_OP_POLL_TIMEOUT = DEFAULT_LRO_RETRY.poll_deadline_s
# Retry-log throttle: emit the full line at most once per interval (per
# client); suppressed retries are logged at DEBUG. Prevents a 429 storm
# from flooding the logs.
_RETRY_LOG_INTERVAL = 30.0

# Fallback maxOutputTokens when a caller does not pass a per-call
# ``max_generation_tokens`` to ``sample()``. The generation cap is a
# single-tier per-call knob (mirroring ``thinking_level``): there is no
# sampler-constructor override, so this module constant is the sole
# fallback and preserves the historical 32768 default on the wire.
_DEFAULT_MAX_OUTPUT_TOKENS = 32768

logger = logging.getLogger(__name__)

# Broken-channel give-up ceilings for wait_for_lro: streak count and
# streak wall time; whichever trips first fails the poll. A single
# successful poll resets both. Prevents burning the 10h outer budget
# on a dead socket that will never recover.
_POLL_MAX_CONSECUTIVE_FAILURES = 20
_POLL_MAX_FAILURE_WINDOW = 300.0  # 5 min

# Optional raw request/response capture (env MSRL_RAW_CAPTURE=<path.jsonl>).
_RAW_CAPTURE_LOCK = threading.Lock()


def _raw_capture(path: str, record: Dict[str, Any]) -> None:
    """Append one record as a JSON line. Thread-safe."""
    try:
        line = json.dumps(record, default=str)
    except Exception:
        return
    with _RAW_CAPTURE_LOCK:
        with open(path, "a") as f:
            f.write(line + "\n")


# ---- Tuning-scope error classification --------------------------------------
# Substring of the tuning-scope rejection when the request exceeds the
# serving prompt cap, e.g. "Prompt length 131109 exceeds maximum allowed
# length 122880." Matched rather than parsed: the inner ``operation``
# envelope carries no structured error code (unlike the outer LRO
# ``error`` handled in ``wait_for_lro``).
# TWO wordings, observed: "... exceeds maximum allowed length" and "Input
# tokens exceed maximum prompt length (max is 122880)". They share no
# substring long enough to match on (`exceed` is not `exceeds`), so both
# are spelled out. Matched case-insensitively against the WHOLE error text,
# so a wrapper prefix ("Operation ... failed: code=3, message=...") does
# not defeat them.
_PROMPT_TOO_LONG_MARKERS = (
    "exceeds maximum allowed length",
    "exceed maximum prompt length",
    "exceeds maximum prompt length",
)


def _is_prompt_too_long_text(text: str) -> bool:
    """True iff ``text`` is one of the serving stack's prompt-length refusals."""
    lowered = text.lower()
    return any(m in lowered for m in _PROMPT_TOO_LONG_MARKERS)


def _detect_prompt_too_long(lro_response: Dict[str, Any]) -> bool:
    """True iff a terminal LRO's ``response`` body is a prompt-length rejection.

    The tuning-scope RPC reports failure *inside* the LRO response
    (``operation.isFailed`` + ``operation.error``) rather than via the
    standard LRO ``error`` field, and returns no candidates. Callers use
    this to set ``SamplingContext.is_prompt_too_long`` so the agent can
    classify the rollout instead of seeing an unexplained empty result.
    """
    op_info = lro_response.get("operation") or {}
    if not op_info.get("isFailed"):
        return False
    return _is_prompt_too_long_text(str(op_info.get("error") or ""))


# ---- HTTP status → retry policy mapping -------------------------------------
# Permanent: identical retry can never succeed; fail fast so the 2h
#   _OP_START_RETRY_DEADLINE isn't burned on a 400.
# Retryable: transient server / rate-limit; retry with annealed backoff.
# Anything else (transport exceptions, unmapped codes): retry too —
#   servers occasionally misclassify transients under unexpected codes.

_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_PERMANENT_STATUS = frozenset({400, 401, 403, 404, 405, 409, 501})


class _AnnealedBackoff:
    """Jittered, annealed sleep schedule."""

    def __init__(
        self,
        initial: float,
        maximum: float,
        *,
        factor: float = _BACKOFF_FACTOR,
        jitter: float = _BACKOFF_JITTER,
    ):
        self._interval = float(initial)
        self._maximum = float(maximum)
        self._factor = float(factor)
        self._jitter = float(jitter)

    @property
    def interval(self) -> float:
        """Current (un-jittered) interval, for logging."""
        return self._interval

    async def sleep(self) -> None:
        lo, hi = 1.0 - self._jitter, 1.0 + self._jitter
        await asyncio.sleep(self._interval * random.uniform(lo, hi))
        self._interval = min(self._interval * self._factor, self._maximum)


class _AsyncRateLimiter:
    """Token-bucket limiter capping outgoing REST QPM.

    Caps this process's share of the per-project
    ``resource_management_requests`` quota.
    """

    def __init__(self, qpm: float, *, burst: Optional[float] = None):
        self._rate = float(qpm) / 60.0  # tokens per second
        self._capacity = float(burst) if burst is not None else max(1.0, self._rate)
        self._tokens = self._capacity
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                self._tokens = min(
                    self._capacity,
                    self._tokens + (now - self._updated) * self._rate,
                )
                self._updated = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                await asyncio.sleep((1.0 - self._tokens) / self._rate)


class VertexMsrlRestClient:
    """Async httpx client for the Vertex MSRL REST surface.

    Owns the shared ``httpx.AsyncClient``, ADC-backed credentials +
    refresh, retries, LRO polling, and the create/get/cancel tuning-job
    lifecycle. Consumed by :class:`VertexMsrlRestSampler` and
    :class:`VertexMsrlRestTrainer` (shared instance so they share the
    connection pool and auth state).
    """

    # Tuning-scope custom-method verb suffixes on the tuning-job
    # resource (AIP-136 style). camelCase of the gRPC method name.
    _URL_GENERATE_CONTENT_TUNING_SCOPE = ":generateContentTuningScope"
    _URL_TRAIN_STEP = ":trainStep"

    def __init__(
        self,
        *,
        project: str,
        location: str = DEFAULT_LOCATION,
        endpoint: str = DEFAULT_ENDPOINT,
        tuning_job_id: str = "",
        request_timeout: float = 600.0,
        quota_qpm: Optional[float] = None,
        http_client: Optional[httpx.AsyncClient] = None,
        api_version: str = "v1beta1",
        service_account: Optional[str] = None,
        use_user_credentials: bool = False,
    ):
        if not project:
            raise ValueError("project is required")
        if request_timeout <= 0:
            raise ValueError(f"request_timeout must be > 0, got {request_timeout}")
        self.project = str(project)
        self.location = location
        self.endpoint = endpoint
        self.tuning_job_id = tuning_job_id
        # Populated by create_tuning_job on the new-job path, or refreshed
        # by wait_for_job_running for pre-existing jobs. Read by the
        # sampler to stamp ``SamplingContext.model_version``.
        self.base_model: Optional[str] = None
        self.parent = f"projects/{self.project}/locations/{self.location}"
        self.api_version = api_version
        self._request_timeout = request_timeout
        self._retry_log_last = 0.0
        self._retry_log_suppressed = 0
        self._limiter = (
            _AsyncRateLimiter(quota_qpm) if quota_qpm and quota_qpm > 0 else None
        )
        if self._limiter is not None:
            logger.info(
                "[vertex_msrl_rest] Outgoing RPC rate limiter enabled: "
                "~%.0f req/min (this process's "
                "resource_management_requests budget).",
                quota_qpm,
            )
        self._service_account = service_account
        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        if use_user_credentials:
            self._creds, _ = google.auth.default(scopes=scopes)
        else:
            self._creds = get_call_credentials(
                target_service_account=service_account, scopes=scopes
            )
        self._auth_lock = asyncio.Lock()
        self._client = (
            http_client
            if http_client is not None
            else build_http_client(read_timeout=request_timeout)
        )
        self._base_url = f"https://{self.endpoint}/{self.api_version}"

    @property
    def tuning_job_name(self) -> str:
        if not self.tuning_job_id:
            raise RuntimeError(
                "tuning_job_id is not set; call create_tuning_job() first or "
                "pass tuning_job_id at construction."
            )
        return f"{self.parent}/tuningJobs/{self.tuning_job_id}"

    async def _get_headers(self) -> Dict[str, str]:
        async with self._auth_lock:
            if not self._creds.valid:
                await asyncio.to_thread(
                    self._creds.refresh,
                    google.auth.transport.requests.Request(),
                )
            token = self._creds.token
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "x-goog-user-project": self.project,
        }

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

    async def _request(
        self,
        method: str,
        url: str,
        *,
        rpc_name: str,
        retry_deadline: Optional[float],
        timeout: Optional[float] = None,
        json: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Invoke an httpx request with deadline-based annealed retry.

        Returns the parsed JSON body (or ``None`` for 2xx no-content).

        ``retry_deadline`` (seconds):
          * ``0``    -> single attempt (caller owns retrying)
          * ``None`` -> unbounded retry
          * ``> 0``  -> retry until this much wall-clock elapsed
        """
        request_timeout = self._request_timeout if timeout is None else timeout
        start = time.monotonic()
        backoff = _AnnealedBackoff(_RETRY_BACKOFF_INITIAL, _RETRY_BACKOFF_MAX)
        attempt = 0
        while True:
            attempt += 1
            if self._limiter is not None:
                await self._limiter.acquire()
            status: Optional[int] = None
            err_detail: str = ""
            exc: Optional[BaseException] = None
            try:
                headers = await self._get_headers()
                resp = await self._client.request(
                    method,
                    url,
                    json=json,
                    headers=headers,
                    timeout=request_timeout,
                )
                status = resp.status_code
                if status in _PERMANENT_STATUS:
                    logger.error(
                        "[vertex_msrl_rest] %s failed (attempt %d, %.0fs "
                        "elapsed): HTTP %s, body=%s; permanent error, "
                        "not retrying",
                        rpc_name,
                        attempt,
                        time.monotonic() - start,
                        status,
                        resp.text[:500],
                    )
                    resp.raise_for_status()
                if status in _RETRYABLE_STATUS:
                    err_detail = resp.text[:500]
                    raise httpx.HTTPStatusError(
                        f"retryable HTTP {status}",
                        request=resp.request,
                        response=resp,
                    )
                resp.raise_for_status()
                # 2xx: return parsed JSON if any, else None.
                if not resp.content:
                    return None
                return resp.json()
            except httpx.HTTPStatusError as e:
                exc = e
                if status is None and e.response is not None:
                    status = e.response.status_code
                    err_detail = e.response.text[:500]
                # Permanent statuses were logged above; re-raise so we
                # don't burn the retry deadline. Retryable path falls
                # through to the retry logic below.
                if status in _PERMANENT_STATUS:
                    raise
            except Exception as e:  # noqa: BLE001 - transport hiccups
                exc = e
                err_detail = repr(e)

            elapsed = time.monotonic() - start
            log = (
                logger.warning
                if (status in _RETRYABLE_STATUS or status is None)
                else logger.error
            )
            exhausted = retry_deadline == 0 or (
                retry_deadline is not None and elapsed >= retry_deadline
            )
            if exhausted:
                if retry_deadline == 0:
                    logger.debug(
                        "[vertex_msrl_rest] %s failed (attempt %d, %.0fs): "
                        "status=%s, detail=%s; single attempt, caller retries",
                        rpc_name,
                        attempt,
                        elapsed,
                        status,
                        err_detail,
                    )
                else:
                    log(
                        "[vertex_msrl_rest] %s failed (attempt %d, %.0fs "
                        "elapsed): status=%s, detail=%s; giving up",
                        rpc_name,
                        attempt,
                        elapsed,
                        status,
                        err_detail,
                    )
                assert exc is not None
                raise exc
            deadline_str = (
                "unlimited" if retry_deadline is None else f"{retry_deadline:.0f}s"
            )
            emit, suppressed = self._should_log_retry()
            if emit:
                log(
                    "[vertex_msrl_rest] %s failed (attempt %d, %.0fs/%s): "
                    "status=%s, detail=%s; retrying in ~%.0fs (+%d more "
                    "retries suppressed in the last ~%.0fs)",
                    rpc_name,
                    attempt,
                    elapsed,
                    deadline_str,
                    status,
                    err_detail,
                    backoff.interval,
                    suppressed,
                    _RETRY_LOG_INTERVAL,
                )
            else:
                logger.debug(
                    "[vertex_msrl_rest] %s failed (attempt %d, %.0fs): "
                    "status=%s, detail=%s; retrying in ~%.0fs",
                    rpc_name,
                    attempt,
                    elapsed,
                    status,
                    err_detail,
                    backoff.interval,
                )
            await backoff.sleep()

    async def create_tuning_job(
        self,
        *,
        base_model: str,
        description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        tuned_model_display_name: Optional[str] = None,
        learning_rate_multiplier: Optional[float] = None,
        adapter_size: Optional[str] = None,
        checkpoint_interval: Optional[int] = None,
    ) -> str:
        """Create an MSRL tuning job and return its short id."""
        hp: Dict[str, Any] = {}
        if learning_rate_multiplier is not None:
            hp["learningRateMultiplier"] = float(learning_rate_multiplier)
        if adapter_size is not None:
            hp["adapterSize"] = (
                str(adapter_size)
                if isinstance(adapter_size, str)
                else int(adapter_size)
            )
        if checkpoint_interval is not None:
            hp["checkpointInterval"] = int(checkpoint_interval)

        tuning_job: Dict[str, Any] = {
            "baseModel": base_model,
            "multiStepReinforcementTuningSpec": {"hyperParameters": hp},
        }
        if description is not None:
            tuning_job["description"] = description
        if tuned_model_display_name is not None:
            tuning_job["tunedModelDisplayName"] = tuned_model_display_name
        if labels:
            tuning_job["labels"] = dict(labels)

        url = f"{self._base_url}/{self.parent}/tuningJobs"
        logger.info(
            "[vertex_msrl_rest] CreateTuningJob (parent=%s, base_model=%s)",
            self.parent,
            base_model,
        )
        resp = await self._request(
            "POST",
            url,
            rpc_name="CreateTuningJob",
            retry_deadline=_CRUD_RETRY_DEADLINE,
            json=tuning_job,
        )
        name = resp.get("name") if isinstance(resp, dict) else None
        if not name:
            raise RuntimeError(f"CreateTuningJob returned no name. Response: {resp!r}")
        self.tuning_job_id = name.rsplit("/", 1)[-1]
        # Remember the base model; surfaced onto
        # SampleResult.context.model_version so ChatCompletionResponse.model
        # isn't empty.
        self.base_model = str(base_model)
        logger.info(
            "[vertex_msrl_rest] Created tuning job %s (state=%s)",
            name,
            resp.get("state"),
        )
        return self.tuning_job_id

    async def get_tuning_job(
        self,
        *,
        retry_deadline: Optional[float] = _CRUD_RETRY_DEADLINE,
    ) -> Dict[str, Any]:
        url = f"{self._base_url}/{self.tuning_job_name}"
        return await self._request(
            "GET",
            url,
            rpc_name="GetTuningJob",
            retry_deadline=retry_deadline,
        )

    async def count_tuning_jobs_by_state(
        self,
        *,
        page_size: int = 100,
        retry_deadline: Optional[float] = _CRUD_RETRY_DEADLINE,
        max_pages: int = 1000,
    ) -> Dict[str, int]:
        """Return a ``{state: count}`` tally of tuning jobs under ``self.parent``.

        Pages through ListTuningJobs until ``nextPageToken`` is empty or the
        ``max_pages`` safety cap is hit (a server that returns a stable
        non-empty token forever can't hang the caller; a warning is logged and
        the partial counts are returned). Jobs with no/empty ``state`` are
        bucketed as ``JOB_STATE_UNSPECIFIED``. Callers that want a best-effort
        congestion signal should wrap this in try/except -- it can raise like
        any other RPC.
        """
        counts: Dict[str, int] = {}
        page_token = ""
        pages = 0
        while True:
            url = f"{self._base_url}/{self.parent}/tuningJobs?pageSize={int(page_size)}"
            if page_token:
                url += f"&pageToken={quote(page_token, safe='')}"
            resp = await self._request(
                "GET",
                url,
                rpc_name="ListTuningJobs",
                retry_deadline=retry_deadline,
            )
            if not isinstance(resp, dict):
                break
            for job in resp.get("tuningJobs", []) or []:
                state = (
                    (job.get("state") or "JOB_STATE_UNSPECIFIED")
                    if isinstance(job, dict)
                    else "JOB_STATE_UNSPECIFIED"
                )
                counts[state] = counts.get(state, 0) + 1
            pages += 1
            page_token = resp.get("nextPageToken") or ""
            if not page_token:
                break
            if pages >= max_pages:
                logger.warning(
                    "[vertex_msrl_rest] ListTuningJobs stopped after %d pages "
                    "(max_pages cap); counts may be partial.",
                    pages,
                )
                break
        return counts

    async def wait_for_job_running(
        self,
        *,
        timeout: float = 60 * 60,
        polling_interval: float = _POLL_BACKOFF_INITIAL,
        max_polling_interval: float = _POLL_BACKOFF_MAX,
        polling_backoff: float = _POLL_BACKOFF_FACTOR,
    ) -> Dict[str, Any]:
        start = time.monotonic()
        backoff = _AnnealedBackoff(
            polling_interval,
            max_polling_interval,
            factor=polling_backoff,
        )
        last_state = ""
        while time.monotonic() - start < timeout:
            try:
                job = await self.get_tuning_job(retry_deadline=0)
            except Exception as e:  # pylint: disable=broad-except
                logger.warning(
                    "[vertex_msrl_rest] GetTuningJob failed while waiting "
                    "for RUNNING (%s); retrying in ~%.0fs",
                    e,
                    backoff.interval,
                )
                await backoff.sleep()
                continue
            state = job.get("state", "")
            if state != last_state:
                logger.info(
                    "[vertex_msrl_rest] tuning job %s state=%s (after %.0fs)",
                    self.tuning_job_id,
                    state,
                    time.monotonic() - start,
                )
                last_state = state
            # Refresh base_model on every fetch to cover the pre-existing-job
            # path where create_tuning_job was never called here.
            job_base_model = job.get("baseModel")
            if isinstance(job_base_model, str) and job_base_model:
                self.base_model = job_base_model
            if state == _JOB_STATE_RUNNING:
                return job
            if state in _JOB_TERMINAL_BAD_STATES or state in _JOB_TERMINAL_GOOD_STATES:
                raise RuntimeError(
                    f"Tuning job {self.tuning_job_id} reached terminal state "
                    f"{state} before becoming RUNNING."
                )
            await backoff.sleep()
        raise TimeoutError(
            f"Timed out after {timeout:.0f}s waiting for tuning job "
            f"{self.tuning_job_id} to reach JOB_STATE_RUNNING (last state="
            f"{last_state})."
        )

    async def cancel_tuning_job(self) -> None:
        """Best-effort cancel. Logs (does not raise) on failure."""
        try:
            url = f"{self._base_url}/{self.tuning_job_name}:cancel"
            logger.info(
                "[vertex_msrl_rest] CancelTuningJob -> %s",
                self.tuning_job_name,
            )
            await self._request(
                "POST",
                url,
                rpc_name="CancelTuningJob",
                retry_deadline=_CRUD_RETRY_DEADLINE,
                json={},
            )
            logger.info(
                "[vertex_msrl_rest] Cancel request accepted for job %s",
                self.tuning_job_id,
            )
        except Exception as e:  # pylint: disable=broad-except
            logger.error(
                "[vertex_msrl_rest] Failed to cancel tuning job %s: %s",
                self.tuning_job_id,
                e,
            )

    async def wait_for_lro(
        self,
        operation_name: str,
        *,
        retry: LroRetryProfile = DEFAULT_LRO_RETRY,
        poll_max_consecutive_failures: int = _POLL_MAX_CONSECUTIVE_FAILURES,
        poll_max_failure_window: float = _POLL_MAX_FAILURE_WINDOW,
    ) -> Dict[str, Any]:
        """Poll ``Operations.GetOperation`` until done and return the op dict.

        The poll budget and schedule come from ``retry`` (its
        ``poll_deadline_s`` / ``poll_backoff``) so a caller states its
        policy once, at construction, instead of per call.

        Two independent give-up conditions bound the poll:

        * ``retry.poll_deadline_s`` bounds how long we wait for an LRO that
          is legitimately still running (default 10h; a healthy long op
          just keeps returning ``done=false``).
        * ``poll_max_consecutive_failures`` / ``poll_max_failure_window``
          bound how long we keep hammering a *broken* channel (transport
          exceptions on every poll). Whichever ceiling trips first raises
          ``RuntimeError`` so the async loop's buffer isn't
          head-of-line-blocked for hours by a single stuck rollout.

        A successful poll resets both failure counters.
        """
        url = f"{self._base_url}/{operation_name}"
        timeout = retry.poll_deadline_s
        start = time.monotonic()
        backoff = _AnnealedBackoff(
            retry.poll_backoff.initial_s,
            retry.poll_backoff.max_s,
            factor=retry.poll_backoff.factor,
            jitter=retry.poll_backoff.jitter,
        )
        consecutive_failures = 0
        first_failure_time: Optional[float] = None
        while time.monotonic() - start < timeout:
            try:
                op = await self._request(
                    "GET",
                    url,
                    rpc_name="GetOperation",
                    retry_deadline=0,
                )
            except Exception as e:  # pylint: disable=broad-except
                now = time.monotonic()
                consecutive_failures += 1
                if first_failure_time is None:
                    first_failure_time = now
                emit, suppressed = self._should_log_retry()
                if emit:
                    logger.warning(
                        "[vertex_msrl_rest] LRO poll %s transient failure "
                        "(%d consecutive): %s; retrying (~%.0fs; +%d more "
                        "suppressed in the last ~%.0fs)",
                        operation_name,
                        consecutive_failures,
                        e,
                        backoff.interval,
                        suppressed,
                        _RETRY_LOG_INTERVAL,
                    )
                else:
                    logger.debug(
                        "[vertex_msrl_rest] LRO poll %s transient failure "
                        "(%d consecutive): %s; retrying in ~%.0fs",
                        operation_name,
                        consecutive_failures,
                        e,
                        backoff.interval,
                    )
                # Broken-channel give-up so we don't burn the outer
                # timeout polling a dead socket.
                failing_for = now - first_failure_time
                if (
                    consecutive_failures >= poll_max_consecutive_failures
                    or failing_for >= poll_max_failure_window
                ):
                    raise RuntimeError(
                        f"LRO poll {operation_name}: giving up after "
                        f"{consecutive_failures} consecutive transient "
                        f"failures over {failing_for:.0f}s (ceilings: "
                        f"count={poll_max_consecutive_failures}, "
                        f"window={poll_max_failure_window:.0f}s); channel "
                        f"looks broken -- not waiting out the "
                        f"{timeout:.0f}s LRO budget. Last error: {e!r}"
                    ) from e
                await backoff.sleep()
                continue
            # Successful read: reset failure state.
            consecutive_failures = 0
            first_failure_time = None
            if op.get("done"):
                if "error" in op and op["error"]:
                    err = op["error"]
                    raise RuntimeError(
                        f"Operation {operation_name} failed: "
                        f"code={err.get('code')}, message={err.get('message')}"
                    )
                return op
            await backoff.sleep()
        raise TimeoutError(
            f"Timed out after {timeout:.0f}s polling LRO {operation_name}"
        )

    async def aclose(self) -> None:
        await self._client.aclose()


class VertexMsrlRestSampler(Sampler):
    """Vertex MSRL sampler over REST/HTTPS (camelCase JSON wire format)."""

    def __init__(
        self,
        client: VertexMsrlRestClient,
        *,
        retry: LroRetryProfile = DEFAULT_LRO_RETRY,
        request_timeout: float = 60.0,
    ):
        self.client = client
        # One object states the whole policy: how patiently a rejected
        # create is resent, how long an existing op is polled, and on what
        # schedule. Set from the backend config at construction.
        self.retry = retry
        self.request_timeout = request_timeout

    def _build_body(
        self,
        prompt: GenerateContentRequest | str,
        num_samples: int,
        max_generation_tokens: int | None = None,
    ) -> Dict[str, Any]:
        """Build the {name, contentGenerationParameters} JSON body.

        Rehomes wire fields under ``contentGenerationParameters`` (the
        tuning-scope RPC shape) and stamps ``candidateCount`` /
        ``maxOutputTokens`` defaults.

        ``max_generation_tokens`` is the single-tier per-call cap: when
        ``None`` (the default), ``maxOutputTokens`` falls back to the module
        constant ``_DEFAULT_MAX_OUTPUT_TOKENS`` (there is no sampler-level
        override; the cap is threaded per-call, mirroring ``thinking_level``).
        """
        wire = to_wire_dict(prompt)  # camelCase JSON dict
        gen_cfg: Dict[str, Any] = dict(wire.get("generationConfig") or {})
        gen_cfg.setdefault(
            "maxOutputTokens",
            max_generation_tokens
            if max_generation_tokens is not None
            else _DEFAULT_MAX_OUTPUT_TOKENS,
        )
        gen_cfg["candidateCount"] = num_samples

        params: Dict[str, Any] = {"contents": wire.get("contents", [])}
        if wire.get("systemInstruction"):
            params["systemInstruction"] = wire["systemInstruction"]
        if wire.get("tools"):
            params["tools"] = wire["tools"]
        if wire.get("toolConfig"):
            params["toolConfig"] = wire["toolConfig"]
        params["generationConfig"] = gen_cfg
        return {
            "name": self.client.tuning_job_name,
            "contentGenerationParameters": params,
        }

    def _empty_result(
        self,
        prompt: GenerateContentRequest | str,
        *,
        operation_id: Optional[str] = None,
        extras: Optional[Dict[str, Any]] = None,
    ) -> SampleResult:
        """Fail-safe SampleResult.

        ``extras`` forwards ``is_prompt_too_long`` / ``is_malformed_usage``
        onto ``SamplingContext``. Consumers detect the sampling failure via
        ``result.tuning_candidate.candidate_id == ""``.
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
                operation_id=operation_id,
                is_prompt_too_long=bool(
                    (extras or {}).get("is_prompt_too_long", False)
                ),
                is_malformed_usage=bool(
                    (extras or {}).get("is_malformed_usage", False)
                ),
            ),
            train_step_id=None,
        )

    async def sample_n(
        self,
        prompt: GenerateContentRequest | str,
        num_samples: int = 1,
        thinking_level: Optional[str] = None,
        max_generation_tokens: int | None = None,
        capture_ctxs: Sequence[CaptureContext | None] | None = None,
    ) -> List[SampleResult]:
        # Stamp thinking_level onto the prompt when supplied so the
        # wire generationConfig carries it. No-op for falsy inputs.
        if thinking_level:
            from backends.vertex_thinking import apply_thinking_level

            prompt = apply_thinking_level(prompt, thinking_level)
        body = self._build_body(
            prompt, num_samples, max_generation_tokens=max_generation_tokens
        )
        # Normalize per-candidate capture-ctx list; raise on length
        # mismatch (fail loud on drift; see sampler ABC docstring).
        # ValueError, not assert: assert is stripped under `python -O`
        # and this is a contract check the caller must not silently
        # violate.
        ctxs: list[CaptureContext | None] = (
            list(capture_ctxs) if capture_ctxs is not None else [None] * num_samples
        )
        if len(ctxs) != num_samples:
            raise ValueError(
                f"capture_ctxs length {len(ctxs)} does not match num_samples {num_samples}"
            )
        t0 = time.monotonic()
        url = (
            f"{self.client._base_url}/{self.client.tuning_job_name}"
            f"{VertexMsrlRestClient._URL_GENERATE_CONTENT_TUNING_SCOPE}"
        )
        try:
            op_start = await self.client._request(
                "POST",
                url,
                rpc_name="GenerateContentTuningScope",
                retry_deadline=self.retry.create_deadline_s,
                timeout=self.request_timeout,
                json=body,
            )
        except Exception as e:  # pylint: disable=broad-except
            logger.error(
                "[vertex_msrl_rest] GenerateContentTuningScope failed: %s",
                e,
            )
            duration_s = time.monotonic() - t0
            error_payload = {"__error__": f"create_failed: {e}"}
            # One error-status ledger blob per rollout ctx, recording the
            # wire call that never completed.
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

        op_name = op_start.get("name", "") if op_start else ""
        operation_id = op_name.rsplit("/", 1)[-1] if op_name else None
        logger.debug("[vertex_msrl_rest] Sampling op started: %s", op_name)

        try:
            op = await self.client.wait_for_lro(op_name, retry=self.retry)
        except Exception as e:  # pylint: disable=broad-except
            logger.error(
                "[vertex_msrl_rest] LRO poll failed for %s: %s",
                op_name,
                e,
            )
            duration_s = time.monotonic() - t0
            # Include op_name + operation_id in the failure payload so the
            # ledger blob cross-references the server-side operation record.
            error_payload = {
                "__error__": f"lro_poll_failed: {e}",
                "op_name": op_name,
                "operation_id": operation_id,
            }
            raw_uris = await dump_raw_fanout(
                ctxs,
                count=num_samples,
                request=body,
                response=error_payload,
                status="error",
                duration_s=duration_s,
                operation_id=operation_id,
            )
            # THIS PATH CAN BE A PROMPT OVERFLOW TOO: the serving stack also
            # reports the refusal as a FAILED operation ("code=3, message=
            # Input tokens exceed maximum prompt length"), not only as a
            # successful terminal LRO. Unclassified, that failsafe is
            # indistinguishable from a dead transport and gets scored as an
            # infra fault; it is not one -- the request was deterministically
            # rejected for its own size.
            extras = {"is_prompt_too_long": _is_prompt_too_long_text(str(e))}
            failsafes = [
                self._empty_result(
                    prompt, operation_id=operation_id, extras=extras
                )
                for _ in range(num_samples)
            ]
            return with_raw_call_uris(failsafes, raw_uris)

        response = op.get("response") or {}
        # Wall-clock span from create-POST to LRO-terminal, for the new
        # RAW_CAPTURE ledger (separate from the legacy MSRL_RAW_CAPTURE
        # single-file dump below, which is unaffected).
        duration_s = time.monotonic() - t0

        _cap = os.getenv("MSRL_RAW_CAPTURE")
        if _cap:
            _raw_capture(
                _cap,
                {
                    "ts": time.time(),
                    "backend": "vertex_msrl_rest",
                    "operation_id": operation_id,
                    "train_step_id": response.get("trainStepId"),
                    "request": body,
                    "response": response,
                },
            )

        raw_train_step_id = response.get("trainStepId")
        if raw_train_step_id is None or raw_train_step_id == "":
            train_step_id = None
        else:
            train_step_id = int(raw_train_step_id)
        tuning_candidates = response.get("tuningCandidates") or []
        if not tuning_candidates:
            # Deprecated candidates-map shape fallback.
            for cid, cand in (response.get("candidates") or {}).items():
                tuning_candidates.append(
                    {
                        "candidateId": cid,
                        "candidate": cand,
                        "generationTokenCount": None,
                        "thoughtsTokenCount": None,
                    }
                )
        if not tuning_candidates:
            logger.warning(
                "[vertex_msrl_rest] No candidates returned for op %s.",
                op_name,
            )
            # Per-candidate empty-status ledger blobs (one per rollout ctx).
            raw_uris = await dump_raw_fanout(
                ctxs,
                count=num_samples,
                request=body,
                response=response,
                status="empty",
                duration_s=duration_s,
                operation_id=operation_id,
            )
            extras = {"is_prompt_too_long": _detect_prompt_too_long(response)}
            failsafes = [
                self._empty_result(prompt, operation_id=operation_id, extras=extras)
                for _ in range(num_samples)
            ]
            return with_raw_call_uris(failsafes, raw_uris)

        # base_model is populated by create_tuning_job (new-job path) or
        # refreshed by wait_for_job_running.baseModel (existing-job path).
        # None here -> ChatCompletionResponse.model empty (fail-fast in
        # sample_chat_completion).
        _model_version: Optional[str] = getattr(self.client, "base_model", None)

        # Per-candidate ledger: for each tuningCandidate, write ONE blob
        # containing the response envelope + JUST that one candidate.
        # Preserves per-rollout provenance without duplicating siblings
        # into every blob. See the ABC docstring.
        envelope = {k: v for k, v in response.items() if k != "tuningCandidates"}

        # SamplingContext.prompt now stores the same typed shape we received
        # (proto or str); no dict-normalization needed since the type is
        # ``GenerateContentRequest | str | None``.
        #
        # Pre-compute per-candidate response slices (synchronous, no I/O)
        # so the fan-out below writes them without an inline loop body.
        # Slice ordering matches candidate ordering; both are indexed by i.
        cand_response_slices: List[Dict[str, Any]] = [
            {**envelope, "tuningCandidates": [tc]} for tc in tuning_candidates
        ]
        raw_uris: List[str | None] = await dump_raw_fanout(
            ctxs,
            count=len(tuning_candidates),
            request=body,
            response=cand_response_slices,
            status="ok",
            duration_s=duration_s,
            operation_id=operation_id,
        )
        results: List[SampleResult] = []
        for i, tc in enumerate(tuning_candidates):
            cand = tc.get("candidate") or {}
            # Parse the JSON candidate dict into a Candidate proto (same
            # helper gemini_api_endpoint uses).
            candidate_proto = _json_dict_to_candidate(cand)
            # The prompt is shared across candidates, but the service repeats
            # its token count on every tuningCandidate so each entry is
            # self-contained. The response-level usageMetadata that used to
            # carry it has been removed from the tuning-scope proto.
            _raw_pt = tc.get("promptTokenCount")
            _prompt_tokens: Optional[int] = (
                int(_raw_pt) if _raw_pt is not None else None
            )
            results.append(
                SampleResult(
                    tuning_candidate=TuningCandidate(
                        candidate_id=tc.get("candidateId") or "",
                        candidate=candidate_proto,
                        generation_token_count=(tc.get("generationTokenCount") or 0),
                        thoughts_token_count=(tc.get("thoughtsTokenCount") or 0),
                    ),
                    context=SamplingContext(
                        prompt=prompt,
                        operation_id=operation_id,
                        prompt_token_count=_prompt_tokens,
                        model_version=_model_version,
                        raw_call_uri=raw_uris[i],
                    ),
                    train_step_id=train_step_id,
                )
            )
        if any(r.context.prompt_token_count is None for r in results):
            # The candidate reported no prompt-token count. Every request
            # has a prompt, so unlike the generation counts there is no
            # legitimate shape that omits this; accepting None silently
            # disables in-loop compaction (which refuses to guess) and
            # blanks the context-size metrics. Surface it as a sampling
            # failure -- but return the failsafe rather than raising,
            # which would lose the task.
            logger.warning(
                "[vertex_msrl_rest] op %s reported no promptTokenCount "
                "(tuningCandidate keys: %s)",
                op_name,
                sorted(tuning_candidates[0].keys()),
            )
            failsafes = [
                self._empty_result(
                    prompt,
                    operation_id=operation_id,
                    extras={"is_malformed_usage": True},
                )
                for _ in range(num_samples)
            ]
            # The "ok" blobs are already on disk; reuse their URIs. There
            # may be fewer of them than num_samples, and the stamping
            # helper leaves the surplus failsafes unstamped.
            return with_raw_call_uris(failsafes, raw_uris)
        if len(results) < num_samples:
            logger.warning(
                "[vertex_msrl_rest] Requested %d samples, got %d for op %s",
                num_samples,
                len(results),
                op_name,
            )
        return results


def _finish_reason_ratios(
    trajectories: List[ScoredTrajectory],
) -> Dict[str, float]:
    """Per-``finish_reason`` share of agent steps across ``trajectories``.

    Returns ``{"sampler/finish_reason/{reason}": count / total}`` for each
    distinct reason, where ``total`` is the number of agent steps (i.e. the
    sum of all per-reason counts). The ratios therefore sum to 1.0. Steps
    with no recorded finish reason are bucketed as ``OTHER``. Returns an
    empty dict when there are no agent steps (so nothing is emitted).

    Mirrors ``FinishReasonMetricAggregator`` (the bare
    ``sampler/finish_reason/*`` twin) so both shapes agree; emitted by the
    trainer directly because these ``train/sampler/finish_reason/*`` keys
    bypass the batch-stage ``aggregate_by_spec`` reducer.
    """
    fr_counts: Dict[str, int] = {}
    total = 0
    for scored in trajectories:
        for step in scored.trajectory.steps:
            if step.source != "agent":
                continue
            total += 1
            fr = atif_extras.get_finish_reason(step) or "OTHER"
            fr_counts[fr] = fr_counts.get(fr, 0) + 1
    if total == 0:
        return {}
    return {f"sampler/finish_reason/{fr}": c / total for fr, c in fr_counts.items()}


def _extract_checkpoint_endpoint(response: dict[str, Any] | None) -> str | None:
    """Return the deployed checkpoint endpoint from a TrainStep response.

    The server includes a ``checkpoint`` object (with a full
    ``projects/{P}/locations/{L}/endpoints/{E}`` resource path) ONLY on
    checkpoint steps (``step % checkpointInterval == 0``). Its presence is
    the signal to run a checkpoint-endpoint eval; absence -> ``None``.
    """
    if not response:
        return None
    checkpoint = response.get("checkpoint") or {}
    endpoint = checkpoint.get("endpoint") or ""
    return endpoint or None


class VertexMsrlRestTrainer(Trainer):
    """Vertex MSRL trainer over REST/HTTPS."""

    def __init__(
        self,
        client: VertexMsrlRestClient,
        *,
        retry: LroRetryProfile = DEFAULT_LRO_RETRY,
        learning_rate_multiplier: Optional[float] = None,
    ):
        self.client = client
        self.retry = retry
        self.learning_rate_multiplier = learning_rate_multiplier

    async def train_step(
        self,
        trajectories: List[ScoredTrajectory],
        sync_trainer_and_sampler: bool = True,
    ) -> Dict[str, Any]:
        if not trajectories:
            return {}
        examples: List[Dict[str, Any]] = []
        skipped_missing_candidate_id = 0
        for scored in trajectories:
            trajectory_advantage = (
                scored.reward.normalized_reward
                if scored.reward.normalized_reward is not None
                else scored.reward.reward
            )
            for step in scored.trajectory.steps:
                if step.source != "agent":
                    continue
                cid = atif_extras.get_candidate_id(step)
                if not cid:
                    skipped_missing_candidate_id += 1
                    logger.warning(
                        "Missing candidateId; skipping step. %s",
                        simplify_trajectory_step_for_logging(step),
                    )
                    continue
                advantage = float(trajectory_advantage)
                step_advantage = atif_extras.get_advantage(step)
                if step_advantage is not None:
                    advantage += float(step_advantage)
                examples.append(
                    {
                        "candidateId": str(cid),
                        "advantage": advantage,
                    }
                )
        if not examples:
            logger.warning(
                "[vertex_msrl_rest] No valid training examples to send.",
            )
            return {"_train_step_error": "no valid training examples to send"}
        body: Dict[str, Any] = {
            "name": self.client.tuning_job_name,
            "reinforcementTuningTrainingDataBatch": {"examples": examples},
            "skipWeightSync": not sync_trainer_and_sampler,
        }
        if self.learning_rate_multiplier is not None:
            body["reinforcementTuningTrainStepHyperParameters"] = {
                "learningRateMultiplier": float(self.learning_rate_multiplier),
            }
        logger.info(
            "[vertex_msrl_rest] TrainStep examples=%d, skip_weight_sync=%s",
            len(examples),
            body["skipWeightSync"],
        )
        url = (
            f"{self.client._base_url}/{self.client.tuning_job_name}"
            f"{VertexMsrlRestClient._URL_TRAIN_STEP}"
        )
        t_step_start: float = time.perf_counter()
        try:
            op_start = await self.client._request(
                "POST",
                url,
                rpc_name="TrainStep",
                # The module constant, not `self.retry`: a train step is not
                # a sample, and giving the trainer its own create budget is
                # a policy question this structural change does not open.
                retry_deadline=_OP_START_RETRY_DEADLINE,
                json=body,
            )
        except Exception as e:  # pylint: disable=broad-except
            logger.error("[vertex_msrl_rest] TrainStep failed to start: %s", e)
            return {"_train_step_error": f"TrainStep failed to start: {e}"}
        op_name = op_start.get("name", "") if op_start else ""
        if not op_name:
            logger.error(
                "[vertex_msrl_rest] TrainStep returned no operation name: %r",
                op_start,
            )
            return {
                "_train_step_error": (
                    f"TrainStep returned no operation name: {op_start!r}"
                )
            }
        logger.info("[vertex_msrl_rest] TrainStep op started: %s", op_name)
        try:
            final = await self.client.wait_for_lro(op_name, retry=self.retry)
        except Exception as e:  # pylint: disable=broad-except
            logger.error(
                "[vertex_msrl_rest] TrainStep LRO failed for %s: %s",
                op_name,
                e,
            )
            return {"_train_step_error": f"TrainStep LRO failed for {op_name}: {e}"}

        t_train_step_time = time.perf_counter() - t_step_start
        metrics: Dict[str, Any] = {
            # Operation metadata for progress observability (not ML metrics).
            "_train_step_operation_name": op_name,
            "_train_step_num_examples": len(examples),
            "_train_step_duration_seconds": t_train_step_time,
        }
        if skipped_missing_candidate_id:
            metrics["trajectory/missing_candidate_id"] = skipped_missing_candidate_id
        response = final.get("response") or {}
        raw_wire_metrics = {
            m["metricName"]: m.get("value")
            for m in (response.get("metrics", []) or [])
            if m.get("metricName")
        }
        # Undeclared wire metricNames are dropped with a one-time WARN
        # (forward-compat). Add a TrainerMetric entry to TRAINER_METRICS
        # to surface a new one. ``_wire_specs`` is discarded here:
        # ``core.metric_spec.METRIC_SPECS`` is the source of truth once
        # the sink layer ingests them.
        mapped_wire_metrics, _wire_specs = map_trainer_wire_metrics(raw_wire_metrics)
        metrics.update(mapped_wire_metrics)
        if response.get("completedTrainStepId"):
            metrics["completed_train_step_id"] = response["completedTrainStepId"]
        checkpoint_endpoint = _extract_checkpoint_endpoint(response)
        if checkpoint_endpoint:
            # `_`-prefixed: operation metadata for eval scheduling, NOT an
            # ML metric. Consumed by train_and_eval_loop, dropped by the
            # metric-spec sink (like _train_step_operation_name).
            metrics["_checkpoint_endpoint"] = checkpoint_endpoint
        metrics.update(parse_candidate_audit(response))
        # sampler/finish_reason/* distribution from input trajectories, as
        # per-reason RATIOS (share of agent steps) so the family sums to 1.0 --
        # matching the bare ``sampler/finish_reason/*`` twin. Key namespace
        # matches FinishReasonMetricAggregator's prefix.
        metrics.update(_finish_reason_ratios(trajectories))
        return metrics


__all__ = [
    # Endpoint helpers (the production template only).
    "DEFAULT_ENDPOINT",
    "DEFAULT_LOCATION",
    "PROD_ENDPOINT_TEMPLATE",
    "default_endpoint",
    # Public classes.
    "VertexMsrlRestClient",
    "VertexMsrlRestSampler",
    "VertexMsrlRestTrainer",
]
