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

"""Raw-request/response capture for the sampler layer.

Writes each backend's wire-level `(request, response)` pair to
`<trial_dir>/artifacts/raw_llm_calls/call_<NNNNNN>.json`, following the
trial tree's raw-LLM ledger blob shape.

**Default: ENABLED (opt-out).** Every training run captures raw LLM
traffic unless the operator explicitly opts out via ``RAW_CAPTURE=0``
(also accepts ``false``, ``no``, ``off`` case-insensitively;
leading/trailing whitespace is stripped). The default flipped from
opt-in to opt-out as a follow-up to the initial capture-ledger
landing, once operators signed off on the storage cost. Recognized
truthy values (``1``, ``true``, ``yes``, ``on``, case-insensitively)
also enable — same as before. Any unrecognized non-empty value (typo
like ``RAW_CAPTURE=trueee``, ``RAW_CAPTURE=1yes``) is treated as
ENABLED (matching the default) but logged as a WARN at import time so
the mistake is visible at process start rather than silently baking a
misconfigured env into the run.

**IMPORTANT — env-var timing.** ``RAW_CAPTURE`` is read exactly ONCE at
module import time and cached in a module-level constant for hot-path
performance (~1 ns bool check vs ~100 ns ``os.environ`` lookup per
`dump_raw` call). Setting ``RAW_CAPTURE=0`` AFTER the recipe process
has imported this module is a silent no-op — the default-on posture
stays for the process lifetime. Always set ``RAW_CAPTURE=0`` in the
SHELL ENVIRONMENT before invoking the recipe (``export RAW_CAPTURE=0;
python -m ...``), or via the smoke wrappers' ``-e RAW_CAPTURE=0``
forwarding, if you need to opt out. The ``_reload_capture_enabled()``
function exists solely for test setup/teardown and is not a supported
runtime toggle.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import dataclasses
import json
import logging
import os
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Canonical truthy / falsy strings accepted for the RAW_CAPTURE env
# var. Kept as module-level frozensets so `_parse_capture_enabled` is a
# pair of O(1) membership tests. Whitespace-stripping + .lower() happen
# in the parser before lookup. Empty string is NOT in either set — it
# is handled as "unset -> default-on" in the parser.
_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSY_VALUES = frozenset({"0", "false", "no", "off"})


def _parse_capture_enabled() -> bool:
    """Parse the RAW_CAPTURE env var into a bool.

    **Default: ENABLED** (opt-out via ``RAW_CAPTURE=0`` / ``false`` /
    ``no`` / ``off``, case-insensitive).

    Accepts standard truthy strings (case-insensitive, whitespace-
    stripped): ``1``, ``true``, ``yes``, ``on``. UNSET or empty string
    also enables — this matches the ``unset -> os.environ.get("",
    "") -> "" -> enabled`` chain and is the operational default when
    nobody explicitly configures the env var. Only explicit falsy
    values (``0``, ``false``, ``no``, ``off``) disable capture. Any
    other non-empty value is treated as ENABLED (matching the
    default-on posture) but logged as a WARN so typos like
    ``RAW_CAPTURE=trueee`` or ``RAW_CAPTURE=1yes`` are visible at
    process start rather than silently baking a misconfigured env
    into the run.

    Design note: default-on landed as a follow-up to the initial
    capture-ledger change after the opt-in shape shipped and got
    operator sign-off on the storage cost. Every training run now
    captures raw LLM traffic to the trajectory tree unless the
    operator explicitly opts out.
    """
    raw = os.environ.get("RAW_CAPTURE", "").strip().lower()
    if raw == "":  # UNSET or empty -> default ENABLED
        return True
    if raw in _TRUTHY_VALUES:
        return True
    if raw in _FALSY_VALUES:
        return False
    logger.warning(
        "RAW_CAPTURE=%r is not a recognized truthy or falsy value; "
        "treating as ENABLED (default). Set RAW_CAPTURE=0 (or "
        "'false'/'no'/'off') to explicitly disable capture.",
        os.environ.get("RAW_CAPTURE"),
    )
    return True


# Module-level cache of the RAW_CAPTURE env var, read once at import
# time. Refreshed via _reload_capture_enabled() in tests. This keeps the
# hot-path check to a single bool comparison (~1ns) instead of an
# os.environ lookup (~100ns).
_CAPTURE_ENABLED: bool = _parse_capture_enabled()


def _reload_capture_enabled() -> None:
    """Re-read the RAW_CAPTURE env var and refresh the module-level cache.

    Intended for test setup/teardown only; not safe to call while
    `dump_raw` is in flight (a concurrent write may see a stale
    view of the flag). Shares parsing rules with import-time init via
    `_parse_capture_enabled` so both paths agree on what counts as
    truthy/falsy.
    """
    global _CAPTURE_ENABLED
    _CAPTURE_ENABLED = _parse_capture_enabled()


class Counter:
    """Async-safe monotonic counter shared within one CaptureContext.

    Each rollout gets its own Counter so call_NNNNNN.json indices are
    per-rollout (matching the contract's per-trial monotonic ledger).
    Concurrent sample() calls within one rollout are safe.
    """

    def __init__(self) -> None:
        self._n = 0
        self._lock = asyncio.Lock()

    async def next(self) -> int:
        async with self._lock:
            n = self._n
            self._n += 1
            return n


@dataclasses.dataclass(frozen=True)
class CaptureContext:
    """Where to write ledger files, and what to key them by.

    Attributes:
        trial_dir: absolute path to the per-rollout dir
            (`<log_dir>/atif/<recipe>/<job>/<task>-<rollout>/`) under
            which ledger blobs land at
            ``<trial_dir>/artifacts/raw_llm_calls/``. This dir may not
            exist at ``CaptureContext``-construction time — the
            AtifDiskWriter creates it later in
            ``on_trajectory_produced`` (which fires AFTER the rollout
            finishes), whereas ``dump_raw`` runs DURING sampling and
            needs to write blobs before that. As a result, ``dump_raw``
            creates the full path chain (``trial_dir`` and its
            ``artifacts/raw_llm_calls/`` subdir) via
            ``mkdir(parents=True, exist_ok=True)`` on the first ledger
            write of the rollout. Callers do NOT need to pre-create
            ``trial_dir``.
        trajectory_id: value stamped into each ledger blob's
            `trajectory_id` field. Conventionally the ATIF session_id.
        counter: shared monotonic counter for `call_NNNNNN.json` names.
        on_written: optional callback invoked with the local Path of each
            successfully-written ledger blob. Used by AtifDiskWriter to
            enqueue blobs for background GCS upload alongside the
            trajectory JSONs. Called AFTER the local write succeeds;
            failures inside the callback are swallowed (log-and-continue)
            so upload-side issues don't crash the sampler hot path.
        blob_uri_prefix: Optional URI prefix used to build the directly-
            resolvable URI returned by ``dump_raw`` and stamped on
            ``step.extra.llm_call_uri``. Populated by
            ``AtifDiskWriter.capture_ctx_for``:

            - When ``atif_output`` is a ``gs://`` URL, this is the full
              GCS path pointing at the same trial dir GCS uploads land
              in (``gs://<bucket>/<prefix>/<recipe>/<job>/<task>-<rollout>``).
              Chosen so a consumer with just the URI (and no other
              context) can ``gcloud storage cat`` the ledger blob
              directly.
            - When ``atif_output`` is local, this is
              ``str(trial_dir.absolute())`` so the URI opens with
              ``open()`` / ``os.path.exists`` without needing to know a
              trial-dir root or cwd.
            - When ``None`` (e.g. bare tests that construct
              ``CaptureContext`` without going through
              ``AtifDiskWriter``), ``dump_raw`` falls back to the legacy
              trial-dir-relative URI form for backward compatibility.
    """

    trial_dir: Path
    trajectory_id: str
    counter: Counter
    on_written: Callable[[Path], None] | None = None
    blob_uri_prefix: str | None = None


# The CLIENT-form request (OpenAI chat-completions) for the call currently
# in flight, set by ``Sampler.sample_chat_completion``.
#
# Why a ContextVar and not a parameter: ``dump_raw`` is called from deep
# inside each backend's ``sample_n``, which only ever sees the converted
# VENDOR body. Threading the client form down would mean changing the
# Sampler ABC and every backend; stashing it on ``CaptureContext`` would
# mean mutating a frozen dataclass shared across a rollout's calls. A
# ContextVar is per-task under asyncio, so concurrent rollouts cannot see
# each other's request.
_CLIENT_REQUEST: contextvars.ContextVar[dict[str, Any] | None] = (
    contextvars.ContextVar("raw_capture_client_request", default=None)
)


@contextlib.contextmanager
def client_request_scope(request: dict[str, Any] | None) -> Iterator[None]:
    """Record ``request`` as the client-form body for calls made inside."""
    token = _CLIENT_REQUEST.set(request)
    try:
        yield
    finally:
        _CLIENT_REQUEST.reset(token)


def _json_safe(obj: Any) -> Any:
    """Fallback JSON encoder for objects json.dumps doesn't know about.

    Protobuf messages are converted via `MessageToDict` (with a
    `str(obj)` fallback if the proto import fails). All other unknown
    types raise `TypeError` — silently stringifying (e.g. numpy arrays
    to `'[[1 2] [3 4]]'`, datetimes to `'2026-...'`) would corrupt the
    ledger and only surface as a mismatch at replay-diff time.

    `dump_raw`'s outer `except Exception` catches this TypeError,
    logs a WARN, and returns None, so a stray un-encodable object in
    the request/response body will not crash the training run — it
    will just skip the ledger entry for that call, loudly enough that
    an operator can notice in the logs.
    """
    if hasattr(obj, "SerializeToString"):  # protobuf message
        try:
            from google.protobuf.json_format import MessageToDict

            return MessageToDict(obj, preserving_proto_field_name=True)
        except Exception:
            return str(obj)
    raise TypeError(
        f"raw_capture cannot JSON-encode object of type "
        f"{type(obj).__module__}.{type(obj).__qualname__!r}; "
        f"callers must pass plain-dict request/response bodies"
    )


async def dump_raw(
    *,
    ctx: CaptureContext | None,
    request: dict[str, Any],
    response: dict[str, Any],
    status: str,
    duration_s: float,
    operation_id: str | None = None,
) -> str | None:
    """Write one call_NNNNNN.json under ctx.trial_dir; return a resolvable URI.

    Each blob carries the full inline ``request`` and ``response`` from the
    wire call, matching the trial tree's raw-LLM ledger
    blob shape. The returned URI is stamped on the trajectory step as
    ``step.extra.llm_call_uri``.

    Args:
        ctx: The per-rollout capture context. None disables the write.
        request: The wire-level request body the backend POSTed. Must be
            JSON-serializable (with `_json_safe` fallback for proto).
        response: The wire-level response body the backend received, or
            an error-envelope dict. Same serialization contract as request.
        status: One of "ok" | "empty" | "error", per contract.
        duration_s: Wall-clock seconds of the upstream call.

    Returns:
        A directly-resolvable URI for the written blob:

        - ``gs://<bucket>/<prefix>/.../artifacts/raw_llm_calls/call_NNNNNN.json``
          when ``ctx`` was built by ``AtifDiskWriter`` with a
          ``gs://`` ``atif_output`` — the URI resolves via
          ``gcloud storage cat`` to the actual uploaded blob.
        - Absolute local path
          ``/abs/path/.../artifacts/raw_llm_calls/call_NNNNNN.json``
          when ``atif_output`` was local — the URI opens with
          ``open()`` without needing to know the trial-dir root or cwd.
        - Trial-dir-relative
          ``artifacts/raw_llm_calls/call_NNNNNN.json`` as a fallback
          for bare ``CaptureContext`` instances with
          ``blob_uri_prefix=None`` (test setups; not produced in
          normal recipe use).

        ``None`` if capture is disabled (via env var or ``ctx=None``)
        or if the write failed. Never raises — a broken disk must not
        crash a training run (defense-in-depth per user profile).
    """
    if ctx is None or not _CAPTURE_ENABLED:
        return None
    idx = await ctx.counter.next()
    filename = f"call_{idx:06d}.json"
    outdir = ctx.trial_dir / "artifacts" / "raw_llm_calls"
    rel_path = f"artifacts/raw_llm_calls/{filename}"
    uri = f"{ctx.blob_uri_prefix}/{rel_path}" if ctx.blob_uri_prefix else rel_path
    try:
        outdir.mkdir(parents=True, exist_ok=True)
        blob: dict[str, Any] = {
            "trajectory_id": ctx.trajectory_id,
            "call_idx": idx,
            "ts": time.time(),
            "status": status,
            "duration_s": duration_s,
            "request": request,
            "response": response,
        }
        if operation_id is not None:
            blob["operation_id"] = operation_id
        # The request as the AGENT wrote it, before vendor conversion:
        # OpenAI messages and — the reason this exists — the tool schemas
        # in the shape the agent declared them. `request` above is the
        # post-conversion vendor body, so without this there is no record
        # of what the client actually asked for, and tool RENDERING has been
        # measured to be worth ~2.5 points of benchmark pass rate.
        client_request = _CLIENT_REQUEST.get()
        if client_request is not None:
            blob["client_request"] = client_request
        # Blocking write on the event loop stalls concurrent coroutines
        # (single-threaded asyncio). AtifDiskWriter.on_trajectory_produced
        # wraps its writes in asyncio.to_thread for this same reason; keep
        # dump_raw consistent so RAW_CAPTURE doesn't inject a new blocking
        # syscall into the sampler hot path.
        payload = json.dumps(blob, indent=2, default=_json_safe)
        outpath = outdir / filename
        await asyncio.to_thread(outpath.write_text, payload)
    except Exception as e:  # noqa: BLE001 — see docstring
        logger.warning("dump_raw failed for %s: %s", filename, e)
        return None
    # Local write succeeded; invoke the upload callback (best-effort).
    # A failure here (e.g. GCS uploader queue full) must NOT crash the
    # sampler hot path or invalidate the local URI already computed --
    # the ledger blob exists on disk regardless. See design gap fixed
    # by the on_written wire-through: without it, ledger blobs landed
    # locally but the trajectory JSONs uploaded to GCS pointed at paths
    # that only existed on the training VM's disk.
    if ctx.on_written is not None:
        try:
            ctx.on_written(outpath)
        except Exception:  # noqa: BLE001 — best-effort upload callback
            logger.warning(
                "CaptureContext.on_written callback failed for %s",
                outpath,
                exc_info=True,
            )
    return uri


async def dump_raw_fanout(
    ctxs: Sequence[CaptureContext | None] | None,
    *,
    count: int,
    request: dict[str, Any],
    response: Any,
    status: str,
    duration_s: float,
    operation_id: str | None = None,
) -> list[str | None]:
    """One blob per candidate, written concurrently. Returns their URIs.

    Every sampler needs the same four things around :func:`dump_raw` and
    was doing them by hand at thirteen call sites: index ``capture_ctxs``
    per candidate, fan out, gather, and keep the returned URIs aligned with
    the results so each one can be stamped onto its own step. Three bugs
    came out of that duplication -- one backend that captured nothing, a
    capture scope lost to a subclass override, and a replay sampler whose
    zero call sites nobody can classify as deliberate or forgotten.

    ``ctxs`` is POSITIONAL, one per candidate, and is deliberately tolerant
    of being short: a caller that passes a single context while asking for
    four candidates gets capture for candidate 0 and ``None`` for the rest,
    rather than an ``IndexError`` in the middle of a paid-for sampling
    call. Losing a blob must never lose a sample -- the ledger is evidence,
    not the product.

    ``response`` may be either a single object shared by every candidate
    (an error payload, a whole-response envelope) or a SEQUENCE with one
    entry per candidate (per-candidate response slices). The sequence form
    is detected by length, so a shared ``dict`` is never mistaken for one.

    Never raises: :func:`dump_raw` swallows its own errors and returns
    ``None``, and this only gathers them.
    """
    ctxs = ctxs or []
    per_candidate = (
        isinstance(response, (list, tuple)) and len(response) == count
    )
    return list(
        await asyncio.gather(
            *[
                dump_raw(
                    ctx=ctxs[i] if i < len(ctxs) else None,
                    request=request,
                    response=response[i] if per_candidate else response,
                    status=status,
                    duration_s=duration_s,
                    operation_id=operation_id,
                )
                for i in range(count)
            ]
        )
    )
