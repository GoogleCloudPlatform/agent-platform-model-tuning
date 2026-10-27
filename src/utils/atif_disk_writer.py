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

"""On-disk ATIF trajectory writer (``TrajectoryCallback`` implementation).

Writes an ATIF-conformant on-disk tree during rollout and best-effort
uploads each file to a GCS prefix in the background.

Directory layout (rooted at ``<root_dir>``)::

    <root_dir>/
      <recipe>/
        <tuning_job_id>/
          config.json                     # once per job (on_job_started)
          result.json                     # once per job (on_job_completed)
          <task_id>-<rollout_idx>/        # one dir per rollout
            result.json                   # per-trial summary
            agent/trajectory.json         # atif_codec.to_disk(scored)[0]
            verifier/reward.json          # atif_codec.to_disk(scored)[1],
                                          #   OMITTED when scored.reward is
                                          #   None ("never measured")
            verifier/reward-details.json  # optional per-criterion breakdown
            artifacts/                    # bulky per-trial evidence,
              raw_llm_calls/call_*.json   #   written by OTHER components
              env_final_state/            #   env.snapshot() at exit (F4)

``env_final_state/`` is a DIRECTORY, not a single document: what an env
has to show for itself at exit is env-specific and not always JSON — a
container env may leave a file tree, a shell env a transcript. Fixing it
as one `.json` would force every env into a shape only some of them have.

``trial_dir_for`` exposes the trial-dir naming rule to callers that run
before this writer does; the journal calls it to place the
``artifacts/`` entries above next to the trajectory they belong to.

Not thread-safe across processes; each process owns its own instance.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
from collections import deque
import logging
import os
import random
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Awaitable, Callable

from google.api_core import exceptions as gax

from core.atif_codec import to_disk, _nan_to_none
from core.atif_extras import get_phase, get_rollout_status, pop_reward_details
from core.atif_trajectory import Reward, ScoredTrajectory, Trajectory
from core.session import sanitize_path_segment
from utils.logger import _dump_config, _PermissiveJSONEncoder
from utils.raw_capture import CaptureContext, Counter

logger = logging.getLogger(__name__)

# Path segments must match ``^[A-Za-z0-9._\-]+$`` per the ATIF contract;
# everything else is replaced with ``_``.
_UNSAFE_SEGMENT_CHARS = re.compile(r"[^A-Za-z0-9._\-]")

# rollout_idx is the integer just before the trailing run-start ISO stamp
# in a session_id built by ``core.session.build_session_id``:
#   f"{recipe}-{tuning_job_id}-{task_id}-{rollout_idx}-{YYYYmmddTHHMMSSZ}-{seq}"
# The trailing ``-{seq}`` dispatch counter was added to make session_ids
# epoch-unique; the ``(?:-\d+)?`` suffix keeps the regex backward-compatible
# with session_ids that predate the counter.
_ROLLOUT_IDX_RE = re.compile(r"-(\d+)-\d{8}T\d{6}Z(?:-\d+)?$")

# Finalized trials kept on disk before reclaim. Small and constant, so disk
# stays flat however long the run is, while the most recent trials remain
# locally inspectable.
_DEFAULT_TRIAL_GRACE_COUNT = 8

# Hard bound on finalized-but-not-yet-reclaimable trials. Reached only when
# uploads are FAILING rather than lagging; past it the oldest is reclaimed
# regardless, freeing whatever uploaded and keeping the rest.
_MAX_PENDING_FINALIZED = 512

# -- Partial-snapshot constants -------------------------------------------
# Write a partial trajectory snapshot every N agent steps.
# Override via $ATIF_PARTIAL_STEP_INTERVAL for testing.
_PARTIAL_STEP_INTERVAL = int(os.environ.get("ATIF_PARTIAL_STEP_INTERVAL", "2"))
# Minimum wall-clock seconds between partial uploads for the same trial.
# Override via $ATIF_PARTIAL_MIN_INTERVAL_S for testing.
_PARTIAL_MIN_INTERVAL_S = int(os.environ.get("ATIF_PARTIAL_MIN_INTERVAL_S", "300"))

_FALLBACK_TASK_ID = "unknown-task"

# Truthy "disable" sentinels for atif_output; truthy so they survive
# empty-only default refill and give a real off switch. Matched
# case-insensitively after stripping whitespace.
_ATIF_OFF_SENTINELS = frozenset({"off", "none", "disable", "disabled"})

# Default background upload worker pool size. Small trajectory files are
# latency-bound, so throughput scales almost linearly with workers. Measured
# with this module's own uploader (1000 files of 2-20 KB per level, real GCS,
# from a workstation):
#   workers  files/s  p50 ms  p95 ms  retries
#        10     15.0     667     782        0
#        32     45.9     654     852        1
#        64     82.5     620     911        4
#       128    147.6     653    1779        5
# 32 triples the old default's throughput for +9% p95 and ~0.1% retries,
# far below GCS's per-bucket write limits; at 128 queueing starts (p95 x2).
_DEFAULT_MAX_CONCURRENT_UPLOADS = 32

# -- Final-drain sizing ----------------------------------------------------
# At the normal end of a run nothing competes for the uploader any more, so
# the drain runs with _DRAIN_CONCURRENCY_FACTOR x the run's worker count, and
# its timeout is sized to the backlog instead of fixed:
#   timeout = clamp(pending / rate * _DRAIN_TIMEOUT_MARGIN,
#                   _DRAIN_MIN_TIMEOUT_S, max_timeout_s)
# where rate = run workers / mean latency of the last _DRAIN_RATE_WINDOW
# successful uploads. The rate is taken at the RUN worker count, so the extra
# drain workers only make the drain finish earlier than its bound; the margin
# absorbs retries. A large production run, with a backlog queued, sustained
# ~50 files/s at 10 workers (~0.2 s per small upload, i.e. ~160 files/s at the
# 32-worker default); even at the slower ~0.65 s measured from a workstation,
# 32 workers move ~46 files/s, so a ~25k-file backlog needs ~540 s WITHOUT the
# drain's extra workers; the default cap (20 min) leaves >2x margin. Whatever
# is still pending at the cap is reported.
_DRAIN_CONCURRENCY_FACTOR = 2
_DRAIN_MIN_TIMEOUT_S = 60.0
_DEFAULT_DRAIN_MAX_TIMEOUT_S = 1200.0
_DRAIN_TIMEOUT_MARGIN = 1.5
_DRAIN_RATE_WINDOW = 200
# Drain timeout when the draining task is being CANCELLED (a SIGTERM/SIGINT
# turned into task cancellation): a terminating host is typically killed
# ~30 s after the signal, and the caller still has its own cleanup (e.g.
# cancelling the remote job) after the drain, so this stays short.
_CANCELLED_DRAIN_TIMEOUT_S = 10.0


def _add_viewer_facets(config_dict: dict[str, Any], config: Any) -> None:
    """Add the trajectory-viewer's config.json facet keys IN PLACE.

    The viewer populates its Datasets / Agents / Providers / Models / Environment
    columns from a Harbor-shaped ``config.json`` -- top-level ``agents`` /
    ``datasets`` / ``environment`` blocks -- but the emitted ``config.json`` is a
    generic ``RunConfig`` dump (``agent`` / ``backend`` / ``task`` singular), so
    those columns come up empty. This derives the viewer blocks from the
    RunConfig and merges them in without touching the existing dump.

    Shapes the viewer reads (see the traj-interface viewer's ``_parse_config``):
      agents:      [{"name": <agent>, "model_name": "<provider>/<model>",
                     "environment_type": <env>}]
      datasets:    [{"name": <dataset>}]
      environment: {"type": <env>}

    All reads are defensive ``getattr`` / ``dict.get`` with fallbacks because
    ``model`` / ``environment`` / ``eval_data_path`` are recipe-overlay fields,
    not base-config fields -- a recipe missing one just yields a thinner block,
    never an error. No-op if the viewer keys are already present.
    """
    agent = config_dict.get("agent") or {}
    backend = config_dict.get("backend") or {}
    task = config_dict.get("task") or {}

    # provider = the backend's short_name (a @property, so absent from the dump);
    # read it off the live config object. model = the model id the run used.
    provider = ""
    backend_obj = getattr(config, "backend", None)
    if backend_obj is not None:
        provider = str(getattr(backend_obj, "short_name", "") or "")
    model = str(agent.get("model") or backend.get("model") or "")
    env = agent.get("environment")
    env_type = str(env) if env else None

    if "agents" not in config_dict and model:
        model_name = f"{provider}/{model}" if provider else model
        entry: dict[str, Any] = {"name": provider or model, "model_name": model_name}
        if env_type:
            entry["environment_type"] = env_type
        config_dict["agents"] = [entry]

    if "datasets" not in config_dict:
        data_path = task.get("eval_data_path")
        if data_path:
            leaf = str(data_path).rstrip("/").rsplit("/", 1)[-1]
            config_dict["datasets"] = [{"name": leaf}]

    if "environment" not in config_dict and env_type:
        config_dict["environment"] = {"type": env_type}


def _sanitize_path_segment(segment: str) -> str:
    """Log-and-delegate wrapper around ``core.session.sanitize_path_segment``.

    The transformation itself lives at the session-id mint point so the
    id the Agent hands us is already path-safe; this wrapper only adds
    the operator-facing WARNING for the fallback paths that still
    sanitize raw task_ids here.
    """
    safe = sanitize_path_segment(segment)
    if segment and safe != segment:
        logger.warning(
            "AtifDiskWriter: path segment %r rewritten to %r (contained "
            "characters outside [A-Za-z0-9._-])",
            segment,
            safe,
        )
    return safe


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` as ONE indivisible replacement.

    ``Path.write_text`` truncates and then streams, so two writers racing
    the same path do not produce one of the two documents: the shorter
    write lands inside the longer one's bytes and the file ends up as the
    head of one document with the tail of the other glued on. On
    ``agent/trajectory.json`` that reads as the final trajectory followed
    by a mid-rollout snapshot's ``"_partial": true}}`` — unparseable, and
    a whole subtree of the record unreadable with it.

    A sibling temp file plus ``os.replace`` is atomic on POSIX, so every
    reader (the viewer, the GCS uploader, the next test) sees either the
    old bytes or the new ones and never a mix of the two. The temp file is
    created in the destination's own directory because ``os.replace``
    is only atomic within a filesystem.
    """
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(text)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _trial_name(session_id: str | None, task_id: str, rollout_idx: int | None) -> str:
    """Directory name for one rollout's artifacts.

    ``session_id`` IS the trial name: it is minted once per dispatch
    (``core.session.build_session_id``), already sanitized to the ATIF
    path charset, and unique per dispatch -- so re-running the same task
    in a later eval sweep or a later training epoch can no longer
    overwrite the earlier rollout's files.

    Deriving the name from anything else is what this function exists to
    prevent: the ledger, the partial snapshot and the final write each
    resolve the trial dir at different times, and any re-derivation is a
    chance for them to disagree.

    The ``task_id``/``rollout_idx`` fallback preserves the historical
    layout for callers that have no session_id (direct ``rollout()``
    users, tests, trajectories restored from older runs).

    Sanitization is re-applied defensively: ids minted by
    ``build_session_id`` are already safe (no-op), but a trajectory may
    arrive from an external producer that never went through it.
    Sanitizing here -- one deterministic function of one input -- keeps
    the three call sites in agreement either way.
    """
    if session_id:
        return sanitize_path_segment(session_id)
    if rollout_idx is None:
        # Unparseable identity -> unique fallback so concurrent unknowns
        # don't collide into <task_id>-0/.
        return f"{task_id}-unknown-{uuid.uuid4().hex[:8]}"
    return f"{task_id}-{rollout_idx}"


def build_atif_writer(
    *,
    atif_output: str,
    log_dir: str,
    recipe: str,
    tuning_job_id: str,
    max_concurrent_uploads: int = _DEFAULT_MAX_CONCURRENT_UPLOADS,
    uploader_factory: Callable[..., Any] | None = None,
) -> AtifDiskWriter | None:
    """Construct an ``AtifDiskWriter`` from a recipe's ``atif_output``
    config, or ``None`` when logging is disabled.

    ``atif_output`` semantics follow :func:`parse_atif_output`:
    ``""`` / ``"off"`` / ``"none"`` disable (returns ``None``); ``"local"``
    writes locally only; ``"gs://..."`` writes locally + uploads with a
    background pool of up to ``max_concurrent_uploads`` workers. Files are
    rooted at ``<log_dir>/atif``; the writer nests
    ``<recipe>/<tuning_job_id>/`` beneath that.
    """
    if _is_atif_disabled(atif_output):
        return None
    gcs_prefix = parse_atif_output(atif_output)  # raises on bad scheme
    writer = AtifDiskWriter(
        root_dir=f"{log_dir}/atif",
        recipe=recipe,
        tuning_job_id=tuning_job_id,
        gcs_prefix=gcs_prefix,
        max_concurrent_uploads=max_concurrent_uploads,
        uploader_factory=uploader_factory,
    )
    # Register HERE rather than in each entry point: this factory is the one
    # place both the train and the eval path go through, so neither can
    # forget, and a new caller gets crash-recovery without knowing it exists.
    # Registration records the gcs_prefix this run is really using, so a
    # later reclaimer never has to re-derive it from a directory name.
    writer.register_for_reclaim()
    return writer


def _default_task_id_extractor(task_data: Any) -> str:
    """Best-effort ``task_id`` from a TaskData-like object.

    Falls back to ``"unknown-task"`` (with a WARNING) when missing so a
    bad task_id degrades the path rather than crashing the rollout.
    """
    task_id = getattr(task_data, "task_id", None)
    if task_id:
        return str(task_id)
    logger.warning(
        "AtifDiskWriter: task_data has no usable task_id (%r); using %r",
        task_id,
        _FALLBACK_TASK_ID,
    )
    return _FALLBACK_TASK_ID


def _extract_rollout_idx_from_session_id(session_id: str | None) -> int | None:
    """Parse ``rollout_idx`` out of a session_id tail.

    Returns the parsed integer (including a legitimate ``0``), or ``None``
    when unparseable. ``None`` is a distinct signal from ``0`` so
    unparseable ids are routed to a unique fallback dir instead of
    colliding every such rollout into ``<task_id>-0/``.
    """
    if not session_id:
        return None
    m = _ROLLOUT_IDX_RE.search(session_id)
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------
# GCS uploader (best-effort background with retry + accounting)
# --------------------------------------------------------------------------

# Transient errors: retried with exponential backoff + jitter.
_RETRYABLE: tuple[type[BaseException], ...] = (
    gax.TooManyRequests,  # 429 rate limit
    gax.ServiceUnavailable,  # 503
    gax.DeadlineExceeded,  # 504 / timeout
    gax.InternalServerError,  # 500
    ConnectionError,  # network / DNS blips
    TimeoutError,
)

# Permanent errors: fail-fast (no retry), mapped to a ledger category.
_PERMANENT: dict[type[BaseException], str] = {
    gax.Forbidden: "auth",  # 403 IAM missing
    gax.Unauthorized: "auth",  # 401
    gax.NotFound: "not_found",  # bucket / prefix wrong
    FileNotFoundError: "local_gone",  # local file deleted mid-flight
}

# Substring identifying an UNAMBIGUOUSLY-transient 403: the GCS client bills API
# usage to a quota project and intermittently gets "serviceusage.services.use
# denied" under concurrent load (observed ~2.6% of uploads in a wide eval, spread
# evenly, NOT reproducible on retry). This is a flaky quota-project probe, never a
# bucket-ACL denial, so it is ALWAYS retried (independent of the circuit breaker
# below).
_TRANSIENT_FORBIDDEN_MARKER = "serviceusage.services.use"


def _is_transient_forbidden(exc: BaseException) -> bool:
    """True for the intermittent serviceusage-quota 403 (always retryable)."""
    return isinstance(exc, gax.Forbidden) and _TRANSIENT_FORBIDDEN_MARKER in str(exc)


# A bare Forbidden (e.g. "storage.objects.create denied") is AMBIGUOUS: it is the
# identical message for both (a) a genuine "this identity cannot write to this
# bucket" denial -- which denies EVERY upload -- and (b) a flaky server-side authz
# hiccup under heavy concurrency -- which hits a tiny scattered fraction while the
# same identity uploads the rest fine (observed ~0.01% in a 5-config eval). A
# substring cannot tell them apart, so we distinguish by PATTERN via a circuit
# breaker: if ANY upload has ever succeeded, the identity demonstrably has write
# access, so a bare Forbidden is a flake -> retry. If NOTHING has succeeded yet,
# assume a genuine misconfiguration -> fail fast (don't burn the backoff schedule
# on every one of N files). See _is_flaky_forbidden().
def _is_bare_forbidden(exc: BaseException) -> bool:
    return isinstance(exc, gax.Forbidden) and not _is_transient_forbidden(exc)


# Two refinements keep that flake tolerance from turning a DETERMINISTIC denial
# into thousands of retries, each sleeping through the backoff while it holds an
# upload slot:
#
# 1. A 403 naming storage.objects.delete is an OVERWRITE denial. The first upload
#    of a path needs only objects.create, but every re-upload of it (a growing
#    trajectory.json, a partial snapshot replaced by the final one) replaces an
#    existing object and also needs objects.delete. It hits every overwrite, so
#    it fails fast regardless of earlier successes.
# 2. Per-signature breaker: a bare Forbidden is reduced to a signature (the
#    denied storage.objects.* permission, else the message without URLs); once
#    one signature has hit more than _FORBIDDEN_SIGNATURE_TRIP_PATHS distinct
#    paths it is a pattern, not scattered flakes, and fails fast from then on.
#    20: at the observed ~0.01% flake rate even a 100k-upload run expects ~10
#    scattered flakes, spread over different permissions and minutes, while a
#    real denial reaches 20 paths within seconds -- and capping its retried paths
#    at 20 bounds the waste at ~20 x 15 s of backoff instead of one per file.
_OVERWRITE_DENIED_MARKER = "storage.objects.delete"
_FORBIDDEN_PERMISSION_RE = re.compile(r"storage\.objects\.\w+")
_URL_RE = re.compile(r"(?:https?|gs)://\S+")
_FORBIDDEN_SIGNATURE_TRIP_PATHS = 20


def _forbidden_signature(exc: BaseException) -> str:
    """The denied permission named in a 403, else its URL-free message."""
    msg = str(exc)
    m = _FORBIDDEN_PERMISSION_RE.search(msg)
    if m:
        return m.group(0)
    return " ".join(_URL_RE.sub("", msg).split())[:200]


# Backoff schedule (seconds); len -> number of retries after the first try.
_RETRY_DELAYS: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)

_PROGRESS_EVERY_SECONDS = 60.0
_PROGRESS_EVERY_OPS = 200

# Permanent-failure cause logging: the first N failures are logged one by one
# (cause, object path, attempts); after that a summary with counts by
# exception type every SUMMARY_EVERY failures, so a run that fails thousands
# of uploads cannot flood the log. Error text is truncated to MAX_ERROR_CHARS.
_PERMANENT_FAILURE_LOG_FIRST_N = 10
_PERMANENT_FAILURE_SUMMARY_EVERY = 100
_PERMANENT_FAILURE_MAX_ERROR_CHARS = 500


@dataclasses.dataclass(frozen=True)
class DrainReport:
    """What a drain left behind. ``not_uploaded`` is every enqueued upload
    that did not land: failed (by category) + still queued at the deadline
    (``abandoned``) + in flight when the deadline cancelled it.

    Truthiness: ``drain()`` used to return an ``int`` — the number of uploads
    still queued at the deadline. A report is truthy exactly when that number
    (``abandoned``) is non-zero, so ``if await writer.drain():`` behaves as
    before. Failed and cancelled in-flight uploads do not make it truthy; check
    ``not_uploaded`` for the full count."""

    uploaded: int = 0
    failed_by_category: dict[str, int] = dataclasses.field(default_factory=dict)
    abandoned: int = 0
    cancelled_in_flight: int = 0
    timeout_s: float = 0.0
    concurrency: int = 0

    @property
    def failed(self) -> int:
        return sum(self.failed_by_category.values())

    @property
    def not_uploaded(self) -> int:
        return self.failed + self.abandoned + self.cancelled_in_flight

    def __bool__(self) -> bool:
        return self.abandoned > 0


def _current_task_cancelling() -> bool:
    """True when the running task has a pending cancellation (signal path)."""
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


def _parse_gs_url(gcs_prefix: str) -> tuple[str, str]:
    """Split a ``gs://bucket/optional/prefix`` URL into ``(bucket, prefix)``.

    ``prefix`` is returned without leading/trailing slashes ("" if absent).
    Raises ValueError for non-``gs://`` inputs.
    """
    if not gcs_prefix.startswith("gs://"):
        raise ValueError(f"Not a gs:// URL: {gcs_prefix!r}")
    rest = gcs_prefix[len("gs://") :]
    bucket, _, prefix = rest.partition("/")
    return bucket, prefix.strip("/")


def _is_atif_disabled(atif_output: str) -> bool:
    """True if ``atif_output`` means "no logging at all".

    Covers the empty string plus the explicit off/none/disable sentinels
    (case-insensitive, whitespace-stripped).
    """
    v = atif_output.strip().lower()
    return v == "" or v in _ATIF_OFF_SENTINELS


def parse_atif_output(atif_output: str) -> str | None:
    """Map a recipe's ``atif_output`` config value to a GCS prefix (or None).

    Contract:
      * ``""`` / ``"off"`` / ``"none"`` / ``"disable"`` -> disabled (None)
      * ``"local"``  -> local writes only (None: no GCS prefix)
      * ``"gs://..."`` -> that GCS prefix (local + background upload)

    Any other value raises ValueError so a typo'd config fails loudly at
    startup rather than silently disabling upload.
    """
    v = atif_output.strip()
    if _is_atif_disabled(v) or v.lower() == "local":
        return None
    if v.startswith("gs://"):
        return v
    raise ValueError(
        f"atif_output must be '', 'local', 'off'/'none', or a 'gs://...' "
        f"URL; got {atif_output!r}"
    )


def _default_upload_fn(gcs_prefix: str) -> Callable[[Path, str], None]:
    """Build the real GCS upload closure (lazy client construction).

    Returns ``upload_fn(local_path, blob_name)`` that uploads a single file
    via ``google-cloud-storage``. Client + bucket are constructed once and
    reused. Auth is ADC.
    """
    bucket_name, _ = _parse_gs_url(gcs_prefix)
    client_box: dict[str, Any] = {}

    def upload_fn(local_path: Path, blob_name: str) -> None:
        if "bucket" not in client_box:
            from google.cloud import storage  # lazy: heavy import

            client = storage.Client()
            client_box["bucket"] = client.bucket(bucket_name)
        blob = client_box["bucket"].blob(blob_name)
        blob.upload_from_filename(str(local_path))

    return upload_fn


class _GcsUploader:
    """Best-effort background uploader with retry, accounting, and a
    recovery ledger.

    A pool of background ``asyncio.Task``s drains a shared queue; each
    file is uploaded with exponential backoff on transient errors and
    fail-fast on permanent ones. Permanent failures append a JSONL line to
    ``<job_dir>/.upload_failures.jsonl`` for recovery.

    All external effects are injectable for testing: ``upload_fn`` (the GCS
    call), ``sleep_fn`` (retry backoff), ``time_fn`` (progress-log clock),
    and ``jitter_fn`` (backoff jitter).
    """

    def __init__(
        self,
        *,
        gcs_prefix: str,
        job_dir: Path,
        max_concurrent_uploads: int = _DEFAULT_MAX_CONCURRENT_UPLOADS,
        delete_after_upload: bool = True,
        trial_grace_count: int = _DEFAULT_TRIAL_GRACE_COUNT,
        upload_fn: Callable[[Path, str], None] | None = None,
        sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
        time_fn: Callable[[], float] = time.time,
        jitter_fn: Callable[[float], float] | None = None,
    ):
        self._bucket, self._prefix = _parse_gs_url(gcs_prefix)
        self._job_dir = job_dir
        # Blob names are relative to the parent of <recipe>/<job_id> so gs
        # paths mirror on-disk paths exactly.
        self._blob_root = job_dir.parent.parent
        self._upload_fn = upload_fn or _default_upload_fn(gcs_prefix)
        self._sleep_fn = sleep_fn
        self._time_fn = time_fn
        self._jitter_fn = jitter_fn or (lambda d: d * random.uniform(0.5, 1.5))
        self._ledger_path = job_dir / ".upload_failures.jsonl"

        # When True, local files whose upload is CONFIRMED successful are
        # deleted from disk after the run drains (GCS becomes the sole copy).
        # Deletion is deferred to drain() rather than done per-upload because
        # some paths (agent/trajectory.json) are re-written and re-uploaded
        # repeatedly during a run via partial snapshots -- deleting on the
        # first success would remove a still-live file. Files that were never
        # uploaded (e.g. harbor_run.log) or whose upload failed are never
        # touched, so nothing un-backed-up is lost.
        self._delete_after_upload = delete_after_upload
        self._uploaded_paths: set[Path] = set()
        # Enqueued-but-not-yet-attempted uploads, per path. A path can be
        # written and enqueued MORE THAN ONCE -- `agent/trajectory.json`
        # gets a partial snapshot during the rollout and the final
        # trajectory at the end, both at the same path -- so membership in
        # `_uploaded_paths` says "some version of this uploaded", not "the
        # version on disk right now is backed up". Mid-run reclaim needs the
        # stronger statement, and this counter supplies it: a path is safe
        # to delete only when it is confirmed AND has no enqueue in flight.
        #
        # Discarding from `_uploaded_paths` in `enqueue()` instead would not
        # work: an in-flight upload of the OLD content can land afterwards
        # and re-add the path, reopening the same window.
        self._pending_enqueues: dict[Path, int] = {}
        self._reclaimed_files = 0

        # Trials the journal has declared FINAL, newest last. Only these may
        # be reclaimed mid-run: the partial-snapshot hazard above is exactly
        # "the writer cannot tell a finished trajectory from one that is
        # about to be rewritten", and the journal can. `finalize_trial` is
        # the signal; this deque applies the grace window to it.
        self._finalized_trials: deque[Path] = deque()

        # How many finalized trials stay on disk. Disk still stays flat --
        # the window is constant, not proportional to the run -- but the most
        # recent trials are the ones you want when something just went wrong,
        # and having them locally beats a GCS round trip.
        self._trial_grace_count = max(0, int(trial_grace_count))

        # Clamp to >=1: 0/negative would spawn no workers and hang the queue.
        self._max_concurrent_uploads = max(1, int(max_concurrent_uploads))
        # Worker count during the final drain, when nothing else competes.
        self._drain_concurrency = (
            self._max_concurrent_uploads * _DRAIN_CONCURRENCY_FACTOR
        )
        # Uploads currently being attempted (for the drain's cancelled count).
        self._in_flight = 0

        self._queue: asyncio.Queue[Path] = asyncio.Queue()
        self._workers: list[asyncio.Task[None]] = []
        # Dedicated thread pool so blocking google-cloud-storage calls don't
        # contend for the default executor's small thread budget.
        self._executor: ThreadPoolExecutor | None = None

        # Counters are mutated only from the event loop, so no lock needed.
        self._success_count = 0
        self._transient_attempts = 0
        self._by_category: dict[str, int] = {}
        # Permanent failures keyed by exception type name (for log summaries).
        self._by_error_type: dict[str, int] = {}
        # Per-signature Forbidden breaker: distinct paths each signature hit
        # (stops growing once tripped) and the signatures that now fail fast.
        self._forbidden_paths: dict[str, set[Path]] = {}
        self._forbidden_tripped: set[str] = set()
        self._last_progress_log_ts = 0.0
        self._last_progress_log_count = 0
        # Timing / byte accounting for throughput and latency reporting.
        self._total_bytes_uploaded: int = 0
        self._upload_latencies_ms: list[float] = []
        self._start_ts: float = 0.0  # set in start()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Launch the background worker pool (idempotent)."""
        if self._workers:
            return
        self._start_ts = self._time_fn()
        self._last_progress_log_ts = self._time_fn()
        # Sized for the drain's larger worker count; threads are spawned on
        # demand, so only the run's worker count of them exist until then.
        self._executor = ThreadPoolExecutor(
            max_workers=self._drain_concurrency,
            thread_name_prefix="atif-gcs-upload",
        )
        self._workers = [
            asyncio.ensure_future(self._run())
            for _ in range(self._max_concurrent_uploads)
        ]

    def enqueue(self, path: Path) -> None:
        """Non-blocking enqueue of a local file for upload."""
        # Counted BEFORE the put so there is no window in which the queue
        # holds the path and the counter does not: `_is_fully_uploaded`
        # runs on the same event loop, but a caller can enqueue from a
        # thread and the ordering must not depend on that.
        self._pending_enqueues[path] = self._pending_enqueues.get(path, 0) + 1
        self._queue.put_nowait(path)

    def _enqueue_done(self, path: Path) -> None:
        """One enqueue of ``path`` has been attempted (succeeded or not)."""
        n = self._pending_enqueues.get(path, 0) - 1
        if n > 0:
            self._pending_enqueues[path] = n
        else:
            self._pending_enqueues.pop(path, None)

    def finalize_trial(self, trial_dir: Path) -> None:
        """The journal says this trial is COMPLETE; reclaim it when eligible.

        This is the signal the writer cannot derive for itself. A successful
        upload does not mean a file is finished -- partial snapshots rewrite
        and re-upload the same path as a rollout proceeds -- so until now the
        only safe moment to delete anything was after every worker stopped,
        which meant a 120-step run held every trial it had ever written and a
        killed run leaked all of them.

        With the journal declaring finality, a trial can be reclaimed as soon
        as it is done AND past the grace window, so local disk stays flat
        across a run of any length.
        """
        if not self._delete_after_upload:
            return
        self._finalized_trials.append(trial_dir)
        self._drain_finalized()

    def _drain_finalized(self) -> None:
        """Reclaim the oldest finalized trials that are fully uploaded.

        Finality is not the same as durability. ``finalize_trial`` runs
        immediately after the files are ENQUEUED, so their uploads are still
        in flight and a blind delete at that moment would destroy the only
        copy. So the oldest trial is reclaimed only once every file in it is
        confirmed, and otherwise left where it is and retried on the next
        finalize -- the queue drains itself as uploads land.

        The grace window sits on top of that: even fully-uploaded trials keep
        their local copies until ``_trial_grace_count`` newer ones exist.

        ``_MAX_PENDING_FINALIZED`` bounds the case where uploads are failing
        rather than merely slow. Past it the oldest is reclaimed anyway,
        which frees whatever DID upload and leaves the rest on disk -- the
        queue must not grow without limit just because GCS is unhappy.
        """
        while len(self._finalized_trials) > self._trial_grace_count:
            oldest = self._finalized_trials[0]
            forced = len(self._finalized_trials) > _MAX_PENDING_FINALIZED
            if not forced and not self._is_fully_uploaded(oldest):
                # Still in flight. Leave it; a later finalize will retry.
                return
            self._finalized_trials.popleft()
            self._reclaim_trial(oldest)

    def _is_fully_uploaded(self, trial_dir: Path) -> bool:
        """True when every file under ``trial_dir`` is confirmed AND settled.

        Confirmation alone is not enough, and getting that wrong loses data.
        ``agent/trajectory.json`` is written twice at the same path -- a
        partial snapshot mid-rollout, the final trajectory at the end -- so
        once the partial uploads, the path is in ``_uploaded_paths``
        forever. A check on membership alone would call the trial uploaded
        while the FINAL trajectory is still sitting in the queue, reclaim
        would unlink it, its upload would then fail ``local_gone``, and GCS
        would be left holding the ``_partial: True`` snapshot as the only
        surviving copy -- lost from disk and from the bucket.

        So a path counts as uploaded only when it is confirmed and has no
        enqueue in flight.
        """
        try:
            return all(
                p in self._uploaded_paths and p not in self._pending_enqueues
                for p in trial_dir.rglob("*")
                if p.is_file()
            )
        except OSError:
            return False

    def _reclaim_trial(self, trial_dir: Path) -> None:
        """Delete a finalized trial's CONFIRMED-uploaded files.

        Only paths in ``_uploaded_paths`` go; anything whose upload failed or
        never happened stays, so GCS never becomes the only copy of something
        that did not reach it. A trial left holding files is left holding its
        directory too -- an empty dir is cheap, a lost trajectory is not.
        """
        removed = 0
        kept = 0
        for path in sorted(trial_dir.rglob("*")):
            if not path.is_file():
                continue
            if path not in self._uploaded_paths:
                kept += 1
                continue
            try:
                path.unlink()
                self._uploaded_paths.discard(path)
                removed += 1
            except OSError as exc:
                kept += 1
                logger.debug("AtifDiskWriter: could not unlink %s: %s", path, exc)
        if kept == 0:
            self._prune_empty_dirs(trial_dir)
        self._reclaimed_files += removed
        if kept:
            logger.debug(
                "AtifDiskWriter: kept %d un-uploaded file(s) in %s", kept, trial_dir
            )

    async def upload_single(self, path: Path, blob_suffix: str) -> None:
        """Upload one file to an EXPLICIT blob name under the run's prefix.

        Every other upload path derives the blob name from the file's
        position under ``_blob_root``. The run log is not in that tree at all
        -- it lives in the sandbox root, written by the shell wrapper, not by
        us -- so it needs its destination stated rather than computed.
        """
        blob = f"{self._prefix}/{blob_suffix}" if self._prefix else blob_suffix
        await asyncio.get_running_loop().run_in_executor(
            self._executor, self._upload_fn, path, blob
        )

    def _prune_empty_dirs(self, root: Path) -> None:
        """Remove ``root`` and its now-empty subdirs, never above ``job_dir``."""
        try:
            for d in sorted(root.rglob("*"), reverse=True):
                if d.is_dir():
                    with contextlib.suppress(OSError):
                        d.rmdir()
            with contextlib.suppress(OSError):
                root.rmdir()
        except OSError:
            pass

    async def reclaim_tree(self, job_dir: Path) -> dict[str, int]:
        """Finish uploads for an ABANDONED tree, then delete what landed.

        Called by a LATER run against a dead run's directory (see
        ``utils.run_registry``), never against its own. Uploads every file
        found, then removes only the ones GCS confirmed -- upload-then-delete,
        so a crash during reclaim loses nothing.
        """
        stats = {"uploaded": 0, "deleted": 0, "failed": 0}
        files = [p for p in sorted(job_dir.rglob("*")) if p.is_file()]
        if not files:
            return stats
        logger.warning(
            "AtifDiskWriter: reclaiming %d file(s) from abandoned tree %s",
            len(files),
            job_dir,
        )
        for path in files:
            try:
                await self._upload_with_retry(path)
            except Exception as exc:  # noqa: BLE001 - counted, never fatal
                stats["failed"] += 1
                logger.debug("reclaim: upload failed for %s: %s", path, exc)
                continue
            if path not in self._uploaded_paths:
                # _upload_with_retry only records on confirmed success.
                stats["failed"] += 1
                continue
            stats["uploaded"] += 1
            try:
                path.unlink()
                stats["deleted"] += 1
            except OSError:
                pass
        if not stats["failed"]:
            self._prune_empty_dirs(job_dir)
        return stats

    async def _run(self) -> None:
        while True:
            path = await self._queue.get()
            self._in_flight += 1
            try:
                await self._upload_with_retry(path)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "AtifDiskWriter uploader worker swallowed error: %s",
                    exc,
                    exc_info=True,
                )
            finally:
                # Exactly once per enqueue, on every exit path. Must run
                # AFTER the upload attempt: while it is still counted the
                # path is ineligible for mid-run reclaim, which is what
                # keeps a re-written file from being deleted out from under
                # its own queued upload.
                self._in_flight -= 1
                self._enqueue_done(path)
                self._queue.task_done()
                self._maybe_log_progress()

    # -- upload with retry -------------------------------------------------

    def _blob_name(self, path: Path) -> str:
        rel = path.relative_to(self._blob_root).as_posix()
        return f"{self._prefix}/{rel}" if self._prefix else rel

    def _rel_to_job(self, path: Path) -> str:
        try:
            return path.relative_to(self._job_dir).as_posix()
        except ValueError:
            return path.name

    async def _upload_with_retry(self, path: Path) -> None:
        blob_name = self._blob_name(path)
        # The content about to be uploaded is UNPROVEN until this attempt
        # confirms it, so retract any earlier confirmation of this path
        # first. A path can be written and enqueued more than once --
        # `agent/trajectory.json` gets a partial snapshot mid-rollout and
        # the final trajectory at the end -- and without this retraction a
        # stale confirmation from the partial survives an attempt that
        # never lands.
        #
        # It matters specifically for CANCELLATION. A failed upload is
        # retracted by `_record_failure`, but a worker cancelled mid-upload
        # (drain timeout) raises `CancelledError`, which never reaches that
        # path; the confirmation would stand and the drain-time delete would
        # unlink a file whose current content is only in GCS as a
        # `_partial: True` snapshot. Retracting up front makes the invariant
        # hold on every exit: a path is in `_uploaded_paths` only if its
        # LATEST attempt succeeded.
        self._uploaded_paths.discard(path)
        last_exc: BaseException | None = None
        for attempt, delay in enumerate((0.0,) + _RETRY_DELAYS):
            if delay:
                await self._sleep_fn(self._jitter_fn(delay))
            try:
                t0 = self._time_fn()
                await self._do_upload(path, blob_name)
                elapsed_ms = (self._time_fn() - t0) * 1000.0
                self._upload_latencies_ms.append(elapsed_ms)
                try:
                    self._total_bytes_uploaded += path.stat().st_size
                except OSError:
                    pass  # file may have been cleaned up
                self._success_count += 1
                if self._delete_after_upload:
                    # Defer the actual unlink to _delete_uploaded_files() (run
                    # at drain time): a path uploaded now may be a partial
                    # snapshot re-written + re-uploaded before the run ends.
                    self._uploaded_paths.add(path)
                return
            except _RETRYABLE as exc:
                self._transient_attempts += 1
                last_exc = exc
                continue
            except tuple(_PERMANENT.keys()) as exc:
                # The serviceusage-quota 403 is an intermittent probe flake, not
                # a real ACL denial -> retry it with backoff instead of
                # fail-fast (it typically succeeds on a later attempt).
                if _is_transient_forbidden(exc):
                    self._transient_attempts += 1
                    last_exc = exc
                    continue
                # Circuit breaker for a bare/ambiguous Forbidden (e.g.
                # "storage.objects.create denied"): if ANY upload has already
                # succeeded, this identity provably has write access, so treat it
                # as a flake and retry; only fail fast when NOTHING has ever
                # uploaded (a genuine, pervasive ACL denial). This keeps the
                # scattered ~0.01% authz hiccups off the failure ledger without
                # burning the backoff schedule on every file of a truly
                # misconfigured run. A deterministic denial (an overwrite, or a
                # signature recurring across many paths) is never a flake.
                if (
                    _is_bare_forbidden(exc)
                    and self._success_count > 0
                    and not self._forbidden_is_deterministic(path, exc)
                ):
                    self._transient_attempts += 1
                    last_exc = exc
                    continue
                self._record_permanent(
                    path, _PERMANENT[type(exc)], exc, attempts=attempt + 1
                )
                return
            except Exception as exc:
                # Unknown: log once with traceback, then treat as permanent
                # so we don't burn retries on an unrecognized error.
                logger.warning(
                    "AtifDiskWriter: unexpected upload error for %s: %s",
                    path,
                    exc,
                    exc_info=True,
                )
                self._record_permanent(path, "unknown", exc, attempts=attempt + 1)
                return
        # Retries exhausted.
        self._record_permanent(
            path, "retries_exhausted", last_exc, attempts=len(_RETRY_DELAYS) + 1
        )

    async def _do_upload(self, path: Path, blob_name: str) -> None:
        """Run the sync upload in the dedicated thread pool."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._executor, self._upload_fn, path, blob_name)

    def _forbidden_is_deterministic(self, path: Path, exc: BaseException) -> bool:
        """True if this bare Forbidden must fail fast instead of being retried
        as a flake: an overwrite denial, or a signature whose breaker tripped.
        Logs the cause ONCE per signature, when it starts failing fast."""
        sig = _forbidden_signature(exc)
        if sig in self._forbidden_tripped:
            return True
        paths = self._forbidden_paths.setdefault(sig, set())
        paths.add(path)
        overwrite = _OVERWRITE_DENIED_MARKER in str(exc)
        if not overwrite and len(paths) <= _FORBIDDEN_SIGNATURE_TRIP_PATHS:
            return False
        self._forbidden_tripped.add(sig)
        self._forbidden_paths.pop(sig, None)
        hint = (
            "overwriting an existing object needs storage.objects.delete as well "
            "as storage.objects.create, so every re-upload of a path is denied; "
            "grant both (e.g. roles/storage.objectUser) on the bucket"
            if overwrite
            else f"the same denial hit more than {_FORBIDDEN_SIGNATURE_TRIP_PATHS} "
            "distinct files, so it is not a scattered flake"
        )
        logger.warning(
            "AtifDiskWriter: GCS denied %s for %d file(s) so far (%s): %s. "
            "Further uploads denied %s fail fast without retries.",
            sig,
            len(paths),
            hint,
            str(exc)[:_PERMANENT_FAILURE_MAX_ERROR_CHARS].rstrip("."),
            sig,
        )
        return True

    # -- accounting --------------------------------------------------------

    def _record_permanent(
        self,
        path: Path,
        category: str,
        exc: BaseException | None,
        *,
        attempts: int,
    ) -> None:
        # Un-confirm this path: its CURRENT on-disk content failed to upload, so
        # it is NOT backed up -- even if an EARLIER upload of the same path (e.g.
        # a partial agent/trajectory.json snapshot) succeeded and added it to
        # the confirmed set. Without this, _delete_uploaded_files would delete a
        # file whose latest content was never uploaded, losing it from both disk
        # and GCS (the OfficeQA/Claude trajectory-loss bug).
        self._uploaded_paths.discard(path)
        self._by_category[category] = self._by_category.get(category, 0) + 1
        entry = {
            "path": self._rel_to_job(path),
            "category": category,
            "error": repr(exc),
            "ts": self._time_fn(),
        }
        # Append-only; one well-formed JSONL line per failure.
        with self._ledger_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        self._log_permanent(path, category, exc, attempts)

    def _log_permanent(
        self, path: Path, category: str, exc: BaseException | None, attempts: int
    ) -> None:
        """WARNING-log the cause of a permanent failure, rate-limited: the
        first ``_PERMANENT_FAILURE_LOG_FIRST_N`` individually, then a summary
        with counts by exception type every
        ``_PERMANENT_FAILURE_SUMMARY_EVERY`` failures."""
        err_type = type(exc).__name__ if exc is not None else "None"
        self._by_error_type[err_type] = self._by_error_type.get(err_type, 0) + 1
        n_failed = sum(self._by_category.values())
        err_msg = str(exc)[:_PERMANENT_FAILURE_MAX_ERROR_CHARS]
        object_uri = f"gs://{self._bucket}/{self._blob_name(path)}"
        if n_failed <= _PERMANENT_FAILURE_LOG_FIRST_N:
            logger.warning(
                "AtifDiskWriter: upload of %s failed permanently after %d "
                "attempt(s) [%s]: %s: %s%s",
                object_uri,
                attempts,
                category,
                err_type,
                err_msg,
                (
                    f" (further failures are summarized every "
                    f"{_PERMANENT_FAILURE_SUMMARY_EVERY})"
                    if n_failed == _PERMANENT_FAILURE_LOG_FIRST_N
                    else ""
                ),
            )
        elif n_failed % _PERMANENT_FAILURE_SUMMARY_EVERY == 0:
            logger.warning(
                "AtifDiskWriter: %d uploads failed permanently so far "
                "(by error: %s); latest: %s after %d attempt(s) [%s]: %s: %s",
                n_failed,
                self._error_type_counts(),
                object_uri,
                attempts,
                category,
                err_type,
                err_msg,
            )

    def _error_type_counts(self) -> str:
        return (
            ",".join(f"{k}={v}" for k, v in sorted(self._by_error_type.items()))
            or "none"
        )

    def upload_status(self) -> dict[str, Any]:
        """Return a snapshot of upload counters for external consumers.

        Field names match the ``GcsIoHealth`` shape downstream consumers
        expect, so callers can forward the dict as-is.
        """
        failed = sum(self._by_category.values())
        elapsed = max(self._time_fn() - self._start_ts, 0.001)
        latencies = self._upload_latencies_ms
        avg_ms = sum(latencies) / len(latencies) if latencies else 0.0
        p95_ms = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0
        return {
            "uploaded": self._success_count,
            "failed": failed,
            "in_queue": self._queue.qsize(),
            "transient_retries": self._transient_attempts,
            "total_bytes_uploaded": self._total_bytes_uploaded,
            "writes_per_second": round(self._success_count / elapsed, 2),
            "bytes_per_second": round(self._total_bytes_uploaded / elapsed, 2),
            "avg_write_latency_ms": round(avg_ms, 1),
            "p95_write_latency_ms": round(p95_ms, 1),
        }

    def _maybe_log_progress(self) -> None:
        total_ops = self._success_count + sum(self._by_category.values())
        now = self._time_fn()
        due_by_time = (now - self._last_progress_log_ts) >= _PROGRESS_EVERY_SECONDS
        due_by_count = (
            total_ops - self._last_progress_log_count
        ) >= _PROGRESS_EVERY_OPS
        if not (due_by_time or due_by_count):
            return
        if total_ops == self._last_progress_log_count:
            return  # nothing happened since last emit
        failed = sum(self._by_category.values())
        cats = ",".join(f"{k}={v}" for k, v in sorted(self._by_category.items()) if v)
        logger.info(
            "AtifDiskWriter progress: uploaded=%d failed(%s) in_queue=%d",
            self._success_count,
            cats or "none",
            self._queue.qsize(),
        )
        self._last_progress_log_ts = now
        self._last_progress_log_count = total_ops

    # -- drain -------------------------------------------------------------

    def _adaptive_drain_timeout(self, pending: int, max_timeout_s: float) -> float:
        """Normal-exit drain bound sized to the backlog (see the constants)."""
        recent = self._upload_latencies_ms[-_DRAIN_RATE_WINDOW:]
        if not pending or not recent:
            return min(_DRAIN_MIN_TIMEOUT_S, max_timeout_s)
        mean_latency_s = max(sum(recent) / len(recent) / 1000.0, 1e-6)
        rate = self._max_concurrent_uploads / mean_latency_s
        estimate = pending / rate * _DRAIN_TIMEOUT_MARGIN
        return min(max(estimate, _DRAIN_MIN_TIMEOUT_S), max_timeout_s)

    async def drain(
        self,
        timeout_s: float | None = None,
        *,
        max_timeout_s: float = _DEFAULT_DRAIN_MAX_TIMEOUT_S,
    ) -> DrainReport:
        """Upload what is still queued, then stop the workers, log the
        end-of-job summary and return a :class:`DrainReport`.

        The wait is bounded by, in order of precedence: an explicit
        ``timeout_s``; ``_CANCELLED_DRAIN_TIMEOUT_S`` when the calling task is
        being cancelled (signal path -- the host is about to go away); else a
        timeout sized to the backlog at the observed upload rate, capped at
        ``max_timeout_s``. Started workers are first raised to the drain
        concurrency. Whatever is still queued (``abandoned``) or in flight
        (``cancelled_in_flight``) at the deadline is reported, not uploaded;
        local copies and the failure ledger remain for recovery. Deleting the
        confirmed-uploaded local copies runs after the bound, outside it."""
        pending = self._queue.qsize() + self._in_flight
        if timeout_s is not None:
            timeout, why = float(timeout_s), "explicit"
        elif _current_task_cancelling():
            timeout, why = _CANCELLED_DRAIN_TIMEOUT_S, "cancelled"
        else:
            timeout = self._adaptive_drain_timeout(pending, max_timeout_s)
            why = f"adaptive, cap {max_timeout_s:g}s"
        if self._workers and len(self._workers) < self._drain_concurrency:
            self._workers += [
                asyncio.ensure_future(self._run())
                for _ in range(self._drain_concurrency - len(self._workers))
            ]
        concurrency = len(self._workers)
        if pending:
            logger.info(
                "AtifDiskWriter: draining %d pending upload(s) with %d workers, "
                "timeout %.1fs (%s)",
                pending,
                concurrency,
                timeout,
                why,
            )
        t0 = time.monotonic()
        try:
            await asyncio.wait_for(self._queue.join(), timeout=timeout)
        except asyncio.TimeoutError:
            pass
        except asyncio.CancelledError:
            # Cancelled mid-drain (a signal during a long normal drain): stop
            # now, but still report what was left, then propagate.
            await self._finish_drain(timeout, concurrency, time.monotonic() - t0)
            raise
        return await self._finish_drain(timeout, concurrency, time.monotonic() - t0)

    async def _finish_drain(
        self, timeout: float, concurrency: int, elapsed: float
    ) -> DrainReport:
        """Stop the workers, log the summary, delete confirmed local copies."""
        abandoned = self._queue.qsize()
        cancelled_in_flight = self._in_flight
        for w in self._workers:
            w.cancel()
        for w in self._workers:
            try:
                await w
            except (asyncio.CancelledError, Exception):
                pass
        self._workers = []
        if self._executor is not None:
            # Don't block the loop on in-flight threads; they'll finish or
            # die with the process.
            self._executor.shutdown(wait=False)
            self._executor = None

        report = DrainReport(
            uploaded=self._success_count,
            failed_by_category={k: v for k, v in self._by_category.items() if v},
            abandoned=abandoned,
            cancelled_in_flight=cancelled_in_flight,
            timeout_s=timeout,
            concurrency=concurrency,
        )
        cats = ", ".join(
            f"{k}={v}" for k, v in sorted(report.failed_by_category.items())
        )
        logger.info(
            "AtifDiskWriter GCS upload summary: uploaded=%d failed=%d(%s) "
            "transient_retries=%d abandoned_on_timeout=%d cancelled_in_flight=%d "
            "not_uploaded=%d (drain: %.1fs of %.1fs timeout, %d workers)",
            report.uploaded,
            report.failed,
            cats or "none",
            self._transient_attempts,
            abandoned,
            cancelled_in_flight,
            report.not_uploaded,
            elapsed,
            timeout,
            concurrency,
        )
        if report.not_uploaded:
            logger.warning(
                "AtifDiskWriter: %d files not uploaded to GCS (failed by "
                "error: %s; abandoned_on_timeout=%d, cancelled_in_flight=%d). "
                "Local copies at %s. Recovery ledger: %s",
                report.not_uploaded,
                self._error_type_counts(),
                abandoned,
                cancelled_in_flight,
                self._job_dir,
                self._ledger_path,
            )

        # Workers are stopped, so _uploaded_paths is final: delete the local
        # copies of files whose upload was CONFIRMED successful. Files that
        # failed / were abandoned are not in the set and thus never deleted
        # (their local copies survive for recovery, per the warning above).
        if self._delete_after_upload and self._uploaded_paths:
            await asyncio.to_thread(self._delete_uploaded_files)

        return report

    def _delete_uploaded_files(self) -> None:
        """Unlink local files whose upload succeeded, then prune emptied dirs.

        Best-effort: an unlink error (already-gone, permissions) is logged and
        skipped -- never raised, so cleanup can't fail a completed job. Only
        files under this uploader's ``_job_dir`` are removed; empty parent
        directories up to (but excluding) ``_job_dir`` are pruned so no orphan
        dir tree is left. Non-uploaded files in the same tree (e.g.
        ``harbor_run.log``, the failure ledger) are left in place, so their
        dirs stay non-empty and are not pruned.
        """
        deleted = 0
        freed_bytes = 0
        dirs_to_check: set[Path] = set()
        for path in self._uploaded_paths:
            if path in self._pending_enqueues:
                # Re-written and re-enqueued, but that enqueue was never
                # attempted -- drain timed out with it still in the queue,
                # so the confirmation in `_uploaded_paths` belongs to the
                # PREVIOUS content. Deleting here would lose the current
                # version from disk while GCS holds only the older one.
                continue
            try:
                freed_bytes += path.stat().st_size
            except OSError:
                pass
            try:
                path.unlink()
                deleted += 1
                dirs_to_check.add(path.parent)
            except FileNotFoundError:
                pass  # already gone (e.g. duplicate path) -- fine
            except OSError as exc:
                logger.warning(
                    "AtifDiskWriter: could not delete uploaded local file %s: %s",
                    path,
                    exc,
                )
        # Prune now-empty dirs, deepest first, but never at/above _job_dir.
        for d in sorted(dirs_to_check, key=lambda p: len(p.parts), reverse=True):
            cur = d
            while cur != self._job_dir and cur.is_relative_to(self._job_dir):
                try:
                    cur.rmdir()  # only succeeds if empty
                except OSError:
                    break  # non-empty (has un-uploaded files) or gone
                cur = cur.parent
        logger.info(
            "AtifDiskWriter: deleted %d uploaded local files (~%.1f MB freed) "
            "under %s; GCS is now the sole copy of those files.",
            deleted,
            freed_bytes / 1e6,
            self._job_dir,
        )


class AtifDiskWriter:
    """``TrajectoryCallback`` that writes ATIF-conformant files to disk.

    Writes ``<root_dir>/<recipe>/<tuning_job_id>/...`` per the ATIF
    project-tree contract. Local writes are unconditional; if
    ``gcs_prefix`` is set, files are also uploaded in the background.
    """

    def __init__(
        self,
        *,
        root_dir: str,
        recipe: str,
        tuning_job_id: str,
        gcs_prefix: str | None = None,
        max_concurrent_uploads: int = _DEFAULT_MAX_CONCURRENT_UPLOADS,
        delete_after_upload: bool = True,
        uploader_factory: Callable[..., Any] | None = None,
    ):
        """Args:
        root_dir: Base directory for the on-disk tree (e.g. ``<log_dir>/atif``).
        recipe / tuning_job_id: Path segments; sanitized to
            ``[A-Za-z0-9._-]+``.
        gcs_prefix: ``gs://bucket[/prefix]`` for background upload, or
            ``None`` for local-only writes.
        max_concurrent_uploads: Background upload worker pool size (only
            used when ``gcs_prefix`` is set). Clamped to >=1.
        delete_after_upload: When True (default) and ``gcs_prefix`` is set,
            local files whose upload is CONFIRMED successful are deleted at
            drain time so GCS becomes their sole copy. Files never uploaded or
            whose upload failed are always kept. No effect for local-only
            runs (``gcs_prefix=None`` keeps everything).
        uploader_factory: Test seam; defaults to ``_GcsUploader``.
        """
        self._root = Path(root_dir).expanduser()
        self._recipe = _sanitize_path_segment(recipe)
        self._tuning_job_id = _sanitize_path_segment(tuning_job_id)
        self._job_dir = self._root / self._recipe / self._tuning_job_id
        self._job_dir.mkdir(parents=True, exist_ok=True)
        self._gcs_prefix = gcs_prefix
        self._task_id_extractor: Callable[[Any], str] = _default_task_id_extractor
        self._rollout_idx_extractor: Callable[[str | None], int | None] = (
            _extract_rollout_idx_from_session_id
        )
        # Per-session CaptureContext cache (populated lazily by
        # ``capture_ctx_for``). Keyed by session_id so repeated calls
        # within one rollout share the same Counter; distinct
        # session_ids get distinct Counters. The lock is eagerly
        # initialized: under Python 3.10+, ``asyncio.Lock()`` is
        # loop-agnostic (does not bind to a specific event loop at
        # construction time), so it is safe to build here even when
        # AtifDiskWriter is constructed before any loop is running.
        # Eager init avoids a check-then-set TOCTOU race between
        # ``capture_ctx_for`` and ``on_trajectory_produced`` (round-6
        # review fix): under the lazy pattern, concurrent first calls
        # could each observe ``None`` and install separate Lock
        # instances, defeating the mutual exclusion the lock enforces.
        self._capture_ctx_cache: dict[str, CaptureContext] = {}
        self._capture_ctx_lock: asyncio.Lock = asyncio.Lock()

        # Partial-snapshot throttle: maps trial_name → last-upload
        # monotonic timestamp. Used by on_trajectory_step_completed
        # to cap snapshots at 1 per _PARTIAL_MIN_INTERVAL_S per trial.
        self._partial_last_upload: dict[str, float] = {}

        # One lock per trial, guarding the writes that all target that
        # trial's agent/trajectory.json. They come from DIFFERENT tasks --
        # the last step's on_trajectory_step_completed overlaps
        # on_trajectory_produced, and concurrent steps of one agent overlap
        # each other -- so without this they interleave in their worker
        # threads. Created on demand, dropped when the trial finalizes.
        # asyncio.Lock is loop-agnostic under 3.10+, same as the capture-ctx
        # lock above.
        self._trial_write_locks: dict[str, asyncio.Lock] = {}
        # Trials whose FINAL agent/trajectory.json has landed. A snapshot
        # queued behind the final write must not overwrite the finished
        # artifact with a mid-rollout one (complete JSON, but missing the
        # last steps and flagged ``_partial``). Holds one short trial name
        # per completed rollout for the life of the writer: the marker
        # cannot be evicted while a queued snapshot could still fire.
        self._finalized: set[str] = set()

        self._gcs_uploader: Any | None = None
        if gcs_prefix:
            factory = uploader_factory or _GcsUploader
            self._gcs_uploader = factory(
                gcs_prefix=gcs_prefix,
                job_dir=self._job_dir,
                max_concurrent_uploads=max_concurrent_uploads,
                delete_after_upload=delete_after_upload,
            )
            self._gcs_uploader.start()

    @property
    def job_dir(self) -> Path:
        """The per-job directory: ``<root_dir>/<recipe>/<tuning_job_id>/``."""
        return self._job_dir

    def trial_dir_for(
        self,
        *,
        task_id: str,
        rollout_idx: int | None = None,
        session_id: str | None = None,
    ) -> Path:
        """The ``<job>/<task_id>-<rollout_idx>/`` dir for one rollout.

        Public because the trial dir is no longer this class's private
        business: the journal writes its crash WAL
        (``artifacts/journal.wal.jsonl``) and the env-state artifact
        under the SAME directory this writer later fills with
        ``agent/trajectory.json``. Exposing the rule beats copying it — a
        second copy of the naming convention is exactly how a WAL ends up
        orphaned from the trajectory it insures.

        ``rollout_idx=None`` falls back to parsing the session_id tail,
        the same way :meth:`on_trajectory_produced` does, so a caller
        holding only a session id still lands in the right place.
        """
        if not task_id:
            task_id = _FALLBACK_TASK_ID
        if rollout_idx is None:
            rollout_idx = self._rollout_idx_extractor(session_id)
        # DELEGATE to `_trial_name`, the same helper `on_trajectory_produced`
        # and the partial-snapshot path use. An earlier version of this
        # method rebuilt the name as f"{task_id}-{rollout_idx}" instead --
        # which is what `_trial_name` returns only in SOME cases, so the WAL
        # landed in `t-1-3/` while the trajectory landed in
        # `r-job-1-t-1-3-20260101T000000Z-1/`. Two directories, one rollout,
        # and a WAL insuring a trajectory that was never going to be its
        # neighbour. That is precisely the copied-convention failure this
        # method's docstring warns about, committed by the method itself.
        # `rollout_idx` is passed THROUGH, never defaulted. It is the one
        # signal `_trial_name` branches on: `None` means "identity
        # unparseable" and earns the collision-proof
        # `<task_id>-unknown-<uuid>`, while `0` is a real index and earns
        # `<task_id>-0`. `rollout_idx or 0` collapsed the first into the
        # second, so EVERY such concurrent rollout's WAL went to one
        # `<task_id>-0/artifacts/journal.wal.jsonl` -- one inode, N writers --
        # while their trajectories went to distinct `-unknown-<uuid>` dirs.
        # The same two-directories-one-rollout split the docstring above
        # warns about, one argument further along.
        return self._job_dir / _trial_name(
            session_id or "", _sanitize_path_segment(task_id), rollout_idx
        )

    def gcs_upload_status(self) -> dict[str, Any] | None:
        """Return GCS upload counters, or ``None`` if GCS is disabled."""
        if self._gcs_uploader is None:
            return None
        return self._gcs_uploader.upload_status()

    @property
    def gcs_uri(self) -> str | None:
        """Full GCS URI for the per-job directory, or ``None`` if local-only."""
        if not self._gcs_prefix:
            return None
        bucket, prefix = _parse_gs_url(self._gcs_prefix)
        job_rel = f"{self._recipe}/{self._tuning_job_id}"
        blob_path = f"{prefix}/{job_rel}" if prefix else job_rel
        return f"gs://{bucket}/{blob_path}/"

    # -- job lifecycle -----------------------------------------------------

    async def on_job_started(self, config: Any) -> None:
        """Write (and enqueue for upload) the per-job ``config.json``."""
        config_dict = _dump_config(config)
        _add_viewer_facets(config_dict, config)
        path = self._job_dir / "config.json"
        payload = json.dumps(config_dict, indent=2, cls=_PermissiveJSONEncoder)
        await asyncio.to_thread(path.write_text, payload)
        if self._gcs_uploader is not None:
            self._gcs_uploader.enqueue(path)

    async def on_job_completed(self, aggregate_stats: dict[str, Any]) -> None:
        """Write the job-level ``result.json`` and drain the uploader."""
        path = self._job_dir / "result.json"
        payload = json.dumps(
            _nan_to_none(aggregate_stats),
            indent=2,
            allow_nan=False,
            cls=_PermissiveJSONEncoder,
        )
        await asyncio.to_thread(path.write_text, payload)
        if self._gcs_uploader is not None:
            self._gcs_uploader.enqueue(path)
        await self.drain()
        # Drained cleanly: there is nothing for a future run to reclaim, so
        # retire the registration. Left in place on any other exit, which is
        # precisely when someone should come back for this tree.
        self.deregister_from_reclaim()

    # -- crash recovery ----------------------------------------------------

    def register_for_reclaim(self, registry: Any = None) -> None:
        """Record this run so a later one can finish its uploads if we die.

        Best-effort and never raises: a run that cannot register is a run
        that will leak on a crash, which is worse than nothing but far better
        than failing to start.
        """
        if self._gcs_uploader is None:
            # Nothing to upload anywhere, so nothing to reclaim.
            return
        try:
            from utils.run_registry import RunRegistry

            self._registry = registry or RunRegistry()
            self._registry.register(
                run_id=self._job_dir.name,
                job_dir=self._job_dir,
                gcs_prefix=self.gcs_uri or "",
            )
        except Exception as exc:  # noqa: BLE001 - never block a run starting
            logger.warning("AtifDiskWriter: could not register for reclaim: %s", exc)

    def deregister_from_reclaim(self) -> None:
        """Retire this run's registration after a clean drain."""
        registry = getattr(self, "_registry", None)
        if registry is None:
            return
        try:
            registry.deregister(self._job_dir.name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("AtifDiskWriter: could not deregister: %s", exc)

    # NOTE: there is deliberately no `reclaim_abandoned_runs` here. A run
    # records itself (above) and cleans up after ITSELF; it never acts on
    # another run's registration. Doing so from the entry points made run B
    # responsible for run A's data -- B's startup latency, credentials and
    # failures entangled with data B has nothing to do with. Reclaim lives in
    # `python -m utils.run_registry --reclaim`, invoked by a human or a cron
    # line, so the operator chooses when another run's tree is touched.

    def enqueue_for_upload(self, path: Path) -> None:
        """Enqueue an arbitrary local file for background GCS upload.

        No-op when GCS upload is disabled. The file must be under the
        writer's ``_blob_root`` (= ``root_dir``) so ``_blob_name`` can
        compute a relative GCS path. Files outside that tree raise
        ``ValueError`` inside the uploader worker (logged, not raised
        to the caller).
        """
        if self._gcs_uploader is not None:
            self._gcs_uploader.enqueue(path)

    async def drain(self, timeout_s: float | None = None) -> DrainReport:
        """Wait for background uploads to finish; see ``_GcsUploader.drain``
        for how the wait is bounded when ``timeout_s`` is None. No-op (an
        empty report) when GCS upload is disabled."""
        if self._gcs_uploader is None:
            return DrainReport()
        return await self._gcs_uploader.drain(timeout_s=timeout_s)

    # -- per-rollout lifecycle (progress hooks) ----------------------------

    def _trial_write_lock(self, trial_name: str) -> asyncio.Lock:
        """The lock guarding one trial's ``agent/trajectory.json``.

        Created on first use. Safe without a lock of its own: the callbacks
        that reach here all run on one event loop and this method does not
        await, so the get-then-set cannot be interleaved.
        """
        lock = self._trial_write_locks.get(trial_name)
        if lock is None:
            lock = self._trial_write_locks[trial_name] = asyncio.Lock()
        return lock

    async def on_trajectory_started(
        self,
        task_data: Any,
        traj: Trajectory | None = None,
    ) -> None:
        """No-op — disk writer only acts on completed trajectories."""

    async def on_trajectory_step_completed(
        self,
        task_data: Any,
        traj: Trajectory | None = None,
    ) -> None:
        """Write a partial trajectory snapshot every few agent steps.

        Conditions (both must hold):
        1. The current agent-step count (``len(traj.steps) - 1``, i.e.
           minus the leading user step) is a multiple of
           ``_PARTIAL_STEP_INTERVAL`` (every 2nd step by default;
           override with $ATIF_PARTIAL_STEP_INTERVAL).
        2. At least ``_PARTIAL_MIN_INTERVAL_S`` (5 min) have elapsed
           since the last snapshot for this trial (rate-limit to avoid
           GCS spam).

        The snapshot is written to the same ``agent/trajectory.json``
        path that ``on_trajectory_produced`` will eventually overwrite
        with the final trajectory, so the viewer can display
        in-progress rollouts.

        We ``traj.model_copy(deep=True)`` for the snapshot rather than
        reading ``traj.steps`` directly — the agent loop keeps
        mutating ``traj.steps.append(...)`` while this async write is
        in flight, and a shallow read of the shared list would race.

        ``traj=None`` is a no-op (used by tests that don't exercise
        the snapshot path); a real agent always supplies one.
        """
        if traj is None:
            return
        e = traj.extra or {}
        rollout_idx = e.get("rollout_idx", 0)
        live_step = max(0, len(traj.steps) - 1)
        if live_step % _PARTIAL_STEP_INTERVAL != 0:
            return

        task_id = _sanitize_path_segment(self._task_id_extractor(task_data))
        trial_name = _trial_name(traj.session_id, task_id, rollout_idx)

        # The rollout is over and its final artifact is on disk; a snapshot
        # now would replace a finished trajectory with a partial one.
        if trial_name in self._finalized:
            return

        # Throttle: at most 1 upload per _PARTIAL_MIN_INTERVAL_S per trial.
        now = time.monotonic()
        last = self._partial_last_upload.get(trial_name, 0.0)
        if now - last < _PARTIAL_MIN_INTERVAL_S:
            return
        self._partial_last_upload[trial_name] = now

        trial_dir = self._job_dir / trial_name
        agent_dir = trial_dir / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Deep-copy the live traj so subsequent appends don't mutate
            # the object we're about to serialize (this runs in a
            # thread; the loop keeps growing the real traj concurrently).
            partial_traj = traj.model_copy(deep=True)
            # Tag as partial for the viewer to distinguish mid-rollout
            # snapshots from the final artifact.
            if partial_traj.extra is None:
                partial_traj.extra = {}
            partial_traj.extra["_partial"] = True
            num_steps_snap = len(partial_traj.steps)
            body = partial_traj.model_dump(exclude_none=True)
            # Strip _raw from step extras to keep snapshot lean.
            for s in body.get("steps", []):
                if s.get("extra") is not None:
                    s["extra"].pop("_raw", None)
            agent_json = json.dumps(body)
        except Exception:
            logger.warning(
                "AtifDiskWriter: failed to serialize partial snapshot at "
                "step %d for %s (skipped)",
                live_step,
                trial_name,
                exc_info=True,
            )
            return

        agent_path = agent_dir / "trajectory.json"
        async with self._trial_write_lock(trial_name):
            # Re-checked under the lock: the final write may have landed
            # while this snapshot was queued behind it.
            if trial_name in self._finalized:
                return
            await asyncio.to_thread(_atomic_write_text, agent_path, agent_json)

        if self._gcs_uploader is not None:
            self._gcs_uploader.enqueue(agent_path)

        logger.debug(
            "AtifDiskWriter: wrote partial snapshot at step %d for %s "
            "(%d steps, %d bytes)",
            live_step,
            trial_name,
            num_steps_snap,
            len(agent_json),
        )

    # -- per-rollout (completed) -------------------------------------------

    async def on_trajectory_produced(
        self,
        scored: Any,
        task_data: Any,
        traj: Trajectory | None = None,
    ) -> None:
        """Write the 3-file trial dir for one scored trajectory.

        The trial dir IS ``scored.trajectory.session_id`` (see
        :func:`_trial_name`). ``traj`` is accepted for protocol
        conformance; post-mutable-Trajectory refactor it is the same
        object as ``scored.trajectory``.

        ``rollout_idx`` is still parsed out of the session_id, but only
        to record it in ``result.json`` -- it no longer routes anything.
        """
        del traj  # protocol-conformance; scored.trajectory is the same ref
        task_id = _sanitize_path_segment(self._task_id_extractor(task_data))
        session_id = scored.trajectory.session_id
        rollout_idx = self._rollout_idx_extractor(session_id)
        if rollout_idx is None and not session_id:
            logger.warning(
                "AtifDiskWriter: no session_id and no parseable "
                "rollout_idx; writing to a unique fallback dir "
                "(<task_id>-unknown-<uuid>) to avoid collisions"
            )
        trial_name = _trial_name(session_id, task_id, rollout_idx)
        trial_dir = self._job_dir / trial_name
        agent_dir = trial_dir / "agent"
        verifier_dir = trial_dir / "verifier"
        agent_dir.mkdir(parents=True, exist_ok=True)

        # Popped BEFORE to_disk() so the breakdown is not also inlined
        # into agent/trajectory.json; the trial-tree contract keeps it in
        # its own verifier/reward-details.json file.
        reward_details = pop_reward_details(scored.trajectory)
        # ``scored.reward is None`` means NOT MEASURED -- a distinct signal
        # from a NaN reward, which means "measured, and the measurement
        # broke". A curator emits it for a candidate whose arm never
        # validated anything: writing ``verifier/reward.json`` there would
        # put a number under the name of a measurement that never happened,
        # and every consumer that walks trial dirs would read the arm as
        # having scored something. So the file (and the ``verifier/`` dir
        # with it) is OMITTED, and ``result.json`` carries a null reward.
        #
        # ``to_disk`` needs a ``Reward`` to serialize either half, so the
        # unmeasured path hands it a throwaway NaN one and keeps only the
        # agent JSON that comes back.
        measured = getattr(scored, "reward", None) is not None
        agent_json, reward_json = to_disk(
            scored
            if measured
            else ScoredTrajectory(
                trajectory=scored.trajectory, reward=Reward(reward=float("nan"))
            )
        )
        agent_path = agent_dir / "trajectory.json"
        reward_path = verifier_dir / "reward.json"

        result = {
            "task_id": task_id,
            "rollout_idx": rollout_idx,
            "reward": scored.reward.reward if measured else None,
            "status": get_rollout_status(scored.trajectory).value,
            "session_id": session_id,
            # Which pipeline phase produced this rollout. The dir name
            # carries the eval step but not the phase, so record it here.
            "phase": get_phase(scored.trajectory),
        }
        result_path = trial_dir / "result.json"
        result_json = json.dumps(_nan_to_none(result), indent=2, allow_nan=False)

        writes = [
            asyncio.to_thread(_atomic_write_text, agent_path, agent_json),
            asyncio.to_thread(_atomic_write_text, result_path, result_json),
        ]
        paths = [agent_path, result_path]
        if measured:
            verifier_dir.mkdir(parents=True, exist_ok=True)
            writes.append(
                asyncio.to_thread(_atomic_write_text, reward_path, reward_json)
            )
            paths.append(reward_path)
        if measured and reward_details is not None:
            details_path = verifier_dir / "reward-details.json"
            writes.append(
                asyncio.to_thread(
                    _atomic_write_text,
                    details_path,
                    json.dumps(_nan_to_none(reward_details), indent=2, allow_nan=False),
                )
            )
            paths.append(details_path)

        # Push the blocking writes to threads concurrently so the event
        # loop isn't stalled for the IO window. Under the trial's write
        # lock, and with the trial marked finalized first, so no snapshot
        # can interleave with these bytes or land on top of them.
        async with self._trial_write_lock(trial_name):
            self._finalized.add(trial_name)
            await asyncio.gather(*writes)

        if self._gcs_uploader is not None:
            for p in paths:
                self._gcs_uploader.enqueue(p)
            # THIS call is the trial's finality signal, and the only place in
            # the system that has one: the journal fires
            # on_trajectory_produced from `finish()`, so by here the
            # trajectory will not be written again and its partial snapshots
            # are superseded. Handing that fact to the uploader is what lets
            # local copies be reclaimed DURING the run instead of piling up
            # until drain() -- which a killed run never reaches.
            self._gcs_uploader.finalize_trial(trial_dir)

        # Per-rollout CaptureContext is only needed while sampler calls
        # for this rollout are in flight. Once on_trajectory_produced
        # fires, the rollout has produced its trajectory and no further
        # sampler calls will need this ctx. Evict to bound memory in
        # long-running training with RAW_CAPTURE=1 (the cache is keyed
        # by session_id, so without eviction it grows monotonically at
        # ~200 bytes per rollout across the job's lifetime). Guarded by
        # the same lock capture_ctx_for uses so a concurrent
        # capture_ctx_for call for a different session_id can't race
        # with this pop. .pop(..., None) is defensive against sessions
        # that never registered a ctx (e.g., RAW_CAPTURE off).
        async with self._capture_ctx_lock:
            self._capture_ctx_cache.pop(scored.trajectory.session_id, None)

        # Evict the partial-snapshot throttle entry and the write lock for
        # this trial. The rollout is done — no more snapshots will fire for
        # it, and one that is already queued on the old lock object still
        # holds its own reference and will see ``_finalized`` when it wakes.
        self._partial_last_upload.pop(trial_name, None)
        self._trial_write_locks.pop(trial_name, None)

    # -- raw-LLM capture routing ------------------------------------------

    async def capture_ctx_for(
        self,
        *,
        task_id: str,
        rollout_idx: int,
        session_id: str,
    ) -> CaptureContext | None:
        """Return a per-rollout CaptureContext for the given rollout.

        Constructs the same trial_dir that
        :meth:`on_trajectory_produced` uses for this rollout -- i.e.
        ``session_id`` (see :func:`_trial_name`) -- so the raw-LLM ledger
        (``<trial_dir>/artifacts/raw_llm_calls/call_NNNNNN.json``) lands
        alongside ``agent/trajectory.json`` and ``verifier/reward.json``
        for the same rollout.

        Caches per session_id so repeated calls within one rollout return
        the SAME CaptureContext (and thus SAME Counter) -- call_NNNNNN.json
        indices monotonically increase across turns of a multi-turn
        rollout. Distinct session_ids get distinct CaptureContexts with
        independent Counters.

        Args:
            task_id: The task's opaque ID (from ``task_data`` after
                ``ensure_task_id``). Sanitized to the contract-safe
                charset when composing the trial_dir name.
            rollout_idx: The within-group rollout position
                (``0..num_samples-1``).
            session_id: The rollout's identity AND its trial-dir name;
                also stamped verbatim onto each ledger blob's
                ``trajectory_id`` field. Used as-is, never parsed.

        Returns:
            A ``CaptureContext`` with ``trial_dir = self._job_dir /
            session_id`` and a per-rollout ``Counter`` (cached by
            ``session_id`` -- repeat calls within one rollout return the
            SAME Counter object).

        Design note: the trial dir is the session_id *verbatim*, never
        recomposed from parts. An earlier version recomposed it from
        ``task_id`` + ``rollout_idx`` precisely because parsing the
        session_id was unsafe (the recipe the Agent embedded can differ
        from this writer's own recipe param -- see fbab9086). Using the
        whole string as an opaque name sidesteps both hazards: nothing
        is parsed, and nothing is rebuilt. ``task_id``/``rollout_idx``
        remain only for the no-session_id fallback.

        Thread-safety: called from Agent.rollout coroutines on the
        asyncio event loop; the internal cache is guarded by an
        ``asyncio.Lock`` (eagerly initialized in ``__init__``) because
        sibling rollouts may call concurrently for distinct session_ids.
        """
        async with self._capture_ctx_lock:
            cached = self._capture_ctx_cache.get(session_id)
            if cached is not None:
                return cached
            # Match _default_task_id_extractor's fallback so the ledger dir
            # aligns with on_trajectory_produced's trial dir when
            # task_data.task_id is missing (defensive path -- ensure_task_id
            # should prevent this from firing, but a misbehaving adapter
            # might not). Without this branch, on_trajectory_produced would
            # route to "unknown-task-<idx>/" (from _default_task_id_extractor)
            # while capture_ctx_for would route to "unknown-<idx>/" (from
            # _sanitize_path_segment("") -> "unknown"), orphaning the
            # ledger from its trajectory.
            if not task_id:
                task_id = _FALLBACK_TASK_ID
            safe_task_id = _sanitize_path_segment(task_id)
            trial_dir = self._job_dir / _trial_name(
                session_id, safe_task_id, rollout_idx
            )
            # Bind the ledger-blob -> GCS upload callback when an
            # uploader is configured. Without this wire-through, blobs
            # written by dump_raw would land on the training VM's local
            # disk only; the trajectory JSONs uploaded to GCS would
            # point at paths that don't exist for GCS consumers.
            on_written = (
                self._gcs_uploader.enqueue if self._gcs_uploader is not None else None
            )
            # Build the URI prefix so dump_raw returns a directly-
            # resolvable URI on step.extra.llm_call_uri. We derive the
            # GCS prefix by delegating to the uploader's own blob-name
            # rule (_blob_name(trial_dir)) so the URI matches EXACTLY
            # where the file lands in GCS -- if the naming rule ever
            # changes, both sides stay in sync automatically. Without
            # this coupling, consumers with just the URI could not
            # resolve the blob without additional trial-dir context.
            blob_uri_prefix: str | None
            if self._gcs_uploader is not None:
                trial_blob_path = self._gcs_uploader._blob_name(trial_dir)
                blob_uri_prefix = f"gs://{self._gcs_uploader._bucket}/{trial_blob_path}"
            else:
                blob_uri_prefix = str(trial_dir.absolute())
            ctx = CaptureContext(
                trial_dir=trial_dir,
                trajectory_id=session_id,
                counter=Counter(),
                on_written=on_written,
                blob_uri_prefix=blob_uri_prefix,
            )
            self._capture_ctx_cache[session_id] = ctx
            return ctx
