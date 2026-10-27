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

"""Session-id builder for ATIF trajectories.

session_id must be unique per rollout dispatch.  Formed from recipe-owned
task identity + framework-owned rollout index + a process-startup
timestamp + a monotonic dispatch counter (ensures uniqueness when the
same task_id + rollout_idx is dispatched again in a later epoch).

The session_id is ALSO the on-disk artifact directory name: the
``AtifDiskWriter`` uses it verbatim as the trial dir (see
``utils.atif_disk_writer._trial_name``). Two consequences the builder
owns:

* the id is sanitized to the ATIF path-segment charset here, at the
  single mint point, so every consumer can use it as a path without
  re-sanitizing (re-derivation at N call sites is how the trial dir and
  the raw-capture ledger drifted apart before);
* eval dispatches carry an ``s{step:04d}`` segment so the directory
  says which sweep produced it. Training passes no step -- the async
  producer runs ahead of the trainer, so no exact train step exists at
  dispatch time; uniqueness there comes from the dispatch counter.
"""

from __future__ import annotations

import itertools
import re
from datetime import datetime, timezone

from core.task_data import TaskData


_RUN_START_ISO: str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
_DISPATCH_SEQ: itertools.count[int] = itertools.count()

# Path segments must match ``^[A-Za-z0-9._\-]+$`` per the ATIF contract;
# everything else is replaced with ``_``.
_UNSAFE_SEGMENT_CHARS = re.compile(r"[^A-Za-z0-9._\-]")


def sanitize_path_segment(segment: str) -> str:
    """Coerce ``segment`` to the ATIF-safe path charset.

    Empty input maps to ``"unknown"`` so a missing component can never
    produce an empty path segment.
    """
    cleaned = _UNSAFE_SEGMENT_CHARS.sub("_", segment or "")
    return cleaned or "unknown"


def eval_only_tuning_job_id() -> str:
    """Fallback ``tuning_job_id`` for eval-only entry points with no
    real Vertex tuning job. Format ``eval-<_RUN_START_ISO>`` (the
    ``eval-`` prefix distinguishes from real tuning-job ids, which are
    positive-integer strings)."""
    return f"eval-{_RUN_START_ISO}"


def _extract_task_id(task_data: TaskData) -> str | None:
    """Return task_data.task_id as a non-empty string, or None.

    The ``str(...)`` coercion is defense-in-depth for callers that
    bypass typing (e.g. tests passing a non-string via ``type: ignore``).
    Empty string and ``None`` both map to ``None``.
    """
    tid = task_data.task_id
    if tid is None:
        return None
    s = str(tid)
    return s if s else None


def build_session_id(
    *,
    recipe: str,
    tuning_job_id: str,
    task_data: TaskData,
    rollout_idx: int,
    step: int | None = None,
) -> str:
    """Format: ``{recipe}-{tuning_job_id}-[s{step:04d}-]{task_id}-{rollout_idx}-{ISO}-{seq}``.

    ``{seq}`` is a process-wide monotonic counter that increments on
    every call, ensuring that re-dispatching the same ``task_id`` +
    ``rollout_idx`` in a later epoch produces a distinct session_id.

    ``step`` (eval sweeps only) inserts an ``s0003``-style segment so the
    artifact directory named after this id says which sweep produced it.
    It sits BEFORE ``task_id`` so the trailing
    ``-{rollout_idx}-{ISO}-{seq}`` shape -- which
    ``utils.atif_disk_writer`` parses ``rollout_idx`` out of -- is
    unchanged.

    Every component is sanitized to the ATIF path-segment charset: this
    id is used verbatim as a directory name.
    """
    task_id = _extract_task_id(task_data)
    if not task_id:
        raise ValueError(
            f"task_data.task_id is required for session id build; got "
            f"type={type(task_data).__name__}, task_id={task_data.task_id!r}. "
            f"Recipes must call ensure_task_id inside parse_*_item at "
            f"dataset-loading time."
        )
    seq = next(_DISPATCH_SEQ)
    step_seg = f"s{step:04d}-" if step is not None else ""
    return (
        f"{sanitize_path_segment(recipe)}"
        f"-{sanitize_path_segment(tuning_job_id)}"
        f"-{step_seg}{sanitize_path_segment(task_id)}"
        f"-{rollout_idx}-{_RUN_START_ISO}-{seq}"
    )
