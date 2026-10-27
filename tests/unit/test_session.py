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

"""Unit tests for core/session.py."""

import dataclasses
import re

import pytest

from core.session import build_session_id, _extract_task_id
from core.task_data import TaskData

# Matches the new format: ...-{rollout_idx}-{ISO}-{seq}
_SESSION_ID_RE = re.compile(r"^.+-\d+-\d{8}T\d{6}Z-\d+$")


@dataclasses.dataclass
class _TD(TaskData):
    # Test-only subclass to avoid coupling this file to any recipe's
    # concrete TaskData subclass.
    pass


def test_extract_task_id_returns_string():
    assert _extract_task_id(_TD(task_id="puzzle-xyz")) == "puzzle-xyz"


def test_extract_task_id_returns_none_when_empty_string():
    assert _extract_task_id(_TD(task_id="")) is None


def test_extract_task_id_stringifies_non_string_values():
    # Defensive: if a recipe accidentally sets task_id to a non-string
    # (e.g. int 0), we stringify rather than crash. This matches the
    # pre-migration behavior at src/core/session.py:67.
    td_int = _TD(task_id=0)  # type: ignore[arg-type]  # deliberate: testing coercion
    assert _extract_task_id(td_int) == "0"


def test_build_session_id_composes_all_parts():
    session = build_session_id(
        recipe="word_puzzle",
        tuning_job_id="tj-1",
        task_data=_TD(task_id="hash-abc"),
        rollout_idx=3,
    )
    # Format: f"{recipe}-{tj}-{task_id}-{rollout}-{iso}-{seq}"
    assert session.startswith("word_puzzle-tj-1-hash-abc-3-")
    assert _SESSION_ID_RE.match(session), f"unexpected format: {session!r}"


def test_build_session_id_epoch_unique():
    """Two calls with identical kwargs MUST produce different session_ids.

    The monotonic dispatch counter ensures that re-dispatching the same
    task_id + rollout_idx in a later epoch produces a distinct ID,
    preventing disk-path and Firestore-doc collisions.
    """
    kw = dict(
        recipe="r",
        tuning_job_id="tj",
        task_data=_TD(task_id="t"),
        rollout_idx=0,
    )
    id_a = build_session_id(**kw)
    id_b = build_session_id(**kw)
    assert id_a != id_b, "session_ids must differ across dispatches"
    # Both should share the same prefix (everything before the seq counter).
    assert id_a.rsplit("-", 1)[0] == id_b.rsplit("-", 1)[0]


def test_build_session_id_raises_when_task_id_missing():
    with pytest.raises(ValueError, match="task_id"):
        build_session_id(
            recipe="r",
            tuning_job_id="tj",
            task_data=_TD(task_id=""),
            rollout_idx=1,
        )


# --------------------------------------------------------------------------
# session_id doubles as the on-disk trial directory name
# (see ``utils.atif_disk_writer._trial_name``)
# --------------------------------------------------------------------------


def test_build_session_id_step_segment_for_eval():
    """Eval sweeps pass ``step`` so the artifact dir named after this id says
    which sweep produced it. The segment sits BEFORE task_id, leaving the
    ``-{rollout_idx}-{ISO}-{seq}`` tail (which the disk writer parses
    rollout_idx out of) intact."""
    session = build_session_id(
        recipe="word_puzzle",
        tuning_job_id="tj-1",
        task_data=_TD(task_id="hash-abc"),
        rollout_idx=3,
        step=7,
    )
    assert session.startswith("word_puzzle-tj-1-s0007-hash-abc-3-")
    assert _SESSION_ID_RE.match(session), f"unexpected format: {session!r}"


def test_build_session_id_no_step_segment_for_training():
    """Training passes no step (no exact train step exists at dispatch time
    in the async pipeline), so the format is unchanged."""
    session = build_session_id(
        recipe="r", tuning_job_id="tj", task_data=_TD(task_id="t"), rollout_idx=0
    )
    assert "-s0000-" not in session
    assert session.startswith("r-tj-t-0-")


def test_build_session_id_step_zero_is_labelled():
    """step=0 is a real eval step (the base-model sweep), not 'absent'."""
    session = build_session_id(
        recipe="r", tuning_job_id="tj", task_data=_TD(task_id="t"), rollout_idx=0, step=0
    )
    assert "-s0000-" in session


def test_build_session_id_is_path_safe():
    """The id is used verbatim as a directory name, so every component is
    sanitized at the mint point -- no consumer should have to re-sanitize."""
    session = build_session_id(
        recipe="we/ird recipe",
        tuning_job_id="tj:1",
        task_data=_TD(task_id="a/b c:d"),
        rollout_idx=0,
    )
    assert re.fullmatch(r"[A-Za-z0-9._\-]+", session), session


def test_step_sorts_lexically():
    """Zero-padding means lexical order == step order when listing a job dir."""
    ids = [
        build_session_id(
            recipe="r",
            tuning_job_id="tj",
            task_data=_TD(task_id="t"),
            rollout_idx=0,
            step=s,
        )
        for s in (2, 10, 100)
    ]
    assert sorted(ids) == ids
