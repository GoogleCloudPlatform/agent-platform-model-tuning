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

"""Discovery and loading of trials from a published trajectory tree.

A *job root* is the directory holding one run's trials (the level with
``config.json``); each child directory containing ``agent/trajectory.json``
is a trial.

Identity (task_id, rollout_idx) is a two-stage affair, because reading it
is expensive and listing it is not. ``discover_trials`` GUESSES it from
the directory name -- one listing for the whole tree -- and
``load_trial`` REPLACES the guess with the authoritative value from the
trajectory's ``extra`` once the trajectory has been parsed anyway. The
name is a convention, the extra is the contract; the guess exists only so
that discovery over a tree with thousands of trials costs one RPC rather
than one RPC per trial (that shape took 9+ minutes on a 3,901-trial run
before the caller had even applied ``--limit``).
"""

from __future__ import annotations

import dataclasses
import logging
import math
from typing import Any

from core.atif_codec import from_disk
from core.atif_extras import get_rollout_idx, get_task_id
from core.atif_trajectory import ScoredTrajectory, Trajectory

from .tree_reader import TreeReader, TreeReadError, make_reader

logger = logging.getLogger(__name__)

_TRAJECTORY_REL = "agent/trajectory.json"
_REWARD_REL = "verifier/reward.json"
_RESULT_REL = "result.json"


class TrialLoadError(Exception):
    """A trial could not be loaded (missing, malformed, or invalid ATIF)."""


@dataclasses.dataclass(frozen=True)
class TrialRef:
    """Enough to locate and pair a trial without loading it."""

    root: str
    trial: str
    task_id: str
    fanout_idx: int

    @property
    def uri(self) -> str:
        return f"{self.root.rstrip('/')}/{self.trial}"


@dataclasses.dataclass
class TrialView:
    """One loaded trial: the scored trajectory plus its sibling files."""

    ref: TrialRef
    scored: ScoredTrajectory
    result: dict[str, Any]
    reader: TreeReader

    @property
    def trajectory(self) -> Trajectory:  # convenience for metric authors
        return self.scored.trajectory

    @property
    def reward(self) -> float:
        return self.scored.reward.reward

    @property
    def is_scored(self) -> bool:
        """False when the verifier produced no reward for this trial.

        A failed rollout carries ``reward=NaN`` (written to disk as
        ``null``, restored by ``atif_codec.from_disk``). NaN compares
        False against everything, so a pairwise metric that does not
        check this silently scores such a trial as a TIE -- absence of
        signal read as agreement.
        """
        return not math.isnan(self.reward)


def parse_trial_name(name: str) -> tuple[str, int]:
    """Guess ``(task_id, rollout_idx)`` from a trial directory name.

    The name is minted by ``utils.atif_disk_writer._trial_name``, which
    spells the conventional case ``<task_id>-<rollout_idx>``. A single
    ``rsplit("-", 1)`` inverts that: an all-digit tail is the fanout
    index, anything else means the whole name is the task id and the
    index is 0.

    The inversion is ambiguous and deliberately not clever about it: a
    task id that itself ends in ``-<digits>`` cannot be told apart from
    task-plus-rollout, so ``"my-task-3"`` parses as ``("my-task", 3)``
    even if the real task id was ``"my-task-3"``. Names carrying a
    session id, or the ``<task>-unknown-<hex>`` fallback, parse as
    ``(name, 0)``. All of these are only GUESSES -- ``load_trial``
    overwrites them with the trajectory's own ``extra`` -- so the cost of
    a wrong guess is bounded to code paths that never load the trial.

    Never raises: an unconventional name yields ``(name, 0)``.
    """
    head, sep, tail = name.rpartition("-")
    if sep and head and tail.isdigit():
        return head, int(tail)
    return name, 0


def discover_trials(root: str, *, reader: TreeReader | None = None) -> list[TrialRef]:
    """List every candidate trial under a job root, in one listing.

    Performs NO per-trial reads: identity comes from ``parse_trial_name``
    and is therefore PROVISIONAL. ``load_trial`` replaces it with the
    authoritative ``(task_id, rollout_idx)`` from ``trajectory.extra``,
    so callers that pair or group trials must do so on LOADED views, not
    on these refs.

    Every child directory is a candidate; nothing here verifies that it
    holds a trajectory. A directory that turns out not to be a trial
    fails at ``load_trial`` and is reported as a skip. Plain files at
    this level (``config.json`` and friends) are excluded by the listing.
    """
    reader = reader or make_reader(root)
    list_subdirs = getattr(reader, "list_subdirs", None)
    names = list_subdirs("") if callable(list_subdirs) else reader.list_dir("")
    return [
        TrialRef(root=root, trial=name, task_id=task_id, fanout_idx=fanout_idx)
        for name, (task_id, fanout_idx) in (
            (n, parse_trial_name(n)) for n in sorted(names)
        )
    ]


def _authoritative_ref(ref: TrialRef, trajectory: Trajectory) -> TrialRef:
    """``ref`` with the name-derived identity replaced by the real one.

    ``ref.trial`` is left alone: it is the path, not the identity. A
    trajectory that declares no ``task_id`` (or no ``rollout_idx``) keeps
    the corresponding guess -- a missing field is no evidence against it.
    """
    task_id = get_task_id(trajectory)
    rollout_idx = get_rollout_idx(trajectory)
    updates: dict[str, Any] = {}
    if task_id is not None and str(task_id) != ref.task_id:
        updates["task_id"] = str(task_id)
    if rollout_idx is not None and int(rollout_idx) != ref.fanout_idx:
        updates["fanout_idx"] = int(rollout_idx)
    return dataclasses.replace(ref, **updates) if updates else ref


def _rehydrate_recipe_step_stats(trajectory: Trajectory) -> None:
    """Restore the in-memory spelling of the per-step recipe stats.

    The codec renames ``_task_step_stats`` (the in-memory key, holding live
    dataclasses) to ``task_step_stats`` on the way to disk. Every recipe
    aggregator and ``atif_extras.get_step_task_stats`` reads the UNDERSCORE
    key, so a trajectory loaded back from disk looks to them like a rollout
    that produced no stats at all: they return ``{}`` and their metrics
    silently vanish from the report -- no error, no warning, just an absent
    number that a reader would take for "the recipe didn't emit this".

    Aliasing the key back here is what makes an offline aggregator agree
    with the same aggregator run inside the pipeline. Mutates in place; the
    on-disk spelling is left alone so a re-serialization round-trips.
    """
    extra = trajectory.extra
    if not extra:
        return
    if "_task_step_stats" not in extra and "task_step_stats" in extra:
        extra["_task_step_stats"] = extra["task_step_stats"]


def load_trial(ref: TrialRef, *, reader: TreeReader | None = None) -> TrialView:
    """Load one trial into a typed ``TrialView``.

    The returned view's ``ref`` carries the AUTHORITATIVE identity: the
    trajectory has to be parsed here anyway, so any disagreement with the
    name-derived guess from ``discover_trials`` is corrected on the way
    out (see ``_authoritative_ref``).

    Raises ``TrialLoadError`` when a required file is missing, a read
    failed, or the ATIF payload fails validation -- callers decide whether
    to skip or abort. The three are spelled differently in the message on
    purpose: "missing" is a malformed tree the operator should look at,
    "read failed" is usually transient and worth re-running.
    """
    reader = reader or make_reader(ref.root)
    try:
        traj_raw = reader.read_text(f"{ref.trial}/{_TRAJECTORY_REL}")
        reward_raw = reader.read_text(f"{ref.trial}/{_REWARD_REL}")
        result = reader.read_json(f"{ref.trial}/{_RESULT_REL}") or {}
    except TreeReadError as exc:
        raise TrialLoadError(f"{ref.uri}: read failed: {exc}") from exc
    if traj_raw is None:
        raise TrialLoadError(f"{ref.uri}: missing {_TRAJECTORY_REL}")
    if reward_raw is None:
        raise TrialLoadError(f"{ref.uri}: missing {_REWARD_REL}")
    try:
        scored = from_disk(traj_raw, reward_raw)
    except Exception as exc:  # pydantic ValidationError, KeyError, ...
        raise TrialLoadError(f"{ref.uri}: invalid ATIF payload: {exc}") from exc
    _rehydrate_recipe_step_stats(scored.trajectory)
    return TrialView(
        ref=_authoritative_ref(ref, scored.trajectory),
        scored=scored,
        result=result,
        reader=reader,
    )
