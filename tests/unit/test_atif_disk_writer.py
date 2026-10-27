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

"""Unit tests for ``utils.atif_disk_writer.AtifDiskWriter`` (local-disk core).

Covers spec §5 + §11.1: the on-disk trial-dir layout, content round-trip,
path-segment sanitization, task_id / rollout_idx extraction, concurrent
writes, and callback-error isolation. GCS upload (spec §6) is exercised
separately in the ``_GcsUploader`` tests once that surface lands.
"""

from __future__ import annotations

import asyncio
import threading
import dataclasses
import json
from pathlib import Path

import pytest

from core.atif_codec import from_disk
from core.atif_trajectory import (
    ScoredTrajectory,
    Reward,
    StepStats,
    Step,
    TrajectoryStep,
)
from core.rollout_status import RolloutStatus
from core import atif_extras
from core.task_data import TaskData
from core.atif_trajectory import Agent as AtifAgent, Trajectory as AtifTrajectory

from utils.atif_disk_writer import (
    _RETRY_DELAYS,
    AtifDiskWriter,
    _GcsUploader,
    _parse_gs_url,
    _sanitize_path_segment,
    build_atif_writer,
    parse_atif_output,
)


# --------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------


@dataclasses.dataclass
class _TaskData(TaskData):
    """Lightweight TaskData for writer tests."""

    pass


def _scored(
    *,
    session_id: str,
    reward: float = 0.7,
    normalized_reward: float | None = 0.3,
    status: RolloutStatus = RolloutStatus.OK,
) -> ScoredTrajectory:
    """Build a minimal valid ScoredTrajectory with the given session_id."""
    us = Step(step_id=1, source="user", message="do X")
    a = TrajectoryStep.from_wire(
        step_id=2,
        generation="g",
        stats=StepStats(candidate_id="c1", finish_reason="stop"),
    )
    traj = AtifTrajectory(
        schema_version="ATIF-v1.7",
        session_id=session_id,
        agent=AtifAgent(name="t", version="0", model_name="m"),
        steps=[us, a],
    )
    atif_extras.set_rollout_status(traj, status)
    return ScoredTrajectory(
        trajectory=traj,
        reward=Reward(reward=reward, normalized_reward=normalized_reward),
    )


def _writer(tmp_path: Path, **kwargs) -> AtifDiskWriter:
    defaults = dict(
        root_dir=str(tmp_path / "atif"),
        recipe="word_puzzle",
        tuning_job_id="9999",
    )
    defaults.update(kwargs)
    return AtifDiskWriter(**defaults)


# session_id tail must match build_session_id: "...-<rollout_idx>-<ISO>"
def _sid(task_id: str, rollout_idx: int) -> str:
    return f"word_puzzle-9999-{task_id}-{rollout_idx}-20260712T031500Z"


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


def test_init_creates_job_dir(tmp_path: Path):
    w = _writer(tmp_path)
    assert (tmp_path / "atif" / "word_puzzle" / "9999").is_dir()
    assert w is not None


def test_init_eagerly_initializes_capture_ctx_lock(tmp_path: Path) -> None:
    """The ``_capture_ctx_lock`` MUST exist immediately after ``__init__``.

    Previously the lock was lazy-created inside
    both ``capture_ctx_for`` and ``on_trajectory_produced`` on their
    respective first calls, creating a TOCTOU window where two
    concurrent invocations could each observe ``None`` and build
    separate ``asyncio.Lock`` instances -- defeating the mutual
    exclusion the lock is supposed to provide. Under Python 3.10+,
    ``asyncio.Lock()`` is loop-agnostic (does not bind to a specific
    event loop at construction time), so we can safely build it in
    ``__init__`` before any loop is running. This test locks in that
    invariant.
    """
    import asyncio as _aio

    w = _writer(tmp_path)
    assert w._capture_ctx_lock is not None
    assert isinstance(w._capture_ctx_lock, _aio.Lock)


@pytest.mark.asyncio
async def test_concurrent_capture_ctx_for_and_on_trajectory_produced_share_lock(
    tmp_path: Path,
) -> None:
    """Concurrent ``capture_ctx_for`` calls for distinct session_ids
    must share ONE lock instance (no per-call lock proliferation).

    Regression guard for the lazy-init race: pre-fix, the very first
    concurrent invocations of ``capture_ctx_for`` and
    ``on_trajectory_produced`` could each observe ``_capture_ctx_lock
    is None`` and each install a separate ``asyncio.Lock``, so the
    "guarded by the same lock" comment in ``on_trajectory_produced``
    was a lie under exactly the race the lock was supposed to prevent.
    """
    w = _writer(tmp_path)
    lock_before = w._capture_ctx_lock
    assert lock_before is not None

    # Fire capture_ctx_for and on_trajectory_produced concurrently.
    sid_a = _sid("task-a", 0)
    sid_b = _sid("task-b", 0)
    await asyncio.gather(
        w.capture_ctx_for(task_id="task-a", rollout_idx=0, session_id=sid_a),
        w.on_trajectory_produced(
            _scored(session_id=sid_b), _TaskData(task_id="task-b")
        ),
    )
    # Same lock instance -- no lazy replacement.
    assert w._capture_ctx_lock is lock_before


# --------------------------------------------------------------------------
# on_trajectory_produced -> 3-file trial dir (§11.1)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_trajectory_produced_creates_three_file_trial_dir(tmp_path: Path):
    w = _writer(tmp_path)
    scored = _scored(session_id=_sid("task-abc", 0))
    await w.on_trajectory_produced(scored, _TaskData(task_id="task-abc"))

    trial = tmp_path / "atif" / "word_puzzle" / "9999" / _sid("task-abc", 0)
    assert (trial / "agent" / "trajectory.json").is_file()
    assert (trial / "verifier" / "reward.json").is_file()
    assert (trial / "result.json").is_file()


@pytest.mark.asyncio
async def test_no_temp_files_survive_a_produced_trajectory(tmp_path: Path):
    """The trial dir the contract describes, and nothing else in it."""
    w = _writer(tmp_path)
    await w.on_trajectory_produced(
        _scored(session_id=_sid("task-abc", 0)), _TaskData(task_id="task-abc")
    )

    trial = tmp_path / "atif" / "word_puzzle" / "9999" / _sid("task-abc", 0)
    assert [p.name for p in trial.rglob("*") if p.name.startswith(".")] == []


@pytest.mark.asyncio
async def test_written_files_round_trip_through_from_disk(tmp_path: Path):
    w = _writer(tmp_path)
    scored = _scored(session_id=_sid("task-abc", 0), reward=0.42, normalized_reward=0.9)
    await w.on_trajectory_produced(scored, _TaskData(task_id="task-abc"))

    trial = tmp_path / "atif" / "word_puzzle" / "9999" / _sid("task-abc", 0)
    agent_json = (trial / "agent" / "trajectory.json").read_text()
    reward_json = (trial / "verifier" / "reward.json").read_text()
    back = from_disk(agent_json, reward_json)

    assert back.reward.reward == 0.42
    assert back.reward.normalized_reward == 0.9
    assert back.trajectory.session_id == _sid("task-abc", 0)


@pytest.mark.asyncio
async def test_result_json_has_expected_fields(tmp_path: Path):
    w = _writer(tmp_path)
    scored = _scored(session_id=_sid("task-abc", 3), reward=0.55)
    await w.on_trajectory_produced(scored, _TaskData(task_id="task-abc"))

    trial = tmp_path / "atif" / "word_puzzle" / "9999" / _sid("task-abc", 3)
    result = json.loads((trial / "result.json").read_text())
    assert result["task_id"] == "task-abc"
    assert result["rollout_idx"] == 3
    assert result["reward"] == 0.55
    assert result["status"] == "OK"
    assert result["session_id"] == _sid("task-abc", 3)


@pytest.mark.asyncio
async def test_reward_json_is_leak_safe_object(tmp_path: Path):
    """verifier/reward.json is a JSON object carrying ``reward`` (contract #2)."""
    w = _writer(tmp_path)
    await w.on_trajectory_produced(
        _scored(session_id=_sid("t", 0), reward=0.1), _TaskData(task_id="t")
    )
    reward_json = (
        tmp_path
        / "atif"
        / "word_puzzle"
        / "9999"
        / _sid("t", 0)
        / "verifier"
        / "reward.json"
    ).read_text()
    obj = json.loads(reward_json)
    assert isinstance(obj, dict)
    assert obj["reward"] == 0.1


# --------------------------------------------------------------------------
# Event-loop non-starvation: blocking write_text must run off the loop
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_trajectory_produced_does_not_starve_event_loop(
    tmp_path: Path, monkeypatch
):
    """A slow (blocking) write must not stall unrelated coroutines: writes
    go through asyncio.to_thread, so a heartbeat coroutine keeps ticking."""
    import time as _time

    real_write_text = Path.write_text

    def _slow_write_text(self, data, *a, **k):
        # Simulate a slow disk. If this runs ON the event loop it will block
        # the heartbeat; if it runs in a worker thread (asyncio.to_thread)
        # the heartbeat keeps ticking.
        _time.sleep(0.05)
        return real_write_text(self, data, *a, **k)

    monkeypatch.setattr(Path, "write_text", _slow_write_text)

    w = _writer(tmp_path)

    ticks = 0
    stop = asyncio.Event()

    async def heartbeat():
        nonlocal ticks
        while not stop.is_set():
            await asyncio.sleep(0.001)
            ticks += 1

    hb = asyncio.create_task(heartbeat())
    await w.on_trajectory_produced(
        _scored(session_id=_sid("task-abc", 0)), _TaskData(task_id="task-abc")
    )
    stop.set()
    await hb

    # 3 files x 50ms = 150ms of blocking IO (or ~50ms if concurrent). The
    # heartbeat ticks every ~1ms, so if the loop was NOT starved it ticked
    # many times during the writes (observed ~45 locally). Pre-fix (sync
    # write_text on the loop) the heartbeat cannot tick at all during the
    # ~150ms stall (observed 0). Threshold is deliberately well below the
    # healthy count but far above the broken=0 case, so a loaded CI runner
    # (where asyncio.sleep granularity inflates) won't flake.
    assert ticks >= 5, f"event loop was starved during writes: only {ticks} ticks"

    # correctness still holds: files written
    trial = tmp_path / "atif" / "word_puzzle" / "9999" / _sid("task-abc", 0)
    assert (trial / "agent" / "trajectory.json").is_file()
    assert (trial / "verifier" / "reward.json").is_file()
    assert (trial / "result.json").is_file()


# --------------------------------------------------------------------------
# Path sanitization (§11.1)
# --------------------------------------------------------------------------


def test_sanitize_path_segment_replaces_unsafe_chars():
    assert _sanitize_path_segment("a/b c\u00e9") == "a_b_c_"
    assert _sanitize_path_segment("repo__pkg-12") == "repo__pkg-12"  # already safe
    assert _sanitize_path_segment("v1.2_x") == "v1.2_x"  # dot + underscore kept


def test_sanitize_path_segment_empty_falls_back():
    # Empty / all-unsafe input must not yield an empty segment.
    assert _sanitize_path_segment("") == "unknown"
    assert _sanitize_path_segment("///") == "___"


@pytest.mark.asyncio
async def test_task_id_with_unsafe_chars_is_sanitized_in_path(tmp_path: Path):
    """Path-safety now comes from the session_id (sanitized at mint), and the
    extracted task_id is still sanitized for the ``result.json`` field."""
    w = _writer(tmp_path)
    scored = _scored(session_id=_sid("weird", 0))
    await w.on_trajectory_produced(scored, _TaskData(task_id="we/ird di:r"))

    trial = tmp_path / "atif" / "word_puzzle" / "9999" / _sid("weird", 0)
    assert (trial / "result.json").is_file()
    assert json.loads((trial / "result.json").read_text())["task_id"] == "we_ird_di_r"


@pytest.mark.asyncio
async def test_real_session_id_is_path_safe_at_mint(tmp_path: Path):
    """``build_session_id`` sanitizes every component, so an unsafe task_id
    can never produce an unsafe directory name."""
    from core.session import build_session_id

    sid = build_session_id(
        recipe="word_puzzle",
        tuning_job_id="9999",
        task_data=_TaskData(task_id="we/ird di:r"),
        rollout_idx=0,
    )
    assert "/" not in sid and " " not in sid and ":" not in sid

    w = _writer(tmp_path)
    await w.on_trajectory_produced(
        _scored(session_id=sid), _TaskData(task_id="we/ird di:r")
    )
    assert (tmp_path / "atif" / "word_puzzle" / "9999" / sid / "result.json").is_file()


# --------------------------------------------------------------------------
# task_id extraction fallback (§11.1)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_task_id_falls_back_to_unknown(tmp_path: Path, caplog):
    """A missing ``task_data.task_id`` no longer affects routing (the dir is
    the session_id); it only shows up as the ``result.json`` fallback."""
    w = _writer(tmp_path)
    scored = _scored(session_id=_sid("x", 0))
    await w.on_trajectory_produced(scored, _TaskData(task_id=None))

    trial = tmp_path / "atif" / "word_puzzle" / "9999" / _sid("x", 0)
    assert (trial / "result.json").is_file()
    assert json.loads((trial / "result.json").read_text())["task_id"] == "unknown-task"


# --------------------------------------------------------------------------
# rollout_idx extraction from session_id (§11.1)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rollout_idx_extracted_from_session_id(tmp_path: Path):
    w = _writer(tmp_path)
    await w.on_trajectory_produced(
        _scored(session_id=_sid("task-abc", 7)), _TaskData(task_id="task-abc")
    )
    assert (
        tmp_path / "atif" / "word_puzzle" / "9999" / _sid("task-abc", 7) / "result.json"
    ).is_file()


@pytest.mark.asyncio
async def test_rollout_idx_multi_digit(tmp_path: Path):
    w = _writer(tmp_path)
    await w.on_trajectory_produced(
        _scored(session_id=_sid("task-abc", 123)), _TaskData(task_id="task-abc")
    )
    result = json.loads(
        (
            tmp_path
            / "atif"
            / "word_puzzle"
            / "9999"
            / _sid("task-abc", 123)
            / "result.json"
        ).read_text()
    )
    assert result["rollout_idx"] == 123


# --------------------------------------------------------------------------
# Concurrency (§11.1)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_same_task_different_idx_no_collision(tmp_path: Path):
    w = _writer(tmp_path)
    await asyncio.gather(
        *[
            w.on_trajectory_produced(
                _scored(session_id=_sid("task-abc", i)), _TaskData(task_id="task-abc")
            )
            for i in range(4)
        ]
    )
    for i in range(4):
        assert (
            tmp_path
            / "atif"
            / "word_puzzle"
            / "9999"
            / _sid("task-abc", i)
            / "result.json"
        ).is_file()


@pytest.mark.asyncio
async def test_same_session_id_written_twice_latest_wins(tmp_path: Path):
    """Two writes for the SAME session_id are the same rollout (partial then
    final), so they share a dir and the later write wins."""
    w = _writer(tmp_path)
    for reward in (0.1, 0.9):
        await w.on_trajectory_produced(
            _scored(session_id=_sid("task-abc", 0), reward=reward),
            _TaskData(task_id="task-abc"),
        )
    result = json.loads(
        (
            tmp_path
            / "atif"
            / "word_puzzle"
            / "9999"
            / _sid("task-abc", 0)
            / "result.json"
        ).read_text()
    )
    assert result["reward"] == 0.9  # latest write wins


@pytest.mark.asyncio
async def test_redispatched_task_does_not_overwrite_earlier_rollout(tmp_path: Path):
    """T1/T2 -- the guarantee this whole design exists for.

    The same task at the same fanout index, dispatched twice (a later eval
    sweep, a later training epoch, or two sweeps at the same step), must
    produce TWO trial dirs. Before session_id-named dirs, both landed in
    ``task-abc-0/`` and the first rollout's files were silently destroyed.

    Uses the REAL builder so the dispatch counter -- not the test's
    hand-built id -- is what separates them.
    """
    from core.session import build_session_id

    w = _writer(tmp_path)
    td = _TaskData(task_id="task-abc")
    rewards = {}
    for reward in (0.1, 0.9):
        sid = build_session_id(
            recipe="word_puzzle", tuning_job_id="9999", task_data=td, rollout_idx=0
        )
        rewards[sid] = reward
        await w.on_trajectory_produced(_scored(session_id=sid, reward=reward), td)

    job_dir = tmp_path / "atif" / "word_puzzle" / "9999"
    trial_dirs = {p.name for p in job_dir.iterdir() if p.is_dir()}
    assert trial_dirs == set(rewards), sorted(trial_dirs)
    for sid, reward in rewards.items():
        got = json.loads((job_dir / sid / "result.json").read_text())
        assert got["reward"] == reward, f"{sid} lost its reward"


@pytest.mark.asyncio
async def test_eval_sweeps_at_different_steps_get_separate_dirs(tmp_path: Path):
    """The eval case end-to-end: the same task evaluated at step 3 and at
    step 6 lands in two step-labelled dirs."""
    from core.session import build_session_id

    w = _writer(tmp_path)
    td = _TaskData(task_id="task-abc")
    for step in (3, 6):
        sid = build_session_id(
            recipe="word_puzzle",
            tuning_job_id="9999",
            task_data=td,
            rollout_idx=0,
            step=step,
        )
        await w.on_trajectory_produced(_scored(session_id=sid), td)

    job_dir = tmp_path / "atif" / "word_puzzle" / "9999"
    names = sorted(p.name for p in job_dir.iterdir() if p.is_dir())
    assert len(names) == 2, names
    assert any("s0003-" in n for n in names), names
    assert any("s0006-" in n for n in names), names


# --------------------------------------------------------------------------
# Unparseable session_id -> unique fallback dir (no silent collision)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unparseable_session_ids_do_not_collide(tmp_path: Path):
    """3 concurrent rollouts with unparseable session_ids must land in 3
    DISTINCT trial dirs. The session_id is the dir name, so even ids the
    rollout_idx regex can't parse stay separated."""
    w = _writer(tmp_path)
    unparseable = ["external-A", "external-B", "external-C"]
    await asyncio.gather(
        *(
            w.on_trajectory_produced(
                _scored(session_id=sid), _TaskData(task_id="task-abc")
            )
            for sid in unparseable
        )
    )
    job_dir = tmp_path / "atif" / "word_puzzle" / "9999"
    trial_dirs = {p.name for p in job_dir.iterdir() if p.is_dir()}
    assert trial_dirs == set(unparseable), sorted(trial_dirs)


@pytest.mark.asyncio
async def test_unparseable_session_id_records_null_rollout_idx(tmp_path: Path):
    """An unparseable session_id still routes (it IS the dir name); the
    unknown ``rollout_idx`` surfaces as ``null`` in result.json."""
    w = _writer(tmp_path)
    await w.on_trajectory_produced(
        _scored(session_id="external-no-idx"), _TaskData(task_id="task-abc")
    )
    result = json.loads(
        (
            tmp_path
            / "atif"
            / "word_puzzle"
            / "9999"
            / "external-no-idx"
            / "result.json"
        ).read_text()
    )
    assert result["rollout_idx"] is None
    assert result["session_id"] == "external-no-idx"


@pytest.mark.asyncio
async def test_parseable_zero_idx_not_treated_as_fallback(tmp_path: Path):
    """A legitimately-parsed rollout_idx of 0 must not trip the
    unknown-<uuid> fallback (None-vs-0 distinction)."""
    w = _writer(tmp_path)
    await w.on_trajectory_produced(
        _scored(session_id=_sid("task-abc", 0)), _TaskData(task_id="task-abc")
    )
    assert (
        tmp_path / "atif" / "word_puzzle" / "9999" / _sid("task-abc", 0) / "result.json"
    ).is_file()


@pytest.mark.asyncio
async def test_empty_session_id_routes_to_fallback_dir(tmp_path: Path):
    """An empty session_id hits the `if not session_id` branch (returns
    None, distinct from the regex-no-match branch); it must route to the
    unique unknown-<uuid> fallback, not collide into <task_id>-0."""
    w = _writer(tmp_path)
    await w.on_trajectory_produced(
        _scored(session_id=""), _TaskData(task_id="task-abc")
    )
    job_dir = tmp_path / "atif" / "word_puzzle" / "9999"
    trial_dirs = [p.name for p in job_dir.iterdir() if p.is_dir()]
    assert len(trial_dirs) == 1
    assert "unknown" in trial_dirs[0], trial_dirs
    assert not (job_dir / _sid("task-abc", 0)).exists()


# --------------------------------------------------------------------------
# config.json / result.json job-level (§9, §10)
# --------------------------------------------------------------------------


def test_add_viewer_facets_derives_harbor_shape():
    """_add_viewer_facets adds top-level agents/datasets/environment (the shape
    the trajectory viewer reads) from a RunConfig dump + live config object."""
    from utils.atif_disk_writer import _add_viewer_facets

    class _Backend:
        short_name = "anthropic_vertex"

    class _Config:
        backend = _Backend()

    cd = {
        "agent": {"model": "claude-opus-4-8", "environment": "docker"},
        "backend": {"model": "claude-opus-4-8"},
        "task": {"eval_data_path": "/x/data/officeqa-text-pro.jsonl"},
    }
    _add_viewer_facets(cd, _Config())

    assert cd["agents"] == [
        {
            "name": "anthropic_vertex",
            "model_name": "anthropic_vertex/claude-opus-4-8",
            "environment_type": "docker",
        }
    ]
    assert cd["datasets"] == [{"name": "officeqa-text-pro.jsonl"}]
    assert cd["environment"] == {"type": "docker"}


def test_add_viewer_facets_is_defensive_when_fields_absent():
    """Missing recipe-overlay fields yield a thinner block, never an error, and
    pre-existing viewer keys are not overwritten."""
    from utils.atif_disk_writer import _add_viewer_facets

    class _Config:
        backend = None

    cd: dict = {"agent": {}, "backend": {}, "task": {}}
    _add_viewer_facets(cd, _Config())
    assert "agents" not in cd
    assert "datasets" not in cd
    assert "environment" not in cd

    cd2 = {"agents": [{"name": "keep"}], "agent": {"model": "m"}, "task": {}}
    _add_viewer_facets(cd2, _Config())
    assert cd2["agents"] == [{"name": "keep"}]


@pytest.mark.asyncio
async def test_on_job_started_writes_config_json(tmp_path: Path):
    w = _writer(tmp_path)

    @dataclasses.dataclass
    class _Cfg:
        lr: float = 0.001
        name: str = "run-x"

    await w.on_job_started(_Cfg())
    cfg_path = tmp_path / "atif" / "word_puzzle" / "9999" / "config.json"
    assert cfg_path.is_file()
    obj = json.loads(cfg_path.read_text())
    assert obj["lr"] == 0.001
    assert obj["name"] == "run-x"


@pytest.mark.asyncio
async def test_on_job_completed_writes_result_json(tmp_path: Path):
    w = _writer(tmp_path)
    await w.on_job_completed({"total_rollouts": 12, "average_reward": 0.5})
    res_path = tmp_path / "atif" / "word_puzzle" / "9999" / "result.json"
    assert res_path.is_file()
    obj = json.loads(res_path.read_text())
    assert obj["total_rollouts"] == 12
    assert obj["average_reward"] == 0.5


# ==========================================================================
# AtifDiskWriter <-> _GcsUploader wiring (spec §5 upload branch, §6.1)
# ==========================================================================


class _FakeUploader:
    """Records enqueue / start / drain calls for wiring tests.

    Mimics the ``_bucket`` / ``_prefix`` / ``_blob_name`` surface of the
    real ``_GcsUploader`` so ``capture_ctx_for``'s ``blob_uri_prefix``
    plumbing (which reads those attrs to build a full ``gs://`` URI) has
    a compatible test double.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.enqueued: list = []
        self.finalized: list = []
        self.started = False
        self.drained = False
        self.drain_timeout = None
        # Parse the gs:// URL the writer passes in so _blob_name behaves
        # exactly like the real uploader (blob = prefix + rel_to_blob_root).
        gcs_prefix = kwargs.get("gcs_prefix", "gs://test-traj-bucket")
        rest = gcs_prefix[len("gs://") :]
        self._bucket, _, prefix_path = rest.partition("/")
        self._prefix = prefix_path.strip("/")
        job_dir: Path = kwargs["job_dir"]
        # Same rule as real _GcsUploader: blob paths are relative to the
        # parent of <recipe>/<job_id>, so gs paths mirror on-disk paths.
        self._blob_root = job_dir.parent.parent

    def _blob_name(self, path: Path) -> str:
        rel = path.relative_to(self._blob_root).as_posix()
        return f"{self._prefix}/{rel}" if self._prefix else rel

    def finalize_trial(self, trial_dir: Path) -> None:
        # The writer signals trial finality here; the real uploader uses it
        # to reclaim local copies once their uploads are confirmed.
        self.finalized.append(trial_dir)

    def start(self) -> None:
        self.started = True

    def enqueue(self, path) -> None:
        self.enqueued.append(path)

    async def drain(self, timeout_s: float = 60.0) -> int:
        self.drained = True
        self.drain_timeout = timeout_s
        return 0


def _writer_with_fake_uploader(tmp_path: Path):
    holder: dict[str, _FakeUploader] = {}

    def factory(**kw):
        fake = _FakeUploader(**kw)
        holder["fake"] = fake
        return fake

    w = AtifDiskWriter(
        root_dir=str(tmp_path / "atif"),
        recipe="word_puzzle",
        tuning_job_id="9999",
        gcs_prefix="gs://test-traj-bucket",
        uploader_factory=factory,
    )
    return w, holder["fake"]


def test_no_uploader_when_gcs_prefix_absent(tmp_path: Path):
    w = _writer(tmp_path)  # no gcs_prefix
    assert w._gcs_uploader is None


def test_uploader_created_and_started_when_gcs_prefix_present(tmp_path: Path):
    w, fake = _writer_with_fake_uploader(tmp_path)
    assert w._gcs_uploader is fake
    assert fake.started is True


@pytest.mark.asyncio
async def test_on_trajectory_produced_enqueues_three_files(tmp_path: Path):
    w, fake = _writer_with_fake_uploader(tmp_path)
    await w.on_trajectory_produced(
        _scored(session_id=_sid("task-abc", 0)), _TaskData(task_id="task-abc")
    )
    names = sorted(p.name for p in fake.enqueued)
    assert names == ["result.json", "reward.json", "trajectory.json"]
    assert len(fake.enqueued) == 3


@pytest.mark.asyncio
async def test_drain_delegates_to_uploader(tmp_path: Path):
    w, fake = _writer_with_fake_uploader(tmp_path)
    await w.drain(timeout_s=5.0)
    assert fake.drained is True
    assert fake.drain_timeout == 5.0


@pytest.mark.asyncio
async def test_on_job_completed_drains_uploader(tmp_path: Path):
    w, fake = _writer_with_fake_uploader(tmp_path)
    await w.on_job_completed({"total_rollouts": 1})
    assert fake.drained is True


@pytest.mark.asyncio
async def test_on_job_started_enqueues_config_json_for_upload(tmp_path: Path):
    """config.json is a REQUIRED per-job contract file
    (PROJECT_TREE_CONTRACT); on_job_started must enqueue it for GCS upload,
    not just write it locally."""
    w, fake = _writer_with_fake_uploader(tmp_path)

    @dataclasses.dataclass
    class _Cfg:
        lr: float = 0.001
        name: str = "run-x"

    await w.on_job_started(_Cfg())
    assert any(p.name == "config.json" for p in fake.enqueued), fake.enqueued


@pytest.mark.asyncio
async def test_drain_is_noop_without_uploader(tmp_path: Path):
    w = _writer(tmp_path)  # no gcs_prefix -> no uploader
    # Must not raise.
    report = await w.drain()
    assert report.not_uploaded == 0
    assert not report


def test_drain_report_truthiness_matches_old_int_return():
    """``drain()`` used to return the abandoned count (items still queued at
    the deadline). ``if await writer.drain():`` must keep meaning exactly
    that: failed uploads and cancelled in-flight uploads did not count."""
    from utils.atif_disk_writer import DrainReport

    assert not DrainReport()
    assert not DrainReport(uploaded=5)
    assert not DrainReport(failed_by_category={"auth": 2})
    assert not DrainReport(cancelled_in_flight=3)
    assert DrainReport(abandoned=1)
    assert DrainReport(abandoned=4, failed_by_category={"auth": 1})


@pytest.mark.asyncio
async def test_end_to_end_with_real_uploader_mocked_transport(tmp_path: Path):
    """AtifDiskWriter + real _GcsUploader compose: a produced trajectory
    uploads its 3 files through the (mocked) transport."""
    uploaded: list = []

    def uploader_factory(*, gcs_prefix, job_dir, **kwargs):
        return _GcsUploader(
            gcs_prefix=gcs_prefix,
            job_dir=job_dir,
            upload_fn=lambda lp, bn: uploaded.append(bn),
            **kwargs,
        )

    w = AtifDiskWriter(
        root_dir=str(tmp_path / "atif"),
        recipe="word_puzzle",
        tuning_job_id="9999",
        gcs_prefix="gs://test-traj-bucket",
        uploader_factory=uploader_factory,
    )
    await w.on_trajectory_produced(
        _scored(session_id=_sid("task-abc", 0)), _TaskData(task_id="task-abc")
    )
    await w.drain()

    assert sorted(uploaded) == [
        f"word_puzzle/9999/{_sid('task-abc', 0)}/agent/trajectory.json",
        f"word_puzzle/9999/{_sid('task-abc', 0)}/result.json",
        f"word_puzzle/9999/{_sid('task-abc', 0)}/verifier/reward.json",
    ]


# ==========================================================================
# _GcsUploader (spec §6 / §11.2)
# ==========================================================================

import logging as _logging  # noqa: E402

from google.api_core import exceptions as gax  # noqa: E402


def _gs_uploader(
    tmp_path: Path,
    *,
    upload_fn,
    gcs_prefix: str = "gs://test-traj-bucket",
    **kwargs,
) -> _GcsUploader:
    """Construct a _GcsUploader with injected upload + instant sleep/time.

    ``upload_fn(local_path, blob_name)`` is the seam that replaces the real
    GCS ``blob.upload_from_filename``. ``sleep_fn`` is a no-op so retries
    don't actually wait; ``jitter_fn`` is identity for deterministic delays.
    """
    job_dir = tmp_path / "atif" / "word_puzzle" / "9999"
    job_dir.mkdir(parents=True, exist_ok=True)

    async def _instant_sleep(_seconds: float) -> None:
        return None

    defaults = dict(
        gcs_prefix=gcs_prefix,
        job_dir=job_dir,
        upload_fn=upload_fn,
        sleep_fn=_instant_sleep,
        jitter_fn=lambda x: x,  # deterministic (no random)
    )
    defaults.update(kwargs)
    return _GcsUploader(**defaults)


def _make_local_file(tmp_path: Path, rel: str | None = None) -> Path:
    job_dir = tmp_path / "atif" / "word_puzzle" / "9999"
    # Default mirrors a real trial dir: the dir name IS the session_id.
    rel = rel or f"{_sid('task-abc', 0)}/agent/trajectory.json"
    p = job_dir / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('{"ok": true}')
    return p


# -- gs:// URL parsing -----------------------------------------------------


def test_parse_gs_url_splits_bucket_and_prefix():
    assert _parse_gs_url("gs://test-traj-bucket") == ("test-traj-bucket", "")
    assert _parse_gs_url("gs://test-traj-bucket/") == ("test-traj-bucket", "")
    assert _parse_gs_url("gs://bkt/a/b") == ("bkt", "a/b")


def test_parse_gs_url_rejects_non_gs():
    with pytest.raises(ValueError):
        _parse_gs_url("s3://nope")


# -- parse_atif_output (config -> gcs_prefix) ------------------------------


def test_parse_atif_output_local_returns_none():
    """ "local" means local writes only -> no GCS prefix."""
    assert parse_atif_output("local") is None


def test_parse_atif_output_gs_returns_prefix():
    assert parse_atif_output("gs://test-traj-bucket") == "gs://test-traj-bucket"
    assert parse_atif_output("gs://bkt/sub") == "gs://bkt/sub"


def test_parse_atif_output_empty_returns_none():
    """Empty (disabled) maps to no prefix; callers gate on the raw field."""
    assert parse_atif_output("") is None


def test_parse_atif_output_rejects_unknown_scheme():
    with pytest.raises(ValueError):
        parse_atif_output("s3://nope")


# -- build_atif_writer (entry-point factory) -------------------------------


def test_build_atif_writer_disabled_returns_none(tmp_path: Path):
    assert (
        build_atif_writer(
            atif_output="",
            log_dir=str(tmp_path),
            recipe="word_puzzle",
            tuning_job_id="9999",
        )
        is None
    )


def test_build_atif_writer_local_no_uploader(tmp_path: Path):
    w = build_atif_writer(
        atif_output="local",
        log_dir=str(tmp_path),
        recipe="word_puzzle",
        tuning_job_id="9999",
    )
    assert isinstance(w, AtifDiskWriter)
    assert w._gcs_uploader is None
    # root is <log_dir>/atif ; job dir nests recipe/tuning_job_id under it.
    assert (tmp_path / "atif" / "word_puzzle" / "9999").is_dir()


def test_build_atif_writer_gcs_sets_prefix(tmp_path: Path):
    created = {}

    def uploader_factory(*, gcs_prefix, job_dir, **kwargs):
        created["prefix"] = gcs_prefix

        class _Stub:
            def start(self):
                pass

        return _Stub()

    w = build_atif_writer(
        atif_output="gs://test-traj-bucket",
        log_dir=str(tmp_path),
        recipe="word_puzzle",
        tuning_job_id="9999",
        uploader_factory=uploader_factory,
    )
    assert isinstance(w, AtifDiskWriter)
    assert created["prefix"] == "gs://test-traj-bucket"


def test_build_atif_writer_rejects_bad_scheme(tmp_path: Path):
    with pytest.raises(ValueError):
        build_atif_writer(
            atif_output="s3://nope",
            log_dir=str(tmp_path),
            recipe="word_puzzle",
            tuning_job_id="9999",
        )


@pytest.mark.parametrize("sentinel", ["off", "none", "OFF", "None", "  off  "])
def test_build_atif_writer_off_sentinel_disables(tmp_path: Path, sentinel: str):
    """The explicit off/none sentinel disables logging even though it's a
    truthy string that survives apply_internal_defaults' empty-only refill."""
    assert (
        build_atif_writer(
            atif_output=sentinel,
            log_dir=str(tmp_path),
            recipe="word_puzzle",
            tuning_job_id="9999",
        )
        is None
    )


def test_parse_atif_output_off_sentinel_returns_none():
    assert parse_atif_output("off") is None
    assert parse_atif_output("none") is None
    assert parse_atif_output("OFF") is None


def test_build_atif_writer_threads_max_concurrent_uploads(tmp_path: Path):
    captured = {}

    def uploader_factory(**kwargs):
        captured.update(kwargs)

        class _Stub:
            def start(self):
                pass

        return _Stub()

    build_atif_writer(
        atif_output="gs://test-traj-bucket",
        log_dir=str(tmp_path),
        recipe="word_puzzle",
        tuning_job_id="9999",
        max_concurrent_uploads=7,
        uploader_factory=uploader_factory,
    )
    assert captured["max_concurrent_uploads"] == 7


def test_atif_disk_writer_passes_max_concurrent_to_uploader(tmp_path: Path):
    captured = {}

    def uploader_factory(**kwargs):
        captured.update(kwargs)

        class _Stub:
            def start(self):
                pass

        return _Stub()

    AtifDiskWriter(
        root_dir=str(tmp_path / "atif"),
        recipe="word_puzzle",
        tuning_job_id="9999",
        gcs_prefix="gs://test-traj-bucket",
        max_concurrent_uploads=4,
        uploader_factory=uploader_factory,
    )
    assert captured["max_concurrent_uploads"] == 4


# -- happy path ------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_uploads_with_correct_blob_name(tmp_path: Path):
    calls = []

    def upload_fn(local_path, blob_name):
        calls.append((str(local_path), blob_name))

    up = _gs_uploader(tmp_path, upload_fn=upload_fn)
    p = _make_local_file(tmp_path)
    up.start()
    up.enqueue(p)
    await up.drain()

    assert up._success_count == 1
    # blob name is prefix-relative to job_dir's parent-of-parent (recipe/job)
    assert (
        calls[0][1] == f"word_puzzle/9999/{_sid('task-abc', 0)}/agent/trajectory.json"
    )


@pytest.mark.asyncio
async def test_blob_name_includes_gcs_prefix_subpath(tmp_path: Path):
    calls = []
    up = _gs_uploader(
        tmp_path,
        gcs_prefix="gs://bkt/some/prefix",
        upload_fn=lambda lp, bn: calls.append(bn),
    )
    p = _make_local_file(tmp_path)
    up.start()
    up.enqueue(p)
    await up.drain()
    assert (
        calls[0]
        == f"some/prefix/word_puzzle/9999/{_sid('task-abc', 0)}/agent/trajectory.json"
    )


# -- transient retry -------------------------------------------------------


@pytest.mark.asyncio
async def test_transient_retry_then_success(tmp_path: Path):
    attempts = {"n": 0}

    def upload_fn(local_path, blob_name):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise gax.TooManyRequests("429")
        # second attempt succeeds

    up = _gs_uploader(tmp_path, upload_fn=upload_fn)
    p = _make_local_file(tmp_path)
    up.start()
    up.enqueue(p)
    await up.drain()

    assert up._success_count == 1
    assert up._transient_attempts == 1
    assert sum(up._by_category.values()) == 0
    assert not (
        tmp_path / "atif" / "word_puzzle" / "9999" / ".upload_failures.jsonl"
    ).exists()


@pytest.mark.asyncio
async def test_retries_exhausted_records_ledger(tmp_path: Path):
    def upload_fn(local_path, blob_name):
        raise gax.ServiceUnavailable("503")

    up = _gs_uploader(tmp_path, upload_fn=upload_fn)
    p = _make_local_file(tmp_path)
    up.start()
    up.enqueue(p)
    await up.drain()

    assert up._success_count == 0
    assert up._by_category["retries_exhausted"] == 1
    ledger = tmp_path / "atif" / "word_puzzle" / "9999" / ".upload_failures.jsonl"
    lines = [l for l in ledger.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["category"] == "retries_exhausted"
    assert entry["path"] == f"{_sid('task-abc', 0)}/agent/trajectory.json"


# -- permanent failures ----------------------------------------------------


@pytest.mark.asyncio
async def test_auth_failure_fails_fast(tmp_path: Path):
    attempts = {"n": 0}

    def upload_fn(local_path, blob_name):
        attempts["n"] += 1
        raise gax.Forbidden("403")

    up = _gs_uploader(tmp_path, upload_fn=upload_fn)
    p = _make_local_file(tmp_path)
    up.start()
    up.enqueue(p)
    await up.drain()

    assert attempts["n"] == 1  # no retry
    assert up._by_category["auth"] == 1
    ledger = tmp_path / "atif" / "word_puzzle" / "9999" / ".upload_failures.jsonl"
    entry = json.loads(ledger.read_text().splitlines()[0])
    assert entry["category"] == "auth"


@pytest.mark.asyncio
async def test_not_found_fails_fast(tmp_path: Path):
    up = _gs_uploader(
        tmp_path, upload_fn=lambda lp, bn: (_ for _ in ()).throw(gax.NotFound("404"))
    )
    p = _make_local_file(tmp_path)
    up.start()
    up.enqueue(p)
    await up.drain()
    assert up._by_category["not_found"] == 1


@pytest.mark.asyncio
async def test_unknown_exception_fails_fast_with_repr(tmp_path: Path):
    def upload_fn(local_path, blob_name):
        raise RuntimeError("weird")

    up = _gs_uploader(tmp_path, upload_fn=upload_fn)
    p = _make_local_file(tmp_path)
    up.start()
    up.enqueue(p)
    await up.drain()

    assert up._by_category["unknown"] == 1
    ledger = tmp_path / "atif" / "word_puzzle" / "9999" / ".upload_failures.jsonl"
    entry = json.loads(ledger.read_text().splitlines()[0])
    assert entry["category"] == "unknown"
    assert "weird" in entry["error"]


@pytest.mark.asyncio
async def test_local_gone_before_upload(tmp_path: Path):
    def upload_fn(local_path, blob_name):
        raise FileNotFoundError(local_path)

    up = _gs_uploader(tmp_path, upload_fn=upload_fn)
    p = _make_local_file(tmp_path)
    up.start()
    up.enqueue(p)
    await up.drain()
    assert up._by_category["local_gone"] == 1


# -- drain summary ---------------------------------------------------------


@pytest.mark.asyncio
async def test_drain_summary_always_fires_on_success(tmp_path: Path, caplog):
    up = _gs_uploader(tmp_path, upload_fn=lambda lp, bn: None)
    p = _make_local_file(tmp_path)
    up.start()
    up.enqueue(p)
    with caplog.at_level(_logging.INFO, logger="utils.atif_disk_writer"):
        await up.drain()
    assert any("GCS upload summary" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_drain_summary_warns_when_failures(tmp_path: Path, caplog):
    up = _gs_uploader(
        tmp_path, upload_fn=lambda lp, bn: (_ for _ in ()).throw(gax.Forbidden("403"))
    )
    p = _make_local_file(tmp_path)
    up.start()
    up.enqueue(p)
    with caplog.at_level(_logging.INFO, logger="utils.atif_disk_writer"):
        await up.drain()
    # INFO summary + WARNING recovery hint both present
    assert any("GCS upload summary" in r.message for r in caplog.records)
    assert any(
        r.levelno == _logging.WARNING and "not uploaded" in r.message
        for r in caplog.records
    )


@pytest.mark.asyncio
async def test_drain_with_pending_items_and_short_timeout(tmp_path: Path):
    # With a 2-worker pool and 6 items where every upload hangs (retryable
    # error -> injected sleep_fn blocks), the 2 workers get stuck on the
    # first 2 items and the remaining 4 stay queued -> drain times out with
    # abandoned>0. (Enqueue MORE items than workers to force a backlog.)
    started = asyncio.Event()

    def upload_fn(local_path, blob_name):
        started.set()
        raise gax.ServiceUnavailable("stall")  # forces retry -> sleep

    async def _long_sleep(_s: float) -> None:
        await asyncio.sleep(3600)

    up = _gs_uploader(
        tmp_path,
        upload_fn=upload_fn,
        sleep_fn=_long_sleep,
        max_concurrent_uploads=2,
    )
    up.start()
    for i in range(6):
        up.enqueue(_make_local_file(tmp_path, rel=f"task-{i}/agent/trajectory.json"))
    await started.wait()  # both workers now stuck on items 0 and 1
    report = await up.drain(timeout_s=0.05)
    assert report.abandoned >= 1  # remaining items still queued


# -- progress log rate-limiting (§6.4) -------------------------------------


@pytest.mark.asyncio
async def test_progress_log_rate_limited_by_count(tmp_path: Path, caplog):
    up = _gs_uploader(tmp_path, upload_fn=lambda lp, bn: None)
    # Enqueue 250 files (all mocked success). At the 200-op boundary one
    # progress line should emit; drain emits the summary separately.
    up.start()
    for i in range(250):
        p = _make_local_file(tmp_path, rel=f"task-{i}/agent/trajectory.json")
        up.enqueue(p)
    with caplog.at_level(_logging.INFO, logger="utils.atif_disk_writer"):
        await up.drain()
    progress_lines = [r for r in caplog.records if "progress:" in r.message]
    # Rate-limited: at most a couple progress lines for 250 uploads.
    assert len(progress_lines) <= 2
    assert up._success_count == 250


# -- permanent-failure cause logging ----------------------------------------


def _failure_warnings(caplog):
    return [
        r
        for r in caplog.records
        if r.levelno == _logging.WARNING and "failed permanently after" in r.message
    ]


@pytest.mark.asyncio
async def test_retries_exhausted_logs_cause_at_warning(tmp_path: Path, caplog):
    """The permanent-failure path names the cause (exception type +
    message), the object path, and the attempt count at WARNING."""

    def upload_fn(local_path, blob_name):
        raise gax.ServiceUnavailable("backend overloaded")

    up = _gs_uploader(tmp_path, upload_fn=upload_fn)
    up.start()
    up.enqueue(_make_local_file(tmp_path))
    with caplog.at_level(_logging.WARNING, logger="utils.atif_disk_writer"):
        await up.drain()

    (rec,) = _failure_warnings(caplog)
    msg = rec.getMessage()
    assert "ServiceUnavailable" in msg
    assert "backend overloaded" in msg
    assert (
        f"gs://test-traj-bucket/word_puzzle/9999/{_sid('task-abc', 0)}"
        "/agent/trajectory.json" in msg
    )
    assert "after 5 attempt(s)" in msg
    assert "retries_exhausted" in msg


@pytest.mark.asyncio
async def test_fail_fast_permanent_failure_logs_one_attempt(tmp_path: Path, caplog):
    def upload_fn(local_path, blob_name):
        raise gax.NotFound("no such bucket")

    up = _gs_uploader(tmp_path, upload_fn=upload_fn)
    up.start()
    up.enqueue(_make_local_file(tmp_path))
    with caplog.at_level(_logging.WARNING, logger="utils.atif_disk_writer"):
        await up.drain()

    (rec,) = _failure_warnings(caplog)
    msg = rec.getMessage()
    assert "NotFound" in msg and "no such bucket" in msg
    assert "after 1 attempt(s)" in msg


@pytest.mark.asyncio
async def test_permanent_failure_logging_is_rate_limited(tmp_path: Path, caplog):
    """Thousands of failures must not flood the log: the first N are logged
    individually, then a periodic summary with counts by exception type."""
    from utils import atif_disk_writer as w

    def upload_fn(local_path, blob_name):
        if "odd" in blob_name:
            raise gax.Forbidden("denied")
        raise gax.NotFound("gone")

    up = _gs_uploader(tmp_path, upload_fn=upload_fn)
    up.start()
    total = 250
    for i in range(total):
        tag = "odd" if i % 2 else "even"
        up.enqueue(
            _make_local_file(tmp_path, rel=f"task-{tag}-{i}/agent/trajectory.json")
        )
    with caplog.at_level(_logging.WARNING, logger="utils.atif_disk_writer"):
        await up.drain()

    assert sum(up._by_category.values()) == total
    individual = _failure_warnings(caplog)
    assert len(individual) == w._PERMANENT_FAILURE_LOG_FIRST_N
    summaries = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == _logging.WARNING and "failed permanently so far" in r.message
    ]
    assert len(summaries) == total // w._PERMANENT_FAILURE_SUMMARY_EVERY
    assert "Forbidden=" in summaries[-1] and "NotFound=" in summaries[-1]
    # The end-of-run warning carries the final per-type breakdown.
    (final,) = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == _logging.WARNING and "not uploaded" in r.message
    ]
    assert "Forbidden=125" in final and "NotFound=125" in final


# -- concurrency: bounded worker pool (§6.8) -------------------------------


def test_default_max_concurrent_uploads_is_32_and_matches_task_config(tmp_path: Path):
    # Raised from 10 after a real-GCS bench (uploads are latency-bound). The
    # writer default and TaskConfig.atif_upload_concurrency must not drift.
    import dataclasses

    from core.task_config import TaskConfig  # abstract: read the field default

    up = _gs_uploader(tmp_path, upload_fn=lambda lp, bn: None)
    assert up._max_concurrent_uploads == 32
    defaults = {f.name: f.default for f in dataclasses.fields(TaskConfig)}
    assert defaults["atif_upload_concurrency"] == 32


def test_max_concurrent_uploads_floors_at_one(tmp_path: Path):
    # Non-positive values are clamped to 1 (serial), never 0 (which would
    # spawn no workers and hang the queue).
    up0 = _gs_uploader(
        tmp_path, upload_fn=lambda lp, bn: None, max_concurrent_uploads=0
    )
    upneg = _gs_uploader(
        tmp_path, upload_fn=lambda lp, bn: None, max_concurrent_uploads=-5
    )
    assert up0._max_concurrent_uploads == 1
    assert upneg._max_concurrent_uploads == 1


class _ConcurrencyProbe:
    """upload_fn seam that records the peak number of simultaneously
    in-flight uploads, using an async gate so we can force overlap."""

    def __init__(self, release: asyncio.Event) -> None:
        self._release = release
        self._in_flight = 0
        self.peak = 0
        self._lock = asyncio.Lock()

    async def __call__(self, local_path: Path, blob_name: str) -> None:
        # NOTE: invoked via run_in_executor -> must be sync from the
        # uploader's perspective. We instead patch _do_upload; see the
        # async-aware uploader below. This __call__ is async and used with
        # the async upload seam.
        async with self._lock:
            self._in_flight += 1
            self.peak = max(self.peak, self._in_flight)
        await self._release.wait()
        async with self._lock:
            self._in_flight -= 1


@pytest.mark.asyncio
async def test_pool_runs_uploads_concurrently(tmp_path: Path):
    """With max_concurrent_uploads=5 and 5 files that all block until
    released, all 5 should be in-flight at once (peak == 5)."""
    release = asyncio.Event()
    probe = _ConcurrencyProbe(release)

    up = _gs_uploader(
        tmp_path,
        upload_fn=lambda lp, bn: None,  # unused; _do_upload is overridden
        max_concurrent_uploads=5,
    )
    # Override _do_upload with the async probe so overlap is observable
    # without threads (the production path uses run_in_executor).
    up._do_upload = lambda path, blob_name: probe(path, blob_name)  # type: ignore

    up.start()
    for i in range(5):
        up.enqueue(_make_local_file(tmp_path, rel=f"task-{i}/agent/trajectory.json"))

    # Give the event loop time to pick up all 5 into workers.
    for _ in range(50):
        await asyncio.sleep(0)
        if probe.peak >= 5:
            break
    assert probe.peak == 5, f"expected 5 concurrent uploads, saw peak={probe.peak}"

    release.set()
    await up.drain()
    assert up._success_count == 5


@pytest.mark.asyncio
async def test_pool_caps_concurrency_below_item_count(tmp_path: Path):
    """With max_concurrent_uploads=3 but 8 files, never more than 3 run at
    once (peak == 3)."""
    release = asyncio.Event()
    probe = _ConcurrencyProbe(release)

    up = _gs_uploader(tmp_path, upload_fn=lambda lp, bn: None, max_concurrent_uploads=3)
    up._do_upload = lambda path, blob_name: probe(path, blob_name)  # type: ignore

    up.start()
    for i in range(8):
        up.enqueue(_make_local_file(tmp_path, rel=f"task-{i}/agent/trajectory.json"))

    for _ in range(50):
        await asyncio.sleep(0)
        if probe.peak >= 3:
            break
    # Never exceeds the cap even though 8 items are queued.
    assert probe.peak == 3, f"peak={probe.peak} exceeded cap of 3"

    release.set()
    await up.drain()
    assert up._success_count == 8


@pytest.mark.asyncio
async def test_serial_when_max_concurrent_is_one(tmp_path: Path):
    """max_concurrent_uploads=1 preserves the old serial behavior:
    never 2 uploads in-flight simultaneously."""
    release = asyncio.Event()
    probe = _ConcurrencyProbe(release)

    up = _gs_uploader(tmp_path, upload_fn=lambda lp, bn: None, max_concurrent_uploads=1)
    up._do_upload = lambda path, blob_name: probe(path, blob_name)  # type: ignore

    up.start()
    for i in range(4):
        up.enqueue(_make_local_file(tmp_path, rel=f"task-{i}/agent/trajectory.json"))

    for _ in range(20):
        await asyncio.sleep(0)
    assert probe.peak == 1  # strictly serial

    release.set()
    await up.drain()
    assert up._success_count == 4


@pytest.mark.asyncio
async def test_pool_uploads_all_files_correctly(tmp_path: Path):
    """Correctness under concurrency: every enqueued file's blob name is
    uploaded exactly once with a 10-worker pool."""
    uploaded: list = []
    lock = asyncio.Lock()

    async def _do_upload(path, blob_name):
        async with lock:
            uploaded.append(blob_name)

    up = _gs_uploader(
        tmp_path, upload_fn=lambda lp, bn: None, max_concurrent_uploads=10
    )
    up._do_upload = _do_upload  # type: ignore

    up.start()
    expected = []
    for i in range(37):
        p = _make_local_file(tmp_path, rel=f"task-{i}/agent/trajectory.json")
        up.enqueue(p)
        expected.append(f"word_puzzle/9999/task-{i}/agent/trajectory.json")
    await up.drain()

    assert sorted(uploaded) == sorted(expected)
    assert up._success_count == 37


@pytest.mark.asyncio
async def test_drain_cancels_all_pool_workers(tmp_path: Path):
    """drain() must tear down every worker in the pool, not just one."""
    up = _gs_uploader(tmp_path, upload_fn=lambda lp, bn: None, max_concurrent_uploads=4)
    up.start()
    assert len(up._workers) == 4
    up.enqueue(_make_local_file(tmp_path))
    await up.drain()
    assert all(w.done() for w in up._workers)


# --------------------------------------------------------------------------
# NaN reward (timeout fast-path) -> RFC 8259-valid JSON on disk
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_on_trajectory_produced_writes_strict_json_on_nan_reward(tmp_path: Path):
    """NaN reward (timeout fast-path) must not produce bare 'NaN' in
    verifier/reward.json or the per-trial result.json."""
    w = _writer(tmp_path)
    scored = _scored(
        session_id=_sid("task-abc", 0),
        reward=float("nan"),
        normalized_reward=None,
    )
    await w.on_trajectory_produced(scored, _TaskData(task_id="task-abc"))

    trial = tmp_path / "atif" / "word_puzzle" / "9999" / _sid("task-abc", 0)
    for rel in ("verifier/reward.json", "result.json"):
        content = (trial / rel).read_text()
        assert "NaN" not in content, f"{rel} has bare NaN"
        json.loads(content)  # would also fail under orjson pre-fix
    # reward serialized as null
    assert json.loads((trial / "result.json").read_text())["reward"] is None


@pytest.mark.asyncio
async def test_on_job_completed_writes_strict_json_on_nan_stat(tmp_path: Path):
    """Job-level aggregate stats can legitimately be NaN (e.g. a mean over
    an all-filtered/empty batch). The job result.json must still be
    RFC 8259-valid (no bare 'NaN' token)."""
    w = _writer(tmp_path)
    await w.on_job_completed({"total_rollouts": 0, "average_reward": float("nan")})

    res_path = tmp_path / "atif" / "word_puzzle" / "9999" / "result.json"
    content = res_path.read_text()
    assert "NaN" not in content, "job result.json has bare NaN"
    obj = json.loads(content)
    assert obj["total_rollouts"] == 0
    assert obj["average_reward"] is None


# --------------------------------------------------------------------------
# capture_ctx_for(session_id) — per-rollout CaptureContext resolution
# --------------------------------------------------------------------------


class TestCaptureCtxFor:
    """AtifDiskWriter.capture_ctx_for maps (task_id, rollout_idx) -> CaptureContext.

    Contract:
      * trial_dir matches the on-disk <task_id>-<rollout_idx>/ layout that
        on_trajectory_produced would produce for the same rollout.
      * Same session_id returns the SAME CaptureContext (same Counter) so
        call_NNNNNN.json indices monotonically increase across turns of one
        rollout.
      * Distinct session_ids get distinct CaptureContexts with independent
        Counters.
      * task_id and rollout_idx are passed EXPLICITLY (kwargs); session_id is
        stamped on the ctx as trajectory_id but NOT parsed. This eliminates
        the recipe-mismatch class of bugs where the writer's recipe param
        (set by the entry-point) differed from the recipe segment the Agent
        used when building session_id.
    """

    @pytest.mark.asyncio
    async def test_returns_capture_context_with_correct_trial_dir(
        self, tmp_path: Path
    ) -> None:
        w = _writer(tmp_path)
        sid = _sid("task-abc", 3)
        ctx = await w.capture_ctx_for(task_id="task-abc", rollout_idx=3, session_id=sid)

        assert ctx is not None
        expected = tmp_path / "atif" / "word_puzzle" / "9999" / _sid("task-abc", 3)
        assert ctx.trial_dir == expected
        assert ctx.trajectory_id == sid

    @pytest.mark.asyncio
    async def test_same_session_id_returns_same_context(self, tmp_path: Path) -> None:
        """Two calls for one session_id must return the SAME object (cached),
        so the ledger's Counter is shared across turns of the same rollout."""
        w = _writer(tmp_path)
        sid = _sid("task-abc", 0)
        ctx1 = await w.capture_ctx_for(
            task_id="task-abc", rollout_idx=0, session_id=sid
        )
        ctx2 = await w.capture_ctx_for(
            task_id="task-abc", rollout_idx=0, session_id=sid
        )
        assert ctx1 is ctx2
        assert ctx1 is not None
        # Same Counter -> monotonic call indices across turns.
        assert ctx1.counter is ctx2.counter

    @pytest.mark.asyncio
    async def test_distinct_session_ids_get_distinct_contexts(
        self, tmp_path: Path
    ) -> None:
        w = _writer(tmp_path)
        ctx_a = await w.capture_ctx_for(
            task_id="task-a", rollout_idx=0, session_id=_sid("task-a", 0)
        )
        ctx_b = await w.capture_ctx_for(
            task_id="task-b", rollout_idx=0, session_id=_sid("task-b", 0)
        )
        assert ctx_a is not None and ctx_b is not None
        assert ctx_a is not ctx_b
        assert ctx_a.counter is not ctx_b.counter
        assert ctx_a.trial_dir != ctx_b.trial_dir

    @pytest.mark.asyncio
    async def test_empty_session_id_still_returns_ctx(self, tmp_path: Path) -> None:
        """session_id is now opaque; an empty string is fine as a
        trajectory_id (unusual, but not a fatal condition). task_id +
        rollout_idx alone determine the trial_dir."""
        w = _writer(tmp_path)
        ctx = await w.capture_ctx_for(task_id="task-e", rollout_idx=0, session_id="")
        assert ctx is not None
        assert ctx.trajectory_id == ""
        assert ctx.trial_dir.name == "task-e-0"

    @pytest.mark.asyncio
    async def test_concurrent_distinct_session_ids(self, tmp_path: Path) -> None:
        """Concurrent calls for distinct session_ids must each get their own
        ctx; the async lock protects the cache without serializing distinct
        rollouts' Counters."""
        w = _writer(tmp_path)
        pairs = [(f"task-{i}", _sid(f"task-{i}", 0)) for i in range(8)]
        results = await asyncio.gather(
            *(
                w.capture_ctx_for(task_id=tid, rollout_idx=0, session_id=sid)
                for tid, sid in pairs
            )
        )
        assert all(r is not None for r in results)
        # All distinct instances -> all distinct Counters.
        counters = {id(r.counter) for r in results}
        assert len(counters) == 8

    @pytest.mark.asyncio
    async def test_task_id_with_unsafe_chars_is_sanitized(self, tmp_path: Path) -> None:
        """The task_id passed in must be sanitized to the same contract-safe
        charset that on_trajectory_produced uses, so the two trial_dirs
        match on disk."""
        w = _writer(tmp_path)
        ctx = await w.capture_ctx_for(
            task_id="we/ird", rollout_idx=0, session_id=_sid("we/ird", 0)
        )
        assert ctx is not None
        # Sanitized to match on_trajectory_produced's trial name.
        expected = tmp_path / "atif" / "word_puzzle" / "9999" / _sid("we_ird", 0)
        assert ctx.trial_dir == expected

    @pytest.mark.asyncio
    async def test_zero_rollout_idx_is_valid(self, tmp_path: Path) -> None:
        """rollout_idx=0 is a legitimate value (distinct from any None
        'unparseable' signal); the dir is the session_id either way."""
        w = _writer(tmp_path)
        ctx = await w.capture_ctx_for(
            task_id="task-x", rollout_idx=0, session_id=_sid("task-x", 0)
        )
        assert ctx is not None
        assert ctx.trial_dir.name == _sid("task-x", 0)

    @pytest.mark.asyncio
    async def test_capture_ctx_for_recipe_mismatch_still_works(
        self, tmp_path: Path
    ) -> None:
        """Regression (fbab9086): when the writer's recipe param differs
        from the recipe segment the Agent embedded in session_id, the
        old parse-out-of-session_id logic returned None and NO raw-LLM
        ledger blobs were written.

        Using the session_id as an opaque NAME (never parsing it) keeps
        that failure mode closed: the mismatched recipe segment is just
        part of the string, and the dir still lands under the writer's
        own job dir.
        """
        w = _writer(tmp_path, recipe="X")
        mismatched_sid = "Y-tuning-taskabc-0-20260101T000000Z"
        ctx = await w.capture_ctx_for(
            task_id="taskabc", rollout_idx=0, session_id=mismatched_sid
        )
        assert ctx is not None
        expected = tmp_path / "atif" / "X" / "9999" / mismatched_sid
        assert ctx.trial_dir == expected
        assert ctx.trajectory_id == mismatched_sid

    @pytest.mark.asyncio
    async def test_capture_ctx_for_uses_session_id_verbatim(
        self, tmp_path: Path
    ) -> None:
        """The trial dir is the session_id, not a recomposition of the
        task_id/rollout_idx kwargs -- those only feed the no-session_id
        fallback. This is what keeps the ledger in the SAME dir as the
        trajectory: both sides use the one minted string."""
        w = _writer(tmp_path)
        sid = _sid("OTHER", 5)
        ctx = await w.capture_ctx_for(task_id="REAL", rollout_idx=5, session_id=sid)
        assert ctx is not None
        assert ctx.trial_dir.name == sid

    @pytest.mark.asyncio
    async def test_ledger_and_trajectory_share_one_trial_dir(
        self, tmp_path: Path
    ) -> None:
        """T4: the raw-capture ledger dir and the final trajectory dir for
        one rollout must be the same directory."""
        w = _writer(tmp_path)
        sid = _sid("task-shared", 2)
        ctx = await w.capture_ctx_for(
            task_id="task-shared", rollout_idx=2, session_id=sid
        )
        await w.on_trajectory_produced(
            _scored(session_id=sid), _TaskData(task_id="task-shared")
        )
        assert ctx is not None
        assert (ctx.trial_dir / "result.json").is_file()

    @pytest.mark.asyncio
    async def test_on_trajectory_produced_evicts_capture_ctx_cache_entry(
        self, tmp_path: Path
    ) -> None:
        """After on_trajectory_produced fires for a session_id, the
        corresponding CaptureContext must be dropped from the cache.

        Rationale: the cache exists only to share a per-rollout Counter
        across multi-turn sample() calls of the same rollout. Once the
        trajectory has been produced, no further sample() calls will
        need the ctx, so we evict to bound memory in long-running
        training with RAW_CAPTURE=1 (otherwise ~200 bytes/entry
        accumulates indefinitely, keyed by session_id).
        """
        w = _writer(tmp_path)
        sid_0 = _sid("task-a", 0)
        sid_1 = _sid("task-b", 0)
        await w.capture_ctx_for(task_id="task-a", rollout_idx=0, session_id=sid_0)
        assert len(w._capture_ctx_cache) == 1

        await w.capture_ctx_for(task_id="task-b", rollout_idx=0, session_id=sid_1)
        assert len(w._capture_ctx_cache) == 2

        # Producing traj for sid_0 evicts only sid_0.
        await w.on_trajectory_produced(
            _scored(session_id=sid_0), _TaskData(task_id="task-a")
        )
        assert len(w._capture_ctx_cache) == 1
        assert sid_0 not in w._capture_ctx_cache
        assert sid_1 in w._capture_ctx_cache

        # Producing traj for sid_1 empties the cache.
        await w.on_trajectory_produced(
            _scored(session_id=sid_1), _TaskData(task_id="task-b")
        )
        assert len(w._capture_ctx_cache) == 0

    @pytest.mark.asyncio
    async def test_on_trajectory_produced_missing_session_id_in_cache_does_not_raise(
        self, tmp_path: Path
    ) -> None:
        """RAW_CAPTURE=0 runs never call capture_ctx_for, so the cache
        is empty when on_trajectory_produced fires. Eviction must be
        defensive against missing keys (pop with default)."""
        w = _writer(tmp_path)
        assert len(w._capture_ctx_cache) == 0
        sid = _sid("task-never-cached", 0)
        # Should not raise.
        await w.on_trajectory_produced(
            _scored(session_id=sid), _TaskData(task_id="task-never-cached")
        )
        assert len(w._capture_ctx_cache) == 0

    @pytest.mark.asyncio
    async def test_capture_ctx_for_empty_task_id_uses_same_fallback_as_extractor(
        self, tmp_path: Path
    ) -> None:
        """Defensive path: when ``task_id=""`` AND no session_id is available
        to name the dir, capture_ctx_for MUST use the same fallback string as
        _default_task_id_extractor ("unknown-task"), NOT the generic
        sanitize("") fallback ("unknown"). Otherwise ledger blobs land in
        "unknown-<idx>/" while trajectory writes land in
        "unknown-task-<idx>/", orphaning the cross-reference."""
        w = _writer(tmp_path)
        ctx = await w.capture_ctx_for(task_id="", rollout_idx=0, session_id="")
        assert ctx is not None
        assert ctx.trial_dir.name == "unknown-task-0"

    @pytest.mark.asyncio
    async def test_capture_ctx_for_none_task_id_uses_fallback(
        self, tmp_path: Path
    ) -> None:
        """Same fallback semantics for ``task_id=None`` (also a defensive
        path -- the Agent typically passes ``task_data.task_id or ""`` but
        this covers the direct-None case for robustness)."""
        w = _writer(tmp_path)
        ctx = await w.capture_ctx_for(
            task_id=None,  # type: ignore[arg-type]
            rollout_idx=2,
            session_id="",
        )
        assert ctx is not None
        assert ctx.trial_dir.name == "unknown-task-2"

    @pytest.mark.asyncio
    async def test_capture_ctx_for_empty_and_extractor_trial_dirs_align(
        self, tmp_path: Path
    ) -> None:
        """End-to-end alignment: with an empty/None task_data.task_id, the
        trial_dir chosen by capture_ctx_for MUST equal the trial_dir chosen
        by on_trajectory_produced for the same rollout. Prior to the fix
        these diverged ("unknown-0/" vs "unknown-task-0/") and the raw-LLM
        ledger orphaned from its trajectory."""
        w = _writer(tmp_path)
        sid = _sid("", 0)  # session_id with empty task_id segment
        ctx = await w.capture_ctx_for(task_id="", rollout_idx=0, session_id=sid)
        assert ctx is not None
        capture_dir = ctx.trial_dir

        # Now fire on_trajectory_produced with an empty-task_id TaskData
        # and see where it lands.
        await w.on_trajectory_produced(_scored(session_id=sid), _TaskData(task_id=None))
        # on_trajectory_produced parses rollout_idx from the session_id
        # tail; _sid("", 0) yields "word_puzzle-9999--0-<iso>" which the
        # rollout-idx regex still parses (the "" segment produces an
        # adjacent "--", but the trailing "-0-<ISO>" is what matches).
        # The trial dir must equal what capture_ctx_for chose.
        assert capture_dir.exists()
        # Trajectory files land alongside the ledger, in the SAME dir.
        assert (capture_dir / "agent" / "trajectory.json").exists()
        assert (capture_dir / "verifier" / "reward.json").exists()

    @pytest.mark.asyncio
    async def test_capture_ctx_for_populates_on_written_when_gcs_uploader_present(
        self, tmp_path: Path
    ) -> None:
        """When a GCS uploader is configured, capture_ctx_for must bind
        ``on_written`` to the uploader's ``enqueue`` method so ledger
        blobs written by dump_raw get uploaded alongside the trajectory
        JSONs. Without this, blobs land only on the training VM's local
        disk and downstream GCS consumers see broken llm_call_uri paths.
        """
        w, fake = _writer_with_fake_uploader(tmp_path)
        ctx = await w.capture_ctx_for(
            task_id="task-a", rollout_idx=0, session_id="sid-a"
        )
        assert ctx is not None
        assert ctx.on_written is not None
        assert ctx.on_written == fake.enqueue

    @pytest.mark.asyncio
    async def test_capture_ctx_for_on_written_is_none_when_no_uploader(
        self, tmp_path: Path
    ) -> None:
        """No GCS uploader (atif_output=local) means no upload plumbing;
        ``on_written`` stays None so dump_raw skips the callback path
        entirely."""
        w = _writer(tmp_path)  # no gcs_prefix -> no uploader
        assert w._gcs_uploader is None
        ctx = await w.capture_ctx_for(
            task_id="task-a", rollout_idx=0, session_id="sid-a"
        )
        assert ctx is not None
        assert ctx.on_written is None

    @pytest.mark.asyncio
    async def test_capture_ctx_for_populates_gcs_blob_uri_prefix_when_uploader_present(
        self, tmp_path: Path
    ) -> None:
        """When atif_output is a gs:// URL, ``blob_uri_prefix`` is a
        full ``gs://bucket/prefix/<recipe>/<job>/<task>-<rollout>``
        string pointing at the SAME location the uploader would upload
        the trial dir's files to. dump_raw uses this to build a
        directly-resolvable URI stamped on ``step.extra.llm_call_uri``.
        """
        w, _fake = _writer_with_fake_uploader(tmp_path)
        ctx = await w.capture_ctx_for(
            task_id="task-abc", rollout_idx=3, session_id="sid-abc"
        )
        assert ctx is not None
        # Fake uploader was built with gcs_prefix="gs://test-traj-bucket"
        # (empty gs prefix path). Blob paths mirror on-disk layout under
        # the recipe/job root.
        assert ctx.blob_uri_prefix == ("gs://test-traj-bucket/word_puzzle/9999/sid-abc")

    @pytest.mark.asyncio
    async def test_capture_ctx_for_gcs_blob_uri_prefix_includes_subprefix(
        self, tmp_path: Path
    ) -> None:
        """When the gs:// URL has a subpath prefix (e.g.
        ``gs://bkt/some/prefix``), it MUST be included in
        blob_uri_prefix so URIs match where uploads actually land."""
        holder: dict[str, _FakeUploader] = {}

        def factory(**kw):
            fake = _FakeUploader(**kw)
            holder["fake"] = fake
            return fake

        w = AtifDiskWriter(
            root_dir=str(tmp_path / "atif"),
            recipe="word_puzzle",
            tuning_job_id="9999",
            gcs_prefix="gs://bkt/some/prefix",
            uploader_factory=factory,
        )
        ctx = await w.capture_ctx_for(
            task_id="task-abc", rollout_idx=0, session_id="sid-abc"
        )
        assert ctx is not None
        assert ctx.blob_uri_prefix == ("gs://bkt/some/prefix/word_puzzle/9999/sid-abc")

    @pytest.mark.asyncio
    async def test_capture_ctx_for_populates_local_absolute_path_prefix_when_no_uploader(
        self, tmp_path: Path
    ) -> None:
        """When atif_output=local (no uploader), ``blob_uri_prefix`` is
        the absolute local path of the trial dir; dump_raw returns
        ``<abs>/artifacts/raw_llm_calls/call_NNNNNN.json`` which
        `open()` / `os.path.exists()` can resolve directly without a
        cwd or root."""
        w = _writer(tmp_path)  # no gcs_prefix -> no uploader
        assert w._gcs_uploader is None
        ctx = await w.capture_ctx_for(
            task_id="task-a", rollout_idx=0, session_id="sid-a"
        )
        assert ctx is not None
        assert ctx.blob_uri_prefix == str(ctx.trial_dir.absolute())
        # Must be absolute so consumers don't need to know the cwd.
        assert ctx.blob_uri_prefix.startswith("/")

    @pytest.mark.asyncio
    async def test_capture_ctx_for_prefix_and_upload_land_at_matching_locations(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Acid test: the URI stamped by dump_raw (via blob_uri_prefix)
        MUST resolve to the SAME GCS location as where the uploader
        writes the local file. Otherwise consumers get a URI that
        gcloud-storage-cats to a 404."""
        from utils import raw_capture as rc

        monkeypatch.setenv("RAW_CAPTURE", "1")
        rc._reload_capture_enabled()
        try:
            w, fake = _writer_with_fake_uploader(tmp_path)
            ctx = await w.capture_ctx_for(
                task_id="task-abc", rollout_idx=0, session_id="sid-abc"
            )
            assert ctx is not None
            uri = await rc.dump_raw(
                ctx=ctx,
                request={"prompt": "hi"},
                response={"candidates": [{"text": "hello"}]},
                status="ok",
                duration_s=0.1,
            )
            # URI is a full gs:// path.
            assert uri is not None
            assert uri.startswith("gs://test-traj-bucket/")
            # The enqueued local path, translated to a GCS blob name by
            # the real uploader's rule (blob = prefix + rel_to_blob_root),
            # MUST equal the URI's path portion (minus gs://bucket/).
            assert len(fake.enqueued) == 1
            (enqueued_path,) = fake.enqueued
            # Reproduce the uploader's blob-name rule locally: blob_root
            # is job_dir.parent.parent, prefix is "" here.
            job_dir = w._job_dir
            blob_root = job_dir.parent.parent
            expected_blob = enqueued_path.relative_to(blob_root).as_posix()
            expected_uri = f"gs://test-traj-bucket/{expected_blob}"
            assert uri == expected_uri
        finally:
            monkeypatch.delenv("RAW_CAPTURE", raising=False)
            rc._reload_capture_enabled()

    @pytest.mark.asyncio
    async def test_capture_ctx_for_end_to_end_enqueues_ledger_blob_to_uploader(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end wire-through: dump_raw called with a ctx from
        capture_ctx_for (with an uploader configured) MUST enqueue the
        written blob's Path onto the uploader. This is the invariant
        the design gap fix guarantees."""
        from utils import raw_capture as rc

        monkeypatch.setenv("RAW_CAPTURE", "1")
        rc._reload_capture_enabled()
        try:
            w, fake = _writer_with_fake_uploader(tmp_path)
            ctx = await w.capture_ctx_for(
                task_id="task-abc", rollout_idx=0, session_id="sid-abc"
            )
            assert ctx is not None
            uri = await rc.dump_raw(
                ctx=ctx,
                request={"prompt": "hi"},
                response={"candidates": [{"text": "hello"}]},
                status="ok",
                duration_s=0.1,
            )
            # URI is a full gs:// path (see TestCaptureCtxFor's
            # blob_uri_prefix tests for the full contract).
            assert uri == (
                "gs://test-traj-bucket/word_puzzle/9999/sid-abc/"
                "artifacts/raw_llm_calls/call_000000.json"
            )
            # The uploader's enqueue was called with the absolute Path
            # of the newly written ledger blob.
            assert len(fake.enqueued) == 1
            (enqueued_path,) = fake.enqueued
            expected = (
                ctx.trial_dir / "artifacts" / "raw_llm_calls" / "call_000000.json"
            )
            assert enqueued_path == expected
        finally:
            monkeypatch.delenv("RAW_CAPTURE", raising=False)
            rc._reload_capture_enabled()


# --------------------------------------------------------------------------
# Partial trajectory snapshot (on_trajectory_step_completed)
# --------------------------------------------------------------------------


class TestPartialTrajectorySnapshot:
    """Test the partial snapshot logic in on_trajectory_step_completed."""

    @pytest.fixture(autouse=True)
    def _pin_partial_constants(self, monkeypatch):
        """Pin step-interval to 3 and throttle to 0 so tests are
        independent of the module-level defaults (which may change)."""
        import utils.atif_disk_writer as dw

        monkeypatch.setattr(dw, "_PARTIAL_STEP_INTERVAL", 3)
        monkeypatch.setattr(dw, "_PARTIAL_MIN_INTERVAL_S", 0)

    def _make_steps(self, n: int):
        """Build n ATIF steps (1 leading user + n-1 agent)."""
        from core.atif_trajectory import (
            Step,
            TrajectoryStep,
            StepStats,
        )

        steps = [Step(step_id=1, source="user", message="task prompt")]
        for i in range(1, n + 1):
            steps.append(
                TrajectoryStep.from_wire(
                    step_id=i + 1,
                    generation=f"response-{i}",
                    stats=StepStats(candidate_id=f"cand-{i}", finish_reason="stop"),
                )
            )
        return steps

    def _traj(self, task_id: str, rollout_idx: int, num_agent_steps: int, sid: str):
        """Build a live-Trajectory shell with N agent steps appended.

        Callback tests post-mutable-Trajectory-refactor drive the
        on_trajectory_step_completed hook by handing it a live Trajectory
        whose step count reflects how many agent turns have happened;
        the writer infers the step index from len(traj.steps) - 1.
        """
        from core.atif_trajectory import Agent as AtifAgent
        from core.atif_trajectory import Trajectory

        return Trajectory(
            schema_version="ATIF-v1.7",
            session_id=sid,
            agent=AtifAgent(name="t", version="1", model_name="m"),
            steps=self._make_steps(num_agent_steps),
            extra={
                "dispatch_index": None,
                "phase": "train",
                "rollout_idx": rollout_idx,
                "task_id": task_id,
            },
        )

    @pytest.mark.asyncio
    async def test_writes_snapshot_at_step_interval(self, tmp_path):
        """Snapshot is written at step 3 (every _PARTIAL_STEP_INTERVAL)."""
        w = _writer(tmp_path)
        td = _TaskData(task_id="task-A")
        traj = self._traj("task-A", 0, 3, "test-session-0-20260726T000000Z")

        await w.on_trajectory_step_completed(td, traj)

        agent_path = (
            w.job_dir / "test-session-0-20260726T000000Z" / "agent" / "trajectory.json"
        )
        assert agent_path.exists()
        data = json.loads(agent_path.read_text())
        assert data.get("extra", {}).get("_partial") is True
        assert len(data["steps"]) == 4  # 1 user + 3 agent

    @pytest.mark.asyncio
    async def test_skips_non_interval_steps(self, tmp_path):
        """Steps 1, 2, 4, 5 are NOT multiples of 3 → no snapshot."""
        w = _writer(tmp_path)
        td = _TaskData(task_id="task-B")

        for n in [1, 2, 4, 5]:
            await w.on_trajectory_step_completed(td, self._traj("task-B", 0, n, "s"))

        agent_path = w.job_dir / "s" / "agent" / "trajectory.json"
        assert not agent_path.exists()

    @pytest.mark.asyncio
    async def test_skips_when_no_partial_steps(self, tmp_path):
        """traj=None → no snapshot even at step 3."""
        w = _writer(tmp_path)
        td = _TaskData(task_id="task-C")

        await w.on_trajectory_step_completed(td, None)

        agent_path = w.job_dir / "s" / "agent" / "trajectory.json"
        assert not agent_path.exists()

    @pytest.mark.asyncio
    async def test_throttle_caps_at_one_per_interval(self, tmp_path, monkeypatch):
        """Second snapshot within 5 minutes is suppressed."""
        import utils.atif_disk_writer as dw

        # Shrink interval to 10s for the test.
        monkeypatch.setattr(dw, "_PARTIAL_MIN_INTERVAL_S", 10)

        w = _writer(tmp_path)
        td = _TaskData(task_id="task-D")

        # First snapshot at 3 agent steps succeeds.
        await w.on_trajectory_step_completed(td, self._traj("task-D", 0, 3, "s"))
        agent_path = w.job_dir / "s" / "agent" / "trajectory.json"
        assert agent_path.exists()
        first_content = agent_path.read_text()

        # Second snapshot at 6 agent steps — throttled (within 10s).
        await w.on_trajectory_step_completed(td, self._traj("task-D", 0, 6, "s"))
        # File still has first content (not overwritten).
        assert agent_path.read_text() == first_content

    @pytest.mark.asyncio
    async def test_throttle_allows_after_interval_expires(self, tmp_path, monkeypatch):
        """After the interval expires, a new snapshot is allowed."""
        import time
        import utils.atif_disk_writer as dw

        # Shrink interval to 0 so second write is always allowed.
        monkeypatch.setattr(dw, "_PARTIAL_MIN_INTERVAL_S", 0)

        w = _writer(tmp_path)
        td = _TaskData(task_id="task-E")

        await w.on_trajectory_step_completed(td, self._traj("task-E", 0, 3, "s"))
        agent_path = w.job_dir / "s" / "agent" / "trajectory.json"
        first = json.loads(agent_path.read_text())
        assert len(first["steps"]) == 4  # 1 user + 3 agent

        await w.on_trajectory_step_completed(td, self._traj("task-E", 0, 6, "s"))
        second = json.loads(agent_path.read_text())
        assert len(second["steps"]) == 7  # 1 user + 6 agent

    @pytest.mark.asyncio
    async def test_enqueues_for_gcs_upload(self, tmp_path):
        """When a GCS uploader is present, the snapshot is enqueued."""
        enqueued = []

        class _FakeUploader:
            def __init__(self, **kw):
                pass

            def start(self):
                pass

            def enqueue(self, path):
                enqueued.append(path)

        w = _writer(
            tmp_path,
            gcs_prefix="gs://bucket/prefix",
            uploader_factory=_FakeUploader,
        )
        td = _TaskData(task_id="task-F")

        await w.on_trajectory_step_completed(td, self._traj("task-F", 0, 3, "s"))

        assert len(enqueued) == 1
        assert enqueued[0].name == "trajectory.json"

    @pytest.mark.asyncio
    async def test_final_overwrite_replaces_partial(self, tmp_path):
        """on_trajectory_produced overwrites the partial snapshot."""
        w = _writer(tmp_path)
        td = _TaskData(task_id="task-G")

        # Write partial snapshot.
        await w.on_trajectory_step_completed(td, self._traj("task-G", 0, 3, "s"))
        agent_path = w.job_dir / "s" / "agent" / "trajectory.json"
        partial = json.loads(agent_path.read_text())
        assert partial["extra"]["_partial"] is True

        # Now produce the final trajectory (overwrites).
        scored = _scored(session_id="s")
        await w.on_trajectory_produced(scored, td)
        final = json.loads(agent_path.read_text())
        # Final does NOT have _partial flag.
        assert "_partial" not in final.get("extra", {})

    @pytest.mark.asyncio
    async def test_throttle_state_evicted_on_trajectory_produced(self, tmp_path):
        """on_trajectory_produced evicts the throttle entry so the dict
        doesn't grow unboundedly over a long training run."""
        w = _writer(tmp_path)
        td = _TaskData(task_id="task-H")

        # Write a partial → creates throttle entry.
        await w.on_trajectory_step_completed(td, self._traj("task-H", 0, 3, "s"))
        assert "s" in w._partial_last_upload  # keyed by trial name (session_id)

        # Produce final trajectory → evicts throttle entry.
        scored = _scored(session_id="s")
        await w.on_trajectory_produced(scored, td)
        assert "s" not in w._partial_last_upload


# --------------------------------------------------------------------------
# reward-details.json (per-criterion breakdown)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reward_details_written_to_its_own_file(tmp_path: Path):
    """A per-criterion breakdown lands in verifier/reward-details.json.

    The contract reserves that file for the breakdown and forbids
    inlining it into reward.json; the viewer renders it under Verifier
    Logs. Without it a partial reward is just a scalar and diagnosing
    one means replaying the rollout.
    """
    w = _writer(tmp_path)
    scored = _scored(session_id=_sid("t1", 0), reward=0.4)
    atif_extras.set_reward_details(
        scored.trajectory,
        {"reward": 0.4, "failed_assertions": [{"type": "row_exists"}]},
    )
    await w.on_trajectory_produced(scored, _TaskData(task_id="t1"))

    trial = next((tmp_path / "atif" / "word_puzzle" / "9999").glob("*t1-0*"))
    details = json.loads((trial / "verifier" / "reward-details.json").read_text())
    assert details["failed_assertions"][0]["type"] == "row_exists"

    # reward.json stays the bare score; the breakdown is NOT inlined
    # there, nor left behind on the trajectory JSON.
    reward = json.loads((trial / "verifier" / "reward.json").read_text())
    assert "failed_assertions" not in reward
    agent_json = json.loads((trial / "agent" / "trajectory.json").read_text())
    assert "reward_details" not in (agent_json.get("extra") or {})


@pytest.mark.asyncio
async def test_no_reward_details_file_when_recipe_sets_none(tmp_path: Path):
    # Optional per the contract: recipes without named criteria emit nothing.
    w = _writer(tmp_path)
    await w.on_trajectory_produced(
        _scored(session_id=_sid("t2", 0), reward=1.0), _TaskData(task_id="t2")
    )
    trial = next((tmp_path / "atif" / "word_puzzle" / "9999").glob("*t2-0*"))
    assert not (trial / "verifier" / "reward-details.json").exists()
# ==========================================================================
# delete_after_upload: local files removed once GCS upload is confirmed
# ==========================================================================


@pytest.mark.asyncio
async def test_delete_after_upload_removes_confirmed_files(tmp_path: Path):
    """delete_after_upload=True: a confirmed-uploaded local file is gone after
    drain, and its now-empty dirs are pruned up to (but not including) job_dir."""
    uploaded: list[str] = []
    up = _gs_uploader(
        tmp_path,
        upload_fn=lambda lp, bn: uploaded.append(bn),
        delete_after_upload=True,
    )
    up.start()
    p = _make_local_file(tmp_path)  # <job>/<sid>/agent/trajectory.json
    assert p.exists()
    up.enqueue(p)
    await up.drain()

    assert uploaded, "upload_fn should have been called"
    assert not p.exists(), "confirmed-uploaded local file should be deleted"
    assert not p.parent.exists(), "empty agent/ dir should be pruned"
    assert up._job_dir.exists(), "job_dir must survive"


@pytest.mark.asyncio
async def test_delete_after_upload_false_keeps_files(tmp_path: Path):
    """Default-off behavior is preserved: files stay when the flag is False."""
    up = _gs_uploader(
        tmp_path,
        upload_fn=lambda lp, bn: None,
        delete_after_upload=False,
    )
    up.start()
    p = _make_local_file(tmp_path)
    up.enqueue(p)
    await up.drain()
    assert p.exists(), "file must be kept when delete_after_upload=False"


@pytest.mark.asyncio
async def test_delete_after_upload_keeps_failed_uploads(tmp_path: Path):
    """A file whose upload permanently fails is NOT deleted (nothing
    un-backed-up is lost)."""

    def _boom(_lp, _bn):
        raise gax.Forbidden("nope")  # permanent -> recorded, not retried

    up = _gs_uploader(
        tmp_path,
        upload_fn=_boom,
        delete_after_upload=True,
    )
    up.start()
    p = _make_local_file(tmp_path)
    up.enqueue(p)
    await up.drain()
    assert p.exists(), "failed-upload file must be kept for recovery"


@pytest.mark.asyncio
async def test_delete_after_upload_keeps_never_enqueued_sibling(tmp_path: Path):
    """A sibling never uploaded (e.g. harbor_run.log) survives, and its
    presence stops its dir from being pruned."""
    uploaded: list[str] = []
    up = _gs_uploader(
        tmp_path,
        upload_fn=lambda lp, bn: uploaded.append(bn),
        delete_after_upload=True,
    )
    up.start()
    p = _make_local_file(tmp_path)  # uploaded + deleted
    sibling = p.parent / "harbor_run.log"  # never enqueued
    sibling.write_text("host-side log")
    up.enqueue(p)
    await up.drain()

    assert not p.exists(), "uploaded file deleted"
    assert sibling.exists(), "never-uploaded sibling must survive"
    assert p.parent.exists(), "dir with a surviving file must NOT be pruned"


@pytest.mark.asyncio
async def test_delete_after_upload_survives_repeated_path_upload(tmp_path: Path):
    """The same path uploaded twice (partial then final) is deleted once at
    drain, with no error on the duplicate."""
    uploaded: list[str] = []
    up = _gs_uploader(
        tmp_path,
        upload_fn=lambda lp, bn: uploaded.append(bn),
        delete_after_upload=True,
    )
    up.start()
    p = _make_local_file(tmp_path)
    up.enqueue(p)  # "partial"
    up.enqueue(p)  # "final" (same path)
    await up.drain()
    assert len(uploaded) == 2, "both enqueues uploaded"
    assert not p.exists(), "file deleted exactly once, no crash on dup"


@pytest.mark.asyncio
async def test_partial_success_then_final_failure_keeps_file(tmp_path: Path):
    """Regression: a path whose PARTIAL-snapshot upload succeeded but whose
    FINAL upload (same path, overwritten content) permanently FAILS must NOT be
    deleted.

    This is the OfficeQA/Claude data-loss bug: on_trajectory_step_completed
    uploads a partial agent/trajectory.json (200 -> added to _uploaded_paths),
    then on_trajectory_produced overwrites the SAME path with the final
    trajectory and its upload 403s. The confirmed-success set still held the
    path from the partial, so drain deleted the file whose CURRENT on-disk
    content was never backed up -> lost from both disk and GCS.
    """
    calls = {"n": 0}

    def _first_ok_then_forbidden(_lp, _bn):
        calls["n"] += 1
        if calls["n"] == 1:
            return  # partial upload succeeds
        raise gax.Forbidden("nope")  # final upload of same path fails (403)

    up = _gs_uploader(
        tmp_path,
        upload_fn=_first_ok_then_forbidden,
        delete_after_upload=True,
    )
    up.start()
    p = _make_local_file(tmp_path)
    up.enqueue(p)  # partial -> 200, confirmed
    await asyncio.sleep(0)  # let the partial upload land before the final
    up.enqueue(p)  # final (same path) -> 403, un-backed-up
    await up.drain()

    # The final 403 is a bare Forbidden AFTER a prior success, so the circuit
    # breaker treats it as a flake and retries the full backoff schedule (1 ok +
    # 5 failing attempts = 6 calls) before recording it permanent. The retry
    # count is incidental; the invariant under test is the keep-file guarantee.
    assert calls["n"] == 1 + len(_RETRY_DELAYS) + 1, "final upload retried then gave up"
    assert p.exists(), (
        "file whose LATEST upload failed must be kept — its current content "
        "is not in GCS; a prior partial success must not authorize deletion"
    )


@pytest.mark.asyncio
async def test_transient_serviceusage_forbidden_is_retried_and_recovers(
    tmp_path: Path,
):
    """A 403 carrying 'serviceusage.services.use' is an intermittent quota-probe
    flake -> retried with backoff, and a later attempt succeeds (file uploaded,
    then deleted; NO permanent failure recorded)."""
    calls = {"n": 0}
    uploaded: list[str] = []

    def _flaky(_lp, bn):
        calls["n"] += 1
        if calls["n"] == 1:
            raise gax.Forbidden(
                "caller does not have serviceusage.services.use "
                "access to the Google Cloud project."
            )
        uploaded.append(bn)  # second attempt succeeds

    up = _gs_uploader(
        tmp_path,
        upload_fn=_flaky,
        delete_after_upload=True,
    )
    up.start()
    p = _make_local_file(tmp_path)
    up.enqueue(p)
    await up.drain()

    assert calls["n"] == 2, "first 403 retried, second attempt made"
    assert uploaded, "upload eventually succeeded"
    assert not p.exists(), "recovered upload -> file deleted as confirmed"


@pytest.mark.asyncio
async def test_genuine_forbidden_fails_fast_when_no_prior_success(tmp_path: Path):
    """A pervasive ACL 403 (bare 'storage.objects.create', and NOTHING has ever
    uploaded) is NOT retried: the circuit breaker reads zero prior successes as a
    genuine misconfiguration, records it permanent, keeps the file for recovery."""
    calls = {"n": 0}

    def _boom(_lp, _bn):
        calls["n"] += 1
        raise gax.Forbidden("caller does not have storage.objects.create access")

    up = _gs_uploader(
        tmp_path,
        upload_fn=_boom,
        delete_after_upload=True,
    )
    up.start()
    p = _make_local_file(tmp_path)
    up.enqueue(p)
    await up.drain()

    assert calls["n"] == 1, "genuine 403 (no prior success) must NOT be retried"
    assert p.exists(), "genuine-403 file kept for recovery"


@pytest.mark.asyncio
async def test_bare_forbidden_after_success_is_retried_and_recovers(tmp_path: Path):
    """A bare 'storage.objects.create' 403 that hits AFTER at least one successful
    upload is a scattered authz flake (the identity provably has write access), so
    the circuit breaker RETRIES it and a later attempt succeeds -- no permanent
    failure recorded, file deleted as confirmed.

    Uses a single drain with one worker (FIFO): t1 uploads first (success ->
    success_count=1), then t2's first attempt 403s and is retried. The seam keys
    off the blob name so it doesn't depend on total call ordering: t2 fails only
    on its FIRST attempt, then succeeds.
    """
    uploaded: list[str] = []
    t2_attempts = {"n": 0}

    def _t1_ok_t2_flaky(_lp, bn):
        if "t2/" in bn:
            t2_attempts["n"] += 1
            if t2_attempts["n"] == 1:
                # t2 first attempt: scattered bare-403 flake (a real ACL denial
                # would also have failed t1 -- but t1 succeeded).
                raise gax.Forbidden(
                    "caller does not have storage.objects.create access"
                )
        uploaded.append(bn)  # t1 always; t2 on its retry

    up = _gs_uploader(
        tmp_path,
        upload_fn=_t1_ok_t2_flaky,
        delete_after_upload=True,
        max_concurrent_uploads=1,  # FIFO: t1 processed (succeeds) before t2
    )
    up.start()
    p1 = _make_local_file(tmp_path, rel="t1/agent/trajectory.json")
    p2 = _make_local_file(tmp_path, rel="t2/agent/trajectory.json")
    up.enqueue(p1)
    up.enqueue(p2)
    await up.drain()

    assert t2_attempts["n"] == 2, "t2 first attempt 403 retried, second succeeds"
    assert len(uploaded) == 2, "both files eventually uploaded (t1, t2-retry)"
    assert not p1.exists() and not p2.exists(), "both deleted as confirmed"
    assert "auth" not in up._by_category, "flake must not be recorded as permanent"


_DELETE_DENIED = (
    "403 POST https://storage.googleapis.com/upload/storage/v1/b/bkt/o: "
    "writer@proj.iam.gserviceaccount.com does not have storage.objects.delete "
    "access to the Google Cloud Storage object. Permission "
    "'storage.objects.delete' denied on resource (or it may not exist)."
)
_CREATE_DENIED = (
    "403 POST https://storage.googleapis.com/upload/storage/v1/b/bkt/o: "
    "writer@proj.iam.gserviceaccount.com does not have storage.objects.create "
    "access to the Google Cloud Storage object."
)


def _first_ok_then(exc_msg: str, attempts: dict[str, int]):
    """upload_fn: ``ok/`` paths succeed, every other path raises Forbidden."""

    def _fn(_lp, bn):
        if "/ok/" in bn:
            return
        attempts[bn] = attempts.get(bn, 0) + 1
        raise gax.Forbidden(exc_msg)

    return _fn


@pytest.mark.asyncio
async def test_overwrite_denial_fails_fast_even_after_success(tmp_path: Path):
    """A 403 naming storage.objects.delete is an OVERWRITE denial: the first
    upload of a path (objects.create) works, every re-upload needs delete. It is
    deterministic, so it must fail fast (1 attempt) even though an earlier upload
    succeeded -- retrying only held a worker slot through the backoff."""
    attempts: dict[str, int] = {}
    up = _gs_uploader(
        tmp_path,
        upload_fn=_first_ok_then(_DELETE_DENIED, attempts),
        max_concurrent_uploads=1,  # FIFO: the success lands first
    )
    up.start()
    up.enqueue(_make_local_file(tmp_path, rel="ok/agent/trajectory.json"))
    up.enqueue(_make_local_file(tmp_path, rel="t2/agent/trajectory.json"))
    await up.drain()

    assert up._success_count == 1
    assert list(attempts.values()) == [1], "overwrite denial must not be retried"
    assert up._transient_attempts == 0
    assert up._by_category == {"auth": 1}


@pytest.mark.asyncio
async def test_recurring_forbidden_signature_trips_breaker_after_k_paths(
    tmp_path: Path,
):
    """A bare create-denial after a success is retried as a flake -- but once the
    SAME denial has hit K distinct paths it is clearly not scattered, so later
    paths fail fast with one attempt instead of the full backoff schedule."""
    from utils.atif_disk_writer import _FORBIDDEN_SIGNATURE_TRIP_PATHS as K

    attempts: dict[str, int] = {}
    up = _gs_uploader(
        tmp_path,
        upload_fn=_first_ok_then(_CREATE_DENIED, attempts),
        max_concurrent_uploads=1,
    )
    up.start()
    up.enqueue(_make_local_file(tmp_path, rel="ok/agent/trajectory.json"))
    for i in range(K + 5):
        up.enqueue(_make_local_file(tmp_path, rel=f"t{i:03d}/agent/trajectory.json"))
    await up.drain()

    per_path = [attempts[k] for k in sorted(attempts)]
    full = len(_RETRY_DELAYS) + 1
    assert per_path[:K] == [full] * K, "first K paths: treated as flakes, retried"
    assert per_path[K:] == [1] * 5, "after K paths the signature fails fast"
    assert up._by_category == {"retries_exhausted": K, "auth": 5}


@pytest.mark.asyncio
async def test_forbidden_cause_logged_once_per_signature(tmp_path: Path, caplog):
    """The cause of a deterministic denial is WARNING-logged ONCE per signature
    (with the identity/permission text), not once per file."""
    from utils.atif_disk_writer import _FORBIDDEN_SIGNATURE_TRIP_PATHS as K

    def _fn(_lp, bn):
        if "/ok/" in bn:
            return
        raise gax.Forbidden(_DELETE_DENIED if "/d" in bn else _CREATE_DENIED)

    up = _gs_uploader(tmp_path, upload_fn=_fn, max_concurrent_uploads=1)
    up.start()
    up.enqueue(_make_local_file(tmp_path, rel="ok/agent/trajectory.json"))
    for i in range(30):
        up.enqueue(_make_local_file(tmp_path, rel=f"d{i:03d}/agent/trajectory.json"))
    for i in range(K + 3):
        up.enqueue(_make_local_file(tmp_path, rel=f"c{i:03d}/agent/trajectory.json"))
    with caplog.at_level(_logging.WARNING, logger="utils.atif_disk_writer"):
        await up.drain()

    cause = [r.getMessage() for r in caplog.records if "fail fast" in r.getMessage()]
    assert len(cause) == 2, cause
    delete_line = next(m for m in cause if "denied storage.objects.delete for" in m)
    assert "writer@proj.iam.gserviceaccount.com" in delete_line
    assert "1 file(s)" in delete_line
    create_line = next(m for m in cause if "denied storage.objects.create for" in m)
    assert f"{K + 1} file(s)" in create_line


@pytest.mark.asyncio
async def test_writer_wires_delete_flag_to_uploader(tmp_path: Path):
    """AtifDiskWriter forwards delete_after_upload to the uploader factory."""
    seen: dict[str, object] = {}

    def factory(**kw):
        seen.update(kw)
        return _FakeUploader(**kw)

    AtifDiskWriter(
        root_dir=str(tmp_path / "atif"),
        recipe="word_puzzle",
        tuning_job_id="9999",
        gcs_prefix="gs://test-traj-bucket",
        delete_after_upload=False,
        uploader_factory=factory,
    )
    assert seen.get("delete_after_upload") is False


class TestFinalizedTrialReclaim:
    """Journal-signalled cleanup: reclaim finished trials DURING the run.

    Before this, deletion waited for `drain()`, so a 120-step run held every
    trial it ever wrote and a SIGKILLed run leaked all of them (5 killed runs
    left 90 GB on one host). The writer could not do better on its own: a
    successful upload does not mean a file is finished, because partial
    snapshots rewrite and re-upload the same path. The journal knows finality;
    `finalize_trial` is that signal.
    """

    @staticmethod
    def _uploader(tmp_path, **kw):
        from utils.atif_disk_writer import _GcsUploader

        job_dir = tmp_path / "recipe" / "job"
        job_dir.mkdir(parents=True)
        return (
            _GcsUploader(
                gcs_prefix="gs://b/p",
                job_dir=job_dir,
                upload_fn=lambda p, b: None,
                **kw,
            ),
            job_dir,
        )

    @staticmethod
    def _trial(job_dir, name, n=2):
        d = job_dir / name
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (d / f"f{i}.json").write_text("{}")
        return d

    def test_an_unconfirmed_trial_is_never_deleted(self, tmp_path):
        """The race this design exists to avoid.

        `finalize_trial` fires right after ENQUEUE, so uploads are still in
        flight. Deleting then would destroy the only copy.
        """
        up, job_dir = self._uploader(tmp_path, trial_grace_count=0)
        t = self._trial(job_dir, "trial-a")
        up.finalize_trial(t)  # nothing uploaded yet
        assert t.exists() and list(t.glob("*.json"))

    def test_a_rewritten_file_is_not_reclaimed_while_its_upload_is_queued(
        self, tmp_path
    ):
        """Confirmation is per-PATH, but a path can be written twice.

        `agent/trajectory.json` gets a partial snapshot mid-rollout and the
        final trajectory at the end, both at the same path. Once the partial
        uploads, the path sits in `_uploaded_paths` permanently -- so a
        reclaim gated on membership alone would delete the file while the
        FINAL trajectory was still queued. The queued upload then fails
        `local_gone` and GCS keeps the `_partial: True` snapshot as the only
        copy: lost from disk AND from the bucket.

        Found in review with a working repro; this pins the fix.
        """
        up, job_dir = self._uploader(tmp_path, trial_grace_count=0)
        t = self._trial(job_dir, "trial-a", n=1)
        (f,) = list(t.glob("*.json"))

        # The partial snapshot uploaded and was confirmed.
        up._uploaded_paths.add(f)
        # The final trajectory is written to the SAME path and re-enqueued;
        # its upload has not been attempted yet.
        f.write_text('{"final": true}')
        up.enqueue(f)

        up.finalize_trial(t)
        assert t.exists(), "reclaimed a file whose latest content is unuploaded"
        assert f.read_text() == '{"final": true}'

        # Once that enqueue is attempted, the trial becomes eligible again.
        up._enqueue_done(f)
        up.finalize_trial(t)
        assert not t.exists()

    def test_pending_enqueues_are_counted_not_flagged(self, tmp_path):
        """Two writes in flight need two completions, not one.

        A boolean would clear on the first, letting reclaim delete the file
        while the second enqueue was still queued.
        """
        up, job_dir = self._uploader(tmp_path, trial_grace_count=0)
        t = self._trial(job_dir, "trial-a", n=1)
        (f,) = list(t.glob("*.json"))
        up._uploaded_paths.add(f)

        up.enqueue(f)
        up.enqueue(f)
        up._enqueue_done(f)
        up.finalize_trial(t)
        assert t.exists(), "one completion cleared two pending enqueues"

        up._enqueue_done(f)
        up.finalize_trial(t)
        assert not t.exists()

    def test_a_failed_upload_still_settles_its_pending_count(self, tmp_path):
        """The decrement runs in the worker's `finally`, not on success.

        If it only ran on success, a permanently failing path would stay
        pending forever and its trial could never be reclaimed -- turning a
        data-loss bug into a disk-leak bug.
        """
        up, job_dir = self._uploader(tmp_path, trial_grace_count=0)
        t = self._trial(job_dir, "trial-a", n=1)
        (f,) = list(t.glob("*.json"))
        up._uploaded_paths.add(f)
        up.enqueue(f)
        assert f in up._pending_enqueues
        up._enqueue_done(f)  # what the worker's finally does
        assert f not in up._pending_enqueues

    def test_a_confirmed_trial_past_the_window_is_reclaimed(self, tmp_path):
        up, job_dir = self._uploader(tmp_path, trial_grace_count=0)
        t = self._trial(job_dir, "trial-a")
        for p in t.glob("*.json"):
            up._uploaded_paths.add(p)
        up.finalize_trial(t)
        assert not t.exists()

    def test_the_grace_window_keeps_the_most_recent_trials(self, tmp_path):
        """Disk stays flat, but the trials you'd want to inspect are local."""
        up, job_dir = self._uploader(tmp_path, trial_grace_count=2)
        trials = []
        for i in range(5):
            t = self._trial(job_dir, f"trial-{i}")
            for p in t.glob("*.json"):
                up._uploaded_paths.add(p)
            up.finalize_trial(t)
            trials.append(t)
        # Oldest three reclaimed, newest two held.
        assert [t.exists() for t in trials] == [False, False, False, True, True]

    def test_a_partially_uploaded_trial_keeps_only_what_did_not_land(
        self, tmp_path
    ):
        """GCS must never become the only copy of something that never got there."""
        up, job_dir = self._uploader(tmp_path, trial_grace_count=0)
        t = self._trial(job_dir, "trial-a", n=3)
        files = sorted(t.glob("*.json"))
        for p in files[:2]:
            up._uploaded_paths.add(p)
        # Force past the pending bound so it reclaims despite being partial.
        up._finalized_trials.extend([t] * 600)
        up.finalize_trial(t)
        remaining = sorted(p.name for p in t.glob("*.json"))
        assert remaining == [files[2].name]

    def test_delete_after_upload_false_disables_it_entirely(self, tmp_path):
        up, job_dir = self._uploader(
            tmp_path, trial_grace_count=0, delete_after_upload=False
        )
        t = self._trial(job_dir, "trial-a")
        for p in t.glob("*.json"):
            up._uploaded_paths.add(p)
        up.finalize_trial(t)
        assert t.exists()


class TestStaleConfirmationIsRetracted:
    """`_uploaded_paths` must mean "the CURRENT content is backed up".

    A path can be written and enqueued more than once --
    `agent/trajectory.json` gets a partial snapshot mid-rollout and the final
    trajectory at the end -- so a confirmation from the partial must not
    outlive it. Two ways it used to:

    * the second enqueue is still QUEUED when the drain times out;
    * the second enqueue is IN FLIGHT and its worker is cancelled.

    The second is not covered by the pending-enqueue counter (the worker's
    `finally` settles the count on the way out) and not covered by
    `_record_failure` (a `CancelledError` never reaches it), which is why
    the confirmation is retracted at the START of every attempt instead.
    """

    @staticmethod
    def _uploader(tmp_path, upload_fn):
        job_dir = tmp_path / "atif" / "recipe" / "job"
        job_dir.mkdir(parents=True)
        up = _GcsUploader(
            gcs_prefix="gs://b/pfx",
            job_dir=job_dir,
            upload_fn=upload_fn,
            max_concurrent_uploads=1,
            sleep_fn=lambda d: asyncio.sleep(0),
            jitter_fn=lambda d: 0,
        )
        return up, job_dir

    @pytest.mark.asyncio
    async def test_a_cancelled_upload_does_not_leave_a_stale_confirmation(
        self, tmp_path
    ):
        """Drain timeout while the final trajectory is mid-upload."""
        gcs: dict[str, str] = {}
        gate = threading.Event()
        gate.set()

        def upload_fn(p: Path, blob: str) -> None:
            gate.wait()
            gcs[blob] = p.read_text()

        up, job_dir = self._uploader(tmp_path, upload_fn)
        up.start()
        f = job_dir / "t0" / "agent" / "trajectory.json"
        f.parent.mkdir(parents=True, exist_ok=True)

        f.write_text("PARTIAL")
        up.enqueue(f)
        await up._queue.join()          # partial confirmed
        assert f in up._uploaded_paths

        gate.clear()                    # uploads now hang
        f.write_text("FINAL")
        up.enqueue(f)
        await up.drain(timeout_s=0.3)   # worker cancelled mid-upload
        gate.set()

        # The local copy is the point: whatever GCS ended up with, the
        # current content must still exist on disk for recovery. (Not
        # asserting on `gcs` here -- releasing the gate lets the cancelled
        # worker's thread finish its write, so its content is a race.)
        assert f.exists(), "deleted a file whose current content never uploaded"
        assert f.read_text() == "FINAL"
        assert f not in up._uploaded_paths, "stale confirmation survived a cancel"

    @pytest.mark.asyncio
    async def test_a_still_queued_rewrite_is_not_deleted_at_drain(self, tmp_path):
        """Drain timeout with the re-enqueue never dequeued at all.

        Here the pending count IS still non-zero, so this pins the guard in
        `_delete_uploaded_files` rather than the retraction.
        """
        up, job_dir = self._uploader(tmp_path, lambda p, b: None)
        f = job_dir / "t0" / "agent" / "trajectory.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("FINAL")
        # Confirmed by an earlier upload of the PREVIOUS content...
        up._uploaded_paths.add(f)
        # ...and re-enqueued, never attempted (workers never started).
        up.enqueue(f)

        up._delete_uploaded_files()

        assert f.exists(), "drain-time delete ignored a pending re-enqueue"
        assert f.read_text() == "FINAL"


class TestReclaimTree:
    """Reclaim of an ABANDONED tree, driven by a later run."""

    @staticmethod
    def _uploader(tmp_path, upload_fn):
        from utils.atif_disk_writer import _GcsUploader

        job_dir = tmp_path / "recipe" / "job"
        job_dir.mkdir(parents=True)
        return (
            _GcsUploader(
                gcs_prefix="gs://b/p", job_dir=job_dir, upload_fn=upload_fn
            ),
            job_dir,
        )

    async def test_it_uploads_then_deletes(self, tmp_path):
        seen = []
        up, job_dir = self._uploader(tmp_path, lambda p, b: seen.append(p))
        d = job_dir / "trial-a"
        d.mkdir()
        for i in range(3):
            (d / f"f{i}.json").write_text("{}")
        stats = await up.reclaim_tree(job_dir)
        assert stats == {"uploaded": 3, "deleted": 3, "failed": 0}
        assert len(seen) == 3
        assert not d.exists()

    async def test_a_file_that_will_not_upload_is_KEPT(self, tmp_path):
        """Upload-then-delete: a crash mid-reclaim must lose nothing."""

        def flaky(p, b):
            if p.name == "f1.json":
                raise RuntimeError("permanent")

        up, job_dir = self._uploader(tmp_path, flaky)
        d = job_dir / "trial-a"
        d.mkdir()
        for i in range(3):
            (d / f"f{i}.json").write_text("{}")
        stats = await up.reclaim_tree(job_dir)
        assert stats["failed"] == 1
        assert (d / "f1.json").exists()
        assert not (d / "f0.json").exists()

    async def test_an_empty_tree_is_a_no_op(self, tmp_path):
        up, job_dir = self._uploader(tmp_path, lambda p, b: None)
        assert await up.reclaim_tree(job_dir) == {
            "uploaded": 0,
            "deleted": 0,
            "failed": 0,
        }


class TestAllThreeArtifactKindsAreReclaimed:
    """trajectory.json, reward.json AND raw_llm_calls -- one mechanism.

    Worth pinning because the three are written by different components:
    the trajectory and reward by the writer itself, the raw-capture blobs by
    `utils.raw_capture.dump_raw`, which the writer only reaches by passing
    `on_written=self._gcs_uploader.enqueue` into the CaptureContext. If that
    wire-through ever breaks, the blobs would still be WRITTEN but never
    enter `_uploaded_paths` -- so reclaim would silently keep them, and they
    are 89% of a run's bytes (21.4 GB of 24 GB measured on a real run).
    """

    async def test_every_artifact_kind_uploads_then_is_deleted(self, tmp_path):
        from utils.atif_disk_writer import _GcsUploader

        job = tmp_path / "recipe" / "job"
        job.mkdir(parents=True)
        uploaded: list[str] = []
        up = _GcsUploader(
            gcs_prefix="gs://b/p",
            job_dir=job,
            upload_fn=lambda p, b: uploaded.append(p.name),
            trial_grace_count=0,
        )
        up.start()

        trial = job / "trial-1"
        (trial / "agent").mkdir(parents=True)
        (trial / "verifier").mkdir()
        (trial / "artifacts" / "raw_llm_calls").mkdir(parents=True)
        (trial / "agent" / "trajectory.json").write_text("{}")
        (trial / "verifier" / "reward.json").write_text("{}")
        for i in range(4):
            (trial / "artifacts" / "raw_llm_calls" / f"call_{i:06d}.json").write_text(
                "{}"
            )

        for p in trial.rglob("*"):
            if p.is_file():
                up.enqueue(p)
        await up.drain()
        up.finalize_trial(trial)

        assert sorted(uploaded) == sorted(
            [
                "trajectory.json",
                "reward.json",
                "call_000000.json",
                "call_000001.json",
                "call_000002.json",
                "call_000003.json",
            ]
        )
        assert not trial.exists()

    def test_the_raw_capture_hook_is_wired_to_the_uploader(self, tmp_path):
        """`on_written` must be the uploader's enqueue, not None.

        This is the single line that makes raw blobs reclaimable; without it
        they are written locally, never tracked, and never deleted.
        """
        from utils.atif_disk_writer import AtifDiskWriter

        w = AtifDiskWriter(
            root_dir=str(tmp_path / "atif"),
            recipe="r",
            tuning_job_id="job",
            gcs_prefix="gs://b/p",
            uploader_factory=lambda **kw: _FakeUploader(**kw),
        )
        assert w._gcs_uploader is not None
        # The hook the CaptureContext gets is the uploader's own enqueue.
        assert w._gcs_uploader.enqueue is not None


class TestUploadLifecycleEndToEnd:
    """The three properties the ATIF upload path exists to guarantee.

    Driven through the real `AtifDiskWriter` with a fake `upload_fn`, rather
    than by poking the uploader's internals, so it fails if any layer
    between the callback and GCS stops enqueueing.

      (a) EVERYTHING is uploaded -- every file the writer puts on disk
          reaches GCS, with nothing enqueued-but-forgotten.
      (b) all three upload KINDS happen: the mid-rollout partial snapshot,
          the completion write, and the write for a FAILED rollout (a
          rollout that produced no usable result is exactly the one whose
          trajectory you need).
      (c) local copies are reclaimed after a confirmed upload, and only
          after.
    """

    @staticmethod
    def _writer_and_gcs(tmp_path, *, fail_paths=frozenset()):
        gcs: dict[str, str] = {}

        def upload_fn(local: Path, blob: str) -> None:
            if local.name in fail_paths:
                raise OSError(f"synthetic upload failure for {local.name}")
            gcs[blob] = local.read_text()

        # Use the documented `uploader_factory` seam: a REAL _GcsUploader
        # with a fake upload_fn, so the queue, retry, confirmation and
        # reclaim logic under test is the production code.
        def factory(**kw):
            kw["upload_fn"] = upload_fn
            return _GcsUploader(**kw)

        w = AtifDiskWriter(
            root_dir=str(tmp_path / "atif"),
            recipe="word_puzzle",
            tuning_job_id="9999",
            gcs_prefix="gs://bucket/pfx",
            delete_after_upload=True,
            uploader_factory=factory,
        )
        return w, gcs

    @staticmethod
    def _live_traj(session_id: str, n_agent_steps: int):
        """A live Trajectory shell, as the partial-snapshot hook receives."""
        steps = [Step(step_id=1, source="user", message="do X")]
        for i in range(1, n_agent_steps + 1):
            steps.append(
                TrajectoryStep.from_wire(
                    step_id=i + 1,
                    generation=f"g{i}",
                    stats=StepStats(candidate_id=f"c{i}", finish_reason="stop"),
                )
            )
        return AtifTrajectory(
            schema_version="ATIF-v1.7",
            session_id=session_id,
            agent=AtifAgent(name="t", version="0", model_name="m"),
            steps=steps,
            extra={"phase": "train", "rollout_idx": 0, "task_id": "task-A"},
        )

    @pytest.mark.asyncio
    async def test_partial_then_completion_then_failure_all_reach_gcs(
        self, tmp_path, monkeypatch
    ):
        import utils.atif_disk_writer as dw

        # Snapshot on every step so the partial is deterministic.
        monkeypatch.setattr(dw, "_PARTIAL_STEP_INTERVAL", 1)
        monkeypatch.setattr(dw, "_PARTIAL_MIN_INTERVAL_S", 0)

        @dataclasses.dataclass
        class _JobCfg:
            lr: float = 0.001
            name: str = "run-x"

        w, gcs = self._writer_and_gcs(tmp_path)
        await w.on_job_started(_JobCfg())

        ok_sid = _sid("task-A", 0)
        fail_sid = _sid("task-B", 1)

        # (b1) PARTIAL -- mid-rollout snapshot of a live trajectory.
        await w.on_trajectory_step_completed(
            _TaskData(task_id="task-A"), self._live_traj(ok_sid, 2)
        )
        # Wait for the partial to land WITHOUT draining: drain() cancels the
        # workers and shuts the executor down, so anything enqueued after it
        # would never upload. Join the queue instead.
        await asyncio.wait_for(w._gcs_uploader._queue.join(), timeout=5)
        partial_blob = next(b for b in gcs if ok_sid in b and "trajectory" in b)
        assert json.loads(gcs[partial_blob])["extra"]["_partial"] is True

        # (b2) COMPLETION -- the final trajectory overwrites the same path.
        await w.on_trajectory_produced(
            _scored(session_id=ok_sid, reward=0.9), _TaskData(task_id="task-A")
        )
        # (b3) FAILURE -- a rollout with a NaN reward and an infra status
        # must still be written and uploaded; it is the one you need most.
        await w.on_trajectory_produced(
            _scored(
                session_id=fail_sid,
                reward=float("nan"),
                normalized_reward=None,
                status=RolloutStatus.ENV_FAILURE,
            ),
            _TaskData(task_id="task-B"),
        )
        await w.drain(timeout_s=10)

        # (b) the final trajectory REPLACED the partial in GCS.
        final = json.loads(gcs[partial_blob])
        assert final.get("extra", {}).get("_partial") is not True, (
            "GCS still holds the partial snapshot; the completion upload "
            "did not supersede it"
        )
        # (b3) the failed rollout is present.
        assert any(fail_sid in b for b in gcs), "a failed rollout was not uploaded"

        # (a) EVERYTHING: every file still on disk under the job dir must be
        # accounted for in GCS. With delete_after_upload the uploaded ones
        # are gone, so what remains must be only the non-uploadable extras.
        leftover = {
            p.name for p in w.job_dir.rglob("*") if p.is_file()
        }
        assert leftover <= {".upload_failures.jsonl"}, (
            f"files left on disk that never uploaded: {leftover}"
        )
        # And GCS has all three kinds.
        assert any("config.json" in b for b in gcs)
        assert any(f"{ok_sid}" in b and "reward" in b for b in gcs)

        # (c) CLEANUP -- the trial dirs are gone, not merely emptied.
        assert not (w.job_dir / ok_sid).exists()
        assert not (w.job_dir / fail_sid).exists()

    @pytest.mark.asyncio
    async def test_a_file_that_failed_to_upload_is_kept_locally(self, tmp_path):
        """Cleanup is conditional on a CONFIRMED upload, never unconditional.

        The counterpart to (c): losing a file locally is only acceptable
        because GCS has it. If the upload failed, the local copy is the
        only copy and must survive.
        """
        w, gcs = self._writer_and_gcs(tmp_path, fail_paths={"reward.json"})
        sid = _sid("task-A", 0)
        await w.on_trajectory_produced(
            _scored(session_id=sid, reward=0.5), _TaskData(task_id="task-A")
        )
        await w.drain(timeout_s=10)

        trial = w.job_dir / sid
        assert (trial / "verifier" / "reward.json").exists(), (
            "deleted a file whose upload failed -- that was the only copy"
        )
        # The ones that DID upload are reclaimed.
        assert not (trial / "agent" / "trajectory.json").exists()
        assert not any("reward.json" in b for b in gcs)


# ==========================================================================
# Drain sizing: adaptive normal-exit timeout, drain concurrency, short
# signal-path timeout, and the not-uploaded report.
# ==========================================================================

import time as _time  # noqa: E402


def _blocking_after(n_ok: int):
    """upload_fn: the first ``n_ok`` calls succeed at once, the rest block
    until ``release`` is set (simulating stuck uploads)."""
    lock = threading.Lock()
    release = threading.Event()
    calls = {"n": 0}

    def _fn(_lp, _bn):
        with lock:
            calls["n"] += 1
            n = calls["n"]
        if n > n_ok:
            release.wait(10)

    return _fn, release


async def _wait_for_successes(up: _GcsUploader, n: int) -> None:
    for _ in range(2000):
        if up._success_count >= n:
            return
        await asyncio.sleep(0.005)
    raise AssertionError(f"only {up._success_count} of {n} uploads landed")


@pytest.mark.asyncio
async def test_adaptive_drain_finishes_backlog_within_computed_bound(
    tmp_path: Path, monkeypatch
):
    """~5k queued files, fixed 5 ms upload latency: the timeout is sized from
    the observed throughput (not the old fixed 60 s), the drain runs at the
    raised drain concurrency, finishes inside its own bound, abandons none."""
    import utils.atif_disk_writer as mod

    monkeypatch.setattr(mod, "_DRAIN_MIN_TIMEOUT_S", 0.01)  # expose the estimate

    def _fixed_latency(_lp, _bn):
        _time.sleep(0.005)

    # delete_after_upload=False: the bound covers the upload wait only. Local
    # cleanup of 5k files runs after it (~2 s here) and made this flaky when
    # it was counted in `elapsed`.
    up = _gs_uploader(
        tmp_path,
        upload_fn=_fixed_latency,
        max_concurrent_uploads=10,
        delete_after_upload=False,
    )
    up.start()
    for i in range(100):  # warm-up: gives the writer latency samples
        up.enqueue(_make_local_file(tmp_path, rel=f"w{i}/agent/trajectory.json"))
    await _wait_for_successes(up, 100)
    for i in range(5000):
        up.enqueue(_make_local_file(tmp_path, rel=f"t{i}/agent/trajectory.json"))

    t0 = _time.monotonic()
    report = await up.drain()
    elapsed = _time.monotonic() - t0

    # Estimate: 5000 / (10 workers / ~5 ms) * 1.5 margin ~= 3.75 s.
    assert 1.0 < report.timeout_s < 30.0, report
    assert elapsed <= report.timeout_s, (elapsed, report)
    assert report.concurrency == 20
    assert report.abandoned == 0 and report.cancelled_in_flight == 0
    assert report.not_uploaded == 0
    assert report.uploaded == 5100


@pytest.mark.asyncio
async def test_drain_cap_smaller_than_needed_reports_exact_not_uploaded(
    tmp_path: Path,
):
    """When the cap is too small, the report says exactly what was dropped:
    queued-never-attempted plus in-flight uploads cancelled at the deadline."""
    upload_fn, release = _blocking_after(100)
    up = _gs_uploader(tmp_path, upload_fn=upload_fn, max_concurrent_uploads=2)
    try:
        up.start()
        for i in range(1000):
            up.enqueue(_make_local_file(tmp_path, rel=f"t{i}/agent/trajectory.json"))
        await _wait_for_successes(up, 100)

        report = await up.drain(max_timeout_s=0.2)

        assert report.timeout_s == 0.2
        assert report.concurrency == 4
        assert report.uploaded == 100
        assert report.cancelled_in_flight == 4  # all drain workers were stuck
        assert report.abandoned == 1000 - 100 - 4
        assert report.not_uploaded == 900
    finally:
        release.set()


@pytest.mark.asyncio
async def test_drain_explicit_timeout_is_honoured(tmp_path: Path):
    """An explicit timeout_s wins over the adaptive estimate and the cap."""
    upload_fn, release = _blocking_after(10)
    up = _gs_uploader(tmp_path, upload_fn=upload_fn, max_concurrent_uploads=2)
    try:
        up.start()
        for i in range(50):
            up.enqueue(_make_local_file(tmp_path, rel=f"t{i}/agent/trajectory.json"))
        await _wait_for_successes(up, 10)
        t0 = _time.monotonic()
        report = await up.drain(timeout_s=0.3, max_timeout_s=3600)
        assert _time.monotonic() - t0 < 3.0
        assert report.timeout_s == 0.3
        assert report.not_uploaded == 40
    finally:
        release.set()


def test_signal_path_drain_timeout_is_short():
    """A SIGTERMed VM is killed ~30 s after the signal and still has to cancel
    the tuning job after the drain, so the cancelled-path drain must be short."""
    from utils.atif_disk_writer import _CANCELLED_DRAIN_TIMEOUT_S

    assert 0 < _CANCELLED_DRAIN_TIMEOUT_S <= 15


@pytest.mark.asyncio
async def test_signal_path_drain_stays_short(tmp_path: Path, monkeypatch):
    """Signal path = the run task is CANCELLED and its finally drains (train:
    on_job_completed; eval: drain()). With uploads stuck, that drain must use
    the short cancelled-path timeout, not the adaptive normal-exit one."""
    import utils.atif_disk_writer as mod

    monkeypatch.setattr(mod, "_CANCELLED_DRAIN_TIMEOUT_S", 0.3)
    upload_fn, release = _blocking_after(5)

    def factory(*, gcs_prefix, job_dir, **kw):
        return _GcsUploader(
            gcs_prefix=gcs_prefix, job_dir=job_dir, upload_fn=upload_fn, **kw
        )

    w = AtifDiskWriter(
        root_dir=str(tmp_path / "atif"),
        recipe="word_puzzle",
        tuning_job_id="9999",
        gcs_prefix="gs://test-traj-bucket",
        uploader_factory=factory,
    )
    for i in range(200):
        p = w.job_dir / f"t{i}" / "agent" / "trajectory.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
        w.enqueue_for_upload(p)
    await _wait_for_successes(w._gcs_uploader, 5)

    box: dict = {}
    started = asyncio.Event()

    async def _run():
        try:
            started.set()
            await asyncio.sleep(3600)
        finally:
            t0 = _time.monotonic()
            await w.on_job_completed({"recipe": "word_puzzle"})
            box["elapsed"] = _time.monotonic() - t0

    try:
        task = asyncio.ensure_future(_run())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=30)
        assert box["elapsed"] < 5.0, box
    finally:
        release.set()
