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

"""Tests for the offline trajectory analysis tool."""

from __future__ import annotations

import json
import math

import pytest
from google.api_core import exceptions as gax

from core.metric_aggregators import DEFAULT_METRIC_AGGREGATORS
from core.metric_spec import (
    METRIC_SPECS,
    AggregationStrategy,
    MetricSpec,
    MetricValue,
    aggregate_by_spec,
    merge_group_metrics,
)
from tools.trajectory_analysis import analyze_run, analyze_sxs
from tools.trajectory_analysis.loader import (
    TrialLoadError,
    discover_trials,
    load_trial,
    parse_trial_name,
)
from tools.trajectory_analysis.metrics import SXS_METRICS, outcome
from tools.trajectory_analysis.pairing import (
    pair_by_fanout,
    pair_by_task,
    pair_explicit,
)
from core.sinks import MLPlatformSink
from tools.trajectory_analysis.runner import (
    aggregate_pairs,
    aggregate_views,
    delta_key_for,
    group_by_task,
    load_views,
    select_refs,
)
from tools.trajectory_analysis.tree_reader import (
    GcsReader,
    LocalReader,
    TreeReadError,
    parse_gs_url,
)


def _write(root, rel: str, payload: dict) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload), encoding="utf-8")


def test_local_reader_reads_json_and_lists_dirs(tmp_path):
    _write(tmp_path, "trial-a/verifier/reward.json", {"reward": 1.0})
    (tmp_path / "trial-b").mkdir()

    reader = LocalReader(str(tmp_path))

    assert reader.read_json("trial-a/verifier/reward.json") == {"reward": 1.0}
    assert reader.read_json("nope.json") is None
    assert reader.exists("trial-a/verifier/reward.json") is True
    assert reader.exists("trial-a/verifier/missing.json") is False
    assert sorted(reader.list_dir("")) == ["trial-a", "trial-b"]


# --- GcsReader, exercised against a small in-memory fake client -----------


class _FakeBlobRef:
    """Stands in for ``storage.Blob``; reads from a flat name -> text dict."""

    def __init__(self, objects: dict[str, str], name: str) -> None:
        self._objects = objects
        self.name = name

    def exists(self) -> bool:
        return self.name in self._objects

    def download_as_text(self) -> str:
        try:
            return self._objects[self.name]
        except KeyError:
            raise gax.NotFound(f"No such object: {self.name}") from None


class _FakeBucket:
    def __init__(self, objects: dict[str, str]) -> None:
        self._objects = objects

    def blob(self, name: str) -> _FakeBlobRef:
        return _FakeBlobRef(self._objects, name)


class _FakeClient:
    """Fake ``storage.Client`` over a flat ``{blob_name: text}`` namespace."""

    def __init__(self, objects: dict[str, str]) -> None:
        self._objects = objects

    def bucket(self, bucket_name: str) -> _FakeBucket:
        return _FakeBucket(self._objects)

    def list_blobs(self, bucket_name, prefix="", max_results=None, delimiter=None):
        names = sorted(n for n in self._objects if n.startswith(prefix))
        if max_results is not None:
            names = names[:max_results]
        blobs = [_FakeBlobRef(self._objects, n) for n in names]
        if delimiter is None:
            return blobs
        # Mirror the real client's rollup: objects below the delimiter are
        # replaced by their common prefix, exposed on the iterator once it
        # has been drained.
        return _FakeListing(
            [b for b in blobs if delimiter not in b.name[len(prefix) :]],
            {
                prefix + n[len(prefix) :].split(delimiter, 1)[0] + delimiter
                for n in names
                if delimiter in n[len(prefix) :]
            },
        )


class _FakeListing:
    """An iterable of blobs that also carries ``prefixes``, like the real one."""

    def __init__(self, blobs: list[_FakeBlobRef], prefixes: set[str]) -> None:
        self._blobs = blobs
        self._prefixes: set[str] = set()
        self._pending = prefixes

    def __iter__(self):
        # ``prefixes`` is only populated as pages are consumed; a reader
        # that reads it before iterating must see nothing.
        self._prefixes = self._pending
        return iter(self._blobs)

    @property
    def prefixes(self) -> set[str]:
        return self._prefixes


_TREE = {
    "runs/job1/trial-a/verifier/reward.json": json.dumps({"reward": 1.0}),
    "runs/job1/trial-a/bad.json": "{not json",
    "runs/job1/trial-b/verifier/reward.json": json.dumps({"reward": 0.0}),
    # Sibling prefix that shares "runs/job1" as a string prefix but is NOT a
    # child of it.
    "runs/job1-sibling/trial-z/verifier/reward.json": json.dumps({"reward": 0.5}),
}


def _reader(
    url: str = "gs://bkt/runs/job1", objects: dict[str, str] | None = None
) -> GcsReader:
    return GcsReader(
        url, client=_FakeClient(dict(_TREE if objects is None else objects))
    )


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("gs://b", ("b", "")),
        ("gs://b/", ("b", "")),
        ("gs://b/p/q/", ("b", "p/q")),
    ],
)
def test_parse_gs_url(url, expected):
    assert parse_gs_url(url) == expected


def test_parse_gs_url_rejects_non_gs():
    with pytest.raises(ValueError):
        parse_gs_url("/local/path")


def test_gcs_reader_read_json_good_malformed_and_missing():
    reader = _reader()

    assert reader.read_json("trial-a/verifier/reward.json") == {"reward": 1.0}
    # Absent: None. Present but unparseable: an error, not an absence.
    assert reader.read_json("trial-a/nope.json") is None
    with pytest.raises(TreeReadError):
        reader.read_json("trial-a/bad.json")


def test_gcs_reader_listing_does_not_leak_sibling_prefix():
    assert _reader().list_dir("") == ["trial-a", "trial-b"]


def test_gcs_reader_list_subdirs_excludes_files_at_this_level():
    """Discovery wants candidate trials, not the run's own ``config.json``."""
    objects = dict(_TREE) | {"runs/job1/config.json": "{}"}
    reader = _reader(objects=objects)

    assert reader.list_dir("") == ["config.json", "trial-a", "trial-b"]
    assert reader.list_subdirs("") == ["trial-a", "trial-b"]


def test_gcs_reader_normalises_dot_and_slash_rel_paths():
    """Regression: an unnormalised "." or "/" lists NOTHING on flat GCS."""
    reader = _reader()
    children = reader.list_dir("")

    assert children == ["trial-a", "trial-b"]
    assert reader.list_dir(".") == children
    assert reader.list_dir("/") == children
    assert reader.read_json("./trial-a/verifier/reward.json") == {"reward": 1.0}


def test_gcs_reader_exists_covers_objects_and_directories():
    reader = _reader()

    assert reader.exists("trial-a/verifier/reward.json") is True
    # A directory is not an object, but must still report True so the two
    # backends agree.
    assert reader.exists("trial-a") is True
    assert reader.exists("trial-a/verifier") is True
    assert reader.exists("trial-nope") is False
    assert reader.exists("trial-a/verifier/missing.json") is False


def test_gcs_reader_distinguishes_a_failed_read_from_an_absent_object():
    """A 503 is not a missing file.

    Collapsing both to None made a transient GCS error look like a trial
    with no trajectory.json, so a flaky read silently shrank the analyzed
    population under a message naming a file that exists.
    """

    class _BoomBucket(_FakeBucket):
        def blob(self, name):
            raise gax.ServiceUnavailable("backend down")

    reader = _reader()
    assert reader.read_text("trial-a/nope.json") is None  # absent
    reader._bucket = _BoomBucket({})  # noqa: SLF001 - exercising the error path

    with pytest.raises(TreeReadError):
        reader.read_text("trial-a/verifier/reward.json")


# --- discovery + loading --------------------------------------------------


def _trajectory_payload(task_id: str, rollout_idx: int, n_steps: int = 2) -> dict:
    """A minimal ATIF v1.7 trajectory dict for one rollout.

    Field names follow the real ``harbor.models.trajectories`` models:
    ``agent`` is required, and a step is ``step_id`` + ``source`` +
    ``message`` (``role``/``content`` are rejected by ``extra=forbid``).
    """
    return {
        "schema_version": "ATIF-v1.7",
        "session_id": f"{task_id}-{rollout_idx}",
        "agent": {"name": "test-agent", "version": "0"},
        "steps": [
            {
                "step_id": i + 1,
                "source": "agent",
                "message": f"step {i}",
                "extra": {"finish_reason": "STOP"},
            }
            for i in range(n_steps)
        ],
        "extra": {
            "task_id": task_id,
            "rollout_idx": rollout_idx,
            "phase": "eval",
            "is_correct": rollout_idx == 0,
        },
    }


def _make_job_tree(root, trials: list[tuple[str, int, float]]) -> None:
    """Write a job tree: one (task_id, rollout_idx, reward) per trial."""
    for task_id, rollout_idx, reward in trials:
        d = f"{task_id}-{rollout_idx}"
        _write(
            root,
            f"{d}/agent/trajectory.json",
            _trajectory_payload(task_id, rollout_idx),
        )
        _write(root, f"{d}/verifier/reward.json", {"reward": reward})
        _write(root, f"{d}/result.json", {"trial_name": d})


def test_discover_trials_finds_every_trial_dir(tmp_path):
    _make_job_tree(tmp_path, [("t1", 0, 1.0), ("t1", 1, 0.0), ("t2", 0, 1.0)])
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    refs = discover_trials(str(tmp_path))

    assert {r.trial for r in refs} == {"t1-0", "t1-1", "t2-0"}
    assert {(r.task_id, r.fanout_idx) for r in refs} == {
        ("t1", 0),
        ("t1", 1),
        ("t2", 0),
    }


class _CountingReader(LocalReader):
    """A LocalReader that counts the reads it is asked to perform."""

    def __init__(self, root: str) -> None:
        super().__init__(root)
        self.reads = 0
        self.listings = 0

    def read_text(self, rel_path: str) -> str | None:
        self.reads += 1
        return super().read_text(rel_path)

    def read_json(self, rel_path: str):
        self.reads += 1
        return super().read_json(rel_path)

    def list_dir(self, rel_path: str) -> list[str]:
        self.listings += 1
        return super().list_dir(rel_path)

    def list_subdirs(self, rel_path: str) -> list[str]:
        self.listings += 1
        return super().list_subdirs(rel_path)


def test_discover_trials_reads_no_trial_files(tmp_path):
    """Regression guard: discovery is ONE listing, never N object reads.

    The tool was unusable on a real 3,901-trial tree because discovery
    fetched every trial's trajectory.json just to recover its identity --
    before ``--limit`` had a chance to discard 3,881 of them. Identity is
    now guessed from the directory name and corrected at load time.
    """
    _make_job_tree(tmp_path, [("t1", 0, 1.0), ("t1", 1, 0.0), ("t2", 0, 1.0)])
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    reader = _CountingReader(str(tmp_path))

    refs = discover_trials(str(tmp_path), reader=reader)

    assert {r.trial for r in refs} == {"t1-0", "t1-1", "t2-0"}
    assert reader.reads == 0
    assert reader.listings == 1


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("t1-0", ("t1", 0)),
        ("t1-12", ("t1", 12)),
        # No trailing integer: the whole name is the task id.
        ("session_abc", ("session_abc", 0)),
        ("t1-unknown-9ab3f0c1", ("t1-unknown-9ab3f0c1", 0)),
        # Ambiguous by construction, and documented as such: a task id
        # ending in "-<digits>" is indistinguishable from task+rollout.
        ("my-task-3", ("my-task", 3)),
        ("-3", ("-3", 0)),
        ("", ("", 0)),
    ],
)
def test_parse_trial_name(name, expected):
    assert parse_trial_name(name) == expected


def _write_trial(root, dirname: str, task_id: str, rollout_idx: int, reward: float):
    """Write one trial into ``dirname``, which need not match its identity."""
    _write(
        root,
        f"{dirname}/agent/trajectory.json",
        _trajectory_payload(task_id, rollout_idx),
    )
    _write(root, f"{dirname}/verifier/reward.json", {"reward": reward})
    _write(root, f"{dirname}/result.json", {"trial_name": dirname})


def test_load_trial_overrides_a_lying_directory_name(tmp_path):
    """The name is a convention; ``trajectory.extra`` is the contract."""
    _write_trial(tmp_path, "mislabelled-9", "real", 2, 1.0)
    ref = discover_trials(str(tmp_path))[0]
    assert (ref.task_id, ref.fanout_idx) == ("mislabelled", 9)  # the guess

    view = load_trial(ref)

    assert view.ref.task_id == "real"
    assert view.ref.fanout_idx == 2
    # The path is not the identity and must survive the correction.
    assert view.ref.trial == "mislabelled-9"


def test_load_trial_keeps_the_guess_when_the_trajectory_is_silent(tmp_path):
    payload = _trajectory_payload("ignored", 0)
    payload["extra"] = {"phase": "eval"}
    _write(tmp_path, "t7-4/agent/trajectory.json", payload)
    _write(tmp_path, "t7-4/verifier/reward.json", {"reward": 1.0})

    view = load_trial(discover_trials(str(tmp_path))[0])

    assert (view.ref.task_id, view.ref.fanout_idx) == ("t7", 4)


def test_load_views_preserves_input_order_and_records_skips(tmp_path):
    """Concurrency must not reorder results: pairing and rows read this order."""
    trials = [(f"t{i}", 0, i / 10) for i in range(10)]
    _make_job_tree(tmp_path, trials)
    for broken in ("t3-0", "t7-0"):
        (tmp_path / broken / "verifier" / "reward.json").write_text(
            "{}", encoding="utf-8"
        )
    refs = discover_trials(str(tmp_path))
    skipped: list[str] = []

    views = load_views(refs, skipped, max_workers=8)

    survivors = [r.trial for r in refs if r.trial not in {"t3-0", "t7-0"}]
    assert [v.ref.trial for v in views] == survivors
    assert len(skipped) == 2
    assert all(any(bad in s for bad in ("t3-0", "t7-0")) for s in skipped)


def test_load_trial_returns_typed_trajectory_and_reward(tmp_path):
    _make_job_tree(tmp_path, [("t1", 0, 0.75)])
    ref = discover_trials(str(tmp_path))[0]

    view = load_trial(ref)

    assert view.scored.reward.reward == 0.75
    assert len(view.scored.trajectory.steps) == 2
    assert view.result["trial_name"] == "t1-0"


def test_load_trial_raises_for_missing_trajectory(tmp_path):
    _make_job_tree(tmp_path, [("t1", 0, 1.0)])
    ref = discover_trials(str(tmp_path))[0]
    (tmp_path / "t1-0" / "agent" / "trajectory.json").unlink()

    with pytest.raises(TrialLoadError):
        load_trial(ref)


# --- aggregation ----------------------------------------------------------


def test_aggregate_views_uses_hand_computed_group_semantics(tmp_path):
    """Grouping is load-bearing: these numbers differ if trials are fed flat.

    Fixture: task t1 has two rollouts (reward 1.0 is_correct=True, reward 0.0
    is_correct=False); task t2 has one (reward 0.5, is_correct=True).

      learning/reward/mean       WEIGHTED_MEAN over groups, weight = group size
                                 = (0.5*2 + 0.5*1) / 3 = 0.5
      learning/sampling_accuracy MEAN over groups of mean(is_correct)
                                 = (0.5 + 1.0) / 2 = 0.75   [flat would be 0.667]
      learning/pass_at_k         MEAN over groups of any(is_correct)
                                 = (1.0 + 1.0) / 2 = 1.0    [flat would be 0.667]
    """
    _make_job_tree(tmp_path, [("t1", 0, 1.0), ("t1", 1, 0.0), ("t2", 0, 0.5)])
    views = [load_trial(r) for r in discover_trials(str(tmp_path))]

    got = aggregate_views(views)

    assert got["learning/reward/mean"] == pytest.approx(0.5)
    assert got["learning/sampling_accuracy"] == pytest.approx(0.75)
    assert got["learning/pass_at_k"] == pytest.approx(1.0)


def test_group_by_task_groups_rollouts_of_the_same_task(tmp_path):
    _make_job_tree(tmp_path, [("t1", 0, 1.0), ("t1", 1, 0.0), ("t2", 0, 1.0)])
    views = [load_trial(r) for r in discover_trials(str(tmp_path))]

    groups = group_by_task(views)

    assert sorted(len(g) for g in groups) == [1, 2]


def test_aggregate_views_matches_the_pipeline_reduction(tmp_path):
    """Same aggregators, same per-group merge, same reduce as the eval loop."""
    _make_job_tree(tmp_path, [("t1", 0, 1.0), ("t1", 1, 0.0), ("t2", 0, 0.5)])
    views = [load_trial(r) for r in discover_trials(str(tmp_path))]

    got = aggregate_views(views)

    # Reference: mirror of Agent._compute_group_metrics + eval_only's reduce.
    expected_per_group = []
    for group in group_by_task(views):
        outputs = []
        for agg in DEFAULT_METRIC_AGGREGATORS:
            raw = agg([v.scored for v in group])
            outputs.append(
                {
                    str(k): v
                    if isinstance(v, MetricValue)
                    else MetricValue.of_scalar(float(v))
                    for k, v in raw.items()
                }
            )
        expected_per_group.append(merge_group_metrics(outputs))
    assert got == aggregate_by_spec(expected_per_group)


def test_aggregate_views_is_empty_for_no_views():
    assert aggregate_views([]) == {}


# --- pairing --------------------------------------------------------------


def test_pair_by_task_matches_on_task_id_and_reports_unmatched(tmp_path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    _make_job_tree(root_a, [("t1", 0, 1.0), ("t2", 0, 0.0)])
    _make_job_tree(root_b, [("t1", 0, 0.5), ("t3", 0, 1.0)])

    result = pair_by_task(discover_trials(str(root_a)), discover_trials(str(root_b)))

    assert [(a.task_id, b.task_id) for a, b in result.pairs] == [("t1", "t1")]
    assert [r.task_id for r in result.missing_in_b] == ["t2"]
    assert [r.task_id for r in result.missing_in_a] == ["t3"]


def test_pair_by_fanout_pairs_rollouts_within_one_task(tmp_path):
    _make_job_tree(tmp_path, [("t1", 0, 1.0), ("t1", 1, 0.0), ("t2", 0, 1.0)])

    result = pair_by_fanout(discover_trials(str(tmp_path)))

    assert [(a.trial, b.trial) for a, b in result.pairs] == [("t1-0", "t1-1")]
    # t2 has a single rollout: nothing to pair it with.
    assert [r.trial for r in result.unpaired] == ["t2-0"]


def test_pair_explicit_returns_just_the_named_pair(tmp_path):
    _make_job_tree(tmp_path, [("t1", 0, 1.0), ("t2", 0, 0.0)])
    ref_a, ref_b = discover_trials(str(tmp_path))

    result = pair_explicit(ref_a, ref_b)

    assert result.pairs == [(ref_a, ref_b)]
    assert result.missing_in_a == []
    assert result.missing_in_b == []
    assert result.unpaired == []


# --- SxS metrics ----------------------------------------------------------


def test_outcome_scores_win_lose_tie(tmp_path):
    _make_job_tree(tmp_path, [("t1", 0, 1.0), ("t2", 0, 0.0), ("t3", 0, 0.5)])
    views = {
        v.ref.task_id: v
        for v in (load_trial(r) for r in discover_trials(str(tmp_path)))
    }

    win = outcome(views["t2"], views["t1"])  # b better than a
    lose = outcome(views["t1"], views["t2"])  # a better than b
    tie = outcome(views["t3"], views["t3"])

    assert win["sxs/outcome/win_rate"].value == 1.0
    assert lose["sxs/outcome/win_rate"].value == 0.0
    assert tie["sxs/outcome/tie_rate"].value == 1.0


def test_aggregate_pairs_reduces_sxs_metrics_and_deltas(tmp_path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    _make_job_tree(root_a, [("t1", 0, 0.0), ("t2", 0, 1.0)])
    _make_job_tree(root_b, [("t1", 0, 1.0), ("t2", 0, 1.0)])
    views_a = [load_trial(r) for r in discover_trials(str(root_a))]
    views_b = [load_trial(r) for r in discover_trials(str(root_b))]
    pairs = list(zip(views_a, views_b, strict=True))

    got = aggregate_pairs(pairs)

    # one win (t1: 0.0 -> 1.0), one tie (t2: 1.0 -> 1.0)
    assert got["sxs/outcome/win_rate"] == 0.5
    assert got["sxs/outcome/tie_rate"] == 0.5
    assert got["sxs/delta/learning_reward_mean"] == 0.5


def test_aggregate_pairs_is_empty_for_no_pairs():
    assert aggregate_pairs([]) == {}


def test_delta_metrics_survive_the_ml_platform_sink(tmp_path):
    """A delta the experiment tracker drops is a delta nobody sees.

    ``MLPlatformSink`` rejects a metric unless it has a registered spec with
    ``emit_to_ml_platform`` AND at most four "/"-separated segments -- and it
    rejects rather than truncating. That cap is why the delta key flattens its
    source key instead of the naive ``sxs/<source key>/delta``, which is five
    segments for ``learning/reward/mean``.
    """
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    _make_job_tree(root_a, [("t1", 0, 0.0)])
    _make_job_tree(root_b, [("t1", 0, 1.0)])
    pairs = [
        (
            load_trial(discover_trials(str(root_a))[0]),
            load_trial(discover_trials(str(root_b))[0]),
        )
    ]

    got = aggregate_pairs(pairs)

    key = delta_key_for("learning/reward/mean")
    assert key in got
    assert key.count("/") + 1 <= 4
    spec = METRIC_SPECS[key]
    assert spec.emit_to_ml_platform is True
    assert spec.strategy is AggregationStrategy.PASSTHROUGH
    assert MLPlatformSink(backend=object()).should_accept(key, got[key], spec) is True


def test_sxs_registry_contains_outcome():
    assert "outcome" in SXS_METRICS


# --- public API -----------------------------------------------------------


def test_analyze_run_returns_metrics_and_per_trial_rows(tmp_path):
    _make_job_tree(tmp_path, [("t1", 0, 1.0), ("t1", 1, 0.0), ("t2", 0, 0.5)])

    result = analyze_run(str(tmp_path))

    assert result.n_trials == 3
    assert result.skipped == []
    assert "learning/reward/mean" in result.metrics
    assert {row["trial"] for row in result.rows} == {"t1-0", "t1-1", "t2-0"}
    assert result.to_json()["metrics"] == result.metrics


def test_analyze_run_retains_the_loaded_views(tmp_path):
    """Callers that need the trials themselves must not have to reload them."""
    _make_job_tree(tmp_path, [("t1", 0, 1.0), ("t2", 0, 0.5)])

    result = analyze_run(str(tmp_path))

    assert [v.ref.trial for v in result.views] == ["t1-0", "t2-0"]
    assert "views" not in result.to_json()


def test_analyze_sxs_retains_both_sides_of_every_pair(tmp_path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    _make_job_tree(root_a, [("t1", 0, 0.0)])
    _make_job_tree(root_b, [("t1", 0, 1.0)])

    result = analyze_sxs(str(root_a), str(root_b))

    assert [v.reward for v in result.views] == [0.0, 1.0]


def test_analyze_run_skips_unloadable_trials_without_aborting(tmp_path):
    _make_job_tree(tmp_path, [("t1", 0, 1.0), ("t2", 0, 0.5)])
    (tmp_path / "t2-0" / "verifier" / "reward.json").write_text("{}", encoding="utf-8")

    result = analyze_run(str(tmp_path))

    assert result.n_trials == 1
    assert len(result.skipped) == 1
    assert "t2-0" in result.skipped[0]


def test_analyze_sxs_pairs_two_runs_by_task(tmp_path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    _make_job_tree(root_a, [("t1", 0, 0.0), ("t2", 0, 1.0)])
    _make_job_tree(root_b, [("t1", 0, 1.0), ("t3", 0, 1.0)])

    result = analyze_sxs(str(root_a), str(root_b))

    assert result.n_pairs == 1
    assert result.metrics["sxs/outcome/win_rate"] == 1.0
    assert result.unmatched["missing_in_b"] == ["t2-0"]
    assert result.unmatched["missing_in_a"] == ["t3-0"]


def test_analyze_sxs_pairs_on_identity_not_on_directory_name(tmp_path):
    """A directory name that lies must not break the join.

    Discovery guesses identity from the name; if pairing ran on that
    guess, side A's ``mislabelled-9`` would never meet side B's ``t1-0``
    even though both trajectories declare task ``t1`` rollout 0. Pairing
    therefore runs on loaded views, whose refs are authoritative.
    """
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    _write_trial(root_a, "mislabelled-9", "t1", 0, 0.0)
    _make_job_tree(root_b, [("t1", 0, 1.0)])

    result = analyze_sxs(str(root_a), str(root_b))

    assert result.n_pairs == 1
    assert result.rows[0]["task_id"] == "t1"
    assert result.rows[0]["trial_a"] == "mislabelled-9"
    assert result.metrics["sxs/outcome/win_rate"] == 1.0
    assert result.unmatched["missing_in_b"] == []
    assert result.unmatched["missing_in_a"] == []


def test_analyze_sxs_limit_caps_the_trials_loaded_per_side(tmp_path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    trials = [(f"t{i}", 0, 0.0) for i in range(5)]
    _make_job_tree(root_a, trials)
    _make_job_tree(root_b, trials)

    result = analyze_sxs(str(root_a), str(root_b), limit=2)

    assert result.n_pairs == 2
    assert len(result.views) == 4


def test_analyze_sxs_requires_a_second_root_for_task_strategy(tmp_path):
    _make_job_tree(tmp_path, [("t1", 0, 1.0)])

    with pytest.raises(ValueError, match="second root"):
        analyze_sxs(str(tmp_path))


def test_analysis_result_render_lists_metrics_and_unmatched(tmp_path):
    _make_job_tree(tmp_path, [("t1", 0, 1.0), ("t1", 1, 0.0), ("t2", 0, 0.5)])

    rendered = analyze_sxs(str(tmp_path), strategy="fanout").render()

    assert "sxs/outcome/win_rate" in rendered
    assert "unpaired (1): t2-0" in rendered


def test_sxs_specs_are_registered_with_pipeline_grade_docs():
    """Registering into METRIC_SPECS means meeting its bar, not evading it.

    ``aggregate_by_spec`` silently falls back to MEAN for unregistered
    keys, so the registration is what makes the reduction intentional --
    and the 20-word doc bar from ``tests/unit/test_metric_spec.py``
    applies to these keys too, whatever the import order happens to be.
    """
    for name in ("win_rate", "lose_rate", "tie_rate"):
        spec = METRIC_SPECS[f"sxs/outcome/{name}"]
        assert spec.strategy == AggregationStrategy.MEAN
        assert spec.emit_to_ml_platform is True
        assert len(spec.doc.split()) >= 20


# --- recipe stats across the disk boundary ---------------------------------


class _RecipeStyleAggregator:
    """Stand-in for a recipe aggregator (e.g. word_puzzle's SubmitRetry).

    Deliberately reads ``_task_step_stats`` the way every real recipe
    aggregator does, so this test breaks if the offline path stops
    presenting the in-memory spelling.
    """

    KEY = "recipe/fake/retries_mean"

    def __call__(self, trajectories):
        vals = []
        for scored in trajectories:
            stats = (scored.trajectory.extra or {}).get("_task_step_stats") or []
            last = next(
                (s for s in reversed(stats) if isinstance(s, dict) and "retries" in s),
                None,
            )
            vals.append(float(last["retries"]) if last else 0.0)
        if not vals:
            return {}
        return {self.KEY: MetricValue.of_scalar(sum(vals) / len(vals))}

    def specs(self):
        return [
            MetricSpec(
                self.KEY,
                AggregationStrategy.MEAN,
                "Fake recipe metric used only by tests to prove that per-step recipe "
                "stats survive the trip through disk and back into an aggregator.",
            )
        ]


def _scored_with_step_stats(tmp_path, retries: int):
    """Write one trial via the REAL codec, so the rename actually happens."""
    from core.atif_codec import to_disk
    from core.atif_trajectory import Reward, ScoredTrajectory
    from harbor.models.trajectories import Trajectory

    traj = Trajectory.model_validate(_trajectory_payload("t1", 0))
    # The in-memory spelling the pipeline uses; to_disk renames it.
    (traj.extra or {})["_task_step_stats"] = [{"retries": retries}]
    scored = ScoredTrajectory(trajectory=traj, reward=Reward(reward=1.0))

    traj_json, reward_json = to_disk(scored)
    d = tmp_path / "t1-0"
    (d / "agent").mkdir(parents=True, exist_ok=True)
    (d / "verifier").mkdir(parents=True, exist_ok=True)
    (d / "agent/trajectory.json").write_text(traj_json, encoding="utf-8")
    (d / "verifier/reward.json").write_text(reward_json, encoding="utf-8")
    return scored


def test_codec_renames_the_recipe_stats_key_on_the_way_to_disk(tmp_path):
    """Pin the behaviour this whole fix exists for."""
    _scored_with_step_stats(tmp_path, retries=3)

    on_disk = json.loads(
        (tmp_path / "t1-0/agent/trajectory.json").read_text(encoding="utf-8")
    )

    assert "task_step_stats" in on_disk["extra"]
    assert "_task_step_stats" not in on_disk["extra"]


def test_recipe_aggregator_gives_the_same_number_offline_as_online(tmp_path):
    """The guarantee: a recipe metric must not change when replayed offline.

    Before the loader re-aliased the key, the offline side silently scored
    0.0 here -- the aggregator found no stats and emitted a plausible,
    wrong number rather than failing.
    """
    online_scored = _scored_with_step_stats(tmp_path, retries=3)
    agg = _RecipeStyleAggregator()

    online = aggregate_by_spec([agg([online_scored])])

    views = [load_trial(r) for r in discover_trials(str(tmp_path))]
    offline = aggregate_views(views, aggregators=[agg])

    assert online[agg.KEY] == pytest.approx(3.0)
    assert offline[agg.KEY] == pytest.approx(online[agg.KEY])


def test_get_step_task_stats_works_on_a_loaded_trial(tmp_path):
    """The typed accessor reads the underscore key too, so it must see them."""
    from core.atif_extras import get_step_task_stats

    _scored_with_step_stats(tmp_path, retries=2)

    view = load_trial(discover_trials(str(tmp_path))[0])

    assert get_step_task_stats(view.trajectory, 0) == {"retries": 2}


# --- unscored (failed) trials --------------------------------------------


def test_load_trial_reads_a_null_reward_as_an_unscored_nan(tmp_path):
    """A failed rollout's reward is NaN in memory and null on disk.

    ``from_disk`` restores the NaN; reading it back as None made every
    ``reward > 0`` in the aggregators raise TypeError, which is what took
    the tool down on any real run containing a sampling failure.
    """
    _make_job_tree(tmp_path, [("t1", 0, None), ("t2", 0, 1.0)])
    views = {
        v.ref.task_id: v
        for v in (load_trial(r) for r in discover_trials(str(tmp_path)))
    }

    assert math.isnan(views["t1"].reward)
    assert views["t1"].is_scored is False
    assert views["t2"].is_scored is True


def test_aggregate_views_tolerates_an_unscored_trial(tmp_path):
    """Single-run aggregation keeps failed rollouts: the pipeline's own
    aggregators see them too (nanmean, and status counters that exist
    precisely to report the failures)."""
    _make_job_tree(tmp_path, [("t1", 0, None), ("t1", 1, 1.0)])
    views = [load_trial(r) for r in discover_trials(str(tmp_path))]

    got = aggregate_views(views)

    assert got["learning/sampling_accuracy"] == 0.5


def test_aggregate_pairs_drops_a_pair_with_an_unscored_side(tmp_path):
    """NaN loses every comparison, so an unscored pair would score a TIE."""
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    _make_job_tree(root_a, [("t1", 0, 0.0), ("t2", 0, 0.0)])
    _make_job_tree(root_b, [("t1", 0, 1.0), ("t2", 0, None)])
    views_a = [load_trial(r) for r in discover_trials(str(root_a))]
    views_b = [load_trial(r) for r in discover_trials(str(root_b))]

    got = aggregate_pairs(list(zip(views_a, views_b, strict=True)))

    # Only t1 is comparable: one win, no ties. Counting t2 as a tie would
    # halve win_rate and invent a 0.5 tie_rate out of a failed rollout.
    assert got["sxs/outcome/win_rate"] == 1.0
    assert got["sxs/outcome/tie_rate"] == 0.0


def test_aggregate_pairs_is_empty_when_every_pair_is_unscored(tmp_path):
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    _make_job_tree(root_a, [("t1", 0, None)])
    _make_job_tree(root_b, [("t1", 0, 1.0)])
    views_a = [load_trial(r) for r in discover_trials(str(root_a))]
    views_b = [load_trial(r) for r in discover_trials(str(root_b))]

    assert aggregate_pairs(list(zip(views_a, views_b, strict=True))) == {}


def test_analyze_sxs_reports_unscored_pairs_rather_than_dropping_them_silently(
    tmp_path,
):
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    _make_job_tree(root_a, [("t1", 0, 0.0), ("t2", 0, 0.0)])
    _make_job_tree(root_b, [("t1", 0, 1.0), ("t2", 0, None)])

    result = analyze_sxs(str(root_a), str(root_b))

    assert result.n_pairs == 1
    assert [r["task_id"] for r in result.rows] == ["t1"]
    assert result.unmatched["unscored"] == ["t2-0 vs t2-0"]


def test_render_truncates_a_long_unmatched_list(tmp_path):
    """Hundreds of names must not bury the metrics table above them."""
    from tools.trajectory_analysis.runner import AnalysisResult

    result = AnalysisResult(
        root="r", unmatched={"unscored": [f"t{i}-0" for i in range(20)]}
    )

    text = result.render()

    assert "unscored (20):" in text
    assert "(+12 more)" in text
    assert "t19-0" not in text


# --- selection: --limit is a head, --sample is a sample -------------------


def _refs(n: int) -> list:
    from tools.trajectory_analysis.loader import TrialRef

    # Same shape discovery produces on a fanout tree: many rollouts of a
    # few tasks, sorted by NAME, so the head is one task's rollouts.
    return [
        TrialRef(
            root="r",
            trial=f"task{i // 10}-{i % 10}",
            task_id=f"task{i // 10}",
            fanout_idx=i % 10,
        )
        for i in range(n)
    ]


def test_select_refs_limit_is_a_head_not_a_sample():
    got = select_refs(_refs(50), limit=5)

    assert [r.trial for r in got] == [f"task0-{i}" for i in range(5)]
    # The whole point: a head over a fanout tree sees ONE task.
    assert {r.task_id for r in got} == {"task0"}


def test_select_refs_sample_spreads_across_tasks_and_is_reproducible():
    a = select_refs(_refs(50), sample=10, seed=7)
    b = select_refs(_refs(50), sample=10, seed=7)
    c = select_refs(_refs(50), sample=10, seed=8)

    assert [r.trial for r in a] == [r.trial for r in b]  # seeded
    assert [r.trial for r in a] != [r.trial for r in c]  # seed matters
    assert len({r.task_id for r in a}) > 1
    # Listing order preserved so downstream row tables stay stable.
    assert [r.trial for r in a] == sorted(r.trial for r in a)


def test_select_refs_sample_larger_than_the_tree_takes_everything():
    assert len(select_refs(_refs(3), sample=10)) == 3


def test_analyze_run_sample_caps_the_trials_loaded(tmp_path):
    _make_job_tree(tmp_path, [("t1", i, 1.0) for i in range(6)])

    result = analyze_run(str(tmp_path), sample=2, seed=1)

    assert result.n_trials == 2


def test_analyze_sxs_sample_selects_the_same_trials_on_both_sides(tmp_path):
    """Two independent draws would agree on almost nothing."""
    root_a, root_b = tmp_path / "a", tmp_path / "b"
    trials = [("t%d" % i, 0, 1.0) for i in range(20)]
    _make_job_tree(root_a, trials)
    _make_job_tree(root_b, trials)

    result = analyze_sxs(str(root_a), str(root_b), sample=5, seed=3)

    assert result.n_pairs == 5


def test_load_trial_says_read_failed_not_missing_when_a_read_blows_up(tmp_path):
    _make_job_tree(tmp_path, [("t1", 0, 1.0)])
    ref = discover_trials(str(tmp_path))[0]

    class _BoomReader(LocalReader):
        def read_text(self, rel_path: str):
            raise TreeReadError("backend down")

    with pytest.raises(TrialLoadError, match="read failed"):
        load_trial(ref, reader=_BoomReader(str(tmp_path)))
