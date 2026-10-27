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

"""Orchestration: load trials, aggregate metrics, build a report.

Aggregation deliberately reuses the pipeline's own machinery
(``core.metric_aggregators`` + ``core.metric_spec``) so an offline analysis
of a finished run reproduces that run's reported numbers rather than
computing a parallel, subtly different average.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import logging
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from core.metric_aggregators import DEFAULT_METRIC_AGGREGATORS, MetricAggregator
from core.metric_spec import (
    METRIC_SPECS,
    AggregationStrategy,
    MetricSpec,
    MetricValue,
    aggregate_by_spec,
    merge_group_metrics,
)

from .loader import TrialLoadError, TrialRef, TrialView, discover_trials, load_trial
from .metrics import SXS_METRICS
from .pairing import pair_by_fanout, pair_by_task
from .tree_reader import make_reader

logger = logging.getLogger(__name__)

# How many unmatched/unscored trial names ``AnalysisResult.render`` spells
# out before collapsing the rest to a count.
_UNMATCHED_PREVIEW = 8


def select_refs(
    refs: Sequence[TrialRef],
    *,
    limit: int | None = None,
    sample: int | None = None,
    seed: int = 0,
) -> list[TrialRef]:
    """Cut a discovered ref list down to the trials worth loading.

    Two different cuts, and the difference matters more than it looks:

    * ``limit`` is a HEAD -- the first N in listing order, which is
      directory-name order. On a fanout tree that is N rollouts of the
      same one or two tasks (``<task>-0``, ``<task>-1``, ``<task>-10``
      ...), so ``learning/*`` computed over it describes those tasks and
      nothing else. Use it to check plumbing, never to estimate a run.
    * ``sample`` is a uniform random draw over the whole tree, seeded so
      two invocations (and the two sides of an SxS) select reproducibly.
      This is the one to use for an estimate from a big run.

    Sampling happens BEFORE any trial is read, so the cost saved is the
    loading, not just the reporting. The result stays in listing order so
    downstream row tables and pairing behave identically either way.
    """
    if sample is not None:
        if sample < len(refs):
            chosen = random.Random(seed).sample(range(len(refs)), sample)
            return [refs[i] for i in sorted(chosen)]
        return list(refs)
    return list(refs[:limit] if limit is not None else refs)


def group_by_task(views: Sequence[TrialView]) -> list[list[TrialView]]:
    """Group trials into trajectory groups (all rollouts of one task).

    The aggregators are group-shaped: ``learning/pass_at_k`` is
    ``any(is_correct)`` over a group and ``learning/reward/mean`` is
    weighted by group size. Feeding them one trial at a time yields
    different numbers than the pipeline reported.
    """
    buckets: dict[str, list[TrialView]] = defaultdict(list)
    for view in views:
        buckets[view.ref.task_id].append(view)
    return [buckets[k] for k in sorted(buckets)]


def _as_metric_values(raw: Mapping[Any, Any]) -> dict[str, MetricValue]:
    """Wrap a metric function's raw output in ``MetricValue`` carriers.

    A metric may return bare floats for convenience; the reduction path
    only speaks ``MetricValue``, so scalars are lifted here once for both
    the per-group and the per-pair callers.
    """
    return {
        str(k): v if isinstance(v, MetricValue) else MetricValue.of_scalar(float(v))
        for k, v in raw.items()
    }


def _group_metrics(
    group: Sequence[TrialView], aggregators: Sequence[MetricAggregator]
) -> dict[str, MetricValue]:
    """One group's merged aggregator output.

    Mirrors ``Agent._compute_group_metrics``: same scalar wrapping, and the
    same ``merge_group_metrics`` collision guard -- two aggregators emitting
    the same key for one group is an error here exactly as it is in the
    pipeline, not a silent last-writer-wins.
    """
    scored = [v.scored for v in group]
    outputs = [_as_metric_values(aggregator(scored)) for aggregator in aggregators]
    return merge_group_metrics(outputs)


def aggregate_views(
    views: Sequence[TrialView],
    *,
    aggregators: Sequence[MetricAggregator] | None = None,
) -> dict[str, float]:
    """Reduce loaded trials to final metric values, pipeline-style."""
    if not views:
        return {}
    aggregators = aggregators or DEFAULT_METRIC_AGGREGATORS
    per_group = [_group_metrics(g, aggregators) for g in group_by_task(views)]
    return aggregate_by_spec(per_group)


def delta_key_for(source_key: str) -> str:
    """Name of the side-by-side delta metric derived from ``source_key``.

    Always exactly three segments, whatever the source key's depth,
    because the ml-platform sink caps metric names at four ``/`` segments
    and REJECTS over-deep names outright rather than truncating them
    (truncating would collide two metrics onto one series). The naive
    ``sxs/<source_key>/delta`` spelling blows that cap for any source key
    with three segments of its own -- ``learning/reward/mean`` being the
    most important one in the set -- so the source key is flattened into
    a single compound leaf instead.
    """
    return f"sxs/delta/{source_key.replace('/', '_')}"


def _register_delta_spec(delta_key: str, source_key: str) -> None:
    """Register a delta metric so downstream sinks will forward it.

    Delta values are already final -- they are the difference of two
    reduced aggregates -- hence ``PASSTHROUGH``: no further cross-stage
    reduction. Registration is idempotent and mutates the global
    ``METRIC_SPECS``, the same mechanism recipes use to declare their own
    keys at import time; here it happens on first use because the key set
    is not known until the two runs have been aggregated.
    """
    if delta_key in METRIC_SPECS:
        return
    METRIC_SPECS[delta_key] = MetricSpec(
        delta_key,
        AggregationStrategy.PASSTHROUGH,
        f"Side-by-side difference in {source_key} between the two runs being "
        "compared, computed as the B-side aggregate minus the A-side aggregate. "
        "Positive means side B scored higher on that metric; the value is final "
        "and is not reduced further.",
        emit_to_ml_platform=True,
    )


def aggregate_pairs(
    pairs: Sequence[tuple[TrialView, TrialView]],
    *,
    aggregators: Sequence[MetricAggregator] | None = None,
    sxs_metrics: Sequence[str] | None = None,
) -> dict[str, float]:
    """Reduce a list of trial pairs to SxS metric values.

    Two independent contributions, both landing in one dict:

    * pairwise metrics from ``SXS_METRICS`` (win/lose/tie -- irreducibly
      per-pair), reduced through ``aggregate_by_spec`` like every other
      metric, and
    * ``sxs/delta/<flattened key>`` for every core metric, computed as
      the aggregate over the B sides minus the aggregate over the A
      sides. The delta needs no new metric interface: it is the
      subtraction of two ordinary aggregations, so it works for every
      registered metric. Each one registers its own ``MetricSpec`` on
      first use (see ``_register_delta_spec``) so the ml-platform sink
      forwards it rather than silently dropping it.

    A pair with an UNSCORED side (``reward=NaN`` -- the rollout failed) is
    dropped: NaN compares False both ways, so ``outcome`` would score such
    a pair a TIE and inflate ``tie_rate`` with pairs carrying no signal.
    ``analyze_sxs`` drops the same pairs up front and reports them, so its
    ``n_pairs`` matches what was measured; the filter is repeated here
    because ``aggregate_pairs`` is also called directly (the ``--trial``
    path).
    """
    pairs = [(a, b) for a, b in pairs if a.is_scored and b.is_scored]
    if not pairs:
        return {}
    chosen = (
        [SXS_METRICS[n] for n in sxs_metrics]
        if sxs_metrics is not None
        else list(SXS_METRICS.values())
    )
    # Same collision guard as the per-group path: two SxS metrics
    # emitting one key for one pair is an error, not last-writer-wins.
    per_pair = [
        merge_group_metrics([_as_metric_values(fn(a, b)) for fn in chosen])
        for a, b in pairs
    ]
    out = aggregate_by_spec(per_pair)

    agg_a = aggregate_views([a for a, _ in pairs], aggregators=aggregators)
    agg_b = aggregate_views([b for _, b in pairs], aggregators=aggregators)
    for key in sorted(set(agg_a) & set(agg_b)):
        delta_key = delta_key_for(key)
        _register_delta_spec(delta_key, key)
        out[delta_key] = agg_b[key] - agg_a[key]
    return out


@dataclasses.dataclass
class AnalysisResult:
    """Everything one analysis run produced."""

    root: str
    metrics: dict[str, float] = dataclasses.field(default_factory=dict)
    rows: list[dict] = dataclasses.field(default_factory=list)
    skipped: list[str] = dataclasses.field(default_factory=list)
    unmatched: dict[str, list[str]] = dataclasses.field(default_factory=dict)
    n_trials: int = 0
    n_pairs: int = 0
    # The trials this analysis actually loaded, in analysis order (both
    # sides of every pair, for a side-by-side). Retained so a caller that
    # needs the trajectories themselves -- mirroring them somewhere, say --
    # reuses them instead of walking and re-downloading the tree.
    views: list[TrialView] = dataclasses.field(default_factory=list, repr=False)

    def to_json(self) -> dict:
        """The serializable fields only; ``views`` holds loaded objects."""
        return {
            f.name: getattr(self, f.name)
            for f in dataclasses.fields(self)
            if f.name != "views"
        }

    def render(self) -> str:
        """A plain-text table of the aggregate metrics, for stdout."""
        width = max((len(k) for k in self.metrics), default=0)
        lines = [f"{k.ljust(width)}  {v:.4f}" for k, v in sorted(self.metrics.items())]
        if self.skipped:
            lines.append(f"\nskipped {len(self.skipped)} trial(s):")
            lines.extend(f"  {s}" for s in self.skipped)
        for label, refs in sorted(self.unmatched.items()):
            if refs:
                # The count is the number that matters; the names are a
                # sample to start debugging from. An unpaired-heavy SxS can
                # list hundreds, which buries the metrics table above it --
                # the full list is in the --json dump.
                shown = ", ".join(refs[:_UNMATCHED_PREVIEW])
                more = len(refs) - _UNMATCHED_PREVIEW
                suffix = f", ... (+{more} more)" if more > 0 else ""
                lines.append(f"\n{label} ({len(refs)}): {shown}{suffix}")
        return "\n".join(lines)


def load_views(
    refs: Sequence[TrialRef], skipped: list[str], *, max_workers: int = 32
) -> list[TrialView]:
    """Load every ref concurrently, recording per-trial failures.

    Each load is two reads (three with ``result.json``) of a remote
    object, so the work is latency-bound and a bounded pool turns minutes
    into seconds; 32 is deep enough to saturate GCS without inviting
    per-connection throttling.

    Two properties callers depend on:

    * The result is in the SAME ORDER as ``refs``. Completion order is
      not input order, so this uses ``Executor.map`` (ordered) rather
      than ``as_completed``; SxS pairing and the per-trial row tables
      read off this order.
    * One unloadable trial is a recorded skip, never a dead pool.

    One reader is built per distinct root and SHARED across that root's
    loads: ``make_reader`` on a ``gs://`` root constructs a
    ``storage.Client``, which is far more expensive than the read it
    serves. Both backends are safe for concurrent reads.
    """
    if not refs:
        return []
    readers = {ref.root: make_reader(ref.root) for ref in refs}

    def _load(ref: TrialRef) -> TrialView | TrialLoadError:
        try:
            return load_trial(ref, reader=readers[ref.root])
        except TrialLoadError as exc:
            return exc

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, min(max_workers, len(refs)))
    ) as pool:
        loaded = list(pool.map(_load, refs))

    views: list[TrialView] = []
    for outcome in loaded:
        if isinstance(outcome, TrialLoadError):
            logger.warning("Skipping trial: %s", outcome)
            skipped.append(str(outcome))
        else:
            views.append(outcome)
    return views


def analyze_run(
    root: str,
    *,
    aggregators: Sequence[MetricAggregator] | None = None,
    limit: int | None = None,
    sample: int | None = None,
    seed: int = 0,
) -> AnalysisResult:
    """Compute metrics over every trial of one run.

    ``limit`` / ``sample`` cap the trials LOADED and are applied to the
    discovered refs before any of them is read -- a cap that only trimmed
    the output would leave the expensive part of the job untouched. They
    are NOT interchangeable: see ``select_refs``.
    """
    refs = select_refs(discover_trials(root), limit=limit, sample=sample, seed=seed)
    skipped: list[str] = []
    views = load_views(refs, skipped)
    return AnalysisResult(
        root=root,
        metrics=aggregate_views(views, aggregators=aggregators),
        rows=[
            {"trial": v.ref.trial, "task_id": v.ref.task_id, "reward": v.reward}
            for v in views
        ],
        skipped=skipped,
        n_trials=len(views),
        views=views,
    )


def analyze_sxs(
    root_a: str,
    root_b: str | None = None,
    *,
    strategy: str = "task",
    aggregators: Sequence[MetricAggregator] | None = None,
    limit: int | None = None,
    sample: int | None = None,
    seed: int = 0,
) -> AnalysisResult:
    """Compare two runs (``strategy='task'``) or one run's rollouts
    against each other (``strategy='fanout'``).

    Load first, pair second. Discovery only GUESSES identity from the
    directory name (see ``loader.discover_trials``), and pairing two runs
    on a guess mismatches trials whose names lie; the authoritative
    ``(task_id, fanout_idx)`` exists only once a trial is loaded, and
    both sides have to be loaded anyway to compute anything.

    ``limit`` / ``sample`` therefore cap the trials LOADED PER SIDE rather
    than the number of pairs -- capping pairs would require the loading
    they are there to avoid. The pair count that comes out is at most the
    cap and may be smaller when the two sides do not fully overlap.

    The B side is cut to match the A side rather than drawn independently:
    two independent draws over two trees agree on almost nothing, and an
    SxS over 3 surviving pairs is worse than useless. The selection uses
    the NAME-DERIVED identity, which is a guess -- a side whose names lie
    simply pairs less, exactly as it does without a cap.
    """
    skipped: list[str] = []

    def _load(refs: Sequence[TrialRef]) -> list[TrialView]:
        return load_views(refs, skipped)

    refs_a = select_refs(discover_trials(root_a), limit=limit, sample=sample, seed=seed)
    views_a = _load(refs_a)
    views = list(views_a)
    if strategy == "fanout":
        pairing = pair_by_fanout([v.ref for v in views_a])
    else:
        if root_b is None:
            raise ValueError("strategy='task' requires a second root")
        refs_b = discover_trials(root_b)
        if limit is not None or sample is not None:
            wanted = {(r.task_id, r.fanout_idx) for r in refs_a}
            refs_b = [r for r in refs_b if (r.task_id, r.fanout_idx) in wanted]
        views += _load(refs_b)
        pairing = pair_by_task(
            [v.ref for v in views_a], [v.ref for v in views[len(views_a) :]]
        )
    # A ref identifies a path; the path identifies the loaded view. Pairing
    # stays a function of refs alone (one implementation, shared with its
    # own tests) and the views are looked back up here.
    by_path = {(v.ref.root, v.ref.trial): v for v in views}
    matched: list[tuple[TrialView, TrialView]] = [
        (by_path[(a.root, a.trial)], by_path[(b.root, b.trial)])
        for a, b in pairing.pairs
    ]
    # A pair is only comparable if BOTH sides were scored; see
    # aggregate_pairs. Dropped here too (not just inside the aggregation)
    # so n_pairs and the row table describe the pairs actually measured,
    # and the dropped ones are named rather than vanishing.
    pairs = [(a, b) for a, b in matched if a.is_scored and b.is_scored]
    unscored = [
        f"{a.ref.trial} vs {b.ref.trial}"
        for a, b in matched
        if not (a.is_scored and b.is_scored)
    ]
    return AnalysisResult(
        root=root_a,
        metrics=aggregate_pairs(pairs, aggregators=aggregators),
        rows=[
            {
                "task_id": a.ref.task_id,
                "trial_a": a.ref.trial,
                "trial_b": b.ref.trial,
                "reward_a": a.reward,
                "reward_b": b.reward,
                "delta": b.reward - a.reward,
            }
            for a, b in pairs
        ],
        skipped=skipped,
        unmatched={
            "missing_in_a": [r.trial for r in pairing.missing_in_a],
            "missing_in_b": [r.trial for r in pairing.missing_in_b],
            "unpaired": [r.trial for r in pairing.unpaired],
            "unscored": unscored,
        },
        n_pairs=len(pairs),
        views=[v for pair in pairs for v in pair],
    )
