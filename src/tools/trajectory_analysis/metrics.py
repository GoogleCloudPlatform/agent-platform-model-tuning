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

"""The one metric interface this tool adds on top of core's.

Everything expressible as a ``core.metric_aggregators.MetricAggregator``
should BE one -- those are shared with the eval pipeline and their numbers
are directly comparable. Exactly one thing does not fit that mould:
``SxSMetric``, because nothing in core pairs trajectories.

It emits the same ``MetricValue`` carriers and registers the same
``MetricSpec``s as a core aggregator, so both reduce through one
``aggregate_by_spec`` call. There is exactly one aggregation path.

A second interface -- a per-trial metric taking the whole ``TrialView``,
for numbers that need ``result.json`` / ``artifacts/`` / the raw-LLM
ledger rather than just the ``ScoredTrajectory`` -- was defined here and
never used by anything. It is left out until a caller exists; the shape
is a two-line ``Callable`` alias and a registry, cheap to reintroduce
alongside the code that reads it.
"""

from __future__ import annotations

from collections.abc import Callable

from core.metric_spec import METRIC_SPECS, AggregationStrategy, MetricSpec, MetricValue

from .loader import TrialView

SxSMetric = Callable[[TrialView, TrialView], dict[str, MetricValue]]

SXS_METRICS: dict[str, SxSMetric] = {}


def register_sxs_metric(
    name: str, specs: list[MetricSpec]
) -> Callable[[SxSMetric], SxSMetric]:
    """Register a pairwise metric and its aggregation specs."""

    def deco(fn: SxSMetric) -> SxSMetric:
        SXS_METRICS[name] = fn
        METRIC_SPECS.update({s.name: s for s in specs})
        return fn

    return deco


_OUTCOME_SPECS = [
    MetricSpec(
        "sxs/outcome/win_rate",
        AggregationStrategy.MEAN,
        "Fraction of pairs where side B scored higher than side A. "
        "Above 0.5 means B wins more pairs than it loses.",
        emit_to_ml_platform=True,
    ),
    MetricSpec(
        "sxs/outcome/lose_rate",
        AggregationStrategy.MEAN,
        "Fraction of pairs where side B scored lower than side A. "
        "Read it alongside win_rate: the two together bound the signal, "
        "and whatever remains of the pairs are ties.",
        emit_to_ml_platform=True,
    ),
    MetricSpec(
        "sxs/outcome/tie_rate",
        AggregationStrategy.MEAN,
        "Fraction of pairs where both sides scored exactly the same reward. "
        "A high tie rate means the comparison carries little signal and the "
        "win/lose rates are computed over very few decisive pairs.",
        emit_to_ml_platform=True,
    ),
]


@register_sxs_metric("outcome", _OUTCOME_SPECS)
def outcome(a: TrialView, b: TrialView) -> dict[str, MetricValue]:
    """Win/lose/tie for one pair, scored from B's point of view."""
    win = b.reward > a.reward
    lose = b.reward < a.reward
    return {
        "sxs/outcome/win_rate": MetricValue.of_scalar(float(win)),
        "sxs/outcome/lose_rate": MetricValue.of_scalar(float(lose)),
        "sxs/outcome/tie_rate": MetricValue.of_scalar(float(not win and not lose)),
    }
