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

"""Client-side typed trainer metric registry.

The trainer's ``train_step`` wire response carries a ``metrics`` list
of ``{"metricName": str, "value": Any}`` entries. This module declares
the set of metric names the framework understands (each as a
``TrainerMetric`` with aggregation strategy, unit, ml_platform flag)
and maps a wire response through that declaration set.

Unknown wire metrics: ``map_trainer_wire_metrics`` logs a one-time
WARNING per unseen ``wire_name`` per process and drops the value.
Adding a ``TrainerMetric`` entry to ``TRAINER_METRICS`` opts a wire
metric IN (and stops the warning). ``lookup_trainer_metric`` still
raises ``UnknownTrainerMetricError`` — callers wanting lenient batch
behavior should use ``map_trainer_wire_metrics``.

Scope: this module types ONLY server-returned ``metricName``
passthrough. Client-computed trainer keys are owned by other layers.

Scope note: this module types ONLY the server-returned ``metricName``
passthrough. The trainer's *client-computed* keys (``avg_train_step_id`` /
``finishReason/*``; ``audit/*`` counters, extracted by the audit
aggregator) are NOT handled here.

Pure logic: no I/O beyond a WARNING log line, no imports from ``pipeline`` or
``backends``. Imports only ``core.metric_spec`` and ``logging``.
"""

from __future__ import annotations

import dataclasses
import logging

from core.metric_spec import AggregationStrategy, MetricSpec

__all__ = [
    "TrainerMetric",
    "TRAINER_METRICS",
    "UnknownTrainerMetricError",
    "lookup_trainer_metric",
    "map_trainer_wire_metrics",
]

logger = logging.getLogger(__name__)

# Per-process one-time-warn cache; a recurring unknown metric doesn't
# spam the log.
_WARNED_UNKNOWN_WIRE_NAMES: set[str] = set()


class UnknownTrainerMetricError(KeyError):
    """Raised when a wire ``metricName`` has no ``TrainerMetric``
    declaration. Subclasses ``KeyError`` so existing ``except KeyError``
    still catches it, while allowing specific handling at the wire-parse
    boundary."""


@dataclasses.dataclass(frozen=True)
class TrainerMetric:
    """One trainer metric the framework accepts from a backend's wire.

    Attributes:
        wire_name: The server's ``metricName``.
        client_key: The snake_case key used client-side (may differ
            from ``wire_name``).
        strategy: Layer-2 batch aggregation strategy.
        emit_to_ml_platform: Forward to ml_platform sink?
        doc: Human description; propagated onto the derived
            ``MetricSpec`` via ``map_trainer_wire_metrics``.
    """

    wire_name: str
    client_key: str
    strategy: AggregationStrategy
    emit_to_ml_platform: bool
    doc: str

    def __post_init__(self) -> None:
        if not self.wire_name:
            raise ValueError("TrainerMetric.wire_name must be non-empty")
        if not self.client_key:
            raise ValueError("TrainerMetric.client_key must be non-empty")


# The registry. Deliberately closed: any wire ``metricName`` without
# an entry below is dropped with a one-time WARN. Adding a new entry
# opts a wire metric IN with explicit aggregation strategy +
# ml_platform flag.
#
# The registry is deliberately closed: any wire ``metricName`` from a real
# backend that lacks a TrainerMetric entry below is dropped with a one-time
# WARN per wire name (see ``map_trainer_wire_metrics``). Adding a new
# ``TrainerMetric`` opts a wire metric IN (with explicit aggregation
# strategy + ship-to-VE flag) so future server-side additions are surfaced
# via an explicit code change rather than silently accepted.
#
# Adding a wire metric here changes it from "warned + dropped" to "typed +
# aggregated + optionally shipped to VE". The framework surfaces only the
# opted-in set; the registry is the single source of truth for names the
# framework understands.
#
# NOTE: client-computed trainer keys (avg_train_step_id / finishReason/* /
# audit/*) are NOT in this registry -- they are aggregator concerns, not
# wire-passthrough concerns owned by this registry.
#
# Naming and non-collision with ``core.metric_spec.METRIC_SPECS``:
#   * ``client_key`` here is the PRE-PREFIX name (e.g. ``"loss"``). The training
#     loop later prepends ``train/`` to produce the final logged key
#     (``pipeline/train_and_eval.py:283`` does ``f"train/{k}"``); the
#     ``MetricSpec`` synthesized by ``map_trainer_wire_metrics`` therefore has
#     ``name == client_key`` (unprefixed), NOT the prefixed key the sink sees.
#   * Wire-trainer keys MUST NOT be duplicated in ``core.metric_spec.METRIC_SPECS``.
#     This registry is the single source of truth for names that arrive from
#     the trainer wire; ``METRIC_SPECS`` owns framework-emitted keys (learning/*,
#     sampler/*, pipeline/*, recipe/*). Two authoritative specs for one key
#     would confuse downstream sinks.
# ---------------------------------------------------------------------------
# NOTE: no ``loss`` entry.
#   * Real Vertex does not disclose loss values over the training wire.
#   * Alternate-backend synthetic ``loss`` values are also not declared
#     here — they hit the lenient-drop path (warn + drop once
#     per process). Adding a spec back requires a concrete production
#     consumer for the value.
TRAINER_METRICS: tuple[TrainerMetric, ...] = (
    TrainerMetric(
        wire_name="train_step_seconds",
        client_key="train_step_seconds",
        strategy=AggregationStrategy.MEAN,
        emit_to_ml_platform=True,
        doc=(
            "Wall-clock time in seconds for one server-side train step, "
            "as reported by the trainer wire. Averaged across the "
            "aggregation stage; useful cross-run telemetry for spotting "
            "server-side perf regressions (compare to client-side "
            "``train/pipeline/train_step_time`` which includes LRO "
            "polling)."
        ),
    ),
    TrainerMetric(
        wire_name="billable_training_tokens",
        client_key="billable_training_tokens",
        strategy=AggregationStrategy.SUM,
        emit_to_ml_platform=True,
        doc=(
            "Total tokens counted for billing this train step (server-"
            "computed), summed across the aggregation stage so the "
            "aggregate is total billable tokens over the batch. Useful "
            "for cost tracking — divide by ``train_step_seconds`` for a "
            "tokens-per-second-of-billed-work rate."
        ),
    ),
    TrainerMetric(
        wire_name="step",
        client_key="server_train_step",
        strategy=AggregationStrategy.PASSTHROUGH,
        emit_to_ml_platform=False,
        doc=(
            "Server-side monotonically-increasing train step counter, "
            "surfaced under a distinct client_key so it doesn't collide "
            "with the framework's own step column. Kept jsonl-only "
            "because the ml_platform sink already carries an "
            "authoritative step column and a second step-shaped axis "
            "would confuse the timeline; use only for cross-check "
            "during debugging."
        ),
    ),
    TrainerMetric(
        wire_name="training_tokens_before_merge",
        client_key="training_tokens_before_merge",
        strategy=AggregationStrategy.SUM,
        emit_to_ml_platform=True,
        doc=(
            "Tokens the trainer has seen since the last weight-merge "
            "boundary (server-reported). Summed across the aggregation "
            "stage for a total-tokens training-progress plot on the "
            "ml_platform sink's dashboard."
        ),
    ),
    TrainerMetric(
        wire_name="mean_abs_advantage",
        client_key="mean_abs_advantage",
        strategy=AggregationStrategy.MEAN,
        emit_to_ml_platform=True,
        doc=(
            "Mean absolute advantage in the batch (RL diagnostic: how "
            "large the reward signal is after baselining). May arrive "
            "as ``None`` on train steps where the trainer hasn't "
            "computed it; the framework aggregator skips ``None`` "
            "values before taking mean so a run mixing None + numeric "
            "values is safe."
        ),
    ),
)


def _build_index(
    metrics: tuple[TrainerMetric, ...],
) -> dict[str, TrainerMetric]:
    """Build a wire_name -> TrainerMetric index; raise on duplicates.
    Called at import to validate ``TRAINER_METRICS`` (fail-fast)."""
    index: dict[str, TrainerMetric] = {}
    for tm in metrics:
        if tm.wire_name in index:
            raise ValueError(
                f"duplicate TrainerMetric wire_name {tm.wire_name!r} in registry"
            )
        index[tm.wire_name] = tm
    return index


_INDEX: dict[str, TrainerMetric] = _build_index(TRAINER_METRICS)


def lookup_trainer_metric(wire_name: str) -> TrainerMetric:
    """Resolve one wire ``metricName`` to its ``TrainerMetric``.

    Raises:
        UnknownTrainerMetricError: if unknown. Message lists known
            wire names so the fix (add a ``TRAINER_METRICS`` entry) is
            obvious.
    """
    try:
        return _INDEX[wire_name]
    except KeyError:
        known = ", ".join(sorted(_INDEX)) or "(none)"
        raise UnknownTrainerMetricError(
            f"unknown trainer wire metric {wire_name!r}; "
            f"declare it in core.trainer_metrics.TRAINER_METRICS. "
            f"Known: {known}"
        ) from None


def map_trainer_wire_metrics(
    raw: dict[str, object],
) -> tuple[dict[str, object], dict[str, MetricSpec]]:
    """Map a trainer's raw wire ``{metricName: value}`` slice to client
    keys.

    Unknown wire names are DROPPED (not raised) so the training loop
    keeps running when the server emits a not-yet-typed metric. Each
    unseen name logs a one-time WARNING per process (throttled via
    ``_WARNED_UNKNOWN_WIRE_NAMES``). Add a ``TrainerMetric`` entry to
    ``TRAINER_METRICS`` to opt a wire metric IN.

    Args:
        raw: server-returned ``{metricName: value}`` pairs only (no
            client-computed keys).

    Returns:
        ``(client_values, client_specs)`` with entries ONLY for known
        wire metrics. Empty if every wire name is unknown.
    """
    values: dict[str, object] = {}
    specs: dict[str, MetricSpec] = {}
    for wire_name, value in raw.items():
        tm = _INDEX.get(wire_name)
        if tm is None:
            if wire_name not in _WARNED_UNKNOWN_WIRE_NAMES:
                _WARNED_UNKNOWN_WIRE_NAMES.add(wire_name)
                known = ", ".join(sorted(_INDEX)) or "(none)"
                logger.warning(
                    "[trainer_metrics] dropping undeclared trainer wire "
                    "metric %r (value type=%s); add a TrainerMetric entry "
                    "to core.trainer_metrics.TRAINER_METRICS to surface "
                    "it. Known: %s. Warning fires once per unseen "
                    "wire_name per process.",
                    wire_name,
                    type(value).__name__,
                    known,
                )
            continue
        values[tm.client_key] = value
        specs[tm.client_key] = MetricSpec(
            name=tm.client_key,
            strategy=tm.strategy,
            doc=tm.doc,
            emit_to_ml_platform=tm.emit_to_ml_platform,
        )
    return values, specs
