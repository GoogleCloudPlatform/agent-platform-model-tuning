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

"""Tests for core.trainer_metrics — client-side typed trainer metric registry."""

import pytest

import core.trainer_metrics as trainer_metrics_mod
from core.metric_spec import AggregationStrategy, MetricSpec
from core.trainer_metrics import (
    TRAINER_METRICS,
    TrainerMetric,
    UnknownTrainerMetricError,
    _build_index,
    lookup_trainer_metric,
    map_trainer_wire_metrics,
)


class TestTrainerMetric:
    def test_holds_all_fields(self):
        tm = TrainerMetric(
            wire_name="loss",
            client_key="loss",
            strategy=AggregationStrategy.MEAN,
            emit_to_ml_platform=True,
            doc="Optimizer loss for the batch.",
        )
        assert tm.wire_name == "loss"
        assert tm.client_key == "loss"
        assert tm.strategy is AggregationStrategy.MEAN
        assert tm.emit_to_ml_platform is True
        assert tm.doc == "Optimizer loss for the batch."

    def test_is_frozen(self):
        tm = TrainerMetric(
            wire_name="loss",
            client_key="loss",
            strategy=AggregationStrategy.MEAN,
            emit_to_ml_platform=True,
            doc="d",
        )
        with pytest.raises(AttributeError):
            tm.wire_name = "other"  # type: ignore[misc]

    def test_rejects_empty_wire_name(self):
        with pytest.raises(ValueError, match="wire_name"):
            TrainerMetric(
                wire_name="",
                client_key="loss",
                strategy=AggregationStrategy.MEAN,
                emit_to_ml_platform=True,
                doc="d",
            )

    def test_rejects_empty_client_key(self):
        with pytest.raises(ValueError, match="client_key"):
            TrainerMetric(
                wire_name="loss",
                client_key="",
                strategy=AggregationStrategy.MEAN,
                emit_to_ml_platform=True,
                doc="d",
            )


class TestUnknownTrainerMetricError:
    def test_is_key_error_subclass_and_names_the_metric(self):
        err = UnknownTrainerMetricError("mystery_metric")
        assert isinstance(err, KeyError)
        assert "mystery_metric" in str(err)


class TestRegistry:
    def test_loss_is_not_registered(self):
        """``loss`` used to be registered for one alternate backend only
        (real Vertex does not disclose loss over the wire). The entry was
        removed entirely: that backend's synthetic loss now hits the
        lenient-drop path (warn+drop once per process) instead of taking
        up a spec slot no production consumer justified. This test locks
        that decision in — a future re-add should be a deliberate change
        with a rationale, not a silent one."""
        by_wire = {tm.wire_name: tm for tm in TRAINER_METRICS}
        assert "loss" not in by_wire

    # -----------------------------------------------------------------
    # Real-Vertex wire-metric declarations. Locks down the semantics
    # chosen when these were opted in from the previously-dropped WARN
    # pool; a future edit that flips strategy or emit_to_ml_platform should be
    # a deliberate change, not a silent one.
    # -----------------------------------------------------------------

    def test_train_step_seconds_is_registered(self):
        by_wire = {tm.wire_name: tm for tm in TRAINER_METRICS}
        assert "train_step_seconds" in by_wire
        tm = by_wire["train_step_seconds"]
        assert tm.client_key == "train_step_seconds"
        assert tm.strategy is AggregationStrategy.MEAN
        assert tm.emit_to_ml_platform is True

    def test_billable_training_tokens_is_registered(self):
        by_wire = {tm.wire_name: tm for tm in TRAINER_METRICS}
        assert "billable_training_tokens" in by_wire
        tm = by_wire["billable_training_tokens"]
        assert tm.client_key == "billable_training_tokens"
        assert tm.strategy is AggregationStrategy.SUM
        assert tm.emit_to_ml_platform is True

    def test_step_is_registered_under_renamed_client_key_and_not_emitted(self):
        # Distinct client_key to avoid colliding with the framework's
        # own step column on the ml_platform sink's backend;
        # emit_to_ml_platform=False for the same reason.
        by_wire = {tm.wire_name: tm for tm in TRAINER_METRICS}
        assert "step" in by_wire
        tm = by_wire["step"]
        assert tm.client_key == "server_train_step"
        assert tm.strategy is AggregationStrategy.PASSTHROUGH
        assert tm.emit_to_ml_platform is False

    def test_training_tokens_before_merge_is_registered(self):
        by_wire = {tm.wire_name: tm for tm in TRAINER_METRICS}
        assert "training_tokens_before_merge" in by_wire
        tm = by_wire["training_tokens_before_merge"]
        assert tm.client_key == "training_tokens_before_merge"
        assert tm.strategy is AggregationStrategy.SUM
        assert tm.emit_to_ml_platform is True

    def test_mean_abs_advantage_is_registered(self):
        by_wire = {tm.wire_name: tm for tm in TRAINER_METRICS}
        assert "mean_abs_advantage" in by_wire
        tm = by_wire["mean_abs_advantage"]
        assert tm.client_key == "mean_abs_advantage"
        assert tm.strategy is AggregationStrategy.MEAN
        assert tm.emit_to_ml_platform is True

    def test_no_two_declarations_share_a_client_key(self):
        # The `step` -> `server_train_step` rename means client_key may
        # differ from wire_name; make sure no two declarations end up
        # pointing at the same client_key (that would collide
        # downstream in the aggregation stage).
        client_keys = [tm.client_key for tm in TRAINER_METRICS]
        assert len(client_keys) == len(set(client_keys)), (
            f"duplicate client_key in TRAINER_METRICS: {client_keys}"
        )

    def test_index_maps_wire_name_to_declaration(self):
        index = _build_index(TRAINER_METRICS)
        assert index["train_step_seconds"].wire_name == "train_step_seconds"
        assert set(index.keys()) == {tm.wire_name for tm in TRAINER_METRICS}

    def test_build_index_raises_on_duplicate_wire_name(self):
        dup = (
            TrainerMetric(
                wire_name="x",
                client_key="x",
                strategy=AggregationStrategy.MEAN,
                emit_to_ml_platform=False,
                doc="first",
            ),
            TrainerMetric(
                wire_name="x",
                client_key="x2",
                strategy=AggregationStrategy.SUM,
                emit_to_ml_platform=False,
                doc="second",
            ),
        )
        with pytest.raises(ValueError, match="duplicate.*x"):
            _build_index(dup)


class TestLookup:
    def test_hit_returns_declaration(self):
        tm = lookup_trainer_metric("train_step_seconds")
        assert tm.wire_name == "train_step_seconds"
        assert tm.client_key == "train_step_seconds"

    def test_miss_raises_unknown_trainer_metric_error(self):
        with pytest.raises(UnknownTrainerMetricError, match="mystery"):
            lookup_trainer_metric("mystery")

    def test_miss_error_message_lists_known_names(self):
        with pytest.raises(UnknownTrainerMetricError, match="train_step_seconds"):
            lookup_trainer_metric("mystery")


class TestMapWireMetrics:
    def test_maps_known_metric_by_client_key(self):
        client, specs = map_trainer_wire_metrics({"train_step_seconds": 4.2})
        assert client == {"train_step_seconds": 4.2}
        assert set(specs.keys()) == {"train_step_seconds"}

    def test_returned_spec_carries_strategy_and_emit_flag(self):
        # Use train_step_seconds — a real Vertex wire metric with
        # emit_to_ml_platform=True.
        _client, specs = map_trainer_wire_metrics({"train_step_seconds": 1.5})
        spec = specs["train_step_seconds"]
        assert isinstance(spec, MetricSpec)
        assert spec.name == "train_step_seconds"
        assert spec.strategy is AggregationStrategy.MEAN
        assert spec.emit_to_ml_platform is True

    def test_empty_input_returns_empty(self):
        client, specs = map_trainer_wire_metrics({})
        assert client == {}
        assert specs == {}

    def test_realistic_trainer_batch_maps_all_declared_metrics(self):
        # Realistic MSRL train_step wire payload: all five declared
        # real-Vertex wire metrics arrive together, with the observed
        # value types (float / int / int / int / None). Verifies:
        #   1. All five map through (none dropped as unknown).
        #   2. `step` lands under the renamed client_key
        #      `server_train_step`, not under `step`.
        #   3. mean_abs_advantage=None passes through untouched (the
        #      downstream aggregator is where None-skipping happens).
        #   4. Each returned MetricSpec carries the strategy that was
        #      declared for that wire name.
        wire_payload = {
            "train_step_seconds": 4.2,
            "billable_training_tokens": 1024,
            "step": 42,
            "training_tokens_before_merge": 500,
            "mean_abs_advantage": None,
        }
        client, specs = map_trainer_wire_metrics(wire_payload)
        # All five surface, `step` under its renamed client key.
        assert set(client.keys()) == {
            "train_step_seconds",
            "billable_training_tokens",
            "server_train_step",
            "training_tokens_before_merge",
            "mean_abs_advantage",
        }
        assert client["train_step_seconds"] == 4.2
        assert client["billable_training_tokens"] == 1024
        assert client["server_train_step"] == 42
        assert client["training_tokens_before_merge"] == 500
        assert client["mean_abs_advantage"] is None
        # MetricSpec strategies match the declarations.
        assert specs["train_step_seconds"].strategy is AggregationStrategy.MEAN
        assert specs["billable_training_tokens"].strategy is AggregationStrategy.SUM
        assert specs["server_train_step"].strategy is AggregationStrategy.PASSTHROUGH
        assert specs["training_tokens_before_merge"].strategy is AggregationStrategy.SUM
        assert specs["mean_abs_advantage"].strategy is AggregationStrategy.MEAN
        # emit_to_ml_platform: everything except server_train_step ships.
        assert specs["train_step_seconds"].emit_to_ml_platform is True
        assert specs["billable_training_tokens"].emit_to_ml_platform is True
        assert specs["server_train_step"].emit_to_ml_platform is False
        assert specs["training_tokens_before_merge"].emit_to_ml_platform is True
        assert specs["mean_abs_advantage"].emit_to_ml_platform is True

    def test_unknown_metric_is_dropped_not_raised(self, caplog):
        # Lenient policy: unknown wire names get dropped from the returned
        # mapping (they DON'T raise, so the training loop keeps running when
        # the server emits a metric the client hasn't been updated for).
        # Known metrics next to them still map correctly.
        trainer_metrics_mod._WARNED_UNKNOWN_WIRE_NAMES.clear()
        with caplog.at_level("WARNING", logger="core.trainer_metrics"):
            client, specs = map_trainer_wire_metrics(
                {"train_step_seconds": 0.1, "mystery": 9}
            )
        assert client == {"train_step_seconds": 0.1}
        assert set(specs.keys()) == {"train_step_seconds"}
        # The unknown name is called out by the WARN log so drift is visible.
        assert any("mystery" in rec.getMessage() for rec in caplog.records), [
            rec.getMessage() for rec in caplog.records
        ]

    def test_unknown_metric_warning_fires_once_per_wire_name(self, caplog):
        # Recurring unknown wire names (fired every train step in the wild)
        # must NOT spam the log. First call warns; subsequent calls with the
        # same name are silent.
        trainer_metrics_mod._WARNED_UNKNOWN_WIRE_NAMES.clear()
        with caplog.at_level("WARNING", logger="core.trainer_metrics"):
            for _ in range(5):
                map_trainer_wire_metrics({"train_step_seconds": 0.0, "mystery": 1})
        mystery_warns = [rec for rec in caplog.records if "mystery" in rec.getMessage()]
        assert len(mystery_warns) == 1, [r.getMessage() for r in mystery_warns]

    def test_unknown_metric_warning_fires_per_distinct_wire_name(self, caplog):
        # A different unseen wire name gets its own one-time WARN.
        trainer_metrics_mod._WARNED_UNKNOWN_WIRE_NAMES.clear()
        with caplog.at_level("WARNING", logger="core.trainer_metrics"):
            map_trainer_wire_metrics({"alpha": 1})
            map_trainer_wire_metrics({"beta": 2})
            map_trainer_wire_metrics({"alpha": 3})  # no re-WARN
            map_trainer_wire_metrics({"beta": 4})  # no re-WARN
        seen_alpha = sum(1 for rec in caplog.records if "'alpha'" in rec.getMessage())
        seen_beta = sum(1 for rec in caplog.records if "'beta'" in rec.getMessage())
        assert seen_alpha == 1
        assert seen_beta == 1

    def test_all_unknown_returns_empty_and_does_not_raise(self):
        # Edge case: every wire metric is unknown. Returns empty mapping,
        # does NOT raise. Training loop must keep running.
        trainer_metrics_mod._WARNED_UNKNOWN_WIRE_NAMES.clear()
        client, specs = map_trainer_wire_metrics({"unknown_a": 1, "unknown_b": 2})
        assert client == {}
        assert specs == {}

    def test_preserves_value_verbatim(self):
        client, _ = map_trainer_wire_metrics({"train_step_seconds": 0.0})
        assert client["train_step_seconds"] == 0.0


class TestRenameSupport:
    def test_map_emits_under_client_key_not_wire_name(self, monkeypatch):
        renamed = TrainerMetric(
            wire_name="usedCand",
            client_key="audit_used_candidates",
            strategy=AggregationStrategy.SUM,
            emit_to_ml_platform=False,
            doc="Count of used candidates (renamed from wire camelCase).",
        )
        synthetic = (renamed,)
        monkeypatch.setattr(trainer_metrics_mod, "TRAINER_METRICS", synthetic)
        monkeypatch.setattr(
            trainer_metrics_mod,
            "_INDEX",
            trainer_metrics_mod._build_index(synthetic),
        )

        client, specs = trainer_metrics_mod.map_trainer_wire_metrics({"usedCand": 7})

        assert client == {"audit_used_candidates": 7}
        assert set(specs.keys()) == {"audit_used_candidates"}
        assert specs["audit_used_candidates"].strategy is AggregationStrategy.SUM


class TestRegressionFence:
    """A server-added wire metric without a TrainerMetric entry is
    handled leniently by `map_trainer_wire_metrics` (dropped + one-time
    WARN naming the offender and pointing at the registry). The training
    loop must NEVER crash on server-side wire-metric additions.

    `lookup_trainer_metric` -- the single-key semantic lookup -- still
    raises. See `TestLookup` above.
    """

    def test_server_added_metric_drops_and_warns(self, caplog):
        # Simulate a server that started emitting a brand-new metric.
        trainer_metrics_mod._WARNED_UNKNOWN_WIRE_NAMES.clear()
        server_response_metrics: dict[str, object] = {
            "train_step_seconds": 0.2,
            "newServerMetric_v2": 42,
        }
        with caplog.at_level("WARNING", logger="core.trainer_metrics"):
            client, specs = map_trainer_wire_metrics(server_response_metrics)
        # Known metric survives; unknown is dropped.
        assert client == {"train_step_seconds": 0.2}
        assert set(specs.keys()) == {"train_step_seconds"}
        # WARN names the offender AND points callers at the registry so
        # opting the metric IN is discoverable.
        msgs = [rec.getMessage() for rec in caplog.records]
        assert any("newServerMetric_v2" in m for m in msgs), msgs
        assert any("TRAINER_METRICS" in m for m in msgs), msgs

    def test_lookup_still_raises_on_unknown(self):
        # The single-key semantic-lookup path stays strict: callers who
        # ask "give me the TrainerMetric for X" must get an error on
        # X-not-registered, not `None`. Batch translation uses the lenient
        # `map_trainer_wire_metrics` path instead.
        with pytest.raises(UnknownTrainerMetricError, match="TRAINER_METRICS"):
            lookup_trainer_metric("newServerMetric_v2")

    def test_all_known_metrics_map_without_raising(self):
        # Every declared wire_name must map cleanly (guards a typo in a future
        # entry where wire_name is declared but map path can't resolve it).
        all_known = {tm.wire_name: 1.0 for tm in TRAINER_METRICS}
        client, specs = map_trainer_wire_metrics(all_known)
        assert set(client.keys()) == {tm.client_key for tm in TRAINER_METRICS}
        assert set(specs.keys()) == {tm.client_key for tm in TRAINER_METRICS}


class TestDocQuality:
    """Drift alarm for the trainer-metric doc quality bar.

    Every ``TrainerMetric.doc`` string must be a proper 2-sentence
    "what + how to interpret" description.
    The 20-word threshold catches accidental short placeholders like
    the auto-migrated strings without being so strict that
    a legitimately terse two-sentence doc fails.
    """

    def test_every_trainer_metric_doc_is_at_least_20_words(self):
        short_docs: list[tuple[str, int, str]] = []
        for tm in TRAINER_METRICS:
            word_count = len(tm.doc.split())
            if word_count < 20:
                short_docs.append((tm.wire_name, word_count, tm.doc))
        assert not short_docs, (
            "TrainerMetric entries with doc strings under 20 words "
            "(likely placeholders); rewrite to proper 2-sentence "
            f"'what + how to interpret' prose: {short_docs}"
        )
