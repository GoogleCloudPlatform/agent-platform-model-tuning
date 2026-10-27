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

"""Tests for core.metric_spec — typed aggregation strategies.

Recipe-specific spec-registration coverage lives alongside each recipe's
own tree; this file covers the framework spec registry and the typed
aggregation strategies.
"""

import math

import pytest

from core.metric_spec import (
    METRIC_SPECS,
    AggregationStrategy,
    MetricSpec,
    MetricValue,
    _FRAMEWORK_SPEC_KEYS,
    aggregate_by_spec,
    merge_group_metrics,
    reduce_values,
    spec_for,
)


class TestAggregationStrategy:
    def test_has_all_nine_strategies(self):
        assert {s.value for s in AggregationStrategy} == {
            "sum",
            "mean",
            "weighted_mean",
            "max",
            "p50",
            "p95",
            "ratio",
            "ratio_of_sums",
            "passthrough",
        }

    def test_is_str_enum(self):
        # Members are usable interchangeably with their string values.
        assert AggregationStrategy.MEAN == "mean"
        assert isinstance(AggregationStrategy.RATIO, str)


class TestMetricValue:
    def test_of_scalar_holds_value(self):
        mv = MetricValue.of_scalar(3.0)
        assert mv.value == 3.0
        assert mv.weighted_pair is None
        assert mv.samples is None
        assert mv.count_total is None
        assert mv.part_total is None

    def test_of_weighted_holds_pair(self):
        mv = MetricValue.of_weighted(value=4.0, weight=2)
        assert mv.weighted_pair == (4.0, 2)
        assert mv.value is None

    def test_of_samples_holds_list(self):
        mv = MetricValue.of_samples([1.0, 2.0, 3.0])
        assert mv.samples == [1.0, 2.0, 3.0]
        assert mv.value is None

    def test_of_ratio_holds_count_total(self):
        mv = MetricValue.of_ratio(count=3, total=10)
        assert mv.count_total == (3, 10)
        assert mv.value is None

    def test_of_part_total_holds_pair(self):
        mv = MetricValue.of_part_total(part=1.5, total=6.0)
        assert mv.part_total == (1.5, 6.0)
        assert mv.value is None

    def test_is_frozen(self):
        mv = MetricValue.of_scalar(1.0)
        with pytest.raises((AttributeError, TypeError)):
            mv.value = 2.0  # type: ignore[misc]

    def test_rejects_multiple_fields_set(self):
        """A hand-constructed MetricValue with two intermediates set is a bug;
        __post_init__ must raise."""
        with pytest.raises(ValueError, match="exactly one"):
            MetricValue(value=1.0, samples=[1.0, 2.0])

    def test_rejects_zero_fields_set(self):
        with pytest.raises(ValueError, match="exactly one"):
            MetricValue()

    def test_of_scalar_rejects_nan(self):
        with pytest.raises(ValueError, match="NaN"):
            MetricValue.of_scalar(math.nan)

    def test_of_weighted_rejects_nan_value(self):
        with pytest.raises(ValueError, match="weighted_pair.*NaN"):
            MetricValue.of_weighted(math.nan, 3)

    def test_of_samples_rejects_nan_in_list(self):
        with pytest.raises(ValueError, match="samples.*NaN"):
            MetricValue.of_samples([1.0, math.nan, 3.0])

    def test_of_part_total_rejects_nan_in_either_slot(self):
        with pytest.raises(ValueError, match="part_total.*NaN"):
            MetricValue.of_part_total(math.nan, 1.0)
        with pytest.raises(ValueError, match="part_total.*NaN"):
            MetricValue.of_part_total(1.0, math.nan)

    def test_of_ratio_rejects_nan_by_construction(self):
        # ``of_ratio`` casts to int, and ``int(float('nan'))`` raises
        # ValueError at conversion time, so no MetricValue is even
        # constructed. This is documented as intentional in
        # ``__post_init__``'s comment.
        with pytest.raises(ValueError):
            MetricValue.of_ratio(int(math.nan), 4)  # type: ignore[arg-type]


class TestMetricSpec:
    def test_holds_all_fields(self):
        spec = MetricSpec(
            name="learning/reward/mean",
            strategy=AggregationStrategy.WEIGHTED_MEAN,
            doc="one rollout trajectory",
            emit_to_ml_platform=True,
        )
        assert spec.name == "learning/reward/mean"
        assert spec.strategy is AggregationStrategy.WEIGHTED_MEAN
        assert spec.doc == "one rollout trajectory"
        assert spec.emit_to_ml_platform is True

    def test_emit_to_ml_platform_defaults_false(self):
        spec = MetricSpec(
            name="pipeline/iterator_current",
            strategy=AggregationStrategy.PASSTHROUGH,
            doc="one train-step boundary",
        )
        assert spec.emit_to_ml_platform is False

    def test_is_frozen(self):
        spec = MetricSpec(name="x", strategy=AggregationStrategy.SUM, doc="u")
        with pytest.raises((AttributeError, TypeError)):
            spec.name = "y"  # type: ignore[misc]


class TestMetricSpecsRegistry:
    def test_registry_is_keyed_by_name(self):
        for key, spec in METRIC_SPECS.items():
            assert isinstance(spec, MetricSpec)
            # For exact-match entries the key equals the spec name; for
            # prefix entries the key ends with "/*".
            if not key.endswith("/*"):
                assert spec.name == key

    def test_core_reward_metrics_are_weighted_mean_or_mean(self):
        assert spec_for("learning/reward/mean").strategy is (
            AggregationStrategy.WEIGHTED_MEAN
        )
        assert spec_for("learning/sampling_accuracy").strategy is (
            AggregationStrategy.MEAN
        )
        assert spec_for("learning/pass_at_k").strategy is AggregationStrategy.MEAN

    def test_p95_timing_metrics(self):
        assert spec_for("pipeline/sample_time_p95").strategy is (
            AggregationStrategy.P95
        )
        assert spec_for("pipeline/env_time_p95").strategy is AggregationStrategy.P95

    def test_ratio_of_sums_fraction(self):
        assert spec_for("pipeline/sample_time_fraction").strategy is (
            AggregationStrategy.RATIO_OF_SUMS
        )

    def test_dynamic_suffix_family_resolves_by_prefix(self):
        # Any per-reason key under the finish_reason family gets the RATIO
        # strategy without an explicit per-reason registry entry.
        assert spec_for("sampler/finish_reason/STOP").strategy is (
            AggregationStrategy.RATIO
        )
        assert spec_for("sampler/finish_reason/MAX_TOKENS").strategy is (
            AggregationStrategy.RATIO
        )
        assert spec_for("sampler/trajectory_status/OK").strategy is (
            AggregationStrategy.RATIO
        )

    def test_sum_token_metric(self):
        assert spec_for("sampler/generation_tokens_sum").strategy is (
            AggregationStrategy.SUM
        )

    def test_passthrough_pipeline_metric(self):
        assert spec_for("pipeline/train_step_time").strategy is (
            AggregationStrategy.PASSTHROUGH
        )

    def test_unknown_key_raises_keyerror(self):
        with pytest.raises(KeyError, match="no MetricSpec"):
            spec_for("totally/unknown/metric")

    def test_every_spec_doc_is_at_least_20_words(self):
        """Drift alarm for the metric-doc quality bar.

        Every ``MetricSpec.doc`` string in the framework registry must
        be a proper 2-sentence "what + how to interpret" description.
        The 20-word threshold catches accidental short placeholders like
        the auto-migrated ``l1_unit`` strings ("one rollout
        trajectory") without being so strict that a legitimately terse
        two-sentence doc fails.

        This asserts over whatever is registered in METRIC_SPECS at run
        time (framework specs plus any recipe modules already imported).
        Per-recipe doc coverage is enforced alongside each recipe's own
        tree.
        """
        short_docs: list[tuple[str, int, str]] = []
        for key, spec in METRIC_SPECS.items():
            word_count = len(spec.doc.split())
            if word_count < 20:
                short_docs.append((key, word_count, spec.doc))
        assert not short_docs, (
            "MetricSpec entries with doc strings under 20 words "
            "(likely placeholders); rewrite to proper 2-sentence "
            f"'what + how to interpret' prose: {short_docs}"
        )


class TestReducers:
    def test_sum(self):
        vals = [MetricValue.of_scalar(2.0), MetricValue.of_scalar(3.0)]
        assert reduce_values(AggregationStrategy.SUM, vals) == 5.0

    def test_mean(self):
        vals = [MetricValue.of_scalar(1.0), MetricValue.of_scalar(3.0)]
        assert reduce_values(AggregationStrategy.MEAN, vals) == 2.0

    def test_max(self):
        vals = [MetricValue.of_scalar(1.0), MetricValue.of_scalar(9.0)]
        assert reduce_values(AggregationStrategy.MAX, vals) == 9.0

    def test_weighted_mean(self):
        # (value, weight): (10, 3) and (20, 1) -> (10*3 + 20*1)/(3+1) = 50/4
        vals = [MetricValue.of_weighted(10.0, 3), MetricValue.of_weighted(20.0, 1)]
        assert reduce_values(AggregationStrategy.WEIGHTED_MEAN, vals) == 12.5

    def test_p95_pools_samples_not_summaries(self):
        # Two groups each contributing 10 samples. True p95 over the pooled 20
        # values, NOT p95-of-per-group-p95 (obs 8). Pooled = 1..20; p95 ~ 19.05.
        g1 = MetricValue.of_samples([float(i) for i in range(1, 11)])
        g2 = MetricValue.of_samples([float(i) for i in range(11, 21)])
        result = reduce_values(AggregationStrategy.P95, [g1, g2])
        # numpy's linear-interpolation percentile of 1..20 at 95 = 19.05
        assert abs(result - 19.05) < 1e-6

    def test_p50_median_over_pooled(self):
        g1 = MetricValue.of_samples([1.0, 2.0, 3.0])
        g2 = MetricValue.of_samples([4.0])
        # pooled = [1,2,3,4], median = 2.5
        assert reduce_values(AggregationStrategy.P50, [g1, g2]) == 2.5

    def test_ratio_sum_over_sum_not_mean_of_ratios(self):
        # Group A: 1/10 = 0.1. Group B: 9/10 = 0.9. mean-of-ratios = 0.5 (WRONG).
        # Correct ratio-of-sums = (1+9)/(10+10) = 10/20 = 0.5 here by coincidence;
        # use uneven denominators to distinguish:
        # A: 1/2 = 0.5, B: 1/100 = 0.01. mean-of-ratios = 0.255 (WRONG).
        # correct = (1+1)/(2+100) = 2/102 ~ 0.019607...
        vals = [MetricValue.of_ratio(1, 2), MetricValue.of_ratio(1, 100)]
        result = reduce_values(AggregationStrategy.RATIO, vals)
        assert abs(result - (2 / 102)) < 1e-9

    def test_ratio_zero_total_is_zero(self):
        vals = [MetricValue.of_ratio(0, 0)]
        assert reduce_values(AggregationStrategy.RATIO, vals) == 0.0

    def test_ratio_of_sums(self):
        # A: part=1, total=4. B: part=3, total=4. sum(part)/sum(total)=4/8=0.5.
        vals = [
            MetricValue.of_part_total(1.0, 4.0),
            MetricValue.of_part_total(3.0, 4.0),
        ]
        assert reduce_values(AggregationStrategy.RATIO_OF_SUMS, vals) == 0.5

    def test_ratio_of_sums_zero_total_is_zero(self):
        vals = [MetricValue.of_part_total(0.0, 0.0)]
        assert reduce_values(AggregationStrategy.RATIO_OF_SUMS, vals) == 0.0

    def test_passthrough_requires_single_value(self):
        assert (
            reduce_values(AggregationStrategy.PASSTHROUGH, [MetricValue.of_scalar(7.0)])
            == 7.0
        )

    def test_passthrough_multiple_values_raises(self):
        vals = [MetricValue.of_scalar(1.0), MetricValue.of_scalar(2.0)]
        with pytest.raises(ValueError, match="passthrough"):
            reduce_values(AggregationStrategy.PASSTHROUGH, vals)

    def test_wrong_carrier_for_strategy_raises(self):
        # A sum reducer handed a samples-carrying MetricValue is a bug.
        with pytest.raises(ValueError, match="expected"):
            reduce_values(AggregationStrategy.SUM, [MetricValue.of_samples([1.0])])

    def test_empty_list_returns_zero(self):
        assert reduce_values(AggregationStrategy.MEAN, []) == 0.0
        assert reduce_values(AggregationStrategy.RATIO, []) == 0.0


class TestAggregateBySpec:
    def test_reduces_each_key_by_its_strategy(self):
        # Two groups. reward -> weighted_mean; finish_reason -> ratio.
        per_group = [
            {
                "learning/reward/mean": MetricValue.of_weighted(10.0, 2),
                "sampler/finish_reason/STOP": MetricValue.of_ratio(2, 2),
            },
            {
                "learning/reward/mean": MetricValue.of_weighted(20.0, 1),
                "sampler/finish_reason/STOP": MetricValue.of_ratio(1, 3),
            },
        ]
        result = aggregate_by_spec(per_group)
        # weighted mean: (10*2 + 20*1)/(2+1) = 40/3
        assert abs(result["learning/reward/mean"] - (40 / 3)) < 1e-9
        # ratio: (2+1)/(2+3) = 3/5 = 0.6
        assert abs(result["sampler/finish_reason/STOP"] - 0.6) < 1e-9

    def test_key_absent_from_some_groups_ok(self):
        # pass_at_k only present in group 1; still reduces over the one value.
        per_group = [
            {"learning/pass_at_k": MetricValue.of_scalar(1.0)},
            {"learning/sampling_accuracy": MetricValue.of_scalar(0.5)},
        ]
        result = aggregate_by_spec(per_group)
        assert result["learning/pass_at_k"] == 1.0
        assert result["learning/sampling_accuracy"] == 0.5

    def test_unknown_scalar_key_falls_back_to_mean(self):
        # Recipe-emitted keys (e.g. legacy ``env/*`` names that no recipe
        # has migrated + registered yet) that the framework registry
        # doesn't know about must NOT crash the pipeline. The framework
        # doesn't register recipe keys; the batch reducer implicitly
        # averages them (the pre-existing behavior: SUM_METRICS was opt-in
        # and everything else was averaged).
        per_group = [
            {"env/some_unregistered_probe": MetricValue.of_scalar(0.2)},
            {"env/some_unregistered_probe": MetricValue.of_scalar(0.4)},
            {"env/some_unregistered_probe": MetricValue.of_scalar(0.6)},
        ]
        result = aggregate_by_spec(per_group)
        assert abs(result["env/some_unregistered_probe"] - 0.4) < 1e-9

    def test_unknown_non_scalar_key_still_errors(self):
        # Fallback only covers scalar-carrying MetricValues (the shape produced
        # by ``_compute_group_metrics`` wrapping bare aggregator floats).
        # If a recipe explicitly emits a non-scalar MetricValue for an
        # unregistered key, that's a programmer error (they meant to register
        # a spec) and the mean reducer will surface it via _require().
        per_group = [{"unknown/samples": MetricValue.of_samples([1.0, 2.0])}]
        with pytest.raises(ValueError, match="expected"):
            aggregate_by_spec(per_group)

    def test_unknown_key_warns_once(self, caplog: pytest.LogCaptureFixture):
        # Warning fires the first time a given unknown key is seen, then is
        # suppressed to keep training-loop logs quiet.
        import logging

        # Fresh key ensures the module-level "already-warned" cache doesn't
        # swallow the first warning this test cares about.
        key = "test_warn_once/probe_metric_xyz"
        with caplog.at_level(logging.WARNING, logger="core.metric_spec"):
            aggregate_by_spec([{key: MetricValue.of_scalar(1.0)}])
            aggregate_by_spec([{key: MetricValue.of_scalar(2.0)}])
            aggregate_by_spec([{key: MetricValue.of_scalar(3.0)}])
        matches = [r for r in caplog.records if key in r.message]
        assert len(matches) == 1, (
            f"expected exactly one warning for unknown key {key!r}, "
            f"got {len(matches)}: {[r.message for r in matches]}"
        )

    def test_empty_input_returns_empty(self):
        assert aggregate_by_spec([]) == {}

    def test_accepts_bare_floats_from_legacy_recipes(self):
        # Recipe agents that OVERRIDE group_rollout can return
        # dict[str, float] directly rather than routing through
        # Agent._compute_group_metrics's MetricValue wrap step. The two
        # pipeline entry points (rollout_pipeline.log_batch,
        # rollout_pipeline._log_step_sampling_metrics, and eval_only) must
        # therefore accept bare-float dicts without crashing.
        result = aggregate_by_spec(
            [
                {
                    "env/average_reward": 0.5,  # unregistered scalar -> MEAN
                    "learning/sampling_accuracy": 0.3,  # registered as MEAN
                },
                {
                    "env/average_reward": 1.5,
                    "learning/sampling_accuracy": 0.7,
                },
            ]
        )
        assert abs(result["env/average_reward"] - 1.0) < 1e-9
        assert abs(result["learning/sampling_accuracy"] - 0.5) < 1e-9

    def test_accepts_bare_ints_from_legacy_recipes(self):
        # Bare-int values (e.g. count-shaped scalars) also
        # wrap cleanly as MetricValue.of_scalar(float(...)) at the entry.
        # Uses an unregistered scalar-friendly key so the MEAN fallback
        # applies. Bare ints for RATIO-family keys would be caller error
        # (not the wrap layer's job to defend against).
        result = aggregate_by_spec([{"env/steps": 5}, {"env/steps": 7}])
        assert result["env/steps"] == 6.0

    def test_bare_scalar_meeting_non_scalar_strategy_falls_back_to_mean(
        self, caplog: pytest.LogCaptureFixture
    ):
        # A recipe can emit registered non-scalar-strategy keys as bare
        # numerics from its overridden group_rollout (e.g.
        # ``sampler/policy_step`` -> WEIGHTED_MEAN, or
        # ``sampler/trajectory_status/*`` -> RATIO). Once the pipeline was
        # wired through ``aggregate_by_spec``, these keys were reaching the
        # strategy-typed reducers, which read ``.weighted_pair`` /
        # ``.count_total`` etc. and crashed the batch.
        #
        # Contract post-fix: when a bare scalar meets a non-scalar strategy,
        # fall back to MEAN with a one-shot WARNING per key — same
        # philosophy as the unregistered-key MEAN fallback. Semantically
        # imperfect (a bare policy_step means "mean of ts_ids for this
        # group", not "weighted mean across groups"), but it matches the
        # legacy pre-B behavior for these emitters and unblocks such a
        # recipe while it migrates to emit proper MetricValue shapes.
        import logging

        # policy_step is registered WEIGHTED_MEAN. Bare-float emission
        # should coerce to MEAN across groups.
        with caplog.at_level(logging.WARNING, logger="core.metric_spec"):
            result = aggregate_by_spec(
                [
                    {"sampler/policy_step": 100.5},
                    {"sampler/policy_step": 200.5},
                ]
            )
        assert abs(result["sampler/policy_step"] - 150.5) < 1e-9
        # trajectory_status is RATIO strategy. Bare-int emission should
        # coerce to MEAN too.
        result2 = aggregate_by_spec(
            [
                {"sampler/trajectory_status/OK": 2},
                {"sampler/trajectory_status/OK": 4},
            ]
        )
        assert abs(result2["sampler/trajectory_status/OK"] - 3.0) < 1e-9

    def test_bare_scalar_shape_mismatch_warns_once_per_key(
        self, caplog: pytest.LogCaptureFixture
    ):
        # The shape-mismatch fallback warns once per key, throttled via a
        # module-level cache (mirrors the unregistered-key throttle).
        import logging

        # Fresh key to escape any test-order contamination of the throttle.
        # Register it first so we're on the shape-mismatch path, not the
        # unknown-key path.
        from core.metric_spec import METRIC_SPECS

        key = "test_shape_mismatch_warn_once/policy_step_probe"
        METRIC_SPECS[key] = MetricSpec(
            name=key,
            strategy=AggregationStrategy.WEIGHTED_MEAN,
            doc="probe",
        )
        try:
            with caplog.at_level(logging.WARNING, logger="core.metric_spec"):
                aggregate_by_spec([{key: 1.0}])
                aggregate_by_spec([{key: 2.0}])
                aggregate_by_spec([{key: 3.0}])
            # Warning message contains "bare scalars" and the key name.
            matches = [
                r
                for r in caplog.records
                if key in r.message and "bare scalars" in r.message
            ]
            assert len(matches) == 1, (
                f"expected exactly one shape-mismatch warning for {key!r}, "
                f"got {len(matches)}: {[r.message for r in matches]}"
            )
        finally:
            METRIC_SPECS.pop(key, None)

    def test_wrap_bare_scalar_drops_nan_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ):
        # Pre-B ``aggregate_rollout_metrics`` silently ``math.isnan``-skipped
        # NaN bare-float values before summing. Post-B, ``MetricValue.__post_init__``
        # rejects NaN — so wrapping a bare NaN would ValueError. Restore the
        # legacy skip semantic AT THE WRAP LAYER: bare NaN gets dropped
        # (with a one-shot WARNING per key for observability), so
        # aggregate_by_spec doesn't crash the whole batch when a recipe
        # aggregator emits a NaN from a partial-data calculation (e.g.
        # mean of an all-NaN reward list).
        import logging

        with caplog.at_level(logging.WARNING, logger="core.metric_spec"):
            # Mix a NaN and a real value under an unregistered key.
            result = aggregate_by_spec(
                [
                    {"env/partial_nan": math.nan},
                    {"env/partial_nan": 1.5},
                    {"env/partial_nan": 2.5},
                ]
            )
        # NaN dropped -> mean over [1.5, 2.5] = 2.0.
        assert abs(result["env/partial_nan"] - 2.0) < 1e-9

    def test_wrap_bare_scalar_all_nan_omits_key(self):
        # All-NaN contributions are all dropped -> the key never enters
        # by_key -> nothing emitted for it. Matches "no valid data means
        # no metric" semantic (better than emitting a misleading 0.0).
        result = aggregate_by_spec(
            [
                {"env/all_nan_probe": math.nan},
                {"env/all_nan_probe": math.nan},
            ]
        )
        assert "env/all_nan_probe" not in result

    def test_mixed_bare_and_metricvalue_ok(self):
        # A single group-dict can contain both bare floats and MetricValues
        # (e.g. a recipe's overridden group_rollout emits both).
        result = aggregate_by_spec(
            [
                {
                    "learning/sampling_accuracy": 0.5,  # bare float
                    "learning/reward/mean": MetricValue.of_weighted(1.0, 3),
                },
                {
                    "learning/sampling_accuracy": 0.7,
                    "learning/reward/mean": MetricValue.of_weighted(2.0, 1),
                },
            ]
        )
        assert abs(result["learning/sampling_accuracy"] - 0.6) < 1e-9
        # weighted mean: (1*3 + 2*1)/(3+1) = 5/4 = 1.25
        assert abs(result["learning/reward/mean"] - 1.25) < 1e-9


class TestMergeGroupMetrics:
    def test_merges_disjoint_keys(self):
        a = {"learning/reward/mean": MetricValue.of_weighted(1.0, 1)}
        b = {"learning/pass_at_k": MetricValue.of_scalar(1.0)}
        merged = merge_group_metrics([a, b])
        assert set(merged) == {"learning/reward/mean", "learning/pass_at_k"}

    def test_same_stage_collision_raises(self):
        # Two aggregators in the SAME group both emit the same key -> raise.
        a = {"sampler/generation_tokens_sum": MetricValue.of_scalar(5.0)}
        b = {"sampler/generation_tokens_sum": MetricValue.of_scalar(9.0)}
        with pytest.raises(ValueError, match="collision"):
            merge_group_metrics([a, b])

    def test_empty_inputs_merge_to_empty(self):
        assert merge_group_metrics([]) == {}
        assert merge_group_metrics([{}, {}]) == {}


class TestRegistryF6Boundary:
    """Registry boundary: the framework knows nothing about recipe names.

    Recipe MetricSpecs are registered by each recipe itself (via
    ``METRIC_SPECS.update({...})`` at import time), so the framework's
    literal registry must contain zero ``recipe/*`` keys.

    Uses ``_FRAMEWORK_SPEC_KEYS`` — a snapshot taken at framework-module load
    time, BEFORE any recipe module has run its side-effect
    ``METRIC_SPECS.update({...})``. Checking the runtime ``METRIC_SPECS``
    dict directly would fail the moment any test in the session imports a
    recipe aggregator module, because the import performs the registration.
    """

    def test_metric_specs_contains_no_recipe_keys(self):
        recipe_keys = [k for k in _FRAMEWORK_SPEC_KEYS if k.startswith("recipe/")]
        assert recipe_keys == [], (
            f"framework registry must not carry recipe/* keys; "
            f"found: {recipe_keys}. Move these registrations to the owning "
            f"recipe's own aggregator module."
        )


# Recipe-owned spec registrations (each recipe's aggregator module registers
# its ``recipe/<name>/*`` MetricSpec(s) at import time) are verified alongside
# each recipe's own tree.
