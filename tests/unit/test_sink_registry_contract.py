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

"""Contract test: every metric key emitted to SinkDispatcher.log_metrics in
production has a corresponding MetricSpec in the registry.

This is the regression fence that would have caught the pre-fix bug where
train_and_eval.py emits 'train/learning/reward/mean' but the registry
only had 'learning/reward/mean' — causing MLPlatformSink to silently
reject the metric (spec=None path in should_accept).

The keys enumerated here are sourced by manual inventory of every literal
metric key that appears in a production log_metrics(...) call:
  - src/pipeline/train_and_eval.py (Site A)
  - src/pipeline/eval_only.py (Site C)
  - src/pipeline/sync_rollout_pipeline.py (Site B)

Names below are the POST-PREFIX shapes the sink actually sees. Site A
prepends 'train/', Site C prepends 'eval/', Site B mixes categorical +
hard-coded 'train/*' names.

Update the list when adding a new emit or renaming an existing one. If a
key here fails to resolve, EITHER register a MetricSpec in
core.metric_spec.METRIC_SPECS OR fix the emitter to use a registered
name — do NOT delete the entry from this list to make the test pass.
"""

import pytest

from core.metric_spec import spec_for

# =============================================================================
# Site A: src/pipeline/train_and_eval.py:283-302
# Every key gets `f"train/{k}"` prefix before log_metrics.
# =============================================================================
_SITE_A_KEYS = [
    # From _pipeline_passthrough dict (train_and_eval.py:287-292)
    "train/pipeline/train_step_time",
    "train/pipeline/wait_for_rollout_time",
    "train/pipeline/iterator_current",
    # Merged into _pipeline_passthrough from
    # pipeline.sample_pipeline_gauges(); async-only, so a sync run emits
    # neither.
    "train/pipeline/buffer_utilization",
    "train/pipeline/group_drop_ratio",
    "train/pipeline/group_zero_variance_ratio",
    "train/pipeline/group_drop_reason/zero_variance",
    # From step_metrics = _aggregate_step_metrics(...) which calls
    # aggregate_by_spec on per-group metrics from batch_items.
    "train/sampler/policy_step",
    "train/pipeline/rollout_wall_time_avg",
    "train/pipeline/rollout_wall_time_p95",
    "train/learning/reward/mean",
    "train/learning/sampling_accuracy",
    "train/learning/pass_at_k",
    # Derived staleness (train_and_eval.py:296-300)
    "train/staleness_steps",
    # From train_metrics = trainer.train_step(...) return dict — these are
    # registered under their client_key in TRAINER_METRICS. Wrapped with
    # train/ prefix at emit time.
    # NOTE: 'train/loss' is intentionally absent — an alternate backend's
    # synthetic loss hits the lenient-drop path (warn+drop once per
    # process) rather than taking up a spec slot no production consumer
    # justified. Real Vertex does not disclose loss over the wire.
    "train/train_step_seconds",
    "train/billable_training_tokens",
    "train/server_train_step",  # client_key='server_train_step' from step wire metric
    "train/training_tokens_before_merge",
    "train/mean_abs_advantage",
    # From a trainer's train_step (VertexMsrlRestClient and the alternate
    # backends alike): keys the trainer puts on the metrics dict directly
    # (not via the TRAINER_METRICS wire mapping). Missing entries here
    # would silently drop metrics in MLPlatformSink.
    "train/completed_train_step_id",
    "train/trajectory/missing_candidate_id",
    "train/audit/used_candidates",
    "train/audit/rejected_candidates",
    # Dynamic-suffix entries: any suffix works for the fence — the strings
    # below are representative production values (RATE_LIMITED is a common
    # rejection reason, STOP is the default agent finish reason). If a
    # future rename lands upstream, refresh these values rather than
    # deleting the assertion — the fence's purpose is to verify the /*
    # spec resolves, not to pin specific suffix strings.
    "train/audit/rejected_reason/RATE_LIMITED",
    "train/sampler/finish_reason/STOP",
]

# =============================================================================
# Site B: src/pipeline/sync_rollout_pipeline.py:371-375
# aggregate_by_spec(filtered_metrics) output goes through as-is (categorical);
# plus 5 hard-coded train/* keys.
# =============================================================================
_SITE_B_CATEGORICAL_KEYS = [
    # aggregate_by_spec output — framework metrics under categorical names
    "learning/reward/mean",
    "learning/sampling_accuracy",
    "learning/pass_at_k",
    "sampler/policy_step",
    "sampler/finish_reason/STOP",
    "sampler/trajectory_status/OK",
    "sampler/generation_tokens_sum",
    "sampler/peak_context_tokens_mean",
    "pipeline/sample_time_avg",
    "pipeline/env_time_avg",
    "pipeline/sample_time_p95",
    "pipeline/env_time_p95",
    "pipeline/sample_time_fraction",
    "pipeline/rollout_wall_time_avg",
    "pipeline/rollout_wall_time_p95",
    # Site B literals: the filter gauges sync emits from inside next_batch
    # (async emits the train/-prefixed twins at Site A).
    "pipeline/group_drop_ratio",
    "pipeline/group_zero_variance_ratio",
    "pipeline/group_drop_reason/zero_variance",
]

_SITE_B_HARDCODED_TRAIN_KEYS = [
    "train/running_pass_rate",
    "train/rollout_rounds",
    "train/rollout_tasks_total",
    "train/first_round_rollout_tasks",
    "train/pipeline/iterator_current",  # collides in shape with Site A's, same value
]

# =============================================================================
# Site C: src/pipeline/eval_only.py:350
# Every key from aggregate_by_spec(results) gets `f"eval/{k}"` prefix.
# =============================================================================
_SITE_C_KEYS = [
    "eval/learning/reward/mean",
    "eval/learning/sampling_accuracy",
    "eval/learning/pass_at_k",
    "eval/sampler/policy_step",
    "eval/sampler/finish_reason/STOP",
    "eval/sampler/trajectory_status/OK",
    "eval/sampler/generation_tokens_sum",
    "eval/sampler/peak_context_tokens_mean",
    "eval/pipeline/sample_time_avg",
    "eval/pipeline/env_time_avg",
    "eval/pipeline/sample_time_p95",
    "eval/pipeline/env_time_p95",
    "eval/pipeline/sample_time_fraction",
    "eval/pipeline/rollout_wall_time_avg",
    "eval/pipeline/rollout_wall_time_p95",
]

PRODUCTION_EMITTED_KEYS = (
    _SITE_A_KEYS
    + _SITE_B_CATEGORICAL_KEYS
    + _SITE_B_HARDCODED_TRAIN_KEYS
    + _SITE_C_KEYS
)


@pytest.mark.parametrize("key", PRODUCTION_EMITTED_KEYS)
def test_every_production_emit_key_resolves_to_a_spec(key):
    """spec_for() must resolve every key emitted to SinkDispatcher in production.

    Failure means MLPlatformSink will silently reject the key at runtime
    (spec=None path in should_accept, sinks.py:121). Fix by either:
      (a) registering a MetricSpec in core.metric_spec.METRIC_SPECS
          matching this key exactly, or
      (b) changing the production emitter to use a name that is already
          registered.

    Do NOT delete the key from PRODUCTION_EMITTED_KEYS to make this pass —
    that would remove the regression fence and re-open the silent-drop
    class of bug.
    """
    try:
        spec = spec_for(key)
    except KeyError:
        pytest.fail(
            f"Production emit key {key!r} has no MetricSpec. "
            f"MLPlatformSink will silently reject it. Register in "
            f"core.metric_spec.METRIC_SPECS or fix the emitter to use "
            f"a registered name."
        )
    assert spec is not None
    # Sanity check: the resolved spec's name should match either the exact
    # key or (for dynamic-suffix families) a prefix of the key.
    assert spec.name == key or key.startswith(spec.name), (
        f"spec_for({key!r}) returned a spec named {spec.name!r} that "
        f"neither matches the key exactly nor is a prefix of it"
    )
