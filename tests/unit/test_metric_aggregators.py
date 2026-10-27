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

"""Tests for the MetricAggregator protocol + default aggregators."""

import math

from core import atif_extras
from core.atif_trajectory import (
    ScoredTrajectory,
    Reward,
    StepStats,
    Step,
    TrajectoryStep as AtifTrajectoryStep,
)
from core.metric_aggregators import (
    DEFAULT_METRIC_AGGREGATORS,
    CoreRewardMetricAggregator,
)
from core.metric_spec import AggregationStrategy, MetricSpec, MetricValue
from core.rollout_status import RolloutStatus
from core.atif_trajectory import (
    Agent as AtifAgent,
    Trajectory as AtifTrajectory,
)


def _scored(
    *,
    reward: float,
    is_correct: bool | None = None,
    status: RolloutStatus = RolloutStatus.OK,
) -> ScoredTrajectory:
    us = Step(step_id=1, source="user", message="p")
    agent_step = AtifTrajectoryStep.from_wire(
        step_id=2, generation="g", stats=StepStats()
    )
    traj = AtifTrajectory(
        schema_version="ATIF-v1.7",
        session_id="test-session",
        agent=AtifAgent(name="t", version="0", model_name="m"),
        steps=[us, agent_step],
    )
    atif_extras.set_rollout_status(traj, status)
    if is_correct is not None:
        atif_extras.set_is_correct(traj, is_correct)
    return ScoredTrajectory(trajectory=traj, reward=Reward(reward=reward))


def _traj_with_train_step_ids(ids: list[int | None]) -> ScoredTrajectory:
    """ScoredTrajectory with one agent step per entry in ``ids``; each step's
    ``train_step_id`` is set to that entry (``None`` => step carries no
    train_step_id). Lets a single trajectory span multiple policy steps so the
    count-weighted mean can be exercised within one trajectory."""
    us = Step(step_id=1, source="user", message="p")
    agent_steps: list = []
    for i, ts_id in enumerate(ids):
        stats = StepStats(train_step_id=ts_id) if ts_id is not None else StepStats()
        agent_steps.append(
            AtifTrajectoryStep.from_wire(step_id=2 + i, generation="g", stats=stats)
        )
    traj = AtifTrajectory(
        schema_version="ATIF-v1.7",
        session_id="test-session",
        agent=AtifAgent(name="t", version="0", model_name="m"),
        steps=[us, *agent_steps],
    )
    atif_extras.set_rollout_status(traj, RolloutStatus.OK)
    return ScoredTrajectory(trajectory=traj, reward=Reward(reward=0.0))


def test_metric_aggregator_protocol_has_specs():
    """Every default aggregator exposes specs() returning a list of MetricSpec."""
    for agg in DEFAULT_METRIC_AGGREGATORS:
        specs = agg.specs()
        assert isinstance(specs, list)
        for s in specs:
            assert isinstance(s, MetricSpec)


def test_core_reward_aggregator_emits_metric_values():
    """CoreRewardMetricAggregator emits learning/* keys as strategy-typed
    MetricValues: reward as weighted pair, accuracy/pass_at_k as scalars."""
    # 2 trajectories, rewards [2.0, 4.0] -> group mean 3.0, n=2.
    # is_correct [True, False] -> accuracy 0.5, pass_at_k (any) = 1.0.
    trajs = [
        _scored(reward=2.0, is_correct=True),
        _scored(reward=4.0, is_correct=False),
    ]
    result = CoreRewardMetricAggregator()(trajs)
    assert result["learning/reward/mean"] == MetricValue.of_weighted(3.0, 2)
    assert result["learning/sampling_accuracy"] == MetricValue.of_scalar(0.5)
    assert result["learning/pass_at_k"] == MetricValue.of_scalar(1.0)


def test_core_reward_aggregator_specs():
    specs = {s.name: s for s in CoreRewardMetricAggregator().specs()}
    assert specs["learning/reward/mean"].strategy is (AggregationStrategy.WEIGHTED_MEAN)
    assert specs["learning/sampling_accuracy"].strategy is AggregationStrategy.MEAN
    assert specs["learning/pass_at_k"].strategy is AggregationStrategy.MEAN


def test_core_reward_aggregator_does_not_emit_legacy_env_keys():
    """Negative-assertion guard against a silent double-emit under both
    the old ``env/*`` and new ``learning/*`` names. The rename SHOULD
    have deleted the env/* emissions from
    ``CoreRewardMetricAggregator``; if they crept back in, the same value
    would flow to two different sink columns and dashboards would show
    duplicated series.

    Complements ``test_core_reward_aggregator_emits_metric_values`` (which
    asserts the NEW keys ARE emitted).
    """
    trajs = [
        _scored(reward=2.0, is_correct=True),
        _scored(reward=4.0, is_correct=False),
    ]
    result = CoreRewardMetricAggregator()(trajs)
    for legacy_key in ("env/average_reward", "env/sampling_accuracy", "env/pass_at_k"):
        assert legacy_key not in result, (
            f"legacy key {legacy_key!r} should have been renamed to "
            f"learning/*; found double-emit"
        )


def test_core_reward_aggregator_omits_reward_mean_on_all_nan_rewards():
    """When all rewards are NaN (e.g. sampler timeout), the aggregator omits
    learning/reward/mean rather than emitting a NaN-carrying MetricValue.
    This eliminates the need for a NaN skip in Agent._compute_group_metrics."""
    trajs = [
        _scored(reward=math.nan, is_correct=False),
        _scored(reward=math.nan, is_correct=False),
    ]
    result = CoreRewardMetricAggregator()(trajs)
    assert "learning/reward/mean" not in result
    # But accuracy/pass_at_k still emit (based on is_correct, not reward).
    assert "learning/sampling_accuracy" in result
    assert "learning/pass_at_k" in result


def test_sampler_policy_step_aggregator_emits_weighted_metric_value():
    """SamplerPolicyStepMetricAggregator emits an of_weighted
    MetricValue carrying (mean_train_step_id, n_agent_steps). Batch stage
    can then compute a true count-weighted mean across groups -- groups
    with more agent steps contribute proportionally."""
    from core.metric_aggregators import SamplerPolicyStepMetricAggregator

    # 2 trajectories: first has agent steps at train_step_ids [10, 12, 14];
    # second has [16, 18]. Total = 70, count = 5, mean = 14.0.
    trajectories = [
        _traj_with_train_step_ids([10, 12, 14]),
        _traj_with_train_step_ids([16, 18]),
    ]
    result = SamplerPolicyStepMetricAggregator()(trajectories)
    assert result == {"sampler/policy_step": MetricValue.of_weighted(14.0, 5)}


def test_sampler_policy_step_aggregator_specs():
    from core.metric_aggregators import SamplerPolicyStepMetricAggregator

    specs = {s.name: s for s in SamplerPolicyStepMetricAggregator().specs()}
    assert specs["sampler/policy_step"].strategy is (AggregationStrategy.WEIGHTED_MEAN)
    assert specs["sampler/policy_step"].emit_to_ml_platform is False


def _traj_with_finish_reasons(reasons: list[str | None]) -> ScoredTrajectory:
    """ScoredTrajectory with one agent step per entry in ``reasons``; each
    step's ``finish_reason`` is set to that entry (``None`` => no
    finish_reason). Lets a single trajectory span multiple finish reasons so
    ratio math can be exercised within one trajectory."""
    us = Step(step_id=1, source="user", message="p")
    agent_steps: list = []
    for i, fr in enumerate(reasons):
        stats = StepStats(finish_reason=fr) if fr is not None else StepStats()
        agent_steps.append(
            AtifTrajectoryStep.from_wire(step_id=2 + i, generation="g", stats=stats)
        )
    traj = AtifTrajectory(
        schema_version="ATIF-v1.7",
        session_id="test-session",
        agent=AtifAgent(name="t", version="0", model_name="m"),
        steps=[us, *agent_steps],
    )
    atif_extras.set_rollout_status(traj, RolloutStatus.OK)
    return ScoredTrajectory(trajectory=traj, reward=Reward(reward=0.0))


def _traj_with_status(status: RolloutStatus) -> ScoredTrajectory:
    """ScoredTrajectory with a single agent step and given trajectory-level
    ``RolloutStatus`` — for exercising RolloutStatusMetricAggregator."""
    us = Step(step_id=1, source="user", message="p")
    agent_step = AtifTrajectoryStep.from_wire(
        step_id=2, generation="g", stats=StepStats()
    )
    traj = AtifTrajectory(
        schema_version="ATIF-v1.7",
        session_id="test-session",
        agent=AtifAgent(name="t", version="0", model_name="m"),
        steps=[us, agent_step],
    )
    atif_extras.set_rollout_status(traj, status)
    return ScoredTrajectory(trajectory=traj, reward=Reward(reward=0.0))


def test_finish_reason_aggregator_emits_ratio_intermediates():
    """FinishReasonMetricAggregator emits of_ratio(count, total) per reason,
    where total = all agent steps in the group."""
    from core.metric_aggregators import FinishReasonMetricAggregator

    trajs = [_traj_with_finish_reasons(["STOP", "STOP", "MAX_TOKENS"])]
    result = FinishReasonMetricAggregator()(trajs)
    assert result["sampler/finish_reason/STOP"] == MetricValue.of_ratio(2, 3)
    assert result["sampler/finish_reason/MAX_TOKENS"] == MetricValue.of_ratio(1, 3)


def test_finish_reason_batch_ratios_sum_to_one():
    """Two groups where every reason is present in every group; per-reason
    ratios sum to 1.0 across the batch (uneven groups, shared denominator per
    reason)."""
    from core.metric_aggregators import FinishReasonMetricAggregator
    from core.metric_spec import aggregate_by_spec

    # g1 total=3 (STOP,STOP,MAX_TOKENS); g2 total=2 (STOP,MAX_TOKENS).
    # Both reasons appear in both groups so each reason's batch denominator = 5.
    # Batch: STOP = (2+1)/(3+2) = 3/5 = 0.6; MAX_TOKENS = (1+1)/(3+2) = 2/5 = 0.4.
    g1 = FinishReasonMetricAggregator()(
        [_traj_with_finish_reasons(["STOP", "STOP", "MAX_TOKENS"])]
    )
    g2 = FinishReasonMetricAggregator()(
        [_traj_with_finish_reasons(["STOP", "MAX_TOKENS"])]
    )
    batch = aggregate_by_spec([g1, g2])
    assert abs(batch["sampler/finish_reason/STOP"] - 0.6) < 1e-9
    assert abs(batch["sampler/finish_reason/MAX_TOKENS"] - 0.4) < 1e-9
    assert (
        abs(
            batch["sampler/finish_reason/STOP"]
            + batch["sampler/finish_reason/MAX_TOKENS"]
            - 1.0
        )
        < 1e-9
    )


def test_finish_reason_batch_ratios_sum_to_one_sparse_key():
    """A reason present in only SOME groups must still share the global
    agent-step denominator so all per-reason ratios sum to 1.0.

    g1 total=3 (STOP,STOP,MAX_TOKENS); g2 total=2 (STOP,STOP) -- MAX_TOKENS
    is absent from g2. Global agent steps = 5. Correct ratios:
    STOP = 4/5 = 0.8, MAX_TOKENS = 1/5 = 0.2, sum = 1.0.
    (The buggy sparse-key reducer gave MAX_TOKENS = 1/3, summing to >1.)
    """
    from core.metric_aggregators import FinishReasonMetricAggregator
    from core.metric_spec import aggregate_by_spec

    g1 = FinishReasonMetricAggregator()(
        [_traj_with_finish_reasons(["STOP", "STOP", "MAX_TOKENS"])]
    )
    g2 = FinishReasonMetricAggregator()([_traj_with_finish_reasons(["STOP", "STOP"])])
    batch = aggregate_by_spec([g1, g2])
    assert abs(batch["sampler/finish_reason/STOP"] - 0.8) < 1e-9
    assert abs(batch["sampler/finish_reason/MAX_TOKENS"] - 0.2) < 1e-9
    assert (
        abs(
            batch["sampler/finish_reason/STOP"]
            + batch["sampler/finish_reason/MAX_TOKENS"]
            - 1.0
        )
        < 1e-9
    )


def test_trajectory_status_aggregator_emits_ratio_intermediates():
    from core.metric_aggregators import RolloutStatusMetricAggregator

    # 3 trajectories: OK, OK, sampling_failure_timeout. total = 3.
    trajs = [
        _traj_with_status(RolloutStatus.OK),
        _traj_with_status(RolloutStatus.OK),
        _traj_with_status(RolloutStatus.SAMPLING_FAILURE_TIMEOUT),
    ]
    result = RolloutStatusMetricAggregator()(trajs)
    assert result["sampler/trajectory_status/OK"] == MetricValue.of_ratio(2, 3)
    # sampling_failure roll-up still emitted, also as a ratio over total trajs.
    assert result["sampler/trajectory_status/sampling_failure"] == MetricValue.of_ratio(
        1, 3
    )


def _traj_with_timings(
    sample_times: list[float], env_times: list[float]
) -> ScoredTrajectory:
    """ScoredTrajectory whose k-th agent step has
    ``sample_time_s=sample_times[k]`` and ``env_time_s=env_times[k]``.

    ``atif_extras`` has no setters for these fields (getters read
    ``step.extra["sample_time_s"]`` / ``["env_time_s"]``), so we write them
    directly onto ``step.extra`` — matching the on-the-wire shape.
    """
    assert len(sample_times) == len(env_times)
    us = Step(step_id=1, source="user", message="p")
    agent_steps: list = []
    for i, (s, e) in enumerate(zip(sample_times, env_times)):
        step = AtifTrajectoryStep.from_wire(
            step_id=2 + i, generation="g", stats=StepStats()
        )
        if step.extra is None:
            step.extra = {}
        step.extra["sample_time_s"] = s
        step.extra["env_time_s"] = e
        agent_steps.append(step)
    traj = AtifTrajectory(
        schema_version="ATIF-v1.7",
        session_id="test-session",
        agent=AtifAgent(name="t", version="0", model_name="m"),
        steps=[us, *agent_steps],
    )
    atif_extras.set_rollout_status(traj, RolloutStatus.OK)
    return ScoredTrajectory(trajectory=traj, reward=Reward(reward=0.0))


def test_timing_aggregator_emits_samples_for_p95():
    """TimingMetricAggregator emits per-step samples (of_samples) for p95 keys
    and a num/denom pair for sample_time_fraction, so the batch stage
    re-percentiles over pooled samples (fixes obs 8)."""
    from core.metric_aggregators import TimingMetricAggregator

    # One trajectory, 2 agent steps: sample_time [1.0, 3.0], env_time [0.5, 0.5].
    trajs = [_traj_with_timings(sample_times=[1.0, 3.0], env_times=[0.5, 0.5])]
    result = TimingMetricAggregator()(trajs)
    assert result["pipeline/sample_time_p95"] == MetricValue.of_samples([1.0, 3.0])
    assert result["pipeline/env_time_p95"] == MetricValue.of_samples([0.5, 0.5])
    assert result["pipeline/sample_time_avg"] == MetricValue.of_scalar(2.0)
    # fraction: sum_s=4.0, sum_e=1.0 -> num=4.0, denom=5.0
    assert result["pipeline/sample_time_fraction"] == MetricValue.of_part_total(
        4.0, 5.0
    )


def test_timing_p95_batch_pools_across_groups():
    """Two groups' samples pool at batch stage; p95 is over the pooled set,
    not mean-of-per-group-p95 (obs 8)."""
    from core.metric_aggregators import TimingMetricAggregator
    from core.metric_spec import aggregate_by_spec

    g1 = TimingMetricAggregator()(
        [
            _traj_with_timings(
                sample_times=[float(i) for i in range(1, 11)], env_times=[0.0] * 10
            )
        ]
    )
    g2 = TimingMetricAggregator()(
        [
            _traj_with_timings(
                sample_times=[float(i) for i in range(11, 21)], env_times=[0.0] * 10
            )
        ]
    )
    batch = aggregate_by_spec([g1, g2])
    # pooled sample_times = 1..20, p95 = 19.05
    assert abs(batch["pipeline/sample_time_p95"] - 19.05) < 1e-6


def test_timing_aggregator_specs():
    from core.metric_aggregators import TimingMetricAggregator

    specs = {s.name: s for s in TimingMetricAggregator().specs()}
    assert specs["pipeline/sample_time_p95"].strategy is AggregationStrategy.P95
    assert specs["pipeline/env_time_p95"].strategy is AggregationStrategy.P95
    assert specs["pipeline/sample_time_avg"].strategy is AggregationStrategy.MEAN
    assert specs["pipeline/sample_time_fraction"].strategy is (
        AggregationStrategy.RATIO_OF_SUMS
    )
    assert specs["pipeline/rollout_wall_time_avg"].strategy is AggregationStrategy.MEAN
    assert specs["pipeline/rollout_wall_time_p95"].strategy is AggregationStrategy.P95


def test_timing_aggregator_emits_rollout_wall_time_not_derivable_from_steps():
    """``rollout_wall_time_*`` reports the stamped per-TRAJECTORY wall clock,
    which is strictly larger than the sum of that trajectory's per-step
    timings — the gap is exactly what the per-step timers cannot see."""
    from core.metric_aggregators import TimingMetricAggregator

    t1 = _traj_with_timings(sample_times=[1.0, 3.0], env_times=[0.5, 0.5])
    t2 = _traj_with_timings(sample_times=[2.0], env_times=[1.0])
    atif_extras.set_rollout_wall_time_s(t1.trajectory, 30.0)
    atif_extras.set_rollout_wall_time_s(t2.trajectory, 10.0)

    result = TimingMetricAggregator()([t1, t2])
    assert result["pipeline/rollout_wall_time_avg"] == MetricValue.of_scalar(20.0)
    # Raw per-rollout samples, so the batch stage re-percentiles over the
    # pooled set rather than over per-group p95s.
    assert result["pipeline/rollout_wall_time_p95"] == MetricValue.of_samples(
        [30.0, 10.0]
    )
    # The sum-of-step-timings view is 5.0 and 3.0 for these two rollouts
    # (mean 4.0); the wall clock mean is 20.0. Were wall time derived from
    # the steps, the two would agree.
    assert result["pipeline/rollout_wall_time_avg"] != MetricValue.of_scalar(4.0)


def test_timing_aggregator_omits_wall_time_when_unstamped():
    """A trajectory with no ``rollout_wall_time_s`` contributes no sample, and
    the keys are omitted rather than reported as 0.0."""
    from core.metric_aggregators import TimingMetricAggregator

    result = TimingMetricAggregator()(
        [_traj_with_timings(sample_times=[1.0], env_times=[0.5])]
    )
    assert "pipeline/rollout_wall_time_avg" not in result
    assert "pipeline/rollout_wall_time_p95" not in result


def _traj_with_token_counts(
    prompt_counts: list[int | None], generation_counts: list[int | None]
) -> ScoredTrajectory:
    """ScoredTrajectory whose k-th agent step has ``prompt_token_count`` and
    ``generation_token_count`` set to the k-th entries. ``None`` => the
    corresponding field is unset on that step's ``StepStats`` (which
    ``from_wire`` then omits from ``step.extra``, so the getters return
    ``None``)."""
    assert len(prompt_counts) == len(generation_counts)
    us = Step(step_id=1, source="user", message="p")
    agent_steps: list = []
    for i, (p, gtok) in enumerate(zip(prompt_counts, generation_counts)):
        stats = StepStats(prompt_token_count=p, generation_token_count=gtok)
        agent_steps.append(
            AtifTrajectoryStep.from_wire(step_id=2 + i, generation="g", stats=stats)
        )
    traj = AtifTrajectory(
        schema_version="ATIF-v1.7",
        session_id="test-session",
        agent=AtifAgent(name="t", version="0", model_name="m"),
        steps=[us, *agent_steps],
    )
    atif_extras.set_rollout_status(traj, RolloutStatus.OK)
    return ScoredTrajectory(trajectory=traj, reward=Reward(reward=0.0))


def test_sampler_token_sum_aggregator_emits_generation_tokens_sum():
    """SamplerTokenSumMetricAggregator sums per-step generation_token_count
    over every agent step in every trajectory in the group. Emits as
    of_scalar so the SUM strategy reducer (which reads MetricValue.value)
    can sum across groups at batch stage."""
    from core.metric_aggregators import SamplerTokenSumMetricAggregator

    # 1 trajectory, 3 agent steps: gen_counts [100, 200, 50]; sum = 350.
    trajs = [
        _traj_with_token_counts(
            prompt_counts=[10, 20, 30], generation_counts=[100, 200, 50]
        )
    ]
    result = SamplerTokenSumMetricAggregator()(trajs)
    assert result["sampler/generation_tokens_sum"] == MetricValue.of_scalar(350)


def test_generation_tokens_sum_counts_harness_initiated_steps():
    """Context compaction emits a ``system``-sourced step carrying a real
    generation call. Those tokens were paid for, so the token SUM must
    include them even though the agent-step COUNT deliberately excludes
    harness-initiated steps."""
    from core.metric_aggregators import SamplerTokenSumMetricAggregator

    traj = _traj_with_token_counts(prompt_counts=[10, 20], generation_counts=[100, 200])
    compaction = AtifTrajectoryStep.from_wire(
        step_id=99,
        generation="SUMMARY",
        stats=StepStats(prompt_token_count=800, generation_token_count=500),
    )
    compaction.source = "system"
    traj.trajectory.steps.append(compaction)

    result = SamplerTokenSumMetricAggregator()([traj])
    assert result["sampler/generation_tokens_sum"] == MetricValue.of_scalar(800)


def test_sampler_token_sum_aggregator_emits_peak_context_tokens_mean():
    """Peak-context = last-agent-step's (prompt + generation) per trajectory;
    aggregator emits the MEAN across trajectories as of_scalar so the MEAN
    strategy reducer averages across groups at batch."""
    from core.metric_aggregators import SamplerTokenSumMetricAggregator

    # Traj A: last step (prompt=100, gen=50) -> 150.
    # Traj B: last step (prompt=300, gen=100) -> 400.
    # Mean = 275.0.
    trajs = [
        _traj_with_token_counts(prompt_counts=[10, 100], generation_counts=[5, 50]),
        _traj_with_token_counts(
            prompt_counts=[20, 200, 300], generation_counts=[10, 60, 100]
        ),
    ]
    result = SamplerTokenSumMetricAggregator()(trajs)
    assert result["sampler/peak_context_tokens_mean"] == MetricValue.of_scalar(275.0)


def test_sampler_token_sum_aggregator_omits_keys_when_no_token_counts():
    """When no agent step in any trajectory has a token count set, both
    token keys are omitted from the emitted dict (matches the conditional
    emit shape of the pre-migration inline aggregation)."""
    from core.metric_aggregators import SamplerTokenSumMetricAggregator

    trajs = [
        _traj_with_token_counts(
            prompt_counts=[None, None], generation_counts=[None, None]
        ),
        _traj_with_token_counts(prompt_counts=[None], generation_counts=[None]),
    ]
    result = SamplerTokenSumMetricAggregator()(trajs)
    assert "sampler/generation_tokens_sum" not in result
    assert "sampler/peak_context_tokens_mean" not in result


def test_sampler_token_sum_aggregator_is_in_defaults():
    from core.metric_aggregators import SamplerTokenSumMetricAggregator

    assert any(
        isinstance(agg, SamplerTokenSumMetricAggregator)
        for agg in DEFAULT_METRIC_AGGREGATORS
    )


def test_sampler_token_sum_aggregator_specs():
    from core.metric_aggregators import SamplerTokenSumMetricAggregator

    specs = {s.name: s for s in SamplerTokenSumMetricAggregator().specs()}
    assert set(specs.keys()) == {
        "sampler/generation_tokens_sum",
        "sampler/peak_context_tokens_mean",
    }
    assert specs["sampler/generation_tokens_sum"].strategy is AggregationStrategy.SUM
    assert (
        specs["sampler/peak_context_tokens_mean"].strategy is AggregationStrategy.MEAN
    )


# ---------------------------------------------------------------------------
# Context compaction
# ---------------------------------------------------------------------------


def _traj_with_compaction(
    pre_tokens: int, post_tokens: int, *, n_events: int = 1
) -> ScoredTrajectory:
    """ScoredTrajectory carrying ``n_events`` compaction steps, each
    stamped with the pre/post token extras the compaction step writes.

    ``source="agent"``, matching what ``_compact`` emits: the model
    authored the summary and it enters the context, so its candidate
    trains. The extras are therefore the only discriminator."""
    scored = _traj_with_token_counts([1000], [100])
    for i in range(n_events):
        scored.trajectory.steps.append(
            Step(
                step_id=99 + i,
                source="agent",
                message="SUMMARY",
                extra={
                    "compaction_pre_tokens": pre_tokens,
                    "compaction_post_tokens": post_tokens,
                    "compaction_pairs_elided": 8,
                    "compaction_pairs_retained": 2,
                },
            )
        )
    return scored


def test_peak_context_tokens_uses_max_not_last():
    """After a compaction the final prompt is small; the peak must not fall
    just because the last step is cheap."""
    from core.metric_aggregators import SamplerTokenSumMetricAggregator

    traj = _traj_with_token_counts([100_000, 5_000], [10_000, 500])
    out = SamplerTokenSumMetricAggregator()([traj])
    assert out["sampler/peak_context_tokens_mean"].value == 110_000


def test_compaction_metrics_zero_when_never_compacted():
    from core.metric_aggregators import CompactionMetricAggregator

    out = CompactionMetricAggregator()([_traj_with_token_counts([10], [1])])
    assert out["sampler/compactions_mean"].value == 0.0
    assert out["sampler/compactions_any"].value == 0.0
    assert "sampler/compaction_reclaimed_tokens_mean" not in out


def test_compaction_metrics_count_steps_carrying_the_extras():
    from core.metric_aggregators import CompactionMetricAggregator

    traj = _traj_with_compaction(pre_tokens=100_000, post_tokens=20_000)
    out = CompactionMetricAggregator()([traj])
    assert out["sampler/compactions_mean"].value == 1.0
    assert out["sampler/compactions_any"].value == 1.0
    assert out["sampler/compaction_reclaimed_tokens_mean"].value == 80_000


def test_compaction_step_is_agent_sourced_but_is_not_a_solving_step():
    """A compaction is agent-sourced so its candidate trains and its
    tokens count -- but it is not a step the agent spent solving, so the
    step-count metric must exclude it via the extras, not the source."""
    from core.metric_aggregators import (
        AgentStepsMetricAggregator,
        CompactionMetricAggregator,
    )

    # One real agent step (from _traj_with_token_counts) + 2 compactions.
    scored = _traj_with_compaction(100, 40, n_events=2)
    assert [s.source for s in scored.trajectory.steps].count("agent") == 3

    assert CompactionMetricAggregator()([scored])["sampler/compactions_mean"].value == 2
    steps_out = AgentStepsMetricAggregator()([scored])
    assert steps_out["sampler/agent_steps_mean"].value == 1.0


def test_compaction_without_post_tokens_is_counted_but_not_reclaimed():
    """``compaction_post_tokens`` is back-filled from the NEXT sample, so a
    rollout that dies before that sample lands has the event but no post.

    It must still count as a compaction (``pre`` is the marker) while being
    excluded from the reclaim mean -- defaulting the missing post to 0 would
    silently report the whole pre-context as reclaimed.
    """
    from core.metric_aggregators import CompactionMetricAggregator

    complete = _traj_with_compaction(pre_tokens=100_000, post_tokens=20_000)
    partial = _traj_with_compaction(pre_tokens=100_000, post_tokens=20_000)
    del partial.trajectory.steps[-1].extra["compaction_post_tokens"]

    out = CompactionMetricAggregator()([complete, partial])
    assert out["sampler/compactions_mean"].value == 1.0
    assert out["sampler/compactions_any"].value == 1.0
    # Only the complete event contributes; 80_000, not (80_000+100_000)/2.
    assert out["sampler/compaction_reclaimed_tokens_mean"].value == 80_000


def test_compaction_reclaim_omitted_when_no_event_has_post_tokens():
    from core.metric_aggregators import CompactionMetricAggregator

    partial = _traj_with_compaction(pre_tokens=100_000, post_tokens=20_000)
    del partial.trajectory.steps[-1].extra["compaction_post_tokens"]

    out = CompactionMetricAggregator()([partial])
    assert out["sampler/compactions_mean"].value == 1.0
    assert "sampler/compaction_reclaimed_tokens_mean" not in out


def test_system_step_without_compaction_extras_is_not_an_event():
    """The system-instruction step is also ``source="system"`` but carries no
    ``compaction_pre_tokens``; counting by source alone would report a
    compaction on every rollout."""
    from core.metric_aggregators import CompactionMetricAggregator

    scored = _traj_with_token_counts([10], [1])
    scored.trajectory.steps.insert(
        0, Step(step_id=1, source="system", message="You are a helpful agent.")
    )
    out = CompactionMetricAggregator()([scored])
    assert out["sampler/compactions_mean"].value == 0.0
    assert out["sampler/compactions_any"].value == 0.0
    assert "sampler/compaction_reclaimed_tokens_mean" not in out


def test_prompt_too_long_frac_counts_terminal_status():
    from core.metric_aggregators import CompactionMetricAggregator

    overflowed = _traj_with_status(RolloutStatus.SAMPLING_FAILURE_PROMPT_TOO_LONG)
    ok = _traj_with_status(RolloutStatus.OK)
    out = CompactionMetricAggregator()([overflowed, ok])
    assert out["sampler/prompt_too_long_frac"].count_total == (1, 2)


def test_compaction_aggregator_averages_over_rollouts():
    """``compactions_mean`` averages events per rollout; ``compactions_any``
    is the fraction of rollouts with at least one."""
    from core.metric_aggregators import CompactionMetricAggregator

    out = CompactionMetricAggregator()(
        [
            _traj_with_compaction(100, 40, n_events=3),
            _traj_with_token_counts([10], [1]),
        ]
    )
    assert out["sampler/compactions_mean"].value == 1.5
    assert out["sampler/compactions_any"].value == 0.5
    assert out["sampler/compaction_reclaimed_tokens_mean"].value == 60.0


def test_compaction_aggregator_is_in_defaults():
    from core.metric_aggregators import CompactionMetricAggregator

    assert any(
        isinstance(agg, CompactionMetricAggregator)
        for agg in DEFAULT_METRIC_AGGREGATORS
    )


def test_compaction_aggregator_specs():
    from core.metric_aggregators import CompactionMetricAggregator

    specs = {s.name: s for s in CompactionMetricAggregator().specs()}
    assert set(specs.keys()) == {
        "sampler/compactions_mean",
        "sampler/compactions_any",
        "sampler/compaction_reclaimed_tokens_mean",
        "sampler/prompt_too_long_frac",
    }
    assert specs["sampler/compactions_mean"].strategy is AggregationStrategy.MEAN
    assert specs["sampler/compactions_any"].strategy is AggregationStrategy.MEAN
    assert (
        specs["sampler/compaction_reclaimed_tokens_mean"].strategy
        is AggregationStrategy.MEAN
    )
    assert specs["sampler/prompt_too_long_frac"].strategy is AggregationStrategy.RATIO


def test_compaction_metric_names_are_registered_specs():
    """Every emitted key must resolve, plus its train/ and eval/ twins."""
    from core.metric_aggregators import CompactionMetricAggregator
    from core.metric_spec import spec_for

    for spec in CompactionMetricAggregator().specs():
        assert spec_for(spec.name).strategy is spec.strategy
        for prefix in ("train/", "eval/"):
            twin = spec_for(prefix + spec.name)
            assert twin.strategy is spec.strategy
            assert twin.emit_to_ml_platform is True


def _traj_no_step_with_status(status: RolloutStatus) -> ScoredTrajectory:
    """A rollout that ended on a sample producing NO step at all.

    This is the real shape of a part-less candidate: the response converter
    raises before a turn exists, so the trajectory holds only the user step
    and the failure is recorded solely as ``rollout_status``.
    """
    us = Step(step_id=1, source="user", message="p")
    traj = AtifTrajectory(
        schema_version="ATIF-v1.7",
        session_id="test-session",
        agent=AtifAgent(name="t", version="0", model_name="m"),
        steps=[us],
    )
    atif_extras.set_rollout_status(traj, status)
    return ScoredTrajectory(trajectory=traj, reward=Reward(reward=float("nan")))


class TestFinishReasonCountsStepLessGenerations:
    """A generation that left no step is still a generation.

    `sampler/finish_reason/*` declares its unit as "one LLM generation call",
    but it only ever walked agent STEPS. A MALFORMED_FUNCTION_CALL candidate
    arrives with empty `parts` -- measured over 60k real candidates, all 80
    malformed ones were part-less and no STOP ever was -- so the converter
    raises, the rollout ends without a turn, and the metric never saw it.
    On a backend that only emits malformed calls that way, the series read
    0.000 forever while the failures were really happening.
    """

    def test_a_step_less_malformed_rollout_is_counted(self):
        from core.metric_aggregators import FinishReasonMetricAggregator

        trajs = [
            _traj_with_finish_reasons(["STOP", "STOP", "STOP"]),
            _traj_no_step_with_status(
                RolloutStatus.SAMPLING_FAILURE_MALFORMED_FUNCTION_CALL
            ),
        ]
        out = FinishReasonMetricAggregator()(trajs)
        assert "sampler/finish_reason/MALFORMED_FUNCTION_CALL" in out
        # Denominator is CALLS: 3 steps + 1 step-less generation.
        assert out["sampler/finish_reason/MALFORMED_FUNCTION_CALL"] == MetricValue.of_ratio(1, 4)
        assert out["sampler/finish_reason/STOP"] == MetricValue.of_ratio(3, 4)

    def test_max_tokens_and_safety_are_counted_too(self):
        from core.metric_aggregators import FinishReasonMetricAggregator

        for status, reason in [
            (RolloutStatus.SAMPLING_FAILURE_MAX_TOKENS, "MAX_TOKENS"),
            (RolloutStatus.SAMPLING_FAILURE_SAFETY, "SAFETY"),
        ]:
            out = FinishReasonMetricAggregator()(
                [
                    _traj_with_finish_reasons(["STOP"]),
                    _traj_no_step_with_status(status),
                ]
            )
            assert out[f"sampler/finish_reason/{reason}"] == MetricValue.of_ratio(1, 2)

    def test_non_generation_failures_are_NOT_counted(self):
        """TIMEOUT and SAMPLING_CLIENT_ERROR describe the call, not the
        generation. Inventing a finish reason for them would put infra noise
        into a metric about model behaviour."""
        from core.metric_aggregators import FinishReasonMetricAggregator

        for status in (
            RolloutStatus.SAMPLING_FAILURE_TIMEOUT,
            RolloutStatus.SAMPLING_FAILURE_SAMPLING_CLIENT_ERROR,
            RolloutStatus.SAMPLING_FAILURE_UNHANDLED,
        ):
            out = FinishReasonMetricAggregator()(
                [
                    _traj_with_finish_reasons(["STOP"]),
                    _traj_no_step_with_status(status),
                ]
            )
            # Denominator stays 1: the failed call contributed nothing.
            assert out["sampler/finish_reason/STOP"] == MetricValue.of_ratio(1, 1)

    def test_an_ordinary_rollout_is_unchanged(self):
        """OK is not in the map, so normal trajectories behave exactly as
        before -- this must not shift any existing series."""
        from core.metric_aggregators import FinishReasonMetricAggregator

        out = FinishReasonMetricAggregator()(
            [_traj_with_finish_reasons(["STOP", "MAX_TOKENS"])]
        )
        assert out["sampler/finish_reason/STOP"] == MetricValue.of_ratio(1, 2)
        assert out["sampler/finish_reason/MAX_TOKENS"] == MetricValue.of_ratio(1, 2)


class TestOtherIsAlwaysEmitted:
    """`finish_reason/OTHER` is pinned to an explicit zero.

    Every other suffix in this family is data-driven, but OTHER is the
    catch-all for "an agent step reported no finish reason at all". Its
    absence is ambiguous -- every step reported one, or the field stopped
    being recorded? -- where 0.000 is a statement. It also keeps the series
    comparable against a run that does have unattributed steps.
    """

    def test_other_is_present_at_zero_when_every_step_has_a_reason(self):
        from core.metric_aggregators import FinishReasonMetricAggregator

        out = FinishReasonMetricAggregator()(
            [_traj_with_finish_reasons(["STOP", "STOP"])]
        )
        assert "sampler/finish_reason/OTHER" in out
        assert out["sampler/finish_reason/OTHER"] == MetricValue.of_ratio(0, 2)
        assert out["sampler/finish_reason/STOP"] == MetricValue.of_ratio(2, 2)

    def test_other_still_counts_real_unattributed_steps(self):
        """Zero-filling must not mask the thing the bucket is for."""
        from core.metric_aggregators import FinishReasonMetricAggregator

        out = FinishReasonMetricAggregator()(
            [_traj_with_finish_reasons(["STOP", None, None])]
        )
        assert out["sampler/finish_reason/OTHER"] == MetricValue.of_ratio(2, 3)

    def test_a_group_with_no_reasons_at_all_still_emits_nothing(self):
        """The existing empty-output guard is unchanged: a recipe whose steps
        never carry a finish_reason should not start reporting an OTHER of
        1.000 as if that were a finding."""
        from core.metric_aggregators import FinishReasonMetricAggregator

        out = FinishReasonMetricAggregator()(
            [_traj_with_finish_reasons([None, None])]
        )
        assert out == {}


class TestTrajectoryStatusVocabularyIsZeroFilled:
    """Every status is published every run, not only once it happens.

    Left data-driven, the family's shape depends on the run's luck:
    `max_step_limit` is absent until some rollout first exhausts its budget,
    `sampling_failure_TIMEOUT` until the first timeout. Two runs of one
    recipe then expose different series sets and a comparison shows gaps
    that mean nothing. Worse, "never happened" becomes indistinguishable
    from "the classifier stopped emitting it" -- the exact failure this
    family exists to catch.
    """

    def test_every_status_is_present_even_when_only_OK_occurred(self):
        from core.metric_aggregators import RolloutStatusMetricAggregator

        out = RolloutStatusMetricAggregator()([_traj_with_status(RolloutStatus.OK)])
        for st in RolloutStatus:
            if st.is_sampling_failure():
                key = (
                    "sampler/trajectory_status/"
                    f"sampling_failure_{st.sampling_failure_reason()}"
                )
            else:
                key = f"sampler/trajectory_status/{st.value}"
            assert key in out, f"{key} missing -- series shape depends on luck"
        assert "sampler/trajectory_status/sampling_failure" in out

    def test_the_three_that_were_missing_from_a_live_run(self):
        """Regression pin for the exact gap found comparing two real runs."""
        from core.metric_aggregators import RolloutStatusMetricAggregator

        out = RolloutStatusMetricAggregator()([_traj_with_status(RolloutStatus.OK)])
        for k in (
            "max_step_limit",
            "sampling_failure_SAMPLING_CLIENT_ERROR",
            "sampling_failure_TIMEOUT",
        ):
            key = f"sampler/trajectory_status/{k}"
            assert out[key] == MetricValue.of_ratio(0, 1)

    def test_real_counts_are_not_masked_by_the_zero_fill(self):
        from core.metric_aggregators import RolloutStatusMetricAggregator

        out = RolloutStatusMetricAggregator()(
            [
                _traj_with_status(RolloutStatus.OK),
                _traj_with_status(RolloutStatus.MAX_STEP_LIMIT),
            ]
        )
        assert out["sampler/trajectory_status/max_step_limit"] == (
            MetricValue.of_ratio(1, 2)
        )
        assert out["sampler/trajectory_status/OK"] == MetricValue.of_ratio(1, 2)

    def test_empty_group_still_emits_nothing(self):
        from core.metric_aggregators import RolloutStatusMetricAggregator

        assert RolloutStatusMetricAggregator()([]) == {}
