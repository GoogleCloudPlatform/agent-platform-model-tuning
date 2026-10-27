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

"""MetricAggregator protocol + built-in aggregators.

Each aggregator is a pure, total function
``list[ScoredTrajectory] -> dict[str, float | int | MetricValue]``
invoked per group by ``SingleStepAgent.group_rollout``. Outputs are
merged via ``merge_group_metrics`` (same-stage collisions raise) then
reduced at batch stage by ``aggregate_by_spec`` (each key's strategy
comes from its registered ``MetricSpec``). Recipes plug extras in
alongside ``DEFAULT_METRIC_AGGREGATORS``.
"""

import math
import warnings
from typing import Protocol

import numpy as np

from harbor.models.trajectories import Step

from core import atif_extras
from core.atif_trajectory import ScoredTrajectory
from core.metric_spec import AggregationStrategy, MetricSpec, MetricValue
from core.rollout_status import RolloutStatus


class MetricAggregator(Protocol):
    """Pure, total function over a trajectory group.

    ``__call__`` returns ``{metric_key: value}``; values may be bare
    ``float`` / ``int`` or ``MetricValue`` (the agent's group-merge
    wraps bare floats). ``specs`` returns the ``MetricSpec``s this
    aggregator can emit.
    """

    def __call__(
        self, trajectories: list[ScoredTrajectory]
    ) -> dict[str, float | int | MetricValue]: ...

    def specs(self) -> list[MetricSpec]: ...


def _trajectory_is_correct(g: ScoredTrajectory) -> bool:
    """Read trajectory-level ``is_correct``; else ``reward > 0``."""
    signal = atif_extras.get_is_correct(g.trajectory)
    if signal is not None:
        return bool(signal)
    return g.reward.reward > 0


class CoreRewardMetricAggregator:
    """learning/reward/mean, learning/sampling_accuracy, learning/pass_at_k.

    Reward carries a (group_mean, n_trajectories) weighted pair so the
    batch stage computes a true trajectory-weighted mean across uneven
    groups. Accuracy is a per-group scalar meaned at batch. pass_at_k
    is a per-group any-correct 0/1 scalar meaned at batch.

    NaN handling: if ``np.nanmean(rewards)`` is NaN (all-NaN group,
    e.g. sampler timeout), ``learning/reward/mean`` is OMITTED -- a
    missing key legitimately signals "no signal to report" for the
    group. sampling_accuracy / pass_at_k survive because they read
    ``is_correct`` (NaN reward makes ``reward > 0`` False, which is
    correct: a timed-out rollout is not "correct").
    """

    def specs(self) -> list[MetricSpec]:
        return [
            MetricSpec(
                "learning/reward/mean",
                AggregationStrategy.WEIGHTED_MEAN,
                "one rollout trajectory",
                emit_to_ml_platform=True,
            ),
            MetricSpec(
                "learning/sampling_accuracy",
                AggregationStrategy.MEAN,
                "one rollout trajectory",
                emit_to_ml_platform=True,
            ),
            MetricSpec(
                "learning/pass_at_k",
                AggregationStrategy.MEAN,
                "one trajectory group",
                emit_to_ml_platform=True,
            ),
        ]

    def __call__(self, trajectories: list[ScoredTrajectory]) -> dict[str, MetricValue]:
        rewards = [g.reward.reward for g in trajectories]
        is_correct = [_trajectory_is_correct(g) for g in trajectories]
        n = len(trajectories)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            avg_reward = float(np.nanmean(rewards)) if rewards else 0.0
        result: dict[str, MetricValue] = {
            "learning/sampling_accuracy": MetricValue.of_scalar(
                float(np.mean(is_correct)) if is_correct else 0.0
            ),
            "learning/pass_at_k": MetricValue.of_scalar(
                float(any(is_correct)) if is_correct else 0.0
            ),
        }
        # Only emit reward/mean when we have a real numeric mean.
        # All-NaN rewards -> np.nanmean returns NaN -> key omitted.
        if not math.isnan(avg_reward):
            result["learning/reward/mean"] = MetricValue.of_weighted(avg_reward, n)
        return result


class SamplerPolicyStepMetricAggregator:
    """Emits ``sampler/policy_step`` = count-weighted mean of per-step
    ``train_step_id`` across the group (skipping None). Empty output if
    no step has a train_step_id.

    ``train/staleness_steps`` is computed client-side at train-step
    boundary as ``global_step - mean(policy_step)`` (see
    ``train_and_eval.py``); this aggregator's output feeds it.
    """

    def specs(self) -> list[MetricSpec]:
        return [
            MetricSpec(
                "sampler/policy_step",
                AggregationStrategy.WEIGHTED_MEAN,
                "one rollout trajectory",
                emit_to_ml_platform=False,  # ml_platform gets train/staleness_steps instead
            ),
        ]

    def __call__(self, trajectories: list[ScoredTrajectory]) -> dict[str, MetricValue]:
        total = 0
        count = 0
        for g in trajectories:
            for step in g.trajectory.steps:
                if step.source != "agent":
                    continue
                ts_id = atif_extras.get_train_step_id(step)
                if ts_id is None:
                    continue
                total += ts_id
                count += 1
        if count == 0:
            return {}
        return {"sampler/policy_step": MetricValue.of_weighted(total / count, count)}


def _failed_sample_finish_reason(g: ScoredTrajectory) -> str | None:
    """Finish reason of a sample that ended the rollout without a step.

    Read off ``rollout_status``, which is the single fault vocabulary and
    already carries the reason the classifier derived it from -- so nothing
    new has to be written at failure time, and the journal keeps its rule of
    writing NOTHING for a sample that produced no candidate.

    Returns ``None`` for statuses that are not a finish reason (``TIMEOUT``,
    ``SAMPLING_CLIENT_ERROR``) and for rollouts that ended on a real turn --
    ``OK`` is not in the map, so a normal rollout adds nothing here and its
    per-step reasons are counted by the loop above as before.
    """
    return atif_extras.get_rollout_status(g.trajectory).native_finish_reason()


class FinishReasonMetricAggregator:
    """Per-``finish_reason`` ratios over agent steps in the group.

    Emits ``of_ratio(count_for_reason, total_agent_steps)`` per reason
    so the batch stage computes true per-reason fractions across the
    whole batch. Empty output if no step reported a finish_reason.
    """

    def specs(self) -> list[MetricSpec]:
        return [
            MetricSpec(
                "sampler/finish_reason/",  # dynamic-suffix family
                AggregationStrategy.RATIO,
                "one LLM generation call",
                emit_to_ml_platform=True,
            )
        ]

    def __call__(self, trajectories: list[ScoredTrajectory]) -> dict[str, MetricValue]:
        counts: dict[str, int] = {}
        total = 0
        any_finish_reason = False
        for g in trajectories:
            for step in g.trajectory.steps:
                if step.source != "agent":
                    continue
                total += 1
                fr = atif_extras.get_finish_reason(step)
                if fr is not None:
                    any_finish_reason = True
                else:
                    fr = "OTHER"
                counts[fr] = counts.get(fr, 0) + 1
            # A generation that produced NO STEP still happened, and this
            # metric's unit is the CALL, not the step. The case that matters
            # is MALFORMED_FUNCTION_CALL: such a candidate arrives with EMPTY
            # parts -- measured over 60k real candidates, all 80 malformed
            # ones were part-less and no STOP ever was -- so the response
            # converter raises, the rollout ends without a turn, and the loop
            # above never sees it. Counting it in the numerator AND the
            # denominator is what makes this a true per-call ratio; without
            # it the series reads 0.000 forever on a backend that only ever
            # emits malformed calls this way.
            fr = _failed_sample_finish_reason(g)
            if fr is not None:
                any_finish_reason = True
                total += 1
                counts[fr] = counts.get(fr, 0) + 1
        if not any_finish_reason:
            return {}
        # Emit OTHER even at zero. Every other suffix in this family is
        # data-driven -- the reason appeared, so the series exists -- but
        # OTHER is the catch-all for "an agent step reported no finish
        # reason at all", and its ABSENCE is ambiguous in a way a 0.000 is
        # not: you cannot tell whether every step reported a reason or
        # whether the field stopped being recorded. Pinning it to an
        # explicit zero makes "no unattributed steps" a statement the
        # dashboard makes, rather than something inferred from a missing
        # line, and keeps the series comparable against a run that does
        # have unattributed steps.
        counts.setdefault("OTHER", 0)
        return {
            f"sampler/finish_reason/{k}": MetricValue.of_ratio(v, total)
            for k, v in counts.items()
        }


class TimingMetricAggregator:
    """Per-step sample/env timings + per-rollout wall time under ``pipeline/*``.

    Emits raw samples (``of_samples``) for p95 keys so the batch stage
    re-percentiles over ALL pooled values, a mean scalar for avg keys,
    and (part, total) for ``sample_time_fraction`` so batch computes
    ratio_of_sums. A key is omitted entirely when its underlying list is
    empty.

    ``rollout_wall_time_*`` is the odd one out: its samples are
    per-TRAJECTORY (one true end-to-end wall clock per rollout, stamped
    by ``Agent._run_one_rollout``), not per-step. It is deliberately not
    reconstructable from the per-step timings above — those exclude env
    construction, tool/sandbox execution and in-rollout waiting.
    """

    def specs(self) -> list[MetricSpec]:
        return [
            MetricSpec(
                "pipeline/sample_time_avg",
                AggregationStrategy.MEAN,
                "one LLM generation call",
                emit_to_ml_platform=True,
            ),
            MetricSpec(
                "pipeline/env_time_avg",
                AggregationStrategy.MEAN,
                "one LLM generation call",
                emit_to_ml_platform=True,
            ),
            MetricSpec(
                "pipeline/sample_time_p95",
                AggregationStrategy.P95,
                "one LLM generation call",
                emit_to_ml_platform=True,
            ),
            MetricSpec(
                "pipeline/env_time_p95",
                AggregationStrategy.P95,
                "one LLM generation call",
                emit_to_ml_platform=True,
            ),
            MetricSpec(
                "pipeline/sample_time_fraction",
                AggregationStrategy.RATIO_OF_SUMS,
                "one LLM generation call",
                emit_to_ml_platform=True,
            ),
            MetricSpec(
                "pipeline/rollout_wall_time_avg",
                AggregationStrategy.MEAN,
                "one rollout trajectory",
                emit_to_ml_platform=True,
            ),
            MetricSpec(
                "pipeline/rollout_wall_time_p95",
                AggregationStrategy.P95,
                "one rollout trajectory",
                emit_to_ml_platform=True,
            ),
        ]

    def __call__(self, trajectories: list[ScoredTrajectory]) -> dict[str, MetricValue]:
        per_step_sample: list[float] = []
        per_step_env: list[float] = []
        per_rollout_wall: list[float] = []
        for g in trajectories:
            w = atif_extras.get_rollout_wall_time_s(g.trajectory)
            if w is not None:
                per_rollout_wall.append(float(w))
            for step in g.trajectory.steps:
                if step.source != "agent":
                    continue
                s = atif_extras.get_sample_time_s(step)
                e = atif_extras.get_env_time_s(step)
                if s is not None:
                    per_step_sample.append(float(s))
                if e is not None:
                    per_step_env.append(float(e))

        out: dict[str, MetricValue] = {}
        if per_rollout_wall:
            out["pipeline/rollout_wall_time_avg"] = MetricValue.of_scalar(
                float(np.mean(per_rollout_wall))
            )
            out["pipeline/rollout_wall_time_p95"] = MetricValue.of_samples(
                per_rollout_wall
            )
        if per_step_sample:
            out["pipeline/sample_time_avg"] = MetricValue.of_scalar(
                float(np.mean(per_step_sample))
            )
            out["pipeline/sample_time_p95"] = MetricValue.of_samples(per_step_sample)
        if per_step_env:
            out["pipeline/env_time_avg"] = MetricValue.of_scalar(
                float(np.mean(per_step_env))
            )
            out["pipeline/env_time_p95"] = MetricValue.of_samples(per_step_env)
        if per_step_sample and per_step_env:
            sum_s = float(sum(per_step_sample))
            sum_e = float(sum(per_step_env))
            out["pipeline/sample_time_fraction"] = MetricValue.of_part_total(
                sum_s, sum_s + sum_e
            )
        return out


class RolloutStatusMetricAggregator:
    """``sampler/trajectory_status/*`` ratios; splits sampling_failure/REASON.

    Per-status ``of_ratio(count, total_trajectories)``. Both the
    ``sampling_failure`` roll-up and per-reason
    ``sampling_failure_<reason>`` keys are emitted (leaves are a
    breakdown of the roll-up, so leaves + roll-up won't sum to 1).
    """

    def specs(self) -> list[MetricSpec]:
        return [
            MetricSpec(
                "sampler/trajectory_status/",  # dynamic-suffix family
                AggregationStrategy.RATIO,
                "one rollout trajectory",
                emit_to_ml_platform=True,
            )
        ]

    def __call__(self, trajectories: list[ScoredTrajectory]) -> dict[str, MetricValue]:
        status_counts: dict[str, int] = {}
        total = len(trajectories)
        for g in trajectories:
            status = atif_extras.get_rollout_status(g.trajectory)
            try:
                status_enum = RolloutStatus(status)
            except ValueError:
                key = str(status)
                status_counts[key] = status_counts.get(key, 0) + 1
                continue
            if status_enum.is_sampling_failure():
                status_counts["sampling_failure"] = (
                    status_counts.get("sampling_failure", 0) + 1
                )
                key = f"sampling_failure_{status_enum.sampling_failure_reason()}"
                status_counts[key] = status_counts.get(key, 0) + 1
            else:
                status_counts[status_enum.value] = (
                    status_counts.get(status_enum.value, 0) + 1
                )
        if total == 0:
            return {}
        # Zero-fill the whole vocabulary. Without this the family is purely
        # data-driven, so a series exists only once its status has occurred
        # at least once -- which makes the dashboard's shape depend on the
        # run's luck: `max_step_limit` is absent until some rollout first
        # exhausts its budget, `sampling_failure_TIMEOUT` until the first
        # timeout. Two runs of the same recipe then show different series
        # sets, and a comparison reads as a gap where nothing is wrong.
        #
        # It also makes "this never happened" indistinguishable from "the
        # classifier stopped emitting it" -- the exact failure this family
        # exists to catch, and one we hit: malformed candidates were being
        # misfiled and the only visible symptom was an ABSENT series.
        #
        # Derived from the enum, so a new RolloutStatus member is published
        # from its first run rather than from its first occurrence.
        for st in RolloutStatus:
            if st.is_sampling_failure():
                status_counts.setdefault("sampling_failure", 0)
                status_counts.setdefault(
                    f"sampling_failure_{st.sampling_failure_reason()}", 0
                )
            else:
                status_counts.setdefault(st.value, 0)
        return {
            f"sampler/trajectory_status/{k}": MetricValue.of_ratio(v, total)
            for k, v in status_counts.items()
        }


class SamplerTokenSumMetricAggregator:
    """Token-usage metrics.

    - ``sampler/generation_tokens_sum``: SUM over every LLM call of the
      per-step ``generation_token_count`` -- including harness-initiated
      calls (context compaction), which are billed to the rollout even
      though they are not agent steps.
    - ``sampler/peak_context_tokens_mean``: MEAN over trajectories of
      each trajectory's MAX over agent steps of
      ``prompt_token_count + generation_token_count`` (the peak context
      length reached during the rollout). Max rather than last: in-loop
      context compaction resets the running prompt, so a last-step
      reading would report the peak FALLING exactly when contexts grow.
      Without compaction the last step is the max, so this is a strict
      generalization.

    Keys are omitted when no step contributes a non-None token count.
    """

    def specs(self) -> list[MetricSpec]:
        return [
            MetricSpec(
                "sampler/generation_tokens_sum",
                AggregationStrategy.SUM,
                "one LLM generation call",
                emit_to_ml_platform=True,
            ),
            MetricSpec(
                "sampler/peak_context_tokens_mean",
                AggregationStrategy.MEAN,
                "one rollout trajectory",
                emit_to_ml_platform=True,
            ),
        ]

    def __call__(self, trajectories: list[ScoredTrajectory]) -> dict[str, MetricValue]:
        # Every step that made an LLM call, NOT just agent-sourced ones:
        # in-loop context compaction emits a ``system``-sourced step
        # carrying a real generation call, and those tokens were paid
        # for. The agent-step COUNT still excludes them (they are not
        # actions); only the token SUM includes them.
        gen_per_step = [
            atif_extras.get_generation_token_count(s)
            for g in trajectories
            for s in g.trajectory.steps
        ]
        out: dict[str, MetricValue] = {}
        if any(c is not None for c in gen_per_step):
            out["sampler/generation_tokens_sum"] = MetricValue.of_scalar(
                sum(c or 0 for c in gen_per_step)
            )

        # peak_context = MAX over agent steps of (prompt + generation) per
        # traj; skip trajectories with no agent step carrying both counts.
        # Max, not last: a compaction shrinks the running prompt, so a
        # last-step reading would report the peak falling as contexts grew.
        per_traj_totals: list[float] = []
        for g in trajectories:
            per_step: list[float] = []
            for s in g.trajectory.steps:
                if s.source != "agent":
                    continue
                p = atif_extras.get_prompt_token_count(s)
                gt = atif_extras.get_generation_token_count(s)
                if p is not None and gt is not None:
                    per_step.append(float(p + gt))
            if per_step:
                per_traj_totals.append(max(per_step))
        if per_traj_totals:
            out["sampler/peak_context_tokens_mean"] = MetricValue.of_scalar(
                float(np.mean(per_traj_totals))
            )
        return out


def _is_compaction_step(step: Step) -> bool:
    """Whether ``step`` is a context-compaction summarization turn.

    Keyed off the ``compaction_pre_tokens`` extra, NOT off ``source``: a
    compaction step is agent-sourced so that its candidate trains and its
    tokens count, which leaves the extras as the only discriminator.
    """
    return (step.extra or {}).get("compaction_pre_tokens") is not None


class AgentStepsMetricAggregator:
    """Average number of agent steps (LLM turns) per trajectory.

    Emits ``sampler/agent_steps_mean`` — the mean count of agent-sourced
    steps across all trajectories in the group.  Useful for tracking how
    many LLM calls the agent makes per rollout (budget utilization,
    straggler detection, step_limit headroom).

    Compaction steps are agent-sourced (the model authored the summary
    and it enters the context, so it trains) but are NOT solving steps,
    so they are excluded here — otherwise this metric drifts away from
    the step budget it exists to track. They are discriminated by their
    ``compaction_*`` extras rather than by ``source``.
    """

    def specs(self) -> list[MetricSpec]:
        return [
            MetricSpec(
                "sampler/agent_steps_mean",
                AggregationStrategy.MEAN,
                "one rollout trajectory",
                emit_to_ml_platform=True,
            ),
        ]

    def __call__(self, trajectories: list[ScoredTrajectory]) -> dict[str, MetricValue]:
        per_traj_counts: list[float] = []
        for g in trajectories:
            n_agent = sum(
                1
                for s in g.trajectory.steps
                if s.source == "agent" and not _is_compaction_step(s)
            )
            per_traj_counts.append(float(n_agent))
        if per_traj_counts:
            return {
                "sampler/agent_steps_mean": MetricValue.of_scalar(
                    float(np.mean(per_traj_counts))
                ),
            }
        return {}


class CompactionMetricAggregator:
    """Context-compaction volume, effectiveness, and residual failures.

    A compaction emits a step carrying ``compaction_pre_tokens`` /
    ``compaction_post_tokens``, so counting those steps gives the event
    count with no extra plumbing. The extras key is the ONLY
    discriminator: a compaction step is ``source="agent"`` (the model
    authored the summary and it enters the context, so it trains), which
    makes it indistinguishable from a solving step by ``source`` alone.

    - ``sampler/compactions_mean``: mean events per rollout.
    - ``sampler/compactions_any``: fraction of rollouts that compacted.
    - ``sampler/compaction_reclaimed_tokens_mean``: mean tokens reclaimed
      per event. A small number means the summary is not buying headroom
      and the rollout will re-trigger within a turn or two. Omitted when
      no rollout in the group compacted.
    - ``sampler/prompt_too_long_frac``: fraction of rollouts that still
      died on an unfittable context. The success criterion: not ~0 means
      the trigger ratio is too loose.
    """

    def specs(self) -> list[MetricSpec]:
        return [
            MetricSpec(
                "sampler/compactions_mean",
                AggregationStrategy.MEAN,
                "one rollout trajectory",
                emit_to_ml_platform=True,
            ),
            MetricSpec(
                "sampler/compactions_any",
                AggregationStrategy.MEAN,
                "one rollout trajectory",
                emit_to_ml_platform=True,
            ),
            MetricSpec(
                "sampler/compaction_reclaimed_tokens_mean",
                AggregationStrategy.MEAN,
                "one compaction event",
                emit_to_ml_platform=True,
            ),
            MetricSpec(
                "sampler/prompt_too_long_frac",
                AggregationStrategy.RATIO,
                "one rollout trajectory",
                emit_to_ml_platform=True,
            ),
        ]

    def __call__(self, trajectories: list[ScoredTrajectory]) -> dict[str, MetricValue]:
        if not trajectories:
            return {}
        counts: list[int] = []
        reclaimed: list[float] = []
        overflowed = 0
        for g in trajectories:
            events = [s for s in g.trajectory.steps if _is_compaction_step(s)]
            counts.append(len(events))
            for s in events:
                extra = s.extra or {}
                # ``post`` is back-filled by the first call after the
                # compaction, so a rollout that died before that call has
                # the event but no post. Skip it rather than defaulting to
                # 0, which would report the whole pre-context as reclaimed.
                post = extra.get("compaction_post_tokens")
                if post is None:
                    continue
                reclaimed.append(
                    float(extra.get("compaction_pre_tokens", 0)) - float(post)
                )
            if (
                atif_extras.get_rollout_status(g.trajectory)
                == RolloutStatus.SAMPLING_FAILURE_PROMPT_TOO_LONG
            ):
                overflowed += 1
        out: dict[str, MetricValue] = {
            "sampler/compactions_mean": MetricValue.of_scalar(float(np.mean(counts))),
            "sampler/compactions_any": MetricValue.of_scalar(
                float(np.mean([1.0 if c > 0 else 0.0 for c in counts]))
            ),
            "sampler/prompt_too_long_frac": MetricValue.of_ratio(
                overflowed, len(trajectories)
            ),
        }
        if reclaimed:
            out["sampler/compaction_reclaimed_tokens_mean"] = MetricValue.of_scalar(
                float(np.mean(reclaimed))
            )
        return out


DEFAULT_METRIC_AGGREGATORS: list[MetricAggregator] = [
    CoreRewardMetricAggregator(),
    SamplerPolicyStepMetricAggregator(),
    FinishReasonMetricAggregator(),
    TimingMetricAggregator(),
    RolloutStatusMetricAggregator(),
    SamplerTokenSumMetricAggregator(),
    AgentStepsMetricAggregator(),
    CompactionMetricAggregator(),
]
