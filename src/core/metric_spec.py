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

"""Typed metric-aggregation specification.

Each metric has an explicit ``MetricSpec`` declaring its aggregation
``strategy`` and Layer-1 unit; the Layer 2 batch stage reads the spec
to pick the right combining function.

Pure logic: no I/O, no imports from ``pipeline``.

Recipe extensibility. Recipes register their own ``recipe/<name>/*``
specs at import time by mutating ``METRIC_SPECS`` (``.update({...})``)
alongside the aggregator that emits those keys. The framework itself
carries no recipe-scoped keys; each recipe owns its own
``emit_to_ml_platform`` policy.

Unregistered-key fallback. ``aggregate_by_spec`` treats an unregistered
key carrying a scalar ``MetricValue`` as MEAN with a one-shot WARNING,
so a recipe emitting new ad-hoc scalar keys doesn't crash the pipeline.
Non-scalar carriers still fail hard: emitting a specialized carrier
declares specialized-strategy intent and should be paired with a spec.
``spec_for`` itself still raises on unregistered keys.
"""

import dataclasses
import enum
import logging
import math
from collections.abc import Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Per-key one-shot warn caches; keep training-loop logs quiet after the
# first sighting.
_UNKNOWN_KEYS_WARNED: set[str] = set()
_SHAPE_MISMATCH_WARNED: set[str] = set()
_NAN_DROP_WARNED: set[str] = set()


class AggregationStrategy(enum.StrEnum):
    """How a metric's per-group values combine into one batch-stage value.

    Members are ``==`` / hash-equal to their string values, so they can be
    compared against raw strings loaded from config/JSON.
    """

    SUM = "sum"  # Sum values across the stage.
    MEAN = "mean"  # Unweighted arithmetic mean across the stage.
    WEIGHTED_MEAN = "weighted_mean"  # Count-weighted mean (correct for uneven groups).
    MAX = "max"  # Maximum across the stage.
    P50 = "p50"  # Median over ALL raw samples (not median-of-group-medians).
    P95 = "p95"  # 95th percentile over ALL raw samples.
    RATIO = "ratio"  # sum(count) / sum(total) across the stage.
    RATIO_OF_SUMS = "ratio_of_sums"  # sum(num) / sum(denom) across the stage.
    PASSTHROUGH = "passthrough"  # Value is already final; no cross-stage reduction.


@dataclasses.dataclass(frozen=True)
class MetricValue:
    """One metric's per-group contribution, carrying either a final scalar or
    the intermediate data its strategy needs to reduce further at batch stage.

    Exactly ONE of the five fields is set; ``__post_init__`` enforces this.
    Use the named constructors (``of_scalar`` etc.) rather than the raw
    constructor — they document intent and pick the right field.

    | strategy                         | field set     | shape          |
    |----------------------------------|---------------|----------------|
    | sum, mean, max, passthrough      | ``value``     | ``float``      |
    | weighted_mean                    | ``weighted_pair`` | ``(value, weight)`` |
    | p50, p95                         | ``samples``   | ``list[float]``|
    | ratio                            | ``count_total`` | ``(count, total)`` |
    | ratio_of_sums                    | ``part_total`` | ``(part, total)``|
    """

    value: float | None = None
    weighted_pair: tuple[float, int] | None = None
    samples: list[float] | None = None
    count_total: tuple[int, int] | None = None
    part_total: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        set_fields = [
            f.name
            for f in dataclasses.fields(self)
            if getattr(self, f.name) is not None
        ]
        if len(set_fields) != 1:
            raise ValueError(
                "MetricValue requires exactly one field set; "
                f"got {len(set_fields)}: {set_fields}"
            )
        # Reject NaN at construction so reducers never propagate NaN
        # into sinks; framework aggregators filter upstream, but recipe
        # aggregators can't be assumed to.
        if self.value is not None and math.isnan(self.value):
            raise ValueError("MetricValue.value must not be NaN")
        if self.weighted_pair is not None and math.isnan(self.weighted_pair[0]):
            raise ValueError("MetricValue.weighted_pair value must not be NaN")
        if self.samples is not None and any(math.isnan(s) for s in self.samples):
            raise ValueError("MetricValue.samples must not contain NaN")
        if self.part_total is not None and (
            math.isnan(self.part_total[0]) or math.isnan(self.part_total[1])
        ):
            raise ValueError("MetricValue.part_total values must not be NaN")
        # ``count_total`` is (int, int); ``int(float('nan'))`` raises,
        # so ``of_ratio`` is NaN-safe by construction.

    @classmethod
    def of_scalar(cls, value: float) -> "MetricValue":
        """For ``sum``, ``mean``, ``max``, ``passthrough`` — a plain value."""
        return cls(value=float(value))

    @classmethod
    def of_weighted(cls, value: float, weight: int) -> "MetricValue":
        """For ``weighted_mean`` — a ``(value, weight)`` pair."""
        return cls(weighted_pair=(float(value), int(weight)))

    @classmethod
    def of_samples(cls, samples: list[float]) -> "MetricValue":
        """For ``p50`` / ``p95`` — the raw underlying samples for this group."""
        return cls(samples=[float(s) for s in samples])

    @classmethod
    def of_ratio(cls, count: int, total: int) -> "MetricValue":
        """For ``ratio`` — a ``(count, total)`` pair; batch divides sum/sum."""
        return cls(count_total=(int(count), int(total)))

    @classmethod
    def of_part_total(cls, part: float, total: float) -> "MetricValue":
        """For ``ratio_of_sums`` — a ``(part, total)`` pair (a real-valued part
        out of a real-valued total); batch computes ``Σpart / Σtotal``."""
        return cls(part_total=(float(part), float(total)))


@dataclasses.dataclass(frozen=True)
class MetricSpec:
    """Aggregation contract for one metric.

    ``name`` matches the key the aggregator emits. ``strategy`` picks the
    batch-stage reducer. ``doc`` is a 2-sentence human-readable description:
    what the metric measures + how to interpret it. Required (every spec
    must document itself); surfaced by sink WARN messages and tooling.
    ``emit_to_ml_platform`` records whether the metric is forwarded to the
    ``MLPlatformSink`` (an experiment-tracker backend, typically Vertex
    Experiments) — the sink layer reads this; the spec only records it.
    """

    name: str
    strategy: AggregationStrategy
    doc: str
    emit_to_ml_platform: bool = False


def _spec(
    name: str,
    strategy: AggregationStrategy,
    doc: str,
    emit_to_ml_platform: bool = False,
) -> MetricSpec:
    return MetricSpec(
        name=name,
        strategy=strategy,
        doc=doc,
        emit_to_ml_platform=emit_to_ml_platform,
    )


# -----------------------------------------------------------------------
# Metric doc strings.
#
# Twin sharing: the same metric appears under multiple keys — a bare
# categorical name (``learning/reward/mean``), a ``train/*`` prefix used
# by the Site A emit path, and an ``eval/*`` prefix used by the Site C
# emit path. Rather than duplicate identical doc text across the trio,
# we hoist each doc to a module-level constant and reference it from
# every twin. Prevents drift: updating a metric's meaning updates every
# registered variant atomically.
#
# Each doc follows the "what it measures + how to interpret" convention,
# enforced by ``TestMetricSpecsRegistry`` in tests/unit/test_metric_spec.py.
# -----------------------------------------------------------------------

# --- Section 1: Learning metrics ---

_DOC_LEARNING_REWARD_MEAN = (
    "Mean reward across the trajectories in the current batch, weight-1 "
    "per trajectory (the bare form is emitted by "
    "``CoreRewardMetricAggregator``; the ``train/`` twin is the "
    "trainer-ingested slice — identical in sync mode, may differ in "
    "async with dynamic-sampling filtering). Track step-to-step to see "
    "if training is actually improving what you're rewarding; a flat "
    "or falling curve means the loss isn't tracking reward and the "
    "run is worthless — kill it."
)

_DOC_LEARNING_SAMPLING_ACCURACY = (
    "Fraction of trajectories in the current batch that the recipe "
    "judged correct (``atif_extras.get_is_correct(traj)``, or "
    "``reward > 0`` fallback). Complementary to ``reward/mean``: an "
    "agent can produce correct answers with low reward (e.g. long "
    "chains of thought penalized) or high reward with wrong answers "
    "(reward-hacking); watch both. Bare form counts all rollouts; "
    "``train/`` twin counts only what the trainer ingested (differs "
    "in async with dynamic-sampling filtering)."
)

_DOC_LEARNING_PASS_AT_K = (
    "Fraction of trajectory GROUPS where at least one of the "
    "``samples_per_task`` sampled trajectories was correct (per-group "
    "``any``, then batch mean). Higher than ``sampling_accuracy`` by "
    "construction; a widening gap between the two means the model gets "
    "the answer sometimes but is unreliable — sampling more will help, "
    "but the trained policy still varies."
)

# --- Section 2: Sampler metrics (behavioral) ---

_DOC_SAMPLER_POLICY_STEP = (
    "Which policy version served the samples in the current batch "
    "(weighted-mean of per-trajectory ``StepStats.train_step_id``). "
    "Kept jsonl-only because the ml_platform sink gets the derived "
    "``train/staleness_steps`` instead (current_train_step - this "
    "value), which is the actionable signal."
)

_DOC_SAMPLER_FINISH_REASON = (
    "Per-reason share of LLM generation calls (agent steps) that ended with "
    "this ``finish_reason`` (STOP / MAX_TOKENS / …). Denominator is the TOTAL "
    "agent steps across the whole batch (a shared global denominator applied to "
    "every reason, including reasons absent from some groups), so the per-reason "
    "ratios sum to 1.0 even when a reason is sparse. A spike in ``MAX_TOKENS`` "
    "means the model is running out of tokens before finishing — bump the token "
    "cap or investigate why responses are ballooning."
)

_DOC_SAMPLER_TRAJECTORY_STATUS = (
    "Per-status share of rollout trajectories with this "
    "``RolloutStatus`` (OK / max_step_limit / infra_error / …). Denominator is "
    "the TOTAL trajectories across the whole batch (a shared global denominator "
    "applied to every status, including statuses absent from some groups), so "
    "the per-status ratios sum to 1.0. Watch for rising failure statuses "
    "(``infra_error``, ``timeout``) — they signal transient environment issues; "
    "sustained ``max_step_limit`` means multi-step agents can't finish and need "
    "a higher step budget."
)

_DOC_SAMPLER_GENERATION_TOKENS_SUM = (
    "Total generation-side tokens the sampler produced across all steps "
    "in the current batch (sum of per-step "
    "``StepStats.generation_token_count``). Rising = LLM is generating "
    "more per call (longer chains of thought / longer answers); useful "
    "for cost tracking, and a spike often correlates with a "
    "``MAX_TOKENS`` increase in ``finish_reason/*``."
)

_DOC_SAMPLER_PEAK_CONTEXT_TOKENS_MEAN = (
    "Per-trajectory peak context length (the MAXIMUM over a rollout's "
    "agent steps of prompt + generation tokens), averaged across "
    "trajectories in the batch. Max rather than last-step because "
    "in-loop context compaction shrinks the running prompt, so a "
    "last-step reading would report the peak falling exactly when "
    "contexts grow. NOT a running sum — that's "
    "``generation_tokens_sum``; this "
    "is 'how close are conversations getting to the context-window "
    "limit', so a rising value predicts ``MAX_TOKENS`` finish reasons "
    "and eventual context-length truncation errors."
)

# --- Section 2b: Context-compaction metrics ---

_DOC_SAMPLER_COMPACTIONS_MEAN = (
    "Mean number of in-loop context-compaction events per rollout — one "
    "event is a step the agent spent asking the model to summarize the "
    "conversation so far instead of acting. Rising means rollouts are "
    "routinely running out of context window; each event costs a turn "
    "of the step budget and discards detail, so a value much above zero "
    "argues for a bigger window or a shorter task."
)

_DOC_SAMPLER_COMPACTIONS_ANY = (
    "Fraction of rollouts in the batch that compacted their context at "
    "least once, ignoring how many times each did. Read alongside "
    "``compactions_mean``: a high ``any`` with a low ``mean`` means "
    "broad but shallow pressure on the context window, while a low "
    "``any`` with a high ``mean`` means a few pathological rollouts are "
    "compacting repeatedly."
)

_DOC_SAMPLER_COMPACTION_RECLAIMED_TOKENS_MEAN = (
    "Mean tokens reclaimed per compaction event, measured as "
    "``compaction_pre_tokens - compaction_post_tokens`` over the "
    "conversation. This is the headroom each summary actually buys; a "
    "small value means the summary is not shrinking the conversation "
    "enough and the rollout will re-trigger compaction within a turn or "
    "two. Absent when nothing in the batch compacted."
)

_DOC_SAMPLER_PROMPT_TOO_LONG_FRAC = (
    "Fraction of rollouts that still terminated on an unfittable prompt "
    "(``SAMPLING_FAILURE_PROMPT_TOO_LONG``) despite compaction being "
    "available. This is compaction's success criterion: it should sit "
    "at approximately zero, and anything materially above that means "
    "the compaction trigger ratio is too loose to fire before the "
    "backend rejects the request."
)

# --- Section 3a: Pipeline metrics (per-LLM-call perf timings) ---

_DOC_PIPELINE_SAMPLE_TIME_AVG = (
    "Mean per-step sample time (LLM generation call latency) across all "
    "steps in the batch, from ``StepTimings`` (aggregated by "
    "``TimingMetricAggregator``). Compare with ``env_time_avg``: if "
    "sample dominates, the LLM is the bottleneck; if env dominates, "
    "the recipe env is slow."
)

_DOC_PIPELINE_ENV_TIME_AVG = (
    "Mean per-step environment execution time across all steps in the "
    "batch (recipe env's turn: tool calls, judge scoring, etc.). Paired "
    "with ``sample_time_avg`` for latency attribution; a rising "
    "``env_time_avg`` during a run usually means an upstream tool or "
    "judge slowed down."
)

_DOC_PIPELINE_SAMPLE_TIME_P95 = (
    "95th percentile of per-step sample time (LLM generation call "
    "latency), computed from all raw per-step values in the batch (NOT "
    "p95-of-group-p95). Catches tail-latency regressions that "
    "``sample_time_avg`` smooths over — a p95 spike with flat avg means "
    "a small fraction of calls are hanging."
)

_DOC_PIPELINE_ENV_TIME_P95 = (
    "95th percentile of per-step environment execution time, computed "
    "from all raw per-step values in the batch. Same tail-catching "
    "purpose as ``sample_time_p95`` but for the env side; a p95 spike "
    "often indicates a stuck subprocess or a slow tool call blocking a "
    "rollout."
)

_DOC_PIPELINE_ROLLOUT_WALL_TIME_AVG = (
    "Mean true end-to-end wall clock of one rollout, in seconds, "
    "measured across the whole rollout body (env build, every LLM call, "
    "every tool/sandbox turn, retries, scoring) — NOT the sum of the "
    "per-step ``sample_time_avg`` + ``env_time_avg``, which miss "
    "everything between the steps. Compare the two: a large gap is time "
    "the per-step timers cannot see, and is where a slow sandbox or a "
    "retry storm hides."
)

_DOC_PIPELINE_ROLLOUT_WALL_TIME_P95 = (
    "95th percentile of per-rollout end-to-end wall clock, computed "
    "over all raw per-rollout values in the batch (NOT p95-of-group-"
    "p95). This is the straggler metric for async rollout: it sets how "
    "long the trainer can be kept waiting by the slowest group in a "
    "step, so a p95 that pulls away from the avg means a few rollouts "
    "are dominating step latency."
)

_DOC_PIPELINE_SAMPLE_TIME_FRACTION = (
    "Fraction of total step wall time spent in the sampler vs the env, "
    "computed as ``sum(sample_time) / sum(sample_time + env_time)`` "
    "over all steps in the batch (ratio_of_sums, not mean-of-ratios). "
    "High = LLM dominates wall time (add sampler concurrency); low = "
    "env dominates (env compute is the bottleneck)."
)

# --- Section 3b: Pipeline metrics (plumbing / passthrough gauges) ---

_DOC_PIPELINE_TRAIN_STEP_TIME = (
    "Wall-clock time in seconds around the ``trainer.train_step(...)`` "
    "call, including LRO polling wait on real Vertex. Watch step-over-"
    "step: a jump means the server-side step got slower or LRO polling "
    "is hitting a queue; correlate with ``train/train_step_seconds`` "
    "(server-reported) to separate client wait from server work."
)

_DOC_PIPELINE_WAIT_FOR_ROLLOUT_TIME = (
    "Wall-clock time in seconds the trainer waited on "
    "``pipeline.next_batch(...)`` before this train step. In sync mode "
    "it's how long the rollout loop took; in async mode it's how long "
    "the trainer sat idle waiting for the buffer to fill — a sustained "
    "high value in async signals sampler is bottleneck."
)

_DOC_PIPELINE_BUFFER_UTILIZATION = (
    "Async pipeline's inter-thread buffer occupancy sampled at train-"
    "step time (``qsize / maxsize``). Sustained low = sampler is "
    "starving the trainer (need more sampler concurrency); sustained "
    "near 1.0 = trainer is falling behind (back-pressure engaging, need "
    "faster trainer or more sampler headroom)."
)

_DOC_PIPELINE_GROUP_DROP_RATIO = (
    "Fraction of dispatched trajectory groups that got filtered out "
    "before reaching the trainer this step (dropped_groups / "
    "dispatched_groups). Reasons: fewer than 2 valid trajectories in "
    "the group, too many transient failures, or zero-variance rewards; "
    "a rising value shrinks effective batch size and hurts gradient "
    "quality. Sync counts the groups its own accumulate loop inspected; "
    "async counts the rollouts that FINISHED since the previous train "
    "step, because there dispatch and consumption are decoupled and a "
    "step's own batch excludes its drops by construction."
)

_DOC_PIPELINE_GROUP_DROP_REASON = (
    "Per-reason share of the trajectory groups the filter classified "
    "this step (``zero_variance`` / ``too_few_valid_trajectories`` / "
    "``too_many_transient_failures``), breaking down "
    "``pipeline/group_drop_ratio``. Emitted every step including zeros, "
    "so a flat-zero series reads as 'that reason never fires' rather "
    "than 'not measured'. The two structural reasons are infra signals "
    "(rollouts failing, NaN rewards); ``zero_variance`` is the tunable "
    "one, driven by ``zero_variance_keep_prob``. Denominated in "
    "CLASSIFIED groups, so in async a window containing a dispatch that "
    "raised makes the three shares sum to slightly under "
    "``group_drop_ratio`` — a raised dispatch never reached the filter."
)

_DOC_PIPELINE_GROUP_ZERO_VARIANCE_RATIO = (
    "Fraction of trajectory groups this step whose rewards were all "
    "equal (within 1e-5), counted whether or not the group was then "
    "dropped. This is the one to read when asking whether zero-variance "
    "filtering earns its keep on a recipe: at "
    "``zero_variance_keep_prob=1.0`` nothing is ever dropped for it, so "
    "``group_drop_reason/zero_variance`` sits at 0 and says nothing, "
    "while this still shows how often groups tie. Near zero on a "
    "continuous-valued reward (exact N-way ties are rare); substantial "
    "on a binary pass/fail reward, where whole groups routinely land "
    "all-0 or all-1."
)

_DOC_PIPELINE_ITERATOR_CURRENT = (
    "Dataset iterator cursor position at train-step boundary (which "
    "task index the next batch will pull from). MUST stay jsonl-only "
    "because ``resolve_resume_position`` reads this back to resume a "
    "killed run — routing to the ml_platform sink would mix a step-"
    "shaped axis into aggregated dashboards."
)

# --- Section 3c: Site B train/* passthroughs (rollout_pipeline direct) ---

_DOC_TRAIN_STALENESS_STEPS = (
    "Derived at train-step boundary: ``global_step - "
    "mean(sampler/policy_step across batch)`` = how many training steps "
    "ahead the trainer is of the policy version that served this "
    "batch's samples. In sync mode this is ~0; in async mode a rising "
    "value signals the sampler is falling behind and training is on "
    "stale data — dashboard-actionable for tuning async rollout "
    "concurrency."
)

_DOC_TRAIN_RUNNING_PASS_RATE = (
    "EMA of 'fraction of trajectory groups kept' (survived "
    "``TrajectoryGroupFilter``), updated per group inspected across all "
    "rounds of the dynamic sampling loop for this step. Used to predict "
    "``est_to_read`` for the NEXT step's round-1 task pull; a "
    "collapsing value inflates read estimates and pushes "
    "``rollout_rounds`` up."
)

_DOC_TRAIN_ROLLOUT_ROUNDS = (
    "How many iterations of the dynamic-sampling outer loop it took to "
    "accumulate ``n`` valid trajectory groups for this training step. "
    "Round count = 1 means round-1 estimate was enough (healthy); "
    "repeated rounds = pass rate collapsed relative to the EMA, so the "
    "estimator kept under-shooting."
)

_DOC_TRAIN_ROLLOUT_TASKS_TOTAL = (
    "Total tasks pulled from the dataset iterator across all rounds of "
    "this train step's rollout loop. Divide by ``rollout_rounds`` for "
    "average tasks per round; complement of ``first_round_rollout_tasks``."
)

_DOC_TRAIN_FIRST_ROUND_ROLLOUT_TASKS = (
    "Tasks pulled on round 1 only (subset of ``rollout_tasks_total``). "
    "When this equals ``rollout_tasks_total``, the EMA was well-"
    "calibrated and one round sufficed; when much smaller, the pass "
    "rate collapsed and multiple rounds were needed — useful for "
    "diagnosing whether ``running_pass_rate`` is chasing a moving "
    "target."
)

# --- Section 4: Trainer wire metrics (TRAINER_METRICS twins) ---
# (loss is deliberately absent: real Vertex never emits it, and the
# alternate-backend synthetic loss doesn't earn the doc/spec bookkeeping.)

_DOC_TRAIN_TRAIN_STEP_SECONDS = (
    "Wall-clock time in seconds for one server-side train step, as "
    "reported by the trainer wire. Averaged across the aggregation "
    "stage; useful cross-run telemetry for spotting server-side perf "
    "regressions (compare to client-side "
    "``train/pipeline/train_step_time`` which includes LRO polling)."
)

_DOC_TRAIN_BILLABLE_TRAINING_TOKENS = (
    "Total tokens counted for billing this train step (server-computed), "
    "summed across the aggregation stage so the aggregate is total "
    "billable tokens over the batch. Useful for cost tracking — divide "
    "by ``train_step_seconds`` for a tokens-per-second-of-billed-work "
    "rate."
)

_DOC_TRAIN_SERVER_TRAIN_STEP = (
    "Server-side monotonically-increasing train step counter, surfaced "
    "under a distinct client_key so it doesn't collide with the "
    "framework's own step column. Kept jsonl-only because the "
    "ml_platform sink already carries an authoritative step column and "
    "a second step-shaped axis would confuse the timeline; use only "
    "for cross-check during debugging."
)

_DOC_TRAIN_TRAINING_TOKENS_BEFORE_MERGE = (
    "Tokens the trainer has seen since the last weight-merge boundary "
    "(server-reported). Summed across the aggregation stage for a "
    "total-tokens training-progress plot on the ml_platform sink's "
    "dashboard."
)

_DOC_TRAIN_MEAN_ABS_ADVANTAGE = (
    "Mean absolute advantage in the batch (RL diagnostic: how large the "
    "reward signal is after baselining). May arrive as ``None`` on "
    "train steps where the trainer hasn't computed it; the framework "
    "aggregator skips ``None`` values before taking mean so a run "
    "mixing None + numeric values is safe."
)

# --- Site A extension: trainer-returned keys not routed via TRAINER_METRICS ---

_DOC_TRAIN_COMPLETED_TRAIN_STEP_ID = (
    "Server-side completed train-step counter, mirrored from the wire "
    "response's ``completedTrainStepId`` field. Not forwarded to the "
    "ml_platform sink because the framework already carries an "
    "authoritative step column and a second step-shaped axis would "
    "confuse the timeline."
)

_DOC_TRAIN_TRAJECTORY_MISSING_CANDIDATE_ID = (
    "Count of trajectories the trainer skipped in this batch because "
    "their trajectory-level ``candidate_id`` was missing (client-side "
    "count computed by whichever backend implementation is wired in). A "
    "rising value signals a producer/schema drift."
)

_DOC_TRAIN_AUDIT_USED_CANDIDATES = (
    "Count of candidates the trainer accepted into this train step, "
    "parsed from the wire response's ``candidateAudit`` block by "
    "``core.trainer_audit.parse_candidate_audit``. Complements "
    "``train/audit/rejected_candidates`` for a full accept/reject "
    "breakdown per step."
)

_DOC_TRAIN_AUDIT_REJECTED_CANDIDATES = (
    "Total count of candidates the trainer rejected in this train "
    "step. See ``train/audit/rejected_reason/*`` for the per-reason "
    "breakdown (top-N reasons only)."
)

_DOC_TRAIN_AUDIT_REJECTED_REASON = (
    "Per-reason count of rejected candidates in this train step "
    "(dynamic-suffix family). The suffix is the rejection reason "
    "string from the wire response's candidateAudit; capped at top-5 "
    "most frequent reasons per ``parse_candidate_audit`` to keep the "
    "metrics dict bounded when a degenerate run produces many distinct "
    "reasons."
)

_DOC_TRAIN_SAMPLER_FINISH_REASON = (
    "Per-finish-reason SHARE of agent steps over the trajectories the trainer "
    "ingested this step (train-step twin of ``sampler/finish_reason/*``). "
    "Emitted by ``Trainer`` implementations (``VertexMsrlRestClient.train_step`` "
    "and alternate-backend equivalents) as ratios where the denominator is the "
    "total agent-step count (the sum of all per-reason counts), so the per-reason "
    "ratios sum to 1.0. Computed directly at the trainer emit site, not via the "
    "batch-stage reducer. A spike in ``MAX_TOKENS`` means the model is running "
    "out of tokens before finishing."
)


# Registry of framework metric specs.
#
# Keys ending in "/*" are DYNAMIC-SUFFIX FAMILIES: any concrete key
# with that prefix resolves to the family spec via ``spec_for`` (e.g.
# "sampler/finish_reason/STOP" -> the "sampler/finish_reason/*"
# entry). The family spec's ``name`` is a template, not an emitted key.
METRIC_SPECS: dict[str, MetricSpec] = {
    # --- Section 1: Learning metrics ---
    "learning/reward/mean": _spec(
        "learning/reward/mean",
        AggregationStrategy.WEIGHTED_MEAN,
        _DOC_LEARNING_REWARD_MEAN,
        emit_to_ml_platform=True,
    ),
    "learning/sampling_accuracy": _spec(
        "learning/sampling_accuracy",
        AggregationStrategy.MEAN,
        _DOC_LEARNING_SAMPLING_ACCURACY,
        emit_to_ml_platform=True,
    ),
    "learning/pass_at_k": _spec(
        "learning/pass_at_k",
        AggregationStrategy.MEAN,  # batch stage: fraction of groups w/ any-correct
        _DOC_LEARNING_PASS_AT_K,
        emit_to_ml_platform=True,
    ),
    # --- Section 2: Sampler metrics (behavioral) ---
    "sampler/policy_step": _spec(
        "sampler/policy_step",
        AggregationStrategy.WEIGHTED_MEAN,
        _DOC_SAMPLER_POLICY_STEP,
        emit_to_ml_platform=False,  # ml_platform gets train/staleness_steps instead
    ),
    "sampler/finish_reason/*": _spec(
        "sampler/finish_reason/",
        AggregationStrategy.RATIO,
        _DOC_SAMPLER_FINISH_REASON,
        emit_to_ml_platform=True,
    ),
    "sampler/trajectory_status/*": _spec(
        "sampler/trajectory_status/",
        AggregationStrategy.RATIO,
        _DOC_SAMPLER_TRAJECTORY_STATUS,
        emit_to_ml_platform=True,
    ),
    "sampler/generation_tokens_sum": _spec(
        "sampler/generation_tokens_sum",
        AggregationStrategy.SUM,
        _DOC_SAMPLER_GENERATION_TOKENS_SUM,
        emit_to_ml_platform=True,
    ),
    "sampler/peak_context_tokens_mean": _spec(
        "sampler/peak_context_tokens_mean",
        AggregationStrategy.MEAN,
        _DOC_SAMPLER_PEAK_CONTEXT_TOKENS_MEAN,
        emit_to_ml_platform=True,
    ),
    "sampler/agent_steps_mean": _spec(
        "sampler/agent_steps_mean",
        AggregationStrategy.MEAN,
        "Average number of agent steps (LLM turns) per rollout trajectory. "
        "Tracks budget utilization and straggler behavior. A rollout that "
        "hits ``step_limit`` will show ``agent_steps_mean ≈ step_limit``; "
        "easy tasks submit early and show a lower mean.",
        emit_to_ml_platform=True,
    ),
    "sampler/compactions_mean": _spec(
        "sampler/compactions_mean",
        AggregationStrategy.MEAN,
        _DOC_SAMPLER_COMPACTIONS_MEAN,
        emit_to_ml_platform=True,
    ),
    "sampler/compactions_any": _spec(
        "sampler/compactions_any",
        AggregationStrategy.MEAN,
        _DOC_SAMPLER_COMPACTIONS_ANY,
        emit_to_ml_platform=True,
    ),
    "sampler/compaction_reclaimed_tokens_mean": _spec(
        "sampler/compaction_reclaimed_tokens_mean",
        AggregationStrategy.MEAN,
        _DOC_SAMPLER_COMPACTION_RECLAIMED_TOKENS_MEAN,
        emit_to_ml_platform=True,
    ),
    "sampler/prompt_too_long_frac": _spec(
        "sampler/prompt_too_long_frac",
        AggregationStrategy.RATIO,
        _DOC_SAMPLER_PROMPT_TOO_LONG_FRAC,
        emit_to_ml_platform=True,
    ),
    # --- Section 3a: Pipeline metrics (per-LLM-call perf timings) ---
    "pipeline/sample_time_avg": _spec(
        "pipeline/sample_time_avg",
        AggregationStrategy.MEAN,
        _DOC_PIPELINE_SAMPLE_TIME_AVG,
        emit_to_ml_platform=True,
    ),
    "pipeline/env_time_avg": _spec(
        "pipeline/env_time_avg",
        AggregationStrategy.MEAN,
        _DOC_PIPELINE_ENV_TIME_AVG,
        emit_to_ml_platform=True,
    ),
    "pipeline/sample_time_p95": _spec(
        "pipeline/sample_time_p95",
        AggregationStrategy.P95,
        _DOC_PIPELINE_SAMPLE_TIME_P95,
        emit_to_ml_platform=True,
    ),
    "pipeline/env_time_p95": _spec(
        "pipeline/env_time_p95",
        AggregationStrategy.P95,
        _DOC_PIPELINE_ENV_TIME_P95,
        emit_to_ml_platform=True,
    ),
    "pipeline/sample_time_fraction": _spec(
        "pipeline/sample_time_fraction",
        AggregationStrategy.RATIO_OF_SUMS,
        _DOC_PIPELINE_SAMPLE_TIME_FRACTION,
        emit_to_ml_platform=True,
    ),
    "pipeline/rollout_wall_time_avg": _spec(
        "pipeline/rollout_wall_time_avg",
        AggregationStrategy.MEAN,
        _DOC_PIPELINE_ROLLOUT_WALL_TIME_AVG,
        emit_to_ml_platform=True,
    ),
    "pipeline/rollout_wall_time_p95": _spec(
        "pipeline/rollout_wall_time_p95",
        AggregationStrategy.P95,
        _DOC_PIPELINE_ROLLOUT_WALL_TIME_P95,
        emit_to_ml_platform=True,
    ),
    # --- Section 3b: Pipeline metrics (plumbing / passthrough gauges) ---
    "pipeline/train_step_time": _spec(
        "pipeline/train_step_time",
        AggregationStrategy.PASSTHROUGH,
        _DOC_PIPELINE_TRAIN_STEP_TIME,
        emit_to_ml_platform=True,
    ),
    "pipeline/wait_for_rollout_time": _spec(
        "pipeline/wait_for_rollout_time",
        AggregationStrategy.PASSTHROUGH,
        _DOC_PIPELINE_WAIT_FOR_ROLLOUT_TIME,
        emit_to_ml_platform=True,
    ),
    "pipeline/buffer_utilization": _spec(
        "pipeline/buffer_utilization",
        AggregationStrategy.PASSTHROUGH,
        _DOC_PIPELINE_BUFFER_UTILIZATION,
        emit_to_ml_platform=True,
    ),
    "pipeline/group_drop_ratio": _spec(
        "pipeline/group_drop_ratio",
        AggregationStrategy.RATIO,
        _DOC_PIPELINE_GROUP_DROP_RATIO,
        emit_to_ml_platform=True,
    ),
    "pipeline/group_drop_reason/*": _spec(
        "pipeline/group_drop_reason/",
        AggregationStrategy.RATIO,
        _DOC_PIPELINE_GROUP_DROP_REASON,
        emit_to_ml_platform=True,
    ),
    "pipeline/group_zero_variance_ratio": _spec(
        "pipeline/group_zero_variance_ratio",
        AggregationStrategy.RATIO,
        _DOC_PIPELINE_GROUP_ZERO_VARIANCE_RATIO,
        emit_to_ml_platform=True,
    ),
    "pipeline/iterator_current": _spec(
        "pipeline/iterator_current",
        AggregationStrategy.PASSTHROUGH,
        _DOC_PIPELINE_ITERATOR_CURRENT,
        emit_to_ml_platform=False,  # MUST stay jsonl-only (resume marker)
    ),
    # --- Site A: keys emitted by train_and_eval.py with the `train/` prefix
    # applied at emit time. These are the CONCRETE keys the SinkDispatcher
    # sees at runtime, one per corresponding categorical spec above. Duplicating
    # them under the `train/*` shape keeps MLPlatformSink accepting them
    # (spec resolution requires an exact / prefix match). The strategy and
    # unit mirror the categorical spec they wrap; emit_to_ml_platform mirrors
    # the wrapped counterpart's. This dual-shape registration was chosen over
    # a per-metric sink-side rewrite because it needs the least code churn.
    "train/pipeline/train_step_time": _spec(
        "train/pipeline/train_step_time",
        AggregationStrategy.PASSTHROUGH,
        _DOC_PIPELINE_TRAIN_STEP_TIME,
        emit_to_ml_platform=True,
    ),
    "train/pipeline/wait_for_rollout_time": _spec(
        "train/pipeline/wait_for_rollout_time",
        AggregationStrategy.PASSTHROUGH,
        _DOC_PIPELINE_WAIT_FOR_ROLLOUT_TIME,
        emit_to_ml_platform=True,
    ),
    "train/pipeline/buffer_utilization": _spec(
        "train/pipeline/buffer_utilization",
        AggregationStrategy.PASSTHROUGH,
        _DOC_PIPELINE_BUFFER_UTILIZATION,
        emit_to_ml_platform=True,
    ),
    "train/pipeline/group_drop_ratio": _spec(
        "train/pipeline/group_drop_ratio",
        AggregationStrategy.RATIO,
        _DOC_PIPELINE_GROUP_DROP_RATIO,
        emit_to_ml_platform=True,
    ),
    "train/pipeline/group_drop_reason/*": _spec(
        "train/pipeline/group_drop_reason/",
        AggregationStrategy.RATIO,
        _DOC_PIPELINE_GROUP_DROP_REASON,
        emit_to_ml_platform=True,
    ),
    "train/pipeline/group_zero_variance_ratio": _spec(
        "train/pipeline/group_zero_variance_ratio",
        AggregationStrategy.RATIO,
        _DOC_PIPELINE_GROUP_ZERO_VARIANCE_RATIO,
        emit_to_ml_platform=True,
    ),
    "train/pipeline/iterator_current": _spec(
        "train/pipeline/iterator_current",
        AggregationStrategy.PASSTHROUGH,
        _DOC_PIPELINE_ITERATOR_CURRENT,
        emit_to_ml_platform=False,  # MUST stay jsonl-only (resume marker)
    ),
    "train/sampler/policy_step": _spec(
        "train/sampler/policy_step",
        AggregationStrategy.WEIGHTED_MEAN,
        _DOC_SAMPLER_POLICY_STEP,
        emit_to_ml_platform=False,  # ml_platform gets train/staleness_steps instead
    ),
    # --- train/learning/* twins: per-train-step reward/accuracy logged
    # by ``train_and_eval.py`` under the ``train/`` prefix.  Mirrors the
    # ``eval/learning/*`` pattern (Site C) so the Vertex Experiment UI
    # shows both train-time and eval-time learning curves.
    "train/learning/reward/mean": _spec(
        "train/learning/reward/mean",
        AggregationStrategy.WEIGHTED_MEAN,
        _DOC_LEARNING_REWARD_MEAN,
        emit_to_ml_platform=True,
    ),
    "train/learning/sampling_accuracy": _spec(
        "train/learning/sampling_accuracy",
        AggregationStrategy.MEAN,
        _DOC_LEARNING_SAMPLING_ACCURACY,
        emit_to_ml_platform=True,
    ),
    "train/learning/pass_at_k": _spec(
        "train/learning/pass_at_k",
        AggregationStrategy.MEAN,
        _DOC_LEARNING_PASS_AT_K,
        emit_to_ml_platform=True,
    ),
    "train/staleness_steps": _spec(
        "train/staleness_steps",
        AggregationStrategy.PASSTHROUGH,
        _DOC_TRAIN_STALENESS_STEPS,
        emit_to_ml_platform=True,
    ),
    # Trainer wire-metrics. client_key from TRAINER_METRICS gets
    # wrapped with `train/` at emit time by train_and_eval_loop (Site A).
    "train/train_step_seconds": _spec(
        "train/train_step_seconds",
        AggregationStrategy.MEAN,
        _DOC_TRAIN_TRAIN_STEP_SECONDS,
        emit_to_ml_platform=True,
    ),
    "train/billable_training_tokens": _spec(
        "train/billable_training_tokens",
        AggregationStrategy.SUM,
        _DOC_TRAIN_BILLABLE_TRAINING_TOKENS,
        emit_to_ml_platform=True,
    ),
    "train/server_train_step": _spec(
        "train/server_train_step",
        AggregationStrategy.PASSTHROUGH,
        _DOC_TRAIN_SERVER_TRAIN_STEP,
        emit_to_ml_platform=False,  # framework already carries a step column
    ),
    "train/training_tokens_before_merge": _spec(
        "train/training_tokens_before_merge",
        AggregationStrategy.SUM,
        _DOC_TRAIN_TRAINING_TOKENS_BEFORE_MERGE,
        emit_to_ml_platform=True,
    ),
    "train/mean_abs_advantage": _spec(
        "train/mean_abs_advantage",
        AggregationStrategy.MEAN,
        _DOC_TRAIN_MEAN_ABS_ADVANTAGE,
        emit_to_ml_platform=True,
    ),
    # --- Site A extension: keys returned directly by trainer.train_step
    # (not via TRAINER_METRICS wire mapping) that ALSO get `train/`
    # wrapped at emit time by train_and_eval.py:283. These are populated
    # by ``Trainer`` implementations (``VertexMsrlRestClient.train_step``
    # and alternate-backend equivalents). Missing entries here would
    # silently drop metrics in MLPlatformSink.
    "train/completed_train_step_id": _spec(
        "train/completed_train_step_id",
        AggregationStrategy.PASSTHROUGH,
        _DOC_TRAIN_COMPLETED_TRAIN_STEP_ID,
        emit_to_ml_platform=False,
    ),
    "train/trajectory/missing_candidate_id": _spec(
        "train/trajectory/missing_candidate_id",
        AggregationStrategy.PASSTHROUGH,
        _DOC_TRAIN_TRAJECTORY_MISSING_CANDIDATE_ID,
        emit_to_ml_platform=True,
    ),
    "train/audit/used_candidates": _spec(
        "train/audit/used_candidates",
        AggregationStrategy.PASSTHROUGH,
        _DOC_TRAIN_AUDIT_USED_CANDIDATES,
        emit_to_ml_platform=True,
    ),
    "train/audit/rejected_candidates": _spec(
        "train/audit/rejected_candidates",
        AggregationStrategy.PASSTHROUGH,
        _DOC_TRAIN_AUDIT_REJECTED_CANDIDATES,
        emit_to_ml_platform=True,
    ),
    # NOTE: both dynamic-suffix specs below reach the SinkDispatcher
    # DIRECTLY from ``trainer.train_step()``'s return dict (not through
    # ``aggregate_by_spec``), so the batch reducer never runs for them.
    # ``train/audit/rejected_reason/*`` keeps a PASSTHROUGH placeholder
    # (count semantics, one value per key). ``train/sampler/finish_reason/*``
    # is RATIO to document its emitted-value contract: the trainer emits
    # per-reason SHARES (count / total agent steps, summing to 1.0) — see
    # ``_finish_reason_ratios`` in ``backends/vertex_msrl_rest.py``. If a
    # future refactor routes trainer-emitted metrics through the aggregation
    # seam, revisit these: PASSTHROUGH would raise on multiple values per
    # key, and the RATIO reducer expects ``of_ratio(count, total)`` carriers
    # rather than the bare ratio floats emitted here.
    "train/audit/rejected_reason/*": _spec(
        "train/audit/rejected_reason/",
        AggregationStrategy.PASSTHROUGH,
        _DOC_TRAIN_AUDIT_REJECTED_REASON,
        emit_to_ml_platform=False,
    ),
    "train/sampler/finish_reason/*": _spec(
        "train/sampler/finish_reason/",
        AggregationStrategy.RATIO,
        _DOC_TRAIN_SAMPLER_FINISH_REASON,
        emit_to_ml_platform=True,
    ),
    "train/sampler/trajectory_status/*": _spec(
        "train/sampler/trajectory_status/",
        AggregationStrategy.RATIO,
        _DOC_SAMPLER_TRAJECTORY_STATUS,
        emit_to_ml_platform=True,
    ),
    "train/sampler/generation_tokens_sum": _spec(
        "train/sampler/generation_tokens_sum",
        AggregationStrategy.SUM,
        _DOC_SAMPLER_GENERATION_TOKENS_SUM,
        emit_to_ml_platform=True,
    ),
    "train/sampler/peak_context_tokens_mean": _spec(
        "train/sampler/peak_context_tokens_mean",
        AggregationStrategy.MEAN,
        _DOC_SAMPLER_PEAK_CONTEXT_TOKENS_MEAN,
        emit_to_ml_platform=True,
    ),
    "train/sampler/agent_steps_mean": _spec(
        "train/sampler/agent_steps_mean",
        AggregationStrategy.MEAN,
        "Average number of agent steps (LLM turns) per rollout trajectory "
        "(train-step twin of ``sampler/agent_steps_mean``). A rollout that "
        "hits ``step_limit`` shows a value near the limit; easy tasks "
        "submit early and produce a lower mean.",
        emit_to_ml_platform=True,
    ),
    "train/sampler/compactions_mean": _spec(
        "train/sampler/compactions_mean",
        AggregationStrategy.MEAN,
        _DOC_SAMPLER_COMPACTIONS_MEAN,
        emit_to_ml_platform=True,
    ),
    "train/sampler/compactions_any": _spec(
        "train/sampler/compactions_any",
        AggregationStrategy.MEAN,
        _DOC_SAMPLER_COMPACTIONS_ANY,
        emit_to_ml_platform=True,
    ),
    "train/sampler/compaction_reclaimed_tokens_mean": _spec(
        "train/sampler/compaction_reclaimed_tokens_mean",
        AggregationStrategy.MEAN,
        _DOC_SAMPLER_COMPACTION_RECLAIMED_TOKENS_MEAN,
        emit_to_ml_platform=True,
    ),
    "train/sampler/prompt_too_long_frac": _spec(
        "train/sampler/prompt_too_long_frac",
        AggregationStrategy.RATIO,
        _DOC_SAMPLER_PROMPT_TOO_LONG_FRAC,
        emit_to_ml_platform=True,
    ),
    # --- train/pipeline/* timing twins: train_and_eval.py prepends `train/`
    # to the five per-LLM-call timing keys from aggregate_by_spec(...) output.
    # Register the train-side shape so the SinkDispatcher sees a spec.
    "train/pipeline/sample_time_avg": _spec(
        "train/pipeline/sample_time_avg",
        AggregationStrategy.MEAN,
        _DOC_PIPELINE_SAMPLE_TIME_AVG,
        emit_to_ml_platform=True,
    ),
    "train/pipeline/env_time_avg": _spec(
        "train/pipeline/env_time_avg",
        AggregationStrategy.MEAN,
        _DOC_PIPELINE_ENV_TIME_AVG,
        emit_to_ml_platform=True,
    ),
    "train/pipeline/sample_time_p95": _spec(
        "train/pipeline/sample_time_p95",
        AggregationStrategy.P95,
        _DOC_PIPELINE_SAMPLE_TIME_P95,
        emit_to_ml_platform=True,
    ),
    "train/pipeline/env_time_p95": _spec(
        "train/pipeline/env_time_p95",
        AggregationStrategy.P95,
        _DOC_PIPELINE_ENV_TIME_P95,
        emit_to_ml_platform=True,
    ),
    "train/pipeline/sample_time_fraction": _spec(
        "train/pipeline/sample_time_fraction",
        AggregationStrategy.RATIO_OF_SUMS,
        _DOC_PIPELINE_SAMPLE_TIME_FRACTION,
        emit_to_ml_platform=True,
    ),
    "train/pipeline/rollout_wall_time_avg": _spec(
        "train/pipeline/rollout_wall_time_avg",
        AggregationStrategy.MEAN,
        _DOC_PIPELINE_ROLLOUT_WALL_TIME_AVG,
        emit_to_ml_platform=True,
    ),
    "train/pipeline/rollout_wall_time_p95": _spec(
        "train/pipeline/rollout_wall_time_p95",
        AggregationStrategy.P95,
        _DOC_PIPELINE_ROLLOUT_WALL_TIME_P95,
        emit_to_ml_platform=True,
    ),
    # --- Site B: keys emitted by sync_rollout_pipeline.py:371-375 as hard-coded
    # `train/*` names (historical shape, not covered by F2 target-state
    # registry). Registering as identity so MLPlatformSink accepts them.
    "train/running_pass_rate": _spec(
        "train/running_pass_rate",
        AggregationStrategy.PASSTHROUGH,
        _DOC_TRAIN_RUNNING_PASS_RATE,
        emit_to_ml_platform=True,
    ),
    "train/rollout_rounds": _spec(
        "train/rollout_rounds",
        AggregationStrategy.PASSTHROUGH,
        _DOC_TRAIN_ROLLOUT_ROUNDS,
        emit_to_ml_platform=True,
    ),
    "train/rollout_tasks_total": _spec(
        "train/rollout_tasks_total",
        AggregationStrategy.PASSTHROUGH,
        _DOC_TRAIN_ROLLOUT_TASKS_TOTAL,
        emit_to_ml_platform=True,
    ),
    "train/first_round_rollout_tasks": _spec(
        "train/first_round_rollout_tasks",
        AggregationStrategy.PASSTHROUGH,
        _DOC_TRAIN_FIRST_ROUND_ROLLOUT_TASKS,
        emit_to_ml_platform=True,
    ),
    # --- Site C: keys emitted by eval_only.py:350 with the `eval/` prefix
    # applied at emit time. One per corresponding categorical spec, mirroring
    # the Site A `train/*` dual-shape pattern. Each twin shares the doc
    # constant with its bare counterpart.
    "eval/learning/reward/mean": _spec(
        "eval/learning/reward/mean",
        AggregationStrategy.WEIGHTED_MEAN,
        _DOC_LEARNING_REWARD_MEAN,
        emit_to_ml_platform=True,
    ),
    "eval/learning/sampling_accuracy": _spec(
        "eval/learning/sampling_accuracy",
        AggregationStrategy.MEAN,
        _DOC_LEARNING_SAMPLING_ACCURACY,
        emit_to_ml_platform=True,
    ),
    "eval/learning/pass_at_k": _spec(
        "eval/learning/pass_at_k",
        AggregationStrategy.MEAN,
        _DOC_LEARNING_PASS_AT_K,
        emit_to_ml_platform=True,
    ),
    "eval/sampler/policy_step": _spec(
        "eval/sampler/policy_step",
        AggregationStrategy.WEIGHTED_MEAN,
        _DOC_SAMPLER_POLICY_STEP,
        emit_to_ml_platform=False,  # jsonl-only, same as train side
    ),
    "eval/sampler/finish_reason/*": _spec(
        "eval/sampler/finish_reason/",
        AggregationStrategy.RATIO,
        _DOC_SAMPLER_FINISH_REASON,
        emit_to_ml_platform=True,
    ),
    "eval/sampler/trajectory_status/*": _spec(
        "eval/sampler/trajectory_status/",
        AggregationStrategy.RATIO,
        _DOC_SAMPLER_TRAJECTORY_STATUS,
        emit_to_ml_platform=True,
    ),
    "eval/sampler/generation_tokens_sum": _spec(
        "eval/sampler/generation_tokens_sum",
        AggregationStrategy.SUM,
        _DOC_SAMPLER_GENERATION_TOKENS_SUM,
        emit_to_ml_platform=True,
    ),
    "eval/sampler/peak_context_tokens_mean": _spec(
        "eval/sampler/peak_context_tokens_mean",
        AggregationStrategy.MEAN,
        _DOC_SAMPLER_PEAK_CONTEXT_TOKENS_MEAN,
        emit_to_ml_platform=True,
    ),
    "eval/sampler/agent_steps_mean": _spec(
        "eval/sampler/agent_steps_mean",
        AggregationStrategy.MEAN,
        "Average number of agent steps (LLM turns) per rollout trajectory. "
        "Tracks budget utilization and straggler behavior. A rollout that "
        "hits ``step_limit`` shows a value near the limit; easy tasks "
        "submit early and produce a lower mean.",
        emit_to_ml_platform=True,
    ),
    "eval/sampler/compactions_mean": _spec(
        "eval/sampler/compactions_mean",
        AggregationStrategy.MEAN,
        _DOC_SAMPLER_COMPACTIONS_MEAN,
        emit_to_ml_platform=True,
    ),
    "eval/sampler/compactions_any": _spec(
        "eval/sampler/compactions_any",
        AggregationStrategy.MEAN,
        _DOC_SAMPLER_COMPACTIONS_ANY,
        emit_to_ml_platform=True,
    ),
    "eval/sampler/compaction_reclaimed_tokens_mean": _spec(
        "eval/sampler/compaction_reclaimed_tokens_mean",
        AggregationStrategy.MEAN,
        _DOC_SAMPLER_COMPACTION_RECLAIMED_TOKENS_MEAN,
        emit_to_ml_platform=True,
    ),
    "eval/sampler/prompt_too_long_frac": _spec(
        "eval/sampler/prompt_too_long_frac",
        AggregationStrategy.RATIO,
        _DOC_SAMPLER_PROMPT_TOO_LONG_FRAC,
        emit_to_ml_platform=True,
    ),
    "eval/pipeline/sample_time_avg": _spec(
        "eval/pipeline/sample_time_avg",
        AggregationStrategy.MEAN,
        _DOC_PIPELINE_SAMPLE_TIME_AVG,
        emit_to_ml_platform=True,
    ),
    "eval/pipeline/env_time_avg": _spec(
        "eval/pipeline/env_time_avg",
        AggregationStrategy.MEAN,
        _DOC_PIPELINE_ENV_TIME_AVG,
        emit_to_ml_platform=True,
    ),
    "eval/pipeline/sample_time_p95": _spec(
        "eval/pipeline/sample_time_p95",
        AggregationStrategy.P95,
        _DOC_PIPELINE_SAMPLE_TIME_P95,
        emit_to_ml_platform=True,
    ),
    "eval/pipeline/env_time_p95": _spec(
        "eval/pipeline/env_time_p95",
        AggregationStrategy.P95,
        _DOC_PIPELINE_ENV_TIME_P95,
        emit_to_ml_platform=True,
    ),
    "eval/pipeline/sample_time_fraction": _spec(
        "eval/pipeline/sample_time_fraction",
        AggregationStrategy.RATIO_OF_SUMS,
        _DOC_PIPELINE_SAMPLE_TIME_FRACTION,
        emit_to_ml_platform=True,
    ),
    "eval/pipeline/rollout_wall_time_avg": _spec(
        "eval/pipeline/rollout_wall_time_avg",
        AggregationStrategy.MEAN,
        _DOC_PIPELINE_ROLLOUT_WALL_TIME_AVG,
        emit_to_ml_platform=True,
    ),
    "eval/pipeline/rollout_wall_time_p95": _spec(
        "eval/pipeline/rollout_wall_time_p95",
        AggregationStrategy.P95,
        _DOC_PIPELINE_ROLLOUT_WALL_TIME_P95,
        emit_to_ml_platform=True,
    ),
}

# Snapshot of framework-declared spec keys, captured at module load
# BEFORE any recipe runs its ``METRIC_SPECS.update({...})`` side
# effect. Used by a boundary test to assert the framework literal
# carries zero ``recipe/*`` keys. Runtime mutations don't affect this.
_FRAMEWORK_SPEC_KEYS: frozenset[str] = frozenset(METRIC_SPECS)


_EVAL_PREFIX = "eval/"
_CKPT_PREFIX = "eval_ckpt/"


def spec_for(name: str) -> MetricSpec:
    """Return the ``MetricSpec`` for a concrete metric ``name``.

    Exact-match keys win first. If none match, try dynamic-suffix families:
    a registered ``"<prefix>/*"`` entry matches any ``name`` that starts with
    ``<prefix>/``.

    ``eval_ckpt/*`` (the inline checkpoint-endpoint sweep) falls back to its
    ``eval/*`` twin: the sweep reuses the SAME ``EvalPipeline`` and emits the
    same keys, only under a different prefix. The fallback is resolved at
    lookup time rather than by pre-deriving twins into ``METRIC_SPECS``, so it
    also covers recipe specs — recipes register their ``eval/*`` keys at recipe
    import, long after this module is loaded.

    Raises ``KeyError`` if nothing matches — an unregistered metric is a
    programming error (add a spec), not a silent fallthrough.
    """
    spec = _lookup(name)
    if spec is not None:
        return spec
    if name.startswith(_CKPT_PREFIX):
        twin = _lookup(_EVAL_PREFIX + name[len(_CKPT_PREFIX) :])
        if twin is not None:
            return dataclasses.replace(
                twin, name=twin.name.replace(_EVAL_PREFIX, _CKPT_PREFIX, 1)
            )
    raise KeyError(f"no MetricSpec registered for metric {name!r}")


def _lookup(name: str) -> MetricSpec | None:
    """Exact then dynamic-suffix-family lookup; ``None`` if unregistered."""
    exact = METRIC_SPECS.get(name)
    if exact is not None:
        return exact
    for key, spec in METRIC_SPECS.items():
        if key.endswith("/*"):
            prefix = key[:-1]  # strip the trailing "*", keep the "/"
            if name.startswith(prefix):
                return spec
    return None


def _require(values: list[MetricValue], field: str) -> list:  # noqa: ANN201 -- heterogenous field type
    """Extract ``field`` from each MetricValue; raise if unset.

    Return type is bare ``list`` because ``field`` picks between
    MetricValue slots of different types; reducers know which dtype to
    expect from their strategy.
    """
    out = []
    for v in values:
        got = getattr(v, field)
        if got is None:
            raise ValueError(
                f"reducer expected MetricValue.{field} to be set, got {v!r}"
            )
        out.append(got)
    return out


def _reduce_sum(values: list[MetricValue]) -> float:
    return float(sum(_require(values, "value")))


def _reduce_mean(values: list[MetricValue]) -> float:
    vs = _require(values, "value")
    return float(sum(vs) / len(vs)) if vs else 0.0


def _reduce_max(values: list[MetricValue]) -> float:
    vs = _require(values, "value")
    return float(max(vs)) if vs else 0.0


def _reduce_weighted_mean(values: list[MetricValue]) -> float:
    pairs = _require(values, "weighted_pair")
    total_w = sum(w for _, w in pairs)
    if total_w == 0:
        return 0.0
    return float(sum(v * w for v, w in pairs) / total_w)


def _reduce_percentile(values: list[MetricValue], q: float) -> float:
    pooled: list[float] = []
    for samples in _require(values, "samples"):
        pooled.extend(samples)
    if not pooled:
        return 0.0
    return float(np.percentile(pooled, q))


def _reduce_ratio(values: list[MetricValue]) -> float:
    pairs = _require(values, "count_total")
    total = sum(t for _, t in pairs)
    if total == 0:
        return 0.0
    return float(sum(c for c, _ in pairs) / total)


def _reduce_ratio_of_sums(values: list[MetricValue]) -> float:
    pairs = _require(values, "part_total")
    total = sum(t for _, t in pairs)
    if total == 0:
        return 0.0
    return float(sum(p for p, _ in pairs) / total)


def _reduce_passthrough(values: list[MetricValue]) -> float:
    vs = _require(values, "value")
    if len(vs) != 1:
        raise ValueError(
            f"passthrough strategy expects exactly one value, got {len(vs)}"
        )
    return float(vs[0])


_REDUCERS = {
    AggregationStrategy.SUM: _reduce_sum,
    AggregationStrategy.MEAN: _reduce_mean,
    AggregationStrategy.MAX: _reduce_max,
    AggregationStrategy.WEIGHTED_MEAN: _reduce_weighted_mean,
    AggregationStrategy.P50: lambda vs: _reduce_percentile(vs, 50.0),
    AggregationStrategy.P95: lambda vs: _reduce_percentile(vs, 95.0),
    AggregationStrategy.RATIO: _reduce_ratio,
    AggregationStrategy.RATIO_OF_SUMS: _reduce_ratio_of_sums,
    AggregationStrategy.PASSTHROUGH: _reduce_passthrough,
}


def reduce_values(strategy: AggregationStrategy, values: list[MetricValue]) -> float:
    """Reduce a metric's per-group ``MetricValue``s to a final batch scalar,
    using the reducer for ``strategy``. Empty input yields 0.0 (except
    passthrough, which requires exactly one value)."""
    if not values and strategy is not AggregationStrategy.PASSTHROUGH:
        return 0.0
    return _REDUCERS[strategy](values)


# Strategies whose reducers read the ``.value`` scalar slot. Strategies
# NOT in this set read specialized slots and would ValueError on a
# bare-scalar wrap; ``aggregate_by_spec`` degrades those to MEAN with
# a warning.
_SCALAR_STRATEGIES: frozenset[AggregationStrategy] = frozenset(
    {
        AggregationStrategy.SUM,
        AggregationStrategy.MEAN,
        AggregationStrategy.MAX,
        AggregationStrategy.PASSTHROUGH,
    }
)


def _wrap_bare_scalar(v: MetricValue | float | int, key: str) -> MetricValue | None:
    """Wrap a bare int/float as ``MetricValue.of_scalar``; pass
    ``MetricValue``s through unchanged. Returns ``None`` for bare NaN
    to signal "drop this contribution" — mirrors the wrap step in
    ``Agent._compute_group_metrics``, and lets recipe aggregators
    that emit NaN from partial-data calculations keep the batch alive
    (``MetricValue.__post_init__`` itself rejects NaN, so we can't
    just wrap-and-store). Bare-NaN drops warn once per key.
    """
    if isinstance(v, MetricValue):
        return v
    f = float(v)
    if math.isnan(f):
        if key not in _NAN_DROP_WARNED:
            _NAN_DROP_WARNED.add(key)
            logger.warning(
                "dropping NaN bare-float contribution for metric %r; the "
                "value will not participate in this batch's aggregation. "
                "This warning fires once per key. If the NaN is expected "
                "(partial-data calculation), consider omitting the key "
                "upstream instead.",
                key,
            )
        return None
    return MetricValue.of_scalar(f)


def _ratio_family_prefixes() -> list[str]:
    """Prefixes (with trailing ``/``) of dynamic-suffix RATIO families, e.g.
    ``sampler/finish_reason/``. Derived from the registry so new RATIO
    ``<prefix>/*`` families get shared-denominator handling automatically."""
    out: list[str] = []
    for key, spec in METRIC_SPECS.items():
        if key.endswith("/*") and spec.strategy is AggregationStrategy.RATIO:
            out.append(key[:-1])  # strip trailing "*", keep the "/"
    return out


def _shared_ratio_family_denominators(
    per_group_metrics: Sequence[Mapping[str, "MetricValue | float | int"]],
) -> dict[str, float]:
    """Global denominator per dynamic-suffix RATIO family.

    Within one group, every suffix of a family shares the same
    ``group_total`` (the ``total`` of its ``of_ratio(count, total)`` pair), so
    the family's contribution to the global denominator from that group is that
    single ``group_total`` counted ONCE. Summing across groups gives a shared
    denominator identical for every suffix -- including suffixes a group never
    emitted -- which is what makes the family's ratios sum to 1.0.
    """
    prefixes = _ratio_family_prefixes()
    if not prefixes:
        return {}
    denoms: dict[str, float] = {}
    for group in per_group_metrics:
        # Per group, capture each family's single group_total once.
        seen_this_group: dict[str, float] = {}
        for key, raw in group.items():
            if not isinstance(raw, MetricValue) or raw.count_total is None:
                continue
            for pfx in prefixes:
                if key.startswith(pfx):
                    # All suffixes in this group share one total; record it once.
                    seen_this_group[pfx] = float(raw.count_total[1])
                    break
        for pfx, group_total in seen_this_group.items():
            denoms[pfx] = denoms.get(pfx, 0.0) + group_total
    return denoms


def _lookup_shared_denom(key: str, family_denoms: Mapping[str, float]) -> float | None:
    """Shared denominator for a concrete family key, or None if ``key`` is not
    part of any shared-denominator RATIO family."""
    for pfx, denom in family_denoms.items():
        if key.startswith(pfx):
            return denom
    return None


def aggregate_by_spec(
    per_group_metrics: Sequence[Mapping[str, MetricValue | float | int]],
) -> dict[str, float]:
    """Batch-stage reduction: combine per-group ``MetricValue`` dicts
    into one final ``dict[str, float]``, each key reduced by its
    registered strategy. Keys present in only some groups reduce over
    just the groups that emitted them.

    Accepts ``MetricValue | float | int``: bare numerics are wrapped
    as ``MetricValue.of_scalar`` at entry.

    Three fallbacks keep the reducer safe against imperfect emitters
    (each warns once per key):

    1. **Unregistered key**: fall back to MEAN.
    2. **Shape mismatch** (registered non-scalar strategy meets a
       bare-scalar wrap): fall back to MEAN instead of crashing.
    3. **Bare NaN**: dropped from the contribution list before reducing.

    Non-scalar carriers for an unregistered key still fail hard: the
    emitter clearly meant a specialized strategy.
    """
    by_key: dict[str, list[MetricValue]] = {}
    for group in per_group_metrics:
        for key, raw in group.items():
            wrapped = _wrap_bare_scalar(raw, key)
            if wrapped is None:  # bare NaN dropped
                continue
            by_key.setdefault(key, []).append(wrapped)

    # Shared-denominator RATIO families (dynamic-suffix, e.g.
    # ``sampler/finish_reason/*``): a per-group aggregator emits an
    # ``of_ratio(count, group_total)`` only for the suffixes it observed in
    # that group, so the plain per-key reducer (``sum(count)/sum(total)``)
    # would drop a group's ``group_total`` from the denominator of any suffix
    # that group didn't emit -- inflating sparse suffixes so the family's
    # ratios sum to > 1.0. Fix here, where group boundaries are still visible:
    # compute ONE global denominator per family = sum of each group's family
    # total (all suffixes in a group share the same ``group_total``), then
    # divide each suffix's summed count by that shared denominator so the
    # family's ratios sum to exactly 1.0.
    family_shared_denoms = _shared_ratio_family_denominators(per_group_metrics)

    result: dict[str, float] = {}
    for key, values in by_key.items():
        try:
            strategy = spec_for(key).strategy
        except KeyError:
            if key not in _UNKNOWN_KEYS_WARNED:
                _UNKNOWN_KEYS_WARNED.add(key)
                logger.warning(
                    "no MetricSpec registered for metric %r; falling back to "
                    "MEAN across the batch. Register a spec in METRIC_SPECS "
                    "if this key needs a different aggregation strategy.",
                    key,
                )
            strategy = AggregationStrategy.MEAN
        # Shape-mismatch fallback: registered non-scalar strategy but
        # every contribution is a bare-scalar wrap. Degrade to MEAN
        # instead of crashing.
        if strategy not in _SCALAR_STRATEGIES and all(
            v.value is not None for v in values
        ):
            if key not in _SHAPE_MISMATCH_WARNED:
                _SHAPE_MISMATCH_WARNED.add(key)
                logger.warning(
                    "metric %r registered with strategy %s but the emitter "
                    "produced only bare scalars (no %s-shaped MetricValue); "
                    "falling back to MEAN across the batch. Migrate the "
                    "emitter to produce the strategy-correct MetricValue "
                    "shape (e.g. MetricValue.of_weighted / of_ratio / "
                    "of_samples / of_part_total).",
                    key,
                    strategy,
                    strategy,
                )
            strategy = AggregationStrategy.MEAN
        # Shared-denominator override for dynamic-suffix RATIO families: use the
        # family's global denominator (sum of every group's total) rather than
        # only the groups that emitted THIS suffix, so the family sums to 1.0.
        if strategy is AggregationStrategy.RATIO:
            shared = _lookup_shared_denom(key, family_shared_denoms)
            if shared is not None:
                denom = shared
                numer = sum(c for c, _ in _require(values, "count_total"))
                result[key] = float(numer / denom) if denom else 0.0
                continue
        result[key] = reduce_values(strategy, values)
    return result


def merge_group_metrics(
    aggregator_outputs: list[dict[str, MetricValue]],
) -> dict[str, MetricValue]:
    """Merge the ``MetricValue`` dicts from several aggregators for one
    group. Two aggregators emitting the same key for the same group
    is a same-stage collision and raises.

    Runs inside ``_compute_group_metrics``; an ``Agent`` subclass that
    builds a flat dict in ``group_rollout`` without going through that
    seam bypasses this guard.
    """
    merged: dict[str, MetricValue] = {}
    for output in aggregator_outputs:
        for key, mv in output.items():
            if key in merged:
                raise ValueError(
                    f"same-stage metric-key collision on {key!r}: two "
                    f"aggregators emitted it for the same group"
                )
            merged[key] = mv
    return merged
