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

"""MultiStepAgent — multi-turn agent loop base class.

Generalizes the multi-turn agent pattern into a reusable
``Agent`` subclass. Sits alongside ``SingleStepAgent`` as a direct
``Agent`` subclass.

Per-turn RL phases: sample the action → check for a usable action →
execute in the env → append the turn → repeat until ``done`` or
``max_steps``. Overridable hooks let recipe subclasses customize error
triage and action extraction without reimplementing the core loop.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import time
from typing import Any, Callable, Generic, TypeVar

from google.cloud.aiplatform_v1.types import (
    Content,
    GenerateContentRequest,
    Part,
)
from google.protobuf.json_format import MessageToDict
from core.atif_trajectory import Trajectory as AtifTrajectory

from core import atif_extras, context_compaction, protocols
from core.agent import (
    Agent,
    RolloutContext,
    SampleFn,
    TrajectoryCallback,
)
from core.atif_trajectory import (
    GenerationDict,
    ScoredTrajectory,
    Reward,
    StepStats,
    StepTimings,
    TrajectoryStep,
    content_to_generation_dict,
)
from core.metric_aggregators import MetricAggregator
from core.rollout_status import RolloutStatus
from core.sample_result import SampleResult
from core.task_data import TaskData

logger = logging.getLogger(__name__)

_DEFAULT_LOWEST_REWARD = -1.0

T = TypeVar("T", bound=TaskData)


@dataclasses.dataclass
class _Action:
    """One sampling call's outcome. ``action is None`` iff the sample is unusable."""

    result: SampleResult | None
    action: protocols.Action | None
    time_s: float


@dataclasses.dataclass
class _TurnCtx:
    """Per-rollout constants + per-turn accumulators threaded through the loop."""

    observation: GenerateContentRequest
    traj: AtifTrajectory
    task_data: TaskData
    per_step_task_stats: list[Any] = dataclasses.field(default_factory=list)
    episode_reward: float = float("nan")
    episode_done: bool = False
    sampling_error_status: RolloutStatus | None = None
    is_correct: bool | None = None
    last_prompt_tokens: int | None = None
    last_generation_tokens: int | None = None
    compactions_used: int = 0
    total_generation_tokens: int = 0
    generation_tokens_seen: bool = False
    # The compaction step awaiting its ``compaction_post_tokens``. The
    # rebuilt context has not been sampled when the step is written, so
    # the size AFTER compaction is only measurable by the next call;
    # ``sample_action`` stamps it. Stays None if no such call lands.
    pending_compaction_step: TrajectoryStep | None = None

    def charge_generation(self, tokens: int | None) -> None:
        """Add one call's generation tokens to the rollout total.

        Charged for calls the rollout is billed for: the actions it took
        and the harness-initiated compaction summaries it needed. NOT
        charged for recovered non-STOP samples, which produce no action.
        """
        if tokens is None:
            return
        self.total_generation_tokens += int(tokens)
        self.generation_tokens_seen = True

    @property
    def rollout_generation_tokens(self) -> int | None:
        """Rollout token total, or ``None`` if no call ever reported one."""
        return self.total_generation_tokens if self.generation_tokens_seen else None

    def mark_terminal(self, *, reward: float, status: RolloutStatus) -> None:
        """Close the episode from a sampling-failure path."""
        self.episode_done = True
        self.episode_reward = reward
        self.sampling_error_status = status

    def update_from_env(self, env_response: protocols.EnvStepResult) -> None:
        """Fold one ``EnvStepResult`` into the turn's accumulators."""
        self.per_step_task_stats.append(env_response.task_stats)
        self.episode_done = env_response.done
        if env_response.reward is not None:
            self.episode_reward = env_response.reward
        if env_response.is_correct is not None:
            self.is_correct = env_response.is_correct

    def append_turn(
        self,
        result: SampleResult,
        env_response: protocols.EnvStepResult,
    ) -> None:
        """Append model turn + env feedback (proto-JSON) to ``observation``.

        All function responses for the turn go into ONE ``Content``, one
        part each, in env order. Parallel function calls are answered by
        a single tool turn whose parts line up positionally with the
        model turn's ``functionCall`` parts; splitting them across
        several ``Content`` entries breaks that pairing.
        """
        content = result.tuning_candidate.candidate.content
        if content is not None:
            self.observation.contents.append(content)
        if env_response.function_responses:
            parts = [
                Part.from_json(json.dumps(fr)) for fr in env_response.function_responses
            ]
            self.observation.contents.append(Content(role="user", parts=parts))


class MultiStepAgent(Agent, Generic[T]):
    """Multi-turn agent loop: sample → check → execute → repeat.

    Parameterized on ``T`` (a ``TaskData`` subclass) so ``env_factory``
    / ``extract_prompt_text`` agree on the recipe-specific shape.

    Hooks for recipe customization:

    * ``_triage_sampling_error(result, finish_reason)`` — classify a
      non-STOP finish reason into ``(reward, RolloutStatus)``.
    * ``_extract_action(result)`` — build an ``Action`` from a
      ``SampleResult`` (called inside ``sample_action``).

    Sample recovery is NOT a hook: it is data, so it is configuration.
    ``recoverable_finish_reasons`` names the non-STOP reasons worth a
    fresh draw against the unchanged observation, and the two retry
    budgets bound it. A retry is not a step -- see ``_run_multi_step_loop``.
    """

    def __init__(
        self,
        sampler: protocols.Sampler,
        env_factory: Callable[[T], protocols.Env],
        *,
        max_steps: int,
        recipe_name: str = "",
        tuning_job_id: str = "",
        extract_prompt_text: Callable[[T], str],
        agent_name: str = "multi-step-agent",
        agent_version: str = "1.0.0",
        agent_rollout_timeout_seconds: float | None = None,
        metric_aggregators: list[MetricAggregator] | None = None,
        trajectory_callback: TrajectoryCallback | None = None,
        lowest_reward: float = _DEFAULT_LOWEST_REWARD,
        recoverable_finish_reasons: frozenset[str] = frozenset(),
        max_sample_retries: int = 3,
        max_sample_retries_per_rollout: int | None = 8,
        max_context_tokens: int | None = None,
        compaction_trigger_ratio: float = 0.7,
        max_compactions: int = 0,
        compaction_tail_pairs: int = context_compaction.MAX_TAIL_PAIRS,
    ):
        super().__init__(
            sampler,
            env_factory,
            recipe_name=recipe_name,
            tuning_job_id=tuning_job_id,
            agent_name=agent_name,
            agent_version=agent_version,
            agent_rollout_timeout_seconds=agent_rollout_timeout_seconds,
            metric_aggregators=metric_aggregators,
            trajectory_callback=trajectory_callback,
        )
        self.max_steps = max_steps
        self._extract_prompt_text = extract_prompt_text
        self.lowest_reward = lowest_reward
        self.recoverable_finish_reasons = recoverable_finish_reasons
        self.max_sample_retries = max_sample_retries
        self.max_sample_retries_per_rollout = max_sample_retries_per_rollout
        self.max_context_tokens = max_context_tokens
        self.compaction_trigger_ratio = compaction_trigger_ratio
        self.max_compactions = max_compactions
        self.compaction_tail_pairs = compaction_tail_pairs

    # ---- Overridable hooks -------------------------------------------------

    def _triage_sampling_error(
        self,
        result: SampleResult,
        finish_reason: str,
    ) -> tuple[float, RolloutStatus]:
        """Classify a non-STOP sampler response into (reward, status)."""
        del finish_reason
        if result.context.is_prompt_too_long:
            return self.lowest_reward, RolloutStatus.SAMPLING_FAILURE_PROMPT_TOO_LONG
        if result.context.is_malformed_usage:
            # Our own inability to read the response, not the model's
            # failure, so NaN (excluded from the reward mean) rather than
            # lowest_reward -- but under a status of its own so the rate
            # is visible in trajectory_status/*.
            return (
                float("nan"),
                RolloutStatus.SAMPLING_FAILURE_SAMPLING_CLIENT_ERROR,
            )
        return float("nan"), RolloutStatus.SAMPLING_FAILURE_UNHANDLED

    def _extract_action(self, result: SampleResult) -> protocols.Action:
        """Build an ``Action`` from a ``SampleResult``.

        Default: extracts content, finish_reason, generation_token_count,
        and candidate_id from the tuning candidate.
        """
        return protocols.Action(
            content=result.tuning_candidate.candidate.content,
            finish_reason=result.finish_reason,
            generation_token_count=result.tuning_candidate.generation_token_count,
            candidate_id=result.tuning_candidate.candidate_id or None,
        )

    def _finalize_timeout_trajectory(
        self,
        traj: AtifTrajectory,
    ) -> ScoredTrajectory:
        """Build a NaN-reward trajectory tagged TIMEOUT."""
        sample_time = self.agent_rollout_timeout_seconds or 0.0
        agent_step = TrajectoryStep.from_wire(
            step_id=len(traj.steps) + 1,
            stats=StepStats(status=RolloutStatus.SAMPLING_FAILURE_TIMEOUT),
            generation="",
            timings=StepTimings(sample_time_s=sample_time, env_time_s=0.0),
        )
        traj.steps.append(agent_step)
        atif_extras.finalize_trajectory(
            traj, status=RolloutStatus.SAMPLING_FAILURE_TIMEOUT
        )
        return ScoredTrajectory(trajectory=traj, reward=Reward(reward=float("nan")))

    # ---- Per-turn RL phases -----------------------------------------------

    async def sample_action(
        self,
        sample_fn: SampleFn,
        turn_ctx: _TurnCtx,
    ) -> _Action:
        """Sample the LLM; extract an Action if finish_reason is STOP."""
        t = time.perf_counter()
        sample_res = await sample_fn(turn_ctx.observation)
        sample_time = time.perf_counter() - t
        action: protocols.Action | None = None
        if sample_res is not None and sample_res.finish_reason == "STOP":
            action = self._extract_action(sample_res)
        if sample_res is not None:
            turn_ctx.last_prompt_tokens = sample_res.context.prompt_token_count
            self._backfill_compaction_post_tokens(turn_ctx, sample_res)
            turn_ctx.last_generation_tokens = (
                sample_res.tuning_candidate.generation_token_count
            )
            # Only an actionable sample is counted: a recovered non-STOP
            # sample is resampled and took no action.
            if action is not None:
                turn_ctx.charge_generation(
                    sample_res.tuning_candidate.generation_token_count
                )
        return _Action(result=sample_res, action=action, time_s=sample_time)

    @staticmethod
    def _backfill_compaction_post_tokens(
        turn_ctx: _TurnCtx,
        sample_res: SampleResult,
    ) -> None:
        """Stamp the pending compaction step with the measured post size.

        The first call after a compaction is the first measurement of the
        rebuilt context, and it is request-scope exactly like
        ``compaction_pre_tokens`` -- so ``pre - post`` stays a like-for-like
        reclaim figure. Left absent (never 0) when no count arrives: a 0
        would read as a total reclaim.
        """
        step = turn_ctx.pending_compaction_step
        if step is None:
            return
        post = sample_res.context.prompt_token_count
        if post is None:
            return
        step.extra = dict(step.extra or {})
        step.extra["compaction_post_tokens"] = post
        turn_ctx.pending_compaction_step = None

    async def execute(
        self,
        env: protocols.Env,
        sample: _Action,
        step_idx: int,
        turn_ctx: _TurnCtx,
    ) -> protocols.EnvStepResult:
        """env.step + record ATIF step + notify observers + fold into turn_ctx."""
        assert sample.action is not None and sample.result is not None
        # Inclusive of the action being dispatched, so an env computing a
        # cost-shaped reward on a terminal step sees the full rollout.
        env.set_rollout_generation_tokens(turn_ctx.rollout_generation_tokens)
        t_env_start = time.perf_counter()
        env_response = await env.step(sample.action, step_count=step_idx + 1)
        env_time = time.perf_counter() - t_env_start
        self._record_ok_step(turn_ctx, sample, env_response, env_time)
        # Live-step notify so dashboards see progress in real time and
        # the disk writer can upload partial snapshots.
        await self._notify_trajectory_step(turn_ctx.task_data, turn_ctx.traj)
        turn_ctx.update_from_env(env_response)
        return env_response

    def _record_ok_step(
        self,
        turn_ctx: _TurnCtx,
        sample: _Action,
        env_response: protocols.EnvStepResult,
        env_time: float,
    ) -> None:
        """Append the ATIF step for a happy-path sample + env round."""
        assert sample.result is not None
        content = sample.result.tuning_candidate.candidate.content
        agent_step = TrajectoryStep.from_wire(
            step_id=len(turn_ctx.traj.steps) + 1,
            stats=sample.result.to_step_stats(),
            generation=content_to_generation_dict(content),
            tool_responses=env_response.function_responses,
            timings=StepTimings(sample_time_s=sample.time_s, env_time_s=env_time),
            raw_call_uri=sample.result.context.raw_call_uri,
        )
        turn_ctx.traj.steps.append(agent_step)

    def _can_recover_sample(
        self,
        sample: _Action,
        *,
        turn_retries: int,
        rollout_retries: int,
    ) -> bool:
        """Whether this unusable sample is worth one more draw.

        Three independent gates, all of which must hold:

        1. the finish reason is in ``recoverable_finish_reasons`` -- i.e.
           a stochastic sampler outcome a fresh draw might avoid, not a
           deterministic failure that would simply repeat;
        2. the per-turn budget (``max_sample_retries``) has room, counted
           consecutively and reset by any successful sample;
        3. the rollout-wide budget (``max_sample_retries_per_rollout``) has
           room, so pathological runs cannot cost
           ``max_steps * max_sample_retries`` sampler calls. ``None``
           disables this gate.

        ``sample.action`` is always ``None`` here (the loop only asks
        about unusable samples) and ``sample.result`` is always non-None
        (the loop rejects empty responses before asking).
        """
        if sample.result is None:
            return False
        if sample.result.finish_reason not in self.recoverable_finish_reasons:
            return False
        if turn_retries >= self.max_sample_retries:
            return False
        if (
            self.max_sample_retries_per_rollout is not None
            and rollout_retries >= self.max_sample_retries_per_rollout
        ):
            return False
        return True

    def _record_recovery_step(
        self,
        turn_ctx: _TurnCtx,
        sample: _Action,
    ) -> None:
        """Append the failed sample as a ``source="system"`` ATIF step.

        Sister method to ``_record_sampling_failure``. Used by the recovery
        path in ``_run_multi_step_loop``: we still want a full record of
        the non-STOP sample in the trajectory (so downstream analysis sees
        the recovery streak, and the raw call stays reachable), but the
        loop resamples against the unchanged observation next iteration.

        ``source="system"``, NOT ``"agent"``, and that is the whole point:
        this generation was discarded. Its output never entered the
        model's context -- the retry re-prompts the identical
        observation -- so it is not an action the rollout took. Every
        consumer that means "the actions in this trajectory" filters on
        ``source == "agent"``, most importantly the trainer
        (``vertex_msrl_rest.train_step`` builds one training example per
        agent step), so tagging it ``"agent"`` would submit a dead
        candidate for a gradient under the rollout's final advantage.
        The token-sum, timing and step-count aggregators drop it for the
        same reason.

        Only reachable when ``sample.result is not None`` (an empty
        sample cannot yield a useful generation dict to record); the loop
        enforces that before choosing the recovery path.
        """
        assert sample.result is not None, (
            "_record_recovery_step called with empty sample; empty "
            "responses are unrecoverable and must go through "
            "_record_sampling_failure."
        )
        content = sample.result.tuning_candidate.candidate.content
        step = TrajectoryStep.from_wire(
            step_id=len(turn_ctx.traj.steps) + 1,
            stats=sample.result.to_step_stats(),
            generation=content_to_generation_dict(content),
            timings=StepTimings(sample_time_s=sample.time_s, env_time_s=0.0),
            raw_call_uri=sample.result.context.raw_call_uri,
            # Via from_wire, not `step.source = ...`: a system step may not
            # carry reasoning_content, so from_wire keeps thoughts in message.
            source="system",
        )
        turn_ctx.traj.steps.append(step)

    def _record_sampling_failure(
        self,
        turn_ctx: _TurnCtx,
        sample: _Action,
    ) -> None:
        """Append the failure ATIF step + mark the episode terminal.

        Two paths, merged: ``sample.result is None`` → NaN reward /
        SAMPLING_FAILURE_UNHANDLED; else recipe ``_triage_sampling_error``
        picks (reward, status) from the non-STOP finish_reason.
        """
        step_id = len(turn_ctx.traj.steps) + 1
        timings = StepTimings(sample_time_s=sample.time_s, env_time_s=0.0)
        if sample.result is None:
            turn_ctx.traj.steps.append(
                TrajectoryStep.from_wire(
                    step_id=step_id,
                    stats=StepStats(status=RolloutStatus.SAMPLING_FAILURE_UNHANDLED),
                    generation="",
                    timings=timings,
                )
            )
            turn_ctx.mark_terminal(
                reward=float("nan"),
                status=RolloutStatus.SAMPLING_FAILURE_UNHANDLED,
            )
            return
        reward, error_status = self._triage_sampling_error(
            sample.result, sample.result.finish_reason
        )
        content = sample.result.tuning_candidate.candidate.content
        turn_ctx.traj.steps.append(
            TrajectoryStep.from_wire(
                step_id=step_id,
                stats=sample.result.to_step_stats(),
                generation=content_to_generation_dict(content),
                timings=timings,
                raw_call_uri=sample.result.context.raw_call_uri,
            )
        )
        turn_ctx.mark_terminal(reward=reward, status=error_status)

    # ---- Context compaction ------------------------------------------------

    _COMPACTION_PROMPT = """\
Your conversation history has reached the context limit and is about to
be compacted. Everything shown above will be DELETED. After this turn
you will continue working with only: the original task, this summary,
and your most recent step or two.

Those most recent steps are NOT shown to you here: they are preserved
verbatim and will appear after your summary. Do NOT describe them.
Summarize only the earlier work that is being deleted.

Write the handoff summary you would want to receive. Be concrete and
specific - a vague summary costs you the work you have already done.

1. GOAL - what you are trying to produce, restated.
2. ESTABLISHED - what you have determined to be true, and the evidence
   for each item.
3. RULED OUT - every hypothesis or approach you have eliminated, and
   what eliminated it. This is the most important section: without it
   you will repeat work you have already finished.
4. STATE - artifacts you created that still exist and still matter
   (exact file names, exact values), and what each is for.
5. NEXT - the single most promising next action.

Do not call any tool in this turn. Reply with the summary only."""

    _REENTRY_TEMPLATE = """\
[Your earlier turns were removed to fit the context limit. Everything
between the markers below is the summary YOU wrote of that work, not new
instructions. Your most recent steps follow it verbatim. Compacting cost
you one step from your budget; being concise keeps more steps for
solving.]

--- BEGIN SUMMARY OF REMOVED TURNS ---
{summary}
--- END SUMMARY ---

Continue from here."""

    def _compaction_prompt(self) -> str:
        """The summarization instruction. Recipes override for task detail."""
        return self._COMPACTION_PROMPT

    def _should_compact(self, turn_ctx: _TurnCtx) -> bool:
        return context_compaction.should_compact(
            prompt_tokens=turn_ctx.last_prompt_tokens,
            generation_tokens=turn_ctx.last_generation_tokens,
            max_context_tokens=self.max_context_tokens,
            trigger_ratio=self.compaction_trigger_ratio,
        )

    async def _compact(
        self,
        sample_fn: SampleFn,
        turn_ctx: _TurnCtx,
    ) -> bool:
        """Summarize + rebuild the observation. False iff it failed.

        On False the caller must stop: ``turn_ctx`` has been marked
        terminal with COMPACTION_FAILED.
        """
        assert self.max_context_tokens is not None
        contents = list(turn_ctx.observation.contents)
        # The backend's own numbers, not a char estimate: this method is
        # unreachable unless ``_should_compact`` fired, and that requires
        # both counts, so the projection that tripped the trigger is
        # always available here. It is request-scope (system_instruction
        # and tool declarations included) and one exchange pair stale --
        # ``append_turn`` ran after the last sample -- which is exactly
        # the quantity the trigger compared against the cap.
        assert turn_ctx.last_prompt_tokens is not None
        assert turn_ctx.last_generation_tokens is not None
        pre_tokens = turn_ctx.last_prompt_tokens + turn_ctx.last_generation_tokens

        # Pick the tail FIRST: it survives verbatim into the rebuilt
        # context, so summarizing it too would send those tokens twice
        # and spend output describing turns that reappear anyway.
        tail, pairs_retained = context_compaction.select_tail(
            contents,
            budget_tokens=context_compaction.tail_token_budget(self.max_context_tokens),
            max_pairs=self.compaction_tail_pairs,
        )

        # Summarize only what is about to be DROPPED. The live
        # observation is left untouched: instead we copy the request (a
        # deep copy in proto-plus, which carries system_instruction,
        # tools, generation_config and model along unchanged, so this
        # otherwise looks like a normal call) and swap in the reduced
        # contents.
        summarization_request = GenerateContentRequest(turn_ctx.observation)
        dropped = contents[: len(contents) - len(tail)]
        del summarization_request.contents[:]
        summarization_request.contents.extend(dropped)
        summarization_request.contents.append(
            Content(role="user", parts=[Part(text=self._compaction_prompt())])
        )
        t = time.perf_counter()
        result = await sample_fn(summarization_request)
        sample_time = time.perf_counter() - t
        if result is not None:
            turn_ctx.charge_generation(result.tuning_candidate.generation_token_count)

        summary = ""
        if result is not None and result.finish_reason == "STOP":
            content = result.tuning_candidate.candidate.content
            summary = "".join(
                p.text for p in (content.parts if content is not None else []) if p.text
            ).strip()
        if not summary:
            # Keep the diagnostics when a result exists: MAX_TOKENS vs a
            # functionCall-instead-of-text is exactly what tells us
            # whether to strip tool declarations from this call.
            # ``to_step_stats()`` does not set status, so override it.
            stats = result.to_step_stats() if result is not None else StepStats()
            stats.status = RolloutStatus.SAMPLING_FAILURE_COMPACTION_FAILED
            turn_ctx.traj.steps.append(
                TrajectoryStep.from_wire(
                    step_id=len(turn_ctx.traj.steps) + 1,
                    stats=stats,
                    generation="",
                    timings=StepTimings(sample_time_s=sample_time, env_time_s=0.0),
                    raw_call_uri=(
                        result.context.raw_call_uri if result is not None else None
                    ),
                )
            )
            turn_ctx.mark_terminal(
                reward=self.lowest_reward,
                status=RolloutStatus.SAMPLING_FAILURE_COMPACTION_FAILED,
            )
            return False

        rebuilt = [contents[0]] if contents else []
        rebuilt.append(
            Content(
                role="user",
                parts=[Part(text=self._REENTRY_TEMPLATE.format(summary=summary))],
            )
        )
        rebuilt.extend(tail)
        del turn_ctx.observation.contents[:]
        turn_ctx.observation.contents.extend(rebuilt)

        pairs_total = max(len(contents) - 1, 0) // 2
        step = TrajectoryStep.from_wire(
            step_id=len(turn_ctx.traj.steps) + 1,
            stats=result.to_step_stats(),
            generation=summary,
            timings=StepTimings(sample_time_s=sample_time, env_time_s=0.0),
            raw_call_uri=result.context.raw_call_uri,
        )
        # ``source="agent"`` (the from_wire default), deliberately.
        # Harness-INITIATED, but the model authored it and the output
        # ENTERS THE STATE: the summary is spliced into the rebuilt
        # observation via _REENTRY_TEMPLATE, so every later turn is
        # conditioned on it and a bad summary loses the rollout. That
        # makes summarizing a capability the task reward should train,
        # so the candidate must reach ``vertex_msrl_rest.train_step``
        # (which filters on ``source == "agent"``), and its tokens and
        # latency belong in the rollout's totals.
        #
        # Consumers that must NOT count it as a solving step key off the
        # ``compaction_*`` extras below, not off ``source``.
        #
        # ``compaction_post_tokens`` is deliberately NOT set here: the
        # rebuilt context has not been sampled, so nothing has measured
        # it yet. ``_backfill_compaction_post_tokens`` stamps it from the
        # next call. ``compaction_pre_tokens`` therefore stays the sole
        # always-present marker of a compaction step.
        step.extra = dict(step.extra or {})
        step.extra.update(
            {
                "compaction_pre_tokens": pre_tokens,
                "compaction_pairs_elided": max(pairs_total - pairs_retained, 0),
                "compaction_pairs_retained": pairs_retained,
            }
        )
        turn_ctx.traj.steps.append(step)
        turn_ctx.pending_compaction_step = step
        # Live-step notify so dashboards see progress in real time and
        # the disk writer can upload partial snapshots.
        await self._notify_trajectory_step(turn_ctx.task_data, turn_ctx.traj)
        turn_ctx.compactions_used += 1
        # The rebuilt observation has not been sampled yet, so the stale
        # projection must not re-trigger compaction on the next pass.
        turn_ctx.last_prompt_tokens = None
        turn_ctx.last_generation_tokens = None
        return True

    def _record_context_overflow(self, turn_ctx: _TurnCtx) -> None:
        """Terminate when the context cannot be shrunk any further."""
        turn_ctx.traj.steps.append(
            TrajectoryStep.from_wire(
                step_id=len(turn_ctx.traj.steps) + 1,
                stats=StepStats(status=RolloutStatus.SAMPLING_FAILURE_PROMPT_TOO_LONG),
                generation="",
                timings=StepTimings(sample_time_s=0.0, env_time_s=0.0),
            )
        )
        turn_ctx.mark_terminal(
            reward=self.lowest_reward,
            status=RolloutStatus.SAMPLING_FAILURE_PROMPT_TOO_LONG,
        )

    def _resolve_rollout_status(
        self,
        *,
        sampling_error_status: RolloutStatus | None,
        episode_done: bool,
        episode_reward: float,
    ) -> tuple[RolloutStatus, float]:
        """Pick terminal (RolloutStatus, reward). Max-step exit pins reward=lowest."""
        if sampling_error_status is not None:
            return sampling_error_status, episode_reward
        if not episode_done:
            logger.warning("Max steps (%d) reached without done=True", self.max_steps)
            return RolloutStatus.MAX_STEP_LIMIT, self.lowest_reward
        return RolloutStatus.OK, episode_reward

    # ---- Core rollout loop -------------------------------------------------

    async def _rollout_internal(
        self,
        task_data: TaskData,
        ctx: RolloutContext,
    ) -> ScoredTrajectory:
        """Multi-step rollout: thin timeout wrapper around the turn loop."""
        if self.agent_rollout_timeout_seconds is None:
            return await self._run_multi_step_loop(task_data, ctx)
        try:
            return await asyncio.wait_for(
                self._run_multi_step_loop(task_data, ctx),
                timeout=self.agent_rollout_timeout_seconds,
            )
        except asyncio.TimeoutError:
            return self._finalize_timeout_trajectory(traj=ctx.traj)

    async def _run_multi_step_loop(
        self,
        task_data: TaskData,
        ctx: RolloutContext,
    ) -> ScoredTrajectory:
        """The RL loop: sample \u2192 check \u2192 execute \u2192 append \u2192 repeat.

        ``step_idx`` counts BUDGETED steps, and the two ``continue``
        paths differ on whether they spend one:

        * A compaction spends a step, deliberately and visibly -- the
          re-entry prompt tells the model it just cost one, so the model
          can trade conciseness for budget.
        * A sampling retry does NOT. It is plumbing to make one step
          succeed: the observation is unchanged, no action was taken, and
          charging it would silently shrink both ``max_steps`` and the
          ``steps_remaining`` the env echoes for something the model
          neither did nor was told about. Retries are bounded by
          ``max_sample_retries`` (consecutive, per turn) and
          ``max_sample_retries_per_rollout`` (whole rollout) instead.
        """
        # ctx.env may be None on bare-rollout() test paths; fall back so
        # the loop stays self-contained.
        env = ctx.env if ctx.env is not None else self.env_factory(task_data)
        assert ctx.traj is not None, (
            "ctx.traj is None \u2014 build one via Agent._new_rollout_trajectory "
            "before calling _rollout_internal directly."
        )
        turn_ctx = _TurnCtx(
            observation=env.get_prompt(),
            traj=ctx.traj,
            task_data=task_data,
        )

        step_idx = 0
        turn_retries = 0
        rollout_retries = 0
        while step_idx < self.max_steps:
            if self._should_compact(turn_ctx):
                # Compacting on the last step pays for a summary the
                # agent never gets to use, and a budget-exhausted
                # rollout provably cannot fit -- both end it here.
                if (
                    turn_ctx.compactions_used >= self.max_compactions
                    or self.max_steps - step_idx < 2
                ):
                    self._record_context_overflow(turn_ctx)
                    break
                if not await self._compact(ctx.sample_fn, turn_ctx):
                    break
                # This iteration WAS the compaction: it consumes a step
                # from the budget, which also shifts solve_step_count and
                # the steps_remaining the model is shown.
                step_idx += 1
                continue
            sample = await self.sample_action(ctx.sample_fn, turn_ctx)
            if sample.action is None:
                # Recovery path: record the discarded sample and resample
                # the UNCHANGED observation. Only valid when there is a
                # real result to record -- an empty response is a hard
                # error, not a stochastic truncation -- and only while
                # both retry budgets hold. Deliberately does NOT advance
                # ``step_idx``: see this method's docstring.
                if sample.result is not None and self._can_recover_sample(
                    sample, turn_retries=turn_retries, rollout_retries=rollout_retries
                ):
                    self._record_recovery_step(turn_ctx, sample)
                    turn_retries += 1
                    rollout_retries += 1
                    continue
                self._record_sampling_failure(turn_ctx, sample)
                break
            turn_retries = 0
            env_response = await self.execute(env, sample, step_idx, turn_ctx)
            step_idx += 1
            if turn_ctx.episode_done:
                break
            # Only append when we're rolling on to another turn \u2014 the
            # last-turn observation would otherwise leak into disk
            # snapshots as if it were mid-conversation.
            assert sample.result is not None
            turn_ctx.append_turn(sample.result, env_response)

        status, episode_reward = self._resolve_rollout_status(
            sampling_error_status=turn_ctx.sampling_error_status,
            episode_done=turn_ctx.episode_done,
            episode_reward=turn_ctx.episode_reward,
        )
        atif_extras.finalize_trajectory(
            turn_ctx.traj,
            status=status,
            is_correct=turn_ctx.is_correct,
            per_step_stats=turn_ctx.per_step_task_stats,
        )
        return ScoredTrajectory(
            trajectory=turn_ctx.traj, reward=Reward(reward=episode_reward)
        )
