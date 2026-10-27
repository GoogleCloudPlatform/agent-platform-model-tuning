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

"""SingleStepAgent — single-sample-per-rollout concrete Agent.

Extracted from ``core.agent`` in the action-type refactor to
reduce ``core.agent`` from ~700 to ~260 lines and give the concrete
implementation its own module. Behavior unchanged from the pre-split
version; all consumers updated in the same PR.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Generic, TypeVar, cast

from google.cloud.aiplatform_v1.types import GenerateContentRequest
from google.protobuf.json_format import MessageToDict
from core.atif_trajectory import Trajectory as AtifTrajectory
from utils.raw_capture import CaptureContext

from core import atif_extras, protocols
from core.agent import (
    _ATIF_SCHEMA_VERSION,
    Agent,
    RolloutContext,
    TrajectoryCallback,
)
from core.atif_trajectory import (
    ScoredTrajectory,
    Reward,
    StepTimings,
    TrajectoryStep as AtifTrajectoryStep,
    build_leading_steps,
)
from core.metric_aggregators import MetricAggregator
from core.rollout_status import RolloutStatus
from core.sample_result import SampleResult
from core.task_data import TaskData


T = TypeVar("T", bound=TaskData)


class SingleStepAgent(Agent, Generic[T]):
    """Single sample per rollout; one-step ScoredTrajectory out.

    Parameterized on ``T`` (a ``TaskData`` subclass) so ``env_factory`` /
    ``extract_prompt_text`` agree on the recipe-specific shape. The
    ``rollout`` / ``group_rollout`` signatures keep the base ``TaskData``
    type to match the ABC; ``T`` narrows only the construction seam, not
    the dispatch seam where the pipeline feeds in base-typed rows.
    """

    def __init__(
        self,
        sampler: protocols.Sampler,
        env_factory: Callable[[T], protocols.Env],
        *,
        recipe_name: str,
        tuning_job_id: str,
        extract_prompt_text: Callable[[T], str],
        agent_name: str = "single-step-agent",
        agent_rollout_timeout_seconds: float | None = None,
        metric_aggregators: list[MetricAggregator] | None = None,
        trajectory_callback: TrajectoryCallback | None = None,
    ):
        """Args:
        recipe_name: Recipe slug for session_id / ATIF metadata.
        tuning_job_id: Tuning job ID for session_id / ATIF metadata.
        extract_prompt_text: Extracts prompt text from task_data for
            the leading user step.
        agent_name: ATIF Agent metadata name (default
            ``"single-step-agent"``).
        agent_rollout_timeout_seconds: Per-rollout wallclock timeout in
            seconds. On timeout, affected samples get ``reward=NaN`` and
            status TIMEOUT so ``TrajectoryGroupFilter`` can drop them.
            ``None`` disables the wrapper.
        metric_aggregators: Ordered list of ``MetricAggregator`` callables
            run after each ``group_rollout``. ``None`` uses
            ``DEFAULT_METRIC_AGGREGATORS``.
        trajectory_callback: Optional per-trajectory publish hook fired
            from ``group_rollout``. Callback exceptions are swallowed.
        """
        super().__init__(
            sampler,
            env_factory,
            recipe_name=recipe_name,
            tuning_job_id=tuning_job_id,
            agent_name=agent_name,
            agent_rollout_timeout_seconds=agent_rollout_timeout_seconds,
            metric_aggregators=metric_aggregators,
            trajectory_callback=trajectory_callback,
        )
        self._extract_prompt_text = extract_prompt_text

    def _build_scored(
        self,
        *,
        agent_step: AtifTrajectoryStep | None,
        reward: float,
        status: RolloutStatus,
        is_correct: bool | None,
        step_task_stats: Any = None,
        traj: AtifTrajectory,
    ) -> ScoredTrajectory:
        """Finalize a single-step rollout: append agent step + stamp status.

        ``traj`` is the live shell built by ``_group_rollout_internal`` via
        the base ``Agent._new_rollout_trajectory`` path, so leading steps
        + identity extras are already populated.
        """
        if agent_step is not None:
            traj.steps.append(agent_step)
        atif_extras.set_rollout_status(traj, status)
        if is_correct is not None:
            atif_extras.set_is_correct(traj, is_correct)
        if step_task_stats is not None:
            if traj.extra is None:
                traj.extra = {}
            traj.extra["_task_step_stats"] = [step_task_stats]
        return ScoredTrajectory(trajectory=traj, reward=Reward(reward=reward))

    def _finalize_timeout_trajectory(
        self,
        task_data: TaskData,
        *,
        rollout_idx: int,
        traj: AtifTrajectory,
    ) -> ScoredTrajectory:
        """NaN-reward trajectory tagged TIMEOUT: append one empty agent step."""
        from core.atif_trajectory import StepStats as _StepStats

        del task_data, rollout_idx  # traj is fully populated by the caller
        sample_time = self.agent_rollout_timeout_seconds or 0.0
        step = AtifTrajectoryStep.from_wire(
            step_id=len(traj.steps) + 1,
            stats=_StepStats(status=RolloutStatus.SAMPLING_FAILURE_TIMEOUT),
            generation="",
            timings=StepTimings(sample_time_s=sample_time, env_time_s=0.0),
        )
        return self._build_scored(
            agent_step=step,
            reward=float("nan"),
            status=RolloutStatus.SAMPLING_FAILURE_TIMEOUT,
            is_correct=None,
            traj=traj,
        )

    def _build_agent_step(
        self,
        res: SampleResult,
        *,
        sample_time_s: float,
        env_time_s: float,
        step_id: int,
        status: RolloutStatus | None = None,
    ) -> AtifTrajectoryStep:
        """Normalize a SampleResult into an ATIF Step."""
        content = res.tuning_candidate.candidate.content
        if content and (content.parts or content.role):
            gen: Any = MessageToDict(
                type(content).pb(content), preserving_proto_field_name=True
            )
        else:
            gen = res.completion_text
        stats = res.to_step_stats()
        if status is not None:
            stats.status = status
        return AtifTrajectoryStep.from_wire(
            step_id=step_id,
            stats=stats,
            generation=gen,
            timings=StepTimings(sample_time_s=sample_time_s, env_time_s=env_time_s),
            raw_call_uri=res.context.raw_call_uri,
        )

    async def _scored_from_sample(
        self,
        res: SampleResult,
        env: protocols.Env,
        task_data: TaskData,
        sample_time: float = 0.0,
        *,
        rollout_idx: int,
        traj: AtifTrajectory,
    ) -> ScoredTrajectory:
        del task_data, rollout_idx  # traj is fully populated by the caller
        step_id = len(traj.steps) + 1
        # Sampling-infra failure: drop via NaN reward.
        if res.tuning_candidate.candidate_id == "":
            failed_step = self._build_agent_step(
                res,
                sample_time_s=sample_time,
                env_time_s=0.0,
                step_id=step_id,
                status=RolloutStatus.SAMPLING_FAILURE_SAMPLING_CLIENT_ERROR,
            )
            return self._build_scored(
                agent_step=failed_step,
                reward=float("nan"),
                status=RolloutStatus.SAMPLING_FAILURE_SAMPLING_CLIENT_ERROR,
                is_correct=None,
                traj=traj,
            )

        # ``content`` is always a real proto ``Content`` (never None):
        # proto submessages default to empty instances, and
        # SingleStepAgent.rollout short-circuits ``if not results: ...``
        # before reaching here. Empty-parts cases (e.g. MAX_TOKENS where
        # the whole budget went to reasoning) are handled naturally by
        # envs iterating an empty ``content.parts``; no dict-shape
        # normalization is needed.
        content = res.tuning_candidate.candidate.content
        action = protocols.Action(
            content=content,
            finish_reason=res.finish_reason,
            generation_token_count=res.tuning_candidate.generation_token_count,
            candidate_id=res.tuning_candidate.candidate_id or None,
        )

        t_env_start = time.perf_counter()
        step_result = await env.step(action, step_count=1)
        env_time = time.perf_counter() - t_env_start
        agent_step = self._build_agent_step(
            res,
            sample_time_s=sample_time,
            env_time_s=env_time,
            step_id=step_id,
        )
        return self._build_scored(
            agent_step=agent_step,
            reward=step_result.reward if step_result.reward is not None else 0.0,
            status=RolloutStatus.OK,
            is_correct=step_result.is_correct,
            step_task_stats=step_result.task_stats,
            traj=traj,
        )

    # ---- Template-method overrides -----------------------------------------

    async def _rollout_internal(
        self,
        task_data: TaskData,
        ctx: RolloutContext,
    ) -> ScoredTrajectory:
        """Not supported — use ``group_rollout``.

        All single-step rollout logic lives in ``_group_rollout_internal``
        so there is exactly one implementation to maintain; a per-task
        entry point would be a second one.
        """
        raise RuntimeError("Rollout shouldn't be called in single step agent.")

    async def _group_rollout_internal(
        self,
        task_data: TaskData,
        num_samples: int,
        thinking_level: str | None = None,
        max_generation_tokens: int | None = None,
        on_trajectory_done: Callable[[], None] | None = None,
        phase: str = "train",
        dispatch_index: int | None = None,
        step: int | None = None,
    ) -> list[ScoredTrajectory]:
        prompt = self.env_factory(task_data).get_prompt()

        # Mint ONE session_id per sample: it identifies the trajectory AND
        # names the rollout's artifact directory, so the capture context
        # (raw-LLM ledger) and the trajectory MUST be handed the same
        # value. These were minted separately before, yielding two
        # different ids per sample -- harmless while the dir name came
        # from task_id+rollout_idx, but it would orphan the ledger now
        # that the dir name IS the session_id.
        session_ids = [
            self._make_session_id(task_data, i, step) for i in range(num_samples)
        ]

        # Resolve N capture contexts for batched-fanout sampling.
        capture_ctxs: list[CaptureContext | None] = []
        for i in range(num_samples):
            ctx = await self._resolve_capture_ctx(
                task_id=task_data.task_id,
                rollout_idx=i,
                session_id=session_ids[i],
            )
            capture_ctxs.append(ctx)

        leading_steps = build_leading_steps(prompt)
        generation_params = self._generation_params(
            prompt, thinking_level, max_generation_tokens
        )
        trajs: list[AtifTrajectory] = [
            self._new_rollout_trajectory(
                task_data=task_data,
                session_id=session_ids[i],
                rollout_idx=i,
                phase=phase,
                dispatch_index=dispatch_index,
                leading_steps=leading_steps,
                generation_params=generation_params,
            )
            for i in range(num_samples)
        ]
        for i in range(num_samples):
            await self._notify_trajectory_started(task_data, trajs[i])

        t_sample_start = time.perf_counter()
        try:
            coro = self._sampler.sample_n(
                prompt,
                num_samples,
                thinking_level=thinking_level,
                max_generation_tokens=max_generation_tokens,
                capture_ctxs=capture_ctxs,
            )
            if self.agent_rollout_timeout_seconds is None:
                results = await coro
            else:
                results = await asyncio.wait_for(
                    coro,
                    timeout=self.agent_rollout_timeout_seconds,
                )
        except asyncio.TimeoutError:
            scored_list: list[ScoredTrajectory] = [
                self._finalize_timeout_trajectory(
                    task_data, rollout_idx=i, traj=trajs[i]
                )
                for i in range(num_samples)
            ]
            for scored in scored_list:
                await self._publish_trajectory(scored, task_data)
            if on_trajectory_done is not None:
                for _ in range(num_samples):
                    on_trajectory_done()
            return scored_list
        sample_time = time.perf_counter() - t_sample_start

        # Empty response: sampler returned zero candidates (infra failure).
        if not results:
            scored_list = [
                self._build_scored(
                    agent_step=None,
                    reward=0.0,
                    status=RolloutStatus.SAMPLING_FAILURE_UNHANDLED,
                    is_correct=None,
                    traj=trajs[i],
                )
                for i in range(num_samples)
            ]
            for scored in scored_list:
                await self._publish_trajectory(scored, task_data)
            if on_trajectory_done is not None:
                for _ in range(num_samples):
                    on_trajectory_done()
            return scored_list

        async def _run(idx: int, res: SampleResult) -> ScoredTrajectory:
            try:
                env = self.env_factory(task_data)
                scored = await self._scored_from_sample(
                    res,
                    env,
                    task_data,
                    sample_time=sample_time,
                    rollout_idx=idx,
                    traj=trajs[idx],
                )
                await self._publish_trajectory(scored, task_data)
                return scored
            finally:
                if on_trajectory_done is not None:
                    on_trajectory_done()

        results_or_excs = await asyncio.gather(
            *(_run(i, res) for i, res in enumerate(results)),
            return_exceptions=True,
        )
        scored_list = [r for r in results_or_excs if isinstance(r, ScoredTrajectory)]
        first_exc = next(
            (r for r in results_or_excs if isinstance(r, BaseException)), None
        )
        if first_exc is not None:
            raise first_exc
        return scored_list
