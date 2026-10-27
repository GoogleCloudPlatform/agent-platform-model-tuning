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

"""Agent ABC + shared adapter/callback protocols."""

from __future__ import annotations

import abc
import asyncio
import contextlib
import dataclasses
import logging
import time
from collections.abc import Awaitable, Sequence
from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterator,
    Protocol,
    runtime_checkable,
)

from google.cloud.aiplatform_v1.types import GenerateContentRequest

from core import atif_extras, candidate_id_utils, protocols
from core.atif_trajectory import (
    ScoredTrajectory,
    Trajectory,
    TrajectoryGroup,
    build_leading_steps,
)
from core.metric_aggregators import (
    DEFAULT_METRIC_AGGREGATORS,
    MetricAggregator,
)
from core.metric_spec import MetricValue, merge_group_metrics
from core.sample_result import SampleResult
from core.task_data import TaskData
from utils.raw_capture import CaptureContext
from utils.rollout_log import bind_rollout_log

logger = logging.getLogger(__name__)

_ATIF_SCHEMA_VERSION = "ATIF-v1.7"


# (prompt) -> SampleResult | None. Pre-bound sample closure passed via RolloutContext.
SampleFn = Callable[..., Awaitable[SampleResult | None]]


@dataclasses.dataclass(frozen=True)
class RolloutContext:
    """Per-rollout context passed to ``_rollout_internal``.

    Subclasses call ``await ctx.sample_fn(prompt)`` for LLM output. ``traj`` is
    the live Trajectory built at rollout START (leading step + identity
    stamped); ``_rollout_internal`` mutates ``traj.steps`` in place. ``env``
    is built ONCE by the base template and reused for leading-step inspection
    + the loop. Both may be ``None`` on bare-``rollout()`` test paths;
    subclasses fall back to constructing their own.
    """

    sample_fn: SampleFn
    rollout_idx: int
    session_id: str
    traj: Trajectory
    thinking_level: str | None = None
    max_generation_tokens: int | None = None
    phase: str = "train"
    dispatch_index: int | None = None
    env: protocols.Env | None = None


@runtime_checkable
class TrajectoryCallback(Protocol):
    """Observer hooks fired by an Agent during rollout.

    All methods are ``async`` and safe to await concurrently. Exceptions from
    any hook are logged and swallowed by the caller. ``traj`` (when passed) is
    the same live reference across started/step/produced/dropped hooks;
    observers key on ``traj.session_id`` / ``traj.extra`` for identity. The
    agent loop keeps mutating ``traj.steps`` while these async hooks run, so
    snapshotting must use ``traj.model_copy(deep=True)``.
    """

    async def on_job_started(self, config: Any) -> None:
        """Once per job, before any rollouts."""
        ...

    async def on_trajectory_started(
        self,
        task_data: Any,
        traj: Trajectory | None = None,
    ) -> None:
        """Fired before a rollout begins; ``traj`` has leading step + identity stamped."""
        ...

    async def on_trajectory_step_completed(
        self,
        task_data: Any,
        traj: Trajectory | None = None,
    ) -> None:
        """Fired after each agent step in a multi-step rollout."""
        ...

    async def on_trajectory_produced(
        self,
        scored: ScoredTrajectory,
        task_data: Any,
        traj: Trajectory | None = None,
    ) -> None:
        """Fired after each scored trajectory is built (``traj is scored.trajectory``)."""
        ...

    async def on_trajectory_dropped(
        self,
        task_data: Any,
        traj: Trajectory | None = None,
        reason: str = "",
    ) -> None:
        """Fired when a trajectory group is rejected by the pipeline filter."""
        ...

    async def on_batch_assembled(
        self,
        step: int,
        trajectory_ids: list[str],
        mean_reward: float | None = None,
    ) -> None:
        """Fired when a batch of trajectories is selected for a trainStep."""
        ...

    async def on_job_completed(self, aggregate_stats: dict[str, Any]) -> None:
        """Once per job, from the entry-point's shutdown path."""
        ...

    async def capture_ctx_for(
        self,
        *,
        task_id: str,
        rollout_idx: int,
        session_id: str,
    ) -> CaptureContext | None:
        """Return a per-rollout CaptureContext (or None if not managed).

        REQUIRED on the Protocol \u2014 a missing implementation raises
        AttributeError at rollout time rather than silently losing raw-LLM
        capture. Callbacks that own a per-rollout output tree return a
        CaptureContext whose ``trial_dir`` matches that tree.
        """
        ...


@contextlib.contextmanager
def _maybe_bind_rollout_log(path: Path | None) -> Iterator[None]:
    """``bind_rollout_log`` when a path is available, else a no-op."""
    if path is None:
        yield
        return
    with bind_rollout_log(path):
        yield


def _enqueue_rollout_log(capture_ctx: Any, path: Path | None) -> None:
    """Hand the finished rollout log to the uploader. Never raises."""
    if capture_ctx is None or path is None or capture_ctx.on_written is None:
        return
    if not path.exists() or path.stat().st_size == 0:
        return  # nothing was logged for this rollout
    try:
        capture_ctx.on_written(path)
    except Exception as exc:  # noqa: BLE001 - upload is best-effort
        logger.warning("could not enqueue rollout log %s: %s", path, exc)


class Agent(abc.ABC):
    """Orchestrates generation and env interaction to collect trajectories.

    ``rollout()`` / ``group_rollout()`` are concrete templates; subclasses
    implement ``_rollout_internal(task_data, ctx)`` only. ``task_data.task_id``
    must be populated at dataset-loading time (via ``ensure_task_id`` in the
    recipe's ``parse_*_item``); empty task_id raises ``ValueError``.
    """

    def __init__(
        self,
        sampler: protocols.Sampler | None,
        env_factory: Callable[[Any], protocols.Env],
        *,
        recipe_name: str = "",
        tuning_job_id: str = "",
        agent_name: str = "agent",
        agent_version: str = "1.0.0",
        agent_rollout_timeout_seconds: float | None = None,
        metric_aggregators: list[MetricAggregator] | None = None,
        trajectory_callback: TrajectoryCallback | None = None,
    ):
        self._sampler = sampler  # private — subclasses use ctx.sample_fn
        self.env_factory = env_factory
        self.recipe_name = recipe_name
        self.tuning_job_id = tuning_job_id
        self.agent_name = agent_name
        self.agent_version = agent_version
        # Base does NOT enforce timeout; subclasses apply it inside
        # _rollout_internal / _group_rollout_internal.
        self.agent_rollout_timeout_seconds = agent_rollout_timeout_seconds
        self.metric_aggregators = (
            DEFAULT_METRIC_AGGREGATORS
            if metric_aggregators is None
            else metric_aggregators
        )
        self._trajectory_callback = trajectory_callback

    @property
    def model_name(self) -> str:
        """``sampler.model_name`` if available, else ``"unknown"``."""
        candidate = getattr(self._sampler, "model_name", None)
        if isinstance(candidate, str) and candidate:
            return candidate
        return "unknown"

    def _atif_agent(self, generation_params: dict[str, Any] | None = None) -> Any:
        """Build ATIF ``Agent`` metadata at rollout time (picks up model swaps)."""
        from core.atif_trajectory import Agent as AtifAgent

        return AtifAgent(
            name=self.agent_name,
            version=self.agent_version,
            model_name=self.model_name,
            extra=generation_params or None,
        )

    @staticmethod
    def _generation_params(
        prompt: GenerateContentRequest,
        thinking_level: str | None,
        max_generation_tokens: int | None,
    ) -> dict[str, Any]:
        """The rollout's generation knobs, recorded in ``Trajectory.agent.extra``."""
        params: dict[str, Any] = {
            "thinking_level": thinking_level,
            "max_generation_tokens": max_generation_tokens,
        }
        thinking = prompt.generation_config.thinking_config
        if "include_thoughts" in thinking:
            params["include_thoughts"] = thinking.include_thoughts
        return {k: v for k, v in params.items() if v is not None}

    def _make_session_id(
        self, task_data: TaskData, rollout_idx: int, step: int | None = None
    ) -> str:
        """Mint the session_id for one rollout.

        Call this EXACTLY ONCE per rollout: the counter inside
        ``build_session_id`` advances on every call, and the returned id
        also names the rollout's artifact directory -- two calls would
        route the trajectory and its raw-capture ledger to different
        dirs.

        ``step`` is set by eval sweeps only (see ``core.session``).
        """
        from core.session import build_session_id

        return build_session_id(
            recipe=self.recipe_name,
            tuning_job_id=self.tuning_job_id,
            task_data=task_data,
            rollout_idx=rollout_idx,
            step=step,
        )

    def _new_rollout_trajectory(
        self,
        *,
        task_data: TaskData,
        session_id: str,
        rollout_idx: int,
        phase: str,
        dispatch_index: int | None,
        leading_steps: list[Any],
        generation_params: dict[str, Any] | None = None,
    ) -> Trajectory:
        """Build a live Trajectory (leading steps + identity) for one rollout."""
        if not leading_steps:
            raise ValueError(
                "_new_rollout_trajectory requires >= 1 leading step "
                "(Trajectory.steps has min_length=1)."
            )
        traj = Trajectory(
            schema_version=_ATIF_SCHEMA_VERSION,
            session_id=session_id,
            agent=self._atif_agent(generation_params),
            steps=list(leading_steps),
        )
        atif_extras.set_rollout_identity(
            traj,
            dispatch_index=dispatch_index,
            phase=phase,
            rollout_idx=rollout_idx,
            task_id=task_data.task_id,
        )
        return traj

    def _build_rollout_env(self, task_data: TaskData) -> protocols.Env:
        """Build the per-rollout Env once. Logs and re-raises on failure."""
        try:
            return self.env_factory(task_data)
        except Exception as e:
            logger.error("Agent._build_rollout_env: env_factory raised %s", e)
            raise

    @abc.abstractmethod
    async def _rollout_internal(
        self,
        task_data: TaskData,
        ctx: RolloutContext,
    ) -> ScoredTrajectory:
        """Recipe-specific rollout logic. Timeout handling is the subclass's job."""
        ...

    def _compute_group_metrics(
        self, scored_trajectories: list[ScoredTrajectory]
    ) -> dict[str, MetricValue]:
        """Merge outputs from ``self.metric_aggregators`` (dupe keys \u2192 raise)."""
        wrapped_outputs: list[dict[str, MetricValue]] = []
        for aggregator in self.metric_aggregators:
            raw = aggregator(scored_trajectories)
            wrapped: dict[str, MetricValue] = {}
            for key, val in raw.items():
                wrapped[str(key)] = (
                    val
                    if isinstance(val, MetricValue)
                    else MetricValue.of_scalar(float(val))
                )
            wrapped_outputs.append(wrapped)
        return merge_group_metrics(wrapped_outputs)

    async def _notify_trajectory_started(
        self,
        task_data: Any,
        traj: Trajectory,
    ) -> None:
        """Notify callback: rollout about to begin."""
        if self._trajectory_callback is None:
            return
        try:
            await self._trajectory_callback.on_trajectory_started(task_data, traj)
        except Exception as exc:  # noqa: BLE001 - callback isolation
            logger.warning(
                "trajectory_callback.on_trajectory_started raised (swallowed): %s",
                exc,
            )

    async def _notify_trajectory_step(
        self,
        task_data: Any,
        traj: Trajectory,
    ) -> None:
        """Notify callback: step completed within a rollout."""
        if self._trajectory_callback is None:
            return
        try:
            await self._trajectory_callback.on_trajectory_step_completed(
                task_data, traj
            )
        except Exception as exc:  # noqa: BLE001 - callback isolation
            logger.warning(
                "trajectory_callback.on_trajectory_step_completed raised (swallowed): %s",
                exc,
            )

    async def _publish_trajectory(
        self,
        scored: ScoredTrajectory,
        task_data: Any,
    ) -> None:
        """Fire on_trajectory_produced; swallow exceptions (callback isolation)."""
        if self._trajectory_callback is None:
            return
        try:
            await self._trajectory_callback.on_trajectory_produced(
                scored,
                task_data,
                scored.trajectory,
            )
        except Exception as exc:  # noqa: BLE001 - callback isolation
            logger.warning(
                "trajectory_callback.on_trajectory_produced raised (swallowed): %s",
                exc,
                exc_info=True,
            )

    async def _resolve_capture_ctx(
        self,
        *,
        task_id: str,
        rollout_idx: int,
        session_id: str,
    ) -> CaptureContext | None:
        """Resolve per-rollout CaptureContext from the callback (None if no callback)."""
        cb = self._trajectory_callback
        if cb is None:
            return None
        return await cb.capture_ctx_for(
            task_id=task_id,
            rollout_idx=rollout_idx,
            session_id=session_id,
        )

    # ---- Template methods -----------------------------------------

    async def rollout(
        self,
        task_data: TaskData,
        thinking_level: str | None = None,
        max_generation_tokens: int | None = None,
        *,
        rollout_idx: int = 0,
        phase: str = "train",
        dispatch_index: int | None = None,
        step: int | None = None,
        session_id: str | None = None,
    ) -> ScoredTrajectory | None:
        """Template: validate → build ctx (auto-build traj/env if missing) → delegate.

        ``step`` (eval sweeps only) is folded into the session_id, which
        is also this rollout's artifact directory name.

        ``session_id`` lets a caller (``_run_one_rollout``, which mints one
        early to place ``agent/agent.log``) thread its already-minted id in so
        the log and the trajectory share ONE trial dir. When ``None`` (bare
        ``rollout()`` callers), mint here. ``build_session_id`` advances a
        per-call counter, so minting twice for one rollout would produce two
        trial dirs — an orphan ``agent.log``-only stub plus the real one.
        """
        if not task_data.task_id:
            raise ValueError(
                f"task_data.task_id must be populated at dataset-loading "
                f"time (via ensure_task_id inside parse_*_item); got "
                f"type={type(task_data).__name__}, task_id={task_data.task_id!r}."
            )
        if session_id is None:
            session_id = self._make_session_id(task_data, rollout_idx, step)
        env = self._build_rollout_env(task_data)
        raw_prompt = env.get_prompt()
        traj = self._new_rollout_trajectory(
            task_data=task_data,
            session_id=session_id,
            rollout_idx=rollout_idx,
            phase=phase,
            dispatch_index=dispatch_index,
            leading_steps=build_leading_steps(raw_prompt),
            generation_params=self._generation_params(
                raw_prompt, thinking_level, max_generation_tokens
            ),
        )
        await self._notify_trajectory_started(task_data, traj)
        capture_ctx = await self._resolve_capture_ctx(
            task_id=task_data.task_id,
            rollout_idx=rollout_idx,
            session_id=session_id,
        )

        async def _sample(prompt: Any) -> SampleResult | None:
            return await self._sampler.sample(
                prompt,
                thinking_level=thinking_level,
                max_generation_tokens=max_generation_tokens,
                capture_ctx=capture_ctx,
            )

        ctx = RolloutContext(
            sample_fn=_sample,
            rollout_idx=rollout_idx,
            session_id=session_id,
            thinking_level=thinking_level,
            max_generation_tokens=max_generation_tokens,
            phase=phase,
            dispatch_index=dispatch_index,
            traj=traj,
            env=env,
        )
        return await self._rollout_internal(task_data, ctx)

    async def group_rollout(
        self,
        task_data: TaskData,
        num_samples: int,
        thinking_level: str | None = None,
        max_generation_tokens: int | None = None,
        on_trajectory_done: Callable[[], None] | None = None,
        phase: str = "train",
        dispatch_index: int | None = None,
        step: int | None = None,
    ) -> tuple[TrajectoryGroup, dict[str, Any]]:
        """Template: validate -> fan out N rollouts -> publish per-rollout -> aggregate.

        ``on_trajectory_done`` fires once per trajectory at completion (success
        OR exception) for per-trajectory resource release; MUST NOT block.

        ``step`` is passed by eval sweeps so each sweep's artifacts land
        in their own directories; training leaves it ``None``.
        """
        if not task_data.task_id:
            raise ValueError(
                f"task_data.task_id must be populated at dataset-loading "
                f"time (via ensure_task_id inside parse_*_item); got "
                f"type={type(task_data).__name__}, task_id={task_data.task_id!r}."
            )

        t_group_start = time.monotonic()
        scored_list = await self._group_rollout_internal(
            task_data,
            num_samples,
            thinking_level=thinking_level,
            max_generation_tokens=max_generation_tokens,
            on_trajectory_done=on_trajectory_done,
            phase=phase,
            dispatch_index=dispatch_index,
            step=step,
        )
        # Wall-time fallback for fan-out implementations that never go
        # through ``_run_one_rollout`` (batched sampling: one call serves
        # all N samples, so the group's interval IS each sample's). A
        # per-rollout value already stamped by ``_run_one_rollout`` wins.
        group_wall_time = time.monotonic() - t_group_start
        for scored in scored_list:
            if atif_extras.get_rollout_wall_time_s(scored.trajectory) is None:
                atif_extras.set_rollout_wall_time_s(scored.trajectory, group_wall_time)

        # Per-rollout publish happens inside _run_one_rollout so
        # survivors are published even when a sibling raises.
        candidate_id_utils.warn_if_untrainable(scored_list)
        group = TrajectoryGroup(task_data=task_data, scored_trajectories=scored_list)
        return group, self._compute_group_metrics(scored_list)

    async def _run_one_rollout(
        self,
        task_data: TaskData,
        idx: int,
        *,
        thinking_level: str | None,
        max_generation_tokens: int | None,
        phase: str,
        dispatch_index: int | None,
        on_trajectory_done: Callable[[], None] | None,
        step: int | None = None,
    ) -> ScoredTrajectory:
        """Env + traj + started callback + rollout + publish; done-callback in finally.

        The session_id is minted here rather than inside ``rollout`` because
        this rollout's log file has to be bound BEFORE ``rollout`` runs, and
        the log lives in the trial dir the session_id names. The SAME id is
        then threaded into ``rollout(session_id=...)`` so the log and the
        trajectory land in one trial dir -- ``build_session_id`` advances a
        per-call counter, so letting ``rollout`` mint its own would yield a
        second id and an orphan ``agent.log``-only stub dir per rollout.

        The log hand-off is in a ``finally`` on purpose: a rollout that raised
        is precisely the one whose log is worth reading.

        ``rollout_wall_time_s`` is clocked across this whole body (env build,
        every LLM call, every tool/sandbox turn, scoring) and stamped on the
        trajectory before publish, so observers see it. A rollout that raised
        produces no trajectory and therefore contributes no sample.
        """
        t_rollout_start = time.monotonic()
        sid = self._make_session_id(task_data, idx, step)
        capture_ctx = await self._resolve_capture_ctx(
            task_id=task_data.task_id,
            rollout_idx=idx,
            session_id=sid,
        )
        log_path = (
            Path(capture_ctx.trial_dir) / "agent" / "agent.log"
            if capture_ctx is not None
            else None
        )
        try:
            with _maybe_bind_rollout_log(log_path):
                result = await self.rollout(
                    task_data,
                    thinking_level=thinking_level,
                    max_generation_tokens=max_generation_tokens,
                    rollout_idx=idx,
                    phase=phase,
                    dispatch_index=dispatch_index,
                    step=step,
                    session_id=sid,
                )
                assert result is not None
                atif_extras.set_rollout_wall_time_s(
                    result.trajectory, time.monotonic() - t_rollout_start
                )
                await self._publish_trajectory(result, task_data)
                return result
        finally:
            _enqueue_rollout_log(capture_ctx, log_path)
            if on_trajectory_done is not None:
                on_trajectory_done()

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
        """Fan out N ``_run_one_rollout`` calls. Override for batched sampling."""
        results = await asyncio.gather(
            *(
                self._run_one_rollout(
                    task_data,
                    i,
                    thinking_level=thinking_level,
                    max_generation_tokens=max_generation_tokens,
                    phase=phase,
                    dispatch_index=dispatch_index,
                    on_trajectory_done=on_trajectory_done,
                    step=step,
                )
                for i in range(num_samples)
            ),
            return_exceptions=True,
        )
        scored_list = [r for r in results if isinstance(r, ScoredTrajectory)]
        first_exc = next((r for r in results if isinstance(r, BaseException)), None)
        if first_exc is not None:
            raise first_exc
        return scored_list
