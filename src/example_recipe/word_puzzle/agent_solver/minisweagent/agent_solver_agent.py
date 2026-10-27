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

"""``AgentSolverAgent``: a ``core.agent.Agent`` adapter over the mini loop.

The word-puzzle agent-solver runs a multi-turn tool-using loop through the mini
runtime (``AsyncDefaultAgent.run()``), which is eval-only and returns a mini
transcript rather than an ATIF ``ScoredTrajectory``. This adapter makes that
loop trainable: ``rollout()`` runs one mini episode and converts its retained
transcript into an ATIF ``ScoredTrajectory`` (via ``mini_trace_to_scored``),
and ``group_rollout()`` runs N independent episodes and aggregates them the way
the framework trainer expects.

The mini loop itself is UNCHANGED; this class only wraps it in the
``core.agent.Agent`` contract. The Docker sandbox env stays entirely inside the
mini agent (the framework ``Env`` protocol is not on this agent's path), so
``env_factory`` is ``None``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable

from core.agent import Agent, RolloutContext
from core.atif_trajectory import (
    ScoredTrajectory,
    Reward,
    Step,
    Trajectory,
)
from core import atif_extras
from core.rollout_status import RolloutStatus
from core.sample_result import SampleResult
from core.task_data import TaskData

from example_recipe.word_puzzle.agent_solver.minisweagent.mini_atif_converter import (
    mini_trace_to_scored,
)
from example_recipe.word_puzzle.reward import WordPuzzleRewardFn


logger = logging.getLogger(__name__)

# A factory (task_data, rollout_idx) -> a mini agent exposing
# ``async run(task) -> exit_extra`` and a ``.messages`` transcript. Injectable
# so tests can supply a fake without Docker/network.
MiniAgentFactory = Callable[[TaskData, int], Any]


class AgentSolverAgent(Agent):
    """Trainable ``core.agent.Agent`` wrapping the mini agent-solver loop."""

    def __init__(
        self,
        sampler: Any,
        *,
        reward_fn: WordPuzzleRewardFn,
        mini_agent_factory: MiniAgentFactory,
        prompt_of: Callable[[TaskData], str] | None = None,
        agent_rollout_timeout_seconds: float | None = None,
        trajectory_callback: Any = None,
        job_id: str = "",
    ) -> None:
        # env_factory=None: the mini agent owns its own (Docker) env; the
        # framework Env seam is not used by this agent. Type: ignore because
        # ``rollout()`` is overridden below so the base template's env-build
        # path never runs; ``_build_rollout_env`` is likewise overridden to
        # raise so any accidental base-path call is loud.
        super().__init__(
            sampler,
            None,  # type: ignore[arg-type]
            recipe_name="word_puzzle_agent_solver",
            tuning_job_id=job_id,
            agent_name="word_puzzle_agent_solver",
            agent_version="1",
            agent_rollout_timeout_seconds=agent_rollout_timeout_seconds,
            trajectory_callback=trajectory_callback,
        )
        self._reward_fn = reward_fn
        self._mini_agent_factory = mini_agent_factory
        self._prompt_of = prompt_of or (lambda td: getattr(td, "question", "") or "")
        self._job_id = job_id
        # Per-rollout step counter for the training prober. Maps a rollout
        # key (e.g. "{task_id}-{rollout_idx}") to the number of LLM query()
        # calls completed so far. Populated/cleared by rollout(); read by the
        # pipeline prober via ``active_step_counts`` (duck-typed, no protocol
        # change needed). Thread-safe: all access is on the event loop.
        self.active_step_counts: dict[str, int] = {}

    def _build_rollout_env(self, task_data: TaskData) -> Any:  # pragma: no cover
        """Overridden to raise: AgentSolverAgent has no framework Env.

        The mini agent owns its own Docker env; the framework ``Env`` protocol
        is not on this agent's path. This override exists only to be loud if
        the base template's env-build path is ever reached by mistake (the
        overridden ``rollout()`` below skips it entirely)."""
        raise RuntimeError(
            "AgentSolverAgent has no framework env; the mini agent owns its "
            "Docker env. This method should never be called (rollout() is "
            "overridden)."
        )

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
        """Overridden template: build shell traj from task prompt (no env), then
        delegate to ``_rollout_internal``.

        The base ``Agent.rollout()`` builds the shell trajectory from
        ``env.get_prompt()``, but AgentSolverAgent has no framework env. We
        build a single-user-step shell directly from ``self._prompt_of``
        instead; everything else (started callback, capture ctx, sample_fn
        closure, ctx construction) mirrors the base template exactly.

        ``session_id`` is threaded in by ``_run_one_rollout`` (which mints one
        early to place ``agent/agent.log``) so the log and the trajectory share
        ONE trial dir; minting a second here would orphan an ``agent.log``-only
        stub dir. When ``None`` (bare ``rollout()`` callers), mint here.
        """
        if not task_data.task_id:
            raise ValueError(
                f"task_data.task_id must be populated at dataset-loading "
                f"time (via ensure_task_id inside parse_*_item); got "
                f"type={type(task_data).__name__}, task_id={task_data.task_id!r}."
            )
        if session_id is None:
            session_id = self._make_session_id(task_data, rollout_idx, step)
        prompt_text = self._prompt_of(task_data)
        leading_steps = [Step(step_id=1, source="user", message=prompt_text)]
        traj = self._new_rollout_trajectory(
            task_data=task_data,
            session_id=session_id,
            rollout_idx=rollout_idx,
            phase=phase,
            dispatch_index=dispatch_index,
            leading_steps=leading_steps,
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
            env=None,
        )
        return await self._rollout_internal(task_data, ctx)

    def _stamp_identity(self, scored: ScoredTrajectory, ctx: RolloutContext) -> None:
        """Copy identity from ``ctx.traj.extra`` onto the produced trajectory."""
        if ctx.traj is None or ctx.traj.extra is None:
            return
        target = scored.trajectory
        if target.extra is None:
            target.extra = {}
        for k in ("dispatch_index", "phase", "rollout_idx", "task_id"):
            if k in ctx.traj.extra:
                target.extra[k] = ctx.traj.extra[k]

    async def _rollout_internal(
        self,
        task_data: TaskData,
        ctx: RolloutContext,
    ) -> ScoredTrajectory:
        mini_agent = self._mini_agent_factory(task_data, ctx.rollout_idx)

        # Resolve per-rollout CaptureContext. AgentSolverAgent injects it
        # directly into the mini model (not via ctx.sample_fn).
        capture_ctx = await self._resolve_capture_ctx(
            task_id=str(task_data.task_id),
            rollout_idx=ctx.rollout_idx,
            session_id=ctx.session_id,
        )
        model = getattr(mini_agent, "model", None)
        if model is not None and hasattr(model, "_capture_ctx"):
            model._capture_ctx = capture_ctx

        # Wire per-rollout step counter for the training prober.
        step_key = f"{task_data.task_id}-{ctx.rollout_idx}"
        self.active_step_counts[step_key] = 0
        if model is not None and hasattr(model, "_on_query"):

            def _inc_step() -> None:
                self.active_step_counts[step_key] = (
                    self.active_step_counts.get(step_key, 0) + 1
                )

            model._on_query = _inc_step

        timeout_s = self.agent_rollout_timeout_seconds
        t_start = time.monotonic()
        try:
            try:
                if timeout_s is None:
                    exit_extra = await mini_agent.run(task=self._prompt_of(task_data))
                else:
                    exit_extra = await asyncio.wait_for(
                        mini_agent.run(task=self._prompt_of(task_data)),
                        timeout=timeout_s,
                    )
            except asyncio.TimeoutError:
                scored = await self._salvage_timeout_trajectory(
                    mini_agent,
                    task_data,
                    ctx,
                    elapsed_s=time.monotonic() - t_start,
                )
                self._stamp_identity(scored, ctx)
                return scored

            scored = await mini_trace_to_scored(
                mini_agent.messages,
                exit_extra,
                task_data,
                ctx.rollout_idx,
                reward_fn=self._reward_fn,
                job_id=self._job_id,
            )
            self._stamp_identity(scored, ctx)
            return scored
        finally:
            self.active_step_counts.pop(step_key, None)
            env = getattr(mini_agent, "env", None)
            if env is not None and hasattr(env, "cleanup"):
                try:
                    await asyncio.to_thread(env.cleanup)
                except Exception:
                    pass

    async def _salvage_timeout_trajectory(
        self,
        mini_agent: Any,
        task_data: TaskData,
        ctx: RolloutContext,
        *,
        elapsed_s: float,
    ) -> ScoredTrajectory:
        """Convert a timed-out rollout's PARTIAL transcript into a trajectory."""
        messages = list(getattr(mini_agent, "messages", None) or [])
        n_turns = sum(1 for m in messages if m.get("role") == "assistant")
        last_role = messages[-1].get("role") if messages else None
        logger.warning(
            "Rollout TIMEOUT after %.1fs (task_id=%s rollout_idx=%s): salvaging "
            "partial transcript -- %d completed agent turn(s), %d total "
            "message(s), last_role=%s. Trajectory tagged TIMEOUT + NaN reward "
            "(excluded from training).",
            elapsed_s,
            task_data.task_id,
            ctx.rollout_idx,
            n_turns,
            len(messages),
            last_role,
        )
        synthetic_exit = {"exit_status": "TIMEOUT", "submission": ""}
        try:
            scored = await mini_trace_to_scored(
                messages,
                synthetic_exit,
                task_data,
                ctx.rollout_idx,
                reward_fn=self._reward_fn,
                job_id=self._job_id,
            )
        except Exception:
            return self._finalize_timeout_trajectory(ctx.traj)

        # Force training-exclusion regardless of what the partial submission
        # would have scored: the rollout did not legitimately finish.
        atif_extras.set_rollout_status(
            scored.trajectory, RolloutStatus.SAMPLING_FAILURE_TIMEOUT
        )
        return ScoredTrajectory(
            trajectory=scored.trajectory, reward=Reward(reward=float("nan"))
        )

    def _finalize_timeout_trajectory(self, traj: Trajectory) -> ScoredTrajectory:
        """A rollout that exceeded ``agent_rollout_timeout_seconds`` -> a
        NaN-reward TIMEOUT trajectory (dropped by the variance filter)."""
        atif_extras.set_rollout_status(traj, RolloutStatus.SAMPLING_FAILURE_TIMEOUT)
        return ScoredTrajectory(trajectory=traj, reward=Reward(reward=float("nan")))
