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

"""Convert a mini-agent trace into an ATIF ``ScoredTrajectory`` for training.

The agent-solver runs its multi-turn loop through the mini-agent runtime,
which retains a flat message list (``system``, ``user``, then repeated
``assistant``/``tool`` turns, ended by an ``exit`` message). That list already
carries every per-turn training id in each assistant message's ``extra``
(``candidate_id``, ``operation_id``, ``finish_reason``, token counts) -- but as
mini's own shape, not ATIF.

This module maps that message list into an ATIF ``ScoredTrajectory`` built with
the repo's own :func:`core.atif_trajectory.TrajectoryStep.from_wire`, so the
per-step ids are stamped into ``step.extra`` exactly the way the trainer reads
them. The single episode reward (from the reused ``WordPuzzleRewardFn``) is
attached at the trajectory level and shared across agent steps.

Why the ``str()`` coercion on ``candidate_id`` matters: mini stores the
candidate index as an ``int`` (commonly ``0``), and the trainer skips any agent
step where ``candidate_id`` is falsy (``if not cid``). An unconverted ``0``
would be silently dropped from training. Coercing to ``str`` ("0") keeps it.
"""

from __future__ import annotations

from typing import Any

from core.atif_trajectory import (
    FunctionCallDict,
    GenerationDict,
    ScoredTrajectory,
    PartDict,
    Reward,
    StepStats,
    StepTimings,
    TrajectoryStep,
)
from core import atif_extras
from core.rollout_status import RolloutStatus
from core.session import build_session_id
from core.task_data import TaskData
from core.atif_trajectory import Agent as AtifAgent
from core.atif_trajectory import Step, Trajectory

from example_recipe.word_puzzle.agent_solver.minisweagent.extra_keys import (
    ENV_ELAPSED_SECONDS,
)
from example_recipe.word_puzzle.reward import WordPuzzleRewardFn
from example_recipe.word_puzzle.reward_signals import WordPuzzleRewardSignals

_ATIF_SCHEMA_VERSION = "ATIF-v1.7"
_AGENT_NAME = "word_puzzle_agent_solver"


# Mini exit_status -> ATIF RolloutStatus. ``Submitted`` is a normal completion
# (env produced a reward, right or wrong). Budget exhaustions map to
# MAX_STEP_LIMIT (there is no dedicated wall-time member). Format failure maps
# to the malformed sampling-failure member; anything unrecognized (e.g. a
# ``Crash:*`` from the runner) falls through to the unhandled sentinel.
_EXIT_STATUS_TO_ROLLOUT: dict[str, RolloutStatus] = {
    "Submitted": RolloutStatus.OK,
    "LimitsExceeded": RolloutStatus.MAX_STEP_LIMIT,
    "LimitsExceededEscaped": RolloutStatus.MAX_STEP_LIMIT,
    "TimeExceeded": RolloutStatus.MAX_STEP_LIMIT,
    "RepeatedFormatError": RolloutStatus.SAMPLING_FAILURE_MALFORMED_FUNCTION_CALL,
}


def _rollout_status_for(exit_status: str) -> RolloutStatus:
    return _EXIT_STATUS_TO_ROLLOUT.get(
        exit_status, RolloutStatus.SAMPLING_FAILURE_UNHANDLED
    )


def _coerce_candidate_id(raw: Any) -> str | None:
    """Coerce a mini candidate_id to a truthy string, or None.

    mini stores the candidate index as an int (often 0). The trainer skips
    agent steps whose candidate_id is falsy, so int 0 MUST become "0" (which
    is truthy) rather than being left as 0 (falsy -> dropped). None stays None
    (the failsafe empty-result path); an empty string also normalizes to None.
    """
    if raw is None:
        return None
    s = str(raw)
    return s if s != "" else None


def _actions_to_generation(
    content: str, actions: list[dict[str, Any]]
) -> GenerationDict:
    """Build a Gemini-shaped ``generation`` dict from mini's parsed actions.

    ``from_wire`` -> ``_parse_generation_dict`` expects
    ``{"parts": [{"function_call": {"name", "args"}}, {"text": ...}]}`` and
    emits one ATIF ``ToolCall`` per ``function_call`` part. Mini stores its
    actions as ``[{"tool_name": ..., "command": ...}]`` on the assistant
    message's ``extra["actions"]`` (``content`` is usually empty for a pure
    tool-call turn), so we translate each action into a ``function_call`` part.
    Without this, the tool call the model made is lost and the ATIF step
    records only the tool RESULT with no CALL.
    """
    parts: list[PartDict] = []
    if content:
        parts.append(PartDict(text=content))
    for action in actions:
        tool_name = action.get("tool_name")
        if not tool_name:
            continue
        # Everything except tool_name is the call's arguments (mini's bash
        # tool carries a single ``command`` key; keep it generic).
        args = {k: v for k, v in action.items() if k != "tool_name"}
        parts.append(
            PartDict(function_call=FunctionCallDict(name=str(tool_name), args=args))
        )
    return GenerationDict(parts=parts)


def _agent_step_from_message(
    *,
    step_id: int,
    message: dict[str, Any],
    observation: str | None,
) -> TrajectoryStep:
    """Build one ATIF agent step from a mini ``assistant`` message."""
    extra = message.get("extra") or {}
    stats = StepStats(
        finish_reason=extra.get("finish_reason"),
        candidate_id=_coerce_candidate_id(extra.get("candidate_id")),
        operation_id=extra.get("operation_id"),
        prompt_token_count=extra.get("prompt_token_count"),
        generation_token_count=extra.get("generation_token_count"),
    )
    # ENV_ELAPSED_SECONDS is stamped by ``AsyncDefaultAgent.execute_actions``
    # around the turn's tool execution. Read defensively: a turn that ran no
    # tool never sets it, and a transcript recorded before the key existed has
    # none at all -- both correctly score 0.0 rather than raising.
    timings = StepTimings(
        sample_time_s=float(extra.get("elapsed_seconds") or 0.0),
        env_time_s=float(extra.get(ENV_ELAPSED_SECONDS) or 0.0),
    )
    # The tool (bash) output for this turn, if any, becomes the env
    # observation. mini stores it as the following ``tool`` message's content;
    # the caller passes it in via ``observation``.
    function_responses = (
        [{"functionResponse": {"response": {"output": observation}}}]
        if observation is not None
        else None
    )
    generation = _actions_to_generation(
        message.get("content", "") or "", extra.get("actions") or []
    )
    return TrajectoryStep.from_wire(
        step_id=step_id,
        generation=generation,
        stats=stats,
        timings=timings,
        tool_responses=function_responses,
        raw_call_uri=extra.get("raw_call_uri"),
    )


async def mini_trace_to_scored(
    messages: list[dict[str, Any]],
    exit_extra: dict[str, Any],
    task_data: TaskData,
    rollout_idx: int,
    reward_fn: WordPuzzleRewardFn,
    *,
    job_id: str = "",
) -> ScoredTrajectory:
    """Convert a mini message list + exit info into an ATIF ScoredTrajectory.

    Args:
        messages: mini's retained message list (``agent.messages``).
        exit_extra: the dict returned by ``AsyncDefaultAgent.run()`` -- the
            final exit message's ``extra`` (``{exit_status, submission, ...}``).
        task_data: the task; its ``reference`` is the ground-truth answer.
        rollout_idx: within-group index (0..N-1); currently informational.
        reward_fn: the reused ``WordPuzzleRewardFn``.

    Returns:
        A ``ScoredTrajectory`` whose agent steps each carry a stringified
        ``candidate_id`` and whose reward is the single episode reward.
    """
    steps: list[Step] = []
    step_id = 1
    # Track the last agent step's info so the malformed/finish-reason of the
    # FINAL turn drives the reward.
    last_finish_reason: str | None = None
    last_gen_tokens: int | None = None

    n = len(messages)
    i = 0
    while i < n:
        msg = messages[i]
        role = msg.get("role")
        if role == "system":
            steps.append(
                Step(step_id=step_id, source="system", message=msg.get("content", ""))
            )
            step_id += 1
            i += 1
        elif role == "user":
            steps.append(
                Step(step_id=step_id, source="user", message=msg.get("content", ""))
            )
            step_id += 1
            i += 1
        elif role == "assistant":
            # Peek at the following tool message (if any) for this turn's
            # observation.
            observation: str | None = None
            if i + 1 < n and messages[i + 1].get("role") == "tool":
                observation = str(messages[i + 1].get("content", ""))
            steps.append(
                _agent_step_from_message(
                    step_id=step_id, message=msg, observation=observation
                )
            )
            step_id += 1
            extra = msg.get("extra") or {}
            last_finish_reason = extra.get("finish_reason")
            last_gen_tokens = extra.get("generation_token_count")
            # Skip the tool message we already consumed.
            i += 2 if observation is not None else 1
        else:
            # ``tool`` (already consumed with its assistant), ``exit``, or any
            # other bookkeeping message: skip.
            i += 1

    # ---- reward from the final submission ----
    submission = exit_extra.get("submission", "") or ""
    reference = getattr(task_data, "reference", "") or ""
    is_correct = submission == reference and submission != ""
    signals = WordPuzzleRewardSignals(
        is_correct=is_correct,
        generation_token_count=last_gen_tokens,
        finish_reason=last_finish_reason,
        max_output_tokens=None,
    )
    episode_reward, _stats = await reward_fn(signals)

    trajectory = Trajectory(
        schema_version=_ATIF_SCHEMA_VERSION,
        session_id=build_session_id(
            recipe=_AGENT_NAME,
            tuning_job_id=job_id,
            task_data=task_data,
            rollout_idx=rollout_idx,
        ),
        agent=AtifAgent(name=_AGENT_NAME, version="1"),
        steps=steps,
    )
    atif_extras.set_rollout_status(
        trajectory, _rollout_status_for(exit_extra.get("exit_status", ""))
    )
    atif_extras.set_is_correct(trajectory, is_correct)
    return ScoredTrajectory(
        trajectory=trajectory, reward=Reward(reward=float(episode_reward))
    )
