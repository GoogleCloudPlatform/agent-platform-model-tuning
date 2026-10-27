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

"""Typed `.extra` contract for Trajectory/Step/FinalMetrics/Agent.

Two things live here:

1. **Typed pydantic models** (``Cloudside*Extras``) — the declared shape of
   each ``.extra`` dict. ATIF's own schema uses ``extra: dict[str, Any] |
   None`` (deliberately loose); these models tighten that surface for this
   repo: every key we write/read is declared with a type, and
   ``extra="forbid"`` catches typos at construction. Consumed by the codec.
2. **Typed accessors** — ~3-line getter/setter helpers over the conventional
   ``.extra`` keys, keeping dict-access noise out of the hot
   metric-aggregator and trainer paths. All accessors tolerate ``extra=None``.
"""

from __future__ import annotations

from typing import Any

from core.atif_trajectory import Step, Trajectory
from pydantic import BaseModel

from core.rollout_status import RolloutStatus


class CloudsideStepExtras(BaseModel):
    """Typed shape of Step.extra.

    NOTE: ``_raw`` (underscore prefix) is an opaque escape hatch for raw
    sampler wire data (may contain non-JSON payloads). It is NOT
    validated here; the codec strips it before validation. Do not add
    other underscore-prefixed keys.
    """

    candidate_id: str | None = None
    train_step_id: str | None = None
    # The backend's own enum name ("STOP" / "MAX_TOKENS" /
    # "MALFORMED_FUNCTION_CALL"). There is no ``native_finish_reason``
    # companion: every Sampler maps its vendor vocabulary into the
    # ``Candidate`` proto at its own boundary, so this field is already
    # native for every backend. See ``StepStats``.
    finish_reason: str | None = None
    # Sibling child trials referenced by this turn:
    # [{"trajectory_path": "../solve-17-k2/agent/trajectory.json",
    #   "session_id": "solve-17-k2"}]. A relative sibling path, not an
    # inlined copy — nesting children under the parent's agent/ duplicated
    # hundreds of MB per multi-agent run.
    subagent_refs: list[dict[str, str]] | None = None
    llm_call_uri: str | None = None
    operation_id: str | None = None
    advantage: float | None = None
    sample_time_s: float | None = None
    env_time_s: float | None = None
    # Wall time of the whole tool fan-out for this step. Differs from
    # ``env_time_s`` (the summed per-tool time) whenever the dispatcher
    # runs tool calls in parallel.
    tool_time_total_s: float | None = None
    prompt_token_count: int | None = None
    generation_token_count: int | None = None
    # Reasoning tokens. They are billed and they have no typed slot in
    # harbor's Metrics (prompt/completion/cached only).
    thought_token_count: int | None = None
    # Context-compaction bookkeeping. Present only on the summarization
    # step a compaction emits; absent on every other step. That step is
    # source="agent" (the model authored the summary and it enters the
    # context, so its candidate trains), which makes these keys the ONLY
    # way to tell a compaction turn from a solving turn.
    # The compaction_ prefix is load-bearing: .extra is one FLAT namespace
    # that already holds prompt_token_count, so a bare pre_tokens would not
    # say pre-WHAT, and offline consumers flatten .extra to CSV/BigQuery
    # columns where the step's source is not visible at all.
    # Both are BACKEND-REPORTED, request-scope counts (system_instruction
    # and tool declarations included), so pre - post is like-for-like.
    # pre is the projection that tripped the trigger and is always present
    # on a compaction step -- which is why it, not post, is the marker.
    # post is the prompt count of the FIRST call after the compaction and
    # is stamped a turn later, so it is absent when no such call lands.
    compaction_pre_tokens: int | None = None
    compaction_post_tokens: int | None = None
    compaction_pairs_elided: int | None = None
    # Kept deliberately even though select_tail currently caps the tail at 2
    # pairs with a 1-pair floor (~1 bit of information): if MAX_TAIL_PAIRS
    # ever becomes configurable, this recorded count is what keeps already
    # written trajectories interpretable. Do not delete it as "derivable".
    compaction_pairs_retained: int | None = None
    # Per-step terminal status, as a ``RolloutStatus`` VALUE. Absent means
    # ``OK`` (``from_wire`` omits the default). It is also what a
    # ``source="system"`` step records for a generation the loop DISCARDED
    # — there is no separate ``fault`` key, because the status is the only
    # place a fault is named (see ``core.rollout_status``).
    step_status: str | None = None
    # Which sampling attempt produced this step. Only meaningful alongside
    # a discard: attempt 1 is the unremarkable case and is omitted.
    attempt: int | None = None

    model_config = {"extra": "forbid"}


class CloudsideFinalMetricsExtras(BaseModel):
    total_sample_time_s: float | None = None
    total_env_time_s: float | None = None

    model_config = {"extra": "forbid"}


class CloudsideAgentExtras(BaseModel):
    """Typed shape of Trajectory.agent.extra: the rollout's generation knobs."""

    thinking_level: str | None = None
    max_generation_tokens: int | None = None
    include_thoughts: bool | None = None

    model_config = {"extra": "forbid"}


class CloudsideTrajectoryExtras(BaseModel):
    """Typed shape of Trajectory.extra.

    Absence of `reward`/`reward_source`/`analysis` is intentional —
    those are verifier/analyzer-owned. The `extra="forbid"` config on this
    model rejects them (and any other undeclared key) at validation time.

    ``dispatch_index`` / ``phase`` / ``rollout_idx`` / ``task_id`` are the
    per-rollout identity fragments the pipeline stamps at rollout START
    (see ``core.agent.Agent._new_rollout_trajectory``). Observers read
    them off the live trajectory rather than accepting them as separate
    callback kwargs. Persisting them to disk/wire is deliberate —
    trajectory JSONs are then self-describing enough that a lone JSON
    can be traced back to its dispatch without re-consulting the run's
    pipeline log.
    """

    is_correct: bool | None = None
    rollout_status: str | None = None
    task_stats: dict[str, Any] | None = None
    task_step_stats: list[dict[str, Any]] | None = None
    # Per-criterion score breakdown. Carried here only in transit: the
    # ATIF disk writer POPS it and emits it as `verifier/reward-details.json`,
    # the slot the contract reserves for it (DESIGN.md: it "must NOT be
    # inlined into" reward.json) and the one the viewer renders under its
    # Verifier Logs tab. A recipe that scores against named criteria sets
    # it so a partial reward is readable without replaying the rollout.
    reward_details: dict[str, Any] | None = None
    dispatch_index: int | None = None
    phase: str | None = None
    rollout_idx: int | None = None
    task_id: str | None = None
    # True end-to-end wall clock for this rollout, in seconds. NOT the sum
    # of the per-step ``sample_time_s`` + ``env_time_s``: it also contains
    # env construction, tool/sandbox execution, retries, and any other
    # in-rollout waiting those per-step timers never see.
    rollout_wall_time_s: float | None = None
    # The recipe's whole declarative surface as `AgentSpec.config_snapshot()`
    # produced it (scalars and names, never pickled callables). Written by
    # `TrajectoryJournal.finish`, read by the viewer's config drawer; this
    # is what makes a run reproducible from its artifact alone.
    agent_config: dict[str, Any] | None = None
    # Set on a CHILD trial only: the session_id of the orchestrator that
    # spawned it. The child is written as a SIBLING directory
    # (`<job>/<child-session>/`), so this is the only thing inside the
    # child's own artifact that says whose child it is — the parent's half
    # of the link is `step.extra["subagent_refs"]`. Whatever later assigns
    # credit across a parent/child chain reads it to decide whether the
    # child's samples are trained on.
    #
    # It has to be DECLARED even though nothing in the single-agent path
    # writes it: these models are `extra="forbid"`, and the multi-agent
    # writers wrap the write in `contextlib.suppress`, so an undeclared
    # key does not raise — it makes the whole child trajectory silently
    # vanish.
    parent_session_id: str | None = None
    # Samples we PAID for and threw away: a candidate the loop re-drew
    # rather than entered into the state. Two things write it — an
    # unusable-but-returned turn (which also leaves a `source="system"`
    # step) and a serving flake that never became a turn at all (which
    # cannot, because there is no generation to record).
    #
    # A trajectory-level COUNT rather than a step per attempt on purpose. A
    # re-drawn flake carries no text and no `candidate_id`, so a step for it
    # would be almost all envelope; what anyone actually asks of this data
    # is "how often, and why", and that is two scalars. Without them a
    # rollout that burned ten sampler calls on flakes is indistinguishable
    # on disk from one that sampled cleanly once.
    discarded_samples: int | None = None
    # The `RolloutStatus` VALUE of the last discard. Not a free-text
    # reason: the status vocabulary is the only place a fault is named.
    last_discard_reason: str | None = None

    model_config = {"extra": "forbid"}


def get_train_step_id(step: Step) -> int | None:
    v = (step.extra or {}).get("train_step_id")
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def get_candidate_id(step: Step) -> str | None:
    return (step.extra or {}).get("candidate_id")


def get_operation_id(step: Step) -> str | None:
    return (step.extra or {}).get("operation_id")


def get_finish_reason(step: Step) -> str | None:
    return (step.extra or {}).get("finish_reason")


def get_advantage(step: Step) -> float | None:
    return (step.extra or {}).get("advantage")


def set_advantage(step: Step, v: float | None) -> None:
    if step.extra is None:
        step.extra = {}
    if v is None:
        step.extra.pop("advantage", None)
    else:
        step.extra["advantage"] = float(v)


def get_sample_time_s(step: Step) -> float | None:
    return (step.extra or {}).get("sample_time_s")


def get_env_time_s(step: Step) -> float | None:
    return (step.extra or {}).get("env_time_s")


def get_tool_time_total_s(step: Step) -> float | None:
    """Wall time of the whole tool fan-out for this step.

    Differs from ``env_time_s`` (the summed per-tool time) whenever the
    dispatcher runs tool calls in parallel.
    """
    return (step.extra or {}).get("tool_time_total_s")


def get_step_status(step: Step) -> RolloutStatus:
    """Per-step status. Absent means ``OK`` (``from_wire`` omits the default)."""
    v = (step.extra or {}).get("step_status")
    return RolloutStatus(v) if v is not None else RolloutStatus.OK


# Token counts live in TWO places and a reader must accept both:
#   * ``step.extra`` — what ``TrajectoryStep.from_wire`` writes (the proto
#     path, and every trajectory written before the trainable-agent core);
#   * ``step.metrics`` — ATIF's own typed slot, which ``TrajectoryJournal``
#     writes because a typed field beats an ``extra`` key.
# Reading only ``extra`` made ``sampler/generation_tokens_sum`` and
# ``sampler/peak_context_tokens_mean`` vanish from a ported recipe's run —
# no error, the keys are simply omitted when nothing reports a count.
# Check ``extra`` first so existing trajectories stay bit-identical.


def get_prompt_token_count(step: Step) -> int | None:
    v = (step.extra or {}).get("prompt_token_count")
    if v is not None:
        return v
    metrics = getattr(step, "metrics", None)
    return getattr(metrics, "prompt_tokens", None) if metrics else None


def get_generation_token_count(step: Step) -> int | None:
    v = (step.extra or {}).get("generation_token_count")
    if v is not None:
        return v
    metrics = getattr(step, "metrics", None)
    return getattr(metrics, "completion_tokens", None) if metrics else None


def get_rollout_status(traj: Trajectory) -> RolloutStatus:
    v = (traj.extra or {}).get("rollout_status")
    return RolloutStatus(v) if v is not None else RolloutStatus.OK


def set_rollout_status(traj: Trajectory, status: RolloutStatus) -> None:
    if traj.extra is None:
        traj.extra = {}
    traj.extra["rollout_status"] = status.value


def get_is_correct(traj: Trajectory) -> bool | None:
    """Per spec Q3: is_correct is trajectory-level."""
    return (traj.extra or {}).get("is_correct")


def set_is_correct(traj: Trajectory, v: bool) -> None:
    if traj.extra is None:
        traj.extra = {}
    traj.extra["is_correct"] = bool(v)


def set_reward_details(traj: Trajectory, details: dict[str, Any]) -> None:
    """Attach the per-criterion breakdown for `verifier/reward-details.json`."""
    if traj.extra is None:
        traj.extra = {}
    traj.extra["reward_details"] = details


def pop_reward_details(traj: Trajectory) -> dict[str, Any] | None:
    """Take the breakdown off the trajectory for separate serialization.

    Popped rather than read so it does not ALSO get inlined into
    `agent/trajectory.json`, which the contract forbids.
    """
    if not traj.extra:
        return None
    v = traj.extra.pop("reward_details", None)
    return v if isinstance(v, dict) else None


def get_rollout_wall_time_s(traj: Trajectory) -> float | None:
    return (traj.extra or {}).get("rollout_wall_time_s")


def set_rollout_wall_time_s(traj: Trajectory, seconds: float) -> None:
    if traj.extra is None:
        traj.extra = {}
    traj.extra["rollout_wall_time_s"] = float(seconds)


def get_task_stats(traj: Trajectory) -> Any:
    return (traj.extra or {}).get("task_stats")


def get_step_task_stats(traj: Trajectory, i: int) -> Any:
    lst = (traj.extra or {}).get("_task_step_stats") or []
    return lst[i] if 0 <= i < len(lst) else None


def finalize_trajectory(
    traj: Trajectory,
    *,
    status: RolloutStatus,
    is_correct: bool | None = None,
    per_step_stats: list[Any] | None = None,
    task_stats: Any = None,
) -> None:
    """Stamp the terminal status + optional aggregates on a live trajectory.

    Collapses the "finalize" block that every agent runs at the end of a
    rollout: set the terminal ``rollout_status``, optionally the
    trajectory-level ``is_correct``, and optionally the per-step +
    per-trajectory task_stats aggregates. Kept as one call so an agent's
    end-of-loop finalize is a single line rather than 5-8 lines of setter
    plumbing that drift out of sync across recipes.
    """
    set_rollout_status(traj, status)
    if is_correct is not None:
        set_is_correct(traj, is_correct)
    if traj.extra is None:
        traj.extra = {}
    if task_stats is not None:
        traj.extra["task_stats"] = task_stats
    if per_step_stats is not None:
        traj.extra["_task_step_stats"] = per_step_stats


# ---- Per-rollout identity accessors ------------------------------------
#
# These live in traj.extra so a single ``Trajectory`` reference carries
# all the identity fragments an observer needs — no more threading them
# as separate callback kwargs. Set once at rollout START by
# ``Agent._new_rollout_trajectory``; observers only read.


def set_rollout_identity(
    traj: Trajectory,
    *,
    dispatch_index: int | None,
    phase: str,
    rollout_idx: int,
    task_id: str,
) -> None:
    """Stamp the per-rollout identity fragments into ``traj.extra``.

    Idempotent; safe to call before or after other ``.extra`` writes.
    """
    if traj.extra is None:
        traj.extra = {}
    traj.extra["dispatch_index"] = dispatch_index
    traj.extra["phase"] = phase
    traj.extra["rollout_idx"] = rollout_idx
    traj.extra["task_id"] = task_id


def get_dispatch_index(traj: Trajectory) -> int | None:
    return (traj.extra or {}).get("dispatch_index")


def get_phase(traj: Trajectory) -> str | None:
    return (traj.extra or {}).get("phase")


def get_rollout_idx(traj: Trajectory) -> int | None:
    return (traj.extra or {}).get("rollout_idx")


def get_task_id(traj: Trajectory) -> str | None:
    return (traj.extra or {}).get("task_id")
