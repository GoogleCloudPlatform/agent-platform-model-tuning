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

"""ATIF-native in-memory trajectory types.

This module is the single seam for the underlying trajectory schema:
the harbor types (``Trajectory``, ``Step``, ``Agent``, ``ToolCall``,
``Observation``, ``ObservationResult``, ``FinalMetrics``) are imported
here and re-exported so downstream code depends only on
``core.atif_trajectory``. Callers should NOT import
``harbor.models.trajectories`` directly.

Convenience aliases ``AtifTrajectory``/``AtifAgent`` are also
re-exported to match the existing call-site naming.
"""

from __future__ import annotations

import copy
import enum
import json
import math
import random
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

import chz
import numpy as np
from google.cloud.aiplatform_v1.types import Content, GenerateContentRequest
from google.protobuf.json_format import MessageToDict
from harbor.models.trajectories import (
    Agent,
    FinalMetrics,
    Step,
    ToolCall,
    Trajectory,
)
from harbor.models.trajectories.observation import Observation
from harbor.models.trajectories.observation_result import ObservationResult
# The parent half of a multi-agent link, and the ONLY spelling of it any
# consumer outside this repo reads: the viewer counts and renders
# `ObservationResult.subagent_trajectory_ref`, the shipped multi-agent
# example is built on it, and the validator type-checks it. Re-exported
# here rather than imported from harbor at the call site because this
# module is the single seam for the schema (see the module docstring) --
# and because a writer that reaches around the seam is how the field came
# to be missing from every trajectory this repo emitted.
from harbor.models.trajectories.subagent_trajectory_ref import (
    SubagentTrajectoryRef,
)

# Convenience aliases so downstream callers can keep the existing
# `AtifTrajectory` / `AtifAgent` naming without an `import ... as`.
AtifTrajectory = Trajectory
AtifAgent = Agent

# Re-export the canonical RolloutStatus for back-compat imports.
from core.rollout_status import RolloutStatus  # noqa: F401,E402 (re-export)
from core.task_data import TaskData  # noqa: E402
from core import atif_extras  # noqa: E402
from core import gemini_proto_utils

__all__ = [
    # harbor schema re-exports
    "Agent",
    "AtifAgent",
    "AtifTrajectory",
    "FinalMetrics",
    "Observation",
    "ObservationResult",
    "Step",
    "SubagentTrajectoryRef",
    "ToolCall",
    "Trajectory",
    # local types defined below
    "StepStats",
    "StepTimings",
    "RolloutStatus",
    "TaskData",
    "ScoredTrajectory",
    "build_leading_steps",
]


@dataclass
class StepStats:
    """Ingestion-only: consumed by TrajectoryStep.from_wire. ``is_correct``
    is omitted here because it is a trajectory-level property."""

    finish_reason: str | None = None
    candidate_id: str | None = None
    operation_id: str | None = None
    prompt_token_count: int | None = None
    generation_token_count: int | None = None
    train_step_id: int | None = None
    status: RolloutStatus = RolloutStatus.OK

    # Thinking-only subset of ``generation_token_count``, and the cached
    # prompt-token count, when the backend reported them. ``None`` rather
    # than 0 when unreported: a cost-shaped reward reads a missing count as
    # "unknown" (no credit) and a real 0 as "free" (full credit).
    thought_token_count: int | None = None

    # NOTE there is deliberately no ``native_finish_reason`` companion
    # field. ``finish_reason`` above already holds the backend's own enum
    # name ("STOP" / "MAX_TOKENS" / "MALFORMED_FUNCTION_CALL"), because
    # every ``Sampler`` maps its vendor vocabulary into the ``Candidate``
    # proto at its own boundary and ``SampleResult.finish_reason`` reads
    # that enum's name. The OpenAI wire spelling ("stop" / "length") is a
    # downstream projection applied when converting to the OAI response
    # shape; it is never stored, and storing it here would fork the live
    # ``sampler/finish_reason/*`` metric family in two.


@dataclass
class StepTimings:
    sample_time_s: float
    env_time_s: float


@dataclass
class Reward:
    """Verifier-owned. Held SEPARATELY from the trajectory (ATIF forbids inline reward).

    Three decoupled quantities, and conflating any two of them is the
    historical Pain 9:

    * ``raw_reward`` -- the ground-truth benchmark metric, exactly as the
      grader measured it from the environment's final state. Never floored,
      never shaped, never normalized. This is the number a paper quotes.
    * ``reward`` -- the FINAL reward after the fault policy (NaN for infra,
      floor for a model fault when the recipe opts in). This is what the
      trainer turns into an advantage.
    * ``normalized_reward`` -- the batch advantage, written in place by a
      ``RewardNormalizer`` across a ``TrajectoryGroup``.

    ``details`` is the grader's named dimensions (``{"partial_credit": .6,
    "tests_passed": 17}``), logged verbatim to ``verifier/reward.json`` so a
    benchmark score can be recomputed without re-running anything.
    """

    reward: float
    normalized_reward: float | None = None
    raw_reward: float | None = None
    details: dict[str, Any] = field(default_factory=dict)


# Structural types for the snake_case MessageToDict shape of a Gemini
# ``Candidate.content`` proto. Documentation + static-check aids only;
# TypedDicts have no runtime presence, so ``_parse_generation_dict``
# still validates every field at runtime (input is untrusted wire data
# and may be malformed). ``total=False`` because a part carries exactly
# one of ``text`` / ``function_call`` (plus opaque extras like
# ``thought_signature``) and a generation may omit ``parts`` entirely.
# The wire->ATIF boundary stays proto-agnostic.


class FunctionCallDict(TypedDict, total=False):
    """The ``function_call`` sub-dict of a Gemini part."""

    name: str
    args: dict[str, Any]


class PartDict(TypedDict, total=False):
    """One entry of a generation's ``parts`` list. Well-formed parts
    carry exactly one of ``text`` / ``function_call``; a malformed part
    carrying both contributes to both outputs (lossless but the
    per-part ordering isn't preserved)."""

    text: str
    thought: bool
    function_call: FunctionCallDict


class GenerationDict(TypedDict, total=False):
    """The snake_case MessageToDict of a ``Candidate.content`` proto."""

    role: str
    parts: list[PartDict]


def content_to_generation_dict(content: Any) -> GenerationDict:
    """Convert a ``Content`` proto to a snake_case dict for ATIF.

    Returns ``{}`` when ``content`` is None or has no parts/role.
    """
    if content and (content.parts or content.role):
        return MessageToDict(
            type(content).pb(content), preserving_proto_field_name=True
        )
    return {}


def _parse_generation_dict(
    gen: GenerationDict,
    *,
    include_thoughts: bool = False,
) -> tuple[str, list[ToolCall] | None, list[dict[str, Any]] | None]:
    """Split a Gemini generation ``content`` dict into ATIF ``message`` (joined
    non-thought text parts), ``tool_calls`` and ``tool_responses``.

    ``thought=True`` text parts are reasoning, not the reply, and are left
    out of ``message``; see ``_extract_reasoning``. ``include_thoughts=True``
    keeps them in ``message`` in part order, for steps that may not carry
    ``reasoning_content`` (harbor allows it only on ``source="agent"``).

    Returns ``(message, tool_calls, tool_responses)``.
    """
    parts = gen.get("parts") or []
    texts: list[str] = []
    tool_calls: list[ToolCall] = []
    tool_responses: list[dict[str, Any]] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str) and (include_thoughts or not part.get("thought")):
            texts.append(text)
        fc = part.get("function_call")
        if isinstance(fc, dict):
            args = fc.get("args")
            tool_calls.append(
                ToolCall(
                    tool_call_id=uuid.uuid4().hex,
                    function_name=str(fc.get("name", "")),
                    arguments=args if isinstance(args, dict) else {},
                )
            )
        fr = part.get("function_response")
        if isinstance(fr, dict):
            response = fr.get("response")
            if isinstance(response, dict):
                tool_responses.append(response)

    message = "".join(texts)
    return message, (tool_calls or None), (tool_responses or None)


def _render_discarded_calls(message: str, tool_calls: list[ToolCall]) -> str:
    """Append ``tool_calls`` to ``message`` for a non-agent step, which may
    not carry ``tool_calls``: one ``[discarded function_call] name(<json
    args>)`` line per call, in part order, after the text (no leading
    newline when the text is empty)."""
    lines = [message] if message else []
    lines += [
        f"[discarded function_call] {tc.function_name}"
        f"({json.dumps(tc.arguments, ensure_ascii=False)})"
        for tc in tool_calls
    ]
    return "\n".join(lines)


def _extract_reasoning(gen: GenerationDict) -> str | None:
    """Join the ``thought=True`` text parts of a generation; ``None`` if none."""
    texts: list[str] = []
    for part in gen.get("parts") or []:
        if not isinstance(part, dict) or not part.get("thought"):
            continue
        text = part.get("text")
        if isinstance(text, str):
            texts.append(text)
    return "".join(texts) if texts else None


def _stringify_function_response(fr: dict[str, Any]) -> str:
    """Extract a readable string from one Gemini ``function_response``
    wire dict. Returns the JSON-serialized ``response`` payload; falls
    back to the whole dict on unexpected shape. Never drops information
    and never raises on a malformed entry.

    Accepts either the proto-JSON camelCase key (``functionResponse``, what
    ``Part.from_json`` requires and what envs emit into
    ``EnvStepResult.function_responses``) or the snake_case ``function_response``
    form (older / non-proto callers).
    """
    import json as _json

    fr_inner: Any = None
    if isinstance(fr, dict):
        fr_inner = fr.get("functionResponse") or fr.get("function_response")
    payload = fr_inner.get("response") if isinstance(fr_inner, dict) else None
    if payload is None:
        payload = fr
    if isinstance(payload, str):
        return payload
    try:
        return _json.dumps(payload, default=str)
    except (TypeError, ValueError):
        return str(payload)


def _build_observation(
    tool_responses: list[dict[str, Any]] | None,
    tool_calls: list[ToolCall] | None,
) -> Observation | None:
    """Build an ATIF ``Observation`` from the env's ``tool_responses``.

    Responses are paired to ``tool_calls`` BY POSITION (Gemini returns
    them in the same order). Surplus responses (more responses than
    calls) carry ``source_call_id=None`` rather than being dropped.
    Returns ``None`` when there are no responses.
    """
    if not tool_responses:
        return None
    calls = tool_calls or []
    results: list[ObservationResult] = []
    for i, fr in enumerate(tool_responses):
        source_call_id = calls[i].tool_call_id if i < len(calls) else None
        results.append(
            ObservationResult(
                source_call_id=source_call_id,
                content=_stringify_function_response(fr),
            )
        )
    return Observation(results=results)


class TrajectoryStep(Step):
    """ATIF Step + factory + typed extras accessors.

    ``message`` is populated from generation (non-thought text parts joined
    for dict-form generations, or the raw string for text-only generations);
    ``thought=True`` parts go to ``reasoning_content``.
    """

    @classmethod
    def from_wire(
        cls,
        *,
        step_id: int,
        stats: StepStats,
        generation: str | GenerationDict,
        timings: StepTimings | None = None,
        tool_responses: list[dict[str, Any]] | None = None,
        raw_call_uri: str | None = None,
        source: Literal["agent", "system"] = "agent",
    ) -> TrajectoryStep:
        """Build an ATIF-shaped agent Step from raw sampler I/O + StepStats.

        ``generation`` may be a plain str (text-only backends) or a
        ``GenerationDict`` (tool-using backends). A dict is split via
        ``_parse_generation_dict`` into ATIF ``message`` + ``tool_calls``.

        `tool_responses` are the env's tool results for this step's
        tool_calls (Gemini `function_response` wire dicts). When provided
        they populate the ATIF `observation` field, with each result linked
        to the tool call it answers via `source_call_id` (see
        `_build_observation`). Absent/empty -> `observation=None`.

        `raw_call_uri` is the relative path to the archived raw sampler
        request/response (written by `utils.raw_capture.dump_raw` when
        `RAW_CAPTURE=1`), sourced from `SamplingContext.raw_call_uri`.
        When non-None it is stamped as `step.extra["llm_call_uri"]`; when
        None the key is omitted to keep `extra` minimal (same convention
        as the `StepStats` token-count fields).

        `source` defaults to "agent". Harbor's `Step` rejects every
        agent-only field (`model_name`, `reasoning_effort`,
        `reasoning_content`, `tool_calls`, `metrics`) on non-agent steps, so
        a "system" step (e.g. a discarded recovery sample) carries none of
        them: thought parts stay in `message`, joined in part order, and
        each function_call is appended to `message` as one line
        `[discarded function_call] <name>(<json args>)` (see
        `_render_discarded_calls`). This method never sets `model_name`,
        `reasoning_effort` or `metrics`. Pass `source` here rather than
        assigning `.source` afterwards: assignment skips validation and
        the step would fail to reload.
        """
        tool_calls: list[ToolCall] | None = None
        reasoning_content: str | None = None
        if isinstance(generation, str):
            message = generation
        elif isinstance(generation, dict):
            # only expect message and tool_calls from model response
            agent = source == "agent"
            message, tool_calls, _ = _parse_generation_dict(
                generation, include_thoughts=not agent
            )
            if agent:
                reasoning_content = _extract_reasoning(generation)
            elif tool_calls:
                message = _render_discarded_calls(message, tool_calls)
                tool_calls = None
        else:
            raise TypeError(
                f"generation must be str or dict, got {type(generation).__name__}"
            )

        extra: dict[str, Any] = {}
        if stats.finish_reason is not None:
            extra["finish_reason"] = stats.finish_reason
        if stats.candidate_id is not None:
            extra["candidate_id"] = stats.candidate_id
        if stats.operation_id is not None:
            extra["operation_id"] = stats.operation_id
        if stats.train_step_id is not None:
            extra["train_step_id"] = str(stats.train_step_id)
        if stats.prompt_token_count is not None:
            extra["prompt_token_count"] = stats.prompt_token_count
        if stats.generation_token_count is not None:
            extra["generation_token_count"] = stats.generation_token_count
        if stats.thought_token_count is not None:
            extra["thought_token_count"] = stats.thought_token_count
        if stats.status != RolloutStatus.OK:
            # Per-step status was computed and then dropped on the floor
            # before this line existed. ``OK`` is the default and is
            # omitted, same minimal-``extra`` convention as the fields
            # above: absent means OK. Serialize the enum VALUE (as
            # ``set_rollout_status`` does) — ``RolloutStatus`` is a
            # ``(str, Enum)``, NOT a ``StrEnum``, so ``str(status)``
            # would write "RolloutStatus.MAX_STEP_LIMIT".
            extra["step_status"] = RolloutStatus(stats.status).value
        if timings is not None:
            extra["sample_time_s"] = timings.sample_time_s
            extra["env_time_s"] = timings.env_time_s
        if raw_call_uri is not None:
            extra["llm_call_uri"] = raw_call_uri
        observation = _build_observation(tool_responses, tool_calls)
        return cls(
            step_id=step_id,
            source=source,
            message=message,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
            observation=observation,
            extra=extra,
        )

    @property
    def candidate_id(self) -> str | None:
        return (self.extra or {}).get("candidate_id")

    @property
    def finish_reason(self) -> str | None:
        return (self.extra or {}).get("finish_reason")

    @property
    def train_step_id(self) -> int | None:
        v = (self.extra or {}).get("train_step_id")
        if v is None:
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    @property
    def advantage(self) -> float | None:
        return (self.extra or {}).get("advantage")

    @advantage.setter
    def advantage(self, v: float | None) -> None:
        if self.extra is None:
            self.extra = {}
        if v is None:
            self.extra.pop("advantage", None)
        else:
            self.extra["advantage"] = float(v)


def _role_to_source(role: str) -> str:
    role = role.lower().strip()
    match role:
        case "user":
            return "user"
        case "model" | "assistant":
            return "agent"
        case _:
            raise ValueError(f"unrecognized role: {role}")


def _merge_consecutive_contents_with_same_role(
    request: GenerateContentRequest,
) -> list[Content]:
    """Return contents with adjacent same-role entries merged into one.

    Multi-turn Gemini prompts sometimes split what is logically one turn
    across several ``Content`` entries with the same ``role``; downstream
    reconstruction is simpler when each role transition maps to exactly
    one merged entry.
    """
    contents: list[Content] = []
    cur_content: Content | None = None
    for content in request.contents:
        if cur_content is None:
            cur_content = copy.deepcopy(content)
            continue
        if cur_content.role == content.role:
            cur_content.parts.extend(content.parts)
            continue
        contents.append(cur_content)
        cur_content = copy.deepcopy(content)
    if cur_content is not None:
        contents.append(cur_content)
    return contents


def build_leading_steps(request: GenerateContentRequest) -> list[Step]:
    """Reconstruct ATIF leading steps from a (possibly partial) GenerateContentRequest.

    Every turn in ``request.contents`` becomes an ATIF step:

    - Optional leading ``system`` step (from ``system_instruction``).
    - Required leading ``user`` step (from ``contents[0]``, which MUST have
      ``role='user'``).
    - Zero or more ``agent`` steps reconstructed from pairs of subsequent
      contents: an agent turn (``role='model'`` / ``'assistant'``) buffers
      its message + tool_calls, and the following user turn's
      ``tool_responses`` flushes them into a single ATIF step with an
      ``observation``. A trailing agent turn without a follow-up response
      is emitted as a text-only agent step.

    Reconstructed agent steps carry only the fields present in the wire
    request (message, reasoning_content from ``thought`` parts, tool_calls,
    observation). They lack the extras a
    live-produced step would have (StepStats, timings, raw_call_uri) —
    that's expected: these steps are historical, not just-executed.

    Raises ``ValueError`` on malformed input (empty contents, non-user
    first turn, or a tail buffered tool_call without a matching response).
    """
    contents = _merge_consecutive_contents_with_same_role(request)
    steps: list[Step] = []
    if len(contents) == 0 or _role_to_source(contents[0].role) != "user":
        raise ValueError(f"Invalid request, first role is not user, {request=}")
    user_message = gemini_proto_utils.extract_text_from_content(contents[0])
    si = gemini_proto_utils.extract_system_instruction_text(request)
    if si:
        steps.append(Step(step_id=len(steps) + 1, source="system", message=si))
    steps.append(Step(step_id=len(steps) + 1, source="user", message=user_message))

    # Reconstruct agent steps from the remaining conversation history.
    cur_message: str | None = None
    cur_reasoning: str | None = None
    cur_tool_calls: list[ToolCall] | None = None
    for content in contents[1:]:
        gen = content_to_generation_dict(content)
        message, tool_calls, tool_responses = _parse_generation_dict(gen)
        if tool_responses:
            if cur_tool_calls is None:
                raise ValueError(
                    "tool_response content without a preceding agent turn "
                    f"buffering tool_calls; {request=}"
                )
            steps.append(
                TrajectoryStep(
                    step_id=len(steps) + 1,
                    source="agent",
                    message=cur_message or "",
                    reasoning_content=cur_reasoning,
                    tool_calls=cur_tool_calls,
                    observation=_build_observation(tool_responses, cur_tool_calls),
                )
            )
            cur_message = None
            cur_reasoning = None
            cur_tool_calls = None
        else:
            cur_tool_calls = tool_calls
            cur_message = message
            cur_reasoning = _extract_reasoning(gen)
    if cur_message is not None:
        steps.append(
            TrajectoryStep(
                step_id=len(steps) + 1,
                source="agent",
                message=cur_message,
                reasoning_content=cur_reasoning,
            )
        )
    if cur_tool_calls:
        raise ValueError(
            f"Tail tool calls without matching tool_response, "
            f"{cur_tool_calls=} {request=}"
        )
    return steps


@dataclass
class ScoredTrajectory:
    trajectory: Trajectory
    reward: Reward


@dataclass
class TrajectoryGroup:
    task_data: TaskData | None
    scored_trajectories: list[ScoredTrajectory] = field(default_factory=list)


def compute_timing_metrics(
    scored_trajectories: list[ScoredTrajectory],
) -> dict[str, float]:
    """Aggregates per-step and per-trajectory sample/env times.

    Reads timings out of each agent step's ``extra["sample_time_s"]`` /
    ``extra["env_time_s"]`` (populated by ``TrajectoryStep.from_wire``).
    """
    per_step_sample_times: list[float] = []
    per_step_env_times: list[float] = []
    per_traj_total_sample: list[float] = []
    per_traj_total_env: list[float] = []
    for g in scored_trajectories:
        s_total = 0.0
        e_total = 0.0
        for step in g.trajectory.steps:
            if step.source != "agent":
                continue
            s = atif_extras.get_sample_time_s(step)
            e = atif_extras.get_env_time_s(step)
            if s is not None:
                per_step_sample_times.append(float(s))
                s_total += float(s)
            if e is not None:
                per_step_env_times.append(float(e))
                e_total += float(e)
        per_traj_total_sample.append(s_total)
        per_traj_total_env.append(e_total)

    metrics: dict[str, float] = {}
    if per_step_sample_times:
        metrics["env/sample_time_avg"] = float(np.mean(per_step_sample_times))
        metrics["env/sample_time_p95"] = float(np.percentile(per_step_sample_times, 95))
    if per_step_env_times:
        metrics["env/env_time_avg"] = float(np.mean(per_step_env_times))
        metrics["env/env_time_p95"] = float(np.percentile(per_step_env_times, 95))
    if per_traj_total_sample:
        metrics["env/total_sample_time_avg"] = float(np.mean(per_traj_total_sample))
    if per_traj_total_env:
        metrics["env/total_env_time_avg"] = float(np.mean(per_traj_total_env))
    if per_traj_total_sample and per_traj_total_env:
        tot_s = sum(per_traj_total_sample)
        tot_e = sum(per_traj_total_env)
        if tot_s + tot_e > 0:
            metrics["env/sample_time_fraction"] = tot_s / (tot_s + tot_e)
    return metrics


class GroupFilterOutcome(str, enum.Enum):
    """How ``TrajectoryGroupFilter`` classified one trajectory group.

    Values double as metric-name leaves
    (``pipeline/group_drop_reason/<value>``) and as the per-trajectory
    ``drop_reason`` reported to observers, so they are snake_case and
    must stay stable once dashboards key off them. ``KEPT`` is the
    empty string so ``if outcome:`` still reads as "was dropped" at
    call sites that only care about that.

    Inherits ``str`` so ``GroupFilterOutcome.ZERO_VARIANCE ==
    "zero_variance"`` is True.
    """

    KEPT = ""
    #: Zero-variance group that survived the ``zero_variance_keep_prob``
    #: coin flip. Distinguished from plain ``KEPT`` so the zero-variance
    #: RATE stays observable at ``keep_prob=1.0``, where the drop count
    #: is identically zero and therefore says nothing about how often
    #: groups actually tie.
    KEPT_ZERO_VARIANCE = "kept_zero_variance"
    ZERO_VARIANCE = "zero_variance"
    TOO_FEW_VALID_TRAJECTORIES = "too_few_valid_trajectories"
    TOO_MANY_TRANSIENT_FAILURES = "too_many_transient_failures"

    @property
    def is_drop(self) -> bool:
        """Whether this outcome means the group was filtered out."""
        return self not in (
            GroupFilterOutcome.KEPT,
            GroupFilterOutcome.KEPT_ZERO_VARIANCE,
        )

    @property
    def is_zero_variance(self) -> bool:
        """Whether the group's rewards were all equal, kept or not."""
        return self in (
            GroupFilterOutcome.ZERO_VARIANCE,
            GroupFilterOutcome.KEPT_ZERO_VARIANCE,
        )

    @property
    def is_structural_failure(self) -> bool:
        """Whether this drop usually means something is actually broken
        (logged at WARNING). Zero-variance is routine RL variance."""
        return self in (
            GroupFilterOutcome.TOO_FEW_VALID_TRAJECTORIES,
            GroupFilterOutcome.TOO_MANY_TRANSIENT_FAILURES,
        )


#: Outcomes that mean "dropped", in the order metrics report them.
GROUP_DROP_OUTCOMES: tuple[GroupFilterOutcome, ...] = tuple(
    o for o in GroupFilterOutcome if o.is_drop
)


@chz.chz
class TrajectoryGroupFilterConfig:
    """Config for ``TrajectoryGroupFilter``; factory via ``.make()``."""

    zero_variance_keep_prob: float = 0.3
    max_transient_failure_fraction: float | None = 0.15

    def make(self) -> "TrajectoryGroupFilter":
        return TrajectoryGroupFilter(self)


class TrajectoryGroupFilter:
    """Filters NaN-reward scored trajectories in place and classifies the
    group as keep/drop. Mutates ``group.scored_trajectories`` in place.
    """

    def __init__(self, config: TrajectoryGroupFilterConfig):
        self._config = config

    def __call__(self, group: TrajectoryGroup) -> tuple[bool, GroupFilterOutcome]:
        cfg = self._config
        if cfg.max_transient_failure_fraction is not None and group.scored_trajectories:
            n_transient = sum(
                1
                for g in group.scored_trajectories
                if atif_extras.get_rollout_status(g.trajectory).is_transient_failure()
            )
            fraction = n_transient / len(group.scored_trajectories)
            if fraction > cfg.max_transient_failure_fraction:
                return False, GroupFilterOutcome.TOO_MANY_TRANSIENT_FAILURES

        valid = [
            g
            for g in group.scored_trajectories
            if not (isinstance(g.reward.reward, float) and math.isnan(g.reward.reward))
        ]
        group.scored_trajectories = valid

        if len(group.scored_trajectories) < 2:
            return False, GroupFilterOutcome.TOO_FEW_VALID_TRAJECTORIES

        rewards = [g.reward.reward for g in group.scored_trajectories]
        is_zero_variance = all(
            math.isclose(r, rewards[0], abs_tol=1e-5) for r in rewards
        )
        if is_zero_variance:
            if random.random() > cfg.zero_variance_keep_prob:
                return False, GroupFilterOutcome.ZERO_VARIANCE
            return True, GroupFilterOutcome.KEPT_ZERO_VARIANCE
        return True, GroupFilterOutcome.KEPT
