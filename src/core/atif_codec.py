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

"""Serialization codec for ATIF trajectories.

Every wire/disk path MUST go through this module. Core invariant:

Typed extras validation via Cloudside*Extras (extra="forbid") rejects
typos, type drift, AND verifier/analyzer-owned keys (reward,
reward_source, analysis) that must not appear in the trajectory.
"""

from __future__ import annotations

import dataclasses
import json
import math
from typing import Any

from core.atif_trajectory import Trajectory

from core.atif_extras import (
    CloudsideAgentExtras,
    CloudsideFinalMetricsExtras,
    CloudsideStepExtras,
    CloudsideTrajectoryExtras,
)
from core.atif_trajectory import ScoredTrajectory, Reward


def _nan_to_none(obj: Any) -> Any:
    """Recursively replace non-finite floats (NaN / +-Inf) with None so
    ``json.dumps(..., allow_nan=False)`` emits RFC 8259-valid JSON.

    Timeout trajectories carry ``reward=float('nan')``; strict parsers
    reject the bare ``NaN`` token the default json.dumps emits.
    """
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_to_none(v) for v in obj]
    return obj


def _json_safe(value: Any) -> Any:
    """Recursively coerce a value into JSON-serializable form.

    Handles non-JSON payloads found under ``step.extra['_raw']``: proto
    messages -> MessageToDict, dataclasses -> asdict, containers ->
    recurse, primitives unchanged, anything else -> ``str(value)`` so
    ``to_disk`` never raises on an exotic object.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_safe(dataclasses.asdict(value))
    # proto-plus message (has _pb + type(x).pb) or raw protobuf message.
    pb = getattr(value, "_pb", None)
    if pb is not None:
        try:
            from google.protobuf.json_format import MessageToDict

            return MessageToDict(pb, preserving_proto_field_name=True)
        except Exception:  # noqa: BLE001 - fall through to str()
            pass
    else:
        try:
            from google.protobuf.message import Message
            from google.protobuf.json_format import MessageToDict

            if isinstance(value, Message):
                return MessageToDict(value, preserving_proto_field_name=True)
        except Exception:  # noqa: BLE001
            pass
    return str(value)


def _normalize_step_task_stat(s: Any) -> dict[str, Any]:
    """Coerce one per-step task-stat entry into a schema-valid dict.

    Positions are preserved (None -> {}, never dropped) so the on-disk
    ``task_step_stats`` list stays index-aligned with the agent steps.
    The in-memory ``get_step_task_stats`` accessor still sees the real
    None (it reads ``_task_step_stats``); only the disk/wire artifact
    carries the {} placeholder.
    """
    if dataclasses.is_dataclass(s) and not isinstance(s, type):
        return dataclasses.asdict(s)
    if isinstance(s, dict):
        return s
    return {}


def _normalize_reward_details(details: Any) -> dict[str, Any]:
    """Coerce a grader's ``details`` mapping into JSON-able shape.

    ``Reward.details`` is the grader's named dimensions and is typed
    ``dict[str, Any]`` on purpose -- a recipe puts its own typed stats
    dataclass in there (a recipe's own ``<Recipe>Stats``).
    Before ``details`` reached ``verifier/reward.json`` that was harmless;
    now it is a hard ``TypeError: not JSON serializable`` inside
    ``json.dumps``, which the callback fan-out swallows as a WARNING -- so
    the symptom is not a crash but **no ATIF output at all** for every
    rollout, while the eval reports a clean score.

    Same rule as ``task_stats``: a dataclass becomes a dict, one level
    deep, and anything still unserializable is stringified rather than
    taking the whole trajectory down with it.
    """
    if not details:
        return {}
    out: dict[str, Any] = {}
    for key, value in dict(details).items():
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            out[key] = dataclasses.asdict(value)
        else:
            out[key] = value
    try:
        json.dumps(out, allow_nan=False)
    except (TypeError, ValueError):
        out = {k: _jsonable(v) for k, v in out.items()}
    return out


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value, allow_nan=False)
        return value
    except (TypeError, ValueError):
        return str(value)


def _normalize_traj_extra(
    extra: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Normalize recipe-owned stats in ``Trajectory.extra`` into the
    schema-conformant JSON shape declared by ``CloudsideTrajectoryExtras``
    (see ``core.atif_extras``):

      * ``task_stats``: dataclass -> ``dataclasses.asdict``.
      * ``_task_step_stats`` (in-memory, underscore) -> ``task_step_stats``
        (the schema's real field); each element coerced via
        ``_normalize_step_task_stat``.

    Returns a NEW dict; the input trajectory is never mutated.
    """
    if not extra:
        return extra
    out = dict(extra)
    ts = out.get("task_stats")
    if dataclasses.is_dataclass(ts) and not isinstance(ts, type):
        out["task_stats"] = dataclasses.asdict(ts)
    if "_task_step_stats" in out:
        steps = out.pop("_task_step_stats") or []
        out["task_step_stats"] = [_normalize_step_task_stat(s) for s in steps]
    return out


def _validate_typed_extras(traj: Trajectory) -> None:
    """Validate every .extra dict against its schema. Underscore-prefixed
    step.extra keys are opaque escape hatches and bypass validation;
    recipe-owned traj.extra stats are normalized first.
    """
    if traj.extra is not None:
        CloudsideTrajectoryExtras.model_validate(_normalize_traj_extra(traj.extra))
    if traj.final_metrics is not None and traj.final_metrics.extra is not None:
        CloudsideFinalMetricsExtras.model_validate(traj.final_metrics.extra)
    if traj.agent.extra is not None:
        CloudsideAgentExtras.model_validate(traj.agent.extra)
    for step in traj.steps:
        if step.extra is not None:
            # Strip underscore-prefixed opaque keys before typed validation.
            e = {k: v for k, v in step.extra.items() if not k.startswith("_")}
            CloudsideStepExtras.model_validate(e)


def to_disk(scored: ScoredTrajectory) -> tuple[str, str]:
    """Returns (agent_trajectory_json, verifier_reward_json).

    Unlike ``to_wire``, the on-disk agent JSON PRESERVES step-level
    underscore keys (``_raw`` raw prompts). Trajectory-level recipe
    stats are still normalized to their schema shape so the artifact
    validates on ``from_disk``.
    """
    _validate_typed_extras(scored.trajectory)
    # Plain (Python-mode) dump so proto payloads under step.extra['_raw']
    # survive as objects; _json_safe then converts them. mode="json"
    # would eagerly fail on the proto before we can coerce.
    body = scored.trajectory.model_dump(exclude_none=True)
    for step in body.get("steps", []):
        if step.get("extra") is not None:
            step["extra"] = _json_safe(step["extra"])
            # Strip _raw from serialized output: the full raw request/response
            # is captured in the per-call blob referenced by llm_call_uri,
            # making the inline _raw redundant bloat on disk.
            step["extra"].pop("_raw", None)
    body["extra"] = _normalize_traj_extra(body.get("extra"))
    if body.get("extra") is None:
        body.pop("extra", None)
    agent_json = json.dumps(body)
    reward_json = json.dumps(
        _nan_to_none(
            {
                "reward": scored.reward.reward,
                "normalized_reward": scored.reward.normalized_reward,
                # The un-shaped benchmark metric and the grader's named
                # dimensions. Omitted when a grader supplied neither, so
                # pre-existing files round-trip byte-identically.
                **(
                    {"raw_reward": scored.reward.raw_reward}
                    if scored.reward.raw_reward is not None
                    else {}
                ),
                **(
                    {"details": _normalize_reward_details(scored.reward.details)}
                    if scored.reward.details
                    else {}
                ),
            }
        ),
        allow_nan=False,
    )
    return agent_json, reward_json


def from_disk(
    agent_trajectory_json: str, verifier_reward_json: str
) -> ScoredTrajectory:
    """Inverse of ``to_disk``, NaN reward included.

    ``Reward.reward`` is typed ``float`` and is never legitimately null:
    a failed rollout carries ``float('nan')``, which ``to_disk`` writes as
    ``null`` only because ``allow_nan=False`` forbids the bare ``NaN``
    token. Reading that back as ``None`` is what makes an offline consumer
    diverge from the pipeline -- ``nanmean`` and ``reward > 0`` both
    tolerate NaN, and both raise ``TypeError`` on None -- so the write-side
    ``_nan_to_none`` is inverted here.

    ``normalized_reward`` is deliberately NOT restored: it is typed
    ``float | None``, so a null there is ambiguous between "was NaN" and
    "never normalized", and None is its legitimate resting value.
    """
    traj = Trajectory.model_validate_json(agent_trajectory_json)
    _validate_typed_extras(traj)
    r = json.loads(verifier_reward_json)
    reward = r["reward"]
    return ScoredTrajectory(
        trajectory=traj,
        reward=Reward(
            reward=float("nan") if reward is None else reward,
            normalized_reward=r.get("normalized_reward"),
            raw_reward=r.get("raw_reward"),
            details=r.get("details") or {},
        ),
    )


def _dump_wire_safe(traj: Trajectory) -> dict[str, Any]:
    """Serialize a Trajectory to a JSON-safe dict.

    Strips underscore-prefixed step.extra keys (may hold non-JSON
    payloads like protos) and normalizes trajectory-level recipe stats
    to their schema shape. The input trajectory is not mutated.
    """
    body = traj.model_dump(exclude_none=True)
    for step in body.get("steps", []):
        extra = step.get("extra")
        if extra is None:
            continue
        step["extra"] = {k: v for k, v in extra.items() if not k.startswith("_")}
    body["extra"] = _normalize_traj_extra(body.get("extra"))
    if body.get("extra") is None:
        body.pop("extra", None)
    return body


def to_wire(scored: ScoredTrajectory) -> dict[str, Any]:
    """Serialize into the MSRL training-request body shape.

    Output is JSON-serializable: underscore-prefixed step.extra keys
    (which may hold protos) are stripped. Callers wanting raw prompts
    should use to_disk instead.
    """
    _validate_typed_extras(scored.trajectory)
    return {
        "trajectory": _dump_wire_safe(scored.trajectory),
        "reward": scored.reward.reward,
        "normalized_reward": scored.reward.normalized_reward,
        **(
            {"raw_reward": scored.reward.raw_reward}
            if scored.reward.raw_reward is not None
            else {}
        ),
        **(
            {"details": _normalize_reward_details(scored.reward.details)}
            if scored.reward.details
            else {}
        ),
    }


def from_wire(body: dict[str, Any]) -> ScoredTrajectory:
    traj = Trajectory.model_validate(body["trajectory"])
    _validate_typed_extras(traj)
    return ScoredTrajectory(
        trajectory=traj,
        reward=Reward(
            reward=body["reward"],
            normalized_reward=body.get("normalized_reward"),
            raw_reward=body.get("raw_reward"),
            details=body.get("details") or {},
        ),
    )
