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

"""chz CLI layer: train vs eval override surfaces.

Parses the small launch-env override set from argv, then merges the
non-None overrides onto a checked-in plain config object.  Sentinel is
``None`` (not ``""``/``0``) so an explicitly-passed value is always
distinguishable from "flag not passed" (design Risk 1).

Train and eval share a common base (Vertex coordinates, backend knobs)
but carry disjoint launch-only fields: train has resume/log/data knobs,
eval has ``target`` and ``jsonl_output_path``.  The ``target`` arg (on
both surfaces) is parsed via :func:`core.target_parser.parse_target`
into project/location/job/endpoint/model components that override the
corresponding backend fields.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import chz

from core.target_parser import TargetInfo, parse_target


@chz.chz
class _BaseLaunchOverrides:
    config_name: str
    target: str | None = None
    vertex_project: str | None = None
    vertex_location: str | None = None


@chz.chz
class TrainLaunchOverrides(_BaseLaunchOverrides):
    vertex_existing_job_id: str | None = None
    vertex_base_model: str | None = None
    log_dir: str | None = None
    resume_from_train_step: int | None = None
    resume_metrics_path: str | None = None
    train_data_path: str | None = None
    eval_data_path: str | None = None
    num_train_steps: int | None = None
    # Size every live-sampler eval sweep of the run (step-0 baseline,
    # cadence, final): tasks per sweep (-1 = all) and samples per task.
    max_eval_items: int | None = None
    eval_samples_per_task: int | None = None


@chz.chz
class EvalLaunchOverrides(_BaseLaunchOverrides):
    eval_data_path: str | None = None
    jsonl_output_path: str | None = None
    max_eval_items: int | None = None


# ── Routing tables ──────────────────────────────────────────────────
# Which sub-config each override field targets.  Keys are override field
# names; values are (sub_config_attr, target_field_name).

_BASE_ROUTING: dict[str, tuple[str, str]] = {
    "vertex_project": ("backend", "project"),
    "vertex_location": ("backend", "location"),
}

_TRAIN_ROUTING: dict[str, tuple[str, str]] = {
    **_BASE_ROUTING,
    "vertex_existing_job_id": ("backend", "vertex_existing_job_id"),
    "vertex_base_model": ("backend", "vertex_base_model"),
    "log_dir": ("train", "log_dir"),
    "resume_from_train_step": ("train", "resume_from_train_step"),
    "resume_metrics_path": ("train", "resume_metrics_path"),
    "train_data_path": ("task", "train_data_path"),
    "eval_data_path": ("task", "eval_data_path"),
    "num_train_steps": ("train", "num_train_steps"),
    "max_eval_items": ("eval", "max_eval_items"),
    "eval_samples_per_task": ("eval", "eval_samples_per_task"),
}

_EVAL_ROUTING: dict[str, tuple[str, str]] = {
    **_BASE_ROUTING,
    "eval_data_path": ("task", "eval_data_path"),
    "max_eval_items": ("eval", "max_eval_items"),
    # jsonl_output_path is NOT routed into RunConfig — it's consumed
    # by the eval entrypoint directly and passed to run_eval().
}


def _apply_target(config: Any, info: TargetInfo) -> Any:
    """Apply parsed target fields into the backend sub-config.

    Only sets fields that the concrete backend class actually declares
    (e.g. ``model`` exists on ``GeminiApiBackendConfig`` but not on
    ``VertexMsrlBackendConfig``), so a target form that doesn't match
    the backend is silently a no-op for those fields — the caller is
    expected to have chosen a compatible backend.
    """
    backend = config.backend
    backend_fields = {f.name for f in dataclasses.fields(backend)}

    changes: dict[str, Any] = {}
    _MAPPING = {
        "project": info.project,
        "location": info.location,
        "vertex_existing_job_id": info.vertex_existing_job_id,
        "endpoint_id": info.endpoint_id,
        "model": info.model,
    }
    for field_name, value in _MAPPING.items():
        if value is not None and field_name in backend_fields:
            changes[field_name] = value

    if not changes:
        return config
    backend = dataclasses.replace(backend, **changes)
    return dataclasses.replace(config, backend=backend)


def apply_overrides(config: Any, overrides: _BaseLaunchOverrides) -> Any:
    """Return a copy of ``config`` with non-None overrides applied into
    the routed sub-configs via dataclasses.replace."""
    # Determine routing table from override class.
    if isinstance(overrides, TrainLaunchOverrides):
        routing = _TRAIN_ROUTING
    elif isinstance(overrides, EvalLaunchOverrides):
        routing = _EVAL_ROUTING
    else:
        routing = _BASE_ROUTING

    # 1:1 field routing.
    per_sub: dict[str, dict[str, Any]] = {}
    for ov_field, (sub_attr, tgt_field) in routing.items():
        val = getattr(overrides, ov_field, None)
        if val is None:
            continue
        per_sub.setdefault(sub_attr, {})[tgt_field] = val
    replacements: dict[str, Any] = {}
    for sub_attr, changes in per_sub.items():
        sub = getattr(config, sub_attr)
        replacements[sub_attr] = dataclasses.replace(sub, **changes)
    if replacements:
        config = dataclasses.replace(config, **replacements)

    # Target parsing (may override project/location set above — resource
    # path is more specific than the separate CLI args).
    if overrides.target is not None:
        config = _apply_target(config, parse_target(overrides.target))

    return config
