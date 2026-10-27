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

"""Shared JSONL serialization for eval output.

Each JSONL row is one eval group (one task × N rollouts). The row
carries three keys:

- ``task_data``: ``{"task_id": ..., ...}`` (recipe-agnostic subset).
- ``metrics``: per-group metrics dict (bare scalars or MetricValue
  JSON shapes — the downstream ``_extract_metric_values`` in the
  smoke harness handles both).
- ``trajectories``: list of per-rollout dicts (reward, step count,
  rollout status). Intentionally shallow — the full per-step details
  live in the ATIF disk artifacts, not the JSONL.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _metric_value_to_json(v: Any) -> Any:
    """Unwrap a ``MetricValue`` (or bare scalar) into a JSON-safe form.

    Bare int/float/bool/str/None pass through unchanged.  ``MetricValue``
    instances are converted to a small dict matching the shape
    ``_scalar_from_json_metric`` in the smoke harness can parse.
    """
    if isinstance(v, (int, float, bool, str, type(None))):
        return v
    # MetricValue has exactly one populated field.
    if dataclasses.is_dataclass(v):
        d = dataclasses.asdict(v)
        # Drop None-valued fields for compactness.
        return {k: val for k, val in d.items() if val is not None}
    return v


def serialize_group(
    task_data: Any,
    metrics: dict[str, Any],
    scored_trajectories: Any = None,
) -> dict[str, Any]:
    """Serialize one eval group into a JSONL-ready dict.

    ``task_data`` is the recipe's ``TaskData`` (any object with a
    ``task_id`` attribute).  ``metrics`` is the per-group metric dict
    from ``agent.group_rollout``.  ``scored_trajectories`` is the list
    of ``ScoredTrajectory`` objects (optional — omitted when the caller
    only has the metrics).
    """
    td: dict[str, Any] = {}
    if task_data is not None:
        td["task_id"] = getattr(task_data, "task_id", None)
        instance_id = getattr(task_data, "instance_id", None)
        if instance_id is not None:
            td["instance_id"] = instance_id

    trajectories: list[dict[str, Any]] = []
    if scored_trajectories is not None:
        for gt in scored_trajectories:
            traj_dict: dict[str, Any] = {
                "reward": gt.reward.reward if gt.reward else None,
                "num_steps": len(
                    [s for s in gt.trajectory.steps if s.source == "agent"]
                ),
            }
            trajectories.append(traj_dict)

    return {
        "task_data": td,
        "metrics": {k: _metric_value_to_json(v) for k, v in metrics.items()},
        "trajectories": trajectories,
    }


def write_eval_jsonl(
    path: Path,
    groups: list[tuple[Any, dict[str, Any]]],
) -> None:
    """Write eval groups to a JSONL file (one line per group).

    ``groups`` is a list of ``(task_data, metrics)`` tuples — the
    per-group output from ``EvalPipeline.run``'s ``on_group`` callback.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for task_data, metrics in groups:
            row = serialize_group(task_data, metrics)
            f.write(json.dumps(row) + "\n")
    logger.info("Wrote %d eval groups to %s", len(groups), path)
