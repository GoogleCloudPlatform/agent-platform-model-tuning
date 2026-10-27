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

"""Tests for pipeline.eval_serializer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pipeline.eval_serializer import serialize_group, write_eval_jsonl


class _FakeTaskData:
    def __init__(self, task_id: str, instance_id: str | None = None):
        self.task_id = task_id
        if instance_id is not None:
            self.instance_id = instance_id


def test_serialize_group_basic():
    td = _FakeTaskData("t1")
    metrics = {"acc": 0.5, "reward": 1.0}
    row = serialize_group(td, metrics)
    assert row["task_data"]["task_id"] == "t1"
    assert row["metrics"] == {"acc": 0.5, "reward": 1.0}
    assert row["trajectories"] == []


def test_serialize_group_with_instance_id():
    td = _FakeTaskData("t1", instance_id="inst-42")
    row = serialize_group(td, {})
    assert row["task_data"]["instance_id"] == "inst-42"


def test_serialize_group_none_task_data():
    row = serialize_group(None, {"k": 1.0})
    assert row["task_data"] == {}
    assert row["metrics"] == {"k": 1.0}


def test_serialize_group_bare_scalars_pass_through():
    metrics: dict[str, Any] = {"a": 1, "b": 2.5, "c": True, "d": "ok", "e": None}
    row = serialize_group(None, metrics)
    assert row["metrics"] == metrics


def test_write_eval_jsonl(tmp_path: Path):
    out = tmp_path / "eval.jsonl"
    groups = [
        (_FakeTaskData("t1"), {"acc": 0.5}),
        (_FakeTaskData("t2"), {"acc": 1.0}),
    ]
    write_eval_jsonl(out, groups)
    lines = out.read_text().strip().split("\n")
    assert len(lines) == 2
    r0 = json.loads(lines[0])
    assert r0["task_data"]["task_id"] == "t1"
    assert r0["metrics"]["acc"] == 0.5
    r1 = json.loads(lines[1])
    assert r1["task_data"]["task_id"] == "t2"
    assert r1["metrics"]["acc"] == 1.0


def test_write_eval_jsonl_creates_parent_dirs(tmp_path: Path):
    out = tmp_path / "deep" / "nested" / "eval.jsonl"
    write_eval_jsonl(out, [(_FakeTaskData("t1"), {"x": 1.0})])
    assert out.exists()
    assert len(out.read_text().strip().split("\n")) == 1
