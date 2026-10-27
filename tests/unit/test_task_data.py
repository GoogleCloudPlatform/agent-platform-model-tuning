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

"""Unit tests for the core TaskData base dataclass."""

import dataclasses

import pytest

from core.task_data import TaskData


def test_task_data_requires_task_id():
    """task_id is a required field (no default)."""
    with pytest.raises(TypeError):
        TaskData()  # type: ignore[call-arg]


def test_task_data_accepts_explicit_task_id():
    td = TaskData(task_id="abc-123")
    assert td.task_id == "abc-123"


def test_task_data_is_a_dataclass():
    # Downstream eval tooling will asdict(td) for JSONL. Confirm
    # the base is dataclasses.is_dataclass-detectable.
    assert dataclasses.is_dataclass(TaskData)
    assert dataclasses.is_dataclass(TaskData(task_id="t"))


def test_task_data_subclass_inherits_required_task_id():
    @dataclasses.dataclass
    class _SubTD(TaskData):
        extra: str = ""

    sub = _SubTD(task_id="t1", extra="hello")
    assert sub.task_id == "t1"
    assert sub.extra == "hello"
    assert dataclasses.is_dataclass(sub)


def test_task_data_asdict_roundtrips_via_json():
    # JSONL serialization path used by the eval tooling. Any dataclass
    # subclass MUST survive asdict -> json.dumps -> json.loads round-trip.
    import json

    td = TaskData(task_id="abc-123")
    payload = json.dumps(dataclasses.asdict(td))
    assert json.loads(payload) == {"task_id": "abc-123"}
