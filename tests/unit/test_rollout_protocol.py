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

import dataclasses

import pytest

from core.rollout_protocol import BatchItem, RolloutPipeline


def test_batchitem_is_dataclass_with_expected_fields():
    assert dataclasses.is_dataclass(BatchItem)
    fields = {f.name for f in dataclasses.fields(BatchItem)}
    assert "group" in fields and "metrics" in fields


def test_batchitem_is_frozen():
    item = BatchItem(group=object(), metrics={})
    with pytest.raises(dataclasses.FrozenInstanceError):
        item.group = None  # type: ignore[misc]


def test_rollout_pipeline_runtime_check_enforces_shape():
    class Good:
        async def run(self): ...
        def stop(self): ...
        async def next_batch(self, n, *, step_num=None): ...
        def sample_pipeline_gauges(self): ...
        def schedule_eval(self, step_val): ...
        def schedule_checkpoint_eval(self, step_val, endpoint): ...
        def schedule_base_model_eval(self, step_val=0): ...

    assert isinstance(Good(), RolloutPipeline)
    assert not isinstance(object(), RolloutPipeline)
