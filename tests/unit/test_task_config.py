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

"""Tests for TaskConfig ABC."""

from __future__ import annotations

import pytest

from core.task_config import TaskConfig


def test_base_is_abstract():
    with pytest.raises(TypeError):
        TaskConfig()  # abstract: make() not implemented


def test_concrete_subclass_make_returns_datasets():
    class _FakeTaskConfig(TaskConfig):
        def make(self):
            return ("train_ds", "eval_ds")

    assert _FakeTaskConfig().make() == ("train_ds", "eval_ds")
