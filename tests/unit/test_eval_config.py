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

"""Tests for EvalConfig construction-time validation."""

from __future__ import annotations

import pytest

from core.eval_config import EvalConfig


def test_eval_config_validates_on_construction():
    with pytest.raises(ValueError):
        EvalConfig(eval_samples_per_task=0)
    # A valid construction does not raise.
    EvalConfig()


def test_eval_config_defaults():
    e = EvalConfig()
    assert e.eval_samples_per_task == 5
    assert e.eval_every_n_train_steps == 1
    assert e.max_eval_items == -1


def test_checkpoint_eval_defaults_off():
    cfg = EvalConfig()
    assert cfg.run_checkpoint_endpoint_eval is False
    assert cfg.checkpoint_eval_max_generation_tokens == 65535


def test_checkpoint_eval_fields_settable():
    cfg = EvalConfig(
        run_checkpoint_endpoint_eval=True,
        checkpoint_eval_max_generation_tokens=32768,
    )
    assert cfg.run_checkpoint_endpoint_eval is True
    assert cfg.checkpoint_eval_max_generation_tokens == 32768
