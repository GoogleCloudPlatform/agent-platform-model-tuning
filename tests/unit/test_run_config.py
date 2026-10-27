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

"""Tests for RunConfig + TrainConfig (the composed config skeleton)."""

from __future__ import annotations

import pytest

from core.run_config import RunConfig, TrainConfig


def test_train_config_defaults():
    t = TrainConfig()
    assert t.num_train_steps == 100
    assert t.sync_every_n_train_steps == 8
    assert t.resume_from_train_step == 0
    assert t.log_dir == "logs"


def test_train_config_validates_on_construction():
    with pytest.raises(ValueError):
        TrainConfig(sync_every_n_train_steps=0)
    with pytest.raises(ValueError):
        TrainConfig(num_train_steps=-1)
    # A valid construction does not raise.
    TrainConfig()


def test_train_config_reward_normalizer_defaults_to_centering_config():
    from core.reward_normalizer_config import CenteringNormalizerConfig

    t = TrainConfig()
    assert isinstance(t.reward_normalizer, CenteringNormalizerConfig)


def test_train_config_accepts_grpo_normalizer_config():
    from core.reward_normalizer_config import GRPONormalizerConfig

    t = TrainConfig(reward_normalizer=GRPONormalizerConfig())
    assert isinstance(t.reward_normalizer, GRPONormalizerConfig)


def test_run_config_has_expected_fields():
    assert set(RunConfig.__dataclass_fields__) == {
        "train",
        "rollout",
        "eval",
        "agent",
        "reward",
        "backend",
        "task",
    }
