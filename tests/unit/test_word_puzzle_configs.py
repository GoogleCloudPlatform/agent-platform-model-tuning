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

"""Registered word_puzzle operating-point RunConfig instances (D2).

Importing the two config modules populates the two-level config registry,
after which ``get`` resolves each operating point by ``family/variant``
path. The pinned numbers are behavior-preserving copies of the A-era
async/sync tunings.
"""

from __future__ import annotations

from backends.backend_config import VertexMsrlBackendConfig
from core.eval_config import EvalConfig
from config_registry import get_config
from pipeline.async_rollout_pipeline import AsyncRolloutConfig
from pipeline.sync_rollout_pipeline import SyncRolloutConfig
from core.run_config import RunConfig, TrainConfig


def test_word_puzzle_async_operating_point():
    cfg = get_config("word_puzzle/async")
    assert isinstance(cfg, RunConfig)
    assert isinstance(cfg.train, TrainConfig)
    assert isinstance(cfg.rollout, AsyncRolloutConfig)
    assert isinstance(cfg.eval, EvalConfig)
    assert isinstance(cfg.backend, VertexMsrlBackendConfig)
    # Async tunings (behavior-preserving from A-era config_model.py):
    assert cfg.train.sync_every_n_train_steps == 4
    assert cfg.rollout.max_pending_rollout_tasks == 128  # 32 * 4
    assert cfg.eval.eval_every_n_train_steps == 10
    # Framework defaults left implicit:
    assert cfg.rollout.samples_per_task == 16


def test_word_puzzle_sync_operating_point():
    cfg = get_config("word_puzzle/sync")
    assert isinstance(cfg, RunConfig)
    assert isinstance(cfg.rollout, SyncRolloutConfig)
    assert isinstance(cfg.backend, VertexMsrlBackendConfig)
    # Sync tunings (behavior-preserving from A-era config_model.py):
    assert cfg.train.sync_every_n_train_steps == 1
    assert cfg.rollout.max_concurrent_samples == 1024
    assert cfg.eval.eval_samples_per_task == 5
    assert cfg.eval.eval_every_n_train_steps == 5
    assert cfg.eval.max_eval_items == -1
