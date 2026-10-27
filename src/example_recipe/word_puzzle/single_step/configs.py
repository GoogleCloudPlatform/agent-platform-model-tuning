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

"""Word-puzzle operating points: all variants in one dict.

Each variant composes a :class:`RunConfig` with different train/rollout/eval
knobs but shares the same agent, reward, backend, and task configs.

Adding a new variant = one dict entry here. The registry resolves
``word_puzzle/<variant>`` by importing this module and looking up
``CONFIGS[variant]``.
"""

from __future__ import annotations

from backends.backend_config import VertexMsrlBackendConfig
from core.eval_config import EvalConfig
from pipeline.async_rollout_pipeline import AsyncRolloutConfig
from pipeline.sync_rollout_pipeline import SyncRolloutConfig
from core.run_config import RunConfig, TrainConfig
from example_recipe.word_puzzle.config_model import (
    WordPuzzleRewardConfig,
    WordPuzzleTaskConfig,
)
from example_recipe.word_puzzle.single_step.config_model import (
    WordPuzzleAgentConfig,
)

_SHARED = dict(
    agent=WordPuzzleAgentConfig(),
    reward=WordPuzzleRewardConfig(),
    backend=VertexMsrlBackendConfig(),
    task=WordPuzzleTaskConfig(),
)

CONFIGS: dict[str, RunConfig] = {
    # Tighter weight-sync cadence than the framework default (8) to stay
    # closer to on-policy for the async loop.  Look-ahead window narrowed
    # (32 * 4 = 128).  Full eval sweep every 10 grad steps.
    "async": RunConfig(
        train=TrainConfig(sync_every_n_train_steps=4),
        rollout=AsyncRolloutConfig(max_pending_rollout_tasks=32 * 4),
        eval=EvalConfig(eval_every_n_train_steps=10),
        **_SHARED,
    ),
    # Sync loop: on-policy by construction, weight-sync every step.
    "sync": RunConfig(
        train=TrainConfig(sync_every_n_train_steps=1),
        rollout=SyncRolloutConfig(),
        eval=EvalConfig(
            eval_samples_per_task=5,
            eval_every_n_train_steps=5,
        ),
        **_SHARED,
    ),
}
