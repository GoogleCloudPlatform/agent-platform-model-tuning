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

"""RunConfig: the single framework config type (never subclassed).

A run is described by ONE ``RunConfig`` composing 7 typed modules. Each operating
point is a registered ``RunConfig`` *instance*; recipe-ness is carried by the
field instances, not by subclassing.

Behavior modules (``rollout``/``agent``/``reward``/``backend``/``task``) expose a
``make()`` that builds a runtime object. Pure-data modules (``train``/``eval``)
carry no ``make()``; they validate their invariants in ``__post_init__`` and are
read field-by-field. ``TrainConfig`` (the training schedule) lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from backends.backend_config import BackendConfig
from core.agent_config import AgentConfig
from core.rollout_protocol import RolloutConfig
from core.eval_config import EvalConfig
from core.reward import RewardConfig
from core.task_config import TaskConfig
from core.reward_normalizer_config import (
    RewardNormalizerConfig,
    CenteringNormalizerConfig,
)


@dataclass(frozen=True)
class TrainConfig:
    num_train_steps: int = 100
    tasks_per_train_step: int = 32
    sync_every_n_train_steps: int = 8
    reward_normalizer: RewardNormalizerConfig = field(
        default_factory=CenteringNormalizerConfig
    )
    resume_from_train_step: int = 0  # CLI
    resume_metrics_path: str = ""  # CLI
    log_dir: str = "logs"  # CLI

    def __post_init__(self) -> None:
        # frozen=True: __post_init__ may read + raise, but must not assign.
        if self.sync_every_n_train_steps < 1:
            raise ValueError("sync_every_n_train_steps must be >= 1")
        if self.num_train_steps < 0:
            raise ValueError("num_train_steps must be >= 0")


@dataclass(frozen=True)
class RunConfig:
    train: TrainConfig  # pure data; validated in __post_init__
    rollout: RolloutConfig  # ABC; Async/Sync subclass; make() -> RolloutPipeline
    eval: EvalConfig  # pure data; validated in __post_init__
    agent: AgentConfig  # ABC; recipe subclass; make() -> Agent
    reward: RewardConfig  # ABC; recipe subclass; make() -> RewardFn
    backend: BackendConfig  # ABC; per-backend subclass; make() -> BackendBundle
    task: TaskConfig  # ABC; recipe subclass; make() -> datasets
