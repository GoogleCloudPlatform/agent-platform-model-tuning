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

"""EvalConfig: eval-sweep knobs, validated on construction via __post_init__."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalConfig:
    eval_samples_per_task: int = 5
    eval_every_n_train_steps: int = 1
    max_eval_items: int = -1
    # When False, the live-sampler eval sweep (``eval/*``) is skipped for
    # the whole training run -- the step-0 baseline, the cadence sweeps,
    # and the final-step sweep. Useful when only the checkpoint-endpoint
    # sweep (``eval_ckpt/*``) is wanted, so eval rollouts stop competing
    # with training for the shared concurrency budget.
    run_sampler_eval: bool = True
    # Opt-in: when True, and the backend surfaces a deployed checkpoint
    # endpoint in the TrainStep response, run an additional inline eval
    # sweep against that frozen endpoint (logged under eval_ckpt/*).
    run_checkpoint_endpoint_eval: bool = False
    # Per-call generation-token cap for the checkpoint-endpoint sweep only
    # (the live-sampler eval keeps config.agent.max_generation_tokens).
    checkpoint_eval_max_generation_tokens: int = 65535

    def __post_init__(self) -> None:
        if self.eval_samples_per_task < 1:
            raise ValueError("eval_samples_per_task must be >= 1")
