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

"""Config + operating points for the Docker-free simple word-puzzle agent.

``SimpleAgentConfig.make()`` builds a :class:`WordPuzzleSimpleAgent` wired
to a :class:`WordPuzzleSimpleEnv` per rollout. No Docker dependency — bash
runs via ``asyncio.create_subprocess_shell`` in a tempdir.

``CONFIGS`` exposes ``async`` / ``smoke`` operating points that compose a
:class:`RunConfig` around this agent, reusing the shared word_puzzle
reward + task modules, plus the opt-in, eval-only ``eval_agent_format``
variant (``async`` with the agent-format eval set and the answer-only
submit contract).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any

from backends.backend_config import VertexMsrlBackendConfig
from core.agent_config import AgentConfig
from core.eval_config import EvalConfig
from core.run_config import RunConfig, TrainConfig
from pipeline.async_rollout_pipeline import AsyncRolloutConfig
from pipeline.sync_rollout_pipeline import SyncRolloutConfig

from example_recipe.word_puzzle.agent_solver.simple_agent.env import (
    WordPuzzleSimpleEnv,
)
from example_recipe.word_puzzle.agent_solver.simple_agent.simple_agent import (
    WordPuzzleSimpleAgent,
)
from example_recipe.word_puzzle.config_model import (
    WordPuzzleRewardConfig,
    WordPuzzleTaskConfig,
)
from example_recipe.word_puzzle.dataset.task_data import WordPuzzleTaskData

# Dataset path: simple_agent/ -> agent_solver/ -> word_puzzle/ -> dataset/
_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "dataset",
)
_TRAIN_JSONL = os.path.join(_DATA_DIR, "combined_training_word_transformation_v1.jsonl")
_EVAL_JSONL = os.path.join(_DATA_DIR, "combined_eval_word_transformation_v1.jsonl")
# Opt-in eval set for the ``eval_agent_format`` variant. The submit tool is
# graded by exact match on its `answer` argument, but the single-step v1
# prompts above tell the model to answer as $\text{...}$; the *_agent_*
# datasets omit that instruction (same set as the minisweagent solver). There
# is no agent-format training set in this tree yet.
_AGENT_EVAL_JSONL = os.path.join(
    _DATA_DIR, "combined_eval_word_transformation_withhold_agent_v1.jsonl"
)


@dataclass(frozen=True)
class SimpleAgentConfig(AgentConfig):
    """Agent module for the Docker-free word-puzzle simple agent.

    Inherits ``thinking_level`` / ``agent_rollout_timeout_seconds`` /
    ``max_generation_tokens`` from ``AgentConfig``. Adds the step limit
    and bash timeout that shape per-rollout behavior.
    """

    step_limit: int = 10
    bash_timeout_seconds: int = 30
    # State the submit contract ("answer only, graded by exact match") in the
    # submit tool and instance prompt. Off by default so existing runs keep
    # their prompts; on in the ``eval_agent_format`` variant, whose dataset
    # prompts ask the model to explain its reasoning.
    answer_only_submit: bool = False

    def make(
        self,
        *,
        sampler: Any,
        job_id: Any,
        reward_fn: Any = None,
        atif_writer: Any = None,
        **kwargs: Any,
    ) -> WordPuzzleSimpleAgent:
        def env_factory(task_data: WordPuzzleTaskData) -> WordPuzzleSimpleEnv:
            return WordPuzzleSimpleEnv(
                task_data=task_data,
                reward_fn=reward_fn,
                step_limit=self.step_limit,
                bash_timeout=self.bash_timeout_seconds,
                answer_only_submit=self.answer_only_submit,
            )

        return WordPuzzleSimpleAgent(
            sampler=sampler,
            env_factory=env_factory,
            max_steps=self.step_limit,
            recipe_name="word_puzzle_simple",
            tuning_job_id=str(job_id) if job_id else "",
            agent_rollout_timeout_seconds=self.agent_rollout_timeout_seconds,
            trajectory_callback=atif_writer,
        )


_SHARED: dict[str, Any] = dict(
    agent=SimpleAgentConfig(),
    reward=WordPuzzleRewardConfig(),
    backend=VertexMsrlBackendConfig(),
    task=WordPuzzleTaskConfig(
        train_data_path=_TRAIN_JSONL,
        eval_data_path=_EVAL_JSONL,
    ),
)

CONFIGS: dict[str, RunConfig] = {
    "async": RunConfig(
        train=TrainConfig(sync_every_n_train_steps=4),
        rollout=AsyncRolloutConfig(max_pending_rollout_tasks=32 * 4),
        eval=EvalConfig(eval_every_n_train_steps=10),
        **_SHARED,
    ),
    "smoke": RunConfig(
        train=TrainConfig(
            num_train_steps=3,
            tasks_per_train_step=4,
            sync_every_n_train_steps=1,
        ),
        rollout=SyncRolloutConfig(),
        eval=EvalConfig(eval_every_n_train_steps=0),
        **_SHARED,
    ),
}

# Opt-in: agent-format eval set + the answer-only submit contract. Derived from
# ``async`` via ``replace`` so its train/rollout/eval/reward/backend config
# cannot drift; the defaults above are unchanged.
#
# Intended for EVAL ONLY (the ``eval`` entry point). Its train set is still
# the v1 set, whose prompts demand a ``$\text{...}$`` answer, contradicting the
# answer-only submit tool text; training with this variant would mix those
# contradictory prompt formats. Do not train with it until an agent-format
# training set exists.
CONFIGS["eval_agent_format"] = replace(
    CONFIGS["async"],
    agent=replace(CONFIGS["async"].agent, answer_only_submit=True),
    task=replace(CONFIGS["async"].task, eval_data_path=_AGENT_EVAL_JSONL),
)
