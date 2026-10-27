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

"""Single-step word_puzzle agent config module.

:class:`WordPuzzleAgentConfig` is the ``AgentConfig`` subclass for the
single-step (one model call per rollout) word_puzzle variant.  Its
``make()`` wires ``WordPuzzleEnv`` + ``SingleStepAgent`` into a runnable
agent.

Shared reward/task configs stay in the parent
``example_recipe.word_puzzle.config_model``; this module only owns the
agent-construction knob.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.agent_config import AgentConfig
from core.reward import RewardFn
from core.single_step_agent import SingleStepAgent
from example_recipe.word_puzzle import WordPuzzleEnv
from example_recipe.word_puzzle.dataset.task_data import WordPuzzleTaskData
from example_recipe.word_puzzle.env import (
    word_puzzle_extract_prompt_text,
)


@dataclass(frozen=True)
class WordPuzzleAgentConfig(AgentConfig):
    """Agent module for word_puzzle.

    Inherits the per-call generation knobs from ``AgentConfig``
    (``thinking_level``, ``agent_rollout_timeout_seconds``). ``make()`` wires the env
    factory + ATIF adapter into a ``SingleStepAgent``. The ``reward_fn`` is
    supplied by the caller (built from ``RunConfig.reward.make()``), so this
    module needs no knowledge of the reward config.
    """

    # Output-token budget handed to ``WordPuzzleEnv`` as ``max_output_tokens``
    # for length-aware reward shaping. This is a REWARD signal, not the
    # per-call generation cap (that is threaded through ``sample()``; see
    # ``AgentConfig``'s module docstring). Kept as an own field because the
    # base ``AgentConfig`` no longer carries it.
    max_generation_tokens: int = 32768

    def make(
        self,
        *,
        sampler,
        job_id,
        reward_fn: RewardFn,
        atif_writer=None,
    ) -> SingleStepAgent:
        def env_factory(task_data: WordPuzzleTaskData) -> WordPuzzleEnv:
            return WordPuzzleEnv(
                task_data=task_data,
                reward_fn=reward_fn,
                max_output_tokens=self.max_generation_tokens,
            )

        return SingleStepAgent[WordPuzzleTaskData](
            sampler,
            env_factory,
            recipe_name="word_puzzle",
            tuning_job_id=job_id,
            extract_prompt_text=word_puzzle_extract_prompt_text,
            agent_name="word-puzzle-rl",
            agent_rollout_timeout_seconds=self.agent_rollout_timeout_seconds,
            trajectory_callback=atif_writer,
        )
