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

"""WordPuzzleSimpleAgent — thin MultiStepAgent subclass for word puzzles.

Docker-free multi-turn agent using bash+submit tools. Inherits the
complete multi-step loop from ``MultiStepAgent``; only provides the
recipe-specific ``extract_prompt_text`` (reads
``WordPuzzleTaskData.question``).
"""

from __future__ import annotations

import logging
from typing import Callable

from core.multi_step_agent import MultiStepAgent
from core.protocols import Env
from example_recipe.word_puzzle.dataset.task_data import WordPuzzleTaskData

logger = logging.getLogger(__name__)


class WordPuzzleSimpleAgent(MultiStepAgent[WordPuzzleTaskData]):
    """Multi-turn word-puzzle agent using bash+submit tools, no Docker."""

    def __init__(
        self,
        sampler,
        env_factory: Callable[[WordPuzzleTaskData], Env],
        *,
        max_steps: int = 10,
        recipe_name: str = "word_puzzle_simple",
        tuning_job_id: str = "",
        agent_rollout_timeout_seconds: float | None = None,
        metric_aggregators=None,
        trajectory_callback=None,
    ) -> None:
        super().__init__(
            sampler=sampler,
            env_factory=env_factory,
            max_steps=max_steps,
            recipe_name=recipe_name,
            tuning_job_id=tuning_job_id,
            extract_prompt_text=lambda td: td.question,
            agent_rollout_timeout_seconds=agent_rollout_timeout_seconds,
            metric_aggregators=metric_aggregators,
            trajectory_callback=trajectory_callback,
        )
