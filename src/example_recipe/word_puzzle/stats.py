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

"""Recipe-owned typed per-step stats for word_puzzle.

Attached to ``TrajectoryStep.task_stats`` by ``WordPuzzleEnv`` and
consumed by the recipe reward function.

Pure dataclass module: no runtime deps on the rest of the recipe or the
framework.
"""

import dataclasses


@dataclasses.dataclass
class WordPuzzleStepStats:
    """Per-step stats emitted by the word_puzzle env.

    Attributes:
        generation_token_count: Env-side view of the generation length.
        malformed_function_call: True if the model's action could not be
            parsed as a tool call and the env short-circuited.
        submit_retries_used: Number of graded-retry submits consumed
            before the terminal submit (agent-solver retry path); 0 for
            recipes without submit retries.
        notes_used: Number of no-op "thinking" tool calls the rollout
            spent before the terminal submit; 0 for recipes without the
            note tool.
        solve_step_count: Number of agent steps the rollout took to reach
            its terminal submit; 0 when the env reports no step budget.
        step_bonus: Step-efficiency bonus actually added to the reward
            (0.0 when the bonus is disabled or the answer was wrong).
        token_bonus: Token-efficiency bonus actually added to the reward
            (0.0 when the bonus is disabled or the answer was wrong).
        rollout_generation_token_count: Every generation token the rollout
            was billed for -- its actions plus the context-compaction
            summaries it caused -- stamped by multi-step envs; 0 for envs
            the agent never reports a total to. Unlike
            ``generation_token_count`` (terminal step only) this is the
            quantity the token bonus is computed from.
    """

    generation_token_count: int = 0
    malformed_function_call: bool = False
    submit_retries_used: int = 0
    notes_used: int = 0
    solve_step_count: int = 0
    step_bonus: float = 0.0
    token_bonus: float = 0.0
    rollout_generation_token_count: int = 0
