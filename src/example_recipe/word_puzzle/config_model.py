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

"""word_puzzle recipe config modules shared across all agent variants.

Two plain frozen dataclasses, each an ABC subclass carrying the
recipe's knobs and a ``make()`` that builds a live component:

* :class:`WordPuzzleRewardConfig` (``RewardConfig``) -> ``RewardFn``
* :class:`WordPuzzleTaskConfig`   (``TaskConfig``)   -> ``(train, eval)`` datasets

Agent-variant configs live in their own sub-packages (e.g.
``single_step/config_model.py``,
``agent_solver/minisweagent/agent_solver_config.py``).

A ``RunConfig`` composes these (one module per config seam); the
entry points build the live components by calling ``.make()`` on each.
Generic vocabulary (src/); no framework-loop knobs live here anymore.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from core.dataset import JsonlDataset
from core.reward import RewardConfig, RewardFn
from core.task_config import TaskConfig
from example_recipe.word_puzzle import parse_word_puzzle_item
from example_recipe.word_puzzle.reward import WordPuzzleRewardConfig as _RewardKnobs
from example_recipe.word_puzzle.reward import WordPuzzleRewardFn

# Resolve default dataset paths relative to this module so runs launched
# from any CWD load correctly. Caller-supplied paths are used verbatim.
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")


@dataclass(frozen=True)
class WordPuzzleRewardConfig(RewardConfig):
    """Reward-shaping knobs. ``make()`` builds the recipe ``RewardFn``.

    Holds the same knobs the reward fn consumes; ``make()`` adapts them
    into the internal reward knob bag the ``WordPuzzleRewardFn`` takes.
    """

    malformed_output_reward: float = -4.0
    correct_answer_reward: float = 1.0
    incorrect_answer_reward: float = -1.0
    max_step_reward: float = -10.0
    # Step-efficiency bonus for CORRECT answers only; 0.0 = off. See
    # ``WordPuzzleRewardFn.step_bonus_for`` for the ramp between the two
    # step thresholds.
    step_bonus_max: float = 0.0
    full_bonus_at_steps: int = 8
    zero_bonus_at_steps: int = 24
    # Token-efficiency bonus for CORRECT answers only; 0.0 = off. See
    # ``WordPuzzleRewardFn.token_bonus_for`` for the LOG ramp between the
    # two token thresholds, and the bucketing that keeps the reward moving
    # in ~1 KiB steps.
    token_bonus_max: float = 0.0
    full_bonus_at_tokens: int = 8192
    zero_bonus_at_tokens: int = 131072
    token_bucket_size: int = 1024

    def make(self) -> RewardFn:
        return WordPuzzleRewardFn(
            _RewardKnobs(
                malformed_output_reward=self.malformed_output_reward,
                correct_answer_reward=self.correct_answer_reward,
                incorrect_answer_reward=self.incorrect_answer_reward,
                max_step_reward=self.max_step_reward,
                step_bonus_max=self.step_bonus_max,
                full_bonus_at_steps=self.full_bonus_at_steps,
                zero_bonus_at_steps=self.zero_bonus_at_steps,
                token_bonus_max=self.token_bonus_max,
                full_bonus_at_tokens=self.full_bonus_at_tokens,
                zero_bonus_at_tokens=self.zero_bonus_at_tokens,
                token_bucket_size=self.token_bucket_size,
            )
        )


@dataclass(frozen=True)
class WordPuzzleTaskConfig(TaskConfig):
    """Recipe-owned task inputs + trajectory-logging knobs.

    ``make()`` resolves the data paths into the loaded ``(train, eval)``
    dataset pair the pipeline/eval path consume.
    """

    train_data_path: str = os.path.join(
        _DATA_DIR, "combined_training_word_transformation_v1.jsonl"
    )
    eval_data_path: str = os.path.join(
        _DATA_DIR, "combined_eval_word_transformation_v1.jsonl"
    )
    # Shuffle the EVAL split (seeded, so the order is reproducible) before
    # ``max_eval_items`` truncates it. The splits are written in depth
    # order, so a capped sweep over an unshuffled split silently reads only
    # the shallowest rows; turn this on whenever ``max_eval_items`` is
    # smaller than the file and the sample is meant to be representative.
    eval_shuffle: bool = False
    # Recipe-agnostic logging knobs (atif_output, atif_upload_concurrency,
    # vertex_experiment*) are inherited from the base TaskConfig.

    def make(self) -> tuple[JsonlDataset, JsonlDataset]:
        train = JsonlDataset(
            file_path=self.train_data_path,
            transform_fn=parse_word_puzzle_item,
            shuffle=True,
        )
        eval_ = JsonlDataset(
            file_path=self.eval_data_path,
            transform_fn=parse_word_puzzle_item,
            shuffle=self.eval_shuffle,
        )
        return (train, eval_)

    def make_eval(self) -> JsonlDataset:
        """Load ONLY the eval dataset.

        The eval path never touches the train split, and datasets load
        eagerly, so eval-only runs must not require the train file to exist
        (it may be absent, e.g. when attaching to a checkpoint for a pure
        eval sweep). Eval orchestrators call this instead of make()[1].
        """
        return JsonlDataset(
            file_path=self.eval_data_path,
            transform_fn=parse_word_puzzle_item,
            shuffle=self.eval_shuffle,
        )
