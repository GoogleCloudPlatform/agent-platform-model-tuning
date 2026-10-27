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

"""Tests for the word_puzzle RunConfig-module config model.

config_model exposes two shared recipe config modules, each an ABC subclass
with a ``make()`` that builds a live component:

* ``WordPuzzleRewardConfig(RewardConfig)`` -> ``RewardFn``
* ``WordPuzzleTaskConfig(TaskConfig)``     -> ``(train_ds, eval_ds)``

The agent config (``WordPuzzleAgentConfig``) lives in
``single_step/config_model.py`` and is tested here too for continuity.
"""

import os

import pytest

from core.agent_config import AgentConfig
from core.dataset import JsonlDataset
from core.reward import RewardConfig
from core.single_step_agent import SingleStepAgent
from core.task_config import TaskConfig
from example_recipe.word_puzzle.config_model import (
    WordPuzzleRewardConfig,
    WordPuzzleTaskConfig,
)
from example_recipe.word_puzzle.single_step.config_model import (
    WordPuzzleAgentConfig,
)
from example_recipe.word_puzzle.reward_signals import WordPuzzleRewardSignals


# ── RewardConfig ────────────────────────────────────────────────────────
def test_reward_is_reward_config_subclass():
    assert isinstance(WordPuzzleRewardConfig(), RewardConfig)


def test_reward_defaults():
    r = WordPuzzleRewardConfig()
    assert r.correct_answer_reward == 1.0
    assert r.malformed_output_reward == -4.0
    assert r.incorrect_answer_reward == -1.0


async def test_reward_make_builds_callable_reward_fn():
    reward_fn = WordPuzzleRewardConfig().make()
    signals = WordPuzzleRewardSignals(
        is_correct=True,
        generation_token_count=5,
        finish_reason=None,
        max_output_tokens=100,
    )
    reward, stats = await reward_fn(signals)
    assert reward == 1.0  # correct_answer_reward
    # Second element is a WordPuzzleStepStats-shaped object.
    assert stats.generation_token_count == 5
    assert stats.malformed_function_call is False


async def test_reward_make_honors_overridden_knobs():
    reward_fn = WordPuzzleRewardConfig(incorrect_answer_reward=-2.5).make()
    signals = WordPuzzleRewardSignals(
        is_correct=False,
        generation_token_count=3,
        finish_reason=None,
        max_output_tokens=100,
    )
    reward, _ = await reward_fn(signals)
    assert reward == -2.5


# ── TaskConfig ──────────────────────────────────────────────────────────
def test_task_is_task_config_subclass():
    assert isinstance(WordPuzzleTaskConfig(), TaskConfig)


def test_task_defaults():
    t = WordPuzzleTaskConfig()
    assert t.vertex_experiment == ""
    assert t.vertex_experiment_tensorboard == ""
    assert t.vertex_experiment_run_name == ""
    assert t.atif_output == ""
    assert t.atif_upload_concurrency == 32
    assert t.train_data_path.endswith("combined_training_word_transformation_v1.jsonl")
    assert t.eval_data_path.endswith("combined_eval_word_transformation_v1.jsonl")


def test_task_vertex_experiment_knobs_settable():
    t = WordPuzzleTaskConfig(
        vertex_experiment_tensorboard="disable",
        vertex_experiment_run_name="myrun",
    )
    assert t.vertex_experiment_tensorboard == "disable"
    assert t.vertex_experiment_run_name == "myrun"


def test_task_make_returns_two_jsonl_datasets():
    cfg = WordPuzzleTaskConfig()
    # make() loads eagerly, so it needs the generated splits on disk, and not
    # every checkout ships them -- some ship only dataset/*_gen.py.
    for p in (cfg.train_data_path, cfg.eval_data_path):
        if not os.path.exists(p):
            pytest.skip(f"word_puzzle splits not shipped in this checkout: {p}")
    train, eval_ = cfg.make()
    assert isinstance(train, JsonlDataset)
    assert isinstance(eval_, JsonlDataset)
    assert len(train) > 0
    assert len(eval_) > 0


# ── AgentConfig ─────────────────────────────────────────────────────────
def test_agent_is_agent_config_subclass():
    assert isinstance(WordPuzzleAgentConfig(), AgentConfig)


def test_agent_inherits_base_defaults():
    a = WordPuzzleAgentConfig()
    assert a.max_generation_tokens == 32768
    assert a.thinking_level == "HIGH"
    assert a.agent_rollout_timeout_seconds == 7200.0  # inherited AgentConfig default


def test_agent_make_builds_single_step_agent():
    fake_sampler = object()

    async def reward_fn(signals: WordPuzzleRewardSignals) -> tuple[float, None]:
        return (0.0, None)

    agent = WordPuzzleAgentConfig().make(
        sampler=fake_sampler,
        job_id="j",
        reward_fn=reward_fn,
    )
    assert isinstance(agent, SingleStepAgent)
    assert (
        agent.agent_rollout_timeout_seconds == 7200.0
    )  # inherited AgentConfig default
