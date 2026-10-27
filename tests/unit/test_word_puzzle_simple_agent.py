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

"""Tests for WordPuzzleSimpleAgent with the real WordPuzzleSimpleEnv.

Uses a mock sampler returning function_call Content protos; the env
runs real bash and submit handlers (no Docker).
"""

from __future__ import annotations

import asyncio
import math
from typing import TYPE_CHECKING, Any

import pytest

from core import atif_extras, protocols
from core.rollout_status import RolloutStatus
from example_recipe.word_puzzle.agent_solver.simple_agent.simple_agent import (
    WordPuzzleSimpleAgent,
)
from example_recipe.word_puzzle.agent_solver.simple_agent.env import (
    WordPuzzleSimpleEnv,
)
from example_recipe.word_puzzle.agent_solver.simple_agent.prompt import (
    TOOLS_ANSWER_ONLY,
)
from example_recipe.word_puzzle.dataset.task_data import WordPuzzleTaskData
from example_recipe.word_puzzle.reward import WordPuzzleRewardConfig, WordPuzzleRewardFn
from tests.unit._sample_result_factories import make_sample_result

if TYPE_CHECKING:
    from example_recipe.word_puzzle.agent_solver.simple_agent.config_model import (
        SimpleAgentConfig,
    )
    from example_recipe.word_puzzle.config_model import WordPuzzleTaskConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reward_fn() -> WordPuzzleRewardFn:
    return WordPuzzleRewardFn(WordPuzzleRewardConfig())


def _task_data(
    question: str = "What is 2+2?", reference: str = "4"
) -> WordPuzzleTaskData:
    return WordPuzzleTaskData(task_id="t-1", question=question, reference=reference)


def _fc_content(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Build a Content dict with a single function_call part.

    Uses camelCase ``functionCall`` because ``Content.from_json`` (called
    by ``make_sample_result``) speaks standard proto-JSON.
    """
    return {
        "role": "model",
        "parts": [{"functionCall": {"name": name, "args": args}}],
    }


def _submit_result(answer: str) -> Any:
    """SampleResult whose content is a submit(answer=...) function_call."""
    return make_sample_result(
        content=_fc_content("submit", {"answer": answer}),
        finish_reason="STOP",
        candidate_id="cand-submit",
        generation_token_count=10,
    )


def _bash_result(command: str) -> Any:
    """SampleResult whose content is a bash(command=...) function_call."""
    return make_sample_result(
        content=_fc_content("bash", {"command": command}),
        finish_reason="STOP",
        candidate_id="cand-bash",
        generation_token_count=5,
    )


class _SequenceSampler(protocols.Sampler):
    """Sampler that returns a pre-set sequence of SampleResults."""

    def __init__(self, results: list[Any]):
        self._queue = list(results)
        self.model_name = "test-model"

    async def sample_n(
        self,
        prompt,
        num_samples=1,
        **kwargs,
    ):
        if not self._queue:
            return []
        return [self._queue.pop(0)]


def _env_factory(reward_fn: WordPuzzleRewardFn | None = None):
    fn = reward_fn or _reward_fn()

    def factory(td: WordPuzzleTaskData) -> WordPuzzleSimpleEnv:
        return WordPuzzleSimpleEnv(
            task_data=td,
            reward_fn=fn,
            step_limit=10,
            bash_timeout=5,
        )

    return factory


def _run(agent: WordPuzzleSimpleAgent, td: WordPuzzleTaskData | None = None):
    """Run a single rollout and return the ScoredTrajectory."""
    td = td or _task_data()
    return asyncio.run(agent.rollout(task_data=td, rollout_idx=0))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDirectSubmitCorrect:
    """Sampler returns submit('4') for '2+2?' — correct answer."""

    def test_reward_positive(self):
        sampler = _SequenceSampler([_submit_result("4")])
        agent = WordPuzzleSimpleAgent(
            sampler=sampler,
            env_factory=_env_factory(),
            max_steps=10,
        )
        scored = _run(agent)

        assert scored.reward.reward > 0, (
            f"Expected positive reward for correct answer, got {scored.reward.reward}"
        )

    def test_status_ok(self):
        sampler = _SequenceSampler([_submit_result("4")])
        agent = WordPuzzleSimpleAgent(
            sampler=sampler,
            env_factory=_env_factory(),
            max_steps=10,
        )
        scored = _run(agent)
        status = atif_extras.get_rollout_status(scored.trajectory)
        assert status == RolloutStatus.OK

    def test_one_agent_step(self):
        sampler = _SequenceSampler([_submit_result("4")])
        agent = WordPuzzleSimpleAgent(
            sampler=sampler,
            env_factory=_env_factory(),
            max_steps=10,
        )
        scored = _run(agent)
        agent_steps = [s for s in scored.trajectory.steps if s.source == "agent"]
        assert len(agent_steps) == 1


class TestBashThenSubmit:
    """Sampler returns bash('echo 4') then submit('4')."""

    def test_two_agent_steps(self):
        sampler = _SequenceSampler([_bash_result("echo 4"), _submit_result("4")])
        agent = WordPuzzleSimpleAgent(
            sampler=sampler,
            env_factory=_env_factory(),
            max_steps=10,
        )
        scored = _run(agent)
        agent_steps = [s for s in scored.trajectory.steps if s.source == "agent"]
        assert len(agent_steps) == 2

    def test_reward_positive(self):
        sampler = _SequenceSampler([_bash_result("echo 4"), _submit_result("4")])
        agent = WordPuzzleSimpleAgent(
            sampler=sampler,
            env_factory=_env_factory(),
            max_steps=10,
        )
        scored = _run(agent)
        assert scored.reward.reward > 0

    def test_status_ok(self):
        sampler = _SequenceSampler([_bash_result("echo 4"), _submit_result("4")])
        agent = WordPuzzleSimpleAgent(
            sampler=sampler,
            env_factory=_env_factory(),
            max_steps=10,
        )
        scored = _run(agent)
        status = atif_extras.get_rollout_status(scored.trajectory)
        assert status == RolloutStatus.OK


class TestMaxStepsExhausted:
    """Sampler always returns bash — never submits; max_steps hit."""

    def test_lowest_reward(self):
        # Always bash, never submit → max steps exhausted
        sampler = _SequenceSampler([_bash_result("echo x")] * 3)
        agent = WordPuzzleSimpleAgent(
            sampler=sampler,
            env_factory=_env_factory(),
            max_steps=3,
        )
        scored = _run(agent)
        # Default lowest_reward is -1.0
        assert scored.reward.reward == -1.0

    def test_status_max_step_limit(self):
        sampler = _SequenceSampler([_bash_result("echo x")] * 3)
        agent = WordPuzzleSimpleAgent(
            sampler=sampler,
            env_factory=_env_factory(),
            max_steps=3,
        )
        scored = _run(agent)
        status = atif_extras.get_rollout_status(scored.trajectory)
        assert status == RolloutStatus.MAX_STEP_LIMIT

    def test_three_agent_steps(self):
        sampler = _SequenceSampler([_bash_result("echo x")] * 3)
        agent = WordPuzzleSimpleAgent(
            sampler=sampler,
            env_factory=_env_factory(),
            max_steps=3,
        )
        scored = _run(agent)
        agent_steps = [s for s in scored.trajectory.steps if s.source == "agent"]
        assert len(agent_steps) == 3


class TestIncorrectSubmit:
    """Sampler submits a wrong answer."""

    def test_reward_negative(self):
        sampler = _SequenceSampler([_submit_result("wrong")])
        agent = WordPuzzleSimpleAgent(
            sampler=sampler,
            env_factory=_env_factory(),
            max_steps=10,
        )
        scored = _run(agent)
        assert scored.reward.reward < 0


class TestConfigMake:
    def test_atif_writer_is_wired_as_trajectory_callback(self):
        # Regression: make() accepted atif_writer but dropped it, so
        # simple_agent runs wrote no per-trial trajectories.
        from example_recipe.word_puzzle.agent_solver.simple_agent.config_model import (
            SimpleAgentConfig,
        )

        writer = object()
        agent = SimpleAgentConfig().make(
            sampler=_SequenceSampler([]),
            job_id="j",
            reward_fn=_reward_fn(),
            atif_writer=writer,
        )
        assert agent._trajectory_callback is writer

    def test_eval_agent_format_variant_enables_both(self):
        # The opt-in variant: agent-format eval prompts (no $\text{...}$
        # instruction, which the exact-match submit grader cannot parse) and
        # the answer-only submit contract, wired through to the env.
        import json

        from example_recipe.word_puzzle.agent_solver.simple_agent.config_model import (
            CONFIGS,
        )

        cfg = CONFIGS["eval_agent_format"]
        with open(cfg.task.eval_data_path) as f:
            prompts = [
                p.get("text", "")
                for line in f
                if line.strip()
                for c in json.loads(line)["contents"]
                for p in c["parts"]
            ]
        assert prompts
        assert not any("\\text{" in t for t in prompts)

        agent = cfg.agent.make(sampler=_SequenceSampler([]), job_id="j", reward_fn=_reward_fn())
        env = agent.env_factory(_task_data())
        assert env.answer_only_submit is True
        assert list(env.get_prompt().tools) == list(TOOLS_ANSWER_ONLY)

    def test_eval_agent_format_variant_is_async_plus_two_fields(self):
        # The variant is derived from ``async``; everything except the
        # answer-only flag and the eval set must match it exactly.
        import dataclasses

        from example_recipe.word_puzzle.agent_solver.simple_agent.config_model import (
            CONFIGS,
        )

        base = CONFIGS["async"]
        variant = CONFIGS["eval_agent_format"]
        base_agent, base_task = _typed_parts(base)
        variant_agent, variant_task = _typed_parts(variant)
        assert variant == dataclasses.replace(
            base,
            agent=dataclasses.replace(base_agent, answer_only_submit=True),
            task=dataclasses.replace(
                base_task, eval_data_path=variant_task.eval_data_path
            ),
        )
        assert variant_agent.answer_only_submit is True
        assert variant_task.eval_data_path != base_task.eval_data_path
        assert variant_task.train_data_path == base_task.train_data_path


# ---------------------------------------------------------------------------
# Golden regression: default variants must match base commit 4e68b830
# ---------------------------------------------------------------------------

# All literals below are copied from base commit
# 4e68b8306960b1a6148cb81ad804600d952a9868
# (src/example_recipe/word_puzzle/agent_solver/simple_agent/prompt.py and
# config_model.py), NOT from the current modules, so any drift in
# SYSTEM_TEMPLATE, INSTANCE_TEMPLATE, the bash/submit tool declarations, the
# step_limit wiring or the dataset paths fails here.
_GOLDEN_SYSTEM_TEXT = (
    "You are a math puzzle solver.\n\nYou have access to two tools:\n"
    "- **bash**: execute a bash command and see the output.\n"
    "- **submit**: submit your final answer.\n\n"
    "You have at most 10 tool-use turns. Use bash to explore the problem and "
    "verify your answer, then call submit with your final answer."
)
_GOLDEN_USER_TEXT = (
    "Solve the following puzzle. When you are confident, call the submit tool "
    "with your answer.\n\nWhat is 2+2?"
)
_GOLDEN_TRAIN_BASENAME = "combined_training_word_transformation_v1.jsonl"
_GOLDEN_EVAL_BASENAME = "combined_eval_word_transformation_v1.jsonl"


def _golden_tools() -> list[Any]:
    """The base commit's ``TOOLS``, as literal proto objects."""
    from google.cloud.aiplatform_v1.types import (
        FunctionDeclaration,
        Schema,
        Tool,
        Type,
    )

    bash = FunctionDeclaration(
        name="bash",
        description="Run a bash command in a sandbox and return stdout/stderr.",
        parameters=Schema(
            type=Type.OBJECT,
            properties={
                "command": Schema(
                    type=Type.STRING,
                    description="The bash command to execute.",
                ),
            },
            required=["command"],
        ),
    )
    submit = FunctionDeclaration(
        name="submit",
        description="Submit your final answer to the puzzle.",
        parameters=Schema(
            type=Type.OBJECT,
            properties={
                "answer": Schema(
                    type=Type.STRING,
                    description="Your final answer.",
                ),
            },
            required=["answer"],
        ),
    )
    return [Tool(function_declarations=[bash, submit])]


def _golden_request() -> Any:
    """The full base-commit request for the golden task, as a literal proto."""
    from google.cloud.aiplatform_v1.types import Content, GenerateContentRequest, Part

    return GenerateContentRequest(
        contents=[Content(role="user", parts=[Part(text=_GOLDEN_USER_TEXT)])],
        system_instruction=Content(parts=[Part(text=_GOLDEN_SYSTEM_TEXT)]),
        tools=_golden_tools(),
    )


def _typed_parts(cfg: Any) -> tuple[SimpleAgentConfig, WordPuzzleTaskConfig]:
    """Narrow a RunConfig's agent/task to the simple_agent word-puzzle types."""
    from example_recipe.word_puzzle.agent_solver.simple_agent.config_model import (
        SimpleAgentConfig as _SimpleAgentConfig,
    )
    from example_recipe.word_puzzle.config_model import (
        WordPuzzleTaskConfig as _WordPuzzleTaskConfig,
    )

    assert isinstance(cfg.agent, _SimpleAgentConfig)
    assert isinstance(cfg.task, _WordPuzzleTaskConfig)
    return cfg.agent, cfg.task


@pytest.mark.parametrize("name", ["async", "smoke"])
class TestDefaultVariantsGolden:
    def _prompt(self, name: str) -> Any:
        from example_recipe.word_puzzle.agent_solver.simple_agent.config_model import (
            CONFIGS,
        )

        agent = CONFIGS[name].agent.make(
            sampler=_SequenceSampler([]), job_id="j", reward_fn=_reward_fn()
        )
        assert agent is not None and agent.env_factory is not None
        td = WordPuzzleTaskData(task_id="golden", question="What is 2+2?", reference="4")
        return agent.env_factory(td).get_prompt()

    def test_prompt_matches_base(self, name: str):
        prompt = self._prompt(name)
        assert prompt.system_instruction.parts[0].text == _GOLDEN_SYSTEM_TEXT
        assert prompt.contents[0].parts[0].text == _GOLDEN_USER_TEXT
        # Whole request (roles, part layout, no extra fields) by proto equality.
        assert prompt == _golden_request()

    def test_tools_match_base(self, name: str):
        prompt = self._prompt(name)
        assert list(prompt.tools) == _golden_tools()

    def test_datasets_match_base(self, name: str):
        # Base resolved both paths to <word_puzzle package>/dataset/<basename>.
        import os

        from example_recipe.word_puzzle import config_model as wp_config_model
        from example_recipe.word_puzzle.agent_solver.simple_agent.config_model import (
            CONFIGS,
        )

        data_dir = os.path.join(
            os.path.dirname(os.path.abspath(wp_config_model.__file__)), "dataset"
        )
        _, task = _typed_parts(CONFIGS[name])
        assert os.path.basename(task.train_data_path) == _GOLDEN_TRAIN_BASENAME
        assert os.path.basename(task.eval_data_path) == _GOLDEN_EVAL_BASENAME
        assert task.train_data_path == os.path.join(data_dir, _GOLDEN_TRAIN_BASENAME)
        assert task.eval_data_path == os.path.join(data_dir, _GOLDEN_EVAL_BASENAME)

    def test_run_config_matches_base(self, name: str):
        # Rebuild the base commit's RunConfig literally; only the new
        # ``answer_only_submit=False`` field is allowed to differ (it is the
        # default, so SimpleAgentConfig() already carries it).
        from backends.backend_config import VertexMsrlBackendConfig
        from core.eval_config import EvalConfig
        from core.run_config import RunConfig, TrainConfig
        from example_recipe.word_puzzle.agent_solver.simple_agent.config_model import (
            CONFIGS,
            SimpleAgentConfig,
        )
        # The recipe's reward config class (not reward.WordPuzzleRewardConfig).
        from example_recipe.word_puzzle.config_model import (
            WordPuzzleRewardConfig as RecipeRewardConfig,
            WordPuzzleTaskConfig,
        )
        from pipeline.async_rollout_pipeline import AsyncRolloutConfig
        from pipeline.sync_rollout_pipeline import SyncRolloutConfig

        cfg = CONFIGS[name]
        agent, task = _typed_parts(cfg)
        shared: dict[str, Any] = dict(
            agent=SimpleAgentConfig(step_limit=10, bash_timeout_seconds=30),
            reward=RecipeRewardConfig(),
            backend=VertexMsrlBackendConfig(),
            task=WordPuzzleTaskConfig(
                train_data_path=task.train_data_path,
                eval_data_path=task.eval_data_path,
            ),
        )
        if name == "async":
            expected = RunConfig(
                train=TrainConfig(sync_every_n_train_steps=4),
                rollout=AsyncRolloutConfig(max_pending_rollout_tasks=32 * 4),
                eval=EvalConfig(eval_every_n_train_steps=10),
                **shared,
            )
        else:
            expected = RunConfig(
                train=TrainConfig(
                    num_train_steps=3,
                    tasks_per_train_step=4,
                    sync_every_n_train_steps=1,
                ),
                rollout=SyncRolloutConfig(),
                eval=EvalConfig(eval_every_n_train_steps=0),
                **shared,
            )
        assert cfg == expected
        assert agent.answer_only_submit is False
