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

"""Tests for AgentSolverAgentConfig.make() and the CONFIGS entries.

make() is exercised with a fake sampler + fake reward fn; it must return an
AgentSolverAgent WITHOUT constructing any Docker container (the mini-agent
factory is lazy -- only invoked per rollout, which these tests never trigger).
"""

from __future__ import annotations

from typing import Any

from example_recipe.word_puzzle.agent_solver.minisweagent.agent_solver_agent import (
    AgentSolverAgent,
)
from example_recipe.word_puzzle.agent_solver.minisweagent.agent_solver_config import (
    AgentSolverAgentConfig,
)
from example_recipe.word_puzzle.reward import (
    WordPuzzleRewardConfig,
    WordPuzzleRewardFn,
)


def test_make_returns_agent_solver_agent_without_docker() -> None:
    cfg = AgentSolverAgentConfig()
    reward_fn = WordPuzzleRewardFn(WordPuzzleRewardConfig())
    agent = cfg.make(
        sampler=object(),  # never called: factory is lazy
        job_id="job-1",
        reward_fn=reward_fn,
        atif_writer=None,
    )
    assert isinstance(agent, AgentSolverAgent)


def test_make_passes_timeout_through() -> None:
    # rollout timeout must stay STRICTLY above wall (invariant); use a value
    # comfortably above the default wall (7000).
    cfg = AgentSolverAgentConfig(agent_rollout_timeout_seconds=8000.0)
    reward_fn = WordPuzzleRewardFn(WordPuzzleRewardConfig())
    agent = cfg.make(sampler=object(), job_id="j", reward_fn=reward_fn)
    assert agent.agent_rollout_timeout_seconds == 8000.0


def test_container_outlives_rollout_budget() -> None:
    """Regression (prod: 68% of rollouts hit 'No such container'): the sandbox
    container MUST outlive the rollout's hard upper bound, or its `sleep`
    expires mid-rollout and --rm deletes it -> the agent can't run tools,
    never submits, scores the -1.0 floor. Enforce wall <= rollout < container
    for every default config."""
    cfg = AgentSolverAgentConfig()
    rollout = cfg.agent_rollout_timeout_seconds
    assert rollout is not None
    # STRICT: equal wall/rollout race and cause mid-call hard cancels.
    assert cfg.wall_time_limit_seconds < rollout, "wall must be strictly below rollout"
    assert cfg.container_timeout_seconds() > rollout, (
        "container must outlive the rollout budget"
    )


def test_container_timeout_falls_back_to_wall_when_no_rollout_timeout() -> None:
    """If the framework rollout timeout is unset, the container tracks the mini
    wall clock instead (still > it, never smaller)."""
    cfg = AgentSolverAgentConfig(
        agent_rollout_timeout_seconds=None,
        wall_time_limit_seconds=900,
    )
    assert cfg.container_timeout_seconds() > 900


def test_configs_dict_has_agent_operating_points() -> None:
    from example_recipe.word_puzzle.agent_solver.minisweagent.agent_solver_config import (
        CONFIGS,
    )

    assert "agent_async" in CONFIGS
    assert "agent_sync" in CONFIGS
    assert "agent_smoke" in CONFIGS
    # Each is a RunConfig whose agent module is an AgentSolverAgentConfig, and
    # the timer invariant (wall <= rollout < container) holds so the sandbox
    # can't die mid-rollout.
    for name in ("agent_async", "agent_sync", "agent_smoke"):
        rc = CONFIGS[name]
        agent = rc.agent
        assert isinstance(agent, AgentSolverAgentConfig)
        rollout = agent.agent_rollout_timeout_seconds
        assert rollout is not None
        assert agent.wall_time_limit_seconds < rollout
        assert agent.container_timeout_seconds() > rollout, (
            f"{name}: container must outlive the rollout budget"
        )


def test_vertex_experiment_enabled_by_default() -> None:
    """Every operating point streams to a Vertex Experiment by default (a
    non-empty vertex_experiment enables the logger); a regression to '' would
    silently drop the browsable reward/metric curves."""
    from example_recipe.word_puzzle.agent_solver.minisweagent.agent_solver_config import (
        CONFIGS,
    )

    for name in ("agent_async", "agent_sync", "agent_smoke"):
        assert CONFIGS[name].task.vertex_experiment == "word-puzzle-agent-solver", (
            f"{name} must default to the shared Vertex experiment"
        )


def test_agent_smoke_is_a_small_no_eval_first_run() -> None:
    """agent_smoke exists to validate the real-MSRL path cheaply: few steps,
    few tasks, and NO eval sweeps. Pin those knobs so the "small first run"
    contract can't silently regress into an expensive run."""
    from example_recipe.word_puzzle.agent_solver.minisweagent.agent_solver_config import (
        CONFIGS,
    )

    rc = CONFIGS["agent_smoke"]
    assert rc.train.num_train_steps == 3
    assert rc.train.tasks_per_train_step == 4
    # Eval disabled (0 => never eval) so the smoke never triggers a sweep.
    assert rc.eval.eval_every_n_train_steps == 0


def test_registry_resolves_agent_solver_configs() -> None:
    import config_registry

    # Recipe namespace is "word_puzzle_agent" (config_registry.py:20); the
    # older "word_puzzle_minisweagent" spelling no longer resolves.
    for variant in ("agent_async", "agent_sync", "agent_smoke"):
        rc = config_registry.get_config(f"word_puzzle_agent/{variant}")
        assert isinstance(rc.agent, AgentSolverAgentConfig)
