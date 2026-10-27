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

"""Config + operating points that make the word-puzzle agent-solver trainable.

``AgentSolverAgentConfig.make()`` builds an :class:`AgentSolverAgent` whose
per-rollout mini-agent factory constructs the hardened Docker sandbox +
``AsyncDefaultAgent`` (the same wiring the standalone runner uses). The factory
is lazy -- no container is created at ``make()`` time, only per rollout -- so
config construction is cheap and unit-testable without Docker.

``CONFIGS`` exposes ``agent_async`` / ``agent_sync`` operating points that
compose a :class:`RunConfig` around this agent, reusing the word_puzzle reward,
backend, and task modules (pointed at the agent-mode withhold JSONL).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from backends.backend_config import VertexMsrlBackendConfig
from core.agent_config import AgentConfig
from core.eval_config import EvalConfig
from core.run_config import RunConfig, TrainConfig
from core.task_data import TaskData
from pipeline.async_rollout_pipeline import AsyncRolloutConfig
from pipeline.sync_rollout_pipeline import SyncRolloutConfig

from example_recipe.word_puzzle.agent_solver.minisweagent.agent_solver_agent import (
    AgentSolverAgent,
)
from example_recipe.word_puzzle.agent_solver.minisweagent.async_default_agent import (
    AsyncDefaultAgent,
)
from example_recipe.word_puzzle.agent_solver.minisweagent.docker_run_args import (
    SANDBOX_IMAGE,
    SANDBOX_RUN_ARGS,
)
from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
    SamplerChatCompletionModel,
)
from example_recipe.word_puzzle.agent_solver.minisweagent.word_puzzle_task import (
    INSTANCE_TEMPLATE,
    SYSTEM_TEMPLATE,
)
from example_recipe.word_puzzle.config_model import (
    WordPuzzleRewardConfig,
    WordPuzzleTaskConfig,
)

# Agent-mode withhold OOD eval set (structured submit; no LaTeX trailer).
# __file__ = .../agent_solver/minisweagent/agent_solver_config.py
# Go up 3 levels to reach .../word_puzzle/, then into dataset/.
_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "dataset",
)
_AGENT_JSONL = os.path.join(
    _DATA_DIR, "combined_eval_word_transformation_withhold_agent_v1.jsonl"
)


@dataclass(frozen=True)
class AgentSolverAgentConfig(AgentConfig):
    """Agent module for the trainable word-puzzle agent-solver.

    Inherits ``thinking_level`` / ``agent_rollout_timeout_seconds`` /
    ``max_generation_tokens`` from ``AgentConfig``. Adds the mini-agent runtime
    limits that are CLI flags on the standalone runner.

    Note: the model/endpoint are NOT configured here. The base model and
    routing are owned entirely by the backend (the injected sampler / its
    tuning job); the tuning-scope endpoint ignores any per-request model
    field. ``SamplerChatCompletionModel`` keeps its own defaulted
    ``model_name`` / ``env`` purely for trajectory provenance.
    """

    step_limit: int = 20
    wall_time_limit_seconds: int = 7000
    cost_limit: float = -1  # disable
    submit_retries: int = 3

    def __post_init__(self) -> None:
        # Enforce the STRICT timeout nesting at construction so a future edit
        # can't silently reintroduce the equal-values race that hard-cancelled
        # rollouts mid-call. (frozen dataclass: read-only validation only.)
        parent_post = getattr(super(), "__post_init__", None)
        if parent_post is not None:
            parent_post()
        rollout = self.agent_rollout_timeout_seconds
        if rollout is not None:
            if not (self.wall_time_limit_seconds < rollout):
                raise ValueError(
                    "wall_time_limit_seconds "
                    f"({self.wall_time_limit_seconds}) must be STRICTLY less "
                    f"than agent_rollout_timeout_seconds ({rollout}); equal "
                    "values race and cause mid-call hard cancels."
                )

    def container_timeout_seconds(self) -> int:
        """Sandbox container lifetime, in seconds.

        MUST exceed the rollout's hard upper bound so the container is never
        --rm'd out from under an in-progress rollout (which causes
        "No such container"). The upper bound is the framework's
        ``agent_rollout_timeout_seconds`` (the framework won't abandon a
        rollout before then); fall back to the mini wall clock if unset. Add
        slack on top so a rollout finishing right at the deadline still has a
        live container.
        """
        rollout_budget = (
            int(self.agent_rollout_timeout_seconds)
            if self.agent_rollout_timeout_seconds
            else self.wall_time_limit_seconds
        )
        return rollout_budget + 120

    def make(
        self,
        *,
        sampler: Any,
        job_id: Any,
        reward_fn: Any,
        atif_writer: Any = None,
    ) -> AgentSolverAgent:
        # model_name/env intentionally omitted: routing is fixed by the
        # injected sampler, not this config. SamplerChatCompletionModel
        # defaults them for provenance.
        model_cfg = {
            "thinking_level": (self.thinking_level or "").lower() or None,
            "max_generation_tokens": self.max_generation_tokens or 65536,
        }

        def mini_agent_factory(
            task_data: TaskData, rollout_idx: int
        ) -> AsyncDefaultAgent:
            # Import mini's DockerEnvironment lazily so config construction /
            # unit tests don't require the docker python package or a daemon.
            from minisweagent.environments.docker import DockerEnvironment

            env = DockerEnvironment(
                image=SANDBOX_IMAGE,
                cwd="/workspace",
                timeout=30,
                container_timeout=f"{self.container_timeout_seconds()}s",
                run_args=SANDBOX_RUN_ARGS,
            )
            model = SamplerChatCompletionModel(sampler=sampler, **model_cfg)
            return AsyncDefaultAgent(
                model=model,
                env=env,
                system_template=SYSTEM_TEMPLATE,
                instance_template=INSTANCE_TEMPLATE,
                step_limit=self.step_limit,
                wall_time_limit_seconds=self.wall_time_limit_seconds,
                cost_limit=self.cost_limit,
                submit_retries=self.submit_retries,
                expected_answer=getattr(task_data, "reference", "") or "",
                task_seed=rollout_idx,
            )

        return AgentSolverAgent(
            sampler,
            reward_fn=reward_fn,
            mini_agent_factory=mini_agent_factory,
            prompt_of=lambda td: getattr(td, "question", "") or "",
            agent_rollout_timeout_seconds=self.agent_rollout_timeout_seconds,
            trajectory_callback=atif_writer,
            job_id=str(job_id) if job_id else "",
        )


# Vertex AI Experiments logger name shared by all agent-solver operating
# points. Enabled by default so every real run streams reward/metric curves
# to a browsable Vertex Experiment; each run is a distinct ExperimentRun keyed
# by tuning job id (compose_vertex_run_name prepends it), so concurrent runs
# sharing this experiment name don't collide. Override to "" to disable.
_VERTEX_EXPERIMENT = "word-puzzle-agent-solver"

_SHARED: dict[str, Any] = dict(
    agent=AgentSolverAgentConfig(
        thinking_level="MINIMAL",
        step_limit=20,
    ),
    reward=WordPuzzleRewardConfig(),
    backend=VertexMsrlBackendConfig(),
    task=WordPuzzleTaskConfig(
        train_data_path=_AGENT_JSONL,
        eval_data_path=_AGENT_JSONL,
        vertex_experiment=_VERTEX_EXPERIMENT,
    ),
)

# Small first-run variant for validating the real-MSRL training path cheaply:
# few train steps, few tasks/step, MINIMAL thinking + tight rollout limits, and
# NO eval sweeps. Use this before the full-scale agent_async/agent_sync. The
# knobs baked here are NOT CLI-overridable (train.py only overrides
# backend/log/data), so a dedicated variant is the safe way to run small.
_SMOKE_SHARED: dict[str, Any] = dict(
    agent=AgentSolverAgentConfig(
        thinking_level="MINIMAL",
        step_limit=15,
    ),
    reward=WordPuzzleRewardConfig(),
    backend=VertexMsrlBackendConfig(),
    task=WordPuzzleTaskConfig(
        train_data_path=_AGENT_JSONL,
        eval_data_path=_AGENT_JSONL,
        vertex_experiment=_VERTEX_EXPERIMENT,
    ),
)

CONFIGS: dict[str, RunConfig] = {
    "agent_async": RunConfig(
        train=TrainConfig(sync_every_n_train_steps=4),
        rollout=AsyncRolloutConfig(max_pending_rollout_tasks=32 * 4),
        eval=EvalConfig(eval_every_n_train_steps=10),
        **_SHARED,
    ),
    "agent_sync": RunConfig(
        train=TrainConfig(sync_every_n_train_steps=1),
        rollout=SyncRolloutConfig(),
        eval=EvalConfig(
            eval_samples_per_task=5,
            eval_every_n_train_steps=5,
        ),
        **_SHARED,
    ),
    # First-run smoke: 3 train steps x 4 tasks, sync weight-sync every step,
    # no eval. Proves real weight-sync + Docker-at-scale + variance survival.
    "agent_smoke": RunConfig(
        train=TrainConfig(
            num_train_steps=3,
            tasks_per_train_step=4,
            sync_every_n_train_steps=1,
        ),
        rollout=SyncRolloutConfig(),
        eval=EvalConfig(eval_every_n_train_steps=0),
        **_SMOKE_SHARED,
    ),
}
