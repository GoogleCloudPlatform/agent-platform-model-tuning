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

"""Generate a word-puzzle eval set on the fly and run the eval plug-in.

Generates N tasks via ``puzzle_dataset_gen.generate_dataset``, writes
them to a temp JSONL, then runs the composed eval entry point
(``eval_config_entry.run_eval``) against that file.

Example (base-model gemini-3.5-flash on prod):

    uv run python -m example_recipe.word_puzzle.dataset.run_generated_eval \\
        config.num_tasks=50 \\
        config.num_transformations=5 \\
        config.num_few_shot_examples=5 \\
        config.operator_set=extended \\
        config.seed=42 \\
        config.backend=gemini_endpoint \\
        config.gemini_endpoint_model=gemini-3.5-flash \\
        config.gemini_endpoint_env=prod \\
        config.gemini_endpoint_project=<your-gcp-project> \\
        config.eval_samples_per_task=1

The temp JSONL lives under ``/tmp/`` and is removed unless
``config.keep_temp=True``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import tempfile
from typing import Optional

import chz

logger = logging.getLogger(__name__)

from backends.backend_config import (
    GeminiApiBackendConfig,
    SyncRetryProfile,
    VertexMsrlBackendConfig,
)
from core.eval_config import EvalConfig
from pipeline.async_rollout_pipeline import AsyncRolloutConfig
from core.run_config import RunConfig, TrainConfig
import eval_config_entry
from example_recipe.word_puzzle.dataset import puzzle_dataset_gen as pdg
from example_recipe.word_puzzle.config_model import (
    WordPuzzleRewardConfig,
    WordPuzzleTaskConfig,
)
from example_recipe.word_puzzle.single_step.config_model import (
    WordPuzzleAgentConfig,
)


@chz.chz
class GeneratedEvalConfig:
    # ── On-the-fly dataset generation ────────────────────────────────
    num_tasks: int = 50
    num_transformations: int = 5
    num_few_shot_examples: int = 5
    operator_set: str = "extended"  # basic | extended | withhold | all
    min_word_length: int = 5
    word_list_path: str = ""  # "" -> bundled word_allowlist.txt
    seed: int = 42
    # If True, also generate encrypt-direction tasks (doubles the count).
    flip_reverse_task: bool = False
    explain_encryption: bool = True
    keep_temp: bool = False

    # ── Eval-side knobs (forwarded to the composed RunConfig) ────────
    backend: str = "gemini_endpoint"
    gemini_endpoint_model: str = "gemini-3.5-flash"
    # Your GCP project (required when using the gemini_endpoint backend).
    gemini_endpoint_project: str = ""
    gemini_endpoint_env: str = "prod"
    gemini_endpoint_location: str = "global"
    gemini_endpoint_use_user_credentials: bool = False
    gemini_endpoint_api_version: str = "v1beta1"
    gemini_endpoint_max_tries: int = 3
    gemini_endpoint_id: str = ""

    # vertex_msrl backend fields (used only if backend="vertex_msrl"):
    vertex_existing_job_id: str = ""
    vertex_project: str = ""
    vertex_location: str = "us-central1"
    vertex_base_model: str = "gemini-3.5-flash"
    vertex_lr_multiplier: float = 0.0
    vertex_adapter_size: str = ""
    vertex_checkpoint_interval: int = 0
    vertex_job_running_timeout_seconds: int = 30 * 60
    vertex_request_timeout: float = 600.0
    vertex_request_qpm: float = 1800.0

    # Sampling / logging knobs (forwarded)
    eval_samples_per_task: int = 1
    thinking_level: str = ""
    max_concurrent_samples: int = 64
    max_generation_tokens: int = 32768
    agent_rollout_timeout_seconds: Optional[float] = None
    log_dir: str = "logs/eval_generated"
    clear_log_dir: bool = False
    eval_step: int = 0
    vertex_experiment: str = ""
    vertex_experiment_tensorboard: str = ""
    vertex_experiment_run_name: str = ""
    service_account: str = ""


def _generate_temp_jsonl(config: GeneratedEvalConfig) -> str:
    """Generate the eval JSONL and return its temp path."""
    if config.operator_set not in pdg.OPERATOR_SETS:
        raise ValueError(
            f"Unknown operator_set {config.operator_set!r}; expected one of"
            f" {sorted(pdg.OPERATOR_SETS)}"
        )
    operators = pdg.OPERATOR_SETS[config.operator_set]
    word_candidates = pdg.load_word_candidates(
        path=config.word_list_path or None,
        min_length=config.min_word_length,
    )
    if not word_candidates:
        raise ValueError(
            "Empty word candidate list; check config.word_list_path and "
            "config.min_word_length."
        )
    rng = random.Random(config.seed)
    tasks = pdg.generate_dataset(
        num_examples=config.num_tasks,
        num_transformations=config.num_transformations,
        num_few_shot_examples=config.num_few_shot_examples,
        word_candidates=word_candidates,
        operators=operators,
        explain_encryption=config.explain_encryption,
        flip_reverse_task=config.flip_reverse_task,
        rng=rng,
    )
    fd, path = tempfile.mkstemp(
        prefix=f"word_puzzle_gen_{config.num_tasks}_", suffix=".jsonl", dir="/tmp"
    )
    os.close(fd)
    pdg.write_jsonl(tasks, path)
    logger.info(
        "Wrote %d generated tasks (operator_set=%s, "
        "num_transformations=%d, num_few_shot_examples=%d, seed=%d) to %s",
        len(tasks),
        config.operator_set,
        config.num_transformations,
        config.num_few_shot_examples,
        config.seed,
        path,
    )
    return path


def _forward_to_run_config(
    config: GeneratedEvalConfig, generated_path: str
) -> RunConfig:
    """Map GeneratedEvalConfig -> composed RunConfig (additive path).

    The generated temp JSONL becomes the eval dataset. The flat ``backend``
    string selects the backend-config subclass TYPE (the composed model has no
    string switch); the flat per-group backend fields map onto that subclass's
    counterparts. Some flat logger-only knobs (``clear_log_dir``, ``eval_step``,
    ``vertex_experiment_tensorboard``, ``vertex_experiment_run_name``) have no
    home in the composed modules and are dropped here, matching how the composed
    eval entry point already handles them.
    """
    backend_kind = config.backend.strip().lower()
    if backend_kind == "vertex_msrl":
        backend = VertexMsrlBackendConfig(
            project=config.vertex_project,
            location=config.vertex_location,
            vertex_base_model=config.vertex_base_model,
            vertex_existing_job_id=config.vertex_existing_job_id,
            vertex_checkpoint_interval=config.vertex_checkpoint_interval,
            vertex_job_running_timeout_seconds=config.vertex_job_running_timeout_seconds,
            vertex_request_qpm=config.vertex_request_qpm,
            request_timeout=config.vertex_request_timeout,
            service_account=config.service_account,
        )
    elif backend_kind == "gemini_endpoint":
        backend = GeminiApiBackendConfig(
            model=config.gemini_endpoint_model,
            project=config.gemini_endpoint_project,
            env=config.gemini_endpoint_env,
            location=config.gemini_endpoint_location,
            use_user_credentials=config.gemini_endpoint_use_user_credentials,
            api_version=config.gemini_endpoint_api_version,
            retry=SyncRetryProfile(max_attempts=config.gemini_endpoint_max_tries),
            endpoint_id=config.gemini_endpoint_id,
            request_timeout=config.vertex_request_timeout,
            service_account=config.service_account,
        )
    else:
        raise ValueError(
            f"Unsupported backend {config.backend!r}; expected 'vertex_msrl' or"
            " 'gemini_endpoint'."
        )

    return RunConfig(
        train=TrainConfig(log_dir=config.log_dir),
        rollout=AsyncRolloutConfig(
            max_concurrent_samples=config.max_concurrent_samples
        ),
        eval=EvalConfig(
            eval_samples_per_task=config.eval_samples_per_task,
            max_eval_items=-1,  # use all generated tasks
        ),
        agent=WordPuzzleAgentConfig(
            max_generation_tokens=config.max_generation_tokens,
            thinking_level=config.thinking_level,
            agent_rollout_timeout_seconds=config.agent_rollout_timeout_seconds,
        ),
        reward=WordPuzzleRewardConfig(),
        backend=backend,
        task=WordPuzzleTaskConfig(
            eval_data_path=generated_path,
            vertex_experiment=config.vertex_experiment,
        ),
    )


def main(config: GeneratedEvalConfig) -> None:
    generated_path = _generate_temp_jsonl(config)
    run_config = _forward_to_run_config(config, generated_path)
    try:
        asyncio.run(eval_config_entry.run_eval(run_config, recipe="word_puzzle"))
    finally:
        if not config.keep_temp:
            try:
                os.remove(generated_path)
                logger.info("Removed temp file %s", generated_path)
            except OSError as e:
                logger.warning("Could not remove %s: %s", generated_path, e)
        else:
            logger.info("Kept temp file %s", generated_path)


if __name__ == "__main__":
    chz.nested_entrypoint(main)
