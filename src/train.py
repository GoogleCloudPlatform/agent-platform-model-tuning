#!/usr/bin/env python3
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
"""Training entry point (composed RunConfig + launch overrides).

Single CLI surface: a composed :class:`~core.run_config.RunConfig` is
selected by ``config_name`` and driven through the recipe-agnostic
``run_training`` orchestrator. The launch-env essentials (Vertex project /
location, log dir, resume knobs, data-path overrides) are top-level CLI
args, so they are passed BARE::

    python src/train.py config_name=word_puzzle/async \\
        vertex_project=my-proj log_dir=/tmp/run resume_from_train_step=0 ...

``main`` destructures those args into a :class:`TrainLaunchOverrides`,
applies the non-None fields onto the checked-in config, and runs the
composed pipeline via the config's ``make()`` seam.
"""

from __future__ import annotations

import asyncio
import signal
import sys

import chz

from core.launch_overrides import TrainLaunchOverrides, apply_overrides
from config_registry import get_config, recipe_label
from train_config_entry import run_training


def main(
    config_name: str,
    *,
    target: str | None = None,
    vertex_existing_job_id: str | None = None,
    vertex_project: str | None = None,
    vertex_location: str | None = None,
    vertex_base_model: str | None = None,
    log_dir: str | None = None,
    resume_from_train_step: int | None = None,
    resume_metrics_path: str | None = None,
    train_data_path: str | None = None,
    eval_data_path: str | None = None,
    num_train_steps: int | None = None,
    max_eval_items: int | None = None,
    eval_samples_per_task: int | None = None,
) -> None:
    """Load the config named ``config_name``, apply the non-None launch
    overrides onto it, and drive ``run_training``."""
    overrides = TrainLaunchOverrides(
        config_name=config_name,
        target=target,
        vertex_existing_job_id=vertex_existing_job_id,
        vertex_project=vertex_project,
        vertex_location=vertex_location,
        vertex_base_model=vertex_base_model,
        log_dir=log_dir,
        resume_from_train_step=resume_from_train_step,
        resume_metrics_path=resume_metrics_path,
        train_data_path=train_data_path,
        eval_data_path=eval_data_path,
        num_train_steps=num_train_steps,
        max_eval_items=max_eval_items,
        eval_samples_per_task=eval_samples_per_task,
    )
    config = apply_overrides(get_config(config_name), overrides)
    asyncio.run(run_training(config, recipe=recipe_label(config_name)))


def _install_signal_handlers() -> None:
    """Translate SIGTERM/SIGHUP into SystemExit so finally blocks
    (backend bundle aclose / tuning-job cancel) run on termination."""

    def _handler(signum: int, _frame: object) -> None:
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGHUP, _handler)


if __name__ == "__main__":
    _install_signal_handlers()
    chz.entrypoint(main)
