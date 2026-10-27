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
"""Standalone eval entry point (composed RunConfig + launch overrides).

Attaches to an existing Vertex tuning job (or a base model / deployed
endpoint) and runs ONE eval sweep on a dataset.  No training, no rollout
buffer, no gradient updates.  Logs ``eval/*`` metrics.

Single CLI surface: a composed :class:`~core.run_config.RunConfig` is
selected by ``config_name`` and driven through the recipe-agnostic
``run_eval`` orchestrator.  The ``target`` arg accepts full Vertex
resource paths or short model names::

    python src/eval.py config_name=word_puzzle/async \\
        target=projects/P/locations/L/tuningJobs/ID \\
        eval_data_path=... jsonl_output_path=eval.jsonl

``main`` destructures those args into an :class:`EvalLaunchOverrides`,
applies the non-None fields onto the checked-in config, and runs ONE
eval sweep via the config's ``make()`` seam.
"""

from __future__ import annotations

import asyncio
import signal
import sys

import chz

from core.launch_overrides import EvalLaunchOverrides, apply_overrides
from config_registry import get_config, recipe_label
from eval_config_entry import run_eval


def main(
    config_name: str,
    *,
    target: str | None = None,
    vertex_project: str | None = None,
    vertex_location: str | None = None,
    eval_data_path: str | None = None,
    jsonl_output_path: str | None = None,
    max_eval_items: int | None = None,
) -> None:
    """Load the config named ``config_name``, apply the non-None launch
    overrides onto it, and drive one ``run_eval`` sweep."""
    overrides = EvalLaunchOverrides(
        config_name=config_name,
        target=target,
        vertex_project=vertex_project,
        vertex_location=vertex_location,
        eval_data_path=eval_data_path,
        jsonl_output_path=jsonl_output_path,
        max_eval_items=max_eval_items,
    )
    config = apply_overrides(get_config(config_name), overrides)
    asyncio.run(
        run_eval(
            config,
            recipe=recipe_label(config_name),
            jsonl_output_path=overrides.jsonl_output_path,
        )
    )


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
