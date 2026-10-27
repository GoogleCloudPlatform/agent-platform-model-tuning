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

"""Composed RunConfig forwarding path for the generated-eval wrapper.

Additive sibling coverage for ``run_generated_eval.py``: exercises the
``_forward_to_run_config`` mapper (flat ``GeneratedEvalConfig`` -> composed
``RunConfig``) and ``main`` (generate temp jsonl -> forward ->
run_eval -> temp cleanup). The mapper tests need no network / dataset-gen;
they just construct a ``GeneratedEvalConfig`` and assert the composed shape.
"""

from __future__ import annotations

import pytest

from backends.backend_config import (
    GeminiApiBackendConfig,
    VertexMsrlBackendConfig,
)
from pipeline.async_rollout_pipeline import AsyncRolloutConfig
from core.run_config import RunConfig
import example_recipe.word_puzzle.dataset.run_generated_eval as rge


def test_forward_to_run_config_gemini_endpoint_backend():
    cfg = rge.GeneratedEvalConfig(
        backend="gemini_endpoint",
        gemini_endpoint_model="gemini-3.5-flash",
        gemini_endpoint_project="my-proj",
        gemini_endpoint_env="prod",
        gemini_endpoint_id="ep-42",
    )
    run_config = rge._forward_to_run_config(cfg, "/tmp/x.jsonl")

    assert isinstance(run_config, RunConfig)
    assert isinstance(run_config.backend, GeminiApiBackendConfig)
    assert run_config.backend.model == "gemini-3.5-flash"
    assert run_config.backend.project == "my-proj"
    assert run_config.backend.env == "prod"
    assert run_config.backend.endpoint_id == "ep-42"


def test_forward_to_run_config_vertex_msrl_backend():
    cfg = rge.GeneratedEvalConfig(
        backend="vertex_msrl",
        vertex_project="vproj",
        vertex_existing_job_id="job-99",
    )
    run_config = rge._forward_to_run_config(cfg, "/tmp/x.jsonl")

    assert isinstance(run_config.backend, VertexMsrlBackendConfig)
    assert run_config.backend.project == "vproj"
    assert run_config.backend.vertex_existing_job_id == "job-99"


def test_forward_to_run_config_bogus_backend_raises():
    cfg = rge.GeneratedEvalConfig(backend="bogus")
    with pytest.raises(ValueError):
        rge._forward_to_run_config(cfg, "/tmp/x.jsonl")


def test_forward_to_run_config_maps_task_eval_agent_rollout_train():
    cfg = rge.GeneratedEvalConfig(
        backend="gemini_endpoint",
        gemini_endpoint_project="p",
        eval_samples_per_task=3,
        max_concurrent_samples=17,
        max_generation_tokens=1234,
        thinking_level="HIGH",
        log_dir="logs/custom",
        vertex_experiment="exp-1",
    )
    run_config = rge._forward_to_run_config(cfg, "/tmp/x.jsonl")

    assert run_config.task.eval_data_path == "/tmp/x.jsonl"
    assert run_config.task.vertex_experiment == "exp-1"
    assert run_config.eval.eval_samples_per_task == cfg.eval_samples_per_task
    assert run_config.eval.max_eval_items == -1
    assert run_config.rollout.max_concurrent_samples == cfg.max_concurrent_samples
    assert isinstance(run_config.rollout, AsyncRolloutConfig)
    assert run_config.agent.max_generation_tokens == cfg.max_generation_tokens
    assert run_config.agent.thinking_level == cfg.thinking_level
    assert run_config.train.log_dir == cfg.log_dir


def test_main_forwards_and_cleans_up(monkeypatch, tmp_path):
    generated = tmp_path / "gen.jsonl"
    generated.write_text("{}\n")

    monkeypatch.setattr(rge, "_generate_temp_jsonl", lambda config: str(generated))

    captured = {}

    async def fake_run_eval(run_config, **kwargs):
        captured["run_config"] = run_config

    monkeypatch.setattr(rge.eval_config_entry, "run_eval", fake_run_eval)

    cfg = rge.GeneratedEvalConfig(
        backend="gemini_endpoint",
        gemini_endpoint_project="p",
        keep_temp=False,
    )
    rge.main(cfg)

    assert isinstance(captured["run_config"], RunConfig)
    assert captured["run_config"].task.eval_data_path == str(generated)
    # Temp file cleaned up (keep_temp=False).
    assert not generated.exists()


def test_main_keeps_temp_when_requested(monkeypatch, tmp_path):
    generated = tmp_path / "gen.jsonl"
    generated.write_text("{}\n")

    monkeypatch.setattr(rge, "_generate_temp_jsonl", lambda config: str(generated))

    async def fake_run_eval(run_config, **kwargs):
        pass

    monkeypatch.setattr(rge.eval_config_entry, "run_eval", fake_run_eval)

    cfg = rge.GeneratedEvalConfig(
        backend="gemini_endpoint",
        gemini_endpoint_project="p",
        keep_temp=True,
    )
    rge.main(cfg)

    # Temp file preserved (keep_temp=True).
    assert generated.exists()
