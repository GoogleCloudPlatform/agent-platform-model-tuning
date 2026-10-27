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

"""Composed RunConfig standalone-eval orchestrator: ``run_eval(RunConfig)``.

Additive sibling of ``test_train_config_entry`` for the eval path. Unit tests
drive the ``make()``-sequence with the backend / task / reward / agent /
logging / eval seams monkeypatched to stand-ins so no real GCP / IO happens,
and assert that ``EvalPipeline`` is constructed with the right agent,
eval_dataset, config, and ml_logger from the composed ``RunConfig``.
"""

from __future__ import annotations

import dataclasses

import pytest

from config_registry import get_config

import eval_config_entry as ece


class _FakeSampler:
    pass


class _FakeTrainer:
    pass


class _FakeBundle:
    def __init__(self):
        self.sampler = _FakeSampler()
        self.trainer = _FakeTrainer()
        self.job_id = "job-123"
        self.closed = False

    async def aclose(self):
        self.closed = True


class _FakeDataset:
    def __len__(self) -> int:
        return 5


class _FakeAgent:
    pass


class _FakeMlLogger:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def _patch_common_seams(monkeypatch, cfg, bundle, *, agent, recorded):
    """Patch the shared make()/logging/eval seams used by every test."""

    async def fake_backend_make(self):
        recorded["resume_from_train_step"] = 0
        return bundle

    train_ds, eval_ds = _FakeDataset(), _FakeDataset()
    reward_fn = object()

    monkeypatch.setattr(type(cfg.backend), "make", fake_backend_make)
    monkeypatch.setattr(type(cfg.task), "make", lambda self: (train_ds, eval_ds))
    # The eval path calls make_eval() (eval-only; never loads the train split).
    monkeypatch.setattr(type(cfg.task), "make_eval", lambda self: eval_ds)
    monkeypatch.setattr(type(cfg.reward), "make", lambda self: reward_fn)

    def fake_agent_make(self, *, sampler, job_id, reward_fn, atif_writer=None):
        recorded["agent_make_sampler"] = sampler
        recorded["agent_make_job_id"] = job_id
        recorded["agent_make_reward_fn"] = reward_fn
        return agent

    monkeypatch.setattr(type(cfg.agent), "make", fake_agent_make)

    # Mock EvalPipeline: capture constructor args + mock run().
    class _FakeEvalPipeline:
        def __init__(self, *, agent, config, ml_logger, eval_dataset=None, **kw):
            recorded["eval_agent"] = agent
            recorded["eval_dataset"] = eval_dataset
            recorded["report_view"] = config
            recorded["ml_logger"] = ml_logger

        async def run(self, step_num=0, *, on_group=None):
            return {"eval/accuracy": 1.0}

    monkeypatch.setattr(ece, "EvalPipeline", _FakeEvalPipeline)
    monkeypatch.setattr(ece, "build_atif_writer", lambda **kw: None)

    fake_logger = _FakeMlLogger()
    monkeypatch.setattr(ece, "setup_logging", lambda **kw: fake_logger)

    recorded["_train_ds"] = train_ds
    recorded["_eval_ds"] = eval_ds
    recorded["_reward_fn"] = reward_fn
    recorded["_fake_logger"] = fake_logger
    return recorded


@pytest.mark.asyncio
async def test_run_eval_drives_make_sequence(monkeypatch):
    cfg = get_config("word_puzzle/async")
    bundle = _FakeBundle()
    agent = _FakeAgent()
    recorded: dict = {}
    _patch_common_seams(monkeypatch, cfg, bundle, agent=agent, recorded=recorded)

    await ece.run_eval(cfg, recipe="word_puzzle")

    # Backend.make() drove the bundle; resume defaults to 0 on the eval path.
    assert recorded["resume_from_train_step"] == 0
    # agent.make(...) got the bundle's sampler, the bundle job_id, and the
    # reward_fn built from config.reward.make().
    assert recorded["agent_make_sampler"] is bundle.sampler
    assert recorded["agent_make_job_id"] == bundle.job_id
    assert recorded["agent_make_reward_fn"] is recorded["_reward_fn"]
    # eval_and_report got the composed eval_dataset (the SECOND of the pair)
    # and the agent produced by agent.make().
    assert recorded["eval_dataset"] is recorded["_eval_ds"]
    assert recorded["eval_agent"] is agent
    assert recorded["ml_logger"] is recorded["_fake_logger"]
    # lifecycle: bundle closed, ml_logger closed.
    assert bundle.closed is True
    assert recorded["_fake_logger"].closed is True


@pytest.mark.asyncio
async def test_run_eval_passes_run_config_directly(monkeypatch):
    cfg = get_config("word_puzzle/async")
    bundle = _FakeBundle()
    recorded: dict = {}
    _patch_common_seams(monkeypatch, cfg, bundle, agent=_FakeAgent(), recorded=recorded)

    await ece.run_eval(cfg, recipe="word_puzzle")

    config_passed = recorded["report_view"]
    # eval_and_report now receives the RunConfig directly (no flat adapter).
    assert hasattr(config_passed, "eval")
    assert hasattr(config_passed, "rollout")
    assert hasattr(config_passed, "agent")
    # Verify the nested fields are readable through the config.
    assert config_passed.eval.eval_samples_per_task == cfg.eval.eval_samples_per_task
    assert config_passed.agent.thinking_level == cfg.agent.thinking_level
    assert config_passed.eval.max_eval_items == cfg.eval.max_eval_items
    assert (
        config_passed.rollout.max_concurrent_samples
        == cfg.rollout.max_concurrent_samples
    )


@pytest.mark.asyncio
async def test_run_eval_does_not_pass_explicit_log_step(monkeypatch):
    """The composed eval orchestrator no longer passes an explicit log_step;
    run_eval auto-derives the x-axis from the response's trainStepId
    (tested in test_eval_only.py)."""
    cfg = get_config("word_puzzle/async")
    bundle = _FakeBundle()
    recorded: dict = {}
    _patch_common_seams(monkeypatch, cfg, bundle, agent=_FakeAgent(), recorded=recorded)

    await ece.run_eval(cfg, recipe="word_puzzle")

    assert "log_step" not in recorded


@pytest.mark.asyncio
async def test_run_eval_passes_resolved_vertex_tensorboard_and_run_name(monkeypatch):
    """The composed word_puzzle eval path resolves the task's
    ``vertex_experiment_tensorboard`` string via
    ``resolve_vertex_tensorboard_config`` ("disable" -> False) and passes the
    task's ``vertex_experiment_run_name`` straight through into
    ``setup_logging``."""
    cfg = get_config("word_puzzle/async")
    cfg = dataclasses.replace(
        cfg,
        task=dataclasses.replace(
            cfg.task,
            vertex_experiment_tensorboard="disable",
            vertex_experiment_run_name="myrun",
        ),
    )
    bundle = _FakeBundle()
    recorded: dict = {}
    _patch_common_seams(monkeypatch, cfg, bundle, agent=_FakeAgent(), recorded=recorded)

    captured = {}

    def fake_setup_logging(**kw):
        captured.update(kw)
        return _FakeMlLogger()

    monkeypatch.setattr(ece, "setup_logging", fake_setup_logging)

    await ece.run_eval(cfg, recipe="word_puzzle")

    assert captured["vertex_tensorboard"] is False
    # Run name is prefixed with the tuning job id (bundle.job_id="job-123")
    # for at-a-glance traceability: "{job_id}-{run_name}".
    assert captured["vertex_run_name"] == "job-123-myrun"


@pytest.mark.asyncio
async def test_run_eval_passes_recipe_to_atif(monkeypatch):
    """The ``recipe`` arg flows through to the ATIF writer (build_atif_writer
    ``recipe=``) and the on_job_completed payload — the eval orchestrator is
    recipe-agnostic; the label is supplied by the caller."""
    cfg = get_config("word_puzzle/async")
    bundle = _FakeBundle()
    recorded: dict = {}
    _patch_common_seams(monkeypatch, cfg, bundle, agent=_FakeAgent(), recorded=recorded)

    captured = {}

    class _RecordingAtif:
        gcs_uri: str = ""
        job_dir: str = "/tmp/fake-atif"

        async def on_job_started(self, config):
            pass

        async def on_job_completed(self, payload):
            captured["on_job_completed"] = payload

        # The entry point drains on every exit, including the failure path.
        # It deliberately does NOT reclaim other runs (cross-job coupling).
        async def drain(self, timeout_s: float = 60.0):
            captured["drained"] = True
            return 0

        def deregister_from_reclaim(self):
            captured["deregistered"] = True

    def fake_build_atif_writer(*, recipe, **kw):
        captured["build_recipe"] = recipe
        return _RecordingAtif()

    monkeypatch.setattr(ece, "build_atif_writer", fake_build_atif_writer)

    await ece.run_eval(cfg, recipe="word_puzzle")

    assert captured["build_recipe"] == "word_puzzle"
    assert captured["on_job_completed"]["recipe"] == "word_puzzle"
