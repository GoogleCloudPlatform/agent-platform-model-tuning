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

"""RunConfig orchestrator (D5): ``run_training(RunConfig)``.

Unit tests drive the ``make()``-sequence with the backend / task / logging
seams monkeypatched to stand-ins so no real GCP / IO happens, and assert
that ``train(...)`` is called with a pipeline-ready composed config plus the
agent/trainer produced by the backend bundle.
"""

from __future__ import annotations

import dataclasses
import logging

import pytest

from config_registry import get_config

import train_config_entry as rce


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
    pass


class _FakeMlLogger:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_run_training_drives_make_sequence(monkeypatch):
    cfg = get_config("word_puzzle/async")

    bundle = _FakeBundle()

    async def fake_backend_make(self):
        return bundle

    train_ds, eval_ds = _FakeDataset(), _FakeDataset()

    monkeypatch.setattr(type(cfg.backend), "make", fake_backend_make)
    monkeypatch.setattr(type(cfg.task), "make", lambda self: (train_ds, eval_ds))

    recorded = {}

    async def fake_train(
        *,
        agent,
        config,
        train_dataset,
        eval_dataset,
        trainer,
        ml_logger,
        eval_agent=None,
        progress_observer=None,
    ):
        recorded["agent"] = agent
        recorded["config"] = config
        recorded["train_dataset"] = train_dataset
        recorded["eval_dataset"] = eval_dataset
        recorded["trainer"] = trainer
        recorded["ml_logger"] = ml_logger

    monkeypatch.setattr(rce, "train", fake_train)
    monkeypatch.setattr(rce, "build_atif_writer", lambda **kw: None)

    fake_logger = _FakeMlLogger()
    monkeypatch.setattr(rce, "setup_logging", lambda **kw: fake_logger)

    await rce.run_training(cfg, recipe="word_puzzle")

    # train() got the bundle's trainer + the composed datasets:
    assert recorded["trainer"] is bundle.trainer
    assert recorded["train_dataset"] is train_ds
    assert recorded["eval_dataset"] is eval_ds
    # train() got a pipeline-ready RunConfig (has rollout.make):
    assert hasattr(recorded["config"].rollout, "make")
    assert recorded["ml_logger"] is fake_logger
    # lifecycle: bundle closed, ml_logger closed.
    assert bundle.closed is True
    assert fake_logger.closed is True


@pytest.mark.asyncio
async def test_run_training_passes_resolved_vertex_tensorboard_and_run_name(
    monkeypatch,
):
    """The composed word_puzzle training path resolves the task's
    ``vertex_experiment_tensorboard`` string via
    ``resolve_vertex_tensorboard_config`` ("disable" -> False) and passes the
    task's ``vertex_experiment_run_name`` straight through into
    ``setup_logging`` (parity with the flat path)."""
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

    async def fake_backend_make(self):
        return bundle

    monkeypatch.setattr(type(cfg.backend), "make", fake_backend_make)
    monkeypatch.setattr(
        type(cfg.task), "make", lambda self: (_FakeDataset(), _FakeDataset())
    )

    async def fake_train(**kw):
        pass

    monkeypatch.setattr(rce, "train", fake_train)
    monkeypatch.setattr(rce, "build_atif_writer", lambda **kw: None)

    captured = {}

    def fake_setup_logging(**kw):
        captured.update(kw)
        return _FakeMlLogger()

    monkeypatch.setattr(rce, "setup_logging", fake_setup_logging)

    await rce.run_training(cfg, recipe="word_puzzle")

    # "disable" -> resolve_vertex_tensorboard_config -> False
    assert captured["vertex_tensorboard"] is False
    # Run name is prefixed with the tuning job id (bundle.job_id="job-123")
    # for at-a-glance traceability: "{job_id}-{run_name}".
    assert captured["vertex_run_name"] == "job-123-myrun"


@pytest.mark.asyncio
async def test_run_training_passes_recipe_to_atif(monkeypatch):
    """The ``recipe`` arg flows through to the ATIF writer (build_atif_writer
    ``recipe=``) and the on_job_completed payload — the orchestrator is
    recipe-agnostic; the label is supplied by the caller."""
    cfg = get_config("word_puzzle/async")
    bundle = _FakeBundle()

    async def fake_backend_make(self):
        return bundle

    monkeypatch.setattr(type(cfg.backend), "make", fake_backend_make)
    monkeypatch.setattr(
        type(cfg.task), "make", lambda self: (_FakeDataset(), _FakeDataset())
    )

    captured = {}

    class _RecordingAtif:
        gcs_uri = ""
        job_dir = "/fake/job_dir"

        async def on_job_started(self, config):
            pass

        async def on_job_completed(self, payload):
            captured["on_job_completed"] = payload

        def deregister_from_reclaim(self):
            # Retired once the drain above succeeded: nothing is left for a
            # future `--reclaim` to pick up.
            captured["deregistered"] = True

    def fake_build_atif_writer(*, recipe, **kw):
        captured["build_recipe"] = recipe
        return _RecordingAtif()

    async def fake_train(**kw):
        pass

    monkeypatch.setattr(rce, "build_atif_writer", fake_build_atif_writer)
    monkeypatch.setattr(rce, "train", fake_train)
    monkeypatch.setattr(rce, "setup_logging", lambda **kw: _FakeMlLogger())

    await rce.run_training(cfg, recipe="word_puzzle")

    assert captured["build_recipe"] == "word_puzzle"
    assert captured["on_job_completed"]["recipe"] == "word_puzzle"


# ── Half-migration end-to-end acceptance (D5) ────────────────────────


class _FakeIter:
    current = 0


class _FakeAgent:
    pass


class _FakeDatasetWithIter:
    def build_iter(self, **kwargs):
        return _FakeIter()


def test_end_to_end_config_drives_async_pipeline_construction():
    """get -> apply_overrides -> config.rollout.make(...) builds a
    real AsyncRolloutPipeline (with fakes for agent/datasets/ml_logger)
    WITHOUT the None.make() AttributeError. Validates the
    trajectory_group_filter construction-readiness fix."""
    from core.launch_overrides import TrainLaunchOverrides, apply_overrides

    cfg = get_config("word_puzzle/async")
    merged = apply_overrides(
        cfg,
        TrainLaunchOverrides(
            config_name="word_puzzle/async",
            log_dir="/tmp/e2e",
            vertex_project="proj",
        ),
    )

    pipeline = merged.rollout.make(
        run_config=merged,
        agent=_FakeAgent(),
        train_dataset=_FakeDatasetWithIter(),
        eval_dataset=_FakeDatasetWithIter(),
        eval_agent=None,
        ml_logger=_FakeMlLogger(),
        resume_train_iter_position=None,
    )

    from pipeline.async_rollout_pipeline import AsyncRolloutPipeline

    assert isinstance(pipeline, AsyncRolloutPipeline)


# ---------------------------------------------------------------------------
# The experiment record: it must be emitted on the ONE-backend path too
# ---------------------------------------------------------------------------


class _RecordingObserver:
    """Captures the kwargs `on_experiment_started` was called with."""

    def __init__(self):
        self.started: dict | None = None

    async def on_experiment_started(self, **fields):
        self.started = fields

    async def on_job_completed(self, *args, **kwargs):
        pass


@pytest.mark.asyncio
async def test_single_backend_run_still_registers_the_experiment(monkeypatch):
    """No `auxiliary_backend()` hook must still produce an experiment record.

    The observer call is wrapped in `except Exception: warn`, so a bug in the
    field-building block degrades to one WARNING line and the run proceeds
    with no experiment in the monitor AT ALL. That is invisible to every
    other test here, none of which passes a `progress_observer`. It has
    already happened once (a name bound only on the two-backend branch, read
    unconditionally), so pin it: fires, and the auxiliary fields are empty.
    """
    cfg = get_config("word_puzzle/async")
    bundle = _FakeBundle()

    async def fake_backend_make(self):
        return bundle

    monkeypatch.setattr(type(cfg.backend), "make", fake_backend_make)
    monkeypatch.setattr(
        type(cfg.task), "make", lambda self: (_FakeDataset(), _FakeDataset())
    )

    async def fake_train(**kw):
        pass

    monkeypatch.setattr(rce, "train", fake_train)
    monkeypatch.setattr(rce, "build_atif_writer", lambda **kw: None)
    monkeypatch.setattr(rce, "setup_logging", lambda **kw: _FakeMlLogger())

    # The recipe under test has no auxiliary backend, which is the point.
    assert not hasattr(cfg.agent, "auxiliary_backend")

    observer = _RecordingObserver()
    await rce.run_training(cfg, recipe="word_puzzle", progress_observer=observer)

    assert observer.started is not None, "on_experiment_started never fired"
    assert observer.started["tuning_job_id"] == bundle.job_id
    assert observer.started["auxiliary_job_id"] == ""
    assert observer.started["auxiliary_backend"] == ""
    assert observer.started["auxiliary_env"] == ""


def test_experiment_fields_name_the_second_job_when_there_is_one():
    """A two-backend run's record must describe BOTH jobs.

    The id alone is not enough: the second backend is configured separately,
    so it can sit in a different serving environment, and a tuning job's
    resource name is identical across environments.
    """
    from types import SimpleNamespace

    from utils.experiment_fields import build_experiment_fields

    cfg = get_config("word_puzzle/async")
    aux = SimpleNamespace(short_name="vertex_msrl", vertex_env="staging")

    fields = build_experiment_fields(
        cfg,
        effective_job_id="job-123-20260101_000000",
        recipe="word_puzzle",
        bundle_job_id="job-123",
        total_train_steps=1,
        auxiliary_job_id="job-456",
        auxiliary_backend_config=aux,
    )

    assert fields["tuning_job_id"] == "job-123"
    assert fields["auxiliary_job_id"] == "job-456"
    assert fields["auxiliary_backend"] == "vertex_msrl"
    assert fields["auxiliary_env"] == "staging"


def test_experiment_field_paths_use_the_writers_sanitised_recipe():
    """A recipe label with a separator must not be interpolated raw.

    `AtifDiskWriter` stores under the sanitised segment, so a raw name here
    advertises a GCS dir and a viewer URL nothing was written to -- and the
    viewer route takes one path segment, not two.
    """
    from utils.experiment_fields import build_experiment_fields

    cfg = get_config("word_puzzle/async")
    cfg = dataclasses.replace(
        cfg, task=dataclasses.replace(cfg.task, atif_output="gs://bucket")
    )

    fields = build_experiment_fields(
        cfg,
        effective_job_id="job-1",
        recipe="family/bench",
        bundle_job_id="job-1",
        total_train_steps=1,
        atif_viewer_base_url="https://viewer.example.com",
    )

    assert fields["gcs_output_dir"] == "gs://bucket/family_bench/job-1"
    assert (
        fields["atif_viewer_base_url"]
        == "https://viewer.example.com/projects/family_bench/jobs/job-1"
    )
