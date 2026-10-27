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

import dataclasses

import pytest

from backends.backend_config import VertexMsrlBackendConfig
from core.launch_overrides import (
    EvalLaunchOverrides,
    TrainLaunchOverrides,
    apply_overrides,
)
from config_registry import get_config
from core.run_config import TrainConfig


@dataclasses.dataclass(frozen=True)
class _FakeTop:
    """Minimal stand-in mirroring the new RunConfig sub-module names the
    ``_ROUTING`` table targets (``train`` / ``backend``)."""

    train: TrainConfig
    backend: VertexMsrlBackendConfig


def _base():
    return _FakeTop(train=TrainConfig(), backend=VertexMsrlBackendConfig())


# ── TrainLaunchOverrides ─────────────────────────────────────────────


def test_unset_overrides_leave_config_untouched():
    ov = TrainLaunchOverrides(config_name="x")  # all essentials None
    merged = apply_overrides(_base(), ov)
    assert merged.backend.project == ""  # baked default kept
    assert merged.train.resume_from_train_step == 0


def test_set_override_replaces_only_that_field():
    ov = TrainLaunchOverrides(
        config_name="x", vertex_project="my-proj", log_dir="/tmp/run"
    )
    merged = apply_overrides(_base(), ov)
    assert merged.backend.project == "my-proj"
    assert merged.train.log_dir == "/tmp/run"
    assert merged.backend.location == "us-central1"  # untouched


def test_resume_zero_is_distinguishable_from_unset():
    ov = TrainLaunchOverrides(config_name="x", resume_from_train_step=0)
    merged = apply_overrides(_base(), ov)
    assert merged.train.resume_from_train_step == 0
    ov2 = TrainLaunchOverrides(config_name="x", resume_from_train_step=40)
    assert apply_overrides(_base(), ov2).train.resume_from_train_step == 40


def test_vertex_existing_job_id_routes_to_backend():
    ov = TrainLaunchOverrides(config_name="x", vertex_existing_job_id="999")
    merged = apply_overrides(_base(), ov)
    assert merged.backend.vertex_existing_job_id == "999"


def test_num_train_steps_routes_to_train():
    """num_train_steps overrides TrainConfig.num_train_steps."""
    ov = TrainLaunchOverrides(config_name="x", num_train_steps=35)
    merged = apply_overrides(_base(), ov)
    assert merged.train.num_train_steps == 35


def test_eval_size_overrides_route_to_eval():
    """max_eval_items / eval_samples_per_task on TrainLaunchOverrides land on
    the composed config's EvalConfig, which sizes every live-sampler sweep of
    a training run (step-0 baseline, cadence, final)."""
    cfg = get_config("word_puzzle/async")
    merged = apply_overrides(
        cfg,
        TrainLaunchOverrides(
            config_name="word_puzzle/async",
            max_eval_items=3,
            eval_samples_per_task=1,
        ),
    )
    assert merged.eval.max_eval_items == 3
    assert merged.eval.eval_samples_per_task == 1
    assert merged.eval.eval_every_n_train_steps == cfg.eval.eval_every_n_train_steps


def test_eval_size_overrides_unset_keep_recipe_values():
    cfg = get_config("word_puzzle/async")
    merged = apply_overrides(
        cfg, TrainLaunchOverrides(config_name="word_puzzle/async")
    )
    assert merged.eval == cfg.eval


def test_max_eval_items_minus_one_means_all():
    cfg = get_config("word_puzzle/async")
    merged = apply_overrides(
        cfg,
        TrainLaunchOverrides(config_name="word_puzzle/async", max_eval_items=-1),
    )
    assert merged.eval.max_eval_items == -1


def test_eval_samples_per_task_zero_rejected_by_eval_config_validation():
    with pytest.raises(ValueError, match="eval_samples_per_task must be >= 1"):
        apply_overrides(
            get_config("word_puzzle/async"),
            TrainLaunchOverrides(
                config_name="word_puzzle/async", eval_samples_per_task=0
            ),
        )


def test_customer_train_entry_point_exposes_every_train_override():
    """src/train.py main() takes every TrainLaunchOverrides field as a
    keyword arg, so the launch surface cannot silently drift."""
    import importlib.util
    import inspect
    import pathlib

    import chz

    path = pathlib.Path(__file__).resolve().parents[2] / "src" / "train.py"
    spec = importlib.util.spec_from_file_location("customer_train_entry", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    fields = set(chz.chz_fields(TrainLaunchOverrides))
    params = set(inspect.signature(mod.main).parameters)
    assert fields <= params, fields - params


def test_customer_train_main_forwards_eval_size_overrides(monkeypatch):
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[2] / "src" / "train.py"
    spec = importlib.util.spec_from_file_location("customer_train_entry2", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    captured = {}

    async def fake_run_training(config, *, recipe, **kwargs):
        captured["config"] = config

    def fake_asyncio_run(coro):
        try:
            coro.send(None)
        except StopIteration:
            pass

    monkeypatch.setattr(mod, "run_training", fake_run_training)
    monkeypatch.setattr(mod.asyncio, "run", fake_asyncio_run)
    mod.main(
        config_name="word_puzzle/async", max_eval_items=2, eval_samples_per_task=1
    )
    assert captured["config"].eval.max_eval_items == 2
    assert captured["config"].eval.eval_samples_per_task == 1


# ── EvalLaunchOverrides ──────────────────────────────────────────────


def test_eval_overrides_route_eval_data_path():
    """eval_data_path on EvalLaunchOverrides routes to task.eval_data_path."""
    cfg = get_config("word_puzzle/async")
    ov = EvalLaunchOverrides(
        config_name="word_puzzle/async", eval_data_path="/data/eval.jsonl"
    )
    merged = apply_overrides(cfg, ov)
    assert merged.task.eval_data_path == "/data/eval.jsonl"


def test_eval_overrides_have_no_train_fields():
    """EvalLaunchOverrides should NOT have training-only fields."""
    assert not hasattr(EvalLaunchOverrides, "vertex_base_model")
    assert not hasattr(EvalLaunchOverrides, "log_dir")
    assert not hasattr(EvalLaunchOverrides, "resume_from_train_step")
    assert not hasattr(EvalLaunchOverrides, "resume_metrics_path")
    assert not hasattr(EvalLaunchOverrides, "train_data_path")
    assert not hasattr(EvalLaunchOverrides, "num_train_steps")


def test_eval_overrides_have_jsonl_output_path():
    ov = EvalLaunchOverrides(config_name="x", jsonl_output_path="/out/eval.jsonl")
    assert ov.jsonl_output_path == "/out/eval.jsonl"


# ── target parsing + routing ─────────────────────────────────────────


def test_target_tuning_job_overrides_backend():
    """A full tuning-job resource path sets project, location, job id."""
    cfg = get_config("word_puzzle/async")
    ov = EvalLaunchOverrides(
        config_name="word_puzzle/async",
        target="projects/my-proj/locations/us-central1/tuningJobs/12345",
    )
    merged = apply_overrides(cfg, ov)
    assert merged.backend.project == "my-proj"
    assert merged.backend.location == "us-central1"
    assert merged.backend.vertex_existing_job_id == "12345"


def test_target_short_model_sets_model():
    """A short model name sets backend.model on GeminiApiBackendConfig."""
    from backends.backend_config import GeminiApiBackendConfig

    @dataclasses.dataclass(frozen=True)
    class _GeminiTop:
        backend: GeminiApiBackendConfig

    cfg = _GeminiTop(backend=GeminiApiBackendConfig(model="old-model"))
    ov = EvalLaunchOverrides(
        config_name="x",
        target="gemini-3.5-flash",
    )
    merged = apply_overrides(cfg, ov)
    assert merged.backend.model == "gemini-3.5-flash"


def test_target_short_model_noop_on_vertex_msrl():
    """A short model name is silently ignored on VertexMsrlBackendConfig
    (which has no ``model`` field)."""
    cfg = get_config("word_puzzle/async")
    ov = EvalLaunchOverrides(
        config_name="word_puzzle/async",
        target="gemini-3.5-flash",
    )
    merged = apply_overrides(cfg, ov)
    # No crash, and no model field set — backend is unchanged.
    assert not hasattr(merged.backend, "model")


def test_target_overrides_explicit_project():
    """target's embedded project wins over explicit vertex_project."""
    cfg = get_config("word_puzzle/async")
    ov = EvalLaunchOverrides(
        config_name="word_puzzle/async",
        vertex_project="from-explicit-arg",
        target="projects/from-target/locations/us-west1/tuningJobs/42",
    )
    merged = apply_overrides(cfg, ov)
    # target is applied AFTER explicit fields, so it wins:
    assert merged.backend.project == "from-target"
    assert merged.backend.location == "us-west1"


# ── Config registry ─────────────────────────────────────────────────


def test_get_returns_registered_config():
    cfg = get_config("word_puzzle/async")
    assert cfg is not None


def test_unknown_config_name_fails_loudly():
    with pytest.raises(KeyError):
        get_config("does_not_exist_xyz")


# ── Routing onto the real composed RunConfig ─────────────────────────


def test_apply_overrides_routes_to_new_modules():
    cfg = get_config("word_puzzle/async")
    merged = apply_overrides(
        cfg,
        TrainLaunchOverrides(
            config_name="word_puzzle/async",
            vertex_project="my-proj",
            log_dir="/tmp/x",
            resume_from_train_step=3,
        ),
    )

    assert merged.backend.project == "my-proj"
    assert merged.train.log_dir == "/tmp/x"
    assert merged.train.resume_from_train_step == 3
    # Untouched fields keep their registered values (async tuning):
    assert merged.train.sync_every_n_train_steps == 4
    assert merged.backend.location == cfg.backend.location
    # The original config is not mutated (replace returns a copy):
    assert cfg.train.log_dir != "/tmp/x"


def test_apply_overrides_routes_task_and_backend_fields():
    cfg = get_config("word_puzzle/async")
    merged = apply_overrides(
        cfg,
        TrainLaunchOverrides(
            config_name="word_puzzle/async",
            train_data_path="/tmp/train.jsonl",
            eval_data_path="/tmp/eval.jsonl",
            vertex_project="my-project",
            vertex_existing_job_id="999",
        ),
    )
    assert merged.task.train_data_path == "/tmp/train.jsonl"
    assert merged.task.eval_data_path == "/tmp/eval.jsonl"
    assert merged.backend.project == "my-project"
    assert merged.backend.vertex_existing_job_id == "999"


def test_registered_config_has_real_trajectory_group_filter():
    """Construction-readiness: the registered RunConfig carries a real
    trajectory_group_filter (not None) so the pipeline's ``.make()`` call
    on it does not hit AttributeError."""
    cfg = get_config("word_puzzle/async")
    assert cfg.rollout.trajectory_group_filter is not None
    # It must be constructible into a live filter:
    assert callable(cfg.rollout.trajectory_group_filter.make())
