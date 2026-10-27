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

"""Tests for the BackendConfig ABC and concrete subclasses."""

from __future__ import annotations

import asyncio
import getpass

import pytest

from backends.backend_config import (
    BackendConfig,
    GeminiApiBackendConfig,
    SyncRetryProfile,
    VertexMsrlBackendConfig,
    resolve_job_author,
)
from backends.factory import BackendBundle
from utils.experiment_fields import resolve_owner


@pytest.fixture(autouse=True)
def _no_owner_env(monkeypatch):
    """Launchers export VMSRL_OWNER / VERTEX_JOB_AUTHOR on VMs; no test in
    this module may inherit them."""
    monkeypatch.delenv("VMSRL_OWNER", raising=False)
    monkeypatch.delenv("VERTEX_JOB_AUTHOR", raising=False)


def test_base_is_abstract():
    with pytest.raises(TypeError):
        BackendConfig()  # abstract: make() not implemented


def test_base_field_defaults():
    class _FakeBackendConfig(BackendConfig):
        async def make(self) -> BackendBundle:
            return BackendBundle(sampler=None, trainer=None, job_id="j")

    c = _FakeBackendConfig()
    assert c.request_timeout == 600.0
    assert c.service_account == ""


def test_concrete_make_returns_backend_bundle():
    class _FakeBackendConfig(BackendConfig):
        async def make(self) -> BackendBundle:
            return BackendBundle(sampler="s", trainer="t", job_id="j")

    bundle = asyncio.run(_FakeBackendConfig().make())
    assert isinstance(bundle, BackendBundle)
    assert bundle.job_id == "j"


# ---- VertexMsrlBackendConfig -------------------------------------------


class _FakeSampler:
    """Module-scope stub for VertexMsrlRestSampler used across make() tests."""

    def __init__(self, client, **kwargs):
        self.client = client
        self.kwargs = kwargs


class _FakeTrainer:
    """Module-scope stub for VertexMsrlRestTrainer used across make() tests."""

    def __init__(self, client, **kwargs):
        self.client = client
        self.kwargs = kwargs


def test_vertex_msrl_is_concrete_backend_config():
    cfg = VertexMsrlBackendConfig()
    assert isinstance(cfg, BackendConfig)


def test_vertex_msrl_field_defaults():
    cfg = VertexMsrlBackendConfig()
    # Inherited base fields.
    assert cfg.request_timeout == 600.0
    assert cfg.service_account == ""
    # Vertex fields.
    assert cfg.project == ""
    assert cfg.location == "us-central1"
    assert cfg.vertex_base_model == "gemini-3.5-flash"
    assert cfg.vertex_existing_job_id == ""
    assert cfg.vertex_job_running_timeout_seconds == 7200
    assert cfg.vertex_checkpoint_interval == 0
    assert cfg.vertex_request_qpm == 1800.0


def test_vertex_msrl_make_returns_bundle_with_stubbed_seams(monkeypatch):
    """make() returns a BackendBundle when the client/sampler/trainer
    constructors are stubbed out (no real GCP infra needed)."""
    import backends.vertex_msrl_rest as vmr

    captured = {}

    class _FakeClient:
        tuning_job_id = "12345"

        @property
        def tuning_job_name(self):
            return f"projects/p/locations/l/tuningJobs/{self.tuning_job_id}"

        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs

        async def create_tuning_job(self, **kwargs):
            captured["create_kwargs"] = kwargs

        async def wait_for_job_running(self, timeout):
            captured["wait_timeout"] = timeout

        async def cancel_tuning_job(self):
            pass

        async def aclose(self):
            pass

    monkeypatch.setattr(vmr, "VertexMsrlRestClient", _FakeClient)
    monkeypatch.setattr(vmr, "VertexMsrlRestSampler", _FakeSampler)
    monkeypatch.setattr(vmr, "VertexMsrlRestTrainer", _FakeTrainer)

    cfg = VertexMsrlBackendConfig(project="p")
    bundle = asyncio.run(cfg.make())

    assert isinstance(bundle, BackendBundle)
    assert bundle.job_id == "12345"
    assert isinstance(bundle.sampler, _FakeSampler)
    assert isinstance(bundle.trainer, _FakeTrainer)
    # max_generation_tokens is no longer a backend field: the sampler is
    # built WITHOUT a config-sourced override.
    assert "max_generation_tokens" not in bundle.sampler.kwargs


def test_vertex_msrl_make_logs_precreate_job_counts(monkeypatch, caplog):
    """The create path logs a RUNNING/PENDING congestion summary, calling
    the count RPC exactly once."""
    import backends.vertex_msrl_rest as vmr

    calls = {"count": 0}

    class _FakeClient:
        tuning_job_id = ""
        parent = "projects/p/locations/us-central1"
        base_model = None

        @property
        def tuning_job_name(self):
            return f"{self.parent}/tuningJobs/{self.tuning_job_id}"

        def __init__(self, **kwargs):
            pass

        async def count_tuning_jobs_by_state(self, **kwargs):
            calls["count"] += 1
            return {"JOB_STATE_RUNNING": 3, "JOB_STATE_PENDING": 1}

        async def create_tuning_job(self, **kwargs):
            type(self).tuning_job_id = "job-123"
            return "job-123"

        async def wait_for_job_running(self, **kwargs):
            return {"state": "JOB_STATE_RUNNING"}

        async def cancel_tuning_job(self):
            pass

        async def aclose(self):
            pass

    monkeypatch.setattr(vmr, "VertexMsrlRestClient", _FakeClient)
    monkeypatch.setattr(vmr, "VertexMsrlRestSampler", _FakeSampler)
    monkeypatch.setattr(vmr, "VertexMsrlRestTrainer", _FakeTrainer)

    cfg = VertexMsrlBackendConfig(project="p")
    with caplog.at_level("INFO"):
        bundle = asyncio.run(cfg.make())

    assert calls["count"] == 1
    assert bundle.job_id == "job-123"
    joined = " ".join(r.message for r in caplog.records)
    assert "RUNNING=3" in joined or "JOB_STATE_RUNNING=3" in joined


def test_vertex_msrl_make_precreate_count_failure_does_not_block(monkeypatch):
    """A listing failure in the pre-create congestion check must never block
    the launch."""
    import backends.vertex_msrl_rest as vmr

    class _FakeClient:
        tuning_job_id = ""
        parent = "projects/p/locations/us-central1"
        base_model = None

        @property
        def tuning_job_name(self):
            return f"{self.parent}/tuningJobs/{self.tuning_job_id}"

        def __init__(self, **kwargs):
            pass

        async def count_tuning_jobs_by_state(self, **kwargs):
            raise RuntimeError("list failed")

        async def create_tuning_job(self, **kwargs):
            type(self).tuning_job_id = "job-xyz"
            return "job-xyz"

        async def wait_for_job_running(self, **kwargs):
            return {"state": "JOB_STATE_RUNNING"}

        async def cancel_tuning_job(self):
            pass

        async def aclose(self):
            pass

    monkeypatch.setattr(vmr, "VertexMsrlRestClient", _FakeClient)
    monkeypatch.setattr(vmr, "VertexMsrlRestSampler", _FakeSampler)
    monkeypatch.setattr(vmr, "VertexMsrlRestTrainer", _FakeTrainer)

    cfg = VertexMsrlBackendConfig(project="p")
    bundle = asyncio.run(cfg.make())  # must NOT raise
    assert bundle.job_id == "job-xyz"


def test_vertex_msrl_make_precreate_count_failure_logs_warning(monkeypatch, caplog):
    """A listing failure logs a diagnostic warning (and still launches)."""
    import backends.vertex_msrl_rest as vmr

    class _FakeClient:
        tuning_job_id = ""
        parent = "projects/p/locations/us-central1"
        base_model = None

        @property
        def tuning_job_name(self):
            return f"{self.parent}/tuningJobs/{self.tuning_job_id}"

        def __init__(self, **kwargs):
            pass

        async def count_tuning_jobs_by_state(self, **kwargs):
            raise RuntimeError("list failed")

        async def create_tuning_job(self, **kwargs):
            type(self).tuning_job_id = "job-warn"
            return "job-warn"

        async def wait_for_job_running(self, **kwargs):
            return {"state": "JOB_STATE_RUNNING"}

        async def cancel_tuning_job(self):
            pass

        async def aclose(self):
            pass

    monkeypatch.setattr(vmr, "VertexMsrlRestClient", _FakeClient)
    monkeypatch.setattr(vmr, "VertexMsrlRestSampler", _FakeSampler)
    monkeypatch.setattr(vmr, "VertexMsrlRestTrainer", _FakeTrainer)

    cfg = VertexMsrlBackendConfig(project="p")
    with caplog.at_level("WARNING"):
        bundle = asyncio.run(cfg.make())

    assert bundle.job_id == "job-warn"
    joined = " ".join(r.message for r in caplog.records)
    assert "congestion check skipped" in joined


def test_vertex_msrl_make_reuse_path_skips_count(monkeypatch):
    """Reusing an existing job id must skip the pre-create count entirely."""
    import backends.vertex_msrl_rest as vmr

    calls = {"count": 0}

    class _FakeClient:
        parent = "projects/p/locations/us-central1"
        base_model = None

        @property
        def tuning_job_name(self):
            return f"{self.parent}/tuningJobs/{self.tuning_job_id}"

        def __init__(self, **kwargs):
            type(self).tuning_job_id = kwargs.get("tuning_job_id", "")

        async def count_tuning_jobs_by_state(self, **kwargs):
            calls["count"] += 1
            return {}

        async def create_tuning_job(self, **kwargs):
            raise AssertionError("must not create on reuse path")

        async def wait_for_job_running(self, **kwargs):
            return {"state": "JOB_STATE_RUNNING"}

        async def cancel_tuning_job(self):
            pass

        async def aclose(self):
            pass

    monkeypatch.setattr(vmr, "VertexMsrlRestClient", _FakeClient)
    monkeypatch.setattr(vmr, "VertexMsrlRestSampler", _FakeSampler)
    monkeypatch.setattr(vmr, "VertexMsrlRestTrainer", _FakeTrainer)

    cfg = VertexMsrlBackendConfig(project="p", vertex_existing_job_id="job-existing")
    asyncio.run(cfg.make())
    assert calls["count"] == 0


# ---- GeminiApiBackendConfig --------------------------------------------


def test_gemini_api_is_concrete_backend_config():
    cfg = GeminiApiBackendConfig()
    assert isinstance(cfg, BackendConfig)


def test_gemini_api_field_defaults():
    cfg = GeminiApiBackendConfig()
    # Inherited base fields.
    assert cfg.request_timeout == 600.0
    assert cfg.service_account == ""
    # Backend-specific fields.
    assert cfg.model == "gemini-3.5-flash"
    assert cfg.project == ""
    assert cfg.env == "prod"
    assert cfg.location == "global"
    assert cfg.use_user_credentials is False
    assert cfg.api_version == "v1beta1"
    assert cfg.endpoint_id == ""
    # Retry policy, by shape. `max_tries: int = 10` became
    # `retry.max_attempts`, same value, same schedule.
    assert cfg.retry == SyncRetryProfile()
    assert cfg.retry.max_attempts == 10
    assert cfg.retry.deadline_s is None


def test_vertex_msrl_retry_profile_defaults_to_the_poll_cap():
    """The 10h poll cap is the ONE stop every rollout tier is matched to.

    If this default moves, every recipe deadline that was matched to it --
    and the timeout reference describing them -- is silently wrong.
    """
    cfg = VertexMsrlBackendConfig()
    assert cfg.retry.poll_deadline_s == 10 * 60 * 60.0
    assert cfg.retry.create_deadline_s == 2 * 60 * 60.0
    assert cfg.retry.poll_backoff.max_s == 90.0


def test_gemini_api_missing_project_raises():
    cfg = GeminiApiBackendConfig(project="", model="m")
    with pytest.raises(ValueError, match="project"):
        asyncio.run(cfg.make())


def test_gemini_api_missing_model_raises():
    cfg = GeminiApiBackendConfig(project="p", model="")
    with pytest.raises(ValueError, match="model"):
        asyncio.run(cfg.make())


def test_gemini_api_make_returns_bundle_with_stubbed_sampler(monkeypatch):
    """make() returns a BackendBundle when the sampler constructor is
    stubbed out (no real auth/credentials needed)."""
    import backends.gemini_api_endpoint as gae

    captured = {}
    closed = {"count": 0}

    class _FakeSampler:
        def __init__(self, **kwargs):
            captured["sampler_kwargs"] = kwargs

        async def aclose(self):
            closed["count"] += 1

    monkeypatch.setattr(gae, "GeminiApiSampler", _FakeSampler)

    cfg = GeminiApiBackendConfig(project="p", model="m", endpoint_id="")
    bundle = asyncio.run(cfg.make())

    assert isinstance(bundle, BackendBundle)
    assert isinstance(bundle.sampler, _FakeSampler)
    # No endpoint id -> job_id falls back to the model name.
    assert bundle.job_id == "m"
    # location defaults to "global" and is passed through directly.
    assert captured["sampler_kwargs"]["location"] == "global"
    # endpoint_id "" -> passed as None (target the publisher model).
    assert captured["sampler_kwargs"]["endpoint_id"] is None
    # max_generation_tokens is no longer a backend field; not forwarded.
    assert "max_generation_tokens" not in captured["sampler_kwargs"]
    # aclose closure tears the sampler down.
    asyncio.run(bundle.aclose())
    assert closed["count"] == 1


def test_gemini_api_make_job_id_uses_endpoint_id_when_set(monkeypatch):
    import backends.gemini_api_endpoint as gae

    class _FakeSampler:
        def __init__(self, **kwargs):
            pass

        async def aclose(self):
            pass

    monkeypatch.setattr(gae, "GeminiApiSampler", _FakeSampler)

    cfg = GeminiApiBackendConfig(project="p", model="m", endpoint_id="98765")
    bundle = asyncio.run(cfg.make())
    # Endpoint id wins over model name for the log-namespacing job id.
    assert bundle.job_id == "98765"


def test_gemini_api_make_trainer_refuses_training(monkeypatch):
    """The eval-only backend has no real trainer; its bundle.trainer must
    refuse to run a training step rather than silently no-op."""
    import backends.gemini_api_endpoint as gae

    class _FakeSampler:
        def __init__(self, **kwargs):
            pass

        async def aclose(self):
            pass

    monkeypatch.setattr(gae, "GeminiApiSampler", _FakeSampler)

    cfg = GeminiApiBackendConfig(project="p", model="m")
    bundle = asyncio.run(cfg.make())
    with pytest.raises(RuntimeError):
        asyncio.run(bundle.trainer.train_step([]))


# ── job attribution ──────────────────────────────────────────────────


def test_resolve_job_author_prefers_env_var(monkeypatch):
    monkeypatch.setenv("VERTEX_JOB_AUTHOR", "someone")
    monkeypatch.setattr(getpass, "getuser", lambda: "login-user")
    assert resolve_job_author() == "someone"


def test_resolve_job_author_falls_back_to_login_user(monkeypatch):
    """Without a fallback this returned "" and the attribution silently
    vanished, which is what made the feature dead in practice."""
    monkeypatch.delenv("VERTEX_JOB_AUTHOR", raising=False)
    monkeypatch.setattr(getpass, "getuser", lambda: "login-user")
    assert resolve_job_author() == "login-user"


def test_resolve_job_author_tolerates_no_passwd_entry(monkeypatch):
    monkeypatch.delenv("VERTEX_JOB_AUTHOR", raising=False)

    def _boom():
        raise KeyError("no passwd entry")

    monkeypatch.setattr(getpass, "getuser", _boom)
    assert resolve_job_author() == ""


# ── run owner (experiment-doc `owner`) ───────────────────────────────


def test_factory_vertex_msrl_author_uses_owner_resolution(monkeypatch):
    """build_backend's vertex_msrl path names the same human as the
    experiment owner (VMSRL_OWNER wins, OS Login suffix stripped)."""
    import types

    import backends.vertex_msrl_rest as vmr
    from backends.factory import _build_vertex_msrl

    captured = {}

    class _FakeClient:
        tuning_job_id = "1"
        tuning_job_name = "projects/p/locations/l/tuningJobs/1"

        def __init__(self, **kwargs):
            pass

        async def create_tuning_job(self, **kwargs):
            captured.update(kwargs)

        async def wait_for_job_running(self, timeout):
            pass

        async def aclose(self):
            pass

    monkeypatch.setattr(vmr, "VertexMsrlRestClient", _FakeClient)
    monkeypatch.setattr(vmr, "VertexMsrlRestSampler", _FakeSampler)
    monkeypatch.setattr(vmr, "VertexMsrlRestTrainer", _FakeTrainer)
    monkeypatch.setenv("VMSRL_OWNER", "alice@example.com")
    monkeypatch.setenv("VERTEX_JOB_AUTHOR", "bob")
    cfg = types.SimpleNamespace(
        project="p",
        vertex_base_model="m",
        location="us-central1",
        vertex_existing_job_id="",
        vertex_job_running_timeout_seconds=1.0,
        vertex_lr_multiplier=0.0,
        vertex_adapter_size="",
        vertex_checkpoint_interval=0,
        vertex_request_timeout=1.0,
        vertex_request_qpm=0.0,
        max_generation_tokens=0,
    )
    asyncio.run(_build_vertex_msrl(cfg))
    assert "author=alice)" in captured["description"]


def test_resolve_owner_prefers_vmsrl_owner(monkeypatch):
    monkeypatch.setenv("VMSRL_OWNER", "alice")
    monkeypatch.setenv("VERTEX_JOB_AUTHOR", "bob")
    monkeypatch.setattr(getpass, "getuser", lambda: "root")
    assert resolve_owner() == "alice"


def test_resolve_owner_then_job_author(monkeypatch):
    monkeypatch.setenv("VERTEX_JOB_AUTHOR", "bob")
    monkeypatch.setattr(getpass, "getuser", lambda: "root")
    assert resolve_owner() == "bob"


def test_resolve_owner_blank_env_falls_through(monkeypatch):
    monkeypatch.setenv("VMSRL_OWNER", "  ")
    monkeypatch.setattr(getpass, "getuser", lambda: "carol")
    assert resolve_owner() == "carol"


def test_resolve_owner_root_without_env_stays_root(monkeypatch):
    """A run started as root with no launcher hint has no better answer."""
    monkeypatch.setattr(getpass, "getuser", lambda: "root")
    assert resolve_owner() == "root"


def test_resolve_owner_strips_email_domain(monkeypatch):
    monkeypatch.setenv("VMSRL_OWNER", "gina@example.com")
    monkeypatch.setattr(getpass, "getuser", lambda: "root")
    assert resolve_owner() == "gina"


def test_resolve_owner_tolerates_no_passwd_entry(monkeypatch):
    def _boom():
        raise KeyError("no passwd entry")

    monkeypatch.setattr(getpass, "getuser", _boom)
    assert resolve_owner() == ""


def test_job_author_shares_owner_resolution(monkeypatch):
    monkeypatch.setenv("VMSRL_OWNER", "alice")
    monkeypatch.setattr(getpass, "getuser", lambda: "frank")
    assert resolve_job_author() == "alice"
    monkeypatch.delenv("VMSRL_OWNER")
    assert resolve_job_author() == "frank"


def test_vertex_msrl_leaves_job_naming_to_the_server():
    """The default stays as-is: no display name is sent."""
    assert VertexMsrlBackendConfig()._tuned_model_display_name() is None
