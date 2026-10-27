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

"""Tests for src/core/rollout_config.py RolloutConfig ABC and subclasses.

Verifies that AsyncRolloutConfig.make() / SyncRolloutConfig.make() dispatch to
the correct pipeline class and pass the FULL run config straight through. The
pipeline classes are monkeypatched with recording stand-ins so the test asserts
dispatch WITHOUT constructing a real pipeline (a real pipeline currently reads
config.loop.* which a bare run config does not yet expose -- a known
half-migration that is out of scope here).
"""

from __future__ import annotations

import pytest

import core.run_config as rc
from pipeline.async_rollout_pipeline import AsyncRolloutConfig
from pipeline.sync_rollout_pipeline import SyncRolloutConfig


def test_async_make_builds_async_pipeline(monkeypatch):
    calls = {}

    class FakeAsync:
        def __init__(self, **kw):
            calls["cls"] = "async"
            calls["kw"] = kw

    monkeypatch.setattr(
        "pipeline.async_rollout_pipeline.AsyncRolloutPipeline", FakeAsync
    )
    cfg = AsyncRolloutConfig()
    stub_run_config = object()
    p = cfg.make(
        run_config=stub_run_config,
        agent=None,
        train_dataset=None,
        eval_dataset=None,
        eval_agent=None,
        ml_logger=None,
        resume_train_iter_position=None,
    )
    assert calls["cls"] == "async"
    assert calls["kw"]["config"] is stub_run_config  # passes the FULL run config
    assert isinstance(p, FakeAsync)


def test_sync_make_builds_sync_pipeline(monkeypatch):
    calls = {}

    class FakeSync:
        def __init__(self, **kw):
            calls["cls"] = "sync"
            calls["kw"] = kw

    monkeypatch.setattr("pipeline.sync_rollout_pipeline.SyncRolloutPipeline", FakeSync)
    cfg = SyncRolloutConfig()
    stub_run_config = object()
    p = cfg.make(
        run_config=stub_run_config,
        agent=None,
        train_dataset=None,
        eval_dataset=None,
        eval_agent=None,
        ml_logger=None,
        resume_train_iter_position=None,
    )
    assert calls["cls"] == "sync"
    assert calls["kw"]["config"] is stub_run_config  # passes the FULL run config
    assert isinstance(p, FakeSync)


def test_base_is_abstract():
    with pytest.raises(TypeError):
        rc.RolloutConfig()  # abstract, cannot instantiate
