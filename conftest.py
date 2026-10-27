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

"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

# Make the repo root importable when pytest is launched from elsewhere.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture(autouse=True)
def _reset_msrl_env_vars(monkeypatch: pytest.MonkeyPatch):
    """Ensure environment toggles can't leak between tests."""
    for key in (
        "MSRL_DEBUG_TRAJECTORY",
        "DISABLE_GENERATE_CONTENT",
    ):
        monkeypatch.delenv(key, raising=False)
    # No test may publish a launch branch to the real remote: the launch
    # entry points commit the code they run, so a test that drives main()
    # would otherwise push refs/launch/... for real — and, since
    # that step is now mandatory, would FAIL when it could not.
    # git_snapshot's OWN tests call create_snapshot directly against a
    # throwaway remote, which this does not touch.
    monkeypatch.setenv("VMSRL_SKIP_CODE_SNAPSHOT", "1")
    yield


@pytest.fixture(autouse=True)
def _reset_root_logger():
    """`utils.logger.configure_logging_module` mutates the root logger; isolate tests."""
    root = logging.getLogger()
    saved_level = root.level
    saved_handlers = list(root.handlers)
    yield
    root.handlers = saved_handlers
    root.setLevel(saved_level)
