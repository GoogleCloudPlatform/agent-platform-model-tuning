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

"""Code-provenance fields on `TaskConfig`.

The launcher stamps the snapshot commit it pushed onto the composed config
just before the tuning job is created; from there the fields ride along for
free into the Vertex Experiments params and the ATIF `config.json` written to
GCS. These tests pin the contract that makes that free ride work: the three
fields live on the recipe-agnostic base, they are flat scalars (a nested
dataclass would not survive the params flattening), and they default to a
benign "no snapshot" state so unsnapshotted launches still validate.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any

# Keep the module importable when pytest is invoked without the repo's
# `pythonpath = ["src", ...]` ini setting.
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.task_config import TaskConfig  # noqa: E402

_SNAPSHOT_FIELDS = {
    "code_snapshot_sha": (str, ""),
    "code_snapshot_branch": (str, ""),
    "code_snapshot_remote": (str, ""),
}


@dataclasses.dataclass(frozen=True)
class _FakeTaskConfig(TaskConfig):
    """Minimal concrete recipe config (TaskConfig is an ABC)."""

    train_data_path: str = "/tmp/train.jsonl"

    def make(self) -> Any:
        return ("train", "eval")


def test_snapshot_fields_declared_on_the_base() -> None:
    """The fields are recipe-agnostic, so they belong to TaskConfig itself."""
    declared = {f.name for f in dataclasses.fields(TaskConfig)}
    assert set(_SNAPSHOT_FIELDS) <= declared


def test_snapshot_field_types_and_defaults() -> None:
    """Defaults describe an unsnapshotted launch and must not be required."""
    fields = {f.name: f for f in dataclasses.fields(_FakeTaskConfig)}
    cfg = _FakeTaskConfig()  # no snapshot kwargs: must construct
    for name, (expected_type, expected_default) in _SNAPSHOT_FIELDS.items():
        assert fields[name].default == expected_default, name
        value = getattr(cfg, name)
        assert value == expected_default, name
        assert type(value) is expected_type, name


def test_snapshot_fields_are_flat_scalars() -> None:
    """Nested dataclasses would be lost when the config is flattened."""
    for name in _SNAPSHOT_FIELDS:
        value = getattr(_FakeTaskConfig(), name)
        assert not dataclasses.is_dataclass(value)
        assert isinstance(value, (str, bool))


def test_launcher_stamps_provenance_via_replace() -> None:
    """TaskConfig is frozen; the launcher stamps it with dataclasses.replace."""
    cfg = _FakeTaskConfig(vertex_experiment_run_name="run-42")
    sha = "a" * 40
    stamped = dataclasses.replace(
        cfg,
        code_snapshot_sha=sha,
        code_snapshot_branch="alice/feature",
        code_snapshot_remote="git@example.com:org/repo.git",
    )

    assert stamped.code_snapshot_sha == sha
    assert len(stamped.code_snapshot_sha) == 40
    assert stamped.code_snapshot_branch == "alice/feature"
    assert stamped.code_snapshot_remote == "git@example.com:org/repo.git"

    # Stamping touches nothing else, and leaves the original untouched.
    assert stamped.vertex_experiment_run_name == "run-42"
    assert stamped.train_data_path == cfg.train_data_path
    assert cfg.code_snapshot_sha == ""


def test_the_stamp_is_sha_branch_remote_and_nothing_else() -> None:
    """Three fields, and the two that were dropped stay dropped.

    ``code_snapshot_ref`` is not recorded: a server serves a commit by
    sha (``git fetch <remote> <sha>``), so sha + remote already recovers
    the code, and the ref name is mechanically derivable. ``dirty`` is
    not recorded because a launch commits what it runs.
    """
    fields = {f.name for f in dataclasses.fields(_FakeTaskConfig)}
    assert {"code_snapshot_sha", "code_snapshot_branch", "code_snapshot_remote"} <= (
        fields
    )
    assert "code_snapshot_ref" not in fields
    assert "code_snapshot_dirty" not in fields


def test_provenance_survives_config_flattening() -> None:
    """asdict() is what feeds Vertex Experiments params / ATIF config.json."""
    stamped = dataclasses.replace(
        _FakeTaskConfig(),
        code_snapshot_sha="b" * 40,
        code_snapshot_branch="main",
        code_snapshot_remote="git@example.com:org/repo.git",
    )
    flat = dataclasses.asdict(stamped)

    assert set(_SNAPSHOT_FIELDS) <= set(flat)
    assert flat["code_snapshot_sha"] == "b" * 40
    for name in _SNAPSHOT_FIELDS:
        assert isinstance(flat[name], (str, bool)), name
