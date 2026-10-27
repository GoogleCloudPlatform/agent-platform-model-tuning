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

"""TaskConfig: abstract base for a recipe's data module.

A recipe subclass carries the recipe-specific data paths + parse fn and
implements make(), which resolves them into the loaded datasets the
pipeline/eval path consume. The recipe-agnostic trajectory/experiment-logging
knobs (ATIF output + Vertex Experiment settings) live on this base since they
are identical across recipes and read only by the generic entry points.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TaskConfig(abc.ABC):
    """Recipe data module. make() resolves data paths + parse fn into datasets.

    The framework-level logging knobs below are recipe-agnostic (read by the
    entry points via ``config.task.*``); recipe subclasses add only their
    ``train_data_path`` / ``eval_data_path`` (+ any parse fn) and ``make()``.
    """

    # ATIF on-disk trajectory logging:
    #   "gs://..."      -- local write + background upload
    #   "local"         -- local writes only, under {log_dir}/atif/
    #   "off" / "none"  -- disable
    atif_output: str = ""
    # Background upload worker-pool size (only when atif_output is gs://).
    # Small-file uploads are latency-bound (~0.2-0.65 s each), so throughput
    # scales with workers; see _DEFAULT_MAX_CONCURRENT_UPLOADS.
    atif_upload_concurrency: int = 32
    # Vertex AI Experiments logger name; empty = disabled.
    vertex_experiment: str = ""
    # Backing TensorBoard for the Vertex Experiment. "" = auto-provision the
    # default; "disable" = no TensorBoard time-series (log_metrics still works);
    # else a full TensorBoard resource name. Resolved via
    # resolve_vertex_tensorboard_config() at logging setup.
    vertex_experiment_tensorboard: str = ""
    # Run name within the Vertex Experiment. "" = auto-derive from the log-dir
    # subdir (timestamp); else the given name.
    vertex_experiment_run_name: str = ""
    # GCP region for the Vertex Experiment and its backing TensorBoard.
    # Independent of backend.location (which selects the API endpoint for
    # sampling). Defaults to us-central1 (where all current TB resources are).
    vertex_experiment_location: str = "us-central1"
    # GCP project for the Vertex Experiment and its backing TensorBoard.
    # Independent of backend.project (which owns the tuning job), so a run
    # whose tuning job lives in another project can keep logging to one
    # long-lived experiment. "" = fall back to backend.project.
    vertex_experiment_project: str = ""

    # Code-provenance stamp for the launch. Filled in by whichever launcher
    # started the run, just before the tuning job is created: the launcher
    # commits the tree it is about to run and publishes it, so these name a
    # commit that really exists on the remote. Left at the defaults only
    # when provenance was explicitly waived. Flat scalars on purpose: the
    # whole config is flattened into Vertex Experiments params and into the
    # ATIF config.json, and a nested dataclass would not survive that.
    #
    # There is deliberately no "dirty" field: a launch commits what it
    # runs, so "the tree had uncommitted changes" is not a state a
    # recorded run can be in. Nor is the ref the commit was published as
    # recorded: a server serves a commit by sha, so sha + remote is
    # already a complete recovery path.
    #
    # 40-char sha of the commit whose tree is exactly what ran.
    code_snapshot_sha: str = ""
    # Branch the launch was made from. Kept because it is the one fact
    # nothing can reconstruct later -- a commit does not remember which
    # branch it was made from.
    code_snapshot_branch: str = ""
    # URL of the remote the commit was published to, i.e. where to fetch
    # the sha from.
    code_snapshot_remote: str = ""

    @abc.abstractmethod
    def make(self) -> Any: ...  # -> (train_dataset, eval_dataset) or a Datasets bundle
