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

"""Single source of truth for config -> experiment-doc field mapping.

Both ``run_training()`` and ``run_eval()`` call
``build_experiment_fields()`` to produce the kwargs dict for
``ProgressObserver.on_experiment_started()``.  Centralising the
mapping here prevents the two orchestrators from diverging.
"""

from __future__ import annotations

import getpass
import os
import socket
from typing import Any

# Code-provenance plumbing.
#
# The launcher commits the tree it is about to run, publishes that commit
# and stamps the result onto ``config.task`` as three flat
# ``code_snapshot_*`` scalars.  We read them back here so the provenance
# facts (sha / branch / remote) ride along into the Firestore
# ExperimentDoc, Vertex Experiments params and the ATIF ``config.json`` —
# every sink that already consumes this dict.
#
# There is no "dirty" field: a launch commits what it runs, so a recorded
# run cannot have had uncommitted changes.
#
# Flat scalars, not a nested container: the whole config is flattened
# into Vertex Experiments params and into config.json, and a nested
# dataclass would not survive that flattening.
#
# The read is deliberately tolerant: a config carrying no stamp (a run
# that waived provenance, or one from before this existed) must still
# produce a well-formed field dict rather than raising.
#
# Output field name -> source attribute on ``config.task``.
_SNAPSHOT_STR_FIELDS: dict[str, str] = {
    "snapshot_sha": "code_snapshot_sha",
    "snapshot_branch": "code_snapshot_branch",
    "snapshot_remote_url": "code_snapshot_remote",
}


# Launcher-set names for the human behind a run, strongest first. The login
# user is a weak signal: a VM may run the trainer as root, or as an OS Login
# account named `<user>_<domain>_<tld>`; a launcher that knows better sets
# VMSRL_OWNER.
_OWNER_ENV_VARS = ("VMSRL_OWNER", "VERTEX_JOB_AUTHOR")


def resolve_owner() -> str:
    """Return the username a run should be attributed to.

    ``VMSRL_OWNER`` > ``VERTEX_JOB_AUTHOR`` > the login user, with an email
    ``<user>@domain`` reduced to ``<user>``. Returns ``""`` when nothing
    resolves (no env and no passwd entry).
    """
    owner = ""
    for var in _OWNER_ENV_VARS:
        owner = os.environ.get(var, "").strip()
        if owner:
            break
    if not owner:
        try:
            owner = getpass.getuser().strip()
        except Exception:  # pylint: disable=broad-except
            # No passwd entry and no USER/LOGNAME (some containers).
            return ""
    # A launcher may pass an email; owner filters match the bare username.
    return owner.split("@", 1)[0] or owner


def _read(obj: Any, name: str) -> Any:
    """Attribute-or-key read that never raises."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def build_snapshot_fields(config: Any) -> dict[str, Any]:
    """Extract the three code-provenance fields from a composed config.

    Reads the flat ``code_snapshot_*`` scalars off ``config.task``, or
    off the top-level config when the stamp landed there instead.  All
    three come from the SAME holder — the one carrying the sha — so a
    stamp can never be spliced together from two different objects.

    Returns the same keys unconditionally so every ExperimentDoc has a
    stable shape: empty strings when the run carries no stamp.
    """
    fields: dict[str, Any] = {name: "" for name in _SNAPSHOT_STR_FIELDS}

    for holder in (_read(config, "task"), config):
        if not _read(holder, _SNAPSHOT_STR_FIELDS["snapshot_sha"]):
            continue
        for out_name, source_name in _SNAPSHOT_STR_FIELDS.items():
            value = _read(holder, source_name)
            fields[out_name] = str(value) if value else ""
        break
    return fields


def _short_name(backend_cfg: Any) -> str:
    """The backend's own label, e.g. ``vertex_msrl``."""
    return str(getattr(backend_cfg, "short_name", "") or "")


def _env_of(backend_cfg: Any) -> str:
    """The serving environment: ``vertex_env`` on the Vertex-specific
    backend config, ``env`` on the shared one.

    Shared by the trainee and the auxiliary backend deliberately. Described
    two different ways, the pair stops being comparable -- and comparing
    them is the entire reason the auxiliary fields exist.
    """
    return str(
        getattr(backend_cfg, "vertex_env", getattr(backend_cfg, "env", "prod")) or ""
    )


def build_experiment_fields(
    config: Any,
    *,
    effective_job_id: str,
    recipe: str,
    bundle_job_id: str,
    total_train_steps: int,
    total_eval_items: int = 0,
    config_name: str = "",
    experiment_group: str = "",
    atif_viewer_base_url: str = "",
    vertex_run_name: str = "",
    auxiliary_job_id: str = "",
    auxiliary_backend_config: Any = None,
) -> dict[str, Any]:
    """Build the kwargs dict for ``on_experiment_started()``.

    Args:
        config: Composed ``RunConfig``.
        effective_job_id: Timestamped job identifier (Firestore doc key).
        recipe: ATIF recipe label.
        bundle_job_id: Raw ``BackendBundle.job_id``.
        total_train_steps: 0 for eval-only, ``config.train.num_train_steps``
            for training.
        total_eval_items: Number of eval tasks (0 when unknown or
            training-only).
        config_name: Registered config name (informational).
        experiment_group: Firestore grouping label.  Falls back to
            ``config.task.vertex_experiment or recipe``.
        atif_viewer_base_url: Resolved ATIF viewer host (may be empty).
        vertex_run_name: The run name ACTUALLY registered with Vertex. Pass
            it whenever the caller does not let ``setup_logging`` derive the
            name from ``config.task.vertex_experiment_run_name`` (the
            checkpoint-eval campaign builds its own ``eval-<job>-<id>``
            name), otherwise the deep-link below points at a run that does
            not exist. Empty -> fall back to the derived name.

    Returns:
        Dict suitable for ``await observer.on_experiment_started(**fields)``.
    """
    atif_output = getattr(config.task, "atif_output", "")
    # THE SAME SANITISER THE WRITER USES, and it must stay the same one:
    # `AtifDiskWriter` stores under `sanitize_path_segment(recipe)`, so a
    # recipe label containing a separator lands somewhere other than its raw
    # name says. Interpolating the raw name here advertised a GCS dir and a
    # viewer URL no trajectory was written to (and the viewer route
    # `/projects/{p}/jobs/{j}` accepts one segment, not two). Imported rather
    # than re-implemented -- a second copy is how the two drifted apart.
    from core.session import sanitize_path_segment

    recipe_path = sanitize_path_segment(recipe)
    gcs_output_dir = (
        f"{atif_output}/{recipe_path}/{effective_job_id}"
        if atif_output.startswith("gs://")
        else ""
    )
    viewer_url = (
        f"{atif_viewer_base_url}/projects/{recipe_path}/jobs/{effective_job_id}"
        if atif_viewer_base_url
        else ""
    )
    tags_raw = os.environ.get("EXPERIMENT_TAGS", "")
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()]

    # Construct Vertex Experiments console URL deep-linking to the
    # experiment RUN (not just the experiment) so the link opens
    # the charts page directly.  The run name is computed identically to
    # setup_logging's compose_vertex_run_name; the Vertex SDK then
    # prepends "{experiment_name}-" when registering the run, so the
    # console URL must include that prefix.  The console path is
    # /agent-platform/experiments/... (not /vertex-ai/experiments/).
    vx_name = getattr(config.task, "vertex_experiment", "")
    vx_location = getattr(config.task, "vertex_experiment_location", "us-central1")
    # The experiment may live in a different project than the tuning job, so
    # this must match what setup_logging was given -- a deep link built from
    # backend.project points at a project with no such experiment.
    vx_project = getattr(config.task, "vertex_experiment_project", "") or getattr(
        config.backend, "project", ""
    )
    vertex_experiment_url = ""
    if vx_name and vx_name != "disable" and vx_project:
        # Prefer the run name the caller actually registered; only fall back
        # to re-deriving it when the caller relied on setup_logging's default.
        # Re-deriving unconditionally produced a 404 deep-link for the
        # checkpoint-eval campaign, whose real name carries an ``eval-`` prefix.
        if vertex_run_name:
            vx_run_name = vertex_run_name
        else:
            from utils.logger import compose_vertex_run_name

            vx_run_name_cfg = (
                getattr(config.task, "vertex_experiment_run_name", "") or None
            )
            vx_run_name = compose_vertex_run_name(bundle_job_id, vx_run_name_cfg)
        # The SDK prepends the experiment name to the run name.
        vx_full_run_name = f"{vx_name}-{vx_run_name}"
        vertex_experiment_url = (
            f"https://console.cloud.google.com/agent-platform/experiments"
            f"/locations/{vx_location}/experiments/{vx_name}"
            f"/runs/{vx_full_run_name}/charts"
            f"?project={vx_project}"
        )

    fields: dict[str, Any] = {
        "doc_id": effective_job_id,
        "experiment_group": experiment_group
        or getattr(config.task, "vertex_experiment", "")
        or recipe,
        "name": effective_job_id,
        "recipe": recipe,
        "config_name": config_name,
        "backend": _short_name(config.backend),
        "env": _env_of(config.backend),
        "tuning_job_id": bundle_job_id or "",
        # THE SECOND JOB, when the recipe asked for one; empty otherwise,
        # which is nearly every run. Recorded rather than left to the log:
        # the id is needed exactly when the run (and its VM) is gone, and a
        # two-backend run that dies hard leaves this job RUNNING.
        "auxiliary_job_id": auxiliary_job_id or "",
        # ...and WHAT it is, not just its id. The second backend is
        # configured separately from the first, so it can differ in kind or
        # in serving environment -- a launch-time override reaches
        # `config.backend` and not a nested backend config. Nothing else in
        # the record would show it: tuning-job resource names are identical
        # across environments.
        "auxiliary_backend": (
            _short_name(auxiliary_backend_config) if auxiliary_backend_config else ""
        ),
        "auxiliary_env": (
            _env_of(auxiliary_backend_config) if auxiliary_backend_config else ""
        ),
        "vm_host": socket.gethostname(),
        "total_train_steps": total_train_steps,
        "total_eval_items": total_eval_items,
        "gcs_output_dir": gcs_output_dir,
        "vertex_experiment_url": vertex_experiment_url,
        "atif_viewer_base_url": viewer_url,
        "owner": resolve_owner(),
        "tags": tags,
    }
    # Code provenance rides on every experiment doc / params payload.
    fields.update(build_snapshot_fields(config))
    return fields
