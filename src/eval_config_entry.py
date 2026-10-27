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

"""RunConfig orchestrator: run one standalone eval sweep from a composed config.

Recipe-agnostic eval entry point — the eval-path sibling of the
training-path ``train_config_entry.py``. It sources every live component from
the composed :class:`~core.run_config.RunConfig` via the uniform ``make()``
seam, then runs ONE eval sweep via ``pipeline.eval_only.EvalPipeline``
(logging ``eval/*`` metrics; no training).

The ``recipe`` label used for ATIF trajectory tagging is supplied by the
caller — one orchestrator serves every recipe, so the tag is a VALUE rather
than a per-recipe constant. The ATIF ``tuning_job_id`` segment is
timestamped (``{job_id}-{YYYYMMDD_HHMMSS}``) so successive eval runs
don't overwrite each other in GCS.

The backend is selected purely by the ``config.backend`` subclass TYPE — a
checkpoint-attach backend or a base-model / deployed-endpoint backend both
expose ``make()``, so there is no string switch. Backend-specific logger
fields (project / location) and the eval x-axis step are read defensively via
``getattr`` so the path works for every backend type.

The backing-TensorBoard and run-name knobs are sourced from the task config
(``config.task.vertex_experiment_tensorboard`` /
``vertex_experiment_run_name``): the TensorBoard string is resolved via
``resolve_vertex_tensorboard_config`` and an empty run name auto-derives
from the log-dir subdir.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from core.protocols import ProgressObserver

logger = logging.getLogger(__name__)
from core.run_config import RunConfig
from core.task_data import TaskData
from pipeline.eval_only import EvalPipeline
from pipeline.eval_serializer import write_eval_jsonl
from utils.atif_disk_writer import build_atif_writer
from utils.logger import (
    compose_vertex_run_name,
    resolve_vertex_tensorboard_config,
    setup_logging,
)


def _get_log_subdir(log_dir: str, job_id: str) -> str:
    """Nest the run's logs under ``{log_dir}/{job_id}/{timestamp}``."""
    time_str = time.strftime("%m%d_%H%M%S")
    return f"{log_dir}/{job_id}/{time_str}"


async def run_eval(
    config: RunConfig,
    *,
    recipe: str,
    config_name: str = "",
    experiment_group: str = "",
    jsonl_output_path: str | None = None,
    progress_observer: ProgressObserver | None = None,
    callback_wrapper: Any = None,
    atif_viewer_base_url: str = "",
) -> dict[str, Any]:
    """Run one standalone eval sweep described by ``config``.

    Builds the backend bundle, reward fn, eval dataset, ATIF writer, agent,
    and ml_logger from ``config`` via ``make()``, then drives one
    ``EvalPipeline.run()`` sweep. All resources are closed in ``finally``
    blocks.

    The caller owns launch-env overrides (including ``log_dir``): they are
    baked into ``config.train.log_dir`` via ``apply_overrides`` before this
    orchestrator is invoked, so ``config`` is authoritative here.

    Args:
        config: The composed run config (e.g. from
            ``load_config(...)``, optionally merged with launch overrides).
        recipe: The recipe label used to tag ATIF trajectories (supplied by
            the caller, e.g. derived from the config name).
        jsonl_output_path: If set, per-group eval results are written as
            JSONL to this path after the sweep completes.

    Returns:
        A dict with ``"metrics"`` (the aggregated ``eval/*`` dict),
        ``"job_id"`` (backend bundle's job identifier), and
        ``"log_dir"`` (the resolved log sub-directory).
    """
    _t0 = time.monotonic()
    _eval_failed = False  # Track whether eval raised, for on_experiment_completed.
    # Log backend identity before the (potentially slow) make() call.
    _job_id = getattr(config.backend, "vertex_existing_job_id", "") or ""
    _proj = config.backend.project
    _loc = config.backend.location
    if _job_id:
        _resource = f"projects/{_proj}/locations/{_loc}/tuningJobs/{_job_id}"
        logger.info("Building backend ... target=%s", _resource)
    else:
        logger.info("Building backend ... project=%s location=%s", _proj, _loc)
    # Backend selected by subclass TYPE; resume defaults to 0 on the eval path
    # (there is no training step to resume from).
    bundle = await config.backend.make()
    # Bound before the try so the finally can always test it: it is assigned
    # part-way down, and anything that raises before then would otherwise
    # turn a real error into a NameError from the cleanup path.
    atif_writer = None
    try:
        reward_fn = config.reward.make()
        # Eval-only: build just the eval dataset. make() would also eagerly load
        # the train split (datasets open their file at construction), which an
        # eval run never uses and which may not exist for a pure eval.
        eval_dataset = config.task.make_eval()

        # ATIF on-disk trajectory logging (opt-in via config.task.atif_output).
        # None when disabled -> no-op callback. Timestamped so successive
        # eval runs don't overwrite each other in GCS.
        import time as _time

        effective_job_id = f"{bundle.job_id}-{_time.strftime('%Y%m%d_%H%M%S')}"
        atif_writer = build_atif_writer(
            atif_output=config.task.atif_output,
            log_dir=config.train.log_dir,
            recipe=recipe,
            tuning_job_id=effective_job_id,
            max_concurrent_uploads=config.task.atif_upload_concurrency,
        )

        # NOTE: this run does NOT reclaim other runs' leftovers. An earlier
        # draft swept abandoned trees here, which made run B responsible for
        # run A's data -- B's startup latency, B's credentials and B's
        # failures all entangled with data B has nothing to do with, and a
        # bug in the sweep could take out a tree the operator still wanted.
        # A run cleans up after ITSELF (finalize_trial during the run, drain
        # on every exit) and registers itself so that a human or a cron line
        # can reclaim it later via `python -m utils.run_registry --reclaim`.

        # Compute total eval items from config cap + dataset size.
        _max = config.eval.max_eval_items
        _ds_len = len(eval_dataset) if eval_dataset is not None else 0
        _total_eval_items = _ds_len if _max < 0 else min(_max, _ds_len)

        # Notify progress observer AFTER build_atif_writer so the viewer URL
        # uses the same effective_job_id (includes the timestamp suffix).
        if progress_observer is not None:
            import logging as _logging

            from utils.experiment_fields import build_experiment_fields

            _obs_logger = _logging.getLogger(__name__)
            try:
                fields = build_experiment_fields(
                    config,
                    effective_job_id=effective_job_id,
                    recipe=recipe,
                    bundle_job_id=bundle.job_id or "",
                    total_train_steps=0,  # eval-only, no training
                    total_eval_items=_total_eval_items,
                    config_name=config_name,
                    experiment_group=experiment_group,
                    atif_viewer_base_url=atif_viewer_base_url or "",
                )
                await progress_observer.on_experiment_started(**fields)
            except Exception:
                _obs_logger.warning(
                    "Progress observer on_experiment_started failed", exc_info=True
                )

        # Wrap callbacks if a wrapper was provided (a caller may compose
        # e.g. [AtifDiskWriter, ProgressSink] via CompositeTrajectoryCallback).
        effective_callback = atif_writer
        if callback_wrapper is not None:
            effective_callback = callback_wrapper(atif_writer)

        agent = config.agent.make(
            sampler=bundle.sampler,
            job_id=bundle.job_id,
            reward_fn=reward_fn,
            atif_writer=effective_callback,
        )

        log_subdir = _get_log_subdir(config.train.log_dir, bundle.job_id)
        ml_logger = setup_logging(
            log_dir=log_subdir,
            config=config,
            do_configure_logging_module=True,
            vertex_experiment=config.task.vertex_experiment or None,
            vertex_project=(
                config.task.vertex_experiment_project or config.backend.project
            ),
            vertex_location=config.task.vertex_experiment_location,
            vertex_tensorboard=resolve_vertex_tensorboard_config(
                config.task.vertex_experiment_tensorboard
            ),
            vertex_run_name=compose_vertex_run_name(
                bundle.job_id, config.task.vertex_experiment_run_name or None
            ),
        )

        if effective_callback is not None:
            await effective_callback.on_job_started(config)

        # Collect per-group (task_data, metrics) for optional JSONL output.
        collected_groups: list[tuple[Any, dict[str, Any]]] = []

        def _collect_group(task_data: TaskData, group_metrics: dict[str, Any]) -> None:
            collected_groups.append((task_data, group_metrics))

        try:
            eval_pipeline = EvalPipeline(
                agent=agent,
                config=config,
                ml_logger=ml_logger,
                eval_dataset=eval_dataset,
                progress_observer=progress_observer,
            )
            metrics = await eval_pipeline.run(
                on_group=_collect_group if jsonl_output_path else None,
            )
            if jsonl_output_path and collected_groups:
                write_eval_jsonl(Path(jsonl_output_path), collected_groups)
            logger.info("Eval complete. Metrics: %s", metrics)
        finally:
            ml_logger.close()
            if effective_callback is not None:
                await effective_callback.on_job_completed(
                    {"recipe": recipe, "tuning_job_id": bundle.job_id}
                )
                # Log ATIF info from the underlying writer.
                if atif_writer is not None:
                    if atif_writer.gcs_uri:
                        logger.info(
                            "ATIF trajectories uploaded to: %s",
                            atif_writer.gcs_uri,
                        )
                    logger.info("ATIF local dir: %s", atif_writer.job_dir)
    except Exception:
        _eval_failed = True
        raise
    finally:
        # Drain on EVERY exit, not just the happy one. `on_job_completed`
        # drains, but it runs on the success path only, so an exception left
        # the whole local tree behind with uploads still queued.
        #
        # A signal CANCELS this task (utils.signals.cancel_on_signal), and
        # a cancelled task's finally still gets to await -- measured: one or
        # two cancels both let this block finish. `asyncio.shield` here was
        # tried and removed; it changed nothing at any cancel count, so it
        # was complexity claiming a protection it did not provide. A third
        # cancel kills cleanup either way, which is what SIGKILL and the run
        # registry are for.
        if atif_writer is not None:
            try:
                await atif_writer.drain()
                atif_writer.deregister_from_reclaim()
            except Exception:  # noqa: BLE001 - teardown must not mask the real error
                logger.warning("ATIF drain on exit failed", exc_info=True)
        if progress_observer is not None:
            _wall = time.monotonic() - _t0
            try:
                await progress_observer.on_experiment_completed(
                    status="failed" if _eval_failed else "completed",
                    wall_time_seconds=_wall,
                )
            except Exception:
                pass
        # This is what cancels the tuning job -- the single most important
        # thing to survive a `pkill`, since a live job holds serving capacity
        # until it ages out.
        await bundle.aclose()

    return {
        "metrics": metrics,
        "job_id": bundle.job_id,
        "log_dir": log_subdir,
        # Where this run's ATIF tree landed. The tuning_job_id segment carries
        # a run-time timestamp, so a caller cannot derive these -- reporting
        # them is the only reliable handoff to post-run tooling (a verifier, or
        # a human trying to find the tree after the VM is reclaimed).
        "effective_job_id": effective_job_id,
        "atif_gcs_uri": atif_writer.gcs_uri if atif_writer is not None else None,
        "atif_local_dir": (
            str(atif_writer.job_dir) if atif_writer is not None else None
        ),
    }
