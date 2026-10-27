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

"""RunConfig orchestrator: run a full training job from a composed config.

Recipe-agnostic training entry point. It sources every live component from
the composed :class:`~core.run_config.RunConfig` via the uniform ``make()``
seam, then hands the assembled pieces to
``pipeline.train_and_eval.train(...)`` (which itself builds the rollout
pipeline via ``config.rollout.make(...)``).

The ``recipe`` label used for ATIF trajectory tagging is supplied by the
caller — one orchestrator serves every recipe, so the tag is a VALUE rather
than a per-recipe constant.

The backing-TensorBoard and run-name knobs are sourced from the task config
(``config.task.vertex_experiment_tensorboard`` /
``vertex_experiment_run_name``): the TensorBoard string is resolved via
``resolve_vertex_tensorboard_config`` and an empty run name auto-derives
from the log-dir subdir.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from core.protocols import ProgressObserver
from core.run_config import RunConfig
from pipeline.train_and_eval import train
from utils.atif_disk_writer import build_atif_writer
from utils.logger import (
    compose_vertex_run_name,
    resolve_vertex_tensorboard_config,
    setup_logging,
)

logger = logging.getLogger(__name__)


def _get_log_subdir(log_dir: str, job_id: str) -> str:
    """Nest the run's logs under ``{log_dir}/{job_id}/{timestamp}``."""
    time_str = time.strftime("%m%d_%H%M%S")
    return f"{log_dir}/{job_id}/{time_str}"


async def run_training(
    config: RunConfig,
    *,
    recipe: str,
    config_name: str = "",
    experiment_group: str = "",
    progress_observer: ProgressObserver | None = None,
    callback_wrapper: Any = None,
    atif_viewer_base_url: str = "",
) -> None:
    """Run one training job described by ``config``.

    Builds the backend bundle, reward fn, datasets, ATIF writer, agent, and
    ml_logger from ``config`` via ``make()``, then drives
    ``train(...)``. All resources are closed in ``finally`` blocks.

    The caller owns launch-env overrides (including ``log_dir``): they are
    baked into ``config.train.log_dir`` via ``apply_overrides`` before this
    orchestrator is invoked, so ``config`` is authoritative here.

    Args:
        config: The composed run config (e.g. from
            ``load_config(...)``, optionally merged with launch overrides).
        recipe: The recipe label used to tag ATIF trajectories (supplied by
            the caller, e.g. derived from the config name).
    """
    _t0 = time.monotonic()
    # Print backend identity before the (potentially slow) make() call so
    # the operator can see what tuning job / endpoint will be used.
    _job_id = getattr(config.backend, "vertex_existing_job_id", "") or ""
    _proj = config.backend.project
    _loc = config.backend.location
    if _job_id:
        _resource = f"projects/{_proj}/locations/{_loc}/tuningJobs/{_job_id}"
        logger.info("Building backend ... target=%s", _resource)
    else:
        logger.info(
            "Building backend ... project=%s location=%s"
            " (new tuning job will be created)",
            _proj,
            _loc,
        )
    bundle = await config.backend.make()
    # Both initialised before the `try`: they are read unconditionally below,
    # on every run, and most runs never set them.
    aux_bundle = None
    # `Any` throughout: `auxiliary_backend` is an OPTIONAL, duck-typed seam --
    # the base `AgentConfig` does not declare it, and neither the hook nor the
    # backend config it returns has a static type here.
    aux_cfg: Any = None
    try:
        # A multi-agent recipe may need a SECOND backend for a sub-agent that
        # must not share the trainee's weights -- a frozen instrument on the
        # same serving path. The recipe asks for one by exposing
        # `auxiliary_backend()`; every other agent config returns nothing here
        # and the run is unchanged.
        #
        # Created HERE, and INSIDE this `try`, for LIFETIME: a backend that
        # provisions a tuning job must be closed or the job outlives the run
        # burning quota, this is the only scope with a `finally` wide enough
        # to do it, and if the auxiliary backend fails to come up that
        # `finally` must still close the trainee's job.
        aux_backend: Any = getattr(config.agent, "auxiliary_backend", None)
        if callable(aux_backend):
            aux_cfg = aux_backend()
            if aux_cfg is not None:
                aux_bundle = await aux_cfg.make()

        reward_fn = config.reward.make()
        train_dataset, eval_dataset = config.task.make()

        # ATIF on-disk trajectory logging (opt-in via config.task.atif_output).
        # None when disabled -> no-op callback. Timestamped so successive
        # runs don't overwrite each other in GCS.
        import time as _time

        effective_job_id = f"{bundle.job_id}-{_time.strftime('%Y%m%d_%H%M%S')}"
        atif_writer = build_atif_writer(
            atif_output=config.task.atif_output,
            log_dir=config.train.log_dir,
            recipe=recipe,
            tuning_job_id=effective_job_id,
            max_concurrent_uploads=config.task.atif_upload_concurrency,
        )

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
                    total_train_steps=config.train.num_train_steps,
                    config_name=config_name,
                    experiment_group=experiment_group,
                    atif_viewer_base_url=atif_viewer_base_url or "",
                    # The second job, so the record names BOTH -- its id, and
                    # the config saying what it is and where it runs. Empty on
                    # a one-backend run, which is nearly all of them.
                    auxiliary_job_id=(
                        (aux_bundle.job_id or "") if aux_bundle is not None else ""
                    ),
                    auxiliary_backend_config=aux_cfg,
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
            # Passed only when the recipe asked for one, so the kwargs never
            # reach an AgentConfig that does not expect them. The auxiliary
            # bundle's TRAINER is deliberately never handed on -- `train(...)`
            # below gets `bundle.trainer` and nothing else, which is what
            # keeps the sub-agent frozen. The id travels WITH the sampler: the
            # sub-agent stamps its trajectories (and hence its ATIF trial dirs
            # and the monitor's per-job view) with the job that served them.
            **(
                {
                    "auxiliary_sampler": aux_bundle.sampler,
                    "auxiliary_job_id": aux_bundle.job_id or "",
                }
                if aux_bundle is not None
                else {}
            ),
        )

        # Also streams hparams + metrics to Vertex AI Experiments when
        # ``config.task.vertex_experiment`` is set. The backing TensorBoard and
        # run-name knobs are sourced from the task config (resolved / derived).
        ml_logger = setup_logging(
            log_dir=_get_log_subdir(config.train.log_dir, bundle.job_id),
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

        try:
            await train(
                agent=agent,
                config=config,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                trainer=bundle.trainer,
                ml_logger=ml_logger,
                progress_observer=progress_observer,
            )
            logger.info("Training loop completed.")
        finally:
            ml_logger.close()
            # A signal CANCELS this task (utils.signals.cancel_on_signal);
            # a cancelled task's finally still gets to await, which is what
            # makes `pkill` clean up rather than leak a live tuning job.
            if effective_callback is not None:
                try:
                    await effective_callback.on_job_completed(
                        {"recipe": recipe, "tuning_job_id": bundle.job_id}
                    )
                except Exception:  # noqa: BLE001 - never mask the real error
                    logger.warning("ATIF drain on exit failed", exc_info=True)
            if atif_writer is not None:
                # Drained above, so there is nothing left for a future
                # reclaim to pick up.
                atif_writer.deregister_from_reclaim()
                if atif_writer.gcs_uri:
                    logger.info(
                        "ATIF trajectories uploaded to: %s",
                        atif_writer.gcs_uri,
                    )
                logger.info("ATIF local dir: %s", atif_writer.job_dir)
    finally:
        if progress_observer is not None:
            _wall = time.monotonic() - _t0
            try:
                await progress_observer.on_experiment_completed(
                    status="completed",
                    wall_time_seconds=_wall,
                )
            except Exception:
                pass
        if aux_bundle is not None:
            # Closed FIRST and in its own `try`: `aclose` cancels a job this
            # run created, and letting an auxiliary failure propagate here
            # would skip the trainee's close and leak the bigger of the two
            # jobs.
            try:
                await aux_bundle.aclose()
            except Exception:
                logger.exception("Auxiliary backend aclose failed.")
        # This is what cancels the tuning job -- the single most important
        # thing to survive a `pkill`, since a live job holds serving capacity
        # until it ages out.
        await bundle.aclose()
