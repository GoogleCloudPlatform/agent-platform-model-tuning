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

"""Pipeline-agnostic training + eval loop.

``train(config=..., ...)`` is the library entry point recipes call. It
owns the trainer control flow (batch loop, reward normalization,
``trainer.train_step``, per-step metric logging, eval-cadence
scheduling); pipelines own the rollout subsystem. Pipeline construction
is delegated to ``config.rollout.make(...)`` — the ``RolloutConfig``
subclass type (async vs sync) selects the concrete pipeline.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import math
import time
from typing import Any

from core.agent import Agent
from core.metric_spec import aggregate_by_spec
from core.protocols import ProgressObserver
from core.rollout_protocol import BatchItem, RolloutPipeline
from core.dataset import JsonlDataset
from core.protocols import RewardNormalizer, TrainConfigLike, Trainer
from core.reward_normalizer_config import (
    RewardNormalizerConfig,
    CenteringNormalizerConfig,
)
from utils.logger import Logger

logger = logging.getLogger(__name__)

# Consecutive trainer failures tolerated before the run is aborted. A dead
# backend (an unreachable trainer server, say) fails every step the same
# way, so carrying on just burns the remaining budget against it — and each
# attempt can cost hours, since the backend only reports failure after
# exhausting its own retries.
_MAX_CONSECUTIVE_TRAIN_STEP_FAILURES = 3


__all__ = [
    "resolve_resume_position",
    "train",
    "train_and_eval_loop",
]


def resolve_resume_position(
    *,
    resume_from_step: int,
    resume_metrics_path: str,
) -> int | None:
    """Read ``train/pipeline/iterator_current`` at ``resume_from_step`` from
    the metrics file. Returns the position, or ``None`` for a fresh start
    (``resume_from_step == 0``). Raises ``RuntimeError`` if the step
    isn't in the file — we fail loud rather than guess, because a wrong
    estimate silently corrupts training.
    """
    if resume_from_step <= 0:
        return None
    # Deferred import: utils.logger pulls in ML deps.
    from utils.logger import restore_iterator_state

    position = restore_iterator_state(resume_metrics_path, resume_from_step)
    if position is None:
        raise RuntimeError(
            f"Cannot resume from step {resume_from_step}: metrics file"
            f" {resume_metrics_path!r} has no 'train/pipeline/iterator_current'"
            f" entry for that step. Either resume from a step present"
            f" in the metrics file, or start fresh (resume_from_step=0)."
        )
    logger.info(
        f"Resuming from train step {resume_from_step}; train_iter position={position}."
    )
    return position


def _aggregate_step_metrics(batch_items: list[BatchItem]) -> dict[str, Any]:
    """Aggregate per-group metrics across ``batch_items`` for the train step.

    Uses ``aggregate_by_spec`` for spec-driven reduction of learning,
    sampler, and pipeline metrics (``learning/reward/mean``,
    ``learning/sampling_accuracy``, ``learning/pass_at_k``,
    ``sampler/finish_reason/*``, timing, etc.).  The caller stamps a
    ``train/`` prefix on every key before logging.

    ``sampler/policy_step`` is pulled out separately so the caller can
    compute ``train/staleness_steps = global_step - mean(policy_step)``.
    """
    _NON_METRIC_KEYS = frozenset({"dropped", "task_idx"})
    filtered = [
        {
            k: v
            for k, v in item.metrics.items()
            if k not in _NON_METRIC_KEYS and v is not None
        }
        for item in batch_items
    ]
    step_metrics: dict[str, Any] = dict(aggregate_by_spec(filtered))
    return step_metrics


def _summarize_batch_for_observer(
    batch_items: list[BatchItem],
) -> tuple[list[str], float | None]:
    """Pull out (session_ids, mean_reward) from batch for ``on_batch_assembled``.

    session_ids match placeholder docs written by ``on_trajectory_started``.
    An empty session_id is a real edge case (a caller bypassing the base
    template) worth surfacing rather than silently emitting a ghost id.
    Mean reward is denormalised here so observers (e.g. the Firestore sink)
    can write a per-batch scalar without fetching every trajectory doc; NaN
    (timeout sentinel) is excluded from the mean.
    """
    session_ids: list[str] = []
    rewards: list[float] = []
    for item in batch_items:
        for scored in item.group.scored_trajectories:
            sid = scored.trajectory.session_id or ""
            if sid:
                session_ids.append(sid)
            else:
                td = item.group.task_data
                logger.warning(
                    "on_batch_assembled: skipping scored trajectory with "
                    "empty session_id (task_id=%s)",
                    td.task_id if td is not None else "unknown",
                )
            r = scored.reward.reward
            if not math.isnan(r):
                rewards.append(r)
    mean_reward = sum(rewards) / len(rewards) if rewards else None
    return session_ids, mean_reward


@dataclasses.dataclass
class _StepOutcome:
    """Result of one train step.

    ``global_step`` is the (possibly unchanged) global step counter after
    the step. ``advanced`` is False when the step produced nothing to train
    on (empty batch or trainer failure) — the caller then leaves
    ``global_step`` untouched and does not schedule eval.

    ``checkpoint_endpoint`` is set to the deployed checkpoint endpoint
    resource path when the backend surfaced one this step (checkpoint
    steps only); ``None`` otherwise. The loop uses it to schedule a
    checkpoint-endpoint eval.

    ``failed`` separates a *trainer* failure from an empty batch — both
    leave ``advanced`` False, but only the former counts toward the
    consecutive-failure abort. ``error`` carries its message.
    """

    global_step: int
    advanced: bool
    checkpoint_endpoint: str | None = None
    failed: bool = False
    error: str | None = None


async def _run_one_train_step(
    *,
    train_step: int,
    num_train_steps: int,
    global_step: int,
    pipeline: RolloutPipeline,
    trainer: Trainer,
    ml_logger: Logger,
    normalizer: RewardNormalizer,
    tasks_per_train_step: int,
    sync_every_n_train_steps: int,
    progress_observer: ProgressObserver | None = None,
) -> _StepOutcome:
    """Run a single train step: pull a batch, normalize rewards, call
    ``trainer.train_step``, log ``train/*`` metrics, and advance the global
    step counter.

    Returns ``_StepOutcome(global_step, advanced)``. ``advanced`` is False
    (with ``global_step`` unchanged) when the batch has no valid
    trajectories or the trainer returns no metrics — the loop then retries
    the same step without scheduling eval.
    """
    t_wait_start = time.perf_counter()
    logger.info(f"\n===== Train step {train_step}/{num_train_steps} =====")
    if progress_observer is not None:
        await progress_observer.on_train_step_started(train_step, num_train_steps)
    logger.info(f"Waiting for {tasks_per_train_step} tasks...")
    # Sampled BEFORE the block below, so buffer occupancy reads "what the
    # trainer found when it asked for data" -- the quantity that pairs
    # with pipeline/wait_for_rollout_time. Exactly one call per step: the
    # drop ratio it returns is a per-step delta.
    pipeline_gauges = pipeline.sample_pipeline_gauges()
    # step_num lets sync key its per-step sampling metrics at the
    # same step as train_metrics below; async ignores it.
    batch_items, train_iter_position = await pipeline.next_batch(
        tasks_per_train_step, step_num=train_step
    )
    logger.info(f"Obtained {len(batch_items)} tasks...")
    trajectory_groups = [item.group for item in batch_items]

    normalizer.normalize(trajectory_groups)  # in-place

    step_metrics = _aggregate_step_metrics(batch_items)
    wait_time = time.perf_counter() - t_wait_start

    step_trajectories = []
    for g in trajectory_groups:
        step_trajectories.extend(g.scored_trajectories)
    if not step_trajectories:
        return _StepOutcome(global_step, advanced=False)  # nothing valid this step

    if progress_observer is not None:
        trajectory_ids, mean_reward = _summarize_batch_for_observer(batch_items)
        try:
            await progress_observer.on_batch_assembled(
                step=train_step,
                trajectory_ids=trajectory_ids,
                mean_reward=mean_reward,
            )
        except Exception:
            logger.warning("on_batch_assembled callback failed", exc_info=True)

    t_step_start = time.perf_counter()
    sync_trainer_and_sampler = (global_step + 1) % sync_every_n_train_steps == 0
    logger.info(
        f"train_step_num:{global_step},"
        f" sync_trainer_and_sampler:{sync_trainer_and_sampler}"
    )
    train_metrics = await trainer.train_step(
        step_trajectories, sync_trainer_and_sampler
    )
    t_step_time = time.perf_counter() - t_step_start
    # A backend signals failure either with an explicit ``_train_step_error``
    # or with a bare falsy dict. Both must be reported: without the
    # ``on_train_step_completed`` call below the step doc is stranded at
    # "training" forever, its error is discarded, and every trajectory in
    # the batch stays TRAINING because phase 2 never runs.
    error = (train_metrics or {}).get("_train_step_error")
    if error or not train_metrics:
        failed_metrics = dict(train_metrics or {})
        error = failed_metrics.setdefault(
            "_train_step_error", "trainer returned no metrics"
        )
        logger.error(
            f"Train step {train_step}/{num_train_steps} FAILED after"
            f" {t_step_time:.1f} seconds: {error}"
        )
        if progress_observer is not None:
            await progress_observer.on_train_step_completed(
                train_step, global_step, train_metrics=failed_metrics
            )
        return _StepOutcome(global_step, advanced=False, failed=True, error=error)

    # Only a step that actually trained gets the "completed" line: the
    # integration smokes count finished steps by parsing it
    # (``Train step (\d+)/(\d+) completed in``), so emitting it on failure
    # both over-counts there and reads as a lie next to the error above.
    logger.info(
        f"Train step {train_step}/{num_train_steps} completed in"
        f" {t_step_time:.1f} seconds"
    )

    # Follow the backend's completed_train_step_id when emitted so
    # the counter is self-repairing across trainer restarts and
    # out-of-order backend updates. Fall back to the client counter
    # for backends that don't emit it; warn once per run so a
    # regression on a backend that SHOULD emit it stays visible.
    backend_step_id = train_metrics.get("completed_train_step_id")
    if backend_step_id is not None:
        global_step = int(backend_step_id)
    else:
        if not getattr(_run_one_train_step, "_warned_missing_step_id", False):
            logger.warning(
                "Backend did not emit completed_train_step_id in"
                " train_metrics; falling back to client-side counter."
                " This warning fires once per run."
            )
            _run_one_train_step._warned_missing_step_id = True
        global_step += 1

    metrics_to_log = {f"train/{k}": v for k, v in train_metrics.items()}
    # Emit these under their bare ``pipeline/*`` names (matching
    # METRIC_SPECS) plus the uniform ``train/`` prefix, so a future
    # spec-aware sink layer can look them up via ``spec_for``.
    _pipeline_passthrough = {
        "pipeline/train_step_time": t_step_time,
        "pipeline/wait_for_rollout_time": wait_time,
        "pipeline/iterator_current": train_iter_position,
        **pipeline_gauges,
    }
    metrics_to_log.update({f"train/{k}": v for k, v in _pipeline_passthrough.items()})
    metrics_to_log.update({f"train/{k}": v for k, v in step_metrics.items()})

    # Client-side staleness: global_step - mean(policy_step across batch).
    if "sampler/policy_step" in step_metrics:
        metrics_to_log["train/staleness_steps"] = (
            global_step - step_metrics["sampler/policy_step"]
        )

    ml_logger.log_metrics(metrics_to_log, step=global_step)
    if progress_observer is not None:
        await progress_observer.on_train_step_completed(
            train_step, global_step, train_metrics=train_metrics
        )
    logger.info(f"Train Step {global_step} Metrics: {train_metrics}")
    return _StepOutcome(
        global_step,
        advanced=True,
        checkpoint_endpoint=train_metrics.get("_checkpoint_endpoint"),
    )


async def train_and_eval_loop(
    *,
    pipeline: RolloutPipeline,
    trainer: Trainer,
    ml_logger: Logger,
    num_train_steps: int,
    tasks_per_train_step: int,
    eval_every_n_train_steps: int,
    reward_normalizer: RewardNormalizerConfig = CenteringNormalizerConfig(),
    initial_global_step: int = 0,
    sync_every_n_train_steps: int = 1,
    run_sampler_eval: bool = True,
    run_checkpoint_endpoint_eval: bool = False,
    progress_observer: ProgressObserver | None = None,
) -> None:
    """Drive the trainer against any ``RolloutPipeline`` for ``num_train_steps``
    gradient updates.

    Per step: pull a batch, normalize rewards, call
    ``trainer.train_step``, log ``train/*`` metrics, and schedule eval
    on cadence. An initial eval fires at ``initial_global_step`` before
    training starts; a final eval fires at the last step regardless of
    cadence. ``pipeline.stop()`` runs in ``finally`` so the paired
    ``pipeline.run()`` in ``asyncio.gather`` exits cleanly.

    ``run_sampler_eval=False`` suppresses every live-sampler ``eval/*``
    sweep (initial, cadence, final) — the checkpoint-endpoint sweep, if
    enabled, still runs.

    ``sync_trainer_and_sampler`` (passed to ``trainer.train_step``) is
    ``(global_step + 1) % sync_every_n_train_steps == 0``.
    """
    # Initial value from config; overwritten below by the backend's
    # ``completed_train_step_id`` on the first successful train_step
    # (or incremented client-side when the backend omits that field).
    global_step = initial_global_step
    normalizer = reward_normalizer.make()
    if run_sampler_eval:
        pipeline.schedule_eval(global_step)  # step-0 baseline
    # Step-0 twin of the checkpoint sweep: no checkpoint exists yet, so
    # the eval_ckpt/* series starts from the untuned base model. Skipped
    # on resume (global_step > 0) -- a base-model score logged at step N
    # would be a lie about that step.
    if run_checkpoint_endpoint_eval and global_step == 0:
        pipeline.schedule_base_model_eval(global_step)

    consecutive_failures = 0
    try:
        for train_step in range(global_step + 1, num_train_steps + 1):
            outcome = await _run_one_train_step(
                train_step=train_step,
                num_train_steps=num_train_steps,
                global_step=global_step,
                pipeline=pipeline,
                trainer=trainer,
                ml_logger=ml_logger,
                normalizer=normalizer,
                tasks_per_train_step=tasks_per_train_step,
                sync_every_n_train_steps=sync_every_n_train_steps,
                progress_observer=progress_observer,
            )
            if not outcome.advanced:
                # Empty batch or trainer failure: don't advance global_step
                # or schedule eval. Note the next iteration pulls a FRESH
                # batch -- this step's batch is not re-sent.
                if outcome.failed:
                    consecutive_failures += 1
                    if (
                        consecutive_failures
                        >= _MAX_CONSECUTIVE_TRAIN_STEP_FAILURES
                    ):
                        raise RuntimeError(
                            f"Aborting run after {consecutive_failures}"
                            " consecutive train step failures; last error:"
                            f" {outcome.error}"
                        )
                continue
            consecutive_failures = 0
            global_step = outcome.global_step

            if run_sampler_eval and (
                global_step % eval_every_n_train_steps == 0
                or global_step == num_train_steps
            ):
                pipeline.schedule_eval(global_step)

            # Checkpoint-endpoint eval fires whenever the backend surfaced a
            # deployed checkpoint endpoint this step (its own server-side
            # cadence), independent of the eval cadence above. Additive:
            # logs under eval_ckpt/* alongside the live-sampler eval/*.
            if run_checkpoint_endpoint_eval and outcome.checkpoint_endpoint:
                pipeline.schedule_checkpoint_eval(
                    global_step, outcome.checkpoint_endpoint
                )
    finally:
        pipeline.stop()


async def train(
    *,
    agent: Agent,
    config: TrainConfigLike,
    train_dataset: JsonlDataset,
    eval_dataset: JsonlDataset | None,
    trainer: Trainer,
    ml_logger: Logger,
    eval_agent: Agent | None = None,
    progress_observer: ProgressObserver | None = None,
) -> None:
    """Delegate pipeline construction to ``config.rollout.make(...)``
    and run ``asyncio.gather(pipeline.run(), train_and_eval_loop(...))``.

    The ``config.rollout`` subclass TYPE selects the concrete pipeline:
    an ``AsyncRolloutConfig`` yields an ``AsyncRolloutPipeline``, a
    ``SyncRolloutConfig`` a ``SyncRolloutPipeline``. Mode-specific
    invariants (e.g. the async
    ``train.tasks_per_train_step <= rollout.max_pending_rollout_tasks``
    sizing check) live inside each pipeline ``__init__`` alongside the
    fields they read.

    The composed config exposes training-cadence knobs under
    ``config.train`` (``num_train_steps`` / ``tasks_per_train_step`` /
    ``resume_from_train_step`` / ``sync_every_n_train_steps`` /
    ``reward_normalizer``) and eval-cadence knobs under ``config.eval``
    (``eval_every_n_train_steps``).

    Resume: ``resolve_resume_position`` recovers the train_iter position
    from ``config.train.resume_metrics_path`` at
    ``config.train.resume_from_train_step``.
    """
    pipeline = config.rollout.make(
        run_config=config,
        agent=agent,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_agent=eval_agent,
        ml_logger=ml_logger,
        resume_train_iter_position=resolve_resume_position(
            resume_from_step=config.train.resume_from_train_step,
            resume_metrics_path=config.train.resume_metrics_path,
        ),
    )
    if progress_observer is not None:
        pipeline.set_progress_observer(progress_observer)
    await asyncio.gather(
        pipeline.run(),
        train_and_eval_loop(
            pipeline=pipeline,
            trainer=trainer,
            ml_logger=ml_logger,
            num_train_steps=config.train.num_train_steps,
            tasks_per_train_step=config.train.tasks_per_train_step,
            eval_every_n_train_steps=config.eval.eval_every_n_train_steps,
            reward_normalizer=config.train.reward_normalizer,
            initial_global_step=config.train.resume_from_train_step,
            sync_every_n_train_steps=config.train.sync_every_n_train_steps,
            run_sampler_eval=config.eval.run_sampler_eval,
            run_checkpoint_endpoint_eval=config.eval.run_checkpoint_endpoint_eval,
            progress_observer=progress_observer,
        ),
    )
