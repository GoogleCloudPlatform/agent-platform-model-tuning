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

"""Bottom-layer pipeline abstractions.

Holds the pipeline abstractions the trainer and rollout subsystem depend
on, at the very bottom of the dependency graph so nothing here imports
``pipeline.*``. This module also imports NO ``chz`` — it is pure
``core.*`` + stdlib, which lets ``pipeline.*`` (and concrete configs that
carry chz fields) depend on it without creating a ``core -> pipeline``
import cycle.

Three public things live here:

- ``BatchItem`` — the frozen ``(trajectory_group, per-rollout-metrics)``
  pair yielded by a pipeline.
- ``RolloutPipeline`` — the runtime-checkable Protocol the shared trainer
  relies on; both concrete rollout pipelines satisfy it.
- ``RolloutConfig`` — the ABC whose subclass TYPE selects async vs sync
  pipeline; ``make()`` builds a ``RolloutPipeline``. Concrete subclasses
  live next to the pipelines they construct
  (``pipeline.async_rollout_pipeline.AsyncRolloutConfig``,
  ``pipeline.sync_rollout_pipeline.SyncRolloutConfig``).
"""

from __future__ import annotations

import abc
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from core.atif_trajectory import TrajectoryGroup, TrajectoryGroupFilterConfig

__all__ = [
    "BatchItem",
    "RolloutConfig",
    "RolloutPipeline",
]


@dataclasses.dataclass(frozen=True, slots=True)
class BatchItem:
    """One (trajectory_group, per-rollout-metrics) pair yielded by a RolloutPipeline.

    Both pipelines yield lists of these from ``next_batch()``;
    ``train_and_eval_loop`` unpacks them into the training path
    (``.group``) and the observability path (``.metrics``).

    Frozen so accidental mutation of an already-buffered item (e.g. by a
    reward normalizer running in-place on ``.group.trajectories``) can't
    leak to other consumers.
    """

    group: TrajectoryGroup
    metrics: dict[str, Any]


@runtime_checkable
class RolloutPipeline(Protocol):
    """Contract the shared trainer relies on. Both pipelines satisfy it."""

    async def run(self) -> None:
        """Long-running for async (spawns producer + prober + drains
        evals). Minimal for sync (waits on stop(); drains pending
        evals). Called via
        ``asyncio.gather(pipeline.run(), train_and_eval_loop(...))``.
        """
        ...

    def stop(self) -> None:
        """Signal shutdown. Idempotent."""
        ...

    async def next_batch(
        self, n: int, *, step_num: int | None = None
    ) -> tuple[list[BatchItem], int]:
        """Return ``(items, train_iter_position)``.

        Async: blocks until buffer has n contiguous items in dispatch order.
        Sync: runs dynamic-sampling accumulate loop inline; returns when n
        valid groups accumulated. ``step_num`` is used by sync to key its
        pipeline-side sampling metrics; ignored by async.
        """
        ...

    def sample_pipeline_gauges(self) -> dict[str, float]:
        """Sample pipeline-internal gauges at the train-step boundary.

        Called by the trainer immediately BEFORE it blocks on
        ``next_batch`` — that instant is the question these gauges
        answer ("when the trainer asked for data, was any waiting?"), so
        the sampling point is the trainer's, not the pipeline's.

        Returns bare ``pipeline/*`` keys; the caller stamps the
        ``train/`` prefix. A key is omitted when the pipeline has no
        meaningful value for it this step. NOT pure: implementations
        that report a per-step delta advance their window here, so call
        it exactly once per train step.
        """
        ...

    def schedule_eval(self, step_val: int) -> None:
        """Trigger an eval sweep at ``step_val``. Non-blocking.

        Async: spawns background task; drained by ``run()``'s finally.
        Sync: spawns background task; drained by ``next_batch()`` at the
        start of its next call — giving "eval blocks next training step"
        semantics without making ``schedule_eval`` itself async.
        """
        ...

    def schedule_checkpoint_eval(self, step_val: int, endpoint: str) -> None:
        """Trigger an eval sweep against a deployed checkpoint endpoint.

        Non-blocking; same draining semantics as ``schedule_eval``. Builds
        a one-off sampler bound to ``endpoint`` and logs under the
        ``eval_ckpt/*`` prefix. No-op when no eval dataset is configured.
        """
        ...

    def schedule_base_model_eval(self, step_val: int = 0) -> None:
        """Trigger the step-0 ``eval_ckpt/*`` sweep against the base model.

        Same semantics as ``schedule_checkpoint_eval``, but targets the
        publisher (untuned) model — before the first gradient update no
        checkpoint endpoint exists yet. No-op when no eval dataset is
        configured or the backend names no base model.
        """
        ...


@dataclass(frozen=True)
class RolloutConfig(abc.ABC):
    """Async-vs-sync pipeline selector. Subclass TYPE determines the pipeline.

    Concrete subclasses (``AsyncRolloutConfig``, ``SyncRolloutConfig``)
    live in their respective pipeline modules; ``make()`` builds the
    pipeline directly.
    """

    samples_per_task: int = 16
    max_concurrent_samples: int = 1024
    trajectory_group_filter: TrajectoryGroupFilterConfig = field(
        default_factory=TrajectoryGroupFilterConfig
    )

    @abc.abstractmethod
    def make(
        self,
        *,
        run_config,
        agent,
        train_dataset,
        eval_dataset,
        eval_agent,
        ml_logger,
        resume_train_iter_position=None,
    ) -> RolloutPipeline: ...
