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

"""CompositeTrajectoryCallback — fan-out wrapper for multiple TrajectoryCallbacks.

Wraps a list of ``TrajectoryCallback`` instances and delegates each
lifecycle method to all of them. Exceptions from individual callbacks
are logged and swallowed so one failing callback doesn't block others.
"""

from __future__ import annotations

import logging
from typing import Any

from core.agent import TrajectoryCallback
from core.atif_trajectory import Trajectory
from utils.raw_capture import CaptureContext

logger = logging.getLogger(__name__)


class CompositeTrajectoryCallback:
    """Delegates TrajectoryCallback methods to multiple wrapped callbacks.

    Satisfies the ``TrajectoryCallback`` protocol — the Agent receives
    this as its single callback, unaware that multiple sinks are behind it.
    """

    def __init__(self, callbacks: list[TrajectoryCallback]) -> None:
        self._callbacks = list(callbacks)

    async def _fanout(self, hook: str, *args: Any, **kwargs: Any) -> None:
        """Await ``hook`` on every callback that implements it.

        The lifecycle hooks are best-effort observers and callbacks may
        implement only the subset they care about (e.g. ``AtifDiskWriter``
        has no ``on_trajectory_dropped``). Missing hooks are skipped rather
        than raising ``AttributeError``; genuine failures are logged with
        the offending callback's class name and swallowed so one bad sink
        can't block the others.
        """
        for cb in self._callbacks:
            fn = getattr(cb, hook, None)
            if fn is None:
                continue
            try:
                await fn(*args, **kwargs)
            except Exception:
                logger.warning(
                    "Callback %s.%s failed", type(cb).__name__, hook, exc_info=True
                )

    async def on_job_started(self, config: Any) -> None:
        await self._fanout("on_job_started", config)

    async def on_trajectory_started(
        self,
        task_data: Any,
        traj: Trajectory | None = None,
    ) -> None:
        await self._fanout("on_trajectory_started", task_data, traj)

    async def on_trajectory_step_completed(
        self,
        task_data: Any,
        traj: Trajectory | None = None,
    ) -> None:
        await self._fanout("on_trajectory_step_completed", task_data, traj)

    async def on_trajectory_produced(
        self,
        scored: Any,
        task_data: Any,
        traj: Trajectory | None = None,
    ) -> None:
        await self._fanout("on_trajectory_produced", scored, task_data, traj)

    async def on_trajectory_dropped(
        self,
        task_data: Any,
        traj: Trajectory | None = None,
        reason: str = "",
    ) -> None:
        await self._fanout("on_trajectory_dropped", task_data, traj, reason)

    async def on_batch_assembled(
        self,
        step: int,
        trajectory_ids: list[str],
        mean_reward: float | None = None,
    ) -> None:
        await self._fanout(
            "on_batch_assembled", step, trajectory_ids, mean_reward=mean_reward
        )

    async def on_job_completed(self, aggregate_stats: dict[str, Any]) -> None:
        await self._fanout("on_job_completed", aggregate_stats)

    async def capture_ctx_for(
        self,
        *,
        task_id: str,
        rollout_idx: int,
        session_id: str,
    ) -> CaptureContext | None:
        """Return the first non-None CaptureContext from the wrapped callbacks."""
        for cb in self._callbacks:
            fn = getattr(cb, "capture_ctx_for", None)
            if fn is None:
                continue
            try:
                ctx = await fn(
                    task_id=task_id, rollout_idx=rollout_idx, session_id=session_id
                )
                if ctx is not None:
                    return ctx
            except Exception:
                logger.warning(
                    "Callback %s.capture_ctx_for failed",
                    type(cb).__name__,
                    exc_info=True,
                )
        return None
