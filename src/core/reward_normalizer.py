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

import numpy as np

from core import atif_trajectory
from core.protocols import RewardNormalizer

# Re-export ``RewardNormalizer`` for back-compat; the canonical
# definition lives in ``core.protocols``.
__all__ = [
    "RewardNormalizer",
    "GRPONormalizer",
    "CenteringNormalizer",
]


class GRPONormalizer(RewardNormalizer):
    """GRPO-style per-prompt reward normalization.

    For each TrajectoryGroup, subtracts the per-prompt mean reward and
    divides by the population std (``ddof=0``); matches the DeepSeek
    GRPO advantage estimator. Zero-variance groups are floored to
    ``epsilon = 1e-10``, so identical rewards produce ``normalized_reward=0``.
    """

    _EPSILON = 1e-10

    def normalize(
        self,
        trajectory_groups: list[atif_trajectory.TrajectoryGroup],
    ) -> list[atif_trajectory.TrajectoryGroup]:
        if not trajectory_groups:
            return trajectory_groups

        for group in trajectory_groups:
            rewards = [g.reward.reward for g in group.scored_trajectories]
            if not rewards:
                continue
            mean_reward = sum(rewards) / len(rewards)
            std = max(float(np.std(rewards, ddof=0)), self._EPSILON)
            for g in group.scored_trajectories:
                g.reward.normalized_reward = (g.reward.reward - mean_reward) / std

        return trajectory_groups


class CenteringNormalizer(RewardNormalizer):
    """Centers rewards per prompt by subtracting the per-prompt mean
    (no standardization)."""

    def normalize(
        self,
        trajectory_groups: list[atif_trajectory.TrajectoryGroup],
    ) -> list[atif_trajectory.TrajectoryGroup]:
        if not trajectory_groups:
            return trajectory_groups

        for group in trajectory_groups:
            rewards = [g.reward.reward for g in group.scored_trajectories]
            if not rewards:
                continue
            mean_reward = sum(rewards) / len(rewards)
            for g in group.scored_trajectories:
                g.reward.normalized_reward = g.reward.reward - mean_reward

        return trajectory_groups
