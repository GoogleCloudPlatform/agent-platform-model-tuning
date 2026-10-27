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

"""Tests for core.reward_normalizer (post-ScoredTrajectory migration)."""

import math

import pytest
from core.atif_trajectory import Agent as AtifAgent, Trajectory as AtifTrajectory

from core.atif_trajectory import (
    ScoredTrajectory,
    Reward,
    StepStats,
    TrajectoryGroup,
    Step,
    TrajectoryStep as AtifTrajectoryStep,
)
from core.reward_normalizer import (
    CenteringNormalizer,
    GRPONormalizer,
    RewardNormalizer,
)


def _scored(reward: float) -> ScoredTrajectory:
    us = Step(step_id=1, source="user", message="p")
    a = AtifTrajectoryStep.from_wire(step_id=2, generation="g", stats=StepStats())
    traj = AtifTrajectory(
        schema_version="ATIF-v1.7",
        session_id="s",
        agent=AtifAgent(name="t", version="0", model_name="m"),
        steps=[us, a],
    )
    return ScoredTrajectory(trajectory=traj, reward=Reward(reward=reward))


def _group(rewards) -> TrajectoryGroup:
    return TrajectoryGroup(
        task_data=None,
        scored_trajectories=[_scored(r) for r in rewards],
    )


class TestGRPONormalizer:
    """GRPO: per-prompt (mean, std) normalization within each TrajectoryGroup."""

    def test_empty_input_returns_empty(self):
        assert GRPONormalizer().normalize([]) == []

    def test_single_group_two_trajectories(self):
        g = _group([0.0, 1.0])
        GRPONormalizer().normalize([g])
        assert g.scored_trajectories[0].reward.normalized_reward == pytest.approx(-1.0)
        assert g.scored_trajectories[1].reward.normalized_reward == pytest.approx(1.0)

    def test_multi_group_uses_per_group_std_not_global(self):
        g1 = _group([0.0, 2.0])
        g2 = _group([1.0, 1.0])
        GRPONormalizer().normalize([g1, g2])
        assert g1.scored_trajectories[0].reward.normalized_reward == pytest.approx(-1.0)
        assert g1.scored_trajectories[1].reward.normalized_reward == pytest.approx(1.0)
        assert g2.scored_trajectories[0].reward.normalized_reward == pytest.approx(0.0)
        assert g2.scored_trajectories[1].reward.normalized_reward == pytest.approx(0.0)

    def test_zero_variance_uses_epsilon_floor(self):
        g = _group([0.5, 0.5, 0.5])
        GRPONormalizer().normalize([g])
        for gt in g.scored_trajectories:
            assert gt.reward.normalized_reward == pytest.approx(0.0)

    def test_three_trajectories_matches_population_std(self):
        g = _group([1.0, 2.0, 3.0])
        GRPONormalizer().normalize([g])
        std = math.sqrt(2 / 3)
        assert g.scored_trajectories[0].reward.normalized_reward == pytest.approx(
            -1 / std
        )
        assert g.scored_trajectories[1].reward.normalized_reward == pytest.approx(0.0)
        assert g.scored_trajectories[2].reward.normalized_reward == pytest.approx(
            1 / std
        )

    def test_empty_group_skipped(self):
        empty = TrajectoryGroup(task_data=None, scored_trajectories=[])
        nonempty = _group([0.0, 1.0])
        out = GRPONormalizer().normalize([empty, nonempty])
        assert out is not None
        assert empty.scored_trajectories == []
        assert nonempty.scored_trajectories[
            0
        ].reward.normalized_reward == pytest.approx(-1.0)
        assert nonempty.scored_trajectories[
            1
        ].reward.normalized_reward == pytest.approx(1.0)

    def test_returns_same_list_object(self):
        groups = [_group([0.0, 1.0])]
        out = GRPONormalizer().normalize(groups)
        assert out is groups


class TestCenteringNormalizer:
    """Centering-only: subtract per-prompt mean; no std scaling."""

    def test_empty_input_returns_empty(self):
        assert CenteringNormalizer().normalize([]) == []

    def test_single_group_centers_around_mean(self):
        g = _group([0.0, 1.0])
        CenteringNormalizer().normalize([g])
        assert g.scored_trajectories[0].reward.normalized_reward == pytest.approx(-0.5)
        assert g.scored_trajectories[1].reward.normalized_reward == pytest.approx(0.5)

    def test_multi_group_centers_each_independently(self):
        g1 = _group([0.0, 2.0])
        g2 = _group([10.0, 12.0])
        CenteringNormalizer().normalize([g1, g2])
        assert g1.scored_trajectories[0].reward.normalized_reward == pytest.approx(-1.0)
        assert g1.scored_trajectories[1].reward.normalized_reward == pytest.approx(1.0)
        assert g2.scored_trajectories[0].reward.normalized_reward == pytest.approx(-1.0)
        assert g2.scored_trajectories[1].reward.normalized_reward == pytest.approx(1.0)

    def test_zero_variance_yields_zeros(self):
        g = _group([0.5, 0.5, 0.5])
        CenteringNormalizer().normalize([g])
        for gt in g.scored_trajectories:
            assert gt.reward.normalized_reward == pytest.approx(0.0)

    def test_empty_group_skipped(self):
        empty = TrajectoryGroup(task_data=None, scored_trajectories=[])
        nonempty = _group([0.0, 1.0])
        CenteringNormalizer().normalize([empty, nonempty])
        assert empty.scored_trajectories == []
        assert nonempty.scored_trajectories[
            0
        ].reward.normalized_reward == pytest.approx(-0.5)
        assert nonempty.scored_trajectories[
            1
        ].reward.normalized_reward == pytest.approx(0.5)

    def test_returns_same_list_object(self):
        groups = [_group([0.0, 1.0])]
        out = CenteringNormalizer().normalize(groups)
        assert out is groups


class TestRewardNormalizerConfig:
    def test_centering_config_make_returns_centering_normalizer(self):
        from core.reward_normalizer_config import CenteringNormalizerConfig

        n = CenteringNormalizerConfig().make()
        assert isinstance(n, CenteringNormalizer)
        assert isinstance(n, RewardNormalizer)

    def test_grpo_config_make_returns_grpo_normalizer(self):
        from core.reward_normalizer_config import GRPONormalizerConfig

        n = GRPONormalizerConfig().make()
        assert isinstance(n, GRPONormalizer)
        assert isinstance(n, RewardNormalizer)

    def test_make_returns_fresh_instances(self):
        from core.reward_normalizer_config import GRPONormalizerConfig

        cfg = GRPONormalizerConfig()
        assert cfg.make() is not cfg.make()

    def test_configs_are_frozen(self):
        import dataclasses

        from core.reward_normalizer_config import (
            CenteringNormalizerConfig,
            GRPONormalizerConfig,
        )

        for cls in (CenteringNormalizerConfig, GRPONormalizerConfig):
            assert dataclasses.is_dataclass(cls)
            assert cls.__dataclass_params__.frozen is True

    def test_configs_are_reward_normalizer_config_subclasses(self):
        from core.reward_normalizer_config import (
            CenteringNormalizerConfig,
            GRPONormalizerConfig,
            RewardNormalizerConfig,
        )

        assert issubclass(CenteringNormalizerConfig, RewardNormalizerConfig)
        assert issubclass(GRPONormalizerConfig, RewardNormalizerConfig)
