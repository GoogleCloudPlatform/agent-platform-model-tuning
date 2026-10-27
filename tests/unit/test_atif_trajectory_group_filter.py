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

"""Tests for src/core/atif_trajectory.py: TrajectoryGroupFilter.

Ports the invariants from the deleted tests/unit/test_rl_trajectory.py
(deleted in commit 7999f3aa as part of the ScoredTrajectory migration).
The filter's semantics are unchanged from the legacy version; only the
field-access shape changed (``scored.reward.reward`` instead of
``t.reward`` and ``atif_extras.get_rollout_status(g.trajectory)``
instead of ``t.stats.status``).

Notes on actual filter behavior (differs slightly from a naive spec):
  * The filter is invoked via ``__call__`` (not ``.filter()``) and
    returns ``(kept, GroupFilterOutcome)``. ``GroupFilterOutcome.KEPT``
    is the empty string, so ``reason == ""`` still reads as "kept,
    nothing to report" — except for a zero-variance group that survived
    the coin flip, which reports ``KEPT_ZERO_VARIANCE``.
  * ``max_transient_failure_fraction`` default is **0.15**; comparison
    is strictly ``fraction > threshold`` (not ``>=``).
  * The zero-variance gate is *probabilistic*: it drops only if
    ``random.random() > zero_variance_keep_prob``. Tests here pin the
    probability to 0.0 (always drop) or 1.0 (always keep) to be
    deterministic.
"""

from __future__ import annotations

import math

from core.atif_trajectory import Agent, Trajectory

from core import atif_extras
from core.atif_trajectory import (
    GroupFilterOutcome,
    ScoredTrajectory,
    Reward,
    StepStats,
    TrajectoryGroup,
    TrajectoryGroupFilter,
    TrajectoryGroupFilterConfig,
    Step,
    TrajectoryStep,
)
from core.rollout_status import RolloutStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scored(
    *, reward: float, status: RolloutStatus = RolloutStatus.OK
) -> ScoredTrajectory:
    """Build a minimal ScoredTrajectory with the given reward + status."""
    us = Step(step_id=1, source="user", message="task")
    agent_step = TrajectoryStep.from_wire(
        step_id=2,
        generation="g",
        stats=StepStats(status=status),
    )
    traj = Trajectory(
        schema_version="ATIF-v1.7",
        session_id="s",
        agent=Agent(name="t", version="0", model_name="m"),
        steps=[us, agent_step],
    )
    # The filter reads the group-level status via atif_extras; make sure it
    # matches the status baked into the last agent step.
    atif_extras.set_rollout_status(traj, status)
    return ScoredTrajectory(trajectory=traj, reward=Reward(reward=reward))


def _make_group(
    rewards: list[float],
    statuses: list[RolloutStatus] | None = None,
) -> TrajectoryGroup:
    """Build a TrajectoryGroup from parallel reward/status lists."""
    if statuses is None:
        statuses = [RolloutStatus.OK] * len(rewards)
    assert len(rewards) == len(statuses)
    scored = [_make_scored(reward=r, status=s) for r, s in zip(rewards, statuses)]
    return TrajectoryGroup(task_data=None, scored_trajectories=scored)


def _make_filter(**overrides) -> TrajectoryGroupFilter:
    return TrajectoryGroupFilterConfig(**overrides).make()


# ---------------------------------------------------------------------------
# 1. Transient-failure fraction gate
# ---------------------------------------------------------------------------


def test_transient_failure_fraction_below_threshold_keeps_group():
    """1 transient out of 10 (10%) < default 15% => group survives transient gate."""
    statuses = [RolloutStatus.SAMPLING_FAILURE_TIMEOUT] + [RolloutStatus.OK] * 9
    # Vary rewards so the zero-variance gate can't fire.
    rewards = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    group = _make_group(rewards=rewards, statuses=statuses)
    # NaN reward for the transient one so it gets NaN-dropped rather than
    # skewing variance; but non-transient trajectories keep varied rewards.
    group.scored_trajectories[0].reward.reward = math.nan
    kept, reason = _make_filter(zero_variance_keep_prob=0.0)(group)
    assert kept is True
    assert reason == ""


def test_transient_failure_fraction_above_threshold_drops_group():
    """4 transient out of 10 (40%) > default 15% => dropped as 'Too many transient failures'."""
    statuses = [RolloutStatus.SAMPLING_FAILURE_TIMEOUT] * 4 + [RolloutStatus.OK] * 6
    rewards = [1.0] * 10
    group = _make_group(rewards=rewards, statuses=statuses)
    kept, reason = _make_filter()(group)
    assert kept is False
    assert reason is GroupFilterOutcome.TOO_MANY_TRANSIENT_FAILURES


def test_transient_failure_fraction_configurable():
    """Custom threshold 0.5 lets 4/10 transient failures (40%) pass the gate."""
    statuses = [RolloutStatus.SAMPLING_FAILURE_TIMEOUT] * 4 + [RolloutStatus.OK] * 6
    # Give surviving trajectories varied rewards so we don't hit the
    # variance gate incidentally.
    rewards = [math.nan] * 4 + [0.0, 1.0, 0.5, 0.25, 0.75, 0.1]
    group = _make_group(rewards=rewards, statuses=statuses)
    kept, reason = _make_filter(
        max_transient_failure_fraction=0.5, zero_variance_keep_prob=0.0
    )(group)
    assert kept is True
    assert reason == ""


def test_transient_gate_disabled_when_threshold_is_none():
    """max_transient_failure_fraction=None disables the transient gate entirely."""
    # 100% transient failures — would otherwise trip the gate immediately.
    statuses = [RolloutStatus.SAMPLING_FAILURE_TIMEOUT] * 5
    rewards = [math.nan] * 5
    group = _make_group(rewards=rewards, statuses=statuses)
    kept, reason = _make_filter(max_transient_failure_fraction=None)(group)
    # Transient gate skipped -> NaN drop leaves 0 -> "<2 valid" reason.
    assert kept is False
    assert reason is GroupFilterOutcome.TOO_FEW_VALID_TRAJECTORIES


def test_transient_gate_uses_original_count_not_post_nan_filter():
    """Transient fraction is measured on ORIGINAL group size, not post-NaN drop.

    Otherwise NaN-dropping (which typically co-occurs with transient
    failures) would artificially inflate the denominator's shrinkage and
    make the gate flap.
    """
    # 2/10 = 20% > 15% default -> should drop on the transient gate.
    # If the filter mistakenly measured 2 / (10 - NaN_count), the number
    # would look different, but here both formulations agree on "drop",
    # so the discriminator is the *reason*: it must be the transient
    # reason, produced before any NaN drop even happens.
    statuses = [RolloutStatus.SAMPLING_FAILURE_TIMEOUT] * 2 + [RolloutStatus.OK] * 8
    rewards = [math.nan] * 2 + [1.0] * 8
    group = _make_group(rewards=rewards, statuses=statuses)
    kept, reason = _make_filter()(group)
    assert kept is False
    assert reason is GroupFilterOutcome.TOO_MANY_TRANSIENT_FAILURES


def test_non_transient_sampling_failures_do_not_count_toward_transient_fraction():
    """MALFORMED_FUNCTION_CALL is a sampling failure but NOT transient.

    It must NOT contribute to the transient-failure fraction; it should
    fall through to whatever downstream gate applies (here: zero variance).
    """
    statuses = [RolloutStatus.SAMPLING_FAILURE_MALFORMED_FUNCTION_CALL] * 5 + [
        RolloutStatus.OK
    ] * 5
    rewards = [1.0] * 10  # all identical -> zero variance
    group = _make_group(rewards=rewards, statuses=statuses)
    kept, reason = _make_filter(zero_variance_keep_prob=0.0)(group)
    # Transient gate: 0 transient -> passes. NaN drop: none. <2 check: 10 -> passes.
    # Zero-variance gate with keep_prob=0.0 -> always drops.
    assert kept is False
    assert reason is GroupFilterOutcome.ZERO_VARIANCE


# ---------------------------------------------------------------------------
# 2. NaN-reward drop (in-place mutation)
# ---------------------------------------------------------------------------


def test_nan_rewards_are_dropped_in_place():
    """Trajectories with NaN reward are removed from ``group.scored_trajectories``."""
    group = _make_group(rewards=[1.0, math.nan, 0.5, math.nan, 0.7])
    orig_len = len(group.scored_trajectories)
    _make_filter(zero_variance_keep_prob=1.0)(group)
    # After filter, NaN entries are gone (3 valid survive).
    assert len(group.scored_trajectories) == 3
    assert len(group.scored_trajectories) < orig_len
    for g in group.scored_trajectories:
        assert not math.isnan(g.reward.reward)


# ---------------------------------------------------------------------------
# 3. <2 valid gate
# ---------------------------------------------------------------------------


def test_less_than_2_valid_drops_group():
    """After NaN drop, if <2 remain: drop with 'Less than 2 valid trajectories'."""
    group = _make_group(rewards=[1.0, math.nan, math.nan, math.nan])
    kept, reason = _make_filter()(group)
    assert kept is False
    assert reason is GroupFilterOutcome.TOO_FEW_VALID_TRAJECTORIES


def test_exactly_2_valid_survives_the_gate():
    """Boundary: exactly 2 valid rewards (with variance) pass through."""
    group = _make_group(rewards=[1.0, 0.5, math.nan])
    kept, reason = _make_filter(zero_variance_keep_prob=0.0)(group)
    assert kept is True
    assert reason == ""


# ---------------------------------------------------------------------------
# 4. Zero-variance gate (probabilistic)
# ---------------------------------------------------------------------------


def test_zero_variance_drops_group_when_keep_prob_zero():
    """All-same-reward group is dropped when ``zero_variance_keep_prob=0``."""
    group = _make_group(rewards=[1.0, 1.0, 1.0, 1.0])
    kept, reason = _make_filter(zero_variance_keep_prob=0.0)(group)
    assert kept is False
    assert reason is GroupFilterOutcome.ZERO_VARIANCE


def test_zero_variance_kept_when_keep_prob_one():
    """All-same-reward group is deterministically KEPT when keep_prob=1.

    The outcome is ``KEPT_ZERO_VARIANCE``, not plain ``KEPT``: at
    keep_prob=1 nothing is ever dropped for zero variance, so this is
    the only remaining signal that the group tied at all.
    """
    group = _make_group(rewards=[1.0, 1.0, 1.0, 1.0])
    kept, reason = _make_filter(zero_variance_keep_prob=1.0)(group)
    assert kept is True
    assert reason is GroupFilterOutcome.KEPT_ZERO_VARIANCE
    assert reason.is_zero_variance is True
    assert reason.is_drop is False


def test_nonzero_variance_keeps_group_even_with_keep_prob_zero():
    """Any real variance => group survives regardless of keep_prob."""
    # Delta larger than the isclose(abs_tol=1e-5) tolerance in the filter.
    group = _make_group(rewards=[1.0, 1.01, 1.0])
    kept, reason = _make_filter(zero_variance_keep_prob=0.0)(group)
    assert kept is True
    assert reason == ""


# ---------------------------------------------------------------------------
# 5. Gate order
# ---------------------------------------------------------------------------


def test_transient_gate_fires_before_variance_gate():
    """If BOTH transient>threshold AND zero variance would apply, transient wins."""
    statuses = [RolloutStatus.SAMPLING_FAILURE_TIMEOUT] * 4 + [RolloutStatus.OK] * 6
    # After NaN drop, the 6 surviving rewards would all be 1.0 -> zero variance.
    rewards = [math.nan] * 4 + [1.0] * 6
    group = _make_group(rewards=rewards, statuses=statuses)
    kept, reason = _make_filter(zero_variance_keep_prob=0.0)(group)
    assert kept is False
    assert reason is GroupFilterOutcome.TOO_MANY_TRANSIENT_FAILURES


def test_less_than_2_gate_fires_before_variance_gate():
    """If BOTH <2 valid AND (trivially) zero variance apply, <2 reason wins."""
    group = _make_group(rewards=[1.0, math.nan, math.nan])
    kept, reason = _make_filter(zero_variance_keep_prob=0.0)(group)
    assert kept is False
    assert reason is GroupFilterOutcome.TOO_FEW_VALID_TRAJECTORIES


# ---------------------------------------------------------------------------
# 6. Config knob smoke
# ---------------------------------------------------------------------------


def test_filter_config_defaults():
    """Sanity: config exposes documented defaults."""
    cfg = TrajectoryGroupFilterConfig()
    assert cfg.max_transient_failure_fraction == 0.15
    assert cfg.zero_variance_keep_prob == 0.3
