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

"""Reward seam for the RunConfig model.

Two distinct things:

* ``RewardFn`` is the EXISTING reward contract (``signals -> (reward, stats)``,
  one scalar per trajectory), canonical in ``core.protocols`` and re-exported
  here unchanged. Per-step credit assignment is the separate ``advantage`` seam
  (``step.extra["advantage"]``), NOT this.
* ``RewardConfig`` is the knob-config module held on ``RunConfig.reward``. Its
  ``make()`` builds the ``RewardFn``, parallel to AgentConfig/BackendConfig/
  RolloutConfig.make().
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

from core.protocols import RewardFn

__all__ = ["RewardFn", "RewardConfig"]


@dataclass(frozen=True)
class RewardConfig(abc.ABC):
    """Knob config for a reward fn. make() builds the RewardFn (1 scalar/traj).

    Carries NO reward-value fields itself: a reward floor / ceiling is a
    reward-fn-specific concept, not a universal one. A reward fn that assigns a
    single failure floor to no-signal trajectories (and needs a consumer -- e.g.
    an agent short-circuiting a structural failure before the reward fn runs --
    to apply the SAME floor) declares its own ``lowest_reward`` field on its
    concrete config, and exposes it on the built reward fn as the single source
    of truth. A reward fn that assigns rewards by discrete outcome (distinct
    correct / incorrect / malformed payoffs) declares no such field. Keeping the
    base ABC field-free avoids forcing an unused ``lowest_reward`` onto
    categorical-reward recipes.

    Frozen dataclass ABC, mirroring ``AgentConfig``: subclasses are likewise
    ``@dataclass(frozen=True)`` and declare their own defaulted fields.
    """

    @abc.abstractmethod
    def make(self) -> RewardFn: ...
