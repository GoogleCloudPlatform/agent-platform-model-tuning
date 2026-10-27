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

"""Tests for the reward seam: RewardFn re-export + RewardConfig ABC."""

from __future__ import annotations

import pytest

from core.reward import RewardConfig, RewardFn


def test_rewardfn_reexport_is_protocol_from_protocols():
    from core.protocols import RewardFn as CanonicalRewardFn

    assert RewardFn is CanonicalRewardFn


def test_reward_config_is_abstract():
    with pytest.raises(TypeError):
        RewardConfig()  # abstract: make() not implemented


async def test_concrete_reward_config_make_returns_reward_fn():
    class _FakeRewardFn:
        async def __call__(self, signals):
            return (1.0, signals)

    class _FakeRewardConfig(RewardConfig):
        def make(self) -> RewardFn:
            return _FakeRewardFn()

    fn = _FakeRewardConfig().make()
    reward, stats = await fn("sig")
    assert reward == 1.0 and stats == "sig"
