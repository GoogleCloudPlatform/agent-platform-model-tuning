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

"""Framework-level tests for the RewardFn protocol.

The protocol is generic and structural (typing.Protocol), so these tests
are mostly documentation-shaped: they instantiate hand-rolled RewardFn
implementations, exercise them at runtime, and confirm the top-level
re-export works.
"""

import dataclasses

import pytest

from core.reward import RewardFn
from core.reward import RewardFn as RewardFnFromModule


@dataclasses.dataclass
class _FakeSignals:
    x: float


@dataclasses.dataclass
class _FakeStats:
    doubled: float


def test_protocol_re_exported_from_top_level():
    assert RewardFn is RewardFnFromModule


async def test_concrete_class_satisfies_protocol_and_returns_tuple():
    class _Doubler:
        async def __call__(self, signals: _FakeSignals) -> tuple[float, _FakeStats]:
            r = signals.x * 2.0
            return r, _FakeStats(doubled=r)

    fn: RewardFn[_FakeSignals, _FakeStats] = _Doubler()
    reward, stats = await fn(_FakeSignals(x=3.0))
    assert reward == pytest.approx(6.0)
    assert stats.doubled == pytest.approx(6.0)


async def test_async_function_satisfies_protocol():
    async def _fake(signals: _FakeSignals) -> tuple[float, _FakeStats]:
        return (signals.x, _FakeStats(doubled=signals.x))

    fn: RewardFn[_FakeSignals, _FakeStats] = _fake
    reward, stats = await fn(_FakeSignals(x=1.5))
    assert reward == pytest.approx(1.5)
    assert stats.doubled == pytest.approx(1.5)
