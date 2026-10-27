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

"""Pure-function tests for WordPuzzleRewardFn.

These call the RewardFn directly with fabricated WordPuzzleRewardSignals
- no env, no agent, no sampler. Behavior must exactly match the previous
in-env reward math when WordPuzzleRewardConfig defaults are used
(regression-pinned).
"""

import pytest

from example_recipe.word_puzzle.reward import (
    WordPuzzleRewardConfig,
    WordPuzzleRewardFn,
)
from example_recipe.word_puzzle.reward_signals import WordPuzzleRewardSignals
from example_recipe.word_puzzle.stats import WordPuzzleStepStats


def _signals(
    is_correct: bool = True,
    generation_token_count: int | None = 100,
    finish_reason: str | None = "STOP",
    max_output_tokens: int | None = None,
) -> WordPuzzleRewardSignals:
    return WordPuzzleRewardSignals(
        is_correct=is_correct,
        generation_token_count=generation_token_count,
        finish_reason=finish_reason,
        max_output_tokens=max_output_tokens,
    )


class TestDefaults:
    """Defaults reproduce the previously-hardcoded module constants."""

    def setup_method(self):
        self.fn = WordPuzzleRewardFn(WordPuzzleRewardConfig())

    async def test_correct_short_generation(self):
        r, stats = await self.fn(_signals(is_correct=True, generation_token_count=100))
        assert r == pytest.approx(1.0)  # +1.0 - 0.0
        assert stats == WordPuzzleStepStats(generation_token_count=100)
        # Explicit malformed check even though the equality above covers
        # it (defaults to False in the dataclass) - guards against future
        # dataclass refactors that might change the default.
        assert stats.malformed_function_call is False

    async def test_incorrect_short_generation(self):
        r, stats = await self.fn(_signals(is_correct=False, generation_token_count=100))
        assert r == pytest.approx(-1.0)
        assert stats.malformed_function_call is False

    async def test_malformed_finish_reason_short_circuits(self):
        r, stats = await self.fn(
            _signals(finish_reason="MALFORMED_FUNCTION_CALL", is_correct=True)
        )
        # is_correct=True is IGNORED once malformed fires: reward reflects
        # the malformed short-circuit, not the raw correctness bit.
        assert r == pytest.approx(-4.0)
        assert stats.malformed_function_call is True

    async def test_none_generation_token_count_no_penalty(self):
        r, stats = await self.fn(_signals(is_correct=True, generation_token_count=None))
        assert r == pytest.approx(1.0)
        assert stats.generation_token_count == 0  # None -> 0 in stats


class TestConfigOverrides:
    """Non-default configs let customers tune every previously-hardcoded knob."""

    async def test_custom_correct_reward(self):
        cfg = WordPuzzleRewardConfig(correct_answer_reward=5.0)
        r, _ = await WordPuzzleRewardFn(cfg)(
            _signals(is_correct=True, generation_token_count=100)
        )
        assert r == pytest.approx(5.0)

    async def test_custom_incorrect_reward(self):
        cfg = WordPuzzleRewardConfig(incorrect_answer_reward=-10.0)
        r, _ = await WordPuzzleRewardFn(cfg)(
            _signals(is_correct=False, generation_token_count=100)
        )
        assert r == pytest.approx(-10.0)

    async def test_custom_malformed_reward(self):
        cfg = WordPuzzleRewardConfig(malformed_output_reward=-100.0)
        r, _ = await WordPuzzleRewardFn(cfg)(
            _signals(finish_reason="MALFORMED_FUNCTION_CALL")
        )
        assert r == pytest.approx(-100.0)

    async def test_custom_malformed_finish_reasons(self):
        cfg = WordPuzzleRewardConfig(malformed_finish_reasons=("STOP", "MAX_TOKENS"))
        # STOP now counts as malformed.
        r, stats = await WordPuzzleRewardFn(cfg)(_signals(finish_reason="STOP"))
        assert r == pytest.approx(-4.0)
        assert stats.malformed_function_call is True
