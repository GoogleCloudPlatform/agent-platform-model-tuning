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

"""Recipe-owned reward config + RewardFn for word_puzzle."""

import dataclasses
import math

from core.rollout_status import RolloutStatus
from example_recipe.word_puzzle.reward_signals import WordPuzzleRewardSignals
from example_recipe.word_puzzle.stats import WordPuzzleStepStats


@dataclasses.dataclass
class WordPuzzleRewardConfig:
    """Tunable knobs for word_puzzle reward shaping."""

    # Malformed-output short-circuit.
    malformed_finish_reasons: tuple[str, ...] = ("MALFORMED_FUNCTION_CALL",)
    malformed_output_reward: float = -4.0
    # Answer payoff.
    correct_answer_reward: float = 1.0
    incorrect_answer_reward: float = -1.0
    # Structural-failure penalty for a rollout that exhausts the step
    # budget without a terminal submit (``status == MAX_STEP_LIMIT``).
    max_step_reward: float = -10.0
    # ---- Step-efficiency bonus (correct answers only) -------------------
    # Added on top of ``correct_answer_reward`` when a CORRECT answer is
    # submitted in few steps, so fast solves outrank slow ones instead of
    # tying at a flat +1. Linear ramp: full bonus at or below
    # ``full_bonus_at_steps``, zero at or above ``zero_bonus_at_steps``.
    #
    # ``step_bonus_max = 0.0`` disables it (the default), so every existing
    # operating point keeps its current reward shape. Keep the bonus well
    # under ``correct_answer_reward - incorrect_answer_reward`` so a fast
    # WRONG answer can never outrank a slow CORRECT one.
    step_bonus_max: float = 0.0
    full_bonus_at_steps: int = 4
    zero_bonus_at_steps: int = 30
    # ---- Token-efficiency bonus (correct answers only) ------------------
    # Same idea as the step bonus, but on the quantity actually paid for:
    # generation tokens summed over the whole rollout. The ramp is
    # LOGARITHMIC because rollout token totals are log-uniform (roughly
    # 1k..100k), so a linear ramp would spend its whole dynamic range on
    # the most expensive quartile.
    #
    # ``token_bonus_max = 0.0`` disables it (the default). Keep
    # ``token_bonus_max + step_bonus_max`` well under
    # ``correct_answer_reward - incorrect_answer_reward`` so no bonus can
    # let a cheap WRONG answer outrank an expensive CORRECT one.
    token_bonus_max: float = 0.0
    full_bonus_at_tokens: int = 8192
    zero_bonus_at_tokens: int = 131072
    # Token counts are floored to a multiple of this before the ramp, so
    # the reward moves in ~1 KiB steps instead of jittering per token.
    token_bucket_size: int = 1024


class WordPuzzleRewardFn:
    """Pure reward-shaping function for word_puzzle.

    Reward =
      * ``max_step_reward`` if the rollout hit ``MAX_STEP_LIMIT``.
      * ``malformed_output_reward`` if malformed finish_reason.
      * else ``correct_answer_reward if is_correct else incorrect_answer_reward``,
        plus a step-efficiency bonus and a token-efficiency bonus when the
        answer is correct (see ``step_bonus_for`` / ``token_bonus_for``; each
        is zero unless its ``*_max`` knob is configured).
    """

    def __init__(self, config: WordPuzzleRewardConfig):
        self._config = config

    def step_bonus_for(self, step_count: int | None) -> float:
        """Bonus for solving in ``step_count`` steps; 0.0 when disabled.

        Linear ramp between the two thresholds: full bonus at or below
        ``full_bonus_at_steps``, nothing at or above ``zero_bonus_at_steps``.
        Callers apply this to CORRECT answers only.
        """
        cfg = self._config
        if cfg.step_bonus_max <= 0.0 or step_count is None:
            return 0.0
        full_at, zero_at = cfg.full_bonus_at_steps, cfg.zero_bonus_at_steps
        if zero_at <= full_at:
            # Degenerate/misconfigured ramp: treat it as a step function so
            # the knob still behaves monotonically instead of dividing by <=0.
            return cfg.step_bonus_max if step_count <= full_at else 0.0
        frac = (zero_at - step_count) / (zero_at - full_at)
        return cfg.step_bonus_max * min(1.0, max(0.0, frac))

    def token_bonus_for(self, token_count: int | None) -> float:
        """Bonus for solving with ``token_count`` generation tokens.

        ``token_count`` is the WHOLE-rollout total. It is floored to a
        ``token_bucket_size`` multiple first, then mapped through a
        logarithmic ramp: full bonus at or below ``full_bonus_at_tokens``,
        nothing at or above ``zero_bonus_at_tokens``. Callers apply this
        to CORRECT answers only.

        ``None`` means no step reported a token count. It earns nothing:
        treating a missing count as 0 tokens would hand out the full
        bonus for absent data.
        """
        cfg = self._config
        if cfg.token_bonus_max <= 0.0 or token_count is None:
            return 0.0
        bucket = max(int(cfg.token_bucket_size), 1)
        bucketed = (int(token_count) // bucket) * bucket
        full_at, zero_at = cfg.full_bonus_at_tokens, cfg.zero_bonus_at_tokens
        if bucketed <= full_at:
            return cfg.token_bonus_max
        # Degenerate/misconfigured ramp (or a non-positive threshold, which
        # log() cannot take): fall back to a step function so the knob still
        # behaves monotonically.
        if bucketed >= zero_at or zero_at <= full_at or full_at <= 0:
            return 0.0
        frac = (math.log(zero_at) - math.log(bucketed)) / (
            math.log(zero_at) - math.log(full_at)
        )
        return cfg.token_bonus_max * min(1.0, max(0.0, frac))

    async def __call__(
        self, signals: WordPuzzleRewardSignals
    ) -> tuple[float, WordPuzzleStepStats]:
        cfg = self._config
        gen_tokens_int = int(signals.generation_token_count or 0)
        if signals.status == RolloutStatus.MAX_STEP_LIMIT:
            return (
                cfg.max_step_reward,
                WordPuzzleStepStats(
                    generation_token_count=gen_tokens_int,
                    malformed_function_call=False,
                ),
            )
        if (
            signals.finish_reason is not None
            and signals.finish_reason in cfg.malformed_finish_reasons
        ):
            return (
                cfg.malformed_output_reward,
                WordPuzzleStepStats(
                    malformed_function_call=True,
                    generation_token_count=gen_tokens_int,
                ),
            )
        # Efficiency bonus applies to correct answers only: a wrong answer
        # must never profit from being fast.
        step_bonus = (
            self.step_bonus_for(signals.step_count) if signals.is_correct else 0.0
        )
        token_bonus = (
            self.token_bonus_for(signals.rollout_generation_token_count)
            if signals.is_correct
            else 0.0
        )
        answer_reward = (
            cfg.correct_answer_reward
            if signals.is_correct
            else cfg.incorrect_answer_reward
        )
        return (
            answer_reward + step_bonus + token_bonus,
            WordPuzzleStepStats(
                generation_token_count=gen_tokens_int,
                malformed_function_call=False,
                solve_step_count=int(signals.step_count or 0),
                step_bonus=step_bonus,
                token_bonus=token_bonus,
                rollout_generation_token_count=int(
                    signals.rollout_generation_token_count or 0
                ),
            ),
        )
