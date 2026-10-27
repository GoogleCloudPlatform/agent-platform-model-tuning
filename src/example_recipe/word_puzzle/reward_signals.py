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

"""Raw signals input to WordPuzzleRewardFn.

Populated by ``WordPuzzleEnv.step`` before reward shaping and consumed by
``WordPuzzleRewardFn.__call__`` to produce
``(reward, WordPuzzleStepStats)``.
"""

import dataclasses

from core.rollout_status import RolloutStatus


@dataclasses.dataclass
class WordPuzzleRewardSignals:
    """What the env observes before reward shaping.

    Attributes:
        is_correct: Env-computed exact-match between parsed_answer and
            reference; the RewardFn maps it to a scalar payoff.
        generation_token_count: Best-effort generation length; None if
            unavailable.
        finish_reason: Sampler-reported finish_reason (e.g. "STOP",
            "MAX_TOKENS", "MALFORMED_FUNCTION_CALL"), or None. Used by
            the malformed short-circuit.
        max_output_tokens: The env's configured max_output_tokens (or
            None); available for RewardFn implementations that need to
            reason about the generation-length cap.
        depth: The task's ``depth`` tag (i.e. ``num_transformations``),
            or None if the task_data has no tags. Available for RewardFn
            implementations that want to apply depth-conditioned shaping
            (e.g. per-depth length bonuses for correct answers). Populated
            from ``task_data.tags['depth']`` by ``WordPuzzleEnv.step``.
        status: Rollout status this signal is being shaped for. Defaults
            to ``OK`` for env-terminated rollouts (real submit call). The
            agent-solver reward configs shape a distinct
            ``max_step_reward`` for ``MAX_STEP_LIMIT``.
        step_count: 1-based index of the step that terminated the rollout
            (i.e. how many agent steps the task took to solve). ``None``
            for single-step envs that have no step budget. Available for
            RewardFn implementations that reward step efficiency.
        rollout_generation_token_count: Every generation token the rollout
            was billed for, not just the terminating step
            (``generation_token_count`` above is the terminal step only,
            which in a multi-step rollout is a few percent of the total).
            Covers the actions taken PLUS harness-initiated calls the
            rollout caused -- notably context-compaction summaries, which
            never reach an env as an ``Action``. Excludes recovered
            non-STOP samples, which took no action. ``None`` when no call
            reported a count, or for envs the agent never reports one to;
            a RewardFn must treat ``None`` as "unknown", never as zero
            tokens.
    """

    is_correct: bool
    generation_token_count: int | None
    finish_reason: str | None
    max_output_tokens: int | None
    depth: int | None = None
    status: RolloutStatus = RolloutStatus.OK
    step_count: int | None = None
    rollout_generation_token_count: int | None = None
