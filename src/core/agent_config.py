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

"""AgentConfig: abstract base for a recipe's agent builder.

A recipe subclass carries the per-call generation knobs (thinking_level,
agent_rollout_timeout_seconds, max_generation_tokens) and implements make(),
which constructs the recipe's Agent (wiring its Env, reward fn, etc.).
Behavior module: make() builds a live component, parallel to
RolloutConfig/BackendConfig/RewardConfig.

``max_generation_tokens`` is the per-call cap on generated tokens, threaded
through ``group_rollout(max_generation_tokens=...)`` →
``sample(max_generation_tokens=...)``. ``None`` (default) falls back to the
sampler's module-level default (32768 on both backends). Recipes override
this to a tighter budget when appropriate (e.g. some recipes use 8192).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig(abc.ABC):
    thinking_level: str = "HIGH"
    # Per-rollout wallclock ceiling (seconds). Defaults to 2h so a sampler
    # operation that wedges server-side (a long-running generate op that
    # stays pending indefinitely) can't block a rollout -- and, on the sync
    # pipeline, the whole training step -- for the sampler's much larger
    # internal poll budget. When the ceiling trips, the agent converts the
    # rollout into a TIMEOUT trajectory (the group is then dropped by the
    # variance filter) instead of hanging. Set to None to disable the
    # ceiling entirely (unbounded wait on the sampler's own poll timeout);
    # only do that if a legitimate rollout can exceed an hour.
    agent_rollout_timeout_seconds: float | None = 7200.0
    # Per-call cap on generated tokens. None = sampler default (32768).
    max_generation_tokens: int | None = None
    # Non-STOP finish reasons the multi-step loop recovers from by
    # resampling the UNCHANGED observation instead of aborting the
    # rollout (e.g. {"MAX_TOKENS", "MALFORMED_FUNCTION_CALL"} -- both are
    # stochastic sampler outcomes that plausibly succeed on a fresh
    # draw). The retry relies on that stochasticity; nothing is appended
    # to the conversation, so the model never sees a "you failed" turn.
    # Empty (the default) = no recovery, abort on the first non-STOP
    # sample.
    #
    # A retry is plumbing to make ONE step succeed, not a step of its
    # own: it does NOT consume the step budget, does not move the
    # ``steps_remaining`` the env echoes, and does not inflate the solve
    # step count a reward fn sees. The two budgets below bound it
    # instead.
    recoverable_finish_reasons: frozenset[str] = frozenset()
    # Consecutive retries allowed for a SINGLE turn before the rollout
    # gives up via the normal sampling-failure path. Reset on every
    # successful sample.
    max_sample_retries: int = 3
    # Rollout-wide ceiling across all turns, so a run that fails on most
    # turns cannot cost ``max_steps * max_sample_retries`` sampler calls.
    # None = bounded by the per-turn budget alone.
    max_sample_retries_per_rollout: int | None = 8
    # Backend prompt cap in tokens. None (default) disables in-loop
    # context compaction entirely. Not discoverable at runtime and it
    # differs per backend, so a config that targets a capped backend
    # must set it explicitly; a stale value means no protection.
    max_context_tokens: int | None = None
    # Fraction of max_context_tokens at which compaction fires, measured
    # against the projected next request (last prompt + last generation).
    # 0.7 leaves room for one maximal generation plus the summarization
    # call; a tighter ratio can be blown through by a single long turn.
    compaction_trigger_ratio: float = 0.7
    # Per-rollout ceiling on compaction events. The summarization call
    # consumes a step, but this bounds pathological repeat-compaction.
    max_compactions: int = 0
    # Most recent model/tool exchange pairs kept verbatim after a compaction
    # (everything older is replaced by the summary). Still bounded by the
    # tail token budget (20% of max_context_tokens); the newest pair is kept
    # even past that budget unless this is 0.
    compaction_tail_pairs: int = 2

    @abc.abstractmethod
    def make(self, *, sampler, job_id, **kwargs): ...
