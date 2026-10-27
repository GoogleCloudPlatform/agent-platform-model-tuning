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

"""Per-backend retry profiles, named by the SHAPE of the call.

A backend's retry policy is not a tuning choice, it is decided by how the
backend talks:

* **synchronous** request/response (``generateContent``) — a failed call
  left nothing behind, so the only correct response to an outage is to
  resend (:class:`SyncRetryProfile`).
* **long-running operation** (create an op, then poll it) — once the create
  lands the work is queued SERVER-SIDE and survives the outage. Resending
  the create duplicates it and spends a second concurrency permit; polling
  can never do harm. Hence two very different budgets on one profile
  (:class:`LroRetryProfile`).

There is one profile class per shape rather than one class with optional
groups: with everything ``Optional`` a ``None`` cannot distinguish "wrong
shape for this backend" from "not configured", and something has to police
the difference. A backend config carries exactly one ``retry`` field, of
its own shape.

A profile is set at CONSTRUCTION, from the config — per backend, per loop.
Nothing threads a remaining budget through ``sample()``: each loop states
its own timeouts for the backend it actually talks to. A direct API
answering in ~1.7 s and an LRO backend taking ~349 s per turn must not be
forced to share one derived number.

Pure value objects: no I/O, no imports beyond the stdlib, so any layer can
depend on this without dragging a transport in.

**Every default here reproduces the module constant it replaced.** This is
a structural change: the numbers move from private constants inside two
sampler modules to a value a config can state, override and test. Arguing
about what the numbers SHOULD be is a separate change.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

__all__ = [
    "Backoff",
    "SyncRetryProfile",
    "LroRetryProfile",
    "DEFAULT_SYNC_RETRY",
    "DEFAULT_LRO_RETRY",
]


@dataclass(frozen=True)
class Backoff:
    """Jittered, annealed wait schedule.

    ``initial_s`` doubles (``factor``) up to ``max_s``, each wait scaled by
    ``1 +/- jitter`` so a fleet of clients retrying the same outage does not
    synchronize into a thundering herd.
    """

    initial_s: float
    max_s: float
    factor: float = 2.0
    jitter: float = 0.5

    def __post_init__(self) -> None:
        if self.initial_s <= 0:
            raise ValueError(f"initial_s must be > 0, got {self.initial_s}")
        if self.max_s < self.initial_s:
            raise ValueError(
                f"max_s ({self.max_s}) must be >= initial_s ({self.initial_s})"
            )
        if self.factor < 1.0:
            raise ValueError(f"factor must be >= 1.0, got {self.factor}")
        if not 0.0 <= self.jitter < 1.0:
            raise ValueError(f"jitter must be in [0, 1), got {self.jitter}")

    def interval_for(self, attempt: int) -> float:
        """Un-jittered interval before attempt ``attempt`` (1-based)."""
        return min(
            self.initial_s * (self.factor ** max(0, attempt - 1)),
            self.max_s,
        )

    def delay_for(self, attempt: int) -> float:
        """Jittered seconds to wait before attempt ``attempt`` (1-based)."""
        raw = self.interval_for(attempt)
        lo, hi = 1.0 - self.jitter, 1.0 + self.jitter
        return raw * random.uniform(lo, hi)


@dataclass(frozen=True)
class SyncRetryProfile:
    """Retry policy for a synchronous request/response backend.

    **The defaults reproduce the previous module constants exactly**: 10
    attempts on a ``2 → 30 s`` schedule with +/-20% jitter, and no
    wall-clock stop. This type is the interface; whether those numbers are
    the right policy is a separate question and a separate change.

    ``deadline_s`` is the new knob the constants could not express: a call's
    total budget in SECONDS. It defaults to ``None`` — unbounded, i.e. the
    count is the only stop, as before — so setting it is opt-in. When both
    are set, whichever trips first ends the call.
    """

    deadline_s: float | None = None
    max_attempts: int = 10
    backoff: Backoff = field(default_factory=lambda: Backoff(2.0, 30.0, jitter=0.2))

    def __post_init__(self) -> None:
        if self.deadline_s is not None and self.deadline_s <= 0:
            raise ValueError(
                f"deadline_s must be > 0 or None (unbounded), got {self.deadline_s}"
            )
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {self.max_attempts}")


@dataclass(frozen=True)
class LroRetryProfile:
    """Retry policy for a create-then-poll (long-running operation) backend.

    Two budgets, because the two halves of an LRO are not the same risk:
    once the operation exists the candidate is queued server-side and will
    complete whenever the service comes back, so waiting is free, while a
    resent create may duplicate work that already landed and spend a second
    concurrency permit.

    **The defaults reproduce the previous module constants exactly** —
    ``_OP_START_RETRY_DEADLINE`` (2 h), ``_OP_POLL_TIMEOUT`` (10 h) and the
    ``10 → 90 s`` poll schedule. This type is the interface; tightening the
    create budget or slowing the poll are policy changes and belong in their
    own change, where they can be argued and reverted on their own.
    """

    # Retry budget for the create call.
    create_deadline_s: float = 2 * 60 * 60.0
    # Total budget for polling one LRO. The single stop every rollout tier
    # above is matched to, so a recipe deadline shorter than it silently
    # truncates the backend's own retry.
    poll_deadline_s: float = 10 * 60 * 60.0
    # Poll schedule. Ops run seconds-to-minutes and thousands can be in
    # flight, so it is deliberately slower than the RPC-retry schedule:
    # steady-state GetOperation QPS is concurrency/interval and draws on the
    # quota-constrained `resource_management_requests` pool.
    poll_backoff: Backoff = field(default_factory=lambda: Backoff(10.0, 90.0))

    def __post_init__(self) -> None:
        if self.create_deadline_s < 0:
            raise ValueError(
                "create_deadline_s must be >= 0 (0 = single attempt), got "
                f"{self.create_deadline_s}"
            )
        if self.poll_deadline_s <= 0:
            raise ValueError(f"poll_deadline_s must be > 0, got {self.poll_deadline_s}")


# Module-level defaults so a sampler constructed without a config (tests,
# scripts) gets the same policy the configs hand out.
DEFAULT_SYNC_RETRY = SyncRetryProfile()
DEFAULT_LRO_RETRY = LroRetryProfile()
