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

"""Rollout / trajectory status — the ONE fault vocabulary.

Centralized vocabulary for the trajectory-level ``rollout_status`` extra /
``StepStats.status``. Inherits ``str`` so members compare equal to
their underlying string values (drop-in for legacy string-literal
comparisons).

``RolloutStatus`` is the only place a fault is named. Nothing downstream
may carry a parallel ``fault`` field beside a status: two names for one
question drift, and when they drift the metric says one thing while the
gradient does another. Everything the reward path needs is derivable from
the member via the three predicates below.

Categories:

* ``OK``: normal completion; the env produced a reward.
* ``MAX_STEP_LIMIT``: the agent ran out of STEPS without the env signaling
  ``done``. Multi-step agents only.
Three different clocks can run out, and each names its own tier. Reading
the status must tell you WHICH budget was exhausted, because the remedy
differs every time — raise the rollout budget, chase backend latency, or
chase the environment:

* ``ROLLOUT_TIMEOUT``: the whole-rollout wall clock
  (``agent_rollout_timeout_seconds``). The wall-clock twin of
  ``MAX_STEP_LIMIT`` and classified the same way — in none of the failure
  sets, because the budget was spent by the policy's own choices and
  whatever it left behind in the env is real, gradeable work. Formerly
  ``ROLLOUT_TIMEOUT``, renamed because "deadline" named no tier and was
  read as "a call timed out".
* ``SAMPLING_FAILURE_TIMEOUT``: ONE sampler call stalled past the turn
  budget. Our serving stack misbehaving rather than the policy being
  slow, so NaN.
* ``ENV_STEP_TIMEOUT``: one ``env.step`` stalled past its budget. The
  environment misbehaving — a wedged sandbox, an executor that accepted
  the request and never answered — so NaN, like the sampling twin.

  Caveat worth stating: ``ROLLOUT_TIMEOUT`` is only honestly a policy
  outcome when the rollout budget is sized so it does not bind routinely.
  Measured at 349 s/turn against a 1800 s budget, 77% of rollouts tripped
  it and the metric was reporting backend latency as policy behaviour.
  A high ``rollout_timeout`` rate means the budget is wrong, not the
  policy.
* ``SAMPLING_FAILURE_*``: the sampler produced no usable candidate.
* ``GRADING_FAILURE``: the verifier itself broke. Infra, so NaN.
* ``SAMPLING_FAILURE_UNHANDLED``: fallthrough sentinel; an unclassified
  response means the triage table needs a new branch. It is in NO set on
  purpose, so a classifier gap shows up in metrics instead of being
  absorbed silently — consumers log an ERROR when they see it.

Three disjoint failure sets, and what each means for reward:

===================== ============================== ==================
predicate             meaning                        reward treatment
===================== ============================== ==================
is_transient_failure  our stack broke                NaN (not evidence)
is_input_failure      the request was unusable       floor (evidence)
is_model_failure      the generation was unusable    floor (evidence)
===================== ============================== ==================

What this vocabulary deliberately does NOT encode is **retriability**.
"Infra, but do not retry" (wire skew: a deploy bug we must see, not spin
on) is a real state, and it is a different question from "what should the
trainer learn from this". Retry decisions are a local, behavioural concern
of whoever is doing the retrying; they belong on that caller's own failure
object, not on the published status.

Nor does it encode **finish reasons or exception triage** any more. This
module used to own a finish-reason table, a resampleable/terminal split
and ``classify_sampling_exception``; the agent core now folds the native
finish reason on the sampled candidate straight onto a member here, in
one place (``agentcore.status_hook.decide``), and exceptions raised by
the sampling call likewise (``agentcore.status_hook.decide_exception``).
Keeping a second, string-keyed table beside it is how the two came to
disagree.
"""

from __future__ import annotations

import enum

__all__ = [
    "RolloutStatus",
]


class RolloutStatus(str, enum.Enum):
    """Terminal status for a single trajectory / rollout. Inherits
    ``str`` so ``RolloutStatus.OK == "OK"`` is True.

    NOTE it is ``(str, Enum)`` and not ``StrEnum``: ``f"{status}"`` renders
    the *member* repr, not the value. Always write ``.value`` when
    formatting.
    """

    OK = "OK"
    MAX_STEP_LIMIT = "max_step_limit"
    # Honest exhaustion of the ROLLOUT wall clock. Sibling of
    # MAX_STEP_LIMIT, not of sampling_failure/TIMEOUT — see the module
    # docstring.
    ROLLOUT_TIMEOUT = "rollout_timeout"
    # One env.step stalled past its own budget. The twin of
    # sampling_failure/TIMEOUT on the environment side, and infra for the
    # same reason: a wedged sandbox is not the policy being slow.
    ENV_STEP_TIMEOUT = "env_step_timeout"

    # Infra-flakiness failures (transient; eligible for retry-fraction gate).
    SAMPLING_FAILURE_SAMPLING_CLIENT_ERROR = "sampling_failure/SAMPLING_CLIENT_ERROR"
    SAMPLING_FAILURE_TIMEOUT = "sampling_failure/TIMEOUT"

    # Deterministic-input failures (NOT transient).
    SAMPLING_FAILURE_PROMPT_TOO_LONG = "sampling_failure/PROMPT_TOO_LONG"
    # Context compaction ran but its summarization call did not produce a
    # usable summary, so the conversation cannot be shortened — i.e. the
    # request we are about to send does not fit. Same family as
    # PROMPT_TOO_LONG: deterministic in the input, not our stack falling
    # over.
    SAMPLING_FAILURE_COMPACTION_FAILED = "sampling_failure/COMPACTION_FAILED"

    # Model-output failures (NOT transient). The call succeeded and the
    # prompt was fine; what came back is unusable.
    SAMPLING_FAILURE_MAX_TOKENS = "sampling_failure/MAX_TOKENS"
    SAMPLING_FAILURE_MALFORMED_FUNCTION_CALL = (
        "sampling_failure/MALFORMED_FUNCTION_CALL"
    )
    # The safety family: SAFETY / RECITATION / BLOCKLIST /
    # PROHIBITED_CONTENT / SPII / IMAGE_SAFETY. The call succeeded and the
    # prompt was fine; the model declined or was filtered. Without a member
    # of its own this lands on UNHANDLED, which is in no set, so a
    # content-filtered rollout gets GRADED at the grader's headline instead
    # of floored.
    SAMPLING_FAILURE_SAFETY = "sampling_failure/SAFETY"

    # The server answered 200, said it stopped cleanly, and returned NOTHING.
    #
    # The observed shape, unanimous across 19 of 19 occurrences in one
    # measured agentic sweep (2026-09-26)::
    #
    #     "candidates": [{"finishReason": "STOP",
    #                     "content": {"role": "model",
    #                                 "parts": [{"text": ""}]}}]
    #     "usageMetadata": {"promptTokenCount": 9859, "totalTokenCount": 9859}
    #
    # A text part holding the EMPTY STRING, ``finishReason: STOP``, and no
    # ``candidatesTokenCount`` at all -- the server itself reports zero output
    # tokens. ``chat_completion._partition_parts`` cannot place such a part
    # (``bool("")`` is False, no function call, no thought signature) and
    # raises "no recognizable payload". It appears deep in long multi-turn
    # rollouts (never before the 8th call in that sweep), not at the start.
    #
    # IN NO FAULT SET, deliberately, and this is the whole point of the
    # member: the rollout TERMINATES, but the work the agent already did is
    # real and ``grade_on_exit`` has already scored it, so that score is KEPT
    # (``is_evidence()`` is True -> rule 0 in ``grading`` does not NaN it).
    # Same treatment, and the same reasoning, as ``ROLLOUT_TIMEOUT``.
    #
    # Before this member existed the condition fell through the classifier's
    # catch-all to SAMPLING_CLIENT_ERROR, which IS transient -- so the
    # trajectory was NaN'd and silently dropped from the denominator, AND the
    # event was filed as a transport fault it is not (the backend's retry
    # budget never fires: the HTTP call succeeded). That cost 2.99% of 4080
    # rollouts on one sweep, invisible in every metric except a log line.
    #
    # It earns a member rather than reusing UNHANDLED so the abnormality is
    # COUNTED under its own name -- `sampler/trajectory_status/
    # sampling_failure_EMPTY_PART` -- instead of being averaged into a
    # classifier-gap bucket that means "we do not know what this is". We do
    # know what this is.
    SAMPLING_FAILURE_EMPTY_PART = "sampling_failure/EMPTY_PART"

    # The ENVIRONMENT broke mid-rollout: ``Env.step`` raised. Infra, so NaN.
    #
    # A failing TOOL is NOT this. The dispatcher turns a tool error into an
    # observation the model can act on, which is an ordinary turn. This is
    # the env itself falling over — the sandbox died, the executor service
    # 500s — and it says nothing about the policy.
    #
    # It earns a member because the alternative was measured and is bad:
    # such an exception reached the agent's generic handler, got
    # ``SAMPLING_FAILURE_UNHANDLED`` — which is in no fault bucket — and was
    # therefore GRADED at the grader's headline. A dead executor training as
    # bad policy is the exact failure this vocabulary exists to prevent.
    ENV_FAILURE = "env_failure"

    # The verifier broke while scoring. Infra — a grader crash says nothing
    # about the policy, so it must be NaN and not a zero. NOT a
    # ``sampling_failure/*``: nothing was wrong with the sampling.
    #
    # There is deliberately no "the agent's output was ungradable" status.
    # That is not an error at all; it is a SCORE of ``floor``, returned as
    # an ordinary reward. Only a broken verifier raises.
    GRADING_FAILURE = "grading_failure"

    # Fallthrough / classifier-gap sentinel. In NO set, so a rollout that
    # lands here is graded at the grader's headline and whoever routed it
    # here logs an ERROR. If you find yourself reaching for this for a case
    # you understand, add the member instead.
    SAMPLING_FAILURE_UNHANDLED = "sampling_failure/UNHANDLED"

    def is_sampling_failure(self) -> bool:
        """True iff this status is a ``sampling_failure/<REASON>`` variant."""
        return self.value.startswith("sampling_failure/")

    def sampling_failure_reason(self) -> str | None:
        """The ``REASON`` part of ``sampling_failure/REASON``, or None."""
        if not self.is_sampling_failure():
            return None
        return self.value.split("/", 1)[1]

    def is_transient_failure(self) -> bool:
        """True iff OUR STACK broke: infra, therefore NaN, not evidence.

        Used by ``TrajectoryGroupFilter``'s retry-fraction gate.
        Intentionally hard-coded (not configurable) so recipes can't
        mark deterministic failures as retriable and mask real bugs;
        add new infra failure modes to the enum AND this set.
        """
        return self in _TRANSIENT_FAILURES

    def is_model_failure(self) -> bool:
        """True iff the model itself produced an unusable generation.

        The call succeeded and the prompt was fine, so the trajectory is
        real evidence about the policy (finite reward), never NaN. Add
        new model-output failure modes to the enum AND this set.
        """
        return self in _MODEL_FAILURES

    def is_input_failure(self) -> bool:
        """True iff the request itself was unusable (e.g. prompt too long).

        Deterministic in the input, so resampling is pointless; like a
        model failure it is evidence, not infra noise. Add new input
        failure modes to the enum AND this set.
        """
        return self in _INPUT_FAILURES

    def native_finish_reason(self) -> str | None:
        """The backend finish reason this status came from, if it is one.

        Some sampling failures ARE a finish reason the generation ended on
        (``MALFORMED_FUNCTION_CALL``, ``MAX_TOKENS``, ``SAFETY``); others
        describe the call rather than the generation (``TIMEOUT``,
        ``SAMPLING_CLIENT_ERROR``, ``PROMPT_TOO_LONG``) and have no finish
        reason at all. Only the first kind returns a value.

        This exists because such a generation may leave NO STEP behind. A
        ``MALFORMED_FUNCTION_CALL`` candidate arrives with empty ``parts``
        -- measured on 60k real candidates, all 80 malformed ones were
        part-less and no STOP ever was -- so the response converter raises
        and the rollout ends without a turn. ``sampler/finish_reason/*``
        counts agent STEPS, so those calls were invisible to it: the
        metric silently under-reported both the numerator and its own
        denominator, which its spec defines as "one LLM generation call".
        """
        return _STATUS_TO_FINISH_REASON.get(self)

    def is_evidence(self) -> bool:
        """Whether this trajectory says anything about the policy.

        False only for infra: the trainer must see NaN, so the group
        filter drops it rather than learning from our own outage.
        ``UNHANDLED`` answers True, which is the deliberately loud
        direction to be wrong in — it is also logged as an ERROR.
        """
        return not self.is_transient_failure()


# Module-level frozensets backing the predicates above. Every member
# belongs to AT MOST ONE of them: ``OK`` / ``MAX_STEP_LIMIT`` /
# ``ROLLOUT_TIMEOUT`` are not failures at all (running out of steps or of
# wall clock is an outcome of the policy, not a fault), and ``UNHANDLED``
# is deliberately unclassified so a classifier gap surfaces in metrics
# instead of being absorbed into a bucket. Enforced by
# ``tests/unit/test_rollout_status.py``.
_TRANSIENT_FAILURES: frozenset[RolloutStatus] = frozenset(
    {
        RolloutStatus.SAMPLING_FAILURE_SAMPLING_CLIENT_ERROR,
        RolloutStatus.SAMPLING_FAILURE_TIMEOUT,
        RolloutStatus.ENV_FAILURE,
        # A wedged env.step is the same kind of fault as an env that
        # raises -- the environment failed to answer -- so it sits beside
        # ENV_FAILURE, not beside ROLLOUT_TIMEOUT. NaN, never graded.
        RolloutStatus.ENV_STEP_TIMEOUT,
        RolloutStatus.GRADING_FAILURE,
    }
)

_MODEL_FAILURES: frozenset[RolloutStatus] = frozenset(
    {
        RolloutStatus.SAMPLING_FAILURE_MAX_TOKENS,
        RolloutStatus.SAMPLING_FAILURE_MALFORMED_FUNCTION_CALL,
        RolloutStatus.SAMPLING_FAILURE_SAFETY,
    }
)

_INPUT_FAILURES: frozenset[RolloutStatus] = frozenset(
    {
        RolloutStatus.SAMPLING_FAILURE_PROMPT_TOO_LONG,
        RolloutStatus.SAMPLING_FAILURE_COMPACTION_FAILED,
    }
)

# Statuses that ARE a backend finish reason, mapped to its native spelling.
# Exactly the model failures: each of these names the reason the generation
# ended on, which is why the classifier produced this status in the first
# place. The infra/input failures are absent on purpose -- TIMEOUT and
# SAMPLING_CLIENT_ERROR describe the CALL, not the generation, and there is
# no finish reason to report for them.
_STATUS_TO_FINISH_REASON: dict[RolloutStatus, str] = {
    RolloutStatus.SAMPLING_FAILURE_MAX_TOKENS: "MAX_TOKENS",
    RolloutStatus.SAMPLING_FAILURE_MALFORMED_FUNCTION_CALL: "MALFORMED_FUNCTION_CALL",
    RolloutStatus.SAMPLING_FAILURE_SAFETY: "SAFETY",
}
