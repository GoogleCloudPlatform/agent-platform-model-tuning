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

"""Tests for ``core.rollout_status`` — the ONE fault vocabulary.

Covers the enum surface and the backward-compat guarantee that
``RolloutStatus.OK == "OK"`` so legacy string-comparing consumers keep
working, plus the invariants that make the vocabulary safe to be the only
carrier of fault:

* a member in two failure sets is a member whose reward treatment depends
  on which predicate the caller happened to ask;
* a member in NO set is silently graded at the grader's headline, which is
  how a broken serving stack trains as bad policy — so the unclassified
  set is pinned to an explicit list rather than left to drift.

What is NOT here any more: the finish-reason table
(``FINISH_REASON_STATUS`` / ``status_for_finish_reason`` / the
terminal-vs-resampleable split) and ``classify_sampling_exception``. This
module is now the vocabulary and its predicates ONLY; the mapping from a
vendor's finish reason or a transport exception onto a member lives in one
place each — the backends' shared ``fold_finish_reason``, the agent-side
sampler's ``classify`` and its status hook — and is tested beside them.
"""

from __future__ import annotations

from core.rollout_status import RolloutStatus


class TestRolloutStatusMembers:
    def test_all_expected_members_present(self):
        # Pinned deliberately: this vocabulary is published (dashboards key
        # on the values, `verifier/reward.json` carries them), so adding a
        # member is a decision, not a refactor. If this fails, add the
        # member to a failure set too — an unclassified member is graded at
        # the grader's headline.
        expected_values = {
            "OK",
            "max_step_limit",
            "rollout_timeout",
            "sampling_failure/SAMPLING_CLIENT_ERROR",
            "sampling_failure/TIMEOUT",
            "sampling_failure/PROMPT_TOO_LONG",
            "sampling_failure/MAX_TOKENS",
            "sampling_failure/MALFORMED_FUNCTION_CALL",
            "sampling_failure/COMPACTION_FAILED",
            "sampling_failure/SAFETY",
            # 200 OK, finishReason STOP, and a part holding the empty string
            # with zero output tokens billed. A SERVING anomaly, not a
            # transport fault and not a model fault: the rollout ends, the
            # abnormality is counted under its own name, and the work already
            # done stays graded.
            "sampling_failure/EMPTY_PART",
            "sampling_failure/UNHANDLED",
            "env_failure",
            # One env.step stalled. The environment twin of
            # sampling_failure/TIMEOUT, and infra for the same reason.
            "env_step_timeout",
            "grading_failure",
        }
        assert {s.value for s in RolloutStatus} == expected_values

    def test_the_three_timeout_tiers_are_distinguishable(self):
        """Each clock that can run out names its own tier.

        Reading the status must say WHICH budget was exhausted, because the
        remedy differs every time: raise the rollout budget, chase backend
        latency, or chase the environment. They also differ in fault class,
        so a merged name would grade an infra hang as policy behaviour --
        which is what `deadline_exceeded` did when a 349s/turn backend met
        a 1800s budget and 77% of rollouts were scored as slow policy.
        """
        assert RolloutStatus.ROLLOUT_TIMEOUT.value == "rollout_timeout"
        assert RolloutStatus.SAMPLING_FAILURE_TIMEOUT.value == (
            "sampling_failure/TIMEOUT"
        )
        assert RolloutStatus.ENV_STEP_TIMEOUT.value == "env_step_timeout"
        # The rollout budget is the policy's own; the other two are ours.
        assert RolloutStatus.ROLLOUT_TIMEOUT.is_evidence() is True
        assert RolloutStatus.SAMPLING_FAILURE_TIMEOUT.is_evidence() is False
        assert RolloutStatus.ENV_STEP_TIMEOUT.is_evidence() is False

    def test_str_equality_backward_compat(self):
        # Enum members == their string values. This is load-bearing for
        # consumers that were written against string literals before the
        # enum landed.
        assert RolloutStatus.OK == "OK"
        assert RolloutStatus.SAMPLING_FAILURE_TIMEOUT == "sampling_failure/TIMEOUT"
        assert "OK" == RolloutStatus.OK  # symmetric


# The members deliberately in no failure set, and why. Anything else
# landing here is a bug: a fault we understand being graded at the
# headline.
_NOT_A_FAULT = {
    RolloutStatus.OK,
    RolloutStatus.MAX_STEP_LIMIT,
    RolloutStatus.ROLLOUT_TIMEOUT,
}
_CLASSIFIER_GAP = {RolloutStatus.SAMPLING_FAILURE_UNHANDLED}
# Graded on purpose, and NOT a classifier gap -- we know exactly what it is.
# The server answered 200 with `finishReason: STOP` and an empty text part, so
# the rollout ends; but the work done before that turn is real and
# grade-on-exit has already scored it. Same treatment as ROLLOUT_TIMEOUT, for
# the same reason. Being in no set is what keeps that score.
_GRADED_SERVING_ANOMALY = {RolloutStatus.SAMPLING_FAILURE_EMPTY_PART}


class TestSetInvariants:
    def test_every_member_is_in_at_most_one_failure_set(self):
        for status in RolloutStatus:
            memberships = [
                status.is_transient_failure(),
                status.is_input_failure(),
                status.is_model_failure(),
            ]
            assert sum(memberships) <= 1, (
                f"{status.value} is in {sum(memberships)} failure sets; a "
                "member's reward treatment must not depend on which "
                "predicate is asked"
            )

    def test_only_the_named_members_are_unclassified(self):
        unclassified = {
            s
            for s in RolloutStatus
            if not (
                s.is_transient_failure()
                or s.is_input_failure()
                or s.is_model_failure()
            )
        }
        assert unclassified == _NOT_A_FAULT | _CLASSIFIER_GAP | _GRADED_SERVING_ANOMALY, (
            "a new member that is in no set will be GRADED at the grader's "
            "headline. Put it in a set, or add it here with a reason."
        )

    def test_empty_part_is_evidence_and_therefore_still_scored(self):
        """The whole point of the member: the rollout ends, the score stays.

        If this ever flips, an empty-part rollout goes back to being NaN'd and
        silently dropped from the denominator -- which is the bug it exists to
        fix, and which is invisible in every metric except a log line.
        """
        empty = RolloutStatus.SAMPLING_FAILURE_EMPTY_PART
        assert empty.is_evidence()
        assert not empty.is_transient_failure()
        assert not empty.is_model_failure()
        assert not empty.is_input_failure()
        # It IS a sampling failure by name, so it is counted under
        # `trajectory_status/sampling_failure_EMPTY_PART` like its siblings.
        assert empty.is_sampling_failure()
        assert empty.value == "sampling_failure/EMPTY_PART"

    def test_infra_is_the_only_non_evidence(self):
        for status in RolloutStatus:
            assert status.is_evidence() is not status.is_transient_failure()

    def test_deadline_exceeded_is_an_outcome_not_a_sampling_failure(self):
        # The distinction that earns it a member: a rollout that spent its
        # wall clock left real, gradeable work behind, whereas a stalled
        # sampler call is our stack and must be NaN.
        deadline = RolloutStatus.ROLLOUT_TIMEOUT
        assert not deadline.is_sampling_failure()
        assert deadline.is_evidence()
        assert RolloutStatus.SAMPLING_FAILURE_TIMEOUT.is_transient_failure()
        assert not RolloutStatus.SAMPLING_FAILURE_TIMEOUT.is_evidence()

    def test_env_failure_is_infra_and_not_a_sampling_failure(self):
        # A dead sandbox says nothing about the policy. Before this member
        # existed the exception reached the agent's generic handler as
        # UNHANDLED -- in no fault bucket -- and was GRADED at the headline,
        # i.e. a broken executor trained as bad policy.
        assert RolloutStatus.ENV_FAILURE.is_transient_failure()
        assert not RolloutStatus.ENV_FAILURE.is_evidence()
        assert not RolloutStatus.ENV_FAILURE.is_sampling_failure()

    def test_grading_failure_is_infra_and_not_a_sampling_failure(self):
        assert RolloutStatus.GRADING_FAILURE.is_transient_failure()
        assert not RolloutStatus.GRADING_FAILURE.is_evidence()
        assert not RolloutStatus.GRADING_FAILURE.is_sampling_failure()


class TestIsSamplingFailure:
    def test_negative(self):
        assert not RolloutStatus.OK.is_sampling_failure()
        assert not RolloutStatus.MAX_STEP_LIMIT.is_sampling_failure()
        assert not RolloutStatus.ROLLOUT_TIMEOUT.is_sampling_failure()
        assert not RolloutStatus.GRADING_FAILURE.is_sampling_failure()

    def test_positive_covers_all_sampling_failure_members(self):
        for s in RolloutStatus:
            if s.value.startswith("sampling_failure/"):
                assert s.is_sampling_failure(), s


class TestSamplingFailureReason:
    def test_non_failures_return_none(self):
        assert RolloutStatus.OK.sampling_failure_reason() is None
        assert RolloutStatus.MAX_STEP_LIMIT.sampling_failure_reason() is None
        assert RolloutStatus.GRADING_FAILURE.sampling_failure_reason() is None

    def test_failures_return_reason_suffix(self):
        assert (
            RolloutStatus.SAMPLING_FAILURE_TIMEOUT.sampling_failure_reason()
            == "TIMEOUT"
        )
        assert (
            RolloutStatus.SAMPLING_FAILURE_MAX_TOKENS.sampling_failure_reason()
            == "MAX_TOKENS"
        )
        assert (
            RolloutStatus.SAMPLING_FAILURE_MALFORMED_FUNCTION_CALL.sampling_failure_reason()
            == "MALFORMED_FUNCTION_CALL"
        )


class TestIsTransientFailure:
    def test_outcomes_are_not_transient(self):
        assert not RolloutStatus.OK.is_transient_failure()
        assert not RolloutStatus.MAX_STEP_LIMIT.is_transient_failure()
        assert not RolloutStatus.ROLLOUT_TIMEOUT.is_transient_failure()

    def test_infra_failures_are_transient(self):
        assert (
            RolloutStatus.SAMPLING_FAILURE_SAMPLING_CLIENT_ERROR.is_transient_failure()
        )
        assert RolloutStatus.SAMPLING_FAILURE_TIMEOUT.is_transient_failure()

    def test_deterministic_and_model_output_failures_are_not_transient(self):
        # Load-bearing invariant: deterministic-input and model-output
        # failures must NOT be classified as transient, or a bad prompt
        # would silently retry forever instead of dropping via the
        # ``<2 valid`` check with its own reason. If a new failure mode
        # gets added and this test starts failing, DELIBERATELY decide
        # whether it should be transient before flipping the assertion.
        assert not RolloutStatus.SAMPLING_FAILURE_PROMPT_TOO_LONG.is_transient_failure()
        assert not RolloutStatus.SAMPLING_FAILURE_MAX_TOKENS.is_transient_failure()
        assert not RolloutStatus.SAMPLING_FAILURE_MALFORMED_FUNCTION_CALL.is_transient_failure()
        assert not RolloutStatus.SAMPLING_FAILURE_SAFETY.is_transient_failure()

    def test_unhandled_is_not_transient(self):
        # UNHANDLED is a classifier-gap sentinel indicating the triage
        # function needs a new branch, NOT a retriable infra failure.
        assert not RolloutStatus.SAMPLING_FAILURE_UNHANDLED.is_transient_failure()


class TestCompactionFailedStatus:
    def test_compaction_failed_member_exists(self):
        assert (
            RolloutStatus.SAMPLING_FAILURE_COMPACTION_FAILED.value
            == "sampling_failure/COMPACTION_FAILED"
        )

    def test_compaction_failed_is_sampling_failure(self):
        assert RolloutStatus.SAMPLING_FAILURE_COMPACTION_FAILED.is_sampling_failure()
        assert (
            RolloutStatus.SAMPLING_FAILURE_COMPACTION_FAILED.sampling_failure_reason()
            == "COMPACTION_FAILED"
        )

    def test_compaction_failed_is_an_input_failure(self):
        """Deterministic: a context that will not fit does not fit on retry."""
        assert (
            not RolloutStatus.SAMPLING_FAILURE_COMPACTION_FAILED.is_transient_failure()
        )
        assert RolloutStatus.SAMPLING_FAILURE_COMPACTION_FAILED.is_input_failure()


class TestTheSafetyFamily:
    def test_the_safety_family_is_a_model_fault(self):
        # Without this, a content-filtered rollout lands on UNHANDLED and is
        # graded at the headline as though the model had answered. The six
        # native reasons (SAFETY / RECITATION / BLOCKLIST /
        # PROHIBITED_CONTENT / SPII / IMAGE_SAFETY) collapse onto this one
        # member at the ADAPTER boundary now; what this module still owns is
        # what the member MEANS to the reward path.
        assert RolloutStatus.SAMPLING_FAILURE_SAFETY.is_model_failure()
        assert not RolloutStatus.SAMPLING_FAILURE_SAFETY.is_transient_failure()
        assert RolloutStatus.SAMPLING_FAILURE_SAFETY.is_evidence()
