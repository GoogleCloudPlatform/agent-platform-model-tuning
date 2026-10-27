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

"""Retry state-machine tests for ``AsyncDefaultAgent``.

Motivation: When ``submit_retries > 1``, a wrong ``submit(answer=...)``
call should NOT terminate the rollout. Instead the agent:

  1. Checks the submitted answer against ``expected_answer``.
  2. If wrong AND retries remain: draws the next 2 words from a
     task-local shuffled ``extra_word_pool`` (fixed rng seeded at
     ``run()`` start), applies ``rule`` to them (respecting
     ``reverse_task`` direction), appends a function-response-shaped
     observation carrying the new (input, output) pairs, decrements
     the retry budget, and lets the run-loop continue.
  3. If correct OR retries exhausted: raises ``Submitted`` as before.

This gives the model up to N total submit attempts per task with 2
fresh worked examples injected after each wrong attempt.

Tests use a tiny in-memory rule (``[("shift", 2)]``) and a fake
event-loop-free stand-in for env / model so the state machine can be
exercised in isolation. Docker + real sampler are covered by higher-
level runs.
"""

from __future__ import annotations

import asyncio
import json

import pytest

# The agent_solver package imports ``minisweagent`` at module scope.
pytest.importorskip("minisweagent")

# Import after the gate.
from example_recipe.word_puzzle.agent_solver.minisweagent.async_default_agent import (  # noqa: E402
    AsyncDefaultAgent,
)
from example_recipe.word_puzzle.dataset.puzzle_dataset_gen import apply_transform  # noqa: E402
from minisweagent.exceptions import Submitted  # noqa: E402


# --------------------------------------------------------------------------
# Fixtures: fake model + env
# --------------------------------------------------------------------------


class _FakeModel:
    """Records ``format_observation_messages`` inputs; returns pass-through."""

    def format_message(self, **kwargs):
        return {**kwargs}

    def format_observation_messages(self, message, outputs, template_vars=None):
        # Not used by the retry path (retry constructs its own message
        # directly), but ``execute_actions`` still calls this for bash
        # actions. Return one user message per output, mirroring the
        # real GeminiSamplerModel shape.
        return [
            {"role": "user", "content": str(o), "extra": o.get("extra", {})}
            for o in outputs
        ]

    def get_template_vars(self):
        return {}


class _NullEnv:
    """Env stand-in; retry path never calls ``execute``."""

    def execute(self, action):
        raise AssertionError("env.execute should not be called on the retry path")

    def cleanup(self):
        pass


def _make_agent(
    *,
    submit_retries: int,
    expected_answer: str,
    rule=(("shift", 2),),
    extra_word_pool=("apple", "banjo", "crate", "dwarf", "eagle", "fable"),
    reverse_task: bool = False,
    task_seed: int = 42,
) -> AsyncDefaultAgent:
    """Build an AsyncDefaultAgent wired to fake model + env.

    ``rule`` / ``extra_word_pool`` / ``reverse_task`` are the retry
    knobs; ``submit_retries=1`` disables the mechanic entirely (single
    attempt, raises Submitted on first submit regardless of correctness).
    """
    agent = AsyncDefaultAgent(
        model=_FakeModel(),
        env=_NullEnv(),
        system_template="sys",
        instance_template="inst",
        step_limit=100,
        wall_time_limit_seconds=0,
        cost_limit=0.0,
        output_path=None,
        submit_retries=submit_retries,
        expected_answer=expected_answer,
        rule=list(rule),
        extra_word_pool=list(extra_word_pool),
        reverse_task=reverse_task,
        task_seed=task_seed,
    )
    # Simulate a run() having initialised these; execute_actions needs
    # ``messages`` mutable and the retry rng seeded.
    agent.messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "inst"},
    ]
    agent._init_retry_state()
    return agent


def _submit_message(answer: str, tool_call_id: str = "call_abc") -> dict:
    """Assistant message shaped like GeminiSamplerModel.query output."""
    return {
        "role": "assistant",
        "content": "",
        "extra": {
            "actions": [
                {
                    "tool_name": "submit",
                    "answer": answer,
                    "tool_call_id": tool_call_id,
                }
            ]
        },
    }


# --------------------------------------------------------------------------
# Backwards compat: retries=1 preserves original behaviour
# --------------------------------------------------------------------------


def test_submit_retries_1_raises_submitted_on_correct_answer():
    """submit_retries=1 with a correct answer: raises Submitted (same as before)."""
    agent = _make_agent(submit_retries=1, expected_answer="hello")
    with pytest.raises(Submitted) as exc:
        asyncio.run(agent.execute_actions(_submit_message("hello")))
    assert exc.value.messages[0]["extra"]["submission"] == "hello"


def test_submit_retries_1_raises_submitted_on_wrong_answer():
    """submit_retries=1 with a wrong answer STILL raises Submitted.

    With only one attempt allowed, wrong submissions terminate the
    rollout the same as correct ones -- the run-loop breaks and the
    grader scores it wrong. No retry, no injection.
    """
    agent = _make_agent(submit_retries=1, expected_answer="hello")
    with pytest.raises(Submitted) as exc:
        asyncio.run(agent.execute_actions(_submit_message("goodbye")))
    # Whatever the model submitted is recorded, even though it's wrong.
    assert exc.value.messages[0]["extra"]["submission"] == "goodbye"


# --------------------------------------------------------------------------
# Retry path: correct answer always terminates
# --------------------------------------------------------------------------


def test_correct_answer_raises_submitted_even_with_retries_remaining():
    """A correct answer terminates immediately regardless of remaining budget."""
    agent = _make_agent(submit_retries=3, expected_answer="hello")
    with pytest.raises(Submitted):
        asyncio.run(agent.execute_actions(_submit_message("hello")))


# --------------------------------------------------------------------------
# Retry path: wrong answer with budget -> inject + continue
# --------------------------------------------------------------------------


def test_wrong_answer_with_budget_does_not_raise_submitted():
    """Wrong submit with retries remaining must NOT terminate the run.

    The run-loop's break condition is ``messages[-1].role == 'exit'``.
    A retry-injected observation is role='tool' (mini's toolcall
    convention; also routes to Vertex ``functionResponse`` via the
    generic OpenAI->Gemini converter), not 'exit', so the loop
    continues into the next query() call.
    """
    agent = _make_agent(submit_retries=3, expected_answer="hello")
    # No exception raised: retry path swallows it and mutates messages.
    result = asyncio.run(agent.execute_actions(_submit_message("goodbye")))
    # execute_actions returns the injected observation list.
    assert result, "execute_actions should return the injected observation(s)"
    # role='tool' — not 'user' (which would become a plain text Part on
    # the next Vertex turn, drifting the model out of tool-calling
    # mode) and not 'exit' (which would break the run-loop).
    assert result[-1]["role"] == "tool"
    assert result[-1]["role"] != "exit"
    # Top-level tool_call_id (mini's convention) — required so the
    # OpenAI->Gemini converter's id->name map populates the paired
    # functionResponse Part's ``name`` field correctly.
    assert result[-1]["tool_call_id"] is not None


def test_wrong_answer_injects_two_new_pairs_from_pool():
    """Injected observation carries exactly 2 (input, output) pairs from the pool.

    Sequential slicing of the task-shuffled pool: first retry consumes
    positions 0-1 (post-shuffle).
    """
    pool = ("apple", "banjo", "crate", "dwarf", "eagle", "fable")
    rule = (("shift", 2),)
    agent = _make_agent(
        submit_retries=3,
        expected_answer="hello",
        rule=rule,
        extra_word_pool=pool,
        task_seed=42,
    )
    asyncio.run(agent.execute_actions(_submit_message("goodbye")))
    inj = agent.messages[-1]
    pairs = inj["extra"]["new_pairs"]
    assert len(pairs) == 2, f"expected 2 pairs, got {len(pairs)}: {pairs}"
    # Each pair is (input, output). reverse_task=False means the
    # puzzle's QUESTION direction is "given encrypted, produce
    # original", so a retry pair mirrors that: input=encrypted(w),
    # output=w. (Note: the demo section in the prompt always shows
    # (raw, encrypted); reverse_task only flips the question, not
    # the demo.)
    for input_word, output_word in pairs:
        assert apply_transform(output_word, list(rule)) == input_word, (
            f"pair ({input_word!r}, {output_word!r}) is not consistent "
            f"with rule={rule} at reverse_task=False (input must be "
            f"apply_transform(output))"
        )
        # Words drawn from the pool.
        assert output_word in pool, f"output word {output_word!r} not in pool"


def test_wrong_answer_pair_direction_respects_reverse_task_true():
    """reverse_task=True flips the pair direction: input=w, output=encrypted(w).

    With reverse_task=True the puzzle's QUESTION direction is "given
    original, produce encrypted", so a retry pair mirrors that:
    input=w (original), output=encrypted(w). (The demo section in
    the prompt still shows (raw, encrypted); reverse_task only flips
    which side of that pair the model must produce.)
    """
    rule = (("shift", 2),)
    pool = ("apple", "banjo", "crate", "dwarf")
    agent = _make_agent(
        submit_retries=3,
        expected_answer="hello",
        rule=rule,
        extra_word_pool=pool,
        reverse_task=True,
    )
    asyncio.run(agent.execute_actions(_submit_message("goodbye")))
    inj = agent.messages[-1]
    pairs = inj["extra"]["new_pairs"]
    for input_word, output_word in pairs:
        assert apply_transform(input_word, list(rule)) == output_word, (
            f"reverse_task=True: pair ({input_word!r}, {output_word!r}) "
            f"must satisfy output = apply_transform(input)"
        )
        assert input_word in pool


def test_wrong_answer_records_retry_metadata_in_extra():
    """Trajectory extra carries retry_attempt + wrong_answer + attempts_remaining."""
    agent = _make_agent(submit_retries=3, expected_answer="hello")
    asyncio.run(agent.execute_actions(_submit_message("goodbye", tool_call_id="tc1")))
    extra = agent.messages[-1]["extra"]
    assert extra["retry_attempt"] == 1
    assert extra["wrong_answer"] == "goodbye"
    # submit_retries=3, first wrong attempt consumed => 2 attempts left.
    assert extra["attempts_remaining"] == 2
    assert extra["tool_call_id"] == "tc1"
    assert extra["tool_name"] == "submit"


def test_wrong_answer_content_is_json_dump_of_payload():
    """The message ``content`` string is a JSON dump of the same payload sent to Gemini.

    Rationale: trace-viewer readability without losing fidelity;
    grepping traces for 'wrong_answer' or 'attempts_remaining' should
    hit both the extra and content fields.
    """
    agent = _make_agent(submit_retries=3, expected_answer="hello")
    asyncio.run(agent.execute_actions(_submit_message("goodbye")))
    content = agent.messages[-1]["content"]
    payload = json.loads(content)
    assert payload["status"] == "wrong_answer"
    assert payload["wrong_answer"] == "goodbye"
    assert payload["attempts_remaining"] == 2
    assert len(payload["additional_examples"]) == 2
    # Pair objects use input/output keys, not encrypted/original.
    for ex in payload["additional_examples"]:
        assert set(ex.keys()) == {"input", "output"}
    assert "submit again" in payload["instruction"].lower()


# --------------------------------------------------------------------------
# Retry path: consecutive wrong answers consume sequential slices
# --------------------------------------------------------------------------


def test_two_consecutive_wrong_answers_draw_disjoint_pairs():
    """Retry N and retry N+1 draw non-overlapping words from the pool.

    Sequential slicing: retry 1 -> pool[0:2], retry 2 -> pool[2:4], etc.
    """
    pool = ("apple", "banjo", "crate", "dwarf", "eagle", "fable")
    agent = _make_agent(
        submit_retries=3,
        expected_answer="hello",
        extra_word_pool=pool,
        task_seed=42,
    )
    asyncio.run(agent.execute_actions(_submit_message("wrong1")))
    first_pairs = agent.messages[-1]["extra"]["new_pairs"]
    asyncio.run(agent.execute_actions(_submit_message("wrong2")))
    second_pairs = agent.messages[-1]["extra"]["new_pairs"]

    first_words = {w for _, w in first_pairs}
    second_words = {w for _, w in second_pairs}
    assert first_words.isdisjoint(second_words), (
        f"consecutive retries drew overlapping words: {first_words} & {second_words}"
    )
    # And the second attempt's metadata reflects the decrement.
    assert agent.messages[-1]["extra"]["retry_attempt"] == 2
    assert agent.messages[-1]["extra"]["attempts_remaining"] == 1


def test_shuffle_is_deterministic_across_agent_instances_with_same_seed():
    """Same task_seed + same pool -> same shuffled order -> same first pair."""
    pool = ("apple", "banjo", "crate", "dwarf", "eagle", "fable")
    a1 = _make_agent(
        submit_retries=3, expected_answer="hello", extra_word_pool=pool, task_seed=7
    )
    a2 = _make_agent(
        submit_retries=3, expected_answer="hello", extra_word_pool=pool, task_seed=7
    )
    asyncio.run(a1.execute_actions(_submit_message("wrong")))
    asyncio.run(a2.execute_actions(_submit_message("wrong")))
    assert (
        a1.messages[-1]["extra"]["new_pairs"] == a2.messages[-1]["extra"]["new_pairs"]
    )


def test_shuffle_differs_across_seeds():
    """Different task_seed -> different shuffle (full ordering differs).

    Uses a 20-word pool + a seed pair verified deterministically to
    produce different FULL orderings (not just different first pairs).
    The full-ordering assertion is far stronger than a "first pair
    differs" check -- with 20! ~= 2.4e18 possible orderings the
    collision probability for any specific seed pair is astronomically
    low, but more importantly, the seeds (1, 99) have been checked
    empirically and produce distinguishable orderings, so this test
    is deterministic under the current Python random.Random
    implementation.

    If Python's random.Random.shuffle implementation changes and this
    test starts failing, pick a new seed pair using the verification
    script in the commit message rather than relaxing the assertion.
    """
    # 20-word pool; large enough to make coincidental full-shuffle
    # matches astronomically unlikely + verified deterministically
    # to differ for the chosen seeds.
    pool = (
        "apple",
        "banjo",
        "crate",
        "dwarf",
        "eagle",
        "fable",
        "grape",
        "honey",
        "igloo",
        "jelly",
        "kayak",
        "lemon",
        "mango",
        "noble",
        "olive",
        "panda",
        "quiet",
        "raven",
        "sugar",
        "tulip",
    )
    a1 = _make_agent(
        submit_retries=3, expected_answer="hello", extra_word_pool=pool, task_seed=1
    )
    a2 = _make_agent(
        submit_retries=3, expected_answer="hello", extra_word_pool=pool, task_seed=99
    )
    # Compare the FULL shuffled pool orderings, not just the first pair.
    # ``_init_retry_state`` already shuffled the pools; inspect them
    # directly rather than triggering a retry (which only surfaces
    # the first 2 words).
    assert a1._shuffled_pool != a2._shuffled_pool, (
        f"seeds 1 and 99 must produce distinguishable full shuffled "
        f"orderings; got identical orderings {a1._shuffled_pool}. "
        f"If Python's random.Random.shuffle implementation changed, "
        f"verify with a fresh seed pair rather than relaxing this "
        f"assertion."
    )


# --------------------------------------------------------------------------
# Retry path: budget exhaustion -> raise Submitted
# --------------------------------------------------------------------------


def test_final_wrong_answer_raises_submitted_when_budget_exhausted():
    """When attempts_remaining hits 0, next wrong submit raises Submitted."""
    agent = _make_agent(submit_retries=2, expected_answer="hello")
    # First wrong attempt: injected, no raise.
    asyncio.run(agent.execute_actions(_submit_message("wrong1")))
    # Second (final) wrong attempt: budget exhausted, raises.
    with pytest.raises(Submitted) as exc:
        asyncio.run(agent.execute_actions(_submit_message("wrong2")))
    assert exc.value.messages[0]["extra"]["submission"] == "wrong2"


# --------------------------------------------------------------------------
# Retry path: empty / short pool
# --------------------------------------------------------------------------


def test_empty_pool_falls_through_to_submitted_on_wrong_answer():
    """No extra_word_pool means no retry mechanism, wrong submit terminates.

    Defensive: some tasks may be generated without ``--include-retry-metadata``
    and reach a submit_retries>1 runner. The agent must not crash trying
    to slice an empty pool; it degrades to single-attempt behaviour.
    """
    agent = _make_agent(submit_retries=3, expected_answer="hello", extra_word_pool=())
    with pytest.raises(Submitted):
        asyncio.run(agent.execute_actions(_submit_message("goodbye")))


def test_init_retry_state_publishes_submit_retries_to_template_vars():
    """``_init_retry_state`` MUST expose ``submit_retries`` in template vars.

    Regression test for the smoke-test bug where the system_template
    referenced ``{{ submit_retries }}`` but ``run()`` rendered the
    template BEFORE calling ``_init_retry_state()``, so jinja2's
    StrictUndefined raised ``'submit_retries' is undefined`` and the
    entire rollout crashed before any model call.

    This test pins the invariant that after ``_init_retry_state`` runs,
    the var is available -- regardless of whether the feature is on
    or off, so the template can safely gate on ``submit_retries > 1``
    without a defensive ``is defined`` check.
    """
    agent = _make_agent(submit_retries=1, expected_answer="hello")
    assert "submit_retries" in agent.extra_template_vars, (
        "submit_retries must be in extra_template_vars so the system "
        "template's `{% if submit_retries > 1 %}` guard doesn't raise "
        "StrictUndefined"
    )
    assert agent.extra_template_vars["submit_retries"] == 1


# --------------------------------------------------------------------------
# Constructor-time validation of retry_metadata correctness
# --------------------------------------------------------------------------


def test_init_rejects_rule_with_unknown_operator_when_feature_on():
    """Unknown ops in ``rule`` must fail at __init__, not mid-rollout.

    Rationale: without this guard, a malformed ``retry_metadata.rule``
    (e.g. from a stale JSONL or a hand-edit typo) only surfaces when
    the FIRST wrong submit fires -- silently, as a per-task
    ``ValueError`` swallowed by the runner's broad exception handler.
    That makes the diagnostic invisible until several minutes of eval
    time have burned. Fail-early in __init__ turns it into a startup
    error with a clear message so the operator sees it before the
    first LLM call.

    Only applies when the retry feature is on (submit_retries > 1);
    when it's off, ``rule`` is unused and we don't touch it.
    """
    with pytest.raises(ValueError, match="unknown operator"):
        _make_agent(
            submit_retries=3,
            expected_answer="hello",
            rule=(("bogus_op", None),),
        )


def test_init_accepts_rule_with_unknown_op_when_feature_off():
    """When submit_retries=1 (feature off), ``rule`` is never consulted.

    Don't reject rows with bad rules at construction time when the
    feature is off -- some runners may pass ``rule`` through
    unconditionally, and rejecting them would break the
    "retries=1 preserves original behaviour" contract.
    """
    # Should not raise.
    agent = _make_agent(
        submit_retries=1,
        expected_answer="hello",
        rule=(("bogus_op", None),),
    )
    assert agent is not None


def test_init_accepts_rule_with_unknown_op_when_pool_empty():
    """Empty pool = feature effectively off; skip rule validation.

    Same rationale as retries=1: if the pool is empty, the rule is
    never applied, so we shouldn't crash startup on it.
    """
    # Should not raise.
    agent = _make_agent(
        submit_retries=3,
        expected_answer="hello",
        rule=(("bogus_op", None),),
        extra_word_pool=(),
    )
    assert agent is not None


def test_init_rejects_empty_expected_answer_when_feature_on():
    """Empty ``expected_answer`` must fail at __init__ when feature is on.

    Without this guard, a caller enabling ``submit_retries > 1`` but
    forgetting to pass ``expected_answer`` (leaving the default "")
    would have every submit scored wrong (nothing == "" is False for
    any real answer), silently burning the entire retry budget on
    every task and manifesting as "the retry feature doesn't help".
    Failing loud at construction time turns this into a startup error
    with a clear message.

    Only enforced when the feature is on (retries > 1 AND pool
    non-empty); when off, expected_answer is unused and the empty
    default is fine.
    """
    with pytest.raises(ValueError, match="expected_answer"):
        _make_agent(
            submit_retries=3,
            expected_answer="",  # BUG: forgot to pass it
        )


def test_init_accepts_empty_expected_answer_when_feature_off():
    """When submit_retries=1 (feature off), expected_answer is never consulted."""
    # Should not raise.
    agent = _make_agent(submit_retries=1, expected_answer="")
    assert agent is not None


def test_init_accepts_empty_expected_answer_when_pool_empty():
    """Empty pool = feature effectively off; skip expected_answer check."""
    # Should not raise.
    agent = _make_agent(submit_retries=3, expected_answer="", extra_word_pool=())
    assert agent is not None


def test_init_accepts_valid_rule_including_json_list_form():
    """List-of-lists rules (JSON round-trip shape) must validate cleanly.

    Regression: earlier the rule was typed ``Sequence[tuple[str,Any]]``,
    but JSONL rows deserialize into list-of-lists. Validation must
    accept both, mirroring apply_transform's widened signature.
    """
    # Should not raise (rule is valid, feature is on).
    agent = _make_agent(
        submit_retries=3,
        expected_answer="hello",
        rule=[["shift", 2], ["mirror", True]],  # list, not tuple
    )
    assert agent is not None


def test_short_pool_uses_all_it_can_then_terminates():
    """If pool has fewer than 2*retries words, later retries fall through.

    Example: pool of 3 words + submit_retries=3 (=> up to 2 retries).
    First retry consumes pool[0:2]. Second retry would need pool[2:4]
    but only pool[2:3] is available -- can't form 2 pairs, degrades to
    Submitted rather than emitting a malformed 1-pair observation.
    """
    pool = ("apple", "banjo", "crate")
    agent = _make_agent(submit_retries=3, expected_answer="hello", extra_word_pool=pool)
    # First wrong: consumes 2 words, injects normally.
    asyncio.run(agent.execute_actions(_submit_message("wrong1")))
    assert len(agent.messages[-1]["extra"]["new_pairs"]) == 2
    # Second wrong: only 1 word left in pool, can't form 2 pairs -> terminates.
    with pytest.raises(Submitted):
        asyncio.run(agent.execute_actions(_submit_message("wrong2")))
