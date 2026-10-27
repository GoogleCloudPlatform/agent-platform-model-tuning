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

"""Tests for word_puzzle (extract_answer_fn / Env / Dataset / Iterator)."""

import dataclasses
import json
import os
import tempfile
import unittest

import pytest
from google.cloud.aiplatform_v1.types import Content, Part

from core.dataset import BatchIterator, JsonlDataset
from core.protocols import Action
from example_recipe.word_puzzle import (
    WordPuzzleEnv,
    extract_answer_fn,
    parse_word_puzzle_item,
)
from example_recipe.word_puzzle.reward import (
    WordPuzzleRewardConfig,
    WordPuzzleRewardFn,
)
from example_recipe.word_puzzle.stats import WordPuzzleStepStats
from example_recipe.word_puzzle.dataset.task_data import WordPuzzleTaskData


def _make_action(
    text: str = "",
    *,
    thought: bool = False,
    finish_reason: str = "STOP",
    generation_token_count: int = 42,
    parts: list[Part] | None = None,
) -> Action:
    """Build a minimal Action for word_puzzle tests.

    ``text`` + ``thought`` compose a single Part; pass ``parts`` for
    multi-part / thought-signature scenarios.
    """
    if parts is None:
        parts = [Part(text=text, thought=thought)]
    return Action(
        content=Content(role="model", parts=parts),
        finish_reason=finish_reason,
        generation_token_count=generation_token_count,
    )


def WordPuzzleDataset(file_path):
    return JsonlDataset(file_path, transform_fn=parse_word_puzzle_item)


WordPuzzleIterator = BatchIterator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_jsonl(path, records):
    """Write each record (dict or raw str) as one JSONL line."""
    with open(path, "w") as f:
        for rec in records:
            if isinstance(rec, str):
                f.write(rec + "\n")
            else:
                f.write(json.dumps(rec) + "\n")


def _dataset_record(text, reference):
    """Build a minimal raw input record consumed by WordPuzzleDataset."""
    return {
        "contents": [{"role": "user", "parts": [{"text": text}]}],
        "references": {"reference": reference},
    }


class _FakeDataset:
    """Minimal len/getitem dataset used by iterator tests."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ---------------------------------------------------------------------------
# extract_answer_fn
# ---------------------------------------------------------------------------
class TestExtractAnswerFn(unittest.TestCase):
    def test_single_match(self):
        action = _make_action(text=r"This is a prompt. \text{hello world} output.")
        self.assertEqual(extract_answer_fn(action), "hello world")

    def test_multiple_matches_returns_last(self):
        action = _make_action(text=r"\text{first} and \text{last}")
        self.assertEqual(extract_answer_fn(action), "last")

    def test_no_match_returns_empty(self):
        action = _make_action(text="No answer here")
        self.assertEqual(extract_answer_fn(action), "")

    def test_empty_braces(self):
        action = _make_action(text=r"\text{}")
        self.assertEqual(extract_answer_fn(action), "")

    def test_preserves_inner_whitespace(self):
        action = _make_action(text=r"\text{ abc }")
        self.assertEqual(extract_answer_fn(action), " abc ")

    def test_accepts_multi_part_content(self):
        action = _make_action(
            parts=[
                Part(text=r"This is a prompt. \text{hello "),
                Part(text=r"world} output."),
            ]
        )
        self.assertEqual(extract_answer_fn(action), "hello world")

    def test_filters_thought_parts(self):
        # Thought parts must not contribute to the parsed answer, so a
        # ``\text{...}`` block inside a thought should be ignored even if it
        # appears later than the real answer.
        action = _make_action(
            parts=[
                Part(text=r"reasoning... \text{wrong}", thought=True),
                Part(text=r"final \text{right}"),
            ]
        )
        self.assertEqual(extract_answer_fn(action), "right")

    def test_thought_only_returns_empty(self):
        action = _make_action(parts=[Part(text=r"\text{hidden}", thought=True)])
        self.assertEqual(extract_answer_fn(action), "")

    def test_keeps_thought_signature_answer_part(self):
        # The thought_signature rides the answer/action chunk (not the thought
        # summary), so it must be kept. Only ``thought=True`` is filtered.
        action = _make_action(
            parts=[
                Part(text=r"summary \text{wrong}", thought=True),
                Part(text=r"final \text{right}", thought_signature=b"CvcE"),
            ]
        )
        self.assertEqual(extract_answer_fn(action), "right")

    def test_thought_signature_only_is_kept_as_answer(self):
        # A lone text chunk carrying only a thought_signature is the answer.
        action = _make_action(
            parts=[Part(text=r"\text{hidden}", thought_signature=b"CvcE")]
        )
        self.assertEqual(extract_answer_fn(action), "hidden")


# ---------------------------------------------------------------------------
# WordPuzzleEnv
# ---------------------------------------------------------------------------
class TestWordPuzzleEnv(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _default_reward_fn() -> WordPuzzleRewardFn:
        return WordPuzzleRewardFn(WordPuzzleRewardConfig())

    def _make_env(self, **overrides):
        fields = {"task_id": "t", "question": "What is 1+1?", "reference": "2"}
        fields.update(overrides)
        task = WordPuzzleTaskData(**fields)
        return WordPuzzleEnv(task, reward_fn=self._default_reward_fn())

    def test_get_prompt(self):
        env = self._make_env()
        prompt = env.get_prompt()
        self.assertEqual(prompt.contents[0].role, "user")
        self.assertEqual(prompt.contents[0].parts[0].text, "What is 1+1?")

    async def test_step_correct_answer_rewards_positive(self):
        env = self._make_env()
        result = await env.step(_make_action(text=r"the answer is \text{2}"))
        self.assertTrue(result.done)
        self.assertEqual(result.reward, 1.0)
        self.assertTrue(result.is_correct)

    async def test_step_incorrect_answer_rewards_negative(self):
        env = self._make_env()
        result = await env.step(_make_action(text=r"the answer is \text{3}"))
        self.assertTrue(result.done)
        self.assertEqual(result.reward, -1.0)
        self.assertFalse(result.is_correct)

    async def test_step_missing_reference_is_incorrect(self):
        env = WordPuzzleEnv(
            WordPuzzleTaskData(task_id="t", question="Q"),  # no reference
            reward_fn=self._default_reward_fn(),
        )
        result = await env.step(_make_action(text=r"\text{anything}"))
        self.assertEqual(result.reward, -1.0)
        self.assertFalse(result.is_correct)

    async def test_step_malformed_function_call_returns_neg_four(self):
        # Vertex Gemini reports an unparseable function call: short-circuit
        # with the fixed penalty regardless of the rest of the action.
        env = self._make_env()
        action = _make_action(
            text=r"\text{2}",
            finish_reason="MALFORMED_FUNCTION_CALL",
            generation_token_count=10,
        )
        result = await env.step(action)
        self.assertEqual(result.reward, -4.0)
        self.assertTrue(result.done)
        # Recipe-side signal that the malformed short-circuit fired.
        self.assertTrue(result.task_stats.malformed_function_call)

    async def test_step_malformed_fc_short_circuits_before_correctness_check(
        self,
    ):
        # Even when the answer text would have matched the reference, the
        # malformed-FC signal still wins because the upstream parser already
        # judged the output unparseable.
        env = self._make_env()  # reference == "2"
        action = _make_action(
            text=r"the answer is \text{2}",
            finish_reason="MALFORMED_FUNCTION_CALL",
            generation_token_count=10,
        )
        result = await env.step(action)
        self.assertEqual(result.reward, -4.0)
        # Critical: even with a textually correct answer, malformed FC means
        # the parsed answer is not trusted. The env AND-s the raw text-match
        # with ``not malformed_function_call`` so ``EnvStepResult.is_correct``
        # (which feeds ``learning/sampling_accuracy`` / ``learning/pass_at_k``)
        # stays False on the malformed short-circuit path.
        self.assertFalse(result.is_correct)
        self.assertTrue(result.task_stats.malformed_function_call)

    async def test_step_normal_finish_reason_does_not_trigger_malformed_path(
        self,
    ):
        env = self._make_env()
        action = _make_action(
            text=r"\text{2}", finish_reason="STOP", generation_token_count=10
        )
        result = await env.step(action)
        self.assertEqual(result.reward, 1.0)
        self.assertFalse(result.task_stats.malformed_function_call)

    async def test_step_max_tokens_with_empty_content_picks_correct_token_count(
        self,
    ):
        """MAX_TOKENS responses arrive with empty parts but the per-candidate
        generation_token_count is still populated. The token count must flow
        through unchanged, not silently zeroed.
        """
        env = self._make_env()
        # Realistic MAX_TOKENS shape: the candidate has no decodable parts,
        # only the truncated finish_reason plus the token count.
        action = _make_action(
            parts=[],  # truncated -- no decodable content
            finish_reason="MAX_TOKENS",
            generation_token_count=32768,
        )
        result = await env.step(action)
        # The reported generation_token_count is the truncated candidate's
        # own count.
        self.assertEqual(result.task_stats.generation_token_count, 32768)
        # Empty parts -> parsed answer is empty string -> not equal to "2" ->
        # is_correct=0; reward = -1 (wrong answer; this recipe applies no
        # length-based reward shaping).
        self.assertFalse(result.is_correct)
        self.assertEqual(result.reward, -1.0)

    async def test_step_plumbs_depth_from_tags_to_signals(self):
        """`WordPuzzleEnv.step` must forward `task_data.tags['depth']` into
        WordPuzzleRewardSignals.depth so depth-conditioned reward fns can
        see it. Uses a spy reward_fn to capture the signals.
        """
        captured: list = []

        class _SpyRewardFn:
            async def __call__(self, signals):
                captured.append(signals)
                return (0.0, WordPuzzleStepStats(generation_token_count=0))

        task = WordPuzzleTaskData(
            task_id="t",
            question="Q",
            reference="2",
            tags={"depth": 4, "operator_set": "extended_v2"},
        )
        env = WordPuzzleEnv(task, reward_fn=_SpyRewardFn())
        await env.step(_make_action(text=r"\text{2}"))
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].depth, 4)

    async def test_step_depth_is_none_when_tags_missing(self):
        """Legacy datasets without tags must not crash; depth is None."""
        captured: list = []

        class _SpyRewardFn:
            async def __call__(self, signals):
                captured.append(signals)
                return (0.0, WordPuzzleStepStats(generation_token_count=0))

        task = WordPuzzleTaskData(task_id="t", question="Q", reference="2")
        env = WordPuzzleEnv(task, reward_fn=_SpyRewardFn())
        await env.step(_make_action(text=r"\text{2}"))
        self.assertIsNone(captured[0].depth)


# ---------------------------------------------------------------------------
# WordPuzzleDataset
# ---------------------------------------------------------------------------
class TestWordPuzzleDataset(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.path = os.path.join(self._tmpdir.name, "data.jsonl")

    def test_len(self):
        _write_jsonl(
            self.path,
            [_dataset_record("q1", "r1"), _dataset_record("q2", "r2")],
        )
        self.assertEqual(len(WordPuzzleDataset(self.path)), 2)

    def test_getitem_extracts_question_and_reference(self):
        _write_jsonl(
            self.path,
            [_dataset_record("q1", "r1"), _dataset_record("q2", "r2")],
        )
        dataset = WordPuzzleDataset(self.path)
        td0 = dataset[0]
        td1 = dataset[1]
        self.assertIsInstance(td0, WordPuzzleTaskData)
        self.assertEqual(td0.question, "q1")
        self.assertEqual(td0.reference, "r1")
        self.assertIsInstance(td1, WordPuzzleTaskData)
        self.assertEqual(td1.question, "q2")
        self.assertEqual(td1.reference, "r2")

    def test_shuffle_preserves_length(self):
        _write_jsonl(
            self.path,
            [_dataset_record("q1", "r1"), _dataset_record("q2", "r2")],
        )
        dataset = WordPuzzleDataset(self.path)
        dataset.shuffle(seed=42)
        self.assertEqual(len(dataset), 2)

    def test_skips_blank_lines(self):
        _write_jsonl(self.path, ["", _dataset_record("q", "r"), ""])
        self.assertEqual(len(WordPuzzleDataset(self.path)), 1)


# ---------------------------------------------------------------------------
# WordPuzzleIterator
# ---------------------------------------------------------------------------
class TestWordPuzzleIterator(unittest.TestCase):
    def test_no_repeat_yields_full_then_partial_batch(self):
        it = WordPuzzleIterator(
            _FakeDataset([1, 2, 3, 4, 5]), batch_size=2, repeat=False
        )
        self.assertEqual(next(it), [1, 2])
        self.assertEqual(next(it), [3, 4])
        self.assertEqual(next(it), [5])

    def test_no_repeat_raises_stop_iteration_when_exhausted(self):
        it = WordPuzzleIterator(_FakeDataset([1, 2]), batch_size=2, repeat=False)
        next(it)
        with self.assertRaises(StopIteration):
            next(it)

    def test_repeat_wraps_around(self):
        it = WordPuzzleIterator(_FakeDataset([1, 2, 3]), batch_size=2, repeat=True)
        self.assertEqual(next(it), [1, 2])
        self.assertEqual(next(it), [3, 1])
        self.assertEqual(next(it), [2, 3])

    def test_resume_via_current_attribute(self):
        it = WordPuzzleIterator(_FakeDataset([1, 2, 3, 4]), batch_size=2, repeat=False)
        it.current = 2
        self.assertEqual(next(it), [3, 4])
        with self.assertRaises(StopIteration):
            next(it)


def test_word_puzzle_metrics_aggregator_deleted():
    """The word_puzzle_metrics aggregator is deleted (both keys were
    framework duplicates)."""
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("example_recipe.word_puzzle.metrics")


# ---------------------------------------------------------------------------
# Producer/aggregator seam cleanup
# ---------------------------------------------------------------------------
def test_word_puzzle_step_stats_has_no_is_correct_field():
    """WordPuzzleStepStats no longer carries the duplicate is_correct field
    (framework StepStats.is_correct is authoritative)."""
    from example_recipe.word_puzzle.stats import WordPuzzleStepStats

    field_names = {f.name for f in dataclasses.fields(WordPuzzleStepStats)}
    assert "is_correct" not in field_names
    # submit_retries_used is an ours-side addition for the retry-on-wrong-answer
    # feature (see word_puzzle env retry_metadata plumbing). notes_used counts
    # no-op "thinking" tool calls for recipes that enable that tool.
    # solve_step_count / step_bonus back the step-efficiency reward bonus;
    # rollout_generation_token_count / token_bonus back the token-efficiency
    # one (whole-rollout tokens, not the terminal step's).
    assert field_names == {
        "generation_token_count",
        "malformed_function_call",
        "submit_retries_used",
        "notes_used",
        "solve_step_count",
        "step_bonus",
        "token_bonus",
        "rollout_generation_token_count",
    }


if __name__ == "__main__":
    unittest.main()
