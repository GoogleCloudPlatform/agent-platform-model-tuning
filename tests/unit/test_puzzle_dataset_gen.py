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

"""Tests for ``example_recipe.word_puzzle.dataset.puzzle_dataset_gen``.

Focus areas:

1. **Determinism** -- the SAME ``seed`` must produce byte-identical output,
   both for ``generate_one_task`` and for the batched ``generate_dataset``
   entry point that the CLI wrapper (``run_generated_eval.py``) drives. This
   is a load-bearing contract: any regression that inserts nondeterministic
   randomness (a bare ``random.random()``, a set-ordering iteration, a dict
   iteration order dependency, ...) would silently break the ability to
   reproduce an eval run.
2. **Seed sensitivity** -- different seeds must produce different tasks
   (guards against a stub that accidentally ignores ``rng`` and returns a
   fixed task).
3. **Round-trip correctness** -- ``apply_transform(original, rule) ==
   encrypted`` for every emitted (original, encrypted) pair, so the
   generated tasks are self-consistent.
4. **JSONL wire-shape parity** -- the emitted records match the schema of
   ``combined_{training,eval}_word_transformation_v1.jsonl`` exactly.
"""

from __future__ import annotations

import json
import os
import random
import sys
import tempfile
import unittest

from example_recipe.word_puzzle.dataset import puzzle_dataset_gen as pdg
from example_recipe.word_puzzle.dataset.task_data import WordPuzzleTaskData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# A fixed, ordered word list -- avoids depending on the bundled list or the
# system dictionary. Ordering matters for reproducibility (see the seed-sensitivity
# test below).
_WORDS = [
    "python",
    "javascript",
    "programming",
    "algorithm",
    "cryptography",
    "transformation",
    "computer",
    "science",
    "language",
    "framework",
    "database",
    "network",
    "developer",
    "engineer",
    "software",
    "hardware",
    "compiler",
    "interpreter",
    "function",
    "variable",
]


def _task_signature(task: pdg.PuzzleTask) -> tuple[object, ...]:
    """Structural fingerprint of a task. Order-sensitive, covers all the
    fields randomness can affect."""
    return (
        task.original_word,
        task.encrypted_word,
        task.expected_answer,
        tuple(task.rule),
        task.template_id,
        tuple(task.few_shot_examples),
        task.reverse_task,
        task.explain_encryption,
        task.num_transformations,
        task.num_few_shot_examples,
        task.prompt,  # last -- catches template-format-string regressions
    )


def _gen_dataset(
    seed: int,
    *,
    n: int = 8,
    operators: list[str] | None = None,
    flip: bool = False,
) -> list[pdg.PuzzleTask]:
    return pdg.generate_dataset(
        num_examples=n,
        num_transformations=3,
        num_few_shot_examples=4,
        word_candidates=_WORDS,
        operators=operators or pdg.EXTENDED_OPERATORS,
        explain_encryption=True,
        flip_reverse_task=flip,
        rng=random.Random(seed),
    )


# ---------------------------------------------------------------------------
# 1. Determinism -- same seed -> byte-identical output
# ---------------------------------------------------------------------------
class TestDeterminism(unittest.TestCase):
    """The core contract: seed -> deterministic tasks."""

    def test_generate_one_task_is_deterministic(self):
        """Two calls with fresh ``Random(seed)`` produce identical tasks."""
        t1 = pdg.generate_one_task(
            num_transformations=3,
            num_few_shot_examples=4,
            word_candidates=_WORDS,
            operators=pdg.EXTENDED_OPERATORS,
            rng=random.Random(42),
        )
        t2 = pdg.generate_one_task(
            num_transformations=3,
            num_few_shot_examples=4,
            word_candidates=_WORDS,
            operators=pdg.EXTENDED_OPERATORS,
            rng=random.Random(42),
        )
        self.assertEqual(_task_signature(t1), _task_signature(t2))

    def test_generate_dataset_is_deterministic(self):
        """Batched generator with same seed -> byte-identical batch."""
        d1 = _gen_dataset(seed=42, n=10)
        d2 = _gen_dataset(seed=42, n=10)
        self.assertEqual(len(d1), len(d2))
        self.assertEqual(len(d1), 10)
        for a, b in zip(d1, d2):
            self.assertEqual(_task_signature(a), _task_signature(b))

    def test_generate_dataset_is_deterministic_with_flip_reverse_task(self):
        """flip_reverse_task=True doubles the batch; determinism must hold
        across the encrypt+decrypt halves."""
        d1 = _gen_dataset(seed=7, n=5, flip=True)
        d2 = _gen_dataset(seed=7, n=5, flip=True)
        self.assertEqual(len(d1), 10)  # 5 forward + 5 reverse
        self.assertEqual(
            [_task_signature(t) for t in d1],
            [_task_signature(t) for t in d2],
        )
        # And the two halves must have opposite reverse_task flags.
        self.assertTrue(all(not t.reverse_task for t in d1[:5]))
        self.assertTrue(all(t.reverse_task for t in d1[5:]))

    def test_jsonl_bytes_are_identical_across_runs(self):
        """The written JSONL file is byte-for-byte reproducible.

        This is the strongest form of the contract and the one the
        ``run_generated_eval.py`` wrapper actually relies on.
        """
        with tempfile.TemporaryDirectory() as td:
            p1 = os.path.join(td, "run1.jsonl")
            p2 = os.path.join(td, "run2.jsonl")
            pdg.write_jsonl(_gen_dataset(seed=123, n=6), p1)
            pdg.write_jsonl(_gen_dataset(seed=123, n=6), p2)
            with open(p1, "rb") as f1, open(p2, "rb") as f2:
                self.assertEqual(f1.read(), f2.read())

    def test_determinism_across_all_operator_sets(self):
        """The seed contract must hold regardless of which operator pool is
        chosen (basic / extended / withhold / all)."""
        for name, ops in pdg.OPERATOR_SETS.items():
            with self.subTest(operator_set=name):
                d1 = _gen_dataset(seed=99, n=4, operators=ops)
                d2 = _gen_dataset(seed=99, n=4, operators=ops)
                self.assertEqual(
                    [_task_signature(t) for t in d1],
                    [_task_signature(t) for t in d2],
                    f"non-determinism detected for operator_set={name!r}",
                )


# ---------------------------------------------------------------------------
# 2. Seed sensitivity -- different seeds produce different tasks
# ---------------------------------------------------------------------------
class TestSeedSensitivity(unittest.TestCase):
    def test_different_seeds_produce_different_batches(self):
        """seed=42 vs seed=43 must diverge for at least one of the 10 tasks.

        Guards against a bug that silently ignores ``rng`` (would produce
        the same first task regardless of seed).
        """
        d1 = _gen_dataset(seed=42, n=10)
        d2 = _gen_dataset(seed=43, n=10)
        sigs1 = [_task_signature(t) for t in d1]
        sigs2 = [_task_signature(t) for t in d2]
        # At least one task must differ. In practice ALL of them differ,
        # but we assert the weaker property to avoid depending on the
        # specific PRNG implementation.
        self.assertNotEqual(sigs1, sigs2)
        # Stronger: the FIRST task must differ (catches any code path that
        # samples the first task before touching the seeded rng).
        self.assertNotEqual(sigs1[0], sigs2[0])

    def test_no_rng_argument_is_nondeterministic(self):
        """Passing ``rng=None`` (the default) should NOT be deterministic
        across process-internal calls -- otherwise the module has a hidden
        global-state singleton, which is a footgun."""
        # Run each call in isolation so the process-global random state
        # progresses between them.
        t1 = pdg.generate_one_task(
            num_transformations=3,
            num_few_shot_examples=4,
            word_candidates=_WORDS,
            operators=pdg.EXTENDED_OPERATORS,
        )
        t2 = pdg.generate_one_task(
            num_transformations=3,
            num_few_shot_examples=4,
            word_candidates=_WORDS,
            operators=pdg.EXTENDED_OPERATORS,
        )
        # With no seed and independent fresh Random() instances inside the
        # function, the two calls should differ with overwhelming
        # probability. If this test EVER flakes we have a real bug (the
        # module is using a fixed default seed somewhere).
        self.assertNotEqual(_task_signature(t1), _task_signature(t2))


# ---------------------------------------------------------------------------
# 3. Round-trip correctness -- rule applied to originals recovers encrypteds
# ---------------------------------------------------------------------------
class TestRoundTrip(unittest.TestCase):
    def test_rule_applied_to_originals_yields_encrypted(self):
        """For every few-shot pair and every task word, applying the emitted
        rule to the original must produce the emitted encrypted string."""
        tasks = _gen_dataset(seed=2026, n=15)
        for i, t in enumerate(tasks):
            with self.subTest(task_index=i, original=t.original_word):
                self.assertEqual(
                    pdg.apply_transform(t.original_word, t.rule),
                    t.encrypted_word,
                )
                for orig, enc in t.few_shot_examples:
                    self.assertEqual(
                        pdg.apply_transform(orig, t.rule),
                        enc,
                        f"few-shot pair mismatch for {orig!r} in task {i}",
                    )


# ---------------------------------------------------------------------------
# 4. JSONL wire-shape parity with the checked-in dataset
# ---------------------------------------------------------------------------
class TestJsonlWireShape(unittest.TestCase):
    def test_to_jsonl_record_matches_checked_in_schema(self):
        """Emitted record must have the same top-level and nested keys as
        ``combined_{training,eval}_word_transformation_v1.jsonl``, which is
        what ``WordPuzzleEnv`` / ``parse_word_puzzle_item`` consume."""
        task = pdg.generate_one_task(
            num_transformations=2,
            num_few_shot_examples=3,
            word_candidates=_WORDS,
            operators=pdg.EXTENDED_OPERATORS,
            rng=random.Random(0),
        )
        rec = task.to_jsonl_record()
        self.assertEqual(set(rec.keys()), {"contents", "references"})
        self.assertEqual(len(rec["contents"]), 1)
        self.assertEqual(set(rec["contents"][0].keys()), {"role", "parts"})
        self.assertEqual(rec["contents"][0]["role"], "user")
        self.assertEqual(len(rec["contents"][0]["parts"]), 1)
        self.assertEqual(set(rec["contents"][0]["parts"][0].keys()), {"text"})
        self.assertIsInstance(rec["contents"][0]["parts"][0]["text"], str)
        self.assertEqual(set(rec["references"].keys()), {"reference"})
        # The reference is the expected answer (a plain string, matching
        # what WordPuzzleEnv.step compares against).
        self.assertEqual(rec["references"]["reference"], task.expected_answer)

    def test_write_jsonl_round_trips_via_parse_word_puzzle_item(self):
        """Written JSONL must be loadable by the same helper the training
        pipeline uses (``parse_word_puzzle_item``)."""
        from example_recipe.word_puzzle import parse_word_puzzle_item

        tasks = _gen_dataset(seed=1, n=3)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "out.jsonl")
            pdg.write_jsonl(tasks, path)
            with open(path) as f:
                lines = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(lines), len(tasks))
        for task, raw in zip(tasks, lines):
            parsed = parse_word_puzzle_item(raw)
            self.assertIsInstance(parsed, WordPuzzleTaskData)
            self.assertEqual(parsed.question, task.prompt)
            self.assertEqual(parsed.reference, task.expected_answer)


# ---------------------------------------------------------------------------
# 5. max_expanding_ops -- caps expanding-op count per rule
# ---------------------------------------------------------------------------
class TestMaxExpandingOps(unittest.TestCase):
    """Pins the ``max_expanding_ops`` constraint that prevents prompt-length
    blowup when composing deep rules over the withhold set. Without this
    cap, a random 7-op rule over the withhold pool can turn a 9-char word
    into 5-10 KB of encoded string (base64+hex+binary compound
    multiplicatively). See ``EXPANDING_OPERATORS`` in
    ``puzzle_dataset_gen`` for the operator set this constrains.
    """

    def _count_expanding(self, rule: list[tuple[str, object]]) -> int:
        return sum(1 for op, _ in rule if op in pdg.EXPANDING_OPERATORS)

    def test_max_expanding_ops_zero_produces_no_expanding_ops(self):
        """``max_expanding_ops=0`` must yield rules containing zero
        expanding ops. The length-preserving withhold subset is small
        enough (5 ops) that this is always achievable."""
        for k in [3, 5, 8, 12]:
            with self.subTest(num_transformations=k):
                for seed in range(10):
                    rule = pdg.generate_random_rule(
                        num_transformations=k,
                        max_rotate=5,
                        operators=pdg.WITHHOLD_OPERATORS,
                        rng=random.Random(seed),
                        max_expanding_ops=0,
                    )
                    self.assertEqual(len(rule), k)
                    self.assertEqual(
                        self._count_expanding(rule),
                        0,
                        f"seed={seed}: {rule}",
                    )

    def test_max_expanding_ops_caps_at_specified_bound(self):
        """Over 100 random seeds, every rule respects
        ``expanding_count <= max_expanding_ops``. Also verifies the cap is
        non-trivial (at least one rule uses the maximum), so we know the
        knob actually controls behavior."""
        for cap in [1, 2, 3]:
            with self.subTest(max_expanding_ops=cap):
                counts = []
                for seed in range(100):
                    rule = pdg.generate_random_rule(
                        num_transformations=8,
                        max_rotate=5,
                        operators=pdg.WITHHOLD_OPERATORS,
                        rng=random.Random(seed),
                        max_expanding_ops=cap,
                    )
                    n_exp = self._count_expanding(rule)
                    self.assertLessEqual(
                        n_exp,
                        cap,
                        f"seed={seed}: got {n_exp} expanding ops, cap={cap}",
                    )
                    counts.append(n_exp)
                # Confirm cap is exercised: at least one seed hit the cap
                # AND at least one seed produced fewer than the cap
                # (proves the sampling is genuinely uniform in [0, cap]).
                self.assertEqual(max(counts), cap, "cap never reached")
                self.assertLess(min(counts), cap, "cap always hit")

    def test_max_expanding_ops_none_is_unconstrained(self):
        """``max_expanding_ops=None`` (the default) preserves the legacy
        behavior: rules can contain any number of expanding ops."""
        # With 8 ops over the 12-op withhold pool (7 of which are
        # expanding), the probability of getting 0 expanding is
        # (5/12)^8 ~= 0.001, and the probability of getting >= 4 is very
        # high. Over 100 seeds, we should see a wide distribution.
        counts = []
        for seed in range(100):
            rule = pdg.generate_random_rule(
                num_transformations=8,
                max_rotate=5,
                operators=pdg.WITHHOLD_OPERATORS,
                rng=random.Random(seed),
                # max_expanding_ops omitted -> default None
            )
            counts.append(self._count_expanding(rule))
        # Should span a wide range and often exceed 1.
        self.assertGreater(
            max(counts),
            1,
            "unconstrained mode never produced > 1 expanding op",
        )

    def test_max_expanding_ops_errors_on_impossible_constraint(self):
        """Setting max_expanding_ops < num_transformations on an
        all-expanding pool must raise, not silently violate the
        constraint."""
        all_expanding = list(pdg.EXPANDING_OPERATORS)
        with self.assertRaises(ValueError):
            pdg.generate_random_rule(
                num_transformations=5,
                max_rotate=5,
                operators=all_expanding,
                rng=random.Random(0),
                max_expanding_ops=2,  # 2 < 5, no fill available
            )

    def test_generate_one_task_forwards_max_expanding_ops(self):
        """The ``max_expanding_ops`` param on ``generate_one_task`` reaches
        the rule generator (integration test for the plumbing)."""
        task = pdg.generate_one_task(
            num_transformations=8,
            num_few_shot_examples=3,
            word_candidates=_WORDS,
            operators=pdg.WITHHOLD_OPERATORS,
            max_expanding_ops=1,
            rng=random.Random(42),
        )
        self.assertLessEqual(self._count_expanding(task.rule), 1)


# ---------------------------------------------------------------------------
# 6. Tags passthrough -- optional metadata survives end-to-end
# ---------------------------------------------------------------------------
class TestTagsPassthrough(unittest.TestCase):
    """The ``tags`` field is a caller-supplied metadata dict (e.g. difficulty
    tier, operator-set name) that must survive from ``generate_one_task``
    all the way to the eval-results JSONL for per-slice performance
    breakdowns. This class pins the three touch-points:

      1. ``PuzzleTask.tags`` stores what the caller passed
      2. ``PuzzleTask.to_jsonl_record()`` emits a top-level ``tags`` key
         iff tags are non-empty (backward-compat with legacy records)
      3. ``parse_word_puzzle_item(record)`` passes ``tags`` through into
         the task dict that ``WordPuzzleEnv.task_data`` receives
    """

    def _make_task(self, tags):  # type: ignore[no-untyped-def]
        # (tags: Optional[Mapping[str, Any]] — annotation omitted to avoid
        #  pyright dict-invariance friction; runtime accepts any dict.)
        return pdg.generate_one_task(
            num_transformations=2,
            num_few_shot_examples=3,
            word_candidates=_WORDS,
            operators=pdg.EXTENDED_OPERATORS,
            tags=tags,
            rng=random.Random(7),
        )

    def test_tags_are_stored_on_puzzle_task(self):
        tags = {"tier": "A", "operator_set": "withhold", "num_transforms": 3}
        task = self._make_task(tags)
        self.assertEqual(task.tags, tags)

    def test_to_jsonl_record_includes_tags_when_non_empty(self):
        tags = {"tier": "B"}
        task = self._make_task(tags)
        rec = task.to_jsonl_record()
        self.assertIn("tags", rec)
        self.assertEqual(rec["tags"], tags)
        # Everything else still matches the wire schema.
        self.assertEqual(set(rec.keys()) - {"tags"}, {"contents", "references"})

    def test_to_jsonl_record_omits_tags_when_empty(self):
        """Legacy shape (no tags) must be preserved when caller passes none
        or an empty dict. Guards against downstream tools that check for
        exact top-level key set."""
        for tags_input in [None, {}]:
            with self.subTest(tags_input=tags_input):
                task = self._make_task(tags_input)
                rec = task.to_jsonl_record()
                self.assertNotIn("tags", rec)
                self.assertEqual(set(rec.keys()), {"contents", "references"})

    def test_parse_word_puzzle_item_passes_tags_through(self):
        """The pipeline's schema-transform function preserves tags."""
        from example_recipe.word_puzzle import parse_word_puzzle_item

        tags = {"tier": "C", "num_transformations": 12}
        task = self._make_task(tags)
        rec = task.to_jsonl_record()
        parsed = parse_word_puzzle_item(rec)
        # parse_word_puzzle_item returns a WordPuzzleTaskData dataclass;
        # ``tags`` is a plain attribute.
        self.assertEqual(parsed.tags, tags)
        self.assertEqual(parsed.question, task.prompt)
        self.assertEqual(parsed.reference, task.expected_answer)

    def test_parse_word_puzzle_item_backward_compat_no_tags(self):
        """Records without a ``tags`` key (the shape of the checked-in
        legacy JSONLs) parse successfully with ``tags=None``, matching
        the ``WordPuzzleTaskData`` default."""
        from example_recipe.word_puzzle import parse_word_puzzle_item

        legacy_record = {
            "contents": [{"role": "user", "parts": [{"text": "Q?"}]}],
            "references": {"reference": "A"},
        }
        parsed = parse_word_puzzle_item(legacy_record)
        self.assertEqual(parsed.question, "Q?")
        self.assertEqual(parsed.reference, "A")
        self.assertIsNone(parsed.tags)

    def test_tags_are_deep_copied_not_aliased(self):
        """Mutating the caller's tags dict after generation must not
        affect the stored task or its emitted record (guards against
        the classic shared-mutable-state footgun)."""
        tags = {"tier": "A"}
        task = self._make_task(tags)
        rec = task.to_jsonl_record()
        tags["tier"] = "MUTATED"
        tags["new_key"] = "leaked"
        self.assertEqual(task.tags, {"tier": "A"})
        self.assertEqual(rec["tags"], {"tier": "A"})


# ---------------------------------------------------------------------------
# 6. Reproducibility across FRESH process boundaries (regression guard)
# ---------------------------------------------------------------------------
class TestCrossProcessReproducibility(unittest.TestCase):
    """The strongest guarantee: `seed=X` in one process must equal `seed=X`
    in a subprocess with a fresh interpreter state. Guards against hidden
    module-level state (e.g. an import that mutates the process-global
    ``random`` seed based on time.time())."""

    def test_subprocess_with_same_seed_produces_identical_first_task(self):
        import subprocess
        import sys

        # Locate the src/ dir (repo root's src) so the subprocess can
        # import `example_recipe`. pytest sets this via pyproject.toml's
        # pythonpath, but a bare subprocess doesn't inherit it.
        # Locate src/ from the puzzle_dataset_gen module file (a real
        # module with __file__, unlike the namespace package
        # `example_recipe` which may not).
        mod_file = pdg.__file__
        assert mod_file is not None, "puzzle_dataset_gen must have __file__"
        # Layout: <src_root>/example_recipe/word_puzzle/dataset/puzzle_dataset_gen.py
        src_root = os.path.dirname(  # -> <src_root>
            os.path.dirname(  # -> example_recipe/
                os.path.dirname(  # -> example_recipe/word_puzzle/
                    os.path.dirname(mod_file)  # -> example_recipe/word_puzzle/dataset/
                )
            )
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = src_root + os.pathsep + env.get("PYTHONPATH", "")

        code = (
            "import random\n"
            "from example_recipe.word_puzzle.dataset import puzzle_dataset_gen as pdg\n"
            "words = " + repr(_WORDS) + "\n"
            "task = pdg.generate_one_task(\n"
            "    num_transformations=3, num_few_shot_examples=4,\n"
            "    word_candidates=words, operators=pdg.EXTENDED_OPERATORS,\n"
            "    rng=random.Random(2026),\n"
            ")\n"
            "print(task.original_word, task.template_id,\n"
            "      task.encrypted_word, task.expected_answer)\n"
            "print(task.rule)\n"
        )
        r1 = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        r2 = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        self.assertEqual(r1.stdout, r2.stdout)
        # And confirm the seed=X in-process reproduces the subprocess
        # output too (no hidden interpreter-state dependence).
        in_proc = pdg.generate_one_task(
            num_transformations=3,
            num_few_shot_examples=4,
            word_candidates=_WORDS,
            operators=pdg.EXTENDED_OPERATORS,
            rng=random.Random(2026),
        )
        expected = (
            f"{in_proc.original_word} {in_proc.template_id} "
            f"{in_proc.encrypted_word} {in_proc.expected_answer}\n"
            f"{in_proc.rule}\n"
        )
        self.assertEqual(r1.stdout, expected)


class TestIncludeAnswerFormat(unittest.TestCase):
    """Pins the ``include_answer_format`` knob on ``generate_prompt`` and
    ``generate_one_task``.

    Background: every prompt template ends with an "answer format"
    instruction:

        Put your final answer in the following format:
        $\\text{your answer}$

    Written for the direct-answer eval mode where the model's completion
    is text-parsed and ``$\\text{...}$`` delimits the answer. In agent
    mode (see ``example_recipe/word_puzzle/agent_solver/``) the answer
    comes back as a structured ``submit(answer=...)`` tool call, so the
    LaTeX wrapper becomes part of the string the grader compares against
    and causes exact-match failures. Agent-mode callers pass
    ``include_answer_format=False`` to omit that trailer entirely;
    direct-eval callers keep the default ``True`` for byte-identical
    legacy behavior.

    Contracts pinned:
      * default ``True`` keeps every template byte-identical to legacy;
      * ``False`` strips the LaTeX line AND the "Put your final answer"
        line from every template, across BOTH the ``explain_encryption``
        on/off branches and the special template 7 branch;
      * the rest of the prompt (persona, examples block, main question,
        explanation-request line if present) is byte-identical between
        the two modes.
    """

    _LATEX_LINE = "$\\text{your answer}$"
    _PUT_YOUR_FINAL_ANSWER = "Put your final answer in the following format:"

    def _all_prompts(
        self, *, include_answer_format: bool, explain: bool
    ) -> dict[int, str]:
        """Render one prompt per template id (1..10) with the given options.

        Uses a fixed (word, examples, rng) so any change in the trailer
        semantics is the only variable across calls.
        """
        examples = [("cat", "dog"), ("run", "swim")]
        prompts: dict[int, str] = {}
        for tid in range(1, 11):
            prompt, got_tid = pdg.generate_prompt(
                examples=examples,
                task_word="ant",
                force_template_id=tid,
                explain_encryption=explain,
                reverse_task=False,
                include_answer_format=include_answer_format,
                rng=random.Random(0),
            )
            self.assertEqual(got_tid, tid)
            prompts[tid] = prompt
        return prompts

    def test_default_includes_answer_format_for_all_templates(self):
        """Default (no kwarg) must include the LaTeX trailer on every
        template. Backwards-compat guard: any regression here would
        change every checked-in JSONL byte."""
        prompts = self._all_prompts(include_answer_format=True, explain=False)
        for tid, p in prompts.items():
            with self.subTest(template_id=tid):
                self.assertIn(self._PUT_YOUR_FINAL_ANSWER, p)
                self.assertIn(self._LATEX_LINE, p)

    def test_include_answer_format_false_strips_trailer_from_all_templates(self):
        """``include_answer_format=False`` must remove BOTH the "Put your
        final answer" line and the ``$\\text{...}$`` line from every
        template, in both ``explain_encryption`` branches."""
        for explain in [False, True]:
            with self.subTest(explain_encryption=explain):
                prompts = self._all_prompts(
                    include_answer_format=False, explain=explain
                )
                for tid, p in prompts.items():
                    with self.subTest(template_id=tid):
                        self.assertNotIn(
                            self._PUT_YOUR_FINAL_ANSWER,
                            p,
                            f"tid={tid} explain={explain}: {p!r}",
                        )
                        self.assertNotIn(
                            self._LATEX_LINE,
                            p,
                            f"tid={tid} explain={explain}: {p!r}",
                        )

    def test_include_answer_format_false_preserves_everything_else(self):
        """The prompt body (persona, examples, main question, and the
        explanation-request line when enabled) must be byte-identical
        between the True and False variants -- only the trailer differs.
        """
        for explain in [False, True]:
            with self.subTest(explain_encryption=explain):
                with_fmt = self._all_prompts(
                    include_answer_format=True, explain=explain
                )
                without_fmt = self._all_prompts(
                    include_answer_format=False, explain=explain
                )
                for tid in range(1, 11):
                    # The "with-format" prompt is the "without-format"
                    # prompt PLUS the trailer. So strip the trailer from
                    # the with_fmt version and compare.
                    stripped = with_fmt[tid]
                    for line in (
                        self._LATEX_LINE,
                        self._PUT_YOUR_FINAL_ANSWER,
                    ):
                        idx = stripped.find(line)
                        if idx >= 0:
                            stripped = stripped[:idx]
                    # Both variants should now share a common prefix
                    # (whitespace at the tail may differ; compare after
                    # right-strip).
                    self.assertEqual(
                        stripped.rstrip(),
                        without_fmt[tid].rstrip(),
                        f"tid={tid} explain={explain}: body diverged",
                    )

    def test_generate_one_task_forwards_include_answer_format(self):
        """``generate_one_task(include_answer_format=False)`` must
        produce a PuzzleTask whose ``prompt`` field has the trailer
        stripped. This is the end-to-end wire for the agent-mode
        eval-dataset generator."""
        task = pdg.generate_one_task(
            num_transformations=3,
            num_few_shot_examples=3,
            word_candidates=_WORDS,
            operators=pdg.EXTENDED_OPERATORS,
            include_answer_format=False,
            rng=random.Random(42),
        )
        self.assertNotIn(self._PUT_YOUR_FINAL_ANSWER, task.prompt)
        self.assertNotIn(self._LATEX_LINE, task.prompt)

    def test_generate_one_task_default_still_includes_trailer(self):
        """Regression guard: default kwarg on ``generate_one_task`` still
        yields prompts with the trailer, so pre-existing callers /
        checked-in JSONLs are byte-compatible."""
        task = pdg.generate_one_task(
            num_transformations=3,
            num_few_shot_examples=3,
            word_candidates=_WORDS,
            operators=pdg.EXTENDED_OPERATORS,
            rng=random.Random(42),
        )
        self.assertIn(self._PUT_YOUR_FINAL_ANSWER, task.prompt)
        self.assertIn(self._LATEX_LINE, task.prompt)


class TestIncludeRetryMetadata(unittest.TestCase):
    """Pins the ``include_retry_metadata`` knob on ``generate_one_task``
    and its ``to_jsonl_record`` output.

    Background: the agent-mode "scored feedback" retry feature lets the
    model call ``submit`` multiple times per task; on each wrong answer,
    the runner injects 2 FRESH (input, output) demonstration pairs
    generated by re-applying the puzzle's rule to unseen words. That
    requires the runner to know:
      * the exact ``rule`` used to build this task (so it can re-encode)
      * a pool of ``extra_word_pool`` words disjoint from the demo /
        task words (so it can pick fresh inputs)
      * the ``reverse_task`` direction (so it knows which side of the
        pair the model would receive as "input")

    The ``include_retry_metadata`` knob (default False) controls whether
    ``to_jsonl_record()`` emits a top-level ``retry_metadata`` key
    containing these three fields. Direct-answer / training pipelines
    keep the default (no leak of the ground-truth rule). Agent-mode
    eval generation opts in.

    Contracts pinned:
      * default False keeps every ``to_jsonl_record()`` byte-identical
        to legacy;
      * True emits a well-shaped ``retry_metadata`` block with the
        required fields;
      * word pool is disjoint from demo + task words (no cheat);
      * word pool has enough words for the maximum retry count
        (at least 6 = 3 retries * 2 pairs, plus a safety margin);
      * word pool is deterministic per rng.
    """

    _MIN_POOL_SIZE = 6  # 3 retries * 2 pairs per retry

    def _make_task(self, *, include_retry_metadata: bool, rng_seed: int = 42):
        """Build one task via ``generate_one_task`` with the given knob."""
        return pdg.generate_one_task(
            num_transformations=3,
            num_few_shot_examples=3,
            word_candidates=_WORDS,
            operators=pdg.EXTENDED_OPERATORS,
            include_retry_metadata=include_retry_metadata,
            rng=random.Random(rng_seed),
        )

    def test_default_omits_retry_metadata_from_jsonl(self):
        """Default (no kwarg) must NOT emit ``retry_metadata`` in the
        JSONL record, matching legacy behavior. This is load-bearing:
        the checked-in direct-answer JSONL must not accidentally start
        containing rule metadata (which would leak ground truth to
        anyone reading the file).
        """
        task = pdg.generate_one_task(
            num_transformations=3,
            num_few_shot_examples=3,
            word_candidates=_WORDS,
            operators=pdg.EXTENDED_OPERATORS,
            rng=random.Random(42),
        )
        rec = task.to_jsonl_record()
        self.assertNotIn("retry_metadata", rec)

    def test_include_retry_metadata_emits_block_with_required_fields(self):
        """``include_retry_metadata=True`` puts a well-shaped block in
        the JSONL record. Required keys: ``rule``, ``extra_word_pool``,
        ``reverse_task``.
        """
        task = self._make_task(include_retry_metadata=True)
        rec = task.to_jsonl_record()
        self.assertIn("retry_metadata", rec)
        rm = rec["retry_metadata"]
        for k in ("rule", "extra_word_pool", "reverse_task"):
            self.assertIn(k, rm, f"missing key: {k}")

    def test_retry_metadata_rule_matches_task_rule(self):
        """The ``rule`` field in retry_metadata IS the task's own rule
        (same op_name / param pairs, in the same order). Runner
        re-applies this rule to new words at wrong-submit time; if it
        drifts from the actual task rule the extra examples would be
        wrong.
        """
        task = self._make_task(include_retry_metadata=True)
        rec = task.to_jsonl_record()
        # Rule is serialized as list-of-lists (json compatible), so
        # compare via list conversion. Each entry is [op_name, param].
        rule_from_rec = rec["retry_metadata"]["rule"]
        self.assertEqual(len(rule_from_rec), len(task.rule))
        for (op_rec, param_rec), (op_task, param_task) in zip(rule_from_rec, task.rule):
            self.assertEqual(op_rec, op_task)
            self.assertEqual(param_rec, param_task)

    def test_retry_metadata_word_pool_is_disjoint_from_task_words(self):
        """The extra_word_pool MUST NOT overlap with the puzzle's demo
        words OR the task word. Overlap would leak the answer (for the
        task word) or duplicate an existing demo (for demo words),
        making the "here are 2 MORE examples" feedback either trivially
        answer-revealing or unhelpful.
        """
        task = self._make_task(include_retry_metadata=True)
        pool = set(task.to_jsonl_record()["retry_metadata"]["extra_word_pool"])
        # Demo words are the .0 slot of each few-shot pair; task word
        # is the original.
        demo_words = {orig for orig, _enc in task.few_shot_examples}
        forbidden = demo_words | {task.original_word}
        overlap = pool & forbidden
        self.assertFalse(
            overlap,
            f"pool overlaps with task/demo words: {overlap}",
        )

    def test_retry_metadata_word_pool_has_min_size(self):
        """Pool must have enough words for the maximum retry scenario
        (3 retries * 2 pairs = 6 words minimum). We use 20 as the
        target size for safety margin (RNG picks without exhausting).
        """
        task = self._make_task(include_retry_metadata=True)
        pool = task.to_jsonl_record()["retry_metadata"]["extra_word_pool"]
        self.assertGreaterEqual(
            len(pool),
            self._MIN_POOL_SIZE,
            f"pool has only {len(pool)} words, need >= {self._MIN_POOL_SIZE}",
        )

    def test_retry_metadata_reverse_task_matches_task(self):
        """The reverse_task flag in retry_metadata must match the
        task's own reverse_task (needed by the runner to know which
        side of the pair is the model's 'input')."""
        for reverse in (False, True):
            with self.subTest(reverse_task=reverse):
                task = pdg.generate_one_task(
                    num_transformations=3,
                    num_few_shot_examples=3,
                    word_candidates=_WORDS,
                    operators=pdg.EXTENDED_OPERATORS,
                    reverse_task=reverse,
                    include_retry_metadata=True,
                    rng=random.Random(42),
                )
                rec = task.to_jsonl_record()
                self.assertEqual(
                    rec["retry_metadata"]["reverse_task"],
                    reverse,
                )

    def test_retry_metadata_is_deterministic_by_seed(self):
        """Same seed \u2192 same pool. Load-bearing: without determinism,
        regenerating the JSONL from the same seed would produce a
        different pool and break byte-comparability."""
        task_a = self._make_task(include_retry_metadata=True, rng_seed=42)
        task_b = self._make_task(include_retry_metadata=True, rng_seed=42)
        self.assertEqual(
            task_a.to_jsonl_record()["retry_metadata"]["extra_word_pool"],
            task_b.to_jsonl_record()["retry_metadata"]["extra_word_pool"],
        )


# ---------------------------------------------------------------------------
# Reversibility: apply_inverse_transform + check_rule_reversible + resample loop
# ---------------------------------------------------------------------------
class TestReversibility(unittest.TestCase):
    """Every op in the v2 sets has a working inverse; generate_one_task only
    emits rules whose forward+inverse round-trips to the original word.

    Catches the two shipped bugs that prompted this machinery:
    (a) `_shift_char` used to map every non-letter to a single letter (via
        `find()==-1`), collapsing rules like [ascii_values_as_string, shift(7)]
        to all-one-letter output;
    (b) `_interleave_split` dropped the middle char on odd-length input,
        making the op non-injective.
    """

    def test_shift_passes_non_letters_through(self):
        """Regression test for `_shift_char`: digits/hyphens must not
        collapse to a single letter."""
        # ascii_values_as_string output: only digits + hyphens => unchanged.
        self.assertEqual(pdg._shift_char("101-99-111", 7), "101-99-111")
        # hex_encoding output: letters shift, digits stay.
        # 'f'(5)+3->'i'; 'd'(3)+3->'g'; 'm'(12)+3->'p'; 'e'(4)+3->'h'; digits unchanged.
        self.assertEqual(pdg._shift_char("6f63746d65", 3), "6i63746g65")
        # Letters shift normally.
        self.assertEqual(pdg._shift_char("abc", 7), "hij")

    def test_interleave_is_bijective_on_odd_lengths(self):
        """Regression test for `_interleave_split`: odd-length input must
        preserve every source character."""
        for w in ["abc", "abcde", "abcdefg", "abcdefghi"]:
            out = pdg._interleave_split(w)
            self.assertEqual(len(out), len(w), w)
            self.assertEqual(sorted(out), sorted(w), w)

    def test_every_v2_op_has_a_working_inverse(self):
        """Each op in extended_v2 + withhold_v2: apply forward then inverse
        recovers the original."""
        r = random.Random(2026)
        words = ["abcde", "programming", "python", "cryptography", "hello"]
        parametric_samples = {
            "shift": [r.randint(1, 51) for _ in range(3)],
            "rotate": [r.randint(1, 4) for _ in range(3)],
            "mirror": [True, False],
        }
        all_ops = set(pdg.EXTENDED_OPERATORS_V2) | set(pdg.WITHHOLD_OPERATORS_V2)
        for op in sorted(all_ops):
            with self.subTest(op=op):
                params = parametric_samples.get(op, [None])
                for param in params:
                    for w in words:
                        rule = [[op, param]]
                        try:
                            fwd = pdg.apply_transform(w, rule)
                            inv = pdg.apply_inverse_transform(fwd, rule)
                        except Exception as e:
                            self.fail(f"{op}({param}) on {w!r}: {e}")
                        self.assertEqual(
                            inv, w, f"{op}({param}) on {w!r}: fwd={fwd!r} inv={inv!r}"
                        )

    def test_check_rule_reversible_catches_bug_shape(self):
        """The pre-fix pathological composition — shift after a
        charset-changing op — round-trips cleanly on the FIXED code, since
        `_shift_char` now leaves non-letters alone."""
        rule = [
            ["ascii_values_as_string", None],
            ["shift", 7],
        ]
        self.assertTrue(pdg.check_rule_reversible("cohoes", rule))

    def test_check_rule_reversible_returns_false_on_non_invertible(self):
        """Sanity check: a non-invertible rule (fake op that drops info)
        should be reported as not reversible."""
        # Register a fake destructive op just for this test.
        pdg.TRANSFORMATION_FUNCTIONS["_test_drop_last"] = lambda s: s[:-1] if s else s
        try:
            self.assertFalse(
                pdg.check_rule_reversible("abc", [["_test_drop_last", None]])
            )
        finally:
            del pdg.TRANSFORMATION_FUNCTIONS["_test_drop_last"]

    def test_generate_one_task_only_emits_reversible_rules(self):
        """Round-trip on many generated tasks: every emitted rule +
        (task_word, demo_words) must round-trip cleanly."""
        rng = random.Random(2026)
        for depth in [1, 3, 5, 7]:
            for opset in [pdg.EXTENDED_OPERATORS_V2, pdg.WITHHOLD_OPERATORS_V2]:
                for _ in range(10):
                    task = pdg.generate_one_task(
                        num_transformations=depth,
                        num_few_shot_examples=3,
                        word_candidates=_WORDS,
                        operators=opset,
                        max_expanding_ops=1 if depth >= 3 else None,
                        rng=rng,
                    )
                    # Task word round-trips.
                    self.assertTrue(
                        pdg.check_rule_reversible(task.original_word, task.rule),
                        f"task_word {task.original_word!r} not reversible with rule {task.rule!r}",
                    )
                    # Every demo also round-trips.
                    for demo, _ in task.few_shot_examples:
                        self.assertTrue(
                            pdg.check_rule_reversible(demo, task.rule),
                            f"demo {demo!r} not reversible with rule {task.rule!r}",
                        )


class TrainingOperatorsV211Test(unittest.TestCase):
    """The v2.1.1 TRAINING pool (`extended_v2_1_1`).

    v2.1.1's 20-operator `withhold_v2_1_1` pool is a permanent held-out eval
    set, so training data must never draw from it. These tests pin the two
    properties that keep that true: the training pool is disjoint from the
    withhold pool, and every new operator is individually invertible.
    """

    WORDS = ["abcde", "programming", "python", "cryptography", "hello", "zebra"]

    def test_training_registry_has_twelve_ops(self):
        self.assertEqual(len(pdg.TRAINING_TRANSFORMATION_FUNCTIONS_V2_1_1), 12)

    def test_training_pool_is_disjoint_from_withhold_v2_1_1(self):
        """The load-bearing invariant: no training operator may appear in the
        held-out eval pool, or v2.1.1 stops measuring transfer."""
        train = set(pdg.OPERATOR_SETS["extended_v2_1_1"])
        withhold = set(pdg.OPERATOR_SETS["withhold_v2_1_1"])
        self.assertEqual(train & withhold, set())

    def test_training_pool_extends_extended_v2(self):
        """extended_v2_1_1 = the 14 extended_v2 ops + the 12 new ones."""
        self.assertEqual(
            set(pdg.OPERATOR_SETS["extended_v2_1_1"]),
            set(pdg.EXTENDED_OPERATORS_V2)
            | set(pdg.TRAINING_TRANSFORMATION_FUNCTIONS_V2_1_1),
        )

    def test_pre_existing_pools_are_unchanged(self):
        """Adding the training pool must not perturb any shipped pool, or
        previously generated datasets stop reproducing."""
        self.assertEqual(len(pdg.OPERATOR_SETS["withhold_v2_1_1"]), 20)
        self.assertEqual(len(pdg.OPERATOR_SETS["withhold_v2"]), 9)
        self.assertEqual(len(pdg.OPERATOR_SETS["extended_v2"]), 14)

    def test_every_new_op_has_a_working_inverse(self):
        for op in sorted(pdg.TRAINING_TRANSFORMATION_FUNCTIONS_V2_1_1):
            with self.subTest(op=op):
                for w in self.WORDS:
                    rule = [[op, None]]
                    fwd = pdg.apply_transform(w, rule)
                    self.assertEqual(pdg.apply_inverse_transform(fwd, rule), w)

    def test_every_new_op_is_length_preserving(self):
        """All 12 are permutations or in-place letter maps, so a deep rule
        cannot blow up prompt length the way the expanding encodings do."""
        for op in sorted(pdg.TRAINING_TRANSFORMATION_FUNCTIONS_V2_1_1):
            with self.subTest(op=op):
                for w in self.WORDS:
                    self.assertEqual(len(pdg.apply_transform(w, [[op, None]])), len(w))

    def test_no_new_op_is_expanding(self):
        self.assertEqual(
            set(pdg.TRAINING_TRANSFORMATION_FUNCTIONS_V2_1_1) & pdg.EXPANDING_OPERATORS,
            set(),
        )

    def test_content_dependent_ops_always_act(self):
        """A content-dependent op must not silently be the identity on a large
        share of words: at depth 1 that yields a task teaching nothing.
        `branch_on_vowel_parity` exists precisely because the naive "reverse
        only when the vowel count is odd" was a no-op on 54% of words."""
        words = [w for w in self.WORDS if len(w) > 2]
        for op in ("branch_on_vowel_parity", "rotate_by_vowel_count"):
            with self.subTest(op=op):
                unchanged = sum(
                    1 for w in words if pdg.apply_transform(w, [[op, None]]) == w
                )
                self.assertLess(unchanged, len(words))

    def test_composed_training_rules_round_trip(self):
        rng = random.Random(2026)
        ops = sorted(pdg.TRAINING_TRANSFORMATION_FUNCTIONS_V2_1_1)
        for depth in (2, 5, 8):
            with self.subTest(depth=depth):
                for _ in range(20):
                    rule = [[rng.choice(ops), None] for _ in range(depth)]
                    w = rng.choice(self.WORDS)
                    self.assertTrue(pdg.check_rule_reversible(w, rule), (w, rule))

    def test_exclude_same_length_demos_is_off_by_default(self):
        """Default behaviour must not change, or v2.0/v2.1 stop reproducing."""
        kwargs = dict(
            num_transformations=2,
            num_few_shot_examples=4,
            word_candidates=_WORDS,
            operators=pdg.EXTENDED_OPERATORS_V2,
        )
        a = pdg.generate_one_task(rng=random.Random(5), **kwargs)
        b = pdg.generate_one_task(rng=random.Random(5), **kwargs)
        self.assertEqual(a.few_shot_examples, b.few_shot_examples)
        self.assertEqual(a.original_word, b.original_word)

    def test_exclude_same_length_demos_drops_same_length_demos(self):
        """With the flag on, no demo may share the task word's length.

        A same-length demo lets the solver skip the rule entirely and copy
        the demo's transformation positionally -- measured on a 5-op chain
        solved by copying one demo (mueang -> aunemg).
        """
        words = [
            "alpha",
            "bravos",
            "charlie",
            "deltaxy",
            "echoing",
            "foxtrots",
            "golf",
            "hotelier",
            "indiaxx",
            "juliette",
            "kilogram",
            "limaxxxxx",
        ]
        for seed in range(8):
            task = pdg.generate_one_task(
                num_transformations=2,
                num_few_shot_examples=3,
                word_candidates=words,
                operators=pdg.EXTENDED_OPERATORS_V2,
                exclude_same_length_demos=True,
                rng=random.Random(seed),
            )
            with self.subTest(seed=seed):
                n = len(task.original_word)
                for demo, _ in task.few_shot_examples:
                    self.assertNotEqual(len(demo), n, (task.original_word, demo))

    def test_exclude_same_length_demos_still_fills_all_demo_slots(self):
        words = [
            "alpha",
            "bravos",
            "charlie",
            "deltaxy",
            "echoing",
            "foxtrots",
            "golf",
            "hotelier",
            "indiaxx",
            "juliette",
            "kilogram",
            "limaxxxxx",
        ]
        task = pdg.generate_one_task(
            num_transformations=2,
            num_few_shot_examples=3,
            word_candidates=words,
            operators=pdg.EXTENDED_OPERATORS_V2,
            exclude_same_length_demos=True,
            rng=random.Random(1),
        )
        self.assertEqual(len(task.few_shot_examples), 3)

    def test_generate_random_rule_accepts_the_new_pool(self):
        rng = random.Random(7)
        rule = pdg.generate_random_rule(
            4, max_rotate=3, operators=pdg.OPERATOR_SETS["extended_v2_1_1"], rng=rng
        )
        self.assertEqual(len(rule), 4)
        for op, _ in rule:
            self.assertIn(op, pdg.OPERATOR_SETS["extended_v2_1_1"])


class FewShotJitterTest(unittest.TestCase):
    """Opt-in per-task jitter on the few-shot demo count.

    Without it, ``num_few_shot`` is a fixed function of depth (``depth + 2``),
    so the two co-vary perfectly and "depth 8 is hard" cannot be separated
    from "depth 8 shows 10 worked examples".
    """

    KW = dict(
        num_transformations=2,
        num_few_shot_examples=5,
        word_candidates=_WORDS,
        operators=pdg.EXTENDED_OPERATORS_V2,
    )

    def test_default_is_unjittered_and_byte_identical(self):
        """Default must not move: v2.0/v2.1 have to keep reproducing."""
        for seed in range(4):
            a = pdg.generate_one_task(rng=random.Random(seed), **self.KW)
            b = pdg.generate_one_task(rng=random.Random(seed), **self.KW)
            with self.subTest(seed=seed):
                self.assertEqual(len(a.few_shot_examples), 5)
                self.assertEqual(a.few_shot_examples, b.few_shot_examples)
                self.assertEqual(a.original_word, b.original_word)

    def test_jitter_varies_the_count_within_the_requested_range(self):
        counts = {
            len(
                pdg.generate_one_task(
                    rng=random.Random(seed), few_shot_jitter=(-1, 3), **self.KW
                ).few_shot_examples
            )
            for seed in range(40)
        }
        self.assertGreater(len(counts), 1, "jitter produced a constant count")
        self.assertTrue(counts <= {4, 5, 6, 7, 8}, counts)

    def test_jitter_never_drops_below_the_solvability_floor(self):
        """A task with too few demos is unsolvable as posed -- there is not
        enough evidence to infer the rule. depth 1 with jitter -1 would
        otherwise reach 0 demos."""
        for seed in range(40):
            task = pdg.generate_one_task(
                num_transformations=1,
                num_few_shot_examples=1,
                word_candidates=_WORDS,
                operators=pdg.EXTENDED_OPERATORS_V2,
                few_shot_jitter=(-1, 3),
                rng=random.Random(seed),
            )
            with self.subTest(seed=seed):
                self.assertGreaterEqual(
                    len(task.few_shot_examples), pdg.MIN_FEW_SHOT_EXAMPLES
                )

    def test_jittered_task_reports_its_actual_count(self):
        """``num_few_shot_examples`` on the task -- and therefore the emitted
        tag -- must be the REALISED count, not the requested base, or every
        downstream per-few-shot analysis silently reads the wrong number."""
        for seed in range(25):
            task = pdg.generate_one_task(
                rng=random.Random(seed), few_shot_jitter=(-1, 3), **self.KW
            )
            with self.subTest(seed=seed):
                self.assertEqual(
                    task.num_few_shot_examples, len(task.few_shot_examples)
                )

    def test_jitter_is_deterministic_by_seed(self):
        a = [
            len(
                pdg.generate_one_task(
                    rng=random.Random(11), few_shot_jitter=(-1, 3), **self.KW
                ).few_shot_examples
            )
            for _ in range(5)
        ]
        b = [
            len(
                pdg.generate_one_task(
                    rng=random.Random(11), few_shot_jitter=(-1, 3), **self.KW
                ).few_shot_examples
            )
            for _ in range(5)
        ]
        self.assertEqual(a, b)

    def test_emitted_tag_reports_the_realised_count(self):
        """The JSONL ``tags.num_few_shot`` is what every downstream analysis
        reads. Callers set it from the REQUESTED base, so with jitter on it
        must be corrected to the realised count or the tag silently lies."""
        seen = set()
        for seed in range(30):
            task = pdg.generate_one_task(
                rng=random.Random(seed),
                few_shot_jitter=(-1, 3),
                tags={"depth": 2, "num_few_shot": 5},
                **self.KW,
            )
            seen.add(task.tags["num_few_shot"])
            with self.subTest(seed=seed):
                self.assertEqual(task.tags["num_few_shot"], len(task.few_shot_examples))
        self.assertGreater(len(seen), 1, "tag never varied -- jitter not applied")

    def test_tag_is_untouched_when_caller_did_not_supply_one(self):
        task = pdg.generate_one_task(rng=random.Random(1), tags={"depth": 2}, **self.KW)
        self.assertNotIn("num_few_shot", task.tags)

    def test_generate_dataset_threads_the_jitter_through(self):
        tasks = pdg.generate_dataset(
            num_examples=25,
            num_transformations=2,
            num_few_shot_examples=5,
            word_candidates=_WORDS,
            operators=pdg.EXTENDED_OPERATORS_V2,
            flip_reverse_task=False,
            few_shot_jitter=(-1, 3),
            rng=random.Random(3),
        )
        counts = {len(t.few_shot_examples) for t in tasks}
        self.assertGreater(len(counts), 1)
        for task in tasks:
            self.assertEqual(task.num_few_shot_examples, len(task.few_shot_examples))


class LoadWordCandidatesTest(unittest.TestCase):
    def test_default_source_is_the_bundled_allowlist(self):
        words = pdg.load_word_candidates(min_length=5)
        with open(pdg.DEFAULT_WORD_LIST, encoding="utf-8") as f:
            bundled = {line.strip() for line in f}
        self.assertGreater(len(words), 100_000)
        self.assertTrue(set(words) <= bundled)
        self.assertTrue(all(len(w) >= 5 for w in words))

    def test_bundled_allowlist_is_sorted_unique_lowercase_ascii(self):
        with open(pdg.DEFAULT_WORD_LIST, encoding="utf-8") as f:
            lines = f.read().splitlines()
        self.assertEqual(lines, sorted(set(lines)))
        bad = [w for w in lines if not (w.isascii() and w.isalpha() and w.islower())]
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
