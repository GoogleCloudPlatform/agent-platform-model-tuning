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

"""Generate the tiered withhold-operator eval dataset.

Produces ``combined_eval_word_transformation_withhold_v1.jsonl``. Tests
generalization to operators NOT seen during training (base64, rot13,
atbash, binary encoding, ...).

Difficulty tiers (150 rows total, 50 per tier):

    Tier | num_transformations | few_shot | max_expanding_ops | rows
    -----+---------------------+----------+-------------------+-----
      A  | 3-4  sampled        | 3        | unconstrained     |  50
      B  | 5-8  sampled        | 5        | 1                 |  50
      C  | 9-15 sampled        | 8        | 1                 |  50

All tiers draw from the whole `withhold` pool -- 15 operators today
(it was 12 when the checked-in v1 bytes were generated). Tiers B and C
apply ``max_expanding_ops=1`` (see
``puzzle_dataset_gen.EXPANDING_OPERATORS``) to prevent multiplicative
prompt-length blowup from composing multiple hex/base64/binary
encodings.

Every row carries a ``tags`` dict passed through by
``parse_word_puzzle_item`` for per-slice breakdowns:

    {"tier": "A"|"B"|"C",
     "num_transformations": <int>,
     "num_few_shot": <int>,
     "operator_set": "withhold",
     "max_expanding_ops": null | 1,
     "reverse_task": <bool>}

Deterministic given ``--seed`` (default 42).

Usage:
    PYTHONPATH=src uv run python -m example_recipe.word_puzzle.dataset.gen_withhold_eval \\
        --output src/example_recipe/word_puzzle/dataset/combined_eval_word_transformation_withhold_v1.jsonl \\
        --seed 42
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from typing import Optional, Sequence

from example_recipe.word_puzzle.dataset import puzzle_dataset_gen as pdg

logger = logging.getLogger(__name__)


# Tier config: (name, transforms_lo, transforms_hi, num_few_shot, rows,
#               max_expanding_ops). None = unconstrained. Lo/hi inclusive.
TIERS: list[tuple[str, int, int, int, int, Optional[int]]] = [
    ("A", 3, 4, 3, 50, None),
    ("B", 5, 8, 5, 50, 1),
    ("C", 9, 15, 8, 50, 1),
]

# Depth schedule for v2 factorial design: fixed depth per slice, few-shot
# scales as depth + 2, max_expanding_ops capped at 1 for depth >= 3.
# (depth, num_few_shot, num_rows, max_expanding_ops)
DEPTH_SCHEDULE_V2: list[tuple[int, int, int, int | None]] = [
    (1, 3, 20, None),
    (2, 4, 20, None),
    (3, 5, 20, 1),
    (4, 6, 20, 1),
    (5, 7, 20, 1),
    (6, 8, 10, 1),
    (7, 9, 10, 1),
    (8, 10, 10, 1),
]

# Extended v2 schedule: same as DEPTH_SCHEDULE_V2 with depths 9-10 appended.
# Appending at the end (rather than interleaving) keeps rows for depths 1-8
# byte-identical to the v2 schedule under the same seed — the RNG is threaded
# through the schedule in order.
DEPTH_SCHEDULE_V2_EXTENDED: list[tuple[int, int, int, int | None]] = (
    DEPTH_SCHEDULE_V2 + [(9, 11, 10, 1), (10, 12, 10, 1)]
)


def generate_tiered_eval(
    word_candidates: Sequence[str],
    *,
    seed: int = 42,
    tiers: list[tuple[str, int, int, int, int, Optional[int]]] = TIERS,
    explain_encryption: bool = True,
    include_answer_format: bool = True,
    include_retry_metadata: bool = False,
    version: str = "v1",
) -> list[pdg.PuzzleTask]:
    """Generate the full tiered eval set (150 tasks by default).

    Args:
      word_candidates: pool of words to sample from.
      seed: RNG seed for reproducibility.
      tiers: tier configuration list.
      explain_encryption: ask the model to describe the rule before answering.
      include_answer_format: if True, each puzzle body ends with the
        ``$\\text{your answer}$`` trailer (direct-answer eval). Agent-mode
        eval should pass False so the LaTeX wrapper doesn't end up inside
        the submitted answer.
      include_retry_metadata: if True, emit a top-level ``retry_metadata``
        block per row so the agent runner can generate fresh demonstration
        pairs on wrong-submit. Off by default to avoid leaking the
        ground-truth rule into the direct-answer JSONL.
      version: ``"v1"`` uses the encoding-heavy ``withhold`` pool;
        ``"v2"`` uses the 7 property-matched withhold ops.
    """
    rng = random.Random(seed)
    ops = pdg.WITHHOLD_OPERATORS_V2 if version == "v2" else pdg.WITHHOLD_OPERATORS
    all_tasks: list[pdg.PuzzleTask] = []
    for tier_name, k_lo, k_hi, n_few_shot, n_rows, max_expanding in tiers:
        for i in range(n_rows):
            num_transformations = rng.randint(k_lo, k_hi)
            # Split each tier evenly across decrypt/encrypt directions.
            reverse_task = i >= (n_rows // 2)
            tags = {
                "tier": tier_name,
                "num_transformations": num_transformations,
                "num_few_shot": n_few_shot,
                "operator_set": "withhold_v2" if version == "v2" else "withhold",
                "max_expanding_ops": max_expanding,
                "reverse_task": reverse_task,
            }
            task = pdg.generate_one_task(
                num_transformations=num_transformations,
                num_few_shot_examples=n_few_shot,
                word_candidates=word_candidates,
                operators=ops,
                explain_encryption=explain_encryption,
                reverse_task=reverse_task,
                tags=tags,
                max_expanding_ops=max_expanding,
                rng=rng,
                include_answer_format=include_answer_format,
                include_retry_metadata=include_retry_metadata,
            )
            all_tasks.append(task)
    return all_tasks


def generate_depth_scheduled_eval(
    word_candidates: Sequence[str],
    *,
    seed: int = 42,
    schedule: list[tuple[int, int, int, int | None]] = DEPTH_SCHEDULE_V2,
    explain_encryption: bool = True,
    include_answer_format: bool = True,
    include_retry_metadata: bool = False,
    version: str = "v2",
) -> list[pdg.PuzzleTask]:
    """Generate eval tasks from an explicit depth schedule.

    Each entry in ``schedule`` is ``(depth, num_few_shot, num_rows,
    max_expanding_ops)``. Rows are tagged with depth and few-shot count
    for per-depth analysis.
    """
    rng = random.Random(seed)
    ops = pdg.WITHHOLD_OPERATORS_V2 if version == "v2" else pdg.WITHHOLD_OPERATORS
    all_tasks: list[pdg.PuzzleTask] = []
    for depth, n_few_shot, n_rows, max_expanding in schedule:
        for i in range(n_rows):
            reverse_task = i >= (n_rows // 2)
            tags = {
                "depth": depth,
                "num_transformations": depth,
                "num_few_shot": n_few_shot,
                "operator_set": "withhold_v2" if version == "v2" else "withhold",
                "max_expanding_ops": max_expanding,
                "reverse_task": reverse_task,
            }
            task = pdg.generate_one_task(
                num_transformations=depth,
                num_few_shot_examples=n_few_shot,
                word_candidates=word_candidates,
                operators=ops,
                explain_encryption=explain_encryption,
                reverse_task=reverse_task,
                tags=tags,
                max_expanding_ops=max_expanding,
                rng=rng,
                include_answer_format=include_answer_format,
                include_retry_metadata=include_retry_metadata,
            )
            all_tasks.append(task)
    return all_tasks


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generate the tiered withhold-operator eval dataset (150 rows, "
            "3 difficulty tiers, per-row tags)."
        )
    )
    p.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to output JSONL file (typically "
        "src/example_recipe/word_puzzle/dataset/"
        "combined_eval_word_transformation_withhold_v1.jsonl).",
    )
    p.add_argument(
        "--word-list",
        default=None,
        help="Path to a plain-text word list (one word per line). If "
        "omitted, uses the bundled vetted allowlist (word_allowlist.txt).",
    )
    p.add_argument(
        "--min-word-length",
        type=int,
        default=5,
        help="Drop candidates shorter than this. 5 matches the "
        "checked-in eval set's convention.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for reproducibility.",
    )
    p.add_argument(
        "--version",
        choices=["v1", "v2"],
        default="v1",
        help="Operator split version: v1 (the `withhold` pool, encoding-heavy) "
        "or v2 (7 property-matched ops). Defaults to v1.",
    )
    p.add_argument(
        "--mode",
        choices=["tiered", "depth-schedule"],
        default="tiered",
        help="Generation mode: 'tiered' (original A/B/C tiers, default) "
        "or 'depth-schedule' (v2 factorial: fixed depth per slice, "
        "few-shot = depth + 2).",
    )
    p.add_argument(
        "--schedule",
        choices=["v2", "v2-extended"],
        default="v2",
        help="Depth schedule to use with --mode depth-schedule: "
        "'v2' (default) = DEPTH_SCHEDULE_V2 (depths 1-8, 130 rows); "
        "'v2-extended' = DEPTH_SCHEDULE_V2_EXTENDED (depths 1-10, 150 rows). "
        "Rows for depths 1-8 are byte-identical between the two schedules "
        "under the same seed.",
    )
    p.add_argument(
        "--no-explain-encryption",
        dest="explain_encryption",
        action="store_false",
        default=True,
        help="Do NOT ask the model to describe the rule before answering.",
    )
    p.add_argument(
        "--no-answer-format",
        dest="include_answer_format",
        action="store_false",
        default=True,
        help=(
            "Omit the 'Put your final answer in the following format: "
            "$\\text{your answer}$' trailer from each puzzle body. Use "
            "for agent-mode (structured tool-call submit) eval, where "
            "the LaTeX wrapper would end up inside the submitted "
            "answer and cause exact-match failures."
        ),
    )
    p.add_argument(
        "--include-retry-metadata",
        dest="include_retry_metadata",
        action="store_true",
        default=False,
        help=(
            "Emit a top-level 'retry_metadata' block on every JSONL "
            "record with the puzzle's rule, a pool of 20 fresh words "
            "disjoint from the demo/task words, and the reverse_task "
            "direction. Consumed by the agent runner for graded-feedback "
            "retries. Off by default to avoid leaking the ground-truth "
            "rule into the eval JSONL; enable ONLY for agent-mode evals."
        ),
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    word_candidates = pdg.load_word_candidates(
        path=args.word_list, min_length=args.min_word_length
    )
    if not word_candidates:
        logger.error("word candidate list is empty.")
        return 2
    logger.info(
        "Loaded %d word candidates (min_length=%d, source=%s)",
        len(word_candidates),
        args.min_word_length,
        args.word_list or pdg.DEFAULT_WORD_LIST,
    )
    wh_ops = (
        pdg.WITHHOLD_OPERATORS_V2 if args.version == "v2" else pdg.WITHHOLD_OPERATORS
    )
    n_expanding = len(pdg.EXPANDING_OPERATORS & set(wh_ops))
    logger.info(
        "Withhold operator pool (%s): %d ops (%d expanding, %d length-preserving)",
        args.version,
        len(wh_ops),
        n_expanding,
        len(wh_ops) - n_expanding,
    )
    if args.mode == "depth-schedule":
        schedule = (
            DEPTH_SCHEDULE_V2_EXTENDED
            if args.schedule == "v2-extended"
            else DEPTH_SCHEDULE_V2
        )
        logger.info(
            "Mode: depth-schedule (%s, %d slices)",
            args.schedule,
            len(schedule),
        )
        for depth, n_fs, n_rows, max_exp in schedule:
            exp_str = "unconstrained" if max_exp is None else f"<= {max_exp}"
            logger.info(
                "  depth=%d: %d rows, few_shot=%d, max_expanding=%s",
                depth,
                n_rows,
                n_fs,
                exp_str,
            )
        tasks = generate_depth_scheduled_eval(
            word_candidates=word_candidates,
            seed=args.seed,
            schedule=schedule,
            explain_encryption=args.explain_encryption,
            include_answer_format=args.include_answer_format,
            include_retry_metadata=args.include_retry_metadata,
            version=args.version,
        )
    else:
        for name, k_lo, k_hi, n_few_shot, n_rows, max_exp in TIERS:
            exp_str = (
                "unconstrained" if max_exp is None else f"<= {max_exp} expanding op"
            )
            logger.info(
                "  Tier %s: %d rows, num_transformations \u2208 [%d, %d], "
                "num_few_shot=%d, max_expanding_ops=%s",
                name,
                n_rows,
                k_lo,
                k_hi,
                n_few_shot,
                exp_str,
            )
        tasks = generate_tiered_eval(
            word_candidates=word_candidates,
            seed=args.seed,
            explain_encryption=args.explain_encryption,
            include_answer_format=args.include_answer_format,
            include_retry_metadata=args.include_retry_metadata,
            version=args.version,
        )
    pdg.write_jsonl(tasks, args.output)
    logger.info("Wrote %d tasks to %s", len(tasks), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
