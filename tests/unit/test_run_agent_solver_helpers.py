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

"""Unit tests for helper functions in ``run_agent_solver``.

Kept separate from ``test_async_default_agent_*`` because these test the
runner-level plumbing (task loading, retry_metadata validation, etc.),
not the agent state machine.
"""

from __future__ import annotations

import pytest

pytest.importorskip("minisweagent")

import argparse  # noqa: E402

from example_recipe.word_puzzle.agent_solver.minisweagent.run_agent_solver import (  # noqa: E402
    _positive_int,
    _validate_retry_metadata,
)


# --------------------------------------------------------------------------
# _validate_retry_metadata: partial retry_metadata blocks must fail loud
# --------------------------------------------------------------------------


def test_validate_retry_metadata_accepts_none():
    """``retry_metadata`` absent (None) is fine -- feature simply won't fire.

    JSONL rows generated without ``--include-retry-metadata`` have no
    ``retry_metadata`` key. ``row.get("retry_metadata")`` returns None;
    the agent's empty-pool short-circuit handles it. No validation
    error to raise here.
    """
    # Should not raise.
    _validate_retry_metadata(None)


def test_validate_retry_metadata_accepts_empty_dict():
    """Empty ``retry_metadata: {}`` is treated the same as None."""
    # Should not raise.
    _validate_retry_metadata({})


def test_validate_retry_metadata_accepts_complete_block():
    """Complete block with all 3 required fields passes."""
    complete = {
        "rule": [["shift", 2]],
        "extra_word_pool": ["apple", "banjo"],
        "reverse_task": False,
    }
    # Should not raise.
    _validate_retry_metadata(complete)


def test_validate_retry_metadata_rejects_missing_reverse_task():
    """Partial block missing ``reverse_task`` must fail loud.

    Without this check, the runner defaults ``reverse_task=False``
    silently, potentially generating pairs in the wrong direction for
    tasks that were actually reversed. The generator always emits all
    3 fields together, so a partial block here indicates JSONL
    corruption or hand-editing.
    """
    partial = {
        "rule": [["shift", 2]],
        "extra_word_pool": ["apple", "banjo"],
        # reverse_task missing
    }
    with pytest.raises(ValueError, match="reverse_task"):
        _validate_retry_metadata(partial)


def test_validate_retry_metadata_rejects_missing_rule():
    """Partial block missing ``rule`` must fail loud."""
    partial = {
        "extra_word_pool": ["apple", "banjo"],
        "reverse_task": False,
        # rule missing
    }
    with pytest.raises(ValueError, match="rule"):
        _validate_retry_metadata(partial)


def test_validate_retry_metadata_rejects_missing_extra_word_pool():
    """Partial block missing ``extra_word_pool`` must fail loud."""
    partial = {
        "rule": [["shift", 2]],
        "reverse_task": False,
        # extra_word_pool missing
    }
    with pytest.raises(ValueError, match="extra_word_pool"):
        _validate_retry_metadata(partial)


def test_validate_retry_metadata_accepts_reverse_task_true():
    """``reverse_task: True`` is a valid non-default value."""
    complete = {
        "rule": [["shift", 2]],
        "extra_word_pool": ["apple"],
        "reverse_task": True,
    }
    # Should not raise.
    _validate_retry_metadata(complete)


# --------------------------------------------------------------------------
# _positive_int: argparse validator for --num-tasks (and similar)
# --------------------------------------------------------------------------
#
# Regression guard against the finding that
# ``--num-tasks 0`` was silently accepted, then flowed through to
# ``asyncio.gather(*[])`` -> ``_print_summary_table`` printing
# "0/0 correct". The validator rejects at the CLI boundary so the
# nonsense input never reaches downstream code.


def test_positive_int_accepts_positive():
    """Positive integers pass through as ints."""
    assert _positive_int("1") == 1
    assert _positive_int("10") == 10
    assert _positive_int("999999") == 999999


def test_positive_int_rejects_zero():
    """Zero is the specific case that was flagged."""
    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        _positive_int("0")


def test_positive_int_rejects_negative():
    """Negative integers are also nonsense for --num-tasks."""
    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        _positive_int("-1")
    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        _positive_int("-100")


def test_positive_int_rejects_non_integer():
    """Non-integer strings raise a distinct message so the CLI error is clear."""
    with pytest.raises(argparse.ArgumentTypeError, match="integer"):
        _positive_int("abc")
    with pytest.raises(argparse.ArgumentTypeError, match="integer"):
        _positive_int("1.5")
    with pytest.raises(argparse.ArgumentTypeError, match="integer"):
        _positive_int("")
