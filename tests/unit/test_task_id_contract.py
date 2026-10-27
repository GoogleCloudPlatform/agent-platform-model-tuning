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

"""Tests for the per-recipe task_id contract (spec Q2).

Each recipe's task_data must expose a `task_id` (attr or dict key) so
core.session.build_session_id can consume it opaquely.

This file covers the word_puzzle recipe; other recipes' coverage lives
alongside their own trees.
"""

from __future__ import annotations

import pytest

from core.session import build_session_id
from example_recipe.word_puzzle.dataset.task_data import WordPuzzleTaskData


def test_word_puzzle_task_id_helper_hashes_question():
    """word_puzzle task_data has no natural id; hash of question is stable."""
    from example_recipe.word_puzzle.env import ensure_task_id

    # ``tags`` is a dict[str, Any] | None per the dataclass; empty dict
    # is fine (the parser normalizes falsy tags to None but the
    # dataclass ctor takes what it's given).
    td = WordPuzzleTaskData(
        task_id="", question="spell the word CAT backwards", reference="TAC", tags={}
    )
    td_with_id = ensure_task_id(td)
    assert isinstance(td_with_id.task_id, str)
    assert len(td_with_id.task_id) > 0
    # Same question → same id (deterministic).
    td2 = ensure_task_id(
        WordPuzzleTaskData(task_id="", question="spell the word CAT backwards")
    )
    assert td2.task_id == td_with_id.task_id
    # Different question → different id.
    td3 = ensure_task_id(WordPuzzleTaskData(task_id="", question="spell DOG backwards"))
    assert td3.task_id != td_with_id.task_id


def test_word_puzzle_task_id_idempotent():
    """Calling ensure_task_id twice yields the same id."""
    from example_recipe.word_puzzle.env import ensure_task_id

    td = ensure_task_id(WordPuzzleTaskData(task_id="", question="q1"))
    tid = td.task_id
    td_again = ensure_task_id(td)
    assert td_again.task_id == tid


def test_word_puzzle_ensure_task_id_empty_question_raises_valueerror():
    """Locks in the ValueError contract from the WordPuzzleTaskData migration.

    Pre-migration, ensure_task_id raised KeyError implicitly via
    `task_data["question"]` on a dict without the key. Post-migration,
    the dataclass's empty-string default means the guard is `if not
    task_data.question:` and the explicit raise is ValueError. This test
    prevents accidental regression if a future refactor either removes
    the empty-string default (silently allowing empty questions through)
    or reverts the error type.
    """
    from example_recipe.word_puzzle.env import ensure_task_id

    with pytest.raises(ValueError, match="missing required 'question'"):
        ensure_task_id(WordPuzzleTaskData(task_id="", question=""))
    with pytest.raises(ValueError, match="missing required 'question'"):
        ensure_task_id(WordPuzzleTaskData(task_id="", question=""))


def test_word_puzzle_task_data_flows_through_session_builder():
    from example_recipe.word_puzzle.env import ensure_task_id

    td = ensure_task_id(
        WordPuzzleTaskData(task_id="", question="spell CAT backwards", reference="TAC")
    )
    sid = build_session_id(
        recipe="word_puzzle",
        tuning_job_id="tj-1",
        task_data=td,
        rollout_idx=2,
    )
    # Ends with -2-<iso_stamp>; contains the recipe and the hashed id.
    assert "word_puzzle" in sid
    assert td.task_id is not None
    assert td.task_id in sid
