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

"""Unit tests for WordPuzzleTaskData + parse_word_puzzle_item transform."""

import dataclasses
import json

import pytest

from core.task_data import TaskData
from example_recipe.word_puzzle.dataset.task_data import WordPuzzleTaskData
from example_recipe.word_puzzle.env import parse_word_puzzle_item


def test_word_puzzle_task_data_is_task_data_subclass():
    td = WordPuzzleTaskData(task_id="t", question="q", reference="r")
    assert isinstance(td, TaskData)
    assert td.task_id == "t"
    assert td.question == "q"
    assert td.reference == "r"
    assert td.tags is None


def test_word_puzzle_task_data_accepts_tags():
    td = WordPuzzleTaskData(
        task_id="t", question="q", reference="r", tags={"difficulty": "hard"}
    )
    assert td.tags == {"difficulty": "hard"}


def test_parse_word_puzzle_item_returns_dataclass():
    item = {
        "contents": [{"parts": [{"text": "spell CAT backwards"}]}],
        "references": {"reference": "TAC"},
    }
    td = parse_word_puzzle_item(item)
    assert isinstance(td, WordPuzzleTaskData)
    assert td.question == "spell CAT backwards"
    assert td.reference == "TAC"
    assert td.tags is None
    assert td.task_id  # populated by ensure_task_id at parse time


def test_parse_word_puzzle_item_preserves_tags_when_present():
    item = {
        "contents": [{"parts": [{"text": "q"}]}],
        "references": {"reference": "r"},
        "tags": {"difficulty": "easy"},
    }
    td = parse_word_puzzle_item(item)
    assert td.tags == {"difficulty": "easy"}


def test_parse_word_puzzle_item_ignores_empty_tags():
    # Empty-dict tags (a producer edge case) should normalize to None so
    # downstream `if td.tags:` checks are consistent. Missing-key and
    # None cases are covered by test_parse_word_puzzle_item_returns_dataclass
    # (which omits `tags` entirely, hitting the same `item.get("tags") or None`
    # code path as tags=None).
    item = {
        "contents": [{"parts": [{"text": "q"}]}],
        "references": {"reference": "r"},
        "tags": {},
    }
    td = parse_word_puzzle_item(item)
    assert td.tags is None


def test_word_puzzle_task_data_json_roundtrips():
    td = WordPuzzleTaskData(
        task_id="hash-abc",
        question="q",
        reference="r",
        tags={"k": "v"},
    )
    payload = json.dumps(dataclasses.asdict(td))
    assert json.loads(payload) == {
        "task_id": "hash-abc",
        "question": "q",
        "reference": "r",
        "tags": {"k": "v"},
        # retry_metadata is an ours-side addition (agent-mode retry feature).
        "retry_metadata": None,
    }
