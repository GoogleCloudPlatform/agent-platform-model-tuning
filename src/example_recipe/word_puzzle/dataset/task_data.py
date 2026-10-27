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

"""Word_puzzle-typed subclass of the framework ``TaskData``."""

from __future__ import annotations

import dataclasses
from typing import Any

from core.task_data import TaskData


@dataclasses.dataclass
class WordPuzzleTaskData(TaskData):
    """Task payload for the word_puzzle recipe.

    Attributes:
        question: the puzzle prompt (defaulted to "" so the dataclass
            can be constructed field-by-field in tests).
        reference: the ground-truth answer string (may be empty when
            producers omit it).
        tags: optional per-task tag dict for offline slicing. Empty /
            missing values are normalized to ``None`` by
            ``parse_word_puzzle_item`` so downstream truthiness checks
            are uniform.

    Note: on-disk JSONL uses the nested ``contents``/``references``
    shape; ``parse_word_puzzle_item`` reshapes into these flat fields.
    ``eval_tool`` output rows are ``dataclasses.asdict`` dumps and are
    NOT re-parseable through ``parse_word_puzzle_item``.
    """

    question: str = ""
    reference: str = ""
    tags: dict[str, Any] | None = None
    retry_metadata: dict[str, Any] | None = None
