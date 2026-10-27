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

from __future__ import annotations

import hashlib as _hashlib
import re
from typing import TYPE_CHECKING, Any, Dict, Optional

from google.cloud.aiplatform_v1.types import Content, GenerateContentRequest, Part

from core.protocols import Action, Env, EnvStepResult


from example_recipe.word_puzzle.reward import (
    WordPuzzleRewardFn,
)
from example_recipe.word_puzzle.reward_signals import WordPuzzleRewardSignals
from example_recipe.word_puzzle.dataset.task_data import WordPuzzleTaskData


def ensure_task_id(task_data: WordPuzzleTaskData) -> WordPuzzleTaskData:
    """Attach a ``task_id`` derived from a SHA-1 prefix of ``question``.

    Idempotent; mutates ``task_data`` in place and returns it. Raises
    ``ValueError`` if ``question`` is empty.
    """
    if task_data.task_id:
        return task_data
    if not task_data.question:
        raise ValueError(
            "word_puzzle task_data missing required 'question'; "
            "cannot derive task_id for session id builder."
        )
    task_data.task_id = _hashlib.sha1(task_data.question.encode("utf-8")).hexdigest()[
        :12
    ]
    return task_data


def extract_answer_fn(action: Action) -> str:
    """Extract the answer text from an Action.

    Walks ``action.content.parts`` for non-thought text parts,
    concatenates their ``.text``, then extracts the last ``\\text{...}``
    match. Empty-parts and no-match cases return ``""``.
    """
    sample_text = "".join(
        part.text for part in action.content.parts if part.text and not part.thought
    )
    matches = list(re.finditer(r"\\text{(.*?)}", sample_text))
    if matches:
        last_match = matches[-1]
        return last_match.group(1) if last_match.groups() else ""
    return ""


class WordPuzzleEnv(Env):
    """Word-puzzle env; delegates all reward shaping to ``reward_fn``."""

    def __init__(
        self,
        task_data: WordPuzzleTaskData,
        reward_fn: WordPuzzleRewardFn,
        max_output_tokens: Optional[int] = None,
    ):
        self.task_data = task_data
        self.reward_fn = reward_fn
        self.max_output_tokens = max_output_tokens

    def get_prompt(self) -> GenerateContentRequest:
        return GenerateContentRequest(
            contents=[
                Content(role="user", parts=[Part(text=self.task_data.question)]),
            ],
        )

    async def step(self, action: Action, *, step_count: int = 0) -> EnvStepResult:
        # word_puzzle is single-step; step_count accepted for Env ABC
        # compat and ignored.
        del step_count

        finish_reason = action.finish_reason
        generation_token_count = action.generation_token_count

        parsed_answer = extract_answer_fn(action)
        reference = self.task_data.reference.strip()
        is_correct = parsed_answer == reference

        # Pass through the task's depth tag if present so depth-conditioned
        # RewardFns can shape per-depth. `tags` may be None for legacy datasets;
        # `depth` may be missing from tags; both cases -> None.
        tags = self.task_data.tags or {}
        depth_val = tags.get("depth") if isinstance(tags, dict) else None
        depth: int | None = int(depth_val) if isinstance(depth_val, int) else None

        signals = WordPuzzleRewardSignals(
            is_correct=is_correct,
            generation_token_count=generation_token_count,
            finish_reason=finish_reason,
            max_output_tokens=self.max_output_tokens,
            depth=depth,
        )
        reward, task_stats = await self.reward_fn(signals)

        # Preserve pre-PR semantic: a rollout that short-circuits on
        # MALFORMED_FUNCTION_CALL is NOT counted as correct on
        # ``learning/sampling_accuracy`` / ``learning/pass_at_k``, even if
        # the text happens to match ``reference``. The reward fn already
        # forces ``reward=malformed_output_reward`` (-4) for these; treat
        # the correctness signal consistently.
        env_is_correct = is_correct and not task_stats.malformed_function_call

        return EnvStepResult(
            function_responses=[],
            reward=reward,
            done=True,
            is_correct=env_is_correct,
            task_stats=task_stats,
        )


def word_puzzle_extract_prompt_text(td: WordPuzzleTaskData) -> str:
    """Extract the user prompt text from word-puzzle task data."""
    return td.question


def parse_word_puzzle_item(item: Dict[str, Any]) -> WordPuzzleTaskData:
    """Schema transform consumed by ``JsonlDataset`` for word-puzzle data.

    Reshapes ``{"contents": [{"parts": [{"text": ...}]}], "references":
    {"reference": ...}}`` into a ``WordPuzzleTaskData``. Optional
    top-level ``tags`` is passed through; empty/missing is normalized to
    ``None``.
    """
    question = item["contents"][0]["parts"][0]["text"]
    reference = item["references"]["reference"]
    tags = item.get("tags") or None
    retry_metadata = item.get("retry_metadata") or None
    td = WordPuzzleTaskData(
        task_id="",
        question=question,
        reference=reference,
        tags=tags,
        retry_metadata=retry_metadata,
    )
    return ensure_task_id(td)
