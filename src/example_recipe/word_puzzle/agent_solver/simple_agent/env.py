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

"""Multi-turn env for the simple word-puzzle agent solver.

Handles two tools:
- **bash**: runs a command via ``asyncio.create_subprocess_shell`` in a
  temp directory with a timeout.
- **submit**: extracts the answer, computes reward via
  ``WordPuzzleRewardFn``, and terminates the episode.

A single turn may carry several function calls: bash calls run concurrently,
submit is deferred to the end of the turn, and every call gets exactly one
response in the model's original order.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from typing import Any

from google.cloud.aiplatform_v1.types import (
    Content,
    GenerateContentRequest,
    Part,
)

from core.protocols import Action, Env, EnvStepResult
from example_recipe.word_puzzle.agent_solver.simple_agent.prompt import (
    TOOLS,
    TOOLS_ANSWER_ONLY,
    render_instance_prompt,
    render_system_prompt,
)
from example_recipe.word_puzzle.dataset.task_data import WordPuzzleTaskData
from example_recipe.word_puzzle.reward import WordPuzzleRewardFn
from example_recipe.word_puzzle.reward_signals import WordPuzzleRewardSignals

logger = logging.getLogger(__name__)

_BASH_OUTPUT_LIMIT = 10_000  # chars
_BASH_TIMEOUT_DEFAULT = 30  # seconds


def _extract_function_calls(action: Action) -> list[dict[str, Any]]:
    """Extract every function_call part from an action's content, in order.

    Returns a list of ``{"name": ..., "args": {...}}``; empty when the
    turn carries no function call.

    Proto ``function_call`` is always present as a default submessage, so
    "name OR args populated" is the set-vs-unset test. A call with args
    but an empty name is kept, not skipped: it must still get a response,
    or the index-based call/response pairing in the ATIF observation
    shifts.
    """
    return [
        {
            "name": part.function_call.name,
            "args": dict(part.function_call.args),
        }
        for part in action.content.parts
        if part.function_call.name or part.function_call.args
    ]


class WordPuzzleSimpleEnv(Env):
    """Multi-turn env with bash execution and submit for word puzzles."""

    def __init__(
        self,
        task_data: WordPuzzleTaskData,
        reward_fn: WordPuzzleRewardFn,
        *,
        step_limit: int = 10,
        bash_timeout: int = _BASH_TIMEOUT_DEFAULT,
        answer_only_submit: bool = False,
    ) -> None:
        self.task_data = task_data
        self.reward_fn = reward_fn
        self.step_limit = step_limit
        self.bash_timeout = bash_timeout
        # Opt-in: state the submit contract ("answer only, exact match") in the
        # tool declaration and instance prompt. See prompt.py.
        self.answer_only_submit = answer_only_submit
        self._tmp_dir: tempfile.TemporaryDirectory[str] | None = None

    # ------------------------------------------------------------------
    # Env protocol
    # ------------------------------------------------------------------

    def get_prompt(self) -> GenerateContentRequest:
        """Build the initial ``GenerateContentRequest`` with tools."""
        system_text = render_system_prompt(step_limit=self.step_limit)
        user_text = render_instance_prompt(
            task=self.task_data.question, answer_only=self.answer_only_submit
        )

        return GenerateContentRequest(
            contents=[
                Content(role="user", parts=[Part(text=user_text)]),
            ],
            system_instruction=Content(parts=[Part(text=system_text)]),
            tools=TOOLS_ANSWER_ONLY if self.answer_only_submit else TOOLS,
        )

    async def step(self, action: Action, *, step_count: int = 0) -> EnvStepResult:
        """Dispatch every function call in the turn; one response per call.

        Parallel-tool-calls policy (same as the minisweagent solver): all
        ``bash`` calls run concurrently, then a deferred ``submit`` is
        graded, so ``[submit, bash]`` still runs the bash. The first
        submit wins; later ones are answered with a duplicate notice.
        Responses are returned in the model's original part order because
        the ATIF observation pairs responses to calls by index.
        """
        calls = _extract_function_calls(action)
        if not calls:
            return self._handle_text_only(action)

        responses: list[dict[str, Any]] = [{} for _ in calls]
        bash_idxs: list[int] = []
        submit_idx: int | None = None
        for i, fc in enumerate(calls):
            name = fc["name"]
            if name == "bash":
                bash_idxs.append(i)
            elif name == "submit":
                if submit_idx is None:
                    submit_idx = i
                else:
                    responses[i] = self._duplicate_submit_response()
            else:
                logger.warning("unknown tool name: %s", name)
                responses[i] = self._unknown_response(name)

        bash_responses = await asyncio.gather(
            *(self._run_bash(calls[i]["args"]) for i in bash_idxs)
        )
        for i, response in zip(bash_idxs, bash_responses, strict=True):
            responses[i] = response

        if submit_idx is None:
            return EnvStepResult(
                function_responses=responses,
                reward=None,
                done=False,
            )

        submit_response, reward, is_correct, task_stats = await self._run_submit(
            calls[submit_idx]["args"], action
        )
        responses[submit_idx] = submit_response
        return EnvStepResult(
            function_responses=responses,
            reward=reward,
            done=True,
            is_correct=is_correct,
            task_stats=task_stats,
        )

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------

    async def _run_submit(
        self, args: dict[str, Any], action: Action
    ) -> tuple[dict[str, Any], float, bool, Any]:
        """Grade a submit. Returns (response, reward, is_correct, task_stats)."""
        submitted = str(args.get("answer", ""))
        reference = self.task_data.reference.strip()
        is_correct = submitted.strip() == reference

        signals = WordPuzzleRewardSignals(
            is_correct=is_correct,
            generation_token_count=action.generation_token_count,
            finish_reason=action.finish_reason,
            max_output_tokens=None,
        )
        reward, task_stats = await self.reward_fn(signals)

        response = {
            "functionResponse": {
                "name": "submit",
                "response": {
                    "output": "Answer submitted.",
                    "is_correct": is_correct,
                },
            }
        }
        return response, reward, is_correct, task_stats

    async def _run_bash(self, args: dict[str, Any]) -> dict[str, Any]:
        """Run one bash command; return its functionResponse dict."""
        command = str(args.get("command", ""))

        if self._tmp_dir is None:
            self._tmp_dir = tempfile.TemporaryDirectory()

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=self._tmp_dir.name,
            )
            stdout_bytes, _ = await asyncio.wait_for(
                proc.communicate(), timeout=self.bash_timeout
            )
            output = stdout_bytes.decode("utf-8", errors="replace")
            exit_code = proc.returncode or 0
        except asyncio.TimeoutError:
            # Kill the process on timeout
            try:
                proc.kill()  # type: ignore[possibly-undefined]
                await proc.wait()  # type: ignore[possibly-undefined]
            except ProcessLookupError:
                pass
            output = f"Command timed out after {self.bash_timeout}s."
            exit_code = 124

        # Truncate long output
        if len(output) > _BASH_OUTPUT_LIMIT:
            output = (
                output[:_BASH_OUTPUT_LIMIT]
                + f"\n... (truncated at {_BASH_OUTPUT_LIMIT} chars)"
            )

        return {
            "functionResponse": {
                "name": "bash",
                "response": {
                    "output": output,
                    "exit_code": exit_code,
                },
            }
        }

    def _unknown_response(self, name: str) -> dict[str, Any]:
        return {
            "functionResponse": {
                "name": name if name else "unknown",
                "response": {
                    "output": (
                        f"Unknown tool '{name}'. "
                        "Please use the bash or submit tool."
                    ),
                },
            }
        }

    def _duplicate_submit_response(self) -> dict[str, Any]:
        return {
            "functionResponse": {
                "name": "submit",
                "response": {
                    "output": (
                        "Duplicate submit ignored: only the first submit "
                        "in a turn is processed."
                    ),
                },
            }
        }

    def _handle_text_only(self, action: Action) -> EnvStepResult:
        return EnvStepResult(
            function_responses=[
                {
                    "functionResponse": {
                        "name": "_system",
                        "response": {
                            "output": "Please use the bash or submit tool.",
                        },
                    }
                }
            ],
            reward=None,
            done=False,
        )
