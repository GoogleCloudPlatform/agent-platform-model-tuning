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

"""Tests for WordPuzzleSimpleEnv (agent-solver simple_agent)."""

from __future__ import annotations

import asyncio
import unittest

from google.cloud.aiplatform_v1.types import (
    Content,
    FunctionCall,
    GenerateContentRequest,
    Part,
)

from core.protocols import Action, EnvStepResult
from example_recipe.word_puzzle.agent_solver.simple_agent.env import (
    WordPuzzleSimpleEnv,
    _extract_function_calls,
)
from example_recipe.word_puzzle.dataset.task_data import WordPuzzleTaskData
from example_recipe.word_puzzle.reward import (
    WordPuzzleRewardConfig,
    WordPuzzleRewardFn,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _task_data(
    question: str = "What is 2+2?", reference: str = "4"
) -> WordPuzzleTaskData:
    return WordPuzzleTaskData(
        task_id="test-task", question=question, reference=reference
    )


def _reward_fn() -> WordPuzzleRewardFn:
    return WordPuzzleRewardFn(WordPuzzleRewardConfig())


def _make_env(**kwargs) -> WordPuzzleSimpleEnv:
    return WordPuzzleSimpleEnv(
        task_data=kwargs.pop("task_data", _task_data()),
        reward_fn=kwargs.pop("reward_fn", _reward_fn()),
        **kwargs,
    )


def _submit_action(answer: str) -> Action:
    """Build an Action with a submit function_call."""
    return Action(
        content=Content(
            role="model",
            parts=[
                Part(function_call=FunctionCall(name="submit", args={"answer": answer}))
            ],
        ),
        finish_reason="STOP",
        generation_token_count=10,
    )


def _bash_action(command: str) -> Action:
    """Build an Action with a bash function_call."""
    return Action(
        content=Content(
            role="model",
            parts=[
                Part(function_call=FunctionCall(name="bash", args={"command": command}))
            ],
        ),
        finish_reason="STOP",
        generation_token_count=10,
    )


def _text_action(text: str = "Let me think...") -> Action:
    """Build an Action with only a text Part (no function_call)."""
    return Action(
        content=Content(role="model", parts=[Part(text=text)]),
        finish_reason="STOP",
        generation_token_count=10,
    )


def _unknown_tool_action(name: str = "calculator") -> Action:
    """Build an Action with an unknown tool function_call."""
    return Action(
        content=Content(
            role="model",
            parts=[Part(function_call=FunctionCall(name=name, args={"expr": "2+2"}))],
        ),
        finish_reason="STOP",
        generation_token_count=10,
    )


def _parallel_action(calls: list[tuple[str, dict]]) -> Action:
    """Build an Action whose Content has one function_call Part per call."""
    return Action(
        content=Content(
            role="model",
            parts=[
                Part(function_call=FunctionCall(name=name, args=args))
                for name, args in calls
            ],
        ),
        finish_reason="STOP",
        generation_token_count=10,
    )


def _names(result: EnvStepResult) -> list[str]:
    """functionResponse names, in response order."""
    return [fr["functionResponse"]["name"] for fr in result.function_responses]


def _outputs(result: EnvStepResult) -> list[str]:
    """functionResponse ``output`` strings, in response order."""
    return [
        fr["functionResponse"]["response"]["output"]
        for fr in result.function_responses
    ]


# ---------------------------------------------------------------------------
# Tests: _extract_function_calls
# ---------------------------------------------------------------------------


class TestExtractFunctionCalls(unittest.TestCase):
    """Tests for the _extract_function_calls helper."""

    def test_returns_empty_for_text_only(self):
        action = _text_action("hello")
        self.assertEqual(_extract_function_calls(action), [])

    def test_extracts_submit(self):
        calls = _extract_function_calls(_submit_action("4"))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "submit")
        self.assertEqual(calls[0]["args"]["answer"], "4")

    def test_extracts_bash(self):
        calls = _extract_function_calls(_bash_action("echo hi"))
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "bash")
        self.assertEqual(calls[0]["args"]["command"], "echo hi")

    def test_extracts_all_calls_in_order(self):
        action = _parallel_action(
            [("bash", {"command": "echo hi"}), ("submit", {"answer": "4"})]
        )
        calls = _extract_function_calls(action)
        self.assertEqual([c["name"] for c in calls], ["bash", "submit"])

    def test_keeps_call_with_args_but_no_name(self):
        """A nameless call must still be extracted: it needs a response slot
        so the index-based call/response pairing stays aligned."""
        action = _parallel_action([("", {"x": "1"})])
        calls = _extract_function_calls(action)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "")
        self.assertEqual(calls[0]["args"]["x"], "1")


# ---------------------------------------------------------------------------
# Tests: get_prompt
# ---------------------------------------------------------------------------


class TestGetPrompt(unittest.TestCase):
    """Tests for WordPuzzleSimpleEnv.get_prompt()."""

    def test_returns_generate_content_request(self):
        env = _make_env()
        prompt = env.get_prompt()
        self.assertIsInstance(prompt, GenerateContentRequest)

    def test_contains_question(self):
        env = _make_env(task_data=_task_data(question="What is 3*3?"))
        prompt = env.get_prompt()
        user_text = prompt.contents[0].parts[0].text
        self.assertIn("What is 3*3?", user_text)

    def test_has_system_instruction(self):
        env = _make_env(step_limit=5)
        prompt = env.get_prompt()
        sys_text = prompt.system_instruction.parts[0].text
        self.assertIn("math puzzle solver", sys_text)
        self.assertIn("5", sys_text)

    def test_has_tools(self):
        env = _make_env()
        prompt = env.get_prompt()
        self.assertTrue(len(prompt.tools) > 0)
        decl_names = [fd.name for fd in prompt.tools[0].function_declarations]
        self.assertIn("bash", decl_names)
        self.assertIn("submit", decl_names)

    def test_default_prompt_keeps_original_submit_wording(self):
        # answer_only_submit is opt-in: existing runs keep their prompts.
        prompt = _make_env().get_prompt()
        submit = next(fd for fd in prompt.tools[0].function_declarations if fd.name == "submit")
        self.assertEqual(submit.description, "Submit your final answer to the puzzle.")
        self.assertEqual(submit.parameters.properties["answer"].description, "Your final answer.")
        self.assertIn(
            "call the submit tool with your answer.", prompt.contents[0].parts[0].text
        )

    def test_submit_contract_says_answer_only(self):
        # Opt-in: submit is graded by exact match on `answer`, and dataset
        # prompts may ask the model to explain its reasoning; the tool
        # contract must say the explanation does not go in `answer`.
        env = _make_env(answer_only_submit=True)
        prompt = env.get_prompt()
        submit = next(fd for fd in prompt.tools[0].function_declarations if fd.name == "submit")
        answer_desc = submit.parameters.properties["answer"].description
        self.assertIn("only the final answer", answer_desc.lower())
        self.assertIn("exact", submit.description.lower())
        self.assertIn("only your final answer", prompt.contents[0].parts[0].text.lower())


# ---------------------------------------------------------------------------
# Tests: submit
# ---------------------------------------------------------------------------


class TestSubmit(unittest.TestCase):
    """Tests for the submit tool handler."""

    def test_correct_answer(self):
        env = _make_env()
        result = asyncio.run(env.step(_submit_action("4")))
        self.assertTrue(result.done)
        self.assertTrue(result.is_correct)
        self.assertIsNotNone(result.reward)
        self.assertGreater(result.reward, 0)

    def test_wrong_answer(self):
        env = _make_env()
        result = asyncio.run(env.step(_submit_action("5")))
        self.assertTrue(result.done)
        self.assertFalse(result.is_correct)
        self.assertIsNotNone(result.reward)
        self.assertLess(result.reward, 0)

    def test_function_response_shape(self):
        env = _make_env()
        result = asyncio.run(env.step(_submit_action("4")))
        self.assertEqual(len(result.function_responses), 1)
        fr = result.function_responses[0]["functionResponse"]
        self.assertEqual(fr["name"], "submit")
        self.assertIn("output", fr["response"])
        self.assertIn("is_correct", fr["response"])
        self.assertTrue(fr["response"]["is_correct"])

    def test_submit_with_whitespace(self):
        """Submitted answer is stripped before comparison."""
        env = _make_env()
        result = asyncio.run(env.step(_submit_action("  4  ")))
        self.assertTrue(result.done)
        self.assertTrue(result.is_correct)


# ---------------------------------------------------------------------------
# Tests: bash
# ---------------------------------------------------------------------------


class TestBash(unittest.TestCase):
    """Tests for the bash tool handler."""

    def test_echo_command(self):
        env = _make_env()
        result = asyncio.run(env.step(_bash_action("echo hello")))
        self.assertFalse(result.done)
        self.assertIsNone(result.reward)
        fr = result.function_responses[0]["functionResponse"]
        self.assertEqual(fr["name"], "bash")
        self.assertIn("hello", fr["response"]["output"])
        self.assertEqual(fr["response"]["exit_code"], 0)

    def test_python_command(self):
        env = _make_env()
        result = asyncio.run(env.step(_bash_action("python3 -c 'print(2+2)'")))
        self.assertFalse(result.done)
        fr = result.function_responses[0]["functionResponse"]
        self.assertIn("4", fr["response"]["output"])

    def test_timeout(self):
        env = _make_env(bash_timeout=1)
        result = asyncio.run(env.step(_bash_action("sleep 10")))
        self.assertFalse(result.done)
        fr = result.function_responses[0]["functionResponse"]
        self.assertIn("timed out", fr["response"]["output"])
        self.assertEqual(fr["response"]["exit_code"], 124)

    def test_bash_then_submit(self):
        """A bash step followed by a submit step."""
        env = _make_env()

        async def _run():
            r1 = await env.step(_bash_action("echo 4"))
            r2 = await env.step(_submit_action("4"))
            return r1, r2

        r1, r2 = asyncio.run(_run())
        self.assertFalse(r1.done)
        self.assertTrue(r2.done)
        self.assertTrue(r2.is_correct)


# ---------------------------------------------------------------------------
# Tests: text-only
# ---------------------------------------------------------------------------


class TestTextOnly(unittest.TestCase):
    """Tests for text-only (no function_call) actions."""

    def test_text_only_returns_nudge(self):
        env = _make_env()
        result = asyncio.run(env.step(_text_action()))
        self.assertFalse(result.done)
        self.assertIsNone(result.reward)
        fr = result.function_responses[0]["functionResponse"]
        self.assertEqual(fr["name"], "_system")
        self.assertIn("bash or submit", fr["response"]["output"])


# ---------------------------------------------------------------------------
# Tests: unknown tool
# ---------------------------------------------------------------------------


class TestUnknownTool(unittest.TestCase):
    """Tests for unknown tool actions."""

    def test_unknown_tool(self):
        env = _make_env()
        result = asyncio.run(env.step(_unknown_tool_action("calculator")))
        self.assertFalse(result.done)
        self.assertIsNone(result.reward)
        fr = result.function_responses[0]["functionResponse"]
        self.assertEqual(fr["name"], "calculator")
        self.assertIn("Unknown tool", fr["response"]["output"])


# ---------------------------------------------------------------------------
# Tests: parallel function calls
# ---------------------------------------------------------------------------


class TestParallelFunctionCalls(unittest.TestCase):
    """Every function_call Part gets executed and answered, in order."""

    def test_two_bash_calls_both_run(self):
        env = _make_env(step_limit=10, bash_timeout=5)
        action = _parallel_action(
            [("bash", {"command": "echo one"}), ("bash", {"command": "echo two"})]
        )
        result = asyncio.run(env.step(action, step_count=1))

        self.assertEqual(len(result.function_responses), 2)
        self.assertEqual([o.strip() for o in _outputs(result)], ["one", "two"])
        self.assertFalse(result.done)

    def test_bash_then_submit_runs_both_and_terminates(self):
        env = _make_env(step_limit=10, bash_timeout=5)
        action = _parallel_action(
            [("bash", {"command": "echo hi"}), ("submit", {"answer": "4"})]
        )
        result = asyncio.run(env.step(action, step_count=1))

        self.assertEqual(_names(result), ["bash", "submit"])
        self.assertEqual(_outputs(result)[0].strip(), "hi")
        self.assertTrue(result.done)
        self.assertTrue(result.is_correct)
        self.assertGreater(result.reward, 0)

    def test_submit_then_bash_still_runs_the_bash(self):
        """Submit is terminal at end-of-turn, not terminal-immediately."""
        env = _make_env(step_limit=10, bash_timeout=5)
        action = _parallel_action(
            [("submit", {"answer": "4"}), ("bash", {"command": "echo hi"})]
        )
        result = asyncio.run(env.step(action, step_count=1))

        # Responses stay in the model's original Part order.
        self.assertEqual(_names(result), ["submit", "bash"])
        self.assertEqual(_outputs(result)[1].strip(), "hi")
        self.assertTrue(result.done)

    def test_second_submit_is_ignored(self):
        env = _make_env(step_limit=10, bash_timeout=5)
        action = _parallel_action(
            [("submit", {"answer": "4"}), ("submit", {"answer": "5"})]
        )
        result = asyncio.run(env.step(action, step_count=1))

        self.assertEqual(len(result.function_responses), 2)
        # The FIRST submit decided the episode; it was the correct answer.
        self.assertTrue(result.is_correct)
        self.assertIn("Duplicate submit ignored", _outputs(result)[1])

    def test_unknown_tool_alongside_bash_both_answered(self):
        env = _make_env(step_limit=10, bash_timeout=5)
        action = _parallel_action(
            [("calculator", {"expr": "2+2"}), ("bash", {"command": "echo hi"})]
        )
        result = asyncio.run(env.step(action, step_count=1))

        self.assertEqual(_names(result), ["calculator", "bash"])
        self.assertIn("Unknown tool", _outputs(result)[0])
        self.assertFalse(result.done)

    def test_nameless_call_answered_as_unknown(self):
        env = _make_env(step_limit=10, bash_timeout=5)
        action = _parallel_action(
            [("", {"x": "1"}), ("bash", {"command": "echo hi"})]
        )
        result = asyncio.run(env.step(action, step_count=1))

        self.assertEqual(_names(result), ["unknown", "bash"])
        self.assertEqual(_outputs(result)[1].strip(), "hi")
        self.assertFalse(result.done)


if __name__ == "__main__":
    unittest.main()
