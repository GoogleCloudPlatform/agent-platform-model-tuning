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

"""Tests for core.protocols.

These are contract tests: the ABCs have no behavior, but instantiating an
incomplete subclass should raise TypeError, and the EnvStepResult dataclass
should validate its shape.
"""

import dataclasses
import inspect

import pytest
from google.cloud.aiplatform_v1.types import Content

from core import protocols
from core.protocols import Action, Sampler


class TestEnvStepResult:
    def test_construct_with_all_fields(self):
        result = protocols.EnvStepResult(
            function_responses=["a", "b"],
            reward=0.5,
            done=True,
        )
        assert result.function_responses == ["a", "b"]
        assert result.reward == 0.5
        assert result.done is True

    def test_env_step_result_has_no_metrics_field(self):
        """EnvStepResult no longer carries the free-form metrics dict (all
        producers write typed task_stats; aggregation is Layer 2's job)."""
        field_names = {f.name for f in dataclasses.fields(protocols.EnvStepResult)}
        assert "metrics" not in field_names
        assert field_names == {
            "function_responses",
            "reward",
            "done",
            "is_correct",
            "task_stats",
        }

    def test_env_step_result_construction_rejects_metrics_kwarg(self):
        """Passing metrics= is now a TypeError (field deleted)."""
        with pytest.raises(TypeError):
            protocols.EnvStepResult(
                function_responses=[],
                reward=1.0,
                done=True,
                metrics={"env/x": 1.0},
            )


class TestEnv:
    def test_cannot_instantiate_without_step_and_get_prompt(self):
        with pytest.raises(TypeError):
            protocols.Env()  # type: ignore[abstract]

    def test_subclass_with_all_abstract_methods_works(self):
        class _E(protocols.Env):
            async def step(self, action, *, step_count: int = 0):
                del step_count
                return protocols.EnvStepResult(
                    function_responses=[], reward=0.0, done=True
                )

            def get_prompt(self):
                return "p"

        assert _E().get_prompt() == "p"


class TestSampler:
    def test_cannot_instantiate_without_sample(self):
        with pytest.raises(TypeError):
            protocols.Sampler()  # type: ignore[abstract]

    def test_subclass_with_sample_n_works(self):
        class _S(protocols.Sampler):
            async def sample_n(
                self,
                prompt,
                num_samples=1,
                thinking_level=None,
                max_generation_tokens=None,
                capture_ctxs=None,
            ):
                return []

        assert isinstance(_S(), protocols.Sampler)

    def test_subclass_can_accept_thinking_level_kwarg(self):
        # A concrete sampler must be able to accept the new thinking_level
        # kwarg from the ABC. Ignoring it in the implementation is fine.
        class _S(protocols.Sampler):
            async def sample_n(
                self,
                prompt,
                num_samples=1,
                thinking_level=None,
                max_generation_tokens=None,
                capture_ctxs=None,
            ):
                return [{"thinking_level": thinking_level, "num_samples": num_samples}]

        import asyncio

        s = _S()
        out = asyncio.run(s.sample_n("p", num_samples=1, thinking_level="MINIMAL"))
        assert out[0]["thinking_level"] == "MINIMAL"

    def test_sample_n_contract_has_max_generation_tokens(self):
        params = inspect.signature(Sampler.sample_n).parameters
        assert "max_generation_tokens" in params
        assert params["max_generation_tokens"].default is None
        names = list(params)
        assert (
            names.index("thinking_level")
            < names.index("max_generation_tokens")
            < names.index("capture_ctxs")
        )

    def test_subclass_can_accept_max_generation_tokens_kwarg(self):
        # A concrete sampler must be able to accept the new
        # max_generation_tokens kwarg from the ABC. Ignoring it is fine.
        class _S(protocols.Sampler):
            async def sample_n(
                self,
                prompt,
                num_samples=1,
                thinking_level=None,
                max_generation_tokens=None,
                capture_ctxs=None,
            ):
                return [{"max_generation_tokens": max_generation_tokens}]

        import asyncio

        s = _S()
        out = asyncio.run(s.sample_n("p", max_generation_tokens=256))
        assert out[0]["max_generation_tokens"] == 256


class TestTrainer:
    def test_cannot_instantiate_without_train_step(self):
        with pytest.raises(TypeError):
            protocols.Trainer()  # type: ignore[abstract]

    def test_subclass_with_train_step_works(self):
        class _T(protocols.Trainer):
            async def train_step(self, trajectories, sync_trainer_and_sampler=True):
                return {}

        assert isinstance(_T(), protocols.Trainer)


class TestAction:
    def test_action_is_frozen_dataclass(self):
        c = Content(role="model", parts=[])
        a = Action(
            content=c,
            finish_reason="STOP",
            generation_token_count=42,
            candidate_id="c1",
        )
        assert a.content is c
        assert a.finish_reason == "STOP"
        assert a.generation_token_count == 42
        assert a.candidate_id == "c1"
        # frozen: assignment raises
        import dataclasses

        with pytest.raises(dataclasses.FrozenInstanceError):
            a.finish_reason = "MAX_TOKENS"

    def test_action_candidate_id_defaults_to_none(self):
        c = Content(role="model", parts=[])
        a = Action(content=c, finish_reason="STOP", generation_token_count=1)
        assert a.candidate_id is None

    def test_an_action_carries_either_a_proto_or_tool_calls(self):
        # `content` used to be required. It is now the PROTO-PATH half of
        # the seam: recipes on the chat-completions path carry the turn in
        # `tool_calls`/`text` and never build a Gemini Content. Both halves
        # construct, and neither is mandatory -- an empty Action is a
        # legitimate STOP-with-no-output.
        from core.atif_trajectory import ToolCall

        proto_side = Action(
            content=Content(role="model", parts=[]),
            finish_reason="STOP",
            generation_token_count=1,
        )
        assert proto_side.tool_calls == ()

        oai_side = Action(
            tool_calls=(ToolCall(tool_call_id="c1", function_name="f", arguments={}),),
            text="thinking out loud",
            finish_reason="STOP",
        )
        assert oai_side.content is None
        assert oai_side.tool_calls[0].function_name == "f"

    def test_with_calls_replaces_the_tool_calls_and_keeps_everything_else(self):
        # The dispatcher coerces arguments and then hands the env a NEW
        # Action carrying the coerced calls -- the candidate_id and finish
        # reason must survive that swap or the env loses the training handle.
        from core.atif_trajectory import ToolCall

        original = Action(
            tool_calls=(ToolCall(tool_call_id="a", function_name="f", arguments={}),),
            finish_reason="STOP",
            candidate_id="cand-7",
            generation_token_count=42,
        )
        coerced = original.with_calls(
            [ToolCall(tool_call_id="a", function_name="f", arguments={"n": 5})]
        )
        assert coerced.tool_calls[0].arguments == {"n": 5}
        assert coerced.candidate_id == "cand-7"
        assert coerced.generation_token_count == 42
        assert original.tool_calls[0].arguments == {}  # unchanged
