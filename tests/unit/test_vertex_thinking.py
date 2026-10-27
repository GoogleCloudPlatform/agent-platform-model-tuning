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

"""Tests for backends.vertex_thinking.apply_thinking_level."""

from __future__ import annotations

import pytest
from google.cloud.aiplatform_v1.types import GenerateContentRequest

from backends import vertex_thinking


class TestApplyThinkingLevel:
    def test_none_is_noop_on_string(self):
        assert vertex_thinking.apply_thinking_level("hello", None) == "hello"

    def test_none_is_noop_on_proto(self):
        prompt = GenerateContentRequest(
            contents=[{"role": "user", "parts": [{"text": "hi"}]}]
        )
        # Should return the same proto identity when thinking_level is falsy.
        assert vertex_thinking.apply_thinking_level(prompt, None) is prompt

    def test_empty_string_is_noop(self):
        # Falsy sentinel; must not attempt validation.
        assert vertex_thinking.apply_thinking_level("hello", "") == "hello"

    def test_string_prompt_wrapped_into_proto(self):
        out = vertex_thinking.apply_thinking_level("hello", "MINIMAL")
        assert isinstance(out, GenerateContentRequest)
        assert len(out.contents) == 1
        assert out.contents[0].role == "user"
        assert out.contents[0].parts[0].text == "hello"
        # Enum value: "MINIMAL" resolves to the ThinkingLevel enum member;
        # proto stores it as int, but MessageToDict(default camelCase)
        # marshals back to the string name.
        assert out.generation_config.thinking_config.thinking_level.name == "MINIMAL"

    def test_proto_prompt_copied_not_mutated(self):
        prompt = GenerateContentRequest(
            contents=[{"role": "user", "parts": [{"text": "hi"}]}]
        )
        out = vertex_thinking.apply_thinking_level(prompt, "LOW")
        assert out is not prompt  # copy, not mutation
        # Source's thinking_config still unset (default enum = 0)
        assert int(prompt.generation_config.thinking_config.thinking_level) == 0
        assert out.generation_config.thinking_config.thinking_level.name == "LOW"

    def test_preserves_existing_generation_config(self):
        prompt = GenerateContentRequest(
            contents=[],
            generation_config={
                "temperature": 0.7,
                "thinking_config": {"include_thoughts": True},
            },
        )
        out = vertex_thinking.apply_thinking_level(prompt, "MEDIUM")
        assert out.generation_config.temperature == pytest.approx(0.7)
        assert out.generation_config.thinking_config.include_thoughts is True
        assert out.generation_config.thinking_config.thinking_level.name == "MEDIUM"

    def test_case_insensitive(self):
        out = vertex_thinking.apply_thinking_level("x", "high")
        assert out.generation_config.thinking_config.thinking_level.name == "HIGH"
        out2 = vertex_thinking.apply_thinking_level("x", " Low ")
        assert out2.generation_config.thinking_config.thinking_level.name == "LOW"

    def test_invalid_level_raises_valueerror_listing_valid_options(self):
        with pytest.raises(ValueError) as exc_info:
            vertex_thinking.apply_thinking_level("x", "TURBO")
        msg = str(exc_info.value)
        assert "TURBO" in msg
        for level in vertex_thinking.VALID_THINKING_LEVELS:
            assert level in msg

    def test_valid_levels_frozenset_contract(self):
        assert vertex_thinking.VALID_THINKING_LEVELS == frozenset(
            {"MINIMAL", "LOW", "MEDIUM", "HIGH"}
        )
