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

"""Tests for backends.chat_completion.

Each test class describes a group of related behaviors of
``convert_openai_request_to_gemini`` — the OpenAI Chat Completions ->
Vertex ``generateContent`` request-side translation. Test docstrings
describe the wire-level behavior being verified in terms of the
OpenAI and Vertex API surfaces:

  * OpenAI Chat Completions:
    https://platform.openai.com/docs/api-reference/chat
  * Vertex AI ``generateContent``:
    https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference

Scaffolding scope (this commit): the simplest happy-path cases plus
basic function calling. Follow-up commits will add tests for
multimodal parts, thought-tag markers, response_format,
reasoning_effort, and extra_body handling.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from backends.chat_completion import (
    ChatCompletionResponse,
    Choice,
    CompletionTokensDetails,
    ConversionContext,
    FunctionCall,
    GEMINI_MODEL_KIND_FLASH,
    GEMINI_MODEL_KIND_FLASH_LITE,
    GEMINI_MODEL_KIND_NONE,
    GEMINI_MODEL_KIND_PRO,
    Message,
    PromptTokensDetails,
    ProviderSpecificFields,
    ThinkingBlock,
    ToolCall,
    Usage,
    convert_openai_request_to_gemini,
    get_gemini_model_kind,
    use_per_part_media_resolution,
    use_thinking_level,
)


class TestGeminiModelKind:
    """Covers get_gemini_model_kind + use_per_part_media_resolution + use_thinking_level."""

    def test_flash_variant(self):
        assert (
            get_gemini_model_kind("google/gemini-flash-latest")
            == GEMINI_MODEL_KIND_FLASH
        )
        assert get_gemini_model_kind("gemini-2.5-flash") == GEMINI_MODEL_KIND_FLASH

    def test_pro_variant(self):
        assert get_gemini_model_kind("gemini-pro") == GEMINI_MODEL_KIND_PRO
        assert get_gemini_model_kind("google/gemini-pro-002") == GEMINI_MODEL_KIND_PRO

    def test_flash_lite_matches_before_flash(self):
        # `flash-lite` is a substring of neither `flash` nor `pro`
        # alone; the classifier must check flash-lite first so `flash`
        # substring match doesn't win.
        assert (
            get_gemini_model_kind("gemini-flash-lite") == GEMINI_MODEL_KIND_FLASH_LITE
        )
        assert (
            get_gemini_model_kind("google/gemini-3-flash-lite")
            == GEMINI_MODEL_KIND_FLASH_LITE
        )

    def test_none_for_unknown(self):
        assert get_gemini_model_kind("llama-3") == GEMINI_MODEL_KIND_NONE
        assert get_gemini_model_kind("") == GEMINI_MODEL_KIND_NONE

    def test_thinking_level_and_per_part_only_for_gemini_3(self):
        # 3.x -> new-shape ThinkingLevel enum + per-Part media resolution.
        assert use_thinking_level("google/gemini-3-flash-lite") is True
        assert use_per_part_media_resolution("google/gemini-3-flash-lite") is True
        # 2.x / older -> False (uses thinkingBudget int + request-level media res).
        assert use_thinking_level("google/gemini-2.5-flash") is False
        assert use_per_part_media_resolution("google/gemini-2.5-flash") is False


class TestConvertRequestEmptyInputs:
    """Empty-ish input still produces a well-formed Vertex request.

    Verifies that:

      * An empty ``messages`` list still yields one empty-text user
        Content so downstream Vertex has something to receive.
      * Every "empty-ish" content shape (empty string, empty array,
        array of empty text/refusal parts, explicit ``null``, missing
        ``content`` key) collapses to a single empty-text Part.
    """

    def test_empty_messages_list_produces_single_empty_user_content(self):
        # Sub-case 1: empty messages list still produces a request with
        # one empty-text user Content, so downstream Vertex has
        # something to receive. Model + stream fields are honored.
        result = convert_openai_request_to_gemini(
            context=None,
            json_input={"model": "test", "stream": True, "messages": []},
        )
        assert result == {
            "model": "test",
            "contents": [{"role": "user", "parts": [{"text": ""}]}],
        }

    def test_empty_and_null_content_variants_all_produce_empty_text_parts(self):
        # 12 messages exercising every "empty-ish" content shape
        # (string "", array [], array with empty text/refusal,
        # explicit null, missing content field). All collapse to
        # empty-text Parts. Consecutive same-role messages merge.
        # System/system/system in the middle populates systemInstruction
        # regardless of the surrounding conversation.
        result = convert_openai_request_to_gemini(
            context=None,
            json_input={
                "model": "test",
                "stream": True,
                "messages": [
                    {"role": "user", "content": ""},
                    {"role": "user", "content": []},
                    {"role": "user", "content": [{"type": "text", "text": ""}, ""]},
                    {"role": "system", "content": ""},
                    {"role": "system", "content": []},
                    {"role": "system", "content": [{"type": "text", "text": ""}, ""]},
                    {"role": "assistant", "content": []},
                    {"role": "assistant", "content": ""},
                    {"role": "assistant", "content": None},
                    {"role": "assistant"},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "refusal", "refusal": ""},
                            {"type": "text", "text": ""},
                            "",
                        ],
                    },
                    {"role": "user", "content": []},
                ],
            },
        )
        # Expected shape: system messages get consolidated into
        # systemInstruction. Adjacent same-role user messages merge
        # into one Content. Empty content entries within a merged
        # Content compact down to a single empty-text Part per
        # Content (rather than one Part per input message) — the
        # empty-Part skip in ``_iter_content_parts_from_message``
        # combined with the post-loop empty-Content padding produces
        # exactly one ``{"text": ""}`` per empty Content, giving:
        #   contents { role: "user"      parts { text: "" } }
        #   contents { role: "assistant" parts { text: "" } }
        #   contents { role: "user"      parts { text: "" } }
        assert result == {
            "model": "test",
            "contents": [
                {"role": "user", "parts": [{"text": ""}]},
                {"role": "model", "parts": [{"text": ""}]},
                {"role": "user", "parts": [{"text": ""}]},
            ],
            "systemInstruction": {
                "role": "system",
                "parts": [{"text": ""}],
            },
        }


class TestConvertRequestBasicMessages:
    """Text-only multi-message conversations translate correctly.

    Verifies role classification (system vs. user vs. assistant vs.
    developer), the merging of consecutive same-role messages, and
    the extraction of system/developer messages into
    ``systemInstruction``. Multimodal parts (``image_url``,
    ``input_audio``, ``audio``) are covered in the multimodal
    follow-up; here they'd raise NotImplementedError.
    """

    def test_text_only_messages_translate_with_role_merging(self):
        # Text-only multi-message conversation: system messages merge
        # into systemInstruction; consecutive same-role user/assistant
        # messages merge into one Content each.
        result = convert_openai_request_to_gemini(
            context=None,
            json_input={
                "model": "google/llama2-70b",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Answer the following question:"},
                    {
                        "role": "system",
                        "content": [
                            "Be",
                            {"type": "text", "text": " as concise as possible."},
                        ],
                    },
                    {"role": "user", "content": "What is 2 + 2?"},
                    {
                        "role": "assistant",
                        "content": [
                            "The",
                            {"type": "text", "text": " answer is 4."},
                            {"type": "refusal", "refusal": "No"},
                        ],
                    },
                    {"role": "user", "content": "What is that number times 8?"},
                    {"role": "assistant", "content": "The answer is 32."},
                    {"role": "user", "content": "Thanks!"},
                ],
            },
        )
        # Expected: consecutive same-role messages collapse into a
        # single Content whose parts are the concatenation. Because
        # the last user message ("Thanks!") is immediately after
        # "The answer is 32." assistant, and system messages don't
        # reset the role-tracking, "Thanks!" starts a new user
        # Content. All user messages that are actually adjacent
        # (system messages don't split roles at the Content level --
        # they populate systemInstruction) merge; here the assistant
        # between "What is that number times 8?" and "Thanks!" keeps
        # them separate.
        assert result == {
            "model": "google/llama2-70b",
            "systemInstruction": {
                "role": "system",
                "parts": [
                    {"text": "You are a helpful assistant."},
                    {"text": "Be"},
                    {"text": " as concise as possible."},
                ],
            },
            "contents": [
                # First user block: "Answer..." + (system message
                # doesn't split) + "What is 2 + 2?" -> merged.
                {
                    "role": "user",
                    "parts": [
                        {"text": "Answer the following question:"},
                        {"text": "What is 2 + 2?"},
                    ],
                },
                {
                    "role": "model",
                    "parts": [
                        {"text": "The"},
                        {"text": " answer is 4."},
                        {"text": "No"},
                    ],
                },
                {"role": "user", "parts": [{"text": "What is that number times 8?"}]},
                {"role": "model", "parts": [{"text": "The answer is 32."}]},
                {"role": "user", "parts": [{"text": "Thanks!"}]},
            ],
        }

    def test_developer_role_treated_as_system(self):
        # OpenAI's newer `developer` role (formerly `system`) must map
        # to Vertex systemInstruction identically to `system`.
        result = convert_openai_request_to_gemini(
            context=None,
            json_input={
                "model": "m",
                "messages": [
                    {"role": "developer", "content": "hi"},
                    {"role": "user", "content": "hello"},
                ],
            },
        )
        assert result["systemInstruction"] == {
            "role": "system",
            "parts": [{"text": "hi"}],
        }
        assert result["contents"] == [{"role": "user", "parts": [{"text": "hello"}]}]


class TestConvertRequestGenerationConfig:
    """Top-level generation-config knobs translate to camelCase Vertex fields.

    Verifies the OpenAI request-level fields (temperature, top_p, n,
    max_tokens, max_completion_tokens, penalties, seed, logprobs,
    modalities, metadata) get mapped to their Vertex
    ``generationConfig`` / ``labels`` equivalents. The extra_body
    case is skipped — extra_body handling is a follow-up.
    """

    def test_generation_config_knobs_map_camelcase(self):
        result = convert_openai_request_to_gemini(
            context=None,
            json_input={
                "model": "google/llama2-70b",
                "messages": [{"role": "user", "content": "Hello!"}],
                "temperature": 0.9,
                "n": 3,
                "max_tokens": 1023,
                "frequency_penalty": -0.053,
                "presence_penalty": 2,
                "top_p": 0.2,
                "seed": 1023,
                "logprobs": True,
                "top_logprobs": 4,
                "metadata": {"A": "B", "C": "D"},
                "modalities": ["text", "audio", "image"],
            },
        )
        assert result == {
            "model": "google/llama2-70b",
            "contents": [{"role": "user", "parts": [{"text": "Hello!"}]}],
            "generationConfig": {
                "temperature": 0.9,
                "candidateCount": 3,
                "maxOutputTokens": 1023,
                "topP": 0.2,
                "frequencyPenalty": -0.053,
                "presencePenalty": 2.0,
                "seed": 1023,
                "responseLogprobs": True,
                "logprobs": 4,
                "responseModalities": ["TEXT", "AUDIO", "IMAGE"],
            },
            "labels": {"A": "B", "C": "D"},
        }

    def test_max_completion_tokens_alone_maps_to_max_output_tokens(self):
        # When only max_completion_tokens is present (no max_tokens),
        # it maps to Vertex maxOutputTokens. Precedence when BOTH are
        # present is covered by test_max_tokens_wins_when_both_present_for_backward_compat.
        result = convert_openai_request_to_gemini(
            context=None,
            json_input={
                "model": "google/llama2-70b",
                "messages": [{"role": "user", "content": "Hello!"}],
                "max_completion_tokens": 1023,
            },
        )
        assert result == {
            "model": "google/llama2-70b",
            "contents": [{"role": "user", "parts": [{"text": "Hello!"}]}],
            "generationConfig": {"maxOutputTokens": 1023},
        }

    def test_max_tokens_wins_when_both_present_for_backward_compat(self):
        # When both max_tokens and max_completion_tokens are set,
        # ``max_tokens`` wins (checked first; ``max_completion_tokens``
        # is the elif branch). This is the OPPOSITE of the OpenAI
        # spec, which deprecates ``max_tokens`` in favor of
        # ``max_completion_tokens``, but the current backend behavior
        # is preserved for wire-format compatibility.
        result = convert_openai_request_to_gemini(
            context=None,
            json_input={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100,
                "max_completion_tokens": 200,
            },
        )
        assert result["generationConfig"]["maxOutputTokens"] == 100


class TestConvertRequestStopSequences:
    """OpenAI ``stop`` field maps to Vertex ``stopSequences``."""

    def test_stop_string_wraps_to_single_element_list(self):
        result = convert_openai_request_to_gemini(
            context=None,
            json_input={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stop": "END",
            },
        )
        assert result["generationConfig"]["stopSequences"] == ["END"]

    def test_stop_list_passes_through(self):
        result = convert_openai_request_to_gemini(
            context=None,
            json_input={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "stop": ["END", "STOP", "\n\n"],
            },
        )
        assert result["generationConfig"]["stopSequences"] == ["END", "STOP", "\n\n"]

    def test_stop_dict_raises(self):
        # Regression guard: neither OpenAI nor Vertex accept dict for
        # stop; we reject with ValueError rather than pass it through
        # and let Vertex 400.
        with pytest.raises(ValueError, match="stop must be str or list"):
            convert_openai_request_to_gemini(
                context=None,
                json_input={
                    "model": "m",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stop": {"nope": True},
                },
            )


class TestConvertRequestContext:
    """publisher_model_name + use_model_role behavior on ConversionContext."""

    def test_publisher_model_name_overrides_input_model_field(self):
        result = convert_openai_request_to_gemini(
            context=ConversionContext(
                publisher_model_name="google/gemini-flash-latest"
            ),
            json_input={
                "model": "input-model",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert result["model"] == "google/gemini-flash-latest"

    def test_use_model_role_rewrites_assistant_to_model(self):
        result = convert_openai_request_to_gemini(
            context=ConversionContext(use_model_role=True),
            json_input={
                "model": "m",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ],
            },
        )
        assert result["contents"] == [
            {"role": "user", "parts": [{"text": "hi"}]},
            {"role": "model", "parts": [{"text": "hello"}]},
        ]

    def test_use_model_role_false_keeps_assistant(self):
        result = convert_openai_request_to_gemini(
            context=ConversionContext(use_model_role=False),
            json_input={
                "model": "m",
                "messages": [
                    {"role": "assistant", "content": "hello"},
                ],
            },
        )
        assert result["contents"][0]["role"] == "assistant"


class TestConvertRequestUnsupportedFeatures:
    """Every top-level and message-level feature not yet ported must raise
    NotImplementedError so callers see a loud, greppable failure rather
    than getting silently-partial output.
    """

    @pytest.mark.parametrize(
        "field,value",
        [
            ("response_format", {"type": "json_object"}),
            ("reasoning_effort", "minimal"),
            ("stream_options", {"include_usage": True}),
        ],
    )
    def test_top_level_unsupported_field_raises(self, field: str, value: object):
        with pytest.raises(NotImplementedError, match="not yet ported"):
            convert_openai_request_to_gemini(
                context=None,
                json_input={
                    "model": "m",
                    "messages": [{"role": "user", "content": "hi"}],
                    field: value,
                },
            )

    def test_extra_body_raises(self):
        with pytest.raises(NotImplementedError, match="extra_body"):
            convert_openai_request_to_gemini(
                context=None,
                json_input={"model": "m", "messages": []},
                extra_body={"session_id": "X"},
            )


class TestConvertRequestTools:
    """Modern ``tools`` array + legacy ``functions`` array translate correctly.

    Verifies:

      * Multiple tool declarations collapse to one Vertex ``tools``
        entry containing a ``functionDeclarations`` list.
      * OpenAI ``parameters`` (an OpenAPI-style JSON schema) is
        emitted as Vertex ``parametersJsonSchema`` with recursive
        camelCase renames and ``nullable: True`` fold-into-type.
      * Modern ``tools`` and legacy ``functions`` produce identical
        output shapes; when both are set, modern wins.

    Extra_body-provided tools (computer_use) and web_search_options
    are deferred to the extra_body / web-search follow-up.
    """

    def test_tools_array_collapses_to_single_tools_entry_with_declarations(self):
        # 3 tool declarations: name-only, name+description, and
        # name+description+parameters (an ARRAY-typed schema). Output
        # collapses to ONE tools entry containing all 3 declarations,
        # with parameters -> parametersJsonSchema (renamed camelCase).
        result = convert_openai_request_to_gemini(
            context=None,
            json_input={
                "model": "google/llama2-70b",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [
                    {"type": "function", "function": {"name": "rand"}},
                    {
                        "type": "function",
                        "function": {
                            "name": "output_6",
                            "description": "Outputs the number 6.",
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "add",
                            "description": "Adds a list of numbers together",
                            "parameters": {
                                "type": "ARRAY",
                                "description": "List of numbers to add",
                                "items": {
                                    "type": "NUMBER",
                                    "format": "double",
                                    "description": "A number to add",
                                },
                                "nullable": False,
                            },
                        },
                    },
                ],
            },
        )
        # Expected: one Vertex ``tools`` entry containing three
        # functionDeclarations, with nullable: False stripped and
        # camelCase field renames applied to the OpenAPI schema.
        assert result == {
            "model": "google/llama2-70b",
            "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
            "tools": [
                {
                    "functionDeclarations": [
                        {"name": "rand"},
                        {
                            "name": "output_6",
                            "description": "Outputs the number 6.",
                        },
                        {
                            "name": "add",
                            "description": "Adds a list of numbers together",
                            "parametersJsonSchema": {
                                "type": "ARRAY",
                                "description": "List of numbers to add",
                                "items": {
                                    "type": "NUMBER",
                                    "format": "double",
                                    "description": "A number to add",
                                },
                            },
                        },
                    ]
                }
            ],
        }

    def test_legacy_functions_array_same_shape_without_wrapper(self):
        # Legacy ``functions`` entries ARE function declarations
        # directly — no {"type": "function", "function": ...}
        # wrapper. Output shape matches the modern tools[] variant.
        result = convert_openai_request_to_gemini(
            context=None,
            json_input={
                "model": "google/llama2-70b",
                "messages": [{"role": "user", "content": "hello"}],
                "functions": [
                    {"name": "rand"},
                    {"name": "output_6", "description": "Outputs the number 6."},
                    {
                        "name": "add",
                        "description": "Adds a list of numbers together",
                        "parameters": {
                            "type": "ARRAY",
                            "description": "List of numbers to add",
                            "items": {
                                "type": "NUMBER",
                                "format": "double",
                                "description": "A number to add",
                            },
                            "nullable": False,
                        },
                    },
                ],
            },
        )
        assert result == {
            "model": "google/llama2-70b",
            "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
            "tools": [
                {
                    "functionDeclarations": [
                        {"name": "rand"},
                        {
                            "name": "output_6",
                            "description": "Outputs the number 6.",
                        },
                        {
                            "name": "add",
                            "description": "Adds a list of numbers together",
                            "parametersJsonSchema": {
                                "type": "ARRAY",
                                "description": "List of numbers to add",
                                "items": {
                                    "type": "NUMBER",
                                    "format": "double",
                                    "description": "A number to add",
                                },
                            },
                        },
                    ]
                }
            ],
        }

    def test_tools_wins_over_legacy_functions(self):
        # if/elif precedence: modern ``tools`` wins when both are
        # present; legacy ``functions`` is silently ignored.
        result = convert_openai_request_to_gemini(
            context=None,
            json_input={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "modern"}}],
                "functions": [{"name": "legacy"}],
            },
        )
        assert result["tools"] == [{"functionDeclarations": [{"name": "modern"}]}]

    def test_nullable_true_produces_null_in_type_list(self):
        # Regression guard for _standardize_openapi_field_names'
        # nullable handling: ``nullable: True`` folds "null" into the
        # ``type`` field (converting a scalar type into a list).
        result = convert_openai_request_to_gemini(
            context=None,
            json_input={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "f",
                            "parameters": {"type": "STRING", "nullable": True},
                        },
                    }
                ],
            },
        )
        decl = result["tools"][0]["functionDeclarations"][0]
        assert decl["parametersJsonSchema"] == {"type": ["STRING", "null"]}

    def test_nullable_false_is_dropped_from_schema(self):
        # ``nullable: False`` just removes the nullable field without
        # touching type — no "null" gets folded in.
        result = convert_openai_request_to_gemini(
            context=None,
            json_input={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "f",
                            "parameters": {"type": "STRING", "nullable": False},
                        },
                    }
                ],
            },
        )
        decl = result["tools"][0]["functionDeclarations"][0]
        assert decl["parametersJsonSchema"] == {"type": "STRING"}


class TestConvertRequestToolChoice:
    """OpenAI ``tool_choice`` / legacy ``function_call`` translate to Vertex ``toolConfig``.

    Verifies each mode-string value (validated / auto / none /
    required / specific-function) and the legacy shape counterparts
    (auto / none / specific-function). When both fields are present,
    the modern ``tool_choice`` wins.
    """

    def _base_input(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        base = {
            "model": "google/llama2-70b",
            "messages": [{"role": "user", "content": "hello"}],
        }
        base.update(extra)
        return base

    def _expected_config(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "model": "google/llama2-70b",
            "contents": [{"role": "user", "parts": [{"text": "hello"}]}],
            "toolConfig": {"functionCallingConfig": cfg},
        }

    def test_tool_choice_validated_maps_to_validated_mode(self):
        assert convert_openai_request_to_gemini(
            None, self._base_input({"tool_choice": "validated"})
        ) == self._expected_config({"mode": "VALIDATED"})

    def test_tool_choice_auto_maps_to_auto_mode(self):
        assert convert_openai_request_to_gemini(
            None, self._base_input({"tool_choice": "auto"})
        ) == self._expected_config({"mode": "AUTO"})

    def test_tool_choice_none_maps_to_none_mode(self):
        assert convert_openai_request_to_gemini(
            None, self._base_input({"tool_choice": "none"})
        ) == self._expected_config({"mode": "NONE"})

    def test_tool_choice_required_maps_to_any_mode(self):
        # OpenAI "required" -> Vertex ANY (Vertex has no separate
        # "required" mode; ANY means "must call at least one tool").
        assert convert_openai_request_to_gemini(
            None, self._base_input({"tool_choice": "required"})
        ) == self._expected_config({"mode": "ANY"})

    def test_tool_choice_specific_function_object(self):
        # tool_choice as object -> mode ANY + allowedFunctionNames.
        assert convert_openai_request_to_gemini(
            None,
            self._base_input(
                {
                    "tool_choice": {
                        "type": "function",
                        "function": {"name": "add"},
                    }
                }
            ),
        ) == self._expected_config({"mode": "ANY", "allowedFunctionNames": ["add"]})

    def test_legacy_function_call_auto_maps_to_auto_mode(self):
        assert convert_openai_request_to_gemini(
            None, self._base_input({"function_call": "auto"})
        ) == self._expected_config({"mode": "AUTO"})

    def test_legacy_function_call_none_maps_to_none_mode(self):
        assert convert_openai_request_to_gemini(
            None, self._base_input({"function_call": "none"})
        ) == self._expected_config({"mode": "NONE"})

    def test_legacy_function_call_specific_function_object(self):
        # Legacy shape: {"name": "add"} (no outer "function" wrapper).
        assert convert_openai_request_to_gemini(
            None, self._base_input({"function_call": {"name": "add"}})
        ) == self._expected_config({"mode": "ANY", "allowedFunctionNames": ["add"]})

    def test_tool_choice_wins_over_legacy_function_call(self):
        # Same if/elif precedence as tools/functions.
        assert convert_openai_request_to_gemini(
            None,
            self._base_input({"tool_choice": "auto", "function_call": "none"}),
        ) == self._expected_config({"mode": "AUTO"})


class TestConvertRequestMultiTurnToolCalls:
    """Multi-turn function-calling conversations translate end-to-end.

    Exercises the full FC round trip (user prompt -> model tool_call
    -> tool response -> user follow-up), the tool_call_id -> name
    lookup used to populate ``functionResponse.name``, parallel tool
    calls sharing one assistant Content, and the legacy
    ``function_call`` field on an assistant message.
    """

    def test_full_tool_call_round_trip(self):
        result = convert_openai_request_to_gemini(
            context=None,
            json_input={
                "model": "google/llama2-70b",
                "messages": [
                    {"role": "user", "content": "hello"},
                    {
                        "role": "model",
                        "tool_calls": [
                            {
                                "id": "function-call-UUID",
                                "type": "function",
                                "function": {
                                    "name": "test",
                                    "arguments": '{"key": "value"}',
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "function-call-UUID",
                        "content": '{"rkey": "rvalue"}',
                    },
                    {"role": "user", "content": "Who are you?"},
                ],
            },
        )
        assert result == {
            "model": "google/llama2-70b",
            "contents": [
                {"role": "user", "parts": [{"text": "hello"}]},
                {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "id": "function-call-UUID",
                                "name": "test",
                                "args": {"key": "value"},
                            }
                        }
                    ],
                },
                # Tool-role Content has NO role field.
                {
                    "parts": [
                        {
                            "functionResponse": {
                                "name": "test",
                                "response": {"rkey": "rvalue"},
                                "id": "function-call-UUID",
                            }
                        }
                    ]
                },
                {"role": "user", "parts": [{"text": "Who are you?"}]},
            ],
        }

    def test_parallel_function_calls_share_one_assistant_content(self):
        # Multiple tool_calls in one assistant message -> multiple
        # functionCall Parts under ONE model Content. Two adjacent
        # tool-role responses -> ONE Content with two
        # functionResponse Parts.
        result = convert_openai_request_to_gemini(
            context=None,
            json_input={
                "model": "google/gemini-1.5-pro-001",
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "What is difference in temperature in New Delhi and San Francisco?"
                        ),
                    },
                    {
                        "role": "model",
                        "tool_calls": [
                            {
                                "id": "function-call-UUID-1",
                                "type": "function",
                                "function": {
                                    "name": "get_current_weather",
                                    "arguments": '{"location": "New Delhi"}',
                                },
                            },
                            {
                                "id": "function-call-UUID-2",
                                "type": "function",
                                "function": {
                                    "name": "get_current_weather_2",
                                    "arguments": '{"location": "San Francisco"}',
                                },
                            },
                        ],
                    },
                    {
                        "role": "tool",
                        "content": '{"temperature": 30.5, "unit": "C"}',
                        "tool_call_id": "function-call-UUID-1",
                    },
                    {
                        "role": "tool",
                        "content": "20 degrees C",
                        "tool_call_id": "function-call-UUID-2",
                    },
                ],
            },
        )
        # Model Content: 2 functionCall Parts, in order.
        model_content = result["contents"][1]
        assert model_content["role"] == "model"
        assert model_content["parts"] == [
            {
                "functionCall": {
                    "id": "function-call-UUID-1",
                    "name": "get_current_weather",
                    "args": {"location": "New Delhi"},
                }
            },
            {
                "functionCall": {
                    "id": "function-call-UUID-2",
                    "name": "get_current_weather_2",
                    "args": {"location": "San Francisco"},
                }
            },
        ]
        # Tool-role Content: 2 functionResponse Parts under ONE
        # Content (adjacent-same-role merging). Second tool response
        # is plain text -> wrapped as {"result": "..."}.
        tool_content = result["contents"][2]
        assert "role" not in tool_content
        assert tool_content["parts"] == [
            {
                "functionResponse": {
                    "name": "get_current_weather",
                    "response": {"temperature": 30.5, "unit": "C"},
                    "id": "function-call-UUID-1",
                }
            },
            {
                "functionResponse": {
                    "name": "get_current_weather_2",
                    "response": {"result": "20 degrees C"},
                    "id": "function-call-UUID-2",
                }
            },
        ]

    def test_tool_response_with_unknown_id_uses_empty_name(self):
        # If a tool response arrives without a matching prior
        # tool_call (id not in the tool_call_id_to_name map), emit
        # an empty ``name`` string on the functionResponse.
        result = convert_openai_request_to_gemini(
            context=None,
            json_input={
                "model": "m",
                "messages": [
                    {
                        "role": "tool",
                        "tool_call_id": "orphan-id",
                        "content": '{"x": 1}',
                    }
                ],
            },
        )
        assert result["contents"] == [
            {
                "parts": [
                    {
                        "functionResponse": {
                            "name": "",
                            "response": {"x": 1},
                            "id": "orphan-id",
                        }
                    }
                ]
            }
        ]

    def test_legacy_function_call_field_on_assistant_message(self):
        # Legacy: assistant message with a `function_call` field (dict
        # form, {"name": ..., "arguments": ...}). Emits a
        # functionCall Part just like tool_calls but no id.
        result = convert_openai_request_to_gemini(
            context=None,
            json_input={
                "model": "m",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "function_call": {
                            "name": "get_weather",
                            "arguments": '{"city": "NYC"}',
                        },
                    },
                ],
            },
        )
        assert result["contents"][1] == {
            "role": "model",
            "parts": [
                {
                    "functionCall": {
                        "name": "get_weather",
                        "args": {"city": "NYC"},
                    }
                }
            ],
        }

    def test_empty_arguments_string_becomes_empty_args_dict(self):
        # OpenAI clients sometimes send arguments="" for zero-arg
        # functions. Result: args={} rather than an error.
        result = convert_openai_request_to_gemini(
            context=None,
            json_input={
                "model": "m",
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "x",
                                "type": "function",
                                "function": {"name": "no_args", "arguments": ""},
                            }
                        ],
                    }
                ],
            },
        )
        fn_call = result["contents"][0]["parts"][0]["functionCall"]
        assert fn_call == {"id": "x", "name": "no_args", "args": {}}

    def test_invalid_json_arguments_raises(self):
        # Malformed JSON in arguments -> ValueError (no wrap_invalid
        # fallback for function_call args; that's tool-response only).
        with pytest.raises(ValueError, match="valid JSON"):
            convert_openai_request_to_gemini(
                context=None,
                json_input={
                    "model": "m",
                    "messages": [
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "x",
                                    "type": "function",
                                    "function": {
                                        "name": "f",
                                        "arguments": "{not valid json",
                                    },
                                }
                            ],
                        }
                    ],
                },
            )


class TestConvertRequestThoughtSignature:
    """thought_signature + thought preservation on Parts.

    Verifies the following behaviors of the OpenAI Google-extension
    namespace (``extra_content.google.{thought, thought_signature}``)
    translating into Vertex Part fields (``thoughtSignature``,
    ``thought``):

      * Thought-only assistant messages (no content) yield a
        synthetic Part carrying just the thought/signature. A
        thought-only-with-NO-signature message is dropped (empty
        text + no signature + no functionCall/functionResponse
        would produce an empty Part).
      * Message-level and per-tool_call extra_content apply
        independently: message-level lands on text/refusal Parts,
        per-tool_call lands on functionCall Parts. Message-level
        does NOT propagate to functionCall Parts.
      * The dummy-signature bypass value
        ``"skip_thought_signature_validator"`` is SKIPPED (not written
        to the Part's ``thoughtSignature`` field) — the sentinel string
        is not valid base64, and ``Part.thought_signature`` is a proto
        ``bytes`` field, so writing it would be rejected by
        ``GenerateContentRequest.from_json``.
      * ``thought: false`` is a no-op — the Part must not carry a
        ``thought`` key.

    Wire shape:

      * ``extra_content.google.thought_signature`` is a
        base64-encoded string on the wire; passed through verbatim
        (NOT decoded) to the Part's ``thoughtSignature`` field, because
        proto3 JSON encodes bytes fields as base64 strings, so the b64
        input IS already the correct wire form.
      * ``extra_content.google.thought`` is a bool. Only emitted on
        the Part when true (Vertex's default is false, so explicit
        false would just add noise).
      * Empty signature (``""``) → not emitted (downstream Vertex
        ignores empty ``thoughtSignature`` fields anyway).
    """

    def test_thought_signature_and_thought_only_parts(self):
        """Thought-only assistant messages: preserved iff a signature is set."""
        result = convert_openai_request_to_gemini(
            context=None,
            json_input={
                "model": "google/gemini-2.5-flash",
                "messages": [
                    {"role": "system", "content": "A"},
                    {"role": "user", "content": "B"},
                    {"role": "user", "content": "C"},
                    {
                        "role": "assistant",
                        "extra_content": {"google": {"thought": True}},
                    },
                    {
                        "role": "assistant",
                        "extra_content": {
                            "google": {
                                "thought": True,
                                "thought_signature": "QQ==",  # base64 "A"
                            }
                        },
                    },
                    {"role": "assistant", "content": "E"},
                ],
            },
        )
        assert result["model"] == "google/gemini-2.5-flash"
        assert result["systemInstruction"] == {
            "role": "system",
            "parts": [{"text": "A"}],
        }
        # First thought-only-no-sig assistant message is dropped
        # (empty text + no signature + no fc/fr = empty Part);
        # second (thought: true + sig "A") becomes 1 Part; third
        # (text "E") becomes 1 Part; both merge into ONE assistant
        # Content (same role).
        assert result["contents"] == [
            {"role": "user", "parts": [{"text": "B"}, {"text": "C"}]},
            {
                "role": "model",
                "parts": [
                    # thoughtSignature is written verbatim as base64 —
                    # proto3 JSON encodes bytes fields ("QQ=="), NOT
                    # decoded to raw text ("A").
                    {"thought": True, "thoughtSignature": "QQ=="},
                    {"text": "E"},
                ],
            },
        ]

    def test_thought_signature_with_content_and_tool_call_success(self):
        """Content + tool_call messages carrying message-level + per-tool_call extras.

        Exercises message-level + per-tool_call extra_content, the
        dummy-signature bypass, and combined thought: true +
        thought_signature on the same Part.
        """
        result = convert_openai_request_to_gemini(
            context=None,
            json_input={
                "model": "google/gemini-2.5-flash",
                "messages": [
                    {
                        "role": "assistant",
                        "content": "Content 1",
                        "extra_content": {
                            "google": {
                                # Dummy signature: passed through verbatim,
                                # NOT base64-decoded.
                                "thought_signature": (
                                    "skip_thought_signature_validator"
                                )
                            }
                        },
                    },
                    {
                        "role": "assistant",
                        "content": "Content 2",
                        "extra_content": {
                            "google": {
                                "thought_signature": "Qg==",  # base64 "B"
                                "thought": True,
                            }
                        },
                    },
                    {
                        "role": "assistant",
                        "content": "Content 3",
                        "extra_content": {"google": {"thought": True}},
                    },
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "type": "function",
                                "id": "tool_id_0",
                                "index": 0,
                                "function": {
                                    "name": "thoughtful_log",
                                    "arguments": ('{"msg":"Thoughtful Hello World."}'),
                                },
                                "extra_content": {"google": {"thought": True}},
                            }
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": "Content",
                        "tool_calls": [
                            {
                                "type": "function",
                                "id": "tool_id_1",
                                "index": 0,
                                "function": {
                                    "name": "log",
                                    "arguments": '{"msg":"Hello World."}',
                                },
                            }
                        ],
                        "extra_content": {
                            "google": {"thought_signature": "Qw=="}  # "C"
                        },
                    },
                ],
            },
        )
        # All 5 assistant messages merge into ONE Content (adjacent same-role).
        assert len(result["contents"]) == 1
        content = result["contents"][0]
        # use_model_role defaults to True -> assistant is rewritten to model.
        assert content["role"] == "model"
        assert content["parts"] == [
            # msg 1: text + dummy signature — bypass sentinel is
            # SKIPPED (not written to thoughtSignature). The sentinel
            # is not valid base64, and Part.thought_signature is a
            # proto bytes field, so writing the sentinel string would
            # be rejected by GenerateContentRequest.from_json.
            {"text": "Content 1"},
            # msg 2: text + b64 signature verbatim + thought.
            # thoughtSignature is written as the base64 string itself
            # (proto3 JSON wire form for bytes fields), NOT decoded to
            # raw text.
            {"text": "Content 2", "thought": True, "thoughtSignature": "Qg=="},
            # msg 3: text + thought (no signature key)
            {"text": "Content 3", "thought": True},
            # msg 4: functionCall with per-tool_call thought: true.
            # The tool_call id ("tool_id_0") is carried onto the
            # Part's functionCall payload.
            {
                "functionCall": {
                    "id": "tool_id_0",
                    "name": "thoughtful_log",
                    "args": {"msg": "Thoughtful Hello World."},
                },
                "thought": True,
            },
            # msg 5: content + message-level signature -> text Part gets the
            # signature; functionCall Part does NOT (per-tool_call was empty).
            {"text": "Content", "thoughtSignature": "Qw=="},
            {
                "functionCall": {
                    "id": "tool_id_1",
                    "name": "log",
                    "args": {"msg": "Hello World."},
                }
            },
        ]

    def test_thought_false_is_noop(self):
        """``thought: false`` on a message → Part must NOT carry a ``thought`` key.

        Vertex's default for ``thought`` is false, so explicitly
        emitting it would just add wire-format noise.
        """
        result = convert_openai_request_to_gemini(
            context=None,
            json_input={
                "model": "google/gemini-2.5-flash",
                "messages": [
                    {
                        "role": "assistant",
                        "content": "Content",
                        "tool_calls": [
                            {
                                "type": "function",
                                "id": "tool_id_1",
                                "index": 0,
                                "function": {
                                    "name": "log",
                                    "arguments": '{"msg":"Hello World."}',
                                },
                            }
                        ],
                        # thought:false must NOT propagate as a key.
                        "extra_content": {"google": {"thought": False}},
                    }
                ],
            },
        )
        assert result["contents"] == [
            {
                "role": "model",
                "parts": [
                    {"text": "Content"},  # No `thought` key.
                    {
                        "functionCall": {
                            "id": "tool_id_1",
                            "name": "log",
                            "args": {"msg": "Hello World."},
                        }
                    },
                ],
            }
        ]

    def test_empty_extra_content_is_noop(self):
        """No extra_content on any message → no thought/thoughtSignature keys."""
        result = convert_openai_request_to_gemini(
            context=None,
            json_input={
                "model": "m",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ],
            },
        )
        for content in result["contents"]:
            for part in content["parts"]:
                assert "thought" not in part
                assert "thoughtSignature" not in part

    def test_empty_signature_string_is_skipped(self):
        """thought_signature: "" must NOT emit a thoughtSignature key.

        Empty signatures are dropped because downstream Vertex
        ignores empty ``thoughtSignature`` fields anyway.
        """
        result = convert_openai_request_to_gemini(
            context=None,
            json_input={
                "model": "m",
                "messages": [
                    {
                        "role": "assistant",
                        "content": "hi",
                        "extra_content": {"google": {"thought_signature": ""}},
                    }
                ],
            },
        )
        part = result["contents"][0]["parts"][0]
        assert part == {"text": "hi"}

    def test_thought_signature_roundtrip_preserves_fc_alignment_with_text_parts(self):
        """Regression: response emits text+fc parts each with their own
        signature; only the FC-part signature should appear in
        provider_specific_fields.thought_signatures (matching tool_calls
        order), so the request-side bridge attaches the RIGHT signature
        to the RIGHT tool_call.

        Pre-fix behavior emitted signatures in parts-order, so a
        multi-turn history with a text-part signature would attach
        text_sig to tool_calls[0] and Vertex would reject the request
        with "signature does not match" on the functionCall Part.
        """
        import base64
        from google.cloud.aiplatform_v1.types import (
            Candidate,
            Content,
            FunctionCall,
            GenerateContentResponse,
            Part,
        )
        from google.protobuf.struct_pb2 import Struct
        from backends.chat_completion import (
            convert_gemini_response_to_openai,
        )

        text_sig = b"text_signature_bytes"
        fc_sig = b"fc_signature_bytes"
        fc_sig_b64 = base64.b64encode(fc_sig).decode("ascii")

        args = Struct()
        args.update({"command": "ls"})
        text_part = Part(text="Some reasoning", thought_signature=text_sig)
        fc_part = Part(
            function_call=FunctionCall(name="bash", args=args),
            thought_signature=fc_sig,
        )
        candidate = Candidate(
            content=Content(role="model", parts=[text_part, fc_part]),
            finish_reason=Candidate.FinishReason.STOP,
        )
        response = GenerateContentResponse()
        response.candidates.append(candidate)
        response.usage_metadata.prompt_token_count = 1
        response.usage_metadata.candidates_token_count = 1
        response.usage_metadata.total_token_count = 2
        response.model_version = "gemini-2.5-pro"
        response.response_id = "resp-mixed-parts"

        # 1. Response side must emit exactly ONE signature (the fc one).
        chat = convert_gemini_response_to_openai(response)
        msg = chat.choices[0].message
        assert msg.tool_calls and len(msg.tool_calls) == 1

        assert msg.provider_specific_fields is not None
        sigs = msg.provider_specific_fields.thought_signatures
        assert sigs == [fc_sig_b64], (
            f"Expected exactly [fc_sig_b64], got {sigs}. Response side "
            f"must emit only fc-Part signatures in thought_signatures "
            f"so the request-side bridge maps signatures to tool_calls "
            f"correctly."
        )

        # 2. Full roundtrip: feed this response back into a next-turn
        # request and verify the outbound functionCall Part carries the
        # FC signature (NOT the text signature).
        chat_dict = chat.to_dict()
        assistant_dict = chat_dict["choices"][0]["message"]
        req_turn2 = {
            "model": "g",
            "messages": [
                {"role": "user", "content": "hi"},
                assistant_dict,
                {
                    "role": "tool",
                    "tool_call_id": assistant_dict["tool_calls"][0]["id"],
                    "content": "out",
                },
                {"role": "user", "content": "next"},
            ],
        }
        vertex_dict = convert_openai_request_to_gemini(
            context=None, json_input=req_turn2
        )
        # Find the assistant turn (may be role="assistant" or "model"
        # depending on ConversionContext.use_model_role; default is
        # "assistant"). Filter by "role" presence + value to skip the
        # tool-response turn (which is emitted without an explicit role).
        assistant_turns = [
            c
            for c in vertex_dict["contents"]
            if c.get("role") in ("assistant", "model")
        ]
        assert len(assistant_turns) == 1
        fc_parts_out = [p for p in assistant_turns[0]["parts"] if "functionCall" in p]
        assert len(fc_parts_out) == 1
        got_sig_b64 = fc_parts_out[0].get("thoughtSignature")
        assert got_sig_b64 is not None, (
            "Outbound functionCall Part missing thoughtSignature after "
            "roundtrip; bridge did not attach the fc signature."
        )
        assert base64.b64decode(got_sig_b64) == fc_sig, (
            f"Wrong signature on fc part after roundtrip. Got "
            f"{base64.b64decode(got_sig_b64)!r}, expected {fc_sig!r}. "
            f"This means text-part signatures leaked into the "
            f"thought_signatures list and misaligned the tc_index bridge."
        )

    def test_provider_specific_sigs_align_across_non_dict_tool_call_entries(self):
        """Regression: with interleaved non-dict tool_calls entries like
        ``[dict1, "garbage", dict2]``, the length check (which compares
        against the *filtered* dict-only list) passes ``2 == 2``, and
        the loop must therefore also iterate the filtered list so
        ``tc_index`` maps ``dict1 -> sigA`` and ``dict2 -> sigB``.

        Pre-fix: the loop iterated the unfiltered ``tool_calls_field``,
        skipped the non-dict via ``continue``, and read
        ``provider_specific_thought_sigs[2]`` for dict2 (out of bounds,
        silently None). Result: dict2 landed with no signature and sigB
        was silently dropped.

        Low practical risk (real OpenAI clients never emit non-dict
        entries), but the code was misleading — the length check
        validated an alignment the loop didn't preserve.
        """
        import base64

        sig_a_bytes = b"sig-A-bytes"
        sig_b_bytes = b"sig-B-bytes"
        sig_a_b64 = base64.b64encode(sig_a_bytes).decode("ascii")
        sig_b_b64 = base64.b64encode(sig_b_bytes).decode("ascii")
        req = {
            "model": "g",
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_a",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        },
                        "garbage-non-dict-entry",
                        {
                            "id": "call_b",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        },
                    ],
                    "provider_specific_fields": {
                        # Two sigs, two dict-shaped tool_calls → length
                        # check passes; the loop must map sigA→dict1
                        # (tc_index=0 on the filtered list) and sigB→
                        # dict2 (tc_index=1 on the filtered list).
                        "thought_signatures": [sig_a_b64, sig_b_b64],
                    },
                },
                {"role": "tool", "tool_call_id": "call_a", "content": "outA"},
                {"role": "tool", "tool_call_id": "call_b", "content": "outB"},
                {"role": "user", "content": "next"},
            ],
        }
        vertex_dict = convert_openai_request_to_gemini(context=None, json_input=req)
        assistant_turns = [
            c
            for c in vertex_dict["contents"]
            if c.get("role") in ("assistant", "model")
        ]
        assert len(assistant_turns) == 1
        fc_parts_out = [p for p in assistant_turns[0]["parts"] if "functionCall" in p]
        assert len(fc_parts_out) == 2, (
            f"expected 2 functionCall Parts (one per dict tool_call), got "
            f"{len(fc_parts_out)}"
        )
        # First fc Part must carry sigA.
        sig0_b64 = fc_parts_out[0].get("thoughtSignature")
        assert sig0_b64 is not None, (
            "first functionCall Part missing thoughtSignature; sigA was "
            "not attached to dict1"
        )
        assert base64.b64decode(sig0_b64) == sig_a_bytes, (
            f"first fc Part signature mismatch: got "
            f"{base64.b64decode(sig0_b64)!r}, expected {sig_a_bytes!r}"
        )
        # Second fc Part must carry sigB (pre-fix would have silently
        # dropped this because the unfiltered loop read sigs[2] here).
        sig1_b64 = fc_parts_out[1].get("thoughtSignature")
        assert sig1_b64 is not None, (
            "second functionCall Part missing thoughtSignature; sigB was "
            "silently dropped because the loop iterated the unfiltered "
            "tool_calls list and read out-of-bounds at index 2"
        )
        assert base64.b64decode(sig1_b64) == sig_b_bytes, (
            f"second fc Part signature mismatch: got "
            f"{base64.b64decode(sig1_b64)!r}, expected {sig_b_bytes!r}"
        )

    def test_provider_specific_thought_sigs_length_mismatch_raises(self):
        """Defensive check: if the caller hands us a message whose
        provider_specific_fields.thought_signatures list length does
        not equal the number of tool_calls, raise ValueError rather
        than silently attach the wrong signature.

        Prevents pre-fix corrupted histories (or callers who mutated
        tool_calls without updating the sig list) from being sent to
        Vertex where they'd be rejected with an opaque server error.
        """
        req = {
            "model": "g",
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_a",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ],
                    "provider_specific_fields": {
                        # Two sigs but only one tool_call → mismatch.
                        "thought_signatures": ["QQ==", "Qg=="],
                    },
                },
                {"role": "tool", "tool_call_id": "call_a", "content": "out"},
                {"role": "user", "content": "next"},
            ],
        }
        with pytest.raises(ValueError, match="thought_signatures length"):
            convert_openai_request_to_gemini(context=None, json_input=req)

    def test_provider_specific_thought_sigs_non_string_entry_raises(self):
        """Defensive check: a mid-list None / non-string entry in
        provider_specific_fields.thought_signatures MUST raise
        ValueError at the read boundary, not silently drop the
        signature for that tool_call.

        Without this guard, the request-side loop's
        ``isinstance(sig_from_psf, str)`` check would silently skip
        the None entry and produce a functionCall Part with no
        thoughtSignature while adjacent Parts kept theirs — a wrong-
        signature-attachment class of bug that Vertex would then
        reject with an opaque "signature mismatch" on the affected
        Part. Fail loud at read boundary so the caller can fix their
        envelope construction locally instead of chasing an opaque
        server error.
        """
        req = {
            "model": "g",
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_a",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        },
                        {
                            "id": "call_b",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        },
                    ],
                    "provider_specific_fields": {
                        # length matches tool_calls but the middle entry
                        # is None instead of a string.
                        "thought_signatures": ["QQ==", None],
                    },
                },
                {"role": "tool", "tool_call_id": "call_a", "content": "out"},
                {"role": "tool", "tool_call_id": "call_b", "content": "out"},
                {"role": "user", "content": "next"},
            ],
        }
        with pytest.raises(
            ValueError, match="thought_signatures\\[1\\] must be a string"
        ):
            convert_openai_request_to_gemini(context=None, json_input=req)


class TestConvertRequestMultimodal:
    """Verifies multimodal content-part translation.

    Scope is intentionally limited to the shapes Vertex's
    ``generateContent`` API accepts:
      * ``data:<mime>;base64,<payload>`` URIs for inline blobs.
      * ``gs://<bucket>/<path>`` URIs for referenced files.

    Other URL forms (``http://``, ``https://``, YouTube URLs, string
    shorthand) are rejected at the boundary with a clear ``ValueError``
    so callers get immediate feedback instead of a downstream Vertex
    400.

    Covers: image_url data URI happy paths + payload validation,
    image_url gs:// + rejections, ``detail`` handling, input_audio,
    and assistant.audio.
    """

    def test_image_url_data_uri_png(self):
        """PNG data URI decodes to inlineData with the correct MIME + blob."""
        result = convert_openai_request_to_gemini(
            context=None,
            json_input={
                "model": "m",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": ("data:image/png;base64,iVBORw0KGgo=")
                                },
                            }
                        ],
                    }
                ],
            },
        )
        part = result["contents"][0]["parts"][0]
        assert part == {
            "inlineData": {
                "mimeType": "image/png",
                "data": "iVBORw0KGgo=",
            }
        }

    def test_image_url_data_uri_various_mime_types(self):
        """MIME parsing covers jpeg, gif, webp."""
        for mime, payload in [
            ("image/jpeg", "/9j/4AAQ"),
            ("image/gif", "R0lGODlh"),
            ("image/webp", "UklGRg=="),
        ]:
            result = convert_openai_request_to_gemini(
                context=None,
                json_input={
                    "model": "m",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime};base64,{payload}"
                                    },
                                }
                            ],
                        }
                    ],
                },
            )
            part = result["contents"][0]["parts"][0]
            assert part["inlineData"]["mimeType"] == mime
            assert part["inlineData"]["data"] == payload

    def test_image_url_data_uri_invalid_base64(self):
        """Malformed base64 payload → ValueError."""
        with pytest.raises(ValueError, match="valid base64"):
            convert_openai_request_to_gemini(
                context=None,
                json_input={
                    "model": "m",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": ("data:image/png;base64,!!!not-base64")
                                    },
                                }
                            ],
                        }
                    ],
                },
            )

    def test_image_url_data_uri_empty_payload(self):
        """Data URI with empty base64 payload → ValueError."""
        with pytest.raises(ValueError, match="empty base64 payload"):
            convert_openai_request_to_gemini(
                context=None,
                json_input={
                    "model": "m",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "data:image/png;base64,"},
                                }
                            ],
                        }
                    ],
                },
            )

    def test_image_url_gs_uri(self):
        """gs:// URI → fileData Part with default image/png MIME."""
        result = convert_openai_request_to_gemini(
            context=None,
            json_input={
                "model": "m",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "gs://bucket/img.png"},
                            }
                        ],
                    }
                ],
            },
        )
        part = result["contents"][0]["parts"][0]
        assert part == {
            "fileData": {
                "mimeType": "image/png",
                "fileUri": "gs://bucket/img.png",
            }
        }

    def test_image_url_https_raises(self):
        """https:// URL → ValueError (Vertex requires data: or gs://)."""
        with pytest.raises(ValueError, match="data:.*base64.*gs://"):
            convert_openai_request_to_gemini(
                context=None,
                json_input={
                    "model": "m",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "https://example.com/img.png"},
                                }
                            ],
                        }
                    ],
                },
            )

    def test_image_url_youtube_raises(self):
        """YouTube URL → ValueError (no auto-video-metadata treatment)."""
        with pytest.raises(ValueError, match="data:.*base64.*gs://"):
            convert_openai_request_to_gemini(
                context=None,
                json_input={
                    "model": "m",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": "https://youtu.be/dQw4w9WgXcQ"
                                    },
                                }
                            ],
                        }
                    ],
                },
            )

    def test_image_url_string_shorthand_raises(self):
        """image_url as bare string (not object) → ValueError."""
        with pytest.raises(ValueError, match="image_url must be an object"):
            convert_openai_request_to_gemini(
                context=None,
                json_input={
                    "model": "m",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": "gs://bucket/img.png",
                                }
                            ],
                        }
                    ],
                },
            )

    def test_image_url_missing_url_field_raises(self):
        """image_url object without a `url` sub-field → ValueError."""
        with pytest.raises(ValueError, match="image_url.url is required"):
            convert_openai_request_to_gemini(
                context=None,
                json_input={
                    "model": "m",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"detail": "high"},
                                }
                            ],
                        }
                    ],
                },
            )

    def test_image_url_detail_per_part_on_gemini_3(self):
        """Gemini 3+ model → mediaResolution lands on the Part itself."""
        result = convert_openai_request_to_gemini(
            context=ConversionContext(large_model_name="gemini-3-pro"),
            json_input={
                "model": "gemini-3-pro",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "gs://b/a.png",
                                    "detail": "high",
                                },
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "gs://b/b.png",
                                    "detail": "low",
                                },
                            },
                        ],
                    }
                ],
            },
        )
        parts = result["contents"][0]["parts"]
        assert parts[0] == {
            "fileData": {"mimeType": "image/png", "fileUri": "gs://b/a.png"},
            "mediaResolution": {"level": "MEDIA_RESOLUTION_HIGH"},
        }
        assert parts[1] == {
            "fileData": {"mimeType": "image/png", "fileUri": "gs://b/b.png"},
            "mediaResolution": {"level": "MEDIA_RESOLUTION_LOW"},
        }
        # Nothing on generationConfig for Gemini 3+.
        assert "mediaResolution" not in result.get("generationConfig", {})

    def test_image_url_detail_request_level_on_gemini_2(self):
        """Older Gemini → detail promoted to generationConfig.mediaResolution."""
        result = convert_openai_request_to_gemini(
            context=ConversionContext(large_model_name="gemini-2.5-flash"),
            json_input={
                "model": "gemini-2.5-flash",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "gs://b/a.png",
                                    "detail": "high",
                                },
                            }
                        ],
                    }
                ],
            },
        )
        # detail is NOT on the Part.
        assert "mediaResolution" not in result["contents"][0]["parts"][0]
        # It IS on generationConfig.
        assert result["generationConfig"]["mediaResolution"] == "MEDIA_RESOLUTION_HIGH"

    def test_image_url_detail_auto_or_missing_no_op(self):
        """detail: auto (or absent) → no mediaResolution anywhere."""
        for detail_kwarg in [{}, {"detail": "auto"}]:
            image_url_obj = {"url": "gs://b/a.png", **detail_kwarg}
            result = convert_openai_request_to_gemini(
                context=None,
                json_input={
                    "model": "m",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": image_url_obj,
                                }
                            ],
                        }
                    ],
                },
            )
            assert "mediaResolution" not in result["contents"][0]["parts"][0]
            assert "mediaResolution" not in result.get("generationConfig", {})

    def test_image_url_detail_inconsistent_on_gemini_2_raises(self):
        """Two images with different `detail` on Gemini 2.x → ValueError."""
        with pytest.raises(ValueError, match="inconsistent.*detail"):
            convert_openai_request_to_gemini(
                context=ConversionContext(large_model_name="gemini-2.5-flash"),
                json_input={
                    "model": "gemini-2.5-flash",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": "gs://b/a.png",
                                        "detail": "high",
                                    },
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": "gs://b/b.png",
                                        "detail": "low",
                                    },
                                },
                            ],
                        }
                    ],
                },
            )

    def test_input_audio_mp3_base64(self):
        """format:mp3 + base64 data → inlineData with audio/mpeg MIME."""
        result = convert_openai_request_to_gemini(
            context=None,
            json_input={
                "model": "m",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": "aGVsbG8=",  # "hello"
                                    "format": "mp3",
                                },
                            }
                        ],
                    }
                ],
            },
        )
        assert result["contents"][0]["parts"][0] == {
            "inlineData": {"mimeType": "audio/mpeg", "data": "aGVsbG8="}
        }

    def test_input_audio_wav_base64(self):
        """format:wav + base64 data → inlineData with audio/wav MIME."""
        result = convert_openai_request_to_gemini(
            context=None,
            json_input={
                "model": "m",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": "UklGRg==",
                                    "format": "wav",
                                },
                            }
                        ],
                    }
                ],
            },
        )
        assert result["contents"][0]["parts"][0] == {
            "inlineData": {"mimeType": "audio/wav", "data": "UklGRg=="}
        }

    def test_input_audio_custom_mime(self):
        """format:<mime-string> passes through as MIME verbatim."""
        result = convert_openai_request_to_gemini(
            context=None,
            json_input={
                "model": "m",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": "Zm9vYmFy",  # "foobar"
                                    "format": "audio/flac",
                                },
                            }
                        ],
                    }
                ],
            },
        )
        assert result["contents"][0]["parts"][0] == {
            "inlineData": {"mimeType": "audio/flac", "data": "Zm9vYmFy"}
        }

    def test_input_audio_gs_uri(self):
        """gs:// URI in input_audio.data → fileData with the format-derived MIME."""
        result = convert_openai_request_to_gemini(
            context=None,
            json_input={
                "model": "m",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": "gs://bucket/foo.mp3",
                                    "format": "mp3",
                                },
                            }
                        ],
                    }
                ],
            },
        )
        assert result["contents"][0]["parts"][0] == {
            "fileData": {
                "mimeType": "audio/mpeg",
                "fileUri": "gs://bucket/foo.mp3",
            }
        }

    def test_input_audio_https_raises(self):
        """https:// URL in input_audio.data → ValueError."""
        with pytest.raises(ValueError, match="base64.*or.*gs://"):
            convert_openai_request_to_gemini(
                context=None,
                json_input={
                    "model": "m",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_audio",
                                    "input_audio": {
                                        "data": "https://example.com/foo.mp3",
                                        "format": "mp3",
                                    },
                                }
                            ],
                        }
                    ],
                },
            )

    def test_input_audio_invalid_base64_raises(self):
        """Malformed base64 in input_audio.data → ValueError."""
        with pytest.raises(ValueError, match="not valid base64"):
            convert_openai_request_to_gemini(
                context=None,
                json_input={
                    "model": "m",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_audio",
                                    "input_audio": {
                                        "data": "!!!not-base64!!!",
                                        "format": "mp3",
                                    },
                                }
                            ],
                        }
                    ],
                },
            )

    def test_assistant_audio_id_base64(self):
        """Assistant message with audio.id (base64) → inlineData Part."""
        result = convert_openai_request_to_gemini(
            context=None,
            json_input={
                "model": "m",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "content": "here you go",
                        "audio": {"id": "SGVsbG8gV29ybGQ="},  # "Hello World"
                    },
                ],
            },
        )
        # Assistant Content has 2 Parts: the text + the audio.
        parts = result["contents"][1]["parts"]
        assert parts[0] == {"text": "here you go"}
        assert parts[1] == {
            "inlineData": {
                "mimeType": "audio/mpeg",
                "data": "SGVsbG8gV29ybGQ=",
            }
        }

    def test_assistant_audio_id_invalid_base64_raises(self):
        """Assistant audio.id that isn't valid base64 → ValueError."""
        with pytest.raises(ValueError, match="not valid base64"):
            convert_openai_request_to_gemini(
                context=None,
                json_input={
                    "model": "m",
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "here you go",
                            "audio": {"id": "!!!not-base64!!!"},
                        }
                    ],
                },
            )

    def test_thought_signature_with_image_part_raises(self):
        """A message with thought_signature AND a media Part → ValueError.

        Thought signatures are model-output semantics that only apply
        to text/refusal Parts. A message combining message-level
        extra_content.google.thought_signature with an image/audio
        Part is ambiguous and rejected explicitly.
        """
        with pytest.raises(ValueError, match="thought_signature.*text/refusal"):
            convert_openai_request_to_gemini(
                context=None,
                json_input={
                    "model": "m",
                    "messages": [
                        {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "look at this"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "gs://b/a.png"},
                                },
                            ],
                            "extra_content": {"google": {"thought_signature": "QQ=="}},
                        }
                    ],
                },
            )

    def test_thought_signature_with_assistant_audio_raises(self):
        """A message with thought_signature AND an audio field → ValueError.

        Same rationale as test_thought_signature_with_image_part_raises,
        but exercises the assistant-message-level ``audio`` field code
        path rather than the ``image_url`` content-part path.
        """
        with pytest.raises(ValueError, match="thought_signature.*text/refusal"):
            convert_openai_request_to_gemini(
                context=None,
                json_input={
                    "model": "m",
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "here you go",
                            "audio": {"id": "SGVsbG8gV29ybGQ="},
                            "extra_content": {"google": {"thought_signature": "QQ=="}},
                        }
                    ],
                },
            )


class TestResponseDataclasses:
    """Verify dataclass definitions and .to_dict() None-pruning behavior."""

    def test_thinking_block_default_type(self):
        from backends.chat_completion import ThinkingBlock

        tb = ThinkingBlock(thinking="reasoning...")
        assert tb.type == "thinking"
        assert tb.thinking == "reasoning..."
        assert tb.signature is None

    def test_function_call_required_fields(self):
        from backends.chat_completion import FunctionCall

        fc = FunctionCall(name="do_thing", arguments='{"x": 1}')
        assert fc.name == "do_thing"
        assert fc.arguments == '{"x": 1}'

    def test_tool_call_default_type(self):
        from backends.chat_completion import ToolCall, FunctionCall

        tc = ToolCall(id="call_abc", function=FunctionCall(name="foo", arguments="{}"))
        assert tc.type == "function"

    def test_provider_specific_fields_default_none(self):
        from backends.chat_completion import ProviderSpecificFields

        psf = ProviderSpecificFields()
        assert psf.thought_signatures is None

    def test_message_defaults(self):
        from backends.chat_completion import Message

        m = Message()
        assert m.role == "assistant"
        assert m.content is None
        assert m.tool_calls is None
        assert m.thinking_blocks is None
        assert m.provider_specific_fields is None

    def test_choice_required_fields(self):
        from backends.chat_completion import Choice, Message

        c = Choice(index=0, message=Message(content="hi"), finish_reason="stop")
        assert c.index == 0
        assert c.finish_reason == "stop"

    def test_usage_optional_details(self):
        from backends.chat_completion import Usage

        u = Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3)
        assert u.completion_tokens_details is None
        assert u.prompt_tokens_details is None

    def test_chat_completion_response_defaults(self):
        from backends.chat_completion import ChatCompletionResponse, Choice, Message

        r = ChatCompletionResponse(
            id="chatcmpl-x",
            created=1000,
            model="gemini-x",
            choices=[
                Choice(index=0, message=Message(content="hi"), finish_reason="stop")
            ],
        )
        assert r.object == "chat.completion"
        assert r.usage is None
        assert r.system_fingerprint is None

    def test_to_dict_drops_none_recursively(self):
        from backends.chat_completion import (
            ChatCompletionResponse,
            Choice,
            Message,
        )

        r = ChatCompletionResponse(
            id="x",
            created=1,
            model="m",
            choices=[
                Choice(index=0, message=Message(content="hi"), finish_reason="stop")
            ],
        )
        d = r.to_dict()
        # Present, non-None:
        assert d["id"] == "x"
        assert d["object"] == "chat.completion"
        assert d["choices"][0]["message"]["content"] == "hi"
        # None-valued fields must be dropped:
        assert "usage" not in d
        assert "system_fingerprint" not in d
        assert "tool_calls" not in d["choices"][0]["message"]
        assert "thinking_blocks" not in d["choices"][0]["message"]
        assert "provider_specific_fields" not in d["choices"][0]["message"]

    def test_to_dict_nested_dataclasses_unwrapped(self):
        from backends.chat_completion import (
            ChatCompletionResponse,
            Choice,
            Message,
            ToolCall,
            FunctionCall,
            ThinkingBlock,
            ProviderSpecificFields,
            Usage,
            CompletionTokensDetails,
            PromptTokensDetails,
        )

        r = ChatCompletionResponse(
            id="x",
            created=1,
            model="m",
            choices=[
                Choice(
                    index=0,
                    message=Message(
                        content=None,
                        tool_calls=[
                            ToolCall(
                                id="call_1",
                                function=FunctionCall(name="foo", arguments='{"a":1}'),
                            )
                        ],
                        thinking_blocks=[ThinkingBlock(thinking="t", signature="sig")],
                        provider_specific_fields=ProviderSpecificFields(
                            thought_signatures=["sig"],
                        ),
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=Usage(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                completion_tokens_details=CompletionTokensDetails(reasoning_tokens=2),
                prompt_tokens_details=PromptTokensDetails(cached_tokens=3),
            ),
        )
        d = r.to_dict()
        msg = d["choices"][0]["message"]
        # Tool call is dict, not dataclass:
        assert msg["tool_calls"][0]["function"]["name"] == "foo"
        # Thinking block:
        assert msg["thinking_blocks"][0]["type"] == "thinking"
        assert msg["thinking_blocks"][0]["signature"] == "sig"
        # PSF:
        assert msg["provider_specific_fields"]["thought_signatures"] == ["sig"]
        # Usage details:
        assert d["usage"]["completion_tokens_details"]["reasoning_tokens"] == 2
        assert d["usage"]["prompt_tokens_details"]["cached_tokens"] == 3
        # content is None so dropped even though the field is set:
        assert "content" not in msg

    def test_to_dict_pruned_recurses_into_dicts(self):
        from backends.chat_completion import (
            _to_dict_pruned,
            ThinkingBlock,
        )

        # Manual test: a dict containing a dataclass value must unwrap it.
        # None of the current dataclasses have Dict[str, X] fields, so the
        # helper's dict branch is exercised directly here.
        input_data = {
            "key1": ThinkingBlock(thinking="t", signature="s"),
            "key2": [ThinkingBlock(thinking="u")],  # list-of-dc inside dict
        }
        out = _to_dict_pruned(input_data)
        assert out == {
            "key1": {"type": "thinking", "thinking": "t", "signature": "s"},
            "key2": [{"type": "thinking", "thinking": "u"}],  # signature=None dropped
        }


def _mk_part(
    *,
    text=None,
    thought=False,
    thought_signature=b"",
    function_call=None,
    inline_data=None,
):
    """Helper: build a Part proto with only the fields you name."""
    from google.cloud.aiplatform_v1.types import Part, FunctionCall as ProtoFC, Blob

    kwargs = {}
    if text is not None:
        kwargs["text"] = text
    if thought:
        kwargs["thought"] = True
    if thought_signature:
        kwargs["thought_signature"] = thought_signature
    if function_call is not None:
        # function_call is a dict {"name": str, "args": dict}
        from google.protobuf.struct_pb2 import Struct

        struct = Struct()
        struct.update(function_call.get("args", {}))
        kwargs["function_call"] = ProtoFC(name=function_call["name"], args=struct)
    if inline_data is not None:
        kwargs["inline_data"] = Blob(
            mime_type=inline_data["mime"], data=inline_data["data"]
        )
    return Part(**kwargs)


def _load_fixture_as_proto(name: str):
    """Load a recorded chat_completion fixture JSON and reshape it into
    a GenerateContentResponse proto matching what Vertex would return."""
    import json
    from pathlib import Path
    import base64
    from google.cloud.aiplatform_v1.types import (
        GenerateContentResponse,
        Candidate,
        Content,
        Part,
        FunctionCall as ProtoFC,
    )
    from google.protobuf.struct_pb2 import Struct

    # NOTE: GenerateContentResponse has a *nested* UsageMetadata type
    # (google.cloud.aiplatform_v1.types.prediction_service.
    # GenerateContentResponse.UsageMetadata), which is distinct from the
    # top-level ``aiplatform_v1.types.UsageMetadata`` used elsewhere. Only
    # the nested one is accepted by GenerateContentResponse(usage_metadata=...).
    UsageMetadata = GenerateContentResponse.UsageMetadata

    fixture_path = Path(__file__).parent / "fixtures" / "chat_completion" / name
    with fixture_path.open() as f:
        data = json.load(f)

    candidates = []
    for i, sample in enumerate(data["response"]):
        completion = sample.get("completion", {})
        parts_in = completion.get("parts", []) or []
        parts_out = []
        for p in parts_in:
            kwargs = {}
            if "text" in p:
                kwargs["text"] = p["text"]
            if p.get("thought"):
                kwargs["thought"] = True
            if "thoughtSignature" in p:
                kwargs["thought_signature"] = base64.b64decode(p["thoughtSignature"])
            if "functionCall" in p:
                fc = p["functionCall"]
                struct = Struct()
                struct.update(fc.get("args", {}))
                kwargs["function_call"] = ProtoFC(name=fc["name"], args=struct)
            parts_out.append(Part(**kwargs))

        reason_str = sample.get("finishReason", "STOP")
        reason = Candidate.FinishReason[reason_str]
        candidates.append(
            Candidate(
                index=i,
                content=Content(
                    role=completion.get("role", "model"),
                    parts=parts_out,
                ),
                finish_reason=reason,
            )
        )

    # Usage metadata: recorded at each sample; take from the first sample
    # (they should agree across samples of the same call).
    first_sample = data["response"][0] if data["response"] else {}
    um_in = first_sample.get("extras", {}).get("usageMetadata", {})
    usage_metadata = UsageMetadata(
        prompt_token_count=um_in.get("promptTokenCount", 0),
        candidates_token_count=um_in.get("candidatesTokenCount", 0),
        total_token_count=um_in.get("totalTokenCount", 0),
        thoughts_token_count=um_in.get("thoughtsTokenCount", 0),
        cached_content_token_count=um_in.get("cachedContentTokenCount", 0),
    )
    return GenerateContentResponse(
        candidates=candidates,
        usage_metadata=usage_metadata,
        model_version="gemini-2.5-pro",  # synthetic; real fixtures don't record it
    )


class TestResponseHelpers:
    """Unit tests for the response-side private helpers."""

    # ── _partition_parts ────────────────────────────────────────────

    def test_partition_empty(self):
        from backends.chat_completion import _partition_parts

        p = _partition_parts([])
        assert p.text_parts == [] and p.thought_parts == []
        assert p.fc_parts == [] and p.fc_signatures == []

    def test_partition_text_only(self):
        from backends.chat_completion import _partition_parts

        parts = [_mk_part(text="hello"), _mk_part(text="world")]
        p = _partition_parts(parts)
        assert len(p.text_parts) == 2
        assert p.thought_parts == [] and p.fc_parts == []
        assert p.fc_signatures == []

    def test_partition_thought_signature_dropped_from_fc_signatures(self):
        """Thought-Part signatures are NOT collected into fc_signatures.

        Only ``function_call`` Parts contribute — dropping thought/text
        signatures keeps ``fc_signatures`` one-to-one with ``fc_parts``
        (and therefore with ``tool_calls`` on the outbound side), which
        is what the request-side bridge assumes.
        """
        from backends.chat_completion import _partition_parts

        parts = [_mk_part(text="reasoning", thought=True, thought_signature=b"sig1")]
        p = _partition_parts(parts)
        assert p.text_parts == []
        assert len(p.thought_parts) == 1
        # Signature on a thought part is intentionally dropped from
        # fc_signatures (the flat carrier); the thought part still
        # surfaces its signature via ThinkingBlock.signature in
        # _build_thinking_blocks.
        assert p.fc_signatures == []

    def test_partition_function_call(self):
        from backends.chat_completion import _partition_parts

        parts = [
            _mk_part(
                function_call={"name": "f", "args": {"a": 1}}, thought_signature=b"sig2"
            )
        ]
        p = _partition_parts(parts)
        assert len(p.fc_parts) == 1
        assert p.fc_signatures == ["c2lnMg=="]  # base64 of b"sig2"

    def test_partition_mixed_only_fc_sigs_kept(self):
        """Regression: with text + thought + fc parts each carrying a
        signature, fc_signatures must contain ONLY the fc-part sig.

        The pre-fix behavior collected all three in parts-order, which
        made the request-side bridge attach the WRONG signature (from
        the text part) to tool_calls[0] on the next turn. This test
        pins the corrected emit shape.
        """
        from backends.chat_completion import _partition_parts

        parts = [
            _mk_part(text="pre", thought_signature=b"text_sig"),
            _mk_part(text="reasoning", thought=True, thought_signature=b"sig1"),
            _mk_part(
                function_call={"name": "f", "args": {}}, thought_signature=b"sig2"
            ),
            _mk_part(text="post"),
        ]
        p = _partition_parts(parts)
        assert len(p.text_parts) == 2  # "pre", "post"
        assert len(p.thought_parts) == 1
        assert len(p.fc_parts) == 1
        # ONLY the fc-part signature. Text-part and thought-part
        # signatures are deliberately dropped (see _partition_parts
        # docstring): they aren't roundtrippable through this carrier
        # and would misalign the tc_index bridge.
        assert p.fc_signatures == ["c2lnMg=="]  # base64 of b"sig2"

    def test_partition_no_signature_no_entry(self):
        from backends.chat_completion import _partition_parts

        # Part with text but empty thought_signature bytes → no sig entry.
        parts = [_mk_part(text="hi")]
        p = _partition_parts(parts)
        assert p.fc_signatures == []

    def test_partition_bare_signature_only_no_fc_entry(self):
        """A part with only a thought_signature (no text, fc, or
        thought=True) is silently absorbed: it produces no per-kind
        list entry AND (post-fix) no fc_signatures entry either.

        Guards the ``elif part.thought_signature: continue`` branch.
        """
        from backends.chat_completion import _partition_parts

        parts = [_mk_part(thought_signature=b"solo")]
        p = _partition_parts(parts)
        assert p.text_parts == [] and p.thought_parts == [] and p.fc_parts == []
        # A bare signature has no fc partner, so it does not enter
        # fc_signatures. Same behavior as a text-only part with a
        # signature (silently dropped from the flat carrier).
        assert p.fc_signatures == []

    def test_partition_unrecognized_part_raises(self):
        from backends.chat_completion import _partition_parts
        from google.cloud.aiplatform_v1.types import Part

        # An empty Part has nothing recognizable.
        with pytest.raises(ValueError, match="no recognizable payload"):
            _partition_parts([Part()])

    def test_partition_media_part_raises(self):
        from backends.chat_completion import _partition_parts

        parts = [_mk_part(inline_data={"mime": "image/png", "data": b"fake"})]
        with pytest.raises(ValueError, match="contains media"):
            _partition_parts(parts)

    # ── _build_content ──────────────────────────────────────────────

    def test_build_content_empty(self):
        from backends.chat_completion import _build_content

        assert _build_content([]) is None

    def test_build_content_single(self):
        from backends.chat_completion import _build_content

        assert _build_content([_mk_part(text="hi")]) == "hi"

    def test_build_content_concatenates(self):
        from backends.chat_completion import _build_content

        parts = [_mk_part(text="a"), _mk_part(text="b"), _mk_part(text="c")]
        assert _build_content(parts) == "abc"

    def test_build_content_empty_string_parts(self):
        from backends.chat_completion import _build_content

        # A part with text="" should not add anything but also not return None.
        parts = [_mk_part(text="x"), _mk_part(text="")]
        assert _build_content(parts) == "x"

    # ── _build_thinking_blocks ──────────────────────────────────────

    def test_build_thinking_blocks_empty(self):
        from backends.chat_completion import _build_thinking_blocks

        assert _build_thinking_blocks([]) is None

    def test_build_thinking_blocks_with_signature(self):
        from backends.chat_completion import _build_thinking_blocks

        parts = [_mk_part(text="thinking hard", thought=True, thought_signature=b"abc")]
        result = _build_thinking_blocks(parts)
        assert result is not None and len(result) == 1
        assert result[0].type == "thinking"
        assert result[0].thinking == "thinking hard"
        assert result[0].signature == "YWJj"  # base64 of b"abc"

    def test_build_thinking_blocks_without_signature(self):
        from backends.chat_completion import _build_thinking_blocks

        parts = [_mk_part(text="thinking", thought=True)]  # no sig bytes
        result = _build_thinking_blocks(parts)
        assert result is not None and result[0].signature is None

    def test_build_thinking_blocks_multiple(self):
        from backends.chat_completion import _build_thinking_blocks

        parts = [
            _mk_part(text="one", thought=True, thought_signature=b"s1"),
            _mk_part(text="two", thought=True, thought_signature=b"s2"),
        ]
        result = _build_thinking_blocks(parts)
        assert result is not None and len(result) == 2
        assert [tb.thinking for tb in result] == ["one", "two"]

    # ── _build_tool_calls ───────────────────────────────────────────

    def test_build_tool_calls_empty(self):
        from backends.chat_completion import _build_tool_calls

        assert _build_tool_calls([]) is None

    def test_build_tool_calls_no_signature(self):
        from backends.chat_completion import (
            _build_tool_calls,
            _LITELLM_TOOL_ID_SEPARATOR,
        )

        parts = [_mk_part(function_call={"name": "search", "args": {"q": "x"}})]
        result = _build_tool_calls(parts)
        assert result is not None and len(result) == 1
        tc = result[0]
        assert tc.type == "function"
        assert tc.id.startswith("call_")
        assert _LITELLM_TOOL_ID_SEPARATOR not in tc.id
        assert tc.function.name == "search"
        # arguments is a JSON string with default separators:
        import json

        assert json.loads(tc.function.arguments) == {"q": "x"}
        # LiteLLM parity: default json.dumps separators (spaces), NOT compact.
        assert tc.function.arguments == '{"q": "x"}'

    def test_build_tool_calls_with_signature_embeds_in_id(self):
        from backends.chat_completion import (
            _build_tool_calls,
            _LITELLM_TOOL_ID_SEPARATOR,
        )

        parts = [
            _mk_part(
                function_call={"name": "f", "args": {}},
                thought_signature=b"mysig",
            )
        ]
        result = _build_tool_calls(parts)
        assert result is not None
        tc = result[0]
        assert _LITELLM_TOOL_ID_SEPARATOR in tc.id
        prefix, b64 = tc.id.split(_LITELLM_TOOL_ID_SEPARATOR, 1)
        assert prefix.startswith("call_")
        import base64 as b

        assert b.b64decode(b64) == b"mysig"

    def test_build_tool_calls_multiple_independent_ids(self):
        from backends.chat_completion import _build_tool_calls

        parts = [
            _mk_part(function_call={"name": "a", "args": {}}),
            _mk_part(function_call={"name": "b", "args": {}}),
        ]
        result = _build_tool_calls(parts)
        assert result is not None
        assert result[0].id != result[1].id  # random uuids diverge

    def test_build_tool_calls_empty_name_raises(self):
        from backends.chat_completion import _build_tool_calls

        # A function_call with name="" via bare Part() has has_fc=False so
        # it would be caught by _partition_parts, not here. But if we ever
        # get one, be defensive. Test the guard directly.
        from google.cloud.aiplatform_v1.types import Part, FunctionCall as ProtoFC

        parts = [Part(function_call=ProtoFC(name=""))]
        with pytest.raises(ValueError, match="empty name"):
            _build_tool_calls(parts)

    # ── _build_tool_calls: NESTED args (proto Struct deep conversion) ──
    #
    # Regression guard. ``dict(fc.args)`` is a SHALLOW proto-Struct
    # conversion: nested values stay as proto ``RepeatedComposite`` /
    # ``MapComposite`` objects that ``json.dumps`` cannot serialize, so any
    # tool call with a list/dict argument blew up with
    #   "args are not JSON-serializable: Object of type RepeatedComposite ..."
    # Observed live: an opencode agent run lost 10/20 LLM calls this way
    # (agent then burned its full timeout retrying).

    def test_build_tool_calls_args_with_list_value(self):
        from backends.chat_completion import _build_tool_calls

        import json

        parts = [
            _mk_part(
                function_call={"name": "edit", "args": {"files": ["a.txt", "b.txt"]}}
            )
        ]
        result = _build_tool_calls(parts)
        assert result is not None
        assert json.loads(result[0].function.arguments) == {
            "files": ["a.txt", "b.txt"]
        }

    def test_build_tool_calls_args_with_nested_dict_value(self):
        from backends.chat_completion import _build_tool_calls

        import json

        parts = [
            _mk_part(
                function_call={"name": "cfg", "args": {"opts": {"deep": True}}}
            )
        ]
        result = _build_tool_calls(parts)
        assert result is not None
        assert json.loads(result[0].function.arguments) == {"opts": {"deep": True}}

    def test_build_tool_calls_args_deeply_nested_mixed(self):
        """List-of-dicts-of-lists must survive intact."""
        from backends.chat_completion import _build_tool_calls

        import json

        parts = [
            _mk_part(
                function_call={
                    "name": "bulk",
                    "args": {
                        "items": [{"ids": [1, 2]}, {"ids": [3]}],
                        "label": "x",
                        "flag": None,
                    },
                }
            )
        ]
        result = _build_tool_calls(parts)
        assert result is not None
        assert json.loads(result[0].function.arguments) == {
            "items": [{"ids": [1, 2]}, {"ids": [3]}],
            "label": "x",
            "flag": None,
        }

    def test_build_tool_calls_args_empty_stays_empty(self):
        from backends.chat_completion import _build_tool_calls

        import json

        parts = [_mk_part(function_call={"name": "noargs", "args": {}})]
        result = _build_tool_calls(parts)
        assert result is not None
        assert json.loads(result[0].function.arguments) == {}

    # ── _build_provider_specific_fields ─────────────────────────────

    def test_build_psf_empty(self):
        from backends.chat_completion import _build_provider_specific_fields

        assert _build_provider_specific_fields([]) is None

    def test_build_psf_populated(self):
        from backends.chat_completion import _build_provider_specific_fields

        result = _build_provider_specific_fields(["a", "b"])
        assert result is not None
        assert result.thought_signatures == ["a", "b"]

    # ── _map_finish_reason ──────────────────────────────────────────

    @pytest.mark.parametrize(
        "reason_name,has_tool_calls,expected",
        [
            # STOP: override to tool_calls when has_tool_calls
            ("STOP", False, "stop"),
            ("STOP", True, "tool_calls"),
            # UNSPECIFIED behaves like STOP
            ("FINISH_REASON_UNSPECIFIED", False, "stop"),
            ("FINISH_REASON_UNSPECIFIED", True, "tool_calls"),
            # MAX_TOKENS -> length (never overridden)
            ("MAX_TOKENS", False, "length"),
            ("MAX_TOKENS", True, "length"),
            # Safety-family -> content_filter
            ("SAFETY", False, "content_filter"),
            ("RECITATION", False, "content_filter"),
            ("BLOCKLIST", False, "content_filter"),
            ("PROHIBITED_CONTENT", False, "content_filter"),
            ("SPII", False, "content_filter"),
            ("MODEL_ARMOR", False, "content_filter"),
            # OTHER -> content_filter (matches LiteLLM); MALFORMED_FUNCTION_CALL -> stop
            ("OTHER", False, "content_filter"),
            ("MALFORMED_FUNCTION_CALL", False, "stop"),
            # Negative override: has_tool_calls=True MUST NOT promote non-STOP mapped-to-stop values
            ("OTHER", True, "content_filter"),  # Note: OTHER is content_filter now
            ("MALFORMED_FUNCTION_CALL", True, "stop"),
            # Negative override: has_tool_calls=True MUST NOT promote length/content_filter
            ("SAFETY", True, "content_filter"),
            ("MODEL_ARMOR", True, "content_filter"),
        ],
    )
    def test_map_finish_reason(self, reason_name, has_tool_calls, expected):
        from google.cloud.aiplatform_v1.types import Candidate
        from backends.chat_completion import _map_finish_reason

        reason = Candidate.FinishReason[reason_name]
        assert _map_finish_reason(reason, has_tool_calls) == expected

    def test_map_finish_reason_unknown_raises(self):
        from backends.chat_completion import _map_finish_reason

        # Cast an out-of-range int through a shim to simulate future proto
        # version drift. Using 999 which is not in the enum.
        class _Fake:
            value = 999
            name = "FUTURE_REASON"

            def __repr__(self):
                return "FUTURE_REASON"

        with pytest.raises(ValueError, match="unknown finish_reason"):
            _map_finish_reason(_Fake(), False)  # type: ignore[arg-type]

    # ── _build_usage ────────────────────────────────────────────────

    def test_build_usage_none_input(self):
        from backends.chat_completion import _build_usage

        assert _build_usage(None) is None

    def test_build_usage_all_zero(self):
        from google.cloud.aiplatform_v1.types import UsageMetadata
        from backends.chat_completion import _build_usage

        u = _build_usage(UsageMetadata())
        assert u is not None
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0
        assert u.total_tokens == 0
        assert u.completion_tokens_details is None
        assert u.prompt_tokens_details is None

    def test_build_usage_basic(self):
        from google.cloud.aiplatform_v1.types import UsageMetadata
        from backends.chat_completion import _build_usage

        u = _build_usage(
            UsageMetadata(
                prompt_token_count=10,
                candidates_token_count=5,
                total_token_count=15,
            )
        )
        assert u is not None
        assert u.prompt_tokens == 10
        assert u.completion_tokens == 5
        assert u.total_tokens == 15
        assert u.completion_tokens_details is None
        assert u.prompt_tokens_details is None

    def test_build_usage_with_reasoning(self):
        from google.cloud.aiplatform_v1.types import UsageMetadata
        from backends.chat_completion import _build_usage

        u = _build_usage(
            UsageMetadata(
                prompt_token_count=10,
                candidates_token_count=5,
                total_token_count=15,
                thoughts_token_count=7,
            )
        )
        assert u is not None
        assert u.completion_tokens_details is not None
        assert u.completion_tokens_details.reasoning_tokens == 7

    def test_build_usage_with_cached(self):
        from google.cloud.aiplatform_v1.types import UsageMetadata
        from backends.chat_completion import _build_usage

        u = _build_usage(
            UsageMetadata(
                prompt_token_count=10,
                candidates_token_count=5,
                total_token_count=15,
                cached_content_token_count=3,
            )
        )
        assert u is not None
        assert u.prompt_tokens_details is not None
        assert u.prompt_tokens_details.cached_tokens == 3

    def test_build_usage_inclusive_candidate_count(self):
        """When candidates_token_count already includes thoughts (prompt+cand==total),
        completion_tokens is reported as-is (no double-counting)."""
        from google.cloud.aiplatform_v1.types import UsageMetadata
        from backends.chat_completion import _build_usage

        # 10 + 15 == 25 -> inclusive
        u = _build_usage(
            UsageMetadata(
                prompt_token_count=10,
                candidates_token_count=15,
                total_token_count=25,
                thoughts_token_count=7,
            )
        )
        assert u is not None
        assert u.completion_tokens == 15  # NOT 15+7=22
        assert u.completion_tokens_details is not None
        assert u.completion_tokens_details.reasoning_tokens == 7

    def test_build_usage_exclusive_candidate_count(self):
        """When candidates_token_count is EXCLUSIVE of thoughts (prompt+cand<total),
        completion_tokens is adjusted to include reasoning."""
        from google.cloud.aiplatform_v1.types import UsageMetadata
        from backends.chat_completion import _build_usage

        # 10 + 15 != 32 -> exclusive; 15+7=22 completion, +10 prompt = 32 total
        u = _build_usage(
            UsageMetadata(
                prompt_token_count=10,
                candidates_token_count=15,
                total_token_count=32,
                thoughts_token_count=7,
            )
        )
        assert u is not None
        assert u.completion_tokens == 22  # 15 + 7 (reasoning added)
        assert u.completion_tokens_details is not None
        assert u.completion_tokens_details.reasoning_tokens == 7

    def test_build_usage_with_both_details(self):
        """Both reasoning and cached details can coexist."""
        from google.cloud.aiplatform_v1.types import UsageMetadata
        from backends.chat_completion import _build_usage

        u = _build_usage(
            UsageMetadata(
                prompt_token_count=10,
                candidates_token_count=15,
                total_token_count=25,
                thoughts_token_count=7,
                cached_content_token_count=3,
            )
        )
        assert u is not None
        assert u.completion_tokens_details is not None
        assert u.completion_tokens_details.reasoning_tokens == 7
        assert u.prompt_tokens_details is not None
        assert u.prompt_tokens_details.cached_tokens == 3


class TestConvertResponseFromFixtures:
    """End-to-end fixture-driven tests exercising the whole pipeline."""

    def test_real_thinkhigh_turn01(self):
        from backends.chat_completion import convert_gemini_response_to_openai

        proto = _load_fixture_as_proto("real_thinkhigh_turn01_text_and_fc.json")
        result = convert_gemini_response_to_openai(
            proto,
            response_id="test-1",
            created=1000000000,
        )
        d = result.to_dict()
        assert d["id"] == "test-1"
        assert d["object"] == "chat.completion"
        assert d["created"] == 1000000000
        assert d["model"] == "gemini-2.5-pro"
        assert len(d["choices"]) == 1
        ch = d["choices"][0]
        assert ch["index"] == 0
        # This fixture has a functionCall part -> STOP overrides to tool_calls.
        assert ch["finish_reason"] == "tool_calls"
        msg = ch["message"]
        assert msg["role"] == "assistant"
        # Has one text part before the fc part.
        assert "content" in msg and isinstance(msg["content"], str)
        assert len(msg["tool_calls"]) == 1
        tc = msg["tool_calls"][0]
        assert tc["id"].startswith("call_") and "__thought__" in tc["id"]
        # Provider-specific catch-all sig list is populated.
        assert msg["provider_specific_fields"]["thought_signatures"]
        # Usage present with reasoning tokens.
        assert d["usage"]["prompt_tokens"] == 881
        assert d["usage"]["total_tokens"] == 1053
        # thoughts_token_count is 124. Check inclusive vs exclusive:
        # 881 + 48 = 929, not 1053 -> exclusive. So completion should be 48+124=172.
        assert d["usage"]["completion_tokens"] == 48 + 124
        assert d["usage"]["completion_tokens_details"]["reasoning_tokens"] == 124

    def test_real_thinkhigh_turn11(self):
        from backends.chat_completion import convert_gemini_response_to_openai

        proto = _load_fixture_as_proto("real_thinkhigh_turn11_text_and_fc.json")
        result = convert_gemini_response_to_openai(proto)
        d = result.to_dict()
        assert d["choices"][0]["finish_reason"] == "tool_calls"
        assert d["choices"][0]["message"]["tool_calls"]

    def test_real_thinkhigh_turn20(self):
        from backends.chat_completion import convert_gemini_response_to_openai

        proto = _load_fixture_as_proto("real_thinkhigh_turn20_text_and_fc.json")
        result = convert_gemini_response_to_openai(proto)
        d = result.to_dict()
        assert d["choices"][0]["finish_reason"] == "tool_calls"

    def test_real_thinkmin_turn01(self):
        from backends.chat_completion import convert_gemini_response_to_openai

        proto = _load_fixture_as_proto("real_thinkmin_turn01_fc_only.json")
        result = convert_gemini_response_to_openai(proto)
        d = result.to_dict()
        ch = d["choices"][0]
        assert ch["finish_reason"] == "tool_calls"
        # thinkmin has NO text part; content should be omitted.
        assert "content" not in ch["message"]
        assert len(ch["message"]["tool_calls"]) == 1

    def test_real_thinkmin_last(self):
        from backends.chat_completion import convert_gemini_response_to_openai

        proto = _load_fixture_as_proto("real_thinkmin_last_fc_only.json")
        result = convert_gemini_response_to_openai(proto)
        d = result.to_dict()
        assert d["choices"][0]["finish_reason"] == "tool_calls"

    def test_synthetic_text_only_stop(self):
        from backends.chat_completion import convert_gemini_response_to_openai

        proto = _load_fixture_as_proto("synthetic_text_only_stop.json")
        result = convert_gemini_response_to_openai(proto)
        d = result.to_dict()
        ch = d["choices"][0]
        # No tool_calls -> STOP stays as 'stop'.
        assert ch["finish_reason"] == "stop"
        assert ch["message"]["content"] == "Hi there!"
        assert "tool_calls" not in ch["message"]
        assert "thinking_blocks" not in ch["message"]
        assert "provider_specific_fields" not in ch["message"]

    def test_synthetic_max_tokens(self):
        from backends.chat_completion import convert_gemini_response_to_openai

        proto = _load_fixture_as_proto("synthetic_max_tokens.json")
        result = convert_gemini_response_to_openai(proto)
        d = result.to_dict()
        assert d["choices"][0]["finish_reason"] == "length"

    def test_synthetic_safety_blocked(self):
        # Safety-blocked responses have empty parts + finish_reason=SAFETY.
        # Per LiteLLM parity: emit a Choice with content_filter finish and
        # None content rather than raising, so clients that key off the
        # finish_reason string work correctly.
        from backends.chat_completion import convert_gemini_response_to_openai

        proto = _load_fixture_as_proto("synthetic_safety_blocked.json")
        result = convert_gemini_response_to_openai(proto)
        d = result.to_dict()
        ch = d["choices"][0]
        assert ch["finish_reason"] == "content_filter"
        # Empty content dropped from wire (Message.content=None -> to_dict drops it).
        assert "content" not in ch["message"]
        assert "tool_calls" not in ch["message"]

    def test_envelope_priority_chain_uses_proto_id(self):
        # When no response_id kwarg, use proto's response_id.
        # Note: GenerateContentResponse's usage_metadata is a *nested*
        # UsageMetadata type — omit it here and let proto default it.
        from google.cloud.aiplatform_v1.types import (
            GenerateContentResponse,
            Candidate,
            Content,
            Part,
        )
        from backends.chat_completion import convert_gemini_response_to_openai

        proto = GenerateContentResponse(
            response_id="vertex-orig-id",
            candidates=[
                Candidate(
                    content=Content(role="model", parts=[Part(text="hi")]),
                    finish_reason=Candidate.FinishReason.STOP,
                )
            ],
        )
        result = convert_gemini_response_to_openai(proto)
        assert result.to_dict()["id"] == "vertex-orig-id"

    def test_envelope_priority_chain_kwarg_wins(self):
        from google.cloud.aiplatform_v1.types import (
            GenerateContentResponse,
            Candidate,
            Content,
            Part,
        )
        from backends.chat_completion import convert_gemini_response_to_openai

        proto = GenerateContentResponse(
            response_id="vertex-orig-id",
            candidates=[
                Candidate(
                    content=Content(role="model", parts=[Part(text="hi")]),
                    finish_reason=Candidate.FinishReason.STOP,
                )
            ],
        )
        result = convert_gemini_response_to_openai(proto, response_id="kwarg-id")
        assert result.to_dict()["id"] == "kwarg-id"

    def test_envelope_falls_back_to_uuid(self):
        from google.cloud.aiplatform_v1.types import (
            GenerateContentResponse,
            Candidate,
            Content,
            Part,
        )
        from backends.chat_completion import convert_gemini_response_to_openai

        proto = GenerateContentResponse(
            candidates=[
                Candidate(
                    content=Content(role="model", parts=[Part(text="hi")]),
                    finish_reason=Candidate.FinishReason.STOP,
                )
            ],
        )
        result = convert_gemini_response_to_openai(proto)
        assert result.to_dict()["id"].startswith("chatcmpl-")

    def test_envelope_model_kwarg_wins(self):
        from google.cloud.aiplatform_v1.types import (
            GenerateContentResponse,
            Candidate,
            Content,
            Part,
        )
        from backends.chat_completion import convert_gemini_response_to_openai

        proto = GenerateContentResponse(
            model_version="proto-model",
            candidates=[
                Candidate(
                    content=Content(role="model", parts=[Part(text="hi")]),
                    finish_reason=Candidate.FinishReason.STOP,
                )
            ],
        )
        result = convert_gemini_response_to_openai(proto, model="kwarg-model")
        assert result.to_dict()["model"] == "kwarg-model"

    def test_envelope_model_from_proto(self):
        from google.cloud.aiplatform_v1.types import (
            GenerateContentResponse,
            Candidate,
            Content,
            Part,
        )
        from backends.chat_completion import convert_gemini_response_to_openai

        proto = GenerateContentResponse(
            model_version="proto-model",
            candidates=[
                Candidate(
                    content=Content(role="model", parts=[Part(text="hi")]),
                    finish_reason=Candidate.FinishReason.STOP,
                )
            ],
        )
        result = convert_gemini_response_to_openai(proto)
        assert result.to_dict()["model"] == "proto-model"

    def test_envelope_created_from_proto(self):
        """C1 regression guard: proto.create_time must be handled correctly."""
        from google.protobuf.timestamp_pb2 import Timestamp
        from google.cloud.aiplatform_v1.types import (
            GenerateContentResponse,
            Candidate,
            Content,
            Part,
        )
        from backends.chat_completion import convert_gemini_response_to_openai

        proto = GenerateContentResponse(
            create_time=Timestamp(seconds=1700000000),
            candidates=[
                Candidate(
                    content=Content(role="model", parts=[Part(text="hi")]),
                    finish_reason=Candidate.FinishReason.STOP,
                )
            ],
        )
        result = convert_gemini_response_to_openai(proto)
        assert result.to_dict()["created"] == 1700000000

    def test_envelope_created_auto_fallback(self):
        import time as _t
        from google.cloud.aiplatform_v1.types import (
            GenerateContentResponse,
            Candidate,
            Content,
            Part,
        )
        from backends.chat_completion import convert_gemini_response_to_openai

        proto = GenerateContentResponse(
            candidates=[
                Candidate(
                    content=Content(role="model", parts=[Part(text="hi")]),
                    finish_reason=Candidate.FinishReason.STOP,
                )
            ],
        )
        before = int(_t.time())
        result = convert_gemini_response_to_openai(proto)
        after = int(_t.time())
        assert before <= result.to_dict()["created"] <= after


class TestResponseErrors:
    """One test per ValueError condition."""

    def test_no_candidates_raises(self):
        from google.cloud.aiplatform_v1.types import GenerateContentResponse
        from backends.chat_completion import convert_gemini_response_to_openai

        with pytest.raises(ValueError, match="no candidates"):
            convert_gemini_response_to_openai(GenerateContentResponse())

    def test_candidate_without_content_raises(self):
        from google.cloud.aiplatform_v1.types import (
            GenerateContentResponse,
            Candidate,
        )
        from backends.chat_completion import convert_gemini_response_to_openai

        # Candidate with no content field (a proto zero-Content is still
        # a Content, but with role="" and parts=[]).
        proto = GenerateContentResponse(candidates=[Candidate()])
        with pytest.raises(ValueError, match="empty parts"):
            convert_gemini_response_to_openai(proto)

    def test_unrecognized_part_payload_raises(self):
        from google.cloud.aiplatform_v1.types import (
            GenerateContentResponse,
            Candidate,
            Content,
            Part,
        )
        from backends.chat_completion import convert_gemini_response_to_openai

        proto = GenerateContentResponse(
            candidates=[Candidate(content=Content(role="model", parts=[Part()]))],
        )
        with pytest.raises(ValueError, match="no recognizable payload"):
            convert_gemini_response_to_openai(proto)

    def test_media_part_raises(self):
        from google.cloud.aiplatform_v1.types import (
            GenerateContentResponse,
            Candidate,
            Content,
            Part,
            Blob,
        )
        from backends.chat_completion import convert_gemini_response_to_openai

        proto = GenerateContentResponse(
            candidates=[
                Candidate(
                    content=Content(
                        role="model",
                        parts=[
                            Part(inline_data=Blob(mime_type="image/png", data=b"x")),
                        ],
                    )
                )
            ],
        )
        with pytest.raises(ValueError, match="contains media"):
            convert_gemini_response_to_openai(proto)

    def test_function_call_empty_name_raises(self):
        # Bare function_call with name="" is treated as "no fc" by
        # _partition_parts (name is falsy) -> falls through to
        # "no recognizable payload". Kept as an error either way.
        from google.cloud.aiplatform_v1.types import (
            GenerateContentResponse,
            Candidate,
            Content,
            Part,
            FunctionCall as ProtoFC,
        )
        from backends.chat_completion import convert_gemini_response_to_openai

        proto = GenerateContentResponse(
            candidates=[
                Candidate(
                    content=Content(
                        role="model",
                        parts=[
                            Part(function_call=ProtoFC(name="")),
                        ],
                    )
                )
            ],
        )
        with pytest.raises(ValueError):
            convert_gemini_response_to_openai(proto)


def _has_litellm() -> bool:
    # We need the specific submodule the parity test drives, not just any
    # `import litellm` -- uv can leave a partially-installed namespace-package
    # shell behind after uninstall (only `litellm/proxy/` remains), and the
    # bare `import litellm` in that state succeeds without the real package.
    try:
        from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (  # noqa: F401
            VertexGeminiConfig,
        )

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _has_litellm(), reason="litellm not installed (dev-only)")
class TestLiteLLMWireParity:
    """Prove wire-format parity against LiteLLM main branch.

    Feeds a fixture through both our converter and LiteLLM's Vertex
    transformer, then compares the serialized dicts with envelope
    fields (id, created) masked. If this test starts failing, either
    LiteLLM's shape drifted or our translator did -- diagnose before
    just updating the expected output.
    """

    # NOTE: synthetic_safety_blocked.json is deliberately excluded. That
    # fixture has parts=[] (safety-blocked response) which exercises our
    # dedicated safety-block code path. LiteLLM's handling of empty parts
    # is unclear (may raise, may emit Choice(content=None,
    # finish_reason='content_filter'), may differ across versions) --
    # investigating that divergence is a separate follow-up.
    @pytest.mark.parametrize(
        "fixture_name",
        [
            "real_thinkhigh_turn01_text_and_fc.json",  # text + FC + thought sig
            "real_thinkmin_turn01_fc_only.json",  # FC-only, no text
            "synthetic_text_only_stop.json",  # pure text, STOP
            "synthetic_max_tokens.json",  # text + MAX_TOKENS finish
        ],
    )
    def test_matches_litellm(self, fixture_name):
        from backends.chat_completion import convert_gemini_response_to_openai

        proto = _load_fixture_as_proto(fixture_name)

        # Our output. Pin envelope fields for later masking.
        ours = convert_gemini_response_to_openai(
            proto,
            response_id="MASKED",
            created=0,
            model="gemini-2.5-pro",
        ).to_dict()

        # LiteLLM's output.
        theirs = _run_through_litellm(proto)

        # Mask envelope fields that legitimately differ (LiteLLM auto-generates
        # id/created; we accept a kwarg override -- see docstring of
        # convert_gemini_response_to_openai for the priority-chain rationale).
        for d in (ours, theirs):
            d["id"] = "MASKED"
            d["created"] = 0
            # LiteLLM may or may not emit system_fingerprint depending on version.
            d.pop("system_fingerprint", None)

        # LiteLLM 1.91+ attaches Vertex-specific metadata to the top-level
        # ModelResponse for observability (grounding, safety, citation,
        # url_context). These are LiteLLM proxy bookkeeping, not part of the
        # OpenAI chat.completion wire body. Assert-then-pop rather than
        # silent-pop so that a future LiteLLM release removing any of these
        # keys surfaces as a test failure -- forcing us to re-evaluate whether
        # the corresponding mask entry can be dropped.
        _LITELLM_ONLY_TOP_LEVEL_KEYS = (
            "vertex_ai_grounding_metadata",
            "vertex_ai_url_context_metadata",
            "vertex_ai_safety_results",
            "vertex_ai_citation_metadata",
        )
        for k in _LITELLM_ONLY_TOP_LEVEL_KEYS:
            assert k in theirs, (
                f"LiteLLM no longer emits top-level {k!r}; the mask entry can "
                f"probably be removed. Verify by checking whether ours emits "
                f"it too, then delete the mask entry accordingly."
            )
            theirs.pop(k)
            # Ours never emits these observability fields, so tolerate absence.
            ours.pop(k, None)

        # Assert the message-level provider_specific_fields (which carries the
        # thought_signatures list) matches BEFORE we strip the per-tool_call
        # psf below -- otherwise the "the per-call copy is redundant because
        # the message-level list is equal" claim is not actually validated.
        for ours_choice, theirs_choice in zip(
            ours.get("choices", []), theirs.get("choices", [])
        ):
            ours_psf = ours_choice.get("message", {}).get("provider_specific_fields")
            theirs_psf = theirs_choice.get("message", {}).get(
                "provider_specific_fields"
            )
            assert ours_psf == theirs_psf, (
                f"message.provider_specific_fields drift: "
                f"ours={ours_psf!r}, theirs={theirs_psf!r}"
            )

        # Mask the random tool_call uuid prefix while keeping the
        # __thought__-suffix and signature intact (that's the wire-parity claim).
        # Also strip LiteLLM-only per-message / per-tool_call additions that
        # aren't part of the OpenAI Chat Completions wire body:
        #   - message.images / message.thinking_blocks: LiteLLM emits empty
        #     defaults; not in OpenAI spec. Symmetric silent-pop because these
        #     may or may not appear depending on fixture shape.
        #   - tool_calls[*].index: LiteLLM adds an index used only for
        #     streaming reassembly; non-streaming responses don't require it.
        #   - tool_calls[*].provider_specific_fields: LiteLLM duplicates the
        #     sig per tool_call. Redundant with the message-level list we
        #     asserted equal directly above.
        for d in (ours, theirs):
            for choice in d.get("choices", []):
                msg = choice.get("message", {})
                msg.pop("images", None)
                msg.pop("thinking_blocks", None)
                for tc in msg.get("tool_calls") or []:
                    if "__thought__" in tc["id"]:
                        _, sig = tc["id"].split("__thought__", 1)
                        tc["id"] = f"call_MASKED__thought__{sig}"
                    else:
                        tc["id"] = "call_MASKED"
                    tc.pop("index", None)
                    tc.pop("provider_specific_fields", None)

        # LiteLLM emits a fuller usage breakdown than the OpenAI wire body:
        # always attaches prompt_tokens_details (often empty {}) and puts a
        # text_tokens breakdown under completion_tokens_details. The core
        # counts (prompt/completion/total/reasoning) are already in both
        # dicts and identical; drop these LiteLLM-only extras so parity is
        # judged on the OpenAI-defined surface. Assert-then-pop for
        # LiteLLM-specific extras we expect to be present, so LiteLLM
        # removing them surfaces as a failure.
        theirs_usage = theirs.get("usage", {})
        assert "prompt_tokens_details" in theirs_usage, (
            "LiteLLM no longer emits usage.prompt_tokens_details; the mask "
            "entry can probably be dropped."
        )
        theirs_usage.pop("prompt_tokens_details")
        # Ours only emits prompt_tokens_details when cached tokens > 0, so
        # tolerate absence on our side.
        ours.get("usage", {}).pop("prompt_tokens_details", None)

        theirs_ctd = theirs_usage.get("completion_tokens_details") or {}
        assert "text_tokens" in theirs_ctd, (
            "LiteLLM no longer emits usage.completion_tokens_details.text_tokens; "
            "the mask entry can probably be dropped."
        )
        theirs_ctd.pop("text_tokens")

        # After removing text_tokens, LiteLLM always leaves reasoning_tokens
        # in completion_tokens_details (even 0), while we omit the whole
        # sub-dict when reasoning tokens are 0. Normalize: if reasoning is 0
        # on either side after mask, drop completion_tokens_details entirely
        # from both. This keeps the parity claim strong on non-zero reasoning
        # (already covered by the thinkhigh fixture) while accepting the
        # equivalent empty representation for zero-reasoning fixtures.
        for d in (ours, theirs):
            usage = d.get("usage", {})
            ctd = usage.get("completion_tokens_details") or {}
            if ctd == {} or ctd == {"reasoning_tokens": 0}:
                usage.pop("completion_tokens_details", None)

        # Diagnostic aid: if this fails, show both sides clearly.
        if ours != theirs:
            import json

            ours_str = json.dumps(ours, indent=2, sort_keys=True)
            theirs_str = json.dumps(theirs, indent=2, sort_keys=True)
            pytest.fail(
                f"Wire-format parity broken for fixture {fixture_name!r}.\n\n"
                f"Ours:\n{ours_str}\n\nTheirs:\n{theirs_str}"
            )


def _run_through_litellm(proto):
    """Feed a GenerateContentResponse proto through LiteLLM's Vertex transformer
    and return the serialized dict LiteLLM would emit over HTTP."""
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
        VertexGeminiConfig,
    )
    from litellm.types.utils import ModelResponse

    # LiteLLM's transform_response ingests the raw REST JSON dict of the
    # Vertex response, so convert our proto back to that shape.
    litellm_input = {
        "candidates": [
            {
                "content": {
                    "role": c.content.role,
                    "parts": [_part_to_rest_dict(p) for p in c.content.parts],
                },
                "finishReason": c.finish_reason.name,
                "index": c.index,
            }
            for c in proto.candidates
        ],
        "usageMetadata": {
            "promptTokenCount": proto.usage_metadata.prompt_token_count,
            "candidatesTokenCount": proto.usage_metadata.candidates_token_count,
            "totalTokenCount": proto.usage_metadata.total_token_count,
            "thoughtsTokenCount": proto.usage_metadata.thoughts_token_count,
        },
        "modelVersion": proto.model_version,
    }
    config = VertexGeminiConfig()
    response = config.transform_response(
        model="gemini-2.5-pro",
        raw_response=_FakeHttpxResponse(litellm_input),
        model_response=ModelResponse(),
        # LiteLLM 1.91 calls logging_obj.post_call() unconditionally, so a
        # None is not tolerated -- pass a minimal no-op stub instead.
        logging_obj=_NoopLoggingObj(),
        request_data={},
        messages=[],
        optional_params={},
        litellm_params={},
        encoding=None,
    )
    return response.model_dump(exclude_none=True, exclude_unset=True)


def _part_to_rest_dict(part):
    """Serialize a Part proto to the REST-shape dict LiteLLM ingests."""
    import base64

    out = {}
    if part.text:
        out["text"] = part.text
    if part.thought:
        out["thought"] = True
    if part.thought_signature:
        out["thoughtSignature"] = base64.b64encode(part.thought_signature).decode(
            "ascii"
        )
    if part.function_call.name:
        out["functionCall"] = {
            "name": part.function_call.name,
            "args": dict(part.function_call.args),
        }
    return out


class _FakeHttpxResponse:
    """Minimal httpx.Response stand-in that LiteLLM's transform_response
    calls .json() on."""

    def __init__(self, data):
        self._data = data
        self.status_code = 200
        self.headers = {}
        self.text = ""

    def json(self):
        return self._data


class _NoopLoggingObj:
    """No-op stand-in for LiteLLM's Logging object.

    LiteLLM's transform_response calls .post_call() on it for
    observability and reads .optional_params off it when partitioning
    tool_calls; we don't need any of that in a wire-parity test.
    """

    optional_params: Dict[str, Any] = {}

    def post_call(self, *args, **kwargs):
        return None


# ── Blocking-gate tests: thought_signature roundtrip ────────────────
# End-to-end guarantee that Google's Part.thought_signature bytes
# survive every conversion in both directions. Multi-turn agent loops
# depend on the signature echoing back correctly, or Vertex rejects
# the second turn with INVALID_ARGUMENT: Function call is missing a
# thought_signature.
#
# Wire-shape notes anchored to the code:
#   * Response side: convert_gemini_response_to_openai emits the
#     signature into message.provider_specific_fields.thought_signatures
#     as a top-level LIST of base64-encoded ASCII strings, one per
#     response Part with a signature (parts-order). This is LiteLLM's
#     canonical envelope. It is ALSO embedded in tool_calls[i].id via
#     the "call_<uuid>__thought__<b64>" convention, but the list is
#     the primary carrier.
#   * Request side: convert_openai_request_to_gemini reads the same
#     provider_specific_fields.thought_signatures list (bridge added
#     alongside the pre-existing extra_content.google.thought_signature
#     scalar path) and writes each entry verbatim to the corresponding
#     functionCall Part's thoughtSignature field. Proto3 JSON encodes
#     bytes fields as base64, so writing the b64 string IS the wire
#     form.
import base64 as _tsig_base64  # local alias avoids shadowing the file's
# existing (module-level or lazy) imports.


def _make_response_with_thought_signature(sig_bytes: bytes):
    """Build a minimal GenerateContentResponse with a function_call Part
    carrying a thought_signature. Used to exercise the response-side
    thought_signature extraction path."""
    from google.cloud.aiplatform_v1.types import (
        Candidate,
        Content,
        FunctionCall,
        GenerateContentResponse,
        Part,
    )
    from google.protobuf.struct_pb2 import Struct

    args = Struct()
    args.update({"command": "ls"})
    part = Part(
        function_call=FunctionCall(name="bash", args=args),
        thought_signature=sig_bytes,
    )
    candidate = Candidate(
        content=Content(role="model", parts=[part]),
        finish_reason=Candidate.FinishReason.STOP,
    )
    response = GenerateContentResponse()
    response.candidates.append(candidate)
    response.usage_metadata.prompt_token_count = 10
    response.usage_metadata.candidates_token_count = 5
    response.usage_metadata.total_token_count = 15
    response.model_version = "gemini-3.5-flash"
    response.response_id = "resp-thought-sig"
    return response


def test_thought_signature_surfaced_on_response_conversion():
    """Response-side: Gemini's Part.thought_signature bytes must appear
    on the OpenAI assistant message as a base64-encoded string in the
    LiteLLM-parity envelope ``provider_specific_fields.thought_signatures``
    (top-level LIST).

    Note the accessor deliberately targets the LIST envelope, NOT
    ``provider_specific_fields.google.thought_signature`` (that shape
    exists only in older LiteLLM proxies and isn't what the converter
    emits — the design chose LiteLLM parity, and LiteLLM's Message
    dataclass has a flat top-level ``thought_signatures`` list).
    """
    from backends.chat_completion import convert_gemini_response_to_openai

    raw_sig = b"\x01\x02\x03opaque_thought_reasoning_token\xff\xfe"
    expected_b64 = _tsig_base64.b64encode(raw_sig).decode("ascii")

    response = _make_response_with_thought_signature(raw_sig)
    chat_response = convert_gemini_response_to_openai(response)

    message = chat_response.choices[0].message
    provider_fields = message.provider_specific_fields
    assert provider_fields is not None, (
        "convert_gemini_response_to_openai must surface thought_signature "
        "via provider_specific_fields.thought_signatures"
    )
    thought_sigs = provider_fields.thought_signatures
    assert thought_sigs is not None, (
        "provider_specific_fields.thought_signatures must not be None "
        "when the response Part carries a thought_signature"
    )
    assert expected_b64 in thought_sigs, (
        f"Expected b64-encoded signature {expected_b64!r} in {thought_sigs!r}. "
        "The bytes MUST roundtrip losslessly through base64."
    )


def test_thought_signature_survives_request_conversion():
    """Request-side: an assistant message with a tool_call and
    ``provider_specific_fields.thought_signatures`` (LiteLLM-parity
    envelope) must be rebuilt into a Gemini Part whose thoughtSignature
    is the same base64 string on the outgoing wire dict, and whose
    base64-decoded bytes exactly match the original raw signature.
    """
    from backends.chat_completion import (
        ConversionContext,
        convert_openai_request_to_gemini,
    )

    raw_sig = b"\x01\x02\x03opaque_thought_reasoning_token\xff\xfe"
    sig_b64 = _tsig_base64.b64encode(raw_sig).decode("ascii")

    openai_request = {
        "model": "gemini-3.5-flash",
        "messages": [
            {"role": "user", "content": "list the files"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "bash",
                            "arguments": '{"command": "ls"}',
                        },
                    }
                ],
                # LiteLLM-parity envelope: top-level list, one entry per
                # tool_call in parts-order. This is the shape the
                # response side emits, so multi-turn agents can feed
                # responses straight back in without translation.
                "provider_specific_fields": {
                    "thought_signatures": [sig_b64],
                },
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc",
                "content": "file1.txt\nfile2.txt\n",
            },
            {"role": "user", "content": "now count them"},
        ],
    }

    vertex_dict = convert_openai_request_to_gemini(ConversionContext(), openai_request)

    # Locate the assistant turn's function_call Part. ConversionContext
    # .use_model_role defaults to True, so assistant is rewritten to
    # "model" on the wire; filter on that.
    contents = vertex_dict["contents"]
    model_turns = [c for c in contents if c.get("role") == "model"]
    assert len(model_turns) == 1, (
        f"Expected 1 model turn, got {len(model_turns)}: {contents!r}"
    )
    parts = model_turns[0]["parts"]
    fc_parts = [p for p in parts if "functionCall" in p]
    assert len(fc_parts) == 1, (
        f"Expected 1 functionCall part, got {len(fc_parts)}: {parts!r}"
    )
    fc_part = fc_parts[0]

    # thoughtSignature is base64-encoded on the wire dict (proto3 JSON
    # encoding for bytes fields). Verify both the string form and the
    # decoded bytes match.
    assert "thoughtSignature" in fc_part, (
        f"functionCall Part missing thoughtSignature. Got keys: {list(fc_part)}"
    )
    wire_sig_b64 = fc_part["thoughtSignature"]
    assert wire_sig_b64 == sig_b64, (
        f"Wire b64 mismatch: expected {sig_b64!r}, got {wire_sig_b64!r}"
    )
    wire_sig_bytes = _tsig_base64.b64decode(wire_sig_b64)
    assert wire_sig_bytes == raw_sig, (
        f"Signature bytes not preserved: expected {raw_sig!r}, got {wire_sig_bytes!r}"
    )


def test_thought_signature_full_roundtrip():
    """End-to-end: start with a proto response carrying a signature;
    convert to OpenAI shape; feed that message back into a next-turn
    request as history; convert the new request to a Gemini wire dict;
    assert the signature bytes are preserved bit-for-bit.

    This is the test that catches "we sent it out on turn 1 but forgot
    to send it back on turn 2" bugs — the class of bug that would fail
    with INVALID_ARGUMENT: Function call is missing a thought_signature
    on the second turn of any multi-turn agent loop.
    """
    from backends.chat_completion import (
        ConversionContext,
        convert_gemini_response_to_openai,
        convert_openai_request_to_gemini,
    )

    raw_sig = b"\x01\x02\x03opaque_thought_reasoning_token\xff\xfe"

    # Turn 1: model returns a response with a function_call + signature.
    response_turn1 = _make_response_with_thought_signature(raw_sig)
    chat_response_turn1 = convert_gemini_response_to_openai(response_turn1)

    # Serialize the full response to a dict (ChatCompletionResponse has
    # .to_dict; individual Message dataclasses don't) and pluck the
    # assistant message. This is exactly what a real client would
    # receive as JSON over the wire.
    response_dict_turn1 = chat_response_turn1.to_dict()
    assistant_msg_turn1 = response_dict_turn1["choices"][0]["message"]

    # Sanity: turn-1 assistant carries the signature under
    # provider_specific_fields.thought_signatures.
    assert assistant_msg_turn1.get("provider_specific_fields", {}).get(
        "thought_signatures"
    ), (
        "Turn-1 assistant message missing provider_specific_fields."
        f"thought_signatures. Got: {assistant_msg_turn1!r}"
    )

    # Turn 2: replay turn-1's assistant + tool result + new follow-up.
    openai_request_turn2 = {
        "model": "gemini-3.5-flash",
        "messages": [
            {"role": "user", "content": "list the files"},
            assistant_msg_turn1,  # carries thought_signatures list
            {
                "role": "tool",
                "tool_call_id": assistant_msg_turn1["tool_calls"][0]["id"],
                "content": "file1.txt\nfile2.txt\n",
            },
            {"role": "user", "content": "now count them"},
        ],
    }
    vertex_dict_turn2 = convert_openai_request_to_gemini(
        ConversionContext(), openai_request_turn2
    )

    # Locate the reconstituted function_call Part. use_model_role
    # defaults to True, so the assistant turn is on the wire as "model".
    contents = vertex_dict_turn2["contents"]
    assistant_turns = [c for c in contents if c.get("role") == "model"]
    assert len(assistant_turns) == 1, (
        f"Expected 1 model turn, got {len(assistant_turns)}: {contents!r}"
    )
    fc_parts = [p for p in assistant_turns[0]["parts"] if "functionCall" in p]
    assert len(fc_parts) == 1, f"Expected 1 functionCall part, got {len(fc_parts)}"
    fc_part = fc_parts[0]

    # The signature MUST be present and MUST decode to the original bytes.
    assert "thoughtSignature" in fc_part, (
        "Full roundtrip lost the thought_signature: response -> OpenAI -> "
        "next-turn request did not preserve it. This will cause "
        "INVALID_ARGUMENT on the second turn of any multi-turn agent."
    )
    roundtripped_bytes = _tsig_base64.b64decode(fc_part["thoughtSignature"])
    assert roundtripped_bytes == raw_sig, (
        f"Full roundtrip corrupted signature bytes: "
        f"expected {raw_sig!r}, got {roundtripped_bytes!r}"
    )


def test_chat_completion_response_from_dict_round_trips():
    original = ChatCompletionResponse(
        id="chatcmpl-abc",
        created=1234567890,
        model="gemini-3.5-flash",
        choices=[
            Choice(
                index=2,
                finish_reason="tool_calls",
                message=Message(
                    content="running the tests",
                    tool_calls=[
                        ToolCall(
                            id="call_1__thought__c2ln",
                            function=FunctionCall(
                                name="bash", arguments='{"command": "ls"}'
                            ),
                        )
                    ],
                    thinking_blocks=[
                        ThinkingBlock(thinking="plan first", signature="c2ln")
                    ],
                    provider_specific_fields=ProviderSpecificFields(
                        thought_signatures=["c2ln"]
                    ),
                ),
            )
        ],
        usage=Usage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            completion_tokens_details=CompletionTokensDetails(reasoning_tokens=7),
            prompt_tokens_details=PromptTokensDetails(cached_tokens=4),
        ),
        system_fingerprint="fp_xyz",
    )

    restored = ChatCompletionResponse.from_dict(original.to_dict())

    assert restored == original


def test_chat_completion_response_from_dict_minimal_payload():
    """Absent optional fields stay None rather than raising."""
    restored = ChatCompletionResponse.from_dict(
        {
            "id": "chatcmpl-min",
            "created": 1,
            "model": "m",
            "choices": [
                {"index": 0, "finish_reason": "stop", "message": {"role": "assistant"}}
            ],
        }
    )

    assert restored.id == "chatcmpl-min"
    assert restored.model == "m"
    assert restored.created == 1
    assert restored.usage is None
    assert restored.choices[0].index == 0
    assert restored.choices[0].finish_reason == "stop"
    assert restored.choices[0].message.content is None
    assert restored.choices[0].message.tool_calls is None
    assert restored.choices[0].message.thinking_blocks is None
    assert restored.choices[0].message.provider_specific_fields is None


@pytest.mark.parametrize(
    "fixture_name",
    [
        "real_thinkhigh_turn01_text_and_fc.json",
        "real_thinkhigh_turn11_text_and_fc.json",
        "real_thinkhigh_turn20_text_and_fc.json",
        "real_thinkmin_last_fc_only.json",
        "real_thinkmin_turn01_fc_only.json",
        "synthetic_max_tokens.json",
        "synthetic_safety_blocked.json",
        "synthetic_text_only_stop.json",
    ],
)
def test_from_dict_round_trips_real_converter_output(fixture_name):
    """from_dict must round-trip actual converter output, not just fixtures."""
    from backends.chat_completion import convert_gemini_response_to_openai

    proto = _load_fixture_as_proto(fixture_name)
    converted = convert_gemini_response_to_openai(
        proto,
        response_id="test-1",
        created=1000000000,
    )

    assert ChatCompletionResponse.from_dict(converted.to_dict()) == converted


def test_from_dict_preserves_present_empty_lists():
    """A present empty list stays [] rather than collapsing to None."""
    restored = ChatCompletionResponse.from_dict(
        {
            "id": "x",
            "created": 1,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "tool_calls": [],
                        "thinking_blocks": [],
                    },
                }
            ],
        }
    )

    message = restored.choices[0].message
    assert message.tool_calls == []
    assert message.thinking_blocks == []


def test_from_dict_raises_on_malformed_choices():
    """A present-but-non-list ``choices`` must fail loudly, not become []."""
    with pytest.raises(ValueError, match="choices"):
        ChatCompletionResponse.from_dict(
            {"id": "x", "created": 1, "model": "m", "choices": 7}
        )


def test_from_dict_raises_on_malformed_usage():
    """A present-but-wrong-typed container must fail loudly, not silently."""
    with pytest.raises(ValueError, match="usage"):
        ChatCompletionResponse.from_dict(
            {"id": "x", "created": 1, "model": "m", "choices": [], "usage": [1, 2, 3]}
        )


# ---------------------------------------------------------------------------
# Reverse direction: convert_gemini_request_to_openai /
# convert_openai_response_to_gemini (driving an OpenAI-compatible server from
# Gemini-shaped callers).
# ---------------------------------------------------------------------------

import json as _json

from google.cloud.aiplatform_v1.types import (
    GenerateContentRequest as _GenerateContentRequest,
    GenerateContentResponse as _GenerateContentResponse,
)

from backends.chat_completion import (
    convert_gemini_request_to_openai,
    convert_gemini_response_to_openai,
    convert_openai_response_to_gemini,
    reasoning_content_to_thinking_blocks,
    thinking_blocks_to_reasoning_content,
)
from backends.prompt_coerce import to_wire_dict as _to_wire_dict

# Shape of a real SGLang-served response (reasoning_content, top-level
# usage.reasoning_tokens), trimmed.
_REV_TOOL_CALL_RESPONSE: Dict[str, Any] = {
    "id": "7bee5e248e1a4763b1a5f9150f587eec",
    "object": "chat.completion",
    "model": "some-org/some-model",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "",
                "reasoning_content": "I should call get_weather.",
                "tool_calls": [
                    {
                        "id": "get_weather:0",
                        "index": 0,
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 179, "completion_tokens": 80, "reasoning_tokens": 37},
}

_REV_TOOLS = [
    {
        "function_declarations": [
            {
                "name": "get_weather",
                "description": "weather",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {"city": {"type": "STRING"}},
                    "required": ["city"],
                },
            }
        ]
    }
]


def _rev_wire(**kwargs: Any) -> Dict[str, Any]:
    return _to_wire_dict(_GenerateContentRequest(**kwargs))


class TestConvertGeminiRequestToOpenai:
    def test_str_prompt(self):
        body = convert_gemini_request_to_openai(_to_wire_dict("hi"), model="m", n=3, max_tokens=7)
        assert body == {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "n": 3,
            "max_tokens": 7,
        }

    def test_system_tools_gen_config_and_tool_choice(self):
        wire = _rev_wire(
            system_instruction={"parts": [{"text": "sys"}]},
            contents=[{"role": "user", "parts": [{"text": "q"}]}],
            tools=_REV_TOOLS,
            tool_config={
                "function_calling_config": {"mode": "ANY", "allowed_function_names": ["get_weather"]}
            },
            generation_config={"temperature": 0.5, "top_p": 0.9, "stop_sequences": ["X"], "seed": 3},
        )
        body = convert_gemini_request_to_openai(wire, model="m", n=1, max_tokens=10)
        assert body["messages"] == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
        ]
        assert body["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ]
        assert body["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}
        assert body["temperature"] == 0.5
        assert body["top_p"] == pytest.approx(0.9)
        assert body["stop"] == ["X"]
        assert body["seed"] == 3

    def test_multi_turn_tool_history_pairs_ids_and_keeps_reasoning(self):
        wire = _rev_wire(
            contents=[
                {"role": "user", "parts": [{"text": "weather in Paris and Rome?"}]},
                {
                    "role": "model",
                    "parts": [
                        {"text": "think", "thought": True},
                        {"function_call": {"name": "get_weather", "args": {"city": "Paris"}}},
                        {"function_call": {"name": "get_time", "args": {}}},
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                        {"function_response": {"name": "get_time", "response": {"t": 1}}},
                        {"function_response": {"name": "get_weather", "response": {"c": 18}}},
                    ],
                },
            ]
        )
        msgs = convert_gemini_request_to_openai(wire, model="m", n=1, max_tokens=1)["messages"]
        assistant = msgs[1]
        assert assistant["reasoning_content"] == "think"
        assert assistant["content"] == ""
        assert [tc["id"] for tc in assistant["tool_calls"]] == ["get_weather:0", "get_time:1"]
        assert _json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {"city": "Paris"}
        # Responses are paired by name, not by order.
        assert msgs[2] == {"role": "tool", "tool_call_id": "get_time:1", "content": '{"t": 1}'}
        assert msgs[3]["tool_call_id"] == "get_weather:0"

    def test_unmatched_function_response_becomes_user_message(self):
        # An env may answer a text-only model turn with a functionResponse
        # that answers no call. Gemini accepts that; OpenAI requires every
        # tool message to answer a tool_call, so it must be a user turn.
        wire = _rev_wire(
            contents=[
                {"role": "user", "parts": [{"text": "q"}]},
                {"role": "model", "parts": [{"text": "my answer is 42"}]},
                {
                    "role": "user",
                    "parts": [
                        {
                            "function_response": {
                                "name": "_system",
                                "response": {"output": "Please use the bash or submit tool."},
                            }
                        }
                    ],
                },
            ]
        )
        msgs = convert_gemini_request_to_openai(wire, model="m", n=1, max_tokens=1)["messages"]
        assert msgs[2] == {"role": "user", "content": "Please use the bash or submit tool."}
        assert not any(m["role"] == "tool" for m in msgs)

    @pytest.mark.parametrize(
        "response, expected",
        [
            # A lone string under Gemini's documented "output" key, or the
            # forward converter's "result" key, is sent as raw text: real
            # newlines and characters, not an escaped JSON string.
            ({"output": "a.txt\nb.txt\n2025年 — café"}, "a.txt\nb.txt\n2025年 — café"),
            ({"result": "plain text"}, "plain text"),
            ({"output": ""}, ""),
        ],
    )
    def test_tool_result_plain_text_is_unwrapped(
        self, response: Dict[str, Any], expected: str
    ) -> None:
        msgs = self._tool_result_messages(response)
        assert msgs[2] == {"role": "tool", "tool_call_id": "run:0", "content": expected}

    @pytest.mark.parametrize(
        "response",
        [
            {"output": "café", "exit_code": "0"},  # more than one key
            {"error": "boom"},  # a key other than output/result
            {"output": {"nested": "é"}},  # not a string
            {},
        ],
    )
    def test_other_tool_results_stay_json_without_escaping(
        self, response: Dict[str, Any]
    ) -> None:
        content = self._tool_result_messages(response)[2]["content"]
        assert _json.loads(content) == response
        assert "\\u" not in content  # non-ASCII kept readable

    def test_json_tool_result_keeps_int_literals(self) -> None:
        # Tool results pass through a protobuf Struct, which stores every
        # number as a double; whole numbers must still reach the model as ints.
        content = self._tool_result_messages({"count": 3, "ratio": 0.5, "items": [1, 2]})[2][
            "content"
        ]
        decoded = _json.loads(content)
        assert decoded == {"count": 3, "ratio": 0.5, "items": [1, 2]}
        assert [type(v) for v in (decoded["count"], *decoded["items"])] == [int, int, int]
        assert type(decoded["ratio"]) is float

    def test_tool_call_args_keep_int_literals_and_characters(self) -> None:
        # A replayed model turn's call args pass through the same Struct. The
        # model must read back the int it generated ("3000", not "3000.0" --
        # models copy the float form into their later calls) and readable
        # non-ASCII text, at the top level and nested.
        args = {
            "WaitMsBeforeAsync": 3000,
            "Ratio": 0.5,
            "Flag": True,
            "Chunks": [{"StartLine": 1, "Target": "a — b"}],
            "Name": "café",
        }
        wire = _rev_wire(
            contents=[
                {"role": "user", "parts": [{"text": "q"}]},
                {"role": "model", "parts": [{"function_call": {"name": "run", "args": args}}]},
            ]
        )
        msgs = convert_gemini_request_to_openai(wire, model="m", n=1, max_tokens=1)["messages"]
        arguments = msgs[1]["tool_calls"][0]["function"]["arguments"]
        decoded = _json.loads(arguments)
        assert decoded == args
        assert type(decoded["WaitMsBeforeAsync"]) is int
        assert type(decoded["Chunks"][0]["StartLine"]) is int
        assert type(decoded["Ratio"]) is float
        assert decoded["Flag"] is True
        assert "\\u" not in arguments

    @staticmethod
    def _tool_result_messages(response: Dict[str, Any]) -> List[Dict[str, Any]]:
        wire = _rev_wire(
            contents=[
                {"role": "user", "parts": [{"text": "q"}]},
                {"role": "model", "parts": [{"function_call": {"name": "run", "args": {}}}]},
                {
                    "role": "user",
                    "parts": [{"function_response": {"name": "run", "response": response}}],
                },
            ]
        )
        return convert_gemini_request_to_openai(wire, model="m", n=1, max_tokens=1)["messages"]

    def test_unsupported_part_raises(self):
        wire = _rev_wire(
            contents=[{"role": "user", "parts": [{"inline_data": {"mime_type": "image/png", "data": b"x"}}]}]
        )
        with pytest.raises(NotImplementedError, match="inlineData"):
            convert_gemini_request_to_openai(wire, model="m", n=1, max_tokens=1)

    def test_round_trip_through_forward_converter(self):
        # The forward converter is the reference: OpenAI -> (forward) Gemini
        # -> (reverse) OpenAI must keep the conversation's structure: roles,
        # texts, tool-call names/args, and each tool result answering the
        # call it was paired with.
        original = {
            "model": "m",
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "list files then read a.txt"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "c1", "type": "function",
                         "function": {"name": "bash", "arguments": '{"command": "ls"}'}},
                        {"id": "c2", "type": "function",
                         "function": {"name": "read", "arguments": '{"path": "a.txt"}'}},
                    ],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "a.txt b.txt"},
                {"role": "tool", "tool_call_id": "c2", "content": "hello"},
                {"role": "user", "content": "thanks"},
            ],
            "tools": [
                {"type": "function", "function": {"name": "bash", "parameters": {
                    "type": "object", "properties": {"command": {"type": "string"}}}}},
                {"type": "function", "function": {"name": "read", "parameters": {
                    "type": "object", "properties": {"path": {"type": "string"}}}}},
            ],
        }
        forward = convert_openai_request_to_gemini(ConversionContext(), dict(original))
        wire = _to_wire_dict(
            _GenerateContentRequest.from_json(_json.dumps(forward), ignore_unknown_fields=True)
        )
        back = convert_gemini_request_to_openai(wire, model="m", n=1, max_tokens=1)

        msgs = back["messages"]
        assert [m["role"] for m in msgs] == ["system", "user", "assistant", "tool", "tool", "user"]
        assert msgs[0]["content"] == "be brief"
        assert msgs[1]["content"] == "list files then read a.txt"
        calls = msgs[2]["tool_calls"]
        assert [(c["function"]["name"], _json.loads(c["function"]["arguments"])) for c in calls] == [
            ("bash", {"command": "ls"}),
            ("read", {"path": "a.txt"}),
        ]
        id_by_name = {c["function"]["name"]: c["id"] for c in calls}
        # Plain-text tool results come back byte-identical (the forward
        # converter wraps them as {"result": ...}; the reverse unwraps).
        assert msgs[3] == {"role": "tool", "tool_call_id": id_by_name["bash"], "content": "a.txt b.txt"}
        assert msgs[4] == {"role": "tool", "tool_call_id": id_by_name["read"], "content": "hello"}
        assert msgs[5]["content"] == "thanks"
        assert sorted(t["function"]["name"] for t in back["tools"]) == ["bash", "read"]


class TestConvertOpenaiResponseToGemini:
    @pytest.mark.parametrize(
        "raw_args, malformed",
        [
            ('{"command": "ls"}', False),
            ('{"command": "ls"', True),  # truncated: missing closing brace
            ("[1, 2]", True),  # valid JSON, not an object
            ("", False),  # no arguments == {}
        ],
    )
    def test_unparseable_tool_arguments_mark_malformed(self, raw_args, malformed):
        # Gemini reports a call it could not form as MALFORMED_FUNCTION_CALL;
        # an OpenAI server says finish_reason "tool_calls" either way, so the
        # converter must flag it, or the turn loop runs it as a clean STOP.
        data = {"choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "tool_calls": [{"id": "c", "type": "function",
                            "function": {"name": "bash", "arguments": raw_args}}]}}]}
        [cand] = convert_openai_response_to_gemini(data)["candidates"]
        assert cand["finishReason"] == ("MALFORMED_FUNCTION_CALL" if malformed else "STOP")
        # The call stays visible in the step either way.
        [part] = cand["content"]["parts"]
        assert part["functionCall"]["name"] == "bash"

    def test_tool_call_and_usage(self):
        out = convert_openai_response_to_gemini(_REV_TOOL_CALL_RESPONSE)
        assert out["candidates"] == [
            {
                "index": 0,
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "I should call get_weather.", "thought": True},
                        {"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}},
                    ],
                },
                "finishReason": "STOP",
                "rawChoice": _REV_TOOL_CALL_RESPONSE["choices"][0],
            }
        ]
        # completion_tokens (80) includes reasoning (37).
        assert out["usageMetadata"] == {
            "candidatesTokenCount": 43,
            "thoughtsTokenCount": 37,
            "promptTokenCount": 179,
        }
        assert out["responseId"] == _REV_TOOL_CALL_RESPONSE["id"]
        raw = out["rawChatCompletion"]
        assert "choices" not in raw
        assert raw["usage"] == _REV_TOOL_CALL_RESPONSE["usage"]

    def test_raw_capture_slices_carry_only_their_own_choice(self):
        # A sampler writes one raw-capture blob per candidate: envelope
        # (everything but "candidates") + that one candidate. With n choices,
        # no blob may carry the other n-1 completions.
        data = {
            "id": "r",
            "choices": [
                {"index": i, "message": {"content": f"answer {i}"}, "finish_reason": "stop"}
                for i in range(3)
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 9},
        }
        out = convert_openai_response_to_gemini(data)
        envelope = {k: v for k, v in out.items() if k != "candidates"}
        assert "answer" not in _json.dumps(envelope)
        for i, cand in enumerate(out["candidates"]):
            blob = _json.dumps({**envelope, "candidates": [cand]})
            assert blob.count("answer ") == 2  # rawChoice + parsed text part
            assert f"answer {i}" in blob

    def test_dict_valued_tool_arguments_are_accepted(self):
        data = {"choices": [{"index": 0, "message": {
            "tool_calls": [{"function": {"name": "f", "arguments": {"a": 1}}}]}}]}
        part = convert_openai_response_to_gemini(data)["candidates"][0]["content"]["parts"][0]
        assert part == {"functionCall": {"name": "f", "args": {"a": 1}}}

    @pytest.mark.parametrize(
        "reason,expected", [("length", "MAX_TOKENS"), ("stop", "STOP"), ("weird", "OTHER")]
    )
    def test_finish_reason_map(self, reason: str, expected: str):
        data = {"choices": [{"index": 0, "message": {"content": "x"}, "finish_reason": reason}]}
        assert convert_openai_response_to_gemini(data)["candidates"][0]["finishReason"] == expected

    def test_malformed_tool_arguments_are_preserved(self):
        data = {"choices": [{"index": 0, "message": {
            "tool_calls": [{"function": {"name": "f", "arguments": "{bad"}}]}}]}
        part = convert_openai_response_to_gemini(data)["candidates"][0]["content"]["parts"][0]
        assert part == {"functionCall": {"name": "f", "args": {"__raw_arguments__": "{bad"}}}

    def test_round_trip_through_forward_converter(self):
        # OpenAI response -> (reverse) Gemini -> (forward) OpenAI must keep
        # text, reasoning, tool calls, finish reason and token counts.
        g = convert_openai_response_to_gemini(_REV_TOOL_CALL_RESPONSE)
        proto = _GenerateContentResponse.from_json(_json.dumps(g), ignore_unknown_fields=True)
        back = convert_gemini_response_to_openai(proto).to_dict()
        choice = back["choices"][0]
        msg = choice["message"]
        assert [tb["thinking"] for tb in msg["thinking_blocks"]] == ["I should call get_weather."]
        assert [(t["function"]["name"], _json.loads(t["function"]["arguments"]))
                for t in msg["tool_calls"]] == [("get_weather", {"city": "Paris"})]
        assert choice["finish_reason"] == "tool_calls"
        assert back["usage"]["prompt_tokens"] == 179
        assert back["usage"]["completion_tokens"] == 80
        assert back["usage"]["completion_tokens_details"]["reasoning_tokens"] == 37


class TestReasoningFieldNormalization:
    def test_reasoning_content_to_thinking_blocks(self):
        raw = _json.loads(_json.dumps(_REV_TOOL_CALL_RESPONSE))
        out = reasoning_content_to_thinking_blocks(raw)
        assert raw == _REV_TOOL_CALL_RESPONSE  # input untouched
        msg = out["choices"][0]["message"]
        assert "reasoning_content" not in msg
        assert msg["thinking_blocks"] == [
            {"type": "thinking", "thinking": "I should call get_weather."}
        ]
        assert out["usage"]["completion_tokens_details"] == {"reasoning_tokens": 37}
        # Parses into the response model with the reasoning kept.
        parsed = ChatCompletionResponse.from_dict(out).to_dict()
        assert parsed["choices"][0]["message"]["thinking_blocks"][0]["thinking"] == (
            "I should call get_weather."
        )

    def test_existing_thinking_blocks_and_details_win(self):
        raw = {
            "choices": [{"message": {"reasoning_content": "r",
                                     "thinking_blocks": [{"type": "thinking", "thinking": "t"}]}}],
            "usage": {"reasoning_tokens": 9, "completion_tokens_details": {"reasoning_tokens": 4}},
        }
        out = reasoning_content_to_thinking_blocks(raw)
        assert out["choices"][0]["message"]["thinking_blocks"][0]["thinking"] == "t"
        assert out["usage"]["completion_tokens_details"]["reasoning_tokens"] == 4

    def test_thinking_blocks_to_reasoning_content(self):
        request = {
            "messages": [
                {"role": "user", "content": "q",
                 "thinking_blocks": [{"type": "thinking", "thinking": "user turns untouched"}]},
                {"role": "assistant", "content": "a",
                 "thinking_blocks": [{"type": "thinking", "thinking": "a"},
                                     {"type": "thinking", "thinking": "b"}]},
                {"role": "assistant", "content": "c", "thinking_blocks": []},
            ],
            "max_completion_tokens": 5,
        }
        out = thinking_blocks_to_reasoning_content(request)
        assert request["messages"][1]["thinking_blocks"]  # input untouched
        assert out["messages"][0] == request["messages"][0]
        assert out["messages"][1] == {"role": "assistant", "content": "a",
                                      "reasoning_content": "ab"}
        assert out["messages"][2] == {"role": "assistant", "content": "c"}
        assert out["max_completion_tokens"] == 5


class TestReverseReviewFixes:
    def test_usage_without_completion_tokens_omits_generation_counts(self):
        data = {"choices": [{"index": 0, "message": {"content": "x"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5}}
        usage_md = convert_openai_response_to_gemini(data)["usageMetadata"]
        assert usage_md == {"promptTokenCount": 5}

    def test_completion_tokens_without_reasoning_omits_thoughts_count(self):
        data = {"choices": [{"index": 0, "message": {"content": "x"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 9}}
        usage_md = convert_openai_response_to_gemini(data)["usageMetadata"]
        assert usage_md == {"candidatesTokenCount": 9, "promptTokenCount": 5}

    def test_parent_sampler_fails_loudly_on_content_without_counts(self):
        import asyncio

        from backends.gemini_api_endpoint import GeminiApiSampler

        data = {"choices": [{"index": 0, "message": {"content": "x"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5}}
        converted = convert_openai_response_to_gemini(data)

        async def _post(body: Dict[str, Any]) -> Dict[str, Any]:
            return converted

        sampler = object.__new__(GeminiApiSampler)
        sampler.include_thoughts = None
        sampler._post_with_retries = _post  # type: ignore[method-assign]
        with pytest.raises(ValueError, match="candidatesTokenCount"):
            asyncio.run(sampler.sample_n("hi", num_samples=1))

    def test_missing_choice_index_falls_back_to_position(self):
        data = {"choices": [{"message": {"content": "a"}}, {"message": {"content": "b"}}]}
        cands = convert_openai_response_to_gemini(data)["candidates"]
        assert [c["index"] for c in cands] == [0, 1]

    def test_multi_part_text_concatenates_without_separator(self):
        wire = _rev_wire(
            system_instruction={"parts": [{"text": "s1"}, {"text": "s2"}]},
            contents=[
                {"role": "user", "parts": [{"text": "a"}, {"text": "b"}]},
                {"role": "model", "parts": [{"text": "t1", "thought": True},
                                            {"text": "t2", "thought": True},
                                            {"text": "c"}, {"text": "d"}]},
            ],
        )
        msgs = convert_gemini_request_to_openai(wire, model="m", n=1, max_tokens=1)["messages"]
        assert msgs[0]["content"] == "s1s2"
        assert msgs[1]["content"] == "ab"
        assert msgs[2]["content"] == "cd"
        assert msgs[2]["reasoning_content"] == "t1t2"

    def test_any_with_multiple_allowed_names_filters_tools(self):
        decl = lambda name: {"name": name, "parameters": {"type": "OBJECT"}}  # noqa: E731
        wire = _rev_wire(
            contents=[{"role": "user", "parts": [{"text": "q"}]}],
            tools=[{"function_declarations": [decl("a"), decl("b"), decl("c")]}],
            tool_config={"function_calling_config": {
                "mode": "ANY", "allowed_function_names": ["a", "c"]}},
        )
        body = convert_gemini_request_to_openai(wire, model="m", n=1, max_tokens=1)
        assert body["tool_choice"] == "required"
        assert [t["function"]["name"] for t in body["tools"]] == ["a", "c"]

    def test_any_without_allowed_names_keeps_all_tools(self):
        decl = lambda name: {"name": name, "parameters": {"type": "OBJECT"}}  # noqa: E731
        wire = _rev_wire(
            contents=[{"role": "user", "parts": [{"text": "q"}]}],
            tools=[{"function_declarations": [decl("a"), decl("b")]}],
            tool_config={"function_calling_config": {"mode": "ANY"}},
        )
        body = convert_gemini_request_to_openai(wire, model="m", n=1, max_tokens=1)
        assert body["tool_choice"] == "required"
        assert [t["function"]["name"] for t in body["tools"]] == ["a", "b"]

    def test_vllm_reasoning_field_becomes_thought_part(self):
        data = {"choices": [{"index": 0, "message": {"content": "x", "reasoning": "r"}}]}
        parts = convert_openai_response_to_gemini(data)["candidates"][0]["content"]["parts"]
        assert parts == [{"text": "r", "thought": True}, {"text": "x"}]

    def test_vllm_reasoning_field_becomes_thinking_blocks(self):
        raw = {"choices": [{"message": {"content": "x", "reasoning": "r"}}]}
        msg = reasoning_content_to_thinking_blocks(raw)["choices"][0]["message"]
        assert msg == {"content": "x", "thinking_blocks": [{"type": "thinking", "thinking": "r"}]}

    def test_reasoning_content_wins_over_reasoning(self):
        data = {"choices": [{"index": 0, "message": {"reasoning_content": "rc", "reasoning": "r"}}]}
        parts = convert_openai_response_to_gemini(data)["candidates"][0]["content"]["parts"]
        assert parts == [{"text": "rc", "thought": True}]
        msg = reasoning_content_to_thinking_blocks(data)["choices"][0]["message"]
        assert msg["thinking_blocks"] == [{"type": "thinking", "thinking": "rc"}]
        assert "reasoning" not in msg


class TestConvertGeminiRequestToOpenaiReviewRound2:
    _Q = [{"role": "user", "parts": [{"text": "q"}]}]

    def _convert(self, **wire: Any) -> Dict[str, Any]:
        return convert_gemini_request_to_openai(
            {"contents": self._Q, **wire}, model="m", n=1, max_tokens=1
        )

    def test_any_with_no_matching_allowed_names_raises(self):
        with pytest.raises(ValueError, match=r"\['nope', 'zzz'\]"):
            self._convert(
                tools=[{"functionDeclarations": [{"name": "a"}, {"name": "b"}]}],
                toolConfig={"functionCallingConfig": {
                    "mode": "ANY", "allowedFunctionNames": ["nope", "zzz"]}},
            )

    def test_any_with_partially_matching_allowed_names_keeps_matches(self):
        body = self._convert(
            tools=[{"functionDeclarations": [{"name": "a"}, {"name": "b"}]}],
            toolConfig={"functionCallingConfig": {
                "mode": "ANY", "allowedFunctionNames": ["a", "nope"]}},
        )
        assert body["tool_choice"] == "required"
        assert [t["function"]["name"] for t in body["tools"]] == ["a"]

    def test_json_mime_without_schema_is_json_object(self):
        body = self._convert(generationConfig={"responseMimeType": "application/json"})
        assert body["response_format"] == {"type": "json_object"}

    def test_response_json_schema_passes_through(self):
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        body = self._convert(generationConfig={
            "responseMimeType": "application/json", "responseJsonSchema": schema})
        assert body["response_format"] == {
            "type": "json_schema", "json_schema": {"name": "response", "schema": schema}}

    def test_response_schema_is_converted_like_function_parameters(self):
        schema = {"type": "OBJECT", "properties": {"x": {"type": "INTEGER"}},
                  "propertyOrdering": ["x"]}
        body = self._convert(generationConfig={
            "responseMimeType": "application/json", "responseSchema": schema})
        assert body["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": {
                "type": "object", "properties": {"x": {"type": "integer"}}}},
        }

    def test_response_schema_via_proto(self):
        wire = _rev_wire(
            contents=self._Q,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": {"type": "OBJECT", "properties": {"x": {"type": "STRING"}}},
            },
        )
        body = convert_gemini_request_to_openai(wire, model="m", n=1, max_tokens=1)
        assert body["response_format"]["json_schema"]["schema"]["properties"] == {
            "x": {"type": "string"}}

    def test_other_mime_type_is_not_implemented(self):
        with pytest.raises(NotImplementedError, match="text/x.enum"):
            self._convert(generationConfig={"responseMimeType": "text/x.enum"})

    @pytest.mark.parametrize("cfg", [{}, {"responseMimeType": "text/plain"}])
    def test_text_or_absent_mime_adds_nothing(self, cfg: Dict[str, Any]):
        assert "response_format" not in self._convert(generationConfig=cfg)

    def test_single_allowed_name_not_declared_raises(self):
        with pytest.raises(ValueError, match=r"\['nope'\]"):
            self._convert(
                tools=[{"functionDeclarations": [{"name": "a"}, {"name": "b"}]}],
                toolConfig={"functionCallingConfig": {
                    "mode": "ANY", "allowedFunctionNames": ["nope"]}},
            )

    def test_single_allowed_name_declared_is_named_choice(self):
        body = self._convert(
            tools=[{"functionDeclarations": [{"name": "a"}, {"name": "b"}]}],
            toolConfig={"functionCallingConfig": {
                "mode": "ANY", "allowedFunctionNames": ["a"]}},
        )
        assert body["tool_choice"] == {"type": "function", "function": {"name": "a"}}

    def test_response_schema_nullable_becomes_null_type(self):
        schema = {"type": "OBJECT", "nullable": True, "properties": {
            "x": {"type": "STRING", "nullable": True},
            "y": {"type": "INTEGER", "nullable": False},
            "z": {"nullable": True, "description": "d"},
        }}
        body = self._convert(generationConfig={
            "responseMimeType": "application/json", "responseSchema": schema})
        assert body["response_format"]["json_schema"]["schema"] == {
            "type": ["object", "null"], "properties": {
                "x": {"type": ["string", "null"]},
                "y": {"type": "integer"},
                "z": {"description": "d"},
            }}

    def test_tool_parameters_nullable_becomes_null_type(self):
        body = self._convert(tools=[{"functionDeclarations": [{
            "name": "f", "parameters": {"type": "OBJECT", "properties": {
                "x": {"type": "ARRAY", "nullable": True, "items": {"type": "STRING"}}}},
        }]}])
        assert body["tools"][0]["function"]["parameters"] == {
            "type": "object", "properties": {
                "x": {"type": ["array", "null"], "items": {"type": "string"}}}}
