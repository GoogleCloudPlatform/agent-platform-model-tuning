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

"""Tests for the shared proto conversion helper."""

from __future__ import annotations

import pytest
from google.cloud.aiplatform_v1.types import Candidate

from backends._proto_utils import _json_dict_to_candidate


class TestJsonDictToCandidate:
    def test_standard_stop_response(self):
        raw = {
            "index": 0,
            "finishReason": "STOP",
            "content": {"role": "model", "parts": [{"text": "hello"}]},
        }
        cand = _json_dict_to_candidate(raw)
        assert isinstance(cand, Candidate)
        assert cand.content.parts[0].text == "hello"
        assert cand.finish_reason == Candidate.FinishReason.STOP

    def test_empty_content_parts(self):
        raw = {"index": 0, "finishReason": "STOP", "content": {"role": "model"}}
        cand = _json_dict_to_candidate(raw)
        assert len(cand.content.parts) == 0

    def test_malformed_function_call(self):
        raw = {
            "index": 0,
            "finishReason": "MALFORMED_FUNCTION_CALL",
            "content": {"role": "model", "parts": [{"text": "broken"}]},
        }
        cand = _json_dict_to_candidate(raw)
        assert cand.finish_reason == Candidate.FinishReason.MALFORMED_FUNCTION_CALL

    def test_thought_part(self):
        raw = {
            "index": 0,
            "finishReason": "STOP",
            "content": {
                "role": "model",
                "parts": [
                    {"text": "thinking", "thought": True},
                    {"text": "answer"},
                ],
            },
        }
        cand = _json_dict_to_candidate(raw)
        assert cand.content.parts[0].thought is True
        assert cand.content.parts[1].thought is False

    def test_unknown_fields_ignored(self):
        raw = {
            "index": 0,
            "finishReason": "STOP",
            "content": {"role": "model", "parts": [{"text": "x"}]},
            "someFutureField": "backend-added-later",
        }
        cand = _json_dict_to_candidate(raw)
        assert cand.content.parts[0].text == "x"

    def test_empty_dict(self):
        """Empty dict produces an empty Candidate (used in _empty_result paths)."""
        cand = _json_dict_to_candidate({})
        assert cand.finish_reason == Candidate.FinishReason.FINISH_REASON_UNSPECIFIED
        assert len(cand.content.parts) == 0
