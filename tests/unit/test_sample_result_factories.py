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

"""Tests for the make_sample_result test factory helper."""

from __future__ import annotations

from core.sample_result import SampleResult
from tests.unit._sample_result_factories import make_sample_result


class TestMakeSampleResult:
    def test_defaults(self):
        r = make_sample_result()
        assert isinstance(r, SampleResult)
        assert r.completion_text == ""
        assert r.finish_reason == "STOP"
        assert r.tuning_candidate.candidate_id == "cand-0"
        assert r.train_step_id is None

    def test_text_populates_content(self):
        r = make_sample_result(text="hello")
        assert r.completion_text == "hello"

    def test_finish_reason(self):
        r = make_sample_result(finish_reason="MAX_TOKENS")
        assert r.finish_reason == "MAX_TOKENS"

    def test_candidate_id(self):
        r = make_sample_result(candidate_id="cid-42")
        assert r.tuning_candidate.candidate_id == "cid-42"

    def test_train_step_id(self):
        r = make_sample_result(train_step_id=99)
        assert r.train_step_id == 99

    def test_context_kwargs(self):
        r = make_sample_result(
            operation_id="op-1",
            response_id="resp-42",
            is_prompt_too_long=True,
        )
        assert r.context.operation_id == "op-1"
        assert r.context.response_id == "resp-42"
        assert r.context.is_prompt_too_long is True

    def test_failure_result(self):
        r = make_sample_result(candidate_id="", text="")
        assert r.tuning_candidate.candidate_id == ""
        assert r.completion_text == ""
        assert r.finish_reason == "OTHER"

    def test_generation_token_count(self):
        r = make_sample_result(generation_token_count=42, thoughts_token_count=5)
        assert r.tuning_candidate.generation_token_count == 42
        assert r.tuning_candidate.thoughts_token_count == 5
