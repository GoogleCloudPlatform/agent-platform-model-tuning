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

"""Terse construction of SampleResult instances for consumer tests.

Consumer tests need many SampleResult fixtures; without a factory, each
fixture is a 6-8 line constructor call. This helper flattens the common
case to a single ``make_sample_result(text="...", finish_reason="STOP")``.

Prefixed with underscore so pytest doesn't try to collect it as a test module.
"""

from __future__ import annotations

import json
from typing import Any

from google.cloud.aiplatform_v1.types import (
    Candidate,
    Content,
    GenerateContentRequest,
    Part,
)

from core.sample_result import (
    SampleResult,
    SamplingContext,
    TuningCandidate,
)


def make_sample_result(
    *,
    text: str = "",
    content: dict[str, Any] | None = None,
    finish_reason: str = "STOP",
    candidate_id: str = "cand-0",
    train_step_id: int | None = None,
    generation_token_count: int = 0,
    thoughts_token_count: int = 0,
    prompt: GenerateContentRequest | str | None = None,
    operation_id: str | None = None,
    response_id: str | None = None,
    model_version: str | None = None,
    is_prompt_too_long: bool = False,
    is_malformed_usage: bool = False,
    prompt_token_count: int | None = None,
    raw_call_uri: str | None = None,
) -> SampleResult:
    """Build a SampleResult with sensible defaults for tests.

    Two ways to specify the model output:
      - ``text="hello"`` — quick single-text-part Content.
      - ``content={"role": "model", "parts": [{"function_call": {...}}]}`` —
        arbitrary proto-JSON Content dict (parsed via ``Content.from_json``,
        so ``function_call``/``functionResponse`` etc. work).

    When ``candidate_id=""``, the resulting SampleResult has an empty
    Candidate whose finish_reason is FINISH_REASON_UNSPECIFIED (accessor
    reports ``"OTHER"``). Matches the shape of ``_empty_result`` from
    both backends.
    """
    if candidate_id == "":
        cand = Candidate()
    else:
        if content is not None:
            cand_content = Content.from_json(
                json.dumps(content), ignore_unknown_fields=True
            )
        else:
            parts = [Part(text=text)] if text else []
            cand_content = Content(role="model", parts=parts)
        cand = Candidate(
            content=cand_content,
            finish_reason=Candidate.FinishReason[finish_reason],
        )
    return SampleResult(
        tuning_candidate=TuningCandidate(
            candidate_id=candidate_id,
            candidate=cand,
            generation_token_count=generation_token_count,
            thoughts_token_count=thoughts_token_count,
        ),
        context=SamplingContext(
            prompt=prompt,
            operation_id=operation_id,
            response_id=response_id,
            model_version=model_version,
            is_prompt_too_long=is_prompt_too_long,
            is_malformed_usage=is_malformed_usage,
            prompt_token_count=prompt_token_count,
            raw_call_uri=raw_call_uri,
        ),
        train_step_id=train_step_id,
    )
