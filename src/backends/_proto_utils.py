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

"""Shared proto conversion utilities for backend samplers.

Two directions:

* ``_json_dict_to_candidate``: raw proto-JSON candidate dict (as returned
  by Vertex ``generateContent`` HTTP responses) -> typed ``Candidate`` proto.
* ``reconstitute_generate_content_response``: ``list[SampleResult]`` ->
  ``GenerateContentResponse`` proto. Used by ``Sampler.sample_chat_completion``
  to bridge back to the proto shape response-side converters expect.

Class-agnostic; depends only on ``google.cloud.aiplatform_v1.types``.
"""

from __future__ import annotations

import json
from typing import Any

from google.cloud.aiplatform_v1.types import Candidate, GenerateContentResponse

from core.sample_result import SampleResult


def _json_dict_to_candidate(raw: dict[str, Any]) -> Candidate:
    """Parse a proto-JSON candidate dict into a ``Candidate`` proto.

    Uses ``Candidate.from_json`` (handles camelCase + string enum values).
    ``ignore_unknown_fields=True`` gives forward-compat against Vertex
    adding response fields not yet modelled here.
    """
    return Candidate.from_json(json.dumps(raw), ignore_unknown_fields=True)


def reconstitute_generate_content_response(
    results: list[SampleResult],
    sampler_name: str = "<unknown sampler>",
) -> GenerateContentResponse:
    """Rebuild a ``GenerateContentResponse`` proto from ``list[SampleResult]``.

    Fail-fast on missing request-scoped metadata (``prompt_token_count``,
    ``response_id``/``operation_id``, ``model_version``): silent defaults
    for numerically load-bearing fields would corrupt downstream
    billing / token-count math.

    Args:
        results: Non-empty list; ``results[0].context`` supplies the
            request-scoped metadata (all candidates share one request).
        sampler_name: Calling sampler's class name, used in error messages.

    Returns:
        A ``GenerateContentResponse`` proto with ``candidates``,
        ``usage_metadata``, ``model_version``, and ``response_id``
        populated. ``create_time`` is left unset (the response-side
        converter falls back to ``time.time()``).

    Raises:
        ValueError: ``results`` is empty, or a required field on
            ``results[0].context`` is None/empty.
    """
    if not results:
        raise ValueError(
            "Cannot reconstitute GenerateContentResponse from empty "
            "results list (sampler returned no candidates)."
        )

    ctx = results[0].context

    if ctx.prompt_token_count is None:
        raise ValueError(
            f"{sampler_name} did not populate "
            f"SamplingContext.prompt_token_count; required for "
            f"ChatCompletionResponse.usage.prompt_tokens (fail-fast "
            f"contract forbids silent 0)."
        )
    resp_id = ctx.response_id or ctx.operation_id
    if not resp_id:
        raise ValueError(
            f"{sampler_name} populated neither response_id nor "
            f"operation_id on SamplingContext; one of them is required "
            f"for ChatCompletionResponse.id."
        )
    if not ctx.model_version:
        raise ValueError(
            f"{sampler_name} did not populate "
            f"SamplingContext.model_version; required for "
            f"ChatCompletionResponse.model."
        )

    response = GenerateContentResponse()
    for r in results:
        response.candidates.append(r.tuning_candidate.candidate)
    response.usage_metadata.prompt_token_count = int(ctx.prompt_token_count)
    response.usage_metadata.candidates_token_count = sum(
        r.tuning_candidate.generation_token_count for r in results
    )
    response.usage_metadata.thoughts_token_count = sum(
        r.tuning_candidate.thoughts_token_count for r in results
    )
    response.usage_metadata.total_token_count = (
        response.usage_metadata.prompt_token_count
        + response.usage_metadata.candidates_token_count
    )
    response.model_version = ctx.model_version
    response.response_id = resp_id
    return response
