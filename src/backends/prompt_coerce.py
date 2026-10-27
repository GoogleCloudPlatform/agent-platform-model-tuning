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

"""Boundary helpers for prompt coercion and wire marshaling.

Prompts are carried as ``GenerateContentRequest | str`` (proto plus the
legacy str shortcut for single-turn text). Untyped payloads (dict
literals in tests, JSON from a dataset row) are normalized via
``to_request``; backends about to marshal to REST JSON call
``to_wire_dict`` for the camelCase dict Vertex expects.

Boundaries::

    dataset dict --(env.get_prompt / to_request)--> GenerateContentRequest
    GenerateContentRequest --(to_wire_dict)--> camelCase JSON dict --> REST

``to_wire_dict`` always returns a fresh dict; callers may mutate it
(e.g. to stamp ``candidateCount`` / ``maxOutputTokens``) safely.
"""

from __future__ import annotations

from typing import Any, Dict, Union

from google.cloud.aiplatform_v1.types import GenerateContentRequest
from google.protobuf.json_format import MessageToDict


Prompt = Union[GenerateContentRequest, str]


def to_request(prompt: Any) -> Prompt:
    """Normalize any of ``str | dict | GenerateContentRequest`` to ``Prompt``.

    * ``str`` / ``GenerateContentRequest``: passthrough.
    * ``dict``: coerced via ``GenerateContentRequest(dict)``. Requires
      snake_case field names (the proto constructor validates against
      proto names, not JSON aliases).

    Raises ``TypeError`` on anything else.
    """
    if isinstance(prompt, (str, GenerateContentRequest)):
        return prompt
    if isinstance(prompt, dict):
        return GenerateContentRequest(prompt)
    raise TypeError(
        f"Cannot coerce prompt of type {type(prompt).__name__} to "
        "GenerateContentRequest | str; got value: "
        f"{prompt!r:.120}"
    )


def to_wire_dict(prompt: Prompt) -> Dict[str, Any]:
    """Marshal a prompt to the camelCase JSON dict Vertex REST expects.

    Returns a fresh dict on every call; safe for callers to mutate
    without aliasing back to the input proto.

    A bare ``str`` is wrapped into the canonical single-turn shape
    ``{"contents": [{"role": "user", "parts": [{"text": s}]}]}``.
    """
    if isinstance(prompt, str):
        return {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    return MessageToDict(prompt._pb, preserving_proto_field_name=False)


__all__ = ["Prompt", "to_request", "to_wire_dict"]
