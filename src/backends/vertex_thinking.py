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

"""Attach a Vertex ``thinking_config.thinking_level`` to a request prompt.

Vertex-specific request-shape knowledge; called from concrete Sampler
implementations in ``src/backends/``, not from agents or the generic
RL framework.
"""

from __future__ import annotations

from typing import Optional, Union

from google.cloud.aiplatform_v1.types import GenerateContentRequest

# Matches ``GenerationConfig.ThinkingConfig.ThinkingLevel`` enum names.
# Input is case-insensitive; normalized upper.
VALID_THINKING_LEVELS = frozenset({"MINIMAL", "LOW", "MEDIUM", "HIGH"})


def apply_thinking_level(
    prompt: Union[GenerateContentRequest, str],
    thinking_level: Optional[str],
) -> Union[GenerateContentRequest, str]:
    """Return a prompt carrying ``generation_config.thinking_config.thinking_level``.

    No-op when ``thinking_level`` is falsy. A str prompt is wrapped into
    a minimal single-turn ``GenerateContentRequest``; a proto prompt is
    copy-constructed so the caller's object is never mutated.

    ``MessageToDict`` later marshals ``thinking_config.thinking_level``
    to camelCase ``thinkingConfig.thinkingLevel`` on the outbound REST
    JSON (the field name Vertex expects).
    """
    if not thinking_level:
        return prompt
    level = str(thinking_level).strip().upper()
    if level not in VALID_THINKING_LEVELS:
        raise ValueError(
            f"invalid thinking_level {thinking_level!r}; expected one of "
            f"{sorted(VALID_THINKING_LEVELS)}"
        )
    if isinstance(prompt, str):
        p = GenerateContentRequest(
            contents=[{"role": "user", "parts": [{"text": prompt}]}]
        )
    else:
        # Copy so we don't mutate the caller's proto.
        p = GenerateContentRequest(prompt)
    p.generation_config.thinking_config.thinking_level = level
    return p
