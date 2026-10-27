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

"""Base ``TaskData`` dataclass for framework-typed task payloads.

Recipes subclass this with recipe-specific fields.  The framework only
reads ``task_id``; everything else is recipe-opaque and travels through
recipe-owned callbacks (the ``extract_prompt_text`` callable passed to
``SingleStepAgent``, ``env_factory``, etc.).

``task_id`` is **required** — it must be populated at dataset-loading
time (inside the recipe's ``parse_*_item`` transform function, which
calls ``ensure_task_id``).  The Agent template methods reject empty
``task_id`` with a ``ValueError`` as a defence-in-depth check.

Nested per-recipe containers stay typed as ``dict[str, Any]`` on the
subclass (not further dataclasses) to preserve JSON-dump / read-back
round-trip in recipe-owned eval and visualization tools.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class TaskData:
    """Framework base for typed per-task payloads. Recipes subclass this."""

    task_id: str
