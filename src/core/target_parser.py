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

"""Parse the unified ``target`` CLI arg into typed routing info."""

from __future__ import annotations

import re
from dataclasses import dataclass

_TUNING_JOB_RE = re.compile(r"^projects/([^/]+)/locations/([^/]+)/tuningJobs/([^/]+)$")
_ENDPOINT_RE = re.compile(r"^projects/([^/]+)/locations/([^/]+)/endpoints/([^/]+)$")
_PUBLISHER_MODEL_RE = re.compile(
    r"^projects/([^/]+)/locations/([^/]+)/publishers/[^/]+/models/([^/]+)$"
)


@dataclass(frozen=True)
class TargetInfo:
    """Parsed result of a ``target`` CLI arg.

    Exactly one of ``vertex_existing_job_id``, ``endpoint_id``, or
    ``model`` is set (unless ``target`` was None, in which case all are
    None).  ``project`` and ``location`` are set when the target was a
    full resource path; None for short model names.
    """

    project: str | None = None
    location: str | None = None
    vertex_existing_job_id: str | None = None
    endpoint_id: str | None = None
    model: str | None = None


def parse_target(target: str | None) -> TargetInfo:
    """Parse a ``target`` string into a :class:`TargetInfo`.

    Accepted forms:

    - ``projects/{p}/locations/{l}/tuningJobs/{id}``
    - ``projects/{p}/locations/{l}/endpoints/{id}``
    - ``projects/{p}/locations/{l}/publishers/{pub}/models/{m}``
    - Short model name (e.g. ``gemini-3.5-flash``)
    - ``None`` (returns an all-None TargetInfo)
    """
    if target is None:
        return TargetInfo()

    m = _TUNING_JOB_RE.match(target)
    if m:
        return TargetInfo(
            project=m.group(1),
            location=m.group(2),
            vertex_existing_job_id=m.group(3),
        )

    m = _ENDPOINT_RE.match(target)
    if m:
        return TargetInfo(
            project=m.group(1),
            location=m.group(2),
            endpoint_id=m.group(3),
        )

    m = _PUBLISHER_MODEL_RE.match(target)
    if m:
        return TargetInfo(
            project=m.group(1),
            location=m.group(2),
            model=m.group(3),
        )

    # If it looks like a resource path but didn't match any known form,
    # reject it rather than silently treating it as a model name.
    if target.startswith("projects/"):
        raise ValueError(
            f"Unrecognized target resource path: {target!r}. Expected "
            "projects/{p}/locations/{l}/tuningJobs/{id}, "
            "projects/{p}/locations/{l}/endpoints/{id}, or "
            "projects/{p}/locations/{l}/publishers/{pub}/models/{m}."
        )

    # Short model name.
    return TargetInfo(model=target)
