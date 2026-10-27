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

"""Trainability semantics for ``Step.extra["candidate_id"]``.

A trainable candidate_id is an opaque handle produced by a tuning-scope
backend (e.g. a 32-hex string) that ``train_step`` can resolve back to
the exact sampled candidate. An inference-only sampler instead emits
``str(candidate_index)`` \u2014 '0', '1', ... \u2014 which the tuning backend
CANNOT resolve; steps carrying those bare indices are silently dropped
or mis-resolved at training time.

These helpers surface that mismatch loudly once per group so an
inference-only run can't masquerade as trainable in dashboards. The
schema-level access (reading ``candidate_id`` off ``Step.extra``) lives
in ``core.atif_extras``; the semantic interpretation of that value lives
here.
"""

from __future__ import annotations

import logging

from core.atif_extras import get_candidate_id
from core.atif_trajectory import ScoredTrajectory

logger = logging.getLogger(__name__)


def is_index_like(cid: str) -> bool:
    """A bare positional index ('0'..'99') is NOT a trainable candidate handle."""
    return cid.isdigit() and len(cid) <= 2


def warn_if_untrainable(scored_list: list[ScoredTrajectory]) -> None:
    """Log a warning if agent-step candidate_ids look like bare indices."""
    cids = [
        cid
        for g in scored_list
        for step in g.trajectory.steps
        if step.source == "agent"
        for cid in [get_candidate_id(step)]
        if cid
    ]
    if not cids:
        return
    index_like = [c for c in cids if is_index_like(c)]
    if index_like:
        logger.warning(
            "%d/%d agent-step candidate_id(s) look like bare "
            "INDICES (e.g. %r), NOT trainable handles. Rollouts must sample "
            "through a tuning-scope backend for training; an inference-only "
            "sampler emits candidate indices that train_step cannot resolve.",
            len(index_like),
            len(cids),
            index_like[0],
        )
