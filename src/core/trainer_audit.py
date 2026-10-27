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

"""Trainer audit-response parsing helper.

Pure function so the audit-parsing logic is unit-testable in isolation
and portable across backends. Not a MetricAggregator (this reads from
a trainer wire response, not trajectories).

The ``audit/*`` namespace is a framework contract:
``audit/used_candidates`` / ``audit/rejected_candidates`` are
per-train-step counters; ``audit/rejected_reason/{reason}`` is a
per-reason breakdown capped at the top-N so a degenerate run producing
many unique reasons doesn't blow up the metrics dict.

Backends reuse this helper by shaping their response into
``{"usedCandidates": [...], "rejectedCandidates": [{"reason": str, ...}]}``.
"""

from __future__ import annotations

from typing import Any

# Distinct rejection reasons surfaced as per-reason metric keys;
# beyond this cap, reasons pool into the roll-up counter without a
# dedicated metric.
_TOP_N_REJECTED_REASONS = 5


def parse_candidate_audit(response: dict[str, Any]) -> dict[str, int]:
    """Parse a trainer wire response's ``candidateAudit`` block into metrics.

    Returns a dict of {metric_name: count} with:

    - ``audit/used_candidates``: always present, count of accepted
      candidates for this train step. 0 if the response has no
      ``candidateAudit`` block or no ``usedCandidates`` entry.
    - ``audit/rejected_candidates``: always present, total count of
      rejected candidates. 0 when absent.
    - ``audit/rejected_reason/{reason}``: 0 to _TOP_N_REJECTED_REASONS
      keys, one per most-frequent rejection reason. Omitted when the
      response has no rejected candidates.

    ``response`` is the raw parsed-JSON body of the train_step LRO's
    completed response. Callers should call this AFTER extracting the
    parsed response body from the LRO result envelope.
    """
    audit = response.get("candidateAudit") or {}
    used = audit.get("usedCandidates") or []
    rejected = audit.get("rejectedCandidates") or []

    metrics: dict[str, int] = {
        "audit/used_candidates": len(used),
        "audit/rejected_candidates": len(rejected),
    }

    if rejected:
        reason_counts: dict[str, int] = {}
        for r in rejected:
            reason = r.get("reason", "unknown")
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        # Descending count, ascending name: stable tie-breaking at the
        # top-N cap (otherwise identical-count reasons would emit in
        # arrival-order-dependent dict insertion order).
        for reason, count in sorted(
            reason_counts.items(), key=lambda kv: (-kv[1], kv[0])
        )[:_TOP_N_REJECTED_REASONS]:
            metrics[f"audit/rejected_reason/{reason}"] = count

    return metrics
