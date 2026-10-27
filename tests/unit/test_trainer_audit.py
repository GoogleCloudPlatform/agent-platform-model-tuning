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

"""Unit tests for the trainer_audit response-parsing helper."""

import pytest

from core.trainer_audit import parse_candidate_audit


class TestParseCandidateAudit:
    def test_missing_audit_block_returns_zero_counters(self):
        """Response without candidateAudit still emits both counter keys."""
        assert parse_candidate_audit({}) == {
            "audit/used_candidates": 0,
            "audit/rejected_candidates": 0,
        }
        assert parse_candidate_audit({"candidateAudit": None}) == {
            "audit/used_candidates": 0,
            "audit/rejected_candidates": 0,
        }

    def test_empty_audit_lists_emit_zero_counters(self):
        response = {
            "candidateAudit": {
                "usedCandidates": [],
                "rejectedCandidates": [],
            }
        }
        assert parse_candidate_audit(response) == {
            "audit/used_candidates": 0,
            "audit/rejected_candidates": 0,
        }

    def test_used_and_rejected_counts(self):
        response = {
            "candidateAudit": {
                "usedCandidates": [{}, {}, {}],
                "rejectedCandidates": [
                    {"reason": "malformed"},
                    {"reason": "timeout"},
                    {"reason": "malformed"},
                    {"reason": "malformed"},
                ],
            }
        }
        assert parse_candidate_audit(response) == {
            "audit/used_candidates": 3,
            "audit/rejected_candidates": 4,
            "audit/rejected_reason/malformed": 3,
            "audit/rejected_reason/timeout": 1,
        }

    def test_missing_reason_field_becomes_unknown(self):
        response = {
            "candidateAudit": {
                "usedCandidates": [],
                "rejectedCandidates": [{}],
            }
        }
        assert parse_candidate_audit(response) == {
            "audit/used_candidates": 0,
            "audit/rejected_candidates": 1,
            "audit/rejected_reason/unknown": 1,
        }

    def test_top_5_reason_cap(self):
        """More than 5 distinct reasons: keep the top 5 by count."""
        rejected = []
        for reason, n in (
            ("r1", 10),
            ("r2", 8),
            ("r3", 6),
            ("r4", 4),
            ("r5", 2),
            ("r6", 1),
            ("r7", 1),
        ):
            rejected.extend([{"reason": reason}] * n)
        response = {
            "candidateAudit": {
                "usedCandidates": [],
                "rejectedCandidates": rejected,
            }
        }
        out = parse_candidate_audit(response)
        assert out["audit/used_candidates"] == 0
        assert out["audit/rejected_candidates"] == 32
        # Top 5: r1..r5. r6 and r7 dropped despite being present in the
        # input (the total count is still accurate; just no per-reason
        # bucket for them).
        top5_keys = {
            f"audit/rejected_reason/{r}" for r in ("r1", "r2", "r3", "r4", "r5")
        }
        assert top5_keys.issubset(out.keys())
        assert "audit/rejected_reason/r6" not in out
        assert "audit/rejected_reason/r7" not in out

    def test_top_5_reason_cap_deterministic_tie_break(self):
        """Ties at the top-N boundary resolve by ascending reason name.

        Prevents non-deterministic emission when many reasons share the
        same count (would otherwise depend on dict-insertion order which
        can drift across runs / trajectory arrival orders).
        """
        # Six distinct reasons, all tied at count=1. Insertion order is
        # deliberately reverse-alphabetical to force the sort to reorder.
        rejected = [{"reason": r} for r in ("rz", "ry", "rx", "rc", "rb", "ra")]
        response = {
            "candidateAudit": {
                "usedCandidates": [],
                "rejectedCandidates": rejected,
            }
        }
        out = parse_candidate_audit(response)
        # Top 5 by (-count, +name) = ra, rb, rc, rx, ry (rz dropped).
        for keeper in ("ra", "rb", "rc", "rx", "ry"):
            assert f"audit/rejected_reason/{keeper}" in out
        assert "audit/rejected_reason/rz" not in out


class TestParseCandidateAuditMalformed:
    """Malformed / off-shape audit-block coverage.

    Real-world callers pass server-validated JSON so these shapes are
    rare, but the helper should either handle them gracefully (returning
    zero-counter dicts) or raise a clear ``AttributeError`` /
    ``TypeError`` rather than silently corrupting metrics.
    """

    def test_candidate_audit_none_treated_as_missing(self):
        assert parse_candidate_audit({"candidateAudit": None}) == {
            "audit/used_candidates": 0,
            "audit/rejected_candidates": 0,
        }

    def test_used_candidates_none_treated_as_empty_list(self):
        response = {
            "candidateAudit": {
                "usedCandidates": None,
                "rejectedCandidates": [{"reason": "x"}],
            }
        }
        out = parse_candidate_audit(response)
        assert out["audit/used_candidates"] == 0
        assert out["audit/rejected_candidates"] == 1
        assert out["audit/rejected_reason/x"] == 1

    def test_rejected_candidates_none_treated_as_empty_list(self):
        response = {
            "candidateAudit": {
                "usedCandidates": [{}, {}],
                "rejectedCandidates": None,
            }
        }
        out = parse_candidate_audit(response)
        assert out == {
            "audit/used_candidates": 2,
            "audit/rejected_candidates": 0,
        }

    def test_rejected_entry_missing_reason_key_becomes_unknown(self):
        response = {
            "candidateAudit": {
                "usedCandidates": [],
                "rejectedCandidates": [{}, {"other_field": "x"}, {"reason": "y"}],
            }
        }
        out = parse_candidate_audit(response)
        assert out["audit/rejected_reason/unknown"] == 2
        assert out["audit/rejected_reason/y"] == 1

    def test_candidate_audit_not_a_dict_raises(self):
        """Non-dict candidateAudit fails fast — server is contract-broken."""
        with pytest.raises(AttributeError):
            parse_candidate_audit({"candidateAudit": "not-a-dict"})
