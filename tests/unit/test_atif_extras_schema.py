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

"""Tests for the typed extras models in src/core/atif_extras.py."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.atif_extras import (
    CloudsideFinalMetricsExtras,
    CloudsideStepExtras,
    CloudsideTrajectoryExtras,
)


def test_step_extras_accepts_all_documented_keys():
    m = CloudsideStepExtras(
        candidate_id="c1",
        train_step_id="42",
        finish_reason="stop",
        llm_call_uri="grpc://x",
        operation_id="op-1",
        advantage=0.5,
        sample_time_s=0.4,
        env_time_s=0.1,
        prompt_token_count=100,
        generation_token_count=25,
    )
    assert m.candidate_id == "c1"
    assert m.prompt_token_count == 100
    assert m.generation_token_count == 25


def test_step_extras_rejects_unknown_key():
    with pytest.raises(ValidationError):
        CloudsideStepExtras(train_step_ids="42")  # typo (plural)


def test_step_extras_rejects_type_mismatch():
    with pytest.raises(ValidationError):
        CloudsideStepExtras(advantage="not-a-float")


def test_train_step_id_must_be_string_per_atif_spec():
    """ATIF v1.7 requires str for training-endpoint identifiers."""
    with pytest.raises(ValidationError):
        CloudsideStepExtras(train_step_id=42)  # int, not str


def test_trajectory_extras_accepts_is_correct_and_status():
    m = CloudsideTrajectoryExtras(
        is_correct=True,
        rollout_status="OK",
        task_stats={"score": 1.0},
        task_step_stats=[{"tool": "grep"}],
    )
    assert m.is_correct is True


def test_trajectory_extras_rejects_reward_leak():
    with pytest.raises(ValidationError):
        CloudsideTrajectoryExtras(reward=0.7)


def test_final_metrics_extras_types():
    m = CloudsideFinalMetricsExtras(total_sample_time_s=1.5, total_env_time_s=0.3)
    assert m.total_sample_time_s == 1.5


def test_step_extras_does_not_have_raw_field():
    """B1 fix: RawWire and CloudsideStepExtras.raw were dead code
    (never populated by any producer). Removed. Test locks in absence."""
    assert "raw" not in CloudsideStepExtras.model_fields


def test_compaction_keys_are_declared():
    """Context-compaction bookkeeping keys must be known to the schema,
    otherwise extra="forbid" would reject the compaction step.

    The compaction_ prefix is part of the contract: .extra is one flat
    namespace already holding prompt_token_count, so a bare pre_tokens
    would be ambiguous for offline consumers that read .extra as columns.
    """
    fields = set(CloudsideStepExtras.model_fields)
    assert {
        "compaction_pre_tokens",
        "compaction_post_tokens",
        "compaction_pairs_elided",
        "compaction_pairs_retained",
    } <= fields


def test_compaction_extras_round_trip():
    m = CloudsideStepExtras(
        compaction_pre_tokens=116_962,
        compaction_post_tokens=19_500,
        compaction_pairs_elided=19,
        compaction_pairs_retained=2,
    )
    assert m.compaction_pre_tokens == 116_962
    assert m.compaction_post_tokens == 19_500
    assert m.compaction_pairs_elided == 19
    assert m.compaction_pairs_retained == 2


def test_compaction_keys_default_to_none():
    """None (not 0) is the correct default: it means "this step is not a
    compaction", whereas 0 would assert "a compaction that elided zero
    pairs" — a different, and false, claim about every ordinary step.

    The default is also load-bearing for artifact size: to_disk dumps with
    exclude_none=True (atif_codec.py), so a None default keeps these keys
    off non-compaction steps entirely, while a 0 default would materialize
    four extra keys on EVERY step of EVERY trajectory artifact.
    """
    m = CloudsideStepExtras()
    assert m.compaction_pre_tokens is None
    assert m.compaction_post_tokens is None
    assert m.compaction_pairs_elided is None
    assert m.compaction_pairs_retained is None


def test_compaction_int_keys_reject_non_numeric_but_coerce_numeric_strings():
    """Mirrors test_step_extras_rejects_type_mismatch for the int fields.

    Pinning ACTUAL pydantic v2 lax-mode behaviour, not the intuitive one:
    a non-numeric string is rejected, and so is a lossy float (19.5), but
    a numeric string "19" IS coerced to int 19. Callers must therefore not
    rely on this model to catch a stringly-typed token count.
    """
    with pytest.raises(ValidationError):
        CloudsideStepExtras(compaction_pre_tokens="not-an-int")
    with pytest.raises(ValidationError):
        CloudsideStepExtras(compaction_pairs_elided=19.5)  # lossy float

    coerced = CloudsideStepExtras(compaction_pairs_elided="19")
    assert coerced.compaction_pairs_elided == 19
