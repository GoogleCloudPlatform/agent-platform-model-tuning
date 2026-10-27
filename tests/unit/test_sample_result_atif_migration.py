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

"""SampleResult.to_step_stats returns atif_trajectory.StepStats.

Regression test: after migration, to_step_stats() must return the NEW
StepStats (from core.atif_trajectory), not the OLD one (from core.trajectory).
The new one is consumable by TrajectoryStep.from_wire().
"""

from __future__ import annotations

from unittest.mock import MagicMock

from core.atif_trajectory import StepStats as AtifStepStats
from core.atif_trajectory import TrajectoryStep


def _make_sample_result_stub(**kwargs):
    """Build a minimal SampleResult-shaped object that to_step_stats can consume.

    We stub rather than construct a real SampleResult to keep the test
    hermetic. If SampleResult.to_step_stats() reads a field we haven't
    stubbed, the test will fail loudly.
    """
    from core.sample_result import SampleResult

    sr = MagicMock(spec=SampleResult)
    # Delegate to real to_step_stats implementation by binding it:
    sr.to_step_stats = SampleResult.to_step_stats.__get__(sr, SampleResult)

    # Fields to_step_stats() reads. Defaults keep everything None-ish.
    sr.finish_reason = kwargs.get("finish_reason")
    sr.tuning_candidate = MagicMock()
    sr.tuning_candidate.candidate_id = kwargs.get("candidate_id") or ""
    sr.context = MagicMock()
    sr.context.operation_id = kwargs.get("operation_id")
    sr.prompt_token_count = kwargs.get("prompt_token_count")
    sr.generation_token_count = kwargs.get("generation_token_count")
    sr.train_step_id = kwargs.get("train_step_id")
    return sr


def test_to_step_stats_returns_atif_stepstats_not_legacy():
    """The critical migration guarantee."""
    sr = _make_sample_result_stub(
        finish_reason="stop",
        candidate_id="c-9",
        train_step_id=42,
        operation_id="op-x",
        prompt_token_count=100,
        generation_token_count=25,
    )
    stats = sr.to_step_stats()
    assert isinstance(stats, AtifStepStats)
    assert stats.finish_reason == "stop"
    assert stats.candidate_id == "c-9"
    assert stats.train_step_id == 42
    assert stats.operation_id == "op-x"
    assert stats.prompt_token_count == 100
    assert stats.generation_token_count == 25


def test_to_step_stats_output_feeds_from_wire_cleanly():
    """The new StepStats must be consumable by TrajectoryStep.from_wire."""
    sr = _make_sample_result_stub(
        finish_reason="stop",
        candidate_id="c-1",
        train_step_id=5,
        prompt_token_count=50,
        generation_token_count=10,
    )
    stats = sr.to_step_stats()
    step = TrajectoryStep.from_wire(
        step_id=2,
        generation="g",
        stats=stats,
    )
    assert step.candidate_id == "c-1"
    assert step.train_step_id == 5
    assert step.extra["prompt_token_count"] == 50
    assert step.extra["generation_token_count"] == 10


def test_atif_stepstats_has_no_is_correct_field():
    """Per spec Q3: is_correct is trajectory-level, must not exist here."""
    assert not hasattr(AtifStepStats(), "is_correct")
