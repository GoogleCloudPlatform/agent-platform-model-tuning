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

"""Unit tests for _extract_checkpoint_endpoint (pure wire-parse helper)."""

from __future__ import annotations

from backends.vertex_msrl_rest import _extract_checkpoint_endpoint

# Checkpoint-step response body, shaped as the TrainStep LRO returns it.
_CKPT_RESPONSE = {
    "@type": "type.googleapis.com/google.cloud.aiplatform.v1beta1.TrainStepResponse",
    "completedTrainStepId": "30",
    "checkpoint": {
        "checkpointId": "3",
        "step": "30",
        "endpoint": "projects/123456789012/locations/us/endpoints/9876543210987654321",
    },
}


def test_extracts_endpoint_on_checkpoint_step():
    assert (
        _extract_checkpoint_endpoint(_CKPT_RESPONSE)
        == "projects/123456789012/locations/us/endpoints/9876543210987654321"
    )


def test_returns_none_when_checkpoint_absent():
    # Ordinary (non-checkpoint) train step: no `checkpoint` key.
    assert _extract_checkpoint_endpoint({"completedTrainStepId": "31"}) is None


def test_returns_none_when_checkpoint_has_no_endpoint():
    # Defensive: checkpoint object present but endpoint missing/empty.
    assert _extract_checkpoint_endpoint({"checkpoint": {"step": "30"}}) is None
    assert _extract_checkpoint_endpoint({"checkpoint": {"endpoint": ""}}) is None


def test_returns_none_on_empty_or_none_response():
    assert _extract_checkpoint_endpoint({}) is None
    assert _extract_checkpoint_endpoint(None) is None
