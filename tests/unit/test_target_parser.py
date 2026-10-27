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

"""Tests for target resource-path parsing."""

import pytest
from core.target_parser import parse_target, TargetInfo


class TestParseTarget:
    def test_tuning_job_full_path(self):
        t = parse_target(
            "projects/123456789012/locations/us-central1/tuningJobs/1234567890123456789"
        )
        assert t.project == "123456789012"
        assert t.location == "us-central1"
        assert t.vertex_existing_job_id == "1234567890123456789"
        assert t.endpoint_id is None
        assert t.model is None

    def test_endpoint_full_path(self):
        t = parse_target("projects/123456789012/locations/us-central1/endpoints/12345")
        assert t.project == "123456789012"
        assert t.location == "us-central1"
        assert t.endpoint_id == "12345"
        assert t.vertex_existing_job_id is None
        assert t.model is None

    def test_publisher_model_full_path(self):
        t = parse_target(
            "projects/test-project/locations/global/publishers/google/models/gemini-3.5-flash"
        )
        assert t.project == "test-project"
        assert t.location == "global"
        assert t.model == "gemini-3.5-flash"
        assert t.endpoint_id is None
        assert t.vertex_existing_job_id is None

    def test_short_model_name(self):
        t = parse_target("gemini-3.5-flash")
        assert t.model == "gemini-3.5-flash"
        assert t.project is None
        assert t.location is None
        assert t.endpoint_id is None
        assert t.vertex_existing_job_id is None

    def test_none_returns_empty(self):
        t = parse_target(None)
        assert t.project is None
        assert t.model is None

    def test_invalid_resource_path_raises(self):
        with pytest.raises(ValueError, match="Unrecognized target"):
            parse_target("projects/foo/locations/bar/somethingElse/baz")
