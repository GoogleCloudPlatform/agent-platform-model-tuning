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

"""Tests for AgentConfig ABC."""

from __future__ import annotations

import pytest

from core.agent_config import AgentConfig


def test_base_is_abstract():
    with pytest.raises(TypeError):
        AgentConfig()  # abstract: make() not implemented


def test_concrete_subclass_defaults_and_make():
    class _FakeAgentConfig(AgentConfig):
        def make(self, *, sampler, job_id, **kwargs):
            return ("agent", sampler, job_id)

    c = _FakeAgentConfig()
    assert c.thinking_level == "HIGH"
    assert (
        c.agent_rollout_timeout_seconds == 7200.0
    )  # AgentConfig default per-rollout ceiling
    assert c.make(sampler="s", job_id="j") == ("agent", "s", "j")


def test_base_max_generation_tokens_defaults_to_none():
    # The per-call generation cap defaults to None (sampler default 32768).
    # Recipes may override to a tighter budget (e.g. 8192).
    assert AgentConfig.__dataclass_fields__["max_generation_tokens"].default is None


def test_compaction_defaults_are_inert():
    """Compaction must be off unless a config opts in EXPLICITLY.

    Two independent switches, both off: no prompt cap to measure against,
    and a zero event budget. Either alone disables compaction; requiring
    both means a config that sets only ``max_context_tokens`` (e.g. to
    document a backend limit) does not silently start compacting.
    """

    class _FakeAgentConfig(AgentConfig):
        def make(self, *, sampler, job_id, **kwargs):
            return ("agent", sampler, job_id)

    cfg = _FakeAgentConfig()
    assert cfg.max_context_tokens is None
    assert cfg.compaction_trigger_ratio == 0.7
    assert cfg.max_compactions == 0
    # Retained-tail size: the pre-config hard-coded value, so existing
    # configs keep their behavior.
    assert cfg.compaction_tail_pairs == 2
