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

"""Regression: session_id carries real ``tuning_job_id`` (from adapter)
and real ``rollout_idx`` (from group_rollout's enumerate position).

Pre-migration state was sentinels: ``tuning_job_id="unknown-tj"`` on
the adapter default, and ``rollout_idx`` from a per-Agent monotonic
counter (bumped in whatever order coroutines happened to hit the
``_build_scored`` line, with no relation to group position).

Post-migration contract:
  1. ``tuning_job_id`` is a REQUIRED kwarg on ``word_puzzle_atif_adapter``;
     there is no default.
  2. ``rollout_idx`` reflects the within-group position (0..N-1),
     assigned by ``group_rollout``'s ``enumerate``. Duplicate-free
     within a single ``group_rollout`` call.

``session_id`` format (from ``core.session.build_session_id``):
  ``{recipe}-{tuning_job_id}-{task_id}-{rollout_idx}-{_RUN_START_ISO}``
"""

from __future__ import annotations

import dataclasses
import re

import pytest

from core.protocols import Action, Env, EnvStepResult, Sampler
from core.task_data import TaskData
from google.cloud.aiplatform_v1.types import Content, GenerateContentRequest


def _prompt_gcr(text: str) -> GenerateContentRequest:
    """Wrap a plain string prompt into a minimal GenerateContentRequest."""
    return GenerateContentRequest(
        contents=[Content(role="user", parts=[{"text": text}])]
    )


# This file covers the SingleStepAgent path; other recipe agents' coverage
# lives alongside their own trees.


@dataclasses.dataclass
class _TestTaskData(TaskData):
    """Lightweight TaskData subclass for session_id-threading tests."""

    pass


class _FakeEnv(Env):
    def get_prompt(self) -> GenerateContentRequest:
        return _prompt_gcr("p")

    async def step(self, action: Action, *, step_count: int = 0) -> EnvStepResult:  # type: ignore[override]
        return EnvStepResult(
            function_responses=[],
            reward=0.5,
            done=True,
            is_correct=True,
            task_stats=None,
        )


class _FakeSampler(Sampler):
    """Sampler that returns N canned SampleResults on demand.

    Uses the shared test factory so the SampleResult carries a real
    tuning_candidate.candidate.content and passes the SingleStepAgent
    happy-path checks (candidate_id != "" etc.).
    """

    model_name = "test-model"

    def __init__(self, n: int):
        self._n = n

    async def sample_n(
        self,
        prompt,
        num_samples: int = 1,
        **kwargs,
    ):
        from tests.unit._sample_result_factories import make_sample_result

        return [
            make_sample_result(candidate_id=f"c{i}", text=f"a{i}")
            for i in range(num_samples)
        ]


def _agent_kwargs(tuning_job_id: str) -> dict:
    return dict(
        recipe_name="rcp",
        tuning_job_id=tuning_job_id,
        extract_prompt_text=lambda td: "x",
    )


# --------------------------------------------------------------------------
# tuning_job_id threading
# --------------------------------------------------------------------------


def test_single_step_agent_requires_tuning_job_id():
    """``tuning_job_id`` is a required kwarg: constructing without it
    raises TypeError."""
    from core.single_step_agent import SingleStepAgent

    with pytest.raises(TypeError):
        SingleStepAgent(
            _FakeSampler(n=1),
            lambda td: _FakeEnv(),
            recipe_name="rcp",
            extract_prompt_text=lambda td: "x",
        )  # type: ignore[call-arg]  # missing tuning_job_id


@pytest.mark.asyncio
async def test_single_step_agent_session_id_carries_tuning_job_id():
    """``session_id`` string embeds the adapter's ``tuning_job_id``."""
    from core.single_step_agent import SingleStepAgent

    sampler = _FakeSampler(n=1)
    agent = SingleStepAgent(
        sampler,
        lambda td: _FakeEnv(),
        **_agent_kwargs("tj-42"),
    )
    group, _ = await agent.group_rollout(_TestTaskData(task_id="t1"), num_samples=1)
    sid = group.scored_trajectories[0].trajectory.session_id
    assert sid is not None
    assert "-tj-42-" in sid, f"tuning_job_id absent from session_id: {sid!r}"


# --------------------------------------------------------------------------
# rollout_idx threading (within-group position 0..N-1)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_step_group_rollout_assigns_positional_rollout_idx():
    """``group_rollout(num_samples=4)`` yields session_ids with tails
    ``-0-<iso>``, ``-1-<iso>``, ``-2-<iso>``, ``-3-<iso>`` — positions
    0..N-1 exactly, no duplicates, no shift."""
    from core.single_step_agent import SingleStepAgent

    sampler = _FakeSampler(n=4)
    agent = SingleStepAgent(
        sampler,
        lambda td: _FakeEnv(),
        **_agent_kwargs("tj-1"),
    )
    group, _ = await agent.group_rollout(_TestTaskData(task_id="t1"), num_samples=4)
    sids = [g.trajectory.session_id for g in group.scored_trajectories]

    # Extract the rollout_idx component from each session_id. Format is
    # rcp-tj-1-t1-<idx>-<iso>; we want the integer right before the ISO.
    # Match "-<digits>-<20-char-ISO>" at the end.
    tails = []
    for sid in sids:
        assert sid is not None
        m = re.search(r"-(\d+)-\d{8}T\d{6}Z(?:-\d+)?$", sid)
        assert m is not None, f"session_id shape unexpected: {sid!r}"
        tails.append(int(m.group(1)))

    assert sorted(tails) == [0, 1, 2, 3], (
        f"expected group positions 0..3, got {sorted(tails)} from {sids}"
    )


# --------------------------------------------------------------------------
# One mint per rollout (the session_id also names the artifact directory)
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_step_agent_ledger_and_trajectory_share_one_session_id():
    """Regression: ``_group_rollout_internal`` used to call
    ``_make_session_id`` twice per sample -- once for the raw-capture
    context, once for the trajectory -- yielding two DIFFERENT ids (the
    builder's counter advances on every call).

    Harmless while the trial dir came from ``task_id + rollout_idx``, but
    the dir is now the session_id, so a second mint would put the raw-LLM
    ledger in a different directory than its own trajectory.
    """
    from core.single_step_agent import SingleStepAgent

    seen_ctx_session_ids: list[str] = []

    class _RecordingCallback:
        async def capture_ctx_for(self, *, task_id, rollout_idx, session_id):
            seen_ctx_session_ids.append(session_id)
            return None

        async def on_trajectory_started(self, task_data, traj=None): ...
        async def on_trajectory_step_completed(self, task_data, traj=None): ...
        async def on_trajectory_produced(self, scored, task_data, traj=None): ...
        async def on_trajectory_dropped(self, *a, **kw): ...

    agent = SingleStepAgent(
        _FakeSampler(n=3),
        lambda td: _FakeEnv(),
        trajectory_callback=_RecordingCallback(),
        **_agent_kwargs("tj-1"),
    )
    group, _ = await agent.group_rollout(_TestTaskData(task_id="t1"), num_samples=3)

    traj_sids = sorted(g.trajectory.session_id for g in group.scored_trajectories)
    assert sorted(seen_ctx_session_ids) == traj_sids, (
        "capture ctx and trajectory must share one minted session_id per "
        f"rollout; ctx={sorted(seen_ctx_session_ids)} traj={traj_sids}"
    )


@pytest.mark.asyncio
async def test_run_one_rollout_log_and_trajectory_share_one_session_id():
    """Regression (HarborAgentBase path): ``_run_one_rollout`` minted a
    session_id for the ``agent/agent.log`` trial dir, then the base
    ``rollout`` template minted ANOTHER for the trajectory -- two different
    ids (the builder's counter advances per call), so the log landed in one
    trial dir and the trajectory in a sibling. Since the trial dir IS the
    session_id, that produced an orphan stub dir (agent.log only) per rollout.

    Contract: the id used to bind agent.log == the produced trajectory's id.
    """
    from core.agent import Agent
    from core.atif_trajectory import Reward, ScoredTrajectory

    log_sids: list[str] = []
    trajectory_sids: list[str] = []

    class _BasePathAgent(Agent):
        """Uses the base ``rollout`` template (the HarborAgentBase-style path
        that goes through ``_run_one_rollout``), not SingleStepAgent's override.
        """

        async def _rollout_internal(self, task_data, ctx):  # type: ignore[override]
            # ctx.traj.session_id is what actually names the trajectory dir.
            trajectory_sids.append(ctx.traj.session_id)
            return ScoredTrajectory(trajectory=ctx.traj, reward=Reward(reward=0.0))

        async def _resolve_capture_ctx(self, *, task_id, rollout_idx, session_id):  # type: ignore[override]
            # Called once by _run_one_rollout (to place agent.log) and again by
            # the base rollout() template (for the trajectory's capture ctx).
            # Record EVERY session_id it's called with, in order.
            log_sids.append(session_id)
            return None

    agent = _BasePathAgent(
        _FakeSampler(n=1),
        lambda td: _FakeEnv(),
        recipe_name="rcp",
        tuning_job_id="tj-1",
    )
    await agent._run_one_rollout(
        _TestTaskData(task_id="t1"),
        0,
        thinking_level=None,
        max_generation_tokens=None,
        phase="train",
        dispatch_index=None,
        on_trajectory_done=None,
    )

    # The FIRST capture-ctx call is _run_one_rollout's agent.log placement.
    # It must equal the trajectory's session_id, i.e. exactly ONE id is minted
    # across the whole rollout (no orphan agent.log-only trial dir).
    trajectory_sid = trajectory_sids[0]
    assert all(sid == trajectory_sid for sid in log_sids), (
        "agent.log and trajectory must share ONE session_id per rollout "
        f"(no orphan stub dir); capture-ctx ids={log_sids} "
        f"trajectory={trajectory_sid!r}"
    )


@pytest.mark.asyncio
async def test_single_step_agent_step_reaches_session_id():
    """``step`` (eval sweeps) is folded into every sample's session_id."""
    from core.single_step_agent import SingleStepAgent

    agent = SingleStepAgent(
        _FakeSampler(n=2),
        lambda td: _FakeEnv(),
        **_agent_kwargs("tj-1"),
    )
    group, _ = await agent.group_rollout(
        _TestTaskData(task_id="t1"), num_samples=2, step=12
    )
    for g in group.scored_trajectories:
        assert "-s0012-" in (g.trajectory.session_id or "")
