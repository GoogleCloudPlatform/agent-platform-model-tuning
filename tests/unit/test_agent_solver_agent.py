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

"""Tests for AgentSolverAgent (the core.agent.Agent adapter).

No Docker, no network: the mini agent is injected via a fake factory that
returns a canned transcript, so these tests exercise ONLY the adapter's
contract -- one scored trajectory per rollout, N per group, merged metrics,
and the timeout path -- not the mini loop itself (covered by the converter
tests + the runner smoke).
"""

from __future__ import annotations

import asyncio
from typing import Any

from core.atif_trajectory import ScoredTrajectory, TrajectoryGroup
from core import atif_extras
from example_recipe.word_puzzle.agent_solver.minisweagent.agent_solver_agent import (
    AgentSolverAgent,
)
from example_recipe.word_puzzle.reward import (
    WordPuzzleRewardConfig,
    WordPuzzleRewardFn,
)
from example_recipe.word_puzzle.dataset.task_data import WordPuzzleTaskData


REWARD_FN = WordPuzzleRewardFn(WordPuzzleRewardConfig())


class _FakeMiniAgent:
    """Stand-in for AsyncDefaultAgent: canned transcript + exit."""

    def __init__(
        self,
        *,
        candidate_ids: list[Any],
        submission: str,
        exit_status: str = "Submitted",
    ) -> None:
        self._candidate_ids = candidate_ids
        self._submission = submission
        self._exit_status = exit_status
        self.messages: list[dict[str, Any]] = []

    async def run(self, task: str = "", **kwargs: Any) -> dict[str, Any]:
        self.messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": task or "puzzle"},
        ]
        for cid in self._candidate_ids:
            self.messages.append(
                {
                    "role": "assistant",
                    "content": "ok",
                    "tool_calls": [
                        {
                            "id": "c",
                            "type": "function",
                            "function": {"name": "bash", "arguments": "{}"},
                        }
                    ],
                    "extra": {
                        "candidate_id": cid,
                        "finish_reason": "tool_calls",
                        "generation_token_count": 5,
                    },
                }
            )
            self.messages.append(
                {"role": "tool", "content": "out", "tool_call_id": "c", "extra": {}}
            )
        exit_extra = {"exit_status": self._exit_status, "submission": self._submission}
        self.messages.append(
            {"role": "exit", "content": self._submission, "extra": exit_extra}
        )
        return exit_extra


def _make_agent(mini_factory: Any) -> AgentSolverAgent:
    return AgentSolverAgent(
        sampler=object(),  # unused: the fake mini agent bypasses it
        reward_fn=REWARD_FN,
        mini_agent_factory=mini_factory,
    )


def _task(reference: str = "answer") -> WordPuzzleTaskData:
    return WordPuzzleTaskData(task_id="t1", question="q", reference=reference)


# ----------------------------------------------------------------------
# T7 -- rollout / group_rollout contract
# ----------------------------------------------------------------------


async def test_t7_rollout_returns_one_scored_trajectory():
    def factory(task_data: Any, rollout_idx: int) -> _FakeMiniAgent:
        return _FakeMiniAgent(candidate_ids=[0, 1], submission="answer")

    agent = _make_agent(factory)
    scored = await agent.rollout(_task(reference="answer"), rollout_idx=0)
    assert isinstance(scored, ScoredTrajectory)
    agent_steps = [s for s in scored.trajectory.steps if s.source == "agent"]
    assert len(agent_steps) == 2
    # correct submission -> +1
    assert scored.reward.reward == 1.0
    # every agent step trainable (candidate_id survives the trainer filter)
    assert all(atif_extras.get_candidate_id(s) for s in agent_steps)


async def test_t7_group_rollout_returns_n_trajectories_and_metrics():
    def factory(task_data: Any, rollout_idx: int) -> _FakeMiniAgent:
        # vary correctness by rollout so the group has reward variance
        submission = "answer" if rollout_idx % 2 == 0 else "wrong"
        return _FakeMiniAgent(candidate_ids=[rollout_idx], submission=submission)

    agent = _make_agent(factory)
    group, metrics = await agent.group_rollout(_task(reference="answer"), 4)
    assert isinstance(group, TrajectoryGroup)
    assert len(group.scored_trajectories) == 4
    assert isinstance(metrics, dict)


# ----------------------------------------------------------------------
# task_id normalization -- regression for the prod None-task_id bug
# ----------------------------------------------------------------------


async def test_rollout_rejects_empty_task_id() -> None:
    """task_id must be populated at dataset-loading time (via ensure_task_id
    inside parse_*_item). The agent rejects empty task_id with ValueError."""

    def factory(task_data: Any, rollout_idx: int) -> _FakeMiniAgent:
        return _FakeMiniAgent(candidate_ids=[0], submission="answer")

    import pytest as _pytest

    agent = _make_agent(factory)
    task = WordPuzzleTaskData(task_id="", question="q", reference="answer")
    with _pytest.raises(ValueError, match="task_id must be populated"):
        await agent.rollout(task, rollout_idx=3)


async def test_group_rollout_rejects_empty_task_id() -> None:
    """group_rollout must also reject empty task_id."""

    def factory(task_data: Any, rollout_idx: int) -> _FakeMiniAgent:
        return _FakeMiniAgent(candidate_ids=[rollout_idx], submission="answer")

    import pytest as _pytest

    agent = _make_agent(factory)
    task = WordPuzzleTaskData(task_id="", question="q", reference="answer")
    with _pytest.raises(ValueError, match="task_id must be populated"):
        await agent.group_rollout(task, 2)


# ----------------------------------------------------------------------
# T8 -- group-level training obligations
# ----------------------------------------------------------------------


async def test_t8_mixed_group_has_reward_variance():
    def factory(task_data: Any, rollout_idx: int) -> _FakeMiniAgent:
        submission = "answer" if rollout_idx % 2 == 0 else "wrong"
        return _FakeMiniAgent(candidate_ids=[rollout_idx], submission=submission)

    agent = _make_agent(factory)
    group, _ = await agent.group_rollout(_task(reference="answer"), 4)
    rewards = {g.reward.reward for g in group.scored_trajectories}
    assert len(rewards) > 1  # variance -> not dropped by the filter


async def test_t8_all_wrong_group_is_zero_variance_but_valid():
    def factory(task_data: Any, rollout_idx: int) -> _FakeMiniAgent:
        return _FakeMiniAgent(candidate_ids=[rollout_idx], submission="wrong")

    agent = _make_agent(factory)
    group, _ = await agent.group_rollout(_task(reference="answer"), 3)
    rewards = {g.reward.reward for g in group.scored_trajectories}
    assert rewards == {-1.0}  # all wrong; handled (kept 30% by the filter)


# ----------------------------------------------------------------------
# Guardrail -- warn when candidate_ids look like non-trainable indices
# ----------------------------------------------------------------------


async def test_index_candidate_ids_trigger_untrainable_warning(caplog: Any) -> None:
    # Integer candidate_ids -> "0","1" (bare indices) as the gemini_api
    # sampler emits: NOT trainable handles. group_rollout must warn.
    def factory(task_data: Any, rollout_idx: int) -> _FakeMiniAgent:
        return _FakeMiniAgent(candidate_ids=[rollout_idx], submission="answer")

    agent = _make_agent(factory)
    import logging

    with caplog.at_level(logging.WARNING):
        await agent.group_rollout(_task(reference="answer"), 2)
    assert any(
        "candidate_id" in r.message and "trainable" in r.message.lower()
        for r in caplog.records
    )


async def test_handle_candidate_ids_do_not_warn(caplog: Any) -> None:
    # 32-hex handles (as MSRL and other backends emit) must NOT warn.
    def factory(task_data: Any, rollout_idx: int) -> _FakeMiniAgent:
        return _FakeMiniAgent(
            candidate_ids=["a55e92791ee3462f85a6c21dabe59921"],
            submission="answer",
        )

    agent = _make_agent(factory)
    import logging

    with caplog.at_level(logging.WARNING):
        await agent.group_rollout(_task(reference="answer"), 2)
    assert not any(
        "candidate_id" in r.message and "trainable" in r.message.lower()
        for r in caplog.records
    )


# ----------------------------------------------------------------------
# on_trajectory_done -- per-trajectory permit-release callback
# ----------------------------------------------------------------------


class _RaisingMiniAgent(_FakeMiniAgent):
    """Mini agent whose run() raises, to exercise the per-trajectory
    on_trajectory_done firing on the exception path."""

    async def run(self, task: str = "", **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("boom")


async def test_group_rollout_fires_on_trajectory_done_once_per_trajectory():
    """The callback must fire exactly once per trajectory on the success
    path so the caller can release one permit as each sample finishes,
    instead of holding all N until the group's slowest straggler."""

    def factory(task_data: Any, rollout_idx: int) -> _FakeMiniAgent:
        return _FakeMiniAgent(candidate_ids=[rollout_idx], submission="answer")

    agent = _make_agent(factory)
    calls = 0

    def _on_done() -> None:
        nonlocal calls
        calls += 1

    await agent.group_rollout(_task(reference="answer"), 4, on_trajectory_done=_on_done)
    assert calls == 4


async def test_group_rollout_fires_on_trajectory_done_on_failure():
    """Even when a per-sample rollout raises, its coroutine's finally must
    still fire on_trajectory_done so the caller's permit accounting stays
    balanced (the caller re-raises after, but the callback already fired)."""

    def factory(task_data: Any, rollout_idx: int) -> _FakeMiniAgent:
        return _RaisingMiniAgent(candidate_ids=[rollout_idx], submission="x")

    agent = _make_agent(factory)
    calls = 0

    def _on_done() -> None:
        nonlocal calls
        calls += 1

    import pytest

    with pytest.raises(RuntimeError):
        await agent.group_rollout(
            _task(reference="answer"), 3, on_trajectory_done=_on_done
        )
    # Every per-sample coroutine ran its finally even though group_rollout
    # re-raised the first exception.
    assert calls == 3


async def test_group_rollout_on_trajectory_done_optional():
    """on_trajectory_done defaults to None -- omitting it must not error
    (external single-group callers don't pass it)."""

    def factory(task_data: Any, rollout_idx: int) -> _FakeMiniAgent:
        return _FakeMiniAgent(candidate_ids=[rollout_idx], submission="answer")

    agent = _make_agent(factory)
    group, _ = await agent.group_rollout(_task(reference="answer"), 2)
    assert len(group.scored_trajectories) == 2


# ----------------------------------------------------------------------
# Timeout path -- the TIMEOUT sentinel trajectory MUST be schema-valid
# ----------------------------------------------------------------------


class _HangingMiniAgent(_FakeMiniAgent):
    """Mini agent whose run() never returns, to force the rollout timeout."""

    async def run(self, task: str = "", **kwargs: Any) -> dict[str, Any]:
        await asyncio.sleep(3600)  # far longer than the test timeout
        return {}


def _make_agent_with_timeout(mini_factory: Any, timeout_s: float) -> AgentSolverAgent:
    return AgentSolverAgent(
        sampler=object(),
        reward_fn=REWARD_FN,
        mini_agent_factory=mini_factory,
        agent_rollout_timeout_seconds=timeout_s,
    )


async def test_rollout_timeout_yields_valid_trajectory_not_crash() -> None:
    """End-to-end: a hanging mini agent past ``agent_rollout_timeout_seconds``
    must yield a schema-valid TIMEOUT trajectory via ``rollout()``,
    NOT raise a ValidationError."""
    import math

    def factory(task_data: Any, rollout_idx: int) -> _HangingMiniAgent:
        return _HangingMiniAgent(candidate_ids=[0], submission="x")

    agent = _make_agent_with_timeout(factory, timeout_s=0.05)
    scored = await agent.rollout(_task(), rollout_idx=0)
    assert len(scored.trajectory.steps) >= 1
    assert math.isnan(scored.reward.reward)


async def test_group_rollout_survives_timeouts() -> None:
    """A group where every sample times out must still return N schema-valid
    (NaN-reward) trajectories instead of raising -- the dispatcher then drops
    the group via the variance filter rather than logging a ValidationError."""

    def factory(task_data: Any, rollout_idx: int) -> _HangingMiniAgent:
        return _HangingMiniAgent(candidate_ids=[0], submission="x")

    agent = _make_agent_with_timeout(factory, timeout_s=0.05)
    group, _ = await agent.group_rollout(_task(), 3)
    assert len(group.scored_trajectories) == 3


# ----------------------------------------------------------------------
# Timeout SALVAGE -- partial transcript must be recovered, not discarded
# ----------------------------------------------------------------------


class _PartialThenHangMiniAgent(_FakeMiniAgent):
    """Mini agent that completes a few real turns into ``.messages`` and THEN
    hangs forever -- simulating a rollout that made real progress before a
    single LLM call hung and tripped the hard rollout timeout. Salvage must
    recover the completed turns rather than throwing them away.
    """

    async def run(self, task: str = "", **kwargs: Any) -> dict[str, Any]:
        # Seed the transcript exactly as the real loop would: system + user,
        # then two completed assistant/tool turns.
        self.messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": task or "puzzle"},
        ]
        for cid in self._candidate_ids:
            self.messages.append(
                {
                    "role": "assistant",
                    "content": "ok",
                    "tool_calls": [
                        {
                            "id": "c",
                            "type": "function",
                            "function": {"name": "bash", "arguments": "{}"},
                        }
                    ],
                    "extra": {
                        "candidate_id": cid,
                        "finish_reason": "tool_calls",
                        "generation_token_count": 5,
                    },
                }
            )
            self.messages.append(
                {"role": "tool", "content": "out", "tool_call_id": "c", "extra": {}}
            )
        # Now hang mid-rollout (as if the next model.query() never returns).
        await asyncio.sleep(3600)
        return {}


async def test_rollout_timeout_salvages_partial_transcript() -> None:
    """Option A salvage: when a rollout times out AFTER completing real turns,
    ``rollout()`` must convert the partial ``mini_agent.messages`` into a
    trajectory that PRESERVES those turns as agent steps -- not a prompt-only
    tombstone. It must still carry NaN reward + TIMEOUT status so the group
    filter excludes it from training.
    """
    import math

    from core.rollout_status import RolloutStatus

    def factory(task_data: Any, rollout_idx: int) -> _PartialThenHangMiniAgent:
        # two completed agent turns before the hang
        return _PartialThenHangMiniAgent(candidate_ids=[0, 1], submission="")

    agent = _make_agent_with_timeout(factory, timeout_s=0.05)
    # Guard: rollout() must self-limit via agent_rollout_timeout_seconds. If it
    # doesn't (the pre-fix code applied the timeout only in the base wrapper,
    # not in rollout()), this wait_for fails fast instead of hanging forever.
    scored = await asyncio.wait_for(
        agent.rollout(_task(reference="answer"), rollout_idx=0), timeout=5.0
    )

    steps = scored.trajectory.steps
    agent_steps = [s for s in steps if s.source == "agent"]
    # The two completed turns must survive -- this is the whole point of
    # salvage (prompt-only would have 0 agent steps).
    assert len(agent_steps) == 2, (
        f"expected 2 salvaged agent steps, got {len(agent_steps)} "
        "(partial transcript was discarded instead of salvaged)"
    )
    # Excluded from training: NaN reward + TIMEOUT status.
    assert math.isnan(scored.reward.reward), (
        "salvaged timeout must be NaN (not trained)"
    )
    assert (
        atif_extras.get_rollout_status(scored.trajectory)
        == RolloutStatus.SAMPLING_FAILURE_TIMEOUT
    )


async def test_rollout_timeout_zero_turns_falls_back_to_tombstone() -> None:
    """When the rollout times out with NO completed dialogue turns (the first
    LLM call hung), there's nothing to salvage -- fall back to the schema-valid
    prompt-only tombstone (>=1 step, NaN, TIMEOUT). Must not crash."""
    import math

    def factory(task_data: Any, rollout_idx: int) -> _HangingMiniAgent:
        return _HangingMiniAgent(candidate_ids=[0], submission="x")

    agent = _make_agent_with_timeout(factory, timeout_s=0.05)
    scored = await asyncio.wait_for(agent.rollout(_task(), rollout_idx=0), timeout=5.0)
    assert len(scored.trajectory.steps) >= 1
    assert math.isnan(scored.reward.reward)


# ----------------------------------------------------------------------
# Session-id format -- MUST match AtifDiskWriter's _ROLLOUT_IDX_RE
# ----------------------------------------------------------------------
import re

# Mirror of AtifDiskWriter._ROLLOUT_IDX_RE (src/utils/atif_disk_writer.py:57).
# session_id format: "{recipe}-{job_id}-{task_id}-{rollout_idx}-{ISO}-{seq}"
_ROLLOUT_IDX_RE = re.compile(r"-(\d+)-\d{8}T\d{6}Z(?:-\d+)?$")


def _make_agent_with_job_id(
    mini_factory: Any, *, job_id: str = "12345", timeout_s: float | None = None
) -> AgentSolverAgent:
    return AgentSolverAgent(
        sampler=object(),
        reward_fn=REWARD_FN,
        mini_agent_factory=mini_factory,
        job_id=job_id,
        agent_rollout_timeout_seconds=timeout_s,
    )


async def test_session_id_matches_disk_writer_regex() -> None:
    """Regression: session_id MUST match ``_ROLLOUT_IDX_RE`` from
    ``atif_disk_writer.py`` so trajectories land in per-rollout dirs instead
    of the ``unknown`` fallback. The regex expects the format
    ``…-{rollout_idx}-{YYYYmmddTHHMMSSZ}`` at the tail. Before the fix,
    ``mini_atif_converter`` produced ``recipe-task_id-rollout_idx`` (no
    timestamp, no job_id) which never matched."""

    def factory(task_data: Any, rollout_idx: int) -> _FakeMiniAgent:
        return _FakeMiniAgent(candidate_ids=[0], submission="answer")

    agent = _make_agent_with_job_id(factory, job_id="99999")
    scored = await agent.rollout(_task(reference="answer"), rollout_idx=3)
    sid = scored.trajectory.session_id
    assert sid is not None, "session_id must not be None"
    assert _ROLLOUT_IDX_RE.search(sid), (
        f"session_id {sid!r} doesn't match _ROLLOUT_IDX_RE — "
        "AtifDiskWriter will dump to unknown/ fallback"
    )
    # job_id must appear in the session_id
    assert "99999" in sid, f"job_id not threaded into session_id: {sid!r}"


async def test_salvage_session_id_matches_regex() -> None:
    """The salvaged timeout trajectory must also have a conforming session_id."""

    def factory(task_data: Any, rollout_idx: int) -> _PartialThenHangMiniAgent:
        return _PartialThenHangMiniAgent(candidate_ids=[0, 1], submission="")

    agent = _make_agent_with_job_id(factory, job_id="77777", timeout_s=0.05)
    scored = await asyncio.wait_for(
        agent.rollout(_task(reference="answer"), rollout_idx=2), timeout=5.0
    )
    sid = scored.trajectory.session_id
    assert sid is not None, "salvage session_id must not be None"
    assert _ROLLOUT_IDX_RE.search(sid), (
        f"salvage session_id {sid!r} doesn't match _ROLLOUT_IDX_RE"
    )
