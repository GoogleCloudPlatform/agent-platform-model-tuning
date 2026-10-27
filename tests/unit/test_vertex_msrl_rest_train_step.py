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

"""``VertexMsrlRestTrainer.train_step`` — the wire body it POSTs.

The batch-assembly arithmetic moved out of the trainer into
the trainer's own inline batch assembly (CORE_DESIGN.md §1.2 step
4). These tests pin the ONE thing that must not have moved with it: the
exact ``reinforcementTuningTrainingDataBatch.examples`` list on the wire.
A drift here does not fail loudly — it trains on the wrong advantages, or
on a fraction of the batch, and the run still looks healthy.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from harbor.models.trajectories.agent import Agent

from backends.vertex_msrl_rest import VertexMsrlRestTrainer
from core.atif_trajectory import Reward, ScoredTrajectory, Step, Trajectory

AGENT = Agent(name="test-agent", version="0")


class _FakeClient:
    """Captures the POST body instead of talking to Vertex.

    Duck-typed rather than a ``VertexMsrlRestClient`` subclass on purpose:
    the trainer only ever touches these four members, and constructing a
    real client drags in auth.
    """

    tuning_job_name = "projects/p/locations/l/tuningJobs/j"
    _base_url = "https://example/v1"

    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []
        self.urls: list[str] = []

    async def _request(self, method, url, *, rpc_name, retry_deadline, json):
        self.urls.append(url)
        self.bodies.append(json)
        return {"name": "operations/1"}

    async def wait_for_lro(self, op_name, **_kwargs):
        return {"response": {"completedTrainStepId": 7}}


def agent_step(
    step_id: int,
    *,
    candidate_id: Any = "c",
    advantage=None,
) -> Step:
    extra: dict[str, Any] = {}
    if candidate_id is not None:
        extra["candidate_id"] = candidate_id
    if advantage is not None:
        extra["advantage"] = advantage
    return Step(step_id=step_id, source="agent", message="m", extra=extra)


def scored(
    *steps: Step,
    reward: float,
    normalized: float | None = None,
    traj_extra: dict[str, Any] | None = None,
):
    return ScoredTrajectory(
        trajectory=Trajectory(
            agent=AGENT, session_id="s", steps=list(steps), extra=traj_extra
        ),
        reward=Reward(reward=reward, normalized_reward=normalized, raw_reward=reward),
    )


@pytest.mark.asyncio
async def test_wire_body_is_one_example_per_agent_step():
    client = _FakeClient()
    trainer = VertexMsrlRestTrainer(client)  # type: ignore[arg-type]

    metrics = await trainer.train_step(
        [
            scored(
                Step(step_id=1, source="user", message="task"),
                agent_step(2, candidate_id="c1"),
                # Non-str cid -> stringified. Deliberately NOT a small
                # integer: an index-shaped id is now rejected by the freeze
                # guard (see the foreign-id test below).
                agent_step(3, candidate_id=42042),
                reward=0.5,
                normalized=-1.25,  # normalized_reward wins
            ),
            scored(
                agent_step(1, candidate_id="c3", advantage=0.25),  # step hook adds on
                agent_step(2, candidate_id=""),  # falsy cid -> untrainable
                reward=2.0,
            ),
        ]
    )

    assert client.bodies == [
        {
            "name": client.tuning_job_name,
            "reinforcementTuningTrainingDataBatch": {
                "examples": [
                    {"candidateId": "c1", "advantage": -1.25},
                    {"candidateId": "42042", "advantage": -1.25},
                    {"candidateId": "c3", "advantage": 2.25},
                ]
            },
            "skipWeightSync": False,
        }
    ]
    assert client.urls == [f"{client._base_url}/{client.tuning_job_name}:trainStep"]
    # The silent-data-loss counter still reaches the metric sink under the
    # same key it had when the extraction was inlined.
    assert metrics["trajectory/missing_candidate_id"] == 1
    assert metrics["_train_step_num_examples"] == 3
    assert metrics["completed_train_step_id"] == 7


@pytest.mark.asyncio
async def test_skip_weight_sync_and_lr_multiplier_still_ride_along():
    client = _FakeClient()
    trainer = VertexMsrlRestTrainer(client, learning_rate_multiplier=0.5)  # type: ignore[arg-type]

    await trainer.train_step(
        [scored(agent_step(1), reward=1.0)], sync_trainer_and_sampler=False
    )

    body = client.bodies[0]
    assert body["skipWeightSync"] is True
    assert body["reinforcementTuningTrainStepHyperParameters"] == {
        "learningRateMultiplier": 0.5
    }


@pytest.mark.asyncio
async def test_a_batch_with_no_trainable_step_sends_nothing():
    """No examples -> nothing POSTed, and no metric a caller could mistake
    for a completed step.

    That is what makes ``train_and_eval`` retry WITHOUT advancing
    global_step. The returned dict is not literally empty -- main added a
    ``_train_step_error`` diagnostic after this test was written, which is
    an improvement: a silent ``{}`` said nothing about WHY the step was
    skipped. What matters is the absence of real metrics, and that the
    wire stayed quiet.
    """
    client = _FakeClient()
    trainer = VertexMsrlRestTrainer(client)  # type: ignore[arg-type]

    metrics = await trainer.train_step([scored(agent_step(1, candidate_id=None), reward=1.0)])

    assert client.bodies == []
    assert metrics == {"_train_step_error": "no valid training examples to send"}
    assert not any(not k.startswith("_") for k in metrics), metrics


# --------------------------------------------------------------------------
# Freeze guard: nothing sampled from another (frozen) job may be trained
# --------------------------------------------------------------------------


