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

"""Backend capture tests for VertexMsrlRestSampler (per-candidate).

Verifies that sample() wires the raw-request/response ledger correctly
under the RAW_CAPTURE=1 env-var gate, and is a no-op when disabled.
Also asserts per-candidate fanout: with ``num_samples > 1``, each ctx
gets its own blob (post per-candidate-capture-ctxs reshape).

The vertex_msrl_rest backend uses a two-phase LRO: create (POST) then
poll (wait_for_lro). We stub both via MagicMock on the client so the
tests don't hit the network.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from backends.retry_profile import Backoff, LroRetryProfile
from backends.vertex_msrl_rest import (
    VertexMsrlRestClient,
    VertexMsrlRestSampler,
    VertexMsrlRestTrainer,
    _detect_prompt_too_long,
)
from core.atif_trajectory import (
    Agent as AtifAgent,
    Reward,
    ScoredTrajectory,
    Step,
    StepStats,
    Trajectory as AtifTrajectory,
    TrajectoryStep as AtifTrajectoryStep,
)
from utils.raw_capture import CaptureContext, Counter, _reload_capture_enabled


@pytest.fixture
def enable_capture(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("RAW_CAPTURE", "1")
    _reload_capture_enabled()
    yield
    monkeypatch.delenv("RAW_CAPTURE", raising=False)
    _reload_capture_enabled()


def _canned_lro_response() -> dict:
    """Minimally-valid terminal LRO response body with 1 tuning candidate."""
    return {
        "trainStepId": "42",
        "tuningCandidates": [
            {
                "candidateId": "cand-abc-0",
                "candidate": {
                    "content": {"parts": [{"text": "hello"}]},
                    "finishReason": "STOP",
                },
                "generationTokenCount": 3,
                "thoughtsTokenCount": 0,
                "promptTokenCount": 5,
            }
        ],
    }


def _canned_lro_response_2() -> dict:
    """Terminal LRO response body with 2 tuning candidates."""
    return {
        "trainStepId": "42",
        "tuningCandidates": [
            {
                "candidateId": "cand-abc-0",
                "candidate": {
                    "content": {"parts": [{"text": "hello"}]},
                    "finishReason": "STOP",
                },
                "generationTokenCount": 3,
                "thoughtsTokenCount": 0,
                "promptTokenCount": 5,
            },
            {
                "candidateId": "cand-abc-1",
                "candidate": {
                    "content": {"parts": [{"text": "world"}]},
                    "finishReason": "STOP",
                },
                "generationTokenCount": 3,
                "thoughtsTokenCount": 0,
                "promptTokenCount": 5,
            },
        ],
    }


def _overflow_lro_response() -> dict:
    """Terminal LRO ``response`` body for a prompt-length rejection."""
    return {
        "trainStepId": "0",
        "tuningCandidates": [],
        "operation": {
            "operationId": "00000000-0000-4000-8000-000000000001",
            "finished": True,
            "isFailed": True,
            "error": (
                "[REX]: request_id: 00000000-0000-4000-8000-000000000001, "
                "Prompt length 131109 exceeds maximum allowed length 122880."
            ),
        },
        "generateContentInTuningScopeResponse": None,
    }


def _build_stubbed_sampler() -> VertexMsrlRestSampler:
    """Build a VertexMsrlRestSampler with client stubbed for offline tests.

    Returns a sampler whose client._request AND client.wait_for_lro
    can be stubbed per-test via AsyncMock assignment.
    """
    client = MagicMock(spec=VertexMsrlRestClient)
    client._base_url = "https://fake"
    client.tuning_job_name = "projects/x/locations/y/tuningJobs/z"
    client.base_model = "gemini-3.5-flash"
    # AsyncMocks for the async methods sample() awaits.
    client._request = AsyncMock()
    client.wait_for_lro = AsyncMock()
    sampler = VertexMsrlRestSampler.__new__(VertexMsrlRestSampler)
    sampler.client = client
    sampler.request_timeout = 60.0
    sampler.retry = LroRetryProfile(
        poll_deadline_s=600.0, poll_backoff=Backoff(1.0, 30.0)
    )
    return sampler


class TestVertexMsrlCaptureIntegration:
    async def test_sample_stamps_raw_call_uri_when_capture_enabled(
        self, enable_capture: None, tmp_path: Path
    ) -> None:
        trial_dir = tmp_path / "trial"
        trial_dir.mkdir()
        ctx = CaptureContext(
            trial_dir=trial_dir,
            trajectory_id="test-msrl-session",
            counter=Counter(),
        )
        sampler = _build_stubbed_sampler()
        sampler.client._request.return_value = {"name": "operations/op-123"}
        sampler.client.wait_for_lro.return_value = {
            "response": _canned_lro_response(),
        }

        results = await sampler.sample_n(prompt="hi", num_samples=1, capture_ctxs=[ctx])

        assert len(results) == 1
        assert (
            results[0].context.raw_call_uri
            == "artifacts/raw_llm_calls/call_000000.json"
        )
        blob = json.loads((trial_dir / results[0].context.raw_call_uri).read_text())
        assert blob["trajectory_id"] == "test-msrl-session"
        assert blob["status"] == "ok"
        # Per-candidate slice: envelope preserved + the one tuningCandidate.
        assert blob["response"]["trainStepId"] == "42"
        assert len(blob["response"]["tuningCandidates"]) == 1
        assert blob["response"]["tuningCandidates"][0]["candidateId"] == "cand-abc-0"

    async def test_sample_stamps_none_when_capture_disabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Explicit opt-out — RAW_CAPTURE defaults ENABLED since the PR
        # A follow-up flip, so unsetting no longer disables. Force
        # capture off with RAW_CAPTURE=0.
        monkeypatch.setenv("RAW_CAPTURE", "0")
        _reload_capture_enabled()
        trial_dir = tmp_path / "trial"
        trial_dir.mkdir()
        ctx = CaptureContext(
            trial_dir=trial_dir,
            trajectory_id="s",
            counter=Counter(),
        )
        sampler = _build_stubbed_sampler()
        sampler.client._request.return_value = {"name": "operations/op"}
        sampler.client.wait_for_lro.return_value = {
            "response": _canned_lro_response(),
        }

        results = await sampler.sample_n(prompt="hi", num_samples=1, capture_ctxs=[ctx])

        assert results[0].context.raw_call_uri is None
        assert not (trial_dir / "artifacts").exists()

    async def test_sample_no_ctxs_stamps_none(self) -> None:
        _reload_capture_enabled()
        sampler = _build_stubbed_sampler()
        sampler.client._request.return_value = {"name": "operations/op"}
        sampler.client.wait_for_lro.return_value = {
            "response": _canned_lro_response(),
        }
        results = await sampler.sample_n(prompt="hi", num_samples=1)
        assert results[0].context.raw_call_uri is None

    async def test_sample_create_failure_captures_error_status(
        self, enable_capture: None, tmp_path: Path
    ) -> None:
        trial_dir = tmp_path / "trial"
        trial_dir.mkdir()
        ctx = CaptureContext(
            trial_dir=trial_dir,
            trajectory_id="s",
            counter=Counter(),
        )
        sampler = _build_stubbed_sampler()
        # Simulate the create POST failing.
        sampler.client._request.side_effect = RuntimeError("create failed")

        results = await sampler.sample_n(prompt="hi", num_samples=1, capture_ctxs=[ctx])
        assert results[0].tuning_candidate.candidate_id == ""
        assert (
            results[0].context.raw_call_uri
            == "artifacts/raw_llm_calls/call_000000.json"
        )
        blob = json.loads((trial_dir / results[0].context.raw_call_uri).read_text())
        assert blob["status"] == "error"

    async def test_sample_lro_poll_failure_captures_error_status(
        self, enable_capture: None, tmp_path: Path
    ) -> None:
        trial_dir = tmp_path / "trial"
        trial_dir.mkdir()
        ctx = CaptureContext(
            trial_dir=trial_dir,
            trajectory_id="s",
            counter=Counter(),
        )
        sampler = _build_stubbed_sampler()
        sampler.client._request.return_value = {"name": "operations/op-999"}
        sampler.client.wait_for_lro.side_effect = RuntimeError("poll failed")

        results = await sampler.sample_n(prompt="hi", num_samples=1, capture_ctxs=[ctx])
        assert results[0].tuning_candidate.candidate_id == ""
        assert (
            results[0].context.raw_call_uri
            == "artifacts/raw_llm_calls/call_000000.json"
        )
        blob = json.loads((trial_dir / results[0].context.raw_call_uri).read_text())
        assert blob["status"] == "error"

    async def test_lro_poll_fail_ledger_blob_includes_op_name_and_operation_id(
        self, enable_capture: None, tmp_path: Path
    ) -> None:
        """The LRO poll-failure ledger payload MUST include ``op_name`` and
        ``operation_id`` for cross-referencing to server-side logs. Prior to
        the fix, only ``__error__`` was written and the ledger blob was
        orphaned from the server-side operation record."""
        trial_dir = tmp_path / "trial"
        trial_dir.mkdir()
        ctx = CaptureContext(
            trial_dir=trial_dir,
            trajectory_id="s",
            counter=Counter(),
        )
        sampler = _build_stubbed_sampler()
        sampler.client._request.return_value = {"name": "operations/op-abc-999"}
        sampler.client.wait_for_lro.side_effect = RuntimeError("poll failed")

        results = await sampler.sample_n(prompt="hi", num_samples=1, capture_ctxs=[ctx])
        assert (
            results[0].context.raw_call_uri
            == "artifacts/raw_llm_calls/call_000000.json"
        )
        blob = json.loads((trial_dir / results[0].context.raw_call_uri).read_text())
        response = blob["response"]
        assert response["__error__"].startswith("lro_poll_failed:")
        assert response["op_name"] == "operations/op-abc-999"
        assert response["operation_id"] == "op-abc-999"

    async def test_sample_no_candidates_captures_empty_status(
        self, enable_capture: None, tmp_path: Path
    ) -> None:
        trial_dir = tmp_path / "trial"
        trial_dir.mkdir()
        ctx = CaptureContext(
            trial_dir=trial_dir,
            trajectory_id="s",
            counter=Counter(),
        )
        sampler = _build_stubbed_sampler()
        sampler.client._request.return_value = {"name": "operations/op-e"}
        # Terminal LRO response with no candidates.
        sampler.client.wait_for_lro.return_value = {
            "response": {"trainStepId": "0", "tuningCandidates": []},
        }

        results = await sampler.sample_n(prompt="hi", num_samples=1, capture_ctxs=[ctx])
        assert results[0].tuning_candidate.candidate_id == ""
        assert (
            results[0].context.raw_call_uri
            == "artifacts/raw_llm_calls/call_000000.json"
        )
        blob = json.loads((trial_dir / results[0].context.raw_call_uri).read_text())
        assert blob["status"] == "empty"

    async def test_sample_overflow_sets_is_prompt_too_long_on_context(self) -> None:
        """End-to-end: the overflow envelope reaches SamplingContext.

        Guards the stringly-typed ``extras`` key shared by the writer in
        ``sample_n`` and the reader in ``_empty_result``.
        """
        sampler = _build_stubbed_sampler()
        sampler.client._request.return_value = {"name": "operations/op-overflow"}
        sampler.client.wait_for_lro.return_value = {
            "response": _overflow_lro_response(),
        }

        results = await sampler.sample_n(prompt="hi", num_samples=1)
        assert results[0].tuning_candidate.candidate_id == ""
        assert results[0].context.is_prompt_too_long is True

    async def test_a_poll_FAILURE_overflow_also_sets_is_prompt_too_long(
        self,
    ) -> None:
        """The second arrival shape, and the one that cost a gate.

        `wait_for_lro` RAISES rather than returning a terminal body, and the
        exception text carries the tuning scope's own refusal. This path used
        to pass no `extras` at all -- on a comment asserting overflow could
        only arrive as a successful terminal LRO -- so the failsafe was
        indistinguishable from a dead transport and the rollout was scored
        `SAMPLING_CLIENT_ERROR` (infra) rather than `PROMPT_TOO_LONG` (input).
        Message verbatim from run 20260927T015304Z-curator-gate.
        """
        sampler = _build_stubbed_sampler()
        sampler.client._request.return_value = {"name": "operations/op-of2"}
        sampler.client.wait_for_lro.side_effect = RuntimeError(
            "Operation projects/p/locations/l/tuningJobs/j/operations/505480848"
            " failed: code=3, message=Input tokens exceed maximum prompt length"
            " (max is 122880)."
        )

        results = await sampler.sample_n(prompt="hi", num_samples=1)
        assert results[0].tuning_candidate.candidate_id == ""
        assert results[0].context.is_prompt_too_long is True

    async def test_a_poll_failure_that_is_NOT_an_overflow_leaves_flag_false(
        self,
    ) -> None:
        """The flag must stay a PROMPT signal, not "the poll raised"."""
        sampler = _build_stubbed_sampler()
        sampler.client._request.return_value = {"name": "operations/op-of3"}
        sampler.client.wait_for_lro.side_effect = RuntimeError(
            "Operation .../operations/1 failed: code=14, message=UNAVAILABLE"
        )

        results = await sampler.sample_n(prompt="hi", num_samples=1)
        assert results[0].tuning_candidate.candidate_id == ""
        assert results[0].context.is_prompt_too_long is False

    async def test_sample_plain_no_candidates_leaves_flag_false(self) -> None:
        sampler = _build_stubbed_sampler()
        sampler.client._request.return_value = {"name": "operations/op-e"}
        sampler.client.wait_for_lro.return_value = {
            "response": {"trainStepId": "0", "tuningCandidates": []},
        }

        results = await sampler.sample_n(prompt="hi", num_samples=1)
        assert results[0].tuning_candidate.candidate_id == ""
        assert results[0].context.is_prompt_too_long is False

    async def test_sample_num_samples_2_writes_two_ledger_blobs(
        self, enable_capture: None, tmp_path: Path
    ) -> None:
        """Batched fanout: 2 candidates, 2 ctxs -> 2 distinct blobs."""
        trial_dir0 = tmp_path / "trial0"
        trial_dir1 = tmp_path / "trial1"
        trial_dir0.mkdir()
        trial_dir1.mkdir()
        ctx0 = CaptureContext(
            trial_dir=trial_dir0, trajectory_id="sid-0", counter=Counter()
        )
        ctx1 = CaptureContext(
            trial_dir=trial_dir1, trajectory_id="sid-1", counter=Counter()
        )
        sampler = _build_stubbed_sampler()
        sampler.client._request.return_value = {"name": "operations/op-multi"}
        sampler.client.wait_for_lro.return_value = {
            "response": _canned_lro_response_2(),
        }

        results = await sampler.sample_n(
            prompt="hi", num_samples=2, capture_ctxs=[ctx0, ctx1]
        )
        assert len(results) == 2
        blob0 = json.loads(
            (trial_dir0 / "artifacts/raw_llm_calls/call_000000.json").read_text()
        )
        blob1 = json.loads(
            (trial_dir1 / "artifacts/raw_llm_calls/call_000000.json").read_text()
        )
        assert blob0["trajectory_id"] == "sid-0"
        assert blob1["trajectory_id"] == "sid-1"
        assert len(blob0["response"]["tuningCandidates"]) == 1
        assert len(blob1["response"]["tuningCandidates"]) == 1
        assert blob0["response"]["tuningCandidates"][0]["candidateId"] == "cand-abc-0"
        assert blob1["response"]["tuningCandidates"][0]["candidateId"] == "cand-abc-1"

    async def test_sample_capture_ctxs_length_mismatch_raises(
        self, enable_capture: None, tmp_path: Path
    ) -> None:
        trial_dir = tmp_path / "trial"
        trial_dir.mkdir()
        ctx = CaptureContext(trial_dir=trial_dir, trajectory_id="s", counter=Counter())
        sampler = _build_stubbed_sampler()
        sampler.client._request.return_value = {"name": "operations/op"}
        sampler.client.wait_for_lro.return_value = {
            "response": _canned_lro_response(),
        }
        with pytest.raises(ValueError, match="does not match num_samples"):
            await sampler.sample_n(
                prompt="hi",
                num_samples=2,
                capture_ctxs=[ctx],  # length 1 != 2
            )

    async def test_sample_capture_ctxs_none_element_skips_that_candidate(
        self, enable_capture: None, tmp_path: Path
    ) -> None:
        trial_dir0 = tmp_path / "trial0"
        trial_dir0.mkdir()
        ctx0 = CaptureContext(
            trial_dir=trial_dir0, trajectory_id="sid-0", counter=Counter()
        )
        sampler = _build_stubbed_sampler()
        sampler.client._request.return_value = {"name": "operations/op-none"}
        sampler.client.wait_for_lro.return_value = {
            "response": _canned_lro_response_2(),
        }

        results = await sampler.sample_n(
            prompt="hi", num_samples=2, capture_ctxs=[ctx0, None]
        )
        assert len(results) == 2
        assert (
            results[0].context.raw_call_uri
            == "artifacts/raw_llm_calls/call_000000.json"
        )
        assert results[1].context.raw_call_uri is None
        assert (trial_dir0 / "artifacts/raw_llm_calls/call_000000.json").exists()

    async def test_sample_create_failure_fans_out_to_all_ctxs(
        self, enable_capture: None, tmp_path: Path
    ) -> None:
        """create-POST failure with num_samples=2 -> both ctxs get an
        error-status blob."""
        trial_dir0 = tmp_path / "trial0"
        trial_dir1 = tmp_path / "trial1"
        trial_dir0.mkdir()
        trial_dir1.mkdir()
        ctx0 = CaptureContext(
            trial_dir=trial_dir0, trajectory_id="sid-0", counter=Counter()
        )
        ctx1 = CaptureContext(
            trial_dir=trial_dir1, trajectory_id="sid-1", counter=Counter()
        )
        sampler = _build_stubbed_sampler()
        sampler.client._request.side_effect = RuntimeError("create failed")

        results = await sampler.sample_n(
            prompt="hi", num_samples=2, capture_ctxs=[ctx0, ctx1]
        )
        assert len(results) == 2
        for i, td in enumerate((trial_dir0, trial_dir1)):
            blob = json.loads(
                (td / "artifacts/raw_llm_calls/call_000000.json").read_text()
            )
            assert blob["status"] == "error"
            assert blob["trajectory_id"] == f"sid-{i}"

    async def test_sample_parallel_dumps_all_candidates_get_blobs(
        self, enable_capture: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Per-candidate ``dump_raw`` calls fire concurrently, not serially.

        With N=4 and a 0.1s sleep per dump, serial baseline is ~0.4s;
        concurrent gather should complete in ~0.1s. Assert wallclock
        < 0.3s to catch a serial regression (which would be >= 0.4s)
        with headroom for CI jitter.
        """
        import asyncio
        import time

        from utils import raw_capture

        trial_dirs = [tmp_path / f"trial{i}" for i in range(4)]
        for td in trial_dirs:
            td.mkdir()
        ctxs = [
            CaptureContext(trial_dir=td, trajectory_id=f"sid-{i}", counter=Counter())
            for i, td in enumerate(trial_dirs)
        ]
        # Build canned response with 4 tuning candidates.
        canned = {
            "trainStepId": "42",
            "tuningCandidates": [
                {
                    "candidateId": f"cand-{i}",
                    "candidate": {
                        "content": {"parts": [{"text": f"c{i}"}]},
                        "finishReason": "STOP",
                    },
                    "generationTokenCount": 3,
                    "thoughtsTokenCount": 0,
                    "promptTokenCount": 5,
                }
                for i in range(4)
            ],
        }
        sampler = _build_stubbed_sampler()
        sampler.client._request.return_value = {"name": "operations/op-x"}
        sampler.client.wait_for_lro.return_value = {"response": canned}

        delay = 0.1
        real_dump_raw = raw_capture.dump_raw

        async def slow_dump_raw(**kwargs: object) -> str | None:
            await asyncio.sleep(delay)
            return await real_dump_raw(**kwargs)  # type: ignore[arg-type]

        # Patched where ``dump_raw_fanout`` looks it up: the sampler now
        # calls the fan-out helper rather than ``dump_raw`` directly.
        monkeypatch.setattr("utils.raw_capture.dump_raw", slow_dump_raw)

        t0 = time.monotonic()
        results = await sampler.sample_n(prompt="hi", num_samples=4, capture_ctxs=ctxs)
        wall_s = time.monotonic() - t0

        assert len(results) == 4
        assert wall_s < 4 * delay * 0.75, (
            f"expected concurrent dumps (< {4 * delay * 0.75:.2f}s), got {wall_s:.3f}s"
        )
        for td in trial_dirs:
            assert (td / "artifacts/raw_llm_calls/call_000000.json").exists()

    async def test_sample_parallel_dumps_preserve_order(
        self, enable_capture: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Order preservation under gather: results[i] carries the URI
        whose blob contains candidate[i], not a shuffled URI."""
        import asyncio

        from utils import raw_capture

        trial_dirs = [tmp_path / f"trial{i}" for i in range(4)]
        for td in trial_dirs:
            td.mkdir()
        ctxs = [
            CaptureContext(trial_dir=td, trajectory_id=f"sid-{i}", counter=Counter())
            for i, td in enumerate(trial_dirs)
        ]
        canned = {
            "trainStepId": "42",
            "tuningCandidates": [
                {
                    "candidateId": f"cand-{i}",
                    "candidate": {
                        "content": {"parts": [{"text": f"UNIQUE_{i}"}]},
                        "finishReason": "STOP",
                    },
                    "generationTokenCount": 3,
                    "thoughtsTokenCount": 0,
                    "promptTokenCount": 5,
                }
                for i in range(4)
            ],
        }
        sampler = _build_stubbed_sampler()
        sampler.client._request.return_value = {"name": "operations/op-y"}
        sampler.client.wait_for_lro.return_value = {"response": canned}

        real_dump_raw = raw_capture.dump_raw

        async def reversed_delay_dump_raw(
            *, ctx: CaptureContext | None, **kwargs: object
        ) -> str | None:
            if ctx is not None:
                sid = int(ctx.trajectory_id.rsplit("-", 1)[-1])
                await asyncio.sleep(0.02 * (4 - sid))
            return await real_dump_raw(ctx=ctx, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr("utils.raw_capture.dump_raw", reversed_delay_dump_raw)

        results = await sampler.sample_n(prompt="hi", num_samples=4, capture_ctxs=ctxs)
        assert len(results) == 4
        for i, result in enumerate(results):
            assert result.context.raw_call_uri is not None
            blob = json.loads((trial_dirs[i] / result.context.raw_call_uri).read_text())
            assert blob["trajectory_id"] == f"sid-{i}"
            assert (
                blob["response"]["tuningCandidates"][0]["candidate"]["content"][
                    "parts"
                ][0]["text"]
                == f"UNIQUE_{i}"
            )


def _count_client() -> VertexMsrlRestClient:
    """Real client with credentials stubbed so construction stays offline.

    ``__init__`` eagerly resolves call credentials; patch that seam so the
    constructor never touches ADC or the metadata server. ``_request`` is
    then replaced with an ``AsyncMock`` per the file's existing pattern.
    """
    with patch(
        "backends.vertex_msrl_rest.get_call_credentials",
        return_value=MagicMock(),
    ):
        c = VertexMsrlRestClient(
            project="p",
            location="us-central1",
            endpoint="us-central1-aiplatform.googleapis.com",
            http_client=httpx.AsyncClient(),
        )
    c._request = AsyncMock()
    return c


class TestCountTuningJobsByState:
    async def test_count_tuning_jobs_single_page(self) -> None:
        c = _count_client()
        c._request.return_value = {
            "tuningJobs": [
                {"state": "JOB_STATE_RUNNING"},
                {"state": "JOB_STATE_RUNNING"},
                {"state": "JOB_STATE_PENDING"},
            ]
        }
        counts = await c.count_tuning_jobs_by_state()
        assert counts == {"JOB_STATE_RUNNING": 2, "JOB_STATE_PENDING": 1}
        assert c._request.await_count == 1

    async def test_count_tuning_jobs_paginates(self) -> None:
        c = _count_client()
        c._request.side_effect = [
            {
                "tuningJobs": [{"state": "JOB_STATE_RUNNING"}],
                "nextPageToken": "tok",
            },
            {"tuningJobs": [{"state": "JOB_STATE_SUCCEEDED"}]},
        ]
        counts = await c.count_tuning_jobs_by_state()
        assert counts == {"JOB_STATE_RUNNING": 1, "JOB_STATE_SUCCEEDED": 1}
        assert c._request.await_count == 2

    async def test_count_tuning_jobs_empty(self) -> None:
        c = _count_client()
        c._request.return_value = {}
        counts = await c.count_tuning_jobs_by_state()
        assert counts == {}

    async def test_count_tuning_jobs_missing_state_key_bucketed_unknown(
        self,
    ) -> None:
        c = _count_client()
        c._request.return_value = {"tuningJobs": [{}, {"state": ""}]}
        counts = await c.count_tuning_jobs_by_state()
        assert counts == {"JOB_STATE_UNSPECIFIED": 2}

    async def test_count_tuning_jobs_empty_list_present(self) -> None:
        c = _count_client()
        c._request.return_value = {"tuningJobs": [], "nextPageToken": ""}
        counts = await c.count_tuning_jobs_by_state()
        assert counts == {}

    async def test_count_tuning_jobs_max_pages_cap_terminates(self) -> None:
        c = _count_client()
        # Server that never stops handing back a non-empty token; the cap
        # must break the loop after exactly ``max_pages`` requests.
        c._request.return_value = {
            "tuningJobs": [{"state": "JOB_STATE_RUNNING"}],
            "nextPageToken": "loop",
        }
        counts = await c.count_tuning_jobs_by_state(max_pages=3)
        assert counts == {"JOB_STATE_RUNNING": 3}
        assert c._request.await_count == 3


# --- _finish_reason_ratios: trainer-emitted train/sampler/finish_reason/* ---
# These keys are emitted as per-reason SHARES (count / total agent steps) so
# the family sums to 1.0, matching the bare sampler/finish_reason/* twin.

from core import atif_extras as _atif_extras  # noqa: E402
from core.atif_trajectory import (  # noqa: E402
    ScoredTrajectory as _GradedTrajectory,
    Reward as _Reward,
    StepStats as _StepStats,
    Step as _Step,
    TrajectoryStep as _AtifTrajectoryStep,
)
from core.rollout_status import RolloutStatus as _RolloutStatus  # noqa: E402
from core.atif_trajectory import (  # noqa: E402
    Agent as _AtifAgent,
    Trajectory as _AtifTrajectory,
)
from backends.vertex_msrl_rest import _finish_reason_ratios  # noqa: E402


def _traj_with_finish_reasons(reasons: list[str | None]) -> _GradedTrajectory:
    """ScoredTrajectory with one agent step per entry in ``reasons`` (None =>
    no finish_reason recorded, which buckets as OTHER)."""
    us = _Step(step_id=1, source="user", message="p")
    agent_steps: list = []
    for i, fr in enumerate(reasons):
        stats = _StepStats(finish_reason=fr) if fr is not None else _StepStats()
        agent_steps.append(
            _AtifTrajectoryStep.from_wire(step_id=2 + i, generation="g", stats=stats)
        )
    traj = _AtifTrajectory(
        schema_version="ATIF-v1.7",
        session_id="test-session",
        agent=_AtifAgent(name="t", version="0", model_name="m"),
        steps=[us, *agent_steps],
    )
    _atif_extras.set_rollout_status(traj, _RolloutStatus.OK)
    return _GradedTrajectory(trajectory=traj, reward=_Reward(reward=0.0))


class TestFinishReasonRatios:
    def test_emits_ratios_summing_to_one(self) -> None:
        trajs = [_traj_with_finish_reasons(["STOP", "STOP", "STOP", "MAX_TOKENS"])]
        m = _finish_reason_ratios(trajs)
        assert m["sampler/finish_reason/STOP"] == 0.75
        assert m["sampler/finish_reason/MAX_TOKENS"] == 0.25
        assert abs(sum(m.values()) - 1.0) < 1e-9

    def test_denominator_is_total_across_trajectories(self) -> None:
        # 5 agent steps total across two trajectories: STOP x3, MAX_TOKENS x1,
        # OTHER x1 (the None). Denominator = 5.
        trajs = [
            _traj_with_finish_reasons(["STOP", "STOP", "MAX_TOKENS"]),
            _traj_with_finish_reasons(["STOP", None]),
        ]
        m = _finish_reason_ratios(trajs)
        assert m["sampler/finish_reason/STOP"] == 3 / 5
        assert m["sampler/finish_reason/MAX_TOKENS"] == 1 / 5
        assert m["sampler/finish_reason/OTHER"] == 1 / 5
        assert abs(sum(m.values()) - 1.0) < 1e-9

    def test_empty_when_no_agent_steps(self) -> None:
        assert _finish_reason_ratios([]) == {}
        assert _finish_reason_ratios([_traj_with_finish_reasons([])]) == {}


class TestDetectPromptTooLong:
    def test_detect_prompt_too_long_true_on_length_error(self) -> None:
        """isFailed=true plus the length message in operation.error."""
        assert _detect_prompt_too_long(_overflow_lro_response()) is True

    def test_detect_prompt_too_long_false_on_other_error(self) -> None:
        response = {"operation": {"isFailed": True, "error": "[REX]: internal error"}}
        assert _detect_prompt_too_long(response) is False

    def test_detect_prompt_too_long_false_on_success(self) -> None:
        response = {"operation": {"isFailed": False, "error": None}}
        assert _detect_prompt_too_long(response) is False

    def test_detect_prompt_too_long_false_on_missing_operation(self) -> None:
        assert _detect_prompt_too_long({}) is False


def _real_shape_lro_response(num_candidates: int = 1) -> dict:
    """Terminal LRO ``response`` body in the shape the live service sends.

    The prompt is shared by every candidate, but its token count is
    repeated on each ``tuningCandidates`` entry so the entry is
    self-contained. The response-level ``usageMetadata`` that used to
    carry it is gone (``reserved 4`` in the tuning-scope proto).
    """
    return {
        "trainStepId": "72",
        "tuningCandidates": [
            {
                "candidateId": f"cand-{i}",
                "candidate": {
                    "content": {"parts": [{"text": f"hello-{i}"}]},
                    "finishReason": "STOP",
                },
                "generationTokenCount": 14065,
                "thoughtsTokenCount": 12000,
                "promptTokenCount": 116962,
            }
            for i in range(num_candidates)
        ],
    }


class TestPromptTokenCountFromTuningCandidate:
    """``SamplingContext.prompt_token_count`` must come off the wire.

    In-loop context compaction projects the next request from the
    measured prompt-token count, so a ``None`` here disables the feature
    entirely on this backend.
    """

    async def test_per_candidate_count_populates_prompt_tokens(self) -> None:
        sampler = _build_stubbed_sampler()
        sampler.client._request.return_value = {"name": "operations/op-1"}
        sampler.client.wait_for_lro.return_value = {
            "response": _real_shape_lro_response(),
        }

        results = await sampler.sample_n(prompt="hi", num_samples=1)

        assert len(results) == 1
        assert results[0].context.prompt_token_count == 116962

    async def test_every_candidate_gets_the_shared_prompt_token_count(self) -> None:
        sampler = _build_stubbed_sampler()
        sampler.client._request.return_value = {"name": "operations/op-2"}
        sampler.client.wait_for_lro.return_value = {
            "response": _real_shape_lro_response(num_candidates=2),
        }

        results = await sampler.sample_n(prompt="hi", num_samples=2)

        assert len(results) == 2
        assert [r.context.prompt_token_count for r in results] == [116962, 116962]

    async def test_missing_prompt_token_count_returns_failsafe(self) -> None:
        """A missing count is malformed, not a silent ``None``.

        A ``None`` here disables compaction outright and blanks the
        context-size metrics, so it must be surfaced. We return the
        failsafe result (triaged to SAMPLING_CLIENT_ERROR) rather than
        raising, which would lose the whole task.
        """
        sampler = _build_stubbed_sampler()
        sampler.client._request.return_value = {"name": "operations/op-3"}
        response = _real_shape_lro_response()
        del response["tuningCandidates"][0]["promptTokenCount"]
        sampler.client.wait_for_lro.return_value = {"response": response}

        results = await sampler.sample_n(prompt="hi", num_samples=1)

        assert results[0].tuning_candidate.candidate_id == ""
        assert results[0].context.is_malformed_usage is True
        assert results[0].context.prompt_token_count is None

    async def test_removed_response_level_usage_metadata_is_ignored(self) -> None:
        """A stale server echoing ``usageMetadata`` no longer feeds the count.

        The field is ``reserved`` in the tuning-scope proto; only the
        per-candidate value is read.
        """
        sampler = _build_stubbed_sampler()
        sampler.client._request.return_value = {"name": "operations/op-4"}
        response = _real_shape_lro_response()
        del response["tuningCandidates"][0]["promptTokenCount"]
        response["usageMetadata"] = {"promptTokenCount": 116962}
        sampler.client.wait_for_lro.return_value = {"response": response}

        results = await sampler.sample_n(prompt="hi", num_samples=1)

        assert results[0].context.is_malformed_usage is True
        assert results[0].context.prompt_token_count is None


def _build_stubbed_trainer() -> VertexMsrlRestTrainer:
    """Build a VertexMsrlRestTrainer with its client stubbed for offline tests."""
    client = MagicMock(spec=VertexMsrlRestClient)
    client._base_url = "https://fake"
    client.tuning_job_name = "projects/x/locations/y/tuningJobs/z"
    client._request = AsyncMock()
    client.wait_for_lro = AsyncMock()
    trainer = VertexMsrlRestTrainer.__new__(VertexMsrlRestTrainer)
    trainer.client = client
    trainer.retry = LroRetryProfile(
        poll_deadline_s=600.0, poll_backoff=Backoff(1.0, 30.0)
    )
    trainer.learning_rate_multiplier = None
    return trainer


def _scored(candidate_id: str | None = "c1") -> ScoredTrajectory:
    stats = StepStats(candidate_id=candidate_id) if candidate_id else StepStats()
    traj = AtifTrajectory(
        schema_version="ATIF-v1.7",
        session_id="s",
        agent=AtifAgent(name="t", version="0", model_name="m"),
        steps=[
            Step(step_id=1, source="user", message="p"),
            AtifTrajectoryStep.from_wire(step_id=2, generation="g", stats=stats),
        ],
    )
    return ScoredTrajectory(trajectory=traj, reward=Reward(reward=1.0))


class TestTrainStepFailureMetrics:
    """Every train_step failure path must name itself in
    ``_train_step_error``.

    A bare ``{}`` is indistinguishable from "no metrics", so the caller
    cannot tell a failure from a quiet success and the step is recorded
    as still training.
    """

    @pytest.mark.asyncio
    async def test_reports_error_when_lro_fails(self):
        trainer = _build_stubbed_trainer()
        trainer.client._request.return_value = {"name": "operations/456"}
        trainer.client.wait_for_lro.side_effect = RuntimeError(
            "Operation operations/456 failed: code=13, message=Internal error"
        )

        metrics = await trainer.train_step([_scored()])

        assert "code=13" in metrics["_train_step_error"]

    @pytest.mark.asyncio
    async def test_reports_error_when_train_step_fails_to_start(self):
        trainer = _build_stubbed_trainer()
        trainer.client._request.side_effect = RuntimeError("503 unavailable")

        metrics = await trainer.train_step([_scored()])

        assert "503 unavailable" in metrics["_train_step_error"]

    @pytest.mark.asyncio
    async def test_reports_error_when_no_operation_name_returned(self):
        trainer = _build_stubbed_trainer()
        trainer.client._request.return_value = {}

        metrics = await trainer.train_step([_scored()])

        assert "no operation name" in metrics["_train_step_error"]

    @pytest.mark.asyncio
    async def test_reports_error_when_no_valid_examples(self):
        """Every agent step missing a candidateId leaves nothing to send."""
        trainer = _build_stubbed_trainer()

        metrics = await trainer.train_step([_scored(candidate_id=None)])

        assert "no valid training examples" in metrics["_train_step_error"]
        trainer.client._request.assert_not_called()


# ---------------------------------------------------------------------------
# Outage behaviour: what the per-backend retry profiles are FOR. A rollout
# should survive the tuning backend going away and coming back, without
# resending work the server already has.
# ---------------------------------------------------------------------------


def _outage_sampler(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    retry: LroRetryProfile,
) -> VertexMsrlRestSampler:
    """Real client + sampler over a scripted transport, credentials stubbed.

    Unlike ``_build_stubbed_sampler`` this drives the REAL ``_request`` /
    ``wait_for_lro`` loops, which is the whole point: the behaviour under
    test is the retry, not the call.
    """
    with patch(
        "backends.vertex_msrl_rest.get_call_credentials",
        return_value=MagicMock(valid=True, token="fake"),
    ):
        client = VertexMsrlRestClient(
            project="p",
            location="us-central1",
            endpoint="us-central1-aiplatform.googleapis.com",
            tuning_job_id="job-1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )
    return VertexMsrlRestSampler(client, retry=retry)


_DONE_OP = {
    "name": "projects/p/locations/us-central1/operations/op-1",
    "done": True,
    "response": {
        "trainStepId": "1",
        "tuningCandidates": [
            {
                "candidateId": "cand-1",
                "candidate": {
                    "content": {"parts": [{"text": "hi"}], "role": "model"},
                    "finishReason": "STOP",
                },
                "generationTokenCount": 1,
                "thoughtsTokenCount": 0,
                # Required: a candidate without it is treated as malformed
                # usage and replaced by a failsafe.
                "promptTokenCount": 3,
            }
        ],
    },
}

_FAST_POLL = Backoff(0.001, 0.001, factor=1.0, jitter=0.0)


class TestLroRetryProfileDefaults:
    """The profile must be a pure refactor of the module constants.

    The numbers moved from private constants into a config-settable value;
    if a default drifts, a rollout deadline that was matched to the 10h
    poll cap is silently wrong, and nothing else would catch it.
    """

    def test_defaults_match_the_module_constants(self) -> None:
        from backends import vertex_msrl_rest as vmr

        profile = LroRetryProfile()
        assert profile.create_deadline_s == vmr._OP_START_RETRY_DEADLINE == 2 * 3600
        assert profile.poll_deadline_s == vmr._OP_POLL_TIMEOUT == 10 * 3600
        assert profile.poll_backoff.initial_s == vmr._POLL_BACKOFF_INITIAL == 10.0
        assert profile.poll_backoff.max_s == vmr._POLL_BACKOFF_MAX == 90.0
        assert profile.poll_backoff.factor == vmr._POLL_BACKOFF_FACTOR == 2.0

    def test_sampler_and_trainer_default_to_that_profile(self) -> None:
        client = MagicMock(spec=VertexMsrlRestClient)
        assert VertexMsrlRestSampler(client).retry == LroRetryProfile()
        assert VertexMsrlRestTrainer(client).retry == LroRetryProfile()


class TestLroOutageBehaviour:
    async def test_outage_during_poll_never_recreates_the_operation(self) -> None:
        """The long tail of the budget belongs to polling.

        A create that is resent may duplicate a sample that already landed
        and spends a second server-side permit, so an outage *after* the op
        exists must be ridden out with GetOperation and nothing else.
        """
        calls = {"create": 0, "poll": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith(":generateContentTuningScope"):
                calls["create"] += 1
                return httpx.Response(200, json={"name": _DONE_OP["name"]})
            calls["poll"] += 1
            if calls["poll"] <= 5:  # the outage
                return httpx.Response(503, json={"error": {"code": 503}})
            return httpx.Response(200, json=_DONE_OP)

        sampler = _outage_sampler(
            handler,
            retry=LroRetryProfile(poll_deadline_s=30.0, poll_backoff=_FAST_POLL),
        )
        results = await sampler.sample_n("hello", num_samples=1)

        assert calls["create"] == 1, "the operation must never be recreated"
        assert calls["poll"] == 6
        # And the rollout survived an outage shorter than the budget.
        assert results[0].tuning_candidate.candidate_id == "cand-1"

    async def test_the_poll_budget_is_the_stop_and_yields_the_failsafe(self) -> None:
        """An op that never finishes is abandoned at ``poll_deadline_s``.

        Pinned as-is: the exhausted poll returns the failsafe empty result
        rather than raising, which is what every caller downstream expects
        today. (Whether it SHOULD raise, so the outcome is attributable to
        the sampler instead of arriving as a generic client error, is a
        behaviour question and not part of this refactor.)
        """

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith(":generateContentTuningScope"):
                return httpx.Response(200, json={"name": _DONE_OP["name"]})
            return httpx.Response(200, json={"name": _DONE_OP["name"], "done": False})

        sampler = _outage_sampler(
            handler,
            retry=LroRetryProfile(poll_deadline_s=0.05, poll_backoff=_FAST_POLL),
        )
        results = await sampler.sample_n("hello", num_samples=1)

        assert results[0].tuning_candidate.candidate_id == ""

    async def test_a_refused_create_is_resent(self) -> None:
        """A create is retried within ``create_deadline_s``.

        ``429 RESOURCE_EXHAUSTED: ... there are already N`` is the routine
        permit-pool rejection at training concurrency, and riding those out
        is what the 2h create budget is for.
        """
        calls = {"create": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith(":generateContentTuningScope"):
                calls["create"] += 1
                if calls["create"] < 3:
                    return httpx.Response(
                        429,
                        json={"error": {"code": 429, "status": "RESOURCE_EXHAUSTED"}},
                    )
                return httpx.Response(200, json={"name": _DONE_OP["name"]})
            return httpx.Response(200, json=_DONE_OP)

        sampler = _outage_sampler(
            handler,
            retry=LroRetryProfile(
                create_deadline_s=30.0,
                poll_deadline_s=30.0,
                poll_backoff=_FAST_POLL,
            ),
        )
        results = await sampler.sample_n("hello", num_samples=1)

        assert calls["create"] == 3
        assert results[0].tuning_candidate.candidate_id == "cand-1"

    async def test_an_exhausted_create_budget_yields_the_failsafe(self) -> None:
        """``create_deadline_s=0`` is a single attempt; the rollout survives.

        The knob the constants could not express: a caller that wants one
        create attempt (because a resend may duplicate a sample that
        already landed) can now say so, without that becoming the default
        for everyone.
        """
        calls = {"create": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["create"] += 1
            raise httpx.ConnectTimeout("no answer")

        sampler = _outage_sampler(
            handler,
            retry=LroRetryProfile(
                create_deadline_s=0.0,
                poll_deadline_s=30.0,
                poll_backoff=_FAST_POLL,
            ),
        )
        results = await sampler.sample_n("hello", num_samples=1)

        assert calls["create"] == 1
        # Create failures keep the failsafe contract (no op to attribute to).
        assert results[0].tuning_candidate.candidate_id == ""
