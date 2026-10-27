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

"""Extend gemini_api backend tests for raw_call_uri stamping (per-candidate)."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from backends.gemini_api_endpoint import GeminiApiSampler
from backends.retry_profile import Backoff, SyncRetryProfile
from utils.raw_capture import CaptureContext, Counter, _reload_capture_enabled


@pytest.fixture
def enable_capture(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("RAW_CAPTURE", "1")
    _reload_capture_enabled()
    yield
    monkeypatch.delenv("RAW_CAPTURE", raising=False)
    _reload_capture_enabled()


class TestGeminiApiCaptureIntegration:
    async def test_sample_stamps_raw_call_uri_when_capture_enabled(
        self,
        enable_capture: None,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        trial_dir = tmp_path / "trial"
        trial_dir.mkdir()
        ctx = CaptureContext(
            trial_dir=trial_dir,
            trajectory_id="test-session",
            counter=Counter(),
        )
        sampler = self._build_sampler(monkeypatch)
        canned = {
            "candidates": [
                {
                    "index": 0,
                    "content": {"parts": [{"text": "hello"}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 1,
            },
            "responseId": "resp-abc",
            "modelVersion": "gemini-3.5-flash",
        }
        sampler._post_with_retries = AsyncMock(return_value=canned)

        results = await sampler.sample_n(prompt="hi", num_samples=1, capture_ctxs=[ctx])

        assert len(results) == 1
        assert (
            results[0].context.raw_call_uri
            == "artifacts/raw_llm_calls/call_000000.json"
        )
        blob_path = trial_dir / results[0].context.raw_call_uri
        assert blob_path.exists()
        blob = json.loads(blob_path.read_text())
        assert blob["trajectory_id"] == "test-session"
        assert blob["status"] == "ok"
        # Per-candidate slice: envelope keys preserved + single candidate.
        assert blob["response"]["responseId"] == "resp-abc"
        assert blob["response"]["modelVersion"] == "gemini-3.5-flash"
        assert len(blob["response"]["candidates"]) == 1
        assert blob["response"]["candidates"][0] == canned["candidates"][0]

    async def test_sample_stamps_none_when_capture_disabled(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Explicit opt-out — RAW_CAPTURE is default-on since the
        # follow-up flip, so an unset env now ENABLES capture. Force
        # the disabled path with an explicit RAW_CAPTURE=0.
        monkeypatch.setenv("RAW_CAPTURE", "0")
        _reload_capture_enabled()
        trial_dir = tmp_path / "trial"
        trial_dir.mkdir()
        ctx = CaptureContext(
            trial_dir=trial_dir,
            trajectory_id="test-session",
            counter=Counter(),
        )
        sampler = self._build_sampler(monkeypatch)
        sampler._post_with_retries = AsyncMock(
            return_value={
                "candidates": [
                    {
                        "index": 0,
                        "content": {"parts": [{"text": "hi"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 1,
                    "candidatesTokenCount": 1,
                },
            }
        )
        results = await sampler.sample_n(prompt="hi", num_samples=1, capture_ctxs=[ctx])
        assert results[0].context.raw_call_uri is None
        assert not (trial_dir / "artifacts").exists()

    async def test_sample_no_ctxs_stamps_none(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _reload_capture_enabled()
        sampler = self._build_sampler(monkeypatch)
        sampler._post_with_retries = AsyncMock(
            return_value={
                "candidates": [
                    {
                        "index": 0,
                        "content": {"parts": [{"text": "hi"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 1,
                    "candidatesTokenCount": 1,
                },
            }
        )
        results = await sampler.sample_n(prompt="hi", num_samples=1)
        assert results[0].context.raw_call_uri is None

    async def test_sample_error_status_still_captured(
        self,
        enable_capture: None,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        trial_dir = tmp_path / "trial"
        trial_dir.mkdir()
        ctx = CaptureContext(
            trial_dir=trial_dir,
            trajectory_id="s",
            counter=Counter(),
        )
        sampler = self._build_sampler(monkeypatch)
        sampler._post_with_retries = AsyncMock(return_value=None)
        results = await sampler.sample_n(prompt="hi", num_samples=1, capture_ctxs=[ctx])
        assert results[0].tuning_candidate.candidate_id == ""
        assert (
            results[0].context.raw_call_uri
            == "artifacts/raw_llm_calls/call_000000.json"
        )
        blob = json.loads((trial_dir / results[0].context.raw_call_uri).read_text())
        assert blob["status"] == "error"

    async def test_sample_num_samples_2_writes_two_ledger_blobs(
        self,
        enable_capture: None,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Batched-fanout: 2 candidates, 2 ctxs -> 2 distinct trial_dirs each
        get their own ``call_000000.json``.

        This is the core invariant that closes the gap: batched
        num_samples>1 wire calls now produce N per-rollout ledger blobs.
        """
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
        sampler = self._build_sampler(monkeypatch)
        canned = {
            "candidates": [
                {
                    "index": 0,
                    "content": {"parts": [{"text": "a"}]},
                    "finishReason": "STOP",
                },
                {
                    "index": 1,
                    "content": {"parts": [{"text": "b"}]},
                    "finishReason": "STOP",
                },
            ],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 2,
            },
            "responseId": "resp-xy",
            "modelVersion": "gemini-3.5-flash",
        }
        sampler._post_with_retries = AsyncMock(return_value=canned)

        results = await sampler.sample_n(
            prompt="hi", num_samples=2, capture_ctxs=[ctx0, ctx1]
        )

        assert len(results) == 2
        # Each result's raw_call_uri points into ITS OWN trial_dir.
        assert (
            results[0].context.raw_call_uri
            == "artifacts/raw_llm_calls/call_000000.json"
        )
        assert (
            results[1].context.raw_call_uri
            == "artifacts/raw_llm_calls/call_000000.json"
        )
        # And the blobs actually landed in the respective trial_dirs.
        blob0 = json.loads(
            (trial_dir0 / "artifacts/raw_llm_calls/call_000000.json").read_text()
        )
        blob1 = json.loads(
            (trial_dir1 / "artifacts/raw_llm_calls/call_000000.json").read_text()
        )
        assert blob0["trajectory_id"] == "sid-0"
        assert blob1["trajectory_id"] == "sid-1"
        # Per-candidate slicing: each blob has ONLY its own candidate (not both).
        assert len(blob0["response"]["candidates"]) == 1
        assert len(blob1["response"]["candidates"]) == 1
        assert blob0["response"]["candidates"][0]["index"] == 0
        assert blob1["response"]["candidates"][0]["index"] == 1

    async def test_sample_capture_ctxs_length_mismatch_raises(
        self,
        enable_capture: None,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        trial_dir = tmp_path / "trial"
        trial_dir.mkdir()
        ctx = CaptureContext(trial_dir=trial_dir, trajectory_id="s", counter=Counter())
        sampler = self._build_sampler(monkeypatch)
        sampler._post_with_retries = AsyncMock(return_value={"candidates": []})
        with pytest.raises(ValueError, match="does not match num_samples"):
            await sampler.sample_n(
                prompt="hi",
                num_samples=2,
                capture_ctxs=[ctx],  # length 1 != 2
            )

    async def test_sample_capture_ctxs_none_element_skips_that_candidate(
        self,
        enable_capture: None,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A None element in capture_ctxs skips capture for that candidate
        only; the sibling candidates still get their blob."""
        trial_dir0 = tmp_path / "trial0"
        trial_dir2 = tmp_path / "trial2"
        trial_dir0.mkdir()
        trial_dir2.mkdir()
        ctx0 = CaptureContext(
            trial_dir=trial_dir0, trajectory_id="sid-0", counter=Counter()
        )
        ctx2 = CaptureContext(
            trial_dir=trial_dir2, trajectory_id="sid-2", counter=Counter()
        )
        sampler = self._build_sampler(monkeypatch)
        canned = {
            "candidates": [
                {
                    "index": 0,
                    "content": {"parts": [{"text": "a"}]},
                    "finishReason": "STOP",
                },
                {
                    "index": 1,
                    "content": {"parts": [{"text": "b"}]},
                    "finishReason": "STOP",
                },
                {
                    "index": 2,
                    "content": {"parts": [{"text": "c"}]},
                    "finishReason": "STOP",
                },
            ],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3},
        }
        sampler._post_with_retries = AsyncMock(return_value=canned)

        results = await sampler.sample_n(
            prompt="hi", num_samples=3, capture_ctxs=[ctx0, None, ctx2]
        )

        assert len(results) == 3
        # Element 0 + 2 captured; element 1 skipped.
        assert (
            results[0].context.raw_call_uri
            == "artifacts/raw_llm_calls/call_000000.json"
        )
        assert results[1].context.raw_call_uri is None
        assert (
            results[2].context.raw_call_uri
            == "artifacts/raw_llm_calls/call_000000.json"
        )
        assert (trial_dir0 / "artifacts/raw_llm_calls/call_000000.json").exists()
        assert (trial_dir2 / "artifacts/raw_llm_calls/call_000000.json").exists()

    async def test_sample_error_status_fans_out_to_all_ctxs(
        self,
        enable_capture: None,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """On upstream failure with num_samples=2, EACH of the 2 ctxs gets
        its own error-status blob."""
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
        sampler = self._build_sampler(monkeypatch)
        sampler._post_with_retries = AsyncMock(return_value=None)

        results = await sampler.sample_n(
            prompt="hi", num_samples=2, capture_ctxs=[ctx0, ctx1]
        )
        assert len(results) == 2
        assert all(r.tuning_candidate.candidate_id == "" for r in results)
        for i, td in enumerate((trial_dir0, trial_dir1)):
            blob = json.loads(
                (td / "artifacts/raw_llm_calls/call_000000.json").read_text()
            )
            assert blob["status"] == "error"
            assert blob["trajectory_id"] == f"sid-{i}"

    async def test_sample_parallel_dumps_all_candidates_get_blobs(
        self,
        enable_capture: None,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Per-candidate ``dump_raw`` calls fire concurrently (not serially).

        Uses a monkeypatched ``dump_raw`` that sleeps ``delay`` seconds
        per invocation and records enter/exit timestamps. With N=4 and
        delay=0.1s, the serial baseline is ~0.4s; concurrent gather
        should complete in ~0.1s. We assert wallclock < 0.25s to leave
        headroom for CI jitter while still catching a serial regression
        (which would take at least 4*0.1 = 0.4s).
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
        sampler = self._build_sampler(monkeypatch)
        canned = {
            "candidates": [
                {
                    "index": i,
                    "content": {"parts": [{"text": f"c{i}"}]},
                    "finishReason": "STOP",
                }
                for i in range(4)
            ],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 4},
        }
        sampler._post_with_retries = AsyncMock(return_value=canned)

        delay = 0.1
        call_windows: list[tuple[float, float]] = []
        real_dump_raw = raw_capture.dump_raw

        async def slow_dump_raw(**kwargs: object) -> str | None:
            t_enter = time.monotonic()
            await asyncio.sleep(delay)
            t_exit = time.monotonic()
            call_windows.append((t_enter, t_exit))
            return await real_dump_raw(**kwargs)  # type: ignore[arg-type]

        # Patched where ``dump_raw_fanout`` looks it up: the sampler now
        # calls the fan-out helper rather than ``dump_raw`` directly.
        monkeypatch.setattr("utils.raw_capture.dump_raw", slow_dump_raw)

        t0 = time.monotonic()
        results = await sampler.sample_n(prompt="hi", num_samples=4, capture_ctxs=ctxs)
        wall_s = time.monotonic() - t0

        assert len(results) == 4
        # All 4 dump_raw calls must have overlapping time windows
        # (proving they ran concurrently, not serially). With serial
        # execution the total would be >= 4*delay; concurrent gather
        # collapses to ~delay.
        assert wall_s < 4 * delay * 0.75, (
            f"expected concurrent dumps (< {4 * delay * 0.75:.2f}s), got {wall_s:.3f}s"
        )
        # Every candidate got a ledger blob written.
        for td in trial_dirs:
            assert (td / "artifacts/raw_llm_calls/call_000000.json").exists()

    async def test_sample_parallel_dumps_preserve_order(
        self,
        enable_capture: None,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Order preservation under parallel gather: results[i] carries the
        URI whose ledger blob contains candidate[i]'s content, not a
        shuffled URI from a sibling."""
        import asyncio

        from utils import raw_capture

        trial_dirs = [tmp_path / f"trial{i}" for i in range(4)]
        for td in trial_dirs:
            td.mkdir()
        ctxs = [
            CaptureContext(trial_dir=td, trajectory_id=f"sid-{i}", counter=Counter())
            for i, td in enumerate(trial_dirs)
        ]
        sampler = self._build_sampler(monkeypatch)
        # Distinct per-candidate content so we can verify the URI
        # returned in results[i] matches the ledger blob containing
        # candidate[i].
        canned = {
            "candidates": [
                {
                    "index": i,
                    "content": {"parts": [{"text": f"UNIQUE_CANDIDATE_{i}"}]},
                    "finishReason": "STOP",
                }
                for i in range(4)
            ],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 4},
        }
        sampler._post_with_retries = AsyncMock(return_value=canned)

        # Reverse-order delay: candidate 0 finishes LAST, candidate 3
        # finishes FIRST. If gather results were mis-ordered (indexed
        # by completion order instead of position), results[0]'s URI
        # would point to candidate 3's blob.
        real_dump_raw = raw_capture.dump_raw

        async def reversed_delay_dump_raw(
            *, ctx: CaptureContext | None, **kwargs: object
        ) -> str | None:
            # Delay proportional to sid index (recovered from trajectory_id).
            if ctx is not None:
                sid = int(ctx.trajectory_id.rsplit("-", 1)[-1])
                await asyncio.sleep(0.02 * (4 - sid))
            return await real_dump_raw(ctx=ctx, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr("utils.raw_capture.dump_raw", reversed_delay_dump_raw)

        results = await sampler.sample_n(prompt="hi", num_samples=4, capture_ctxs=ctxs)
        assert len(results) == 4
        # For each result, load the blob it references and verify the
        # blob's response contains the CORRECT sibling candidate.
        for i, result in enumerate(results):
            assert result.context.raw_call_uri is not None
            blob = json.loads((trial_dirs[i] / result.context.raw_call_uri).read_text())
            assert blob["trajectory_id"] == f"sid-{i}"
            assert (
                blob["response"]["candidates"][0]["content"]["parts"][0]["text"]
                == f"UNIQUE_CANDIDATE_{i}"
            )

    def _build_sampler(self, monkeypatch: pytest.MonkeyPatch) -> GeminiApiSampler:
        monkeypatch.setattr(
            "backends.gemini_api_endpoint.get_call_credentials",
            lambda **kw: MagicMock(valid=True, token="fake"),
        )
        return GeminiApiSampler(
            model="gemini-3.5-flash",
            project="test-project",
            env="prod",
        )


class TestPostWithRetries:
    """The 429 retry budget.

    Vertex ``RESOURCE_EXHAUSTED`` is a rate-limit *window*, not a transient
    blip, so the budget has to be long enough to outlast one. These drive the
    real ``_post_with_retries`` loop through an injected transport; only
    ``asyncio.sleep`` is faked, so the backoff schedule under test is the one
    the loop actually computes.
    """

    _OK_BODY = {
        "candidates": [
            {"index": 0, "content": {"parts": [{"text": "hi"}]}, "finishReason": "STOP"}
        ],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
    }

    def _sampler(
        self,
        monkeypatch: pytest.MonkeyPatch,
        handler: Callable[[httpx.Request], httpx.Response],
        **kwargs: Any,
    ) -> GeminiApiSampler:
        monkeypatch.setattr(
            "backends.gemini_api_endpoint.get_call_credentials",
            lambda **kw: MagicMock(valid=True, token="fake"),
        )
        return GeminiApiSampler(
            model="gemini-3.5-flash",
            project="test-project",
            env="prod",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            **kwargs,
        )

    @staticmethod
    def _capture_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
        """Record backoff delays instead of paying them."""
        delays: list[float] = []

        async def _fake_sleep(seconds: float) -> None:
            delays.append(seconds)

        monkeypatch.setattr(asyncio, "sleep", _fake_sleep)
        return delays

    async def test_recovers_from_a_429_burst_longer_than_three_attempts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A quota window outlasting the old 3-try budget must still succeed.

        This is the observed production failure: two rollouts died on a single
        429 because the budget was exhausted in ~6 seconds.
        """
        self._capture_sleeps(monkeypatch)
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            if attempts["n"] < 8:
                return httpx.Response(429, json={"error": {"code": 429}})
            return httpx.Response(200, json=self._OK_BODY)

        sampler = self._sampler(monkeypatch, handler)
        result = await sampler._post_with_retries({"contents": []})

        assert result is not None, "429 burst should have been ridden out"
        assert attempts["n"] == 8

    async def test_backoff_is_capped_at_the_profile_ceiling(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Uncapped ``2**n`` reaches 512s by the 9th wait — an absurd stall.

        The profile is passed explicitly so this pins the CAP regardless of
        what the default budget happens to be.
        """
        delays = self._capture_sleeps(monkeypatch)
        profile = SyncRetryProfile()
        sampler = self._sampler(
            monkeypatch,
            lambda r: httpx.Response(429, json={"error": {}}),
            retry=profile,
        )

        await sampler._post_with_retries({"contents": []})

        assert delays, "expected the loop to back off between attempts"
        # Jitter is applied on top of the cap.
        ceiling = profile.backoff.max_s * (1.0 + profile.backoff.jitter)
        assert max(delays) <= ceiling + 1e-9, f"uncapped backoff: {max(delays)}"

    def test_the_default_profile_is_the_historical_budget(self) -> None:
        """The refactor must not move the policy: 10 tries, 2 -> 30s, +/-20%.

        `SyncRetryProfile` replaced a bare ``max_tries`` int and two module
        constants. The values are the same ones; if a default drifts here,
        every eval that never names a profile silently changes.
        """
        profile = SyncRetryProfile()
        assert profile.max_attempts == 10
        assert profile.deadline_s is None, "a wall-clock stop must stay opt-in"
        assert (profile.backoff.initial_s, profile.backoff.max_s) == (2.0, 30.0)
        assert (profile.backoff.factor, profile.backoff.jitter) == (2.0, 0.2)
        # The schedule the old `min(2 ** (attempt + 1), 30)` produced.
        assert [profile.backoff.interval_for(n) for n in range(1, 7)] == [
            2.0,
            4.0,
            8.0,
            16.0,
            30.0,
            30.0,
        ]

    async def test_a_deadline_can_be_opted_into(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The knob the count could not express: a budget in SECONDS.

        Real (sub-millisecond) sleeps, because a faked clock cannot show a
        deadline binding. The attempt count is set absurdly high so that if
        it is what stops the loop, this fails.
        """
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            return httpx.Response(429, json={"error": {}})

        sampler = self._sampler(
            monkeypatch,
            handler,
            retry=SyncRetryProfile(
                deadline_s=0.05,
                max_attempts=10_000,
                backoff=Backoff(0.005, 0.005, factor=1.0, jitter=0.0),
            ),
        )
        result = await sampler._post_with_retries({"contents": []})

        assert result is None
        assert 1 < attempts["n"] < 10_000, (
            "the deadline should have stopped the loop long before the "
            f"attempt cap; got {attempts['n']} attempts"
        )

    async def test_the_attempt_count_stops_the_loop_with_no_deadline_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default shape: the count is the only stop, exactly as before."""
        self._capture_sleeps(monkeypatch)
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            return httpx.Response(429, json={"error": {}})

        sampler = self._sampler(
            monkeypatch,
            handler,
            retry=SyncRetryProfile(max_attempts=5),
        )
        await sampler._post_with_retries({"contents": []})

        assert attempts["n"] == 5

    async def test_first_retry_is_visible_at_info(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """At DEBUG, retry-exhaustion and never-retried look identical.

        That ambiguity caused a real misdiagnosis of a production 429, so the
        first retry of a burst has to be visible at the default level.
        """
        self._capture_sleeps(monkeypatch)
        caplog.set_level(logging.INFO, logger="backends.gemini_api_endpoint")
        sampler = self._sampler(
            monkeypatch,
            lambda r: httpx.Response(429, json={"error": {}}),
            retry=SyncRetryProfile(max_attempts=4),
        )

        await sampler._post_with_retries({"contents": []})

        retry_lines = [
            r
            for r in caplog.records
            if r.levelno == logging.INFO and "retrying" in r.getMessage()
        ]
        assert retry_lines, "the retry itself must be visible at INFO"

    async def test_a_retry_storm_is_throttled_to_one_line_per_interval(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """48-wide fan-out x 10 tries would otherwise flood the run log."""
        self._capture_sleeps(monkeypatch)
        caplog.set_level(logging.INFO, logger="backends.gemini_api_endpoint")
        sampler = self._sampler(
            monkeypatch,
            lambda r: httpx.Response(429, json={"error": {}}),
            retry=SyncRetryProfile(max_attempts=10),
        )

        await sampler._post_with_retries({"contents": []})

        retry_lines = [
            r
            for r in caplog.records
            if r.levelno == logging.INFO and "retrying" in r.getMessage()
        ]
        # 9 retries happen back-to-back (sleep is faked), so they fall inside
        # one throttle window: exactly one full line, not nine.
        assert len(retry_lines) == 1, f"expected throttling, got {len(retry_lines)}"

    async def test_exhausting_the_budget_still_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The failsafe contract is unchanged: give up, don't raise."""
        self._capture_sleeps(monkeypatch)
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            return httpx.Response(429, json={"error": {}})

        sampler = self._sampler(
            monkeypatch, handler, retry=SyncRetryProfile(max_attempts=4)
        )
        result = await sampler._post_with_retries({"contents": []})

        assert result is None
        assert attempts["n"] == 4, "the attempt cap must stay caller-tunable"

    async def test_a_non_retryable_status_fails_on_the_first_attempt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A wider budget must not turn a 400 into ten pointless calls."""
        self._capture_sleeps(monkeypatch)
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            return httpx.Response(400, json={"error": {"code": 400}})

        sampler = self._sampler(monkeypatch, handler)
        result = await sampler._post_with_retries({"contents": []})

        assert result is None
        assert attempts["n"] == 1
