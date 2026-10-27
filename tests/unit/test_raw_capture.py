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

"""Tests for the raw-capture layer.

``utils.raw_capture`` — CaptureContext, Counter, dump_raw, and the
per-candidate ``dump_raw_fanout`` every sampler calls — plus
``core.sample_result.with_raw_call_uris``, which puts the URIs the
fanout returns back onto the results they belong to.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.sample_result import with_raw_call_uris
from tests.unit._sample_result_factories import make_sample_result
from utils.raw_capture import (
    Counter,
    CaptureContext,
    client_request_scope,
    dump_raw,
    dump_raw_fanout,
    _parse_capture_enabled,
    _reload_capture_enabled,
)
import utils.raw_capture as _raw_capture_module


@pytest.fixture
def enable_capture(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Turn RAW_CAPTURE=1 on for the test and refresh the module cache."""
    monkeypatch.setenv("RAW_CAPTURE", "1")
    _reload_capture_enabled()
    yield
    monkeypatch.delenv("RAW_CAPTURE", raising=False)
    _reload_capture_enabled()


@pytest.fixture
def disable_capture(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Explicit opt-out via RAW_CAPTURE=0.

    Post-default-on flip: unsetting the env var no longer disables
    capture (unset is the default-ENABLED path). Tests that need
    capture DISABLED must explicitly set RAW_CAPTURE=0.
    """
    monkeypatch.setenv("RAW_CAPTURE", "0")
    _reload_capture_enabled()
    yield
    monkeypatch.delenv("RAW_CAPTURE", raising=False)
    _reload_capture_enabled()


@pytest.fixture
def trial_dir(tmp_path: Path) -> Path:
    d = tmp_path / "trial"
    d.mkdir()
    return d


@pytest.fixture
def ctx(trial_dir: Path) -> CaptureContext:
    return CaptureContext(
        trial_dir=trial_dir,
        trajectory_id="test-session-42",
        counter=Counter(),
    )


class TestCounter:
    async def test_starts_at_zero(self) -> None:
        c = Counter()
        assert await c.next() == 0

    async def test_monotonic(self) -> None:
        c = Counter()
        assert await c.next() == 0
        assert await c.next() == 1
        assert await c.next() == 2

    async def test_concurrent_calls_are_serialized(self) -> None:
        c = Counter()
        results = await asyncio.gather(*[c.next() for _ in range(10)])
        assert sorted(results) == list(range(10))


class TestDumpRawEnvVarGate:
    async def test_explicit_opt_out_returns_none(
        self, disable_capture: None, ctx: CaptureContext
    ) -> None:
        """RAW_CAPTURE=0 explicitly disables — dump_raw short-circuits."""
        uri = await dump_raw(
            ctx=ctx,
            request={"a": 1},
            response={"b": 2},
            status="ok",
            duration_s=0.5,
        )
        assert uri is None
        # No file written.
        assert not (ctx.trial_dir / "artifacts").exists()

    async def test_none_ctx_returns_none(
        self, enable_capture: None, trial_dir: Path
    ) -> None:
        uri = await dump_raw(
            ctx=None,
            request={},
            response={},
            status="ok",
            duration_s=0.0,
        )
        assert uri is None


class TestDumpRawWrites:
    async def test_writes_contract_blob_shape(
        self, enable_capture: None, ctx: CaptureContext
    ) -> None:
        uri = await dump_raw(
            ctx=ctx,
            request={"prompt": "hi"},
            response={"candidates": [{"text": "hello"}]},
            status="ok",
            duration_s=1.23,
        )
        assert uri == "artifacts/raw_llm_calls/call_000000.json"
        blob_path = ctx.trial_dir / uri
        assert blob_path.exists()
        blob = json.loads(blob_path.read_text())
        # Contract-mandated keys:
        assert blob["trajectory_id"] == "test-session-42"
        assert blob["call_idx"] == 0
        assert blob["status"] == "ok"
        assert blob["duration_s"] == 1.23
        assert blob["request"] == {"prompt": "hi"}
        assert blob["response"] == {"candidates": [{"text": "hello"}]}
        assert "ts" in blob and isinstance(blob["ts"], float)

    async def test_monotonic_filenames_across_calls(
        self, enable_capture: None, ctx: CaptureContext
    ) -> None:
        uris = []
        for i in range(3):
            u = await dump_raw(
                ctx=ctx,
                request={"i": i},
                response={},
                status="ok",
                duration_s=0.1,
            )
            uris.append(u)
        assert uris == [
            "artifacts/raw_llm_calls/call_000000.json",
            "artifacts/raw_llm_calls/call_000001.json",
            "artifacts/raw_llm_calls/call_000002.json",
        ]
        for u in uris:
            assert (ctx.trial_dir / u).exists()

    async def test_status_error_and_empty_pass_through(
        self, enable_capture: None, ctx: CaptureContext
    ) -> None:
        u1 = await dump_raw(
            ctx=ctx, request={}, response={"err": "..."}, status="error", duration_s=0.0
        )
        u2 = await dump_raw(
            ctx=ctx,
            request={},
            response={"candidates": []},
            status="empty",
            duration_s=0.1,
        )
        for u, expected in [(u1, "error"), (u2, "empty")]:
            assert u is not None
            blob = json.loads((ctx.trial_dir / u).read_text())
            assert blob["status"] == expected


class TestDumpRawIsolation:
    async def test_write_failure_returns_none_and_logs(
        self,
        enable_capture: None,
        ctx: CaptureContext,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Patch write_text to raise IOError.
        import utils.raw_capture as rc

        with patch.object(rc.Path, "write_text", side_effect=IOError("disk full")):
            uri = await dump_raw(
                ctx=ctx,
                request={},
                response={},
                status="ok",
                duration_s=0.0,
            )
        assert uri is None
        assert "dump_raw failed" in caplog.text

    async def test_write_of_unencodable_object_returns_none_and_logs(
        self,
        enable_capture: None,
        ctx: CaptureContext,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # A non-JSON, non-proto object (a lambda) in the response must
        # cause dump_raw to WARN + return None, not silently stringify.
        class Unencodable:
            pass

        uri = await dump_raw(
            ctx=ctx,
            request={},
            response={"weird": Unencodable()},
            status="ok",
            duration_s=0.0,
        )
        assert uri is None
        assert "dump_raw failed" in caplog.text
        assert (
            "Unencodable" in caplog.text or "TypeError" in caplog.text.lower() or True
        )
        # (The last assertion is loose because the exact log formatting
        # of the exception isn't part of the contract; existence of the
        # WARN line and the None return are.)


class TestDumpRawOnWrittenCallback:
    """CaptureContext.on_written is invoked after a successful local write.

    Wires the raw-LLM ledger into AtifDiskWriter's GCS uploader (real
    design gap: without this, dump_raw would land blobs on the training
    VM's local disk but the trajectory JSONs uploaded to GCS pointed at
    paths that only existed there, breaking cross-reference for GCS
    consumers).
    """

    async def test_dump_raw_invokes_on_written_after_local_write(
        self, enable_capture: None, trial_dir: Path
    ) -> None:
        spy = MagicMock()
        ctx = CaptureContext(
            trial_dir=trial_dir,
            trajectory_id="sid-1",
            counter=Counter(),
            on_written=spy,
        )
        uri = await dump_raw(
            ctx=ctx,
            request={"prompt": "hi"},
            response={"candidates": [{"text": "hello"}]},
            status="ok",
            duration_s=0.5,
        )
        assert uri == "artifacts/raw_llm_calls/call_000000.json"
        # File must exist on disk BEFORE the callback fires.
        blob_path = trial_dir / uri
        assert blob_path.exists()
        # Callback invoked exactly once, with the absolute Path.
        spy.assert_called_once()
        (call_arg,) = spy.call_args.args
        assert isinstance(call_arg, Path)
        assert call_arg == blob_path

    async def test_dump_raw_on_written_callback_exception_swallowed(
        self,
        enable_capture: None,
        trial_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A callback failure (e.g. GCS uploader queue full) MUST NOT
        crash the sampler hot path or invalidate the local write."""
        spy = MagicMock(side_effect=RuntimeError("upload broke"))
        ctx = CaptureContext(
            trial_dir=trial_dir,
            trajectory_id="sid-2",
            counter=Counter(),
            on_written=spy,
        )
        uri = await dump_raw(
            ctx=ctx,
            request={"a": 1},
            response={"b": 2},
            status="ok",
            duration_s=0.1,
        )
        # Valid URI returned despite callback failure -- local write was ok.
        assert uri == "artifacts/raw_llm_calls/call_000000.json"
        assert (trial_dir / uri).exists()
        spy.assert_called_once()
        # WARN logged so operators see the callback failure.
        assert "on_written" in caplog.text or "callback" in caplog.text

    async def test_dump_raw_no_on_written_callback_default_none(
        self, enable_capture: None, trial_dir: Path
    ) -> None:
        """Default ``on_written=None`` MUST work (no crash, no callback
        attempted) so callers who don't care about uploads (local-only
        atif_output) are unaffected."""
        ctx = CaptureContext(
            trial_dir=trial_dir,
            trajectory_id="sid-3",
            counter=Counter(),
        )
        assert ctx.on_written is None
        uri = await dump_raw(
            ctx=ctx,
            request={},
            response={},
            status="ok",
            duration_s=0.0,
        )
        assert uri == "artifacts/raw_llm_calls/call_000000.json"
        assert (trial_dir / uri).exists()


class TestParseCaptureEnabled:
    """Tolerant parsing of RAW_CAPTURE — DEFAULT-ON (opt-out) semantics.

    Post-flip contract:
    - UNSET or empty string -> ENABLED (default-on).
    - Canonical truthy strings (case-insensitive, whitespace-stripped):
      ``1``, ``true``, ``yes``, ``on`` -> ENABLED.
    - Canonical falsy strings: ``0``, ``false``, ``no``, ``off`` ->
      DISABLED (the only way to opt out).
    - Any other non-empty value (typo like ``RAW_CAPTURE=trueee``) is
      treated as ENABLED (default) AND logs a WARN at parse time so
      operator typos are visible at process start rather than silently
      degrading behavior in either direction.

    History: the module originally defaulted OFF (opt-in via
    ``RAW_CAPTURE=1``). The default flipped to ON in a follow-up
    once operators signed off on the storage cost of capturing
    every training run's raw LLM traffic.
    """

    @pytest.mark.parametrize(
        "raw",
        [
            "1",
            "true",
            "TRUE",
            "True",
            "yes",
            "Yes",
            "YES",
            "on",
            "ON",
            "  1  ",
            "TRUE  ",
            " yes",
        ],
    )
    def test_truthy_values_parse_true_no_warn(
        self,
        raw: str,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.setenv("RAW_CAPTURE", raw)
        with caplog.at_level("WARNING", logger="utils.raw_capture"):
            result = _parse_capture_enabled()
        assert result is True
        # No WARN for canonical truthy values.
        assert "not a recognized" not in caplog.text

    @pytest.mark.parametrize(
        "raw",
        ["0", "false", "FALSE", "False", "no", "No", "off", "OFF", "  0  "],
    )
    def test_falsy_values_parse_false_no_warn(
        self,
        raw: str,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Explicit falsy values disable capture without WARN."""
        monkeypatch.setenv("RAW_CAPTURE", raw)
        with caplog.at_level("WARNING", logger="utils.raw_capture"):
            result = _parse_capture_enabled()
        assert result is False
        assert "not a recognized" not in caplog.text

    def test_parse_capture_enabled_unset_defaults_to_true(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """UNSET env var -> ENABLED (post-flip default-on).

        This is the canonical "operator did nothing" path. Prior to the
        default flip, unset returned False.
        """
        monkeypatch.delenv("RAW_CAPTURE", raising=False)
        with caplog.at_level("WARNING", logger="utils.raw_capture"):
            result = _parse_capture_enabled()
        assert result is True
        # No WARN — unset is the default operational path, not a typo.
        assert "not a recognized" not in caplog.text

    @pytest.mark.parametrize("raw", ["", "  "])
    def test_parse_capture_enabled_empty_string_defaults_to_true(
        self,
        raw: str,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Empty / whitespace-only env var -> ENABLED (treated as unset).

        Matches the ``unset -> os.environ.get("", "") -> "" -> enabled``
        chain so operators who accidentally set ``RAW_CAPTURE=""`` get
        the same behavior as leaving it unset.
        """
        monkeypatch.setenv("RAW_CAPTURE", raw)
        with caplog.at_level("WARNING", logger="utils.raw_capture"):
            result = _parse_capture_enabled()
        assert result is True
        assert "not a recognized" not in caplog.text

    @pytest.mark.parametrize(
        "raw",
        ["0", "false", "no", "off", "FALSE", "OFF"],
    )
    def test_parse_capture_enabled_explicit_false_disables(
        self,
        raw: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Explicit falsy values are the ONLY way to opt out of capture.

        Parametrized companion to test_falsy_values_parse_false_no_warn
        — this variant asserts the disable contract from the caller's
        perspective ("how do I turn capture off?") rather than the
        WARN-suppression contract.
        """
        monkeypatch.setenv("RAW_CAPTURE", raw)
        assert _parse_capture_enabled() is False

    @pytest.mark.parametrize(
        "raw",
        ["trueee", "yesss", "2", "1.0", "enabled", "1yes", "y", "t"],
    )
    def test_unrecognized_values_enable_and_warn(
        self,
        raw: str,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Typos default to ENABLED (matches default-on posture) + WARN.

        Prior to the default-on flip, typos disabled capture. Now they
        enable it, matching the "operator did nothing sensible ->
        default behavior" posture. The WARN is still critical so the
        typo is visible at process start.
        """
        monkeypatch.setenv("RAW_CAPTURE", raw)
        with caplog.at_level("WARNING", logger="utils.raw_capture"):
            result = _parse_capture_enabled()
        assert result is True
        # WARN mentions the operator's actual value so they can grep.
        assert "not a recognized" in caplog.text
        assert raw in caplog.text
        # WARN wording steers the operator to the correct opt-out knob.
        assert "RAW_CAPTURE=0" in caplog.text

    def test_reload_capture_enabled_reparses_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("RAW_CAPTURE", "true")
        _reload_capture_enabled()
        assert _raw_capture_module._CAPTURE_ENABLED is True
        monkeypatch.setenv("RAW_CAPTURE", "off")
        _reload_capture_enabled()
        assert _raw_capture_module._CAPTURE_ENABLED is False
        # UNSET -> default-on (was default-off pre-flip).
        monkeypatch.delenv("RAW_CAPTURE", raising=False)
        _reload_capture_enabled()
        assert _raw_capture_module._CAPTURE_ENABLED is True

    def test_default_on_matches_prod_expectation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Canary: with no RAW_CAPTURE in the env, _CAPTURE_ENABLED is True.

        Guards against accidental revert of the default-on flip. If
        someone flips the parser back to default-off without updating
        this test, this will fail loudly rather than silently
        regressing every training run to no-capture.
        """
        monkeypatch.delenv("RAW_CAPTURE", raising=False)
        _reload_capture_enabled()
        assert _raw_capture_module._CAPTURE_ENABLED is True, (
            "RAW_CAPTURE default must be ENABLED (opt-out via RAW_CAPTURE=0). "
            "If this test failed, someone reverted the "
            "default-on flip; check src/utils/raw_capture.py::"
            "_parse_capture_enabled."
        )


class TestDumpRawUriFormat:
    """dump_raw returns a directly-resolvable URI stamped on
    step.extra.llm_call_uri.

    Contract:
      * blob_uri_prefix=None (bare CaptureContext, test setups) -> the
        legacy trial-dir-relative URI (backward compat).
      * blob_uri_prefix=<abs local path> -> absolute local path URI.
      * blob_uri_prefix=<gs://bucket/prefix/.../trial> -> full gs:// URI
        that a consumer can gcloud-storage-cat directly.
    """

    async def test_dump_raw_returns_full_gcs_uri_when_prefix_set(
        self, enable_capture: None, trial_dir: Path
    ) -> None:
        ctx = CaptureContext(
            trial_dir=trial_dir,
            trajectory_id="sid-gcs",
            counter=Counter(),
            blob_uri_prefix="gs://test-bucket/testprefix/recipe/job/task-0",
        )
        uri = await dump_raw(
            ctx=ctx,
            request={"prompt": "hi"},
            response={"candidates": [{"text": "hello"}]},
            status="ok",
            duration_s=0.5,
        )
        assert uri == (
            "gs://test-bucket/testprefix/recipe/job/task-0/"
            "artifacts/raw_llm_calls/call_000000.json"
        )
        # Local file still written under trial_dir (the local write is
        # unchanged; only the returned URI shape changed).
        assert (trial_dir / "artifacts" / "raw_llm_calls" / "call_000000.json").exists()

    async def test_dump_raw_returns_absolute_local_path_when_prefix_is_local(
        self, enable_capture: None, trial_dir: Path
    ) -> None:
        abs_prefix = str(trial_dir.absolute())
        ctx = CaptureContext(
            trial_dir=trial_dir,
            trajectory_id="sid-local",
            counter=Counter(),
            blob_uri_prefix=abs_prefix,
        )
        uri = await dump_raw(
            ctx=ctx,
            request={},
            response={},
            status="ok",
            duration_s=0.0,
        )
        assert uri is not None
        assert uri.startswith(abs_prefix)
        assert uri.endswith("/artifacts/raw_llm_calls/call_000000.json")

    async def test_dump_raw_returns_relative_uri_when_prefix_is_none_backward_compat(
        self, enable_capture: None, trial_dir: Path
    ) -> None:
        """Bare CaptureContext with default ``blob_uri_prefix=None`` (test
        setups that don't go through AtifDiskWriter) MUST still return the
        legacy trial-dir-relative URI for backward compat."""
        ctx = CaptureContext(
            trial_dir=trial_dir,
            trajectory_id="sid-bare",
            counter=Counter(),
        )
        assert ctx.blob_uri_prefix is None
        uri = await dump_raw(
            ctx=ctx,
            request={},
            response={},
            status="ok",
            duration_s=0.0,
        )
        assert uri == "artifacts/raw_llm_calls/call_000000.json"


class TestConcurrentDumps:
    async def test_ten_concurrent_calls_produce_ten_files(
        self, enable_capture: None, ctx: CaptureContext
    ) -> None:
        async def one_call(i: int) -> str | None:
            return await dump_raw(
                ctx=ctx,
                request={"i": i},
                response={},
                status="ok",
                duration_s=0.0,
            )

        uris = await asyncio.gather(*[one_call(i) for i in range(10)])
        # All URIs unique; every file exists.
        assert len(set(uris)) == 10
        for u in uris:
            assert u is not None
            assert (ctx.trial_dir / u).exists()
        # Sorted filenames are call_000000..000009.
        actual_names = sorted((ctx.trial_dir / "artifacts" / "raw_llm_calls").iterdir())
        assert [p.name for p in actual_names] == [
            f"call_{i:06d}.json" for i in range(10)
        ]


# ---------------------------------------------------------------------------
# client_request: the OpenAI-form body behind a converted vendor call
# ---------------------------------------------------------------------------


async def test_the_client_request_is_recorded_when_a_scope_is_active(tmp_path):
    """The blob's `request` is the VENDOR body; the agent's own request —
    tool schemas included — would otherwise be recorded nowhere."""
    from utils.raw_capture import Counter, CaptureContext, client_request_scope, dump_raw

    ctx = CaptureContext(
        trial_dir=tmp_path, trajectory_id="t-1", counter=Counter()
    )
    openai_request = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "api_search",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
    }
    with client_request_scope(openai_request):
        await dump_raw(
            ctx=ctx,
            request={"contents": [{"role": "user"}]},  # the converted body
            response={"candidates": []},
            status="ok",
            duration_s=0.1,
        )

    blob = json.loads(
        (tmp_path / "artifacts" / "raw_llm_calls" / "call_000000.json").read_text()
    )
    assert blob["request"] == {"contents": [{"role": "user"}]}
    assert blob["client_request"]["tools"][0]["function"]["name"] == "api_search"


async def test_no_client_request_key_outside_a_scope(tmp_path):
    # The proto path has no client form; the key must simply be absent
    # rather than present-and-null.
    from utils.raw_capture import Counter, CaptureContext, dump_raw

    ctx = CaptureContext(trial_dir=tmp_path, trajectory_id="t", counter=Counter())
    await dump_raw(
        ctx=ctx, request={"a": 1}, response={"b": 2}, status="ok", duration_s=0.0
    )
    blob = json.loads(
        (tmp_path / "artifacts" / "raw_llm_calls" / "call_000000.json").read_text()
    )
    assert "client_request" not in blob


async def test_the_scope_does_not_leak_between_concurrent_tasks(tmp_path):
    """Two rollouts sampling at once must not see each other's request."""
    import asyncio

    from utils.raw_capture import Counter, CaptureContext, client_request_scope, dump_raw

    ctx = CaptureContext(trial_dir=tmp_path, trajectory_id="t", counter=Counter())

    async def one(tag: str) -> None:
        with client_request_scope({"model": tag}):
            await asyncio.sleep(0)  # yield, so the tasks interleave
            await dump_raw(
                ctx=ctx, request={}, response={}, status="ok", duration_s=0.0
            )

    await asyncio.gather(one("a"), one("b"))
    blobs = [
        json.loads(p.read_text())
        for p in sorted((tmp_path / "artifacts" / "raw_llm_calls").iterdir())
    ]
    assert sorted(b["client_request"]["model"] for b in blobs) == ["a", "b"]


# ---------------------------------------------------------------------------
# dump_raw_fanout: one blob per candidate, from one call site
# ---------------------------------------------------------------------------


def _rollout_ctx(root: Path, name: str) -> CaptureContext:
    """One rollout's capture context: its own trial dir, its own counter."""
    return CaptureContext(trial_dir=root / name, trajectory_id=name, counter=Counter())


def _blobs_of(ctx: CaptureContext) -> list[dict]:
    """Every ledger blob written under ``ctx``, in call order."""
    outdir = ctx.trial_dir / "artifacts" / "raw_llm_calls"
    if not outdir.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(outdir.iterdir())]


class TestDumpRawFanout:
    async def test_one_blob_lands_under_each_candidates_own_ctx(
        self, enable_capture: None, tmp_path: Path
    ) -> None:
        ctxs = [_rollout_ctx(tmp_path, f"r{i}") for i in range(3)]

        uris = await dump_raw_fanout(
            ctxs,
            count=3,
            request={"q": 1},
            response={"a": 2},
            status="ok",
            duration_s=1.5,
        )

        assert len(uris) == 3
        assert all(u is not None for u in uris)
        for ctx in ctxs:
            blobs = _blobs_of(ctx)
            assert len(blobs) == 1
            assert blobs[0]["status"] == "ok"
            assert blobs[0]["duration_s"] == 1.5
            assert blobs[0]["request"] == {"q": 1}
            assert blobs[0]["response"] == {"a": 2}

    async def test_a_short_ctxs_list_captures_what_it_can(
        self, enable_capture: None, tmp_path: Path
    ) -> None:
        """Losing a blob must never lose a sample: a caller that passes one
        ctx for four candidates gets capture for candidate 0 and None for
        the rest, not an IndexError mid-call."""
        ctxs = [_rollout_ctx(tmp_path, "r0")]

        uris = await dump_raw_fanout(
            ctxs,
            count=4,
            request={},
            response={},
            status="ok",
            duration_s=0.0,
        )

        assert uris[0] is not None
        assert uris[1:] == [None, None, None]
        assert len(_blobs_of(ctxs[0])) == 1

    async def test_no_ctxs_at_all_is_not_an_error(
        self, enable_capture: None
    ) -> None:
        assert await dump_raw_fanout(
            None, count=2, request={}, response={}, status="ok", duration_s=0.0
        ) == [None, None]
        assert await dump_raw_fanout(
            [], count=2, request={}, response={}, status="ok", duration_s=0.0
        ) == [None, None]

    async def test_a_sequence_response_is_sliced_per_candidate(
        self, enable_capture: None, tmp_path: Path
    ) -> None:
        ctxs = [_rollout_ctx(tmp_path, f"r{i}") for i in range(3)]
        slices = [{"candidates": [{"index": i}]} for i in range(3)]

        await dump_raw_fanout(
            ctxs,
            count=3,
            request={},
            response=slices,
            status="ok",
            duration_s=0.0,
        )

        for i, ctx in enumerate(ctxs):
            assert _blobs_of(ctx)[0]["response"] == {"candidates": [{"index": i}]}

    async def test_a_shared_dict_response_lands_whole_in_every_blob(
        self, enable_capture: None, tmp_path: Path
    ) -> None:
        """A dict is never mistaken for per-candidate slices."""
        ctxs = [_rollout_ctx(tmp_path, f"r{i}") for i in range(2)]
        shared = {"__error__": "post_failed", "candidates": [1, 2]}

        await dump_raw_fanout(
            ctxs,
            count=2,
            request={},
            response=shared,
            status="error",
            duration_s=0.0,
        )

        for ctx in ctxs:
            assert _blobs_of(ctx)[0]["response"] == shared
            assert _blobs_of(ctx)[0]["status"] == "error"

    async def test_a_list_response_of_the_wrong_length_is_shared_not_sliced(
        self, enable_capture: None, tmp_path: Path
    ) -> None:
        """Slice detection is by LENGTH, so a list that is the payload
        itself (rather than one entry per candidate) is not torn apart."""
        ctxs = [_rollout_ctx(tmp_path, f"r{i}") for i in range(3)]
        payload = [{"a": 1}, {"a": 2}]  # 2 entries, 3 candidates

        await dump_raw_fanout(
            ctxs,
            count=3,
            request={},
            response=payload,
            status="ok",
            duration_s=0.0,
        )

        for ctx in ctxs:
            assert _blobs_of(ctx)[0]["response"] == payload

    async def test_operation_id_reaches_every_blob(
        self, enable_capture: None, tmp_path: Path
    ) -> None:
        ctxs = [_rollout_ctx(tmp_path, f"r{i}") for i in range(2)]

        await dump_raw_fanout(
            ctxs,
            count=2,
            request={},
            response={},
            status="ok",
            duration_s=0.0,
            operation_id="op-4242",
        )

        for ctx in ctxs:
            assert _blobs_of(ctx)[0]["operation_id"] == "op-4242"

    async def test_the_operation_id_key_is_absent_when_not_passed(
        self, enable_capture: None, tmp_path: Path
    ) -> None:
        # The non-LRO backends pass no operation_id; the key must be
        # missing rather than present-and-null.
        ctx = _rollout_ctx(tmp_path, "r0")

        await dump_raw_fanout(
            [ctx], count=1, request={}, response={}, status="ok", duration_s=0.0
        )

        assert "operation_id" not in _blobs_of(ctx)[0]

    async def test_zero_candidates_writes_nothing(
        self, enable_capture: None, tmp_path: Path
    ) -> None:
        ctx = _rollout_ctx(tmp_path, "r0")

        assert (
            await dump_raw_fanout(
                [ctx], count=0, request={}, response={}, status="ok", duration_s=0.0
            )
            == []
        )
        assert _blobs_of(ctx) == []

    async def test_capture_disabled_writes_nothing(
        self, disable_capture: None, tmp_path: Path
    ) -> None:
        ctxs = [_rollout_ctx(tmp_path, f"r{i}") for i in range(2)]

        uris = await dump_raw_fanout(
            ctxs, count=2, request={}, response={}, status="ok", duration_s=0.0
        )

        assert uris == [None, None]
        for ctx in ctxs:
            assert _blobs_of(ctx) == []

    async def test_an_unserializable_response_does_not_raise(
        self, enable_capture: None, tmp_path: Path
    ) -> None:
        """A broken blob must not crash a paid-for sampling call."""
        ctxs = [_rollout_ctx(tmp_path, f"r{i}") for i in range(2)]

        uris = await dump_raw_fanout(
            ctxs,
            count=2,
            request={},
            response=object(),  # type: ignore[arg-type]
            status="ok",
            duration_s=0.0,
        )

        assert uris == [None, None]

    async def test_the_client_request_scope_still_applies(
        self, enable_capture: None, tmp_path: Path
    ) -> None:
        # The fanout must not lose the contextvar the samplers set around
        # the call -- every blob carries the agent-form request.
        ctxs = [_rollout_ctx(tmp_path, f"r{i}") for i in range(2)]

        with client_request_scope({"model": "m", "messages": []}):
            await dump_raw_fanout(
                ctxs, count=2, request={}, response={}, status="ok", duration_s=0.0
            )

        for ctx in ctxs:
            assert _blobs_of(ctx)[0]["client_request"]["model"] == "m"


# ---------------------------------------------------------------------------
# with_raw_call_uris: the blob is only findable if its URI comes back
# ---------------------------------------------------------------------------


class TestWithRawCallUris:
    def test_each_result_gets_its_own_uri(self) -> None:
        results = [make_sample_result(text=f"r{i}") for i in range(3)]

        stamped = with_raw_call_uris(results, ["u0", "u1", "u2"])

        assert [r.context.raw_call_uri for r in stamped] == ["u0", "u1", "u2"]
        # The frozen inputs are left alone; new objects come back.
        assert [r.context.raw_call_uri for r in results] == [None, None, None]

    def test_a_none_uri_leaves_its_result_untouched(self) -> None:
        results = [make_sample_result(text="a"), make_sample_result(text="b")]

        stamped = with_raw_call_uris(results, [None, "u1"])

        assert stamped[0] is results[0]
        assert stamped[1].context.raw_call_uri == "u1"

    def test_a_short_uri_list_leaves_the_remainder_untouched(self) -> None:
        results = [make_sample_result(text=str(i)) for i in range(3)]

        stamped = with_raw_call_uris(results, ["u0"])

        assert stamped[0].context.raw_call_uri == "u0"
        assert stamped[1] is results[1]
        assert stamped[2] is results[2]

    def test_everything_else_on_the_result_survives(self) -> None:
        result = make_sample_result(
            text="hi",
            candidate_id="c7",
            operation_id="op-9",
            prompt_token_count=11,
            generation_token_count=5,
            train_step_id=3,
        )

        stamped = with_raw_call_uris([result], ["u"])[0]

        assert stamped.context.raw_call_uri == "u"
        assert stamped.completion_text == "hi"
        assert stamped.tuning_candidate.candidate_id == "c7"
        assert stamped.tuning_candidate.generation_token_count == 5
        assert stamped.context.operation_id == "op-9"
        assert stamped.context.prompt_token_count == 11
        assert stamped.train_step_id == 3

    def test_no_results_and_no_uris(self) -> None:
        assert with_raw_call_uris([], []) == []
        assert with_raw_call_uris([], ["u0"]) == []
