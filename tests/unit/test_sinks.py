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

"""Tests for the sink layer (src/core/sinks.py).

Recipe-specific spec-registration coverage (bare vs eval-prefixed recipe
emits reaching the ML-platform sink) lives alongside each recipe's own
tree; this file covers the framework sink behavior.
"""

import json
import os

import pytest

from core.metric_spec import AggregationStrategy, MetricSpec


def _spec(name: str, *, emit_to_ml_platform: bool = False) -> MetricSpec:
    """Build a MetricSpec for tests without touching the real registry."""
    return MetricSpec(
        name=name,
        strategy=AggregationStrategy.MEAN,
        doc="test doc",
        emit_to_ml_platform=emit_to_ml_platform,
    )


class TestScalarHelper:
    def test_int_and_float_are_scalar(self):
        from core.sinks import _is_scalar

        assert _is_scalar(3) is True
        assert _is_scalar(3.5) is True

    def test_bool_is_not_scalar(self):
        # bool is an int subclass, but a metric value of True/False is a
        # flag, not a numeric series — VE drops it, so treat it as non-scalar.
        from core.sinks import _is_scalar

        assert _is_scalar(True) is False
        assert _is_scalar(False) is False

    def test_str_dict_list_none_are_not_scalar(self):
        from core.sinks import _is_scalar

        assert _is_scalar("hello") is False
        assert _is_scalar({"a": 1}) is False
        assert _is_scalar([1, 2]) is False
        assert _is_scalar(None) is False

    def test_numpy_scalars_are_scalar(self):
        # np.int64 is NOT a subclass of Python int (verified). Without the
        # numbers.Real-based predicate, np.int64 counts would silently miss
        # VE / console / jsonl and land trace-only. Common producers:
        # len(...) returning a numpy shape, argmax(), np.sum(bool_array).
        import numpy as np

        from core.sinks import _is_scalar

        assert _is_scalar(np.int64(5)) is True
        assert _is_scalar(np.int32(5)) is True
        assert _is_scalar(np.float64(5.0)) is True
        assert _is_scalar(np.float32(5.0)) is True
        # numpy bools remain non-scalar (both bool and np.bool_ mean "flag").
        assert _is_scalar(np.bool_(True)) is False


class TestSinkProtocol:
    def test_protocol_is_runtime_checkable(self):
        """A plain class with all four Sink methods satisfies the Protocol."""
        from core.sinks import Sink

        class _FakeSink:
            def should_accept(self, name, value, spec):
                return True

            def write(self, metrics, step):
                pass

            def log_hparams(self, config):
                pass

            def close(self):
                pass

        assert isinstance(_FakeSink(), Sink)

    def test_missing_method_fails_protocol_check(self):
        from core.sinks import Sink

        class _NotASink:
            def should_accept(self, name, value, spec):
                return True

            # no write, no log_hparams, no close

        assert not isinstance(_NotASink(), Sink)


class _FakeBackend:
    """Captures write/close/hparams calls in place of a real utils.logger.Logger."""

    def __init__(self):
        self.writes: list[tuple[dict, int | None]] = []
        self.hparams = None
        self.closed = False

    def log_metrics(self, metrics, step=None):
        self.writes.append((dict(metrics), step))

    def log_hparams(self, config):
        self.hparams = config

    def close(self):
        self.closed = True


class TestJsonlSink:
    def test_accepts_every_metric_regardless_of_spec_or_type(self):
        from core.sinks import JsonlSink

        sink = JsonlSink(_FakeBackend())
        # scalar with a spec
        assert (
            sink.should_accept(
                "train/train_step_seconds", 0.5, _spec("train/train_step_seconds")
            )
            is True
        )
        # scalar without a spec
        assert sink.should_accept("recipe/foo", 1.0, None) is True
        # non-scalar payload
        assert sink.should_accept("trace/answer", {"a": 1}, None) is True
        # bool
        assert sink.should_accept("flag/x", True, None) is True

    def test_write_forwards_to_backend(self):
        from core.sinks import JsonlSink

        backend = _FakeBackend()
        sink = JsonlSink(backend)
        sink.write({"train/train_step_seconds": 0.5, "eval/acc": 0.9}, step=7)
        assert backend.writes == [
            ({"train/train_step_seconds": 0.5, "eval/acc": 0.9}, 7)
        ]

    def test_close_forwards_to_backend(self):
        from core.sinks import JsonlSink

        backend = _FakeBackend()
        sink = JsonlSink(backend)
        sink.close()
        assert backend.closed is True


class TestMLPlatformSink:
    def test_accepts_emit_to_ml_platform_scalar_with_shallow_name(self):
        from core.sinks import MLPlatformSink

        sink = MLPlatformSink(_FakeBackend())
        spec = _spec("train/reward/mean", emit_to_ml_platform=True)  # 3 segments, ok
        assert sink.should_accept("train/reward/mean", 0.5, spec) is True

    def test_rejects_when_emit_to_ml_platform_false(self):
        from core.sinks import MLPlatformSink

        sink = MLPlatformSink(_FakeBackend())
        spec = _spec("sampler/policy_step", emit_to_ml_platform=False)
        assert sink.should_accept("sampler/policy_step", 42.0, spec) is False

    def test_rejects_non_scalar_even_if_emit_to_ml_platform(self):
        from core.sinks import MLPlatformSink

        sink = MLPlatformSink(_FakeBackend())
        spec = _spec("train/payload", emit_to_ml_platform=True)
        assert sink.should_accept("train/payload", {"x": 1}, spec) is False

    def test_rejects_bool_even_if_emit_to_ml_platform(self):
        from core.sinks import MLPlatformSink

        sink = MLPlatformSink(_FakeBackend())
        spec = _spec("train/flag", emit_to_ml_platform=True)
        assert sink.should_accept("train/flag", True, spec) is False

    def test_rejects_unregistered_metric_spec_none(self):
        from core.sinks import MLPlatformSink

        sink = MLPlatformSink(_FakeBackend())
        # No spec => can't confirm VE-eligibility => reject (jsonl catches it).
        assert sink.should_accept("recipe/foo/bar", 1.0, None) is False

    def test_accepts_four_segment_train_pipeline_name(self):
        """4-segment VE metrics (train/pipeline/buffer_utilization etc.)
        are explicitly allowed."""
        from core.sinks import MLPlatformSink

        sink = MLPlatformSink(_FakeBackend())
        spec = _spec("train/pipeline/buffer_utilization", emit_to_ml_platform=True)
        assert (
            sink.should_accept("train/pipeline/buffer_utilization", 0.8, spec) is True
        )

    def test_accepts_five_and_six_segment_names(self):
        """Vertex Experiments stores 5- and 6-segment names verbatim, so
        per-recipe names such as ``train/recipe/<r>/hacking/<label>/score``
        must reach the tracker."""
        from core.sinks import MLPlatformSink

        sink = MLPlatformSink(_FakeBackend())
        for name in (
            "train/recipe/word_puzzle/hacking/flag_rate",
            "train/recipe/word_puzzle/hacking/answer_leak/score",
            "eval/recipe/word_puzzle/hacking/answer_leak/evaluated_rate",
        ):
            spec = _spec(name, emit_to_ml_platform=True)
            assert sink.should_accept(name, 0.5, spec) is True, name

    def test_rejects_and_warns_on_too_deep_name(self, caplog):
        from core.sinks import _MAX_ML_PLATFORM_SEGMENTS, MLPlatformSink

        sink = MLPlatformSink(_FakeBackend())
        # One over the sanity cap.
        name = "/".join(["seg"] * (_MAX_ML_PLATFORM_SEGMENTS + 1))
        spec = _spec(name, emit_to_ml_platform=True)
        with caplog.at_level("WARNING"):
            accepted = sink.should_accept(name, 3.0, spec)
        assert accepted is False
        # Assert on invariants that survive copy edits: the offending
        # metric name appears in the warning, and the warning fires at
        # WARNING level. The exact phrasing of the message (segment
        # count, "too deep" wording, etc.) is not pinned.
        assert name in caplog.text
        assert any(r.levelname == "WARNING" for r in caplog.records)

    def test_write_forwards_only_given_subset(self):
        from core.sinks import MLPlatformSink

        backend = _FakeBackend()
        sink = MLPlatformSink(backend)
        sink.write({"train/reward/mean": 0.5}, step=3)
        assert backend.writes == [({"train/reward/mean": 0.5}, 3)]


class TestConsoleSink:
    def test_accepts_scalar_under_known_prefix(self):
        from core.sinks import ConsoleSink

        sink = ConsoleSink(_FakeBackend())
        for name in [
            "learning/reward/mean",
            "sampler/policy_step",
            "pipeline/buffer_utilization",
            "eval/accuracy",
        ]:
            assert sink.should_accept(name, 1.0, _spec(name)) is True, name

    def test_rejects_scalar_under_unknown_prefix(self):
        from core.sinks import ConsoleSink

        sink = ConsoleSink(_FakeBackend())
        # train/* is a VE framing; on the console the same data appears under
        # learning/* / pipeline/*. A bare train/* key is not consoled.
        assert (
            sink.should_accept(
                "train/train_step_seconds", 0.5, _spec("train/train_step_seconds")
            )
            is False
        )
        assert sink.should_accept("audit/x", 1.0, _spec("audit/x")) is False

    def test_rejects_non_scalar(self):
        from core.sinks import ConsoleSink

        sink = ConsoleSink(_FakeBackend())
        assert (
            sink.should_accept("sampler/payload", {"a": 1}, _spec("sampler/payload"))
            is False
        )

    def test_accepts_known_prefix_even_without_spec(self):
        from core.sinks import ConsoleSink

        # Console policy is prefix-based, not spec-based — an unregistered
        # pipeline/* gauge should still show on the console.
        sink = ConsoleSink(_FakeBackend())
        assert sink.should_accept("pipeline/queue_depth", 12.0, None) is True

    def test_write_forwards_to_backend(self):
        from core.sinks import ConsoleSink

        backend = _FakeBackend()
        sink = ConsoleSink(backend)
        sink.write({"learning/reward/mean": 0.5}, step=2)
        assert backend.writes == [({"learning/reward/mean": 0.5}, 2)]

    def test_emit_live_gauge_is_a_noop_stub(self):
        # Async live-gauge loop is deferred (decision D4). The hook exists so
        # producers can call it today; it currently forwards a single sample.
        from core.sinks import ConsoleSink

        backend = _FakeBackend()
        sink = ConsoleSink(backend)
        sink.emit_live_gauge("pipeline/buffer_utilization", 0.8)
        assert backend.writes == [({"pipeline/buffer_utilization": 0.8}, None)]


class TestTraceLogSink:
    def test_accepts_non_scalar_payloads(self):
        from core.sinks import TraceLogSink

        sink = TraceLogSink(_FakeBackend())
        assert sink.should_accept("trace/answer", {"parsed": "x"}, None) is True
        assert sink.should_accept("trace/sections", ["a", "b"], None) is True
        assert sink.should_accept("trace/note", "free text", None) is True

    def test_rejects_scalars(self):
        from core.sinks import TraceLogSink

        sink = TraceLogSink(_FakeBackend())
        assert (
            sink.should_accept(
                "train/train_step_seconds", 0.5, _spec("train/train_step_seconds")
            )
            is False
        )
        assert sink.should_accept("flag/x", True, None) is False

    def test_write_forwards_to_backend(self):
        from core.sinks import TraceLogSink

        backend = _FakeBackend()
        sink = TraceLogSink(backend)
        payload = {"trace/answer": {"parsed": "foo", "ground_truth": "bar"}}
        sink.write(payload, step=5)
        assert backend.writes == [(payload, 5)]


class TestTraceLogWriter:
    def test_appends_json_line_with_step(self, tmp_path):
        from core.sinks import TraceLogWriter

        writer = TraceLogWriter(tmp_path)
        writer.log_metrics({"trace/answer": {"parsed": "x"}}, step=3)
        lines = (tmp_path / "trace.jsonl").read_text().splitlines()
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec == {"step": 3, "trace/answer": {"parsed": "x"}}

    def test_non_serializable_degrades_not_crashes(self, tmp_path):
        from core.sinks import TraceLogWriter

        class _Weird:
            pass

        writer = TraceLogWriter(tmp_path)
        # Must not raise, even on a non-JSON-serializable payload.
        writer.log_metrics({"trace/obj": {"x": _Weird()}}, step=1)
        line = (tmp_path / "trace.jsonl").read_text().splitlines()[0]
        rec = json.loads(line)
        assert rec["step"] == 1
        assert "trace/obj" in rec
        # Fallback SHAPE assertion: the payload must round-trip as
        # ``str(type(o))`` (the _TracePermissiveEncoder contract). A
        # regression that silently swapped to repr(o) or dropped the key
        # would still satisfy the "no crash + key present" assertions
        # above; this locks in the actual shape.
        assert rec["trace/obj"] == {"x": str(_Weird)}


class TestSinkDispatcher:
    def test_routes_each_metric_to_accepting_sinks_only(self):
        from core.sinks import (
            ConsoleSink,
            JsonlSink,
            MLPlatformSink,
            SinkDispatcher,
        )

        ve_backend = _FakeBackend()
        jsonl_backend = _FakeBackend()
        console_backend = _FakeBackend()
        dispatcher = SinkDispatcher(
            sinks=[
                MLPlatformSink(ve_backend),
                JsonlSink(jsonl_backend),
                ConsoleSink(console_backend),
            ],
            spec_lookup=_fake_spec_lookup(
                {
                    "train/reward/mean": _spec(
                        "train/reward/mean", emit_to_ml_platform=True
                    ),
                    "learning/reward/mean": _spec("learning/reward/mean"),
                }
            ),
        )
        dispatcher.log_metrics(
            {"train/reward/mean": 0.5, "learning/reward/mean": 0.5}, step=1
        )
        # VE only gets the emit_to_ml_platform train/* metric.
        assert ve_backend.writes == [({"train/reward/mean": 0.5}, 1)]
        # jsonl gets both.
        assert jsonl_backend.writes == [
            ({"train/reward/mean": 0.5, "learning/reward/mean": 0.5}, 1)
        ]
        # console gets only the learning/* one (train/* is not a console prefix).
        assert console_backend.writes == [({"learning/reward/mean": 0.5}, 1)]

    def test_unregistered_metric_defaults_to_jsonl_only_with_warning(self, caplog):
        from core.sinks import JsonlSink, MLPlatformSink, SinkDispatcher

        ve_backend = _FakeBackend()
        jsonl_backend = _FakeBackend()
        dispatcher = SinkDispatcher(
            sinks=[MLPlatformSink(ve_backend), JsonlSink(jsonl_backend)],
            spec_lookup=_fake_spec_lookup({}),  # nothing registered
        )
        with caplog.at_level("WARNING"):
            dispatcher.log_metrics({"recipe/foo/bar": 1.0}, step=2)
        assert ve_backend.writes == []  # no spec => VE rejects
        assert jsonl_backend.writes == [({"recipe/foo/bar": 1.0}, 2)]
        assert "recipe/foo/bar" in caplog.text

    def test_unregistered_warning_deduplicated(self, caplog):
        from core.sinks import JsonlSink, SinkDispatcher

        dispatcher = SinkDispatcher(
            sinks=[JsonlSink(_FakeBackend())],
            spec_lookup=_fake_spec_lookup({}),
        )
        with caplog.at_level("WARNING"):
            dispatcher.log_metrics({"recipe/foo": 1.0}, step=1)
            dispatcher.log_metrics({"recipe/foo": 2.0}, step=2)
        # Same key warned only once.
        assert caplog.text.count("recipe/foo") == 1

    def test_unregistered_nonscalar_goes_to_jsonl_and_trace(self):
        from core.sinks import JsonlSink, SinkDispatcher, TraceLogSink

        jsonl_backend = _FakeBackend()
        trace_backend = _FakeBackend()
        dispatcher = SinkDispatcher(
            sinks=[JsonlSink(jsonl_backend), TraceLogSink(trace_backend)],
            spec_lookup=_fake_spec_lookup({}),
        )
        dispatcher.log_metrics({"trace/answer": {"a": 1}}, step=1)
        assert jsonl_backend.writes == [({"trace/answer": {"a": 1}}, 1)]
        assert trace_backend.writes == [({"trace/answer": {"a": 1}}, 1)]

    def test_close_fans_out(self):
        from core.sinks import JsonlSink, SinkDispatcher

        b1, b2 = _FakeBackend(), _FakeBackend()
        dispatcher = SinkDispatcher(
            sinks=[JsonlSink(b1), JsonlSink(b2)],
            spec_lookup=_fake_spec_lookup({}),
        )
        dispatcher.close()
        assert b1.closed and b2.closed

    def test_close_continues_when_a_sink_raises(self, caplog):
        """H-2 regression fence: a raising sink.close() must not prevent
        subsequent sinks from being closed. Exception is logged, not raised."""
        from core.sinks import JsonlSink, SinkDispatcher

        class _RaisingSink:
            def should_accept(self, name, value, spec):
                return False

            def write(self, metrics, step):
                pass

            def close(self):
                raise RuntimeError("simulated VE end_experiment_run failure")

        good_backend = _FakeBackend()
        dispatcher = SinkDispatcher(
            sinks=[_RaisingSink(), JsonlSink(good_backend)],
            spec_lookup=_fake_spec_lookup({}),
        )
        with caplog.at_level("ERROR"):
            dispatcher.close()  # must not raise
        assert good_backend.closed is True
        assert "failed to close cleanly" in caplog.text
        assert "simulated VE end_experiment_run failure" in caplog.text

    def test_empty_metrics_is_a_noop(self):
        from core.sinks import JsonlSink, SinkDispatcher

        b = _FakeBackend()
        dispatcher = SinkDispatcher(
            sinks=[JsonlSink(b)], spec_lookup=_fake_spec_lookup({})
        )
        dispatcher.log_metrics({}, step=1)
        assert b.writes == []

    def test_log_hparams_fans_out_to_forwarding_sinks(self):
        """N-3 regression fence: dispatcher.log_hparams reaches every sink,
        forwarding sinks call their backend.log_hparams, TraceLogSink is a
        no-op. Pins the Sink Protocol's log_hparams contract."""
        from core.sinks import (
            ConsoleSink,
            JsonlSink,
            SinkDispatcher,
            TraceLogSink,
        )

        jsonl_backend = _FakeBackend()
        console_backend = _FakeBackend()
        trace_backend = _FakeBackend()
        dispatcher = SinkDispatcher(
            sinks=[
                JsonlSink(jsonl_backend),
                ConsoleSink(console_backend),
                TraceLogSink(trace_backend),
            ],
            spec_lookup=_fake_spec_lookup({}),
        )
        config = {"lr": 0.01, "epochs": 10}
        dispatcher.log_hparams(config)
        # Forwarding sinks pass through to backend.
        assert jsonl_backend.hparams == config
        assert console_backend.hparams == config
        # TraceLogSink.log_hparams is deliberately a no-op (trace is
        # per-step payloads, not config). Backend must be untouched.
        assert trace_backend.hparams is None


def _fake_spec_lookup(registry: dict):
    """Return a spec_for-compatible callable raising KeyError on a miss."""

    def _lookup(name: str):
        try:
            return registry[name]
        except KeyError:
            raise KeyError(f"no MetricSpec registered for metric {name!r}")

    return _lookup


class TestSetupLoggingReturnsDispatcher:
    def test_setup_logging_returns_sink_dispatcher(self, tmp_path):
        from core.sinks import SinkDispatcher
        from utils.logger import setup_logging

        ml_logger = setup_logging(str(tmp_path), do_configure_logging_module=False)
        assert isinstance(ml_logger, SinkDispatcher)

    def test_dispatcher_routes_to_jsonl_and_trace_backends(self, tmp_path):
        """End-to-end: a scalar lands in metrics.jsonl; a non-scalar lands in
        trace.jsonl; both files are created by the real backends."""
        import json

        from utils.logger import setup_logging

        ml_logger = setup_logging(str(tmp_path), do_configure_logging_module=False)
        # A registered VE-eligible scalar + a non-scalar payload.
        ml_logger.log_metrics(
            {"train/reward/mean": 0.5, "trace/answer": {"parsed": "x"}}, step=1
        )

        metrics_lines = (tmp_path / "metrics.jsonl").read_text().splitlines()
        metrics_rec = json.loads(metrics_lines[-1])
        # jsonl (everything archive) has BOTH.
        assert metrics_rec["train/reward/mean"] == 0.5
        assert metrics_rec["trace/answer"] == {"parsed": "x"}

        trace_lines = (tmp_path / "trace.jsonl").read_text().splitlines()
        trace_rec = json.loads(trace_lines[-1])
        # trace has ONLY the non-scalar.
        assert trace_rec["trace/answer"] == {"parsed": "x"}
        assert "train/reward/mean" not in trace_rec


class TestEvalMetricsRouting:
    def test_eval_metrics_route_to_ve_jsonl_console(self):
        """An eval/* scalar with emit_to_ml_platform reaches VE, jsonl, and console."""
        from core.sinks import (
            ConsoleSink,
            JsonlSink,
            MLPlatformSink,
            SinkDispatcher,
        )

        ve, js, con = _FakeBackend(), _FakeBackend(), _FakeBackend()
        dispatcher = SinkDispatcher(
            sinks=[MLPlatformSink(ve), JsonlSink(js), ConsoleSink(con)],
            spec_lookup=_fake_spec_lookup(
                {"eval/accuracy": _spec("eval/accuracy", emit_to_ml_platform=True)}
            ),
        )
        dispatcher.log_metrics({"eval/accuracy": 0.87}, step=10)
        assert ve.writes == [({"eval/accuracy": 0.87}, 10)]
        assert js.writes == [({"eval/accuracy": 0.87}, 10)]
        assert con.writes == [({"eval/accuracy": 0.87}, 10)]


class TestFourSinkPolicyInvariants:
    """Property tests over the whole dispatcher: the 4-sink policy invariants."""

    def _build(self):
        from core.sinks import (
            ConsoleSink,
            JsonlSink,
            MLPlatformSink,
            SinkDispatcher,
            TraceLogSink,
        )

        self.ve = _FakeBackend()
        self.js = _FakeBackend()
        self.con = _FakeBackend()
        self.tr = _FakeBackend()
        registry = {
            "train/reward/mean": _spec("train/reward/mean", emit_to_ml_platform=True),
            "learning/reward/mean": _spec(
                "learning/reward/mean", emit_to_ml_platform=False
            ),
            "sampler/policy_step": _spec(
                "sampler/policy_step", emit_to_ml_platform=False
            ),
            "pipeline/buffer_utilization": _spec(
                "pipeline/buffer_utilization", emit_to_ml_platform=True
            ),
            "eval/accuracy": _spec("eval/accuracy", emit_to_ml_platform=True),
        }
        return SinkDispatcher(
            sinks=[
                MLPlatformSink(self.ve),
                JsonlSink(self.js),
                ConsoleSink(self.con),
                TraceLogSink(self.tr),
            ],
            spec_lookup=_fake_spec_lookup(registry),
        )

    # A representative metric set covering scalar/non-scalar × emit_to_ml_platform.
    _SAMPLE = {
        "train/reward/mean": 0.5,  # scalar, emit_to_ml_platform -> VE+jsonl (train/ not console)
        "learning/reward/mean": 0.5,  # scalar, no VE -> jsonl+console
        "sampler/policy_step": 42.0,  # scalar, no VE -> jsonl+console
        "pipeline/buffer_utilization": 0.8,  # scalar, VE -> VE+jsonl+console
        "eval/accuracy": 0.9,  # scalar, VE -> VE+jsonl+console
        "trace/answer": {"parsed": "x"},  # non-scalar (unregistered) -> jsonl+trace
        "recipe/unknown": 3.0,  # unregistered scalar -> jsonl only
    }

    def _flatten(self, backend) -> dict:
        merged: dict = {}
        for m, _step in backend.writes:
            merged.update(m)
        return merged

    def test_invariant_1_everything_reaches_jsonl(self):
        d = self._build()
        d.log_metrics(dict(self._SAMPLE), step=1)
        assert set(self._flatten(self.js)) == set(self._SAMPLE)

    def test_invariant_2_ve_only_gets_emit_to_ml_platform(self):
        d = self._build()
        d.log_metrics(dict(self._SAMPLE), step=1)
        assert set(self._flatten(self.ve)) == {
            "train/reward/mean",
            "pipeline/buffer_utilization",
            "eval/accuracy",
        }

    def test_invariant_3_no_nonscalar_on_ve_or_console(self):
        d = self._build()
        d.log_metrics(dict(self._SAMPLE), step=1)
        assert "trace/answer" not in self._flatten(self.ve)
        assert "trace/answer" not in self._flatten(self.con)

    def test_invariant_4_every_nonscalar_reaches_trace(self):
        d = self._build()
        d.log_metrics(dict(self._SAMPLE), step=1)
        assert set(self._flatten(self.tr)) == {"trace/answer"}

    def test_invariant_5_ve_and_console_are_subsets_of_jsonl(self):
        d = self._build()
        d.log_metrics(dict(self._SAMPLE), step=1)
        jsonl_keys = set(self._flatten(self.js))
        assert set(self._flatten(self.ve)) <= jsonl_keys
        assert set(self._flatten(self.con)) <= jsonl_keys


class TestSinkDispatcherWithRealRegistry:
    """End-to-end integration: SinkDispatcher + real spec_for + real
    MLPlatformSink. NO spec_lookup= override.

    Every other test class in this file injects fake specs via
    _fake_spec_lookup(). That misses the class of bugs where production
    key shape doesn't match the registered spec key shape — exactly the
    pre-fix bug where train_and_eval.py:283 emitted 'train/learning/reward/mean'
    but only 'learning/reward/mean' was registered, causing MLPlatformSink
    to silently reject via the spec=None path in should_accept.

    These tests use the REAL METRIC_SPECS registry (via core.metric_spec.spec_for)
    and assert that production-shaped keys survive the full dispatch pipeline
    to MLPlatformSink under the expected forwarded key.
    """

    def _build(self):
        """Build a dispatcher with all four real sinks + fake backends.

        Uses real spec_for (default when spec_lookup= is omitted). Recipe
        registrations are ensured at module load (see the top-of-file
        `import example_recipe.…` side-effects).
        """
        from core.sinks import (
            ConsoleSink,
            JsonlSink,
            MLPlatformSink,
            SinkDispatcher,
            TraceLogSink,
        )

        self.ve = _FakeBackend()
        self.js = _FakeBackend()
        self.con = _FakeBackend()
        self.tr = _FakeBackend()
        return SinkDispatcher(
            [
                MLPlatformSink(self.ve),
                JsonlSink(self.js),
                ConsoleSink(self.con),
                TraceLogSink(self.tr),
            ]
            # NOTE: no spec_lookup= — uses real core.metric_spec.spec_for.
        )

    @staticmethod
    def _flatten(backend):
        keys = set()
        for m, _ in backend.writes:
            keys.update(m.keys())
        return keys

    def test_train_prefixed_pipeline_key_reaches_ml_platform(self):
        """Regression fence for the pre-fix bug: train_and_eval.py:283,293
        emits 'train/pipeline/train_step_time' (via `f"train/{k}"`).
        Pre-fix, this key had no spec, so MLPlatformSink silently
        rejected. Post-fix, the key is registered and MLPlatformSink
        accepts it."""
        d = self._build()
        d.log_metrics({"train/pipeline/train_step_time": 12.5}, step=1)
        assert "train/pipeline/train_step_time" in self._flatten(self.ve), (
            "MLPlatformSink rejected the production emit key — the pre-fix "
            "bug shape has re-appeared. Check core.metric_spec.METRIC_SPECS "
            "for the missing/wrong spec."
        )

    def test_eval_prefixed_learning_key_reaches_ml_platform(self):
        """Site C (eval_only.py:350) emits `f"eval/{k}"`. Post-fix, the
        eval/* shape is registered and MLPlatformSink accepts."""
        d = self._build()
        d.log_metrics({"eval/learning/reward/mean": 0.6}, step=2)
        assert "eval/learning/reward/mean" in self._flatten(self.ve)

    def test_hardcoded_train_key_from_site_b_reaches_ml_platform(self):
        """Site B (sync_rollout_pipeline.py:371-375) emits hard-coded train/*
        keys. Post-fix, they're registered as identity."""
        d = self._build()
        d.log_metrics({"train/running_pass_rate": 0.7}, step=3)
        assert "train/running_pass_rate" in self._flatten(self.ve)

    def test_sampler_policy_step_not_forwarded_to_ml_platform(self):
        """emit_to_ml_platform=False keeps a metric off the ml_platform
        sink. The derived train/staleness_steps replaces it on VE."""
        d = self._build()
        d.log_metrics(
            {"sampler/policy_step": 42.0, "train/sampler/policy_step": 42.0},
            step=4,
        )
        # Both aliases have emit_to_ml_platform=False.
        assert "sampler/policy_step" not in self._flatten(self.ve)
        assert "train/sampler/policy_step" not in self._flatten(self.ve)
        # But both are in jsonl (everything archive).
        assert "sampler/policy_step" in self._flatten(self.js)
        assert "train/sampler/policy_step" in self._flatten(self.js)

    def test_iterator_current_stays_jsonl_only(self):
        """train/pipeline/iterator_current is a resume marker; MUST stay
        jsonl-only per METRIC_SPECS (emit_to_ml_platform=False). Silent
        drift to VE would break resume by mixing the marker into
        aggregated dashboards."""
        d = self._build()
        d.log_metrics({"train/pipeline/iterator_current": 100}, step=5)
        assert "train/pipeline/iterator_current" not in self._flatten(self.ve)
        assert "train/pipeline/iterator_current" in self._flatten(self.js)

    def test_all_train_derived_metrics_reach_ml_platform(self):
        """Regression fence for the full Site A + Site B train-side emit
        set. Every key here would have been silently dropped pre-fix.
        Post-fix, every VE-eligible one reaches the ml_platform sink.

        Uses the actual keys train_and_eval.py:283-302 emits: wire
        metrics wrapped with `train/` (train_step_seconds /
        billable_training_tokens / training_tokens_before_merge /
        mean_abs_advantage — but NOT loss, which is mock-only and
        emit_to_ml_platform=False), pipeline passthroughs wrapped with
        `train/` (train_step_time / wait_for_rollout_time), staleness
        derivation, plus Site B literals (running_pass_rate /
        rollout_rounds / etc)."""
        d = self._build()
        train_side_keys = {
            "train/pipeline/train_step_time": 12.5,
            "train/pipeline/wait_for_rollout_time": 0.5,
            "train/staleness_steps": 3.0,
            "train/train_step_seconds": 1.5,
            "train/billable_training_tokens": 12345.0,
            "train/running_pass_rate": 0.6,
            "train/rollout_rounds": 4.0,
        }
        d.log_metrics(train_side_keys, step=6)
        ve_keys = self._flatten(self.ve)
        for k in train_side_keys:
            assert k in ve_keys, f"{k!r} did not reach MLPlatformSink"

    def test_all_eval_metrics_reach_ml_platform_under_emitted_key(self):
        """Site C emits eval/learning/reward/mean etc. Post-fix all
        VE-eligible eval keys survive dispatch under their emit name."""
        d = self._build()
        eval_side_keys = {
            "eval/learning/reward/mean": 0.5,
            "eval/learning/sampling_accuracy": 0.7,
            "eval/learning/pass_at_k": 0.5,
            "eval/pipeline/sample_time_avg": 1.2,
        }
        d.log_metrics(eval_side_keys, step=7)
        ve_keys = self._flatten(self.ve)
        for k in eval_side_keys:
            assert k in ve_keys, f"{k!r} did not reach MLPlatformSink"


class TestComposeVertexRunName:
    """`compose_vertex_run_name` prepends the tuning job id to the run name
    so a Vertex ExperimentRun is traceable to its job at a glance."""

    def test_prepends_job_id_to_run_name(self):
        from utils.logger import compose_vertex_run_name

        assert (
            compose_vertex_run_name("1234567890123456789", "my-vm-20260101-000000")
            == "1234567890123456789-my-vm-20260101-000000"
        )

    def test_job_id_only_when_run_name_empty(self):
        from utils.logger import compose_vertex_run_name

        assert compose_vertex_run_name("12345", None) == "12345"
        assert compose_vertex_run_name("12345", "") == "12345"

    def test_sanitizes_to_vertex_constraint(self):
        from utils.logger import compose_vertex_run_name

        # Underscores / uppercase / dots are coerced to the
        # ^[a-z0-9][a-z0-9-]{0,127}$ run-name constraint.
        out = compose_vertex_run_name("999", "My_VM.host")
        assert out == "999-my-vm-host"
        import re

        assert re.fullmatch(r"[a-z0-9][a-z0-9-]{0,127}", out)


# ---------------------------------------------------------------------------
# WandbLogger tests
# ---------------------------------------------------------------------------

import types
from unittest.mock import MagicMock


@pytest.fixture
def mock_wandb(monkeypatch):
    """Mock the ``wandb`` module so tests never touch the network."""
    import sys

    fake = types.ModuleType("wandb")
    fake.init = MagicMock(return_value=MagicMock())
    fake.log = MagicMock()
    fake.finish = MagicMock()
    fake.config = MagicMock()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


class TestWandbLogger:
    def test_init_offline_by_default(self, tmp_path, monkeypatch, mock_wandb):
        monkeypatch.delenv("WANDB_MODE", raising=False)
        from utils.logger import WandbLogger

        WandbLogger(project="proj", run_name="run-1", log_dir=str(tmp_path))
        assert os.environ.get("WANDB_MODE") == "offline"
        mock_wandb.init.assert_called_once()
        kwargs = mock_wandb.init.call_args.kwargs
        assert kwargs["project"] == "proj"
        assert kwargs["name"] == "run-1"
        assert kwargs["mode"] == "offline"

    def test_init_missing_wandb_raises_importerror(self, tmp_path, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "wandb", None)
        from utils.logger import WandbLogger

        with pytest.raises(ImportError, match="wandb_logger"):
            WandbLogger(project="p", run_name="r", log_dir=str(tmp_path))

    def test_log_hparams_flattens_and_updates_config(self, tmp_path, mock_wandb):
        from utils.logger import WandbLogger

        wl = WandbLogger(project="p", run_name="r", log_dir=str(tmp_path))
        wl.log_hparams({"nested": {"lr": 1e-3}, "n": 5})
        wl._run.config.update.assert_called_once()
        sent = wl._run.config.update.call_args.args[0]
        assert sent["nested.lr"] == 1e-3
        assert sent["n"] == 5
        # idempotent
        wl.log_hparams({"n": 9})
        wl._run.config.update.assert_called_once()

    def test_log_metrics_numeric_only_with_step(self, tmp_path, mock_wandb):
        from utils.logger import WandbLogger

        wl = WandbLogger(project="p", run_name="r", log_dir=str(tmp_path))
        wl.log_metrics(
            {"train/loss": 0.5, "train/flag": True, "note": "x", "n": 3}, step=7
        )
        wl._run.log.assert_called_once()
        data = wl._run.log.call_args.args[0]
        assert data == {"train/loss": 0.5, "n": 3.0}
        assert wl._run.log.call_args.kwargs["step"] == 7

    def test_log_metrics_empty_after_filter_is_noop(self, tmp_path, mock_wandb):
        from utils.logger import WandbLogger

        wl = WandbLogger(project="p", run_name="r", log_dir=str(tmp_path))
        wl.log_metrics({"only_str": "hi"}, step=1)
        wl._run.log.assert_not_called()

    def test_log_metrics_runtime_error_is_best_effort(self, tmp_path, mock_wandb):
        from utils.logger import WandbLogger

        wl = WandbLogger(project="p", run_name="r", log_dir=str(tmp_path))
        wl._run.log.side_effect = RuntimeError("boom")
        wl.log_metrics({"loss": 1.0}, step=1)  # must NOT raise
        assert wl._failure_count == 1

    def test_close_calls_finish_best_effort(self, tmp_path, mock_wandb):
        from utils.logger import WandbLogger

        wl = WandbLogger(project="p", run_name="r", log_dir=str(tmp_path))
        wl.close()
        wl._run.finish.assert_called_once()
        wl._run.finish.side_effect = RuntimeError("x")
        wl.close()  # no raise


class TestSetupLoggingWandb:
    def test_no_wandb_by_default(self, tmp_path):
        from utils.logger import setup_logging

        ml = setup_logging(str(tmp_path / "logs"), do_configure_logging_module=False)
        # 3 sinks: jsonl, console, trace
        assert len(ml._sinks) == 3

    def test_appends_wandb_when_project_set(self, tmp_path, mock_wandb):
        from core.sinks import MLPlatformSink
        from utils.logger import WandbLogger, setup_logging

        ml = setup_logging(
            str(tmp_path / "logs" / "job" / "0707_120000"),
            config={"a": 1},
            do_configure_logging_module=False,
            wandb_project="my-proj",
        )
        wb_sinks = [
            s
            for s in ml._sinks
            if isinstance(s, MLPlatformSink) and isinstance(s._backend, WandbLogger)
        ]
        assert len(wb_sinks) == 1
        mock_wandb.init.assert_called_once()
        assert mock_wandb.init.call_args.kwargs["name"] == "0707_120000"


class TestCheckpointEvalPrefixRouting:
    """``eval_ckpt/*`` (the inline checkpoint-endpoint sweep prefix) must
    route exactly like its ``eval/*`` twin.

    Regression guard: the prefix is applied in ``EvalPipeline.run`` AFTER
    spec-driven aggregation, so a missing ``eval_ckpt/*`` MetricSpec makes
    ``SinkDispatcher`` resolve ``None`` -> MLPlatformSink/ConsoleSink drop
    the metric and it silently degrades to jsonl-only (never reaching
    Vertex AI Experiments / TensorBoard).
    """

    def test_eval_ckpt_specs_mirror_eval_specs(self):
        from core.metric_spec import METRIC_SPECS, spec_for

        eval_keys = [
            k for k in METRIC_SPECS if k.startswith("eval/") and not k.endswith("/*")
        ]
        assert eval_keys, "expected framework eval/* specs to exist"
        for key in eval_keys:
            src = METRIC_SPECS[key]
            dst = spec_for(key.replace("eval/", "eval_ckpt/", 1))
            assert dst.strategy == src.strategy
            assert dst.emit_to_ml_platform == src.emit_to_ml_platform
            assert dst.name == src.name.replace("eval/", "eval_ckpt/", 1)

        # Concrete lookups (incl. a dynamic-suffix family member) resolve.
        assert spec_for("eval_ckpt/learning/sampling_accuracy").emit_to_ml_platform
        assert spec_for("eval_ckpt/sampler/finish_reason/STOP").emit_to_ml_platform

    def test_eval_ckpt_twin_covers_late_registered_recipe_specs(self):
        """Recipes register their ``eval/*`` specs at recipe-import time, long
        after ``core.metric_spec`` loads. Resolving the ``eval_ckpt/`` twin at
        lookup time (not by pre-deriving at module load) is what keeps recipe
        metrics — e.g. a recipe's own F0.5 — out of the jsonl-only hole.
        """
        from core.metric_spec import (
            METRIC_SPECS,
            AggregationStrategy,
            MetricSpec,
            spec_for,
        )

        key = "eval/recipe/latecomer/score"
        METRIC_SPECS[key] = MetricSpec(
            name=key,
            strategy=AggregationStrategy.MEAN,
            doc="late-registered recipe metric",
            emit_to_ml_platform=True,
        )
        try:
            twin = spec_for("eval_ckpt/recipe/latecomer/score")
            assert twin.name == "eval_ckpt/recipe/latecomer/score"
            assert twin.emit_to_ml_platform
        finally:
            del METRIC_SPECS[key]

    def test_ml_platform_sink_accepts_eval_ckpt_metric(self):
        from core.metric_spec import spec_for
        from core.sinks import MLPlatformSink

        sink = MLPlatformSink(_FakeBackend())
        name = "eval_ckpt/learning/reward/mean"
        # 4 segments (under the MLPlatformSink cap), same shape as eval/* twin.
        assert sink.should_accept(name, 0.5, spec_for(name))

    def test_console_sink_accepts_eval_ckpt_metric(self):
        from core.sinks import ConsoleSink

        sink = ConsoleSink(_FakeBackend())
        assert sink.should_accept("eval_ckpt/learning/reward/mean", 0.5, None)

    def test_dispatcher_routes_eval_ckpt_to_ml_platform(self):
        from core.sinks import JsonlSink, MLPlatformSink, SinkDispatcher

        ml_backend, jsonl_backend = _FakeBackend(), _FakeBackend()
        dispatcher = SinkDispatcher(
            [JsonlSink(jsonl_backend), MLPlatformSink(ml_backend)]
        )
        dispatcher.log_metrics({"eval_ckpt/learning/reward/mean": 0.75}, step=30)

        assert ml_backend.writes, (
            "eval_ckpt/* must reach the ml_platform backend, not jsonl only"
        )
        logged, step = ml_backend.writes[-1]
        assert logged == {"eval_ckpt/learning/reward/mean": 0.75}
        assert step == 30


class TestVertexExperimentLoggerClose:
    """``close()`` must be bounded: ``aiplatform.end_run()`` issues a
    deadline-less gRPC UpdateContext, and when it wedges it used to block
    the teardown ``finally`` forever — SIGTERM never completed and the
    run's terminal status was never written.
    """

    @staticmethod
    def _logger_with_end_run(end_run):
        """Build a VertexExperimentLogger without running __init__.

        __init__ imports aiplatform and performs init()/start_run() network
        calls; close() only needs ``_aiplatform``, ``_run_name`` and
        ``_failure_count``.
        """
        from unittest.mock import MagicMock

        from utils.logger import VertexExperimentLogger

        vl = object.__new__(VertexExperimentLogger)
        vl._aiplatform = MagicMock()
        vl._aiplatform.end_run = end_run
        vl._run_name = "run-1"
        vl._failure_count = 0
        return vl

    def test_close_calls_end_run(self):
        from unittest.mock import MagicMock

        end_run = MagicMock()
        vl = self._logger_with_end_run(end_run)
        vl.close()
        end_run.assert_called_once_with()

    def test_close_swallows_end_run_error(self):
        from unittest.mock import MagicMock

        end_run = MagicMock(side_effect=RuntimeError("boom"))
        vl = self._logger_with_end_run(end_run)
        vl.close()  # must NOT raise
        assert vl._failure_count == 1

    def test_close_returns_when_end_run_blocks(self, caplog):
        """The regression: a hung end_run() must not hang teardown."""
        import logging
        import threading
        import time

        release = threading.Event()
        entered = threading.Event()

        def blocking_end_run():
            entered.set()
            release.wait(30)

        vl = self._logger_with_end_run(blocking_end_run)
        vl._CLOSE_TIMEOUT_SECONDS = 0.2
        try:
            with caplog.at_level(logging.WARNING):
                start = time.monotonic()
                vl.close()
                elapsed = time.monotonic() - start

            assert entered.is_set(), "end_run() was never invoked"
            # Bounded by the timeout, not by the blocked call.
            assert elapsed < 5.0, f"close() took {elapsed:.1f}s — not bounded"
            assert "did not return within" in caplog.text
        finally:
            release.set()

    def test_close_worker_thread_is_daemon(self):
        """A still-blocked end_run() must not keep the interpreter alive."""
        import threading

        seen: list[threading.Thread] = []

        def record_thread():
            seen.append(threading.current_thread())

        vl = self._logger_with_end_run(record_thread)
        vl.close()
        assert len(seen) == 1
        assert seen[0].daemon is True
