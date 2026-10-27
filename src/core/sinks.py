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

"""Sink layer: four sinks with distinct accept-predicates.

Each ``Sink`` is a policy object owning a ``should_accept`` predicate,
forwarding accepted metrics to a backend ``Logger``. The
``SinkDispatcher`` fans a metric dict out to sinks that accept it,
consulting ``MetricSpec`` via ``core.metric_spec.spec_for``.

Pure policy + thin backend delegation; the actual I/O backends live
in ``utils.logger``.
"""

from __future__ import annotations

import json
import logging
import numbers
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from core.metric_spec import MetricSpec

logger = logging.getLogger(__name__)


def _is_scalar(value: Any) -> bool:
    """True iff ``value`` is a numeric metric value.

    Uses ``numbers.Real`` (not ``isinstance(value, (int, float))``) so
    numpy scalars (``np.int64``, ``np.float64``, ...) — common outputs
    of ``len(...)``, ``np.sum(...)``, etc. — are recognized. Excludes
    ``bool`` (a flag, not a time series): the ml_platform sink drops
    it and the trace sink treats it as a non-scalar payload.
    """
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


@runtime_checkable
class Sink(Protocol):
    """One metrics sink: a policy + a backend it forwards accepted metrics to."""

    def should_accept(self, name: str, value: Any, spec: MetricSpec | None) -> bool:
        """Return True iff this sink should write ``name`` this step.

        ``spec`` is the metric's ``MetricSpec`` or ``None`` if the metric
        is not registered (see ``SinkDispatcher``). Predicates that need
        the spec must handle ``None`` gracefully.
        """
        ...

    def write(self, metrics: dict[str, Any], step: int | None) -> None:
        """Write the already-filtered ``metrics`` subset for ``step``."""
        ...

    def log_hparams(self, config: Any) -> None:
        """Log hyperparameters once at the start of a run. Sinks whose
        backend has no hparams surface should implement as an explicit
        no-op (not silently via ``getattr``)."""
        ...

    def close(self) -> None:
        """Flush / release the backend."""
        ...


class JsonlSink:
    """The complete archive: accepts every metric, drops nothing.

    Wraps a backend ``Logger`` appending ``{"step": step, **metrics}``
    lines to ``metrics.jsonl``. Predicate is unconditionally ``True``.
    """

    def __init__(self, backend: Any):
        self._backend = backend

    def should_accept(self, name: str, value: Any, spec: MetricSpec | None) -> bool:
        return True

    def write(self, metrics: dict[str, Any], step: int | None) -> None:
        if not metrics:
            return
        self._backend.log_metrics(metrics, step=step)

    def log_hparams(self, config: Any) -> None:
        """Forward hparams to the backend (writes ``config.json``)."""
        self._backend.log_hparams(config)

    def close(self) -> None:
        close = getattr(self._backend, "close", None)
        if callable(close):
            close()


# Sanity cap on "/"-separated metric-name segments. Vertex Experiments
# stores names up to at least 10 segments verbatim: a probe run
# (tracker-cap-probe-8-10seg-20261008-223951) logged 8-, 9- and 10-segment
# names, e.g. ``train/<group>/<c>/recipe/<r>/hacking/<label>/flag_rate``, and
# read every series back with both steps. The old cap of 4 was dropping real
# series such as ``train/recipe/<r>/hacking/flag_rate``. 10 covers the deepest
# names emitted today (8: per-component copies of per-label series) while
# still catching a runaway f-string.
_MAX_ML_PLATFORM_SEGMENTS = 10


class MLPlatformSink:
    """Curated, cross-run-comparable metrics for an experiment-tracker
    backend.

    Backend-agnostic — anything with a ``log_metrics`` / ``log_hparams``
    surface fits.

    Accept iff: value is a numeric scalar AND the metric has a
    registered ``MetricSpec`` with ``emit_to_ml_platform=True`` AND the
    name has at most ``_MAX_ML_PLATFORM_SEGMENTS`` (10) segments.

    Over-deep ``emit_to_ml_platform`` metrics are REJECTED with a
    warning, never truncated (truncation would collide distinct metrics
    onto one series and silently lose data).
    """

    def __init__(self, backend: Any):
        self._backend = backend

    def should_accept(self, name: str, value: Any, spec: MetricSpec | None) -> bool:
        if not _is_scalar(value):
            return False
        if spec is None or not spec.emit_to_ml_platform:
            return False
        if name.count("/") + 1 > _MAX_ML_PLATFORM_SEGMENTS:
            logger.warning(
                "MLPlatformSink: rejecting %r — %d segments exceeds the "
                "sanity cap of %d. Fix the metric name (collapse "
                "sub-prefixes into a compound leaf) or set "
                "emit_to_ml_platform=False.",
                name,
                name.count("/") + 1,
                _MAX_ML_PLATFORM_SEGMENTS,
            )
            return False
        return True

    def write(self, metrics: dict[str, Any], step: int | None) -> None:
        if not metrics:
            return
        self._backend.log_metrics(metrics, step=step)

    def log_hparams(self, config: Any) -> None:
        """Forward hparams to the backend (typically ``log_params``)."""
        self._backend.log_hparams(config)

    def close(self) -> None:
        close = getattr(self._backend, "close", None)
        if callable(close):
            close()


_CONSOLE_PREFIXES = ("learning/", "sampler/", "pipeline/", "eval/", "eval_ckpt/")


class ConsoleSink:
    """The live progress console: a curated scalar subset.

    Accept iff the value is a numeric scalar AND the name starts with
    one of the categorical-by-intent console prefixes (``learning/``,
    ``sampler/``, ``pipeline/``, ``eval/``). Prefix-based (not
    spec-based) so an unregistered ``pipeline/*`` gauge still shows.

    ``emit_live_gauge`` is a forward-compatible hook (future replacement
    with a live-region update); today it writes a single sample.
    """

    def __init__(self, backend: Any):
        self._backend = backend

    def should_accept(self, name: str, value: Any, spec: MetricSpec | None) -> bool:
        if not _is_scalar(value):
            return False
        return name.startswith(_CONSOLE_PREFIXES)

    def write(self, metrics: dict[str, Any], step: int | None) -> None:
        if not metrics:
            return
        self._backend.log_metrics(metrics, step=step)

    def emit_live_gauge(self, name: str, value: float) -> None:
        """Emit a single wall-clock gauge sample (forward-compatible
        stub for a future live-region update)."""
        self._backend.log_metrics({name: value}, step=None)

    def log_hparams(self, config: Any) -> None:
        """Forward hparams to the backend (prints config summary)."""
        self._backend.log_hparams(config)

    def close(self) -> None:
        close = getattr(self._backend, "close", None)
        if callable(close):
            close()


class _TracePermissiveEncoder(json.JSONEncoder):
    """Degrade non-serializable trace payloads to their type string
    so a stray proto/dataclass never crashes a train step."""

    def default(self, o: Any) -> Any:
        try:
            return super().default(o)
        except TypeError:
            return str(type(o))


class TraceLogWriter:
    """Appends non-scalar trace payloads to ``<log_dir>/trace.jsonl``.
    Record shape mirrors ``JsonLogger``: ``{"step": step, **payload}``.
    """

    def __init__(self, log_dir: str | Path):
        self.log_dir = Path(log_dir).expanduser()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.trace_file = self.log_dir / "trace.jsonl"

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        record = {"step": step} if step is not None else {}
        record.update(metrics)
        with open(self.trace_file, "a") as f:
            f.write(json.dumps(record, cls=_TracePermissiveEncoder) + "\n")

    def close(self) -> None:
        pass


class TraceLogSink:
    """Per-trajectory non-scalar payload sink.

    Accept iff the value is NON-scalar (dict / list / str / None): a
    payload with no numeric ``MetricSpec``. Predicate keys off value
    TYPE, not the spec.
    """

    def __init__(self, backend: Any):
        self._backend = backend

    def should_accept(self, name: str, value: Any, spec: MetricSpec | None) -> bool:
        # The ``isinstance(value, bool)`` guard is load-bearing:
        # ``_is_scalar`` excludes bool by design (a flag, not a time
        # series), so without this line the "not scalar" predicate
        # would silently accept True/False into trace.jsonl as
        # payloads. Do NOT remove as "redundant".
        if _is_scalar(value) or isinstance(value, bool):
            return False
        return True

    def write(self, metrics: dict[str, Any], step: int | None) -> None:
        if not metrics:
            return
        self._backend.log_metrics(metrics, step=step)

    def log_hparams(self, config: Any) -> None:
        """No-op: trace log is per-step payloads, not config."""
        return None

    def close(self) -> None:
        close = getattr(self._backend, "close", None)
        if callable(close):
            close()


class SinkDispatcher:
    """Routes each metric to the sinks whose predicate accepts it.

    For each metric, resolves the ``MetricSpec`` (via ``spec_lookup``,
    default ``core.metric_spec.spec_for``), asks each sink
    ``should_accept``, and forwards each sink its accepted subset.

    Unregistered metrics (``spec_lookup`` raises ``KeyError``) are
    dispatched with ``spec=None``: jsonl still accepts them
    (everything-archive), non-scalars still reach the trace sink, but
    ml_platform and console reject. The unknown name is warned once.
    """

    def __init__(
        self,
        sinks: list[Sink],
        spec_lookup: Callable[[str], MetricSpec] | None = None,
    ):
        self._sinks = list(sinks)
        if spec_lookup is None:
            from core.metric_spec import spec_for

            spec_lookup = spec_for
        self._spec_lookup = spec_lookup
        self._warned_unregistered: set[str] = set()

    def _resolve_spec(self, name: str) -> MetricSpec | None:
        try:
            return self._spec_lookup(name)
        except KeyError:
            if name not in self._warned_unregistered:
                self._warned_unregistered.add(name)
                logger.warning(
                    "SinkDispatcher: no MetricSpec for %r — defaulting to "
                    "jsonl-only (plus trace if non-scalar). Register a spec in "
                    "core.metric_spec.METRIC_SPECS to route it to ml_platform/console.",
                    name,
                )
            return None

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if not metrics:
            return
        per_sink: list[dict[str, Any]] = [{} for _ in self._sinks]
        for name, value in metrics.items():
            spec = self._resolve_spec(name)
            for i, sink in enumerate(self._sinks):
                if sink.should_accept(name, value, spec):
                    per_sink[i][name] = value
        for sink, subset in zip(self._sinks, per_sink):
            if subset:
                sink.write(subset, step)

    def log_hparams(self, config: Any) -> None:
        """Forward hparams to every sink. Exceptions propagate
        (fail-fast at startup)."""
        for sink in self._sinks:
            sink.log_hparams(config)

    def close(self) -> None:
        """Close every sink best-effort: per-sink exceptions are logged
        and remaining sinks still get closed so their file handles /
        buffers don't leak."""
        for sink in self._sinks:
            try:
                sink.close()
            except Exception:
                logger.exception(
                    "SinkDispatcher: sink %r failed to close cleanly; "
                    "continuing with remaining sinks",
                    sink,
                )
