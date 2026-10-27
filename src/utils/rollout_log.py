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

"""Route log records to a per-rollout file chosen by the current context.

The pipeline runs many rollouts concurrently in one process, so the
process-wide log interleaves them and no single rollout's story can be read
back. This module adds a second destination: a file bound to the CURRENT
context, so each rollout gets its own log without any caller threading a
logger object through every call site.

Isolation comes from ``contextvars``: ``ContextVar.set`` mutates only the
current task's copy of the context, ``asyncio`` copies the context when a task
is created, and ``asyncio.to_thread`` runs its target under a copied context.
Concurrent rollouts therefore hold independent values with no locking, and one
rollout can neither read nor clobber another's binding.

The handler is purely additive: records still reach the console and the
process-wide log file exactly as before.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from pathlib import Path
from typing import IO, Iterator

logger = logging.getLogger(__name__)

_rollout_log_path: ContextVar[Path | None] = ContextVar(
    "rollout_log_path", default=None
)

_FORMAT = "%(asctime)s %(name)s:%(lineno)d [%(levelname)s] %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S %z"


class RolloutLogHandler(logging.Handler):
    """Write each record to the log file bound to the current context.

    Records emitted with no binding (pipeline startup, the dispatcher, any
    background worker) are dropped by this handler -- they are not
    rollout-scoped, and they are already in the process-wide log.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        self._streams: dict[Path, IO[str]] = {}
        self._failed: set[Path] = set()
        # How many live bindings reference each path. The SAME path may be
        # bound by two binders concurrently, so closing the stream when the
        # first of them unbinds would yank the file out from under the other.
        # Refcount and close only on the last release.
        self._refs: dict[Path, int] = {}

    def acquire_path(self, path: Path) -> None:
        with self.lock if self.lock is not None else nullcontext():
            self._refs[path] = self._refs.get(path, 0) + 1

    def _stream_for(self, path: Path) -> IO[str] | None:
        """Open (and cache) the stream for ``path``; None if it can't be opened."""
        cached = self._streams.get(path)
        if cached is not None:
            return cached
        if path in self._failed:
            return None  # already warned once for this path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = open(path, "a", encoding="utf-8")
        except OSError as exc:
            # Warn ONCE per path: a per-record warning on a broken path would
            # itself flood the log we are trying to keep readable.
            #
            # ORDER IS LOAD-BEARING. The warning below is itself a record that
            # re-enters this handler for the same (still broken) path, so the
            # path must be marked failed BEFORE warning or the two lines
            # recurse until the record cascade exhausts itself. Measured at
            # 271 warnings from a single failure when the guard was removed.
            self._failed.add(path)
            logger.warning("could not open rollout log %s: %s", path, exc)
            return None
        self._streams[path] = stream
        return stream

    def emit(self, record: logging.LogRecord) -> None:
        path = _rollout_log_path.get()
        if path is None:
            return
        try:
            stream = self._stream_for(path)
            if stream is None:
                return
            stream.write(self.format(record) + "\n")
            stream.flush()  # a crashed rollout must still leave a readable log
        except Exception:  # noqa: BLE001 - a raising handler breaks callers
            self.handleError(record)

    def release_path(self, path: Path) -> None:
        """Drop one reference; close the stream only when the last one goes."""
        with self.lock if self.lock is not None else nullcontext():
            remaining = self._refs.get(path, 1) - 1
            if remaining > 0:
                self._refs[path] = remaining
                return
            self._refs.pop(path, None)
            # Forget a transient failure too, so a later binding of the same
            # path retries instead of being blocked for the process's life.
            self._failed.discard(path)
            stream = self._streams.pop(path, None)
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass

    def close(self) -> None:
        self._refs.clear()
        for path in list(self._streams):
            self.release_path(path)
        super().close()


def install_rollout_log_handler() -> RolloutLogHandler:
    """Add one ``RolloutLogHandler`` to the root logger. Idempotent."""
    root = logging.getLogger()
    for h in root.handlers:
        if isinstance(h, RolloutLogHandler):
            return h
    handler = RolloutLogHandler()
    root.addHandler(handler)
    return handler


def _handlers() -> list[RolloutLogHandler]:
    return [
        h for h in logging.getLogger().handlers if isinstance(h, RolloutLogHandler)
    ]


@contextmanager
def bind_rollout_log(path: Path) -> Iterator[None]:
    """Route records emitted in this context (and its children) to ``path``.

    Re-entrant across contexts: the same path may be bound concurrently by two
    binders. The stream is closed when the last binding exits, not the first.
    """
    resolved = Path(path)
    handlers = _handlers()
    for h in handlers:
        h.acquire_path(resolved)
    token = _rollout_log_path.set(resolved)
    try:
        yield
    finally:
        # A Token may only be reset in the Context that created it. This
        # binder wraps AWAITED work (`core.agent` holds it across
        # `await self.rollout(...)`), so enter and exit can land in
        # different Contexts and `reset` then raises ValueError -- OUT of
        # the `with`, into the rollout, which is scored a failure. Observed
        # 108 times in one run: every group was dropped as
        # `too_few_valid_trajectories`, no train step completed, the run
        # produced no metrics at all and still exited 0.
        #
        # Routing log records must not be able to fail a training rollout,
        # so a foreign token degrades to clearing OUR binding in this
        # Context instead of raising.
        try:
            _rollout_log_path.reset(token)
        except ValueError:
            _rollout_log_path.set(None)
        finally:
            # Nested, because the release used to sit AFTER the reset: when
            # the reset raised, every handler kept its refcount on the path
            # and the stream was never closed. The leak was invisible next
            # to the failed rollout, and it is the reason this is a `try`
            # rather than a bare reset.
            for h in handlers:
                h.release_path(resolved)


def current_rollout_log_path() -> Path | None:
    """The path bound in this context, if any. For tests and diagnostics."""
    return _rollout_log_path.get()
