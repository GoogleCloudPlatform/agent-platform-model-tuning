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

"""Termination that lets async cleanup actually run.

The obvious approach does not work, and it fails silently. Raising
``SystemExit`` from a ``signal.signal`` handler unwinds
``loop.run_until_complete`` and leaves the main coroutine SUSPENDED. Its
``finally`` blocks run later, during GC, when the loop is already closed --
so any ``await`` in them dies with::

    RuntimeError: coroutine ignored GeneratorExit

Everything that matters at shutdown here is async: ``bundle.aclose()`` is
what cancels the Vertex tuning job, and the ATIF drain is what flushes
queued uploads and deletes the local tree. All of it was being skipped, so
a plain ``pkill`` leaked a live tuning job exactly like ``pkill -9`` --
while the handler's own docstring claimed the opposite.

:func:`cancel_on_signal` cancels the TASK instead. ``CancelledError`` is
raised at the task's current await point inside a LIVE loop, so ``finally``
blocks get to run their own awaits.

Cleanup awaits do NOT need ``asyncio.shield``. That was the intuition --
"the task is already cancelling, so the first bare await in a ``finally``
will re-raise" -- and it is wrong. A cancelled task's ``finally`` gets to
await normally; ``CancelledError`` is delivered once, at the await point
that was pending. Measured at one, two and three cancellations, shielded
and not, the results are identical: cleanup completes for one and two, and
fails for three either way. The shields were removed rather than kept as
insurance, because a guard that provably changes nothing is a guard people
trust for protection it does not give.

Three-plus cancellations (an impatient operator sending SIGTERM repeatedly)
does lose cleanup. That is the same territory as ``SIGKILL``, and it is
what the run registry covers.

Lives in ``src/utils`` rather than beside either entry point because the
train and eval paths need identical, non-obvious behaviour, and two copies
of subtle shutdown logic drift.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from typing import Any, Awaitable, Iterable

logger = logging.getLogger(__name__)

# SIGINT included so Ctrl-C on an interactive run cleans up the same way a
# `pkill` does. SIGKILL is absent because it cannot be caught -- that path is
# covered by the run registry instead.
DEFAULT_SIGNALS: tuple[signal.Signals, ...] = (
    signal.SIGTERM,
    signal.SIGHUP,
    signal.SIGINT,
)


async def cancel_on_signal(
    coro: Awaitable[Any],
    *,
    signals: Iterable[signal.Signals] = DEFAULT_SIGNALS,
    label: str = "run",
) -> Any:
    """Await ``coro`` as a task that the given signals CANCEL.

    Returns the coroutine's result, or ``None`` if it was cancelled. The
    cancellation is swallowed deliberately: a signalled shutdown that ran its
    cleanup is an orderly stop, not a crash, and re-raising would turn a
    successful teardown into a traceback.
    """
    task = asyncio.ensure_future(coro)
    loop = asyncio.get_running_loop()
    installed = []
    for sig in signals:
        try:
            loop.add_signal_handler(sig, task.cancel)
            installed.append(sig)
        except (NotImplementedError, RuntimeError, ValueError):
            # Not a Unix main thread. The caller's `signal.signal` fallback
            # still terminates the process; only async cleanup is lost.
            logger.debug("cancel_on_signal: could not install handler for %s", sig)
    try:
        return await task
    except asyncio.CancelledError:
        logger.warning(
            "%s: cancelled by signal -- cleanup ran (tuning job cancelled, "
            "uploads drained)",
            label,
        )
        return None
    finally:
        for sig in installed:
            try:
                loop.remove_signal_handler(sig)
            except (NotImplementedError, RuntimeError, ValueError):
                pass
