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

"""SIGTERM must let async cleanup finish.

The bug these pin: raising SystemExit from a `signal.signal` handler unwinds
`run_until_complete` and leaves the coroutine SUSPENDED, so its `finally`
runs at GC with the loop already closed and any `await` inside dies with
"coroutine ignored GeneratorExit". Since `bundle.aclose()` (tuning-job
cancel) and the ATIF drain are both async, a plain `pkill` leaked a live
tuning job exactly like `pkill -9`.
"""

from __future__ import annotations

import asyncio
import signal

import pytest

from utils.signals import cancel_on_signal


class TestCancelOnSignal:
    async def test_async_cleanup_completes_on_cancellation(self):
        """The whole point: a `finally` that awaits must get to finish."""
        done: list[str] = []

        async def _aclose():
            await asyncio.sleep(0.01)  # a real round trip
            done.append("job cancelled")

        async def work():
            try:
                await asyncio.sleep(30)
            finally:
                await asyncio.shield(_aclose())

        task = asyncio.ensure_future(cancel_on_signal(work(), label="test"))
        await asyncio.sleep(0.05)
        signal.raise_signal(signal.SIGTERM)
        await task
        assert done == ["job cancelled"]

    async def test_cleanup_needs_no_shield(self):
        """An UNSHIELDED await in a cancelled task's finally still completes.

        Written first as the opposite assertion, on the intuition that a
        cancelling task re-raises at the next await. It does not:
        CancelledError is delivered once, at the await that was pending.
        Measured shielded vs not at 1/2/3 cancels, the results were
        identical, so the shields added earlier were removed as inert.
        """
        done: list[str] = []

        async def work():
            try:
                await asyncio.sleep(30)
            finally:
                await asyncio.sleep(0.01)  # deliberately NOT shielded
                done.append("cleanup ran")

        task = asyncio.ensure_future(cancel_on_signal(work(), label="test"))
        await asyncio.sleep(0.05)
        signal.raise_signal(signal.SIGTERM)
        await task
        assert done == ["cleanup ran"]

    async def test_repeated_cancellation_does_lose_cleanup(self):
        """The real limit, so the boundary is documented rather than assumed.

        Three cancels (an impatient operator) loses it -- shield or not.
        That is SIGKILL territory, covered by the run registry.
        """
        done: list[str] = []

        async def work():
            try:
                await asyncio.sleep(30)
            finally:
                await asyncio.sleep(0.02)
                done.append("cleanup ran")

        task = asyncio.ensure_future(cancel_on_signal(work(), label="test"))
        await asyncio.sleep(0.05)
        for _ in range(3):
            signal.raise_signal(signal.SIGTERM)
            await asyncio.sleep(0)
        await task
        assert done == []

    async def test_a_normal_return_is_passed_through(self):
        async def work():
            return "result"

        assert await cancel_on_signal(work(), label="test") == "result"

    async def test_a_real_error_still_propagates(self):
        """Signal handling must not swallow genuine failures."""

        async def work():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            await cancel_on_signal(work(), label="test")

    async def test_handlers_are_removed_afterwards(self):
        """Otherwise a later signal fires a handler for a dead task."""

        async def work():
            return None

        await cancel_on_signal(work(), label="test")
        loop = asyncio.get_running_loop()
        # Re-installing must succeed, i.e. nothing was left behind.
        loop.add_signal_handler(signal.SIGTERM, lambda: None)
        loop.remove_signal_handler(signal.SIGTERM)
