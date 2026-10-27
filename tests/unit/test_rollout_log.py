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

"""Per-rollout log routing: the isolation guarantee under concurrency."""

from __future__ import annotations

import asyncio
import logging

import pytest

from utils.rollout_log import (
    RolloutLogHandler,
    bind_rollout_log,
    install_rollout_log_handler,
)


@pytest.fixture
def handler():
    """Install a handler on root; remove it again so tests don't leak state."""
    h = RolloutLogHandler()
    root = logging.getLogger()
    root.addHandler(h)
    old_level = root.level
    root.setLevel(logging.INFO)
    yield h
    root.removeHandler(h)
    root.setLevel(old_level)
    h.close()


def test_bound_records_go_to_the_bound_file(tmp_path, handler):
    log = logging.getLogger("t1")
    path = tmp_path / "a" / "agent.log"
    with bind_rollout_log(path):
        log.info("hello")
    assert "hello" in path.read_text()


def test_unbound_records_write_no_file(tmp_path, handler):
    logging.getLogger("t2").info("orphan")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.asyncio
async def test_concurrent_rollouts_do_not_interleave(tmp_path, handler):
    """THE isolation claim: N tasks, N files, no cross-contamination."""
    log = logging.getLogger("t3")

    async def rollout(i: int) -> None:
        with bind_rollout_log(tmp_path / f"r{i}" / "agent.log"):
            for turn in range(5):
                log.info("rollout=%d turn=%d", i, turn)
                await asyncio.sleep(0)  # force interleaving

    await asyncio.gather(*(rollout(i) for i in range(3)))

    for i in range(3):
        lines = (tmp_path / f"r{i}" / "agent.log").read_text().splitlines()
        assert len(lines) == 5
        assert all(f"rollout={i} " in ln for ln in lines)


@pytest.mark.asyncio
async def test_to_thread_inherits_the_binding(tmp_path, handler):
    """asyncio.to_thread copies the context, so worker lines route correctly."""
    log = logging.getLogger("t4")
    path = tmp_path / "th" / "agent.log"
    with bind_rollout_log(path):
        await asyncio.to_thread(log.info, "from a worker thread")
    assert "from a worker thread" in path.read_text()


def test_unwritable_path_warns_once_and_never_raises(tmp_path, handler, caplog):
    """A logging handler that raises would break unrelated code.

    ``set_level`` (not ``at_level``) is deliberate: ``at_level(WARNING)`` would
    raise the ROOT level to WARNING and filter the INFO records below before
    they ever reach a handler, so the per-record dedup path -- the thing this
    test exists to pin -- would never execute.
    """
    blocked = tmp_path / "afile"
    blocked.write_text("not a directory")
    log = logging.getLogger("t5")
    caplog.set_level(logging.INFO)
    with bind_rollout_log(blocked / "nested" / "agent.log"):
        log.info("one")
        log.info("two")
    warnings = [
        r for r in caplog.records if "could not open rollout log" in r.getMessage()
    ]
    assert len(warnings) == 1  # once per path, not once per record


def test_binding_without_logging_creates_no_file(tmp_path, handler):
    """A later task's upload guard is ``if not path.exists()``, so a silent
    rollout must not leave an empty file behind."""
    path = tmp_path / "quiet" / "agent.log"
    with bind_rollout_log(path):
        pass
    assert not path.exists()


@pytest.mark.asyncio
async def test_nested_binding_of_the_same_path_keeps_the_stream_open(
    tmp_path, handler
):
    """The proxy binds a rollout's path while the rollout itself holds it.

    If the inner binding closed the stream on exit, the outer one would write
    to a closed file for the rest of the rollout.
    """
    log = logging.getLogger("t6")
    path = tmp_path / "shared" / "agent.log"
    with bind_rollout_log(path):
        log.info("outer-before")
        with bind_rollout_log(path):  # simulates a concurrent proxy request
            log.info("inner")
        log.info("outer-after")
    text = path.read_text()
    assert "outer-before" in text and "inner" in text and "outer-after" in text


def test_install_is_idempotent():
    root = logging.getLogger()
    before = len(root.handlers)
    install_rollout_log_handler()
    install_rollout_log_handler()
    added = len(root.handlers) - before
    assert added == 1
    root.handlers = [h for h in root.handlers if not isinstance(h, RolloutLogHandler)]


def test_configure_logging_module_installs_the_handler(tmp_path):
    """The handler must be live in any process that configures logging."""
    import logging as _logging

    from utils.logger import configure_logging_module

    root = _logging.getLogger()
    saved = list(root.handlers)
    try:
        configure_logging_module(str(tmp_path / "logs.log"))
        assert any(isinstance(h, RolloutLogHandler) for h in root.handlers)
    finally:
        for h in root.handlers:
            if isinstance(h, RolloutLogHandler):
                h.close()
        root.handlers = saved


class TestAForeignContextTokenCannotFailARollout:
    """Enter and exit in DIFFERENT Contexts — the shape `core.agent` uses.

    `core.agent` holds this binder across `await self.rollout(...)`, so the
    generator's enter and exit can be resumed in different Contexts. A
    `ContextVar.Token` may only be reset where it was created, so the exit
    used to raise `ValueError` straight out of the `with` and into the
    rollout, which the pipeline then scored as a failed rollout.

    Live consequence, 2026-09-29: 108 of these in one run. Every group was
    dropped as `too_few_valid_trajectories`, no train step completed, the
    run emitted no metrics at all — and still exited 0.
    """

    def test_exiting_in_another_context_does_not_raise(self, tmp_path, handler):
        import contextvars

        path = tmp_path / "agent.log"
        cm = bind_rollout_log(path)
        # __enter__ (the `set`) runs in a COPY of this Context...
        contextvars.copy_context().run(cm.__enter__)
        # ...and __exit__ (the `reset`) runs out here. Must not raise.
        cm.__exit__(None, None, None)

    def test_the_handler_is_still_released_when_the_token_is_foreign(
        self, tmp_path, handler
    ):
        """The release used to sit after the reset, so a raise leaked it.

        This is the defect the outer `try` exists for: the stream stayed
        open with a live refcount, invisible beside the failed rollout.
        """
        import contextvars

        path = tmp_path / "agent.log"
        cm = bind_rollout_log(path)
        contextvars.copy_context().run(cm.__enter__)
        # `acquire_path` takes a REFCOUNT; the stream is opened lazily on
        # the first record, so the refcount is what proves the leak.
        assert handler._refs.get(path) == 1, "precondition: the path was acquired"
        cm.__exit__(None, None, None)
        assert path not in handler._refs, (
            "handler leaked its refcount on the foreign-token path; the "
            "stream would never be closed"
        )

    def test_the_normal_same_context_path_still_restores_the_previous_value(
        self, tmp_path, handler
    ):
        """The fix must not turn a correct reset into a blanket clear."""
        from utils.rollout_log import bind_rollout_log as bind, current_rollout_log_path

        outer, inner = tmp_path / "outer.log", tmp_path / "inner.log"
        with bind(outer):
            assert current_rollout_log_path() == outer
            with bind(inner):
                assert current_rollout_log_path() == inner
            # Restored to OUTER, not None.
            assert current_rollout_log_path() == outer
        assert current_rollout_log_path() is None
