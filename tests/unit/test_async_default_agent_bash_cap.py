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

"""Bash-output cap tests for ``AsyncDefaultAgent``.

Motivation: On the withhold word_puzzle full eval, task 124 (tierC) ran a
python brute-force script that emitted hundreds of thousands of match
lines. Its trajectory JSON grew to **696 MB** and stalled the entire
asyncio.gather batch for 15+ minutes while other 149 tasks sat idle.

The runner's ``--wall-time-limit-seconds`` did NOT catch it because the
wall-time check happens BETWEEN LLM turns, not while ``subprocess.run``
is buffering stdout. The runaway output landed in the trajectory
message list on the FIRST turn back to the model.

Fix: cap bash-tool output at 64 KB (head + tail truncation with a marker
in the middle) right after ``env.execute`` returns. Keeps the model
informed (it sees the truncation marker and can add ``head`` / ``grep``
/ ``wc -l`` on the next turn) without letting arbitrarily-large output
poison the trajectory.

Tests are isolated to the ``_cap_bash_output`` helper -- no docker, no
mini-agent, no event loop. That's enough to pin the contract; the
call-site wiring is a one-liner that's exercised by higher-level tests
in a companion unit-test file for the sampler adapter model.
"""

from __future__ import annotations

import pytest

# The agent_solver package imports `minisweagent` at module scope, so
# even this bash-cap-only test file transitively needs the extra
# (`uv sync --extra agent_solver`). Skip the module cleanly when the
# extra isn't installed. Must gate BEFORE the agent_solver import below.
pytest.importorskip("minisweagent")

from example_recipe.word_puzzle.agent_solver.minisweagent.async_default_agent import (  # noqa: E402
    _cap_bash_output,
    _CAP_BASH_OUTPUT_MAX_BYTES,
)


# --------------------------------------------------------------------------
# Below-cap: unchanged passthrough
# --------------------------------------------------------------------------


def test_below_cap_output_is_unchanged():
    """Small output (< cap) passes through byte-for-byte."""
    original = {
        "output": "hello world\n",
        "returncode": 0,
        "exception_info": "",
    }
    result = _cap_bash_output(dict(original))  # copy so we can compare
    assert result == original


def test_exactly_cap_output_is_unchanged():
    """Output exactly at the cap must NOT be truncated (boundary check)."""
    text = "x" * _CAP_BASH_OUTPUT_MAX_BYTES
    result = _cap_bash_output({"output": text, "returncode": 0})
    assert result["output"] == text
    assert "truncated" not in result["output"]


# --------------------------------------------------------------------------
# Above-cap: head + tail truncation with marker
# --------------------------------------------------------------------------


def test_above_cap_output_is_truncated_with_marker():
    """Output larger than cap is truncated; a marker is inserted."""
    n = _CAP_BASH_OUTPUT_MAX_BYTES * 4  # comfortably above cap
    text = "A" * n
    result = _cap_bash_output({"output": text, "returncode": 0})
    # The output must be strictly shorter than the original.
    assert len(result["output"]) < n
    # The marker must be present and must include the byte counts.
    assert "truncated" in result["output"]
    assert str(n) in result["output"], (
        "marker should include the original byte count so the model "
        "knows how much it lost"
    )


def test_truncation_preserves_head_and_tail():
    """Truncated output starts with the original's head and ends with the tail.

    This is the whole point of head+tail truncation: tracebacks are usually
    at the END, initial diagnostic setup at the START. Keeping both lets
    the model see the error message AND the setup context.
    """
    # Construct output with distinct head/tail markers so we can verify
    # both survive the cut.
    head = "HEAD_MARKER_"
    tail = "_TAIL_MARKER"
    middle = "M" * (_CAP_BASH_OUTPUT_MAX_BYTES * 4)
    text = head + middle + tail
    result = _cap_bash_output({"output": text, "returncode": 0})
    assert result["output"].startswith(head), (
        "head of output must be preserved so tracebacks with filename+lineno "
        "at the top survive"
    )
    assert result["output"].endswith(tail), (
        "tail of output must be preserved so python's final "
        "'ValueError: <msg>' line survives"
    )


def test_truncation_preserves_non_output_fields():
    """returncode and exception_info must pass through unchanged.

    The cap only touches the ``output`` field; the return-code and any
    caller-provided exception info are semantically independent and
    downstream consumers rely on them.
    """
    text = "X" * (_CAP_BASH_OUTPUT_MAX_BYTES * 2)
    result = _cap_bash_output(
        {
            "output": text,
            "returncode": 137,  # e.g. SIGKILL
            "exception_info": "TimeoutExpired: docker exec timed out",
            "extra": {"exception_type": "TimeoutExpired"},
        }
    )
    assert result["returncode"] == 137
    assert result["exception_info"] == "TimeoutExpired: docker exec timed out"
    assert result["extra"] == {"exception_type": "TimeoutExpired"}


# --------------------------------------------------------------------------
# Robustness: missing / non-string output
# --------------------------------------------------------------------------


def test_missing_output_field_is_tolerated():
    """If output dict has no ``output`` key, don't crash -- pass through."""
    result = _cap_bash_output({"returncode": 0})
    assert "output" not in result
    assert result["returncode"] == 0


def test_none_output_is_tolerated():
    """If ``output`` is None (some backends), don't crash -- pass through."""
    result = _cap_bash_output({"output": None, "returncode": 0})
    assert result["output"] is None


# --------------------------------------------------------------------------
# Byte-count semantics: UTF-8 bytes, not chars
# --------------------------------------------------------------------------


def test_cap_counts_utf8_bytes_not_chars():
    """Multibyte chars count as their UTF-8 byte length, not 1 char each.

    Otherwise a run producing e.g. lots of CJK output would appear
    "well under cap" by char count but blow the trajectory size budget
    by real byte count.
    """
    # Each "\u2603" (snowman) is 3 UTF-8 bytes.
    # Choose char count so char-count < cap but byte-count > cap.
    char_count = (_CAP_BASH_OUTPUT_MAX_BYTES // 3) + 100  # 3 bytes/char
    text = "\u2603" * char_count
    # Sanity: chars < cap but bytes > cap.
    assert len(text) < _CAP_BASH_OUTPUT_MAX_BYTES
    assert len(text.encode("utf-8")) > _CAP_BASH_OUTPUT_MAX_BYTES
    result = _cap_bash_output({"output": text, "returncode": 0})
    assert "truncated" in result["output"], (
        "cap must use utf-8 byte length, not python char count, "
        "so multibyte-heavy output is still capped"
    )
