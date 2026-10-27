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

"""Eval entry point: flattened ``main(config_name, ...)``.

Drives the ``src/eval.py`` entry point (word_puzzle only)
with the composed eval orchestrator + ``asyncio.run`` monkeypatched to
capture stubs, and asserts that ``run_eval`` receives a real ``RunConfig``
with the launch override applied to the routed sub-config and the recipe
label derived from ``config_name``. ``get`` (two-level config registry)
+ ``apply_overrides`` run for real.
"""

from __future__ import annotations

import eval as eval_entry
from core.run_config import RunConfig


def test_main_dispatches_word_puzzle_run_eval_with_override(monkeypatch):
    captured = {}

    async def fake_run_eval(config, *, recipe, jsonl_output_path=None):
        captured["config"] = config
        captured["recipe"] = recipe
        captured["jsonl_output_path"] = jsonl_output_path

    # main imports run_eval at module import time (top-level), so patch it
    # on the eval entry module namespace.
    monkeypatch.setattr(eval_entry, "run_eval", fake_run_eval)

    ran = {}

    def fake_asyncio_run(coro):
        # Drive the coroutine so fake_run_eval records the config, without
        # a real event loop taking over the test.
        ran["called"] = True
        try:
            coro.send(None)
        except StopIteration:
            pass

    monkeypatch.setattr(eval_entry.asyncio, "run", fake_asyncio_run)

    eval_entry.main(config_name="word_puzzle/async", vertex_project="my-proj")

    assert ran["called"] is True
    config = captured["config"]
    assert isinstance(config, RunConfig)
    # The vertex_project override routed onto the backend sub-config.
    assert config.backend.project == "my-proj"
    # The recipe label was derived from config_name.
    assert captured["recipe"] == "word_puzzle"
