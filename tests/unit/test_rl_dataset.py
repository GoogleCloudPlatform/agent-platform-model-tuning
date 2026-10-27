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

"""Tests for core.dataset."""

import collections.abc
import json
from pathlib import Path

import pytest

from core import dataset


def _write_jsonl(tmp_path: Path, lines: list[str], name: str = "d.jsonl") -> Path:
    p = tmp_path / name
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


class TestJsonlDataset:
    def test_parses_each_nonblank_line(self, tmp_path: Path):
        p = _write_jsonl(
            tmp_path,
            [
                json.dumps({"a": 1}),
                "",
                "   ",
                json.dumps({"a": 2}),
            ],
        )
        d = dataset.JsonlDataset(str(p))
        assert len(d) == 2
        assert d[0] == {"a": 1}
        assert d[1] == {"a": 2}

    def test_applies_transform_fn(self, tmp_path: Path):
        p = _write_jsonl(tmp_path, [json.dumps({"raw": 5})])
        d = dataset.JsonlDataset(
            str(p), transform_fn=lambda r: {"doubled": r["raw"] * 2}
        )
        assert d[0] == {"doubled": 10}

    def test_malformed_json_raises_valueerror_with_context(self, tmp_path: Path):
        p = _write_jsonl(tmp_path, [json.dumps({"ok": 1}), "not json{"])
        with pytest.raises(ValueError) as exc_info:
            dataset.JsonlDataset(str(p))
        msg = str(exc_info.value)
        assert str(p) in msg
        assert "line 2" in msg

    def test_shuffle_permutes_deterministically(self, tmp_path: Path):
        p = _write_jsonl(tmp_path, [json.dumps({"i": i}) for i in range(10)])
        d1 = dataset.JsonlDataset(str(p), shuffle=True, seed=42)
        d2 = dataset.JsonlDataset(str(p), shuffle=True, seed=42)
        d3 = dataset.JsonlDataset(str(p), shuffle=True, seed=7)
        assert [x["i"] for x in d1] == [x["i"] for x in d2]
        assert [x["i"] for x in d1] != [x["i"] for x in d3]

    def test_is_a_sequence(self):
        """We inherit the collections.abc.Sequence mixin methods for free."""
        assert collections.abc.Sequence in dataset.JsonlDataset.__mro__

    def test_sequence_mixin_methods_work(self, tmp_path: Path):
        p = _write_jsonl(tmp_path, [json.dumps({"i": i}) for i in range(3)])
        d = dataset.JsonlDataset(str(p))
        # Sequence gives us __contains__, __iter__, index, count.
        assert {"i": 1} in d
        assert list(d) == [{"i": 0}, {"i": 1}, {"i": 2}]
        assert d.index({"i": 2}) == 2


class TestBatchIterator:
    @pytest.fixture
    def seq(self):
        return list(range(7))

    def test_yields_full_batches_when_dataset_size_divisible(self):
        seq = list(range(6))
        it = dataset.BatchIterator(seq, batch_size=2, repeat=False)
        assert list(it) == [[0, 1], [2, 3], [4, 5]]

    def test_non_repeat_final_batch_is_partial(self, seq):
        it = dataset.BatchIterator(seq, batch_size=3, repeat=False)
        assert list(it) == [[0, 1, 2], [3, 4, 5], [6]]

    def test_repeat_wraps_final_partial_batch_to_full(self, seq):
        it = dataset.BatchIterator(seq, batch_size=3, repeat=True)
        b1 = next(it)
        b2 = next(it)
        b3 = next(it)
        assert b1 == [0, 1, 2]
        assert b2 == [3, 4, 5]
        assert b3 == [6, 0, 1]

    def test_repeat_true_never_stops(self):
        it = dataset.BatchIterator([1, 2], batch_size=1, repeat=True)
        got = [next(it) for _ in range(5)]
        assert got == [[1], [2], [1], [2], [1]]

    def test_batch_size_must_be_positive(self):
        with pytest.raises(AssertionError):
            dataset.BatchIterator([1, 2], batch_size=0)

    def test_is_an_iterator(self, seq):
        it = dataset.BatchIterator(seq, batch_size=2)
        assert isinstance(it, collections.abc.Iterator)

    def test_reset_returns_to_start(self, seq):
        it = dataset.BatchIterator(seq, batch_size=2, repeat=False)
        next(it)
        next(it)
        it.reset()
        assert next(it) == [0, 1]

    def test_advance_moves_cursor(self, seq):
        it = dataset.BatchIterator(seq, batch_size=2, repeat=False)
        it.advance(3)
        assert next(it) == [3, 4]

    def test_repeat_with_batch_size_larger_than_dataset(self):
        # Wraparound loop must run multiple times when batch_size exceeds
        # the underlying sequence length.
        it = dataset.BatchIterator([1, 2], batch_size=5, repeat=True)
        assert next(it) == [1, 2, 1, 2, 1]

    def test_iter_restarts_exhausted_iterator(self):
        # After a repeat=False iterator has been exhausted, re-entering
        # iter() should reset the cursor to 0.
        it = dataset.BatchIterator([1, 2, 3, 4], batch_size=2, repeat=False)
        list(it)  # exhaust
        iter(it)  # re-enter iteration
        assert next(it) == [1, 2]
