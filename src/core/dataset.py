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

"""JSONL-backed dataset and a simple batching iterator."""

from __future__ import annotations

import collections.abc
import json
import random
from typing import Any, Callable, Literal, overload


class JsonlDataset(collections.abc.Sequence[Any]):
    """Generic JSONL-backed dataset.

    Each non-blank line is parsed as JSON. An optional ``transform_fn``
    converts the raw record into whatever the transform returns
    (typically a recipe's ``TaskData`` subclass); with no transform,
    records are dicts.
    """

    def __init__(
        self,
        file_path: str,
        transform_fn: Callable[[dict[str, Any]], Any] | None = None,
        shuffle: bool = False,
        seed: int = 42,
    ):
        self.data: list[Any] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Malformed JSON in {file_path} at line {line_no}: {e}"
                    ) from e
                if transform_fn is not None:
                    item = transform_fn(item)
                self.data.append(item)
        if shuffle:
            self.shuffle(seed)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Any:
        return self.data[idx]

    def shuffle(self, seed: int = 42) -> None:
        rng = random.Random(seed)
        rng.shuffle(self.data)

    # Overloads let pyright narrow the return per-call: training passes
    # ``batch_size`` and gets ``BatchIterator``; eval passes ``max_items``
    # and gets ``BatchIterator | None`` (None on empty).
    @overload
    def build_iter(
        self,
        *,
        batch_size: int,
        max_items: Literal[None] = None,
        repeat: bool = True,
        resume_from_position: int | None = None,
    ) -> BatchIterator: ...

    @overload
    def build_iter(
        self,
        *,
        batch_size: Literal[None] = None,
        max_items: int,
        repeat: bool = True,
        resume_from_position: int | None = None,
    ) -> "BatchIterator | None": ...

    def build_iter(
        self,
        *,
        batch_size: int | None = None,
        max_items: int | None = None,
        repeat: bool = True,
        resume_from_position: int | None = None,
    ) -> "BatchIterator | None":
        """Build a ``BatchIterator`` over this dataset.

        Two usage patterns, distinguished by which sizing kwarg you pass:

        * **Training**: pass ``batch_size`` (typically ``repeat=True``).
          Optionally pass ``resume_from_position`` — an absolute
          ``current`` recovered from the metrics file. The pipeline
          has no separate rollout-batch counter; this is the single
          source of truth for resume.

          .. warning::
             ``resume_from_position`` assumes the dataset is in the
             **same deterministic order** as the run that produced the
             saved position. Today this holds (fixed file + constant
             shuffle seed). If seed is made configurable or per-epoch
             reshuffling is added, callers must persist a shuffle
             version and verify on resume.
        * **Eval**: pass ``max_items`` (``-1`` = entire dataset).
          Returns ``None`` if the result would be empty; typically
          used with ``repeat=False``.

        Exactly one of ``batch_size`` / ``max_items`` must be provided.
        """
        if (batch_size is None) == (max_items is None):
            raise ValueError(
                "build_iter: pass exactly one of `batch_size` or `max_items`"
            )
        if max_items is not None:
            if max_items == 0 or len(self) == 0:
                return None
            batch_size = len(self) if max_items < 0 else min(max_items, len(self))
        assert batch_size is not None  # XOR check above guarantees this.
        it = BatchIterator(self, batch_size=batch_size, repeat=repeat)
        if resume_from_position is not None:
            it.current = resume_from_position
        return it


class BatchIterator(collections.abc.Iterator[list[Any]]):
    """Yields fixed-size batches over any Sequence, optionally repeating.

    When ``repeat=True``, wraps around to always emit a full batch. When
    ``repeat=False``, the final batch may be partial and the iterator
    raises ``StopIteration`` once exhausted.
    """

    def __init__(
        self,
        dataset: collections.abc.Sequence[Any],
        batch_size: int,
        repeat: bool = True,
    ):
        assert batch_size > 0, "batch_size must be > 0"
        self.dataset = dataset
        self.batch_size = batch_size
        self.repeat = repeat
        self.current = 0

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self) -> list[Any]:
        if self.current >= len(self.dataset):
            if self.repeat:
                self.current = 0
            else:
                raise StopIteration

        end_idx = min(self.current + self.batch_size, len(self.dataset))
        batch = [self.dataset[i] for i in range(self.current, end_idx)]
        self.current = end_idx

        # Wrap-around to fill the full batch when repeating.
        while self.repeat and len(batch) < self.batch_size:
            self.current = 0
            remaining = self.batch_size - len(batch)
            end_idx = min(remaining, len(self.dataset))
            batch.extend(self.dataset[i] for i in range(self.current, end_idx))
            self.current = end_idx

        return batch

    def reset(self) -> None:
        self.current = 0

    def advance(self, num_items: int) -> None:
        self.current += num_items
        if len(self.dataset) > 0:
            self.current = self.current % len(self.dataset)
