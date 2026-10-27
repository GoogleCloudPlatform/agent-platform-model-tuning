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

"""Strategies for choosing which trials get compared side-by-side.

Comparison itself is a function of a *pair*; these strategies only decide
which trials meet. Unmatched trials are always reported, never dropped --
a silently shrinking comparison set is how an SxS lies.
"""

from __future__ import annotations

import dataclasses
import itertools
from collections import defaultdict
from collections.abc import Sequence

from .loader import TrialRef


@dataclasses.dataclass
class PairingResult:
    """Pairs plus everything that could not be paired."""

    pairs: list[tuple[TrialRef, TrialRef]] = dataclasses.field(default_factory=list)
    missing_in_a: list[TrialRef] = dataclasses.field(default_factory=list)
    missing_in_b: list[TrialRef] = dataclasses.field(default_factory=list)
    unpaired: list[TrialRef] = dataclasses.field(default_factory=list)


def _key(ref: TrialRef) -> tuple[str, int]:
    return (ref.task_id, ref.fanout_idx)


def pair_by_task(
    refs_a: Sequence[TrialRef], refs_b: Sequence[TrialRef]
) -> PairingResult:
    """Join two runs on ``(task_id, fanout_idx)``.

    Tie-breaking on ``fanout_idx`` keeps rollout N of a task compared
    against rollout N of the same task rather than an arbitrary sibling.
    """
    by_key_b = {_key(r): r for r in refs_b}
    seen_b: set[tuple[str, int]] = set()
    result = PairingResult()
    for ref_a in sorted(refs_a, key=_key):
        match = by_key_b.get(_key(ref_a))
        if match is None:
            result.missing_in_b.append(ref_a)
            continue
        seen_b.add(_key(match))
        result.pairs.append((ref_a, match))
    result.missing_in_a = [r for r in sorted(refs_b, key=_key) if _key(r) not in seen_b]
    return result


def pair_by_fanout(refs: Sequence[TrialRef]) -> PairingResult:
    """Pair rollouts of the same task against each other (within one run).

    Measures sampling variance, not model quality. Rollouts are paired in
    fanout order (0,1), (2,3), ...; a task with an odd count leaves its
    last rollout unpaired.
    """
    buckets: dict[str, list[TrialRef]] = defaultdict(list)
    for ref in refs:
        buckets[ref.task_id].append(ref)
    result = PairingResult()
    for task_id in sorted(buckets):
        ordered = sorted(buckets[task_id], key=lambda r: r.fanout_idx)
        pairs = list(itertools.zip_longest(ordered[::2], ordered[1::2]))
        for first, second in pairs:
            if second is None:
                result.unpaired.append(first)
            else:
                result.pairs.append((first, second))
    return result


def pair_explicit(ref_a: TrialRef, ref_b: TrialRef) -> PairingResult:
    """The degenerate strategy: exactly the two trials the caller named."""
    return PairingResult(pairs=[(ref_a, ref_b)])
