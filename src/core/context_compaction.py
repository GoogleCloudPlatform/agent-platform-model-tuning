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

"""Pure helpers for in-loop conversation compaction.

A multi-turn rollout grows its observation until the backend rejects the
request for exceeding its prompt cap. These helpers decide *when* to
compact (from measured token counts) and *what to keep* (whole
model/tool exchange pairs from the tail), leaving the sampling and the
trajectory bookkeeping to the agent loop.

No I/O, no protos constructed -- only inspected -- so every rule here is
directly unit-testable.
"""

from __future__ import annotations

from collections.abc import Sequence

from google.cloud.aiplatform_v1.types import Content

# Serialized-JSON chars per prompt token, measured on a long tool-using
# rollout (362,843 request chars for 116,962 reported prompt tokens).
# Only used to size the retained tail, never the trigger -- the trigger
# uses the backend's own reported counts.
CHARS_PER_TOKEN = 3.1

# Share of the context cap the retained tail may occupy.
TAIL_TOKEN_BUDGET_RATIO = 0.2

# Ceiling on retained exchange pairs. Two is enough to carry the tool
# result the model was mid-way through interpreting.
MAX_TAIL_PAIRS = 2


def should_compact(
    *,
    prompt_tokens: int | None,
    generation_tokens: int | None,
    max_context_tokens: int | None,
    trigger_ratio: float,
) -> bool:
    """True when the *next* request is projected past the trigger point.

    The projection is ``prompt + generation`` from the last completed
    call: the next request replays the whole conversation plus the model
    turn just produced. The tool response that follows is small enough
    to ignore (they run a few hundred chars).

    Returns False when compaction is disabled (``max_context_tokens`` is
    None) or the backend reported no token counts -- guessing at the
    size is worse than letting the existing failure path run.
    """
    if max_context_tokens is None:
        return False
    if prompt_tokens is None or generation_tokens is None:
        return False
    return prompt_tokens + generation_tokens > max_context_tokens * trigger_ratio


def tail_token_budget(max_context_tokens: int) -> int:
    """Token allowance for the retained tail."""
    return int(max_context_tokens * TAIL_TOKEN_BUDGET_RATIO)


def estimate_content_tokens(content: Content) -> int:
    """Rough token cost of one ``Content`` from its serialized length."""
    # indent=None matches how CHARS_PER_TOKEN was measured (compact
    # request JSON); the default pretty-printing inflates it by 2-7%.
    return int(len(Content.to_json(content, indent=None)) / CHARS_PER_TOKEN)


def select_tail(
    contents: Sequence[Content],
    *,
    budget_tokens: int,
    max_pairs: int = MAX_TAIL_PAIRS,
) -> tuple[list[Content], int]:
    """Pick the most recent whole exchange pairs to keep verbatim.

    ``contents[0]`` is the original task and is handled by the caller, so
    it is never part of the tail. Everything after it alternates
    ``model`` (functionCall + thoughtSignature) and ``user``
    (functionResponse); those two together are one *exchange pair* and
    are kept or dropped as a unit -- an orphaned call would leave the
    history structurally invalid.

    Pairing is aligned by role from the end, not by index parity. A
    rollout can end on a *dangling* ``model`` turn (an env step that
    returned no function responses records the call but no response);
    that entry is dropped before pairing, since it cannot be replayed in
    a rebuilt prompt and would otherwise misalign every pair behind it.

    Walks newest-first, stopping at ``max_pairs`` or when the next pair
    would exceed ``budget_tokens``. The most recent pair is always
    retained even when it alone exceeds the budget: it is the one the
    model was mid-way through interpreting, and it is also the pair most
    likely to be large. The one exception is ``max_pairs=0``, which
    honours the caller and retains nothing.

    Returns ``(tail_contents_oldest_first, pairs_retained)``.
    """
    body = contents[1:]
    end = len(body)
    if end and body[end - 1].role == "model":
        # Dangling call with no response yet: it cannot be replayed, and
        # it must not shift the pairing of everything behind it.
        end -= 1
    pairs: list[list[Content]] = []
    used = 0
    i = end
    while i >= 2 and len(pairs) < max_pairs:
        pair = body[i - 2 : i]
        if pair[0].role != "model" or pair[1].role != "user":
            # Genuinely malformed history (not just a dangling tail):
            # stop rather than emit a split or mis-rolled pair.
            break
        cost = sum(estimate_content_tokens(c) for c in pair)
        if pairs and used + cost > budget_tokens:
            break
        pairs.append(pair)
        used += cost
        i -= 2
    pairs.reverse()
    return [c for pair in pairs for c in pair], len(pairs)
