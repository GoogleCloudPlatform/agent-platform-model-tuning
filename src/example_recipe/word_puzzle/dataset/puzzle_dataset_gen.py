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

"""Offline word-puzzle dataset generator.

Reproduces the synthesis pipeline behind
``combined_{training,eval}_word_transformation_v1.jsonl``. Purely local:
no LLM calls, no GCS I/O.

Two entry points:

1. ``generate_one_task(...)`` -- programmatic util. Returns a
   ``PuzzleTask`` dataclass; call ``.to_jsonl_record()`` for the
   ``{contents, references}`` wire shape that ``WordPuzzleEnv`` consumes.

2. ``python -m example_recipe.word_puzzle.dataset.puzzle_dataset_gen ...`` -- CLI
   that writes a JSONL file.

Word candidates come from the bundled ``word_allowlist.txt`` (English words
vetted to exclude slurs, sexual terms and profanity), or from a plain-text
file passed as ``--word-list``.
"""

from __future__ import annotations

import argparse
import base64
import codecs
import dataclasses
import json
import logging
import math
import os
import random
import string
import sys
from collections.abc import Sequence
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

DEFAULT_WORD_LIST = os.path.join(os.path.dirname(__file__), "word_allowlist.txt")

_ASCII_LETTERS = string.ascii_letters
_NUM_LETTERS = len(_ASCII_LETTERS)


# --- Character-level transforms with parameters --------------------------------


def _rotate_char(s: str, rotate: int = 2) -> str:
    rotate = rotate % len(s)
    return s[rotate:] + s[:rotate]


def _mirror_char(s: str, copy_input: bool = False) -> str:
    return (s + s[::-1]) if copy_input else s[::-1]


def _shift_char(s: str, shift: int = 7) -> str:
    # Caesar shift over [a-zA-Z]. Non-letters (digits, punctuation, whitespace)
    # pass through unchanged so that composing shift after a charset-changing
    # op (hex_encoding, ascii_values_as_string, base64_encoding, ...) doesn't
    # silently collapse every non-letter to a single letter via find()==-1.
    shift = shift % _NUM_LETTERS
    out = []
    for c in s:
        idx = _ASCII_LETTERS.find(c)
        if idx < 0:
            out.append(c)
        else:
            out.append(_ASCII_LETTERS[(idx + shift) % _NUM_LETTERS])
    return "".join(out)


# --- Fixed (parameter-free) transforms; the "extended" training set ------------


def _swap_adj(s: str) -> str:
    return "".join(s[i + 1] + s[i] for i in range(0, len(s) - 1, 2)) + (
        s[-1] if len(s) % 2 else ""
    )


def _reverse_blocks_of_3(s: str) -> str:
    return "".join(s[i : i + 3][::-1] for i in range(0, len(s), 3))


def _weave_from_ends(s: str) -> str:
    return "".join(sum(zip(s[: len(s) // 2], s[::-1]), ())) + (
        s[len(s) // 2] if len(s) % 2 else ""
    )


def _move_middle_to_front(s: str) -> str:
    n = len(s)
    if n % 2:
        return s[n // 2 : n // 2 + 1] + s[: n // 2] + s[n // 2 + 1 :]
    return s[n // 2 - 1 : n // 2 + 1] + s[: n // 2 - 1] + s[n // 2 + 1 :]


def _move_middle_to_end(s: str) -> str:
    n = len(s)
    if n % 2:
        return s[: n // 2] + s[n // 2 + 1 :] + s[n // 2 : n // 2 + 1]
    return s[: n // 2 - 1] + s[n // 2 + 1 :] + s[n // 2 - 1 : n // 2 + 1]


def _interleave_split(s: str) -> str:
    # Split s into halves and interleave char-by-char. For odd length, the
    # unpaired middle character goes last. (Previously this appended
    # s[len(s)//2 - 1] on odd length — the char BEFORE the middle — which
    # dropped the middle and duplicated an earlier char, making the op
    # non-injective on odd-length inputs.)
    half = len(s) // 2
    first, second = s[:half], s[half + (len(s) % 2) :]
    paired = "".join("".join(pair) for pair in zip(first, second))
    return paired + (s[half] if len(s) % 2 else "")


def _pull_outer_to_middle(s: str) -> str:
    return s[1 : len(s) // 2] + s[0] + s[-1] + s[len(s) // 2 : -1] if len(s) > 1 else s


TRANSFORMATION_FUNCTIONS: dict[str, Callable[[str], str]] = {
    "swap_adjacent_char_pairs": _swap_adj,
    "reverse_blocks_of_3": _reverse_blocks_of_3,
    "weave_from_ends": _weave_from_ends,
    "move_middle_to_front": _move_middle_to_front,
    "move_middle_to_end": _move_middle_to_end,
    "interleave_split_string": _interleave_split,
    "pull_outer_to_middle": _pull_outer_to_middle,
}


# --- Withhold (eval-only) transforms ------------------------------------------

# Alphabets for the "cycle within a permuted alphabet" family. Each op walks
# ONE step along its own alphabet and wraps, so the output is always drawn
# from that alphabet -- printable by construction, unlike the raw
# ``chr(ord(c) +/- n)`` ops these replaced.
#
# Deliberately NOT a Caesar cipher over a-z: the training set's ``shift`` is
# exactly that, so a plain Caesar would leak. Here the model must infer the
# ALPHABET ORDERING (vowels only / consonants only / QWERTY), not just an
# offset. Each is a single-step cycle, hence non-involutive (order 5, 21 and
# 26 respectively) -- which is what keeps deep compositions from collapsing
# to the identity the way the all-involution set does.
_VOWELS = "aeiou"
_CONSONANTS = "bcdfghjklmnpqrstvwxyz"
_QWERTY = "qwertyuiopasdfghjklzxcvbnm"


def _cycle_in(alphabet: str, step: int) -> Callable[[str], str]:
    """Map each char one ``step`` along ``alphabet`` (wrapping); others pass through."""

    def _apply(s: str) -> str:
        out = []
        for c in s:
            idx = alphabet.find(c)
            out.append(alphabet[(idx + step) % len(alphabet)] if idx >= 0 else c)
        return "".join(out)

    return _apply


def _atbash(s: str) -> str:
    out = []
    for c in s:
        if "a" <= c <= "z":
            out.append(chr(ord("a") + (ord("z") - ord(c))))
        elif "A" <= c <= "Z":
            out.append(chr(ord("A") + (ord("Z") - ord(c))))
        else:
            out.append(c)
    return "".join(out)


def _coprime_a(n: int, a: int = 3) -> int:
    """Multiplier for the affine index map, coerced to be coprime with ``n``.

    Forward and inverse MUST derive this identically -- both call this helper
    rather than duplicating the fallback, so the two can never drift apart.
    """
    if n <= 1:
        return 1
    if math.gcd(a, n) == 1:
        return a
    return next(x for x in range(2, n + 2) if math.gcd(x, n) == 1)


def _affine_index(s: str, a: int = 3, b: int = 1) -> str:
    """Position i of the input lands at (a*i + b) mod n -- a coordinate map on Z_n."""
    n = len(s)
    if n <= 1:
        return s
    a = _coprime_a(n, a)
    out = [""] * n
    for i, ch in enumerate(s):
        out[(a * i + b) % n] = ch
    return "".join(out)


def _invert_affine_index(s: str, a: int = 3, b: int = 1) -> str:
    n = len(s)
    if n <= 1:
        return s
    a = _coprime_a(n, a)
    return "".join(s[(a * i + b) % n] for i in range(n))


def _grid_dims(n: int) -> tuple[int, int]:
    c = math.ceil(math.sqrt(n)) if n else 1
    return math.ceil(n / c) if c else 1, c


def _grid_rot90(s: str) -> str:
    """Write row-wise into a near-square grid, rotate 90 degrees, read row-wise."""
    n = len(s)
    if n <= 1:
        return s
    r, c = _grid_dims(n)
    g = [[""] * c for _ in range(r)]
    for i, ch in enumerate(s):
        g[i // c][i % c] = ch
    out = [[""] * r for _ in range(c)]
    for i in range(r):
        for j in range(c):
            out[j][r - 1 - i] = g[i][j]
    return "".join(ch for row in out for ch in row if ch)


def _invert_grid_rot90(s: str) -> str:
    n = len(s)
    if n <= 1:
        return s
    r, c = _grid_dims(n)
    occ = [[(i * c + j) < n for j in range(c)] for i in range(r)]
    rot = [[""] * r for _ in range(c)]
    k = 0
    for a in range(c):
        for b in range(r):
            if occ[r - 1 - b][a]:
                rot[a][b] = s[k]
                k += 1
    out = [[""] * c for _ in range(r)]
    for i in range(r):
        for j in range(c):
            out[i][j] = rot[j][r - 1 - i]
    return "".join(ch for row in out for ch in row if ch)


def _spiral_order(n: int) -> list[int]:
    """Flat indices of a near-square grid in inward clockwise spiral order."""
    r, c = _grid_dims(n)
    order: list[int] = []
    top, bot, left, right = 0, r - 1, 0, c - 1
    while top <= bot and left <= right:
        for j in range(left, right + 1):
            order.append(top * c + j)
        top += 1
        for i in range(top, bot + 1):
            order.append(i * c + right)
        right -= 1
        if top <= bot:
            for j in range(right, left - 1, -1):
                order.append(bot * c + j)
            bot -= 1
        if left <= right:
            for i in range(bot, top - 1, -1):
                order.append(i * c + left)
            left += 1
    return [i for i in order if i < n]


def _spiral(s: str) -> str:
    n = len(s)
    if n <= 1:
        return s
    return "".join(s[i] for i in _spiral_order(n))


def _invert_spiral(s: str) -> str:
    n = len(s)
    if n <= 1:
        return s
    out = [""] * n
    for ch, i in zip(s, _spiral_order(n)):
        out[i] = ch
    return "".join(out)


def _grid_transpose(s: str) -> str:
    """Write row-wise into the grid, read column-wise -- reflect across (i,j)->(j,i)."""
    n = len(s)
    if n <= 1:
        return s
    _, c = _grid_dims(n)
    return "".join(s[i] for j in range(c) for i in range(j, n, c))


def _grid_diagonal_read(s: str) -> str:
    """Read the grid along successive anti-diagonals (i + j constant)."""
    n = len(s)
    if n <= 1:
        return s
    r, c = _grid_dims(n)
    out = []
    for d in range(r + c - 1):
        for i in range(r):
            j = d - i
            if 0 <= j < c and i * c + j < n:
                out.append(s[i * c + j])
    return "".join(out)


def _boustrophedon(s: str) -> str:
    """Serpentine grid read: row 0 left-to-right, row 1 right-to-left, ..."""
    n = len(s)
    if n <= 1:
        return s
    r, c = _grid_dims(n)
    out = []
    for i in range(r):
        row = [i * c + j for j in range(c) if i * c + j < n]
        out.extend(s[k] for k in (row if i % 2 == 0 else reversed(row)))
    return "".join(out)


def _rail_fence_3(s: str) -> str:
    """Three-rail zigzag transposition: write down-up over 3 rails, read rail by rail."""
    n = len(s)
    if n <= 1:
        return s
    rails: list[list[str]] = [[], [], []]
    rail, step = 0, 1
    for ch in s:
        rails[rail].append(ch)
        if rail == 0:
            step = 1
        elif rail == 2:
            step = -1
        rail += step
    return "".join("".join(r) for r in rails)


def _block_reverse_growing(s: str) -> str:
    """Reverse consecutive blocks of growing size 1, 2, 3, ... (length-dependent)."""
    n = len(s)
    if n <= 1:
        return s
    out = []
    i, size = 0, 1
    while i < n:
        out.extend(reversed(s[i : i + size]))
        i += size
        size += 1
    return "".join(out)


_VIGENERE_KEY = (12, 0, 19, 7)  # "math"


def _vigenere(s: str, sign: int) -> str:
    out = []
    ai = 0
    for ch in s:
        k = _VIGENERE_KEY[ai % len(_VIGENERE_KEY)] * sign
        if "a" <= ch <= "z":
            out.append(chr(ord("a") + (ord(ch) - ord("a") + k) % 26))
            ai += 1
        elif "A" <= ch <= "Z":
            out.append(chr(ord("A") + (ord(ch) - ord("A") + k) % 26))
            ai += 1
        else:
            out.append(ch)
    return "".join(out)


def _fib_shift(s: str, sign: int) -> str:
    """Shift character i by Fibonacci(i) mod 26 -- polyalphabetic, non-obvious period."""
    out = []
    a, b = 1, 1
    for ch in s:
        k = a * sign
        if "a" <= ch <= "z":
            out.append(chr(ord("a") + (ord(ch) - ord("a") + k) % 26))
        elif "A" <= ch <= "Z":
            out.append(chr(ord("A") + (ord(ch) - ord("A") + k) % 26))
        else:
            out.append(ch)
        a, b = b, (a + b) % 26
    return "".join(out)


def _pos_shift(s: str, sign: int, square: bool) -> str:
    """Position-dependent Caesar shift with modular wrap (prompt-safe by design)."""
    out = []
    for i, ch in enumerate(s):
        k = (i * i if square else i) * sign
        if "a" <= ch <= "z":
            out.append(chr(ord("a") + (ord(ch) - ord("a") + k) % 26))
        elif "A" <= ch <= "Z":
            out.append(chr(ord("A") + (ord(ch) - ord("A") + k) % 26))
        else:
            out.append(ch)
    return "".join(out)


def _trithemius(s: str) -> str:
    return _pos_shift(s, 1, False)


def _invert_trithemius(s: str) -> str:
    return _pos_shift(s, -1, False)


def _quad_shift(s: str) -> str:
    return _pos_shift(s, 1, True)


def _invert_quad_shift(s: str) -> str:
    return _pos_shift(s, -1, True)


WITHHOLD_TRANSFORMATION_FUNCTIONS: dict[str, Callable[[str], str]] = {
    "base64_encoding": lambda s: base64.b64encode(s.encode("utf-8")).decode("utf-8"),
    "hex_encoding": lambda s: s.encode("utf-8").hex(),
    "rot13_cipher": lambda s: codecs.encode(s, "rot_13"),
    "word_to_binary_string": lambda s: " ".join(format(ord(c), "08b") for c in s),
    "atbash_cipher": _atbash,
    "hex_and_reverse": lambda s: s.encode("utf-8").hex()[::-1],
    "base64_and_rot13": lambda s: codecs.encode(
        base64.b64encode(s.encode("utf-8")).decode("utf-8"), "rot_13"
    ),
    "ascii_values_as_string": lambda s: "-".join(str(ord(c)) for c in s),
    "shift_chars_by_position": lambda s: "".join(
        chr(ord(s[i]) + i) for i in range(len(s))
    ),
    "alternate_ascii_add_subtract": lambda s: "".join(
        chr(ord(c) + (1 if i % 2 == 0 else -1)) for i, c in enumerate(s)
    ),
    "duplicate_vowels": lambda s: "".join(
        c * 2 if c.lower() in "aeiou" else c for c in s
    ),
    "reverse_vowel_order": (
        lambda s: (
            lambda vowels: "".join(
                vowels.pop(0) if c.lower() in "aeiou" else c for c in s
            )
        )([c for c in s if c.lower() in "aeiou"][::-1])
    ),
    # Non-involutive, printable-by-construction replacements for the two
    # unbounded-``chr`` ops (see the _VOWELS/_CONSONANTS/_QWERTY note above).
    "vowel_cycle": _cycle_in(_VOWELS, 1),
    "consonant_cycle": _cycle_in(_CONSONANTS, 1),
    "keyboard_shift": _cycle_in(_QWERTY, 1),
}


# --- v2.1.1 training-operator primitives --------------------------------------
#
# See TRAINING_TRANSFORMATION_FUNCTIONS_V2_1_1 below for why these exist and
# for the invertibility rule that constrains their design.


def _vowel_count(s: str) -> int:
    return sum(1 for c in s if c.lower() in "aeiou")


def _shift_letter(ch: str, k: int) -> str:
    """Case-preserving modular letter shift; non-letters pass through.

    Matches the convention of ``_pos_shift`` / ``_vigenere``. Using
    ``ch.lower()`` instead would corrupt any upstream charset-changing
    operator's output -- e.g. composing after ``base64_encoding`` would
    lowercase the base64 alphabet and make the rule non-invertible.
    """
    if "a" <= ch <= "z":
        return chr(ord("a") + (ord(ch) - ord("a") + k) % 26)
    if "A" <= ch <= "Z":
        return chr(ord("A") + (ord(ch) - ord("A") + k) % 26)
    return ch


def _rotate_by_vowel_count(s: str) -> str:
    """Rotate left by the vowel count. Rotation preserves that count, so the
    inverse can recompute the shift from the output."""
    if not s:
        return s
    k = _vowel_count(s) % len(s)
    return s[k:] + s[:k]


def _shift_by_first_letter(s: str) -> str:
    """Caesar-shift every letter EXCEPT the first by the first letter's index.

    Position 0 is deliberately left untouched: shifting it too would make the
    output's first letter ``2*first mod 26``, and 2 is not invertible mod 26,
    so the key could not be recovered.
    """
    if not s:
        return s
    k = ord(s[0].lower()) - 97 if s[0].isalpha() else 0
    return s[0] + "".join(_shift_letter(c, k) for c in s[1:])


def _branch_on_vowel_parity(s: str) -> str:
    """Reverse when the vowel count is odd, rotate left by 2 when even.

    Both branches preserve the vowel count, so the inverse can tell which was
    taken. The even branch exists so the op ALWAYS acts: a plain "reverse only
    when odd" is the identity on ~54% of words, which at depth 1 produces a
    task that teaches nothing.
    """
    if _vowel_count(s) % 2:
        return s[::-1]
    return s[2:] + s[:2] if len(s) > 2 else s


def _reverse_blocks(s: str, k: int) -> str:
    return "".join(s[i : i + k][::-1] for i in range(0, len(s), k))


def _block_size_from_vowels(s: str) -> str:
    """Reverse blocks whose size is derived from the vowel count.

    Keyed off the vowel count rather than the first letter because this op is
    a permutation: it moves the first letter, so a first-letter key would be
    unrecoverable, whereas the vowel count is permutation-invariant.
    """
    return _reverse_blocks(s, (_vowel_count(s) % 3) + 2)


def _grid_rot270(s: str) -> str:
    """Read the grid columns right-to-left, top-to-bottom (270-degree turn)."""
    rows, cols = _grid_dims(len(s))
    grid = [list(s[r * cols : (r + 1) * cols].ljust(cols, "\x00")) for r in range(rows)]
    out = [grid[r][c] for c in range(cols - 1, -1, -1) for r in range(rows)]
    return "".join(out).replace("\x00", "")


def _zigzag_columns(s: str) -> str:
    """Read columns alternately downward and upward (serpentine by column)."""
    rows, cols = _grid_dims(len(s))
    grid = [list(s[r * cols : (r + 1) * cols].ljust(cols, "\x00")) for r in range(rows)]
    out: list[str] = []
    for c in range(cols):
        col = [grid[r][c] for r in range(rows)]
        out.extend(col if c % 2 == 0 else col[::-1])
    return "".join(out).replace("\x00", "")


def _rail_fence_4(s: str) -> str:
    """Rail-fence with 4 rails (the withhold pool uses 3)."""
    rails: list[list[str]] = [[] for _ in range(4)]
    row, step = 0, 1
    for ch in s:
        rails[row].append(ch)
        if row == 0:
            step = 1
        elif row == 3:
            step = -1
        row += step
    return "".join("".join(r) for r in rails)


def _triangle_read(s: str) -> str:
    """Lay the word out in rows of 1, 2, 3, ... chars and read bottom-up."""
    rows, i, width = [], 0, 1
    while i < len(s):
        rows.append(s[i : i + width])
        i += width
        width += 1
    return "".join(reversed(rows))


_BEAUFORT_KEY = "puzzle"


def _beaufort(s: str) -> str:
    """Beaufort cipher: c_i -> (key_i - c_i) mod 26. Self-inverse."""
    out = []
    for i, c in enumerate(s):
        k = ord(_BEAUFORT_KEY[i % len(_BEAUFORT_KEY)]) - 97
        if "a" <= c <= "z":
            out.append(chr(ord("a") + (k - (ord(c) - ord("a"))) % 26))
        elif "A" <= c <= "Z":
            out.append(chr(ord("A") + (k - (ord(c) - ord("A"))) % 26))
        else:
            out.append(c)
    return "".join(out)


def _autokey(s: str, decrypt: bool = False) -> str:
    """Autokey cipher: the plaintext itself extends the key after one seed
    letter, so the shift at position i depends on the plaintext at i-1."""
    out: list[str] = []
    key = "k"
    for c in s:
        if not ("a" <= c <= "z" or "A" <= c <= "Z"):
            out.append(c)
            continue
        # The shift is the PREVIOUS plaintext letter, seeded by "k" for the
        # first one -- so decryption must recover plaintext sequentially.
        k = ord(key[-1]) - 97
        if decrypt:
            plain = _shift_letter(c, -k)
            out.append(plain)
            key += plain.lower()
        else:
            out.append(_shift_letter(c, k))
            key += c.lower()
    return "".join(out)


def _first_primes(n: int) -> list[int]:
    primes: list[int] = []
    cand = 2
    while len(primes) < n:
        if all(cand % p for p in primes if p * p <= cand):
            primes.append(cand)
        cand += 1
    return primes


def _prime_shift(s: str, direction: int) -> str:
    """Shift the letter at position i by the i-th prime."""
    primes = _first_primes(len(s))
    return "".join(_shift_letter(c, direction * primes[i]) for i, c in enumerate(s))


def _invert_rotate_by_vowel_count(s: str) -> str:
    if not s:
        return s
    k = _vowel_count(s) % len(s)
    return s[-k:] + s[:-k] if k else s


def _invert_shift_by_first_letter(s: str) -> str:
    if not s:
        return s
    k = ord(s[0].lower()) - 97 if s[0].isalpha() else 0
    return s[0] + "".join(_shift_letter(c, -k) for c in s[1:])


def _invert_branch_on_vowel_parity(s: str) -> str:
    if _vowel_count(s) % 2:
        return s[::-1]
    return s[-2:] + s[:-2] if len(s) > 2 else s


def _alternating_caesar(s: str, direction: int) -> str:
    """+3 on even positions, -3 on odd ones."""
    return "".join(
        _shift_letter(c, direction * (3 if i % 2 == 0 else -3)) for i, c in enumerate(s)
    )


# --- Operator sets exposed via the CLI ----------------------------------------

# Math / geometry candidate operators, under evaluation for a future withhold
# pool. They live in their OWN registry, deliberately NOT in
# ``WITHHOLD_TRANSFORMATION_FUNCTIONS``: ``WITHHOLD_OPERATORS`` below is derived
# from that dict's keys, so adding them there would silently widen the shipped
# v1 ``withhold`` pool (and ``ALL_OPERATORS``) and change what existing configs
# generate. Registering them here makes them *applicable and invertible*
# without placing them in any sampling pool -- a rule must name them explicitly.
#
#   affine_index  i -> (a*i + b) mod n           coordinate map on Z_n
#   grid_rot90    (i,j) -> (j, r-1-i)            90-degree grid rotation
#   spiral        inward clockwise grid read
#   trithemius    c_i -> (c_i + i)   mod 26      position-dependent shift
#   quad_shift    c_i -> (c_i + i^2) mod 26      nonlinear in position
#
# The three geometry ops are pure permutations (multiset-preserving, so the
# solver can see it is positional) and length-dependent (the index map differs
# per word length, so a single map cannot be read off the few-shot demos).
# The two modular ops are polyalphabetic: a chain of monoalphabetic
# substitutions composes to ONE letter->letter table, and adding either of
# these breaks that collapse for the whole rule.
MATH_TRANSFORMATION_FUNCTIONS: dict[str, Callable[[str], str]] = {
    # -- coordinate / grid permutations (multiset-preserving, length-dependent)
    "affine_index": _affine_index,
    "grid_rot90": _grid_rot90,
    "grid_transpose": _grid_transpose,
    "grid_diagonal_read": _grid_diagonal_read,
    "spiral": _spiral,
    "boustrophedon": _boustrophedon,
    "rail_fence_3": _rail_fence_3,
    "block_reverse_growing": _block_reverse_growing,
    # -- polyalphabetic modular shifts (break monoalphabetic collapse)
    "trithemius": _trithemius,
    "quad_shift": _quad_shift,
    "vigenere_math": lambda s: _vigenere(s, 1),
    "fibonacci_shift": lambda s: _fib_shift(s, 1),
}

# --- v2.1.1 TRAINING operators ---------------------------------------------
#
# The 20-operator ``withhold_v2_1_1`` pool is a PERMANENT held-out eval set:
# it is what measures whether training transfers to operator families the
# model has never seen. Training data must therefore never draw from it, so
# these 12 operators are deliberately disjoint from all 20 -- verified by
# name, by per-word output, and against all 400 depth-2 compositions of the
# withhold pool (closest match 33%, no duplicates).
#
# They teach the same three mechanisms the eval pool tests, via different
# primitives:
#
#   content-dependent (4): the transform's PARAMETER is derived from the word
#       itself, so there is no fixed map to read off the few-shot demos.
#   geometry (4):          length-dependent index permutations with readout
#       orders distinct from the eval set's.
#   polyalphabetic (4):    position-dependent letter shifts, which break the
#       monoalphabetic collapse that made deep v2.1 rules trivial.
#
# INVERTIBILITY RULE: a content-dependent PERMUTATION may only key its
# parameter off a permutation-INVARIANT property (vowel count, length),
# because the inverse must recompute that parameter from the TRANSFORMED
# string. ``block_size_from_letter`` keys off vowel count, not the first
# letter, for exactly this reason -- the op moves the first letter, so a
# first-letter key would be unrecoverable. Content-dependent SUBSTITUTIONS
# cannot satisfy this at all (they destroy every letter-derived invariant),
# which is why all 4 polyalphabetic ops below use positional keys.
#
# Own registry, like MATH_TRANSFORMATION_FUNCTIONS: no shipped pool derives
# from it, so adding these cannot change what any existing config generates.
TRAINING_TRANSFORMATION_FUNCTIONS_V2_1_1: dict[str, Callable[[str], str]] = {
    # -- content-dependent (parameter read off the word)
    "rotate_by_vowel_count": _rotate_by_vowel_count,
    "shift_by_first_letter": _shift_by_first_letter,
    "branch_on_vowel_parity": _branch_on_vowel_parity,
    "block_size_from_letter": _block_size_from_vowels,
    # -- geometry (length-dependent permutations)
    "grid_rot270": _grid_rot270,
    "zigzag_columns": _zigzag_columns,
    "rail_fence_4": _rail_fence_4,
    "triangle_read": _triangle_read,
    # -- polyalphabetic (position-dependent shifts)
    "beaufort": _beaufort,
    "autokey": _autokey,
    "prime_shift": lambda s: _prime_shift(s, 1),
    "alternating_caesar": lambda s: _alternating_caesar(s, 1),
}

BASIC_OPERATORS = ["shift", "mirror", "rotate"]
EXTENDED_OPERATORS = BASIC_OPERATORS + list(TRANSFORMATION_FUNCTIONS.keys())
WITHHOLD_OPERATORS = list(WITHHOLD_TRANSFORMATION_FUNCTIONS.keys())
ALL_OPERATORS = EXTENDED_OPERATORS + WITHHOLD_OPERATORS

# --- v2 operator sets: property-matched train/withhold split ----------------
# v1 split was confounded: all training ops were length-preserving + letter-only,
# all withhold ops were encodings (expanding + charset-changing). v2 reshuffles
# so both sets contain a mix of permutations and encodings.
#
# Train v2 (14 ops):
#   Class A — length-preserving, letter-only (10): the 3 basic + 7 transforms
#   Class B — expanding, letter-only (1): duplicate_vowels
#   Class D — expanding, charset-change (3): base64_encoding, hex_encoding,
#       ascii_values_as_string
#
# Withhold v2 (8 ops):
#   Class A — length-preserving, letter-only (4): rot13_cipher, atbash_cipher,
#       alternate_ascii_add_subtract, reverse_vowel_order
#   Class C — length-preserving, charset-change (1): shift_chars_by_position
#   Class D — expanding, charset-change (3): word_to_binary_string,
#       hex_and_reverse, base64_and_rot13

EXTENDED_OPERATORS_V2 = BASIC_OPERATORS + [
    # Class A (permutations) — same 7 as v1
    "swap_adjacent_char_pairs",
    "reverse_blocks_of_3",
    "weave_from_ends",
    "move_middle_to_front",
    "move_middle_to_end",
    "interleave_split_string",
    "pull_outer_to_middle",
    # Class B (expanding, letter-only)
    "duplicate_vowels",
    # Class D (expanding, charset-change)
    "base64_encoding",
    "hex_encoding",
    "ascii_values_as_string",
]

WITHHOLD_OPERATORS_V2 = [
    # Class A (length-preserving, letter-only) -- involutions
    "rot13_cipher",
    "atbash_cipher",
    "reverse_vowel_order",
    # Class B (length-preserving, letter-only) -- NON-involutive alphabet
    # cycles. Without these the length-preserving pool is all involutions,
    # so deep compositions collapse (~21% of depth-2 rules became the
    # identity). See the _VOWELS/_CONSONANTS/_QWERTY note above.
    "vowel_cycle",
    "consonant_cycle",
    "keyboard_shift",
    # Class D (expanding, charset-change)
    "word_to_binary_string",
    "hex_and_reverse",
    "base64_and_rot13",
]
# REMOVED from the sampling pool (still registered above so existing v2.0 /
# v2.1 rules referencing them remain applicable/invertible):
#   alternate_ascii_add_subtract, shift_chars_by_position
# Both did raw ``chr(ord(c) +/- n)`` with no modular wrap, so on ordinary
# words they emitted control bytes (\x7f, C1) and backticks straight into
# the prompt -- corrupting ~1/3 of the v2.1 withhold rows.

# v2.1.1 withhold pool: a harder, better-calibrated successor to withhold_v2.
# ``withhold_v2`` is FROZEN -- it must keep reproducing the shipped v2.1 splits,
# so this is a new set rather than an edit.
#
# Carried over (8): every withhold_v2 operator except word_to_binary_string,
# which measured 0.952 at the base model and +0.012 across a whole training run
# (22 of its 28 rows pinned 1.00 -> 1.00, including depth 8-10). It is at the
# ceiling and cannot discriminate, so it is dropped from sampling -- but stays
# registered/invertible for the shipped v2.1 rules that use it.
# base64_and_rot13 is KEPT despite ~38% of its v2.1 rows never being solved:
# a floor task is dormant (a stronger model lights it up), unlike a ceiling
# task which is permanently dead.
#
# Added (12): the math/geometry operators. Probe-measured on the base model at
# depth 3 against a 3-substitution control:
#     affine_index 0.38   spiral 0.40   rail_fence_3 0.42
#     grid_transpose 0.50 vigenere_math 0.50  block_reverse_growing 0.52
#     grid_diagonal_read 0.52  grid_rot90 0.55
#     fibonacci_shift 0.63  quad_shift 0.63  boustrophedon 0.68
#     trithemius 0.73
# They exist because withhold_v2 is bimodal -- its six substitution operators
# cluster at 0.75-0.80 and base64_and_rot13 sits alone at 0.29, with nothing
# between. Worse, substitutions COMPOSE to a single letter->letter table, so a
# depth-10 chain of them is no harder than depth-2 (measured: a 6-substitution
# composite is still monoalphabetic). Every operator below breaks that
# collapse: the permutations are length-dependent (a different index map per
# word length, so no single map can be read off the few-shot demos) and the
# modular shifts are polyalphabetic.
WITHHOLD_OPERATORS_V2_1_1 = [
    op for op in WITHHOLD_OPERATORS_V2 if op != "word_to_binary_string"
] + list(MATH_TRANSFORMATION_FUNCTIONS)

# v2.1.1 TRAINING pool: the 14 extended_v2 operators plus the 12 new ones.
# ``withhold_v2_1_1`` is treated as a PERMANENT held-out set -- training on it
# would retire it as a transfer probe and make every checkpoint measured
# against it incomparable to future ones -- so this pool is disjoint from it
# by construction (extended_v2 and withhold_v2_1_1 share no operator, and the
# 12 additions were chosen to overlap neither).
EXTENDED_OPERATORS_V2_1_1 = EXTENDED_OPERATORS_V2 + list(
    TRAINING_TRANSFORMATION_FUNCTIONS_V2_1_1
)

OPERATOR_SETS = {
    "basic": BASIC_OPERATORS,
    "extended": EXTENDED_OPERATORS,
    "withhold": WITHHOLD_OPERATORS,
    "extended_v2": EXTENDED_OPERATORS_V2,
    "withhold_v2": WITHHOLD_OPERATORS_V2,
    "withhold_v2_1_1": WITHHOLD_OPERATORS_V2_1_1,
    "extended_v2_1_1": EXTENDED_OPERATORS_V2_1_1,
    "all": ALL_OPERATORS,
}

# Operators whose output is longer than the input; composing multiple in
# one rule multiplies string length. Callers building deep rules cap them
# via ``generate_random_rule(max_expanding_ops=...)`` to keep prompt size
# bounded.
# Fewest worked examples a task may show. Below this there is not enough
# evidence to infer the rule, so the task is unsolvable as posed rather than
# merely hard. Only ``few_shot_jitter`` can push a count downward, so this is
# the floor it clamps to.
MIN_FEW_SHOT_EXAMPLES = 3

EXPANDING_OPERATORS: set[str] = {
    "word_to_binary_string",
    "ascii_values_as_string",
    "hex_encoding",
    "hex_and_reverse",
    "base64_encoding",
    "base64_and_rot13",
    "duplicate_vowels",
}


# --- Inverses: enables round-trip reversibility checking on generated rules ---
#
# Every op used by v2 (both extended_v2 and withhold_v2) has an inverse below.
# `generate_one_task` uses these to verify that `apply_inverse_transform(
# apply_transform(word, rule), rule) == word` for every emitted task,
# resampling the rule when a composition is non-invertible.
#
# For pure permutation ops (7 of them), inverses are computed generically via
# `_invert_permutation` — apply forward to a tagged length-n string, read the
# resulting index permutation, invert it. Avoids writing 7 hand-rolled inverses.


def _invert_permutation(forward_fn: Callable[[str], str], s: str) -> str:
    """Invert a length-preserving permutation op by tagging positions."""
    n = len(s)
    if n == 0:
        return s
    tagged = "".join(chr(0x100 + i) for i in range(n))
    permuted = forward_fn(tagged)
    if len(permuted) != n:
        raise ValueError(f"{forward_fn.__name__} is not length-preserving on n={n}")
    perm_from = [ord(c) - 0x100 for c in permuted]
    result: list[Optional[str]] = [None] * n
    for j, i in enumerate(perm_from):
        result[i] = s[j]
    if any(c is None for c in result):
        raise ValueError(f"{forward_fn.__name__} is not a bijection on n={n}")
    return "".join(result)  # type: ignore[arg-type]


def _invert_mirror(s: str, copy_input: bool) -> str:
    if copy_input:
        # forward: s + s[::-1]  =>  even length, first half is the original
        return s[: len(s) // 2]
    # forward: reverse; self-inverse
    return s[::-1]


def _invert_rotate(s: str, rotate: int) -> str:
    if not s:
        return s
    return _rotate_char(s, (-rotate) % len(s))


def _invert_duplicate_vowels(s: str) -> str:
    # Forward doubles each vowel (case-insensitive). Inverse halves each
    # maximal vowel run. Non-vowels pass through. Raises if a vowel run has
    # odd length (not in the image of the forward op).
    out: list[str] = []
    i = 0
    while i < len(s):
        c = s[i]
        if c.lower() in "aeiou":
            j = i
            while j < len(s) and s[j].lower() in "aeiou" and s[j] == c:
                j += 1
            run = j - i
            if run % 2 != 0:
                raise ValueError(
                    f"duplicate_vowels inverse: odd vowel run of {c!r} at {i}"
                )
            out.append(c * (run // 2))
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _invert_alternate_ascii_add_subtract(s: str) -> str:
    # Forward: chr(ord(c) + (1 if even else -1)). Inverse flips the sign.
    return "".join(chr(ord(c) + (-1 if i % 2 == 0 else 1)) for i, c in enumerate(s))


def _invert_shift_chars_by_position(s: str) -> str:
    # Forward: chr(ord(s[i]) + i). Inverse subtracts i.
    return "".join(chr(ord(s[i]) - i) for i in range(len(s)))


def _invert_reverse_vowel_order(s: str) -> str:
    # Forward: reverses the sequence of vowels in place. Applying it again
    # reverses that reversal — self-inverse.
    vowels = [c for c in s if c.lower() in "aeiou"][::-1]
    return "".join(vowels.pop(0) if c.lower() in "aeiou" else c for c in s)


# Inverses keyed by op name. Basic ops (shift/mirror/rotate) are parametric
# and dispatched in `apply_inverse_transform`; everything else is a fixed
# function of the string.
INVERSE_FUNCTIONS: dict[str, Callable[[str], str]] = {
    # extended_v2 permutations
    "swap_adjacent_char_pairs": _swap_adj,  # self-inverse
    "reverse_blocks_of_3": _reverse_blocks_of_3,  # self-inverse
    "weave_from_ends": lambda s: _invert_permutation(_weave_from_ends, s),
    "move_middle_to_front": lambda s: _invert_permutation(_move_middle_to_front, s),
    "move_middle_to_end": lambda s: _invert_permutation(_move_middle_to_end, s),
    "interleave_split_string": lambda s: _invert_permutation(_interleave_split, s),
    "pull_outer_to_middle": lambda s: _invert_permutation(_pull_outer_to_middle, s),
    # extended_v2 expanding
    "duplicate_vowels": _invert_duplicate_vowels,
    "base64_encoding": lambda s: base64.b64decode(s.encode("utf-8")).decode("utf-8"),
    "hex_encoding": lambda s: bytes.fromhex(s).decode("utf-8"),
    "ascii_values_as_string": lambda s: "".join(chr(int(x)) for x in s.split("-")),
    # withhold_v2 length-preserving letter-only
    "rot13_cipher": lambda s: codecs.encode(s, "rot_13"),  # self-inverse
    "atbash_cipher": _atbash,  # self-inverse
    "alternate_ascii_add_subtract": _invert_alternate_ascii_add_subtract,
    "reverse_vowel_order": _invert_reverse_vowel_order,  # self-inverse
    # Inverse = one step BACKWARD along the same alphabet.
    "vowel_cycle": _cycle_in(_VOWELS, -1),
    "consonant_cycle": _cycle_in(_CONSONANTS, -1),
    "keyboard_shift": _cycle_in(_QWERTY, -1),
    # withhold_v2 length-preserving charset-change
    "shift_chars_by_position": _invert_shift_chars_by_position,
    # withhold_v2 expanding
    "word_to_binary_string": lambda s: "".join(chr(int(b, 2)) for b in s.split(" ")),
    "hex_and_reverse": lambda s: bytes.fromhex(s[::-1]).decode("utf-8"),
    "base64_and_rot13": lambda s: base64.b64decode(
        codecs.encode(s, "rot_13").encode("utf-8")
    ).decode("utf-8"),
    # Math / geometry candidates (MATH_TRANSFORMATION_FUNCTIONS). Listed here so
    # a rule naming them is invertible; they are in no sampling pool.
    "affine_index": _invert_affine_index,
    "grid_rot90": _invert_grid_rot90,
    "spiral": _invert_spiral,
    "trithemius": _invert_trithemius,
    "quad_shift": _invert_quad_shift,
    # Pure position permutations: _invert_permutation derives the inverse from
    # the forward map itself (position tagging), so the two cannot drift apart.
    "grid_transpose": lambda s: _invert_permutation(_grid_transpose, s),
    "grid_diagonal_read": lambda s: _invert_permutation(_grid_diagonal_read, s),
    "boustrophedon": lambda s: _invert_permutation(_boustrophedon, s),
    "rail_fence_3": lambda s: _invert_permutation(_rail_fence_3, s),
    "block_reverse_growing": lambda s: _invert_permutation(_block_reverse_growing, s),
    "vigenere_math": lambda s: _vigenere(s, -1),
    "fibonacci_shift": lambda s: _fib_shift(s, -1),
    # v2.1.1 training operators (TRAINING_TRANSFORMATION_FUNCTIONS_V2_1_1).
    # The content-dependent permutations recompute their parameter from the
    # TRANSFORMED string, which is sound only because vowel count survives
    # any permutation -- see the registry's invertibility note.
    # NOTE the content-dependent four do NOT use _invert_permutation: it
    # derives the index map by running the forward op on a synthetic tagged
    # string, which contains no vowels, so it would compute the map for
    # vowel_count == 0 and silently return the wrong inverse.
    "rotate_by_vowel_count": _invert_rotate_by_vowel_count,
    "shift_by_first_letter": _invert_shift_by_first_letter,
    "branch_on_vowel_parity": _invert_branch_on_vowel_parity,
    "block_size_from_letter": _block_size_from_vowels,  # self-inverse
    "grid_rot270": lambda s: _invert_permutation(_grid_rot270, s),
    "zigzag_columns": lambda s: _invert_permutation(_zigzag_columns, s),
    "rail_fence_4": lambda s: _invert_permutation(_rail_fence_4, s),
    "triangle_read": lambda s: _invert_permutation(_triangle_read, s),
    "beaufort": _beaufort,  # self-inverse
    "autokey": lambda s: _autokey(s, decrypt=True),
    "prime_shift": lambda s: _prime_shift(s, -1),
    "alternating_caesar": lambda s: _alternating_caesar(s, -1),
}


def apply_inverse_transform(encoded: str, rule: Sequence[Sequence[Any]]) -> str:
    """Apply the inverse of ``rule`` to ``encoded``, recovering the original word.

    Steps are undone in reverse order. Raises ``ValueError`` (or the codec's
    own decode error) if any step's inverse fails; ``generate_one_task``
    treats this as "rule not reversible on this word" and resamples.
    """
    out = encoded
    for op, param in reversed(list(rule)):
        if op == "shift":
            out = _shift_char(out, (-param) % _NUM_LETTERS)
        elif op == "mirror":
            out = _invert_mirror(out, param)
        elif op == "rotate":
            out = _invert_rotate(out, param)
        elif op in INVERSE_FUNCTIONS:
            out = INVERSE_FUNCTIONS[op](out)
        else:
            raise ValueError(f"no inverse defined for operator {op!r}")
    return out


def check_rule_reversible(word: str, rule: Sequence[Sequence[Any]]) -> bool:
    """Return True iff apply_inverse_transform(apply_transform(word, rule), rule) == word."""
    try:
        encoded = apply_transform(word, rule)
        recovered = apply_inverse_transform(encoded, rule)
    except Exception:  # decode/permutation errors mean not-reversible
        return False
    return recovered == word


# Characters a transform output must never contain, because the output is
# embedded verbatim in the prompt:
#   * anything outside printable ASCII (0x20-0x7e) -- control/C1 bytes are
#     invisible, survive JSON but not human/model reading, and can be mangled
#     in transport;
#   * a backtick, which terminates the markdown span the templates wrap
#     words in (`` `i`mdt` `` renders as garbage).
# Ops doing raw ``chr(ord(c) +/- n)`` without a modular wrap (e.g.
# ``shift_chars_by_position``, ``alternate_ascii_add_subtract``) escape the
# safe range on ordinary words -- ``'a' - 1`` is a backtick.
_PROMPT_SAFE = frozenset(chr(c) for c in range(0x20, 0x7F)) - {"`"}


def check_rule_nondegenerate(word: str, rule: Sequence[Sequence[Any]]) -> bool:
    """Return True iff the rule actually CHANGES ``word``.

    An identity rule (e.g. ``rot13 . rot13``, or any involution applied to
    itself) is perfectly reversible and perfectly printable, so neither
    sibling check rejects it -- but the resulting puzzle shows
    ``input -> input`` in every demo and asks for the word already printed
    in the question. It is free score, and worse, it teaches "when unsure,
    echo the input" (which also collects the full step-efficiency bonus).
    """
    try:
        return apply_transform(word, rule) != word
    except Exception:
        return False


def check_rule_prompt_safe(word: str, rule: Sequence[Sequence[Any]]) -> bool:
    """Return True iff ``apply_transform(word, rule)`` is safe to put in a prompt.

    Companion to :func:`check_rule_reversible`: reversibility is about the
    rule being well-defined, this is about the *rendered* puzzle being
    readable. A rule can be perfectly invertible and still emit control
    characters, which silently corrupts the prompt.
    """
    try:
        encoded = apply_transform(word, rule)
    except Exception:
        return False
    return all(c in _PROMPT_SAFE for c in encoded)


def apply_transform(word: str, rule: Sequence[Sequence[Any]]) -> str:
    """Apply a sequence of (op_name, param) transformations to ``word``.

    The signature accepts any 2-element iterable per rule step so
    JSON-round-tripped rules (``list[list[...]]``) work unchanged.
    """
    out = word
    for op, param in rule:
        if op == "shift":
            out = _shift_char(out, param)
        elif op == "mirror":
            out = _mirror_char(out, param)
        elif op == "rotate":
            out = _rotate_char(out, param)
        elif op in TRANSFORMATION_FUNCTIONS:
            out = TRANSFORMATION_FUNCTIONS[op](out)
        elif op in WITHHOLD_TRANSFORMATION_FUNCTIONS:
            out = WITHHOLD_TRANSFORMATION_FUNCTIONS[op](out)
        elif op in MATH_TRANSFORMATION_FUNCTIONS:
            out = MATH_TRANSFORMATION_FUNCTIONS[op](out)
        elif op in TRAINING_TRANSFORMATION_FUNCTIONS_V2_1_1:
            out = TRAINING_TRANSFORMATION_FUNCTIONS_V2_1_1[op](out)
        else:
            raise ValueError(f"unknown operator {op!r}")
    return out


def _random_op_param(op: str, max_rotate: int, rng: random.Random) -> tuple[str, Any]:
    """Sample a random parameter for one operator."""
    if op == "shift":
        return (op, rng.randint(1, _NUM_LETTERS - 1))
    if op == "mirror":
        return (op, rng.choice([True, False]))
    if op == "rotate":
        return (op, rng.randint(1, max(1, max_rotate)))
    if (
        op in TRANSFORMATION_FUNCTIONS
        or op in WITHHOLD_TRANSFORMATION_FUNCTIONS
        or op in MATH_TRANSFORMATION_FUNCTIONS
        or op in TRAINING_TRANSFORMATION_FUNCTIONS_V2_1_1
    ):
        return (op, None)
    raise ValueError(f"unknown operator {op!r}")


def generate_random_rule(
    num_transformations: int,
    max_rotate: int,
    operators: Sequence[str] = BASIC_OPERATORS,
    rng: Optional[random.Random] = None,
    max_expanding_ops: Optional[int] = None,
) -> list[tuple[str, Any]]:
    """Build a random rule = list of ``(op_name, param)`` of length ``num_transformations``.

    Operators are cycled + shuffled so each op appears roughly equally often.

    ``max_expanding_ops`` caps the number of ``EXPANDING_OPERATORS`` in
    the rule to bound prompt length; sampling picks uniformly in
    ``[0, max_expanding_ops]`` positions before filling in the operator
    names.
    """
    r: random.Random = rng if rng is not None else random.Random()
    if max_expanding_ops is not None:
        expanding_in_pool = [op for op in operators if op in EXPANDING_OPERATORS]
        non_expanding_in_pool = [
            op for op in operators if op not in EXPANDING_OPERATORS
        ]
        if not non_expanding_in_pool and max_expanding_ops < num_transformations:
            raise ValueError(
                f"max_expanding_ops={max_expanding_ops} but every operator in "
                f"the pool is expanding; can't build a rule of length "
                f"{num_transformations}."
            )
        # Choose how many expanding positions first (uniform in
        # [0, cap]) so "cap of 1" means "0 or 1", not "always 1".
        upper = min(max_expanding_ops, len(expanding_in_pool), num_transformations)
        num_expanding = r.randint(0, upper) if upper > 0 else 0
        chosen_ops: list[str] = []
        if num_expanding > 0:
            chosen_ops.extend(r.choices(expanding_in_pool, k=num_expanding))
        remaining = num_transformations - len(chosen_ops)
        if remaining > 0:
            if not non_expanding_in_pool:
                raise ValueError(
                    f"Cannot fill {remaining} non-expanding slots: pool "
                    f"has no non-expanding operators."
                )
            chosen_ops.extend(r.choices(non_expanding_in_pool, k=remaining))
        r.shuffle(chosen_ops)
    else:
        pool = list(operators) * math.ceil(num_transformations / max(1, len(operators)))
        r.shuffle(pool)
        chosen_ops = pool[:num_transformations]
    return [_random_op_param(op, max_rotate, r) for op in chosen_ops]


_TEMPLATES = [
    # 1 -- Codebreaker
    """You are a master codebreaker, renowned for deciphering the most enigmatic patterns. The following messages have been intercepted, each containing a word that has been systematically scrambled. Your mission is to crack the code.

Here are the successful decryptions from your field notes:
{examples_block}

{main_question}

{explanation_request}Put your final answer in the following format:
$\\text{{your answer}}$""",
    # 2 -- Linguistic Analyst
    """You are a linguistic analyst specializing in algorithmic word transformations. Your task is to identify the pattern in the following examples and apply it to a new word.

Observe the following data points:
{examples_block}

{main_question}

{explanation_request}Put your final answer in the following format:
$\\text{{your answer}}$""",
    # 3 -- Direct Pattern Finder
    """A consistent rule was used to transform an original word into an encrypted word.

Here are {num_examples} examples of this transformation:
{examples_block}

{main_question}

{explanation_request}Put your final answer in the following format:
$\\text{{your answer}}$""",
    # 4 -- Software Engineer
    """You are a senior software engineer debugging a legacy system. You've found a function, `scramble_word()`, that is undocumented. By observing its input and output, you must determine the algorithm it uses.

Here is the data from your test runs:
{examples_block}

{main_question}

{explanation_request}Put your final answer in the following format:
$\\text{{your answer}}$""",
    # 5 -- Game Show
    """Hello and welcome back to "Word Wizards!" Our contestant has to figure out the secret rule that's changing our words!

Let's look at the board:
{examples_block}

{main_question}

{explanation_request}Put your final answer in the following format:
$\\text{{your answer}}$""",
    # 6 -- Historical Cryptographer
    """You are a historian specializing in historical cryptography. A newly discovered manuscript contains a peculiar substitution cipher that seems to operate on a character-level principle rather than simple letter replacement.

Your analysis has so far revealed these pairings:
{examples_block}

{main_question}

{explanation_request}Put your final answer in the following format:
$\\text{{your answer}}$""",
    # 7 -- AI Trainer
    """// SYSTEM PROMPT: PATTERN RECOGNITION TEST

The following examples demonstrate a consistent string transformation rule. Your task is to infer the rule and apply it.

// TRAINING DATA:
{examples_block}

{main_question}

Put your final answer in the following format:
$\\text{{your answer}}$""",
    # 8 -- Xenolinguist
    """You are a xenolinguist aboard the starship "Odyssey," deciphering a communication from a new non-humanoid species. The signal appears to be a language, but each word is passed through a strange structural filter.

Our translation matrix has established the following relationships:
{examples_block}

{main_question}

{explanation_request}Put your final answer in the following format:
$\\text{{your answer}}$""",
    # 9 -- Alchemist
    """You are an apprentice to a master alchemist. You've found a grimoire detailing a "Transmutation of Essence" spell, which changes the form of written words. To understand the spell, you are studying the master's notes.

The notes show the following transmutations:
{examples_block}

{main_question}

{explanation_request}Put your final answer in the following format:
$\\text{{your answer}}$""",
    # 10 -- Minimalist
    """Infer a function `f(x)` given the following examples.

{examples_block}

{main_question}

{explanation_request}Put your final answer in the following format:
$\\text{{your answer}}$""",
]


_QUESTION_ENCRYPT = {  # reverse_task=True (given original, find encrypted)
    1: "Your next mission is to encrypt the word **[{w}]**. What is the resulting coded message?",
    2: "Now, apply that rule to transform the original word `{w}`.",
    3: "What is the encrypted word for `{w}`?",
    4: "What would be the output if the input string was `{w}`?",
    5: "Alright, contestant, for the grand prize: if our wizards get the word `{w}`, what will they turn it into?",
    6: "You must now encode the original text `{w}` using the same principle. What is the resulting coded text?",
    8: "A new concept, `{w}`, must be transmitted. How will it be filtered into a signal?",
    9: "You must now perform a Transmutation of Essence on the substance `{w}`. What is its new form?",
    10: "Given `x = '{w}'`, find `f(x)`.",
}

_QUESTION_DECRYPT = {  # reverse_task=False (given encrypted, find original)
    1: "Now, a new coded message has arrived: **[{w}]**. What is the original word?",
    2: "Use the rule to determine the original word for `{w}`.",
    3: "What is the original word for `{w}`?",
    4: "A new output, `{w}`, was generated by the function. What was the original input?",
    5: "Alright, contestant, for the grand prize: if our wizards give you `{w}`, what word did they start with? Think fast!",
    6: "A key fragment of the coded text reads: `{w}`. What is the original text for this fragment?",
    8: "A priority message has just been received: `{w}`. The ship's survival depends on your translation. What is the original concept?",
    9: "You find a smudged recipe that requires the original essence of a substance written as `{w}`. What was the original substance?",
    10: "Given `y = '{w}'`, find `x` such that `f(x) = y`.",
}

_EXPLANATION_TAIL = {
    1: "Before you give the answer, briefly outline the method you've uncovered.",
    2: "Your response must first describe the transformation rule before providing the final answer.",
    3: "First, state the rule, then provide the answer.",
    4: "First, describe the algorithm you've reverse-engineered, then state the answer.",
    5: "First, tell us the secret rule, and then give us the word!",
    6: "First, describe the character-level principle you have deciphered, then state the answer.",
    8: "First, document the structural filter's logic, then provide the answer.",
    9: "First, describe the principle of this Transmutation of Essence, then reveal the answer.",
    10: "First, define the function, then provide the answer.",
}


def _example_block(examples: Sequence[tuple[str, str]], template_id: int) -> str:
    if template_id == 7:
        rows = [f'  {{ "input": "{raw}", "output": "{enc}" }}' for raw, enc in examples]
        return "[\n" + ",\n".join(rows) + "\n]"
    lines: list[str] = []
    if template_id == 1:
        for i, (raw, enc) in enumerate(examples):
            lines.append(
                f"- **Example {i}:** `[{raw}]` was found to be the original for `[{enc}]`."
            )
    elif template_id == 2:
        for i, (raw, enc) in enumerate(examples):
            lines.append(f"{i + 1}.  Original: `{raw}`, Transformed: `{enc}`")
    elif template_id == 3:
        for raw, enc in examples:
            lines.append(f"- `{raw}` -> `{enc}`")
    elif template_id == 4:
        for raw, enc in examples:
            lines.append(f"- Input: `{raw}` -> Output: `{enc}`")
    elif template_id == 5:
        for raw, enc in examples:
            lines.append(f"- Our wizards turned `{raw}` into `{enc}`!")
    elif template_id == 6:
        for raw, enc in examples:
            lines.append(f"- Original Text: `{raw}`, Coded Text: `{enc}`")
    elif template_id == 8:
        for raw, enc in examples:
            lines.append(f"- Concept `{raw}` is transmitted as signal `{enc}`.")
    elif template_id == 9:
        for raw, enc in examples:
            lines.append(
                f"- {raw.capitalize()} (`{raw}`) becomes {enc.capitalize()} (`{enc}`)."
            )
    elif template_id == 10:
        for raw, enc in examples:
            lines.append(f"f('{raw}') = '{enc}'")
    else:
        raise ValueError(f"invalid template_id {template_id}")
    return "\n".join(lines)


_ANSWER_FORMAT_TRAILER = (
    "Put your final answer in the following format:\n$\\text{your answer}$"
)


def generate_prompt(
    examples: Sequence[tuple[str, str]],
    task_word: str,
    force_template_id: Optional[int] = None,
    explain_encryption: bool = False,
    reverse_task: bool = False,
    rng: Optional[random.Random] = None,
    include_answer_format: bool = True,
) -> tuple[str, int]:
    """Format one prompt.

    Returns ``(prompt_text, template_id)``. Template id is 1..10.

    Args:
      examples: (original, encrypted) demonstration pairs.
      task_word: the input to include in the main question.
      force_template_id: pin a template (1..10) instead of sampling.
      explain_encryption: if True, ask the model to describe the rule
        before answering.
      reverse_task: if False, question is "given encrypted, find original";
        if True, "given original, find encrypted".
      rng: seedable random source.
      include_answer_format: if True (default), every template ends
        with the ``Put your final answer in the following format:
        $\\text{your answer}$`` trailer. Direct-answer eval callers
        MUST keep True; agent-mode callers (structured tool-call
        submit) SHOULD pass False.
    """
    r: random.Random = rng if rng is not None else random.Random()
    if force_template_id is not None:
        if not 1 <= force_template_id <= len(_TEMPLATES):
            raise ValueError(f"force_template_id must be in 1..{len(_TEMPLATES)}")
        tid = force_template_id
    else:
        tid = r.randint(1, len(_TEMPLATES))

    if tid == 7:
        # Template 7 embeds the explain-encryption instruction inline.
        instr = (
            "Determine the transformed output."
            if reverse_task
            else "Determine the original input."
        )
        if explain_encryption:
            expl = "// TASK: Your response should first describe the inferred rule, then provide the answer on a new line."
            main_q = (
                f"// TEST INPUT:\n`{task_word}`\n\n// REQUIRED OUTPUT:\n{expl}\n{instr}"
            )
        else:
            main_q = f"// TEST INPUT:\n`{task_word}`\n\n// REQUIRED OUTPUT:\n{instr}"
        explanation_tail = ""
    else:
        table = _QUESTION_ENCRYPT if reverse_task else _QUESTION_DECRYPT
        main_q = table[tid].format(w=task_word)
        explanation_tail = _EXPLANATION_TAIL[tid] + "\n\n" if explain_encryption else ""

    prompt = _TEMPLATES[tid - 1].format(
        examples_block=_example_block(examples, tid),
        main_question=main_q,
        explanation_request=explanation_tail,
        num_examples=len(examples),
    )
    if not include_answer_format:
        # Strip the trailing answer-format block; every template ends
        # with it, so locate + cut and rstrip to end cleanly.
        idx = prompt.rfind(_ANSWER_FORMAT_TRAILER)
        if idx >= 0:
            prompt = prompt[:idx].rstrip()
    return prompt, tid


@dataclasses.dataclass(kw_only=True)
class PuzzleTask:
    """One synthesized word-puzzle task."""

    prompt: str
    expected_answer: str
    original_word: str
    encrypted_word: str
    few_shot_examples: list[tuple[str, str]]  # list of (original, encrypted)
    rule: list[tuple[str, Any]]  # list of (op_name, param)
    template_id: int
    num_transformations: int
    num_few_shot_examples: int
    explain_encryption: bool
    reverse_task: bool  # False: encrypted->original; True: original->encrypted
    # Optional caller-supplied metadata serialized as top-level ``tags``
    # on the JSONL record; survives into eval output for per-slice
    # breakdowns. Values must be JSON-serializable.
    tags: dict[str, Any] = dataclasses.field(default_factory=dict)
    # Optional pool of extra words disjoint from ``few_shot_examples``
    # and ``original_word``; populated when the task is generated with
    # ``include_retry_metadata=True`` for the agent-mode retry feature.
    # Empty list => the JSONL record omits the ``retry_metadata`` block.
    extra_word_pool: list[str] = dataclasses.field(default_factory=list)

    @property
    def example_id(self) -> str:
        """Stable JSON-serialized id for this task."""
        return json.dumps(
            {
                "t": self.template_id,
                "n": self.num_transformations,
                "nf": self.num_few_shot_examples,
                "ee": self.explain_encryption,
                "rt": self.reverse_task,
                "rule": [{"tr": t, "p": p} for t, p in self.rule],
            }
        )

    def to_jsonl_record(self) -> dict[str, Any]:
        """Convert to the ``{contents, references}`` wire shape consumed
        by ``parse_word_puzzle_item`` in ``env.py``.

        Non-empty ``self.tags`` are emitted as a top-level ``tags`` key.
        Non-empty ``self.extra_word_pool`` emits a top-level
        ``retry_metadata`` block used by the agent-mode retry feature;
        the block is omitted otherwise so direct-answer/training
        pipelines never see the ground-truth rule.
        """
        record: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": self.prompt}]}],
            "references": {"reference": self.expected_answer},
        }
        if self.tags:
            record["tags"] = dict(self.tags)
        if self.extra_word_pool:
            record["retry_metadata"] = {
                # list-of-lists (not tuples) for byte-identical json round-trip.
                "rule": [[op, param] for op, param in self.rule],
                "extra_word_pool": list(self.extra_word_pool),
                "reverse_task": self.reverse_task,
            }
        return record


# Cache for ``exclude_same_length_demos``. Filtering a 118k-word candidate
# list per task cost 47M len() calls over a 200-task slice and dominated
# generation. The number of DISTINCT task-word lengths is ~15, so caching one
# filtered list per excluded length turns a per-task scan into a per-length
# one. Order is preserved, so ``r.sample`` draws exactly what the naive
# comprehension would have -- output stays byte-identical.
#
# Keyed by the candidate list's identity; the entry holds a reference to that
# list so its id() cannot be recycled while the cache is live.
_LENGTH_FILTER_CACHE: dict[tuple[int, int], tuple[Sequence[str], tuple[str, ...]]] = {}


def _candidates_excluding_length(
    word_candidates: Sequence[str], excluded: int
) -> tuple[str, ...]:
    key = (id(word_candidates), excluded)
    hit = _LENGTH_FILTER_CACHE.get(key)
    if hit is not None and hit[0] is word_candidates:
        return hit[1]
    filtered = tuple(w for w in word_candidates if len(w) != excluded)
    _LENGTH_FILTER_CACHE[key] = (word_candidates, filtered)
    return filtered


def generate_one_task(
    *,
    num_transformations: int,
    num_few_shot_examples: int,
    word_candidates: Sequence[str],
    operators: Sequence[str] = BASIC_OPERATORS,
    explain_encryption: bool = False,
    reverse_task: bool = False,
    force_template_id: Optional[int] = None,
    rule: Optional[Sequence[tuple[str, Any]]] = None,
    tags: Optional[dict[str, Any]] = None,
    max_expanding_ops: Optional[int] = None,
    rng: Optional[random.Random] = None,
    include_answer_format: bool = True,
    include_retry_metadata: bool = False,
    retry_extra_pool_size: int = 20,
    exclude_same_length_demos: bool = False,
    few_shot_jitter: Optional[tuple[int, int]] = None,
) -> PuzzleTask:
    """Generate one word-puzzle task.

    Example:
        >>> import random
        >>> from example_recipe.word_puzzle.dataset.puzzle_dataset_gen import (
        ...     generate_one_task, EXTENDED_OPERATORS)
        >>> task = generate_one_task(
        ...     num_transformations=10,
        ...     num_few_shot_examples=5,
        ...     word_candidates=["python", "javascript", "rust", "golang", ...],
        ...     operators=EXTENDED_OPERATORS,
        ...     rng=random.Random(42),
        ... )
        >>> task.rule           # 10 (op_name, param) tuples
        >>> task.expected_answer
        >>> task.to_jsonl_record()  # -> {"contents": [...], "references": {...}}

    Args:
        num_transformations: how many operations to compose into the rule.
        num_few_shot_examples: how many (original, encrypted) pairs to show.
        word_candidates: pool of words to sample from (uses ``num_few_shot_examples
            + 1`` distinct words: N for demonstrations, 1 for the task).
        operators: which operators are allowed in the rule (e.g.
            ``BASIC_OPERATORS`` for shift/mirror/rotate,  ``EXTENDED_OPERATORS``
            for the training-set operators, ``WITHHOLD_OPERATORS`` for
            eval-only operators, or ``ALL_OPERATORS``).
        explain_encryption: if True, ask the model to describe the rule before
            answering.
        reverse_task: if False (default), question is "given encrypted, find
            original"; if True, "given original, find encrypted".
        force_template_id: pin a prompt template (1..10) instead of sampling.
        rule: optionally pass a pre-built rule (skips ``generate_random_rule``).
            Useful for testing or when you want to reuse a rule across tasks.
        tags: optional caller-supplied metadata dict, serialized as a
            top-level ``tags`` key on the JSONL record and passed
            through into the eval output for group-by-tag analysis.
        max_expanding_ops: if set, caps the number of expanding operators
            (see ``EXPANDING_OPERATORS``) in a single rule. Ignored when
            ``rule`` is provided.
        rng: seedable ``random.Random`` instance. If None, uses a fresh
            (non-deterministic) one.
        include_answer_format: if True (default), the emitted prompt ends
            with the ``$\\text{your answer}$`` trailer. Set False for
            agent-mode (structured tool-call submit) eval generation.
        include_retry_metadata: if True, populate
            ``PuzzleTask.extra_word_pool`` and emit a ``retry_metadata``
            block on the JSONL record for the agent-mode retry feature.
            Default False to avoid leaking the ground-truth rule.
        retry_extra_pool_size: pool size when
            ``include_retry_metadata=True``. Ignored otherwise.

    Returns:
        A ``PuzzleTask`` dataclass.
    """
    if num_few_shot_examples < 0:
        raise ValueError("num_few_shot_examples must be >= 0")
    r: random.Random = rng if rng is not None else random.Random()

    if few_shot_jitter is not None:
        # Opt-in. Callers normally derive the demo count from depth
        # (``depth + 2``), which makes the two co-vary perfectly: a per-depth
        # accuracy breakdown then cannot separate "deeper rule" from "more
        # worked examples". Jittering per task breaks that correlation while
        # leaving the mean roughly where it was.
        #
        # Drawn from ``r`` BEFORE the word sample, so a jittered run is still
        # fully determined by the seed. It does shift every subsequent draw,
        # so jittered output is not comparable row-for-row with unjittered
        # output -- which is why this is opt-in and off by default.
        lo, hi = few_shot_jitter
        if lo > hi:
            raise ValueError(f"few_shot_jitter lo > hi: {few_shot_jitter!r}")
        num_few_shot_examples = max(
            MIN_FEW_SHOT_EXAMPLES, num_few_shot_examples + r.randint(lo, hi)
        )

    picked = r.sample(list(word_candidates), num_few_shot_examples + 1)
    demo_words = picked[:num_few_shot_examples]
    task_word = picked[-1]

    if exclude_same_length_demos:
        # A demo of the SAME length as the task word leaks the answer for any
        # positional rule: the solver can copy that demo's index permutation
        # instead of inferring the rule. Proven on a depth-5 chain solved by
        # copying a single same-length demo (mueang -> aunemg); estimated
        # worth 4-6 accuracy points across a split.
        #
        # Opt-in, and applied AFTER the initial sample so the default RNG
        # draw is untouched -- v2.0 / v2.1 must keep reproducing byte-exactly.
        pool = _candidates_excluding_length(word_candidates, len(task_word))
        if len(pool) >= num_few_shot_examples:
            demo_words = r.sample(pool, num_few_shot_examples)
        else:
            raise ValueError(
                "exclude_same_length_demos: only "
                f"{len(pool)} candidate(s) differ in length from task word "
                f"{task_word!r}, need {num_few_shot_examples}"
            )

    if rule is None:
        max_rotate = min((len(w) - 1 for w in demo_words), default=len(task_word) - 1)
        max_rotate = max(1, max_rotate)
        # Resample the rule until, on the task word AND every demo word, it
        # (a) round-trips, (b) renders safely into the prompt, and
        # (c) actually changes the word.
        # (a) guards against non-invertible compositions (e.g. shift after a
        #     charset-changing op producing a degenerate all-one-letter
        #     output, or a permutation op that isn't a bijection on this
        #     length).
        # (b) guards against outputs carrying control characters or a
        #     backtick, which corrupt the rendered puzzle even though the
        #     rule itself is perfectly invertible. See
        #     ``check_rule_prompt_safe``.
        # (c) guards against identity rules (an involution composed with
        #     itself), which are reversible AND printable but ask the model
        #     to echo the word already shown. See
        #     ``check_rule_nondegenerate``.
        _MAX_ATTEMPTS = 200
        for _attempt in range(_MAX_ATTEMPTS):
            candidate = generate_random_rule(
                num_transformations=num_transformations,
                max_rotate=max_rotate,
                operators=operators,
                rng=r,
                max_expanding_ops=max_expanding_ops,
            )
            if all(
                check_rule_reversible(w, candidate)
                and check_rule_prompt_safe(w, candidate)
                and check_rule_nondegenerate(w, candidate)
                for w in picked  # demos + task word
            ):
                rule = candidate
                break
        else:
            raise RuntimeError(
                f"generate_one_task: could not find a reversible, "
                f"prompt-safe, non-degenerate rule of length "
                f"{num_transformations} on operators {list(operators)!r} in "
                f"{_MAX_ATTEMPTS} attempts (task_word={task_word!r})."
            )
    rule = list(rule)

    demos: list[tuple[str, str]] = [(w, apply_transform(w, rule)) for w in demo_words]
    encrypted = apply_transform(task_word, rule)

    prompt, tid = generate_prompt(
        examples=demos,
        task_word=task_word if reverse_task else encrypted,
        force_template_id=force_template_id,
        explain_encryption=explain_encryption,
        reverse_task=reverse_task,
        rng=r,
        include_answer_format=include_answer_format,
    )

    # Sample the retry-feature extra pool disjoint from demo + task
    # words; if candidates run short, take whatever's available.
    extra_word_pool: list[str] = []
    if include_retry_metadata:
        used = set(demo_words) | {task_word}
        pool_candidates = [w for w in word_candidates if w not in used]
        n_pool = min(retry_extra_pool_size, len(pool_candidates))
        extra_word_pool = r.sample(pool_candidates, n_pool)

    return PuzzleTask(
        prompt=prompt,
        expected_answer=encrypted if reverse_task else task_word,
        original_word=task_word,
        encrypted_word=encrypted,
        few_shot_examples=demos,
        rule=rule,
        template_id=tid,
        num_transformations=num_transformations,
        num_few_shot_examples=num_few_shot_examples,
        explain_encryption=explain_encryption,
        reverse_task=reverse_task,
        tags=_tags_with_realised_few_shot(tags, num_few_shot_examples),
        extra_word_pool=extra_word_pool,
    )


def _tags_with_realised_few_shot(
    tags: Optional[dict[str, Any]], realised: int
) -> dict[str, Any]:
    """Correct a caller-supplied ``num_few_shot`` tag to the realised count.

    Callers build that tag from the count they REQUESTED, which
    ``few_shot_jitter`` may then have changed. Without this the emitted JSONL
    would claim a demo count the row does not have, and every downstream
    per-few-shot analysis would silently read the wrong number.

    Only rewrites a key the caller already set, so unjittered output stays
    byte-identical (requested == realised there) and no tag is invented for
    callers that never asked for one.
    """
    out = dict(tags) if tags else {}
    if "num_few_shot" in out:
        out["num_few_shot"] = realised
    return out


def load_word_candidates(
    path: Optional[str] = None,
    *,
    min_length: int = 4,
) -> list[str]:
    """Load ASCII-letter-only word candidates.

    Reads one word per line from ``path``, or from the bundled
    ``DEFAULT_WORD_LIST`` allowlist when ``path`` is None.

    Results are filtered to lowercase-ASCII words of length
    ``>= min_length``, de-duplicated, and sorted for determinism.
    """
    with open(path or DEFAULT_WORD_LIST, "r", encoding="utf-8") as f:
        raw = [line.strip() for line in f if line.strip()]

    keep: set[str] = set()
    for w in raw:
        if len(w) >= min_length and all(c in _ASCII_LETTERS for c in w):
            keep.add(w)
    return sorted(keep)


def generate_dataset(
    *,
    num_examples: int,
    num_transformations: int,
    num_few_shot_examples: int,
    word_candidates: Sequence[str],
    operators: Sequence[str] = BASIC_OPERATORS,
    explain_encryption: bool = True,
    flip_reverse_task: bool = True,
    include_answer_format: bool = True,
    max_expanding_ops: Optional[int] = None,
    rng: Optional[random.Random] = None,
    include_retry_metadata: bool = False,
    retry_extra_pool_size: int = 20,
    base_tags: Optional[dict[str, Any]] = None,
    exclude_same_length_demos: bool = False,
    few_shot_jitter: Optional[tuple[int, int]] = None,
) -> list[PuzzleTask]:
    """Generate ``num_examples`` tasks (or ``2 * num_examples`` if
    ``flip_reverse_task=True``, producing both encrypt- and
    decrypt-direction puzzles).

    Args:
        include_answer_format: if True (default), every task's prompt
            ends with the ``$\\text{your answer}$`` trailer. Set False
            for agent-mode training/eval where the answer is submitted
            via a ``submit()`` tool-call payload. Forwarded to
            ``generate_one_task``.
        max_expanding_ops: if set, caps the number of expanding
            operators (hex/base64/ascii-values/etc.) in each generated
            rule. Forwarded to ``generate_one_task``; see
            ``generate_random_rule`` for the exact semantics.
        include_retry_metadata: if True, each task carries
            ``retry_metadata`` (extra word pool for the agent-mode
            retry feature). Forwarded to ``generate_one_task``.
        retry_extra_pool_size: pool size when
            ``include_retry_metadata=True``. Ignored otherwise.
        base_tags: caller-supplied metadata merged into each task's
            ``tags`` dict. ``reverse_task`` is added per-task
            automatically.
        exclude_same_length_demos: if True, no few-shot demo shares the
            task word's length, closing a shortcut where the solver copies
            a same-length demo's index permutation instead of inferring
            the rule. Off by default so existing splits reproduce.
        few_shot_jitter: if set to ``(lo, hi)``, each task's demo count is
            ``num_few_shot_examples + randint(lo, hi)``, clamped at
            ``MIN_FEW_SHOT_EXAMPLES``. Decorrelates demo count from depth.
            Off by default so existing splits reproduce.
    """
    r: random.Random = rng if rng is not None else random.Random()
    tasks: list[PuzzleTask] = []
    for _ in range(num_examples):
        tags = {**(base_tags or {}), "reverse_task": False}
        tasks.append(
            generate_one_task(
                num_transformations=num_transformations,
                num_few_shot_examples=num_few_shot_examples,
                word_candidates=word_candidates,
                operators=operators,
                explain_encryption=explain_encryption,
                reverse_task=False,
                tags=tags,
                rng=r,
                include_answer_format=include_answer_format,
                max_expanding_ops=max_expanding_ops,
                include_retry_metadata=include_retry_metadata,
                retry_extra_pool_size=retry_extra_pool_size,
                exclude_same_length_demos=exclude_same_length_demos,
                few_shot_jitter=few_shot_jitter,
            )
        )
    if flip_reverse_task:
        for _ in range(num_examples):
            tags = {**(base_tags or {}), "reverse_task": True}
            tasks.append(
                generate_one_task(
                    num_transformations=num_transformations,
                    num_few_shot_examples=num_few_shot_examples,
                    word_candidates=word_candidates,
                    operators=operators,
                    explain_encryption=explain_encryption,
                    reverse_task=True,
                    tags=tags,
                    rng=r,
                    include_answer_format=include_answer_format,
                    max_expanding_ops=max_expanding_ops,
                    include_retry_metadata=include_retry_metadata,
                    retry_extra_pool_size=retry_extra_pool_size,
                    exclude_same_length_demos=exclude_same_length_demos,
                    few_shot_jitter=few_shot_jitter,
                )
            )
    return tasks


def write_jsonl(tasks: Sequence[PuzzleTask], path: str) -> None:
    """Write tasks to ``path`` in the ``{contents, references}`` wire shape."""
    with open(path, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t.to_jsonl_record()) + "\n")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="puzzle_dataset_gen",
        description=(
            "Generate a word-puzzle JSONL dataset compatible with "
            "WordPuzzleEnv (see env.py). Purely offline: no LLM "
            "calls, no GCS I/O."
        ),
    )
    p.add_argument("--output", "-o", required=True, help="Path to output JSONL file.")
    p.add_argument(
        "--num-examples",
        type=int,
        default=100,
        help="Number of decrypt-direction examples (doubled if --flip-reverse-task).",
    )
    p.add_argument(
        "--num-transformations",
        type=int,
        default=2,
        help="How many operations composed into each puzzle's rule.",
    )
    p.add_argument(
        "--num-few-shot-examples",
        type=int,
        default=3,
        help="How many (original, encrypted) demo pairs shown in each prompt.",
    )
    p.add_argument(
        "--operator-set",
        choices=list(OPERATOR_SETS.keys()),
        default="extended",
        help="Operator pool: `basic` = shift/mirror/rotate; `extended` "
        "= basic + 7 fixed transforms; `withhold` = 12 eval-only "
        "encoding/cipher transforms; `all` = union.",
    )
    p.add_argument(
        "--word-list",
        default=None,
        help="Path to a plain-text word list (one word per line). If omitted, "
        "uses the bundled vetted allowlist (word_allowlist.txt).",
    )
    p.add_argument(
        "--min-word-length",
        type=int,
        default=5,
        help="Drop candidates shorter than this. 5 is the safer default "
        "for multi-op rules.",
    )
    p.add_argument(
        "--explain-encryption",
        action="store_true",
        default=True,
        help="Ask the model to describe the rule before answering (default: on).",
    )
    p.add_argument(
        "--no-explain-encryption",
        dest="explain_encryption",
        action="store_false",
        help="Do NOT include the 'first describe the rule' instruction.",
    )
    p.add_argument(
        "--flip-reverse-task",
        action="store_true",
        default=True,
        help="Also generate matching encrypt-direction tasks (doubles the "
        "output count). On by default.",
    )
    p.add_argument(
        "--no-flip-reverse-task",
        dest="flip_reverse_task",
        action="store_false",
        help="Only produce decrypt-direction tasks.",
    )
    p.add_argument(
        "--answer-format",
        action="store_true",
        default=True,
        help="Append a 'Put your final answer in the following format: "
        "$\\text{your answer}$' trailer to every prompt (default: on). "
        "Enables unambiguous answer extraction for single-step eval.",
    )
    p.add_argument(
        "--no-answer-format",
        dest="answer_format",
        action="store_false",
        help="Omit the LaTeX answer-format trailer. Use for agent-mode "
        "training/eval, where the answer is submitted via a submit() "
        "tool-call payload and extract_answer.py does exact-match with "
        "no LaTeX handling (a trailer would teach the model to embed "
        "'$\\text{...}$' inside submit args and fail).",
    )
    p.add_argument(
        "--few-shot-jitter",
        nargs=2,
        type=int,
        metavar=("LO", "HI"),
        default=None,
        help="Randomise each task's demo count as "
        "num_few_shot_examples + randint(LO, HI), clamped at "
        f"{MIN_FEW_SHOT_EXAMPLES}. Without it the count is a fixed function "
        "of depth, so the two co-vary perfectly and a per-depth accuracy "
        "breakdown cannot separate 'deeper rule' from 'more worked "
        "examples'. Off by default so existing splits reproduce.",
    )
    p.add_argument(
        "--exclude-same-length-demos",
        action="store_true",
        help="Never show a few-shot demo whose length matches the task "
        "word's. Without this, a positional rule can be solved by copying "
        "a same-length demo's index permutation instead of inferring the "
        "rule. Off by default so previously generated splits reproduce.",
    )
    p.add_argument(
        "--max-expanding-ops",
        type=int,
        default=None,
        help="Cap on the number of expanding operators (hex/base64/"
        "ascii-values/etc. — ops whose output is longer than input) "
        "per generated rule. Default None = no cap. Set to 1 to match "
        "the cap `gen_withhold_eval.py --mode depth-schedule` "
        "applies at depth >= 3.",
    )
    p.add_argument(
        "--include-retry-metadata",
        action="store_true",
        default=False,
        help="Populate retry_metadata (extra_word_pool) on each task for "
        "agent-mode retry. Default off to avoid leaking ground-truth info.",
    )
    p.add_argument(
        "--retry-extra-pool-size",
        type=int,
        default=20,
        help="Number of extra words in the retry pool (only used with "
        "--include-retry-metadata). Default 20.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for the random number generator. If omitted, the run is "
        "non-deterministic.",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    operators = OPERATOR_SETS[args.operator_set]
    word_candidates = load_word_candidates(
        path=args.word_list, min_length=args.min_word_length
    )
    if not word_candidates:
        logger.error("word candidate list is empty.")
        return 2
    logger.info(
        "Loaded %d word candidates (min_length=%d, source=%s)",
        len(word_candidates),
        args.min_word_length,
        args.word_list or DEFAULT_WORD_LIST,
    )
    logger.info("Operator set: %s (%d ops)", args.operator_set, len(operators))
    rng = random.Random(args.seed) if args.seed is not None else random.Random()
    base_tags = {
        "depth": args.num_transformations,
        "num_transformations": args.num_transformations,
        "num_few_shot": args.num_few_shot_examples,
        "operator_set": args.operator_set,
    }
    tasks = generate_dataset(
        num_examples=args.num_examples,
        num_transformations=args.num_transformations,
        num_few_shot_examples=args.num_few_shot_examples,
        word_candidates=word_candidates,
        operators=operators,
        explain_encryption=args.explain_encryption,
        flip_reverse_task=args.flip_reverse_task,
        include_answer_format=args.answer_format,
        max_expanding_ops=args.max_expanding_ops,
        rng=rng,
        include_retry_metadata=args.include_retry_metadata,
        retry_extra_pool_size=args.retry_extra_pool_size,
        base_tags=base_tags,
        exclude_same_length_demos=args.exclude_same_length_demos,
        few_shot_jitter=(tuple(args.few_shot_jitter) if args.few_shot_jitter else None),
    )
    write_jsonl(tasks, args.output)
    logger.info("Wrote %d tasks to %s", len(tasks), args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
