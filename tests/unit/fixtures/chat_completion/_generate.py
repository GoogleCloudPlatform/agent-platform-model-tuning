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

"""Generate the five wire-shape fixtures in this directory.

The ``real_*.json`` files mirror the on-the-wire shape of a Vertex
``generateContent`` turn as captured by the response recorders: turn
counts, role alternation, part kinds, tool-result variants, finish
reasons and usage metadata. Every payload string is synthetic.

Two fictional conversations are replayed here and exported as prefix
snapshots, which is what keeps the cross-file invariants true by
construction rather than by hand-alignment:

* ``len(contents) == 2 * call_idx + 1``
* an earlier snapshot's ``contents`` is a strict prefix of a later one's
* a snapshot's ``completion`` is the next snapshot's ``contents[1]``

``extras.usageMetadata`` is carried over verbatim from the original
recordings and is deliberately *not* re-derived from the synthetic
prompts: ``test_real_thinkhigh_turn01`` pins those counts, and nothing
recounts tokens.

Run with ``python3 tests/unit/fixtures/chat_completion/_generate.py``.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

Json = dict[str, object]

ELIDED_MARKER = "(truncated)"


def _elide(text: str, keep: int = 5) -> str:
    """Keep the first ``keep`` words and mark the rest as removed.

    A real recording's user turn carries the caller's whole task
    description. Emitting a full one here -- even an invented one --
    leaves a reader having to judge whether it is genuine, so user turns
    are always truncated to this shape instead.
    """
    return " ".join(text.split()[:keep]) + " " + ELIDED_MARKER


SYSTEM_INSTRUCTION = """You are a code exploration assistant. Given a question about a
repository, locate the relevant code and report what you found. You have
one tool, `bash`, which runs a single shell command against a read-only
checkout at /codebase.

Work in small steps: run one command, read its output, then decide the
next command. When you have enough context, write your findings to
/tmp/report.txt and stop."""

TOOLS = [
    {
        "functionDeclarations": [
            {
                "name": "bash",
                "description": "Execute a bash command",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "command": {
                            "type": "STRING",
                            "description": "The bash command to execute",
                        }
                    },
                    "required": ["command"],
                },
            }
        ]
    }
]


@dataclass
class Turn:
    """One model turn plus the tool result it produced."""

    text: str | None
    command: str
    output: str = ""
    truncated: bool = False


@dataclass
class Snapshot:
    """A recorded call exported from a conversation at ``call_idx``."""

    filename: str
    call_idx: int
    duration_s: float
    generation_token_count: int
    usage: Json


@dataclass
class Scenario:
    trajectory_id: str
    thinking_level: str
    problem: str
    turns: list[Turn]
    ts0: float
    sig_lengths: list[int]
    snapshots: list[Snapshot] = field(default_factory=list)


def _blob(key: str, nbytes: int) -> str:
    """Deterministic pseudo-random bytes, base64-encoded.

    Stands in for an opaque server-side thought signature. Only its
    presence is ever asserted, so these are kept far shorter than the
    ones a real backend returns -- they would otherwise be most of the
    bytes in this directory, and no reader can check them anyway.
    """
    seed = hashlib.sha256(key.encode()).digest()
    buf = b""
    counter = 0
    while len(buf) < nbytes:
        buf += hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        counter += 1
    return base64.b64encode(buf[:nbytes]).decode()


def _hex(key: str, nchars: int) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:nchars]


def _model_parts(scenario: Scenario, idx: int) -> list[Json]:
    turn = scenario.turns[idx]
    nbytes = scenario.sig_lengths[idx % len(scenario.sig_lengths)]
    parts: list[Json] = []
    if turn.text is not None:
        parts.append({"text": turn.text})
    parts.append(
        {
            "functionCall": {"name": "bash", "args": {"command": turn.command}},
            "thoughtSignature": _blob(f"{scenario.trajectory_id}:{idx}", nbytes),
        }
    )
    return parts


def _tool_parts(turn: Turn) -> list[Json]:
    if turn.truncated:
        head, tail = turn.output[:220], turn.output[-220:]
        response: Json = {
            "returncode": 0,
            "output_head": head,
            "output_tail": tail,
            "elided_chars": 4793,
            "warning": "Output too long.",
        }
    else:
        response = {"returncode": 0, "output": turn.output}
    return [{"functionResponse": {"name": "unknown", "response": response}}]


def _request(scenario: Scenario, call_idx: int) -> Json:
    contents: list[Json] = [
        {"role": "user", "parts": [{"text": scenario.problem}]},
    ]
    for i in range(call_idx):
        contents.append({"role": "model", "parts": _model_parts(scenario, i)})
        contents.append({"role": "user", "parts": _tool_parts(scenario.turns[i])})
    return {
        "contents": contents,
        "system_instruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
        "tools": TOOLS,
        "generation_config": {
            "thinkingConfig": {"thinkingLevel": scenario.thinking_level}
        },
    }


def _record(scenario: Scenario, snap: Snapshot) -> Json:
    request = _request(scenario, snap.call_idx)
    parts = _model_parts(scenario, snap.call_idx)
    turn = scenario.turns[snap.call_idx]
    return {
        "trajectory_id": scenario.trajectory_id,
        "call_idx": snap.call_idx,
        "ts": scenario.ts0 + snap.call_idx,
        "status": "ok",
        "duration_s": snap.duration_s,
        "request": request,
        "response": [
            {
                "prompt": request,
                "completion": {"role": "model", "parts": parts},
                "completion_text": turn.text or "",
                "finishReason": "STOP",
                "extras": {
                    "candidateId": "real-"
                    + _hex(f"{scenario.trajectory_id}:cand:{snap.call_idx}", 32),
                    "operationId": "real-op-"
                    + _hex(f"{scenario.trajectory_id}:op:{snap.call_idx}", 12),
                    "trainStepId": None,
                    "generationTokenCount": snap.generation_token_count,
                    "usageMetadata": snap.usage,
                },
            }
        ],
    }


def _usage(prompt: int, candidates: int, total: int, thoughts: int | None) -> Json:
    usage: Json = {
        "promptTokenCount": prompt,
        "candidatesTokenCount": candidates,
        "totalTokenCount": total,
        "trafficType": "ON_DEMAND",
        "promptTokensDetails": [{"modality": "TEXT", "tokenCount": prompt}],
        "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": candidates}],
    }
    if thoughts is not None:
        usage["thoughtsTokenCount"] = thoughts
    return usage


# ── Conversation A: HIGH thinking, model narrates before each call ──

THINKHIGH = Scenario(
    trajectory_id="traj-thinkhigh-01__1",
    thinking_level="HIGH",
    problem=_elide("Which code decides the cache eviction policy?"),
    ts0=1700000000.0,
    sig_lengths=[96, 144, 288],
    turns=[
        Turn(
            "I will list the top level of the codebase.",
            "find /codebase -maxdepth 2 -not -path '*/.*'",
            "/codebase\n/codebase/lib\n/codebase/lib/cache\n/codebase/lib/config\n"
            "/codebase/lib/http\n/codebase/docs\n/codebase/tests\n"
            + "\n".join(f"/codebase/lib/mod_{i}.py" for i in range(40)),
            truncated=True,
        ),
        Turn(
            "I will search for cache-related modules.",
            'rg -l "cache" /codebase',
            "/codebase/lib/cache/policy.py\n/codebase/lib/cache/store.py\n"
            "/codebase/lib/config/settings.py\n"
            + "\n".join(f"/codebase/tests/cache/test_{i}.py" for i in range(30)),
            truncated=True,
        ),
        Turn(
            "I will look at the cache package layout.",
            "find /codebase/lib/cache -maxdepth 1",
            "/codebase/lib/cache\n/codebase/lib/cache/__init__.py\n"
            "/codebase/lib/cache/policy.py\n/codebase/lib/cache/store.py\n"
            "/codebase/lib/cache/lru.py\n/codebase/lib/cache/ttl.py\n"
            "/codebase/lib/cache/sweeper.py",
        ),
        Turn(
            "I will read the cache entry point.",
            "sed -n '1,60p' /codebase/lib/cache/__init__.py",
            "from .policy import EvictionPolicy, resolve_policy\n"
            "from .store import CacheStore\n\n"
            '__all__ = ["CacheStore", "EvictionPolicy", "resolve_policy"]\n',
        ),
        Turn(
            "I will find the eviction policy definition.",
            'rg -n "evict" /codebase/lib/cache',
            "/codebase/lib/cache/policy.py:12:class EvictionPolicy(Enum):\n"
            "/codebase/lib/cache/store.py:88:    def evict(self, n: int) -> int:\n"
            "/codebase/lib/cache/sweeper.py:31:        store.evict(batch)\n",
        ),
        Turn(
            "I will read the policy module.",
            "sed -n '1,80p' /codebase/lib/cache/policy.py",
            "class EvictionPolicy(Enum):\n"
            '    LRU = "lru"\n'
            '    TTL = "ttl"\n\n'
            "def resolve_policy(settings):\n"
            "    return EvictionPolicy(settings.cache_policy)\n",
        ),
        Turn(
            "I will check how the policy is selected at startup.",
            'rg -n "CACHE_POLICY" /codebase',
            "/codebase/lib/config/settings.py:41:    cache_policy: str = "
            'os.environ.get("CACHE_POLICY", "lru")\n',
        ),
        Turn(
            "I will read the settings module.",
            "sed -n '1,50p' /codebase/lib/config/settings.py",
            "@dataclass\nclass Settings:\n"
            '    cache_policy: str = os.environ.get("CACHE_POLICY", "lru")\n'
            "    cache_max_entries: int = 10_000\n",
        ),
        Turn(
            "I will look for the default policy value.",
            'rg -n "default_policy" /codebase',
            "/codebase/lib/cache/policy.py:9:default_policy = EvictionPolicy.LRU\n",
        ),
        Turn(
            "I will read the store implementation.",
            "sed -n '1,70p' /codebase/lib/cache/store.py",
            "class CacheStore:\n"
            "    def __init__(self, policy, max_entries):\n"
            "        self._policy = policy\n"
            "        self._max_entries = max_entries\n",
        ),
        Turn(
            "I will check the LRU implementation.",
            "sed -n '1,90p' /codebase/lib/cache/lru.py",
            "class LruIndex:\n"
            "    def touch(self, key): ...\n"
            "    def victims(self, n): ...\n",
        ),
        Turn(
            "I will check the TTL implementation.",
            "sed -n '1,60p' /codebase/lib/cache/ttl.py",
            "class TtlIndex:\n    def expired(self, now): ...\n",
        ),
        Turn(
            "I will look for the eviction call sites.",
            "rg -n 'evict\\(' /codebase --glob '!tests'",
            "/codebase/lib/cache/sweeper.py:31:        store.evict(batch)\n",
        ),
        Turn(
            "I will check the metrics emitted on eviction.",
            'rg -n "cache_evictions" /codebase',
            "",
        ),
        Turn(
            "I will inspect the background sweeper.",
            "sed -n '1,55p' /codebase/lib/cache/sweeper.py",
            "def sweep(store, batch=128):\n    while True:\n        store.evict(batch)\n",
        ),
        Turn(
            "I will look at the cache tests.",
            "find /codebase/tests/cache -maxdepth 1",
            "/codebase/tests/cache\n/codebase/tests/cache/test_policy.py\n"
            "/codebase/tests/cache/test_store.py\n",
        ),
        Turn(
            "I will read the policy tests.",
            "sed -n '1,70p' /codebase/tests/cache/test_policy.py",
            "def test_resolve_policy_defaults_to_lru():\n"
            "    assert resolve_policy(Settings()) is EvictionPolicy.LRU\n"
            + "\n".join(f"def test_case_{i}(): ..." for i in range(60)),
            truncated=True,
        ),
        Turn(
            "I will check the documentation for the policy.",
            'rg -n "eviction" /codebase/docs',
            "/codebase/docs/cache-tuning.md:14:The eviction policy is chosen by "
            "CACHE_POLICY.\n",
        ),
        Turn(
            "I will read the tuning guide.",
            "sed -n '1,40p' /codebase/docs/cache-tuning.md",
            "# Cache tuning\n\nSet CACHE_POLICY to `lru` or `ttl`. The default is "
            "`lru`.\n",
        ),
        Turn(
            "I will write the report.",
            "cat > /tmp/report.txt <<'EOF'\n"
            "lib/cache/policy.py 9-24\n"
            "lib/config/settings.py 38-45\n"
            "EOF\ncat /tmp/report.txt",
        ),
    ],
    snapshots=[
        Snapshot(
            "real_thinkhigh_turn01_text_and_fc.json",
            call_idx=0,
            duration_s=1.9,
            generation_token_count=48,
            usage=_usage(881, 48, 1053, 124),
        ),
        Snapshot(
            "real_thinkhigh_turn11_text_and_fc.json",
            call_idx=10,
            duration_s=2.96,
            generation_token_count=33,
            usage=_usage(14656, 33, 15193, 504),
        ),
        Snapshot(
            "real_thinkhigh_turn20_text_and_fc.json",
            call_idx=19,
            duration_s=1.64,
            generation_token_count=46,
            usage=_usage(23258, 46, 23399, 95),
        ),
    ],
)


# ── Conversation B: MINIMAL thinking, no narration before the call ──

THINKMIN = Scenario(
    trajectory_id="traj-thinkmin-01__1",
    thinking_level="MINIMAL",
    problem=_elide("Where is the default request timeout set?"),
    ts0=1700001000.0,
    sig_lengths=[51],
    turns=[
        Turn(
            None,
            'rg -i "REQUEST_TIMEOUT" /codebase',
            "/codebase/lib/config/defaults.js:7:  REQUEST_TIMEOUT: 30000,\n"
            "/codebase/lib/http/client.js:22:  timeout: config.REQUEST_TIMEOUT,\n",
        ),
        Turn(
            None,
            'rg -i "timeout" /codebase/lib/http',
            "/codebase/lib/http/client.js:22:  timeout: config.REQUEST_TIMEOUT,\n"
            "/codebase/lib/http/retry.js:9:const RETRY_TIMEOUT = 5000;\n",
        ),
        Turn(
            None,
            "sed -n '1,60p' /codebase/lib/http/client.js",
            "const config = require('../config/defaults');\n\n"
            "function createClient(overrides = {}) {\n"
            "  return new Client({ timeout: config.REQUEST_TIMEOUT, ...overrides });\n"
            "}\n" + "\n".join(f"// helper {i}" for i in range(60)),
            truncated=True,
        ),
        Turn(
            None,
            "sed -n '1,40p' /codebase/lib/config/defaults.js",
            "module.exports = {\n  REQUEST_TIMEOUT: 30000,\n  MAX_RETRIES: 3,\n};\n",
        ),
        Turn(
            None,
            'rg -n "DEFAULT_TIMEOUT" /codebase',
            "/codebase/lib/http/retry.js:11:const DEFAULT_TIMEOUT = "
            "config.REQUEST_TIMEOUT;\n",
        ),
        Turn(
            None,
            "cat > /tmp/report.txt <<'EOF'\n"
            "lib/config/defaults.js 1-4\n"
            "EOF\ncat /tmp/report.txt",
        ),
    ],
    snapshots=[
        Snapshot(
            "real_thinkmin_turn01_fc_only.json",
            call_idx=0,
            duration_s=0.99,
            generation_token_count=24,
            usage=_usage(878, 24, 902, None),
        ),
        Snapshot(
            "real_thinkmin_last_fc_only.json",
            call_idx=5,
            duration_s=1.11,
            generation_token_count=86,
            usage=_usage(5615, 86, 5701, None),
        ),
    ],
)


def main() -> None:
    out_dir = Path(__file__).parent
    for scenario in (THINKHIGH, THINKMIN):
        for snap in scenario.snapshots:
            path = out_dir / snap.filename
            path.write_text(json.dumps(_record(scenario, snap)))
            print(f"{snap.filename}  {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
