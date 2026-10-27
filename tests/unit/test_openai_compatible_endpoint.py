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

"""Offline tests for OpenAICompatibleSampler (fake httpx transport)."""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import time
from typing import Any, Callable

import httpx
import pytest
from google.cloud.aiplatform_v1.types import GenerateContentRequest

from backends import openai_compatible_endpoint
from backends.chat_completion import convert_gemini_request_to_openai
from backends.openai_compatible_endpoint import OpenAICompatibleSampler
from backends.prompt_coerce import to_wire_dict
from backends.retry_profile import Backoff, SyncRetryProfile

# Real SGLang response (recorded 2026-09-25), trimmed; model renamed.
_TOOL_CALL_RESPONSE = {
    "id": "7bee5e248e1a4763b1a5f9150f587eec",
    "object": "chat.completion",
    "model": "org/model-x",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "",
                "reasoning_content": "I should call get_weather.",
                "tool_calls": [
                    {
                        "id": "get_weather:0",
                        "index": 0,
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 179, "completion_tokens": 80, "reasoning_tokens": 37},
}

_TOOLS = [
    {
        "function_declarations": [
            {
                "name": "get_weather",
                "description": "weather",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {"city": {"type": "STRING"}},
                    "required": ["city"],
                },
            }
        ]
    }
]


def _fake_token(exp: float) -> str:
    payload = base64.urlsafe_b64encode(json.dumps({"exp": exp}).encode()).decode().rstrip("=")
    return f"h.{payload}.s"


def _sampler(
    handler: Callable[[httpx.Request], httpx.Response], token: str = "tok"
) -> OpenAICompatibleSampler:
    cmd = f"{sys.executable} -c \"print('{token}')\""
    return OpenAICompatibleSampler(
        base_url="https://llm.example/",
        model="org/model-x",
        token_command=cmd,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        retry=SyncRetryProfile(
            max_attempts=3, backoff=Backoff(0.01, 0.01, jitter=0.0)
        ),
    )


def _bare(**kwargs: Any) -> OpenAICompatibleSampler:
    return OpenAICompatibleSampler(base_url="https://llm.example", model="org/model-x", **kwargs)


def _count_mints(sampler: OpenAICompatibleSampler) -> dict[str, int]:
    calls = {"n": 0}
    orig = sampler._mint_token

    async def counting() -> str:
        calls["n"] += 1
        return await orig()

    sampler._mint_token = counting  # type: ignore[method-assign]
    return calls


def _age_last_mint(sampler: OpenAICompatibleSampler) -> None:
    """Pretend the cached token was minted long ago (a 401 then re-mints)."""
    sampler._last_mint_at -= 3600.0


def test_sampler_end_to_end_with_fake_transport():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_TOOL_CALL_RESPONSE)

    sampler = _sampler(handler, token="abc")
    req = GenerateContentRequest(contents=[{"role": "user", "parts": [{"text": "q"}]}], tools=_TOOLS)
    [result] = asyncio.run(sampler.sample_n(req, 1, max_generation_tokens=99))

    assert seen["url"] == "https://llm.example/v1/chat/completions"
    assert seen["auth"] == "Bearer abc"
    assert seen["body"]["max_tokens"] == 99
    assert seen["body"]["model"] == "org/model-x"
    tc = result.tuning_candidate
    assert tc.candidate_id == "0"
    assert tc.generation_token_count == 80
    assert tc.thoughts_token_count == 37
    assert result.context.prompt_token_count == 179
    assert result.context.model_version == "org/model-x"
    assert tc.candidate.content.parts[1].function_call.name == "get_weather"
    assert sampler.model_name == "org/model-x"


# Tool-call arguments as a server writes them: keys not sorted, an int,
# non-ASCII, a nested object, compact separators. The history's protobuf Struct
# loses the key order and turns 3000 into 3000.0, so only the sampler's memory
# can send this back. The separators are what keep the converter from ever
# reproducing it: upb's Struct key order is random per process, and it matched
# this order often enough (~7% of processes) to make the `!= _ARGS` guards flaky.
_ARGS = '{"path":"a — b.txt","WaitMs":3000,"chunks":[{"end":2,"start":1}],"case":true}'


def _reply(arguments: str) -> dict[str, Any]:
    reply = json.loads(json.dumps(_TOOL_CALL_RESPONSE))
    reply["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = arguments
    return reply


def _history(*arguments: str, prompt: str = "q") -> GenerateContentRequest:
    calls = [{"function_call": {"name": "get_weather", "args": json.loads(a)}} for a in arguments]
    return GenerateContentRequest(
        contents=[{"role": "user", "parts": [{"text": prompt}]}, {"role": "model", "parts": calls}]
    )


def _remember(sampler: OpenAICompatibleSampler, arguments: str, prompt: str = "q") -> None:
    """File a reply calling with ``arguments`` to a one-message ``prompt``."""
    request = GenerateContentRequest(contents=[{"role": "user", "parts": [{"text": prompt}]}])
    sampler._remember_tool_call_args(sampler._build_body(request, 1), _reply(arguments))


def test_replayed_tool_call_args_are_sent_back_verbatim():
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_reply(_ARGS))

    sampler = _sampler(handler)
    turn1 = GenerateContentRequest(contents=[{"role": "user", "parts": [{"text": "q"}]}], tools=_TOOLS)

    async def two_turns() -> GenerateContentRequest:
        [result] = await sampler.sample_n(turn1, 1)
        turn2 = GenerateContentRequest(
            contents=[
                *turn1.contents,
                result.tuning_candidate.candidate.content,
                {
                    "role": "user",
                    "parts": [
                        {"function_response": {"name": "get_weather", "response": {"output": "ok"}}}
                    ],
                },
            ],
            tools=_TOOLS,
        )
        await sampler.sample_n(turn2, 1)
        return turn2

    turn2 = asyncio.run(two_turns())
    [call] = bodies[1]["messages"][1]["tool_calls"]
    assert call["function"]["arguments"] == _ARGS
    # The converter alone cannot reproduce it (the test would pass vacuously).
    plain = convert_gemini_request_to_openai(to_wire_dict(turn2), model="m", n=1, max_tokens=1)
    assert plain["messages"][1]["tool_calls"][0]["function"]["arguments"] != _ARGS


def test_same_call_in_two_rollouts_replays_each_ones_own_string():
    # Two rollouts make the same call, spelled differently (key order); each
    # must get back the string it wrote, not the one filed last.
    reordered = json.dumps(dict(reversed(json.loads(_ARGS).items())), ensure_ascii=False)
    sampler = _bare()
    _remember(sampler, _ARGS, prompt="task A")
    _remember(sampler, reordered, prompt="task B")
    for prompt, written in (("task A", _ARGS), ("task B", reordered)):
        body = sampler._build_body(_history(written, prompt=prompt), 1)
        assert body["messages"][1]["tool_calls"][0]["function"]["arguments"] == written


def test_rewritten_history_still_finds_the_call():
    # A history rewritten since the reply (e.g. compacted) no longer repeats
    # the conversation that produced the call; the call alone still matches.
    sampler = _bare()
    _remember(sampler, _ARGS, prompt="the full history")
    body = sampler._build_body(_history(_ARGS, prompt="a summary of it"), 1)
    assert body["messages"][1]["tool_calls"][0]["function"]["arguments"] == _ARGS


def test_tool_call_args_this_sampler_never_returned_use_the_converter(
    caplog: pytest.LogCaptureFixture,
):
    request = _history('{"b": 1, "a": 2}')
    plain = convert_gemini_request_to_openai(to_wire_dict(request), model="m", n=1, max_tokens=1)
    assert _bare()._build_body(request, 1)["messages"] == plain["messages"]
    assert "no remembered argument string" in caplog.text


def test_tool_call_args_memory_is_a_bounded_lru(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(openai_compatible_endpoint, "_TOOL_ARGS_MEMORY_MAX", 4)  # 2 calls
    sampler = _bare()
    # The converter never writes _ARGS's compact separators, so a fallback is visible.
    a, b, c = (_ARGS.replace("3000", str(n)) for n in (1, 2, 3))
    _remember(sampler, a)
    _remember(sampler, b)
    sampler._build_body(_history(a), 1)  # replaying a keeps it fresh
    _remember(sampler, c)  # evicts b, the least recently used
    sent = [
        tc["function"]["arguments"]
        for tc in sampler._build_body(_history(a, b, c), 1)["messages"][1]["tool_calls"]
    ]
    assert sent[0] == a and sent[2] == c
    assert sent[1] != b and json.loads(sent[1]) == json.loads(b)  # converter fallback


def test_sampler_http_failure_returns_failsafe():
    sampler = _sampler(lambda request: httpx.Response(403, json={"error": "forbidden"}))
    [result] = asyncio.run(sampler.sample_n("q", 1))
    assert result.tuning_candidate.candidate_id == ""


def test_retryable_status_is_retried_through_inherited_loop():
    # Exercises the parent's retry/backoff/log path against the sampler's
    # hand-set attributes (catches drift in what that path reads).
    statuses = iter([503, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(statuses)
        return httpx.Response(status, json=_TOOL_CALL_RESPONSE if status == 200 else {})

    [result] = asyncio.run(_sampler(handler).sample_n("q", 1))
    assert result.tuning_candidate.candidate_id == "0"


def test_n2_splits_tokens_and_keeps_candidate_order():
    data = {
        "id": "r",
        "model": "org/model-x",
        "choices": [
            {"index": i, "message": {"content": f"a{i}"}, "finish_reason": "stop"}
            for i in range(2)
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 11, "reasoning_tokens": 3},
    }
    results = asyncio.run(
        _sampler(lambda r: httpx.Response(200, json=data)).sample_n("q", 2)
    )
    assert [r.completion_text for r in results] == ["a0", "a1"]
    assert sum(r.tuning_candidate.generation_token_count for r in results) == 11
    assert sum(r.tuning_candidate.thoughts_token_count for r in results) == 3


def test_401_drops_cached_token_and_retries_once():
    statuses = iter([401, 200])
    auths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        auths.append(request.headers["authorization"])
        status = next(statuses)
        if status == 401:  # the next mint prints a different token
            sampler._token_command = f"{sys.executable} -c \"print('fresh')\""
            _age_last_mint(sampler)  # a 401 on a fresh token is not re-minted
        return httpx.Response(status, json=_TOOL_CALL_RESPONSE if status == 200 else {})

    sampler = _sampler(handler)
    mints = _count_mints(sampler)
    [result] = asyncio.run(sampler.sample_n("q", 1))
    assert result.tuning_candidate.candidate_id == "0"
    assert mints["n"] == 2  # initial + re-mint after the 401
    assert auths == ["Bearer tok", "Bearer fresh"]


def test_retries_reuse_headers_minted_before_the_loop():
    # The refresh time passing mid-retry must not mint inside the parent's
    # loop, where a failing token command would end as a silent failsafe.
    statuses = iter([503, 200])
    auths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        auths.append(request.headers["authorization"])
        status = next(statuses)
        if status == 503:
            sampler._token_refresh_at = 0.0
            sampler._token_command = f"{sys.executable} -c \"import sys; sys.exit(5)\""
        return httpx.Response(status, json=_TOOL_CALL_RESPONSE if status == 200 else {})

    sampler = _sampler(handler)
    [result] = asyncio.run(sampler.sample_n("q", 1))
    assert result.tuning_candidate.candidate_id == "0"
    assert auths == ["Bearer tok", "Bearer tok"]


def test_401_on_one_call_does_not_rerun_a_concurrent_call():
    posts = {"a": 0, "b": 0}

    async def handler(request: httpx.Request) -> httpx.Response:
        who = json.loads(request.content)["messages"][-1]["content"]
        posts[who] += 1
        if who == "a":
            await asyncio.sleep(0.05)
            if posts["a"] == 1:
                sampler._token_command = f"{sys.executable} -c \"print('fresh')\""
                return httpx.Response(401, json={})
            return httpx.Response(200, json=_TOOL_CALL_RESPONSE)
        await asyncio.sleep(0.2)  # still in flight when a's 401 lands
        return httpx.Response(400, json={"error": "bad request"})

    sampler = _sampler(handler)  # type: ignore[arg-type]
    asyncio.run(sampler._get_headers())
    _age_last_mint(sampler)

    async def run() -> Any:
        return await asyncio.gather(sampler.sample_n("a", 1), sampler.sample_n("b", 1))

    [a], [b] = asyncio.run(run())
    assert a.tuning_candidate.candidate_id == "0"
    assert b.tuning_candidate.candidate_id == ""
    assert posts == {"a": 2, "b": 1}


@pytest.fixture
def no_mint_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    import backends.openai_compatible_endpoint as ke

    monkeypatch.setattr(ke, "_TOKEN_MINT_BACKOFF_S", 0.0)


def test_transient_token_command_failure_is_retried(tmp_path: Any, no_mint_backoff: None):
    # First invocation fails, second prints a token (gcloud's hourly network
    # refresh hitting a blip must not fail the task).
    marker = tmp_path / "ran_once"
    script = (
        "import pathlib, sys; p = pathlib.Path(sys.argv[1]); "
        "sys.exit(1) if not p.exists() and not p.touch() else print('tok2')"
    )
    sampler = _bare(token_command=f"{sys.executable} -c \"{script}\" {marker}")
    headers = asyncio.run(sampler._get_headers())
    assert headers["Authorization"] == "Bearer tok2"


def test_401_then_failing_remint_raises_not_failsafe(no_mint_backoff: None):
    # The re-mint after a 401 must also happen outside the parent's retry
    # loop, else a broken gcloud is retried as a transport error and ends as
    # a silent empty result.
    sampler = _sampler(lambda r: httpx.Response(401, json={}))
    asyncio.run(sampler._get_headers())  # cache a valid token first
    _age_last_mint(sampler)
    sampler._token_command = f"{sys.executable} -c \"import sys; sys.exit(4)\""
    with pytest.raises(RuntimeError, match="exit 4"):
        asyncio.run(sampler.sample_n("q", 1))


def test_token_command_timeout_blocks_further_attempts(monkeypatch: pytest.MonkeyPatch, tmp_path: Any):
    import backends.openai_compatible_endpoint as ke

    monkeypatch.setattr(ke, "_TOKEN_COMMAND_TIMEOUT_S", 2.0)
    pid_file = tmp_path / "pid"
    script = f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(30)"
    sampler = _bare(token_command=f"{sys.executable} -c \"{script}\"")
    with pytest.raises(RuntimeError, match="did not finish"):
        asyncio.run(sampler._get_headers())
    # The timed-out child was killed, not orphaned.
    import os

    pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    # Queued callers fail fast instead of each waiting out a new timeout.
    import time

    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="not retrying"):
        asyncio.run(sampler._get_headers())
    assert time.monotonic() - t0 < 0.3


def test_token_command_failure_raises_from_sample_n(no_mint_backoff: None):
    # Must surface to the caller, not be retried as a transport error and
    # turned into an empty (silently failed) result.
    sampler = _bare(
        token_command=f"{sys.executable} -c \"import sys; sys.exit(3)\"",
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json={}))
        ),
    )
    with pytest.raises(RuntimeError, match="exit 3"):
        asyncio.run(sampler.sample_n("q", 1))


def test_token_inside_refresh_margin_is_not_reminted_per_request():
    import time

    sampler = _sampler(lambda r: httpx.Response(200, json={}), token=_fake_token(time.time() + 60))
    mints = _count_mints(sampler)

    async def run() -> None:
        for _ in range(3):
            await sampler._get_headers()

    asyncio.run(run())
    assert mints["n"] == 1


def test_non_object_body_returns_failsafe():
    sampler = _sampler(lambda r: httpx.Response(200, json=["not", "an", "object"]))
    [result] = asyncio.run(sampler.sample_n("q", 1))
    assert result.tuning_candidate.candidate_id == ""


def test_token_cached_until_near_expiry():
    import time

    calls = {"n": 0}
    sampler = _sampler(lambda r: httpx.Response(200, json={}), token=_fake_token(time.time() + 3600))
    orig = sampler._mint_token

    async def counting():
        calls["n"] += 1
        return await orig()

    sampler._mint_token = counting  # type: ignore[method-assign]

    async def run():
        await sampler._get_headers()
        await sampler._get_headers()
        sampler._token_refresh_at = 0.0  # simulate expiry
        await sampler._get_headers()

    asyncio.run(run())
    assert calls["n"] == 2


def test_token_command_failure_is_loud(no_mint_backoff: None):
    sampler = _bare(token_command=f"{sys.executable} -c \"import sys; sys.exit(3)\"")
    with pytest.raises(RuntimeError, match="exit 3"):
        asyncio.run(sampler._get_headers())


def test_persistent_failure_is_remembered_for_queued_callers(no_mint_backoff: None):
    # A fast-failing gcloud (e.g. needs reauth) must not make every queued
    # caller rerun the whole retry sequence under the lock.
    sampler = _bare(token_command=f"{sys.executable} -c \"import sys; sys.exit(3)\"")
    mints = _count_mints(sampler)

    async def run() -> list[BaseException]:
        results = await asyncio.gather(
            *(sampler._get_headers() for _ in range(5)), return_exceptions=True
        )
        return [r for r in results if isinstance(r, BaseException)]

    errors = asyncio.run(run())
    assert len(errors) == 5
    assert mints["n"] == 3  # one retry sequence, not five
    assert sum("not retrying" in str(e) for e in errors) == 4


def test_missing_token_binary_is_not_retried(no_mint_backoff: None):
    sampler = _bare(token_command="definitely-not-a-real-token-binary")
    mints = _count_mints(sampler)
    with pytest.raises(RuntimeError, match="not found on PATH"):
        asyncio.run(sampler._get_headers())
    assert mints["n"] == 1


def test_401_for_a_superseded_token_does_not_invalidate_current_one():
    sampler = _sampler(lambda r: httpx.Response(200, json={}), token="current")
    asyncio.run(sampler._get_headers())
    refresh_at = sampler._token_refresh_at

    def resp_401(bearer: str) -> httpx.Response:
        req = httpx.Request("POST", sampler.url, headers={"authorization": f"Bearer {bearer}"})
        return httpx.Response(401, request=req)

    asyncio.run(sampler._on_response(resp_401("old")))
    assert sampler._token_refresh_at == refresh_at
    asyncio.run(sampler._on_response(resp_401("current")))
    # Minted just now: kept for _TOKEN_MIN_REUSE_S, not re-minted at once.
    assert time.time() < sampler._token_refresh_at < refresh_at
    _age_last_mint(sampler)
    asyncio.run(sampler._on_response(resp_401("current")))
    assert sampler._token_refresh_at <= time.time()


# ---- sample_chat_completion passthrough -------------------------------------


def test_chat_completion_passthrough_forwards_body_and_normalizes_response():
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_TOOL_CALL_RESPONSE)

    request = {
        "model": "caller-model-name",
        "messages": [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": None,
                "thinking_blocks": [{"type": "thinking", "thinking": "earlier thought"}],
                "tool_calls": [{"id": "c1", "type": "function",
                                "function": {"name": "get_weather", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "18C"},
        ],
        "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
        # Rejected by the OpenAI->Gemini converter; must survive the passthrough.
        "response_format": {"type": "json_object"},
    }
    out = asyncio.run(_sampler(handler).sample_chat_completion(request, num_samples=1))

    body = seen["body"]
    assert body["model"] == "org/model-x"  # pinned to the sampler's model
    assert body["n"] == 1
    assert body["max_tokens"] == 32768  # defaulted when the caller sets none
    assert body["response_format"] == {"type": "json_object"}
    assert body["messages"][2] == {"role": "tool", "tool_call_id": "c1", "content": "18C"}
    history = body["messages"][1]
    assert "thinking_blocks" not in history
    assert history["reasoning_content"] == "earlier thought"
    assert request["messages"][1]["thinking_blocks"]  # caller's dict not mutated

    resp = out.response.to_dict()
    msg = resp["choices"][0]["message"]
    assert [tb["thinking"] for tb in msg["thinking_blocks"]] == ["I should call get_weather."]
    assert msg["tool_calls"][0]["id"] == "get_weather:0"  # server's id kept
    assert resp["choices"][0]["finish_reason"] == "tool_calls"
    assert resp["usage"]["completion_tokens_details"]["reasoning_tokens"] == 37

    [result] = out.results
    assert result.tuning_candidate.generation_token_count == 80
    assert result.context.thinking_blocks == (
        {"type": "thinking", "thinking": "I should call get_weather."},
    )


def test_chat_completion_passthrough_failure_raises():
    sampler = _sampler(lambda r: httpx.Response(403, json={"error": "forbidden"}))
    with pytest.raises(RuntimeError, match="upstream sampling failed"):
        asyncio.run(
            sampler.sample_chat_completion({"model": "x", "messages": [{"role": "user", "content": "q"}]})
        )


def test_passthrough_state_is_per_call():
    # Concurrent passthrough calls must each send their own body, and the
    # sample_n path afterwards must convert from Gemini again.
    def handler(request: httpx.Request) -> httpx.Response:
        text = json.loads(request.content)["messages"][-1]["content"]
        return httpx.Response(200, json={
            "id": "r", "model": "m",
            "choices": [{"index": 0, "message": {"content": f"echo {text}"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        })

    sampler = _sampler(handler)

    async def run() -> tuple[list[str], str]:
        outs = await asyncio.gather(*(
            sampler.sample_chat_completion({"model": "x", "messages": [{"role": "user", "content": f"q{i}"}]})
            for i in range(5)
        ))
        [plain] = await sampler.sample_n("gemini path", 1)
        return [o.response.to_dict()["choices"][0]["message"]["content"] for o in outs], plain.completion_text

    texts, plain = asyncio.run(run())
    assert texts == [f"echo q{i}" for i in range(5)]
    assert plain == "echo gemini path"


def test_no_token_command_sends_no_authorization_header():
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = dict(request.headers)
        return httpx.Response(200, json=_TOOL_CALL_RESPONSE)

    sampler = _bare(http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    [result] = asyncio.run(sampler.sample_n("q", 1))
    assert result.tuning_candidate.candidate_id == "0"
    assert "authorization" not in seen["headers"]


def test_passthrough_keeps_caller_max_completion_tokens():
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_TOOL_CALL_RESPONSE)

    asyncio.run(_sampler(handler).sample_chat_completion(
        {"model": "x", "messages": [{"role": "user", "content": "q"}], "max_completion_tokens": 5},
        num_samples=2,
    ))
    assert seen["body"]["max_completion_tokens"] == 5
    assert "max_tokens" not in seen["body"]
    assert (seen["body"]["model"], seen["body"]["n"]) == ("org/model-x", 2)


def test_passthrough_rejects_streaming_without_posting():
    posts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        posts.append(request)
        return httpx.Response(200, json=_TOOL_CALL_RESPONSE)

    with pytest.raises(ValueError, match="streaming is not supported"):
        asyncio.run(_sampler(handler).sample_chat_completion(
            {"model": "x", "messages": [{"role": "user", "content": "q"}], "stream": True}
        ))
    assert posts == []


def test_passthrough_failure_names_the_reason():
    # Choices but no usage.prompt_tokens / completion_tokens -> the error
    # names the missing field.
    for usage, missing in (
        ({"completion_tokens": 3}, "prompt_tokens"),
        ({"prompt_tokens": 3}, "completion_tokens"),
    ):
        data = {**_TOOL_CALL_RESPONSE, "usage": usage}
        sampler = _sampler(lambda r, data=data: httpx.Response(200, json=data))
        with pytest.raises(RuntimeError, match=f"no usage.{missing}"):
            asyncio.run(
                sampler.sample_chat_completion({"model": "x", "messages": [{"role": "user", "content": "q"}]})
            )


# ---- direct passthrough: no sample_n, raw OpenAI capture ---------------------


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Records every dump_raw_fanout call the sampler module makes."""
    import backends.openai_compatible_endpoint as ke
    from utils import raw_capture

    calls: list[dict[str, Any]] = []

    async def fake_dump(ctxs: Any, **kw: Any) -> list[str | None]:
        calls.append({**kw, "ctxs": ctxs, "client_request": raw_capture._CLIENT_REQUEST.get()})
        return [f"uri-{i}" for i in range(kw["count"])]

    monkeypatch.setattr(ke, "dump_raw_fanout", fake_dump)
    return calls


_Q = {"model": "x", "messages": [{"role": "user", "content": "q"}]}


def test_passthrough_hands_capture_ctxs_through(captured: list[dict[str, Any]]):
    ctxs = [object(), object()]
    data = {**_TOOL_CALL_RESPONSE, "choices": [
        {**_TOOL_CALL_RESPONSE["choices"][0], "index": i} for i in range(2)]}
    asyncio.run(_sampler(lambda r: httpx.Response(200, json=data)).sample_chat_completion(
        _Q, num_samples=2, capture_ctxs=ctxs))  # type: ignore[arg-type]
    assert captured[0]["ctxs"] == ctxs


def test_passthrough_capture_ctxs_length_must_match():
    sampler = _sampler(lambda r: httpx.Response(200, json=_TOOL_CALL_RESPONSE))
    with pytest.raises(ValueError, match="capture_ctxs length 2"):
        asyncio.run(sampler.sample_chat_completion(_Q, num_samples=1, capture_ctxs=[None, None]))


def test_passthrough_rejected_response_is_captured_as_error(captured: list[dict[str, Any]]):
    data = {**_TOOL_CALL_RESPONSE, "usage": {"completion_tokens": 3}}  # no prompt_tokens
    with pytest.raises(RuntimeError, match="no usage.prompt_tokens"):
        asyncio.run(_sampler(lambda r: httpx.Response(200, json=data)).sample_chat_completion(_Q))
    [call] = captured
    assert call["status"] == "error"
    assert call["response"][0]["choices"] == data["choices"]  # the raw reply is kept


def test_passthrough_no_choices_writes_one_blob_per_sample(captured: list[dict[str, Any]]):
    data = {"id": "r", "choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 0}}
    with pytest.raises(RuntimeError, match="no choices"):
        asyncio.run(_sampler(lambda r: httpx.Response(200, json=data)).sample_chat_completion(
            _Q, num_samples=2))
    [call] = captured
    assert (call["count"], call["status"], call["response"]) == (2, "error", data)


def test_passthrough_mint_failure_still_captures_the_cause(
    captured: list[dict[str, Any]], no_mint_backoff: None
):
    sampler = _bare(token_command=f"{sys.executable} -c \"import sys; sys.exit(3)\"")
    with pytest.raises(RuntimeError, match="exit 3"):
        asyncio.run(sampler.sample_chat_completion(_Q))
    [call] = captured
    assert call["status"] == "error"
    assert call["response"]["__error__"].startswith("RuntimeError: ")
    assert "exit 3" in call["response"]["__error__"]


def test_passthrough_token_counts_edge_cases():
    # Over-reported reasoning: generation (output + thoughts) never < thoughts.
    over = {**_TOOL_CALL_RESPONSE, "usage": {"prompt_tokens": 1, "completion_tokens": 3,
                                             "reasoning_tokens": 5}}
    [r] = asyncio.run(_sampler(lambda r: httpx.Response(200, json=over))
                      .sample_chat_completion(_Q)).results
    assert (r.tuning_candidate.generation_token_count, r.tuning_candidate.thoughts_token_count) == (5, 5)
    # No completion_tokens but no output either: (0, 0), like the parent.
    empty = {"id": "r", "choices": [{"index": 0, "message": {"content": ""}, "finish_reason": "length"}],
             "usage": {"prompt_tokens": 1}}
    [r] = asyncio.run(_sampler(lambda r: httpx.Response(200, json=empty))
                      .sample_chat_completion(_Q)).results
    assert (r.tuning_candidate.generation_token_count, r.tuning_candidate.thoughts_token_count) == (0, 0)


def test_passthrough_does_not_go_through_sample_n(monkeypatch: pytest.MonkeyPatch):
    sampler = _sampler(lambda r: httpx.Response(200, json=_TOOL_CALL_RESPONSE))

    async def boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("sample_chat_completion must not call sample_n")

    monkeypatch.setattr(sampler, "sample_n", boom)
    out = asyncio.run(sampler.sample_chat_completion(
        {"model": "x", "messages": [{"role": "user", "content": "q"}]}
    ))
    assert out.results[0].tuning_candidate.candidate_id == "0"


def test_passthrough_captures_raw_openai_response_and_client_request(
    captured: list[dict[str, Any]],
):
    request = {"model": "x", "messages": [{"role": "user", "content": "q"}]}
    out = asyncio.run(_sampler(lambda r: httpx.Response(200, json=_TOOL_CALL_RESPONSE))
                      .sample_chat_completion(request))
    [call] = captured
    assert call["status"] == "ok"
    assert call["request"]["model"] == "org/model-x"  # the body actually sent
    assert call["client_request"] == request  # the caller's form
    [slice0] = call["response"]
    assert slice0 == _TOOL_CALL_RESPONSE  # raw OpenAI shape, one choice
    assert out.results[0].context.raw_call_uri == "uri-0"


def test_passthrough_captures_on_http_failure(captured: list[dict[str, Any]]):
    sampler = _sampler(lambda r: httpx.Response(403, json={"error": "forbidden"}))
    with pytest.raises(RuntimeError, match="upstream sampling failed"):
        asyncio.run(sampler.sample_chat_completion(
            {"model": "x", "messages": [{"role": "user", "content": "q"}]}, num_samples=2
        ))
    [call] = captured
    assert (call["status"], call["count"]) == ("error", 2)
    assert call["request"]["n"] == 2


def test_passthrough_n2_splits_tokens_and_slices_capture(captured: list[dict[str, Any]]):
    data = {
        "id": "r", "model": "org/model-x",
        "choices": [
            {"index": i, "message": {"content": f"a{i}", "reasoning_content": f"t{i}"},
             "finish_reason": "stop"}
            for i in range(2)
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 11, "reasoning_tokens": 3},
    }
    out = asyncio.run(_sampler(lambda r: httpx.Response(200, json=data)).sample_chat_completion(
        {"model": "x", "messages": [{"role": "user", "content": "q"}]}, num_samples=2
    ))
    r0, r1 = out.results
    assert [r.tuning_candidate.candidate_id for r in out.results] == ["0", "1"]
    assert (r0.tuning_candidate.generation_token_count, r1.tuning_candidate.generation_token_count) == (6, 5)
    assert (r0.tuning_candidate.thoughts_token_count, r1.tuning_candidate.thoughts_token_count) == (2, 1)
    assert r1.context.thinking_blocks == ({"type": "thinking", "thinking": "t1"},)
    assert (r0.context.prompt_token_count, r0.context.response_id) == (5, "r")
    [call] = captured
    assert [s["choices"][0]["index"] for s in call["response"]] == [0, 1]
    assert all(s["usage"] == data["usage"] for s in call["response"])
    assert [r.context.raw_call_uri for r in out.results] == ["uri-0", "uri-1"]


def test_passthrough_thinking_blocks_come_from_normalized_response(monkeypatch: pytest.MonkeyPatch):
    # Whatever the shared normalizer maps into thinking_blocks (e.g. a
    # server-specific field name) must reach SamplingContext too.
    import backends.openai_compatible_endpoint as ke

    orig = ke.reasoning_content_to_thinking_blocks

    def normalizer(response: dict[str, Any]) -> dict[str, Any]:
        out = orig(response)
        for choice in out["choices"]:
            msg = choice["message"]
            if msg.get("reasoning") and not msg.get("thinking_blocks"):
                msg["thinking_blocks"] = [{"type": "thinking", "thinking": msg.pop("reasoning")}]
        return out

    monkeypatch.setattr(ke, "reasoning_content_to_thinking_blocks", normalizer)
    message = {k: v for k, v in _TOOL_CALL_RESPONSE["choices"][0]["message"].items()
               if k != "reasoning_content"}
    data = {**_TOOL_CALL_RESPONSE,
            "choices": [{"index": 0, "message": {**message, "reasoning": "via reasoning"},
                         "finish_reason": "tool_calls"}]}
    out = asyncio.run(_sampler(lambda r: httpx.Response(200, json=data)).sample_chat_completion(
        {"model": "x", "messages": [{"role": "user", "content": "q"}]}
    ))
    [result] = out.results
    assert result.context.thinking_blocks == ({"type": "thinking", "thinking": "via reasoning"},)


def test_whitespace_token_command_is_a_config_error():
    with pytest.raises(ValueError, match="token_command"):
        _bare(token_command="   ")


def test_non_executable_token_command_is_permanent(tmp_path: Any, no_mint_backoff: None):
    script = tmp_path / "mint.sh"
    script.write_text("#!/bin/sh\necho tok\n")
    script.chmod(0o644)
    sampler = _bare(token_command=str(script))
    mints = _count_mints(sampler)
    with pytest.raises(RuntimeError, match="cannot run"):
        asyncio.run(sampler._get_headers())
    assert mints["n"] == 1
    with pytest.raises(RuntimeError, match="not retrying for 60s"):
        asyncio.run(sampler._get_headers())


# ---- review round 2 ------------------------------------------------------------


def test_unreported_reasoning_tokens_warns_once(caplog: pytest.LogCaptureFixture):
    data = {
        "choices": [{"index": 0, "message": {"content": "a", "reasoning": "r"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 11},
    }
    sampler = _sampler(lambda r: httpx.Response(200, json=data))
    with caplog.at_level("WARNING", logger="backends.openai_compatible_endpoint"):
        asyncio.run(sampler.sample_n("q", 1))
        asyncio.run(sampler.sample_n("q", 1))
    hits = [r for r in caplog.records if "reasoning_tokens" in r.getMessage()]
    assert len(hits) == 1


def test_reported_reasoning_tokens_do_not_warn(caplog: pytest.LogCaptureFixture):
    data = {
        "choices": [{"index": 0, "message": {"content": "a", "reasoning_content": "r"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 11,
                  "completion_tokens_details": {"reasoning_tokens": 3}},
    }
    sampler = _sampler(lambda r: httpx.Response(200, json=data))
    with caplog.at_level("WARNING", logger="backends.openai_compatible_endpoint"):
        asyncio.run(sampler.sample_n("q", 1))
    assert not [r for r in caplog.records if "reasoning_tokens" in r.getMessage()]


def test_401_without_token_command_posts_once():
    posts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        posts["n"] += 1
        return httpx.Response(401, json={})

    sampler = _bare(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        retry=SyncRetryProfile(max_attempts=3, backoff=Backoff(0.01, 0.01, jitter=0.0)),
    )
    [result] = asyncio.run(sampler.sample_n("q", 1))
    assert result.tuning_candidate.candidate_id == ""
    assert posts["n"] == 1  # 401 is not retryable; no second loop


def test_persistent_401_does_not_remint_within_min_reuse_window():
    posts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        posts["n"] += 1
        return httpx.Response(401, json={})

    sampler = _sampler(handler)
    mints = _count_mints(sampler)
    for _ in range(3):
        [result] = asyncio.run(sampler.sample_n("q", 1))
        assert result.tuning_candidate.candidate_id == ""
    assert mints["n"] == 1  # the token was minted < _TOKEN_MIN_REUSE_S ago
    assert posts["n"] == 3  # no pointless retry with the same token


def test_401_on_old_token_remints_once_then_stops():
    posts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        posts["n"] += 1
        # The re-mint prints a different token.
        sampler._token_command = f"{sys.executable} -c \"print('fresh')\""
        return httpx.Response(401, json={})

    sampler = _sampler(handler)
    asyncio.run(sampler._get_headers())
    _age_last_mint(sampler)
    mints = _count_mints(sampler)
    for _ in range(3):
        asyncio.run(sampler.sample_n("q", 1))
    assert mints["n"] == 1  # one re-mint for the stale token, then none
    assert posts["n"] == 4  # call 1: stale + fresh token; calls 2-3: once each


def test_transient_spawn_oserror_is_retried(monkeypatch: pytest.MonkeyPatch, no_mint_backoff: None):
    import errno

    import backends.openai_compatible_endpoint as ke

    real = asyncio.create_subprocess_exec
    calls = {"n": 0}

    async def flaky(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError(errno.EMFILE, "Too many open files")
        return await real(*args, **kwargs)

    monkeypatch.setattr(ke.asyncio, "create_subprocess_exec", flaky)
    sampler = _bare(token_command=f"{sys.executable} -c \"print('tok3')\"")
    headers = asyncio.run(sampler._get_headers())
    assert headers["Authorization"] == "Bearer tok3"
    assert calls["n"] == 2
    assert sampler._mint_blocked_until == 0.0


def test_aclose_closes_client_and_detaches_response_hook():
    # aclose closes the client (injected or not, as GeminiApiSampler.aclose
    # does); the hook is detached too so the dead client does not keep the
    # closed sampler reachable through its event hooks.
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    a = _bare(http_client=client)
    assert a._on_response in client.event_hooks["response"]
    asyncio.run(a.aclose())
    assert a._on_response not in client.event_hooks["response"]
    assert client.is_closed


def test_401_on_throttled_token_logs_error_once_per_window(caplog: pytest.LogCaptureFixture):
    sampler = _sampler(lambda r: httpx.Response(401, json={}))
    with caplog.at_level("ERROR", logger="backends.openai_compatible_endpoint"):
        for _ in range(3):
            asyncio.run(sampler.sample_n("q", 1))
    hits = [
        r for r in caplog.records
        if r.levelname == "ERROR" and "rejected" in r.getMessage()
        and "not re-minting until" in r.getMessage()
    ]
    assert len(hits) == 1  # once for this token's throttle window, not per call
    caplog.clear()
    # A later mint opens a new window: its rejection is reported again.
    sampler._token = None
    with caplog.at_level("ERROR", logger="backends.openai_compatible_endpoint"):
        asyncio.run(sampler.sample_n("q", 1))
    assert [r for r in caplog.records if "not re-minting until" in r.getMessage()]


# ---- full turn loop: real recipe agent + env, scripted OpenAI server ---------
#
# Everything below the HTTP layer is real: the simple_agent word_puzzle agent
# and env (real bash / submit), the Gemini->OpenAI request conversion of the
# growing history, the OpenAI->Gemini response conversion, ATIF steps and
# reward. Only the server is scripted.


def _chat_reply(tool_name: str, raw_args: str, reasoning: str, call_id: str) -> dict[str, Any]:
    return {
        "id": f"r-{call_id}", "object": "chat.completion", "model": "org/model-x",
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": "", "reasoning_content": reasoning,
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": tool_name, "arguments": raw_args}}]}}],
        "usage": {"prompt_tokens": 50, "completion_tokens": 20, "reasoning_tokens": 8},
    }


def _run_simple_agent(
    replies: list[dict[str, Any]],
) -> tuple[Any, list[dict[str, Any]], list[Any]]:
    """Run one rollout; returns (scored, request bodies, SampleResults seen by the agent)."""
    from example_recipe.word_puzzle.agent_solver.simple_agent.env import WordPuzzleSimpleEnv
    from example_recipe.word_puzzle.agent_solver.simple_agent.simple_agent import (
        WordPuzzleSimpleAgent,
    )
    from example_recipe.word_puzzle.dataset.task_data import WordPuzzleTaskData
    from example_recipe.word_puzzle.reward import WordPuzzleRewardConfig, WordPuzzleRewardFn

    requests: list[dict[str, Any]] = []
    queue = list(replies)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=queue.pop(0))

    sampler = _sampler(handler)
    sampled: list[Any] = []
    real_sample_n = sampler.sample_n

    async def spy_sample_n(*args: Any, **kwargs: Any) -> Any:
        results = await real_sample_n(*args, **kwargs)
        sampled.extend(results)
        return results

    sampler.sample_n = spy_sample_n  # type: ignore[method-assign]
    reward_fn = WordPuzzleRewardFn(WordPuzzleRewardConfig())
    agent = WordPuzzleSimpleAgent(
        sampler=sampler,
        env_factory=lambda td: WordPuzzleSimpleEnv(
            task_data=td, reward_fn=reward_fn, step_limit=10, bash_timeout=5
        ),
        max_steps=10,
    )
    td = WordPuzzleTaskData(task_id="t-1", question="What is 2+2?", reference="4")
    return asyncio.run(agent.rollout(task_data=td, rollout_idx=0)), requests, sampled


def test_turn_loop_bash_then_submit_end_to_end():
    from core import atif_extras
    from core.rollout_status import RolloutStatus

    scored, requests, sampled = _run_simple_agent([
        _chat_reply("bash", '{"command": "echo $((2+2))"}', "compute it", "c1"),
        _chat_reply("submit", '{"answer": "4"}', "it printed 4", "c2"),
    ])

    assert atif_extras.get_rollout_status(scored.trajectory) == RolloutStatus.OK
    assert scored.reward.reward > 0
    agent_steps = [s for s in scored.trajectory.steps if s.source == "agent"]
    assert [s.tool_calls[0].function_name for s in agent_steps] == ["bash", "submit"]
    assert [s.reasoning_content for s in agent_steps] == ["compute it", "it printed 4"]
    assert [atif_extras.get_finish_reason(s) for s in agent_steps] == ["STOP", "STOP"]
    # The bash tool really ran and its output came back as the observation.
    assert "4" in agent_steps[0].observation.results[0].content

    # Turn 2's request carries turn 1 as OpenAI history: the assistant call
    # (with its reasoning replayed) and the tool result paired to its id.
    assert len(requests) == 2
    history = requests[1]["messages"]
    assistant = next(m for m in history if m["role"] == "assistant")
    assert assistant["tool_calls"][0]["function"]["name"] == "bash"
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {
        "command": "echo $((2+2))"}
    tool_msg = next(m for m in history if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == assistant["tool_calls"][0]["id"]
    assert "4" in tool_msg["content"]
    assert {t["function"]["name"] for t in requests[1]["tools"]} == {"bash", "submit"}
    # Turn 1's reasoning goes back to the server as reasoning_content (not as
    # visible text), so the model sees its own chain on the next turn.
    assert assistant["reasoning_content"] == "compute it"
    assert assistant["content"] in ("", None)

    # Reasoning tokens: each reply reports completion_tokens=20 incl.
    # reasoning_tokens=8. The agent sees the split on every turn...
    assert [(r.tuning_candidate.generation_token_count, r.tuning_candidate.thoughts_token_count)
            for r in sampled] == [(20, 8), (20, 8)]
    # ...and each ATIF step records the total (reasoning included), like
    # Gemini's steps; StepStats has no separate thoughts field on any backend.
    assert [(s.extra or {}).get("generation_token_count") for s in agent_steps] == [20, 20]


def test_turn_loop_malformed_tool_call_is_flagged_not_executed():
    from core import atif_extras
    from core.rollout_status import RolloutStatus

    # Truncated arguments: missing the closing brace.
    scored, requests, _ = _run_simple_agent([
        _chat_reply("bash", '{"command": "ls"', "list files", "c1"),
    ])

    status = atif_extras.get_rollout_status(scored.trajectory)
    assert status != RolloutStatus.OK and status.is_sampling_failure()
    [step] = [s for s in scored.trajectory.steps if s.source == "agent"]
    # Counted by sampler/finish_reason/MALFORMED_FUNCTION_CALL, like Gemini's.
    assert atif_extras.get_finish_reason(step) == "MALFORMED_FUNCTION_CALL"
    assert step.observation is None  # the broken call was never run
    assert len(requests) == 1  # and the loop did not continue on it


# ---- opt-in guard against degenerate (e.g. [PAD]-run) replies ----------------


def _pad_reply(pads: int) -> dict[str, Any]:
    return {
        "id": "r-pad", "model": "org/model-x",
        "choices": [{"index": 0, "finish_reason": "length", "message": {
            "role": "assistant", "content": "", "reasoning_content": "hm" + "[PAD]" * pads}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 32768, "reasoning_tokens": 32768},
    }


def _guarded(handler: Callable[[httpx.Request], httpx.Response], retries: int) -> OpenAICompatibleSampler:
    return OpenAICompatibleSampler(
        base_url="https://llm.example", model="org/model-x",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        retry=SyncRetryProfile(max_attempts=2, backoff=Backoff(0.01, 0.01, jitter=0.0)),
        degenerate_marker="[PAD]", degenerate_retries=retries,
    )


def test_degenerate_reply_is_retried_then_accepted_when_clean():
    replies = iter([_pad_reply(500), _TOOL_CALL_RESPONSE])
    posts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        posts.append(1)
        return httpx.Response(200, json=next(replies))

    [r] = asyncio.run(_guarded(handler, retries=2).sample_n("q", 1))
    assert (len(posts), r.tuning_candidate.candidate_id, r.finish_reason) == (2, "0", "STOP")


def test_degenerate_reply_fails_the_call_after_retries(caplog: pytest.LogCaptureFixture):
    posts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        posts.append(1)
        return httpx.Response(200, json=_pad_reply(500))

    with caplog.at_level("WARNING", logger="backends.openai_compatible_endpoint"):
        [r] = asyncio.run(_guarded(handler, retries=2).sample_n("q", 1))
    assert len(posts) == 3  # first try + 2 retries
    # Failed like a transport failure (failsafe), not returned as MAX_TOKENS.
    assert r.tuning_candidate.candidate_id == ""
    assert any("degenerate" in rec.getMessage() for rec in caplog.records)
    # The passthrough raises instead, as for any failed call.
    with pytest.raises(RuntimeError, match="upstream sampling failed"):
        asyncio.run(_guarded(handler, retries=0).sample_chat_completion(_Q))


def test_degenerate_guard_is_off_by_default_and_ignores_short_runs():
    # Default sampler: a [PAD] reply is returned as-is (MAX_TOKENS).
    [r] = asyncio.run(_sampler(lambda q: httpx.Response(200, json=_pad_reply(500))).sample_n("q", 1))
    assert r.finish_reason == "MAX_TOKENS"
    # Guard on: a few literal "[PAD]" mentions in normal text are not a run.
    posts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        posts.append(1)
        return httpx.Response(200, json=_pad_reply(3))

    [r] = asyncio.run(_guarded(handler, retries=2).sample_n("q", 1))
    assert (len(posts), r.tuning_candidate.candidate_id) == (1, "0")
