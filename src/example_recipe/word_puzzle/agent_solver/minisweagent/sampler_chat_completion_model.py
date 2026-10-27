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

"""Mini v2 Model adapter that consumes Sampler.sample_chat_completion.

Backend-agnostic wrapper. Any concrete Sampler subclass drops in via
constructor injection. The adapter delegates all wire-format work to
Sampler.sample_chat_completion; this file contains only the mini v2
Model-protocol surface and the SampleResult-side-channel plumbing that
preserves training-relevant metadata onto the trajectory JSON.
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any, NoReturn

from jinja2 import StrictUndefined, Template
from minisweagent.exceptions import FormatError
from minisweagent.models.utils.actions_toolcall import (
    BASH_TOOL,
    format_toolcall_observation_messages,
)
from pydantic import BaseModel

from backends.chat_completion import EmptyCandidatePartsError
from core.protocols import Sampler
from core.sample_result import SampleResult


AVAILABLE_TOOLS: tuple[str, ...] = ("bash", "submit")

SUBMIT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit",
        "description": "Submit your final answer for the puzzle.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The final answer.",
                },
            },
            "required": ["answer"],
        },
    },
}


def _tool_call_to_dict(tc: Any) -> dict[str, Any]:
    """Convert a response-side tool_call into the OpenAI wire dict shape
    that the request-side converter expects.

    Accepts either the ToolCall dataclass (from real converter output)
    or an already-dict-shaped tool_call (from unit-test fixtures or
    LiteLLM-proxy responses that ship dicts on ``choice.message
    .tool_calls`` without dataclass coercion). Shape:
    ``{"id": str, "type": "function", "function":
    {"name": str, "arguments": str}}``. Only these four fields; anything
    else is intentionally dropped so no extras leak into the request.

    Precondition: caller guarantees the tool_call has already been
    validated by ``_parse_tool_calls``. Currently that gates:
      * non-empty ``id`` (string)
      * ``function.name`` in ``AVAILABLE_TOOLS``
      * ``function.arguments`` is valid JSON with the required key
        (``command`` for bash, ``answer`` for submit)

    This helper does not re-validate — it maps the accepted response
    shape into the OpenAI request wire shape. Anything not gated
    upstream (e.g. an unexpected ``type`` value, or missing
    ``function.name``/``arguments`` if the upstream contract loosens)
    lands as its passthrough default and would trip Vertex's
    ``REQUIRED_FIELD_MISSING`` on the next-turn request. Tighten
    upstream, not here.
    """
    tc_id = _tc_field(tc, "id") or ""
    tc_type = _tc_field(tc, "type") or "function"
    fn_name = _tc_field(tc, "function", "name") or ""
    fn_args = _tc_field(tc, "function", "arguments") or ""
    return {
        "id": tc_id,
        "type": tc_type,
        "function": {"name": fn_name, "arguments": fn_args},
    }


def _provider_specific_fields_to_dict(psf: Any) -> dict[str, Any]:
    """Convert ProviderSpecificFields into a pruned dict.

    Accepts either the ProviderSpecificFields dataclass or an
    already-dict-shaped value (parity with ``_tool_call_to_dict``).
    Currently only carries ``thought_signatures`` (a list of base64-
    encoded per-tool_call signatures). Returns an empty dict when the
    input is None or ``thought_signatures`` is missing / an empty list
    — the caller guards with ``if psf_dict:`` (truthy on non-empty
    dicts), so collapsing None and [] to the same empty-return branch
    keeps the emitted wire shape consistent across both cases.

    If ``ProviderSpecificFields`` grows additional fields, extend this
    helper to lift them (and refresh this docstring); the current
    ``thought_signatures``-only shape is the entire contract today.
    """
    if psf is None:
        return {}
    signatures = _tc_field(psf, "thought_signatures")
    # ``not signatures`` covers both ``None`` and ``[]`` — the empty-list
    # case would otherwise emit ``{"thought_signatures": []}``, which
    # is benign at the wire boundary (the request-side reader guards
    # ``if not sigs``) but inconsistent with the None branch.
    if not signatures:
        return {}
    return {"thought_signatures": list(signatures)}


class SamplerChatCompletionModelConfig(BaseModel):
    model_name: str = "gemini-3.5-flash"
    """Model id sent in the OpenAI request. Recorded on the trajectory
    for provenance; the actual backend is fixed by the injected sampler."""

    env: str = "prod"
    """Recorded on trajectory for provenance. The actual endpoint is
    fixed by the injected sampler."""

    max_generation_tokens: int = 65536
    """Recorded on trajectory for provenance. The actual limit is
    fixed by the injected sampler."""

    thinking_level: str | None = "medium"
    """Passed to sampler.sample_chat_completion() on every call."""

    format_error_template: str = (
        "Your last response did not call one of the available tools"
        " ({{ available_tools }}). {{ error }}"
        " Every response MUST invoke at least one tool call."
        "{% if finish_reason == 'MAX_TOKENS' %}"
        " Note: your previous response was truncated at max_tokens;"
        " keep reasoning concise and call the tool sooner.{% endif %}"
        "{% if finish_reason == 'malformed_function_call' %}"
        " Note: the model produced a malformed function call that could"
        " not be parsed; ensure your tool call has valid JSON arguments"
        " and uses one of the available tools.{% endif %}"
    )

    observation_template: str = (
        "<returncode>{{ output.returncode }}</returncode>\n"
        "<output>\n{{ output.output }}</output>"
    )


class SamplerChatCompletionModel:
    """Mini v2 Model adapter over Sampler.sample_chat_completion.

    Backend-agnostic: works with any concrete Sampler subclass. Enables
    the same mini-swe-agent code path to run against any registered
    backend for both cheap integration coverage and production runs.

    Wire path: mini's messages -> sampler.sample_chat_completion() ->
    ChatCompletionSampleResult. tool_calls parsed off
    .response.choices[0].message. The underlying list[SampleResult] is
    threaded into message['extra'] via _sample_result_extras so
    training-relevant metadata (candidate_id, response_id,
    thoughts_token_count, thought_signature) is preserved for the
    trajectory JSON.
    """

    def __init__(
        self,
        sampler: Sampler,
        *,
        on_query: Any = None,
        capture_ctx: Any | None = None,
        **config_kwargs: Any,
    ):
        self._sampler = sampler
        self.config = SamplerChatCompletionModelConfig(**config_kwargs)
        # Optional CaptureContext for raw LLM request/response recording.
        # When set, every query() call passes [capture_ctx] as
        # capture_ctxs to sampler.sample_chat_completion(), which
        # forwards it to sampler.sample(). Each call writes a
        # call_NNNNNN.json blob under capture_ctx.trial_dir.
        self._capture_ctx = capture_ctx
        # Optional callback fired at the START of every query() call (before
        # the sampler round-trip). Used by the training prober to count
        # active LLM steps per rollout. Signature: () -> None.
        self._on_query = on_query
        # Cumulative sampler-call wall time across SUCCESSFUL query()
        # invocations (both content-success and FormatError paths accumulate;
        # RuntimeError / ValueError / other exception paths from the sampler
        # itself DO NOT). Emitted in serialize() under
        # info.config.total_sample_seconds so extract_answer's summary CSV
        # can report per-task LLM latency.
        #
        # Excluding failed-turn latency is deliberate: including it would
        # inflate the t_llm signal asymmetrically depending on per-task
        # failure rate (a task that hit 3 backend errors before succeeding
        # would look 3x slower than a task that got there in one call).
        # Callers who need raw wall time spent trying to sample should
        # add a separate counter around a try/finally.
        self._total_sample_seconds: float = 0.0

    # ------------------------------------------------------------------
    # Model protocol methods
    # ------------------------------------------------------------------

    async def query(self, messages: list[dict], **kwargs: Any) -> dict:
        if self._on_query is not None:
            self._on_query()
        request = {
            "model": self.config.model_name,
            "messages": self._strip_extras(messages),
            "tools": [BASH_TOOL, SUBMIT_TOOL],
            **kwargs,
        }
        t0 = time.perf_counter()
        try:
            ret = await self._sampler.sample_chat_completion(
                request,
                num_samples=1,
                thinking_level=self.config.thinking_level,
                capture_ctxs=([self._capture_ctx] if self._capture_ctx else None),
            )
        except EmptyCandidatePartsError as e:
            # Vertex returned a candidate with ``parts=[]`` (finish_reason
            # was NOT a content-filter type — those are handled by the
            # converter and return ``content=None`` gracefully). Observed
            # in ~13% of long-running rollouts against real Vertex during
            # a baseline eval reproduction (2026-07-14: 19 crashes
            # in 150 tasks, all this same exception message).
            #
            # Emit a zero-actions failsafe message so the rollout can
            # continue: the agent's next ``step()`` sees no tool_calls,
            # hits the format-error retry path, and eventually
            # ``LimitsExceeded``-out with a partial trajectory. Matches
            # the pre-migration ``GeminiSamplerModel._parse_response``
            # behavior which turned this failure mode into
            # ``LimitsExceeded`` (partial trajectory usable for
            # debugging) instead of ``Crash`` (no trajectory at all).
            #
            # NOT catching plain ``ValueError`` here because the
            # sampler layer signals malformed-input BUGS (openai_request
            # missing required fields, etc.) via ``ValueError`` too —
            # those should surface loudly. ``EmptyCandidatePartsError``
            # is the specific subclass added at chat_completion.py:2094
            # for this Vertex-flake pattern.
            elapsed = time.perf_counter() - t0
            self._total_sample_seconds += elapsed
            return self._make_message(
                content=(
                    f"[SamplerChatCompletionModel: Vertex returned an empty-parts"
                    f" candidate ({e}); no actions to execute this turn.]"
                ),
                actions=[],
                elapsed=elapsed,
                finish_reason="OTHER",
                sample_result=None,
            )
        elapsed = time.perf_counter() - t0
        # Accumulate BEFORE any early return / raise so both exit paths
        # (FormatError, success) count toward the per-task latency total
        # that extract_answer reads out of serialize().
        self._total_sample_seconds += elapsed

        # Note: no local failsafe guard here for other failure modes.
        # ``Sampler.sample_chat_completion`` (core/protocols.py) raises
        # ``RuntimeError`` on ``candidate_id == ""`` and
        # ``reconstitute_generate_content_response`` raises ``ValueError``
        # on empty ``results``; both surface as exceptions from the
        # ``await`` above and propagate to the task-level catcher.
        # Suppressing them here with a synthetic "no candidates" message
        # would mask genuine backend failures (HTTP / LRO errors
        # operators need to see in sampler logs). ``EmptyCandidatePartsError``
        # is the ONE narrowly-scoped exception we treat as recoverable
        # because it's an observed Vertex-side flake, not an infra bug.

        choice = ret.response.choices[0]
        finish_reason = choice.finish_reason or "unknown"
        sample_result = ret.results[0]
        tool_calls = choice.message.tool_calls or []

        try:
            actions = _parse_tool_calls(
                tool_calls,
                format_error_template=self.config.format_error_template,
                finish_reason=finish_reason,
            )
        except FormatError as e:
            # Attach SampleResult extras to the FormatError's user
            # message so the trajectory JSON still carries the metadata
            # from the failed step.
            if e.messages:
                e.messages[0]["extra"] = e.messages[0].get("extra", {}) | {
                    "response_finish_reason": finish_reason,
                    **_sample_result_extras(sample_result),
                }
            raise

        return self._make_message(
            content=choice.message.content or "",
            actions=actions,
            elapsed=elapsed,
            finish_reason=finish_reason,
            sample_result=sample_result,
            raw_message=choice.message,
        )

    def format_message(self, **kwargs: Any) -> dict:
        # Text-only pass-through; no multimodal support (mirrors mini's
        # LitellmModel when multimodal_regex is empty).
        return dict(kwargs)

    def format_observation_messages(
        self,
        message: dict,
        outputs: list[dict],
        template_vars: dict | None = None,
    ) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        return format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex="",
        )

    def get_template_vars(self, **_kwargs: Any) -> dict[str, Any]:
        return self.config.model_dump()

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": (
                        f"{self.__class__.__module__}.{self.__class__.__name__}"
                    ),
                    "total_sample_seconds": self._total_sample_seconds,
                    "tools": [BASH_TOOL, SUBMIT_TOOL],
                },
            }
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_extras(messages: list[dict]) -> list[dict]:
        """Drop the mini-only 'extra' key before sending to the sampler.
        Mirrors LitellmModel._prepare_messages_for_api behavior."""
        return [{k: v for k, v in m.items() if k != "extra"} for m in messages]

    def _make_message(
        self,
        *,
        content: str,
        actions: list[dict],
        elapsed: float,
        finish_reason: str,
        sample_result: SampleResult | None,
        raw_message: Any = None,
    ) -> dict:
        # Multi-turn correctness: the assistant message must carry the
        # OpenAI-format ``tool_calls`` list (id + function.name + arguments)
        # so the next-turn request-converter can populate its
        # tool_call_id -> tool_name map. Without this, the follow-up
        # ``role="tool"`` observation converts to a Gemini functionResponse
        # Part with an empty ``name`` field, which Vertex rejects with
        # REQUIRED_FIELD_MISSING. See chat_completion.py:1173 for the
        # lookup site.
        #
        # Also preserve ``provider_specific_fields`` so the thought-signature
        # roundtrip bridge (chat_completion.py, request-side) can echo per-
        # tool_call signatures back to Vertex on the next turn.
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": content,
            "extra": {
                "actions": actions,
                "cost": 0.0,
                "elapsed_seconds": elapsed,
                "finish_reason": finish_reason,
                "timestamp": time.time(),
                **_sample_result_extras(sample_result),
            },
        }
        # Carry this turn's opaque reasoning blocks (LiteLLM ``thinking_blocks``
        # standard) on the assistant message so a backend that replays them
        # across the tool loop (the Anthropic endpoint requires the prior turn's
        # signed thinking to be sent back verbatim during function calling) can
        # find them on the next request. Backend-neutral: providers that don't
        # emit thinking blocks (e.g. Gemini, which uses its own
        # ``provider_specific_fields`` thought-signature carrier) leave this
        # absent, so this is a no-op for them. Must be on EVERY assistant turn
        # (incl. tool-call turns), not gated on tool_calls.
        if sample_result is not None:
            tblocks = getattr(sample_result.context, "thinking_blocks", None)
            if tblocks:
                msg["thinking_blocks"] = list(tblocks)
        if raw_message is not None and getattr(raw_message, "tool_calls", None):
            psf = getattr(raw_message, "provider_specific_fields", None)
            psf_dict = _provider_specific_fields_to_dict(psf)
            raw_sigs = psf_dict.get("thought_signatures") if psf_dict else None
            if isinstance(raw_sigs, list):
                for i, action in enumerate(actions):
                    if action.get("tool_name") == "submit" and i < len(raw_sigs):
                        # Stash the signature onto the action dict so
                        # ``_restore_submit_tool_call`` can put it back
                        # into the correct psf.thought_signatures slot
                        # if this submit turns out to be non-terminal.
                        action["thought_signature"] = raw_sigs[i]

            filtered_tool_calls = [
                _tool_call_to_dict(tc)
                for tc, action in zip(raw_message.tool_calls, actions)
                if action.get("tool_name") != "submit"
            ]
            if filtered_tool_calls:
                msg["tool_calls"] = filtered_tool_calls
                # provider_specific_fields carries thought_signatures
                # ALIGNED to the original tool_calls list. Since we
                # filter tool_calls, we also need to filter the aligned
                # signature list to preserve the tc_index -> signature
                # mapping the request-side bridge relies on.
                if psf_dict:
                    sigs = psf_dict.get("thought_signatures")
                    if isinstance(sigs, list):
                        aligned_sigs = [
                            sig
                            for sig, action in zip(sigs, actions)
                            if action.get("tool_name") != "submit"
                        ]
                        # Collapse to absent if no aligned entry carries
                        # a non-empty signature. This is stricter than
                        # the emit-side ``_build_provider_specific_fields``
                        # (which only drops when the raw list is empty)
                        # — after the submit-filter we can be left with
                        # an all-empty-string list where the strict-drop
                        # emit policy would still emit noise; treat that
                        # as "no useful signatures to echo" and drop.
                        if any(aligned_sigs):
                            msg["provider_specific_fields"] = {
                                "thought_signatures": aligned_sigs
                            }
                    else:
                        # Unreachable today: ``_provider_specific_fields_to_dict``
                        # only returns ``{}`` or ``{"thought_signatures":
                        # list}``, so ``sigs`` is always a list when
                        # ``psf_dict`` is truthy. Kept as a forward-compat
                        # passthrough for when ``ProviderSpecificFields``
                        # grows non-``thought_signatures`` fields — at that
                        # point the extractor helper will return them under
                        # a non-list key and this branch will start firing.
                        msg["provider_specific_fields"] = psf_dict
        return msg


# ----------------------------------------------------------------------
# Tool-call parsing
# ----------------------------------------------------------------------
#
# Kept as a module-level helper (not on the class) so it stays testable
# in isolation and doesn't accidentally depend on adapter state.
#
# Custom parser rationale: mini v2.4.5's parse_toolcall_actions is
# bash-only and expects attribute-access on ToolCall objects. This
# adapter needs to support (a) both 'bash' and 'submit' and (b) both
# dict-shaped and attribute-shaped tool_calls (dicts appear in the
# hermetic unit-test fixtures; ToolCall objects appear in real
# response-side conversion). Rebuilt to cover both cases.


def _tc_field(tc: Any, *path: str) -> Any:
    """Access a nested field on a tool_call whether it's a dict or an
    attribute object. Returns None on any missing intermediate."""
    obj: Any = tc
    for p in path:
        if obj is None:
            return None
        if isinstance(obj, dict):
            obj = obj.get(p)
        else:
            obj = getattr(obj, p, None)
    return obj


def _render_format_error(
    template: str,
    *,
    error: str,
    finish_reason: str,
) -> str:
    return Template(template, undefined=StrictUndefined).render(
        error=error,
        available_tools=", ".join(AVAILABLE_TOOLS),
        finish_reason=finish_reason,
    )


def _raise_format_error(
    template: str,
    *,
    error: str,
    finish_reason: str,
) -> NoReturn:
    raise FormatError(
        {
            "role": "user",
            "content": _render_format_error(
                template, error=error, finish_reason=finish_reason
            ),
            "extra": {
                "interrupt_type": "FormatError",
                "response_finish_reason": finish_reason,
            },
        }
    )


def _parse_tool_calls(
    tool_calls: list[Any],
    *,
    format_error_template: str,
    finish_reason: str,
) -> list[dict[str, Any]]:
    """Parse a response's tool_calls list into mini action dicts.

    Accepts dict-shaped tool_calls (as unit-test fixtures produce) and
    attribute-shaped ToolCall dataclass instances (as the real converter
    produces). Enforces the (``bash``, ``submit``) allowlist.

    ``finish_reason`` is forwarded to the format-error template so the
    rendered message can adapt (e.g. adding a truncation hint when the
    backend cut off at max_tokens).

    Raises FormatError on:
      * empty tool_calls list
      * unknown tool name
      * malformed JSON arguments
      * missing required argument (``command`` for bash, ``answer`` for submit)
    """
    if not tool_calls:
        _raise_format_error(
            format_error_template,
            error=(
                "No tool calls found in the response. Every response MUST"
                " include at least one tool call."
            ),
            finish_reason=finish_reason,
        )

    actions: list[dict[str, Any]] = []
    for tc in tool_calls:
        name = _tc_field(tc, "function", "name")
        raw_args = _tc_field(tc, "function", "arguments")
        tc_id = _tc_field(tc, "id")

        # ``id`` is required by both the OpenAI spec and Vertex's request
        # bridge (``functionResponse.name`` lookup keys off tool_call_id).
        # Missing / empty id here would produce a ``{"id": ""}`` in the
        # assistant message and Vertex would reject the next turn with
        # ``REQUIRED_FIELD_MISSING`` on ``functionResponse.name``. Fail
        # fast so the FormatError feedback loop can steer the model
        # instead of blowing up on the wire two turns later.
        if not isinstance(tc_id, str) or tc_id == "":
            _raise_format_error(
                format_error_template,
                error=(
                    f"Tool call missing 'id' field (got {tc_id!r}); every "
                    f"tool call MUST carry a non-empty string id."
                ),
                finish_reason=finish_reason,
            )

        if name not in AVAILABLE_TOOLS:
            _raise_format_error(
                format_error_template,
                error=(
                    f"Unknown tool {name!r}; must be one of {list(AVAILABLE_TOOLS)}."
                ),
                finish_reason=finish_reason,
            )

        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except (json.JSONDecodeError, TypeError) as exc:
            _raise_format_error(
                format_error_template,
                error=f"Error parsing tool call arguments: {exc}.",
                finish_reason=finish_reason,
            )

        if not isinstance(args, dict):
            _raise_format_error(
                format_error_template,
                error=(
                    f"Tool call arguments must be a JSON object, got "
                    f"{type(args).__name__}."
                ),
                finish_reason=finish_reason,
            )

        if name == "bash":
            if "command" not in args:
                _raise_format_error(
                    format_error_template,
                    error="Missing 'command' argument in bash tool call.",
                    finish_reason=finish_reason,
                )
            actions.append(
                {
                    "tool_name": "bash",
                    "command": args["command"],
                    "tool_call_id": tc_id,
                }
            )
        else:  # submit
            if "answer" not in args:
                _raise_format_error(
                    format_error_template,
                    error="Missing 'answer' argument in submit tool call.",
                    finish_reason=finish_reason,
                )
            actions.append(
                {
                    "tool_name": "submit",
                    "answer": args["answer"],
                    "tool_call_id": tc_id,
                }
            )

    return actions


# ----------------------------------------------------------------------
# SampleResult -> extras conversion
# ----------------------------------------------------------------------


def _sample_result_extras(res: SampleResult | None) -> dict[str, Any]:
    """Extract training-relevant metadata from a SampleResult.

    When res is None (failsafe empty result path), every field is None
    so the trajectory JSON still has the keys but marked as missing.
    Field names + semantics match the ATIF convention used by the
    trajectory JSON writer downstream.
    """
    if res is None:
        return {
            "candidate_id": None,
            "response_id": None,
            "operation_id": None,
            "model_version": None,
            "prompt_token_count": None,
            "generation_token_count": None,
            "thoughts_token_count": None,
            "thought_signatures_b64": [],
        }
    ctx = res.context
    tc = res.tuning_candidate
    extras: dict[str, Any] = {
        "candidate_id": tc.candidate_id,
        "response_id": ctx.response_id,
        "operation_id": ctx.operation_id,
        "model_version": ctx.model_version,
        "prompt_token_count": ctx.prompt_token_count,
        "generation_token_count": tc.generation_token_count,
        "thoughts_token_count": tc.thoughts_token_count,
        "thought_signatures_b64": _extract_thought_signatures_b64(tc.candidate),
    }
    # URI of the raw LLM call blob (written by raw_capture.dump_raw).
    # Stamped on the trajectory step as llm_call_uri per the ATIF contract.
    if ctx.raw_call_uri is not None:
        extras["raw_call_uri"] = ctx.raw_call_uri
    return extras


def _extract_thought_signatures_b64(candidate: Any) -> list[str]:
    """Base64-encode every non-empty Part.thought_signature on this
    candidate, in parts-order. Returns an empty list if no part carries one.

    This is an ANALYTICS carrier only. The authoritative wire-roundtrip
    carrier is the tool_call-aligned ``provider_specific_fields.thought_signatures``
    on the assistant message (see
    ``convert_gemini_response_to_openai``), which preserves per-tool_call
    signature-to-call mapping. This list is a flat parts-order dump
    for observability / debugging (e.g. counting signatures per turn,
    or checking for missing signatures across a run).

    Kept as a small helper so the SampleResult -> extras conversion
    stays testable in isolation.
    """
    content = getattr(candidate, "content", None)
    if not content:
        return []
    parts = getattr(content, "parts", None) or []
    signatures: list[str] = []
    for part in parts:
        raw = bytes(getattr(part, "thought_signature", b"") or b"")
        if raw:
            signatures.append(base64.b64encode(raw).decode("ascii"))
    return signatures
