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

"""Translate OpenAI Chat Completions requests/responses to/from Vertex generateContent.

Pure translation between OpenAI Chat Completions and Vertex
``generateContent`` wire formats. No network I/O, no SDK dependency.

Public entry points:

  * ``convert_openai_request_to_gemini()`` -- OpenAI request dict ->
    Vertex ``generateContent`` request dict (camelCase JSON).
  * ``convert_gemini_response_to_openai()`` -- Vertex
    ``generateContent`` response proto -> OpenAI Chat Completions
    response object (LiteLLM-proxy wire-compatible via ``.to_dict()``).

and the reverse pair, for driving an OpenAI-compatible server (e.g. an
SGLang or vLLM deployment) from Gemini-shaped callers:

  * ``convert_gemini_request_to_openai()`` -- camelCase ``generateContent``
    request dict -> OpenAI Chat Completions request dict.
  * ``convert_openai_response_to_gemini()`` -- OpenAI Chat Completions
    response dict -> ``generateContent`` response JSON shape.

plus ``thinking_blocks_to_reasoning_content()`` /
``reasoning_content_to_thinking_blocks()``, which map this module's
``thinking_blocks`` to and from such servers' ``reasoning_content`` for
OpenAI-native passthrough (see ``backends.openai_compatible_endpoint``).

Scope (forward pair): text messages, function calling (``tools`` / ``tool_choice`` /
``tool_calls`` and the legacy ``functions`` / ``function_call``), and
the Google-specific ``extra_content.google.{thought, thought_signature}``
extension on assistant messages. Multimodal parts, ``response_format``,
``reasoning_effort``, and ``extra_body`` raise ``NotImplementedError``.

Errors: ``ValueError`` on malformed input; ``NotImplementedError`` for
unsupported features. Callers get loud, greppable failures rather than
silent partial output.
"""

from __future__ import annotations

import base64
import copy
import json
import re
import time
import uuid
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Dict, List, Literal, NamedTuple, Optional, Sequence, Tuple

from google.cloud.aiplatform_v1.types import (
    Candidate,
    GenerateContentResponse,
    Part,
    UsageMetadata,
)

# ── OpenAI / Vertex field-name constants ──────────────────────────────
# Plain string constants for OpenAI and Vertex wire schemas.

# Top-level OpenAI request fields.
K_MODEL_FIELD = "model"
K_STREAM_FIELD = "stream"
K_MESSAGES_FIELD = "messages"

# OpenAI reasoning content (the ``<think>...</think>`` prefix convention).
K_REASONING_CONTENT_FIELD = "reasoning_content"
K_REASONING_CONTENT_OPEN_TAG = "<think>"
K_REASONING_CONTENT_CLOSE_TAG = "</think>"

# Message content shape.
K_CONTENT_FIELD = "content"
K_CONTENT_SEPARATOR = "\n"

# Content-part type discriminators.
K_TYPE_FIELD = "type"
K_TYPE_TEXT = "text"
K_TYPE_IMAGE_URL = "image_url"
K_TYPE_INPUT_AUDIO = "input_audio"
K_TYPE_REFUSAL = "refusal"

# Content-part field names.
K_TEXT_FIELD = "text"
K_IMAGE_URL_FIELD = "image_url"
K_URL_FIELD = "url"
K_DETAIL_FIELD = "detail"
K_INPUT_AUDIO_FIELD = "input_audio"
K_DATA_FIELD = "data"
K_FORMAT_FIELD = "format"

# Detail-level values for image_url parts.
K_DETAIL_HIGH = "high"
K_DETAIL_LOW = "low"
K_DETAIL_MEDIUM = "medium"
K_DETAIL_AUTO = "auto"

# Role values.
K_ROLE_FIELD = "role"
K_ROLE_SYSTEM = "system"
K_ROLE_DEVELOPER = "developer"
K_ROLE_ASSISTANT = "assistant"
K_ROLE_MODEL = "model"
K_ROLE_USER = "user"
K_ROLE_TOOL = "tool"
K_ROLE_FUNCTION = "function"

# Tool / function-call fields.
K_TOOL_CALL_ID_FIELD = "tool_call_id"
K_NAME_FIELD = "name"
K_TOOLS_FIELD = "tools"
K_FUNCTIONS_FIELD = "functions"
K_TOOL_CHOICE_FIELD = "tool_choice"
K_FUNCTION_CALL_FIELD = "function_call"

# tool_choice / function_call special values.
K_TOOL_CHOICE_NONE = "none"
K_TOOL_CHOICE_AUTO = "auto"
K_TOOL_CHOICE_REQUIRED = "required"
K_TOOL_CHOICE_VALIDATED = "validated"
K_TOOL_CHOICES = frozenset(
    {
        K_TOOL_CHOICE_NONE,
        K_TOOL_CHOICE_AUTO,
        K_TOOL_CHOICE_REQUIRED,
        K_TOOL_CHOICE_VALIDATED,
    }
)
K_FUNCTION_CALLS = frozenset({K_TOOL_CHOICE_NONE, K_TOOL_CHOICE_AUTO})

# Tool object shape.
K_TOOL_TYPE_FUNCTION = "function"
K_FUNCTION_FIELD = "function"
K_DESCRIPTION_FIELD = "description"
K_PARAMETERS_FIELD = "parameters"
K_TOOL_CALLS_FIELD = "tool_calls"
K_ID_FIELD = "id"
K_ARGUMENTS_FIELD = "arguments"
K_AUDIO_FIELD = "audio"

# OpenAI response envelope fields.
K_OBJECT_FIELD = "object"
K_INDEX_FIELD = "index"
K_DELTA_FIELD = "delta"
K_MESSAGE_FIELD = "message"
K_CHOICES_FIELD = "choices"

# Generation-config knobs (OpenAI request-level).
K_TEMPERATURE_FIELD = "temperature"
K_TOP_P_FIELD = "top_p"
K_N_FIELD = "n"
K_MAX_TOKENS_FIELD = "max_tokens"
K_MAX_COMPLETION_TOKENS_FIELD = "max_completion_tokens"
K_STOP_FIELD = "stop"
K_FREQUENCY_PENALTY_FIELD = "frequency_penalty"
K_PRESENCE_PENALTY_FIELD = "presence_penalty"
K_SEED_FIELD = "seed"
K_TOP_LOG_PROBS_FIELD = "top_logprobs"
K_REASONING_EFFORT_FIELD = "reasoning_effort"

# reasoning_effort values (OpenAI shape). Translation to thinkingConfig
# not yet implemented; currently raises NotImplementedError.
K_REASONING_EFFORT_LOW = "low"
K_REASONING_EFFORT_MEDIUM = "medium"
K_REASONING_EFFORT_HIGH = "high"
K_REASONING_EFFORT_MINIMAL = "minimal"
K_REASONING_EFFORT_CHOICES = frozenset(
    {
        K_REASONING_EFFORT_MINIMAL,
        K_REASONING_EFFORT_LOW,
        K_REASONING_EFFORT_MEDIUM,
        K_REASONING_EFFORT_HIGH,
    }
)

# Metadata (client-supplied labels).
K_METADATA_FIELD = "metadata"

# Response usage-block fields (OpenAI shape).
K_USAGE_FIELD = "usage"
K_COMPLETION_TOKENS_FIELD = "completion_tokens"
K_PROMPT_TOKENS_FIELD = "prompt_tokens"
K_TOTAL_TOKENS_FIELD = "total_tokens"
K_REASONING_TOKENS_FIELD = "reasoning_tokens"

# Finish-reason values (OpenAI shape).
K_FINISH_REASON_FIELD = "finish_reason"
K_FINISH_REASON_OTHER = "other"
K_FINISH_REASON_MALFORMED_FUNCTION_CALL = "malformed_function_call"
K_FINISH_REASON_UNEXPECTED_TOOL_CALL = "unexpected_tool_call"
K_FINISH_REASON_STOP = "stop"
K_FINISH_REASON_LENGTH = "length"
K_FINISH_REASON_CONTENT_FILTER = "content_filter"
K_FINISH_REASON_TOOL_CALLS = "tool_calls"

# response_format shape.
K_RESPONSE_FORMAT_FIELD = "response_format"
K_RESPONSE_FORMAT_TEXT = "text"
K_RESPONSE_FORMAT_JSON_OBJECT = "json_object"
K_RESPONSE_FORMAT_JSON_SCHEMA = "json_schema"
K_JSON_SCHEMA_FIELD = "json_schema"
K_MAX_JSON_SCHEMA_NAME_LENGTH = 64
K_SCHEMA_FIELD = "schema"
K_STRICT_FIELD = "strict"

# Streaming options.
K_STREAM_OPTIONS_FIELD = "stream_options"
K_INCLUDE_USAGE_FIELD = "include_usage"

# Audio format identifiers.
K_FORMAT_MP3 = "mp3"
K_FORMAT_WAV = "wav"

# Modalities.
K_MODALITIES_FIELD = "modalities"
K_MODALITY_AUDIO = "audio"
K_MODALITY_IMAGE = "image"
K_MODALITY_TEXT = "text"
K_MODALITIES = frozenset({K_MODALITY_AUDIO, K_MODALITY_TEXT, K_MODALITY_IMAGE})
K_MIME_TYPE_FIELD = "mime_type"

# Response-envelope misc.
K_SYSTEM_FINGERPRINT_FIELD = "system_fingerprint"
K_CREATED_FIELD = "created"
K_LOG_PROBS_FIELD = "logprobs"

# logprobs per-token detail.
K_TOKEN_FIELD = "token"
K_LOG_PROB_FIELD = "logprob"
K_BYTES_FIELD = "bytes"

# Regex patterns (compiled lazily).
K_IMAGE_DATA_BASE64_PATTERN = r"data:([^/]+/[^;]+);base64,(.*)"
K_YOUTUBE_URL_PREFIX_PATTERN = r"(http(s?)://)?(((www|m)\.)?youtube\.com|youtu\.be)/.*"
K_URL_PREFIX_PATTERN = r"(http(s?)://|gs://)"

# MIME types.
K_IMAGE_MIME_TYPE = "image/png"
K_TEXT_PLAIN_MIME_TYPE = "text/plain"
K_WEBM_MIME_TYPE = "video/webm"
K_AUDIO_MPEG_MIME_TYPE = "audio/mpeg"
K_AUDIO_WAV_MIME_TYPE = "audio/wav"

# input_audio format -> MIME. Unmapped format strings are used verbatim.
K_INPUT_AUDIO_FORMAT_TO_MIME: Dict[str, str] = {
    K_FORMAT_MP3: K_AUDIO_MPEG_MIME_TYPE,
    K_FORMAT_WAV: K_AUDIO_WAV_MIME_TYPE,
}

K_TEXT_MIME_TYPE_PREFIX = "text/"
K_IMAGE_MIME_TYPE_PREFIX = "image/"
K_AUDIO_MIME_TYPE_PREFIX = "audio/"

# Refusal field (assistant response only).
K_REFUSAL_FIELD = "refusal"

# Vertex Part-type discriminators used by multimodal Parts.
K_INLINE_DATA_FIELD = "inlineData"
K_FILE_DATA_FIELD = "fileData"
K_FILE_URI_FIELD = "fileUri"

# URI-scheme prefixes for content-part `url` / `data` fields.
K_DATA_URI_PREFIX = "data:"
K_GS_URI_PREFIX = "gs://"
K_HTTP_URI_PREFIX = "http://"
K_HTTPS_URI_PREFIX = "https://"

# Vertex mediaResolution enum values (wire strings).
K_MEDIA_RESOLUTION_FIELD = "mediaResolution"
K_MEDIA_RESOLUTION_LEVEL_FIELD = "level"
K_MEDIA_RESOLUTION_UNSPECIFIED = "MEDIA_RESOLUTION_UNSPECIFIED"
K_MEDIA_RESOLUTION_LOW = "MEDIA_RESOLUTION_LOW"
K_MEDIA_RESOLUTION_MEDIUM = "MEDIA_RESOLUTION_MEDIUM"
K_MEDIA_RESOLUTION_HIGH = "MEDIA_RESOLUTION_HIGH"

# Detail-level -> mediaResolution enum. K_DETAIL_AUTO deliberately
# absent — "auto" means Vertex default.
K_DETAIL_TO_MEDIA_RESOLUTION: Dict[str, str] = {
    K_DETAIL_HIGH: K_MEDIA_RESOLUTION_HIGH,
    K_DETAIL_MEDIUM: K_MEDIA_RESOLUTION_MEDIUM,
    K_DETAIL_LOW: K_MEDIA_RESOLUTION_LOW,
}

# Google-specific extension fields on Vertex responses.
K_GOOGLE_THOUGHT_SIGNATURE_FIELD = "thought_signature"
K_GOOGLE_THOUGHT_FIELD = "thought"

# Prompt/completion token detail sub-fields.
K_PROMPT_TOKENS_DETAILS_FIELD = "prompt_tokens_details"
K_COMPLETION_TOKENS_DETAILS_FIELD = "completion_tokens_details"
K_AUDIO_TOKENS_FIELD = "audio_tokens"
K_CACHED_TOKENS_FIELD = "cached_tokens"

# Extra-body namespaces used by OpenAI extensions.
K_CHAT_COMPLETIONS_EXTRA_BODY_FIELD = "extra_body"
K_CHAT_COMPLETIONS_EXTRA_CONTENT_FIELD = "extra_content"

# Nested-namespace key for Google-specific extension inside extra_content /
# extra_body (e.g. ``extra_content.google.thought_signature``).
K_OPEN_API_GOOGLE_EXTRA_FIELD = "google"

# LiteLLM-parity envelope for thought signatures. On the response side we
# emit them as ``provider_specific_fields.thought_signatures`` (top-level
# list, one per functionCall Part in tool_calls order); on the request
# side we read from the same envelope so a response flows straight into
# the next-turn request. When present, this envelope takes precedence
# over per-tool_call ``extra_content.google.thought_signature``.
K_CHAT_COMPLETIONS_PROVIDER_SPECIFIC_FIELDS_FIELD = "provider_specific_fields"
K_LITELLM_THOUGHT_SIGNATURES_FIELD = "thought_signatures"

# Bypass sentinel: when a client sends this as ``thought_signature``,
# skip base64 decoding and pass through verbatim onto the Part.
K_DUMMY_THOUGHT_SIGNATURE = "skip_thought_signature_validator"

# Thought-tag marker formatting.
K_THOUGHT_TAG_MARKER_OPEN_TAG_FORMAT = "<%s>\n"
K_THOUGHT_TAG_MARKER_CLOSE_TAG_FORMAT = "\n</%s>\n\n"
K_DEFAULT_THOUGHT_TAG_MARKER = "think"


# ── Gemini model-family classification ────────────────────────────────
# Heuristic substring match on well-known marketing names in the model id.

GEMINI_MODEL_KIND_FLASH = "FLASH"
GEMINI_MODEL_KIND_PRO = "PRO"
GEMINI_MODEL_KIND_FLASH_LITE = "FLASH_LITE"
GEMINI_MODEL_KIND_NONE = "NONE"


def get_gemini_model_kind(model_name: str) -> str:
    """Classify a model name into a ``GEMINI_MODEL_KIND_*`` string.

    The heuristic is order-sensitive: ``flash-lite`` must match before
    ``flash`` (which is a substring), and ``pro`` last.
    """
    name = model_name.lower()
    if "flash-lite" in name or "flash_lite" in name:
        return GEMINI_MODEL_KIND_FLASH_LITE
    if "flash" in name:
        return GEMINI_MODEL_KIND_FLASH
    if "pro" in name:
        return GEMINI_MODEL_KIND_PRO
    return GEMINI_MODEL_KIND_NONE


def use_per_part_media_resolution(model_name: str) -> bool:
    """Whether the model supports per-part media_resolution (Gemini 3+).

    Older Gemini 2.x only supported request-level
    ``generationConfig.mediaResolution``; 3.x adds a per-Part override.
    """
    name = model_name.lower()
    return "gemini-3" in name or "gemini_3" in name


def use_thinking_level(model_name: str) -> bool:
    """Whether the model exposes the ThinkingLevel enum (vs. thinkingBudget int).

    Gemini 3.x + newer Flash variants use the ThinkingLevel enum on
    ``generationConfig.thinkingConfig``; older models take a
    ``thinkingBudget`` integer instead.
    """
    name = model_name.lower()
    return "gemini-3" in name or "gemini_3" in name


# ── Request-conversion context ────────────────────────────────────────


class ConversionContext:
    """Ancillary knobs passed alongside the OpenAI request dict.

    Attributes:
      publisher_model_name: When set, overrides the input ``model`` field
        in the outbound Vertex request. Used by dispatchers that route by
        publisher-model but want a normalized model name outbound.
      use_model_role: When True, rewrite ``assistant`` to ``model`` in
        the Vertex output. Defaults to True: plain Vertex v1beta1 accepts
        both ``assistant`` and ``model``, but the tuning-scope endpoint
        (``GenerateContentTuningScope``, used by the MSRL sampler) rejects
        ``assistant`` -- it only accepts ``user``, ``model``, or empty. So
        the safe default for every path is to emit ``model``. Set False
        only if a caller specifically needs the raw OpenAI ``assistant``
        role preserved on the wire.
      large_model_name: Full model name used for kind classification
        (see ``get_gemini_model_kind``). Independent from the request's
        ``model`` field because dispatchers may resolve a publisher name
        to a specific tuned checkpoint before conversion.
    """

    def __init__(
        self,
        publisher_model_name: Optional[str] = None,
        use_model_role: bool = True,
        large_model_name: str = "",
    ) -> None:
        self.publisher_model_name = publisher_model_name
        self.use_model_role = use_model_role
        self.large_model_name = large_model_name


# ── Internal helpers (single-message translation) ─────────────────────


def _text_part(text: str) -> Dict[str, Any]:
    """Vertex ``Part`` with a text payload."""
    return {"text": text}


def _extract_text_from_content(content_field: Any) -> str:
    """Return the flattened text for a system/user/assistant/tool message.

    Handles plain string, ``None`` (assistant with only tool_calls), and
    list-of-text-parts. Image / audio parts raise ``NotImplementedError``.
    Empty content is preserved as ``""`` so downstream generateContent
    still has something to receive.
    """
    if content_field is None:
        return ""
    if isinstance(content_field, str):
        return content_field
    if isinstance(content_field, list):
        pieces: List[str] = []
        for elem in content_field:
            if isinstance(elem, str):
                pieces.append(elem)
                continue
            if isinstance(elem, dict):
                elem_type = elem.get(K_TYPE_FIELD, "")
                if elem_type == K_TYPE_TEXT:
                    pieces.append(str(elem.get(K_TEXT_FIELD, "")))
                elif elem_type == K_TYPE_REFUSAL:
                    # Vertex has no refusal Part type; emit as plain text.
                    pieces.append(str(elem.get(K_REFUSAL_FIELD, "")))
                elif elem_type in (K_TYPE_IMAGE_URL, K_TYPE_INPUT_AUDIO):
                    raise NotImplementedError(
                        f"content-part type {elem_type!r} not yet ported "
                        "(multimodal-parts follow-up)"
                    )
                else:
                    raise ValueError(f"unknown content-part type {elem_type!r}")
                continue
            raise ValueError(
                f"content-list element must be str or dict, got {type(elem).__name__}"
            )
        return (
            K_CONTENT_SEPARATOR.join(pieces)
            if len(pieces) > 1
            else (pieces[0] if pieces else "")
        )
    raise ValueError(
        f"content must be str, list, or None; got {type(content_field).__name__}"
    )


def _is_empty_part(part: Dict[str, Any]) -> bool:
    """Whether a Vertex Part is empty enough to skip during the message loop.

    A Part is empty when text is empty AND there is no
    thoughtSignature, inlineData, or fileData.
    """
    return (
        not part.get("text")
        and not part.get("thoughtSignature")
        and not part.get(K_INLINE_DATA_FIELD)
        and not part.get(K_FILE_DATA_FIELD)
    )


def _iter_content_parts_from_message(
    content_field: Any,
    *,
    use_per_part_mr: bool = False,
    req_level_mr: Optional[List[Optional[str]]] = None,
) -> List[Dict[str, Any]]:
    """Split a message's content into a list of non-empty Vertex ``Part`` dicts.

    Empty Parts are skipped via ``_is_empty_part``. The caller is
    responsible for the post-loop "pad zero-Part Contents with one
    empty-text Part" step so Vertex never receives a Content with an
    empty ``parts`` array.
    """
    if content_field is None:
        return []
    if isinstance(content_field, str):
        part = _text_part(content_field)
        return [] if _is_empty_part(part) else [part]
    if isinstance(content_field, list):
        parts: List[Dict[str, Any]] = []
        for elem in content_field:
            if isinstance(elem, str):
                part = _text_part(elem)
                if not _is_empty_part(part):
                    parts.append(part)
                continue
            if isinstance(elem, dict):
                elem_type = elem.get(K_TYPE_FIELD, "")
                if elem_type == K_TYPE_TEXT:
                    part = _text_part(str(elem.get(K_TEXT_FIELD, "")))
                elif elem_type == K_TYPE_REFUSAL:
                    part = _text_part(str(elem.get(K_REFUSAL_FIELD, "")))
                elif elem_type == K_TYPE_IMAGE_URL:
                    if req_level_mr is None:
                        req_level_mr = [None]
                    part = _build_image_url_part(
                        elem,
                        use_per_part_mr=use_per_part_mr,
                        req_level_mr=req_level_mr,
                    )
                elif elem_type == K_TYPE_INPUT_AUDIO:
                    part = _build_input_audio_part(elem)
                else:
                    raise ValueError(f"unknown content-part type {elem_type!r}")
                if not _is_empty_part(part):
                    parts.append(part)
                continue
            raise ValueError(
                f"content-list element must be str or dict, got {type(elem).__name__}"
            )
        return parts
    raise ValueError(
        f"content must be str, list, or None; got {type(content_field).__name__}"
    )


# ── Multimodal content-part helpers ──────────────────────────────────


_BASE64_DATA_URI_RE = re.compile(K_IMAGE_DATA_BASE64_PATTERN)


def _parse_base64_data_uri(uri: str) -> Optional[Tuple[str, str]]:
    """Parse a `data:<mime>;base64,<payload>` URI.

    Returns ``(mime, payload)`` on match, ``None`` otherwise. The
    payload is the raw base64 string (not decoded).
    """
    match = _BASE64_DATA_URI_RE.fullmatch(uri)
    if match is None:
        return None
    return match.group(1), match.group(2)


def _build_image_url_part(
    elem: Dict[str, Any],
    *,
    use_per_part_mr: bool,
    req_level_mr: List[Optional[str]],
) -> Dict[str, Any]:
    """Convert an OpenAI ``image_url`` content-part to a Vertex Part.

    Supported URL forms:
      * ``data:<mime>;base64,<payload>`` → ``inlineData`` Part.
      * ``gs://<bucket>/<path>`` → ``fileData`` Part with default
        ``image/png`` MIME type.
    Anything else (http(s), YouTube, string-shorthand) raises ``ValueError``.

    The ``detail`` sub-field translates to ``mediaResolution``:
      * ``"auto"`` (or missing) → no ``mediaResolution`` emitted.
      * On Gemini 3+: per-Part ``mediaResolution.level``.
      * On older Gemini: promoted to request-level via ``req_level_mr[0]``.
        Multiple image Parts must share the same ``detail`` value on
        older models; mismatched values raise ``ValueError``.

    Raises ``ValueError`` on malformed ``image_url`` shape, invalid
    base64 payload, or unsupported URL form.
    """
    image_url_field = elem.get(K_IMAGE_URL_FIELD)
    if not isinstance(image_url_field, dict):
        raise ValueError(
            "image_url must be an object with a 'url' field; got "
            f"{type(image_url_field).__name__}"
        )
    url = image_url_field.get(K_URL_FIELD)
    if not isinstance(url, str) or not url:
        raise ValueError("image_url.url is required and must be a non-empty string")

    detail = image_url_field.get(K_DETAIL_FIELD)
    mr_enum: Optional[str] = None
    if isinstance(detail, str) and detail in K_DETAIL_TO_MEDIA_RESOLUTION:
        mr_enum = K_DETAIL_TO_MEDIA_RESOLUTION[detail]

    def _finalize(part: Dict[str, Any]) -> Dict[str, Any]:
        """Attach per-Part or request-level mediaResolution as configured."""
        if mr_enum is not None:
            if use_per_part_mr:
                part[K_MEDIA_RESOLUTION_FIELD] = {
                    K_MEDIA_RESOLUTION_LEVEL_FIELD: mr_enum
                }
            else:
                # Older Gemini takes a single request-level knob; reject
                # mismatched detail values that can't be reconciled.
                if req_level_mr[0] is None:
                    req_level_mr[0] = mr_enum
                elif req_level_mr[0] != mr_enum:
                    raise ValueError(
                        "found inconsistent 'detail' values across "
                        "image_url content-parts; on this model all "
                        "image parts must share the same detail level "
                        f"(saw {req_level_mr[0]!r} and {mr_enum!r})"
                    )
        return part

    data_uri = _parse_base64_data_uri(url)
    if data_uri is not None:
        mime, payload = data_uri
        if not payload:
            raise ValueError(f"image_url data URI has an empty base64 payload: {url!r}")
        try:
            base64.b64decode(payload, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"image_url data URI does not contain valid base64: {url!r}"
            ) from exc
        return _finalize(
            {
                K_INLINE_DATA_FIELD: {
                    "mimeType": mime,
                    "data": payload,
                }
            }
        )

    if url.startswith(K_GS_URI_PREFIX):
        return _finalize(
            {
                K_FILE_DATA_FIELD: {
                    "mimeType": K_IMAGE_MIME_TYPE,  # "image/png" default
                    K_FILE_URI_FIELD: url,
                }
            }
        )

    raise ValueError(
        "image_url must be a data:<mime>;base64,<payload> URI or a "
        f"gs://<bucket>/<path> URI; got {url!r}"
    )


def _build_input_audio_part(elem: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an OpenAI ``input_audio`` content-part to a Vertex Part.

    Same URL-shape constraint as image_url: only base64 blob data or
    ``gs://`` URIs are accepted; ``http(s)://`` raises ``ValueError``.

    ``format: "mp3"`` -> ``audio/mpeg``; ``format: "wav"`` -> ``audio/wav``;
    any other format string is used verbatim as the MIME type. Missing
    format raises ``ValueError``.
    """
    input_audio_field = elem.get(K_INPUT_AUDIO_FIELD)
    if not isinstance(input_audio_field, dict):
        raise ValueError(
            "input_audio must be an object with 'data' and 'format' "
            f"fields; got {type(input_audio_field).__name__}"
        )
    data = input_audio_field.get(K_DATA_FIELD)
    if not isinstance(data, str) or not data:
        raise ValueError("input_audio.data is required and must be a non-empty string")
    fmt = input_audio_field.get(K_FORMAT_FIELD)
    if not isinstance(fmt, str) or not fmt:
        raise ValueError(
            "input_audio.format is required and must be a non-empty string"
        )
    mime = K_INPUT_AUDIO_FORMAT_TO_MIME.get(fmt, fmt)

    # gs:// -> fileData; http(s):// rejected; else base64 blob -> inlineData.
    if data.startswith(K_GS_URI_PREFIX):
        return {
            K_FILE_DATA_FIELD: {
                "mimeType": mime,
                K_FILE_URI_FIELD: data,
            }
        }
    if data.startswith(K_HTTP_URI_PREFIX) or data.startswith(K_HTTPS_URI_PREFIX):
        raise ValueError(
            "input_audio.data must be a base64-encoded payload or a "
            f"gs://<bucket>/<path> URI; got {data!r}"
        )
    try:
        base64.b64decode(data, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "input_audio.data is not valid base64: "
            f"{data[:40]!r}{'...' if len(data) > 40 else ''}"
        ) from exc
    return {
        K_INLINE_DATA_FIELD: {
            "mimeType": mime,
            "data": data,
        }
    }


def _build_assistant_audio_part(audio_field: Any) -> Dict[str, Any]:
    """Convert an OpenAI assistant-message ``audio`` field to a Vertex Part.

    Vertex has no "identifier for prior response" concept, so ``audio.id``
    is treated as a base64-encoded ``audio/mpeg`` blob and emitted as an
    ``inlineData`` Part. Invalid base64 raises ``ValueError``.
    """
    if not isinstance(audio_field, dict):
        raise ValueError(
            "assistant.audio must be an object with an 'id' field; "
            f"got {type(audio_field).__name__}"
        )
    audio_id = audio_field.get(K_ID_FIELD)
    if not isinstance(audio_id, str) or not audio_id:
        raise ValueError(
            "assistant.audio.id is required and must be a non-empty string"
        )
    try:
        base64.b64decode(audio_id, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "assistant.audio.id is not valid base64: "
            f"{audio_id[:40]!r}{'...' if len(audio_id) > 40 else ''}"
        ) from exc
    return {
        K_INLINE_DATA_FIELD: {
            "mimeType": K_AUDIO_MPEG_MIME_TYPE,
            "data": audio_id,
        }
    }


# ── Function-calling helpers ─────────────────────────────────────────


def _convert_json_string_to_struct(
    json_string: str, wrap_invalid: bool = False
) -> Dict[str, Any]:
    """Parse a JSON-encoded string into a dict.

    Empty input yields ``{}``.

    Args:
      json_string: JSON-encoded string. Empty string -> ``{}``.
      wrap_invalid: When True and the string is not a valid JSON
        object, wrap as ``{"result": <original-string>}``.

    Raises:
      ValueError: when ``wrap_invalid=False`` and the input is not a
        valid JSON object.
    """
    if not json_string:
        return {}
    try:
        parsed = json.loads(json_string)
    except json.JSONDecodeError:
        if wrap_invalid:
            return {"result": json_string}
        raise ValueError(
            f"Expected a valid JSON object; got unparseable string {json_string!r}"
        )
    if isinstance(parsed, dict):
        return parsed
    if wrap_invalid:
        return {"result": json_string}
    raise ValueError(f"Expected a JSON object; parsed {type(parsed).__name__} instead")


# snake_case -> camelCase renames for JSON Schema fields (Vertex expects
# camelCase). Applied recursively via _standardize_openapi_field_names.
_JSON_SCHEMA_FIELD_RENAMES: Dict[str, str] = {
    "ref": "$ref",
    "min_items": "minItems",
    "max_items": "maxItems",
    "defs": "$defs",
    "property_ordering": "propertyOrdering",
    "min_properties": "minProperties",
    "max_properties": "maxProperties",
    "min_length": "minLength",
    "max_length": "maxLength",
    "any_of": "anyOf",
    "all_of": "allOf",
    "one_of": "oneOf",
    "additional_properties": "additionalProperties",
}

# Fields holding a sub-schema that must be recursed into.
_JSON_SCHEMA_RECURSE_FIELDS = frozenset(
    {
        "items",
        "default",
        "additionalProperties",
        "pattern",
        "example",
    }
)

# Fields that are dict-of-name-to-schema (each value recursed).
_JSON_SCHEMA_RECURSE_DICT_OF_SCHEMAS_FIELDS = frozenset({"properties", "$defs"})

# Fields that are list-of-schemas (each element recursed).
_JSON_SCHEMA_RECURSE_LIST_OF_SCHEMAS_FIELDS = frozenset({"anyOf", "allOf", "oneOf"})


def _standardize_openapi_field_names(json_schema: Any) -> Any:
    """Recursively rename schema fields to camelCase + normalize nullable.

    Mutates the input dict in-place AND returns it, so callers can chain.
    Non-dict / non-list inputs are returned unchanged.
    """
    if isinstance(json_schema, list):
        for elem in json_schema:
            _standardize_openapi_field_names(elem)
        return json_schema
    if not isinstance(json_schema, dict):
        return json_schema

    # Renames first so subsequent recursion uses the new names.
    for old, new in _JSON_SCHEMA_FIELD_RENAMES.items():
        if old in json_schema:
            json_schema[new] = json_schema.pop(old)

    # nullable: True -> merge "null" into type; nullable: False is dropped.
    if "nullable" in json_schema:
        nullable_value = json_schema["nullable"]
        if isinstance(nullable_value, bool):
            if nullable_value:
                if "type" in json_schema:
                    type_field = json_schema["type"]
                    if isinstance(type_field, str):
                        json_schema["type"] = [type_field, "null"]
                    elif isinstance(type_field, list):
                        type_field.append("null")
                else:
                    json_schema["type"] = "null"
            json_schema.pop("nullable")

    for field_name in _JSON_SCHEMA_RECURSE_FIELDS:
        if field_name in json_schema:
            _standardize_openapi_field_names(json_schema[field_name])
    for field_name in _JSON_SCHEMA_RECURSE_DICT_OF_SCHEMAS_FIELDS:
        if field_name in json_schema and isinstance(json_schema[field_name], dict):
            for sub_schema in json_schema[field_name].values():
                _standardize_openapi_field_names(sub_schema)
    for field_name in _JSON_SCHEMA_RECURSE_LIST_OF_SCHEMAS_FIELDS:
        if field_name in json_schema:
            _standardize_openapi_field_names(json_schema[field_name])

    return json_schema


def _convert_request_function_call(
    function_call_object: Dict[str, Any],
) -> Dict[str, Any]:
    """Convert an OpenAI function-call spec into a Vertex functionCall Part payload.

    The OpenAI ``arguments`` string is parsed into a dict for Vertex's
    ``args`` field. Missing / empty arguments -> ``{}``.
    """
    return {
        "name": str(function_call_object.get(K_NAME_FIELD, "")),
        "args": _convert_json_string_to_struct(
            str(function_call_object.get(K_ARGUMENTS_FIELD, ""))
        ),
    }


def _convert_request_function_declaration(
    function_declaration_object: Dict[str, Any],
) -> Dict[str, Any]:
    """Convert an OpenAI function-declaration into a Vertex FunctionDeclaration dict.

    Parameters (OpenAPI-style JSON schema) get field-name standardization
    applied before being emitted as ``parametersJsonSchema``.
    """
    ret: Dict[str, Any] = {
        "name": str(function_declaration_object.get(K_NAME_FIELD, "")),
    }
    description = function_declaration_object.get(K_DESCRIPTION_FIELD)
    if isinstance(description, str) and description:
        ret["description"] = description
    parameters = function_declaration_object.get(K_PARAMETERS_FIELD)
    if isinstance(parameters, dict):
        # Deep copy so we don't mutate the caller's input dict.
        params_copy = copy.deepcopy(parameters)
        _standardize_openapi_field_names(params_copy)
        ret["parametersJsonSchema"] = params_copy
    return ret


def _convert_tool_choice(tool_choice_field: Any) -> Optional[Dict[str, Any]]:
    """Convert an OpenAI tool_choice value into a Vertex functionCallingConfig dict.

    Returns None if the input is neither a string nor a dict. Unknown
    string values return ``{}`` so Vertex uses its own default.
    """
    if isinstance(tool_choice_field, str):
        mode_map = {
            K_TOOL_CHOICE_NONE: "NONE",
            K_TOOL_CHOICE_AUTO: "AUTO",
            K_TOOL_CHOICE_REQUIRED: "ANY",
            K_TOOL_CHOICE_VALIDATED: "VALIDATED",
        }
        if tool_choice_field in mode_map:
            return {"mode": mode_map[tool_choice_field]}
        return {}
    if isinstance(tool_choice_field, dict):
        function_field = tool_choice_field.get(K_FUNCTION_FIELD, {})
        name = ""
        if isinstance(function_field, dict):
            name = str(function_field.get(K_NAME_FIELD, ""))
        return {
            "mode": "ANY",
            "allowedFunctionNames": [name],
        }
    return None


def _convert_legacy_function_call(function_call_field: Any) -> Optional[Dict[str, Any]]:
    """Convert a legacy OpenAI function_call value into a functionCallingConfig dict.

    Same shape as ``_convert_tool_choice`` but only ``auto`` / ``none``
    string values are valid.
    """
    if isinstance(function_call_field, str):
        mode_map = {
            K_TOOL_CHOICE_NONE: "NONE",
            K_TOOL_CHOICE_AUTO: "AUTO",
        }
        if function_call_field in mode_map:
            return {"mode": mode_map[function_call_field]}
        return {}
    if isinstance(function_call_field, dict):
        # Legacy function_call object: {"name": "..."} (no "function" wrapper).
        name = str(function_call_field.get(K_NAME_FIELD, ""))
        return {
            "mode": "ANY",
            "allowedFunctionNames": [name],
        }
    return None


# ── thought_signature / thought extraction ───────────────────────────


def _extract_extra_content_google(
    container: Any,
) -> Optional[Dict[str, Any]]:
    """Extract ``extra_content.google.*`` fields from a message or tool_call.

    Returns None when the ``extra_content.google`` namespace is missing
    or not a dict. On success, returns a dict with the lifted fields:

      * ``thought``: bool (only present when originally set).
      * ``thought_signature``: str — the base64-encoded string, passed
        through VERBATIM (``Part.thought_signature`` is a proto ``bytes``
        field and proto3 JSON encodes bytes as base64 strings, so the
        incoming b64 IS the wire form). Validated as real base64 to
        fail fast on malformed values. ``K_DUMMY_THOUGHT_SIGNATURE`` is
        exempt from validation; empty strings are dropped.

    Raises ``ValueError`` on malformed input.
    """
    if not isinstance(container, dict):
        return None
    extra_content = container.get(K_CHAT_COMPLETIONS_EXTRA_CONTENT_FIELD)
    if not isinstance(extra_content, dict):
        return None
    google = extra_content.get(K_OPEN_API_GOOGLE_EXTRA_FIELD)
    if not isinstance(google, dict):
        return None

    result: Dict[str, Any] = {}

    if K_GOOGLE_THOUGHT_FIELD in google:
        thought_value = google[K_GOOGLE_THOUGHT_FIELD]
        if not isinstance(thought_value, bool):
            raise ValueError(
                f"extra_content.google.{K_GOOGLE_THOUGHT_FIELD} must be a "
                f"bool, got {type(thought_value).__name__}"
            )
        result[K_GOOGLE_THOUGHT_FIELD] = thought_value

    if K_GOOGLE_THOUGHT_SIGNATURE_FIELD in google:
        sig_value = google[K_GOOGLE_THOUGHT_SIGNATURE_FIELD]
        if not isinstance(sig_value, str):
            raise ValueError(
                f"extra_content.google.{K_GOOGLE_THOUGHT_SIGNATURE_FIELD} "
                f"must be a string, got {type(sig_value).__name__}"
            )
        if sig_value == K_DUMMY_THOUGHT_SIGNATURE:
            # Bypass sentinel: verbatim, no base64 decode.
            result[K_GOOGLE_THOUGHT_SIGNATURE_FIELD] = sig_value
        elif sig_value == "":
            # Empty signature is a no-op downstream.
            pass
        else:
            # Validate the input is real base64 so malformed values fail
            # fast at conversion time (not deep in from_json), but write
            # the b64 STRING back onto the result -- proto3 JSON encodes
            # ``Part.thought_signature`` (a bytes field) as a base64
            # string on the wire, so the b64 IS the correct wire form.
            # Decoding to text and re-encoding as latin-1 corrupted
            # non-UTF-8 signatures and made GenerateContentRequest.from_json
            # reject valid ASCII test values as invalid base64.
            try:
                base64.b64decode(sig_value, validate=True)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"extra_content.google.{K_GOOGLE_THOUGHT_SIGNATURE_FIELD} "
                    f"is not valid base64: {sig_value!r}"
                ) from exc
            result[K_GOOGLE_THOUGHT_SIGNATURE_FIELD] = sig_value

    return result


def _extract_provider_specific_thought_signatures(
    message: Any,
) -> Optional[List[str]]:
    """Return the ``provider_specific_fields.thought_signatures`` list, if any.

    LiteLLM-parity envelope emitted by ``convert_gemini_response_to_openai``:
    a top-level list of base64-encoded signature strings, one per response
    ``functionCall`` Part with a signature, in the same order as
    ``message.tool_calls`` (see ``_build_provider_specific_fields`` for
    the emit side). Reading it on the request side lets multi-turn agents
    pipe a response straight into the next-turn request without
    translating through ``extra_content.google.thought_signature``.

    Precedence: when both this envelope AND a per-tool_call
    ``extra_content.google.thought_signature`` are present on the same
    assistant message, this envelope WINS (LiteLLM's canonical form). The
    per-tool_call ``extra_content`` path is retained for older callers.

    Returns None when the message is not a dict, has no
    ``provider_specific_fields``, or the envelope is missing / not a
    non-empty list. Base64 validation of individual entries is NOT done
    here — invalid entries fail downstream in
    ``GenerateContentRequest.from_json`` (mirrors LiteLLM's own policy).

    Raises ``ValueError`` if any entry in the list is not a string —
    the request-side loop uses ``isinstance(sig, str)`` as its guard,
    so a mid-list ``None`` / non-string entry would silently drop the
    signature for that tool_call while adjacent tool_calls kept theirs
    (a wrong-signature-attachment class of bug that Vertex would then
    reject with an opaque "signature mismatch" on the corresponding
    functionCall Part). Fail loud at the read boundary instead.
    """
    if not isinstance(message, dict):
        return None
    psf = message.get(K_CHAT_COMPLETIONS_PROVIDER_SPECIFIC_FIELDS_FIELD)
    if not isinstance(psf, dict):
        return None
    sigs = psf.get(K_LITELLM_THOUGHT_SIGNATURES_FIELD)
    if not isinstance(sigs, list) or not sigs:
        return None
    for i, entry in enumerate(sigs):
        if not isinstance(entry, str):
            raise ValueError(
                f"provider_specific_fields.thought_signatures[{i}] must be "
                f"a string (base64-encoded signature), got "
                f"{type(entry).__name__}: {entry!r}. Mid-list non-string "
                f"entries silently misalign the signature-to-tool_call "
                f"mapping downstream; fix the caller."
            )
    return sigs


def _apply_google_extra_to_part(
    part: Dict[str, Any], google_extra: Optional[Dict[str, Any]]
) -> None:
    """Attach ``thought`` / ``thoughtSignature`` to a Part in-place.

    Only ``thought: true`` is emitted; ``thought: false``, empty
    signatures, and the ``K_DUMMY_THOUGHT_SIGNATURE`` bypass sentinel
    are skipped (Vertex ignores empties, and the sentinel is not valid
    base64 so writing it would be rejected by
    ``GenerateContentRequest.from_json``).
    """
    if not google_extra:
        return
    thought = google_extra.get(K_GOOGLE_THOUGHT_FIELD)
    if thought is True:
        part["thought"] = True
    sig = google_extra.get(K_GOOGLE_THOUGHT_SIGNATURE_FIELD)
    if isinstance(sig, str) and sig != "" and sig != K_DUMMY_THOUGHT_SIGNATURE:
        part["thoughtSignature"] = sig


# ── convert_openai_request_to_gemini ─────────────────────────────────


def convert_openai_request_to_gemini(
    context: Optional[ConversionContext],
    json_input: Dict[str, Any],
    extra_body: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Translate an OpenAI Chat Completions request dict to a Vertex generateContent dict.

    Args:
      context: Optional conversion knobs. ``None`` -> defaults.
      json_input: OpenAI Chat Completions request. Must contain at
        minimum ``model`` (str) and ``messages`` (list).
      extra_body: Optional Vertex-side extra body dict. Setting it
        raises ``NotImplementedError`` (not yet handled).

    Returns:
      A dict shaped like a Vertex ``GenerateContentRequest`` in the
      REST wire format (camelCase field names).

    Raises:
      ValueError: input is malformed.
      NotImplementedError: input uses a feature not yet supported
        (``response_format``, ``reasoning_effort``, ``stream_options``,
        ``extra_body``).
    """
    if context is None:
        context = ConversionContext()

    # `req_level_mr` is a 1-element mutable holder used as an out-param
    # from `_build_image_url_part` for the request-level media-resolution
    # promotion path (used on older Gemini; Gemini 3+ emits per-Part).
    model_name_for_kind = context.large_model_name or json_input.get(K_MODEL_FIELD, "")
    use_per_part_mr = use_per_part_media_resolution(model_name_for_kind)
    req_level_mr: List[Optional[str]] = [None]

    if extra_body is not None:
        raise NotImplementedError("extra_body handling not yet ported")

    for unsupported_field in (
        K_RESPONSE_FORMAT_FIELD,
        K_REASONING_EFFORT_FIELD,
        K_STREAM_OPTIONS_FIELD,
    ):
        if unsupported_field in json_input:
            raise NotImplementedError(
                f"top-level {unsupported_field!r} field not yet ported "
                "(scaffolding scope; follow-up will port)"
            )

    ret: Dict[str, Any] = {}

    # Model: publisher_model_name (from context) wins if set.
    if context.publisher_model_name is not None:
        ret[K_MODEL_FIELD] = context.publisher_model_name
    else:
        ret[K_MODEL_FIELD] = str(json_input.get(K_MODEL_FIELD, ""))

    # Merge consecutive same-role messages into one Content; split
    # system/developer to systemInstruction; rewrite assistant -> model
    # when context.use_model_role. tool_call_id_to_name lets later
    # tool/function-role messages populate functionResponse.name.
    contents: List[Dict[str, Any]] = []
    system_instruction: Optional[Dict[str, Any]] = None
    prev_role: str = ""
    tool_call_id_to_name: Dict[str, str] = {}

    for message in json_input.get(K_MESSAGES_FIELD, []):
        raw_role = str(message.get(K_ROLE_FIELD, ""))
        curr_role = raw_role
        if curr_role == K_ROLE_ASSISTANT and context.use_model_role:
            curr_role = K_ROLE_MODEL

        is_tool_role = curr_role in (K_ROLE_TOOL, K_ROLE_FUNCTION)
        is_system_role = curr_role in (K_ROLE_SYSTEM, K_ROLE_DEVELOPER)

        content_field = message.get(K_CONTENT_FIELD)  # None if missing

        # Message-level extra_content.google applies to text/refusal/
        # functionResponse Parts and to the synthetic thought-only Part
        # on assistant messages. Per-tool_call extra overrides for the
        # corresponding functionCall Part only.
        message_google_extra = _extract_extra_content_google(message)

        if is_system_role:
            # Preserve prev_role across system boundaries so
            # user/system/user still merges into one user Content.
            new_parts = _iter_content_parts_from_message(
                content_field,
                use_per_part_mr=use_per_part_mr,
                req_level_mr=req_level_mr,
            )
            if system_instruction is None:
                system_instruction = {
                    K_ROLE_FIELD: K_ROLE_SYSTEM,
                    "parts": list(new_parts),
                }
            else:
                system_instruction["parts"].extend(new_parts)
            continue

        if is_tool_role:
            # Unknown tool_call_id (no matching prior tool_call) -> "".
            tool_call_id = str(message.get(K_TOOL_CALL_ID_FIELD, ""))
            fn_name = tool_call_id_to_name.get(tool_call_id, "")
            # Split tool content into TEXT (-> functionResponse.response Struct) and
            # MEDIA Parts (image/audio) -- e.g. a tool that returns base64
            # screenshots as image_url parts.
            #
            # Media placement follows the official Gemini-3 multimodal function
            # response: media go in functionResponse.parts as nested
            # FunctionResponsePart inline/file data, with text left in
            # functionResponse.response. Text-only tool results (media_parts == [])
            # behave exactly as before.
            tool_parts = _iter_content_parts_from_message(
                content_field,
                use_per_part_mr=use_per_part_mr,
                req_level_mr=req_level_mr,
            )
            text_pieces = [p[K_TEXT_FIELD] for p in tool_parts if K_TEXT_FIELD in p]
            media_parts = [p for p in tool_parts if K_TEXT_FIELD not in p]
            # wrap_invalid=True: plain-text content -> {"result": "..."}.
            content_str = (
                K_CONTENT_SEPARATOR.join(text_pieces)
                if len(text_pieces) > 1
                else (text_pieces[0] if text_pieces else "")
            )
            response_dict = _convert_json_string_to_struct(
                content_str, wrap_invalid=True
            )
            fn_response: Dict[str, Any] = {
                "name": fn_name,
                "response": response_dict,
                "id": tool_call_id,
            }
            if media_parts:
                # Each FunctionResponsePart carries only its inline/file image data.
                fr_parts: List[Dict[str, Any]] = []
                for p in media_parts:
                    if K_INLINE_DATA_FIELD in p:
                        fr_parts.append({K_INLINE_DATA_FIELD: p[K_INLINE_DATA_FIELD]})
                    elif K_FILE_DATA_FIELD in p:
                        fr_parts.append({K_FILE_DATA_FIELD: p[K_FILE_DATA_FIELD]})
                if fr_parts:
                    fn_response["parts"] = fr_parts
            fn_response_part = {"functionResponse": fn_response}
            # Message-level extras are the only source for tool-role.
            _apply_google_extra_to_part(fn_response_part, message_google_extra)
            # tool-role Contents have no role field on the Vertex side.
            # Adjacent tool-role messages collapse into one Content.
            standardized_prev_role = K_ROLE_TOOL
            if contents and prev_role == standardized_prev_role:
                contents[-1]["parts"].append(fn_response_part)
            else:
                contents.append({"parts": [fn_response_part]})
                prev_role = standardized_prev_role
            continue

        # user or assistant/model role.
        new_parts = _iter_content_parts_from_message(
            content_field,
            use_per_part_mr=use_per_part_mr,
            req_level_mr=req_level_mr,
        )
        for part in new_parts:
            _apply_google_extra_to_part(part, message_google_extra)

        # LiteLLM-parity envelope: top-level list of b64 signatures
        # aligned one-to-one with ``tool_calls`` (see
        # ``_extract_provider_specific_thought_signatures``). Read here
        # so multi-turn agents can pipe responses straight into the
        # next-turn request; per-tool_call values override the legacy
        # ``extra_content.google.thought_signature`` path.
        provider_specific_thought_sigs = _extract_provider_specific_thought_signatures(
            message
        )

        # Assistant tool_calls: one functionCall Part per entry, order
        # preserved. Record id -> name mapping for later tool-role lookup.
        tool_calls_field = message.get(K_TOOL_CALLS_FIELD)
        if isinstance(tool_calls_field, list):
            # Filter to dict-shaped entries so ``tc_index`` matches the
            # index used against ``provider_specific_thought_sigs``.
            # Iterating the unfiltered list would let a mid-list garbage
            # entry ``[dict, "garbage", dict]`` slip past a naive length
            # check and misalign the signature index.
            dict_shaped_tool_calls = [
                tc for tc in tool_calls_field if isinstance(tc, dict)
            ]
            if provider_specific_thought_sigs is not None and len(
                provider_specific_thought_sigs
            ) != len(dict_shaped_tool_calls):
                raise ValueError(
                    f"provider_specific_fields.thought_signatures length "
                    f"({len(provider_specific_thought_sigs)}) does not match "
                    f"tool_calls length ({len(dict_shaped_tool_calls)}) on "
                    f"assistant message. Must be one-to-one in tool_calls "
                    f"order (mutated history or older converter output?)."
                )

            for tc_index, tool_call in enumerate(dict_shaped_tool_calls):
                function_object = tool_call.get(K_FUNCTION_FIELD, {})
                if not isinstance(function_object, dict):
                    function_object = {}
                fn_call = _convert_request_function_call(function_object)
                # Record the id -> name map so tool-role messages can
                # populate functionResponse.name.
                tc_id = tool_call.get(K_ID_FIELD)
                if isinstance(tc_id, str):
                    fn_call["id"] = tc_id
                    tool_call_id_to_name[tc_id] = fn_call["name"]
                fn_call_part: Dict[str, Any] = {"functionCall": fn_call}
                # Per-tool_call extras apply to this functionCall Part
                # only; message-level extras don't propagate here.
                tool_call_google_extra = _extract_extra_content_google(tool_call)
                _apply_google_extra_to_part(fn_call_part, tool_call_google_extra)
                # LiteLLM envelope overrides per-tool_call extra_content
                # (skip empty / dummy sentinel entries).
                if provider_specific_thought_sigs is not None and tc_index < len(
                    provider_specific_thought_sigs
                ):
                    sig_from_psf = provider_specific_thought_sigs[tc_index]
                    if (
                        isinstance(sig_from_psf, str)
                        and sig_from_psf != ""
                        and sig_from_psf != K_DUMMY_THOUGHT_SIGNATURE
                    ):
                        fn_call_part["thoughtSignature"] = sig_from_psf
                new_parts.append(fn_call_part)

        # Legacy function_call is mutually exclusive with tool_calls.
        elif isinstance(message.get(K_FUNCTION_CALL_FIELD), dict):
            fn_call = _convert_request_function_call(message[K_FUNCTION_CALL_FIELD])
            new_parts.append({"functionCall": fn_call})

        # Assistant-only audio field; user-role audio arrives via
        # content-parts (input_audio) handled upstream.
        if raw_role == K_ROLE_ASSISTANT:
            audio_field = message.get(K_AUDIO_FIELD)
            if audio_field is not None:
                new_parts.append(_build_assistant_audio_part(audio_field))

        # thought_signature is only defined for text/refusal Parts.
        # Raise if a message with message-level thought_signature also
        # contains an image/audio Part rather than silently misapplying.
        if message_google_extra and message_google_extra.get(
            K_GOOGLE_THOUGHT_SIGNATURE_FIELD
        ):
            for part in new_parts:
                if K_INLINE_DATA_FIELD in part or K_FILE_DATA_FIELD in part:
                    raise ValueError(
                        "thought_signature is only valid on "
                        "text/refusal Parts; found on a message "
                        "containing an image/audio Part"
                    )

        # Thought-only assistant message (no content, no tool_calls,
        # no function_call, but message-level extra_content.google set):
        # emit a synthetic Part. `_is_empty_part` drops it if only
        # `thought: true` was set (no signature).
        if not new_parts and message_google_extra:
            synthetic_part: Dict[str, Any] = {}
            _apply_google_extra_to_part(synthetic_part, message_google_extra)
            if not _is_empty_part(synthetic_part):
                new_parts.append(synthetic_part)

        if contents and prev_role == curr_role:
            contents[-1]["parts"].extend(new_parts)
        else:
            contents.append({K_ROLE_FIELD: curr_role, "parts": list(new_parts)})
            prev_role = curr_role

    # Vertex rejects Contents with an empty ``parts`` array — pad any
    # empty Content (including systemInstruction) with one empty-text Part.
    if system_instruction is not None and not system_instruction["parts"]:
        system_instruction["parts"].append(_text_part(""))
    if not contents:
        contents.append({K_ROLE_FIELD: K_ROLE_USER, "parts": [_text_part("")]})
    else:
        for content in contents:
            if not content["parts"]:
                content["parts"].append(_text_part(""))

    ret["contents"] = contents
    if system_instruction is not None:
        ret["systemInstruction"] = system_instruction

    # generationConfig: only fill fields the caller set; omit if empty.
    gen_cfg: Dict[str, Any] = {}
    if K_TEMPERATURE_FIELD in json_input:
        gen_cfg["temperature"] = float(json_input[K_TEMPERATURE_FIELD])
    if K_TOP_P_FIELD in json_input:
        gen_cfg["topP"] = float(json_input[K_TOP_P_FIELD])
    if K_N_FIELD in json_input:
        gen_cfg["candidateCount"] = int(json_input[K_N_FIELD])
    # Both map to maxOutputTokens; when both are present max_tokens wins
    # (opposite of the OpenAI spec's preference for max_completion_tokens,
    # but preserved for wire-format compatibility with existing callers).
    if K_MAX_TOKENS_FIELD in json_input:
        gen_cfg["maxOutputTokens"] = int(json_input[K_MAX_TOKENS_FIELD])
    elif K_MAX_COMPLETION_TOKENS_FIELD in json_input:
        gen_cfg["maxOutputTokens"] = int(json_input[K_MAX_COMPLETION_TOKENS_FIELD])
    if K_FREQUENCY_PENALTY_FIELD in json_input:
        gen_cfg["frequencyPenalty"] = float(json_input[K_FREQUENCY_PENALTY_FIELD])
    if K_PRESENCE_PENALTY_FIELD in json_input:
        gen_cfg["presencePenalty"] = float(json_input[K_PRESENCE_PENALTY_FIELD])
    if K_SEED_FIELD in json_input:
        gen_cfg["seed"] = int(json_input[K_SEED_FIELD])
    if K_LOG_PROBS_FIELD in json_input:
        gen_cfg["responseLogprobs"] = bool(json_input[K_LOG_PROBS_FIELD])
    if K_TOP_LOG_PROBS_FIELD in json_input:
        gen_cfg["logprobs"] = int(json_input[K_TOP_LOG_PROBS_FIELD])
    if K_STOP_FIELD in json_input:
        stop = json_input[K_STOP_FIELD]
        if isinstance(stop, str):
            gen_cfg["stopSequences"] = [stop]
        elif isinstance(stop, list):
            gen_cfg["stopSequences"] = [str(s) for s in stop]
        else:
            raise ValueError(
                f"stop must be str or list[str]; got {type(stop).__name__}"
            )
    if K_MODALITIES_FIELD in json_input:
        modalities_raw = json_input[K_MODALITIES_FIELD]
        if not isinstance(modalities_raw, list):
            raise ValueError(
                f"modalities must be list; got {type(modalities_raw).__name__}"
            )
        unknown = [m for m in modalities_raw if m not in K_MODALITIES]
        if unknown:
            raise ValueError(
                f"unknown modality values {unknown!r}; expected subset of "
                f"{sorted(K_MODALITIES)}"
            )
        # Vertex expects uppercase modality-enum names.
        gen_cfg["responseModalities"] = [m.upper() for m in modalities_raw]

    # Older-Gemini per-request mediaResolution (set by image_url `detail`
    # in the message loop). Gemini 3+ emits per-Part instead.
    if req_level_mr[0] is not None:
        gen_cfg[K_MEDIA_RESOLUTION_FIELD] = req_level_mr[0]

    if gen_cfg:
        ret["generationConfig"] = gen_cfg

    # Metadata -> Vertex labels (string:string).
    if K_METADATA_FIELD in json_input:
        meta = json_input[K_METADATA_FIELD]
        if not isinstance(meta, dict):
            raise ValueError(f"metadata must be dict; got {type(meta).__name__}")
        ret["labels"] = {str(k): str(v) for k, v in meta.items()}

    # ── Tools + tool_choice / legacy functions + function_call ───────
    # tools/tool_choice takes precedence over legacy functions/function_call
    # when both are set. Both arrays collapse into a single ``tools``
    # entry containing all functionDeclarations.
    tools_field = json_input.get(K_TOOLS_FIELD)
    functions_field = json_input.get(K_FUNCTIONS_FIELD)
    tool_declarations: List[Dict[str, Any]] = []
    if isinstance(tools_field, list):
        for tool_elem in tools_field:
            if not isinstance(tool_elem, dict):
                continue
            function_object = tool_elem.get(K_FUNCTION_FIELD, {})
            if not isinstance(function_object, dict):
                function_object = {}
            tool_declarations.append(
                _convert_request_function_declaration(function_object)
            )
    elif isinstance(functions_field, list):
        for function_elem in functions_field:
            if not isinstance(function_elem, dict):
                continue
            tool_declarations.append(
                _convert_request_function_declaration(function_elem)
            )
    if tool_declarations:
        ret["tools"] = [{"functionDeclarations": tool_declarations}]

    # tool_choice / legacy function_call -> toolConfig.functionCallingConfig.
    tool_choice_field = json_input.get(K_TOOL_CHOICE_FIELD)
    function_call_field = json_input.get(K_FUNCTION_CALL_FIELD)
    function_calling_config: Optional[Dict[str, Any]] = None
    if tool_choice_field is not None and (isinstance(tool_choice_field, (str, dict))):
        function_calling_config = _convert_tool_choice(tool_choice_field)
    elif function_call_field is not None and (
        isinstance(function_call_field, (str, dict))
    ):
        function_calling_config = _convert_legacy_function_call(function_call_field)
    if function_calling_config is not None:
        ret["toolConfig"] = {"functionCallingConfig": function_calling_config}

    return ret


# ── Response-side dataclasses ────────────────────────────────────────
# Model the OpenAI Chat Completions response as emitted by LiteLLM's
# proxy (a superset of OpenAI's own schema, adding thinking_blocks,
# provider_specific_fields, and reasoning_tokens details).


@dataclass(kw_only=True, slots=True)
class ThinkingBlock:
    """One reasoning/thought part.

    Mirrors Anthropic's extended-thinking ``{type, thinking, signature}``
    shape, adopted by LiteLLM as its cross-provider reasoning
    representation.

    A REDACTED block is the same slot with the payload encrypted: it
    carries ``data`` instead of ``thinking``, and no plaintext at all.
    Both forms must be echoed back VERBATIM on the next request during a
    tool loop -- the model verifies its own prior reasoning and the API
    rejects a follow-up that reconstructs or omits them. Modelling only
    the plaintext form is therefore not a simplification but a silent
    truncation of what has to round-trip, so both fields are optional and
    exactly one is set.
    """

    type: Literal["thinking", "redacted_thinking"] = "thinking"
    thinking: Optional[str] = None
    signature: Optional[str] = None
    data: Optional[str] = None


@dataclass(kw_only=True, slots=True)
class FunctionCall:
    """The function payload of a tool call."""

    name: str
    arguments: str  # JSON-encoded object string, per OpenAI spec


@dataclass(kw_only=True, slots=True)
class ToolCall:
    """OpenAI tool call.

    The ``id`` may embed a base64-encoded thought signature via the
    ``__thought__`` separator when the upstream Vertex functionCall
    carried a ``thoughtSignature`` (LiteLLM's wire convention; see
    ``_LITELLM_TOOL_ID_SEPARATOR``).
    """

    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


@dataclass(kw_only=True, slots=True)
class ProviderSpecificFields:
    """LiteLLM escape hatch for non-OpenAI-spec data.

    Only ``thought_signatures`` is populated: the flat catch-all list of
    every thought signature in the response, in parts-order.
    """

    thought_signatures: Optional[List[str]] = None


@dataclass(kw_only=True, slots=True)
class Message:
    """The assistant message on a ``Choice``.

    ``reasoning_content`` / ``audio`` / ``function_call`` exist on
    LiteLLM's Message but are not emitted here (reasoning lives in
    ``thinking_blocks``; audio is out of scope; ``function_call`` is
    the legacy pre-tool_calls API).
    """

    role: Literal["assistant"] = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    thinking_blocks: Optional[List[ThinkingBlock]] = None
    provider_specific_fields: Optional[ProviderSpecificFields] = None


# OpenAI's documented ChatCompletion finish_reason values.
FinishReasonStr = Literal[
    "stop", "length", "tool_calls", "content_filter", "function_call"
]


@dataclass(kw_only=True, slots=True)
class Choice:
    index: int
    message: Message
    finish_reason: FinishReasonStr


@dataclass(kw_only=True, slots=True)
class CompletionTokensDetails:
    reasoning_tokens: Optional[int] = None


@dataclass(kw_only=True, slots=True)
class PromptTokensDetails:
    cached_tokens: Optional[int] = None


@dataclass(kw_only=True, slots=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    completion_tokens_details: Optional[CompletionTokensDetails] = None
    prompt_tokens_details: Optional[PromptTokensDetails] = None


def _require_dict(value: Any, field_name: str) -> Dict[str, Any]:
    """Return ``value`` as a dict, or raise if it is present but not one."""
    if not isinstance(value, dict):
        raise ValueError(
            f"malformed chat-completion payload: {field_name!r} must be an "
            f"object, got {type(value).__name__}"
        )
    return value


def _require_list(value: Any, field_name: str) -> List[Any]:
    """Return ``value`` as a list, or raise if it is present but not one."""
    if not isinstance(value, list):
        raise ValueError(
            f"malformed chat-completion payload: {field_name!r} must be an "
            f"array, got {type(value).__name__}"
        )
    return value


def _thinking_block_from_dict(payload: Dict[str, Any]) -> ThinkingBlock:
    """Build a ThinkingBlock from a ``{thinking, signature}`` dict.

    A missing ``thinking`` becomes ``""``; a missing ``signature``
    becomes ``None``.
    """
    return ThinkingBlock(
        thinking=payload.get("thinking", ""),
        signature=payload.get("signature"),
    )


def _tool_call_from_dict(payload: Dict[str, Any]) -> ToolCall:
    """Build a ToolCall from an ``{id, function: {name, arguments}}`` dict.

    A missing ``id`` / ``function`` / ``name`` / ``arguments`` becomes
    the empty string (or an absent nested function payload).
    """
    function_payload = payload.get(K_FUNCTION_FIELD) or {}
    return ToolCall(
        id=payload.get(K_ID_FIELD, ""),
        function=FunctionCall(
            name=function_payload.get(K_NAME_FIELD, ""),
            arguments=function_payload.get(K_ARGUMENTS_FIELD, ""),
        ),
    )


def _message_from_dict(payload: Dict[str, Any]) -> Message:
    """Build a Message from a ``choices[].message`` dict.

    Missing ``content`` / ``tool_calls`` / ``thinking_blocks`` /
    ``provider_specific_fields`` stay ``None``; a present container of
    the wrong type raises ``ValueError``.
    """
    tool_calls = payload.get(K_TOOL_CALLS_FIELD)
    thinking_blocks = payload.get("thinking_blocks")
    psf = payload.get(K_CHAT_COMPLETIONS_PROVIDER_SPECIFIC_FIELDS_FIELD)
    return Message(
        content=payload.get(K_CONTENT_FIELD),
        tool_calls=(
            [
                _tool_call_from_dict(tc)
                for tc in _require_list(tool_calls, K_TOOL_CALLS_FIELD)
            ]
            if tool_calls is not None
            else None
        ),
        thinking_blocks=(
            [
                _thinking_block_from_dict(tb)
                for tb in _require_list(thinking_blocks, "thinking_blocks")
            ]
            if thinking_blocks is not None
            else None
        ),
        provider_specific_fields=(
            ProviderSpecificFields(
                thought_signatures=_require_dict(
                    psf, K_CHAT_COMPLETIONS_PROVIDER_SPECIFIC_FIELDS_FIELD
                ).get(K_LITELLM_THOUGHT_SIGNATURES_FIELD)
            )
            if psf is not None
            else None
        ),
    )


def _usage_from_dict(payload: Dict[str, Any]) -> Usage:
    """Build a Usage from a ``usage`` dict.

    Missing token counts default to ``0``; missing detail blocks stay
    ``None``; a present detail block of the wrong type raises
    ``ValueError``.
    """
    ctd = payload.get(K_COMPLETION_TOKENS_DETAILS_FIELD)
    ptd = payload.get(K_PROMPT_TOKENS_DETAILS_FIELD)
    return Usage(
        prompt_tokens=payload.get(K_PROMPT_TOKENS_FIELD, 0),
        completion_tokens=payload.get(K_COMPLETION_TOKENS_FIELD, 0),
        total_tokens=payload.get(K_TOTAL_TOKENS_FIELD, 0),
        completion_tokens_details=(
            CompletionTokensDetails(
                reasoning_tokens=_require_dict(
                    ctd, K_COMPLETION_TOKENS_DETAILS_FIELD
                ).get(K_REASONING_TOKENS_FIELD)
            )
            if ctd is not None
            else None
        ),
        prompt_tokens_details=(
            PromptTokensDetails(
                cached_tokens=_require_dict(ptd, K_PROMPT_TOKENS_DETAILS_FIELD).get(
                    K_CACHED_TOKENS_FIELD
                )
            )
            if ptd is not None
            else None
        ),
    )


def _choice_from_dict(payload: Dict[str, Any]) -> Choice:
    """Build a Choice from a ``choices[]`` dict.

    A missing ``index`` becomes ``0`` and a missing ``finish_reason``
    becomes ``"stop"``; a present ``message`` of the wrong type raises
    ``ValueError``.
    """
    message_payload = payload.get(K_MESSAGE_FIELD)
    return Choice(
        index=payload.get(K_INDEX_FIELD, 0),
        finish_reason=payload.get(K_FINISH_REASON_FIELD, K_FINISH_REASON_STOP),
        message=_message_from_dict(
            _require_dict(message_payload, K_MESSAGE_FIELD)
            if message_payload is not None
            else {}
        ),
    )


@dataclass(kw_only=True, slots=True)
class ChatCompletionResponse:
    """Top-level Chat Completions response.

    ``.to_dict()`` matches what a LiteLLM proxy returns for the same
    underlying Vertex response; ``None`` fields are recursively dropped
    (matches LiteLLM's ``exclude_none=True, exclude_unset=True``).
    """

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None
    system_fingerprint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict; recursively drops ``None`` values."""
        return _to_dict_pruned(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ChatCompletionResponse":
        """Rebuild a response object from a wire dict.

        Needed by callers that receive this response over a wire (rather
        than from the in-process converter) and still want the typed
        object.

        This is a lenient reader, not a strict inverse of
        :meth:`to_dict`: missing fields fall back to the type defaults
        (``id=""``, ``created=0``, ``model=""``, ``finish_reason="stop"``,
        ``index=0``, token counts ``0``) and unknown keys are ignored. It
        does round-trip exactly for any value this module's
        ``convert_gemini_response_to_openai`` produces.

        Args:
          payload: A decoded Chat Completions response dict.

        Returns:
          The equivalent :class:`ChatCompletionResponse`.

        Raises:
          ValueError: a container field is present but has the wrong
            type -- an object-typed field (``usage``, a ``choices``
            entry, ``message``, ``provider_specific_fields``, or either
            ``*_tokens_details`` block) that is not an object, or an
            array-typed field (``choices``, ``tool_calls``,
            ``thinking_blocks``) that is not an array.
        """
        choices_payload = payload.get(K_CHOICES_FIELD)
        usage_payload = payload.get(K_USAGE_FIELD)
        return cls(
            id=payload.get(K_ID_FIELD, ""),
            created=payload.get(K_CREATED_FIELD, 0),
            model=payload.get(K_MODEL_FIELD, ""),
            choices=[
                _choice_from_dict(_require_dict(choice, f"{K_CHOICES_FIELD}[{i}]"))
                for i, choice in enumerate(
                    _require_list(choices_payload, K_CHOICES_FIELD)
                    if choices_payload is not None
                    else []
                )
            ],
            usage=(
                _usage_from_dict(_require_dict(usage_payload, K_USAGE_FIELD))
                if usage_payload is not None
                else None
            ),
            system_fingerprint=payload.get(K_SYSTEM_FINGERPRINT_FIELD),
        )


def _to_dict_pruned(obj: Any) -> Any:
    """Recursively convert dataclasses to dicts, pruning ``None`` values.

    Not ``dataclasses.asdict``: that keeps ``None``-valued fields, which
    would diverge from LiteLLM's ``exclude_none`` serialization.
    """
    if is_dataclass(obj) and not isinstance(obj, type):
        result: Dict[str, Any] = {}
        for f in fields(obj):
            value = getattr(obj, f.name)
            if value is None:
                continue
            result[f.name] = _to_dict_pruned(value)
        return result
    if isinstance(obj, list):
        return [_to_dict_pruned(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _to_dict_pruned(v) for k, v in obj.items()}
    return obj


# ── Response-side helpers ────────────────────────────────────────────

# LiteLLM tool-call id separator for embedded thought signatures.
# Consumers keying off `call_<uuid>__thought__<b64sig>` depend on this
# exact spelling.
_LITELLM_TOOL_ID_SEPARATOR = "__thought__"


class _PartitionedParts(NamedTuple):
    """Grouped parts from a single candidate's content.

    text_parts: text-only, non-thought (join into ``message.content``)
    thought_parts: ``thought=True`` text parts (become ``thinking_blocks``)
    fc_parts: parts carrying a ``function_call`` (become ``tool_calls``)
    fc_signatures: base64-encoded thought signatures collected from
      ``function_call`` Parts only, aligned one-for-one with ``fc_parts``
      (become ``provider_specific_fields.thought_signatures``, one entry
      per tool_call). Text-part / thought-part signatures are dropped
      here — see ``_partition_parts`` for why.
    """

    text_parts: List[Part]
    thought_parts: List[Part]
    fc_parts: List[Part]
    fc_signatures: List[str]


def _partition_parts(parts: Sequence[Part]) -> _PartitionedParts:
    """Classify each part and collect function_call-Part thought signatures.

    Signature collection policy: only ``function_call`` Parts contribute
    to ``fc_signatures``, so the list stays aligned index-for-index with
    ``fc_parts`` (and, transitively, with the ``tool_calls`` list built
    by ``_build_tool_calls``). Text and thought Parts' signatures are
    DROPPED from ``fc_signatures`` — emitting them into a flat catch-all
    would misalign the request-side bridge that maps
    ``thought_signatures[i]`` to ``tool_calls[i]``, causing Vertex to
    reject the wrong signature on the wrong Part. Vertex only enforces
    signature echo on ``function_call`` Parts anyway.

    Thought-part signatures ARE still emitted on the response side via
    ``ThinkingBlock.signature`` (see ``_build_thinking_blocks``), but
    this converter's request-side has no reader for
    ``message.thinking_blocks[i].signature`` today — so thought-part
    signatures are effectively emit-only until/unless a request-side
    ``thinking_blocks`` reader is added. This is deliberate: Vertex
    doesn't require thought-part signature echo, so a round-trip has
    no wire-level payoff. Callers who want thought-part signatures
    preserved analytically can read them off ``ThinkingBlock.signature``
    on the response, or off the flat parts-order
    ``thought_signatures_b64`` extra that the word_puzzle adapter
    emits into trajectory JSON.

    fc-part signatures also get embedded in ``tool_calls[i].id`` by
    ``_build_tool_calls`` (LiteLLM's ``__thought__`` convention);
    deliberate duplication for round-tripping through strict OpenAI
    clients that only preserve the ``id`` field.

    Raises ``ValueError`` on parts with no recognized payload, and on
    responses containing media (out of scope).
    """
    text_parts: List[Part] = []
    thought_parts: List[Part] = []
    fc_parts: List[Part] = []
    fc_signatures: List[str] = []

    for j, part in enumerate(parts):
        if part.inline_data.mime_type or part.inline_data.data:
            raise ValueError(
                f"Vertex response part {j} contains media (inline_data); "
                "responses with media are not supported"
            )
        if part.file_data.file_uri or part.file_data.mime_type:
            raise ValueError(
                f"Vertex response part {j} contains media (file_data); "
                "responses with media are not supported"
            )

        has_text = bool(part.text)
        has_fc = bool(part.function_call.name)

        if has_text and part.thought:
            thought_parts.append(part)
        elif has_text:
            text_parts.append(part)
        elif has_fc:
            fc_parts.append(part)
            # Only fc-part signatures feed the flat carrier — see the
            # policy note above for why text/thought sigs are dropped.
            if part.thought_signature:
                fc_signatures.append(
                    base64.b64encode(part.thought_signature).decode("ascii")
                )
        elif part.thought_signature:
            # Bare signature with no other payload is legal but has no
            # fc partner to pair with; produces no per-kind entry.
            continue
        else:
            raise ValueError(f"Vertex candidate part {j} has no recognizable payload")

    return _PartitionedParts(
        text_parts=text_parts,
        thought_parts=thought_parts,
        fc_parts=fc_parts,
        fc_signatures=fc_signatures,
    )


def _build_content(text_parts: Sequence[Part]) -> Optional[str]:
    """Concatenate ``.text`` from all non-thought text parts.

    Returns ``None`` if empty so ``to_dict`` drops ``content`` (LiteLLM parity).
    """
    joined = "".join(p.text for p in text_parts if p.text)
    return joined if joined else None


def _build_thinking_blocks(
    thought_parts: Sequence[Part],
) -> Optional[List[ThinkingBlock]]:
    """One :class:`ThinkingBlock` per thought part.

    ``signature`` is base64-encoded ``thought_signature`` bytes, or
    ``None`` if empty. Returns ``None`` when no thought parts.
    """
    if not thought_parts:
        return None
    blocks: List[ThinkingBlock] = []
    for p in thought_parts:
        sig = (
            base64.b64encode(p.thought_signature).decode("ascii")
            if p.thought_signature
            else None
        )
        blocks.append(ThinkingBlock(thinking=p.text, signature=sig))
    return blocks


def _build_tool_calls(fc_parts: Sequence[Part]) -> Optional[List[ToolCall]]:
    """One :class:`ToolCall` per functionCall part.

    When a part carries a ``thought_signature``, embed it via
    ``f"call_{uuid}__thought__{b64sig}"`` (LiteLLM's convention for
    round-tripping signatures through strict OpenAI clients that only
    preserve the ``id`` field). fc-part signatures are ALSO collected
    into ``provider_specific_fields.thought_signatures`` by
    ``_partition_parts`` — deliberate duplication for parity.
    """
    if not fc_parts:
        return None
    calls: List[ToolCall] = []
    for j, p in enumerate(fc_parts):
        fc = p.function_call
        if not fc.name:
            raise ValueError(f"Vertex candidate function_call part {j} has empty name")
        # proto Struct -> plain dict -> JSON string.
        #
        # MUST be a DEEP conversion. ``dict(fc.args)`` only unwraps the top
        # level: nested values stay as proto ``RepeatedComposite`` (lists) /
        # ``MapComposite`` (objects), which ``json.dumps`` cannot serialize —
        # so any tool call with a list/dict argument raised
        # "args are not JSON-serializable: Object of type RepeatedComposite".
        # proto-plus's ``to_dict`` recurses and yields JSON-native types.
        # ``args`` is absent from ``to_dict`` output when unset, hence the
        # ``.get`` default.
        args_dict = type(fc).to_dict(fc).get("args", {})
        try:
            args_json = json.dumps(args_dict)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Vertex candidate function_call part {j} args are not "
                f"JSON-serializable: {e}"
            ) from e

        call_id = f"call_{uuid.uuid4().hex}"
        if p.thought_signature:
            b64sig = base64.b64encode(p.thought_signature).decode("ascii")
            call_id = f"{call_id}{_LITELLM_TOOL_ID_SEPARATOR}{b64sig}"

        calls.append(
            ToolCall(
                id=call_id,
                function=FunctionCall(name=fc.name, arguments=args_json),
            )
        )
    return calls


def _build_provider_specific_fields(
    fc_signatures: Sequence[str],
) -> Optional[ProviderSpecificFields]:
    """Wrap the fc-part signature list; returns ``None`` when empty."""
    if not fc_signatures:
        return None
    return ProviderSpecificFields(thought_signatures=list(fc_signatures))


# LiteLLM proxy mapping from Vertex FinishReason to OpenAI's finish_reason
# string. STOP is overridden to 'tool_calls' by the caller when the message
# contains tool calls.
_FINISH_MAP: Dict[int, FinishReasonStr] = {
    Candidate.FinishReason.FINISH_REASON_UNSPECIFIED: "stop",
    Candidate.FinishReason.STOP: "stop",
    Candidate.FinishReason.MAX_TOKENS: "length",
    Candidate.FinishReason.SAFETY: "content_filter",
    Candidate.FinishReason.RECITATION: "content_filter",
    Candidate.FinishReason.OTHER: "content_filter",
    Candidate.FinishReason.BLOCKLIST: "content_filter",
    Candidate.FinishReason.PROHIBITED_CONTENT: "content_filter",
    Candidate.FinishReason.SPII: "content_filter",
    Candidate.FinishReason.MALFORMED_FUNCTION_CALL: "stop",
    Candidate.FinishReason.MODEL_ARMOR: "content_filter",
}


def _map_finish_reason(
    vertex_reason: Candidate.FinishReason,
    has_tool_calls: bool,
) -> FinishReasonStr:
    """Map Vertex's finish_reason enum to an OpenAI finish_reason string.

    Applies LiteLLM proxy's STOP -> 'tool_calls' override when the
    message has tool_calls. Raises ``ValueError`` on unknown enum values
    (proto version drift); silently coercing to ``'stop'`` would hide
    legitimately new signal types.
    """
    try:
        mapped = _FINISH_MAP[vertex_reason]
    except KeyError:
        raise ValueError(
            f"Vertex candidate has unknown finish_reason: {vertex_reason!r}"
        ) from None
    if has_tool_calls and vertex_reason in (
        Candidate.FinishReason.STOP,
        Candidate.FinishReason.FINISH_REASON_UNSPECIFIED,
    ):
        return "tool_calls"
    return mapped


def _is_candidates_token_count_inclusive(um: UsageMetadata) -> bool:
    """Whether ``candidates_token_count`` already includes ``thoughts_token_count``.

    Matches LiteLLM's ``VertexGeminiConfig.is_candidate_token_count_inclusive``:
    if ``prompt + candidates == total``, thoughts are inside candidates;
    otherwise they're reported separately and must be added.
    """
    return um.prompt_token_count + um.candidates_token_count == um.total_token_count


def _build_usage(um: Optional[UsageMetadata]) -> Optional[Usage]:
    """Translate Vertex ``UsageMetadata`` proto to :class:`Usage`.

    When Vertex reports ``candidates_token_count`` EXCLUSIVE of thoughts,
    add ``thoughts_token_count`` to ``completion_tokens`` (LiteLLM parity
    via :func:`_is_candidates_token_count_inclusive`); when INCLUSIVE,
    leave it alone. Either way, ``thoughts_token_count`` is separately
    reported via ``completion_tokens_details.reasoning_tokens`` (when > 0).

    Returns ``None`` only if ``um`` itself is ``None``; a proto with
    all-zero fields still yields ``Usage(0, 0, 0)``.
    """
    if um is None:
        return None

    reasoning_tokens = um.thoughts_token_count
    completion_tokens = um.candidates_token_count
    if reasoning_tokens > 0 and not _is_candidates_token_count_inclusive(um):
        completion_tokens = reasoning_tokens + completion_tokens

    completion_details: Optional[CompletionTokensDetails] = None
    if reasoning_tokens > 0:
        completion_details = CompletionTokensDetails(
            reasoning_tokens=reasoning_tokens,
        )
    prompt_details: Optional[PromptTokensDetails] = None
    if um.cached_content_token_count > 0:
        prompt_details = PromptTokensDetails(
            cached_tokens=um.cached_content_token_count,
        )
    return Usage(
        prompt_tokens=um.prompt_token_count,
        completion_tokens=completion_tokens,
        total_tokens=um.total_token_count,
        completion_tokens_details=completion_details,
        prompt_tokens_details=prompt_details,
    )


# ── convert_gemini_response_to_openai ────────────────────────────────


class EmptyCandidatePartsError(ValueError):
    """Vertex returned a candidate with ``parts=[]`` and a non-content-filter
    finish_reason.

    Distinct from ordinary ``ValueError`` (which callers may catch to
    signal a malformed-input bug on our side) because this specific
    condition is an observed Vertex flake: the backend occasionally
    returns an otherwise-well-formed response envelope where a candidate
    has zero parts, typically with ``finish_reason=STOP``. There is
    nothing in the response to convert into an OpenAI ``Choice``.

    Adapter callers (e.g. ``SamplerChatCompletionModel.query``) can
    catch THIS subclass specifically to fall back to a zero-actions
    message (the agent's next step hits the format-error retry path
    and eventually LimitsExceeds out with a partial trajectory) —
    matching the pre-migration ``GeminiSamplerModel._parse_response``
    behavior. Catching plain ``ValueError`` would be too broad: it
    would also swallow malformed-input bugs the sampler layer signals
    via the same base class.

    Subclasses ``ValueError`` for backward compatibility — existing
    ``except ValueError`` blocks still catch it.
    """


def _validate_response_envelope(resp: GenerateContentResponse) -> None:
    """Envelope-level validation; per-part validation is in _convert_candidate."""
    if not resp.candidates:
        raise ValueError("Vertex response has no candidates")


def _convert_candidate(candidate: Candidate, *, index: int) -> Choice:
    """Convert one Vertex ``Candidate`` proto to an OpenAI ``Choice``.

    Empty parts with a content-filter finish reason (SAFETY, RECITATION,
    BLOCKLIST, PROHIBITED_CONTENT, SPII, OTHER, MODEL_ARMOR) is a legitimate
    safety block: emit a Choice with ``content=None``. Empty parts with any
    other finish reason is malformed and raises
    ``EmptyCandidatePartsError`` (a ``ValueError`` subclass so callers
    can catch this specific Vertex-flake pattern separately from
    other input-validation ``ValueError``s from this converter).
    """
    parts = list(candidate.content.parts)
    if not parts:
        finish_reason = _map_finish_reason(
            candidate.finish_reason,
            has_tool_calls=False,
        )
        if finish_reason == "content_filter":
            return Choice(
                index=candidate.index or index,
                message=Message(role="assistant", content=None),
                finish_reason=finish_reason,
            )
        # Include the finish_reason so this flake is self-diagnosing: empty parts
        # only reaches here for finish_reason in {STOP, UNSPECIFIED, MAX_TOKENS,
        # MALFORMED_FUNCTION_CALL} (the non-content-filter set). Distinguishes a
        # malformed-function-call from a truncated/degenerate STOP response.
        raise EmptyCandidatePartsError(
            f"Vertex candidate {index} has empty parts "
            f"(finish_reason={candidate.finish_reason!r} -> {finish_reason!r})"
        )

    # Prefix sub-helper ValueErrors with the candidate index for debuggability.
    try:
        partitioned = _partition_parts(parts)
        content = _build_content(partitioned.text_parts)
        thinking_blocks = _build_thinking_blocks(partitioned.thought_parts)
        tool_calls = _build_tool_calls(partitioned.fc_parts)
        psf = _build_provider_specific_fields(partitioned.fc_signatures)
    except ValueError as e:
        raise ValueError(f"candidate {index}: {e}") from e

    finish_reason = _map_finish_reason(
        candidate.finish_reason,
        has_tool_calls=tool_calls is not None,
    )

    message = Message(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        thinking_blocks=thinking_blocks,
        provider_specific_fields=psf,
    )
    # candidate.index defaults to 0; fall back to loop pos when unset.
    proto_index = candidate.index or index
    return Choice(index=proto_index, message=message, finish_reason=finish_reason)


def convert_gemini_response_to_openai(
    vertex_response: GenerateContentResponse,
    *,
    model: Optional[str] = None,
    response_id: Optional[str] = None,
    created: Optional[int] = None,
) -> ChatCompletionResponse:
    """Translate a Vertex generateContent response proto to a LiteLLM-proxy-
    wire-compatible OpenAI Chat Completions response object.

    ``.to_dict()`` matches what a LiteLLM proxy returns for the same
    underlying Vertex response.

    Envelope priority for id / model / created: explicit kwarg > proto
    field > synthesized fallback. (LiteLLM's proxy always synthesizes
    ``response_id``; we prefer the real Vertex-issued one for traceability.)

    Args:
      vertex_response: The ``GenerateContentResponse`` proto.
      model: Override for the model identifier.
      response_id: Override for the response id.
      created: Override for the unix-timestamp created field.

    Returns:
      A :class:`ChatCompletionResponse` dataclass. Call ``.to_dict()``
      for the JSON-serializable wire dict.

    Raises:
      ValueError: response is malformed.
    """
    _validate_response_envelope(vertex_response)

    choices = [
        _convert_candidate(c, index=i) for i, c in enumerate(vertex_response.candidates)
    ]

    # usage_metadata is always a nested proto (never None);
    # _build_usage handles the all-zero case.
    usage = _build_usage(vertex_response.usage_metadata)

    resolved_id = (
        response_id or vertex_response.response_id or f"chatcmpl-{uuid.uuid4().hex}"
    )
    resolved_model = model or vertex_response.model_version or ""

    if created is not None:
        resolved_created = created
    elif vertex_response.create_time:
        # create_time is a DatetimeWithNanoseconds (datetime subclass) when
        # set or None when unset — NOT a proto Timestamp — so guard on
        # truthiness and use .timestamp().
        resolved_created = int(vertex_response.create_time.timestamp())
    else:
        resolved_created = int(time.time())

    return ChatCompletionResponse(
        id=resolved_id,
        created=resolved_created,
        model=resolved_model,
        choices=choices,
        usage=usage,
    )


# ---------------------------------------------------------------------------
# Reverse direction: Gemini-shaped request -> OpenAI request, and OpenAI
# response -> Gemini-shaped response. Used to drive an OpenAI-compatible
# server (SGLang / vLLM style, which returns ``message.reasoning_content``,
# or ``message.reasoning`` on newer vLLM) from callers that speak
# ``GenerateContentRequest``.
#
# aiplatform_v1 ``FunctionCall`` has no ``id`` field, so tool-call ids are
# synthesized positionally (``<name>:<k>``). Each ``functionResponse`` is
# paired with the pending call of the same name (else the oldest pending
# call); one that answers no call is sent as a plain user turn.
# ---------------------------------------------------------------------------

# OpenAI finish_reason -> Vertex Candidate.FinishReason name. Gemini reports a
# function-calling turn as STOP, so ``tool_calls`` maps there too.
_O2G_FINISH_REASON = {
    "stop": "STOP",
    "tool_calls": "STOP",
    "function_call": "STOP",
    "length": "MAX_TOKENS",
    "content_filter": "SAFETY",
}

# generationConfig (camelCase) -> OpenAI request key.
_G2O_GEN_CFG = {
    "temperature": "temperature",
    "topP": "top_p",
    "topK": "top_k",
    "seed": "seed",
    "presencePenalty": "presence_penalty",
    "frequencyPenalty": "frequency_penalty",
    "stopSequences": "stop",
}

_G2O_PART_FIELDS = ("text", "thought", "thoughtSignature", "functionCall", "functionResponse")


def _g2o_schema(schema: Any) -> Any:
    """Vertex ``Schema`` proto-JSON -> JSON Schema (lowercase ``type`` enums).

    Gemini's ``nullable: true`` (not JSON Schema) becomes ``"null"`` in the
    ``type`` list; ``nullable`` itself is always dropped (also when there is
    no ``type`` to extend). Used for tool parameters and ``responseSchema``.
    """
    if isinstance(schema, list):
        return [_g2o_schema(s) for s in schema]
    if not isinstance(schema, dict):
        return schema
    out: Dict[str, Any] = {}
    for key, value in schema.items():
        if key in ("propertyOrdering", "nullable"):
            continue
        if key == "type" and isinstance(value, str):
            out[key] = value.lower()
        elif key == "properties" and isinstance(value, dict):
            out[key] = {k: _g2o_schema(v) for k, v in value.items()}
        else:
            out[key] = _g2o_schema(value)
    if schema.get("nullable") is True and "type" in out:
        t = out["type"]
        if isinstance(t, str):
            out["type"] = [t, "null"]
        elif isinstance(t, list) and "null" not in t:
            out["type"] = [*t, "null"]
    return out


def _g2o_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for tool in tools:
        for decl in tool.get("functionDeclarations") or []:
            fn: Dict[str, Any] = {"name": decl["name"]}
            if decl.get("description"):
                fn["description"] = decl["description"]
            if "parametersJsonSchema" in decl:
                fn["parameters"] = decl["parametersJsonSchema"]
            elif "parameters" in decl:
                fn["parameters"] = _g2o_schema(decl["parameters"])
            out.append({"type": "function", "function": fn})
        other = sorted(k for k in tool if k != "functionDeclarations")
        if other:
            raise NotImplementedError(f"unsupported tool kinds {other}")
    return out


def _g2o_tool_choice(
    tool_config: Dict[str, Any], declared: Optional[set[str]] = None
) -> Optional[Any]:
    """``toolConfig`` -> ``tool_choice``.

    ``declared``: the declared function names when tools are sent; a single
    ANY-allowed name outside them raises ``ValueError`` (a named choice of an
    undeclared tool is not something the server can honor).
    """
    fcc = tool_config.get("functionCallingConfig") or {}
    mode = fcc.get("mode")
    if mode in (None, "MODE_UNSPECIFIED", "AUTO"):
        return None
    if mode == "NONE":
        return "none"
    if mode == "ANY":
        allowed = fcc.get("allowedFunctionNames") or []
        if len(allowed) == 1:
            if declared is not None and allowed[0] not in declared:
                raise ValueError(
                    f"allowedFunctionNames {sorted(allowed)} match no declared "
                    f"function (declared: {sorted(declared)})"
                )
            return {"type": "function", "function": {"name": allowed[0]}}
        return "required"
    raise NotImplementedError(f"unsupported functionCallingConfig.mode {mode!r}")


def _g2o_text(parts: List[Dict[str, Any]]) -> str:
    # Gemini treats adjacent text parts as one string (as ``_build_content``).
    return "".join(p["text"] for p in parts if p.get("text"))


def _g2o_json(value: Any) -> str:
    """JSON text for a Gemini ``Struct`` value (call args, tool results).

    A ``Struct`` stores every number as a double, so an int the model or tool
    wrote (``3000``) arrives as ``3000.0``; whole-number floats are written as
    ints again (models copy the float form into their later calls). Non-ASCII
    characters are not escaped. Key order is the ``Struct``'s, which may differ
    from the order the model generated.
    """

    def ints(v: Any) -> Any:
        if isinstance(v, float) and v.is_integer():
            return int(v)
        if isinstance(v, dict):
            return {k: ints(x) for k, x in v.items()}
        if isinstance(v, list):
            return [ints(x) for x in v]
        return v

    return json.dumps(ints(value), ensure_ascii=False)


def _g2o_tool_content(response: Dict[str, Any]) -> str:
    """OpenAI tool-message text for a Gemini ``functionResponse.response``.

    Gemini requires ``response`` to be an object, so a tool's plain-text result
    arrives wrapped: under ``"output"`` (the key Gemini's API documents) or
    ``"result"`` (the key ``convert_openai_request_to_gemini`` wraps plain text
    in). A lone string under either key is sent as that raw text, so the model
    reads real newlines and characters instead of an escaped JSON string, and
    plain text survives an OpenAI -> Gemini -> OpenAI round trip unchanged.
    Anything else is sent as JSON (``_g2o_json``).
    """
    if len(response) == 1:
        ((key, value),) = response.items()
        if key in ("output", "result") and isinstance(value, str):
            return value
    return _g2o_json(response)


def _g2o_messages(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    pending: List[Tuple[str, str]] = []  # (call_id, name) awaiting a response
    call_counter = 0
    for content in contents:
        role = content.get("role") or "user"
        parts = content.get("parts") or []
        unsupported = sorted({k for p in parts for k in p if k not in _G2O_PART_FIELDS})
        if unsupported:
            raise NotImplementedError(f"unsupported part fields {unsupported}")

        if role == "model":
            msg: Dict[str, Any] = {
                "role": "assistant",
                "content": _g2o_text([p for p in parts if not p.get("thought")]),
            }
            reasoning = _g2o_text([p for p in parts if p.get("thought")])
            if reasoning:
                msg[K_REASONING_CONTENT_FIELD] = reasoning
            tool_calls = []
            pending = []
            for p in parts:
                fc = p.get("functionCall")
                if fc is None:
                    continue
                call_id = f"{fc['name']}:{call_counter}"
                call_counter += 1
                pending.append((call_id, fc["name"]))
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": fc["name"],
                            "arguments": _g2o_json(fc.get("args") or {}),
                        },
                    }
                )
            if tool_calls:
                msg["tool_calls"] = tool_calls
            messages.append(msg)
            continue

        # user / function turn: tool results first (they answer the previous
        # assistant turn), then any plain text.
        for p in parts:
            fr = p.get("functionResponse")
            if fr is None:
                continue
            tool_content = _g2o_tool_content(fr.get("response") or {})
            if not pending:
                # Gemini accepts a functionResponse that answers no call (e.g.
                # an env nudging a text-only turn); OpenAI requires every tool
                # message to answer a tool_call, so send it as a user turn.
                messages.append({"role": "user", "content": tool_content})
                continue
            idx = next((i for i, (_, n) in enumerate(pending) if n == fr.get("name")), 0)
            call_id, _ = pending.pop(idx)
            messages.append({"role": "tool", "tool_call_id": call_id, "content": tool_content})
        text = _g2o_text([p for p in parts if not p.get("thought")])
        if text:
            messages.append({"role": "user", "content": text})
    return messages


def convert_gemini_request_to_openai(
    wire: Dict[str, Any], *, model: str, n: int, max_tokens: int
) -> Dict[str, Any]:
    """Convert a camelCase ``GenerateContentRequest`` dict to an OpenAI body.

    Thought parts on model turns become ``reasoning_content``. Supported:
    text, thought, function calls/responses, ``tools`` (function
    declarations), ``toolConfig`` and the sampling knobs in
    ``_G2O_GEN_CFG``; other part kinds or tool kinds raise
    ``NotImplementedError``. ``responseMimeType`` ``application/json`` becomes
    ``response_format`` (``json_object``, or ``json_schema`` from
    ``responseJsonSchema`` / ``responseSchema``); other non-text MIME types
    raise ``NotImplementedError``. ``thinkingConfig`` is not translated.
    ``ValueError`` if ``allowedFunctionNames`` (mode ANY) matches no declared
    function.
    """
    messages: List[Dict[str, Any]] = []
    system = _g2o_text((wire.get("systemInstruction") or {}).get("parts") or [])
    if system:
        messages.append({"role": "system", "content": system})
    messages.extend(_g2o_messages(wire.get("contents") or []))

    body: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "n": n,
        "max_tokens": max_tokens,
    }
    gen_cfg = wire.get("generationConfig") or {}
    for src_key, dst_key in _G2O_GEN_CFG.items():
        if src_key in gen_cfg:
            body[dst_key] = gen_cfg[src_key]
    # thinkingConfig is deliberately dropped: chat completions has no
    # thinking-level knob (OpenAICompatibleSampler warns about thinking_level).
    response_format = _g2o_response_format(gen_cfg)
    if response_format is not None:
        body["response_format"] = response_format
    if wire.get("tools"):
        body["tools"] = _g2o_tools(wire["tools"])
    if wire.get("toolConfig"):
        declared_names = (
            {t["function"]["name"] for t in body["tools"]} if body.get("tools") else None
        )
        choice = _g2o_tool_choice(wire["toolConfig"], declared_names)
        if choice is not None:
            body["tool_choice"] = choice
        # ANY restricted to 2+ names: OpenAI's tool_choice names at most one
        # function, so express the restriction by offering only those tools.
        fcc = wire["toolConfig"].get("functionCallingConfig") or {}
        allowed = fcc.get("allowedFunctionNames") or []
        if choice == "required" and allowed and body.get("tools"):
            declared = {t["function"]["name"] for t in body["tools"]}
            if not declared.intersection(allowed):
                # Filtering would leave no tools with tool_choice "required",
                # which the server rejects; fail loudly instead.
                raise ValueError(
                    f"allowedFunctionNames {sorted(allowed)} match no declared "
                    f"function (declared: {sorted(declared)})"
                )
            body["tools"] = [t for t in body["tools"] if t["function"]["name"] in allowed]
    return body


def _g2o_response_format(gen_cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """generationConfig structured-output keys -> OpenAI ``response_format``."""
    mime = gen_cfg.get("responseMimeType")
    if mime in (None, "", "text/plain"):
        return None
    if mime != "application/json":
        raise NotImplementedError(f"unsupported generationConfig.responseMimeType {mime!r}")
    if gen_cfg.get("responseJsonSchema") is not None:
        schema = gen_cfg["responseJsonSchema"]  # already JSON Schema
    elif gen_cfg.get("responseSchema") is not None:
        schema = _g2o_schema(gen_cfg["responseSchema"])
    else:
        return {"type": "json_object"}
    return {"type": "json_schema", "json_schema": {"name": "response", "schema": schema}}


def _o2g_reasoning(message: Dict[str, Any]) -> Any:
    """``reasoning_content``, else ``reasoning`` (newer vLLM's field name)."""
    return message.get(K_REASONING_CONTENT_FIELD) or message.get("reasoning")


def _o2g_candidate(choice: Dict[str, Any], position: int) -> Dict[str, Any]:
    message = choice.get("message") or {}
    parts: List[Dict[str, Any]] = []
    reasoning = _o2g_reasoning(message)
    if reasoning:
        parts.append({"text": reasoning, "thought": True})
    if message.get("content"):
        parts.append({"text": message["content"]})
    malformed = False
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        raw_args = fn.get("arguments") or "{}"
        if isinstance(raw_args, dict):
            args: Any = raw_args
        else:
            try:
                args = json.loads(raw_args)
            except (json.JSONDecodeError, TypeError):
                args = None
        if not isinstance(args, dict):
            # Keep the malformed call visible in the step instead of dropping it.
            args = {"__raw_arguments__": raw_args}
            malformed = True
        parts.append({"functionCall": {"name": fn.get("name", ""), "args": args}})
    return {
        # Fall back to position so candidate ids stay distinct when the
        # server omits ``index``.
        "index": choice.get("index", position),
        "content": {"role": "model", "parts": parts},
        # The server says "tool_calls" even when it could not form the
        # arguments; report that the way Gemini does, so the turn is not
        # treated (and executed / trained on) as a clean STOP.
        "finishReason": "MALFORMED_FUNCTION_CALL" if malformed else _O2G_FINISH_REASON.get(
            choice.get("finish_reason") or "", "OTHER"
        ),
        # This candidate's own raw choice, for per-candidate raw capture. Not
        # a Candidate field; the proto parse ignores it.
        "rawChoice": choice,
    }


def convert_openai_response_to_gemini(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an OpenAI chat.completion body to ``generateContent`` JSON shape.

    ``completion_tokens`` includes reasoning, so output-only tokens are
    ``completion_tokens - reasoning_tokens`` (Gemini's ``candidatesTokenCount``
    excludes thinking); reasoning tokens are read from ``usage.reasoning_tokens``
    or ``usage.completion_tokens_details.reasoning_tokens``. A count the server
    did not report is omitted (not zeroed), so content without generation
    counts still fails loudly downstream. If the server reports no reasoning
    token count but the reply carries reasoning text, all
    ``completion_tokens`` are booked as ``candidatesTokenCount`` and
    ``thoughtsTokenCount`` is absent (reasoning is not split out;
    ``OpenAICompatibleSampler`` warns once when this happens). The raw body minus
    ``choices`` is kept under ``rawChatCompletion`` and each candidate carries
    its own raw choice (``rawChoice``), so a per-candidate raw-capture slice
    holds the true wire response for that candidate only.
    """
    usage = data.get("usage") or {}
    reasoning = usage.get("reasoning_tokens")
    if reasoning is None:
        reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    usage_md: Dict[str, Any] = {}
    if usage.get("completion_tokens") is not None:
        usage_md["candidatesTokenCount"] = max(
            int(usage["completion_tokens"]) - int(reasoning or 0), 0
        )
    if reasoning is not None:
        usage_md["thoughtsTokenCount"] = int(reasoning)
    if usage.get("prompt_tokens") is not None:
        usage_md["promptTokenCount"] = int(usage["prompt_tokens"])
    return {
        "candidates": [
            _o2g_candidate(c, i) for i, c in enumerate(data.get("choices") or [])
        ],
        "usageMetadata": usage_md,
        "responseId": data.get("id"),
        "modelVersion": data.get("model"),
        "rawChatCompletion": {k: v for k, v in data.items() if k != "choices"},
    }


# ---------------------------------------------------------------------------
# Reasoning-field normalization between this module's response model
# (LiteLLM-style ``thinking_blocks``, reasoning tokens under
# ``usage.completion_tokens_details``) and servers that use
# ``message.reasoning_content`` / top-level ``usage.reasoning_tokens``
# (SGLang / vLLM style). Pure; inputs are not mutated.
# ---------------------------------------------------------------------------


def thinking_blocks_to_reasoning_content(openai_request: Dict[str, Any]) -> Dict[str, Any]:
    """Copy of an OpenAI request with assistant ``thinking_blocks`` turned into
    ``reasoning_content`` (the field such servers read back from history)."""
    out = copy.deepcopy(openai_request)
    for msg in out.get(K_MESSAGES_FIELD) or []:
        if msg.get("role") == "assistant" and "thinking_blocks" in msg:
            blocks = msg.pop("thinking_blocks") or []
            text = "".join(
                b.get("thinking") or "" for b in blocks if isinstance(b, dict)
            )
            if text and not msg.get(K_REASONING_CONTENT_FIELD):
                msg[K_REASONING_CONTENT_FIELD] = text
    return out


def reasoning_content_to_thinking_blocks(response: Dict[str, Any]) -> Dict[str, Any]:
    """Copy of an OpenAI response in the shape ``ChatCompletionResponse``
    models: ``reasoning_content`` (else ``reasoning``) -> ``thinking_blocks``,
    and top-level ``usage.reasoning_tokens`` ->
    ``usage.completion_tokens_details``."""
    out = copy.deepcopy(response)
    for choice in out.get("choices") or []:
        msg = choice.get("message") or {}
        # ``reasoning`` is newer vLLM's name for ``reasoning_content``.
        alt = msg.pop("reasoning", None)
        reasoning = msg.pop(K_REASONING_CONTENT_FIELD, None) or alt
        if reasoning and not msg.get("thinking_blocks"):
            msg["thinking_blocks"] = [{"type": "thinking", "thinking": reasoning}]
    usage = out.get("usage")
    if isinstance(usage, dict) and usage.get("reasoning_tokens") is not None:
        details = dict(usage.get("completion_tokens_details") or {})
        details.setdefault("reasoning_tokens", usage["reasoning_tokens"])
        usage["completion_tokens_details"] = details
    return out
