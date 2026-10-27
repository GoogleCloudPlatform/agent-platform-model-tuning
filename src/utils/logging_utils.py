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

from typing import Any, Dict

from google.cloud.aiplatform_v1.types import GenerateContentRequest


def simplify_trajectory_step_for_logging(step) -> Dict[str, Any]:
    """Simplify a TrajectoryStep for logging by truncating large fields.

    Accepts three shapes:
      * atif ``TrajectoryStep`` (pydantic model exposing ``raw_prompt`` /
        ``raw_generation`` properties).
      * a plain ``dict`` with ``"prompt"``/``"generation"`` keys.
      * a legacy dataclass with ``.prompt``/``.generation`` fields.
    """
    import dataclasses

    # atif TrajectoryStep is a Pydantic model, not a dataclass; use its
    # typed raw_prompt / raw_generation properties.
    if hasattr(step, "raw_prompt") and hasattr(step, "raw_generation"):
        data = {
            "prompt": step.raw_prompt,
            "generation": step.raw_generation,
        }
    elif hasattr(step, "__dataclass_fields__"):
        try:
            data = dataclasses.asdict(step)
        except Exception:
            data = {
                "prompt": getattr(step, "prompt", None),
                "generation": getattr(step, "generation", None),
                "extras": getattr(step, "extras", {}),
            }
    elif isinstance(step, dict):
        import copy

        data = copy.deepcopy(step)
    else:
        return {"repr": repr(step)}

    # Convert proto prompts to a camelCase dict up-front so the truncation
    # path below applies uniformly (dataclasses.asdict leaves protos intact).
    if isinstance(data.get("prompt"), GenerateContentRequest):
        # Local import to avoid a package-level cycle.
        from backends.prompt_coerce import to_wire_dict

        data["prompt"] = to_wire_dict(data["prompt"])

    def _truncate(v, max_len=100):
        if isinstance(v, str) and len(v) > max_len:
            return f"{v[: max_len // 2]}...[{len(v) - max_len} chars truncated]...{v[-max_len // 2 :]}"
        return v

    if isinstance(data.get("prompt"), str):
        data["prompt"] = _truncate(data["prompt"])
    elif isinstance(data.get("prompt"), dict):
        if "contents" in data["prompt"]:
            for c in data["prompt"]["contents"]:
                if "parts" in c:
                    for p in c["parts"]:
                        if "text" in p:
                            p["text"] = _truncate(p["text"])

                        fr = p.get("functionResponse") or p.get("function_call")
                        if isinstance(fr, dict) and "response" in fr:
                            resp = fr["response"]
                            if isinstance(resp, dict):
                                if "response" in resp:
                                    inner_resp = resp["response"]
                                    if isinstance(inner_resp, dict):
                                        for k, v in inner_resp.items():
                                            if isinstance(v, str):
                                                inner_resp[k] = _truncate(v)
                                else:
                                    for k, v in resp.items():
                                        if isinstance(v, str):
                                            resp[k] = _truncate(v)

        # Accept both snake_case and camelCase (hand-built dict prompts vs.
        # proto -> wire-dict output).
        si = data["prompt"].get("systemInstruction") or data["prompt"].get(
            "system_instruction"
        )
        if isinstance(si, dict) and "parts" in si:
            for p in si["parts"]:
                if "text" in p:
                    p["text"] = _truncate(p["text"])

    if isinstance(data.get("generation"), str):
        data["generation"] = _truncate(data["generation"])
    elif isinstance(data.get("generation"), dict) and "parts" in data["generation"]:
        for p in data["generation"]["parts"]:
            if "text" in p:
                p["text"] = _truncate(p["text"])

    return data
