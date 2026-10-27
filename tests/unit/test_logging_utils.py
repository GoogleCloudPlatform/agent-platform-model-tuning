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

"""Tests for logging_utils.simplify_trajectory_step_for_logging."""

from utils.logging_utils import simplify_trajectory_step_for_logging


def test_unknown_object_returns_repr():
    out = simplify_trajectory_step_for_logging(42)
    assert "repr" in out


def test_dict_step_with_gemini_contents_truncates_text():
    long = "z" * 300
    step = {
        "prompt": {
            "contents": [
                {
                    "parts": [
                        {"text": long},
                        {
                            "functionResponse": {
                                "response": {
                                    "response": {"stdout": long, "stderr": "ok"}
                                }
                            }
                        },
                    ]
                }
            ],
            "system_instruction": {"parts": [{"text": long}]},
        },
        "generation": {"parts": [{"text": long}]},
    }
    out = simplify_trajectory_step_for_logging(step)
    parts = out["prompt"]["contents"][0]["parts"]
    assert "truncated" in parts[0]["text"]
    inner = parts[1]["functionResponse"]["response"]["response"]
    assert "truncated" in inner["stdout"]
    assert inner["stderr"] == "ok"  # short string untouched
    assert "truncated" in out["prompt"]["system_instruction"]["parts"][0]["text"]
    assert "truncated" in out["generation"]["parts"][0]["text"]


def test_dict_short_strings_untouched():
    step = {"prompt": "hi", "generation": "ok"}
    out = simplify_trajectory_step_for_logging(step)
    assert out == {"prompt": "hi", "generation": "ok"}


def test_function_response_non_nested_response_dict():
    long = "y" * 200
    step = {
        "prompt": {
            "contents": [
                {
                    "parts": [
                        {"functionResponse": {"response": {"foo": long, "bar": "ok"}}}
                    ]
                }
            ]
        },
    }
    out = simplify_trajectory_step_for_logging(step)
    resp = out["prompt"]["contents"][0]["parts"][0]["functionResponse"]["response"]
    assert "truncated" in resp["foo"]
    assert resp["bar"] == "ok"
