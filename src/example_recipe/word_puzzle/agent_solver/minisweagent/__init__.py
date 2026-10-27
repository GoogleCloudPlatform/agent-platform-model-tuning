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

"""mini-swe-agent word_puzzle solver prototype.

Wires mini-swe-agent's DefaultAgent + DockerEnvironment to this repo's
GeminiApiSampler so a gemini-3.5-flash agent can attempt word puzzles
inside a hardened sandbox (no host FS, no network, no credentials).
"""


def __getattr__(name: str):  # noqa: ANN201
    """Lazy-import heavy dependencies to avoid dragging in the third-party
    ``minisweagent`` package when only lightweight submodules (e.g.
    ``extract_answer``) are needed.
    """
    if name in ("SamplerChatCompletionModel", "SamplerChatCompletionModelConfig"):
        from example_recipe.word_puzzle.agent_solver.minisweagent.sampler_chat_completion_model import (
            SamplerChatCompletionModel,
            SamplerChatCompletionModelConfig,
        )

        _exports = {
            "SamplerChatCompletionModel": SamplerChatCompletionModel,
            "SamplerChatCompletionModelConfig": SamplerChatCompletionModelConfig,
        }
        # Cache in module dict so subsequent accesses skip __getattr__.
        globals().update(_exports)
        return _exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SamplerChatCompletionModel",
    "SamplerChatCompletionModelConfig",
]
