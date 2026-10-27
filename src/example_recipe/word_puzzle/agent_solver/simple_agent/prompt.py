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

"""Prompt templates and Gemini tool declarations for the simple agent solver.

Provides Jinja2 system/instance templates and ``TOOLS`` — a list of
``Tool`` protos with ``bash`` and ``submit`` function declarations ready
to pass to ``GenerateContentRequest.tools``.
"""

from __future__ import annotations

import jinja2
from google.cloud.aiplatform_v1.types import (
    FunctionDeclaration,
    Schema,
    Tool,
    Type,
)

# ---------------------------------------------------------------------------
# Jinja2 templates
# ---------------------------------------------------------------------------

SYSTEM_TEMPLATE = jinja2.Template(
    """\
You are a math puzzle solver.

You have access to two tools:
- **bash**: execute a bash command and see the output.
- **submit**: submit your final answer.

You have at most {{ step_limit }} tool-use turns. \
Use bash to explore the problem and verify your answer, \
then call submit with your final answer.""",
    undefined=jinja2.StrictUndefined,
)

INSTANCE_TEMPLATE = jinja2.Template(
    """\
Solve the following puzzle. \
When you are confident, call the submit tool with your answer.

{{ task }}""",
    undefined=jinja2.StrictUndefined,
)

# Opt-in (``answer_only`` / ``SimpleAgentConfig.answer_only_submit``): states
# the submit contract. submit is graded by exact match on `answer`, while some
# dataset prompts ask the model to explain its reasoning; without this, a model
# may put the explanation in `answer`.
INSTANCE_TEMPLATE_ANSWER_ONLY = jinja2.Template(
    """\
Solve the following puzzle. \
When you are confident, call the submit tool with only your final answer; \
if the puzzle asks you to explain your reasoning, do that in your message, \
not in the submit argument.

{{ task }}""",
    undefined=jinja2.StrictUndefined,
)

# ---------------------------------------------------------------------------
# Tool declarations (Gemini proto format)
# ---------------------------------------------------------------------------

BASH_TOOL_DECL = FunctionDeclaration(
    name="bash",
    description="Run a bash command in a sandbox and return stdout/stderr.",
    parameters=Schema(
        type=Type.OBJECT,
        properties={
            "command": Schema(
                type=Type.STRING,
                description="The bash command to execute.",
            ),
        },
        required=["command"],
    ),
)

SUBMIT_TOOL_DECL = FunctionDeclaration(
    name="submit",
    description="Submit your final answer to the puzzle.",
    parameters=Schema(
        type=Type.OBJECT,
        properties={
            "answer": Schema(
                type=Type.STRING,
                description="Your final answer.",
            ),
        },
        required=["answer"],
    ),
)

# Opt-in counterpart of SUBMIT_TOOL_DECL; see INSTANCE_TEMPLATE_ANSWER_ONLY.
SUBMIT_TOOL_DECL_ANSWER_ONLY = FunctionDeclaration(
    name="submit",
    description=(
        "Submit your final answer to the puzzle. The answer is graded by "
        "exact string match."
    ),
    parameters=Schema(
        type=Type.OBJECT,
        properties={
            "answer": Schema(
                type=Type.STRING,
                description=(
                    "Only the final answer string, with no explanation, "
                    "labels or formatting."
                ),
            ),
        },
        required=["answer"],
    ),
)

TOOLS = [Tool(function_declarations=[BASH_TOOL_DECL, SUBMIT_TOOL_DECL])]
TOOLS_ANSWER_ONLY = [
    Tool(function_declarations=[BASH_TOOL_DECL, SUBMIT_TOOL_DECL_ANSWER_ONLY])
]


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------


def render_system_prompt(*, step_limit: int) -> str:
    """Render the system prompt with the given step limit."""
    return SYSTEM_TEMPLATE.render(step_limit=step_limit)


def render_instance_prompt(*, task: str, answer_only: bool = False) -> str:
    """Render the instance (user) prompt with the given task text."""
    template = INSTANCE_TEMPLATE_ANSWER_ONLY if answer_only else INSTANCE_TEMPLATE
    return template.render(task=task)
