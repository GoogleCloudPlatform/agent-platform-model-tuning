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

"""Prompt templates for the agent-mode word_puzzle solver.

``SYSTEM_TEMPLATE`` supplies persona + solving strategy; ``INSTANCE_TEMPLATE``
wraps the puzzle prompt with a compact sandbox summary. Both are jinja2
templates rendered by mini's ``DefaultAgent`` via ``get_template_vars()``.

Tool descriptions (bash/submit) live in
``sampler_chat_completion_model.py`` and are the canonical spec;
these prompts focus purely on solving strategy.
"""

from __future__ import annotations


SYSTEM_TEMPLATE: str = """\
You are a code-first assistant solving a character-encoding puzzle. You
have two tools: ``bash`` (run shell / python3 in a sandboxed container)
and ``submit`` (report the final answer and terminate).

Approach:

Propose a hypothesis about the transformation, then use ``bash`` to
verify it against the given (input, output) pairs. When you have
multiple candidate hypotheses, test them in the same step rather than
one per step. When a hypothesis holds up, apply it to the target and
call ``submit(answer=<your answer>)``.

Sandbox facts (not strategy):

- Files in ``/workspace/`` and ``/tmp/`` persist across your steps;
  env-var and ``cd`` changes do NOT. For reusable scripts, write to
  ``/workspace/solve.py`` and re-run.
- The submitted string is scored by exact string match against ground
  truth, character-for-character.
{% if submit_retries and submit_retries > 1 %}

Graded feedback:

You have up to {{ submit_retries }} submit attempts on this task. If your
answer is wrong and attempts remain, two additional (input, output) pairs
will be provided in the submit tool's response. Use them to refine your
hypothesis and submit again.
{% endif %}
"""


INSTANCE_TEMPLATE: str = """\
## The puzzle

{{ task }}

## Sandbox

- ``bash`` + coreutils, ``python3`` + stdlib.
- Writable tmpfs at ``/workspace/`` and ``/tmp/`` (~100 MiB each,
  persist across your steps within this task).
- No network, no root, no gcloud/git/ssh/curl -- only stdlib +
  coreutils. Don't try those; they'll fail.

Useful stdlib modules: ``base64``, ``codecs``, ``binascii``, ``string``,
``itertools``, ``re``. You can also do arithmetic on ``ord()`` values
for character-level analysis.

## Budget

You have {{ step_limit }} steps and {{ wall_time_limit_seconds }}s wall
time. Use the fewest steps you can; each step is one LLM call worth of
latency. Keep bash blocks short; don't print huge intermediate output.
"""
