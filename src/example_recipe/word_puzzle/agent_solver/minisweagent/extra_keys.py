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

"""Keys the agent stashes on a mini message's ``extra`` dict.

The writer (``async_default_agent``) and the reader
(``mini_atif_converter``) do not import each other, and deliberately so:
the agent's dependency envelope is kept to minisweagent plus stdlib
because it also runs inside the pod sandbox, whereas the converter pulls
in ``core.atif_*`` and the reward functions. A bare string literal on
each side is therefore exactly the shape that drifts apart silently, so
the key lives here instead -- a module with no imports, safe for both.

Read defensively at the consumer: a message may predate a key (a pod
running a previously baked agent image records nothing), and turns that
ran no tool never set ``ENV_ELAPSED_SECONDS`` at all.
"""

from __future__ import annotations

# Wall clock, in seconds, around one turn's tool execution. Written by
# AsyncDefaultAgent.execute_actions; read by mini_atif_converter into
# StepTimings.env_time_s.
ENV_ELAPSED_SECONDS = "env_elapsed_seconds"
