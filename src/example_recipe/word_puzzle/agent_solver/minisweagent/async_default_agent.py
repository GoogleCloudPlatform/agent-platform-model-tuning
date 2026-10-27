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

"""Async subclass of mini-swe-agent's DefaultAgent.

Mini's stock ``DefaultAgent`` is sync all the way down. This subclass
converts the four hot-path methods (``run``, ``step``, ``query``,
``execute_actions``) to ``async def`` so N concurrent rollouts can share
one event loop + one sampler. Everything else is inherited unchanged
(template rendering, trajectory save, limit checks, exit-message
conversion). Overrides keep upstream's control flow for easy porting.

Blocking-call handling: ``model.query`` is awaited directly (async in
our ``SamplerChatCompletionModel``); ``env.execute`` is blocking
``subprocess.run`` so it's wrapped in ``asyncio.to_thread``.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import Sequence
from typing import Any

from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import (
    FormatError,
    InterruptAgentFlow,
    LimitsExceeded,
    Submitted,
    TimeExceeded,
)

from example_recipe.word_puzzle.agent_solver.minisweagent.extra_keys import (
    ENV_ELAPSED_SECONDS,
)
from example_recipe.word_puzzle.dataset.puzzle_dataset_gen import apply_transform


# Cap on bash-tool output bytes per call, applied after ``env.execute``
# returns and before it enters the trajectory. 64 KB is roughly 16k
# tokens; protects against pathological unbounded-output scripts that
# blow the trajectory / context budget. Head+tail slices with a
# "[...truncated N bytes...]" marker preserve both traceback ends and
# expose the byte count so the model can adapt on the next turn.
_CAP_BASH_OUTPUT_MAX_BYTES = 65536


def _cap_bash_output(output: dict, max_bytes: int = _CAP_BASH_OUTPUT_MAX_BYTES) -> dict:
    """Truncate ``output['output']`` if it exceeds ``max_bytes`` UTF-8 bytes.

    Head-and-tail truncation with a ``[...truncated N bytes (M total)...]``
    marker between the halves. Non-string / missing ``output`` fields pass
    through (mini's DockerEnvironment leaves ``output=None`` on some error
    paths); other dict fields are preserved. Byte count uses UTF-8 encoding
    so multibyte chars don't sneak past the cap. Mutates in place AND
    returns for chaining.
    """
    text = output.get("output")
    if not isinstance(text, str):
        return output
    n = len(text.encode("utf-8"))
    if n <= max_bytes:
        return output
    half = max_bytes // 2
    # decode-with-replace so a split mid-multibyte-char yields valid utf-8
    # rather than a UnicodeDecodeError in the trajectory writer.
    b = text.encode("utf-8")
    head = b[:half].decode("utf-8", errors="replace")
    tail = b[-half:].decode("utf-8", errors="replace")
    output["output"] = (
        head + f"\n[...truncated {n - max_bytes} bytes ({n} total)...]\n" + tail
    )
    return output


class AsyncDefaultAgent(DefaultAgent):
    """Async sibling of ``DefaultAgent`` with optional graded-feedback retry.

    Usage (mirrors DefaultAgent, but ``run`` is a coroutine)::

        agent = AsyncDefaultAgent(model=async_model, env=docker_env, ...)
        exit_info = await agent.run(task=puzzle_prompt)

    When ``submit_retries > 1``, a wrong ``submit(answer=...)`` call does
    NOT terminate. The agent draws 2 fresh words from a task-local shuffled
    ``extra_word_pool``, applies ``rule`` (respecting ``reverse_task``),
    and appends a function-response observation carrying the new pairs.
    Retries continue until correct or the budget is exhausted. The retry
    rng is seeded per task (``task_seed``) for reproducibility.
    """

    def __init__(
        self,
        *args: Any,
        submit_retries: int = 1,
        expected_answer: str = "",
        rule: Sequence[Sequence[Any]] | None = None,
        extra_word_pool: Sequence[str] | None = None,
        reverse_task: bool = False,
        task_seed: int = 0,
        **kwargs: Any,
    ) -> None:
        """Extend DefaultAgent.__init__ with the retry-feature knobs.

        Args:
          submit_retries: Total ``submit`` attempts allowed per task. ``1``
            (default) disables the retry mechanic. ``N > 1`` = up to N
            attempts; wrong submits with budget remaining inject 2 fresh
            worked examples.
          expected_answer: Ground-truth answer; used to grade each submit.
            Ignored when ``submit_retries=1``.
          rule: The puzzle's transformation rule as a sequence of
            ``(op_name, param)`` pairs. Used to compute the ``output`` of
            injected pairs. Ignored when the feature is off.
          extra_word_pool: Words available to draw injected pairs from.
            Must be disjoint from the demo words + task word (enforced
            at generation time by ``puzzle_dataset_gen``). Empty pool
            degrades the retry mechanic to single-attempt behaviour
            (wrong submit terminates as if ``submit_retries=1``).
          reverse_task: Direction flag matching the puzzle's question
            direction (which side the model must PRODUCE from the
            other). ``False`` (default) = "given encrypted, produce
            original" -> injected pairs are ``(encrypted(w), w)``.
            ``True`` = "given original, produce encrypted" ->
            injected pairs are ``(w, encrypted(w))``. Note: the demo
            section in the prompt always shows ``(raw, encrypted)``
            regardless of this flag (see
            ``puzzle_dataset_gen._example_block``); what
            ``reverse_task`` changes is only which side of that pair
            the QUESTION asks the model to produce.
          task_seed: Seed for the per-task rng that shuffles the pool
            at run() start. Same seed + same pool = same shuffle =
            same first pair; enables bitwise-reproducible evals.
        """
        super().__init__(*args, **kwargs)
        self._submit_retries = submit_retries
        self._expected_answer = expected_answer
        self._rule = list(rule) if rule else []
        self._extra_word_pool_raw = list(extra_word_pool) if extra_word_pool else []
        self._reverse_task = reverse_task
        self._task_seed = task_seed
        # Fail-early on a malformed ``rule`` so an unknown-operator error
        # surfaces at construction, not mid-rollout on the first wrong
        # submit (where the runner's broad except swallows it as a
        # per-task crash). Skipped when the feature is off so the
        # "retries=1 preserves original behaviour" contract holds for
        # runners that pass ``rule`` through unconditionally.
        if self._submit_retries > 1 and self._extra_word_pool_raw:
            try:
                apply_transform("test", self._rule)
            except ValueError as e:
                raise ValueError(
                    f"AsyncDefaultAgent: retry_metadata.rule contains "
                    f"an operator that apply_transform doesn't "
                    f"recognize (rule={self._rule!r}). Original error: "
                    f"{e}. Regenerate the eval JSONL with a compatible "
                    f"puzzle_dataset_gen version, or pass "
                    f"submit_retries=1 to disable the retry mechanic."
                ) from e
            # Empty ``expected_answer`` when the feature is on is
            # almost certainly a caller bug (default was left
            # unchanged). If we don't guard it here, every submit
            # compares against "" and is scored wrong -- the entire
            # retry budget burns silently on every task, and the only
            # symptom is "the feature doesn't help". Fail loud instead.
            if not self._expected_answer:
                raise ValueError(
                    "AsyncDefaultAgent: submit_retries>1 with a "
                    "non-empty ``extra_word_pool`` requires a "
                    "non-empty ``expected_answer`` (ground truth used "
                    "to grade each submit call). Got expected_answer="
                    f"{self._expected_answer!r}. Pass the row's "
                    "reference answer, empty the pool, or set "
                    "submit_retries=1 to disable the retry mechanic."
                )
        # Retry runtime state; ``_init_retry_state`` populates these at
        # the top of ``run()`` (and directly from tests bypassing run()).
        self._shuffled_pool: list[str] = []
        self._pool_cursor = 0
        self._retries_used = 0

    def _init_retry_state(self) -> None:
        """Reset per-task retry state: shuffle the pool with the task rng.

        Uses a fresh ``random.Random(task_seed)`` (not the global rng) so
        concurrent rollouts on the shared event loop don't race on rng
        state. Also publishes ``submit_retries`` into ``extra_template_vars``
        so ``SYSTEM_TEMPLATE`` can render a retry-aware message.
        """
        rng = random.Random(self._task_seed)
        self._shuffled_pool = list(self._extra_word_pool_raw)
        rng.shuffle(self._shuffled_pool)
        self._pool_cursor = 0
        self._retries_used = 0
        self.extra_template_vars |= {"submit_retries": self._submit_retries}

    async def run(self, task: str = "", **kwargs: Any) -> dict:
        """Async version of ``DefaultAgent.run``."""
        self.extra_template_vars |= {"task": task, **kwargs}
        # MUST run BEFORE the templates below render: publishes
        # ``submit_retries`` into ``extra_template_vars`` which the
        # system_template references under ``{% if submit_retries > 1 %}``.
        # StrictUndefined would raise otherwise.
        self._init_retry_state()
        self.messages = []
        self.add_messages(
            self.model.format_message(
                role="system",
                content=self._render_template(self.config.system_template),
            ),
            self.model.format_message(
                role="user",
                content=self._render_template(self.config.instance_template),
            ),
        )

        self._start_time = time.time()
        while True:
            try:
                await self.step()
                self.n_consecutive_format_errors = 0
            except FormatError as e:
                self.n_consecutive_format_errors += 1
                if (
                    0
                    < self.config.max_consecutive_format_errors
                    <= self.n_consecutive_format_errors
                ):
                    self.add_messages(
                        *e.messages,
                        {
                            "role": "exit",
                            "content": "RepeatedFormatError",
                            "extra": {
                                "exit_status": "RepeatedFormatError",
                                "submission": "",
                            },
                        },
                    )
                else:
                    self.add_messages(*e.messages)
            except InterruptAgentFlow as e:
                self.add_messages(*e.messages)
            except Exception as e:  # noqa: BLE001
                self.handle_uncaught_exception(e)
                raise
            finally:
                self.save(self.config.output_path)
            if self.messages[-1].get("role") == "exit":
                break
        return self.messages[-1].get("extra", {})

    async def step(self) -> list[dict]:
        """Async version of ``DefaultAgent.step``."""
        return await self.execute_actions(await self.query())

    async def query(self) -> dict:
        """Async version of ``DefaultAgent.query`` (awaits the model)."""
        if (
            0 < self.config.step_limit <= self.n_calls
            or 0 < self.config.cost_limit <= self.cost
        ):
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )
        if (
            0
            < self.config.wall_time_limit_seconds
            <= int(time.time() - self._start_time)
        ):
            raise TimeExceeded(
                {
                    "role": "exit",
                    "content": "TimeExceeded",
                    "extra": {"exit_status": "TimeExceeded", "submission": ""},
                }
            )
        self.n_calls += 1
        message = await self.model.query(self.messages)
        self.cost += message.get("extra", {}).get("cost", 0.0)
        self.add_messages(message)
        return message

    async def execute_actions(self, message: dict) -> list[dict]:
        """Dispatch on ``action["tool_name"]`` (``bash`` or ``submit``).

        ``bash`` runs via ``env.execute`` wrapped in
        ``asyncio.to_thread`` (docker exec is blocking). ``submit``
        bypasses the env and either (a) raises ``Submitted`` to
        terminate, or (b) triggers the graded-feedback retry path
        (fresh examples + continue) via
        ``_maybe_build_retry_observation``. Any other tool_name is a
        parser bug and raises.

        Parallel-tool-calls policy: the model
        MAY emit multiple tool_calls per turn (bash + submit together,
        or several bashes together). We run ALL non-submit actions
        first, appending their outputs as one observation message,
        THEN handle the deferred submit (if any). This way:

          * ``[bash, submit]`` runs the bash, appends its output, then
            handles submit -- both actions are visible in the
            trajectory.
          * ``[submit, bash]`` runs the bash first anyway (output
            captured), then handles submit -- submit is treated as
            terminal-at-end-of-turn, not terminal-immediately, so
            leading bash calls aren't silently dropped along with
            their outputs.
          * Multiple submits in one turn: use the FIRST submit's
            answer (deterministic; parallel submits are model
            confusion, not a legitimate use case).

        Interaction with graded-feedback retry: when the
        deferred submit's answer is wrong AND retries remain AND
        enough pool words are left, ``_maybe_build_retry_observation``
        returns a retry observation and we return normally instead of
        raising ``Submitted``. Any prior-run bash observations remain
        on the trajectory (already added above).
        """
        actions = message.get("extra", {}).get("actions", [])
        bash_actions: list[dict] = []
        submit_action: dict | None = None
        for action in actions:
            tool_name = action.get("tool_name")
            if tool_name == "submit":
                # Defer submits until AFTER all non-submit tools have
                # run. First submit wins if the model emits multiple.
                if submit_action is None:
                    submit_action = action
            elif tool_name == "bash":
                bash_actions.append(action)
            else:
                raise ValueError(
                    f"AsyncDefaultAgent.execute_actions: unknown tool_name "
                    f"{tool_name!r} in action {action!r}"
                )

        # Run all bash actions in parallel via asyncio.gather.  Each
        # docker-exec is blocking (env.execute), so we wrap each in
        # asyncio.to_thread.  gather() preserves input order, so the
        # i-th output corresponds to the i-th bash action -- the
        # action/output pairing stays correct for the observation
        # message below.
        if bash_actions:
            t_env_start = time.monotonic()
            raw_outputs = await asyncio.gather(
                *(asyncio.to_thread(self.env.execute, a) for a in bash_actions)
            )
            # Wall clock around the WHOLE gather, not per action: the bash
            # actions of one turn run in parallel, so summing their
            # individual durations would over-count a multi-bash turn.
            # This is the per-turn analogue of the timing the core
            # multi-step agent takes around its single ``env.step``.
            #
            # Stamped on the ASSISTANT message, not the tool message:
            # ``message`` is the same dict ``query`` appended to
            # ``self.messages``, and the ATIF converter builds each agent
            # step from that message's ``extra`` (taking only the CONTENT
            # of the following tool message). A turn with no bash leaves
            # the key unset -- zero env time is the truth there -- and the
            # converter reads it defensively, so a transcript recorded
            # before this key existed still converts.
            message.setdefault("extra", {})[ENV_ELAPSED_SECONDS] = (
                time.monotonic() - t_env_start
            )
            outputs = [_cap_bash_output(o) for o in raw_outputs]
        else:
            outputs = []
        executed_actions = bash_actions

        # Emit the observation message for the non-submit outputs
        # FIRST (regardless of whether a submit is queued), so the
        # trajectory shows every bash call the model asked for. Without
        # this, a ``[bash, submit]`` turn would leave the bash's output
        # off the trajectory even though env.execute ran it.
        #
        # IMPORTANT: pass a shim message whose ``extra.actions`` list
        # contains ONLY the executed (non-submit) actions. The mini
        # helper ``format_toolcall_observation_messages`` zips
        # ``actions`` against ``outputs`` (padding the shorter with a
        # ``not_executed`` placeholder), so passing the unfiltered list
        # here would misalign the tool_call_id -> output pairing in
        # two ways:
        #   * ``[bash, submit]`` -> spurious role=tool message emitted
        #     with tool_call_id=<submit's id> and "action was not
        #     executed" content (submit is never a tool call the
        #     model expects a tool_result back from — its result IS
        #     the exit).
        #   * ``[submit, bash]`` -> outputs list is [bash_output], but
        #     the zip pairs (submit, bash_output) and (bash,
        #     not_executed). Submit's tool_call_id ends up carrying
        #     bash's stdout on disk; bash's tool_call_id gets
        #     "not executed". Trajectory JSON is factually wrong.
        # Filtering to ``executed_actions`` makes the zip 1:1 by
        # construction: |executed_actions| == |outputs| always.
        added: list[dict] = []
        if outputs:
            shim_extra = {**message.get("extra", {}), "actions": executed_actions}
            shim_message = {**message, "extra": shim_extra}
            added = self.add_messages(
                *self.model.format_observation_messages(
                    shim_message, outputs, self.get_template_vars()
                )
            )

        # Now handle the deferred submit, if any. Two paths:
        #   (a) Graded-feedback retry: if the answer is
        #       wrong AND retries remain AND the pool has >=2 words,
        #       inject fresh examples via
        #       _maybe_build_retry_observation and CONTINUE the
        #       rollout. Any bash observations from THIS turn (added
        #       above) stay on the trajectory alongside the retry
        #       observation.
        #   (b) Terminate: raise Submitted with the answer. role="exit"
        #       matches DockerEnvironment._check_finished so the
        #       run-loop's ``messages[-1]["role"] == "exit"`` check
        #       breaks correctly. The Submitted exception bypasses the
        #       env entirely -- the answer is authoritative.
        if submit_action is not None:
            answer = submit_action.get("answer", "")
            retry_obs = self._maybe_build_retry_observation(submit_action, answer)
            if retry_obs is not None:
                # Non-terminal submit: on the next turn the retry
                # observation will convert to a Vertex ``functionResponse``
                # Part. The converter requires the paired ``functionCall``
                # to be present in the prior assistant turn (Vertex rejects
                # unpaired ``functionResponse`` parts).
                #
                # ``SamplerChatCompletionModel._make_message`` filters
                # submit ``tool_calls`` off the assistant message at emit
                # time under the assumption that submit is always terminal
                # (the common ``--submit-retries=1`` case). When retries
                # ARE enabled and this submit is being retried, that
                # assumption is wrong — re-insert the submit ``tool_call``
                # so the next-turn wire pairing holds.
                #
                # ``message`` is the SAME dict object the run-loop
                # appended to ``self.messages`` right before calling
                # ``execute_actions``, so mutating it here is visible on
                # the trajectory on the next ``self.model.query`` call.
                self._restore_submit_tool_call(message, submit_action)
                return self.add_messages(retry_obs)
            raise Submitted(
                {
                    "role": "exit",
                    "content": answer,
                    "extra": {
                        "exit_status": "Submitted",
                        "submission": answer,
                        "tool_call_id": submit_action.get("tool_call_id"),
                    },
                }
            )
        return added

    # ------------------------------------------------------------------
    # Graded-feedback retry helpers
    # ------------------------------------------------------------------

    def _restore_submit_tool_call(self, message: dict, submit_action: dict) -> None:
        """Re-insert the filtered-out submit ``tool_call`` on ``message``.

        Called from ``execute_actions`` on the non-terminal-submit path
        (retry mode). ``SamplerChatCompletionModel._make_message``
        strips submit entries from the assistant message's
        ``tool_calls`` list to keep the common ``--submit-retries=1``
        trajectory replay-safe (unpaired ``functionCall`` would trip the
        converter). When we're about to append a retry observation, the
        submit is NOT terminal after all, so we need to restore its
        ``tool_call`` so the wire history pairs correctly on the next
        turn.

        Reconstruction is lossless: ``submit_action`` (which came from
        ``_parse_tool_calls`` before filtering) preserves both
        ``tool_call_id`` and ``answer``, which are exactly the fields
        the OpenAI ``tool_call`` shape needs. Order matters: if the
        assistant had other (non-submit) tool_calls in the same turn,
        those are already at their original indices from the emit-time
        filter; appending the submit at the end preserves that. This
        also matches the emit-time first-submit-wins rule (a
        model-emitted duplicate submit is discarded, so at most one
        entry is ever added).

        Restores TWO aligned invariants:

          1. ``message["tool_calls"]`` gains the submit entry at the
             end (as described above).

          2. ``message["provider_specific_fields"]["thought_signatures"]``
             — IF present — gains an aligned entry at the SAME index
             using the signature the model returned for this submit
             on the original response. ``_make_message`` stashed that
             signature onto ``submit_action["thought_signature"]``
             specifically so we could echo it back here.

             Why this matters: without the signature, the restored
             submit functionCall Part goes on the wire without a
             ``thoughtSignature`` attribute, and gemini-3.5-flash
             rejects with HTTP 400:
                 "Function call is missing a thought_signature in
                  functionCall parts. This is required for tools to
                  work correctly ..."
             (Triggers only when other functionCalls in the same
             conversation DO have signatures, which is the normal
             case with ``--thinking-level != minimal``.)

             Empirically observed: 14/150 crashes in the first
             ``--submit-retries 3 --thinking-level high`` eval run
             (2026-07-15), all HTTP 400 on ``default_api:submit``.

             Falls back to empty string ("") if the stashed signature
             is missing or empty — matches the pre-fix behavior for
             the rare case where the model returned no signature for
             the submit (thinking off, or an empty-parts response
             fixed up by the failsafe path).

             ``_make_message``'s envelope-preservation invariant:
             even when only the bash slot has a real signature,
             ``if any(aligned_sigs):`` keeps the envelope populated
             (with the bash sig + no submit sig, filtered length 1),
             so this branch fires and we grow the list to length 2.

        No-op if ``message["tool_calls"]`` already contains the submit
        (defensive: makes the helper idempotent, though the call site
        only invokes it once per turn).
        """
        tool_call_id = submit_action.get("tool_call_id")
        answer = submit_action.get("answer", "")
        existing = message.get("tool_calls") or []
        for tc in existing:
            if tc.get("id") == tool_call_id:
                return  # already present; nothing to do.
        submit_tc = {
            "id": tool_call_id,
            "type": "function",
            "function": {
                "name": "submit",
                "arguments": json.dumps({"answer": answer}),
            },
        }
        message["tool_calls"] = list(existing) + [submit_tc]

        # Ensure provider_specific_fields.thought_signatures carries
        # a signature for the restored submit. Uses the stashed
        # signature from ``submit_action["thought_signature"]`` (put
        # there by ``SamplerChatCompletionModel._make_message`` at
        # emit time) so Vertex doesn't reject the retry-turn with
        # ``Function call is missing a thought_signature``.
        #
        # Three cases to handle:
        #   (a) psf envelope is present with a list of length N (= filtered
        #       tool_calls count): extend the list by 1 with the submit's sig.
        #   (b) psf envelope is absent AND the submit had a real signature
        #       stashed: CREATE the envelope with a properly-aligned list.
        #       This is the [submit] solo-tool-call case where
        #       ``_make_message``'s ``if any(aligned_sigs):`` gate collapsed
        #       the envelope after the filter removed the only signed entry.
        #       Without this reconstruction, the restored submit goes on
        #       the wire without a signature (chat_completion.py:1047 drops
        #       empty-string sigs from ``_apply_google_extra_to_part``, and
        #       there's no psf list to emit from). Length of new list is 1
        #       (matches the new tool_calls length: 0 pre-restore + 1 submit).
        #   (c) psf absent AND no stashed sig: leave psf absent. Vertex
        #       tolerates all-signatures-absent when nothing else in the
        #       conversation was signed.
        submit_sig = submit_action.get("thought_signature") or ""
        psf = message.get("provider_specific_fields")
        if isinstance(psf, dict):
            sigs = psf.get("thought_signatures")
            if isinstance(sigs, list):
                # case (a): extend existing list.
                psf["thought_signatures"] = list(sigs) + [submit_sig]
        elif submit_sig:
            # case (b): create envelope so the restored submit's
            # signature reaches the wire. Length aligns to the new
            # tool_calls length (previous filter had removed the sole
            # signed entry along with the submit tool_call, so pre-
            # restore tool_calls was empty; post-restore is [submit]).
            pre_restore_len = len(existing)
            message["provider_specific_fields"] = {
                "thought_signatures": [""] * pre_restore_len + [submit_sig],
            }

    def _maybe_build_retry_observation(self, action: dict, answer: str) -> dict | None:
        """Return a retry observation message, or None if we should terminate.

        Returns None (=> caller raises ``Submitted``) when ANY of:
          * ``submit_retries <= 1`` (feature disabled).
          * ``answer == expected_answer`` (correct: terminate success).
          * this wrong submit would be the final attempt.
          * pool has fewer than 2 words left (a 1-pair observation would
            violate the "2 fresh pairs on wrong submit" contract).

        Otherwise draws the next 2 words from ``_shuffled_pool``, builds
        the observation, increments the retry counter + pool cursor.
        """
        if self._submit_retries <= 1:
            return None
        if answer == self._expected_answer:
            return None
        # Attempts remaining AFTER consuming this wrong attempt.
        attempts_remaining = self._submit_retries - (self._retries_used + 1)
        if attempts_remaining <= 0:
            return None
        if self._pool_cursor + 2 > len(self._shuffled_pool):
            return None

        pair_words = self._shuffled_pool[self._pool_cursor : self._pool_cursor + 2]
        self._pool_cursor += 2
        self._retries_used += 1

        # Build (input, output) pairs matching the puzzle's QUESTION
        # direction (see ``reverse_task`` docstring on ``__init__``
        # above for the full rationale re: demo-vs-question direction).
        new_pairs: list[list[str]] = []
        for w in pair_words:
            encrypted = apply_transform(w, self._rule)
            if self._reverse_task:
                # Task: given original, produce encrypted.
                new_pairs.append([w, encrypted])
            else:
                # Task: given encrypted, produce original.
                new_pairs.append([encrypted, w])

        # Wire contract: this becomes a ``functionResponse`` Part on the
        # next Vertex generateContent turn. The
        # ``SamplerChatCompletionModel`` adapter path relies on the
        # generic ``convert_openai_request_to_gemini`` converter, which
        # only routes a message to ``functionResponse`` when
        # ``role in {"tool", "function"}`` (see chat_completion.py:1170).
        # A ``role="user"`` message becomes a plain text Part instead,
        # which drifts the model out of tool-calling mode.
        #
        # ``tool_call_id`` at message level (mini's OpenAI-tool
        # convention) lets the converter's ``tool_call_id_to_name`` map
        # populate ``functionResponse.name = "submit"`` — required so
        # the prior turn's ``functionCall(submit)`` pairs correctly.
        # That pairing in turn requires the prior assistant message to
        # STILL carry the submit ``tool_call``; see the fixup in
        # ``execute_actions`` right before this observation is
        # appended.
        #
        # ``content`` is a JSON dump of the payload — the converter
        # will re-parse it into ``functionResponse.response`` via
        # ``_convert_json_string_to_struct``.
        payload = {
            "status": "wrong_answer",
            "wrong_answer": answer,
            "attempts_remaining": attempts_remaining,
            "additional_examples": [
                {"input": input_word, "output": output_word}
                for input_word, output_word in new_pairs
            ],
            "instruction": (
                "Additional examples provided. Refine your hypothesis and submit again."
            ),
        }
        return {
            "role": "tool",
            "content": json.dumps(payload),
            "tool_call_id": action.get("tool_call_id"),
            "extra": {
                # Redundant with the top-level ``tool_call_id`` (kept
                # for trace-viewer / ATIF converters that read from
                # ``extra`` uniformly across bash + submit observations).
                "tool_call_id": action.get("tool_call_id"),
                "tool_name": "submit",
                # Retry-audit fields for the trace-viewer / ATIF converter.
                "retry_attempt": self._retries_used,
                "wrong_answer": answer,
                "new_pairs": new_pairs,
                "attempts_remaining": attempts_remaining,
            },
        }
