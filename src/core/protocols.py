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

"""Abstract protocols for envs, samplers, trainers, reward normalizers, and reward fns.

These ABCs/Protocols define the seam between the training loops (which
orchestrate rollouts and train steps) and concrete backends/recipes
(which implement sampling, gradient updates, and reward shaping). They
are pure interfaces with no behavior.
"""

from __future__ import annotations

import abc
import dataclasses
import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TypeVar

from google.cloud.aiplatform_v1.types import Content, GenerateContentRequest

from core.atif_trajectory import ScoredTrajectory, ToolCall, TrajectoryGroup
from core.sample_result import ChatCompletionSampleResult, SampleResult
from utils.raw_capture import CaptureContext


@dataclasses.dataclass
class EnvStepResult:
    """The outcome of a single Env.step call.

    Attributes:
        function_responses: Tool/function-response messages (Vertex
            ``Part``-shaped dicts) appended to the conversation as
            ``{"role": "user", "parts": [...]}`` for the next sampler turn.
        reward: Optional scalar reward for this step.
        done: Whether the episode is finished after this step.
        is_correct: Optional framework-recognized correctness signal;
            set by envs with a natural notion of correctness so the
            framework can aggregate correctness without recipe-specific
            wiring.
        task_stats: Recipe-owned typed per-step stats. Framework-opaque
            (``Any``): only the recipe's aggregator consumes it.
    """

    function_responses: list[dict[str, Any]]
    reward: float | None
    done: bool
    is_correct: bool | None = None
    task_stats: Any = None


@dataclasses.dataclass(frozen=True)
class Action:
    """Typed payload passed to Env.step().

    Replaces the pre-refactor ``dict[str, Any] | str`` seam. Assembled
    at the Agent boundary from a ``SampleResult``; envs consume typed
    attributes instead of ``.get()``-ing a loose dict.

    ``content`` is REQUIRED (non-Optional). Concrete ``Agent``
    subclasses (e.g. ``SingleStepAgent`` in ``core.single_step_agent``)
    short-circuit sampling failures (empty ``candidate_id`` from a
    backend error / timeout) before dispatching to ``env.step``;
    multi-turn recipe agents additionally short-circuit on non-STOP
    terminations. Envs are therefore guaranteed ``action.content`` is
    a real proto — but may still see empty ``content.parts`` on
    legitimate STOP-with-no-output cases (e.g. MAX_TOKENS where the
    whole budget went to reasoning), which envs handle by natural
    iteration (empty for-loop over parts).

    Fields mirror the data that concrete Agents extract from a
    ``SampleResult`` per candidate: the proto ``Content`` plus per-
    candidate sidecar metadata (finish reason, generation token count,
    candidate id) that isn't carried on the proto itself.
    """

    content: Content | None = None  # proto path only; None on the OpenAI path
    finish_reason: str = ""  # proto FinishReason enum name
    generation_token_count: int = 0  # sampler contract: always populated
    candidate_id: str | None = None  # None only when backend omits it

    # -- OpenAI / vendor-neutral path -------------------------------------
    # ``content`` is a Google proto and so hard-codes one vendor. Recipes on
    # the ``TrainableAgent`` path carry the model's turn here instead: the
    # tool calls it emitted (already validated and coerced by the
    # dispatcher) plus any assistant text. An env sees ONE Action per turn
    # and fans out over ``tool_calls`` itself, which is what keeps the
    # transition function -- and therefore world semantics -- inside the env
    # where a tool-surface ablation cannot reach it.
    tool_calls: tuple[ToolCall, ...] = ()
    text: str | None = None
    # ``tool_call_id -> seconds``: the per-call execution budget the tool
    # declared, for the env to ENFORCE. It rides the Action rather than being
    # a keyword on ``step`` because every Env subclass in the tree overrides
    # ``step``, and a keyword most of them ignore would have to be re-declared
    # in each one. A call absent from the mapping declares no budget and the
    # env falls back to its own default -- never the other way round: a
    # container exec silently cut at an env default is indistinguishable from
    # a model failure.
    tool_timeouts: Mapping[str, float] = dataclasses.field(default_factory=dict)

    def with_calls(self, calls: Sequence[ToolCall]) -> "Action":
        """Returns a copy carrying ``calls`` (post-coercion) instead."""
        return dataclasses.replace(self, tool_calls=tuple(calls))

    def timeout_for(self, call_id: str, default: float) -> float:
        """The declared budget for one call, or ``default``.

        ``default`` is also the CEILING an env must clamp any caller-supplied
        override to: without it a model can name its own timeout and pin a
        container for as long as it likes.
        """
        budget = self.tool_timeouts.get(call_id)
        return budget if budget and budget > 0 else default


class Env(abc.ABC):
    """Evaluates an action and returns next state, reward, and done flag."""

    @abc.abstractmethod
    async def step(self, action: Action, *, step_count: int = 0) -> EnvStepResult:
        """Applies the given action and returns the step result.

        ``action`` is a typed :class:`Action` carrying the model's
        response ``content`` (proto ``Content``) plus per-candidate
        sidecar metadata (``finish_reason``, ``generation_token_count``,
        ``candidate_id``). Envs access typed attributes instead of
        dict lookups.

        The Agent short-circuits sampling failures (empty
        ``candidate_id`` from a backend error / timeout) before
        dispatching to ``env.step``; multi-turn recipe agents
        additionally short-circuit on non-STOP terminations. Envs are
        therefore guaranteed ``action.content`` is a real proto — but
        may still see empty ``content.parts`` on legitimate STOP-with-
        no-output cases, which envs handle by natural iteration
        (empty for-loop over parts).

        ``step_count`` is the 1-based count of ``env.step()`` calls so
        far in the current rollout (i.e. ``1`` on the first call, ``2``
        on the second, ...). Provided by the agent's rollout loop; envs
        that need step-index-aware behavior (e.g. a multi-step recipe's
        per-step bonus / decay gate) read it directly. Envs that don't
        need it may ignore it. The default of ``0`` is a benign no-op
        for single-step / stateless envs where the value is never
        consulted.

        ``action.tool_timeouts`` carries the per-call execution budget
        each tool declared, and enforcing it is the ENV's job: it is the
        only layer that knows how to abort the thing it started (kill the
        exec, cancel the coroutine). A caller-supplied ``timeout``
        argument must be clamped to that budget, never allowed to raise
        it. See :meth:`Action.timeout_for`.
        """
        ...

    def handled_tools(self) -> frozenset[str]:
        """Tool names this env can serve. Empty means "no check".

        Declared on the protocol, with a default, so a cross-check at
        rollout start can call it unconditionally. When it was merely a
        convention, the caller had to probe with ``getattr(env,
        "handled_tools", None)`` — and one whole class of env answering
        ``None`` meant the fail-fast was silently skipped for every rollout
        on that path, which is the failure it exists to prevent: a tool the
        env cannot serve becomes N identical error observations at runtime
        instead of one startup error.
        """
        return frozenset()

    def get_prompt(self) -> GenerateContentRequest:
        """Returns the initial prompt for the environment (proto path only).

        Not abstract: an env on the OpenAI path is prompted from
        ``TaskData.instruction`` by the agent, never from a Gemini proto it
        would have to build. Proto-path envs still override it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no proto prompt; it is prompted from "
            f"TaskData on the chat-completions path."
        )

    async def get_state(self) -> Any:
        """Returns the world state for the verifier to grade.

        The env owns the state; the grader reads it through this one
        accessor rather than downcasting to a concrete env class. Default
        returns ``self`` so a grader that genuinely needs the typed surface
        still has it, without every env having to implement a dump.

        This is the GRADER's accessor and nothing else — see
        :meth:`snapshot` for persistence (deviation F4).
        """
        return self

    async def snapshot(self) -> Any:
        """Returns a JSON-able dump of the final world, or ``None``.

        The counterpart to :meth:`get_state`, split off it because one
        method was serving two callers with incompatible needs: the
        grader wants the live typed object, the journal wants bytes it
        can write under ``artifacts/env_final_state/``. Conflating them
        meant "do not persist my world" was expressed as a *serialisation
        failure* swallowed by the journal — one recipe's ~78 KB world
        state stayed out of trajectories by accident, and would have
        started landing in all of them the day someone made it
        serialisable.

        Default ``None`` = "nothing to persist". Opting in is implementing
        this method; the journal size-caps and truncation-marks whatever
        comes back, and never fails a rollout over it.
        """
        return None

    async def aclose(self) -> None:
        """Releases sandbox resources. MUST be idempotent and MUST NOT raise.

        Called by ``RolloutScope`` on every exit path including cancellation,
        inside a shielded window.
        """
        return None

    def set_rollout_generation_tokens(self, total: int | None) -> None:
        """Running total of generation tokens charged to this rollout.

        Called by the agent loop before each ``step``, so the value is
        inclusive of the action about to be dispatched. Unlike
        ``Action.generation_token_count`` (one candidate) this covers the
        WHOLE rollout, including harness-initiated calls that never
        produce an ``Action`` -- notably in-loop context compaction,
        whose summarization call is generation the rollout paid for.

        ``None`` means no call reported a count; envs must keep that
        distinct from ``0``, since a cost-shaped reward reads a missing
        count as "unknown" (no credit) and a real ``0`` as "free"
        (full credit).

        Default: ignore. Envs with a cost-shaped reward override it;
        everything else needs no change.
        """
        del total


class Sampler(abc.ABC):
    """Generates actions (e.g. completions) for a given state (e.g. prompt)."""

    # -- Single-sample convenience (concrete) ---------------------------------

    async def sample(
        self,
        prompt: GenerateContentRequest | str,
        *,
        thinking_level: str | None = None,
        max_generation_tokens: int | None = None,
        capture_ctx: CaptureContext | None = None,
    ) -> SampleResult | None:
        """Generate a single sample. Returns ``None`` on empty candidates.

        Delegates to ``sample_n(prompt, 1, ...)``. Use this for the
        common single-sample path (``Agent.rollout``, multi-turn per-step
        calls).  Use ``sample_n`` directly when you need batched fanout.
        """
        results = await self.sample_n(
            prompt,
            1,
            thinking_level=thinking_level,
            max_generation_tokens=max_generation_tokens,
            capture_ctxs=[capture_ctx] if capture_ctx is not None else None,
        )
        return results[0] if results else None

    # -- Batched N-sample (abstract — backends implement this) ---------------

    @abc.abstractmethod
    async def sample_n(
        self,
        prompt: GenerateContentRequest | str,
        num_samples: int,
        *,
        thinking_level: str | None = None,
        max_generation_tokens: int | None = None,
        capture_ctxs: Sequence[CaptureContext | None] | None = None,
    ) -> list[SampleResult]:
        """Generate ``num_samples`` candidates in a single wire call.

        Args:
            prompt: ``GenerateContentRequest`` proto or bare string.
            num_samples: Number of candidates to generate.
            thinking_level: Optional reasoning-budget hint.
            max_generation_tokens: Per-call token cap; ``None`` uses
                the backend's module-level default.
            capture_ctxs: Per-candidate capture contexts (length must
                equal ``num_samples``).  See ``utils/raw_capture.py``.
        """
        ...

    async def sample_chat_completion(
        self,
        openai_request: dict[str, Any],
        num_samples: int = 1,
        thinking_level: str | None = None,
        capture_ctxs: Sequence[CaptureContext | None] | None = None,
    ) -> ChatCompletionSampleResult:
        """Sample via the OpenAI Chat Completions wire shape.

        Template Method on the ``Sampler`` ABC. Converts
        ``openai_request`` to a Vertex ``GenerateContentRequest``,
        delegates to ``self.sample()``, then reconstitutes a
        ``GenerateContentResponse`` proto and converts back to an
        OpenAI-shape ``ChatCompletionResponse``.

        ``num_samples`` overrides any ``n`` on the incoming request.

        Args:
            openai_request: OpenAI Chat Completions request dict. Must
                contain at minimum ``model`` and ``messages``.
            num_samples: Number of candidates to request. Overrides any
                ``n`` in ``openai_request``.
            thinking_level: Optional reasoning-budget hint; forwarded
                to ``self.sample()`` unchanged.
            capture_ctxs: Optional per-candidate capture contexts;
                forwarded to ``self.sample()`` unchanged. See
                ``Sampler.sample()`` docstring for semantics.

        Returns:
            A ``ChatCompletionSampleResult`` wrapping both the
            OpenAI-shape response (``.response``) and the underlying
            per-candidate ``list[SampleResult]`` (``.results``). Use
            ``.response.to_dict()`` for the JSON-serializable wire dict;
            use ``.results`` for training-relevant metadata (candidate_id,
            response_id, thoughts_token_count, thought_signature) that
            the OpenAI wire shape doesn't preserve.

        Raises:
            ValueError: ``openai_request`` malformed, or the sampler
                failed to populate required request-scoped metadata on
                ``SamplingContext``.
            RuntimeError: underlying ``sample()`` returned a failsafe
                empty result (``candidate_id == ""``); upstream call
                failed and there's no real response to reconstitute.
            NotImplementedError: request uses features the converter
                doesn't yet support (e.g. ``extra_body``,
                ``response_format``, tools).
        """
        # Lazy imports keep non-chat callers from paying the cost at
        # core-package import time and break a would-be cycle.
        from utils import raw_capture

        from backends._proto_utils import reconstitute_generate_content_response
        from backends.chat_completion import (
            ConversionContext,
            convert_gemini_response_to_openai,
            convert_openai_request_to_gemini,
        )

        # num_samples wins over any "n" in the dict.
        req_copy = dict(openai_request)
        req_copy["n"] = num_samples
        vertex_dict = convert_openai_request_to_gemini(ConversionContext(), req_copy)
        # ignore_unknown_fields lets the converter emit
        # forward-compatible keys without breaking parse.
        vertex_request = GenerateContentRequest.from_json(
            json.dumps(vertex_dict), ignore_unknown_fields=True
        )

        # Record the CLIENT-form request (this dict, tool schemas and all)
        # on every raw-capture blob the call below writes. The backends only
        # ever see `vertex_request`, so without this the ledger cannot say
        # what the agent actually asked for — and tool rendering alone has
        # been measured to move a benchmark's pass rate by ~2.5 points. One
        # place, all backends.
        with raw_capture.client_request_scope(req_copy):
            results = await self.sample_n(
                vertex_request,
                num_samples,
                thinking_level=thinking_level,
                capture_ctxs=capture_ctxs,
            )

        # Failsafe empty result: converting it would yield a bogus
        # "success" response with an empty Candidate.
        if results and results[0].tuning_candidate.candidate_id == "":
            # NAME the reason. The failsafe carries WHY it is empty on its
            # context, and this raise is the only thing the caller sees --
            # so anything not repeated here is lost, and a caller
            # classifying the exception has to fall back to "unknown
            # transport error". That is not a cosmetic loss: an over-long
            # prompt is an INPUT fault (deterministic: re-sending it cannot
            # help, and it is evidence about the rollout), while an unknown
            # transport error is INFRA (retriable, and NaN rather than a
            # score). Dropping the reason silently converts the first into
            # the second.
            ctx = results[0].context
            if ctx.is_prompt_too_long:
                reason = "the prompt is too long for this model"
            elif ctx.is_malformed_usage:
                reason = (
                    "the response arrived with unusable usageMetadata "
                    "(no prompt token count)"
                )
            else:
                reason = "upstream sampling failed"
            raise RuntimeError(
                f"{type(self).__name__}.sample() returned failsafe empty "
                f"result — {reason}. Check sampler logs for the underlying "
                f"HTTP / LRO error."
            )

        vertex_response = reconstitute_generate_content_response(
            results, sampler_name=type(self).__name__
        )
        return ChatCompletionSampleResult(
            response=convert_gemini_response_to_openai(vertex_response),
            results=results,
        )


class Trainer(abc.ABC):
    """Takes trajectories and updates model weights."""

    @abc.abstractmethod
    async def train_step(
        self,
        trajectories: list[ScoredTrajectory],
        sync_trainer_and_sampler: bool = True,
    ) -> dict[str, Any]:
        """Executes a training step and returns metrics.

        Returned as ``dict[str, Any]`` because trainers emit a mix of
        numeric scalars and nested per-step breakdowns; the downstream
        aggregation layer is the typed consumer.

        Server-returned ``metricName`` values are typed via
        ``core.trainer_metrics.TRAINER_METRICS``; a backend routes its
        wire ``metrics`` through ``map_trainer_wire_metrics``, which
        drops undeclared names with a one-time WARN per wire_name.
        """
        ...


class RewardNormalizer(abc.ABC):
    """Normalizes rewards across a batch of trajectories.

    Concrete implementations (GRPO, centering, etc.) live in
    ``core.reward_normalizer``; select one via a ``RewardNormalizerConfig``
    subclass in ``core.reward_normalizer_config`` (its ``make()`` builds the
    normalizer).
    """

    @abc.abstractmethod
    def normalize(
        self, trajectory_groups: list[TrajectoryGroup]
    ) -> list[TrajectoryGroup]:
        """Normalizes rewards in trajectories within groups.

        Args:
            trajectory_groups: A list of ``TrajectoryGroup`` objects
                representing a batch.

        Returns:
            The same list (mutated or new) with normalized rewards.
        """
        ...


# The recipe defines its own SignalsT (input) and StatsT (aggregation
# snapshot) dataclasses; both are recipe-typed and opaque to the framework.
SignalsT = TypeVar("SignalsT", contravariant=True)
StatsT = TypeVar("StatsT", covariant=True)


class RewardFn(Protocol[SignalsT, StatsT]):
    """Recipe-plugged reward-shaping hook: ``signals -> (reward, task_stats)``.

    Called by the recipe (in Env.step or the recipe's Agent); the
    framework never invokes RewardFn — this protocol is here only so
    recipes can share type vocabulary and external callers can be typed.

    Framework-owned failure rewards (TIMEOUT NaN, max_steps floor) do
    NOT flow through RewardFn: it shapes SUCCESS rewards from raw
    signals; the Agent independently slots in floors on failure.

    Contract:
      - Purity: pure implementations (no I/O, time, or randomness) SHOULD
        stay pure. Impure implementations (e.g. reward fns that consult
        an external judge) are allowed, but MUST document their impurity
        and MUST provide a documented reward floor for the failure path
        (see "Failure semantics" below).
      - Total: malformed/edge-case inputs get a documented floor, not
        an exception.
      - Determinism: reward shaping that depends on step index or
        task_data must have those fields IN the SignalsT dataclass,
        not read from ambient state. Impure implementations that call
        into a non-deterministic judge are exempt, but the caller must
        be able to tolerate different rewards on repeated calls with
        the same signals.
      - Failure semantics: ``__call__`` MUST NOT raise into the caller.
        Impure implementations catch their own network / parse / judge
        errors internally and return the documented reward floor on
        failure. Pure implementations satisfy this trivially (they
        cannot raise on well-formed input).
    """

    async def __call__(self, signals: SignalsT) -> tuple[float, StatsT]: ...


class ProgressObserver(Protocol):
    """Structural protocol for experiment-level progress hooks.

    Accepted as an optional observer by the eval/training orchestrators
    and pipeline loops, for shipping run progress to an external
    dashboard.  Methods are all async and never raise into the pipeline —
    implementations swallow errors internally.
    """

    async def on_experiment_started(self, **kwargs: Any) -> None: ...

    async def on_experiment_completed(
        self, status: str, wall_time_seconds: float
    ) -> None: ...

    async def on_train_step_started(self, step: int, total_steps: int) -> None: ...

    async def on_train_step_completed(
        self,
        step: int,
        global_step: int,
        train_metrics: dict[str, Any] | None = None,
    ) -> None: ...

    async def on_batch_assembled(
        self,
        step: int,
        trajectory_ids: list[str],
        mean_reward: float | None = None,
    ) -> None: ...

    async def on_eval_started(
        self, eval_step: int, eval_source: str = "sampler"
    ) -> None: ...

    async def on_eval_item_completed(
        self, completed_eval_items: int, total_eval_items: int
    ) -> None: ...

    async def on_eval_completed(self, eval_step: int) -> None: ...


class TrainConfigLike(Protocol):
    """Structural type for the composed ``RunConfig`` the train loop reads.

    The composed config exposes typed sub-configs — ``train``
    (training-cadence knobs), ``rollout`` (a ``RolloutConfig`` whose
    ``make()`` builds the concrete pipeline), and ``eval`` (eval-cadence
    knobs). ``train`` depends only on this structural shape; the
    async-vs-sync pipeline choice is carried by the ``rollout`` subclass
    type, resolved inside ``rollout.make()``.
    """

    train: Any
    rollout: Any
    eval: Any
