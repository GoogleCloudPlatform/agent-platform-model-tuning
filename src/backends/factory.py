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

"""Backend factory for sampler/trainer selection.

Dispatches ``config.backend`` to a concrete ``(Sampler, Trainer)`` pair.
Ships pre-registered with ``"vertex_msrl"``; additional backends can
register themselves via ``register_backend`` at import time (via a
side-effect import shim that callers must import before invoking
``build_backend``).

Async because ``_build_vertex_msrl`` must ``CreateTuningJob`` and wait
for ``JOB_STATE_RUNNING`` before it can hand back a usable pair. Sync
builders are also supported; ``build_backend`` awaits only when the
return value is awaitable.

Always use the returned :class:`BackendBundle` in a try/finally so that
``bundle.aclose()`` runs unconditionally — that's what guarantees the
(expensive) Vertex tuning job gets cancelled on exit.

Example
-------

>>> bundle = await build_backend(config)
>>> try:
...     agent = SingleStepAgent(bundle.sampler, env_factory)
...     await train_and_eval_loop(..., trainer=bundle.trainer, ...)
... finally:
...     await bundle.aclose()
"""

from __future__ import annotations

import logging
import re
import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Protocol, Union


from core.protocols import Sampler, Trainer
from utils.experiment_fields import resolve_owner

logger = logging.getLogger(__name__)


# ---- Per-backend config Protocols ---------------------------------------
#
# Structural (`typing.Protocol`) rather than a shared base class so
# call-site Configs don't have to inherit. Only *required* fields are
# declared; optional knobs are read via `getattr(config, name, default)`.


class BackendConfig(Protocol):
    """Minimum contract for `build_backend`: a `backend` selector."""

    backend: str


class VertexMsrlBuildConfig(Protocol):
    """Fields read by `_build_vertex_msrl`."""

    project: str
    vertex_base_model: str
    location: str
    vertex_existing_job_id: str
    vertex_job_running_timeout_seconds: float
    vertex_lr_multiplier: float
    vertex_adapter_size: str
    vertex_checkpoint_interval: int
    # Per-attempt RPC deadline. Retry counts/backoff live in the
    # backend's `_call` / poll loops (deadline-based).
    vertex_request_timeout: float
    # Optional cap (req/min) on this script's outgoing RPCs against the
    # shared per-project `resource_management_requests` quota. Defaults
    # to 1800 when unset; 0 disables the limiter.
    vertex_request_qpm: float
    max_generation_tokens: int


# ---- Backend identifiers ------------------------------------------------

BACKEND_VERTEX_MSRL = "vertex_msrl"


# ---- BackendBundle -------------------------------------------------------


async def _noop_aclose() -> None:
    """Default `aclose` for backends that have no teardown step."""


@dataclass
class BackendBundle:
    """Container for the per-backend objects the train loop needs.

    Attributes:
      sampler: Concrete Sampler ready to ``await sample(...)``.
      trainer: Concrete Trainer ready to ``await train_step(...)``.
      job_id: Short filesystem-safe id for log namespacing (for
        ``vertex_msrl``, the numeric tuning job id).
      aclose: Async no-arg callable that releases all backend-owned
        resources. For a ``vertex_msrl`` job we created (empty
        ``vertex_existing_job_id``), this CANCELS the tuning job.
    """

    sampler: Sampler
    trainer: Trainer
    job_id: str
    aclose: Callable[[], Awaitable[None]] = field(default=_noop_aclose)


# ---- Backend registry ---------------------------------------------------
# Backends register themselves via ``register_backend(name, builder)``.
# Sync or async builders both work; ``build_backend`` awaits the return
# when it's a coroutine.
BackendBuilder = Callable[[Any], Union[BackendBundle, Awaitable[BackendBundle]]]

_BACKEND_REGISTRY: Dict[str, BackendBuilder] = {}


def register_backend(name: str, builder: BackendBuilder) -> None:
    """Register a backend builder under a dispatch name.

    Idempotent for the same ``(name, builder)`` pair; raises
    ``ValueError`` if a different builder is already registered under
    ``name`` (identities must not silently drift).
    """
    existing = _BACKEND_REGISTRY.get(name)
    if existing is not None and existing is not builder:
        raise ValueError(
            f"Backend {name!r} already registered to a different builder "
            f"({existing!r}); cannot re-register to {builder!r}."
        )
    _BACKEND_REGISTRY[name] = builder


def known_backends() -> frozenset[str]:
    """Return the set of currently-registered backend names."""
    return frozenset(_BACKEND_REGISTRY)


# ---- Helpers -------------------------------------------------------------


_SAFE_PATH_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_job_id(s: str) -> str:
    """Best-effort sanitiser for log subdir names ('/' -> '_' etc.)."""
    s = _SAFE_PATH_RE.sub("_", s).strip("_")
    return s or "unknown"


def _require_nonempty(value: object, name: str) -> None:
    """Raise `ValueError` if `value` is `None` or empty string."""
    if value is None or value == "":
        raise ValueError(f"config.{name} is required for this backend")


# ---- vertex_msrl builder -----------------------------------------------


async def _build_vertex_msrl(config: VertexMsrlBuildConfig) -> BackendBundle:
    """Build the Vertex MSRL sampler/trainer pair.

    If ``config.vertex_existing_job_id`` is set, reuse that job (no
    create, no cancel-on-exit). Otherwise call ``CreateTuningJob`` and
    poll ``GetTuningJob`` until ``JOB_STATE_RUNNING`` (bounded by
    ``vertex_job_running_timeout_seconds``); the returned ``aclose`` will
    cancel this job on exit.

    Ownership: ``aclose`` cancels iff *we* created the job in this call.
    A pre-existing job may still be in use by other drivers, so we never
    cancel it — a Vertex MSRL cancel is irreversible and re-creating one
    is a ~30 min paid operation.
    """
    from backends.vertex_msrl_rest import (
        VertexMsrlRestClient as VertexMsrlClient,
        VertexMsrlRestSampler as VertexMsrlSampler,
        VertexMsrlRestTrainer as VertexMsrlTrainer,
        default_endpoint,
    )

    if int(getattr(config, "resume_from_train_step", 0)) != 0:
        raise ValueError(
            "vertex_msrl backend does not support resume_from_train_step != 0; "
            "Vertex MSRL tuning jobs always start training from step 0."
        )

    project = config.project
    base_model = config.vertex_base_model
    location = config.location
    _require_nonempty(project, "project")
    _require_nonempty(base_model, "vertex_base_model")
    _require_nonempty(location, "location")

    endpoint = default_endpoint(location)
    logger.info(
        "[vertex_msrl] Resolved endpoint %s (location=%s).",
        endpoint,
        location,
    )

    existing_job_id = str(config.vertex_existing_job_id or "")
    max_wait = float(config.vertex_job_running_timeout_seconds)
    request_timeout = float(config.vertex_request_timeout)

    # `0.0` / `""` / `0` in Config means "use server default".
    lr_multiplier = float(config.vertex_lr_multiplier) or None
    adapter_size = config.vertex_adapter_size or None
    ckpt_interval = int(config.vertex_checkpoint_interval) or None

    # Single client shared by sampler and trainer so they share the
    # transport channel, service stubs, and ADC credentials.
    # `vertex_request_qpm` (default 1800; 0 disables) caps this
    # script's share of the per-project
    # `resource_management_requests` quota (matters when several jobs
    # share a project).
    request_qpm = getattr(config, "vertex_request_qpm", 1800.0)
    quota_qpm = float(request_qpm) if request_qpm and float(request_qpm) > 0 else None

    client = VertexMsrlClient(
        project=str(project),
        location=location,
        endpoint=endpoint,
        tuning_job_id=existing_job_id,
        request_timeout=request_timeout,
        quota_qpm=quota_qpm,
    )

    we_created_the_job = False
    if existing_job_id:
        logger.info(
            "[vertex_msrl] Reusing existing tuning job: %s (id=%s);"
            " will NOT cancel on exit.",
            client.tuning_job_name,
            existing_job_id,
        )
    else:
        # Author attribution: the same human the experiment doc names as
        # owner (see resolve_owner). Empty -> omitted.
        author = resolve_owner()
        description = f"msrl client backend (project={project}"
        if author:
            description += f", author={author}"
        description += ")"
        logger.info(
            "[vertex_msrl] Creating new tuning job (project=%s, base_model=%s,"
            " description=%r)",
            project,
            base_model,
            description,
        )
        await client.create_tuning_job(
            base_model=base_model,
            description=description,
            learning_rate_multiplier=lr_multiplier,
            adapter_size=adapter_size,
            checkpoint_interval=ckpt_interval,
        )
        we_created_the_job = True
        # Log the full resource name + id as soon as the job exists so it can
        # be located in the Vertex UI / reused (vertex_existing_job_id=<id>)
        # WITHOUT waiting for it to reach RUNNING.
        logger.info(
            "[vertex_msrl] Created tuning job: %s (id=%s)",
            client.tuning_job_name,
            client.tuning_job_id,
        )

    logger.info(
        "[vertex_msrl] Waiting up to %.0fs for job %s (id=%s) to reach"
        " JOB_STATE_RUNNING...",
        max_wait,
        client.tuning_job_name,
        client.tuning_job_id,
    )
    try:
        await client.wait_for_job_running(timeout=max_wait)
    except BaseException:
        # Catch BaseException so KeyboardInterrupt / SystemExit
        # (from Ctrl-C or SIGTERM) also trigger cleanup.
        if we_created_the_job:
            logger.warning(
                "[vertex_msrl] wait_for_job_running failed for job %s"
                " (we created it); cancelling to avoid orphaned job.",
                client.tuning_job_id,
            )
            try:
                await client.cancel_tuning_job()
            except Exception:
                logger.exception(
                    "[vertex_msrl] Failed to cancel job %s after wait failure.",
                    client.tuning_job_id,
                )
            finally:
                await client.aclose()
        raise
    logger.info(
        "[vertex_msrl] Tuning job %s is RUNNING. Building sampler/trainer.",
        client.tuning_job_id,
    )

    # Sampler and trainer take the default LRO retry profile (10h poll
    # cap, rejection-only create retry). This legacy name-keyed builder
    # has no config field to override it; `VertexMsrlBackendConfig` does.
    sampler = VertexMsrlSampler(client)
    trainer = VertexMsrlTrainer(
        client,
        learning_rate_multiplier=lr_multiplier,
    )

    async def _aclose() -> None:
        # Only cancel jobs we created ourselves (see docstring).
        try:
            if we_created_the_job:
                logger.info(
                    "[vertex_msrl] aclose: cancelling tuning job %s (we created it).",
                    client.tuning_job_id,
                )
                await client.cancel_tuning_job()
            else:
                logger.info(
                    "[vertex_msrl] aclose: leaving job %s alone (pre-existing).",
                    client.tuning_job_id,
                )
        finally:
            # Sampler/Trainer share the client's channel; one close covers both.
            await client.aclose()

    return BackendBundle(
        sampler=sampler,
        trainer=trainer,
        job_id=str(client.tuning_job_id),
        aclose=_aclose,
    )


# ---- Public API ----------------------------------------------------------


async def build_backend(config: BackendConfig) -> BackendBundle:
    """Async dispatch on ``config.backend`` via the backend registry.

    Raises ``ValueError`` for unregistered names. Sync and async builders
    are both supported: awaitable returns are awaited transparently.
    Always pair the call with ``try / finally: await bundle.aclose()``
    so backend resources are released on any exit path.
    """
    backend = str(getattr(config, "backend", "") or "").strip().lower()
    builder = _BACKEND_REGISTRY.get(backend)
    if builder is None:
        raise ValueError(
            f"Unknown backend {backend!r}. Supported: {sorted(_BACKEND_REGISTRY)}."
        )
    result = builder(config)
    if inspect.isawaitable(result):
        return await result
    return result


# ---- Built-in registrations --------------------------------------------
# Additional backends register themselves at import time via their own shims.
register_backend(BACKEND_VERTEX_MSRL, _build_vertex_msrl)


__all__ = [
    "BACKEND_VERTEX_MSRL",
    "BackendBuilder",
    "BackendBundle",
    "BackendConfig",
    "build_backend",
    "known_backends",
    "register_backend",
]
