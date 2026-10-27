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

"""BackendConfig: abstract base for a backend builder.

A concrete subclass IS the backend selector: its ``make()`` builds the sampler +
trainer and returns a ``BackendBundle``. This replaces the older
string-``backend=`` field + registry dispatch.

Only the ABC lives here for now; concrete per-backend subclasses (with their
lifted ``make()`` bodies) land in later tasks; alternate-backend subclasses
extend this generic base.
"""

from __future__ import annotations

import abc
import logging
import os
from dataclasses import dataclass, field

from typing import Any

from backends.factory import BackendBundle

# Re-exported: the retry vocabulary is part of the backend-config surface,
# so a recipe overriding a policy imports it from here.
from backends.retry_profile import (  # noqa: F401
    Backoff,
    LroRetryProfile,
    SyncRetryProfile,
)
from core.protocols import Trainer
from utils.experiment_fields import resolve_owner

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackendConfig(abc.ABC):
    """Abstract base for backend builders.

    ``project`` and ``location`` identify the GCP project and region the
    backend operates in.  Every concrete backend needs them (Vertex MSRL
    for the tuning job, Gemini API for the generateContent URL), so they
    live on the ABC rather than being
    duplicated (with different names) on each subclass.  Subclasses may
    override the defaults (e.g. ``VertexMsrlBackendConfig`` defaults
    ``location`` to ``"us-central1"``).
    """

    project: str = ""
    location: str = ""
    request_timeout: float = 600.0
    service_account: str = ""
    captures_dir: str = ""

    @property
    def short_name(self) -> str:
        """Human-readable short name for this backend (e.g. ``"vertex_msrl"``,
        ``"gemini_api"``). Used in monitoring dashboards and log messages."""
        return "unknown"

    @abc.abstractmethod
    async def make(self) -> BackendBundle: ...


def _require_nonempty(value: object, name: str) -> None:
    """Raise ``ValueError`` if ``value`` is ``None`` or empty string."""
    if value is None or value == "":
        raise ValueError(f"{name} is required for this backend")


def resolve_job_author() -> str:
    """Return who a newly created tuning job should be attributed to.

    The same human the experiment doc names as ``owner``: see
    ``utils.experiment_fields.resolve_owner`` (``VMSRL_OWNER`` >
    ``VERTEX_JOB_AUTHOR`` > login user). Returns ``""`` when nothing
    resolves, so callers can omit the attribution rather than stamping a
    placeholder.
    """
    return resolve_owner()


@dataclass(frozen=True)
class VertexMsrlBackendConfig(BackendConfig):
    """Concrete backend config that builds the Vertex MSRL sampler/trainer.

    A concrete subclass IS the backend selector: ``make()`` builds a
    ``VertexMsrlRestClient`` (creating a tuning job when
    ``vertex_existing_job_id`` is empty, else reusing one), waits for it to
    reach ``JOB_STATE_RUNNING``, and returns a ``BackendBundle`` whose
    ``aclose`` cancels the job iff *we* created it.

    Per-call generation length (``max_generation_tokens``) is no longer a
    backend field; it flows through ``sample(max_generation_tokens=...)``.
    The sampler is therefore built with its own constructor default.
    """

    location: str = "us-central1"  # override base default for Vertex

    @property
    def short_name(self) -> str:
        return "vertex_msrl"

    vertex_base_model: str = "gemini-3.5-flash"
    vertex_existing_job_id: str = ""
    # Timeout (seconds) for how long ``make()`` waits for the tuning job to
    # reach JOB_STATE_RUNNING before giving up. NOT a cap on run duration.
    vertex_job_running_timeout_seconds: float = 7200
    vertex_checkpoint_interval: int = 0
    vertex_request_qpm: float = 1800.0
    # Retry policy for the LRO shape: the create budget, the poll budget
    # and the poll schedule, previously private constants in the sampler
    # module. Defaults are unchanged (2h / 10h / 10->90s);
    # ``poll_deadline_s`` is the stop every rollout tier above is matched
    # to, so a recipe deadline shorter than it truncates the backend's retry.
    retry: LroRetryProfile = field(default_factory=LroRetryProfile)

    def _get_endpoint(self) -> str:
        """Return the API endpoint hostname. Subclasses override for non-prod."""
        from backends.vertex_msrl_rest import default_endpoint

        return default_endpoint(self.location)

    def _tuned_model_display_name(self) -> str | None:
        """Display name to give a tuning job we create, or ``None``.

        ``None`` leaves the field unset, and the server then generates
        ``<base-model>-<uuid>`` -- fine for a project running one job, but
        indistinguishable when several share a project. Subclasses override
        to stamp something identifying.
        """
        return None

    async def make(self) -> BackendBundle:
        from backends.vertex_msrl_rest import (
            VertexMsrlRestClient as VertexMsrlClient,
            VertexMsrlRestSampler as VertexMsrlSampler,
            VertexMsrlRestTrainer as VertexMsrlTrainer,
        )

        project = self.project
        base_model = self.vertex_base_model
        location = self.location
        _require_nonempty(project, "project")
        _require_nonempty(base_model, "vertex_base_model")
        _require_nonempty(location, "location")

        endpoint = self._get_endpoint()
        logger.info(
            "[vertex_msrl] Resolved endpoint %s (location=%s).",
            endpoint,
            location,
        )

        existing_job_id = str(self.vertex_existing_job_id or "")
        max_wait = float(self.vertex_job_running_timeout_seconds)
        request_timeout = float(self.request_timeout)

        # `0` means "use server default".
        ckpt_interval = int(self.vertex_checkpoint_interval) or None

        # Single client shared by sampler and trainer so they share the
        # transport channel, service stubs, and ADC credentials.
        # `vertex_request_qpm` (default 1800; 0 disables) caps this
        # script's share of the per-project
        # `resource_management_requests` quota (matters when several jobs
        # share a project).
        request_qpm = self.vertex_request_qpm
        quota_qpm = (
            float(request_qpm) if request_qpm and float(request_qpm) > 0 else None
        )

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
            # Author attribution: see resolve_job_author(). Empty -> omitted.
            author = resolve_job_author()
            dataset = os.environ.get("VERTEX_JOB_DATASET", "").strip()
            description = f"msrl client backend (project={project}"
            if author:
                description += f", author={author}"
            if dataset:
                description += f", dataset={dataset}"
            description += ")"
            display_name = self._tuned_model_display_name()
            logger.info(
                "[vertex_msrl] Creating new tuning job (project=%s,"
                " base_model=%s, description=%r, display_name=%r)",
                project,
                base_model,
                description,
                display_name,
            )
            # Best-effort pre-create congestion signal: on a shared project a
            # new job may queue in PENDING behind existing RUNNING/PENDING
            # jobs. Never let a listing failure block the launch.
            try:
                counts = await client.count_tuning_jobs_by_state()
                running = counts.get("JOB_STATE_RUNNING", 0)
                pending = counts.get("JOB_STATE_PENDING", 0)
                summary = (
                    ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "(none)"
                )
                logger.info(
                    "[vertex_msrl] Existing tuning jobs under %s: %s "
                    "(RUNNING=%d, PENDING=%d%s)",
                    client.parent,
                    summary,
                    running,
                    pending,
                    (
                        "; new job may queue behind these"
                        if (running or pending)
                        else ""
                    ),
                )
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(
                    "[vertex_msrl] Could not list existing tuning jobs "
                    "(pre-create congestion check skipped): %s",
                    exc,
                )
            await client.create_tuning_job(
                base_model=base_model,
                description=description,
                tuned_model_display_name=display_name,
                checkpoint_interval=ckpt_interval,
            )
            we_created_the_job = True
            # Log the full resource name + id as soon as the job exists so it
            # can be located in the Vertex UI / reused
            # (vertex_existing_job_id=<id>) WITHOUT waiting for RUNNING.
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

        # The 10h per-sample LRO poll cap (and the rest of the retry
        # policy) comes from `self.retry`: it keeps GetOperation QPS sane
        # under high `max_concurrent_rollouts` even when the Operations
        # endpoint is 429-throttled. The trainer shares the same profile.
        #
        # max_generation_tokens is intentionally NOT passed: it is a
        # per-call sample() param now, so the sampler's own constructor
        # default stands here.
        sampler = VertexMsrlSampler(client, retry=self.retry)
        trainer = VertexMsrlTrainer(client, retry=self.retry)

        async def _aclose() -> None:
            # Only cancel jobs we created ourselves (see docstring).
            try:
                if we_created_the_job:
                    logger.info(
                        "[vertex_msrl] aclose: cancelling tuning job %s (we"
                        " created it).",
                        client.tuning_job_id,
                    )
                    await client.cancel_tuning_job()
                else:
                    logger.info(
                        "[vertex_msrl] aclose: leaving job %s alone (pre-existing).",
                        client.tuning_job_id,
                    )
            finally:
                # Sampler/Trainer share the client's channel; one close
                # covers both.
                await client.aclose()

        return BackendBundle(
            sampler=sampler,
            trainer=trainer,
            job_id=str(client.tuning_job_id),
            aclose=_aclose,
        )


class _NoTrainer(Trainer):
    """Trainer that refuses to run for a base-model / eval-only backend.

    ``BackendBundle.trainer`` is a required field, but a base-model backend
    that only calls ``generateContent`` has nothing to train: there is no
    tuning job and no weights to update. Rather than hand back a silently
    no-op trainer (which would let a mis-wired training loop run to a false
    "success"), ``train_step`` raises so the misuse is loud and immediate.
    """

    async def train_step(
        self,
        trajectories: list[Any],
        sync_trainer_and_sampler: bool = True,
    ) -> dict[str, Any]:
        raise RuntimeError(
            "This backend is eval-only (base model / deployed endpoint) and "
            "cannot train: it drives generateContent directly with no tuning "
            "job. Use a training-capable backend if you need train_step()."
        )


@dataclass(frozen=True)
class GeminiApiBackendConfig(BackendConfig):
    """Concrete backend config for direct generateContent evaluation.

    A concrete subclass IS the backend selector: ``make()`` builds a
    :class:`~backends.gemini_api_endpoint.GeminiApiSampler` that calls the
    ordinary synchronous ``generateContent`` API against either a publisher
    (base) model or a deployed endpoint. This is an eval-only backend: it
    has a sampler but no trainer and no tuning job, so the returned
    ``BackendBundle`` pairs the sampler with a trainer that refuses to run.

    Per-call generation length (``max_generation_tokens``) is not a backend
    field; it flows through ``sample(max_generation_tokens=...)``. The
    sampler is therefore built with its own constructor default.
    """

    @property
    def short_name(self) -> str:
        return "gemini_api"

    model: str = "gemini-3.5-flash"
    project: str = ""
    # Selects the data realm: ``env`` identifies the API realm.
    # Combined with ``location`` to resolve the API hostname via the
    # sampler's endpoint registry.
    env: str = "prod"
    # Region for the ``locations/{location}`` URL path segment AND for
    # host resolution. ``env`` and ``location`` together determine the
    # API hostname (e.g. prod+global → ``aiplatform.googleapis.com``,
    # prod+us-central1 → ``us-central1-aiplatform.googleapis.com``).
    location: str = "global"
    # Credential source for the Bearer token on every generateContent call:
    #   * True  -> use end-user Application Default Credentials directly
    #              (google.auth.default()), with NO service-account
    #              impersonation, so the request is attributed to the caller's
    #              own identity (e.g. a dev's `gcloud auth application-default
    #              login` account). ``service_account`` is ignored. Only works
    #              against production endpoints (other realms reject end-user
    #              creds).
    #   * False -> (default) delegate to the shared auth helper: on GCE, use
    #              the VM's attached service-account creds unchanged; off GCE,
    #              impersonate a target SA (from ``service_account``, else the
    #              IMPERSONATION_TARGET_SA env var -- required, else it errors),
    #              which the caller's ADC identity must hold
    #              roles/iam.serviceAccountTokenCreator on.
    # (``service_account`` is thus consulted ONLY when False AND off GCE.)
    use_user_credentials: bool = False
    api_version: str = "v1beta1"
    # Retry policy for the synchronous shape. Replaces the bare
    # ``max_tries`` int, which is now ``retry.max_attempts`` and still
    # defaults to 10 on the same 2->30s schedule; the profile additionally
    # exposes an (opt-in, off by default) wall-clock ``deadline_s``.
    retry: SyncRetryProfile = field(default_factory=SyncRetryProfile)
    # "" -> target the publisher (base) model; else a deployed endpoint id.
    endpoint_id: str = ""

    async def make(self) -> BackendBundle:
        _require_nonempty(self.project, "project")
        _require_nonempty(self.model, "model")

        from backends.gemini_api_endpoint import GeminiApiSampler

        # max_generation_tokens is intentionally NOT passed: it is a
        # per-call sample() param now, so the sampler's own constructor
        # default stands here.
        sampler = GeminiApiSampler(
            model=self.model,
            project=self.project,
            env=self.env,
            location=self.location,
            timeout=float(self.request_timeout),
            service_account=self.service_account or None,
            use_user_credentials=self.use_user_credentials,
            api_version=self.api_version,
            retry=self.retry,
            endpoint_id=self.endpoint_id or None,
        )

        # Endpoint id if targeting a deployed endpoint, else the model name;
        # used only for log namespacing.
        job_id = self.endpoint_id or self.model

        async def _aclose() -> None:
            await sampler.aclose()

        return BackendBundle(
            sampler=sampler,
            trainer=_NoTrainer(),
            job_id=job_id,
            aclose=_aclose,
        )


# The Claude (Anthropic-on-Vertex) eval-only backend config is an
# alternate-backend subclass, defined elsewhere, that subclasses this
# generic BackendConfig ABC (see the module docstring above).
