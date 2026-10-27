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

from typing import Optional
import google.auth
from google.auth import impersonated_credentials
from google.auth.compute_engine import _metadata as gce_metadata
from google.auth.transport.requests import Request as AuthRequest
import logging
import os
import asyncio

logger = logging.getLogger(__name__)

# Per-process cache of the GCE metadata-server probe. None = not yet probed.
_ON_GCE_CACHE: Optional[bool] = None


def _on_gce(timeout: float = 1.0) -> bool:
    """Return True iff the GCE metadata server is reachable.

    Off GCE this fails within ``timeout`` seconds and callers fall back to
    ADC + impersonation. Result is cached per-process.
    """
    global _ON_GCE_CACHE

    if _ON_GCE_CACHE is not None:
        return _ON_GCE_CACHE

    try:
        _ON_GCE_CACHE = bool(gce_metadata.ping(AuthRequest(), timeout=timeout))
    except Exception as e:  # pragma: no cover - defensive
        logger.debug(f"GCE metadata ping raised: {e}")
        _ON_GCE_CACHE = False

    logger.info(
        "Environment detection: %s",
        "on GCE (metadata reachable)" if _ON_GCE_CACHE else "off GCE",
    )
    return _ON_GCE_CACHE


def _resolve_target_service_account(target_service_account: Optional[str]) -> str:
    """Pick the SA to impersonate.

    Resolution order:
      1. explicit ``target_service_account`` arg
      2. ``IMPERSONATION_TARGET_SA`` envvar

    Raises:
      ValueError: if neither is set. Fails fast here with an actionable
        message rather than surfacing later as an opaque
        ``PERMISSION_DENIED`` from the impersonated-credentials refresh.
    """
    sa = (
        target_service_account or os.environ.get("IMPERSONATION_TARGET_SA") or ""
    ).strip()
    if not sa:
        raise ValueError(
            "No target service account resolved. Off-GCE callers must "
            "either pass `target_service_account=` explicitly or set the "
            "`IMPERSONATION_TARGET_SA` envvar to a service account the "
            "caller's ADC identity holds "
            "`roles/iam.serviceAccountTokenCreator` on."
        )
    return sa


def get_call_credentials(
    target_service_account: Optional[str] = None,
    scopes: Optional[list[str]] = None,
):
    """Return google-auth credentials for calling Google APIs (access tokens).

    Companion to ``get_identity_token`` but yields access-token
    credentials (cloud-platform scope by default) suitable for gRPC call
    credentials.

    On GCE the VM's attached-SA ADC credentials are returned unchanged.
    Off GCE (laptop, CI, ...), impersonates the resolved target SA to
    mint a service-account access token; the caller's ADC identity must
    hold ``roles/iam.serviceAccountTokenCreator`` on that SA.
    """
    scopes = scopes or ["https://www.googleapis.com/auth/cloud-platform"]
    source_credentials, _ = google.auth.default(scopes=scopes)
    if _on_gce():
        return source_credentials

    sa = _resolve_target_service_account(target_service_account)
    logger.info("Impersonating service account for call credentials: %s", sa)
    return impersonated_credentials.Credentials(
        source_credentials=source_credentials,
        target_principal=sa,
        target_scopes=scopes,
        lifetime=3600,
    )


async def _get_identity_token_from_metadata(audience: str) -> Optional[str]:
    """Fetch an identity token directly from the GCE metadata server.

    Returns the token on success, or None on any failure so the caller
    can fall back to impersonation.
    """
    try:
        logger.info("Fetching identity token from GCE metadata server...")
        # format=full is a strict superset of the default identity token (it
        # adds the google.compute_engine claim; the standard claims incl.
        # email/aud/exp are unchanged). Request format=full so the standard
        # claim set, including email, is present for services that authorize
        # callers by email.
        token = await asyncio.to_thread(
            gce_metadata.get,
            AuthRequest(),
            f"instance/service-accounts/default/identity?audience={audience}&format=full",
        )
        logger.info("Successfully got identity token from metadata server.")
        return token
    except Exception as e:
        logger.warning(f"Failed to get token from metadata server: {e}")
        return None


async def get_identity_token(
    audience: str,
    target_service_account: Optional[str] = None,
) -> str:
    """Generate a Google-signed identity token for ``audience``.

    On GCE the VM's own attached-SA identity token is used directly
    — UNLESS ``target_service_account`` (or ``IMPERSONATION_TARGET_SA``)
    is set, in which case impersonation is used even on GCE. This
    matters in environments where the metadata server is reachable but
    the VM's default service account lacks the required Cloud Run invoker
    role.

    Off GCE, always uses impersonation via ADC; the caller's ADC
    identity must hold ``roles/iam.serviceAccountTokenCreator`` on the
    target SA.
    """
    # When an explicit SA is requested (directly or via env var), skip
    # the metadata path entirely — the caller wants a specific identity,
    # not the VM's default.
    explicit_sa = (
        target_service_account or os.environ.get("IMPERSONATION_TARGET_SA", "").strip()
    )
    if _on_gce() and not explicit_sa:
        token = await _get_identity_token_from_metadata(audience)
        if token:
            return token
        logger.warning(
            "Metadata-server token fetch failed despite being on GCE; "
            "falling back to impersonation."
        )

    credentials, _ = google.auth.default()
    sa = _resolve_target_service_account(target_service_account)
    logger.info(f"Impersonating service account: {sa}")
    impersonated_creds = impersonated_credentials.Credentials(
        source_credentials=credentials,
        target_principal=sa,
        target_scopes=["https://www.googleapis.com/auth/cloud-platform"],
        lifetime=3600,
    )

    id_token_creds = impersonated_credentials.IDTokenCredentials(
        target_credentials=impersonated_creds,
        target_audience=audience,
        include_email=True,
    )

    try:
        logger.info("Refreshing ID token...")
        await asyncio.to_thread(id_token_creds.refresh, AuthRequest())
        logger.info("Token refresh successful.")
        return id_token_creds.token
    except Exception as e:
        logger.error(f"Failed to refresh ID token: {e}")
        raise
