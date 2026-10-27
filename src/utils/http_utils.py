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

"""Shared httpx client construction for Cloud Run egress.

Centralizes HTTP/2 + connection-pool settings used to talk to Cloud Run
sampling endpoints and per-recipe tool-executor endpoints.

HTTP/2 matters here because the async loop keeps thousands of sampling
LROs in flight plus a stream of env tool-calls. Over HTTP/1.1 each
in-flight request needs its own TCP connection, which under Cloud NAT
exhausts the per-VM source-port budget and produces spurious connection
drops / TLS-EOF / connect timeouts. HTTP/2 multiplexes many concurrent
streams over a single connection, so ``max_connections`` here caps
sockets (and NAT ports), not logical concurrency.

HTTP/2 is always on: the ``h2`` package is a hard dependency (declared as
``httpx[http2]`` in ``pyproject.toml``), so there is no HTTP/1.1 fallback.
"""

from __future__ import annotations

import httpx

# Default ceiling on simultaneous TCP connections per client. Kept well
# under the VM's Cloud NAT per-VM source-port budget; HTTP/2 multiplexing
# means a small pool sustains very high request concurrency.
DEFAULT_MAX_CONNECTIONS = 128


def build_http_client(
    *,
    max_connections: int = DEFAULT_MAX_CONNECTIONS,
    connect_timeout: float = 10.0,
    read_timeout: float = 60.0,
    write_timeout: float = 10.0,
    pool_timeout: float = 10.0,
    keepalive_expiry: float = 300.0,
) -> httpx.AsyncClient:
    """Build an ``httpx.AsyncClient`` tuned for pooled, multiplexed egress.

    Always negotiates HTTP/2 so many concurrent requests share each
    connection (see module docstring for why that matters under Cloud NAT).

    Args:
      max_connections: Max simultaneous TCP connections (= max NAT source
        ports for this client). Keepalive is pinned to the same value so
        connections are reused rather than churned (churn leaves TIME_WAIT
        sockets holding NAT ports for ~120s).
      read_timeout: Fallback per-request read timeout. Callers doing
        quick status polls should pass a short per-request override;
        long-running server work relies on this client-level value.
      keepalive_expiry: How long an idle pooled connection is kept open
        for reuse.
    """
    limits = httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=max_connections,
        keepalive_expiry=keepalive_expiry,
    )
    timeout = httpx.Timeout(
        connect=connect_timeout,
        read=read_timeout,
        write=write_timeout,
        pool=pool_timeout,
    )
    return httpx.AsyncClient(http2=True, limits=limits, timeout=timeout)
