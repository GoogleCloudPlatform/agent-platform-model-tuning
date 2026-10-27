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

"""Isolation flags for mini-swe-agent's DockerEnvironment.

Passed verbatim as ``run_args`` to ``DockerEnvironment(...)``. Each flag
protects a specific class of "agent messes up my host" failure.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Every flag matters -- understand what it protects before removing one.
SANDBOX_RUN_ARGS: list[str] = [
    # Auto-delete the container on exit.
    "--rm",
    # No network at all: blocks API calls, git push, curl exfil, DNS.
    "--network=none",
    # Run as uid 65534 (nobody), NOT root.
    "--user=65534:65534",
    # Immutable root filesystem.
    "--read-only",
    # Writable scratch mounts (ephemeral, capped at 100 MiB).
    #
    # ``mode=1777`` is one HALF of the writability contract; the other
    # half is the mode-1777 mountpoint created in the Dockerfile.
    # Both are required: docker's tmpfs ``mode=`` option does NOT
    # override the underlying directory's permissions — the mount
    # inherits whatever mode the mountpoint had in the image. See
    # the extended comment on the ``install -d -m 1777`` line in the
    # Dockerfile for the empirical evidence.
    #
    # If either half is missing, the tmpfs comes up as
    # ``drwxr-xr-x root root`` at runtime and uid=nobody (--user=65534
    # below) cannot write, contradicting the agent prompt's "Files
    # in /workspace/ persist across your steps" contract.
    #
    # ``noexec`` is deliberately omitted so ``python3 /workspace/foo.py``
    # works.
    "--tmpfs=/workspace:size=100m,mode=1777",
    "--tmpfs=/tmp:size=100m,mode=1777",
    # CPU/RAM caps against runaway compute.
    "--cpus=1",
    "--memory=512m",
    # Blocks fork bombs; a normal task uses <10.
    "--pids-limit=64",
    # Drop all linux capabilities (belt-and-suspenders with --user=nobody).
    "--cap-drop=ALL",
    # Refuse setuid-based privilege escalation.
    "--security-opt=no-new-privileges",
]

# Shared by the runner and the isolation smoke tests.
SANDBOX_IMAGE = "word-puzzle-agent-sandbox:latest"

# Dockerfile lives alongside this module.
_DOCKERFILE_DIR = Path(__file__).resolve().parent


def ensure_sandbox_image(
    image: str = SANDBOX_IMAGE,
    dockerfile_dir: Path | None = None,
) -> None:
    """Build the sandbox Docker image if it does not already exist.

    Safe to call multiple times; a no-op when the image is present.
    Raises ``subprocess.CalledProcessError`` if the build fails.
    """
    try:
        subprocess.run(
            ["docker", "image", "inspect", image],
            check=True,
            capture_output=True,
        )
        logger.info("Sandbox image %s already exists.", image)
        return
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    build_dir = dockerfile_dir or _DOCKERFILE_DIR
    logger.info("Sandbox image %s not found — building from %s ...", image, build_dir)
    subprocess.run(
        ["docker", "build", "-t", image, str(build_dir)],
        check=True,
    )
