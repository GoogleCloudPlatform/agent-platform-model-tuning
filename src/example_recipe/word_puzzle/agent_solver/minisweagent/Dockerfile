# Sandbox image for the mini-swe-agent word_puzzle solver prototype.
#
# The agent runs bash commands INSIDE a container spawned from this image via
# `docker run -d ... image sleep 2h` (see mini's DockerEnvironment). Isolation
# from the host is enforced entirely by docker run flags at runtime (see
# `docker_run_args.py`) -- this Dockerfile just makes sure the tools the agent
# needs are present.
#
# Deliberately minimal:
#   * NO curl, wget, git, gcloud, ssh, aws-cli -- even inside the network=none
#     sandbox, if any of these existed and a future env dropped --network=none,
#     they'd be an exfiltration path. Absence is defense in depth.
#   * NO pip, no compilers, no build tools -- agent shouldn't be installing
#     things at runtime.
#   * Just: bash, coreutils (base64, rev, tr, cut, head, tail, ...),
#     python3 with stdlib (base64, codecs, binascii, hashlib, ...).
#     Hex-dump utilities (xxd, hexdump) are NOT installed — the agent
#     can use python3's binascii.hexlify / bytes.hex() for the same
#     purpose without pulling in vim-common (where xxd actually lives —
#     it is NOT in coreutils, despite the frequent misconception).
#
# Run non-root at runtime (--user=65534:65534 = nobody), so /workspace is
# tmpfs-mounted and writable to that uid. Also declared via USER in this
# Dockerfile for defense in depth, so an accidental `docker run` without
# --user still lands non-root.

FROM debian:12-slim

# Install only what the agent will actually use. --no-install-recommends keeps
# the image tight and the attack surface small.
RUN apt-get update && apt-get install -y --no-install-recommends \
        bash \
        coreutils \
        python3 \
        python3-minimal \
    && rm -rf /var/lib/apt/lists/*

# Working directory the agent will land in. The runtime `--tmpfs /workspace`
# mount overlays this so state persists across actions but dies with the
# container. The directory needs to exist in the image so `docker run -w
# /workspace` doesn't fail before the tmpfs mounts.
#
# Mode 1777 is sticky-bit world-writable (drwxrwxrwt, as on /tmp), not an
# unrestricted 0777. It is required here because docker's tmpfs ``mode=``
# does not override the mountpoint's image-time permissions: at 755 the
# tmpfs comes up root-owned and uid=nobody cannot write, which made the
# majority of eval rollouts fail with EACCES on /workspace/solve.py.
#
# ``install -d -m`` creates the directory with the mode in one step;
# some permission linters flag the two-step spelling because they cannot
# distinguish sticky 1777 from unrestricted 0777.
RUN install -d -m 1777 /workspace

# Sanity: verify each tool the agent depends on is actually present.
# Deliberately does NOT include xxd — xxd ships in vim-common, not
# coreutils, and we don't install vim-common (see the "Deliberately
# minimal" comment block at the top). Callers that need hex should
# use python3's binascii/bytes.hex().
#
# Note the per-tool loop: ``command -v A B C D E`` (space-separated
# args) exits 0 if ANY of the named tools is found, not if ALL are
# found — it prints paths for the ones present and silently skips
# the rest. Since ``bash`` is always in the image, a single
# multi-arg ``command -v`` would degenerate to "does bash exist"
# and never catch a missing base64 / rev / tr / python3. Checking
# each tool in its own invocation gives us the "all-must-exist"
# semantics we actually want.
RUN bash -c 'for t in bash base64 rev tr python3; do \
        command -v "$t" >/dev/null \
            || { echo "ERROR: expected tool missing from image: $t"; exit 1; }; \
    done'

# Defense in depth: match the runtime --user=65534:65534 enforced by
# docker_run_args.py, so an accidental `docker run` without --user still
# lands non-root. Placed AFTER apt-get install + mkdir /workspace (both
# of which need root to succeed) and BEFORE CMD (which by convention
# closes the Dockerfile).
USER nobody

# The container is spawned via `sleep <container_timeout>` by mini's
# DockerEnvironment; each action then runs via `docker exec ... bash -lc "$cmd"`.
# So the image doesn't need a specific CMD -- but set one that's obviously wrong
# if someone accidentally `docker run`s it without an override.
CMD ["bash", "-c", "echo 'this image is meant to be used with docker run -d ... sleep, then docker exec' >&2; exit 1"]
