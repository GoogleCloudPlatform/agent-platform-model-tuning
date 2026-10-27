#!/bin/bash
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
# Generic container-isolation verifier for any training/eval launch that runs
# an agent inside a Docker sandbox.
#
# It runs a fixed battery of "the agent can't touch the host / phone home"
# checks against a given (run_args, image) pair -- the SAME pair a recipe
# passes to its sandbox environment. Run it before shipping any change to a
# sandbox Dockerfile or its run-args, and on a fresh VM before a real training
# run, so the isolation guarantee is proven on that specific host.
#
# The checks are recipe-agnostic: they only assume a Debian-ish base image with
# bash + python3 + coreutils (base64). No closed-source or recipe-specific
# assumptions.
#
# Inputs (two ways to supply the sandbox definition):
#
#   1. From a Python module that exports SANDBOX_RUN_ARGS (list[str]) and
#      SANDBOX_IMAGE (str) -- the single-source-of-truth pattern:
#
#        tools/verify_sandbox_isolation.sh \
#          --from-module example_recipe.word_puzzle.agent_solver.minisweagent.docker_run_args
#
#      (PYTHONPATH must include the module's root; default is `src`.)
#
#   2. Explicitly:
#
#        tools/verify_sandbox_isolation.sh \
#          --run-args '--rm --network=none --user=65534:65534 --read-only ...' \
#          --image my-sandbox:latest
#
# Exit: 0 if all checks pass, 1 if any fail, 2 on bad usage.

set -u

usage() {
  sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-2}"
}
# ---- resolve the (run_args, image) pair from flags -------------------------
FROM_MODULE=""
RUN_ARGS=""
IMAGE=""
PYTHONPATH_ROOT="${PYTHONPATH_ROOT:-src}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from-module) FROM_MODULE="$2"; shift 2 ;;
    --run-args)    RUN_ARGS="$2";    shift 2 ;;
    --image)       IMAGE="$2";       shift 2 ;;
    --pythonpath)  PYTHONPATH_ROOT="$2"; shift 2 ;;
    -h|--help)     usage 0 ;;
    *) echo "unknown arg: $1" >&2; usage 2 ;;
  esac
done

if [[ -n "$FROM_MODULE" ]]; then
  # Load the single-source-of-truth args/image from the module. `uv run` so
  # this works against the project venv without manual activation.
  #
  # NOTE: importing the module can trigger third-party import-time banners on
  # STDOUT (e.g. mini-swe-agent prints a version notice), so we can't rely on
  # line position. Each real value is tagged with a unique sentinel and grepped
  # back out, immune to any surrounding banner noise.
  RESOLVED=$(
    PYTHONPATH="$PYTHONPATH_ROOT" uv run --no-project python -c "
import importlib, shlex
m = importlib.import_module('$FROM_MODULE')
args = getattr(m, 'SANDBOX_RUN_ARGS')
img  = getattr(m, 'SANDBOX_IMAGE')
print('__ISOARGS__ ' + ' '.join(shlex.quote(a) for a in args))
print('__ISOIMG__ '  + img)
" 2>/dev/null
  )
  RUN_ARGS="$(printf '%s\n' "$RESOLVED" | sed -n 's/^__ISOARGS__ //p' | head -1)"
  IMAGE="$(printf '%s\n'   "$RESOLVED" | sed -n 's/^__ISOIMG__ //p'  | head -1)"
  if [[ -z "$RUN_ARGS" || -z "$IMAGE" ]]; then
    echo "FAIL: could not load SANDBOX_RUN_ARGS / SANDBOX_IMAGE from '$FROM_MODULE'" >&2
    echo "      (PYTHONPATH root = '$PYTHONPATH_ROOT'; does the module export both?)" >&2
    exit 1
  fi
fi

if [[ -z "$RUN_ARGS" || -z "$IMAGE" ]]; then
  echo "ERROR: supply either --from-module MOD, or both --run-args '...' and --image IMG." >&2
  usage 2
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker not found on PATH." >&2
  exit 2
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "ERROR: image '$IMAGE' not present locally. Build it first." >&2
  exit 2
fi
# ---- checks -----------------------------------------------------------------
fails=0

# Run a command inside the sandbox and assert its combined output contains an
# expected substring.  $RUN_ARGS is intentionally unquoted so each flag becomes
# a separate argv entry.
run_check() {
  local name="$1" expect="$2" cmd="$3"
  echo "-- $name"
  local actual
  actual=$(docker run $RUN_ARGS "$IMAGE" bash -c "$cmd" 2>&1)
  if [[ "$actual" == *"$expect"* ]]; then
    echo "   PASS"
  else
    echo "   FAIL: expected substring $(printf %q "$expect")"
    echo "   got:      $(echo "$actual" | head -5)"
    fails=$((fails+1))
  fi
}

echo "=== sandbox isolation verification ==="
echo "  image    : $IMAGE"
echo "  run_args : $RUN_ARGS"
echo ""

# 1. No network egress (the load-bearing "can't phone home / exfil" guarantee).
#    With --network=none, DNS resolution itself fails.
run_check "network egress blocked" \
    "Temporary failure in name resolution" \
    'python3 -c "import socket; socket.create_connection((\"google.com\", 80), timeout=3)" 2>&1'

# 2. Runs as an unprivileged user, not root.
run_check "runs as non-root" \
    "uid=" \
    'id'
echo "-- (verify above uid is NOT uid=0/root)"
ROOTCHK=$(docker run $RUN_ARGS "$IMAGE" bash -c 'id -u' 2>&1)
if [[ "$ROOTCHK" == "0" ]]; then
  echo "   FAIL: container runs as root (uid=0)"; fails=$((fails+1))
else
  echo "   PASS: uid=$ROOTCHK (non-root)"
fi

# 3. Root filesystem is read-only (agent can't tamper with the image).
run_check "root filesystem read-only" \
    "Read-only file system" \
    'touch /etc/isolation_probe 2>&1'

# 4. No host credentials leaked into the container.
run_check "no gcloud creds mounted" \
    "No such file" \
    'ls ~/.config/gcloud/ 2>&1'
run_check "no ssh keys mounted" \
    "No such file" \
    'ls ~/.ssh/ 2>&1'

# 5. No exfil/push binaries present in the image.
run_check "no network/push binaries" \
    "absent" \
    'for t in gcloud git ssh curl wget gsutil; do command -v $t >/dev/null 2>&1 && echo "$t: PRESENT" || echo "$t: absent"; done'

echo ""
if [[ $fails -eq 0 ]]; then
  echo "=== ALL ISOLATION CHECKS PASSED ($IMAGE) ==="
  exit 0
else
  echo "=== $fails ISOLATION CHECK(S) FAILED ($IMAGE) ==="
  exit 1
fi
