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
# Isolation smoke test for the word-puzzle-agent-sandbox image.
#
# Thin wrapper around the generic, recipe-agnostic verifier at
# tools/verify_sandbox_isolation.sh -- it just points that verifier at THIS
# recipe's sandbox definition (SANDBOX_RUN_ARGS / SANDBOX_IMAGE in
# docker_run_args.py, the single source of truth).
#
# Run before shipping any change to the Dockerfile or docker_run_args.py --
# these are the load-bearing "agent can't mess up my host" guarantees.
#
# Usage:  ./isolation_smoke_test.sh
# Exit:   0 if all checks pass, 1 if any fail, 2 on setup error.
set -u

# Repo root is 5 levels up from this recipe dir
# (src/example_recipe/word_puzzle/agent_solver/minisweagent -> repo root).
REPO_ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"

# 1) Generic isolation battery (network/root/read-only/creds/binaries).
"$REPO_ROOT/tools/verify_sandbox_isolation.sh" \
  --from-module example_recipe.word_puzzle.agent_solver.minisweagent.docker_run_args \
  --pythonpath "$REPO_ROOT/src"
iso_rc=$?

# 2) Recipe-specific FUNCTIONAL checks: this image's /workspace must be usable
#    by the agent (writable tmpfs, python3 decoders, state persisting across
#    the mini-swe-agent DockerEnvironment's separate `docker exec` calls).
#    These verify the image is USABLE, complementing the isolation checks above.
RUN_ARGS_STR=$(
  PYTHONPATH="$REPO_ROOT/src" uv run --no-project python -c "
import shlex
from example_recipe.word_puzzle.agent_solver.minisweagent.docker_run_args import (
    SANDBOX_RUN_ARGS, SANDBOX_IMAGE)
print('__A__ ' + ' '.join(shlex.quote(a) for a in SANDBOX_RUN_ARGS))
print('__I__ ' + SANDBOX_IMAGE)
" 2>/dev/null
)
RA="$(printf '%s\n' "$RUN_ARGS_STR" | sed -n 's/^__A__ //p' | head -1)"
IMG="$(printf '%s\n' "$RUN_ARGS_STR" | sed -n 's/^__I__ //p' | head -1)"
func_fails=0

echo ""
echo "=== recipe functional checks ($IMG) ==="

echo "-- /workspace writable + base64 roundtrip"
out=$(docker run $RA "$IMG" bash -c \
  'echo hello | base64 > /workspace/x.b64 && cat /workspace/x.b64 | base64 -d' 2>&1)
if [[ "$out" == *"hello"* ]]; then echo "   PASS"; else
  echo "   FAIL: got $(echo "$out" | head -2)"; func_fails=$((func_fails+1)); fi

echo "-- python3 base64 decode"
out=$(docker run $RA "$IMG" python3 -c \
  'import base64; print("d:", base64.b64decode("aGVsbG8=").decode())' 2>&1)
if [[ "$out" == *"d: hello"* ]]; then echo "   PASS"; else
  echo "   FAIL: got $(echo "$out" | head -2)"; func_fails=$((func_fails+1)); fi

echo "-- state persists across docker exec (mini uses exec, not run, per step)"
CID=$(docker run -d $RA "$IMG" sleep 30 2>&1)
if [[ "$CID" =~ ^[0-9a-f]{12,}$ ]]; then
  docker exec "$CID" bash -c 'echo world > /workspace/hello.txt' >/dev/null 2>&1
  out=$(docker exec "$CID" bash -c 'cat /workspace/hello.txt' 2>&1)
  docker kill "$CID" >/dev/null 2>&1
  if [[ "$out" == "world" ]]; then echo "   PASS"; else
    echo "   FAIL: expected 'world', got: $out"; func_fails=$((func_fails+1)); fi
else
  echo "   FAIL: could not start persistent container: $CID"; func_fails=$((func_fails+1))
fi

echo ""
if [[ $iso_rc -eq 0 && $func_fails -eq 0 ]]; then
  echo "=== ALL SMOKE CHECKS PASSED ==="
  exit 0
else
  echo "=== SMOKE FAILED (isolation_rc=$iso_rc, functional_fails=$func_fails) ==="
  exit 1
fi
