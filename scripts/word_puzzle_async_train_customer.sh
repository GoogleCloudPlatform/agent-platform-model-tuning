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
# Self-contained async WORD-PUZZLE RL training wrapper.
#
# This launcher:
#   * puts only `src` on PYTHONPATH,
#   * dispatches through the `src/train.py` entry point, and
#   * passes every tuning knob explicitly (no hidden project defaults).
#
# It runs the `word_puzzle_async` recipe against the Vertex AI tuning
# (`vertex_msrl`) backend.
#
# Steps: (1) create a Vertex tuning job and wait for RUNNING (unless
# VERTEX_EXISTING_JOB_ID is set), (2) run word_puzzle_async against it.
#
# Usage (from the checkout root, with the venv built):
#   bash scripts/word_puzzle_async_train_customer.sh
#
# Reuse an existing RUNNING job (skips job creation):
#   VERTEX_EXISTING_JOB_ID=<id> bash scripts/word_puzzle_async_train_customer.sh
#
# Override individual chz config knobs (forwarded verbatim, last-writer
# wins for chz-style configs), e.g. enable Vertex Experiments logging:
#   bash scripts/word_puzzle_async_train_customer.sh \
#     config.vertex_experiment=my-experiment
set -euo pipefail
export TZ="America/Los_Angeles"
# `src` is the sole entry on PYTHONPATH. The heredoc below (job
# creation) and step 2 (training) both need `backends.*` importable,
# so set it once here before either runs.
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
# Uses the default uv.lock (pypi.org). `uv run` below installs from it.
# Keep this run alive after the launching login session ends, and make an early
# death loud rather than silent (see scripts/session_survival.sh).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/session_survival.sh" 2>/dev/null || true
command -v install_exit_trap >/dev/null 2>&1 && install_exit_trap || true
command -v ensure_linger     >/dev/null 2>&1 && ensure_linger     || true

VERTEX_PROJECT="${VERTEX_PROJECT:-}"
VERTEX_LOCATION="${VERTEX_LOCATION:-us-central1}"
VERTEX_BASE_MODEL="${VERTEX_BASE_MODEL:-gemini-3.5-flash}"


if [ -z "${VERTEX_PROJECT}" ]; then
  echo "[vertex] ERROR: set VERTEX_PROJECT=<your GCP project> before running." >&2
  exit 2
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] [vertex] Environment: project=${VERTEX_PROJECT} location=${VERTEX_LOCATION} base_model=${VERTEX_BASE_MODEL}" >&2

# ---------------------------------------------------------------------------
# Job lifecycle: create (if needed), cancel on failure / signal.
# ---------------------------------------------------------------------------
# WE_CREATED_JOB tracks whether this script created the job (vs reuse).
# Only cancel on exit if we created it.
WE_CREATED_JOB=0
JOB_ID="${VERTEX_EXISTING_JOB_ID:-}"

_cancel_job() {
  if [ "${WE_CREATED_JOB}" -eq 1 ] && [ -n "${JOB_ID}" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] [vertex] Cancelling tuning job ${JOB_ID} (we created it)..." >&2
    uv run python - "${VERTEX_PROJECT}" "${VERTEX_LOCATION}" "${JOB_ID}" <<'CANCEL' 2>&1 | sed 's/^/  /' >&2 || true
import sys, asyncio
from backends.vertex_msrl_rest import VertexMsrlRestClient as C, default_endpoint
project, location, job_id = sys.argv[1], sys.argv[2], sys.argv[3]
async def main():
    ep = default_endpoint(location)
    client = C(project=project, location=location, endpoint=ep, tuning_job_id=job_id)
    try:
        await client.cancel_tuning_job()
        print(f"Cancelled tuning job {job_id}.")
    finally:
        await client.aclose()
asyncio.run(main())
CANCEL
  fi
}

# Trap EXIT so _cancel_job fires on any failure path (set -e, signal, etc.)
trap '_cancel_job' EXIT

if [ -z "${JOB_ID}" ]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] [vertex] Creating tuning job (project=${VERTEX_PROJECT} model=${VERTEX_BASE_MODEL}); waiting for RUNNING..." >&2
  JOB_ID="$(uv run python - "${VERTEX_PROJECT}" "${VERTEX_LOCATION}" "${VERTEX_BASE_MODEL}" <<'PY'
import sys, asyncio, time as _time
def _ts():
    return _time.strftime("%Y-%m-%d %H:%M:%S %Z")
from backends.vertex_msrl_rest import VertexMsrlRestClient as VertexMsrlClient, default_endpoint
project, location, base_model = sys.argv[1], sys.argv[2], sys.argv[3]
async def main():
    ep = default_endpoint(location)
    client = VertexMsrlClient(project=project, location=location, endpoint=ep)
    try:
        job_id = await client.create_tuning_job(
            base_model=base_model,
            description="word_puzzle_async_train_customer.sh",
        )
        print(f"[{_ts()}] [vertex] Tuning job created: {client.tuning_job_name} -- waiting for RUNNING...", file=sys.stderr, flush=True)
        await client.wait_for_job_running()
        print(job_id)  # ONLY the job id goes to stdout; all logs go to stderr
    except BaseException:
        # Cancel the job we just created if wait fails or we're interrupted.
        try:
            await client.cancel_tuning_job()
            print(f"[{_ts()}] [vertex] Cancelled job after wait failure.", file=sys.stderr, flush=True)
        except Exception:
            pass
        raise
    finally:
        await client.aclose()
asyncio.run(main())
PY
)"
  WE_CREATED_JOB=1
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] [vertex] Tuning job RUNNING: ${JOB_ID}" >&2
fi

# Trailing `"$@"` forwards any extra `config.*=...` args to `src/train.py`,
# so callers can override individual knobs without editing the script.
# Later `config.foo=` wins for chz-style configs.
uv run python src/train.py word_puzzle_async \
  config.backend=vertex_msrl \
  config.vertex_project="${VERTEX_PROJECT}" \
  config.vertex_location="${VERTEX_LOCATION}" \
  config.vertex_base_model="${VERTEX_BASE_MODEL}" \
  config.vertex_existing_job_id="${JOB_ID}" \
  config.vertex_job_running_timeout_seconds=7200 \
  config.vertex_request_timeout=600.0 \
  config.num_train_steps=120 \
  config.sync_every_n_train_steps=4 \
  config.tasks_per_rollout_batch=32 \
  config.tasks_per_train_step=32 \
  config.samples_per_task=16 \
  config.max_pending_rollout_batches=128 \
  config.max_concurrent_samples=1024 \
  config.trajectory_group_filter.zero_variance_keep_prob=0.3 \
  config.reward_normalizer=centering \
  config.eval_every_n_train_steps=8 \
  config.eval_samples_per_task=5 \
  config.max_eval_items=-1 \
  config.max_generation_tokens=32768 \
  "$@"

# Training completed normally — don't cancel the job on exit.
WE_CREATED_JOB=0
