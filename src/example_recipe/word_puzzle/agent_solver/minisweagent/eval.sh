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
# Agent-solver eval runner (composed path: eval.py + RunConfig).
#
# Wraps ``src/eval.py`` with the ``word_puzzle_agent/agent_async`` config.
# Gets ATIF trajectory writing, raw LLM capture, session_id, reward, and
# the eval prober for free from the composed pipeline.
#
# Usage:
#   ./eval.sh                                              # base model eval
#   ./eval.sh --tuning-job-id 12345                        # tuning job eval (convenience)
#   ./eval.sh --tuning-job-id 12345 --location us-central1 # with location
#   ./eval.sh target=projects/P/locations/L/tuningJobs/ID  # full chz target
#   ./eval.sh vertex_project=my-proj                       # override project
#   ./eval.sh jsonl_output_path=/tmp/eval.jsonl            # per-group JSONL
#
# Environment variables:
#   GCP_PROJECT     — GCP project (required if not passed via vertex_project=)
#   RAW_CAPTURE     — 1 to capture raw LLM requests (default: 1)
#   LOG_DIR         — output directory (default: /tmp/agent_eval)
#   CONFIG_NAME     — override config (default: word_puzzle_agent/agent_async)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# 5 levels up: src/example_recipe/word_puzzle/agent_solver/minisweagent -> repo root.
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
LOG_DIR="${LOG_DIR:-/tmp/agent_eval}"
CONFIG_NAME="${CONFIG_NAME:-word_puzzle_agent/agent_async}"

# ---- Translate --flag style to chz key=value style -------------------------
# Convenience: accept --tuning-job-id ID [--location LOC] and build the full
# Vertex resource target path automatically.
TUNING_JOB_ID=""
LOCATION="${VERTEX_LOCATION:-us-central1}"
PROJECT="${GCP_PROJECT:-}"
CHZ_PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tuning-job-id)
            TUNING_JOB_ID="$2"; shift 2 ;;
        --location)
            LOCATION="$2"; shift 2 ;;
        --project)
            PROJECT="$2"; shift 2 ;;
        *)
            CHZ_PASSTHROUGH+=("$1"); shift ;;
    esac
done

echo "=== eval.sh (composed path) ==="
echo "  repo:        $REPO_ROOT"
echo "  config:      $CONFIG_NAME"
echo "  log_dir:     $LOG_DIR"
if [[ -n "$TUNING_JOB_ID" ]]; then
    echo "  tuning job:  $TUNING_JOB_ID (location=$LOCATION)"
fi
echo ""

mkdir -p "$LOG_DIR"

cd "$REPO_ROOT"

# Build the chz CLI args.
CHZ_ARGS=()
CHZ_ARGS+=("config_name=$CONFIG_NAME")

# Build target from --tuning-job-id if provided.
if [[ -n "$TUNING_JOB_ID" ]]; then
    if [[ -z "$PROJECT" ]]; then
        echo "ERROR: --tuning-job-id requires GCP_PROJECT env var or --project flag." >&2
        exit 1
    fi
    CHZ_ARGS+=("target=projects/$PROJECT/locations/$LOCATION/tuningJobs/$TUNING_JOB_ID")
elif [[ -n "$PROJECT" ]]; then
    # No tuning job — just pass project through.
    CHZ_ARGS+=("vertex_project=$PROJECT")
fi

# Default to 10 eval tasks (override with max_eval_items=N on the CLI).
HAS_MAX_EVAL=0
for arg in "${CHZ_PASSTHROUGH[@]}"; do
    [[ "$arg" == max_eval_items=* ]] && HAS_MAX_EVAL=1
done
if [[ "$HAS_MAX_EVAL" == "0" ]]; then
    CHZ_ARGS+=("max_eval_items=10")
fi

# Pass remaining user args through (chz key=value style).
CHZ_ARGS+=("${CHZ_PASSTHROUGH[@]}")

echo "  chz args:    ${CHZ_ARGS[*]}"
PYTHONPATH=src RAW_CAPTURE="${RAW_CAPTURE:-1}" \
  .venv/bin/python src/eval.py "${CHZ_ARGS[@]}"
