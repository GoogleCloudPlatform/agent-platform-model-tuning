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
# =============================================================================
# Provision a docker-sandboxed GCE VM and launch REAL MSRL-prod training of a
# word-puzzle agent-solver recipe (config-driven concurrency, tool-calling
# inside a container).
#
# REVIEW THIS BEFORE RUNNING. It creates a billed GCE VM, deploys the current
# local worktree (INCLUDING uncommitted changes), builds the sandbox image, and
# starts a real Vertex MSRL tuning job (spends real compute + tokens).
#
# What it does, in order:
#   1. create the VM (zone-fallback on STOCKOUT)
#   2. install docker + uv; tune fd/pid limits for high-concurrency containers
#   3. rsync src/ from THIS worktree to the VM
#   4. uv-sync the project venv on the VM
#   5. build the word-puzzle-agent-sandbox image
#   6. launch `train.py config_name=<CONFIG_NAME>` under nohup, pointed at
#      the Vertex MSRL prod backend (project/location). Concurrency and all
#      other run-shape knobs come from the config, not this script.
#
# It does NOT delete the VM afterwards (training is long-running); tear down
# manually when done (command printed at the end).
#
# Usage:
#   bash train_sandboxed.sh              # interactive: prompts before create + launch
#   CONFIRM=1 bash train_sandboxed.sh    # skip prompts (non-interactive)
#   DRY_RUN=1 bash train_sandboxed.sh    # print the plan + commands, create nothing
#
# for single-step / direct-answer recipes.
# =============================================================================
set -euo pipefail

# ---- Derived identity (no hardcoded user/project) ---------------------------
# Current unix user (used to namespace the VM so parallel users don't collide).
USER_ID="$(id -un)"
# Active gcloud project; overridable via GCP_PROJECT.
GCP_PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
# Repo root to deploy = the git worktree of the CURRENT directory (run this
# script from inside the checkout you want to deploy). Deliberately NOT derived
# from the script's own path -- the script may live in a sibling data/ dir whose
# git root is a different checkout.
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

# ---- Config (review / override via env) -------------------------------------
VM_NAME="${VM_NAME:-${USER_ID}-agent-train}"
MACHINE_TYPE="${MACHINE_TYPE:-n2-standard-32}"          # 32 vCPU / 128 GB
ZONES="${ZONES:-us-central1-a us-central1-b us-central1-c us-central1-f}"
BOOT_DISK_GB="${BOOT_DISK_GB:-500}"
IMAGE_FAMILY="${IMAGE_FAMILY:-debian-12}"
IMAGE_PROJECT="${IMAGE_PROJECT:-debian-cloud}"

# MSRL PROD backend. Project defaults to the active gcloud project.
VERTEX_PROJECT="${VERTEX_PROJECT:-$GCP_PROJECT}"
VERTEX_LOCATION="${VERTEX_LOCATION:-us-central1}"        # prod: {loc}-aiplatform.googleapis.com
# Reuse an existing tuning job id to skip the ~15min create+RUNNING wait.
# Empty => the backend creates a NEW prod tuning job (real budget).
VERTEX_EXISTING_JOB_ID="${VERTEX_EXISTING_JOB_ID:-}"

# Training target + run knobs.
CONFIG_NAME="${CONFIG_NAME:-word_puzzle_agent/agent_async}"   # concurrency comes from the config's RolloutConfig
REMOTE_DIR="${REMOTE_DIR:-/tmp/agent-train}"
LOG_DIR="${LOG_DIR:-/tmp/agent-train/logs}"

# Source worktree to deploy (this checkout, incl. uncommitted config).
LOCAL_REPO="${LOCAL_REPO:-$REPO_ROOT}"

DRY_RUN="${DRY_RUN:-0}"
CONFIRM="${CONFIRM:-0}"
# Run the sandbox-isolation verifier on the VM after the image build and before
# launching (aborts before any billed tuning job if isolation is broken).
VERIFY_ISOLATION="${VERIFY_ISOLATION:-1}"

# Fail early if we couldn't derive the project (needed for MSRL + gcloud).
if [[ -z "$VERTEX_PROJECT" ]]; then
  echo "ERROR: could not determine GCP project. Set GCP_PROJECT or run" >&2
  echo "       'gcloud config set project <proj>' first." >&2
  exit 1
fi

# Fail early (before creating any billed VM) if the deploy source doesn't look
# like the repo. Guards against running from the wrong cwd -> a late remote
# failure after the VM is already up and billing.
for f in src/train.py src/example_recipe tools/verify_sandbox_isolation.sh \
         tools/install_uv.sh pyproject.toml uv.lock; do
  if [[ ! -e "$LOCAL_REPO/$f" ]]; then
    echo "ERROR: '$LOCAL_REPO' does not look like the repo (missing $f)." >&2
    echo "       Run this script from inside the worktree you want to deploy," >&2
    echo "       or set LOCAL_REPO=<path-to-checkout>." >&2
    exit 1
  fi
done

say() { printf '\n=== %s ===\n' "$*"; }
run() { echo "+ $*"; [[ "$DRY_RUN" == "1" ]] || eval "$@"; }
confirm() {
  [[ "$CONFIRM" == "1" || "$DRY_RUN" == "1" ]] && return 0
  read -r -p "$1 [y/N] " a; [[ "$a" == "y" || "$a" == "Y" ]]
}

# Copy a local script file to the VM and run it with bash. Avoids fragile
# nested-quote escaping through gcloud ssh --command + eval.
run_remote_script() {
  local local_path="$1" remote_path="$2"
  echo "+ scp $local_path -> $VM_NAME:$remote_path && bash $remote_path"
  [[ "$DRY_RUN" == "1" ]] && { echo "--- (dry-run) contents of $local_path ---"; cat "$local_path"; return 0; }
  gcloud compute scp --zone="$ZONE" "$local_path" "$VM_NAME:$remote_path"
  gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="bash $remote_path"
}

ssh_vm() { gcloud compute ssh "$VM_NAME" --zone="$ZONE" --command="$1"; }

# ---- Plan summary -----------------------------------------------------------
say "PLAN"
cat <<EOF
  VM          : $VM_NAME  ($MACHINE_TYPE, ${BOOT_DISK_GB}GB, zones: $ZONES)
  backend     : MSRL PROD  project=$VERTEX_PROJECT location=$VERTEX_LOCATION
  tuning job  : ${VERTEX_EXISTING_JOB_ID:-<create new prod job>}
  config      : $CONFIG_NAME   (concurrency + samples_per_task from the config)
  deploy from : $LOCAL_REPO
  remote dir  : $REMOTE_DIR    log dir: $LOG_DIR
  DRY_RUN=$DRY_RUN  CONFIRM=$CONFIRM
EOF
[[ "$VERTEX_EXISTING_JOB_ID" == "" ]] && \
  echo "  WARNING: no VERTEX_EXISTING_JOB_ID -> creates a NEW real prod tuning job."
confirm "Proceed to create the VM and launch training?" || { echo "aborted."; exit 0; }

# ---- 1. create VM (idempotent: reuse if it already exists) ------------------
# Re-running the script (e.g. to recover from a mid-run failure) must NOT error
# on "instance already exists". First look the VM up by name across the project;
# if found, reuse it (and its actual zone). Only create when truly absent.
say "CREATE VM (or reuse existing)"
ZONE=""
if [[ "$DRY_RUN" != "1" ]]; then
  # Discover the zone of any existing instance with this name (empty if none).
  EXISTING_ZONE="$(gcloud compute instances list \
      --filter="name=($VM_NAME)" --format='value(zone)' 2>/dev/null | head -1)"
  if [[ -n "$EXISTING_ZONE" ]]; then
    ZONE="$EXISTING_ZONE"
    echo "VM '$VM_NAME' already exists in $ZONE -- reusing it (skipping create)."
    echo "  (to force a clean VM: delete it first, or set VM_NAME=<other>)"
  fi
fi

if [[ -z "$ZONE" ]]; then
  for z in $ZONES; do
    echo "trying $MACHINE_TYPE in $z ..."
    if [[ "$DRY_RUN" == "1" ]]; then ZONE="$z"; echo "(dry-run) would create in $z"; break; fi
    if gcloud compute instances create "$VM_NAME" \
        --zone="$z" --machine-type="$MACHINE_TYPE" \
        --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT" \
        --boot-disk-size="${BOOT_DISK_GB}GB" --boot-disk-type=pd-ssd \
        --scopes=cloud-platform 2>/tmp/.create_err; then
      ZONE="$z"; echo "created in $z"; break
    else
      grep -qi "stockout\|does not have enough" /tmp/.create_err \
        && { echo "  stockout in $z, next zone"; continue; }
      # Race: another run created it between our list check and create.
      grep -qi "already exists" /tmp/.create_err \
        && { echo "  VM appeared concurrently; reusing in $z"; ZONE="$z"; break; }
      cat /tmp/.create_err >&2; exit 1
    fi
  done
  [[ -n "$ZONE" ]] || { echo "ERROR: no capacity in any zone: $ZONES" >&2; exit 1; }
  [[ "$DRY_RUN" == "1" ]] || sleep 25   # let sshd + keys come up on a FRESH VM
fi

# ---- 2. install docker + uv; tune fd/pid for high-concurrency containers ----
# Generated as a file + scp'd + run, so the daemon.json quoting survives.
say "SETUP VM (docker, uv, ulimits)"
SETUP_SH="$(mktemp /tmp/agent_train_setup.XXXX.sh)"
cat > "$SETUP_SH" <<EOF
set -e
sudo apt-get update -qq && sudo apt-get install -y -qq docker.io rsync
sudo systemctl start docker && sudo usermod -aG docker \$USER
# fd/pid tuning for high-concurrency container fleets (headroom sized for
# ~1024 concurrent containers on the probe run; safe for smaller runs too).
sudo timedatectl set-timezone America/Los_Angeles
sudo sysctl -w kernel.pid_max=4194304
sudo tee /etc/docker/daemon.json >/dev/null <<'JSON'
{"default-ulimits":{"nofile":{"Name":"nofile","Hard":1048576,"Soft":1048576}}}
JSON
sudo systemctl restart docker
bash /tmp/install_uv.sh
mkdir -p "$REMOTE_DIR" "$LOG_DIR"
echo SETUP_OK
EOF
# uv comes from the shared pinned+checksummed installer, not a curl-piped-to-
# shell one. It is pushed separately because the repo itself is only deployed
# in step 3.
run "gcloud compute scp --zone='$ZONE' '$LOCAL_REPO/tools/install_uv.sh' '$VM_NAME:/tmp/install_uv.sh'"
run_remote_script "$SETUP_SH" "/tmp/setup.sh"

# ---- 3. deploy repo (this worktree, incl uncommitted config) ----------------
# The word_puzzle_agent recipe resolves and runs from `src/` alone; `tools/`
# is deployed too so the on-VM isolation pre-flight can run. PYTHONPATH is
# `src` on the VM.
#
# `scp --recurse` copies files one-per-SSH-round-trip -- ~166 tiny files in
# src/ makes it latency-bound and slow. Instead: tar (excluding __pycache__)
# into ONE archive, scp that single file, untar on the VM. One round-trip.
say "DEPLOY REPO -> $VM_NAME:$REMOTE_DIR (tarball)"
DEPLOY_TGZ="$(mktemp /tmp/agent_train_deploy.XXXX.tgz)"
# DEPLOY_EXTRA_DIRS can be set by a calling wrapper to deploy additional
# directories (e.g. for non-prod env support or monitoring).
DEPLOY_DIRS="src tools pyproject.toml uv.lock ${DEPLOY_EXTRA_DIRS:-}"
run "tar czf '$DEPLOY_TGZ' -C '$LOCAL_REPO' \
  --exclude='__pycache__' --exclude='*.pyc' \
  $DEPLOY_DIRS"
run "gcloud compute scp --zone=$ZONE '$DEPLOY_TGZ' $VM_NAME:$REMOTE_DIR/deploy.tgz"
run "gcloud compute ssh $VM_NAME --zone=$ZONE \
  --command='tar xzf $REMOTE_DIR/deploy.tgz -C $REMOTE_DIR && rm -f $REMOTE_DIR/deploy.tgz && echo DEPLOY_OK'"
[[ "$DRY_RUN" == "1" ]] || rm -f "$DEPLOY_TGZ"

# ---- 4/5/6. uv sync + build image + launch training -------------------------
say "UV SYNC + BUILD IMAGE + LAUNCH TRAINING ($CONFIG_NAME, MSRL prod)"
JOB_FLAG=""
[[ -n "$VERTEX_EXISTING_JOB_ID" ]] && JOB_FLAG="vertex_existing_job_id=$VERTEX_EXISTING_JOB_ID"

# TRAIN_SCRIPT / TRAIN_ENV_PREFIX can be overridden by a calling wrapper
# (e.g. for non-prod env support or monitoring). Default: src/train.py.
TRAIN_SCRIPT="${TRAIN_SCRIPT:-src/train.py}"
TRAIN_ENV_PREFIX="${TRAIN_ENV_PREFIX:-}"
LAUNCH_SH="$(mktemp /tmp/agent_train_launch.XXXX.sh)"
cat > "$LAUNCH_SH" <<EOF
set -e
export PATH="\$HOME/.local/bin:\$PATH"
cd "$REMOTE_DIR"
echo '=== uv sync (with agent_solver extra: pulls mini-swe-agent) ==='
# The agent_solver recipe imports \`minisweagent\`, which lives in the OPTIONAL
# \`agent_solver\` extra (kept out of base install to spare RL-only users its
# heavy transitive stack). Without --extra, \`import minisweagent\` fails on the
# VM. \`uv run\` below repeats the extra so its auto-sync doesn't drop it.
uv sync --extra agent_solver 2>&1 | tail -5
uv run --extra agent_solver python -c 'import minisweagent; print("minisweagent OK", minisweagent.__version__ if hasattr(minisweagent, "__version__") else "")' \\
  || { echo "FATAL: minisweagent still missing after uv sync --extra agent_solver" >&2; exit 1; }
echo '=== build sandbox image ==='
sudo docker build -q -t word-puzzle-agent-sandbox:latest src/example_recipe/word_puzzle/agent_solver/minisweagent/
sudo docker image inspect word-puzzle-agent-sandbox:latest >/dev/null && echo IMAGE_OK
if [[ "$VERIFY_ISOLATION" != "0" ]]; then
  echo '=== verify sandbox isolation on THIS host (before spending prod budget) ==='
  # Proves --network=none / non-root / read-only / no-creds hold on this VM.
  # A failure here aborts BEFORE the (billed) tuning job is created.
  PYTHONPATH_ROOT=src \
    bash tools/verify_sandbox_isolation.sh \
      --from-module example_recipe.word_puzzle.agent_solver.minisweagent.docker_run_args \
      || { echo "FATAL: sandbox isolation checks FAILED -- not launching." >&2; exit 1; }
fi
echo '=== launch training ==='
export VMSRL_VM_NAME="$VM_NAME"
export RAW_CAPTURE="${RAW_CAPTURE:-1}"
nohup env $TRAIN_ENV_PREFIX uv run --extra agent_solver --project . python $TRAIN_SCRIPT \\
    config_name=$CONFIG_NAME \\
    vertex_project=$VERTEX_PROJECT \\
    vertex_location=$VERTEX_LOCATION \\
    log_dir="$LOG_DIR" \\
    $JOB_FLAG \\
    > "$LOG_DIR/train.out" 2>&1 &
echo "train pid \$!"
sleep 10
echo '=== first log lines ==='; tail -20 "$LOG_DIR/train.out"
EOF
run_remote_script "$LAUNCH_SH" "/tmp/launch.sh"

# ---- stream the live training log for a bounded window ----------------------
# Training runs long under nohup, so an unbounded `tail -f` would hang the
# script forever. Instead, follow the log for TAIL_SECONDS (default 60s) so you
# see the run come alive, then detach and print the reference commands. Set
# TAIL_SECONDS=0 to skip streaming.
TAIL_SECONDS="${TAIL_SECONDS:-60}"
if [[ "$DRY_RUN" != "1" && "$TAIL_SECONDS" != "0" ]]; then
  say "STREAMING TRAINING LOG (${TAIL_SECONDS}s, then detaching -- the run keeps going)"
  # `timeout` stops the remote tail; SIGdefault kills only our ssh'd tail, not
  # the nohup'd trainer. `|| true` so the timeout's non-zero exit doesn't abort.
  timeout "${TAIL_SECONDS}" \
    gcloud compute ssh "$VM_NAME" --zone="$ZONE" \
      --command="tail -n +1 -f $LOG_DIR/train.out" || true
fi

say "DONE"
cat <<EOF
Training running on $VM_NAME ($ZONE) under nohup (survives this script exiting).
  tail logs : gcloud compute ssh $VM_NAME --zone=$ZONE --command='tail -f $LOG_DIR/train.out'
  watch cids: grep for candidate_id INDEX warnings in the log (should be NONE on MSRL)
  tear down : gcloud compute instances delete $VM_NAME --zone=$ZONE --quiet
EOF
