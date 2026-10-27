#!/usr/bin/env bash
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
# Shared helpers so a launcher wrapped in `systemd-run --user` (or nohup) keeps
# running after the login session that started it ends. Source this, then call
# `ensure_linger` and `install_exit_trap` early in the launcher.
#
# Root cause this guards against: with `Linger=no` (the DEFAULT on a fresh
# machine / user), systemd tears down the per-user manager -- and every
# `systemd-run --user` unit it owns -- as soon as the last login session closes.
# A training loop launched that way then dies silently seconds/minutes after you
# disconnect (journal shows "Reached target exit.target - Exit the Session"),
# while the server-side tuning job keeps burning resources. Enabling linger keeps
# the user manager (and its units) alive independently of any login session.

# ensure_linger: idempotently enable systemd user-lingering for the current user.
# Tries as-self first (allowed on most systemd boxes via the set-self-linger
# polkit action), then passwordless sudo, then warns LOUDLY instead of failing --
# so the operator learns the one manual step rather than discovering a dead job
# hours later. Safe to call on every launch.
ensure_linger() {
  local u="${USER:-$(id -un)}"
  if [ "$(loginctl show-user "$u" -p Linger --value 2>/dev/null)" = "yes" ]; then
    echo "[survival] systemd linger already enabled for '$u'" >&2
    return 0
  fi
  if loginctl enable-linger "$u" 2>/dev/null || sudo -n loginctl enable-linger "$u" 2>/dev/null; then
    echo "[survival] enabled systemd linger for '$u' -- background jobs will now survive logout" >&2
  else
    echo "[survival] WARNING: could not enable linger for '$u'." >&2
    echo "[survival]   A job launched under 'systemd-run --user' WILL be killed when your" >&2
    echo "[survival]   session ends. Fix once with:  sudo loginctl enable-linger $u" >&2
    echo "[survival]   or run detached under tmux instead:  tmux new -s msrl '<launch cmd>'" >&2
  fi
  return 0
}

# install_exit_trap: make the launcher fail LOUD instead of dying silently. Emits
# a clearly-marked final line (exit status + timestamp) on ANY exit, so a monitor
# tailing the log can distinguish a crash / session-teardown from a healthy run.
install_exit_trap() {
  trap 'rc=$?; echo "[survival] launcher EXIT status=${rc} at $(date "+%Y-%m-%d %H:%M:%S %Z")" >&2' EXIT
  return 0
}
