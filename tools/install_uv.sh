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
# Install a pinned `uv` release, verified against a committed SHA-256.
#
# This exists so no script in this repo has to pipe curl output straight into a
# shell, which executes whatever the remote server happens to serve. We download
# one pinned release artifact, check it against the hash committed below, and
# only then unpack it.
#
# Idempotent: no-op when `uv` is already on PATH.
#
# Environment:
#   UV_INSTALL_DIR   where to install uv/uvx (default: $HOME/.local/bin)
#   UV_INSTALL_SUDO  set to 1 to `sudo install` (needed for /usr/local/bin)
#
# To bump the version: change UV_VERSION and UV_SHA256 together. The hash is
# for the tarball itself, from the release's checksum file:
#   https://github.com/astral-sh/uv/releases/download/<ver>/<tarball>.sha256
set -euo pipefail

UV_VERSION="0.12.3"
UV_TARBALL="uv-x86_64-unknown-linux-gnu.tar.gz"
UV_SHA256="600cf9a742aca00d292673b16b5acffaa7b8c269a364ad0c2e79498dcb1fe101"

install_dir="${UV_INSTALL_DIR:-$HOME/.local/bin}"
use_sudo="${UV_INSTALL_SUDO:-0}"

if command -v uv >/dev/null 2>&1; then
  echo "uv already present ($(uv --version 2>/dev/null)); skipping install"
  exit 0
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

curl -LsSf -o "$tmp/$UV_TARBALL" \
  "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/${UV_TARBALL}"
echo "${UV_SHA256}  ${tmp}/${UV_TARBALL}" | sha256sum -c -
tar -xzf "$tmp/$UV_TARBALL" -C "$tmp"

if [[ "$use_sudo" == "1" ]]; then
  sudo mkdir -p "$install_dir"
  sudo install -m 0755 "$tmp"/uv-*/uv  "$install_dir/uv"
  sudo install -m 0755 "$tmp"/uv-*/uvx "$install_dir/uvx"
else
  mkdir -p "$install_dir"
  install -m 0755 "$tmp"/uv-*/uv  "$install_dir/uv"
  install -m 0755 "$tmp"/uv-*/uvx "$install_dir/uvx"
fi

echo "installed uv ${UV_VERSION} -> ${install_dir}/uv"
