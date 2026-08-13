#!/usr/bin/env bash
# Capture immutable host facts used to select a compatible Jetson container.
# This script intentionally does not install, configure, or restart anything.
set -Eeuo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output=${1:-"${project_root}/artifacts/host_probe.txt"}
mkdir -p "$(dirname "$output")"

run() {
  printf '\n$ %s\n' "$*"
  "$@" 2>&1 || printf '[unavailable or failed: exit %s]\n' "$?"
}

{
  printf 'NEUROSEEK host probe\n'
  printf 'captured_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  run uname -a
  run uname -m
  run cat /etc/os-release
  if [[ -r /etc/nv_tegra_release ]]; then run cat /etc/nv_tegra_release; fi
  run dpkg-query -W '-f=${Package} ${Version}\n' nvidia-l4t-core nvidia-jetpack nvidia-container-toolkit nvidia-container-toolkit-base
  run bash -lc 'for candidate in "$(command -v nvcc 2>/dev/null || true)" /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc; do test -n "$candidate" && test -x "$candidate" || continue; echo "$candidate"; "$candidate" --version; exit 0; done; echo "nvcc not installed"; exit 1'
  run python3 --version
  run bash -lc 'rustc --version; cargo --version'
  run docker --version
  run docker compose version
  run bash -lc 'docker info --format "runtimes={{json .Runtimes}} default={{.DefaultRuntime}}" || sudo -n docker info --format "runtimes={{json .Runtimes}} default={{.DefaultRuntime}}"'
  run df -h "${project_root}"
  run free -h
  run swapon --show
  run nproc
  run lsblk -o NAME,TYPE,SIZE,TRAN,MOUNTPOINTS -e7
  run findmnt -no SOURCE,FSTYPE,TARGET /
  run bash -lc 'command -v nvpmodel && nvpmodel -q'
  run bash -lc 'command -v tegrastats && timeout 2 tegrastats'
  run bash -lc 'for z in /sys/class/thermal/thermal_zone*; do test -r "$z/temp" || continue; printf "%s type=" "$z"; cat "$z/type"; printf "temp_mC="; cat "$z/temp"; done'
} | sed -E 's/[[:space:]]+$//' | tee "$output"
