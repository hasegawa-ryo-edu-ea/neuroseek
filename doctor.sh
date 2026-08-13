#!/usr/bin/env bash
# Read-only preflight.  It may build nothing and never starts a trainer.
set -Eeuo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$root"

./scripts/host_probe.sh "$root/artifacts/host_probe.txt" >/dev/null
python3 scripts/write_environment_manifest.py
fail=0
need() { command -v "$1" >/dev/null 2>&1 || { echo "FAIL: missing $1" >&2; fail=1; }; }
need docker
need python3
need tegrastats
[[ -f compose.yaml ]] || { echo 'FAIL: compose.yaml missing' >&2; fail=1; }
[[ -f Dockerfile ]] || { echo 'FAIL: Dockerfile missing' >&2; fail=1; }
[[ -x scripts/host_probe.sh ]] || { echo 'FAIL: scripts/host_probe.sh is not executable' >&2; fail=1; }

docker_cmd=(docker)
if ! docker info >/dev/null 2>&1; then docker_cmd=(sudo -n docker); fi
if ! "${docker_cmd[@]}" info >/dev/null 2>&1; then
  echo 'FAIL: Docker daemon is not reachable by this user (add the user to docker group or use passwordless sudo).' >&2
  fail=1
else
  runtimes=$("${docker_cmd[@]}" info --format '{{json .Runtimes}}')
  [[ "$runtimes" == *'"nvidia"'* ]] || { echo 'FAIL: NVIDIA Docker runtime is absent.' >&2; fail=1; }
  # This particular Jetson needs host networking because its Docker bridge
  # cannot create the required raw-table rule.  NEUROSEEK intentionally has
  # no server; if a live trainer owns a TCP listener, fail the release gate
  # rather than silently widening its host-network exposure.
  if "${docker_cmd[@]}" compose -f compose.yaml ps --status running --services 2>/dev/null | grep -qx trainer; then
    trainer_id=$("${docker_cmd[@]}" compose -f compose.yaml ps -q trainer)
    trainer_pid=$("${docker_cmd[@]}" inspect --format '{{.State.Pid}}' "$trainer_id")
    if command -v ss >/dev/null 2>&1 && sudo -n ss -ltnp 2>/dev/null | grep -Fq "pid=$trainer_pid,"; then
      echo "FAIL: host-network trainer PID $trainer_pid owns a TCP listener; NEUROSEEK must not expose a network service." >&2
      fail=1
    fi
  fi
fi

for directory in data/raw data/processed cache runs artifacts; do
  mkdir -p "$directory"
  [[ -w "$directory" ]] || { echo "FAIL: $directory is not writable" >&2; fail=1; }
done

if (( fail )); then exit 1; fi
echo "NEUROSEEK doctor passed; host report: artifacts/host_probe.txt"
