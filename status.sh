#!/usr/bin/env bash
set -Eeuo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$root"
docker_cmd=(docker)
docker info >/dev/null 2>&1 || docker_cmd=(sudo -n docker)
"${docker_cmd[@]}" compose -f compose.yaml ps
if [[ -f runs/current.json ]]; then
  printf '\nCurrent run metadata:\n'
  python3 -m json.tool runs/current.json 2>/dev/null || cat runs/current.json
fi
