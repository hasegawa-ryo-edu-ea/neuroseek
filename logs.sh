#!/usr/bin/env bash
set -Eeuo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$root"
docker_cmd=(docker)
docker info >/dev/null 2>&1 || docker_cmd=(sudo -n docker)
exec "${docker_cmd[@]}" compose -f compose.yaml logs -f --tail=200 trainer
