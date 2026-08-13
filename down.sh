#!/usr/bin/env bash
# Stops NEUROSEEK services only.  Bind-mounted datasets, caches, runs, and
# named build caches are intentionally retained for a safe later resume.
set -Eeuo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$root"
docker_cmd=(docker)
docker info >/dev/null 2>&1 || docker_cmd=(sudo -n docker)
"${docker_cmd[@]}" compose -f compose.yaml stop "$@"
