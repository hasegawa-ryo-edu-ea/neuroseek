#!/usr/bin/env bash
# NEUROSEEK's single operator entry point.  It never removes datasets or runs.
set -Eeuo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$root"
mode=full
new_run=0
no_tui=0
while (($#)); do
  case "$1" in
    --smoke) mode=smoke ;;
    --trial) mode=trial ;;
    --doctor) exec ./doctor.sh ;;
    --status) exec ./status.sh ;;
    --new-run) new_run=1 ;;
    --no-tui) no_tui=1 ;;
    *) echo "usage: $0 [--smoke|--trial] [--new-run] [--no-tui] | --doctor | --status" >&2; exit 2 ;;
  esac
  shift
done

docker_cmd=(docker)
if ! docker info >/dev/null 2>&1; then docker_cmd=(sudo -n docker); fi
compose=("${docker_cmd[@]}" compose -f compose.yaml)
runtime_compose() {
  # sudo intentionally drops the caller environment.  Pass the four run
  # selectors explicitly so `--smoke`/`--trial` cannot silently execute the
  # default full configuration inside Compose.
  if [[ "${docker_cmd[0]}" == "sudo" ]]; then
    sudo -n env NEUROSEEK_RUN_MODE="${NEUROSEEK_RUN_MODE:-full}" \
      NEUROSEEK_CONFIG="${NEUROSEEK_CONFIG:-/workspace/config/full.toml}" \
      NEUROSEEK_RUN_DIR="${NEUROSEEK_RUN_DIR:-/workspace/runs/current}" \
      NEUROSEEK_RESUME="${NEUROSEEK_RESUME:-1}" \
      docker compose -f compose.yaml "$@"
  else
    "${compose[@]}" "$@"
  fi
}
./doctor.sh

# Check the run identity before any potentially long prerequisite work.  This
# prevents a harmless `./up.sh` typo after a stopped trial from spending hours
# on a full semantic build only to reject the incompatible checkpoint later.
if [[ "$mode" != smoke && "$new_run" == 0 && -f runs/current.json ]]; then
  current_mode_early=$(python3 -c 'import json; print(json.load(open("runs/current.json"))["mode"])')
  if [[ "$current_mode_early" != "$mode" ]]; then
    echo "current incomplete run is $current_mode_early, requested mode is $mode; refusing incompatible checkpoint resume before prerequisite setup. Use ./up.sh --new-run to create a $mode run." >&2
    exit 1
  fi
fi

# Reattachment must be cheap and non-invasive.  In particular, do not start a
# second CUDA preflight/ANN probe while the 50-hour trainer already owns the
# GPU: that would create avoidable unified-memory contention exactly when the
# operator reconnects over SSH.
trainer_running=0
if "${compose[@]}" ps --status running --services | grep -qx trainer; then trainer_running=1; fi
if [[ "$trainer_running" == 1 ]]; then
  if [[ "$mode" == smoke ]]; then
    echo "trainer is already running; refuse concurrent smoke GPU work. Attach with ./up.sh or stop safely with ./down.sh first." >&2
    exit 1
  fi
  if [[ "$new_run" == 1 ]]; then
    echo "trainer is already running; refuse to change run identity. Use ./down.sh, then ./up.sh --new-run." >&2
    exit 1
  fi
  [[ -f runs/current.json ]] || { echo "running trainer has no runs/current.json; refusing an unsafe attach" >&2; exit 1; }
  run_dir=$(python3 -c 'import json; print(json.load(open("runs/current.json"))["path"])')
  current_mode=$(python3 -c 'import json; print(json.load(open("runs/current.json"))["mode"])')
  if [[ "$current_mode" != "$mode" ]]; then
    echo "trainer is running in $current_mode mode; refuse to attach it as $mode. Stop it first with ./down.sh." >&2
    exit 1
  fi
  [[ -d "$run_dir" ]] || { echo "running trainer has invalid run directory: $run_dir" >&2; exit 1; }
  echo "NEUROSEEK trainer already detached: $run_dir"
  if [[ "$no_tui" == 0 && -t 1 ]]; then
    # runtime_compose is a shell function, not an executable.  Keep the
    # detached trainer independent and run the viewer in this terminal.
    runtime_compose run --rm tui neuroseek-tui "/workspace/$run_dir/metrics.jsonl"
  fi
  exit 0
fi

if [[ "$mode" != smoke && ! -f data/processed/manifest.json ]]; then
  echo "NEUROSEEK: acquiring and compiling the canonical Wikidata5M graph."
  "${compose[@]}" build trainer
  "${compose[@]}" run --rm trainer python3 scripts/download_data.py
  mapfile -t triples < <(find data/raw/wikidata5m -type f -name 'wikidata5m_transductive_*.txt' | sort)
  (( ${#triples[@]} >= 3 )) || { echo "Wikidata5M archive did not provide expected split files" >&2; exit 1; }
  compile_args=()
  for triple in "${triples[@]}"; do compile_args+=(--input "$triple"); done
  "${compose[@]}" run --rm trainer python3 scripts/preprocess.py "${compile_args[@]}" --train-only --output data/processed
fi

"${compose[@]}" build trainer
if [[ "$mode" != smoke ]]; then
  "${compose[@]}" run --rm trainer python3 scripts/verify_data.py data/processed
  "${compose[@]}" run --rm trainer python3 scripts/materialize_task_splits.py --graph data/processed
  if [[ "$mode" == trial ]]; then
    # Trial deliberately permits the bounded, explicitly partial TransE
    # fallback.  It is never promoted to the 50-hour full experiment.
    "${compose[@]}" run --rm trainer python3 scripts/build_semantic.py --graph data/processed --output data/processed/semantic_bounded
  elif [[ ! -f data/processed/semantic_full/semantic_manifest.json ]]; then
    echo "NEUROSEEK: building the full aligned CUDA TransE semantic artifact (all compact entity IDs)."
    semantic_limit=$(awk -F= '/^[[:space:]]*semantic_preparation_max_seconds[[:space:]]*=/{gsub(/[[:space:]]/, "", $2); print $2; exit}' config/full.toml)
    [[ "$semantic_limit" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo "invalid semantic_preparation_max_seconds in config/full.toml" >&2; exit 1; }
    set +e
    "${compose[@]}" run --rm trainer python3 scripts/build_semantic.py --full \
      --graph data/processed --output data/processed/semantic_full --dimension 64 --steps 100000 --batch-size 1024 \
      --checkpoint-every-steps 10000 --max-wall-seconds "$semantic_limit"
    semantic_status=$?
    set -e
    if [[ "$semantic_status" == 124 ]]; then
      echo "NEUROSEEK: full semantic preparation reached its ${semantic_limit}s safety budget after checkpointing. Rerun ./up.sh to resume preparation; training has not started." >&2
      exit 1
    fi
    (( semantic_status == 0 )) || exit "$semantic_status"
  fi
fi
# This explicitly verifies that PyTorch sees Jetson CUDA from inside the chosen
# container; a CPU fallback makes startup fail outside smoke mode.
"${compose[@]}" run --rm trainer python3 -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable in container"; print("CUDA container OK:", torch.cuda.get_device_name(0), torch.version.cuda)'
"${compose[@]}" run --rm trainer bash -lc 'LD_LIBRARY_PATH=/opt/neuroseek/lib neuroseek-cuda-parity'
"${compose[@]}" run --rm trainer python3 -c 'from neuroseek.cuda_backend import CudaExactBackend; b=CudaExactBackend(); assert b.device_count()>0; b.self_test(); print("CUDA exact backend parity OK")'
# The native VM has its own checked JSONL fixture.  This is not a UI demo: it
# proves the Rust NEURO-ISA/parser/proof path can run in the production image.
"${compose[@]}" run --rm trainer neuroseek-native --jsonl /workspace/rust/cli/tests/fixtures/path_proof.jsonl
if [[ "$mode" != smoke ]]; then
  "${compose[@]}" run --rm trainer python3 scripts/production_preflight.py --mode "$mode"
fi

if [[ "$mode" == smoke ]]; then
  run_dir="runs/smoke-$(date -u +%Y%m%dT%H%M%SZ)"
  NEUROSEEK_RUN_MODE=smoke NEUROSEEK_CONFIG=/workspace/config/smoke.toml NEUROSEEK_RUN_DIR="/workspace/$run_dir" NEUROSEEK_RESUME=0 runtime_compose run --rm trainer
  echo "Smoke complete: $run_dir"
  exit 0
fi

create_run() {
  run_id="$(date -u +%Y%m%dT%H%M%SZ)-$(openssl rand -hex 3)"
  run_dir="runs/$run_id"
  mkdir -p "$run_dir"
  python3 - "$run_id" "$run_dir" "$mode" <<'PY'
import json, sys, time
open('runs/current.json','w',encoding='utf-8').write(json.dumps({'run_id':sys.argv[1],'path':sys.argv[2],'mode':sys.argv[3],'updated_at':time.time()},indent=2)+'\n')
PY
}
if [[ "$trainer_running" == 1 && "$new_run" == 1 ]]; then
  echo "trainer is already running; refuse to change run identity. Use ./down.sh, then ./up.sh --new-run." >&2
  exit 1
fi
if [[ "$new_run" == 1 || ! -f runs/current.json ]]; then
  create_run
else
  run_dir=$(python3 -c 'import json; print(json.load(open("runs/current.json"))["path"])')
  current_mode=$(python3 -c 'import json; print(json.load(open("runs/current.json"))["mode"])')
  # A stopped container is not permission to reinterpret its checkpoint.  In
  # particular, a bounded trial must never become the first phase of a full
  # experiment simply because the operator later runs the default command.
  if [[ "$current_mode" != "$mode" ]]; then
    if [[ "$trainer_running" == 1 ]]; then
      echo "trainer is running in $current_mode mode; refuse to attach it as $mode. Stop it first with ./down.sh." >&2
    else
      echo "current incomplete run is $current_mode, requested mode is $mode; refusing incompatible checkpoint resume. Use ./up.sh --new-run to create a $mode run." >&2
    fi
    exit 1
  fi
  if [[ "$trainer_running" == 0 ]] && python3 - "$run_dir/manifest.json" <<'PY'
import json, pathlib, sys
p=pathlib.Path(sys.argv[1])
raise SystemExit(0 if p.is_file() and json.loads(p.read_text()).get('completed_at') else 1)
PY
  then
    echo "NEUROSEEK: previous run is complete; creating a new $mode run."
    create_run
  fi
fi
[[ -d "$run_dir" ]] || { echo "invalid current run directory: $run_dir" >&2; exit 1; }

if [[ "$mode" == trial ]]; then config=/workspace/config/trial.toml; else config=/workspace/config/full.toml; fi
export NEUROSEEK_RUN_MODE="$mode" NEUROSEEK_CONFIG="$config" NEUROSEEK_RUN_DIR="/workspace/$run_dir" NEUROSEEK_RESUME=1
if [[ "$trainer_running" == 0 ]]; then
  runtime_compose up -d trainer
fi
echo "NEUROSEEK trainer detached: $run_dir"

if [[ "$no_tui" == 0 && -t 1 ]]; then
  # runtime_compose is a Bash function, so it cannot be the target of exec.
  # Calling it normally preserves the detached trainer while the TUI owns this
  # terminal; closing SSH only ends this viewer process.
  runtime_compose run --rm tui neuroseek-tui "/workspace/$run_dir/metrics.jsonl"
fi
