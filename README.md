# NEUROSEEK

NEUROSEEK is a local knowledge-graph search system for the Jetson Orin Nano.
It uses a learned controller and CUDA to search a Wikidata5M graph. Every
answer includes graph evidence that the Rust validator checks independently.
It is a research/demo system that runs locally; it is not a hosted API or a
general-purpose chatbot.

Japanese version: [README.ja.md](README.ja.md)

## What it does

- Converts Wikidata5M into immutable, memory-mapped CSR graph files.
- Uses embeddings and CUDA search to propose candidates.
- Uses a learned NEURO-ISA policy to choose graph operations.
- Returns an answer only when the graph evidence passes validation.
- Records configuration, hashes, checkpoints, and append-only metrics for
  reproducible runs.

## Intended use and GUI status

The main use is to extend the knowledge available to a small local LLM. An
application that exposes an API can connect its local LLM to NEUROSEEK through
the included MCP server, then use the returned local graph facts or validated
search results as grounded context for its own response.

The terminal and web GUIs are demonstration tools built quickly with AI
assistance. They are useful for inspecting the model and proofs, but they are
not the primary product interface and should be treated as experimental. For
integration, use the MCP server instead:

```bash
python3 python/neuroseek/mcp_server.py
```

It communicates over standard input/output and provides read-only tools for
local facts and validation searches. It does not generate answers or expose an
HTTP API by itself; the calling LLM application is responsible for API access
and for writing the final response.

## Requirements

The validated environment is a Jetson Orin Nano running Jetson Linux R36.3
(JetPack 6 generation), Docker Compose, and NVIDIA Container Runtime. Keep
about 24 GiB free before a full run. The container image is pinned for Jetson;
do not replace it with an x86 CUDA image or enable CPU fallback.

The Wikidata5M data and model artifacts are distributed as GitHub Release
assets because they are too large for Git. Authenticate with GitHub CLI before
downloading from the private repository.

```bash
git clone https://github.com/hasegawa-ryo-edu-ea/neuroseek.git
cd neuroseek
gh release download initial-data-and-models \
  --repo hasegawa-ryo-edu-ea/neuroseek \
  --pattern 'neuroseek-*.tar.zst'
sha256sum -c data/RELEASE_SHA256SUMS
tar --zstd -xf neuroseek-wikidata5m-raw.tar.zst
tar --zstd -xf neuroseek-processed-data-and-models.tar.zst
```

This restores `data/raw/wikidata5m/` and `data/processed/`, including the
aligned `semantic_full` artifact. Do not place those files under `cache/`.

## Verify the installation

```bash
./up.sh --doctor
PYTHONPATH=python pytest -q
cargo test --workspace -q
```

On the validated target, these checks passed with 54 Python tests and 12 Rust
tests. This confirms the implementation and CUDA/data contracts, not a repeat
of the published benchmark.

## Run training

| Command | Purpose |
| --- | --- |
| `./up.sh --smoke` | Short synthetic functional and CUDA-path check. Not a scientific result. |
| `./up.sh --trial` | Bounded run on a deterministic real-data subgraph. It may use partial semantic coverage. |
| `./up.sh` | Starts or safely resumes the full detached training run. |
| `./up.sh --status` / `./logs.sh` | Shows the trainer state / follows its logs. |
| `./down.sh` | Gracefully stops training while preserving data and checkpoints. |

The first full run may build semantic vectors in the foreground. Keep the SSH
session open until `NEUROSEEK trainer detached:` appears or the monitor opens.
After that, it is safe to disconnect. Run `./up.sh` again to attach the
read-only monitor, or use `./up.sh --no-tui`. Interrupted semantic preparation
resumes from its atomic checkpoint.

For the standard 50-hour curriculum, use `config/full.toml`. A
latency-oriented specialization can start from a parent checkpoint with
`config/latency_optimization_6h.toml`:

```bash
mkdir -p runs/my-specialization
NEUROSEEK_CONFIG=/workspace/config/latency_optimization_6h.toml \
NEUROSEEK_RUN_DIR=/workspace/runs/my-specialization \
NEUROSEEK_RESUME= \
NEUROSEEK_PARENT_CHECKPOINT=/workspace/runs/<parent-run>/checkpoints/final.ckpt \
docker compose -f compose.yaml up -d trainer
```

Use `sudo -n docker compose` instead if your Docker configuration requires it.
Do not resume into the parent run or compare runs that use different graph,
split, or semantic artifacts.

## Query the final model

The query tools require CUDA and are read-only. They do not modify the graph,
checkpoints, or telemetry, but they share the Jetson GPU with training; run
them outside an active training session. The GUI is for demonstration and
inspection; use MCP for local-LLM integration.

```bash
./search
./search tui ja
./search web en
```

The web UI listens only on `http://127.0.0.1:8787`. Use SSH port forwarding to
open it from another machine. The terminal UI supports `r` (run task), `n`
(next task), `1`–`5` (page), `l` (language), and `:run 3`, `:lang ja`,
`:quit`. `Model` and `Path` show the policy output; `Proof` shows the
independent validation result. `./watch` and `./watch ja` display live
training metrics without controlling the trainer.

## Results and limits

The final model was evaluated against its parent on the same fixed 256-task
held-out set, graph, semantic artifact, and CUDA backend.

| Metric | Parent | Final model |
| --- | ---: | ---: |
| Mean end-to-end latency | 31.92 ms | 31.11 ms |
| p95 end-to-end latency | 57.10 ms | 53.02 ms |
| Mean compute credits/task | 233.98 | 146.02 |
| Answer accuracy / valid-proof rate | 98.05% | 97.27% |

The final model is faster and uses fewer compute credits, with a 0.78-point
accuracy decrease (two tasks). These are results from one Jetson and one fixed
test set; they are not confidence intervals, an external-SOTA comparison, or
a natural-language question-answering evaluation. See the [raw comparison]
(runs/latency-optimization-20260813T1255EDT/exports/benchmark_comparison.csv)
and [evaluation report](reports/latency-final-presentation/report.html).

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Training](docs/TRAINING.md)
- [Operator runbook](docs/RUNBOOK.md)
- [Experiment design](docs/EXPERIMENT_DESIGN.md)
- [Reproducibility](docs/REPRODUCIBILITY.md)
- [Graph format](docs/GRAPH_FORMAT.md)
- [Semantic lane contract](docs/SEMANTIC.md)
- [Third-party data and licenses](THIRD_PARTY.md)
