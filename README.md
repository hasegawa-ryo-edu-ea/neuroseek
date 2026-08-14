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

## Architecture and engineering choices

NEUROSEEK is deliberately split into components with different jobs and trust
boundaries. Fast learned or GPU-assisted decisions reduce the search space;
only immutable graph evidence can establish an answer. This separation is the
reason the system can optimize for an edge device without presenting a
similarity score as a fact.

```text
Wikidata5M triples
       │  deterministic preprocessing + manifest/hash
       ▼
immutable mmap forward/reverse CSR ──► Rust NEURO-ISA VM ──► proof validator
       │                                      ▲                     │
       │                                      │                     ▼
       ▼                              bounded policy program   answer only if valid
aligned entity vectors ──► CUDA exact scoring/top-k ──► candidate jumps
       │
       └── semantic candidates guide search; they never replace graph proof
```

### 1. Graph evidence is the source of truth

The offline compiler turns triples into immutable, memory-mapped forward and
reverse CSR arrays. Hot paths use compact numeric representations (`uint64`
offsets, `uint32` node IDs, and `uint16` relation IDs), while human-readable
labels stay in separate TSV lookups. This avoids string processing and large
in-memory object graphs during search, and lets the same processed graph be
shared by training, evaluation, and read-only queries.

The Rust VM executes a constrained NEURO-ISA program under explicit budgets.
It records examined nodes and edges, instructions, ANN calls, credits, frontier
state, and the edges actually observed during execution. Its proof validator
then checks both the proposed answer and every proof edge against the immutable
CSR graph. A candidate that is plausible but lacks valid graph evidence remains
unverified rather than being converted into an answer.

### 2. Semantic retrieval is helpful but never authoritative

The semantic lane aligns one entity vector with each compact graph entity ID.
Its artifact contains the ordered IDs, row-major FP16 embeddings, and a
manifest with dimensions, normalization, source, coverage, and hashes. Full
mode accepts the artifact only when IDs are exactly the complete compact-ID
range; matching vector counts alone are not accepted as alignment evidence.

The current production backend deliberately uses exact CUDA dot-product
scoring in bounded batches and a deterministic host-side top-k merge. That is
a correctness-first trade-off: the backend reports its `cuda_exact` identity
and host merge instead of implying an approximate all-GPU index. CUDA failure
is fatal outside explicitly named smoke/test paths—there is no silent NumPy or
CPU fallback. Bounded TransE artifacts exist for trials, are marked partial in
their manifests, and cannot silently satisfy a full Wikidata5M run.

### 3. Learning is constrained by real execution costs

Python owns the compact query-conditioned navigator, curriculum adapter, and
training loop. The curriculum first collects measured CUDA-search behavior and
fits a durable hardware-cost model, then uses behavior cloning to anchor valid
proof programs before PPO improves the policy. The policy can be penalized for
predicted latency and instruction count, but a valid proof remains the dominant
success criterion; a shorter invalid trace is not rewarded as a successful
answer.

Tasks are structured `QuerySpec` graph problems, not free-form language
questions. Train/validation/test splits have separate seeds and persisted
files. This makes it possible to compare fixed held-out tasks using answer
accuracy, proof validity, nodes/edges examined, instruction count, ANN calls,
credits, latency, and energy when telemetry is available.

### 4. CUDA compatibility is a release gate, not a checkbox

The runtime image is pinned by digest to a Jetson/L4T-compatible NVIDIA image.
This follows a real compatibility constraint: enumerating the GPU in PyTorch
does not prove that custom CUDA code can execute on the host/toolchain pair.
Before training starts, the startup path requires the custom SM87 CUDA
primitives to agree with their host reference. The CUDA code is exposed through
a narrow C ABI, keeping bulk GPU operations separate from Python orchestration
and making parity failures visible early.

### 5. Long edge runs are designed to fail safely and resume visibly

Full semantic preparation is capped separately from the 50-hour curriculum
and writes an initial recovery checkpoint before expensive work. Progress and
training checkpoints are atomically published, fsynced, and validated on
resume; incompatible or corrupt state fails closed rather than being mixed with
a new run. A complete semantic artifact is published only after its manifest
and data are ready, and its temporary progress checkpoint is then removed.

The operator guardrails reserve disk space before large work, preserve
configuration/data/semantic hashes with each run, and use a critical thermal
policy that checkpoints and stops instead of continuing in a critical state.
Metrics are append-only JSONL. The Rust TUI and `watch` consume those durable
events read-only, so reconnecting over SSH can attach a monitor without
starting a duplicate trainer or contending for GPU memory.

### 6. Reproducibility and integration are first-class interfaces

Raw downloads retain source, timestamp, size, and SHA-256; processed graph data
is immutable; generated runs, logs, and checkpoints live separately under
`runs/`. The container digest, locked language dependencies, host probe, and
resolved environment manifest make the executable environment inspectable as
well as the model artifacts. The MCP server mirrors this discipline: it offers
read-only local facts and validated searches over stdio, while the calling LLM
remains responsible for language generation and any network-facing API.

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

- [Architecture](docs/ARCHITECTURE.md) / [日本語](docs/ARCHITECTURE.ja.md)
- [Training](docs/TRAINING.md) / [日本語](docs/TRAINING.ja.md)
- [Operator runbook](docs/RUNBOOK.md) / [日本語](docs/RUNBOOK.ja.md)
- [Experiment design](docs/EXPERIMENT_DESIGN.md) / [日本語](docs/EXPERIMENT_DESIGN.ja.md)
- [Reproducibility](docs/REPRODUCIBILITY.md) / [日本語](docs/REPRODUCIBILITY.ja.md)
- [Graph format](docs/GRAPH_FORMAT.md) / [日本語](docs/GRAPH_FORMAT.ja.md)
- [Semantic lane contract](docs/SEMANTIC.md) / [日本語](docs/SEMANTIC.ja.md)
- [NEURO-ISA](docs/NEURO_ISA.md) / [日本語](docs/NEURO_ISA.ja.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md) / [日本語](docs/TROUBLESHOOTING.ja.md)
- [Data and model assets](data/README.md) / [日本語](data/README.ja.md)
- [Third-party data and licenses](THIRD_PARTY.md) / [日本語](THIRD_PARTY.ja.md)
- [Final evaluation](reports/latency-final-presentation/FINAL_RESULT.en.md) / [日本語](reports/latency-final-presentation/FINAL_RESULT.ja.md)
