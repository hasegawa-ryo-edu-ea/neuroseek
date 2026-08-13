# NEUROSEEK

**A proof-first, GPU-native knowledge-search research system for the Jetson Orin Nano.**

NEUROSEEK explores a simple question: can a compact learned policy decide how
to traverse a very large knowledge graph on an edge GPU, while still returning
evidence that can be independently checked?

It combines a memory-mapped [Wikidata5M](https://deepgraphlearning.github.io/project/wikidata5m)
graph, a Rust virtual machine and proof validator, custom CUDA primitives, a
semantic entity-vector lane, and a PyTorch policy trained with behavior cloning
and PPO. The system is designed as a reproducible research artifact and an
operator-friendly Jetson workload—not as a hosted search API.

> **Current status — research prototype.** CUDA/data gates and the full
> semantic artifact have been validated, but NEUROSEEK does **not** yet claim
> a completed end-to-end 50-hour scientific result or production-grade query
> latency. A latency-specialized model is currently being trained. This README
> intentionally separates its planned method from results that have not yet
> been measured.

日本語版: [README.ja.md](README.ja.md)

## Why NEUROSEEK?

Large knowledge-graph systems usually force a trade-off: graph traversal gives
auditable answers but can be expensive, while semantic retrieval is fast to
prototype but may not prove that a result is true. NEUROSEEK keeps both lanes:

- **Symbolic lane:** compact CSR graph storage and a constrained NEURO-ISA
  program are executed and independently proof-checked.
- **Semantic lane:** aligned entity embeddings propose candidate jumps; they
  never replace graph verification.
- **Learned controller:** a small query-conditioned policy learns which
  operations to use under a bounded execution budget.

The aim is not merely to retrieve an entity. It is to produce an answer with a
valid path through immutable graph evidence, while making the hardware costs
of that search visible and optimizable on an 8 GB-class edge device.

## What is already strong

| Capability | Why it matters |
| --- | --- |
| Proof validation in Rust | A returned answer receives credit only after an independent validator accepts its graph evidence. |
| GPU compatibility is a release gate | CUDA visibility alone is insufficient; a custom CUDA parity test must agree with the host reference before training starts. |
| Full semantic coverage | `semantic_full` is checked for hashes, dimensions, ordering, and coverage of every compact graph entity ID. Partial embeddings are rejected in full mode. |
| Reproducible data path | Wikidata5M is compiled into immutable, memory-mapped CSR arrays; task splits, configuration, and run manifests are persisted. |
| Failure-aware long runs | Atomic checkpoints, resume validation, free-space protection, and critical-temperature checkpoint-and-stop behavior protect a long Jetson run. |
| SSH-safe operation | Training is detached from the terminal. Reattaching the TUI cannot launch a duplicate trainer or contend for GPU memory. |

## Architecture at a glance

```text
Wikidata5M triples
        │ preprocess
        ▼
Memory-mapped CSR graph ──────► Rust NEURO-ISA VM ──────► proof validator
        │                               ▲                        │
        │                               │                        ▼
        │                         learned policy              valid answer
        │                               ▲
        ▼                               │
Aligned entity embeddings ──► CUDA score/top-k candidates
```

The semantic lane can suggest where to look. The symbolic lane is responsible
for establishing why an answer is valid.

## Latency: current reality and the work in progress

Latency is currently a first-class limitation, not a solved claim. The
production semantic backend uses exact CUDA dot-product scoring followed by a
deterministic host-side top-k merge. In addition, policy and VM steps can incur
repeated control-path overhead. These are deliberate correctness-first choices,
but they are not the final low-latency architecture.

### In-progress latency-specialized model

At the time this README was written, a derived Jetson-specialization run is in
progress with an expected completion window of roughly six hours. It starts
from a parent checkpoint and trains with:

- a cost model fitted from observed CUDA search probes;
- a configurable penalty for predicted query latency; and
- a small penalty for NEURO-ISA instruction count, encouraging shorter
  policy/VM interaction traces without rewarding invalid proofs.

The improvement must be reported against fixed held-out tasks and must retain
proof validity. Until that evaluation completes, no latency improvement number
or percentage should be inferred from this repository.

### Results to add after the run

Fill this table only with measurements from the completed fixed held-out
evaluation, including device state and configuration hash.

| Metric | Baseline model | Latency-specialized model | Status |
| --- | ---: | ---: | --- |
| Median end-to-end latency | pending | pending | evaluation pending |
| p95 end-to-end latency | pending | pending | evaluation pending |
| Mean NEURO-ISA instructions/query | pending | pending | evaluation pending |
| Proof validity | pending | pending | evaluation pending |
| Answer accuracy | pending | pending | evaluation pending |
| CUDA/host cost breakdown | pending | pending | evaluation pending |

The implementation and its configuration are deliberately isolated from the
baseline, so the eventual README update can name the parent checkpoint,
evaluation manifest, and exact measured deltas rather than presenting an
unverifiable benchmark.

## Quick start

### 1. Prerequisites

The validated target environment is a **Jetson Orin Nano** running Jetson Linux
R36.3 / JetPack 6 generation with NVIDIA Container Runtime, Docker Compose,
and approximately **24 GiB free disk space** before a full run. The container
image is pinned to a compatible NVIDIA L4T ML CUDA 12.2-generation digest.

You also need the release assets. They are not ordinary Git files because the
corpus and model exceed GitHub's 100 MB file limit.

```bash
git clone https://github.com/hasegawa-ryo-edu-ea/neuroseek.git
cd neuroseek

# For this private repository, authenticate with GitHub CLI first.
gh release download initial-data-and-models \
  --repo hasegawa-ryo-edu-ea/neuroseek \
  --pattern 'neuroseek-*.tar.zst'

sha256sum -c data/RELEASE_SHA256SUMS
tar --zstd -xf neuroseek-wikidata5m-raw.tar.zst
tar --zstd -xf neuroseek-processed-data-and-models.tar.zst
```

This restores `data/raw/wikidata5m/` and `data/processed/`, including the
full `semantic_full` embedding artifact. Do not place them under `cache/`.

### 2. Check the host before consuming GPU time

```bash
./up.sh --doctor
```

The doctor command checks the Docker/NVIDIA runtime prerequisites and refuses
a configuration that would expose an unexpected network listener.

### 3. Choose an execution mode

| Command | Purpose | Bounded? |
| --- | --- | --- |
| `./up.sh --smoke` | Synthetic functional/CUDA-path check. Not scientific evaluation. | Yes |
| `./up.sh --trial` | Deterministic real-data subgraph trial; permits the explicitly partial semantic artifact. | Yes |
| `./up.sh` | Full detached production training/resume using the complete semantic artifact. | 50-hour training budget; setup is separately capped |
| `./up.sh --status` | Inspect the detached trainer. | Yes |
| `./logs.sh` | Follow the current run's logs. | Yes |
| `./down.sh` | Gracefully stop training while preserving data and checkpoints. | Yes |

For a full run, the first semantic preparation may run in the foreground before
the trainer detaches. Wait for `NEUROSEEK trainer detached:` (or for the TUI)
before disconnecting SSH. If preparation is interrupted, run `./up.sh` again;
the atomic semantic checkpoint will resume safely.

### 4. Reattach safely

Once the trainer is detached, it continues independently of your SSH session.
Reconnect and run:

```bash
cd /path/to/neuroseek
./up.sh
```

When a healthy trainer already exists, this opens a read-only terminal
dashboard instead of starting another training job. For a non-interactive
attach path, use `./up.sh --no-tui`.

### Showcase TUI

For a presentation-oriented, high-contrast terminal display, build the
read-only Rust TUI and attach it to the current run:

```bash
./watch
```

Run `./watch ja` for Japanese. The helper builds only when needed, attaches to
the current run, and opens the cinematic full-screen view. It draws a
fixed-height program tree from the latest durable search program, plus
proof/search, policy, and Jetson telemetry. The tree reserves ten stable rows,
so the panels below keep their position while live trace data refreshes. It
reads only `metrics.jsonl`; `Ctrl-C` closes the display and never pauses,
resumes, or otherwise controls the trainer.

Use `1` Explore, `2` Trace, `3` System, and `4` Model to switch pages. Press
`l` to toggle the display language. `/` opens a local Codex-style command bar:
`/help`, `/lang ja`, `/lang en`, and `/quit` are supported. Add `--once` for a
single screenshot-friendly frame or `--no-color` for a plain log capture.

`./up.sh` now opens this monitor automatically when a healthy trainer is
already detached. Use `./up.sh --no-tui` when you only need launcher status.

### Real model-search console

For the video/demo search experience, launch the learned-policy console:

```bash
./search
```

Use `./search ja` for Japanese. This is a separate CPU-only, read-only
process: it loads the immutable presentation checkpoint and the mmap graph but
does not create a CUDA context, touch `metrics.jsonl`, or signal the trainer.
This makes it safe while an independent training job uses the GPU.

Press `r` to execute the loaded model on an immutable validation task, `n` for
the next task, `1`–`5` for Words/Model/Path/Proof/System, `l` for language, and
`:` for commands such as `:run 3`, `:lang ja`, and `:quit`. The displayed
operator lattice, candidates, answer, and proof state come from that actual
policy execution; the reference answer remains outside the model input.

The first **Words** page is for your own terms, not a prerecorded task. Press
`f`, type `Japan` (or `日本`), then press Enter. The console first shows only
results that are **ready to explore** in the downloaded graph, then displays
their real outgoing facts. Use `:use 2` for another result and `:rel capital`
(or `:rel 首都`) to filter facts by relation.

Wikidata may know a name that this fixed demo dataset did not download. That
does not mean the item is wrong or nonexistent; it simply cannot be explored
locally in this experiment. The viewer shows its Wikidata name and description
in a separately labeled context section, never as local graph evidence.
Name resolution uses the public Wikidata API; all graph traversal, targets,
and evidence remain local and read-only. Without network access, Q/P IDs can
still be entered directly.

### What the presentation demonstrates

NEUROSEEK should be demonstrated as an evidence pipeline, not as a generic
chatbot. The viewer deliberately separates three claims:

1. **Words** resolves a Japanese or English term and shows facts that are
   actually present in the local graph.
2. **Model / Path** runs the learned policy on a held-out task and shows the
   exact graph operators it selected.
3. **Proof** independently reconstructs the returned graph evidence and
   accepts or rejects it.

This separation makes it clear which information was resolved online for
display, which evidence was found on the Jetson, and which result was produced
by the trained policy.

## How to interpret a run

NEUROSEEK records append-only metrics and durable manifests under `runs/`.
The early `cuda_search_microbenchmarks` phase is a hardware-cost collection
stage: policy reward and proof metrics may legitimately be zero or `n/a`
there. Meaningful policy metrics begin after the cost-model and behavior-
cloning phases.

For a research result, report at least:

1. the run/configuration manifest and data/semantic hashes;
2. fixed held-out answer accuracy and proof validity;
3. examined nodes/edges, instruction count, ANN calls, and latency; and
4. device power/thermal conditions and any checkpoint-stop event.

Do not present a smoke test, successful container startup, or a partial
semantic artifact as a full Wikidata5M result.

## Repository map

```text
config/     Run budgets, curriculum, hardware safety, and experiment settings
cuda/       Custom SM87 CUDA primitives and host-parity test
data/       Release-asset instructions and checksums
docs/       Architecture, graph format, semantic contract, and runbooks
python/     Policy, training loop, graph/task code, telemetry, evaluation
rust/       NEURO-ISA VM, proof validator, CLI, and read-only TUI
scripts/    Data preparation, semantic build, verification, and preflight
tests/      Python correctness and recovery tests
```

## Validation performed on the target setup

The source tree has passed the following project checks on the validated Jetson
setup:

```bash
./up.sh --doctor
PYTHONPATH=python pytest -q        # 54 tests
cargo test --workspace -q          # 12 tests
```

These checks establish source, data-contract, and CUDA-parity readiness. They
do not substitute for an accepted long-duration scientific run or the pending
latency evaluation.

## Further reading

- [Architecture](docs/ARCHITECTURE.md)
- [Graph format](docs/GRAPH_FORMAT.md)
- [Semantic lane contract](docs/SEMANTIC.md)
- [Training and recovery](docs/TRAINING.md)
- [Experiment discipline](docs/EXPERIMENT_DESIGN.md)
- [Operator runbook](docs/RUNBOOK.md)
- [Japanese README](README.ja.md)

## License and external data

See [THIRD_PARTY.md](THIRD_PARTY.md) for the components and dataset provenance
used by this research project. Verify the applicable Wikidata5M terms before
redistributing the release assets.
