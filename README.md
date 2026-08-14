# NEUROSEEK

**A proof-first, GPU-native knowledge-graph navigator for the Jetson Orin Nano.**

NEUROSEEK is the completed low-latency culmination of this project. Its learned controller searches a large graph on an edge GPU, and an answer is accepted only with graph evidence that an independent validator verifies. It is a reproducible research system and local evidence-bounded search demonstrator—not a hosted API or general chatbot.

日本語版: [README.ja.md](README.ja.md)

## Why this project exists

Knowledge systems often choose between fast semantic retrieval that is hard to audit and symbolic graph traversal that is inspectable but costly. NEUROSEEK makes that trade-off explicit on an 8 GB-class edge device: learning and CUDA decide where to search; immutable graph evidence establishes the final claim.

The goal is more demanding than returning a plausible entity. A result must include a replayable, independently validated path, and reveal its cost in latency, instructions, and graph work.

## Final low-latency result

The final model and its parent were evaluated on the same fixed 256-task held-out test set, graph, semantic provenance, and CUDA exact backend.

| Metric | Parent | Final model | Change |
| --- | ---: | ---: | ---: |
| Mean end-to-end latency | 31.92 ms | **31.11 ms** | **-2.56%** |
| p95 end-to-end latency | 57.10 ms | **53.02 ms** | **-7.14%** |
| Mean compute credits/task | 233.98 | **146.02** | **-37.59%** |
| Answer accuracy / valid-proof rate | **98.05%** | 97.27% | -0.78 points (2 tasks) |

This is an intentional, measured efficiency trade-off—not a claim that every quality dimension improved. Both regressions were in the path family; accuracy was retained for distractor, intersection, semantic-hybrid, and robustness. The 4.66 ms hand-written hybrid baseline is not directly comparable: it receives the relation program, while the learned model chooses its own operator sequence.

Raw result: [benchmark comparison](runs/latency-optimization-20260813T1255EDT/exports/benchmark_comparison.csv). See the [final evaluation report](reports/latency-final-presentation/report.html) for the comparison and limits.

## What makes it strong

| Capability | Why it matters |
| --- | --- |
| Independent proof validation | Rust accepts a result only when the returned graph evidence is valid; semantic similarity is never proof. |
| Evidence-bounded hybrid search | Embeddings and CUDA search propose candidates; the symbolic lane establishes why the answer is valid. |
| Edge-first cost-aware control | A compact query-conditioned policy learns bounded NEURO-ISA programs and was specialized to reduce instructions and device cost. |
| GPU correctness gate | Training requires custom CUDA/host parity, not merely CUDA visibility. |
| Reproducible graph path | Wikidata5M becomes immutable memory-mapped CSR; splits, hashes, config, checkpoints, and append-only metrics persist. |
| Long-run and SSH safety | Atomic recovery, a 24 GiB reserve, thermal checkpoint-and-stop, and a read-only reconnect monitor protect the run. |

## Architecture

~~~
Wikidata5M triples → immutable preprocessing → memory-mapped forward/reverse CSR graph
                                             ↑
        embeddings → CUDA exact score/top-k → candidates
                                             ↑
learned query-conditioned policy → Rust NEURO-ISA VM → Rust proof validator
                                                        ↓
                                            accepted answer + evidence
~~~

Rust owns VM semantics, accounting, and proof validation. CUDA supplies bulk primitives with host-parity tests. Python owns the navigator, PPO/behavior-cloning training, curriculum, checkpointing, telemetry, and evaluation. Semantic search can suggest a jump, but cannot bypass validation.

## Reproduce on your Jetson

The validated target is Jetson Orin Nano with Jetson Linux R36.3 / JetPack 6 generation, Docker Compose, and NVIDIA Container Runtime. Reserve about 24 GiB before a full run. The build pins a compatible NVIDIA L4T ML image; do not substitute an x86 CUDA image or accept CPU fallback.

Large data and model assets are a GitHub Release. Authenticate with GitHub CLI first for the private repository.

~~~
git clone https://github.com/hasegawa-ryo-edu-ea/neuroseek.git
cd neuroseek
gh release download initial-data-and-models \
  --repo hasegawa-ryo-edu-ea/neuroseek \
  --pattern 'neuroseek-*.tar.zst'
sha256sum -c data/RELEASE_SHA256SUMS
tar --zstd -xf neuroseek-wikidata5m-raw.tar.zst
tar --zstd -xf neuroseek-processed-data-and-models.tar.zst
./up.sh --doctor
PYTHONPATH=python pytest -q
cargo test --workspace -q
~~~

The release restores data/raw/wikidata5m/ and data/processed/, including fully aligned semantic_full; do not place them under cache/. On the validated target, the checks passed with 54 Python and 12 Rust tests. They establish implementation and CUDA/data-contract readiness, not reproduction of the final benchmark.

| Command | Purpose |
| --- | --- |
| ./up.sh --smoke | Bounded synthetic functional/CUDA-path check; not a scientific result. |
| ./up.sh --trial | Bounded deterministic real-data subgraph run; explicitly permits partial semantic coverage. |
| ./up.sh | Full detached training or safe resume with the complete semantic artifact. |
| ./up.sh --status / ./logs.sh | Inspect the detached trainer or follow logs. |
| ./down.sh | Gracefully stop while preserving data and checkpoints. |

The first full invocation can prepare semantic vectors in the foreground. Do not disconnect SSH until NEUROSEEK trainer detached: appears (or the monitor opens). Afterwards, reconnect with ./up.sh for the read-only monitor, or use ./up.sh --no-tui. Interrupted semantic preparation resumes from its atomic checkpoint.

## Train or specialize a model

Use config/full.toml for the standard 50-hour curriculum. It records phase budgets, caps semantic preparation separately, checkpoints every five minutes, and saves progress then stops at critical temperature.

For a latency-oriented derivative, copy config/latency_optimization_6h.toml, retain the parent's graph/split/semantic artifacts, and record every changed setting. The supplied configuration starts from a parent checkpoint and applies small latency and instruction penalties while keeping valid-proof reward dominant.

~~~
mkdir -p runs/my-specialization
NEUROSEEK_CONFIG=/workspace/config/latency_optimization_6h.toml \
NEUROSEEK_RUN_DIR=/workspace/runs/my-specialization \
NEUROSEEK_RESUME= \
NEUROSEEK_PARENT_CHECKPOINT=/workspace/runs/<parent-run>/checkpoints/final.ckpt \
docker compose -f compose.yaml up -d trainer
~~~

Replace docker compose with sudo -n docker compose if required. Never resume into the parent run, mix semantic_bounded into a full comparison, or report a changed configuration without its manifest and fixed held-out evaluation. Keep train/validation/test seeds separate and evaluate accuracy, proof validity, instructions, graph work, latency, and device conditions together. See [Training](docs/TRAINING.md), [experiment design](docs/EXPERIMENT_DESIGN.md), and [reproducibility](docs/REPRODUCIBILITY.md).

## Use the final model

The presentation/query tool is CPU-only and read-only, so it can demonstrate an immutable checkpoint without contending with training.

~~~
./search
./search tui ja
./search web en
~~~

The Web UI is at http://127.0.0.1:8787; use ja or en. In the terminal console, r runs an immutable validation task, n selects the next task, 1–5 switch pages, l switches language, and :run 3, :lang ja, and :quit are available. Model and Path show the policy's actual program and candidates; Proof independently validates returned evidence. The reference answer is not an input to the policy.

Words resolves terms such as Japan or 日本 and shows only locally explorable graph facts. Public Wikidata name resolution is labeled external context, never local evidence; Q/P IDs work offline. For live training display, use ./watch (English) or ./watch ja (Japanese); it only tails metrics.jsonl.

## Scope and limits

These latency values are a single-Jetson fixed-test evaluation. They do not provide repeated-run confidence intervals, external-SOTA comparison, or natural-language-to-relation-program evaluation. The online latency cost model was removed from the final reward after its validation error proved too large; the final result uses the safer instruction penalty. Natural-language interpretation and local-LLM answer generation are next integration steps, not claims made here.

## Further reading

- [Architecture](docs/ARCHITECTURE.md)
- [Graph format](docs/GRAPH_FORMAT.md)
- [Semantic lane contract](docs/SEMANTIC.md)
- [Operator runbook](docs/RUNBOOK.md)
- [Third-party data and licenses](THIRD_PARTY.md)
