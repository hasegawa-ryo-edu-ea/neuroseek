# Semantic lane

NEUROSEEK has a symbolic CSR graph lane and a semantic entity-vector lane.  A semantic result is only a candidate jump: a later graph `VERIFY` remains responsible for factual proof.

## Artifact contract

An immutable semantic directory contains `entity_ids.u32`, row-major `embeddings.f16`, and `semantic_manifest.json`.  The manifest records the graph entity count, dimensions, normalization, source, hashes, and whether every compact graph entity ID is covered. `AlignedEmbeddings` refuses incomplete, corrupt, incorrectly sized, or wrongly ordered data by default.

Pretrained embeddings must be converted with their explicit compact entity-ID mapping. A full artifact is accepted only when its IDs are exactly `0..graph_entity_count-1`; matching vector count alone is not evidence of alignment.

## Backend selection

There is currently no cuVS/CAGRA backend in this repository. `CudaExactAnnBackend` is the production fallback: it calls the checked custom CUDA dot-product C ABI in bounded batches, then performs a deterministic host top-k merge. Its reported name is `cuda_exact`, and its stats disclose that the merge is host-side. Failure to load CUDA is an explicit error; it never falls back to NumPy.

`NumpyExactBackend` exists solely for tests and offline development. Its backend name includes `test_only` so it cannot be mistaken for GPU evidence.

## Bounded TransE fallback

`train_bounded_transe(graph, output, TransEConfig(...))` trains a compact TransE-style model over a deterministic induced subset of mmap CSR entities. It uses sparse PyTorch embeddings, bounded steps and bounded entity count to protect the 8 GB Jetson. The resulting artifact is intentionally marked partial and callers must pass `allow_partial=True`; it is suitable for trial semantic jumps, not a claim that all Wikidata5M entities are embedded. The source manifest records the seed, steps actually containing valid subset edges, final observed loss, and device.

Full mode uses sparse CUDA TransE over every compact entity ID and writes a
step-zero recovery checkpoint before its first batch, then atomically publishes
periodic progress checkpoints. A real Jetson container probe successfully
allocated and CPU-exported the 4,594,149 x 64 FP32 entity table; the measured
GPU allocation was 1,176,502,272 bytes and the CPU snapshot was 1,176,102,144
bytes. The bounded probe record is
`artifacts/semantic_checkpoint_memory_probe.json`; it is an allocation/export
test, not a claimed full training result.

`artifacts/full_semantic_resume_probe.json` records a separate CUDA
failure-injection acceptance test: the explicit test-only CLI interruption
wrote the step-zero checkpoint, an ordinary subsequent full build loaded it,
published a complete store, and removed the progress checkpoint only after
publication.

`artifacts/full_semantic_sigterm_probe.json` records the operational signal
path: SIGTERM requests a safe stop, the next full-TransE loop boundary writes
the atomic progress checkpoint, and the process exits with the conventional
signal-derived status. The following ordinary full invocation resumes it.

After graph compilation, the reproducible command-line entry point is:

```bash
python3 scripts/build_semantic.py --graph data/processed --output data/processed/semantic_bounded
```

It runs the bounded CUDA TransE fallback with explicit defaults (64 dimensions,
100,000 entities, 1,000 steps), writes the processed-graph manifest hash into
the embedding source metadata, validates hashes on reuse, and refuses an
implicit replacement. Use `--replace` only to explicitly regenerate that
derived cache.

Before a full-run semantic lane is enabled, use a fully aligned pretrained artifact or explicitly produce and validate full coverage. Do not present the bounded fallback as full Wikidata5M semantic coverage.

`./up.sh --trial` builds/reuses `semantic_bounded` transparently and records its
partial coverage in the run manifest. `./up.sh` deliberately does **not** do
that: it requires `data/processed/semantic_full` and rejects every manifest
whose compact IDs are not exactly `0..graph_entity_count-1`. This makes a
missing full semantic asset a visible release gate instead of silently changing
the research question.

The first full build uses sparse-SGD TransE and writes an atomic trainable-table
checkpoint every 10,000 optimizer steps at
`data/processed/.semantic_full.training.ckpt`. A repeated `./up.sh` resumes
that exact graph/configuration checkpoint; an incompatible or corrupt checkpoint
fails visibly rather than producing a mixed artifact. The checkpoint is removed
only after the complete FP16 artifact and manifest are atomically published.
