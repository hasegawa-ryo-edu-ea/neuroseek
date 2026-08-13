# NEUROSEEK

NEUROSEEK is a learned GPU knowledge processor for a Jetson Orin Nano. It couples memory-mapped Wikidata5M graph traversal, a Rust search VM/proof validator, custom CUDA primitives, and a compact PyTorch policy trainer.

From this directory, start the real production experiment with:

```bash
./up.sh
```

The trainer is detached; closing SSH only closes the dashboard. Re-run `./up.sh`
to reattach: when a trainer is already running, this is a read-only dashboard
attach and does not launch another CUDA preflight or compete for GPU memory.
On a first run, wait for `NEUROSEEK trainer detached:` (or the TUI) before
disconnecting. The earlier full-semantic preparation is a foreground setup
step; if that SSH session is interrupted, rerunning `./up.sh` resumes its
atomic semantic checkpoint before starting the trainer.

Useful bounded commands:

```bash
./up.sh --doctor
./up.sh --smoke
./up.sh --trial
./up.sh --status
./down.sh
```

`--smoke` is explicitly synthetic and bounded. `--trial` requires the processed canonical Wikidata5M graph and is bounded. On its first full invocation, `./up.sh` also builds a streamed, fully aligned 64-dimensional sparse-TransE artifact for every compact Wikidata5M entity ID before the 50-hour schedule begins; this is a real prerequisite and may take substantial time. It does not alter JetPack, firmware, kernel, or power mode.

The production image is pinned to an NVIDIA L4T ML CUDA 12.2-generation
container matched to this R36.3 Jetson. Startup runs custom CUDA parity in the
container; GPU visibility alone is deliberately not accepted as compatibility.

See `docs/RUNBOOK.md` for operations, `docs/GRAPH_FORMAT.md` for storage, `docs/NEURO_ISA.md` for VM semantics, and `docs/EXPERIMENT_DESIGN.md` for evaluation discipline.
