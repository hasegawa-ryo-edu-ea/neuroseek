# Operator runbook

日本語版: [RUNBOOK.ja.md](RUNBOOK.ja.md)

Use `./up.sh --doctor` before a large experiment. `./up.sh --smoke` is the bounded synthetic verification path. `./up.sh --trial` uses a deterministic subgraph compiled from the real dataset and is bounded. `./up.sh` starts or resumes full detached production training and attaches a reader when interactive.

Use `./status.sh` to inspect the detached service and `./logs.sh` for logs. To
return to the live terminal dashboard after an SSH reconnect, run `./up.sh`
again. A running trainer is detected before any build, semantic preparation,
or CUDA preflight, so reattachment does not contend with the training GPU.
On the first invocation, do not intentionally disconnect until the launcher
prints `NEUROSEEK trainer detached:` or presents the TUI: full-semantic
preparation precedes that detached service. An interrupted preparation has an
atomic checkpoint and resumes safely on the next `./up.sh`.
`./down.sh` sends a graceful stop and preserves all persistent data. Never
delete `data/processed/` or `runs/` to recover from a failure; inspect the
crash report and latest checkpoint first.
