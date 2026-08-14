# Training and recovery

日本語版: [TRAINING.ja.md](TRAINING.ja.md)

`full.toml` uses a 180,000-second (50-hour) wall-clock ceiling. The launcher always creates or resumes a detached trainer service; dashboard attachment must not affect it. Initial full semantic preparation has its own six-hour fail-closed budget. If it reaches that limit, it atomically checkpoints and exits before training starts; rerunning `./up.sh` resumes preparation. On this prepared host, this makes the normal first invocation bounded below the requested 60-hour envelope rather than allowing an unbounded setup stage.

Each checkpoint includes model, optimizer, phase, best metric, configuration, RNG state, task-generator RNG/cursor, held-out exclusion set, and phase state. A trial/full resume refuses a checkpoint that lacks task-generator state rather than silently replaying the task stream from its seed. Checkpoints are written to a same-directory temporary file, fsynced, then atomically renamed. `latest.ckpt` and `best.ckpt` are retained independently of periodic retention.

The reward record is componentized in the task adapter. A policy result receives answer credit only after the independent proof validator accepts its graph evidence. Deterministic fixed-seed validation uses held-out split seeds.

The full curriculum has concrete handlers: real CUDA score/CSR probes, observed hardware-cost samples, BC warmup, 2--3 hop RL, 4--6 hop distractors, intersection episodes, semantic-hybrid episodes, robustness distractors, Jetson-specialization PPO, and a materialized-test deterministic baseline export. Unknown phase names fail closed. Phase durations and promotion thresholds belong in the active configuration and run manifest, never in a UI.

At each entered production phase, NEUROSEEK runs the checkpointed policy on a
fixed representative set drawn from the immutable validation artifact (path,
distractor, intersection, semantic-hybrid, and robustness when available).
It writes the actual programs to `runs/<id>/traces/reference-*.json` and a
durable `traces/strategy_evolution.json` index. Reference proofs are retained
only for auditing; the policy execution receives no teacher path.

`semantic_bounded` is allowed for `--trial` only. The default full run refuses to start until `data/processed/semantic_full` passes hash and complete-alignment validation for all compact graph entity IDs. This gate is intentional: partial vectors must not be presented as full Wikidata5M semantic coverage.

In full mode a critical temperature writes a checkpoint and stops the trainer.
This is deliberate: resuming only after cooling is preferable to consuming a
fixed wall-clock budget under sustained thermal throttling.
