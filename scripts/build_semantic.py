#!/usr/bin/env python3
"""Build NEUROSEEK semantic artifacts with explicit coverage contracts.

This is deliberately separate from graph compilation.  It never changes the
immutable CSR and it will not overwrite a published embedding directory
without an explicit ``--replace``.  The full-run gate must not mistake this
bounded TransE artifact for a fully aligned Wikidata5M embedding release.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import signal
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from neuroseek.data.graph import GraphMmap
from neuroseek.semantic import (AlignedEmbeddings, SemanticInterrupted, SemanticTestInterruption, TransEConfig,
                                train_bounded_transe, train_full_transe)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "processed" / "semantic_bounded")
    parser.add_argument("--full", action="store_true", help="train/export every graph entity; required for full production mode")
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--max-entities", type=int, default=100_000)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--checkpoint-every-steps", type=int, default=0,
                        help="full TransE only: atomically retain trainable-table progress at this interval")
    parser.add_argument("--max-wall-seconds", type=float, default=0.0,
                        help="full TransE only: checkpoint and stop safely after this preparation budget (0 disables)")
    parser.add_argument("--test-stop-after-initial-checkpoint", action="store_true",
                        help="TEST ONLY: stop after the durable full-TransE step-zero checkpoint")
    parser.add_argument("--replace", action="store_true",
                        help="explicitly replace this generated semantic artifact")
    args = parser.parse_args()
    interrupted: dict[str, int | None] = {"signal": None}
    def request_stop(signum: int, _frame: object) -> None:
        interrupted["signal"] = signum
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    if not (args.graph / "manifest.json").is_file():
        raise SystemExit(f"graph manifest is absent: {args.graph}")
    if args.dimension <= 0 or args.max_entities < 2 or args.steps <= 0 or args.batch_size <= 0 or args.checkpoint_every_steps < 0:
        raise SystemExit("dimension, max-entities, steps, and batch-size must be positive")
    if args.max_wall_seconds < 0.0:
        raise SystemExit("max-wall-seconds must be non-negative")
    if args.full and args.max_entities != 100_000:
        # Full coverage is inferred from graph manifest, never a fragile count
        # argument that could accidentally publish a partial artifact.
        raise SystemExit("--full does not accept --max-entities; coverage is the processed graph entity count")
    if args.test_stop_after_initial_checkpoint and not args.full:
        raise SystemExit("--test-stop-after-initial-checkpoint requires --full")
    if args.test_stop_after_initial_checkpoint and args.replace:
        raise SystemExit("test interruption cannot be combined with --replace")
    graph_hash = hashlib.sha256((args.graph / "manifest.json").read_bytes()).hexdigest()
    deadline = time.monotonic() + args.max_wall_seconds if args.max_wall_seconds else None
    deadline_exceeded = {"value": False}
    def stop_requested() -> bool:
        if interrupted["signal"] is not None:
            return True
        if deadline is not None and time.monotonic() >= deadline:
            deadline_exceeded["value"] = True
            return True
        return False
    train_kwargs = {"stop_requested": stop_requested} if args.full else {}
    if args.output.exists():
        if not args.replace:
            # A valid existing store is reusable; a corrupt one is never
            # silently trusted or overwritten.
            store = AlignedEmbeddings(args.output,
                                      expected_graph_entities=GraphMmap(args.graph).manifest.entity_count,
                                      allow_partial=not args.full)
            if store.manifest.source.get("graph_manifest_sha256") != graph_hash:
                raise SystemExit("existing semantic artifact belongs to a different processed graph; use --replace to regenerate")
            print(json.dumps({"ok": True, "reused": True, "output": str(args.output),
                              "complete_alignment": store.manifest.complete_alignment,
                              "indexed_entity_count": store.manifest.indexed_entity_count}, sort_keys=True))
            return 0
        backup = args.output.with_name(args.output.name + ".replaced-backup")
        if backup.exists():
            raise SystemExit(f"refusing replacement while retained backup exists: {backup}")
        args.output.replace(backup)
        try:
            trainer = train_full_transe if args.full else train_bounded_transe
            path = trainer(GraphMmap(args.graph), args.output, TransEConfig(
                dimension=args.dimension, max_entities=args.max_entities, steps=args.steps,
                batch_size=args.batch_size, seed=args.seed, device=args.device, graph_manifest_sha256=graph_hash,
                checkpoint_every_steps=args.checkpoint_every_steps,
            ), **train_kwargs, **({"test_stop_after_initial_checkpoint": True} if args.full and args.test_stop_after_initial_checkpoint else {}))
        except Exception:
            args.output.replace(backup)
            raise
        shutil.rmtree(backup)
    else:
        trainer = train_full_transe if args.full else train_bounded_transe
        try:
            path = trainer(GraphMmap(args.graph), args.output, TransEConfig(
                dimension=args.dimension, max_entities=args.max_entities, steps=args.steps,
                batch_size=args.batch_size, seed=args.seed, device=args.device, graph_manifest_sha256=graph_hash,
                checkpoint_every_steps=args.checkpoint_every_steps,
            ), **train_kwargs, **({"test_stop_after_initial_checkpoint": True} if args.full and args.test_stop_after_initial_checkpoint else {}))
        except SemanticTestInterruption as exc:
            checkpoint = args.output.parent / f".{args.output.name}.training.ckpt"
            print(json.dumps({"ok": False, "test_interruption": True, "checkpoint": str(checkpoint),
                              "reason": str(exc)}, sort_keys=True))
            return 75
        except SemanticInterrupted as exc:
            checkpoint = args.output.parent / f".{args.output.name}.training.ckpt"
            if deadline_exceeded["value"]:
                print(json.dumps({"ok": False, "preparation_time_limit_reached": True,
                                  "max_wall_seconds": args.max_wall_seconds, "checkpoint": str(checkpoint),
                                  "reason": str(exc)}, sort_keys=True))
                return 124
            print(json.dumps({"ok": False, "interrupted": True, "signal": interrupted["signal"],
                              "checkpoint": str(checkpoint), "reason": str(exc)}, sort_keys=True))
            return 128 + int(interrupted["signal"] or signal.SIGTERM)
    store = AlignedEmbeddings(path, expected_graph_entities=GraphMmap(args.graph).manifest.entity_count,
                              allow_partial=not args.full)
    print(json.dumps({"ok": True, "reused": False, "output": str(path), "graph_manifest_sha256": graph_hash,
                      "complete_alignment": store.manifest.complete_alignment,
                      "indexed_entity_count": store.manifest.indexed_entity_count,
                      "dimension": store.manifest.dimension, "source": store.manifest.source}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
