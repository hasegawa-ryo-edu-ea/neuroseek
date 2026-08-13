#!/usr/bin/env python3
"""Create immutable, deterministic validation/test episodes for NEUROSEEK."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from neuroseek.data.graph import GraphMmap
from neuroseek.evaluation.baselines import materialize_heldout_tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    if args.count <= 0:
        raise SystemExit("--count must be positive")
    graph = GraphMmap(args.graph)
    destination = args.graph / "task_splits"
    source_manifest = args.graph / "manifest.json"
    payload = {
        "format": 2,
        "graph_manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
        "validation": "validation_v2.jsonl",
        "test": "test_v2.jsonl",
        "count_each": args.count,
        "seed": args.seed,
        "families": ["path", "distractor", "intersection", "semantic_hybrid", "robustness"],
    }
    marker = destination / "manifest.json"
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if marker.exists() and marker.read_text() != encoded:
        previous = json.loads(marker.read_text())
        # v1 artifacts remain on disk for provenance.  The v2 manifest points
        # to new filenames, so this migration never overwrites an evaluation
        # set that was used by an earlier trial.
        if previous.get("format") != 1 or previous.get("graph_manifest_sha256") != payload["graph_manifest_sha256"]:
            raise SystemExit("refuse to replace task split manifest generated for a different graph or format")
    # Validate identity before writing either split.  A caller changing count
    # or seed cannot silently replace one half of an existing held-out set.
    validation = materialize_heldout_tasks(graph, destination / payload["validation"], count=args.count,
                                           seed=args.seed, split="validation")
    test = materialize_heldout_tasks(graph, destination / payload["test"], count=args.count,
                                     seed=args.seed ^ 0x5EED, split="test")
    marker.write_text(encoded)
    print(json.dumps({"ok": True, **payload}, sort_keys=True))


if __name__ == "__main__":
    main()
