#!/usr/bin/env python3
"""Fit a NEUROSEEK latency predictor from real CUDA/Search-VM JSONL records.

Example: ./build/cuda/neuroseek_cuda_bench > runs/current/hardware.jsonl
         python scripts/bench_cost_model.py --input runs/current/hardware.jsonl \
             --output runs/current/hardware_cost_model.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from neuroseek.cost_model.model import CostModelError, OperationRecord, load_records, save_model, train_cost_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--input", type=Path, action="append", help="Measured JSONL file; repeatable")
    sources.add_argument("--stdin", action="store_true", help="Read measured JSONL from standard input")
    parser.add_argument("--output", type=Path, required=True, help="Model artifact JSON path")
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--skip-invalid", action="store_true", help="Skip unrelated runtime events lacking latency instead of failing")
    args = parser.parse_args(argv)
    try:
        if args.stdin:
            # Preserve stdin in a private temporary file so ``load_records``
            # retains per-line provenance and exactly one parser is used.
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".jsonl") as temporary:
                temporary.write(sys.stdin.read())
                temporary.flush()
                records = load_records([Path(temporary.name)], strict=not args.skip_invalid)
                records = [OperationRecord(item.operation, item.latency_ms, item.features,
                                           source="<stdin>", line=item.line) for item in records]
        else:
            records = load_records(args.input, strict=not args.skip_invalid)
        model = train_cost_model(records, ridge=args.ridge, validation_fraction=args.validation_fraction)
        save_model(model, args.output, records=records)
    except (OSError, CostModelError) as exc:
        print(f"cost-model: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output), "records": len(records), "operations": list(model.operations),
                      "training_records": model.training_records, "validation_records": model.validation_records,
                      "metrics": model.metrics}, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
