#!/usr/bin/env python3
"""Re-evaluate two immutable NEUROSEEK checkpoints for presentation analysis.

This tool does not alter checkpoints, manifests, or the canonical final
exports.  It writes a separate, auditable task-level comparison.  The original
final-evaluation latency aggregate remains the authoritative latency result;
the recheck is intended for outcome and search-cost breakdowns by task family.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch

from neuroseek.cuda_backend import CudaExactBackend
from neuroseek.data.graph import GraphMmap
from neuroseek.data.tasks import load_task_jsonl
from neuroseek.models.policy import NavigatorPolicy
from neuroseek.semantic import AlignedEmbeddings, CudaExactAnnBackend
from neuroseek.training.checkpoint import load_checkpoint
from neuroseek.training.trainer import _learned_evaluation_result, _semantic_search_callback


METRICS = ("nodes_visited", "edges_examined", "search_steps", "compute_credits")


def evaluate_checkpoint(checkpoint: Path, task_rows: list[tuple[object, tuple[int, ...] | None]], graph: GraphMmap,
                        device: torch.device, cuda_session: object, semantic_search: object | None) -> dict[str, object]:
    payload = load_checkpoint(checkpoint, device)
    if payload is None:
        raise RuntimeError(f"cannot load checkpoint: {checkpoint}")
    model = NavigatorPolicy().to(device)
    model.load_state_dict(payload["model"])
    rows: dict[str, object] = {}
    for query, proof in task_rows:
        result, _trace, _ranked = _learned_evaluation_result(
            model, graph, query, proof, device, cuda_session, 262144, semantic_search, 256
        )
        rows[query.task_id] = result
    return rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--derived", required=True, type=Path)
    parser.add_argument("--tasks", default=Path("data/processed/task_splits/test_v2.jsonl"), type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; do not silently fall back to CPU")
    graph = GraphMmap("data/processed")
    embeddings = AlignedEmbeddings("data/processed/semantic_full", expected_graph_entities=graph.manifest.entity_count)
    backend = CudaExactBackend(); backend.self_test()
    session = backend.create_graph_session(graph)
    ann_backend = CudaExactAnnBackend()
    semantic_search = _semantic_search_callback(embeddings, ann_backend)
    task_rows = load_task_jsonl(args.tasks)
    try:
        parent = evaluate_checkpoint(args.parent, task_rows, graph, device, session, semantic_search)
        derived = evaluate_checkpoint(args.derived, task_rows, graph, device, session, semantic_search)
    finally:
        session.close()

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    fields = ["task_id", "family", "parent_correct", "derived_correct", "parent_valid_proof", "derived_valid_proof"]
    fields += [f"parent_{name}" for name in METRICS] + [f"derived_{name}" for name in METRICS] + [f"delta_{name}" for name in METRICS]
    records: list[dict[str, object]] = []
    for query, _proof in task_rows:
        old, new = parent[query.task_id], derived[query.task_id]
        row: dict[str, object] = {"task_id": query.task_id, "family": query.family,
                                  "parent_correct": old.answer_correct, "derived_correct": new.answer_correct,
                                  "parent_valid_proof": old.valid_proof, "derived_valid_proof": new.valid_proof}
        for name in METRICS:
            row[f"parent_{name}"] = getattr(old, name)
            row[f"derived_{name}"] = getattr(new, name)
            row[f"delta_{name}"] = getattr(new, name) - getattr(old, name)
        records.append(row)
    with (output / "task_level_recheck.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(records)

    by_family: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in records:
        by_family[str(row["family"])].append(row)
    summary: dict[str, object] = {"task_count": len(records), "families": {}}
    for family, rows in sorted(by_family.items()):
        entry: dict[str, object] = {"task_count": len(rows),
                                     "parent_accuracy": mean([float(row["parent_correct"]) for row in rows]),
                                     "derived_accuracy": mean([float(row["derived_correct"]) for row in rows]),
                                     "parent_proof_validity": mean([float(row["parent_valid_proof"]) for row in rows]),
                                     "derived_proof_validity": mean([float(row["derived_valid_proof"]) for row in rows])}
        for name in METRICS:
            entry[f"parent_mean_{name}"] = mean([float(row[f"parent_{name}"]) for row in rows])
            entry[f"derived_mean_{name}"] = mean([float(row[f"derived_{name}"]) for row in rows])
            entry[f"delta_mean_{name}"] = mean([float(row[f"delta_{name}"]) for row in rows])
        summary["families"][family] = entry
    summary["note"] = "This is a post-completion structural recheck. Canonical end-to-end latency is reported only from each run's original final evaluation."
    (output / "task_family_recheck.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
