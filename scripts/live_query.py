#!/usr/bin/env python3
"""One real, CPU-only NEUROSEEK policy search for the presentation console.

This is intentionally a query *worker*, not a service.  It loads a named,
immutable checkpoint and graph in read-only mode, executes the learned policy
against a materialized validation task, and prints one JSON result.  It never
writes a run directory, telemetry event, checkpoint, or graph artifact.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# The invocation container has CUDA explicitly hidden.  Importing Torch after
# this declaration prevents a video/demo query from joining the trainer's GPU.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from neuroseek.data.graph import GraphMmap
from neuroseek.data.tasks import load_task_jsonl
from neuroseek.models.policy import NavigatorPolicy, OP_NAMES
from neuroseek.search.environment import GraphSearchEnv
from neuroseek.training.checkpoint import load_checkpoint


DEFAULT_CHECKPOINT = Path("runs/presentation-stabilized-20260812T1520EDT/checkpoints/latest.ckpt")
DEFAULT_TASKS = Path("data/processed/task_splits/validation_v2.jsonl")


def _name(graph: GraphMmap, entity: int) -> dict[str, object]:
    return {"id": entity, "identifier": graph.entity_identifier(entity), "label": graph.entity_label(entity)}


def _relation(graph: GraphMmap, relation: int) -> dict[str, object]:
    return {"id": relation, "identifier": graph.relation_identifier(relation), "label": graph.relation_label(relation)}


def _checkpoint_path(value: str | None) -> Path:
    path = Path(value) if value else DEFAULT_CHECKPOINT
    if not path.is_file():
        raise FileNotFoundError(f"immutable presentation checkpoint is absent: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=0, help="zero-based immutable validation task index")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--checkpoint", help="explicit immutable checkpoint; defaults to the presentation baseline")
    parser.add_argument("--max-steps", type=int, default=12)
    args = parser.parse_args()
    if args.index < 0 or args.max_steps < 1:
        raise SystemExit("--index must be non-negative and --max-steps must be positive")
    if not args.tasks.is_file():
        raise FileNotFoundError(f"task artifact is absent: {args.tasks}")

    # CPU is an intentional coexistence boundary, not a fallback.  The worker
    # runs an actual exported policy but never creates a CUDA context.
    device = torch.device("cpu")
    checkpoint = _checkpoint_path(args.checkpoint)
    state = load_checkpoint(checkpoint, device)
    model = NavigatorPolicy().to(device)
    model.load_state_dict(state["model"])
    model.eval()

    graph = GraphMmap("data/processed")
    tasks = load_task_jsonl(args.tasks)
    query, _demonstration = tasks[args.index % len(tasks)]
    env = GraphSearchEnv(graph, query, (), cuda_session=None)
    steps: list[dict[str, object]] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for index in range(args.max_steps):
            observation = torch.as_tensor(env.observation(), dtype=torch.float32, device=device).unsqueeze(0)
            logits, value = model(observation)
            probabilities = torch.softmax(logits[0], dim=0)
            action = int(torch.argmax(probabilities).item())
            result = env.step(action)
            frontier = sorted(env.frontier)
            # The viewer gets a small evidence window, not an unbounded graph dump.
            candidates = [_name(graph, entity) for entity in frontier[:5]]
            steps.append({
                "index": index + 1,
                "operator": OP_NAMES[action],
                "policy_probability": round(float(probabilities[action]), 7),
                "value": round(float(value[0]), 7),
                "trace": result.trace[-1] if result.trace else "",
                "frontier_size": len(frontier),
                "frontier_sample": candidates,
                "credits": result.credits,
                "nodes_visited": result.nodes_visited,
                "edges_examined": result.edges_examined,
            })
            if result.done:
                break
        if not env.done:
            result = env.step(OP_NAMES.index("STOP"))
            steps.append({"index": len(steps) + 1, "operator": "STOP", "policy_probability": None,
                          "value": None, "trace": result.trace[-1], "frontier_size": len(env.frontier),
                          "frontier_sample": [_name(graph, entity) for entity in sorted(env.frontier)[:5]],
                          "credits": result.credits, "nodes_visited": result.nodes_visited,
                          "edges_examined": result.edges_examined})
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    proof = [_name(graph, entity) for entity in env.proof_path]
    payload = {
        "event": "live_model_search",
        "mode": "cpu_read_only_policy",
        "checkpoint": str(checkpoint),
        "checkpoint_global_step": int(state["global_step"]),
        "task_source": str(args.tasks),
        "task_index": args.index % len(tasks),
        "query": {
            "task_id": query.task_id,
            "family": query.family,
            "source": _name(graph, query.source),
            "relations": [_relation(graph, relation) for relation in query.relations],
            "budget": query.budget,
        },
        # Reference answer is deliberately separated from the policy input.
        "reference_answer": _name(graph, query.answer),
        "steps": steps,
        "outcome": {
            "answer": _name(graph, env.answer) if env.answer is not None else None,
            "proof_path": proof,
            "valid_proof": result.valid_proof,
            "answer_correct": result.answer_correct,
            "reward": result.reward,
            "credits": result.credits,
            "nodes_visited": result.nodes_visited,
            "edges_examined": result.edges_examined,
            "elapsed_ms": round(elapsed_ms, 3),
        },
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
