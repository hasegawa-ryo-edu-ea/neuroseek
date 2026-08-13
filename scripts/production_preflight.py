#!/usr/bin/env python3
"""Fast fail-closed gate for a NEUROSEEK long-running trainer.

This intentionally exercises the same immutable mmap graph and CUDA ABI used
by trial/full mode.  It emits a small JSON result only after every mandatory
check succeeds; it never creates a training run or mutates the dataset.
"""
from __future__ import annotations

import json
import hashlib
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MIN_FREE_BYTES = 24 * 1024**3
# Unified-memory Jetson needs headroom for the first real training batch and
# allocator fragmentation after the co-residency probe.  Operators can raise
# this release gate without editing code for a tighter deployment policy.
MIN_FREE_CUDA_BYTES = int(os.environ.get("NEUROSEEK_MIN_FREE_CUDA_BYTES", str(512 * 1024**2)))
sys.path.insert(0, str(ROOT / "python"))

from neuroseek.cuda_backend import CudaExactBackend
from neuroseek.data.graph import GraphMmap
from neuroseek.data.tasks import load_task_jsonl, validate_intersection_proof, validate_path_proof
from neuroseek.models.policy import NavigatorPolicy
from neuroseek.search.environment import GraphSearchEnv
from neuroseek.semantic import AlignedEmbeddings, CudaExactAnnBackend


def _validate_heldout(graph: GraphMmap, root: Path, split: dict) -> tuple[object, tuple[int, ...] | None]:
    """Bind preflight to persisted validation/test episodes, never new draws."""
    selected = None
    for name in ("validation", "test"):
        filename = split.get(name)
        if not isinstance(filename, str):
            raise RuntimeError(f"held-out manifest is missing {name} filename")
        path = root / "task_splits" / filename
        sidecar = path.with_suffix(path.suffix + ".manifest.json")
        if not sidecar.is_file():
            raise RuntimeError(f"held-out task hash sidecar is absent: {sidecar}")
        expected = json.loads(sidecar.read_text()).get("sha256")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected != actual:
            raise RuntimeError(f"held-out task hash mismatch: {path}")
        tasks = load_task_jsonl(path)
        for query, proof in tasks:
            valid = validate_intersection_proof(graph, query) if query.family == "intersection" else proof is not None and validate_path_proof(graph, query, proof)
            if not valid:
                raise RuntimeError(f"persisted held-out task has invalid proof: {query.task_id}")
        if name == "validation":
            selected = tasks[0]
    if selected is None:
        raise RuntimeError("held-out validation task set is empty")
    return selected


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("trial", "full"), default="trial")
    args = parser.parse_args(argv)
    graph_root = ROOT / "data" / "processed"
    if not (graph_root / "manifest.json").is_file():
        raise RuntimeError("processed graph manifest is absent")
    split_manifest = graph_root / "task_splits" / "manifest.json"
    if not split_manifest.is_file():
        raise RuntimeError("held-out task split manifest is absent")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory for trial/full preflight")
    graph = GraphMmap(graph_root)
    split = json.loads(split_manifest.read_text())
    if split.get("graph_manifest_sha256") != hashlib.sha256((graph_root / "manifest.json").read_bytes()).hexdigest():
        raise RuntimeError("held-out task split does not match the processed graph manifest")
    query, expected = _validate_heldout(graph, graph_root, split)
    if expected is None:
        # An intersection's proof is its real constraint edges, so there is no
        # invented linear path to replay.
        expected = ()
    env = GraphSearchEnv(graph, query, expected)
    result = None
    for instruction in env.demonstration():
        result = env.step(instruction)
        if result.done:
            break
    if result is None or not result.done or not result.answer_correct or not result.valid_proof:
        raise RuntimeError("generated mmap graph search did not produce a valid proof")
    backend = CudaExactBackend()
    if backend.device_count() < 1:
        raise RuntimeError("CUDA exact backend reports no device")
    backend.self_test()
    semantic_root = graph_root / ("semantic_full" if args.mode == "full" else "semantic_bounded")
    if not semantic_root.is_dir():
        if args.mode == "full":
            raise RuntimeError("full semantic embedding store is absent: data/processed/semantic_full must provide complete entity alignment; semantic_bounded is trial-only")
        raise RuntimeError("trial semantic embedding store is absent; run scripts/build_semantic.py")
    # A bounded fallback is allowed only because its manifest calls itself
    # partial.  Treating it as fully aligned would turn a coverage limitation
    # into a fabricated production capability.
    embeddings = AlignedEmbeddings(semantic_root, expected_graph_entities=graph.manifest.entity_count,
                                   allow_partial=args.mode != "full")
    if embeddings.manifest.source.get("graph_manifest_sha256") != hashlib.sha256((graph_root / "manifest.json").read_bytes()).hexdigest():
        raise RuntimeError("semantic embedding store belongs to a different graph manifest")
    ann_backend = CudaExactAnnBackend()
    policy = NavigatorPolicy().cuda().eval()
    # Keep the runtime's real long-lived residents co-present: full CSR,
    # mmap semantic store/ANN work buffer and policy.  This catches a unified
    # memory allocation failure that separate sequential probes would miss.
    with backend.create_graph_session(graph) as session:
        expanded = session.expand([query.source], query.relations[0])
        cpu_nodes, cpu_relations = graph.neighbors(query.source)
        cpu = np.asarray([node for node, relation in zip(cpu_nodes, cpu_relations) if int(relation) == query.relations[0]], dtype=np.uint32)
        if not np.array_equal(np.sort(expanded), np.sort(cpu)):
            raise RuntimeError("CUDA graph expansion differs from CPU CSR reference")
        ann_result = embeddings.search(np.asarray(embeddings.vectors[0], dtype=np.float32), 1, ann_backend)
        with torch.no_grad():
            logits, value = policy(torch.as_tensor(env.observation(), device="cuda").unsqueeze(0))
        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        free_cuda, total_cuda = torch.cuda.mem_get_info(0)
    if not len(expanded):
        raise RuntimeError("CUDA graph expansion returned an empty frontier for a generated real query")
    if ann_result.entity_ids.size != 1 or not np.isfinite(ann_result.scores).all():
        raise RuntimeError("CUDA exact semantic search returned an invalid result")
    if not bool(torch.isfinite(logits).all() and torch.isfinite(value).all()):
        raise FloatingPointError("policy forward emitted non-finite values")
    if free_cuda < MIN_FREE_CUDA_BYTES:
        raise RuntimeError(
            f"insufficient free CUDA memory after CSR/ANN/policy co-residency: "
            f"{free_cuda / 1024**2:.1f} MiB < {MIN_FREE_CUDA_BYTES / 1024**2:.1f} MiB"
        )
    free = shutil.disk_usage(ROOT).free
    if free < MIN_FREE_BYTES:
        raise RuntimeError(f"insufficient free disk for retained checkpoints and exports: {free / 1024**3:.1f} GiB < {MIN_FREE_BYTES / 1024**3:.0f} GiB")
    run_root = ROOT / "runs"
    if not run_root.is_dir() or not os.access(run_root, os.W_OK):
        raise RuntimeError("run directory is not writable")
    print(json.dumps({
        "ok": True,
        "device": torch.cuda.get_device_name(0),
        "entities": graph.manifest.entity_count,
        "relations": graph.manifest.relation_count,
        "proof_task": query.task_id,
        "proof_nodes": result.nodes_visited,
        "proof_edges": result.edges_examined,
        "cuda_graph_expand_candidates": int(len(expanded)),
        "ann_backend": ann_result.backend,
        "ann_vectors_examined": ann_result.vectors_examined,
        "semantic_complete_alignment": embeddings.manifest.complete_alignment,
        "semantic_indexed_entities": embeddings.manifest.indexed_entity_count,
        "cuda_allocated_bytes": allocated,
        "cuda_reserved_bytes": reserved,
        "cuda_free_bytes_after_co_residency": free_cuda,
        "cuda_total_bytes": total_cuda,
        "minimum_free_cuda_bytes": MIN_FREE_CUDA_BYTES,
        "free_disk_bytes": free,
        "minimum_free_disk_bytes": MIN_FREE_BYTES,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
