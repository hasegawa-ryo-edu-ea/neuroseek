from __future__ import annotations

import subprocess
import sys
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "python"))

from neuroseek.data.graph import GraphMmap
from neuroseek.data.tasks import QuerySpec
from neuroseek.models.policy import NavigatorPolicy
from neuroseek.semantic import AlignedEmbeddings, NumpyExactBackend, write_embedding_store
from neuroseek.training.trainer import _capture_reference_traces, _run_final_evaluation


class _ScriptedNavigator(NavigatorPolicy):
    """Exercise the evaluator's actual local-GNN TOPK path deterministically."""
    def forward(self, state: torch.Tensor):
        logits = torch.full((state.shape[0], 12), -100.0, device=state.device)
        seed = state[:, 0] == 0
        expand = (~seed) & (state[:, 1] == 0)
        logits[seed, 0] = 100.0          # SEED
        logits[expand, 2] = 100.0        # EXPAND_REL
        logits[~(seed | expand), 8] = 100.0  # TOPK -> invokes Navigator GNN
        return logits, torch.zeros((state.shape[0], 1), device=state.device)


def test_final_evaluation_exports_complete_real_result_set(tmp_path: Path):
    triples = tmp_path / "triples.tsv"
    triples.write_text("Q1\tP1\tQ2\nQ2\tP2\tQ3\nQ1\tP3\tQ3\n", encoding="utf-8")
    graph_root = tmp_path / "graph"
    subprocess.run([sys.executable, str(ROOT / "scripts/preprocess.py"), "--input", str(triples), "--output", str(graph_root)], check=True)
    graph = GraphMmap(graph_root)
    semantic_root = tmp_path / "semantic"
    write_embedding_store(semantic_root, np.arange(3, dtype=np.uint32), np.eye(3, dtype=np.float32), graph_entity_count=3,
                          source={"kind": "unit-test"})
    embeddings = AlignedEmbeddings(semantic_root, expected_graph_entities=3)
    query = QuerySpec("heldout-path", "test", 1, 0, 2, (0, 1), 32)
    run_dir = tmp_path / "run"
    metrics = _run_final_evaluation(_ScriptedNavigator(), graph, [(query, (0, 1, 2))], torch.device("cpu"), None,
                                    4096, embeddings, NumpyExactBackend(), None, run_dir, {"test": "bound"},
                                    evaluated_checkpoint={"filename": "unit.ckpt", "sha256": "unit"})
    assert metrics["evaluation_task_count"] == 1.0
    assert metrics["learned_navigator_ranked_candidates"] > 0
    for name in ("final_metrics.json", "benchmark_comparison.csv", "training_curve.csv", "strategy_evolution.json",
                 "reference_query_traces.json", "hardware_summary.json", "operator_distribution.json", "proof_examples.json",
                 "phase_summary.json", "final_model_manifest.json"):
        assert (run_dir / "exports" / name).is_file()
    manifest = json.loads((run_dir / "exports" / "final_model_manifest.json").read_text())
    assert manifest["checkpoint"] == {"filename": "unit.ckpt", "sha256": "unit"}


def test_reference_trace_capture_is_phase_indexed_and_durable(tmp_path: Path):
    triples = tmp_path / "triples.tsv"
    triples.write_text("Q1\tP1\tQ2\nQ2\tP2\tQ3\n", encoding="utf-8")
    graph_root = tmp_path / "graph"
    subprocess.run([sys.executable, str(ROOT / "scripts/preprocess.py"), "--input", str(triples), "--output", str(graph_root)], check=True)
    graph = GraphMmap(graph_root)
    query = QuerySpec("reference-path", "validation", 1, 0, 2, (0, 1), 32)
    destination = _capture_reference_traces(NavigatorPolicy(), graph, [(query, (0, 1, 2))], torch.device("cpu"), None,
                                            4096, None, 16, tmp_path / "run", phase="behavior_cloning",
                                            global_step=0, elapsed_seconds=0.0)
    record = json.loads(destination.read_text())
    index = json.loads((destination.parent / "strategy_evolution.json").read_text())
    assert record["phase"] == "behavior_cloning"
    assert record["traces"][0]["task"]["task_id"] == "reference-path"
    assert index[0]["file"] == destination.name
