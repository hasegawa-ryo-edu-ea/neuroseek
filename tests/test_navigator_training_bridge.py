from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch
import neuroseek.training.trainer as trainer_module

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "python"))

from neuroseek.data.graph import GraphMmap
from neuroseek.data.tasks import TaskGenerator
from neuroseek.models.policy import NavigatorPolicy
from neuroseek.search.environment import GraphSearchEnv
from neuroseek.training.trainer import _behavior_clone, _navigator_ranker


def _graph(tmp_path: Path) -> GraphMmap:
    triples = tmp_path / "triples.tsv"
    triples.write_text("Q1\tP1\tQ2\nQ1\tP1\tQ3\nQ2\tP2\tQ4\nQ3\tP2\tQ4\n")
    output = tmp_path / "graph"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "preprocess.py"), "--input", str(triples), "--output", str(output)], check=True)
    return GraphMmap(output)


def test_real_frontier_topk_uses_bounded_navigator_and_bc_backpropagates(tmp_path: Path) -> None:
    graph = _graph(tmp_path)
    model = NavigatorPolicy()
    generator = TaskGenerator(graph, "train", 17, 2, 2)
    query, proof = generator.next()
    env = GraphSearchEnv(graph, query, proof, candidate_ranker=_navigator_ranker(model, graph, torch.device("cpu"), 16))
    env.step(0)
    env.step(2)
    result = env.step(8)
    assert result.navigator_ranked_candidates > 0
    assert "NAVIGATOR_RANK" in " ".join(result.trace)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    metrics = _behavior_clone(model, optimizer, graph, TaskGenerator(graph, "train", 18, 2, 2), torch.device("cpu"), 2, None, 4096, 16)
    assert metrics["navigator_loss"] >= 0.0
    assert metrics["success_rate"] == 1.0
    assert metrics["proof_validity"] == 1.0
    assert metrics["live_search_result"] == "VALID"
    assert any(parameter.grad is not None for parameter in model.message_layers.parameters())


def test_behavior_cloning_skips_auxiliary_label_outside_bounded_candidates(tmp_path: Path, monkeypatch) -> None:
    """A proof child outside the local GNN cap must not crash the trainer."""
    graph = _graph(tmp_path)
    model = NavigatorPolicy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    original = trainer_module._navigator_local_scores

    def omit_teacher(*args, **kwargs):
        _candidates, scores = original(*args, **kwargs)
        # The exact teacher identity is deliberately unavailable to the local
        # classifier while remaining in the real environment frontier.
        return [], scores[:0]

    monkeypatch.setattr(trainer_module, "_navigator_local_scores", omit_teacher)
    metrics = _behavior_clone(model, optimizer, graph, TaskGenerator(graph, "train", 19, 2, 2),
                              torch.device("cpu"), 2, None, 4096, 16)
    assert metrics["navigator_loss"] == 0.0
    assert metrics["success_rate"] == 1.0
