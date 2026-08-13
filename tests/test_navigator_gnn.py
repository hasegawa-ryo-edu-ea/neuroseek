"""Unit tests for the bounded query-conditioned relational navigator."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "python"))

from neuroseek.models.policy import LocalCandidateSubgraph, NavigatorPolicy, OP_NAMES


def _subgraph(*, relation_ids: torch.Tensor | None = None, node_batch: torch.Tensor | None = None) -> LocalCandidateSubgraph:
    # Includes a converging path, a duplicate destination, and an isolated
    # candidate; these catch the common scatter/count edge cases.
    nodes = torch.tensor(
        [[1.0, 0.0, 0.5, 0.0], [0.2, 0.7, 0.0, 0.1], [0.3, 0.0, 0.2, 0.9], [0.0, 0.1, 0.0, 1.0], [0.8, 0.2, 0.1, 0.0]],
        dtype=torch.float32,
    )
    return LocalCandidateSubgraph(
        node_features=nodes,
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 2, 2, 3]], dtype=torch.long),
        edge_relations=relation_ids if relation_ids is not None else torch.tensor([0, 1, 1, 2], dtype=torch.long),
        candidate_indices=torch.tensor([2, 3, 4], dtype=torch.long),
        node_batch=node_batch,
    )


def test_legacy_forward_and_candidate_api_are_preserved() -> None:
    model = NavigatorPolicy(feature_dim=4, hidden_dim=12, num_relations=3)
    state = torch.randn(2, 4)
    logits, value = model(state)
    assert logits.shape == (2, len(OP_NAMES))
    assert value.shape == (2,)
    encoded_candidates = torch.randn(2, 12)
    assert model.score_candidates(state, encoded_candidates).shape == (2,)
    # The original module also accepted one state/candidate vector directly.
    assert model.score_candidates(state[0], encoded_candidates[0]).ndim == 0
    # New raw candidate support broadcasts one query over a local list.
    assert model.score_candidates(state[:1], torch.randn(5, 4)).shape == (5,)


def test_local_relational_scores_are_finite_and_differentiable() -> None:
    torch.manual_seed(7)
    model = NavigatorPolicy(feature_dim=4, hidden_dim=16, num_relations=3, message_passing_steps=2)
    scores = model.score_local_candidates(torch.tensor([0.3, 0.6, 0.1, 0.4]), _subgraph())
    assert scores.shape == (3,)
    assert torch.isfinite(scores).all()
    loss = scores.square().mean()
    loss.backward()
    # Every component on the local path has a real, finite gradient.
    for parameter in (
        model.node_encoder[0].weight,
        model.query_encoder[0].weight,
        model.relation_embedding.weight,
        model.message_layers[0].source.weight,
        model.local_candidate_score[-1].weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_relation_ids_affect_message_passing_scores() -> None:
    torch.manual_seed(11)
    model = NavigatorPolicy(feature_dim=4, hidden_dim=16, num_relations=3).eval()
    query = torch.tensor([0.1, 0.2, 0.3, 0.4])
    first = model.score_local_candidates(query, _subgraph(relation_ids=torch.tensor([0, 0, 0, 0], dtype=torch.long)))
    second = model.score_local_candidates(query, _subgraph(relation_ids=torch.tensor([2, 2, 2, 2], dtype=torch.long)))
    assert not torch.allclose(first, second)


def test_batched_queries_map_only_to_their_local_nodes() -> None:
    torch.manual_seed(3)
    model = NavigatorPolicy(feature_dim=4, hidden_dim=12, num_relations=3)
    graph = _subgraph(node_batch=torch.tensor([0, 0, 0, 1, 1], dtype=torch.long))
    scores = model.score_local_candidates(torch.randn(2, 4), graph)
    assert scores.shape == (3,)
    assert torch.isfinite(scores).all()


def test_bad_local_graph_is_rejected_before_scoring() -> None:
    model = NavigatorPolicy(feature_dim=4, hidden_dim=12, num_relations=3)
    graph = _subgraph(relation_ids=torch.tensor([9, 0, 1, 2], dtype=torch.long))
    with pytest.raises(ValueError, match="outside configured"):
        model.score_local_candidates(torch.randn(4), graph)
    graph = _subgraph()
    graph = LocalCandidateSubgraph(graph.node_features, graph.edge_index.to(torch.int32), graph.edge_relations, graph.candidate_indices)
    with pytest.raises(ValueError, match="edge_index"):
        model.score_local_candidates(torch.randn(4), graph)
