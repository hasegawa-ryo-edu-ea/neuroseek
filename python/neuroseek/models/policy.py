"""Query-conditioned navigator and hierarchical search strategist.

The policy intentionally never materialises the full Wikidata graph.  The
``score_local_candidates`` path receives a *bounded*, already sampled local
subgraph from the search runtime and performs relation-aware message passing
only over that subgraph.  ``forward`` remains the compact state-only API used
by the existing PPO/BC loop.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


OP_NAMES = (
    "SEED", "ANN", "EXPAND_REL", "EXPAND_ANY", "FILTER", "INTERSECT",
    "UNION", "PRUNE", "TOPK", "VERIFY", "BACKTRACK", "STOP",
)


@dataclass(frozen=True)
class LocalCandidateSubgraph:
    """Compact tensors describing a sampled local graph.

    ``edge_index`` has shape ``[2, E]`` and uses ``source, destination`` row
    order.  ``edge_relations`` is either relation ids ``[E]`` (``torch.long``)
    or explicit relation features ``[E, feature_dim]``.  ``node_batch`` is
    optional; when supplied it maps every node to one query row.

    This is deliberately only a tensor carrier.  Validation stays in the model
    boundary so malformed CUDA/Rust output cannot silently produce a plausible
    score.
    """

    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_relations: torch.Tensor
    candidate_indices: torch.Tensor
    node_batch: torch.Tensor | None = None


class _RelationalMessageLayer(nn.Module):
    """Small query-gated relational aggregation layer.

    Relation embeddings are added to source messages rather than using one
    dense matrix per relation.  That keeps the 822 Wikidata relations cheap on
    an 8 GB Jetson while still allowing relation-specific propagation.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.source = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.relation = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.query_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid()
        )
        self.update = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        nodes: torch.Tensor,
        edge_index: torch.Tensor,
        relation_features: torch.Tensor,
        query_per_node: torch.Tensor,
    ) -> torch.Tensor:
        if edge_index.shape[1] == 0:
            # A sampled isolated node still needs a finite, differentiable path.
            return self.norm(nodes)
        source, destination = edge_index[0], edge_index[1]
        message = self.source(nodes[source]) + self.relation(relation_features)
        gate = self.query_gate(torch.cat((nodes[source], query_per_node[source]), dim=-1))
        message = message * gate
        aggregate = torch.zeros_like(nodes)
        aggregate.index_add_(0, destination, message)
        counts = torch.zeros((nodes.shape[0], 1), dtype=nodes.dtype, device=nodes.device)
        counts.index_add_(0, destination, torch.ones((destination.numel(), 1), dtype=nodes.dtype, device=nodes.device))
        aggregate = aggregate / counts.clamp_min_(1.0)
        return self.norm(nodes + self.update(torch.cat((nodes, aggregate), dim=-1)))


class NavigatorPolicy(nn.Module):
    """Hierarchical operation policy plus a local relational GNN navigator.

    The constructor defaults preserve the old ``NavigatorPolicy()`` checkpoint
    and trainer call sites.  The extra navigator modules are serialised in new
    checkpoints; loading an old state dict should use ``strict=False`` at a
    migration boundary rather than silently discarding parameters.
    """

    def __init__(
        self,
        feature_dim: int = 32,
        hidden_dim: int = 96,
        *,
        num_relations: int = 1024,
        message_passing_steps: int = 2,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or hidden_dim <= 0 or num_relations <= 0 or message_passing_steps <= 0:
            raise ValueError("feature_dim, hidden_dim, num_relations, and message_passing_steps must be positive")
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU()
        )
        self.actor = nn.Linear(hidden_dim, len(OP_NAMES))
        self.critic = nn.Linear(hidden_dim, 1)
        self.candidate_score = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

        self.node_encoder = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.query_encoder = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.relation_embedding = nn.Embedding(num_relations, hidden_dim)
        self.relation_feature_encoder = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.GELU())
        self.message_layers = nn.ModuleList(_RelationalMessageLayer(hidden_dim) for _ in range(message_passing_steps))
        self.local_candidate_score = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return high-level NEURO-ISA logits and state value.

        ``state`` must be ``[..., feature_dim]``.  Keeping this shape contract
        explicit prevents an accidental CPU/Python-side feature expansion from
        being hidden by broadcasting.
        """
        self._check_feature_tensor("state", state)
        encoded = self.encoder(state)
        return self.actor(encoded), self.critic(encoded).squeeze(-1)

    def score_candidates(self, state: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """Backward-compatible state/candidate score head.

        Candidates may be pre-encoded ``[..., hidden_dim]`` (the original API)
        or raw ``[..., feature_dim]`` features.  A single state is broadcast to
        a candidate list; arbitrary mismatched batches are rejected.
        """
        self._check_feature_tensor("state", state)
        if candidates.ndim not in (1, 2) or candidates.shape[-1] not in (self.feature_dim, self.hidden_dim):
            raise ValueError("candidates must have shape [feature_dim|hidden_dim] or [N, feature_dim|hidden_dim]")
        scalar_candidate = candidates.ndim == 1
        if scalar_candidate:
            candidates = candidates.unsqueeze(0)
        encoded = self.encoder(state)
        candidate_hidden = self.node_encoder(candidates) if candidates.shape[-1] == self.feature_dim else candidates
        if encoded.ndim == 1:
            encoded = encoded.unsqueeze(0)
        elif encoded.ndim != 2:
            raise ValueError("score_candidates expects state with shape [feature_dim] or [B, feature_dim]")
        if encoded.shape[0] == 1 and candidate_hidden.shape[0] != 1:
            encoded = encoded.expand(candidate_hidden.shape[0], -1)
        elif encoded.shape[0] != candidate_hidden.shape[0]:
            raise ValueError("state batch must be one or match candidates")
        result = self.candidate_score(torch.cat((encoded, candidate_hidden), dim=-1)).squeeze(-1)
        return result.squeeze(0) if scalar_candidate else result

    def encode_local_subgraph(
        self,
        query_features: torch.Tensor,
        subgraph: LocalCandidateSubgraph,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a local subgraph and return ``(nodes, query_per_node)``.

        The function is usable by a runtime that wants to apply its own masked
        top-k.  It is deterministic in eval mode and differentiable end-to-end.
        """
        node_features, edge_index, edge_relations, node_batch = self._validate_local_subgraph(query_features, subgraph)
        nodes = self.node_encoder(node_features)
        query_hidden = self.query_encoder(query_features)
        if node_batch is None:
            query_per_node = query_hidden.expand(nodes.shape[0], -1)
        else:
            query_per_node = query_hidden[node_batch]
        relation_features = self._relation_features(edge_relations, nodes.device)
        for layer in self.message_layers:
            nodes = layer(nodes, edge_index, relation_features, query_per_node)
        if not torch.isfinite(nodes).all():
            raise FloatingPointError("non-finite local GNN node representation")
        return nodes, query_per_node

    def score_local_candidates(self, query_features: torch.Tensor, subgraph: LocalCandidateSubgraph) -> torch.Tensor:
        """Score candidate node IDs in a bounded sampled relation graph.

        Returns one score per ``candidate_indices`` in the same order.  This is
        the navigator API used after Rust/CUDA has selected a compact frontier;
        it never accepts a global entity matrix.
        """
        nodes, query_per_node = self.encode_local_subgraph(query_features, subgraph)
        candidate_indices = subgraph.candidate_indices
        candidate_nodes = nodes[candidate_indices]
        candidate_query = query_per_node[candidate_indices]
        score = self.local_candidate_score(torch.cat((candidate_nodes, candidate_query, candidate_nodes * candidate_query), dim=-1)).squeeze(-1)
        if not torch.isfinite(score).all():
            raise FloatingPointError("non-finite local candidate score")
        return score

    def _relation_features(self, values: torch.Tensor, device: torch.device) -> torch.Tensor:
        if values.ndim == 1:
            if values.dtype != torch.long:
                raise ValueError("one-dimensional edge_relations must contain torch.long relation IDs")
            if values.numel() and (int(values.min()) < 0 or int(values.max()) >= self.num_relations):
                raise ValueError("edge relation id is outside configured num_relations")
            return self.relation_embedding(values.to(device=device))
        if values.ndim == 2 and values.shape[1] == self.feature_dim:
            return self.relation_feature_encoder(values.to(device=device, dtype=self.node_encoder[0].weight.dtype))
        raise ValueError("edge_relations must be [E] relation IDs or [E, feature_dim] features")

    def _validate_local_subgraph(
        self, query_features: torch.Tensor, subgraph: LocalCandidateSubgraph
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        self._check_feature_tensor("query_features", query_features)
        if query_features.ndim == 1:
            query_features = query_features.unsqueeze(0)
        elif query_features.ndim != 2:
            raise ValueError("query_features must be [feature_dim] or [batch, feature_dim]")
        nodes = subgraph.node_features
        self._check_feature_tensor("node_features", nodes)
        if nodes.ndim != 2:
            raise ValueError("node_features must be [nodes, feature_dim]")
        if query_features.device != nodes.device:
            raise ValueError("query_features and node_features must be on the same device")
        if query_features.dtype != nodes.dtype:
            raise ValueError("query_features and node_features must use the same dtype")
        edges = subgraph.edge_index
        if edges.dtype != torch.long or edges.ndim != 2 or edges.shape[0] != 2:
            raise ValueError("edge_index must be torch.long with shape [2, edges]")
        if edges.device != nodes.device or subgraph.edge_relations.device != nodes.device:
            raise ValueError("all local graph tensors must be on node_features.device")
        if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= nodes.shape[0]):
            raise ValueError("edge_index contains a node outside node_features")
        if subgraph.edge_relations.shape[0] != edges.shape[1]:
            raise ValueError("edge_relations length must equal edge count")
        candidates = subgraph.candidate_indices
        if candidates.dtype != torch.long or candidates.ndim != 1 or candidates.device != nodes.device:
            raise ValueError("candidate_indices must be a one-dimensional torch.long tensor on node_features.device")
        if candidates.numel() == 0:
            raise ValueError("candidate_indices must not be empty")
        if int(candidates.min()) < 0 or int(candidates.max()) >= nodes.shape[0]:
            raise ValueError("candidate index is outside node_features")
        batch = subgraph.node_batch
        if batch is None:
            if query_features.shape[0] != 1:
                raise ValueError("batched query_features require node_batch")
        else:
            if batch.dtype != torch.long or batch.ndim != 1 or batch.shape[0] != nodes.shape[0] or batch.device != nodes.device:
                raise ValueError("node_batch must be torch.long [nodes] on node_features.device")
            if batch.numel() and (int(batch.min()) < 0 or int(batch.max()) >= query_features.shape[0]):
                raise ValueError("node_batch has a query index outside query_features")
        return nodes, edges, subgraph.edge_relations, batch

    def _check_feature_tensor(self, name: str, tensor: torch.Tensor) -> None:
        if not isinstance(tensor, torch.Tensor) or tensor.ndim < 1 or tensor.shape[-1] != self.feature_dim:
            raise ValueError(f"{name} must end in feature_dim={self.feature_dim}")
        if not tensor.is_floating_point():
            raise ValueError(f"{name} must use a floating dtype")
