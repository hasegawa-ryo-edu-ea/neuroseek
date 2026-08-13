"""A real QuerySpec search environment with independently verified proofs.

Actions are NEURO-ISA operators.  The environment never injects an answer into
the frontier: every returned candidate comes from mmap CSR traversal.  It is
small on purpose; policy choices remain high-level while node work stays in
the graph backend.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from neuroseek.data.graph import GraphMmap
from neuroseek.data.tasks import QuerySpec, validate_intersection_proof, validate_path_proof
from neuroseek.models.policy import OP_NAMES


@dataclass
class SearchResult:
    reward: float
    done: bool
    valid_proof: bool
    answer_correct: bool
    nodes_visited: int
    edges_examined: int
    credits: int
    cuda_expansions: int
    navigator_ranked_candidates: int
    ann_calls: int
    ann_vectors_examined: int
    trace: list[str]


class GraphSearchEnv:
    """One budgeted path query with stable policy and proof semantics.

    Without a ``cuda_session`` this is the deterministic CPU reference used by
    unit tests.  Trial/full training supplies the persistent CUDA CSR session:
    it creates each next frontier on the GPU, while the deliberately small CPU
    scan records only predecessor pointers for independent proof validation.
    """
    def __init__(self, graph: GraphMmap, query: QuerySpec, expected_path: tuple[int, ...], *, cuda_session: object | None = None, max_cuda_expand_edges: int = 262_144, candidate_ranker: Callable[["GraphSearchEnv", list[int]], list[int]] | None = None, semantic_search: Callable[[int, int], object] | None = None) -> None:
        # ``expected_path`` is a teacher-only demonstration.  It must never be
        # used to validate a policy result: validation reconstructs the actual
        # predecessor chain produced by the executed instructions below.
        self.graph, self.query, self.expected_path = graph, query, expected_path
        self.frontier: set[int] = set()
        # F1/F2 are compact VM frontier registers.  They make conjunction and
        # union instructions concrete without smuggling candidates through the
        # query answer oracle.
        self.aux_frontier: set[int] = set()
        self.saved_frontier: set[int] = set()
        self.parents: list[dict[int, int]] = []
        self.parent_relations: list[dict[int, int]] = []
        self.depth = 0
        self.credits = 0
        self.nodes_visited = 0
        self.edges_examined = 0
        self.done = False
        self.trace: list[str] = []
        self.answer: int | None = None
        self.proof_path: tuple[int, ...] = ()
        self.cuda_session = cuda_session
        self.max_cuda_expand_edges = max_cuda_expand_edges
        self.cuda_expansions = 0
        self.candidate_ranker = candidate_ranker
        self.navigator_ranked_candidates = 0
        self.semantic_search = semantic_search
        self.ann_calls = 0
        self.ann_vectors_examined = 0
        self.disabled_edges = frozenset(query.disabled_edges)

    def observation(self) -> np.ndarray:
        state = np.zeros(32, dtype=np.float32)
        state[0] = min(len(self.frontier), 4096) / 4096.0
        state[1] = self.depth / max(1, len(self.query.relations))
        state[2] = max(0, self.query.budget - self.credits) / self.query.budget
        state[3] = min(self.nodes_visited, self.query.budget) / self.query.budget
        if self.depth < len(self.query.relations):
            state[4 + self.query.relations[self.depth] % 16] = 1.0
        return state

    def demonstration(self) -> tuple[int, ...]:
        if self.query.family == "intersection":
            return (0, 2, 5, 9, 11)
        return (0,) + (2,) * len(self.query.relations) + (9, 11)

    def _result(self, reward: float, valid: bool = False, correct: bool = False) -> SearchResult:
        return SearchResult(reward, self.done, valid, correct, self.nodes_visited, self.edges_examined, self.credits, self.cuda_expansions, self.navigator_ranked_candidates, self.ann_calls, self.ann_vectors_examined, list(self.trace))

    def _fail_budget(self) -> SearchResult:
        self.done = True
        self.trace.append("BUDGET_EXHAUSTED")
        return self._result(-2.0)

    def _reconstruct_path(self, answer: int) -> tuple[int, ...]:
        """Return the path actually selected through the layered frontier.

        Each expansion stores one deterministic predecessor per discovered
        node.  Reconstructing from those layers makes proof validity depend on
        the policy's own operations, not on the generator's hidden solution.
        """
        node = answer
        reverse = [node]
        for parent in reversed(self.parents):
            previous = parent.get(node)
            if previous is None:
                return ()
            reverse.append(previous)
            node = previous
        reverse.reverse()
        return tuple(reverse) if reverse and reverse[0] == self.query.source else ()

    def step(self, action: int) -> SearchResult:
        if self.done:
            raise RuntimeError("instruction after STOP")
        if not 0 <= action < len(OP_NAMES):
            raise ValueError("invalid NEURO-ISA action")
        op = OP_NAMES[action]
        base_cost = 1
        if op == "SEED":
            self.frontier = {self.query.source}; self.saved_frontier = set(); self.parents = []; self.parent_relations = []; self.depth = 0; self.proof_path = ()
            self.aux_frontier = {self.query.constraints[1][0]} if self.query.family == "intersection" and len(self.query.constraints) >= 2 else set()
            self.trace.append(f"SEED({self.query.source}, overlay_edges={len(self.disabled_edges)})")
            self.credits += base_cost
            return self._result(-0.01)
        if op == "ANN":
            if self.semantic_search is None:
                self.credits += base_cost; self.trace.append("ANN(unavailable)")
                return self._result(-0.25)
            # Querying with the current seed representation is deliberate: a
            # task answer is never passed to the ANN backend.  The backend
            # returns real indexed entity IDs which still require graph proof.
            try:
                outcome = self.semantic_search(self.query.source, min(64, max(1, self.query.budget // 32)))
            except RuntimeError as exc:
                # Partial trial embeddings legitimately do not cover every
                # graph source.  This is an explicit unavailable ANN outcome,
                # never a hidden substitute retrieval backend or trainer crash.
                self.credits += base_cost
                self.trace.append(f"ANN(unavailable:{exc})")
                return self._result(-0.25)
            ids = getattr(outcome, "entity_ids", None)
            examined = int(getattr(outcome, "vectors_examined", 0))
            if ids is None:
                raise RuntimeError("semantic backend returned no entity_ids")
            self.saved_frontier = set(self.frontier)
            self.frontier = {int(value) for value in np.asarray(ids, dtype=np.uint32)}
            self.ann_calls += 1; self.ann_vectors_examined += examined
            self.nodes_visited += len(self.frontier)
            self.credits += base_cost + max(1, examined // 1024)
            self.trace.append(f"ANN(k={len(self.frontier)}, examined={examined})")
            if self.credits > self.query.budget: return self._fail_budget()
            return self._result(-0.02)
        if op in ("EXPAND_REL", "EXPAND_ANY"):
            if not self.frontier:
                self.credits += base_cost; self.trace.append(f"{op}(empty)")
                return self._result(-0.25)
            target_relation = self.query.relations[self.depth] if self.depth < len(self.query.relations) else None
            ordered_frontier = sorted(self.frontier)
            degrees = [int(self.graph.forward_offsets[node + 1] - self.graph.forward_offsets[node]) for node in ordered_frontier]
            edges_touched = sum(degrees)
            self.edges_examined += edges_touched
            if self.cuda_session is not None:
                if edges_touched > self.max_cuda_expand_edges:
                    self.done = True
                    self.trace.append(f"CUDA_EXPAND_CAP({edges_touched}>{self.max_cuda_expand_edges})")
                    return self._result(-2.0)
                # GPU session owns the frontier expansion.  The small CPU pass
                # below only derives deterministic predecessor pointers for
                # independent proof validation; it never supplies candidates.
                raw = self.cuda_session.expand(np.asarray(ordered_frontier, dtype=np.uint32), target_relation if op == "EXPAND_REL" else None)
                # CUDA owns bulk expansion.  This bounded overlay is a
                # per-episode control-plane mask, so remove only the masked
                # result edges after the kernel returns; immutable CSR stays
                # untouched and the trace reports the mask explicitly.
                blocked_targets = {target for parent, relation, target in self.disabled_edges
                                   if parent in self.frontier and (op == "EXPAND_ANY" or relation == target_relation)}
                next_frontier = {int(child) for child in raw if int(child) not in blocked_targets}
                self.cuda_expansions += 1
            else:
                next_frontier = set()
            parent: dict[int, int] = {}
            parent_relation: dict[int, int] = {}
            for node in ordered_frontier:
                nodes, relations = self.graph.neighbors(node)
                for child, relation in zip(nodes, relations):
                    if op == "EXPAND_REL" and int(relation) != target_relation:
                        continue
                    child = int(child)
                    if (node, int(relation), child) in self.disabled_edges:
                        continue
                    if self.cuda_session is None:
                        next_frontier.add(child)
                    if child in next_frontier and child not in parent:
                        parent[child] = node; parent_relation[child] = int(relation)
            self.credits += base_cost + len(next_frontier)
            if self.credits > self.query.budget: return self._fail_budget()
            self.nodes_visited += len(next_frontier); self.frontier = next_frontier; self.parents.append(parent); self.parent_relations.append(parent_relation); self.depth += 1
            self.trace.append(f"{op}({target_relation if op == 'EXPAND_REL' else '*'}, masked={len(self.disabled_edges)}) -> {len(next_frontier)}")
            # Correct relation progress is a small shaped signal; exhaustive
            # expansion remains allowed but pays measured edge cost.
            return self._result(0.05 if op == "EXPAND_REL" else -0.02)
        if op == "PRUNE" or op == "TOPK":
            keep = max(1, min(64, len(self.frontier)))
            candidates = sorted(self.frontier)
            if self.candidate_ranker is not None and candidates:
                ranked = self.candidate_ranker(self, candidates)
                if len(ranked) != len(candidates) or set(ranked) != set(candidates):
                    raise RuntimeError("candidate ranker must return each current frontier node exactly once")
                candidates = ranked
                self.navigator_ranked_candidates += len(candidates)
                self.trace.append(f"NAVIGATOR_RANK({len(candidates)})")
            self.frontier = set(candidates[:keep]); self.credits += base_cost
            self.trace.append(f"{op}({keep})")
            return self._result(-0.01)
        if op == "FILTER":
            # Keep only candidates that have at least one outgoing edge with
            # the next requested relation.  This is a real graph predicate,
            # not a target-answer membership check.
            wanted = self.query.relations[self.depth] if self.depth < len(self.query.relations) else None
            before = len(self.frontier)
            if wanted is not None:
                self.frontier = {node for node in self.frontier if bool(np.any(self.graph.neighbors(node)[1] == wanted))}
            self.credits += base_cost + before
            self.trace.append(f"FILTER(rel={wanted}, {before}->{len(self.frontier)})")
            if self.credits > self.query.budget: return self._fail_budget()
            return self._result(-0.01)
        if op == "INTERSECT":
            if self.query.family != "intersection" or len(self.query.constraints) < 2 or not self.aux_frontier:
                self.credits += base_cost; self.trace.append("INTERSECT(inapplicable)")
                return self._result(-0.25)
            source, relation = self.query.constraints[1]
            nodes, relations = self.graph.neighbors(source)
            other = {int(node) for node, rel in zip(nodes, relations) if int(rel) == relation}
            before = len(self.frontier)
            self.edges_examined += len(nodes); self.nodes_visited += len(other)
            self.frontier.intersection_update(other)
            self.credits += base_cost + len(other)
            self.trace.append(f"INTERSECT({before}&{len(other)}->{len(self.frontier)})")
            if self.credits > self.query.budget: return self._fail_budget()
            return self._result(0.05 if self.query.answer in self.frontier else -0.05)
        if op == "UNION":
            before = len(self.frontier)
            self.frontier.update(self.saved_frontier or self.aux_frontier)
            self.credits += base_cost + len(self.frontier)
            self.trace.append(f"UNION({before}->{len(self.frontier)})")
            if self.credits > self.query.budget: return self._fail_budget()
            return self._result(-0.02)
        if op == "VERIFY":
            self.credits += base_cost
            if self.query.answer in self.frontier:
                self.answer = self.query.answer
                self.proof_path = self._reconstruct_path(self.answer) if self.query.family != "intersection" else ()
            self.trace.append(f"VERIFY({self.answer}, path={len(self.proof_path)})")
            return self._result(0.1 if self.answer is not None else -0.5)
        if op == "STOP":
            self.credits += base_cost; self.done = True; self.trace.append("STOP")
            valid = self.answer == self.query.answer and (
                validate_intersection_proof(self.graph, self.query) if self.query.family == "intersection"
                else validate_path_proof(self.graph, self.query, self.proof_path)
            )
            if valid and self.query.family != "intersection":
                valid = not any((source, relation, target) in self.disabled_edges
                                for source, relation, target in zip(self.proof_path, self.query.relations, self.proof_path[1:]))
            correct = self.answer == self.query.answer
            return self._result((2.0 if valid else -1.0) - 0.001 * self.credits, valid, correct)
        if op == "BACKTRACK":
            self.frontier = {self.query.source}; self.saved_frontier = set(); self.parents = []; self.parent_relations = []; self.depth = 0; self.proof_path = (); self.credits += base_cost; self.trace.append("BACKTRACK")
            return self._result(-0.1)
        raise RuntimeError(f"unhandled NEURO-ISA operator: {op}")
