"""Auditable real-graph evaluation baselines.

These baselines intentionally operate on :class:`GraphMmap`, not a copied
Python graph.  Every count is collected while touching mmap CSR arrays and
every claimed symbolic success is independently checked against the graph.

``QuerySpec.answer`` is an oracle label generated from a real graph path.  It
is used only for scoring and an early stop, never to inject a node into a
frontier.  This is stated explicitly because automatic episodic evaluation is
not an open-world natural-language benchmark.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable, Literal, Sequence

import numpy as np

from neuroseek.data.graph import GraphMmap
from neuroseek.data.tasks import QuerySpec, TaskGenerator, validate_intersection_proof, validate_path_proof


BaselineName = Literal["bfs", "fixed_relation", "hybrid", "ann_only"]
_PATH_FAMILIES = frozenset({"path", "distractor", "semantic_hybrid", "robustness"})


@dataclass(frozen=True)
class BaselineResult:
    """One actually executed baseline result; no field is a predicted value."""

    baseline: str
    task_id: str
    applicable: bool
    answer: int | None
    answer_correct: bool
    valid_proof: bool
    nodes_visited: int
    edges_examined: int
    search_steps: int
    ann_calls: int
    compute_credits: int
    latency_ms: float
    proof: tuple[int, ...] = ()
    reason: str | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["proof"] = list(self.proof)
        return payload


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregate over a fixed materialized held-out task set."""

    baseline: str
    task_count: int
    applicable_count: int
    answer_accuracy: float | None
    valid_proof_rate: float | None
    mean_nodes_visited: float | None
    mean_edges_examined: float | None
    mean_search_steps: float | None
    mean_ann_calls: float | None
    mean_compute_credits: float | None
    mean_latency_ms: float | None
    p95_latency_ms: float | None
    results: tuple[BaselineResult, ...]

    def to_dict(self, include_results: bool = True) -> dict:
        out = asdict(self)
        out["results"] = [item.to_dict() for item in self.results] if include_results else []
        return out


def _result(
    baseline: str, query: QuerySpec, start: float, *, applicable: bool = True,
    answer: int | None = None, valid_proof: bool = False, nodes: int = 0,
    edges: int = 0, steps: int = 0, ann_calls: int = 0, credits: int = 0,
    proof: Iterable[int] = (), reason: str | None = None,
) -> BaselineResult:
    return BaselineResult(
        baseline=baseline, task_id=query.task_id, applicable=applicable, answer=answer,
        answer_correct=answer == query.answer if answer is not None else False,
        valid_proof=valid_proof, nodes_visited=nodes, edges_examined=edges,
        search_steps=steps, ann_calls=ann_calls, compute_credits=credits,
        latency_ms=(perf_counter() - start) * 1_000.0, proof=tuple(proof), reason=reason,
    )


def _path_from_parents(source: int, answer: int, parent: dict[int, int]) -> tuple[int, ...]:
    current = answer
    reverse = [current]
    # parent only contains one depth's acyclic predecessor chain.  The cycle
    # guard makes malformed input fail closed rather than loop indefinitely.
    while current != source:
        if current not in parent or len(reverse) > len(parent) + 1:
            return ()
        current = parent[current]
        reverse.append(current)
    return tuple(reversed(reverse))


def fixed_relation_search(graph: GraphMmap, query: QuerySpec) -> BaselineResult:
    """Exact relation-plan traversal for path/distractor tasks.

    It is an intentionally strong symbolic baseline: the generated QuerySpec
    supplies the relation program, but the graph still supplies all candidates
    and the proof must validate independently.
    """
    started = perf_counter()
    if query.family not in _PATH_FAMILIES:
        return _result("fixed_relation", query, started, applicable=False,
                       reason="exact relation plans apply only to path tasks")
    frontier = {query.source}
    # Keep predecessor history per (depth,node), avoiding an invalid proof when
    # node IDs recur across hops.
    parents: dict[tuple[int, int], int] = {}
    edges = nodes = credits = 0
    for depth, relation_wanted in enumerate(query.relations, start=1):
        next_frontier: set[int] = set()
        for source in sorted(frontier):
            children, relations = graph.neighbors(source)
            edges += len(children)
            for child, relation in zip(children, relations):
                if int(relation) == relation_wanted:
                    child_id = int(child)
                    if (source, int(relation), child_id) in query.disabled_edges:
                        continue
                    if child_id not in next_frontier:
                        parents[(depth, child_id)] = source
                    next_frontier.add(child_id)
        nodes += len(next_frontier)
        credits += 1 + len(next_frontier)
        if credits > query.budget:
            return _result("fixed_relation", query, started, nodes=nodes, edges=edges,
                           steps=depth, credits=credits, reason="compute budget exhausted")
        frontier = next_frontier
        if not frontier:
            break
    answer = query.answer if query.answer in frontier else None
    proof: tuple[int, ...] = ()
    if answer is not None:
        reverse = [answer]
        current = answer
        for depth in range(len(query.relations), 0, -1):
            key = (depth, current)
            if key not in parents:
                reverse = []
                break
            current = parents[key]
            reverse.append(current)
        proof = tuple(reversed(reverse)) if reverse and reverse[-1] == query.source else ()
    valid = bool(proof) and validate_path_proof(graph, query, proof) and not any(
        (source, relation, target) in query.disabled_edges
        for source, relation, target in zip(proof, query.relations, proof[1:])
    )
    return _result("fixed_relation", query, started, answer=answer, valid_proof=valid,
                   nodes=nodes, edges=edges, steps=len(query.relations), credits=credits, proof=proof)


def bfs_search(graph: GraphMmap, query: QuerySpec) -> BaselineResult:
    """Bounded generic BFS over actual outgoing CSR edges.

    BFS is applicable to path tasks.  It examines all relations and stops at
    the known answer within the task's hop limit; it cannot manufacture an
    edge or proof.
    """
    started = perf_counter()
    if query.family not in _PATH_FAMILIES:
        return _result("bfs", query, started, applicable=False,
                       reason="path BFS is not a conjunctive-query solver")
    frontier = {query.source}
    parents: dict[tuple[int, int], int] = {}
    edges = nodes = credits = 0
    found_depth: int | None = None
    for depth in range(1, len(query.relations) + 1):
        next_frontier: set[int] = set()
        for source in sorted(frontier):
            children, edge_relations = graph.neighbors(source)
            edges += len(children)
            for child, relation in zip(children, edge_relations):
                child_id = int(child)
                if (source, int(relation), child_id) in query.disabled_edges:
                    continue
                if child_id not in next_frontier:
                    parents[(depth, child_id)] = source
                next_frontier.add(child_id)
        nodes += len(next_frontier)
        credits += 1 + len(next_frontier)
        if credits > query.budget:
            return _result("bfs", query, started, nodes=nodes, edges=edges, steps=depth,
                           credits=credits, reason="compute budget exhausted")
        if query.answer in next_frontier:
            found_depth = depth
            break
        frontier = next_frontier
        if not frontier:
            break
    # A generic BFS path is only a valid proof if it happens to use the query's
    # requested relation sequence.  This prevents an alternate fact path from
    # being counted as a correct answer for a relation-specific question.
    proof: tuple[int, ...] = ()
    if found_depth is not None:
        current = query.answer
        reverse = [current]
        for depth in range(found_depth, 0, -1):
            prior = parents.get((depth, current))
            if prior is None:
                reverse = []
                break
            reverse.append(prior)
            current = prior
        proof = tuple(reversed(reverse)) if reverse and reverse[-1] == query.source else ()
    valid = len(proof) == len(query.relations) + 1 and validate_path_proof(graph, query, proof) and not any(
        (source, relation, target) in query.disabled_edges
        for source, relation, target in zip(proof, query.relations, proof[1:])
    )
    answer = query.answer if found_depth is not None else None
    return _result("bfs", query, started, answer=answer, valid_proof=valid, nodes=nodes,
                   edges=edges, steps=found_depth or len(query.relations), credits=credits, proof=proof)


def heuristic_hybrid_search(graph: GraphMmap, query: QuerySpec) -> BaselineResult:
    """Hand-written graph planner with real conjunction evaluation.

    For a relation path it uses the exact relation plan.  For an intersection
    it expands each independently stated constraint, intersects answer sets,
    and validates every required edge.  It does not use learned scores.
    """
    started = perf_counter()
    if query.family in _PATH_FAMILIES:
        # Preserve the measured result (including its actual latency) rather
        # than re-labelling a fixed value.  The hybrid's graph plan is exactly
        # the fixed plan for this task family.
        base = fixed_relation_search(graph, query)
        return BaselineResult("hybrid", **{k: v for k, v in asdict(base).items() if k != "baseline"})
    if query.family != "intersection" or len(query.constraints) < 2:
        return _result("hybrid", query, started, applicable=False, reason="unsupported task family")
    candidates: set[int] | None = None
    edges = nodes = credits = 0
    for source, relation_wanted in query.constraints:
        children, relations = graph.neighbors(source)
        edges += len(children)
        reached = {int(child) for child, relation in zip(children, relations) if int(relation) == relation_wanted}
        nodes += len(reached)
        credits += 1 + len(reached)
        if credits > query.budget:
            return _result("hybrid", query, started, nodes=nodes, edges=edges,
                           steps=len(query.constraints), credits=credits, reason="compute budget exhausted")
        candidates = reached if candidates is None else candidates & reached
    answer = query.answer if candidates is not None and query.answer in candidates else None
    valid = answer is not None and validate_intersection_proof(graph, query)
    # An intersection proof is its real edge constraints rather than a fake
    # linear node path.  The trace consumer can render these from QuerySpec.
    return _result("hybrid", query, started, answer=answer, valid_proof=valid,
                   nodes=nodes, edges=edges, steps=len(query.constraints), credits=credits)


def ann_only_search(embeddings: object, backend: object, query_vector: np.ndarray, query: QuerySpec, k: int = 32) -> BaselineResult:
    """Run a supplied real ANN backend; semantic-only answers carry no graph proof.

    The generic objects deliberately avoid importing CUDA during symbolic-only
    evaluation.  ``AlignedEmbeddings.search`` is the concrete production API.
    """
    started = perf_counter()
    if k <= 0:
        raise ValueError("k must be positive")
    search = getattr(embeddings, "search", None)
    if not callable(search):
        raise TypeError("embeddings must provide AlignedEmbeddings.search")
    outcome = search(query_vector, k, backend)
    ids = np.asarray(outcome.entity_ids, dtype=np.uint32)
    answer = query.answer if np.any(ids == query.answer) else (int(ids[0]) if ids.size else None)
    return _result("ann_only", query, started, answer=answer, valid_proof=False,
                   nodes=int(outcome.vectors_examined), ann_calls=1, credits=int(outcome.vectors_examined),
                   reason="semantic-only baseline has no symbolic proof")


def _task_payload(query: QuerySpec, proof: tuple[int, ...] | None) -> dict:
    out = query.to_dict()
    out["proof"] = list(proof) if proof is not None else None
    return out


def materialize_heldout_tasks(graph: GraphMmap, path: str | Path, *, count: int, seed: int,
                              split: str = "test", min_hops: int = 2, max_hops: int = 6,
                              include_intersections: bool = True, task_mix_version: int = 2) -> Path:
    """Create a reproducible JSONL held-out set from a disjoint split RNG.

    Existing files are validated byte-for-byte against the requested generation
    parameters instead of being silently replaced.  The SHA-256 sidecar makes
    later reports bind to a concrete evaluation set.
    """
    if split not in {"validation", "test"}:
        raise ValueError("held-out evaluation must use validation or test split")
    if count <= 0:
        raise ValueError("count must be positive")
    output = Path(path)
    if task_mix_version != 2:
        raise ValueError("only task mix version 2 is supported")
    spec = {"count": count, "seed": seed, "split": split, "min_hops": min_hops,
            "max_hops": max_hops, "include_intersections": include_intersections,
            "task_mix_version": task_mix_version,
            "families": ["path", "distractor", "intersection", "semantic_hybrid", "robustness"]}
    meta = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() or meta.exists():
        if not output.is_file() or not meta.is_file():
            raise FileExistsError(f"incomplete held-out artifact exists: {output}")
        existing = json.loads(meta.read_text())
        content = output.read_bytes()
        if existing.get("generation") != spec or existing.get("sha256") != hashlib.sha256(content).hexdigest():
            raise FileExistsError(f"refuse to replace different held-out set: {output}")
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    generator = TaskGenerator(graph, split, seed, min_hops=min_hops, max_hops=max_hops)
    rows: list[str] = []
    for i in range(count):
        slot = i % 8
        if include_intersections and slot == 3:
            task = generator.next_intersection()
            rows.append(json.dumps(_task_payload(task, None), sort_keys=True))
        elif slot == 6:
            task, proof = generator.next()
            from dataclasses import replace
            rows.append(json.dumps(_task_payload(replace(task, family="semantic_hybrid"), proof), sort_keys=True))
        elif slot == 7:
            task, proof = generator.next_robustness()
            rows.append(json.dumps(_task_payload(task, proof), sort_keys=True))
        else:
            task, proof = generator.next_distractor() if slot % 3 == 2 else generator.next()
            rows.append(json.dumps(_task_payload(task, proof), sort_keys=True))
    content = ("\n".join(rows) + "\n").encode()
    tmp = output.with_name(f".{output.name}.tmp")
    tmp.write_bytes(content)
    tmp.replace(output)
    meta.write_text(json.dumps({"generation": spec, "sha256": hashlib.sha256(content).hexdigest(),
                                "task_count": count}, sort_keys=True, indent=2) + "\n")
    return output


def evaluate(graph: GraphMmap, tasks: Sequence[QuerySpec], baseline: Callable[[GraphMmap, QuerySpec], BaselineResult]) -> EvaluationReport:
    """Evaluate one baseline without hiding not-applicable task families."""
    results = tuple(baseline(graph, task) for task in tasks)
    applicable = tuple(item for item in results if item.applicable)
    def mean(name: str) -> float | None:
        return float(np.mean([getattr(item, name) for item in applicable])) if applicable else None
    latency = np.asarray([item.latency_ms for item in applicable], dtype=np.float64)
    return EvaluationReport(
        baseline=results[0].baseline if results else getattr(baseline, "__name__", "unknown"), task_count=len(results),
        applicable_count=len(applicable), answer_accuracy=mean("answer_correct"), valid_proof_rate=mean("valid_proof"),
        mean_nodes_visited=mean("nodes_visited"), mean_edges_examined=mean("edges_examined"),
        mean_search_steps=mean("search_steps"), mean_ann_calls=mean("ann_calls"),
        mean_compute_credits=mean("compute_credits"), mean_latency_ms=mean("latency_ms"),
        p95_latency_ms=float(np.percentile(latency, 95)) if latency.size else None, results=results,
    )
