"""Deterministic held-out graph task generation; generated paths remain proofs."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import random
import json
from pathlib import Path
from typing import Any, Iterable
from .graph import GraphMmap


@dataclass(frozen=True)
class QuerySpec:
    task_id: str
    split: str
    seed: int
    source: int
    answer: int
    relations: tuple[int, ...]
    budget: int
    family: str = "path"
    # For intersections, every (source, relation) must reach ``answer``.
    constraints: tuple[tuple[int, int], ...] = ()
    distractor_count: int = 0
    # Environment-only traversal overlay. Immutable CSR is never changed.
    disabled_edges: tuple[tuple[int, int, int], ...] = ()

    def to_dict(self) -> dict:
        result = asdict(self)
        result["relations"] = list(self.relations)
        result["constraints"] = [list(item) for item in self.constraints]
        result["disabled_edges"] = [list(item) for item in self.disabled_edges]
        return result


def query_spec_from_dict(payload: dict[str, Any]) -> QuerySpec:
    """Strict JSONL task decoder used by immutable evaluation artifacts."""
    required = {"task_id", "split", "seed", "source", "answer", "relations", "budget"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"task payload is missing fields: {sorted(missing)}")
    return QuerySpec(
        task_id=str(payload["task_id"]), split=str(payload["split"]), seed=int(payload["seed"]),
        source=int(payload["source"]), answer=int(payload["answer"]),
        relations=tuple(int(item) for item in payload["relations"]), budget=int(payload["budget"]),
        family=str(payload.get("family", "path")),
        constraints=tuple((int(item[0]), int(item[1])) for item in payload.get("constraints", ())),
        distractor_count=int(payload.get("distractor_count", 0)),
        disabled_edges=tuple((int(item[0]), int(item[1]), int(item[2])) for item in payload.get("disabled_edges", ())),
    )


def load_task_jsonl(path: str | Path) -> list[tuple[QuerySpec, tuple[int, ...] | None]]:
    """Load a materialized task set; never regenerate evaluation episodes."""
    tasks: list[tuple[QuerySpec, tuple[int, ...] | None]] = []
    for number, row in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not row:
            continue
        try:
            payload = json.loads(row)
            proof_value = payload.get("proof")
            proof = tuple(int(value) for value in proof_value) if proof_value is not None else None
            tasks.append((query_spec_from_dict(payload), proof))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid task JSONL row {number} in {path}: {exc}") from exc
    if not tasks:
        raise ValueError(f"task JSONL is empty: {path}")
    if len({query.task_id for query, _ in tasks}) != len(tasks):
        raise ValueError(f"task JSONL has duplicate task IDs: {path}")
    return tasks


def validate_path_proof(graph: GraphMmap, query: QuerySpec, path: Iterable[int]) -> bool:
    """Validate a returned node path independently of any learned policy."""
    nodes = tuple(path)
    if len(nodes) != len(query.relations) + 1 or not nodes:
        return False
    if nodes[0] != query.source or nodes[-1] != query.answer:
        return False
    return all(graph.has_edge(a, rel, b) for a, rel, b in zip(nodes, query.relations, nodes[1:]))


def validate_intersection_proof(graph: GraphMmap, query: QuerySpec) -> bool:
    """Validate conjunctions without trusting the generator or the policy."""
    return query.family == "intersection" and len(query.constraints) >= 2 and all(
        graph.has_edge(source, relation, query.answer) for source, relation in query.constraints
    )


class TaskGenerator:
    """Seeded random-walk tasks. Split seed ranges are intentionally disjoint."""
    SPLIT_SALTS = {"train": 0x11A4, "validation": 0x22B5, "test": 0x33C6}

    def __init__(self, graph: GraphMmap, split: str, seed: int, min_hops: int = 2, max_hops: int = 6,
                 forbidden_task_ids: Iterable[str] = ()):
        if split not in self.SPLIT_SALTS or min_hops < 1 or max_hops < min_hops:
            raise ValueError("invalid split or hop bounds")
        self.graph, self.split, self.seed = graph, split, seed
        self.min_hops, self.max_hops = min_hops, max_hops
        self.rng = random.Random(seed ^ self.SPLIT_SALTS[split])
        self.cursor = 0
        # A training stream must never reuse an immutable held-out episode.
        # IDs are random but this explicit set makes the exclusion auditable.
        self.forbidden_task_ids = frozenset(forbidden_task_ids)

    def state_dict(self) -> dict[str, Any]:
        return {"format": 1, "split": self.split, "seed": self.seed,
                "min_hops": self.min_hops, "max_hops": self.max_hops,
                "cursor": self.cursor, "rng_state": self.rng.getstate(),
                "forbidden_task_ids": sorted(self.forbidden_task_ids)}

    def load_state_dict(self, state: dict[str, Any], *, allow_additional_forbidden: bool = False) -> bool:
        if int(state.get("format", 0)) != 1 or state.get("split") != self.split:
            raise ValueError("incompatible task generator checkpoint")
        if int(state.get("seed", -1)) != self.seed:
            raise ValueError("task generator checkpoint differs from configured stream")
        saved_min, saved_max = int(state.get("min_hops", -1)), int(state.get("max_hops", -1))
        if saved_min < 1 or saved_max < saved_min:
            raise ValueError("task generator checkpoint has invalid hop bounds")
        saved_forbidden = frozenset(str(value) for value in state.get("forbidden_task_ids", ()))
        migrated = saved_forbidden != self.forbidden_task_ids
        if migrated and not allow_additional_forbidden:
            raise ValueError("task generator held-out exclusion differs from checkpoint")
        if migrated:
            # A later immutable evaluation-set version can only *add* excluded
            # episode IDs. Keep both sets, preserving the saved RNG/cursor and
            # preventing any old or newly materialized held-out task leakage.
            self.forbidden_task_ids = self.forbidden_task_ids | saved_forbidden
        # Curriculum phase factories intentionally change these bounds. They
        # are generator state, not a static launch option, so restoring them
        # is required for a checkpoint taken during 4--6-hop/semantic phases.
        self.min_hops, self.max_hops = saved_min, saved_max
        self.cursor = int(state["cursor"])
        self.rng.setstate(state["rng_state"])
        return migrated

    def next(self, budget: int = 4096) -> tuple[QuerySpec, tuple[int, ...]]:
        for _ in range(10_000):
            source = self.rng.randrange(self.graph.manifest.entity_count)
            hops = self.rng.randint(self.min_hops, self.max_hops)
            path, rels, current = [source], [], source
            for _ in range(hops):
                nodes, relations = self.graph.neighbors(current)
                if not len(nodes): break
                index = self.rng.randrange(len(nodes))
                current = int(nodes[index]); rels.append(int(relations[index])); path.append(current)
            if len(rels) == hops:
                task_seed = self.rng.getrandbits(63)
                spec = QuerySpec(f"{self.split}-{task_seed:016x}", self.split, task_seed, source, current, tuple(rels), budget)
                if spec.task_id in self.forbidden_task_ids:
                    continue
                self.cursor += 1
                return spec, tuple(path)
        raise RuntimeError("could not generate path task; graph has insufficient traversable edges")

    def next_distractor(self, budget: int = 4096) -> tuple[QuerySpec, tuple[int, ...]]:
        """A valid path whose seed has irrelevant outgoing alternatives.

        No graph mutation is involved: the policy must select the demonstrated
        relation rather than receiving an artificially changed truth graph.
        """
        for _ in range(10_000):
            query, proof = self.next(budget)
            nodes, relations = self.graph.neighbors(query.source)
            alternatives = sum(
                1 for node, relation in zip(nodes, relations)
                if int(node) != proof[1] or int(relation) != query.relations[0]
            )
            if alternatives:
                return QuerySpec(query.task_id, query.split, query.seed, query.source,
                                 query.answer, query.relations, query.budget, "distractor",
                                 query.constraints, alternatives), proof
        raise RuntimeError("could not generate distractor task; sampled seeds lack branching")

    def next_intersection(self, budget: int = 4096) -> QuerySpec:
        """Create A--r1-->answer AND B--r2-->answer from real reverse edges."""
        for _ in range(10_000):
            answer = self.rng.randrange(self.graph.manifest.entity_count)
            sources, relations = self.graph.neighbors(answer, reverse=True)
            if len(sources) < 2:
                continue
            candidates = list(zip(map(int, sources), map(int, relations)))
            self.rng.shuffle(candidates)
            first = candidates.pop()
            second = next((x for x in candidates if x != first), None)
            if second is None:
                continue
            task_seed = self.rng.getrandbits(63)
            task = QuerySpec(f"{self.split}-{task_seed:016x}", self.split, task_seed,
                             first[0], answer, (first[1],), budget, "intersection",
                             (first, second))
            if task.task_id in self.forbidden_task_ids:
                continue
            self.cursor += 1
            return task
        raise RuntimeError("could not generate intersection task; graph lacks reverse fan-in")

    def next_robustness(self, budget: int = 4096) -> tuple[QuerySpec, tuple[int, ...]]:
        """Hide one real *irrelevant* first-hop edge through an overlay.

        The correct demonstrated edge is never disabled.  The overlay is
        carried by the episode rather than mutating mmap CSR, making it safe
        for concurrent held-out evaluation and resume.
        """
        for _ in range(10_000):
            query, proof = self.next_distractor(budget)
            nodes, relations = self.graph.neighbors(query.source)
            alternatives = [(int(node), int(relation)) for node, relation in zip(nodes, relations)
                            if not (int(node) == proof[1] and int(relation) == query.relations[0])]
            if not alternatives:
                continue
            node, relation = alternatives[self.rng.randrange(len(alternatives))]
            return QuerySpec(query.task_id, query.split, query.seed, query.source, query.answer,
                             query.relations, query.budget, "robustness", query.constraints,
                             query.distractor_count, ((query.source, relation, node),)), proof
        raise RuntimeError("could not generate robustness task with a safe edge overlay")
