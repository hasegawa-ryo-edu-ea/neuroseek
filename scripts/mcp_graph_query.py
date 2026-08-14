#!/usr/bin/env python3
"""Read a bounded set of direct, local Wikidata5M graph facts as JSON.

This worker intentionally has no network path, CUDA context, model loading, or
write path.  It is used by the stdio MCP server so a local LLM can distinguish
direct graph evidence from a learned-policy search result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from neuroseek.data.graph import GraphMmap


def _entity(graph: GraphMmap, index: int) -> dict[str, object]:
    return {
        "local_id": index,
        "identifier": graph.entity_identifier(index),
        "label": graph.entity_label(index),
    }


def _relation(graph: GraphMmap, index: int) -> dict[str, object]:
    return {
        "local_id": index,
        "identifier": graph.relation_identifier(index),
        "label": graph.relation_label(index),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", required=True, help="Wikidata Q-ID present in the local graph")
    parser.add_argument("--relation", help="optional Wikidata P-ID filter present in the local graph")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.limit <= 50:
        raise SystemExit("--limit must be between 1 and 50")

    graph_root = Path("data/processed")
    if not graph_root.is_dir():
        raise FileNotFoundError("processed graph artifact is absent: data/processed")
    graph = GraphMmap(graph_root)
    entity_id = args.entity.strip().upper()
    entity = graph.find_entity_identifier(entity_id)
    if entity is None:
        raise LookupError(f"entity is not present in the local graph: {entity_id}")

    relation: int | None = None
    if args.relation:
        relation_id = args.relation.strip().upper()
        relation = graph.find_relation_identifier(relation_id)
        if relation is None:
            raise LookupError(f"relation is not present in the local graph: {relation_id}")

    nodes, relations = graph.neighbors(entity)
    facts = []
    for target, edge_relation in zip(nodes, relations):
        edge_relation = int(edge_relation)
        if relation is not None and edge_relation != relation:
            continue
        facts.append({
            "subject": _entity(graph, entity),
            "predicate": _relation(graph, edge_relation),
            "object": _entity(graph, int(target)),
            "evidence": {"kind": "direct_local_csr_edge", "graph": "Wikidata5M", "direction": "outgoing"},
        })
        if len(facts) >= args.limit:
            break
    print(json.dumps({
        "event": "local_graph_facts",
        "entity": _entity(graph, entity),
        "relation_filter": _relation(graph, relation) if relation is not None else None,
        "returned_facts": facts,
        "returned_count": len(facts),
        "limit": args.limit,
        "truncated": len(nodes) > len(facts),
        "evidence_scope": "Only outgoing edges stored in this local immutable graph are returned.",
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
