import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "python"))

from neuroseek.data import GraphMmap, QuerySpec, TaskGenerator
from neuroseek.evaluation import (
    bfs_search,
    evaluate,
    fixed_relation_search,
    heuristic_hybrid_search,
    materialize_heldout_tasks,
)


def _graph(tmp_path: Path) -> GraphMmap:
    triples = tmp_path / "graph.tsv"
    # Compact IDs preserve lexical Q/P order in the compiler: Q1=0... Q5=4,
    # P1=0... This graph has a relation-valid two-hop path plus distractors
    # and two independent incoming constraints to Q4.
    triples.write_text(
        "Q1\tP1\tQ2\nQ2\tP2\tQ4\nQ1\tP3\tQ3\nQ3\tP4\tQ4\n"
        "Q5\tP5\tQ4\nQ2\tP6\tQ3\n",
        encoding="utf-8",
    )
    output = tmp_path / "processed"
    subprocess.run([sys.executable, str(ROOT / "scripts/preprocess.py"), "--input", str(triples),
                    "--output", str(output)], check=True)
    return GraphMmap(output)


def test_real_mmap_path_baselines_and_metrics(tmp_path):
    graph = _graph(tmp_path)
    # Q1 -P1-> Q2 -P2-> Q4 (IDs follow first appearance in source order).
    query = QuerySpec("heldout-path", "test", 1, 0, 2, (0, 1), 10_000)
    fixed = fixed_relation_search(graph, query)
    hybrid = heuristic_hybrid_search(graph, query)
    bfs = bfs_search(graph, query)
    assert fixed.answer_correct and fixed.valid_proof and fixed.proof == (0, 1, 2)
    assert hybrid.answer_correct and hybrid.valid_proof
    assert bfs.answer_correct
    assert fixed.edges_examined > 0 and fixed.nodes_visited > 0 and fixed.latency_ms >= 0.0
    report = evaluate(graph, [query], fixed_relation_search)
    assert report.task_count == report.applicable_count == 1
    assert report.answer_accuracy == report.valid_proof_rate == 1.0
    assert report.to_dict()["results"][0]["task_id"] == "heldout-path"


def test_intersection_baseline_uses_real_constraints(tmp_path):
    graph = _graph(tmp_path)
    # Q2 -P2-> Q4 and Q5 -P5-> Q4.
    query = QuerySpec("heldout-intersection", "test", 2, 1, 2, (1,), 10_000,
                      family="intersection", constraints=((1, 1), (4, 4)))
    result = heuristic_hybrid_search(graph, query)
    assert result.applicable and result.answer_correct and result.valid_proof
    assert result.proof == ()  # conjunction proof is validated via constraints, not a fake path
    assert not fixed_relation_search(graph, query).applicable


def test_heldout_materialization_is_reproducible_and_nonoverwriting(tmp_path):
    graph = _graph(tmp_path)
    destination = tmp_path / "heldout.jsonl"
    first = materialize_heldout_tasks(graph, destination, count=8, seed=777, min_hops=1, max_hops=1)
    content = first.read_bytes()
    assert materialize_heldout_tasks(graph, destination, count=8, seed=777, min_hops=1, max_hops=1) == first
    assert destination.read_bytes() == content
    manifest = json.loads(destination.with_suffix(".jsonl.manifest.json").read_text())
    assert manifest["task_count"] == 8
    rows = [json.loads(row) for row in destination.read_text().splitlines()]
    assert {row["split"] for row in rows} == {"test"}
    assert any(row["family"] == "intersection" for row in rows)
    assert {"semantic_hybrid", "robustness"}.issubset({row["family"] for row in rows})
    try:
        materialize_heldout_tasks(graph, destination, count=7, seed=777, min_hops=1, max_hops=1)
    except FileExistsError:
        pass
    else:
        raise AssertionError("different held-out set must not overwrite existing artifact")


def test_generator_test_split_does_not_reuse_train_sequence(tmp_path):
    graph = _graph(tmp_path)
    train, _ = TaskGenerator(graph, "train", 55, 1, 1).next()
    test, _ = TaskGenerator(graph, "test", 55, 1, 1).next()
    assert train.task_id != test.task_id
