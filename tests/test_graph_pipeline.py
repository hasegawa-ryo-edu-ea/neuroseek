import json, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "python"))
from neuroseek.data import GraphMmap, TaskGenerator, validate_intersection_proof, validate_path_proof
from neuroseek.data.tasks import QuerySpec
from neuroseek.search.environment import GraphSearchEnv
from neuroseek.models.policy import NavigatorPolicy
from neuroseek.training.trainer import _collect_real_rollouts
import torch

def test_compile_mmap_and_tasks(tmp_path):
    triples = tmp_path / "train.txt"
    triples.write_text("Q1\tP1\tQ2\nQ2\tP2\tQ3\nQ1\tP3\tQ3\n", encoding="utf-8")
    output = tmp_path / "graph"
    subprocess.run([sys.executable, str(ROOT / "scripts/preprocess.py"), "--input", str(triples), "--output", str(output)], check=True)
    graph = GraphMmap(output)
    assert graph.manifest.entity_count == 3
    assert graph.manifest.original_triples == 3
    assert graph.manifest.traversal_edges == 6
    assert graph.has_edge(0, 0, 1)
    assert graph.has_edge(1, 1, 2)
    assert graph.entity_identifier(0) == "Q1"
    assert graph.entity_label(2) == "Q3"
    assert graph.relation_identifier(0) == "P1"
    assert graph.relation_label(1) == "P2"
    query, proof = TaskGenerator(graph, "test", 42, 2, 2).next()
    assert validate_path_proof(graph, query, proof)
    subprocess.run([sys.executable, str(ROOT / "scripts/verify_data.py"), str(output)], check=True)

def test_split_seed_is_deterministic(tmp_path):
    triples = tmp_path / "x.tsv"; triples.write_text("Q1\tP1\tQ2\nQ2\tP1\tQ3\n")
    output = tmp_path / "g"
    subprocess.run([sys.executable, str(ROOT / "scripts/preprocess.py"), "--input", str(triples), "--output", str(output)], check=True)
    graph = GraphMmap(output)
    assert TaskGenerator(graph, "validation", 3, 2, 2).next()[0] == TaskGenerator(graph, "validation", 3, 2, 2).next()[0]

def test_distractor_and_intersection_tasks(tmp_path):
    triples = tmp_path / "x.tsv"
    triples.write_text("Q1\tP1\tQ2\nQ1\tP2\tQ3\nQ4\tP3\tQ3\n")
    output = tmp_path / "g"
    subprocess.run([sys.executable, str(ROOT / "scripts/preprocess.py"), "--input", str(triples), "--output", str(output)], check=True)
    graph = GraphMmap(output)
    generator = TaskGenerator(graph, "train", 7, 1, 1)
    query, proof = generator.next_distractor()
    assert query.family == "distractor" and query.distractor_count > 0
    assert validate_path_proof(graph, query, proof)
    assert validate_intersection_proof(graph, generator.next_intersection())


def test_robustness_task_uses_an_immutable_edge_overlay(tmp_path):
    triples = tmp_path / "robust.tsv"
    triples.write_text("Q1\tP1\tQ2\nQ1\tP2\tQ3\nQ2\tP3\tQ4\nQ3\tP3\tQ4\n")
    output = tmp_path / "g"
    subprocess.run([sys.executable, str(ROOT / "scripts/preprocess.py"), "--input", str(triples), "--output", str(output)], check=True)
    graph = GraphMmap(output)
    query, proof = TaskGenerator(graph, "train", 29, 1, 1).next_robustness()
    assert query.family == "robustness" and len(query.disabled_edges) == 1
    assert (proof[0], query.relations[0], proof[1]) not in query.disabled_edges
    env = GraphSearchEnv(graph, query, proof)
    for action in env.demonstration():
        result = env.step(action)
    assert result.valid_proof
    assert any("masked=1" in row for row in result.trace)


def test_intersection_instruction_has_a_real_verified_transition(tmp_path):
    triples = tmp_path / "intersection.tsv"
    triples.write_text("Q1\tP1\tQ3\nQ2\tP2\tQ3\nQ1\tP3\tQ4\n")
    output = tmp_path / "g"
    subprocess.run([sys.executable, str(ROOT / "scripts/preprocess.py"), "--input", str(triples), "--output", str(output)], check=True)
    graph = GraphMmap(output)
    query = QuerySpec("intersection", "test", 1, 0, 1, (0,), 100, "intersection", ((0, 0), (2, 1)))
    env = GraphSearchEnv(graph, query, ())
    for action in env.demonstration():
        result = env.step(action)
    assert result.done and result.answer_correct and result.valid_proof
    assert any(row.startswith("INTERSECT(") for row in result.trace)


def test_generator_state_continues_task_stream(tmp_path):
    triples = tmp_path / "state.tsv"
    triples.write_text("Q1\tP1\tQ2\nQ2\tP1\tQ3\nQ3\tP1\tQ4\n")
    output = tmp_path / "g"
    subprocess.run([sys.executable, str(ROOT / "scripts/preprocess.py"), "--input", str(triples), "--output", str(output)], check=True)
    graph = GraphMmap(output)
    first = TaskGenerator(graph, "train", 11, 1, 1)
    first.next(); state = first.state_dict(); expected = first.next()[0]
    resumed = TaskGenerator(graph, "train", 11, 1, 1); resumed.load_state_dict(state)
    assert resumed.next()[0] == expected


def test_resume_can_union_new_heldout_exclusions_without_resetting_rng(tmp_path):
    triples = tmp_path / "migration.tsv"
    triples.write_text("Q1\tP1\tQ2\nQ2\tP1\tQ3\nQ3\tP1\tQ4\n")
    output = tmp_path / "g"
    subprocess.run([sys.executable, str(ROOT / "scripts/preprocess.py"), "--input", str(triples), "--output", str(output)], check=True)
    graph = GraphMmap(output)
    first = TaskGenerator(graph, "train", 12, 1, 1, forbidden_task_ids=("old-heldout",))
    first.next(); state = first.state_dict(); expected = first.next()[0]
    resumed = TaskGenerator(graph, "train", 12, 1, 1, forbidden_task_ids=("new-heldout",))
    assert resumed.load_state_dict(state, allow_additional_forbidden=True)
    assert resumed.next()[0] == expected
    assert resumed.forbidden_task_ids == frozenset({"old-heldout", "new-heldout"})


def test_resume_restores_curriculum_hop_bounds(tmp_path):
    triples = tmp_path / "hop-resume.tsv"
    triples.write_text("Q1\tP1\tQ2\nQ2\tP1\tQ3\nQ3\tP1\tQ4\nQ4\tP1\tQ5\n")
    output = tmp_path / "g"
    subprocess.run([sys.executable, str(ROOT / "scripts/preprocess.py"), "--input", str(triples), "--output", str(output)], check=True)
    graph = GraphMmap(output)
    source = TaskGenerator(graph, "train", 13, 2, 3)
    source.min_hops, source.max_hops = 4, 6
    state = source.state_dict()
    resumed = TaskGenerator(graph, "train", 13, 2, 3)
    assert not resumed.load_state_dict(state)
    assert (resumed.min_hops, resumed.max_hops) == (4, 6)


def test_ann_instruction_uses_configured_backend_and_never_cpu_falls_back(tmp_path):
    triples = tmp_path / "ann.tsv"
    triples.write_text("Q1\tP1\tQ2\n")
    output = tmp_path / "g"
    subprocess.run([sys.executable, str(ROOT / "scripts/preprocess.py"), "--input", str(triples), "--output", str(output)], check=True)
    graph = GraphMmap(output)
    query = QuerySpec("ann", "test", 1, 0, 1, (0,), 100, "semantic_hybrid")
    calls = []
    class Result:
        entity_ids = np.asarray([1], dtype=np.uint32)
        vectors_examined = 7
    env = GraphSearchEnv(graph, query, (0, 1), semantic_search=lambda source, k: calls.append((source, k)) or Result())
    env.step(0); result = env.step(1)
    assert calls == [(0, 3)]
    assert result.ann_calls == 1 and result.ann_vectors_examined == 7
    assert 1 in env.frontier


def test_uncovered_partial_ann_is_an_explicit_instruction_failure_not_a_trainer_error(tmp_path):
    triples = tmp_path / "ann-missing.tsv"
    triples.write_text("Q1\tP1\tQ2\n")
    output = tmp_path / "g"
    subprocess.run([sys.executable, str(ROOT / "scripts/preprocess.py"), "--input", str(triples), "--output", str(output)], check=True)
    graph = GraphMmap(output)
    query = QuerySpec("ann-missing", "test", 1, 0, 1, (0,), 100, "semantic_hybrid")
    env = GraphSearchEnv(graph, query, (0, 1), semantic_search=lambda _source, _k: (_ for _ in ()).throw(RuntimeError("semantic vector unavailable")))
    env.step(0); result = env.step(1)
    assert not result.done and result.ann_calls == 0
    assert result.reward < 0 and result.trace[-1].startswith("ANN(unavailable:")


def test_proof_validator_uses_executed_path_not_teacher_path(tmp_path):
    triples = tmp_path / "proof.tsv"
    triples.write_text("Q1\tP9\tQ2\nQ1\tP1\tQ4\nQ2\tP2\tQ3\nQ4\tP2\tQ3\n", encoding="utf-8")
    output = tmp_path / "graph"
    subprocess.run([sys.executable, str(ROOT / "scripts/preprocess.py"), "--input", str(triples), "--output", str(output)], check=True)
    graph = GraphMmap(output)
    # Teacher proof is Q1-P1-Q4-P2-Q3, but the executed ANY step records the
    # first discovered predecessor Q1-P9-Q2 before reaching Q3 through P2.
    # A correct validator must reject that actual, relation-mismatched path.
    query = QuerySpec("proof", "test", 1, 0, 3, (1, 2), 100)
    env = GraphSearchEnv(graph, query, (0, 2, 3))
    for action in (0, 3, 2, 9, 11):
        result = env.step(action)
    assert result.answer_correct
    assert not result.valid_proof


def test_real_rollout_emits_durable_trace_fields(tmp_path):
    triples = tmp_path / "trace.tsv"
    triples.write_text("Q1\tP1\tQ2\nQ2\tP2\tQ3\n", encoding="utf-8")
    output = tmp_path / "graph"
    subprocess.run([sys.executable, str(ROOT / "scripts/preprocess.py"), "--input", str(triples), "--output", str(output)], check=True)
    graph = GraphMmap(output)
    generator = TaskGenerator(graph, "train", 33, 2, 2)
    _states, _actions, _old_logprob, _returns, metrics = _collect_real_rollouts(
        NavigatorPolicy(), graph, generator, torch.device("cpu"), 2, None, 4096, 16,
    )
    assert metrics["live_search_task"]
    assert metrics["live_search_family"] == "path"
    assert isinstance(metrics["live_search_trace"], str)
    assert isinstance(metrics["operator_distribution"], str)
