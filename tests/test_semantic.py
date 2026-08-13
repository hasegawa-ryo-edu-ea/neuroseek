from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "python"))

from neuroseek.data.graph import GraphMmap
from neuroseek.semantic import (AlignedEmbeddings, NumpyExactBackend, SemanticError,
                                TransEConfig, train_bounded_transe, write_embedding_store)


def test_aligned_store_exact_search_and_tie_rule(tmp_path):
    root = tmp_path / "semantic"
    write_embedding_store(root, np.arange(3, dtype=np.uint32), np.array([[1., 0.], [1., 0.], [0., 1.]]),
                          graph_entity_count=3, source={"kind": "unit-test"})
    store = AlignedEmbeddings(root, expected_graph_entities=3)
    result = store.search(np.array([1., 0.]), 2, NumpyExactBackend())
    assert result.entity_ids.tolist() == [0, 1]
    assert result.backend == "numpy_exact_test_only"
    assert result.vectors_examined == 3


def test_partial_store_requires_explicit_opt_in_and_hash_is_checked(tmp_path):
    root = tmp_path / "partial"
    write_embedding_store(root, np.array([1, 3], dtype=np.uint32), np.ones((2, 3)), graph_entity_count=5,
                          source={"kind": "unit-test"})
    with pytest.raises(SemanticError, match="partial"):
        AlignedEmbeddings(root, expected_graph_entities=5)
    store = AlignedEmbeddings(root, expected_graph_entities=5, allow_partial=True)
    assert store.manifest.complete_alignment is False
    with (root / "embeddings.f16").open("r+b") as file:
        file.write(b"BAD!")
    with pytest.raises(SemanticError, match="hash mismatch"):
        AlignedEmbeddings(root, expected_graph_entities=5, allow_partial=True)


def test_bounded_transe_writes_partial_real_graph_artifact(tmp_path):
    triples = tmp_path / "triples.tsv"
    triples.write_text("Q1\tP1\tQ2\nQ2\tP1\tQ3\nQ3\tP2\tQ1\n")
    graph_root = tmp_path / "graph"
    subprocess.run([sys.executable, str(ROOT / "scripts/preprocess.py"), "--input", str(triples), "--output", str(graph_root)], check=True)
    store_root = train_bounded_transe(GraphMmap(graph_root), tmp_path / "transe",
                                        TransEConfig(dimension=8, max_entities=2, steps=16, batch_size=4, device="cpu"))
    store = AlignedEmbeddings(store_root, expected_graph_entities=3, allow_partial=True)
    assert store.manifest.source["kind"] == "bounded_transe_fallback"
    assert store.manifest.dimension == 8
    result = store.search(np.ones(8, dtype=np.float32), 1, NumpyExactBackend())
    assert result.entity_ids.shape == (1,)
