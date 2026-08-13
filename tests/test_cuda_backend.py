"""CUDA graph-core parity tests; never substitute a CPU execution path."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from neuroseek.cuda_backend import CudaBackendError, CudaExactBackend


def _library() -> Path:
    candidate = Path(os.environ.get("NEUROSEEK_CUDA_LIB", "build/cuda/libneuroseek_cuda.so"))
    if not candidate.exists():
        pytest.skip(f"CUDA shared library has not been built: {candidate}")
    return candidate


@pytest.fixture(scope="module")
def cuda() -> CudaExactBackend:
    backend = CudaExactBackend(_library())
    try:
        if backend.device_count() < 1:
            pytest.skip("CUDA library loaded but no device is visible")
    except CudaBackendError as exc:
        pytest.skip(str(exc))
    return backend


def _cpu_expand(offsets: np.ndarray, targets: np.ndarray, relations: np.ndarray, frontier: list[int], relation: int | None) -> np.ndarray:
    values: list[int] = []
    for node in frontier:
        start, end = int(offsets[node]), int(offsets[node + 1])
        for target, current_relation in zip(targets[start:end], relations[start:end]):
            if relation is None or int(current_relation) == relation:
                values.append(int(target))
    return np.asarray(values, dtype=np.uint32)


@pytest.fixture
def csr() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Node 0 is deliberately high degree enough to exercise multiple blocks
    # on the common Orin launch shape; duplicate targets are semantically kept.
    degree = 777
    offsets = np.asarray([0, degree, degree + 2, degree + 2], dtype=np.uint64)
    targets = np.asarray(([i % 3 for i in range(degree)] + [2, 0]), dtype=np.uint32)
    relations = np.asarray(([7 if i % 5 else 9 for i in range(degree)] + [7, 8]), dtype=np.uint16)
    return offsets, targets, relations


@pytest.mark.parametrize("relation", [None, 7, 9])
def test_expand_relation_filter_cpu_parity(cuda: CudaExactBackend, csr: tuple[np.ndarray, np.ndarray, np.ndarray], relation: int | None) -> None:
    offsets, targets, relations = csr
    frontier = [1, 0, 0]  # duplicate frontier ensures no accidental dedup.
    actual = cuda.expand_csr(offsets, targets, relations, frontier, relation)
    expected = _cpu_expand(offsets, targets, relations, frontier, relation)
    # Atomic append order is deliberately unspecified; edge-result multiset is
    # the ABI contract until the VM executes deterministic compaction.
    np.testing.assert_array_equal(np.sort(actual), np.sort(expected))


def test_empty_frontier_is_valid(cuda: CudaExactBackend, csr: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    offsets, targets, relations = csr
    actual = cuda.expand_csr(offsets, targets, relations, [], 7)
    assert actual.dtype == np.uint32
    assert actual.size == 0


def test_capacity_is_never_silent_truncation(cuda: CudaExactBackend, csr: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    offsets, targets, relations = csr
    with pytest.raises(CudaBackendError, match="below required"):
        cuda.expand_csr(offsets, targets, relations, [0], None, output_capacity=3)


def test_gpu_resident_session_lifecycle_and_cpu_parity(cuda: CudaExactBackend, csr: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    offsets, targets, relations = csr
    frontier = [0, 1, 0]
    expected = _cpu_expand(offsets, targets, relations, frontier, 7)
    session = cuda.create_csr_session(offsets, targets, relations)
    try:
        actual = session.expand(frontier, 7)
        np.testing.assert_array_equal(np.sort(actual), np.sort(expected))
        with pytest.raises(CudaBackendError, match="below required"):
            session.expand(frontier, 7, output_capacity=2)
    finally:
        session.close()
    assert session.closed
    with pytest.raises(CudaBackendError, match="already closed"):
        session.expand(frontier, 7)


def test_gpu_resident_session_rejects_corrupt_csr(cuda: CudaExactBackend, csr: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    _offsets, targets, relations = csr
    corrupt = np.asarray([0, 4, 2, targets.size], dtype=np.uint64)
    with pytest.raises(CudaBackendError, match="CUDA status 1"):
        cuda.create_csr_session(corrupt, targets, relations)


@pytest.mark.parametrize(
    ("offsets", "frontier", "message"),
    [
        (np.asarray([0, 2, 1, 1], dtype=np.uint64), [0], "CUDA status 1"),
        (np.asarray([0, 2, 3, 3], dtype=np.uint64), [3], "CUDA status 1"),
    ],
)
def test_invalid_csr_or_frontier_is_rejected(cuda: CudaExactBackend, csr: tuple[np.ndarray, np.ndarray, np.ndarray], offsets: np.ndarray, frontier: list[int], message: str) -> None:
    _old_offsets, targets, relations = csr
    with pytest.raises((ValueError, CudaBackendError), match=message):
        cuda.expand_csr(offsets, targets[: int(offsets[-1])], relations[: int(offsets[-1])], frontier)
