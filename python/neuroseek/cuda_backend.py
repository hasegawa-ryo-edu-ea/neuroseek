"""Checked bindings for NEUROSEEK's mandatory CUDA-exact backend.

The graph expansion entrypoint intentionally accepts NumPy arrays directly so
``GraphMmap``'s read-only mmap buffers can be used without materialising a
Python graph or copying the CSR on the host.  The current C ABI is synchronous
and copies a requested CSR to temporary device buffers; it is a correctness
reference and a safe fallback, not the persistent GPU graph cache used by a
long-running search service.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from neuroseek.data.graph import GraphMmap


CUDA_OK = 0
CUDA_INVALID_ARGUMENT = 1
CUDA_UNAVAILABLE = 2
CUDA_RUNTIME_ERROR = 3
CUDA_OUTPUT_CAPACITY = 4

_U64_P = ctypes.POINTER(ctypes.c_uint64)
_U32_P = ctypes.POINTER(ctypes.c_uint32)
_U16_P = ctypes.POINTER(ctypes.c_uint16)
_F32_P = ctypes.POINTER(ctypes.c_float)


class CudaBackendError(RuntimeError):
    """A CUDA C-ABI failure, including the native diagnostic string."""


def _pointer(array: np.ndarray, pointer_type: type[ctypes._Pointer]) -> ctypes._Pointer:
    return array.ctypes.data_as(pointer_type)


class CudaCsrSession:
    """Owned GPU-resident CSR traversal direction.

    It uploads offsets, target IDs and relation IDs once at construction. Each
    ``expand`` call transfers only a compact frontier plus its result. Close it
    explicitly at phase/run shutdown (or use it as a context manager) to make
    its unified-memory reservation deterministic on an 8 GB Jetson.
    """

    def __init__(self, backend: "CudaExactBackend", handle: ctypes.c_void_p) -> None:
        self._backend = backend
        self._handle: ctypes.c_void_p | None = handle

    @property
    def closed(self) -> bool:
        return self._handle is None

    def close(self) -> None:
        if self._handle is None:
            return
        handle, self._handle = self._handle, None
        self._backend._check(self._backend.lib.neuroseek_cuda_csr_session_destroy(handle), "csr_session_destroy")

    def __enter__(self) -> "CudaCsrSession":
        if self._handle is None:
            raise CudaBackendError("GPU CSR session is already closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:  # best-effort only; production must call close.
        try:
            self.close()
        except (CudaBackendError, OSError):
            return

    def expand(self, frontier: np.ndarray | list[int], relation: int | None = None, *, output_capacity: int | None = None) -> np.ndarray:
        if self._handle is None:
            raise CudaBackendError("GPU CSR session is already closed")
        frontier_array = self._backend._frontier(frontier)
        relation_code = -1 if relation is None else int(relation)
        if relation_code < -1 or relation_code > np.iinfo(np.uint16).max:
            raise ValueError("relation must be None/-1 or a uint16 relation ID")
        required = ctypes.c_uint64()
        status = self._backend.lib.neuroseek_cuda_csr_session_expand(
            self._handle, _pointer(frontier_array, _U32_P) if frontier_array.size else None,
            frontier_array.size, relation_code, None, 0, ctypes.byref(required),
        )
        self._backend._check(status, "csr_session_expand size probe", allowed=(CUDA_OUTPUT_CAPACITY,))
        if output_capacity is not None and output_capacity < int(required.value):
            raise CudaBackendError(
                f"GPU CSR expand output capacity {output_capacity} is below required {required.value}; native result was not truncated"
            )
        capacity = int(required.value) if output_capacity is None else int(output_capacity)
        output = np.empty(capacity, dtype=np.uint32)
        actual = ctypes.c_uint64()
        self._backend._check(self._backend.lib.neuroseek_cuda_csr_session_expand(
            self._handle, _pointer(frontier_array, _U32_P) if frontier_array.size else None,
            frontier_array.size, relation_code, _pointer(output, _U32_P) if capacity else None,
            capacity, ctypes.byref(actual),
        ), "csr_session_expand")
        if actual.value != required.value:
            raise CudaBackendError(f"GPU CSR result cardinality changed between probes: {required.value} -> {actual.value}")
        return output[:actual.value]


class CudaExactBackend:
    """No-fallback CUDA exact scores, top-k, and relation-tagged CSR expansion."""

    def __init__(self, library: str | Path | None = None) -> None:
        location = str(library or os.environ.get("NEUROSEEK_CUDA_LIB", "/opt/neuroseek/lib/libneuroseek_cuda.so"))
        self.library = Path(location)
        self.lib = ctypes.CDLL(location)
        self.lib.neuroseek_cuda_last_error.argtypes = []
        self.lib.neuroseek_cuda_last_error.restype = ctypes.c_char_p
        self.lib.neuroseek_cuda_device_count.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.lib.neuroseek_cuda_device_count.restype = ctypes.c_int
        self.lib.neuroseek_cuda_exact_scores.argtypes = [_F32_P, _F32_P, ctypes.c_uint32, ctypes.c_uint32, _F32_P]
        self.lib.neuroseek_cuda_exact_scores.restype = ctypes.c_int
        self.lib.neuroseek_cuda_topk.argtypes = [_F32_P, ctypes.c_uint32, ctypes.c_uint32, _U32_P, _F32_P]
        self.lib.neuroseek_cuda_topk.restype = ctypes.c_int
        self.lib.neuroseek_cuda_expand.argtypes = [
            _U64_P, _U32_P, _U16_P, ctypes.c_uint32, ctypes.c_uint64,
            _U32_P, ctypes.c_uint32, ctypes.c_int32,
            _U32_P, ctypes.c_uint64, _U64_P,
        ]
        self.lib.neuroseek_cuda_expand.restype = ctypes.c_int
        self.lib.neuroseek_cuda_csr_session_create.argtypes = [_U64_P, _U32_P, _U16_P, ctypes.c_uint32, ctypes.c_uint64, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.neuroseek_cuda_csr_session_create.restype = ctypes.c_int
        self.lib.neuroseek_cuda_csr_session_destroy.argtypes = [ctypes.c_void_p]
        self.lib.neuroseek_cuda_csr_session_destroy.restype = ctypes.c_int
        self.lib.neuroseek_cuda_csr_session_expand.argtypes = [ctypes.c_void_p, _U32_P, ctypes.c_uint32, ctypes.c_int32, _U32_P, ctypes.c_uint64, _U64_P]
        self.lib.neuroseek_cuda_csr_session_expand.restype = ctypes.c_int

    def _detail(self) -> str:
        raw = self.lib.neuroseek_cuda_last_error()
        return raw.decode("utf-8", "replace") if raw else "no native diagnostic"

    def _check(self, status: int, operation: str, *, allowed: tuple[int, ...] = ()) -> None:
        if status != CUDA_OK and status not in allowed:
            raise CudaBackendError(f"{operation} failed with CUDA status {status}: {self._detail()}")

    def device_count(self) -> int:
        count = ctypes.c_int()
        self._check(self.lib.neuroseek_cuda_device_count(ctypes.byref(count)), "device_count")
        return count.value

    @staticmethod
    def _csr_array(value: np.ndarray, dtype: np.dtype, name: str) -> np.ndarray:
        array = np.asarray(value)
        if array.dtype != np.dtype(dtype) or array.ndim != 1 or not array.flags.c_contiguous:
            raise ValueError(f"{name} must be a C-contiguous one-dimensional {np.dtype(dtype)} array")
        return array

    @staticmethod
    def _frontier(value: np.ndarray | list[int]) -> np.ndarray:
        array = np.ascontiguousarray(value, dtype=np.uint32)
        if array.ndim != 1:
            raise ValueError("frontier must be one-dimensional")
        return array

    def expand_csr(
        self,
        offsets: np.ndarray,
        targets: np.ndarray,
        relations: np.ndarray,
        frontier: np.ndarray | list[int],
        relation: int | None = None,
        *,
        output_capacity: int | None = None,
    ) -> np.ndarray:
        """Synchronously expand a compact CSR frontier on CUDA.

        Passing ``output_capacity`` is useful for explicit capacity testing. A
        too-small buffer raises :class:`CudaBackendError` after native code has
        returned the required length; normal callers receive a precisely sized
        result by first issuing a zero-capacity probe. Duplicate output nodes
        are intentional and preserve one result per traversed edge.
        """
        offsets = self._csr_array(offsets, np.uint64, "offsets")
        targets = self._csr_array(targets, np.uint32, "targets")
        relations = self._csr_array(relations, np.uint16, "relations")
        frontier = self._frontier(frontier)
        if offsets.size == 0:
            raise ValueError("offsets must contain at least the sentinel")
        if targets.size != relations.size:
            raise ValueError("targets and relations lengths differ")
        if int(offsets[-1]) != targets.size:
            raise ValueError("offsets final sentinel must equal edge count")
        relation_code = -1 if relation is None else int(relation)
        if relation_code < -1 or relation_code > np.iinfo(np.uint16).max:
            raise ValueError("relation must be None/-1 or a uint16 relation ID")
        node_count, edge_count = offsets.size - 1, targets.size
        if node_count > np.iinfo(np.uint32).max:
            raise ValueError("node_count exceeds CUDA ABI uint32 range")
        required = ctypes.c_uint64()
        # A zero-capacity probe is deliberate: it avoids guessing high-degree
        # output cardinality and prevents silent truncation.
        status = self.lib.neuroseek_cuda_expand(
            _pointer(offsets, _U64_P), _pointer(targets, _U32_P), _pointer(relations, _U16_P),
            node_count, edge_count, _pointer(frontier, _U32_P) if frontier.size else None,
            frontier.size, relation_code, None, 0, ctypes.byref(required),
        )
        self._check(status, "expand size probe", allowed=(CUDA_OUTPUT_CAPACITY,))
        if output_capacity is not None and output_capacity < int(required.value):
            raise CudaBackendError(
                f"expand output capacity {output_capacity} is below required {required.value}; native result was not truncated"
            )
        capacity = int(required.value) if output_capacity is None else int(output_capacity)
        output = np.empty(capacity, dtype=np.uint32)
        actual = ctypes.c_uint64()
        status = self.lib.neuroseek_cuda_expand(
            _pointer(offsets, _U64_P), _pointer(targets, _U32_P), _pointer(relations, _U16_P),
            node_count, edge_count, _pointer(frontier, _U32_P) if frontier.size else None,
            frontier.size, relation_code, _pointer(output, _U32_P) if capacity else None,
            capacity, ctypes.byref(actual),
        )
        self._check(status, "expand")
        if actual.value != required.value:
            raise CudaBackendError(f"expand result cardinality changed between probes: {required.value} -> {actual.value}")
        return output[:actual.value]

    def expand_graph(
        self, graph: "GraphMmap", frontier: np.ndarray | list[int], relation: int | None = None, *, reverse: bool = False
    ) -> np.ndarray:
        """Expand directly from a :class:`GraphMmap` CSR without host copies."""
        if reverse:
            return self.expand_csr(graph.reverse_offsets, graph.reverse_neighbors, graph.reverse_relations, frontier, relation)
        return self.expand_csr(graph.forward_offsets, graph.forward_neighbors, graph.forward_relations, frontier, relation)

    def create_csr_session(self, offsets: np.ndarray, targets: np.ndarray, relations: np.ndarray) -> CudaCsrSession:
        """Validate once and upload one CSR traversal direction to the GPU."""
        offsets = self._csr_array(offsets, np.uint64, "offsets")
        targets = self._csr_array(targets, np.uint32, "targets")
        relations = self._csr_array(relations, np.uint16, "relations")
        if offsets.size == 0 or targets.size != relations.size or int(offsets[-1]) != targets.size:
            raise ValueError("CSR offsets/targets/relations lengths are inconsistent")
        node_count, edge_count = offsets.size - 1, targets.size
        if node_count > np.iinfo(np.uint32).max:
            raise ValueError("node_count exceeds CUDA ABI uint32 range")
        handle = ctypes.c_void_p()
        self._check(self.lib.neuroseek_cuda_csr_session_create(
            _pointer(offsets, _U64_P), _pointer(targets, _U32_P), _pointer(relations, _U16_P),
            node_count, edge_count, ctypes.byref(handle),
        ), "csr_session_create")
        if not handle.value:
            raise CudaBackendError("csr_session_create returned success without a handle")
        return CudaCsrSession(self, handle)

    def create_graph_session(self, graph: "GraphMmap", *, reverse: bool = False) -> CudaCsrSession:
        """Upload one GraphMmap direction once; no Python-object graph is made."""
        if reverse:
            return self.create_csr_session(graph.reverse_offsets, graph.reverse_neighbors, graph.reverse_relations)
        return self.create_csr_session(graph.forward_offsets, graph.forward_neighbors, graph.forward_relations)

    def scores(self, vectors: np.ndarray, query: np.ndarray) -> np.ndarray:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        query = np.ascontiguousarray(query, dtype=np.float32)
        if vectors.ndim != 2 or query.shape != (vectors.shape[1],) or not vectors.shape[0] or not vectors.shape[1]:
            raise ValueError("vectors must be non-empty [rows,dims] and query [dims]")
        if vectors.shape[0] > np.iinfo(np.uint32).max or vectors.shape[1] > np.iinfo(np.uint32).max:
            raise ValueError("vector shape exceeds CUDA ABI uint32 range")
        result = np.empty(vectors.shape[0], dtype=np.float32)
        self._check(self.lib.neuroseek_cuda_exact_scores(_pointer(vectors, _F32_P), _pointer(query, _F32_P), vectors.shape[0], vectors.shape[1], _pointer(result, _F32_P)), "exact_scores")
        return result

    def topk(self, scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        scores = np.ascontiguousarray(scores, dtype=np.float32)
        if scores.ndim != 1 or not 0 < k <= min(scores.size, 1024):
            raise ValueError("k must be in [1, min(rows, 1024)]")
        indices = np.empty(k, dtype=np.uint32)
        values = np.empty(k, dtype=np.float32)
        self._check(self.lib.neuroseek_cuda_topk(_pointer(scores, _F32_P), scores.size, k, _pointer(indices, _U32_P), _pointer(values, _F32_P)), "topk")
        return indices, values

    def self_test(self) -> None:
        vectors = np.asarray([[1., 0., 0.], [0., 2., 0.], [1., 1., 1.]], dtype=np.float32)
        query = np.asarray([1., .5, 0.], dtype=np.float32)
        np.testing.assert_allclose(self.scores(vectors, query), vectors @ query, rtol=1e-5, atol=1e-6)
