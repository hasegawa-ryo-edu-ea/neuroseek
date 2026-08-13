"""Semantic entity retrieval with an explicit exact-CUDA fallback.

This module deliberately does *not* claim an ANN index or cuVS/CAGRA support.
``CudaExactAnnBackend`` scores every indexed vector with NEUROSEEK's checked
CUDA C ABI, in bounded host-to-device batches, and uses a deterministic host
merge for top-k.  This is slower than a GPU-resident index, but means semantic
search remains available when no compatible Jetson ANN package can be built.

Embedding files are self-describing and immutable.  A store is either fully
aligned (one vector for every compact graph entity ID), or explicitly partial
(the bounded TransE fallback); callers must opt in to partial coverage.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np


FORMAT_VERSION = 1


class SemanticError(RuntimeError):
    """An invalid semantic artifact or unavailable explicitly selected backend."""


class SemanticTestInterruption(RuntimeError):
    """Explicit test-only stop after a durable full-store recovery point."""


class SemanticInterrupted(RuntimeError):
    """Graceful external interruption after saving full-TransE progress."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_topk(scores: np.ndarray, k: int) -> np.ndarray:
    """Score descending, row index ascending on ties; no unstable argpartition."""
    if not 0 < k <= scores.size:
        raise ValueError("k must be in [1, number of indexed embeddings]")
    # lexsort makes the tie rule an artifact invariant across NumPy versions.
    return np.lexsort((np.arange(scores.size, dtype=np.uint64), -scores))[:k]


@dataclass(frozen=True)
class EmbeddingManifest:
    format_version: int
    created_utc: str
    graph_entity_count: int
    indexed_entity_count: int
    dimension: int
    dtype: str
    normalized: bool
    complete_alignment: bool
    source: dict[str, Any]
    files: dict[str, dict[str, Any]]

    @classmethod
    def load(cls, root: Path) -> "EmbeddingManifest":
        try:
            raw = json.loads((root / "semantic_manifest.json").read_text())
            manifest = cls(**{field: raw[field] for field in cls.__annotations__})
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise SemanticError(f"invalid semantic manifest in {root}: {exc}") from exc
        if manifest.format_version != FORMAT_VERSION:
            raise SemanticError(f"unsupported semantic format {manifest.format_version}")
        if manifest.dtype != "float16":
            raise SemanticError("only compact float16 semantic vectors are supported")
        if manifest.dimension <= 0 or manifest.indexed_entity_count <= 0:
            raise SemanticError("semantic manifest has invalid shape")
        if manifest.complete_alignment and manifest.indexed_entity_count != manifest.graph_entity_count:
            raise SemanticError("complete semantic store does not cover every graph entity")
        return manifest


@dataclass(frozen=True)
class AnnSearchResult:
    entity_ids: np.ndarray
    scores: np.ndarray
    backend: str
    vectors_examined: int


class AnnBackend(Protocol):
    name: str

    def search(self, vectors: np.ndarray, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]: ...

    def stats(self) -> dict[str, Any]: ...


class CudaExactAnnBackend:
    """Checked custom-CUDA exact dot-product search, never a CPU fallback.

    The current C ABI takes host arrays and synchronizes per invocation.  The
    batch ceiling prevents a full float32 materialization of a multi-million
    FP16 embedding file on an 8 GB Jetson.  Top-k merging is host-side and
    deterministic; this fact is exposed in ``stats`` rather than hidden.
    """

    name = "cuda_exact"

    def __init__(self, library: str | None = None, max_batch_rows: int = 65_536) -> None:
        if max_batch_rows <= 0:
            raise ValueError("max_batch_rows must be positive")
        try:
            from neuroseek.cuda_backend import CudaExactBackend
            self._cuda = CudaExactBackend(library)
            count = self._cuda.device_count()
        except (OSError, RuntimeError) as exc:
            raise SemanticError(f"CUDA_EXACT selected but unavailable: {exc}") from exc
        if count < 1:
            raise SemanticError("CUDA_EXACT selected but CUDA reports no device")
        self.max_batch_rows = max_batch_rows
        self.device_count = count
        self.calls = 0

    def search(self, vectors: np.ndarray, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if vectors.ndim != 2 or query.shape != (vectors.shape[1],):
            raise ValueError("vectors must be [rows, dimensions] and query [dimensions]")
        if not 0 < k <= vectors.shape[0]:
            raise ValueError("invalid k")
        # Keep only k candidates per CUDA batch: O(batch + k*batches) RAM.
        ids: list[np.ndarray] = []
        scores: list[np.ndarray] = []
        for start in range(0, vectors.shape[0], self.max_batch_rows):
            stop = min(vectors.shape[0], start + self.max_batch_rows)
            batch_scores = self._cuda.scores(vectors[start:stop], query)
            self.calls += 1
            take = min(k, batch_scores.size)
            selected = _stable_topk(batch_scores, take)
            ids.append(selected.astype(np.uint64) + start)
            scores.append(batch_scores[selected])
        candidate_ids = np.concatenate(ids)
        candidate_scores = np.concatenate(scores)
        chosen = np.lexsort((candidate_ids, -candidate_scores))[:k]
        return candidate_ids[chosen].astype(np.uint32), candidate_scores[chosen].astype(np.float32)

    def stats(self) -> dict[str, Any]:
        return {"backend": self.name, "cuda_devices": self.device_count, "score_calls": self.calls,
                "max_batch_rows": self.max_batch_rows, "topk_location": "host_deterministic_merge"}


class NumpyExactBackend:
    """Explicit test/development backend.  It is never selected implicitly."""

    name = "numpy_exact_test_only"

    def __init__(self) -> None:
        self.calls = 0

    def search(self, vectors: np.ndarray, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        self.calls += 1
        scores = np.asarray(vectors, dtype=np.float32) @ np.asarray(query, dtype=np.float32)
        selected = _stable_topk(scores, k)
        return selected.astype(np.uint32), scores[selected].astype(np.float32)

    def stats(self) -> dict[str, Any]:
        return {"backend": self.name, "score_calls": self.calls, "warning": "test/development only; no CUDA executed"}


class AlignedEmbeddings:
    """Validated mmap embedding store keyed by compact graph entity ID."""

    def __init__(self, root: str | Path, *, expected_graph_entities: int | None = None,
                 allow_partial: bool = False, verify_hashes: bool = True) -> None:
        self.root = Path(root)
        self.manifest = EmbeddingManifest.load(self.root)
        if expected_graph_entities is not None and self.manifest.graph_entity_count != expected_graph_entities:
            raise SemanticError(f"embedding graph entity count {self.manifest.graph_entity_count} != expected {expected_graph_entities}")
        if not allow_partial and not self.manifest.complete_alignment:
            raise SemanticError("embedding store is partial; pass allow_partial=True only for bounded fallback experiments")
        for name, expected in self.manifest.files.items():
            path = self.root / name
            if not path.is_file() or path.stat().st_size != int(expected["bytes"]):
                raise SemanticError(f"missing or truncated semantic file: {path}")
            if verify_hashes and _sha256(path) != expected["sha256"]:
                raise SemanticError(f"semantic file hash mismatch: {path}")
        n, dim = self.manifest.indexed_entity_count, self.manifest.dimension
        self.entity_ids = np.memmap(self.root / "entity_ids.u32", mode="r", dtype="<u4", shape=(n,))
        self.vectors = np.memmap(self.root / "embeddings.f16", mode="r", dtype="<f2", shape=(n, dim))
        ids = np.asarray(self.entity_ids)
        if np.any(ids[1:] <= ids[:-1]) or int(ids[-1]) >= self.manifest.graph_entity_count:
            raise SemanticError("entity IDs must be unique, strictly increasing, and in graph range")
        if self.manifest.complete_alignment and not np.array_equal(ids, np.arange(n, dtype=np.uint32)):
            raise SemanticError("complete alignment requires entity_ids == [0..N)")

    def search(self, query: np.ndarray, k: int, backend: AnnBackend) -> AnnSearchResult:
        query = np.ascontiguousarray(query, dtype=np.float32)
        if query.shape != (self.manifest.dimension,) or not np.isfinite(query).all():
            raise ValueError("query must be finite and match embedding dimension")
        if self.manifest.normalized:
            norm = float(np.linalg.norm(query))
            if norm == 0.0:
                raise ValueError("normalized store cannot search with a zero vector")
            query = query / norm
        rows, scores = backend.search(self.vectors, query, k)
        return AnnSearchResult(np.asarray(self.entity_ids[rows], dtype=np.uint32), scores, backend.name, int(self.vectors.shape[0]))


def write_embedding_store(root: str | Path, entity_ids: np.ndarray, vectors: np.ndarray, *, graph_entity_count: int,
                          source: dict[str, Any], normalized: bool = True, overwrite: bool = False) -> Path:
    """Atomically publish an immutable semantic store from aligned arrays.

    This helper is intentionally strict: a caller cannot accidentally publish a
    vector file with a different row order than its entity mapping.
    """
    root = Path(root)
    ids = np.ascontiguousarray(entity_ids, dtype=np.uint32)
    values = np.ascontiguousarray(vectors, dtype=np.float32)
    if values.ndim != 2 or ids.ndim != 1 or ids.size != values.shape[0] or values.shape[1] == 0:
        raise ValueError("entity_ids [N] and vectors [N,D] are required")
    if ids.size == 0 or np.any(ids[1:] <= ids[:-1]) or int(ids[-1]) >= graph_entity_count:
        raise ValueError("entity IDs must be nonempty, unique, increasing, and in graph range")
    if not np.isfinite(values).all():
        raise ValueError("embeddings contain NaN or Inf")
    if normalized:
        norms = np.linalg.norm(values, axis=1)
        if np.any(norms == 0):
            raise ValueError("cannot normalize zero embedding")
        values = values / norms[:, None]
    complete = ids.size == graph_entity_count and np.array_equal(ids, np.arange(graph_entity_count, dtype=np.uint32))
    if root.exists() and not overwrite:
        raise FileExistsError(f"refuse to overwrite semantic store: {root}")
    stage = root.with_name(f".{root.name}.stage-{os.getpid()}-{time.time_ns()}")
    stage.mkdir(parents=True)
    try:
        ids.astype("<u4", copy=False).tofile(stage / "entity_ids.u32")
        values.astype("<f2").tofile(stage / "embeddings.f16")
        files = {name: {"bytes": path.stat().st_size, "sha256": _sha256(path)} for name in ("entity_ids.u32", "embeddings.f16") for path in [stage / name]}
        manifest = EmbeddingManifest(FORMAT_VERSION, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), graph_entity_count,
                                     int(ids.size), int(values.shape[1]), "float16", normalized, complete, source, files)
        (stage / "semantic_manifest.json").write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n")
        if root.exists():
            if not overwrite:
                raise FileExistsError(f"refuse to overwrite semantic store: {root}")
            backup = root.with_name(f"{root.name}.previous")
            if backup.exists():
                raise SemanticError(f"retained semantic backup exists: {backup}")
            os.replace(root, backup)
            try:
                os.replace(stage, root)
            except Exception:
                os.replace(backup, root)
                raise
            shutil.rmtree(backup)
        else:
            os.replace(stage, root)
        return root
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


@dataclass(frozen=True)
class TransEConfig:
    """Bounded fallback configuration; ``max_entities`` prevents Jetson OOM."""

    dimension: int = 64
    max_entities: int = 100_000
    steps: int = 1_000
    batch_size: int = 256
    learning_rate: float = 2e-3
    margin: float = 1.0
    seed: int = 1337
    device: str = "cuda"
    graph_manifest_sha256: str = ""
    checkpoint_every_steps: int = 0


def _publish_streamed_full_store(output: str | Path, entity: Any, *, graph_entity_count: int,
                                 dimension: int, source: dict[str, Any], chunk_rows: int) -> Path:
    """Publish a full aligned FP16 store without a full host float32 copy.

    This is the crucial 8 GB Jetson path: the trainable sparse embedding table
    stays on CUDA while one bounded row chunk is normalized, copied and written
    at a time.  A staging directory plus rename preserves the immutable-artifact
    contract even if the device/process is interrupted during export.
    """
    import torch
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    root = Path(output)
    if root.exists():
        raise FileExistsError(f"refuse to overwrite semantic store: {root}")
    stage = root.with_name(f".{root.name}.stage-{os.getpid()}-{time.time_ns()}")
    stage.mkdir(parents=True)
    try:
        np.arange(graph_entity_count, dtype="<u4").tofile(stage / "entity_ids.u32")
        vector_path = stage / "embeddings.f16"
        with vector_path.open("wb") as destination, torch.no_grad():
            for start in range(0, graph_entity_count, chunk_rows):
                stop = min(graph_entity_count, start + chunk_rows)
                values = entity.weight[start:stop]
                values = torch.nn.functional.normalize(values, p=2, dim=1, eps=1e-12)
                np.asarray(values.detach().cpu(), dtype=np.float16).tofile(destination)
            destination.flush()
            os.fsync(destination.fileno())
        files = {name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
                 for name in ("entity_ids.u32", "embeddings.f16") for path in [stage / name]}
        manifest = EmbeddingManifest(FORMAT_VERSION, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                     graph_entity_count, graph_entity_count, dimension, "float16", True,
                                     True, source, files)
        (stage / "semantic_manifest.json").write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n")
        os.replace(stage, root)
        return root
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def train_full_transe(graph: Any, output: str | Path, config: TransEConfig = TransEConfig(), *,
                      export_chunk_rows: int = 65_536,
                      test_stop_after_initial_checkpoint: bool = False,
                      stop_requested: Callable[[], bool] | None = None) -> Path:
    """Train and stream a fully aligned TransE semantic artifact on Jetson.

    Unlike the trial-only bounded fallback, this table has one compact vector
    for every graph entity.  Sparse SGD intentionally has no dense optimizer
    moments, avoiding the multi-gigabyte Adam state that would make an 8 GB
    Orin unreliable.  Every sampled positive is an actual mmap CSR edge;
    negative tails are sampled compact IDs.  Requested/observed steps are
    recorded in the immutable source metadata for later scientific reporting.
    """
    if config.dimension <= 0 or config.steps <= 0 or config.batch_size <= 0 or config.checkpoint_every_steps < 0:
        raise ValueError("invalid full TransE configuration")
    import torch
    n = int(graph.manifest.entity_count)
    if n < 2:
        raise SemanticError("TransE requires at least two graph entities")
    if config.device != "cuda" or not torch.cuda.is_available():
        raise SemanticError("full TransE requires CUDA; CPU full-table training is disallowed on this target")
    device = torch.device("cuda")
    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed); torch.cuda.manual_seed_all(config.seed)
    entity = torch.nn.Embedding(n, config.dimension, sparse=True, device=device)
    relation = torch.nn.Embedding(int(graph.manifest.relation_count), config.dimension, sparse=True, device=device)
    torch.nn.init.uniform_(entity.weight, -0.05, 0.05)
    torch.nn.init.uniform_(relation.weight, -0.05, 0.05)
    optimizer = torch.optim.SGD((entity.weight, relation.weight), lr=config.learning_rate)
    losses: list[float] = []
    sampled_edges = 0
    checkpoint_path = Path(output).parent / f".{Path(output).name}.training.ckpt"
    start_step = 0
    resumed_checkpoint = False
    if checkpoint_path.exists():
        try:
            saved = torch.load(checkpoint_path, map_location="cpu")
            expected = {"format": 1, "entities": n, "relations": int(graph.manifest.relation_count),
                        "dimension": config.dimension, "seed": config.seed,
                        "graph_manifest_sha256": config.graph_manifest_sha256}
            if not isinstance(saved, dict) or any(saved.get(key) != value for key, value in expected.items()):
                raise SemanticError("full TransE checkpoint is incompatible with the requested graph/configuration")
            entity.weight.data.copy_(saved["entity"].to(device=device, dtype=entity.weight.dtype))
            relation.weight.data.copy_(saved["relation"].to(device=device, dtype=relation.weight.dtype))
            rng.bit_generator.state = saved["rng_state"]
            start_step = int(saved["next_step"])
            losses = [float(value) for value in saved.get("losses", [])]
            sampled_edges = int(saved.get("sampled_edges", 0))
            resumed_checkpoint = True
        except (OSError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            raise SemanticError(f"cannot resume full TransE checkpoint {checkpoint_path}: {exc}") from exc

    def save_progress(next_step: int) -> None:
        """Atomically retain the full trainable table for interruption resume."""
        payload = {"format": 1, "entities": n, "relations": int(graph.manifest.relation_count),
                   "dimension": config.dimension, "seed": config.seed,
                   "graph_manifest_sha256": config.graph_manifest_sha256, "next_step": next_step,
                   "entity": entity.weight.detach().cpu(), "relation": relation.weight.detach().cpu(),
                   "rng_state": rng.bit_generator.state, "losses": losses[-256:], "sampled_edges": sampled_edges}
        temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + f".{os.getpid()}.tmp")
        try:
            torch.save(payload, temporary)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, checkpoint_path)
            directory_fd = os.open(checkpoint_path.parent, os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)

    # Publish a recoverable generation before the first expensive batch.  The
    # interval checkpoint remains configurable, but a power loss at step 1
    # must not force an otherwise-valid full-store build back to step zero.
    if start_step == 0 and not checkpoint_path.exists():
        save_progress(0)
        if test_stop_after_initial_checkpoint:
            # This hook is solely for the explicitly named CLI failure test;
            # normal production runs have no test-triggerable interruption.
            raise SemanticTestInterruption(f"test interruption after durable checkpoint {checkpoint_path}")

    for _step in range(start_step, config.steps):
        if stop_requested is not None and stop_requested():
            # The current table/RNG state is sufficient to continue from this
            # exact next step.  Do this before sampling so an interrupted
            # batch can never advance RNG without durable model state.
            save_progress(_step)
            raise SemanticInterrupted(f"full TransE interrupted after checkpointing next_step={_step}")
        heads = rng.integers(0, n, size=config.batch_size, dtype=np.uint32)
        tails = np.empty(config.batch_size, dtype=np.uint32)
        relations = np.empty(config.batch_size, dtype=np.int64)
        valid = np.zeros(config.batch_size, dtype=bool)
        for index, head in enumerate(heads):
            neighbors, rels = graph.neighbors(int(head))
            if not len(neighbors):
                continue
            selected = int(rng.integers(len(neighbors)))
            tails[index], relations[index], valid[index] = neighbors[selected], rels[selected], True
        if not valid.any():
            continue
        head_tensor = torch.as_tensor(heads[valid].astype(np.int64, copy=False), device=device)
        tail_tensor = torch.as_tensor(tails[valid].astype(np.int64, copy=False), device=device)
        relation_tensor = torch.as_tensor(relations[valid], device=device)
        negative_tensor = torch.as_tensor(rng.integers(0, n, size=int(valid.sum()), dtype=np.int64), device=device)
        positive = torch.linalg.vector_norm(entity(head_tensor) + relation(relation_tensor) - entity(tail_tensor), ord=1, dim=1)
        negative = torch.linalg.vector_norm(entity(head_tensor) + relation(relation_tensor) - entity(negative_tensor), ord=1, dim=1)
        loss = torch.relu(config.margin + positive - negative).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("full TransE produced a non-finite loss")
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        losses.append(float(loss.detach().cpu())); sampled_edges += int(valid.sum())
        if config.checkpoint_every_steps and (_step + 1) % config.checkpoint_every_steps == 0:
            save_progress(_step + 1)
    if not losses:
        raise SemanticError("full TransE found no traversable CSR edges")
    result = _publish_streamed_full_store(output, entity, graph_entity_count=n, dimension=config.dimension,
                                          chunk_rows=export_chunk_rows, source={
        "kind": "full_transe_streaming", "algorithm": "TransE margin ranking sparse SGD",
        "seed": config.seed, "steps_requested": config.steps, "steps_with_edges": len(losses),
        "sampled_positive_edges": sampled_edges, "last_loss": losses[-1], "device": str(device),
        "optimizer": "sparse_sgd_no_dense_moments", "export_chunk_rows": export_chunk_rows,
        # A step-zero checkpoint is still a genuine recovery after process
        # interruption; reporting only progress>0 would hide that safeguard.
        "resumed_from_checkpoint": resumed_checkpoint,
        "graph_manifest_sha256": config.graph_manifest_sha256,
    })
    checkpoint_path.unlink(missing_ok=True)
    return result


def train_bounded_transe(graph: Any, output: str | Path, config: TransEConfig = TransEConfig()) -> Path:
    """Train a compact TransE-style *partial* embedding store from mmap CSR.

    It samples an induced subset deterministically, so it does not build a
    Python graph or allocate embeddings for all Wikidata5M entities.  It is a
    dependable fallback/trial path, not a claim of full-graph KGE quality.
    """
    if config.dimension <= 0 or config.max_entities <= 1 or config.steps <= 0 or config.batch_size <= 0:
        raise ValueError("invalid bounded TransE configuration")
    import torch
    n = int(graph.manifest.entity_count)
    if n < 2:
        raise SemanticError("TransE requires at least two graph entities")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise SemanticError("bounded TransE requested CUDA but torch CUDA is unavailable")
    device = torch.device(config.device)
    chosen_count = min(n, config.max_entities)
    # Evenly spaced IDs avoid an accidental source-file-prefix-only sample.
    selected = np.unique(np.linspace(0, n - 1, chosen_count, dtype=np.uint32))
    if selected.size < 2:
        raise SemanticError("bounded TransE entity selection is too small")
    generator = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)
    entity = torch.nn.Embedding(selected.size, config.dimension, sparse=True, device=device)
    relation = torch.nn.Embedding(int(graph.manifest.relation_count), config.dimension, sparse=True, device=device)
    torch.nn.init.uniform_(entity.weight, -0.05, 0.05); torch.nn.init.uniform_(relation.weight, -0.05, 0.05)
    optimizer = torch.optim.SparseAdam(list(entity.parameters()) + list(relation.parameters()), lr=config.learning_rate)

    def index_of(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        slots = np.searchsorted(selected, ids)
        good = slots < selected.size
        good[good] &= selected[slots[good]] == ids[good]
        return slots, good

    losses: list[float] = []
    for _step in range(config.steps):
        heads = generator.choice(selected, size=config.batch_size, replace=True)
        tails = np.empty(config.batch_size, dtype=np.uint32)
        rels = np.empty(config.batch_size, dtype=np.int64)
        valid = np.zeros(config.batch_size, dtype=bool)
        for i, head in enumerate(heads):
            neighbors, relations = graph.neighbors(int(head))
            if len(neighbors) == 0:
                continue
            # Bounded retries retain only edges entirely in the selected index.
            for _ in range(8):
                edge = int(generator.integers(len(neighbors)))
                target = int(neighbors[edge])
                pos = int(np.searchsorted(selected, target))
                if pos < selected.size and int(selected[pos]) == target:
                    tails[i], rels[i], valid[i] = target, int(relations[edge]), True
                    break
        if not valid.any():
            continue
        head_slots, _ = index_of(heads[valid]); tail_slots, _ = index_of(tails[valid])
        neg_slots = generator.integers(selected.size, size=valid.sum(), dtype=np.int64)
        h = torch.as_tensor(head_slots, device=device); t = torch.as_tensor(tail_slots, device=device)
        r = torch.as_tensor(rels[valid], device=device); neg = torch.as_tensor(neg_slots, device=device)
        positive = torch.linalg.vector_norm(entity(h) + relation(r) - entity(t), ord=1, dim=1)
        negative = torch.linalg.vector_norm(entity(h) + relation(r) - entity(neg), ord=1, dim=1)
        loss = torch.relu(config.margin + positive - negative).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("bounded TransE produced a non-finite loss")
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        losses.append(float(loss.detach().cpu()))
    if not losses:
        raise SemanticError("bounded TransE found no edges within its selected entity subset")
    vectors = entity.weight.detach().float().cpu().numpy()
    return write_embedding_store(output, selected, vectors, graph_entity_count=n, normalized=True, source={
        "kind": "bounded_transe_fallback", "algorithm": "TransE margin ranking", "seed": config.seed,
        "steps_requested": config.steps, "steps_with_edges": len(losses), "last_loss": losses[-1],
        "max_entities": config.max_entities, "device": str(device), "partial_coverage_required": True,
        "graph_manifest_sha256": config.graph_manifest_sha256,
    })
