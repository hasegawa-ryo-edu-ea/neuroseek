"""Read-only, memory-mapped compact graph storage used by NEUROSEEK.

The binary layout deliberately contains no Python object graph: offsets are u64,
neighbors u32 and relation IDs u16.  It is also a CPU oracle for native kernels.
"""
from __future__ import annotations
import json
import mmap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import numpy as np


@dataclass(frozen=True)
class GraphManifest:
    entity_count: int
    relation_count: int
    original_triples: int
    traversal_edges: int
    files: dict

    @classmethod
    def load(cls, root: Path) -> "GraphManifest":
        data = json.loads((root / "manifest.json").read_text())
        return cls(**{key: data[key] for key in cls.__annotations__})


class GraphMmap:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest = GraphManifest.load(self.root)
        n = self.manifest.entity_count
        e = self.manifest.original_triples
        self.forward_offsets = np.memmap(self.root / "forward_offsets.u64", dtype="<u8", mode="r", shape=(n + 1,))
        self.forward_neighbors = np.memmap(self.root / "forward_neighbors.u32", dtype="<u4", mode="r", shape=(e,))
        self.forward_relations = np.memmap(self.root / "forward_relations.u16", dtype="<u2", mode="r", shape=(e,))
        self.reverse_offsets = np.memmap(self.root / "reverse_offsets.u64", dtype="<u8", mode="r", shape=(n + 1,))
        self.reverse_neighbors = np.memmap(self.root / "reverse_neighbors.u32", dtype="<u4", mode="r", shape=(e,))
        self.reverse_relations = np.memmap(self.root / "reverse_relations.u16", dtype="<u2", mode="r", shape=(e,))
        # Human-readable lookup is intentionally outside the hot CSR path.
        # A sparse byte-offset index keeps 4.59M labels out of Python objects
        # while making occasional trace/TUI resolution bounded.
        self._entities_text = _TsvLookup(self.root / "entities.tsv", n)
        self._relations_text = _TsvLookup(self.root / "relations.tsv", self.manifest.relation_count)

    def neighbors(self, entity: int, reverse: bool = False) -> tuple[np.ndarray, np.ndarray]:
        if not 0 <= entity < self.manifest.entity_count:
            raise IndexError(f"entity ID out of range: {entity}")
        offsets = self.reverse_offsets if reverse else self.forward_offsets
        nodes = self.reverse_neighbors if reverse else self.forward_neighbors
        relations = self.reverse_relations if reverse else self.forward_relations
        start, end = int(offsets[entity]), int(offsets[entity + 1])
        return nodes[start:end], relations[start:end]

    def has_edge(self, source: int, relation: int, target: int) -> bool:
        nodes, rels = self.neighbors(source)
        return bool(np.any((nodes == target) & (rels == relation)))

    def edges(self, entity: int, reverse: bool = False) -> Iterator[tuple[int, int]]:
        nodes, relations = self.neighbors(entity, reverse)
        yield from ((int(n), int(r)) for n, r in zip(nodes, relations))

    def entity_identifier(self, entity: int) -> str:
        return self._entities_text.row(entity)[1]

    def entity_label(self, entity: int) -> str:
        return self._entities_text.row(entity)[2]

    def relation_identifier(self, relation: int) -> str:
        return self._relations_text.row(relation)[1]

    def relation_label(self, relation: int) -> str:
        return self._relations_text.row(relation)[2]

    def find_entity_identifier(self, identifier: str) -> int | None:
        """Resolve an exact Wikidata Q identifier without materialising labels."""
        return self._entities_text.find_identifier(identifier)

    def find_relation_identifier(self, identifier: str) -> int | None:
        """Resolve an exact Wikidata P identifier without materialising labels."""
        return self._relations_text.find_identifier(identifier)


class _TsvLookup:
    """Sparse mmap index for ordered ``id<TAB>identifier<TAB>label`` rows."""

    _STRIDE = 4096

    def __init__(self, path: Path, expected_rows: int) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"graph display lookup is missing: {path}")
        self.path = path
        self.expected_rows = expected_rows
        self._file = path.open("rb")
        self._mapped = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self._offsets = self._build_sparse_offsets()

    def _build_sparse_offsets(self) -> list[int]:
        offsets = [0]
        offset = 0
        row = 0
        size = self._mapped.size()
        while offset < size:
            if row and row % self._STRIDE == 0:
                offsets.append(offset)
            ending = self._mapped.find(b"\n", offset)
            if ending < 0:
                raise ValueError(f"unterminated display lookup row in {self.path}")
            row += 1
            offset = ending + 1
        if row != self.expected_rows:
            raise ValueError(f"display lookup row count {row} != expected {self.expected_rows}: {self.path}")
        return offsets

    def row(self, identifier: int) -> tuple[str, str, str]:
        if not 0 <= identifier < self.expected_rows:
            raise IndexError(f"display lookup ID out of range: {identifier}")
        block = identifier // self._STRIDE
        offset = self._offsets[block]
        for _ in range(identifier % self._STRIDE + 1):
            ending = self._mapped.find(b"\n", offset)
            if ending < 0:
                raise ValueError(f"truncated display lookup: {self.path}")
            line = self._mapped[offset:ending]
            offset = ending + 1
        fields = line.decode("utf-8").split("\t")
        if len(fields) != 3 or int(fields[0]) != identifier:
            raise ValueError(f"malformed or out-of-order display lookup: {self.path}")
        return fields[0], fields[1], fields[2]

    def find_identifier(self, identifier: str) -> int | None:
        """Find a TSV second-column identifier through the mmap, allocation-free.

        The compact IDs are intentionally not ordered by Wikidata identifier,
        so a binary search is invalid.  ``mmap.find`` scans the immutable text
        directly and keeps an occasional interactive lookup out of RAM.
        """
        marker = f"\t{identifier}\t".encode("utf-8")
        position = self._mapped.find(marker)
        if position < 0:
            return None
        beginning = self._mapped.rfind(b"\n", 0, position)
        beginning = 0 if beginning < 0 else beginning + 1
        ending = self._mapped.find(b"\n", position)
        if ending < 0:
            return None
        fields = self._mapped[beginning:ending].decode("utf-8").split("\t")
        if len(fields) != 3 or fields[1] != identifier:
            return None
        return int(fields[0])
