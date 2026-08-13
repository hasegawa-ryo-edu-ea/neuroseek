# NEUROSEEK graph format

`scripts/preprocess.py` compiles the immutable Wikidata5M TSV split files into a directory published atomically only after successful completion. IDs are compact, zero-based `uint32` nodes and `uint16` relations; the compiler rejects a dataset that exceeds either width.

Forward and reverse CSR are materialized separately. `original_triples` is the exact number of input facts; `traversal_edges` is exactly twice that count and is only a traversal index, not additional facts.

| File | Type | Meaning |
| --- | --- | --- |
| `forward_offsets.u64` | u64, N+1 | CSR boundaries by source |
| `forward_neighbors.u32` / `forward_relations.u16` | E | target/relation for each original fact |
| `reverse_*` | same | source/relation indexed by target |
| `entities.tsv`, `relations.tsv` | text | zero-based ID, original ID, display label |
| `manifest.json` | JSON | source hashes, dimensions, binary hashes, format version |

The runtime uses NumPy memmaps and does not deserialize the graph into Python dictionaries. Pass the optional upstream alias files to the compiler to populate the third display-label column; aliases never enter CUDA/VM hot data.
