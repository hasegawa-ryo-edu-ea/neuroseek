# Architecture

Immutable Wikidata triples compile to compact, memory-mapped forward and reverse CSR. `uint64` offsets, `uint32` node IDs, and `uint16` relation IDs avoid string work in hot paths. Human labels remain TSV lookup files.

Rust owns NEURO-ISA semantics, VM accounting, graph CPU reference, and proof validation. CUDA provides C ABI bulk primitives compiled for SM87; parity tests compare them against host references. Python owns the compact query-conditioned navigator, PPO update, curriculum/task adapter, checkpoint lifecycle, and append-only metrics. The Rust terminal dashboard only tails metrics JSONL.

The ANN interface is designed to select CAGRA only after it passes in-container build/correctness tests. The mandatory initial backend is CUDA exact dot-product/top-k. Failure to load either CUDA backend is fatal outside named smoke mode.

The production image is pinned to NVIDIA L4T ML R36.2 (the JetPack 6 / CUDA
12.2 generation) because the detected Jetson Linux R36.3 host runs CUDA 12.2.
An initially tested generic PyTorch 24.03 image exposed a CUDA 12.4 userspace:
PyTorch could enumerate Orin, but the custom CUDA PTX failed with an unsupported
toolchain error. Container GPU visibility is therefore not treated as a CUDA
compatibility proof; `neuroseek-cuda-parity` is a required startup gate.
