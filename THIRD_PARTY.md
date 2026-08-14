# Third-party components

日本語版: [THIRD_PARTY.ja.md](THIRD_PARTY.ja.md)

- NVIDIA PyTorch container `nvcr.io/nvidia/pytorch:24.03-py3`, selected for the detected Jetson/L4T R36.3 environment.
- PyTorch, NumPy, Rust standard library, CMake, CUDA runtime/toolkit.
- Wikidata5M by Deep Graph Learning. Download provenance and observed SHA-256 values are saved beside the raw files.

No CAGRA/cuVS binary is claimed or bundled. The selected production ANN fallback is the custom CUDA exact score/top-k backend, validated through its C ABI parity test before trainer startup.
