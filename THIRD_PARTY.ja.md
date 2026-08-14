# サードパーティコンポーネント

英語版: [THIRD_PARTY.md](THIRD_PARTY.md)

- 検出された Jetson/L4T R36.3 環境向けに選択した NVIDIA PyTorch コンテナ `nvcr.io/nvidia/pytorch:24.03-py3`。
- PyTorch、NumPy、Rust 標準ライブラリ、CMake、CUDA runtime/toolkit。
- Deep Graph Learning の Wikidata5M。ダウンロードの出所と観測した SHA-256 値は raw ファイルの隣に保存します。

CAGRA/cuVS のバイナリは同梱・主張していません。選択した本番 ANN フォールバックは custom CUDA exact score/top-k backend です。学習器の起動前に C ABI parity test で検証します。
