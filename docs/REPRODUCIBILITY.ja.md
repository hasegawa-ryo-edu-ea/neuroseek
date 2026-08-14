# 再現性

英語版: [REPRODUCIBILITY.md](REPRODUCIBILITY.md)

ホストプローブと解決済み依存関係 manifest は `artifacts/` に保存します。データセットのダウンロードには、URL、時刻、バイト数、SHA-256 を記録します。処理済みデータは `data/processed/` に不変として保存し、実行、ログ、チェックポイントは `runs/` に置きます。

各 run は設定、デバイス選択、データセット manifest、チェックポイント状態を保存します。コンテナはこのホストのプローブ後に選んだ Jetson/L4T 互換イメージに固定され、学習サービス開始前に CUDA smoke コマンドを通過する必要があります。

ビルドイメージは `Dockerfile` で digest 固定です。APT のビルド依存関係は検証済みイメージに実際に入ったバージョンに固定し、Python 依存関係も固定し、Rust アプリケーション依存関係は `Cargo.lock` に固定します。`artifacts/environment_manifest.json` はこの解決済み集合を記録し、ホストまたはツールチェーンを変更した後に `python3 scripts/write_environment_manifest.py` で再生成します。後日の再ビルドは、設定した APT ミラーがいずれかの正確なパッケージを保持していない場合に失敗することがあります。これは新しいコンパイラやネイティブライブラリへ黙って置き換えないための意図した挙動です。
