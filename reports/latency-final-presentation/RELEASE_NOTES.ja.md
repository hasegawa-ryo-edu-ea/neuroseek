# NEUROSEEK Jetson 低レイテンシーモデル

英語版: [RELEASE_NOTES.md](RELEASE_NOTES.md)

このパッケージには、完了したレイテンシー特化 NEUROSEEK チェックポイントと、最終の保持テスト評価を確認するために必要な不変 artifact を含めます。

## 最終保持テスト結果

派生チェックポイントは、CUDA exact backend を使い、Wikidata5M に基づく固定テスト 256 件で評価しました。親チェックポイントと比べ、平均エンドツーエンド遅延は 31.92 ms から 31.11 ms（-2.56%）、p95 遅延は 57.10 ms から 53.02 ms（-7.14%）、平均計算クレジットは 233.98 から 146.02（-37.59%）になりました。回答正確度と独立検証済み証明率は 98.05% から 97.27%（-0.78 ポイント）へ変化しました。

## 内容

- `checkpoints/final.ckpt`: 評価済みの不変チェックポイント
- `config.toml` と `manifest.json`: run 設定と provenance
- `exports/`: 正式な最終評価、ベンチマーク比較、hardware summary、checkpoint binding、trace/operator export
- `analysis/`: 同じ 256 件に対する完了後の task-family 構造再評価。2 回目の遅延測定ではありません。

## 適用範囲

これは構造化グラフタスクの研究 artifact です。自然言語からの relation parsing、外部 SOTA に対する優位性、汎用的な Local LLM の知識更新システムを単独で実証するものではありません。
