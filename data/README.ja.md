# データとモデル asset

英語版: [README.md](README.md)

ソースコードは Git に保存します。Wikidata5M コーパス、処理済みグラフ、semantic モデルは、通常の GitHub ファイルサイズ上限を超えるファイルを含むため、`initial-data-and-models` GitHub Release の asset として公開します。

リポジトリの root で、両方の archive をダウンロードして展開します。

```bash
tar --zstd -xf neuroseek-wikidata5m-raw.tar.zst
tar --zstd -xf neuroseek-processed-data-and-models.tar.zst
```

これにより次を復元します。

- `data/raw/wikidata5m/`: 元の Wikidata5M ファイル
- `data/processed/`: 処理済みグラフ、`semantic_full`、その manifest

runtime cache、build output、学習 run、log、probe artifact は Release に含みません。
