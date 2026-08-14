# NEUROSEEK のグラフ形式

英語版: [GRAPH_FORMAT.md](GRAPH_FORMAT.md)

`scripts/preprocess.py` は不変の Wikidata5M TSV 分割ファイルをコンパイルし、正常終了した場合にだけ出力ディレクトリをアトミックに公開します。ID は 0 始まりのコンパクトな `uint32` ノードと `uint16` 関係です。どちらかの幅を超えるデータセットはコンパイラが拒否します。

順方向と逆方向の CSR は別々に実体化します。`original_triples` は入力された事実の正確な件数です。`traversal_edges` はそのちょうど 2 倍で、探索用のインデックスにすぎず、追加の事実ではありません。

| ファイル | 型 | 意味 |
| --- | --- | --- |
| `forward_offsets.u64` | u64, N+1 | 始点ごとの CSR 境界 |
| `forward_neighbors.u32` / `forward_relations.u16` | E | 各元事実の終点／関係 |
| `reverse_*` | 同上 | 終点から索引した始点／関係 |
| `entities.tsv`, `relations.tsv` | text | 0 始まり ID、元 ID、表示ラベル |
| `manifest.json` | JSON | 元データのハッシュ、次元、バイナリハッシュ、形式バージョン |

実行時は NumPy の memmap を使い、グラフを Python の辞書に逆シリアル化しません。上流の別名ファイルを任意でコンパイラに渡すと、3 列目の表示ラベルを埋められます。別名は CUDA／VM のホットデータには入りません。
