# NEURO-ISA

英語版: [NEURO_ISA.md](NEURO_ISA.md)

NEURO-ISA は NEUROSEEK の探索方策が使う、境界付きの命令言語です。`rust/core` が決定的な CPU 意味論の参照実装です。CUDA カーネルは同等の一括処理を高速化できますが、出力はこの参照実装と照合し、同じリソース計数と証明根拠を保持する必要があります。

## 状態と不変条件

VM はクエリを可変の機械状態の外に置き、frontier レジスタ `F0`〜`F3`、回答候補 `A`、残りクレジット `B`、バックトラック履歴、観測した証明を持ちます。frontier はコンパクトな `uint32` ノード ID とスコアだけを保持します。グラフのホットパスは CSR の `uint64` offset、隣接する `uint32` target、`uint16` relation を使い、ラベルは VM に入りません。

各命令はアトミックです。決定的コストが `B` を超える、ANN プロバイダがない、レジスタ引数が不正などで実行できない場合、frontier、証明、予算、深さ、成功操作カウンタを復元します。拒否は `attempted_instructions` と `failed_instructions` だけに記録します。これにより、エラー後に一部だけ展開した frontier をスケジューラが続行することを防ぎます。

| 命令 | 意味 | 決定的なクレジットモデル |
| --- | --- | --- |
| `SEED(node)` | 検証済みノードをスコア 1.0 で `F0` に書く。 | 1 |
| `ANN(k)` | 設定済み ANN から最大 `k` 件の意味候補を取得し、重複 ID は最大スコアへ正規化して `F0` に書く。 | `k` |
| `EXPAND_REL(r)` | relation `r` を持つ各 `F0` ノードの CSR 隣接を走査し、正規化した target を `F0` に書き、実際にたどった edge を証明へ追加する。 | 出力 edge 数 |
| `EXPAND_ANY` | relation 条件なしの同じ走査。 | 出力 edge 数 |
| `FILTER(r)` | 自身から relation `r` の出力 edge を少なくとも一つ持つ `F0` ノードだけを残す。 | 入力 frontier サイズ |
| `INTERSECT(Fi)` | `F0` と `Fi` に共通するノード ID を残す。 | 両オペランドのサイズ |
| `UNION(Fi)` | `F0` と `Fi` を和集合にし、重複スコアは最大を取る。 | 両オペランドのサイズ |
| `PRUNE(k)` / `TOPK(k)` | スコア降順、ノード ID 昇順で並べ、先頭 `k` 件を残す。 | 入力 frontier サイズ |
| `VERIFY` | `F0` を `A` にコピーし、最初の決定的候補を提案証明の答えにする。 | 1 |
| `BACKTRACK` | 直前に保存した frontier レジスタのスナップショットを復元する。 | 1 |
| `PREFETCH` / `EVICT` | 明示的なキャッシュ意図イベント。CPU 参照では隠れたキャッシュ副作用を持たない。 | 1 |
| `STOP` | プログラムを閉じる。以後の命令は拒否する。 | 1 |

`EXPAND_*` は、タスク生成器のデモ経路ではなく、その命令が実際に選んだ CSR edge をすべて記録します。`VERIFY` は正しさを意味しません。別の `validate_proof()` が、必要なクエリ edge がすべて観測されたこと、証明 edge が CSR に根拠を持つこと、答えが根拠に支えられた許可済み回答であることを検証します。この検証器は方策や教師軌跡から独立しています。

## 可観測性

各受理命令について VM は、frontier サイズの前後、提案回答、証明 edge 数、残りクレジットを含むメモリ内 `VmStep` を出します。また、試行／成功／失敗命令数、訪問ノード、調査 edge、ANN 呼び出し、クレジット、推定 CSR バイト数、最大深さ、バックトラック、prefetch、evict、frontier peak、証明 edge 数を記録します。

CLI はこれらを JSONL として直列化します。壁時計時間と物理 GPU テレメトリは VM が作るのではなく、呼び出し側または学習器が測定します。

## JSONL 受け入れランナー

`neuroseek-native` は改行区切りレコード列を受け取ります。各 `program` レコードより前に有効な `graph` レコードが必要です。プロセスは明示的なクレジットで境界付けられ、`graph_loaded`、受理操作ごとの `vm_step`、`vm_result` を出力します。不正な入力と VM エラーは panic ではなく構造化された `*_error` 行になります。

```bash
cargo run -p neuroseek-native -- --jsonl rust/cli/tests/fixtures/path_proof.jsonl
```

テスト済み fixture は 2 ホップ CSR `0 -7-> 1 -8-> 2` を作り、`SEED`、relation 展開 2 回、`VERIFY`、`STOP` を実行して、回答 `2` に対して `"proof_valid":true` を報告します。これは意図的に小さくした受け入れ／診断経路であり、本番の Wikidata5M ローダーではありません。

入力形式:

```json
{"type":"graph","nodes":3,"edges":[{"src":0,"relation":7,"dst":1}]}
{"type":"program","budget":16,"instructions":[{"op":"SEED","node":0},{"op":"STOP"}]}
```

対応する操作名は `SEED`、`ANN`、`EXPAND_REL`、`EXPAND_ANY`、`FILTER`、`INTERSECT`、`UNION`、`PRUNE`、`TOP_K`、`VERIFY`、`BACKTRACK`、`PREFETCH`、`EVICT`、`STOP` です。この単体の受け入れランナーには意味バックエンドが設定されていないため、`ANN` は意図的にエラーを返します。本番の native 境界はバックエンドを明示的に提供する必要があります。
