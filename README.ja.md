# NEUROSEEK

NEUROSEEKは、Jetson Orin Nanoで動くローカル知識グラフ探索システムです。
学習済みコントローラとCUDAを使ってWikidata5Mのグラフを探索します。返す答えには
グラフ上の根拠が付き、Rustの検証器がその根拠を独立して確認します。これは研究・
デモ用のローカルシステムであり、ホスト型APIや汎用チャットボットではありません。

英語版: [README.md](README.md)

## できること

- Wikidata5Mを、変更しないメモリマップCSRグラフへ変換します。
- 埋め込みとCUDA探索で候補を絞り込みます。
- 学習済みのNEURO-ISA方策がグラフ操作を選びます。
- グラフ上の根拠を検証できた答えだけを返します。
- 設定、ハッシュ、チェックポイント、追記型メトリクスを残し、実験を再現しやすくします。

## 主な用途とGUIの位置付け

主な用途は、小型のローカルLLMが使える知識を増やすことです。APIを持つアプリケーション
から、同梱のMCPサーバー経由でローカルLLMをNEUROSEEKへ接続できます。取得したローカル
グラフの事実や検証済み探索結果を、呼び出し側のLLMが回答の根拠として使います。

ターミナルUIとWeb UIは、AI支援で短期間に作成したデモ・確認用のGUIです。モデルと証拠
を確認する用途には使えますが、主要な製品UIではなく、実験的なものとして扱ってください。
連携にはGUIではなくMCPサーバーを使います。

```bash
python3 python/neuroseek/mcp_server.py
```

MCPサーバーは標準入出力で通信し、ローカル事実の取得と検証用探索を読み取り専用で提供
します。このサーバー自体は回答を生成せず、HTTP APIも提供しません。APIの公開と最終的な
回答生成は、呼び出し側のローカルLLMアプリケーションが担当します。

## アーキテクチャと実装上の工夫

NEUROSEEKは、役割と信頼境界が異なる部品へ意図的に分割しています。学習済み方策やGPUは
探索空間を速く絞り込むために使い、答えを確定できるのは不変なグラフ上の根拠だけです。
この分離により、エッジデバイス向けに最適化しても、類似度スコアを事実として扱いません。

```text
Wikidata5M のトリプル
       │  決定的な前処理 + manifest/hash
       ▼
不変 mmap 順方向/逆方向 CSR ──► Rust NEURO-ISA VM ──► 証明検証器
       │                                      ▲                  │
       │                                      │                  ▼
       ▼                              予算付き方策プログラム  有効時だけ答え
整列済み entity vector ──► CUDA 完全スコア/top-k ──► 候補ジャンプ
       │
       └── semantic候補は探索を案内するだけで、グラフ証明を置き換えません
```

### 1. グラフ上の根拠を唯一の確定情報にする

オフラインコンパイラはトリプルを、不変でメモリマップ可能な順方向・逆方向CSR配列に変換
します。ホットパスでは`uint64`のoffset、`uint32`のnode ID、`uint16`のrelation IDという
コンパクトな数値表現を使い、人間向けラベルは別のTSV lookupに置きます。これにより探索中の
文字列処理や巨大なメモリ内オブジェクトグラフを避け、学習・評価・読み取り専用検索で同じ
処理済みグラフを共有できます。

Rust VMは、明示的な予算内で制約付きNEURO-ISAプログラムを実行します。探索したnode/edge、
命令数、ANN呼び出し、credit、frontier状態、実行中に実際に観測したedgeを記録します。続く
証明検証器が、候補の答えと各proof edgeを不変CSRグラフに照らして確認します。もっともらしく
見えても有効なグラフ根拠がない候補は、答えに変換せず未検証として残します。

### 2. Semantic検索は補助であり、権威にはしない

semantic laneは、各コンパクトgraph entity IDに一つのentity vectorを整列させます。artifactには
順序付きID、row-major FP16 embedding、次元・正規化・生成元・被覆率・hashを含むmanifestを
格納します。フルモードではIDが完全なコンパクトID範囲と厳密に一致した場合だけを受理し、
vector数が同じというだけでは整列の証拠とみなしません。

現在のproduction backendは、境界付きbatchでのCUDA完全内積スコアリングと、決定的な
host-side top-k mergeを意図的に採用しています。これは正しさを優先した選択です。backendは
`cuda_exact`という識別子とhost mergeの事実を報告し、近似・完全GPU indexであるかのようには
見せません。明示的なsmoke/test経路以外でCUDAに失敗した場合は停止し、NumPyやCPUへの黙った
フォールバックはしません。trial向けの境界付きTransE artifactはmanifestでpartialと明示され、
フルWikidata5M実行を暗黙に満たすことはできません。

### 3. 実機で測った探索コストで学習を制約する

Pythonは、コンパクトなquery-conditioned navigator、curriculum adapter、training loopを担当
します。curriculumはまず実測したCUDA探索の挙動を収集して永続的なhardware-cost modelを適合
させ、次にbehavior cloningで有効なproof programをアンカーしてから、PPOで方策を改善します。
方策には予測latencyや命令数のpenaltyを与えられますが、有効なproofが成功の主要条件です。
短くても無効なtraceを成功した答えとして報酬化しません。

タスクは自由文の質問ではなく、構造化した`QuerySpec`グラフ問題です。train/validation/testの
splitは別々のseedと永続化したファイルを持ちます。これにより同じ保持タスク上で、正答率、
証明有効性、調べたnode/edge、命令数、ANN呼び出し、credit、latency、取得可能な場合のenergyを
比較できます。

### 4. CUDA互換性をチェック項目ではなくリリースゲートにする

runtime imageは、Jetson/L4T互換のNVIDIA imageへdigestで固定しています。これは実際の互換性
制約から来ています。PyTorchがGPUを列挙できても、custom CUDA codeがhost/toolchainの組合せで
実行できる証拠にはなりません。学習前の起動経路では、custom SM87 CUDA primitiveとhost reference
が一致することを必須にしています。CUDA codeは細いC ABIで公開し、bulk GPU operationをPythonの
orchestrationから分離することで、parity failureを早期に見える形にします。

### 5. 長時間のエッジ実行を安全に失敗・明示的に再開できるようにする

フルsemantic準備には50時間curriculumとは別の上限を設け、重い処理の前に最初のrecovery
checkpointを書きます。進捗・学習checkpointはatomicにpublishしてfsyncし、再開時に検証します。
互換性のない状態や壊れた状態は、新しいrunと混ぜずにfail closedします。完全semantic artifactは
dataとmanifestがそろってから公開し、その後に一時的な進捗checkpointを削除します。

運用上のguardrailとして、大きな処理の前にdisk空き領域を予約し、各runにconfig/data/semantic
hashを保存します。criticalな温度状態では継続せず、checkpointして停止するthermal policyを使い
ます。metricsは追記型JSONLです。Rust TUIと`watch`はこの永続eventを読み取り専用で消費するため、
SSH再接続時にGPUメモリを競合させたり二重のtrainerを起動したりせず、monitorへ接続できます。

### 6. 再現性と連携を第一級のインターフェースにする

raw downloadにはsource、timestamp、size、SHA-256を残し、処理済みgraph dataは不変にします。
生成されるrun、log、checkpointは`runs/`以下へ分離します。container digest、lockした言語依存、
host probe、解決済みenvironment manifestにより、model artifactだけでなく実行環境も検査できます。
MCP serverもこの方針を踏襲し、stdio上で読み取り専用のローカル事実と検証済み検索を提供します。
言語生成とネットワーク公開APIは、呼び出し側のLLMが担当します。

## 必要な環境

検証済みの環境は、Jetson Linux R36.3（JetPack 6世代）、Docker Compose、NVIDIA
Container Runtimeを備えたJetson Orin Nanoです。フル実行の前に、約24 GiBの空き
領域を確保してください。コンテナイメージはJetson向けに固定されています。x86向け
CUDAイメージへ置き換えたり、CPUフォールバックを有効にしたりしないでください。

Wikidata5Mのデータとモデルは大きいため、GitではなくGitHub Releaseで配布します。
非公開リポジトリから取得する場合は、先にGitHub CLIで認証してください。

```bash
git clone https://github.com/hasegawa-ryo-edu-ea/neuroseek.git
cd neuroseek
gh release download initial-data-and-models \
  --repo hasegawa-ryo-edu-ea/neuroseek \
  --pattern 'neuroseek-*.tar.zst'
sha256sum -c data/RELEASE_SHA256SUMS
tar --zstd -xf neuroseek-wikidata5m-raw.tar.zst
tar --zstd -xf neuroseek-processed-data-and-models.tar.zst
```

展開すると、完全整列済みの`semantic_full`を含む`data/raw/wikidata5m/`と
`data/processed/`が復元されます。これらは`cache/`には置かないでください。

## インストール確認

```bash
./up.sh --doctor
PYTHONPATH=python pytest -q
cargo test --workspace -q
```

検証済み環境では、Pythonの54件とRustの12件のテストが成功しています。これは実装と
CUDA/データ契約の確認であり、公開ベンチマークを再現したことを意味しません。

## 学習を実行する

| コマンド | 用途 |
| --- | --- |
| `./up.sh --smoke` | 合成データによる短時間の機能・CUDAパス確認。科学的な結果ではありません。 |
| `./up.sh --trial` | 実データの決定的な部分グラフで行う時間制限付き実行。semanticの被覆は部分的です。 |
| `./up.sh` | フル学習を開始、または安全に再開します。学習器はバックグラウンドで動きます。 |
| `./up.sh --status` / `./logs.sh` | 学習器の状態確認 / ログの追跡。 |
| `./down.sh` | データとチェックポイントを残して、学習を正常停止します。 |

最初のフル実行では、semanticベクトルの準備が前景で動く場合があります。
`NEUROSEEK trainer detached:`が表示されるかモニタが開くまで、SSHを切断しないで
ください。その後は切断しても大丈夫です。再接続後に`./up.sh`を実行すると読み取り
専用モニタへ接続します。TUIが不要なら`./up.sh --no-tui`を使います。中断した準備は
原子的チェックポイントから再開します。

標準の50時間カリキュラムには`config/full.toml`を使います。親チェックポイントを
使うレイテンシー重視の追加学習には、`config/latency_optimization_6h.toml`を使えます。

```bash
mkdir -p runs/my-specialization
NEUROSEEK_CONFIG=/workspace/config/latency_optimization_6h.toml \
NEUROSEEK_RUN_DIR=/workspace/runs/my-specialization \
NEUROSEEK_RESUME= \
NEUROSEEK_PARENT_CHECKPOINT=/workspace/runs/<parent-run>/checkpoints/final.ckpt \
docker compose -f compose.yaml up -d trainer
```

Dockerの設定によっては、`docker compose`を`sudo -n docker compose`に置き換えて
ください。親runへ再開したり、グラフ・分割・semantic artifactが異なるrunを比較したり
しないでください。

## 最終モデルを検索に使う

検索ツールはCUDA必須・読み取り専用です。グラフ、チェックポイント、テレメトリは
変更しません。ただし学習とJetson GPUを共有するため、学習中とは時間を分けてください。
GUIはデモ・確認用であり、ローカルLLMとの連携にはMCPを使います。

```bash
./search
./search tui ja
./search web en
```

Web UIは`http://127.0.0.1:8787`だけで待ち受けます。別のPCから開く場合はSSHポート
フォワーディングを使ってください。ターミナルUIでは、`r`でタスク実行、`n`で次の
タスク、`1`〜`5`でページ切替、`l`で言語切替を行えます。`:run 3`、`:lang ja`、
`:quit`も使えます。`Model`と`Path`は方策の出力を、`Proof`は独立検証の結果を表示
します。`./watch`（英語）と`./watch ja`（日本語）は学習を制御せず、メトリクスだけを
表示します。

## 結果と制約

最終モデルと親モデルは、同じ256件の固定保持テスト、グラフ、semantic artifact、
CUDAバックエンドで評価しました。

| 指標 | 親モデル | 最終モデル |
| --- | ---: | ---: |
| 平均エンドツーエンド遅延 | 31.92 ms | 31.11 ms |
| p95エンドツーエンド遅延 | 57.10 ms | 53.02 ms |
| 1テスト当たり平均計算クレジット | 233.98 | 146.02 |
| 正答率 / 有効証明率 | 98.05% | 97.27% |

最終モデルは高速で計算クレジットも少ない一方、正答率は0.78ポイント（2件）下がり
ました。この値は単一Jetson・単一の固定テストにおける結果です。信頼区間、外部SOTA
との比較、自然言語QAの評価は含みません。[生データ]
(runs/latency-optimization-20260813T1255EDT/exports/benchmark_comparison.csv)と
[評価レポート](reports/latency-final-presentation/report.html)を参照してください。

## 詳細資料

- [アーキテクチャ](docs/ARCHITECTURE.ja.md) / [English](docs/ARCHITECTURE.md)
- [学習](docs/TRAINING.ja.md) / [English](docs/TRAINING.md)
- [運用手順](docs/RUNBOOK.ja.md) / [English](docs/RUNBOOK.md)
- [実験設計](docs/EXPERIMENT_DESIGN.ja.md) / [English](docs/EXPERIMENT_DESIGN.md)
- [再現性](docs/REPRODUCIBILITY.ja.md) / [English](docs/REPRODUCIBILITY.md)
- [グラフ形式](docs/GRAPH_FORMAT.ja.md) / [English](docs/GRAPH_FORMAT.md)
- [Semantic lane 契約](docs/SEMANTIC.ja.md) / [English](docs/SEMANTIC.md)
- [NEURO-ISA](docs/NEURO_ISA.ja.md) / [English](docs/NEURO_ISA.md)
- [トラブルシューティング](docs/TROUBLESHOOTING.ja.md) / [English](docs/TROUBLESHOOTING.md)
- [データとモデル asset](data/README.ja.md) / [English](data/README.md)
- [第三者データとライセンス](THIRD_PARTY.ja.md) / [English](THIRD_PARTY.md)
- [最終評価](reports/latency-final-presentation/FINAL_RESULT.ja.md) / [English](reports/latency-final-presentation/FINAL_RESULT.en.md)
