# NEUROSEEK（日本語）

**Jetson Orin Nano向けの、証明を第一にするGPUネイティブ知識グラフ・ナビゲータです。**

NEUROSEEKは本プロジェクトの低レイテンシー化した完成形です。学習済みコントローラがエッジGPU上で大規模グラフを探索し、独立した検証器が受理したグラフ証拠とともに答えを返します。再現可能な研究システムであり、根拠の範囲を越えないローカル探索の実証です。ホスト型APIや汎用チャットボットではありません。

英語版: [README.md](README.md)

## なぜこのプロジェクトを始めたのか

知識システムは、速い一方で監査しにくい意味検索と、確認可能だが高コストになりやすい記号的グラフ探索のどちらかを選びがちです。NEUROSEEKは、8 GB級エッジデバイス上でこのトレードオフを明示的に扱うために始めました。学習とCUDAで「どこを探索するか」を決めつつ、最終的な主張は不変のグラフ証拠に結び付けます。

目標は、もっともらしいエンティティを返すことより厳しいものです。結果には再生・独立検証できる経路を含め、その到達コストをレイテンシー、命令数、グラフ探索量として可視化します。

## 最終低レイテンシー結果

最終モデルと親モデルは、同じグラフ、semantic provenance、CUDA exact backendを用い、同一の固定保持テスト256件で評価しました。

| 指標 | 親モデル | 最終モデル | 変化 |
| --- | ---: | ---: | ---: |
| 平均エンドツーエンド遅延 | 31.92 ms | **31.11 ms** | **-2.56%** |
| p95エンドツーエンド遅延 | 57.10 ms | **53.02 ms** | **-7.14%** |
| 1テスト当たり平均計算クレジット | 233.98 | **146.02** | **-37.59%** |
| 正答率 / 有効証明率 | **98.05%** | 97.27% | -0.78ポイント（2件） |

これは、すべての品質軸が上がったという主張ではなく、意図的かつ実測済みの効率トレードオフです。回帰した2件はpath種別に集中し、distractor、intersection、semantic-hybrid、robustnessでは正答率を維持しました。4.66 msの手書きhybridベースラインとは直接比較できません。後者には関係プログラムが与えられますが、学習モデルは演算子列を自ら選択するためです。

生データ: [benchmark comparison](runs/latency-optimization-20260813T1255EDT/exports/benchmark_comparison.csv)。比較と限界は[最終評価レポート](reports/latency-final-presentation/report.html)に記録しています。

## どこが優れているか

| 特長 | 意味 |
| --- | --- |
| 独立した証明検証 | Rustは返されたグラフ証拠が有効なときだけ結果を受理します。意味的な類似だけを証明として扱いません。 |
| 根拠を限定したハイブリッド探索 | 埋め込みとCUDA探索が候補を提案し、記号的レーンが答えの正しさを確立します。 |
| エッジ優先・コスト認識制御 | 小型のクエリ条件付き方策が予算付きNEURO-ISAプログラムを学び、命令数とデバイスコストを下げるよう特化しました。 |
| GPU正しさを起動条件にする | カスタムCUDAとホストのパリティを要求し、CUDAが見えるだけでは開始しません。 |
| 再現可能な大規模グラフ経路 | Wikidata5Mを不変のメモリマップCSRへ変換し、分割、ハッシュ、設定、チェックポイント、追記型メトリクスを残します。 |
| 長時間・SSHでの安全性 | 原子的復旧、24 GiB保護、臨界温度時の保存して停止、読み取り専用再接続モニタでrunを守ります。 |

## アーキテクチャ

~~~
Wikidata5M triples → 不変の前処理 → メモリマップ前方/逆方向CSRグラフ
                                           ↑
 埋め込み → CUDA exact score/top-k → 候補 ↑
                                           ↑
学習済みクエリ条件付き方策 → Rust NEURO-ISA VM → Rust証明検証器
                                                    ↓
                                        受理された答え + 証拠
~~~

RustがVMの意味論、計数、証明検証を担当します。CUDAはホストとのパリティテスト付きバルクプリミティブを実装し、Pythonはナビゲータ、PPO/behavior cloning学習、カリキュラム、チェックポイント、テレメトリ、評価を担当します。意味検索はジャンプ候補を提案できますが、検証を迂回できません。

## 自分のJetsonで再現する

検証済みの対象は、Jetson Linux R36.3 / JetPack 6世代、Docker Compose、NVIDIA Container Runtimeを備えたJetson Orin Nanoです。フル実行の前に約24 GiBの空き領域を確保してください。ビルドは互換性のあるNVIDIA L4T MLイメージを固定します。x86向けCUDAイメージへ置き換えたり、CPUフォールバックを許可したりしないでください。

大きなデータ・モデル資産はGitHub Releaseで配布します。非公開リポジトリでは先にGitHub CLIで認証してください。

~~~
git clone https://github.com/hasegawa-ryo-edu-ea/neuroseek.git
cd neuroseek
gh release download initial-data-and-models \
  --repo hasegawa-ryo-edu-ea/neuroseek \
  --pattern 'neuroseek-*.tar.zst'
sha256sum -c data/RELEASE_SHA256SUMS
tar --zstd -xf neuroseek-wikidata5m-raw.tar.zst
tar --zstd -xf neuroseek-processed-data-and-models.tar.zst
./up.sh --doctor
PYTHONPATH=python pytest -q
cargo test --workspace -q
~~~

Releaseにより完全整列済みsemantic_fullを含むdata/raw/wikidata5m/とdata/processed/が復元されます。cache/には置かないでください。検証済み環境ではPython 54件、Rust 12件が成功しています。これは実装とCUDA/データ契約の準備を示しますが、最終ベンチマークの再現そのものではありません。

| コマンド | 用途 |
| --- | --- |
| ./up.sh --smoke | 合成データでの時間制限付き機能・CUDAパス検証。科学的結果ではありません。 |
| ./up.sh --trial | 実データ部分グラフによる時間制限付き決定的実行。部分semantic被覆を明示的に許可します。 |
| ./up.sh | 完全semantic artifactを使った、分離フル学習または安全な再開。 |
| ./up.sh --status / ./logs.sh | 分離学習器の状態確認、またはログ追跡。 |
| ./down.sh | データセットとチェックポイントを残して正常停止。 |

最初のフル実行ではsemanticベクトルの準備が前景で走ることがあります。NEUROSEEK trainer detached:が出るかモニタが開くまでSSHを切断しないでください。その後は./up.shで読み取り専用モニタへ接続し、TUI不要なら./up.sh --no-tuiを使います。中断したsemantic準備は原子的チェックポイントから再開します。

## 追加学習・特化学習

標準の50時間カリキュラムにはconfig/full.tomlを使います。フェーズ予算を記録し、semantic準備には別上限を設け、5分ごとにチェックポイントを保存し、臨界温度では進捗を保存して停止します。

レイテンシー重視の派生モデルでは、config/latency_optimization_6h.tomlをコピーし、親と同一のグラフ・分割・semantic artifactを保持して、変更したすべての設定を記録してください。提供設定は親チェックポイントから開始し、有効証明の報酬を優勢に保ちながら、小さなレイテンシー・命令数ペナルティを適用します。

~~~
mkdir -p runs/my-specialization
NEUROSEEK_CONFIG=/workspace/config/latency_optimization_6h.toml \
NEUROSEEK_RUN_DIR=/workspace/runs/my-specialization \
NEUROSEEK_RESUME= \
NEUROSEEK_PARENT_CHECKPOINT=/workspace/runs/<parent-run>/checkpoints/final.ckpt \
docker compose -f compose.yaml up -d trainer
~~~

必要ならdocker composeをsudo -n docker composeに置き換えます。親runへ再開したり、semantic_boundedをフル比較に混ぜたり、マニフェストと固定保持テスト評価なしに設定変更を報告したりしないでください。train/validation/testのseedを分け、正答率、証明有効性、命令数、グラフ探索量、レイテンシー、デバイス条件を一緒に評価します。[Training](docs/TRAINING.md)、[実験設計](docs/EXPERIMENT_DESIGN.md)、[再現性](docs/REPRODUCIBILITY.md)を参照してください。

## 最終モデルを使う

発表・クエリツールはCUDA必須かつ読み取り専用です。方策と常駐CSR展開をGPUで実行し、CPUフォールバックを拒否します。データ、チェックポイント、テレメトリは書き換えませんが、Jetson GPUを共有するため学習実行中とは時間を分けてください。

~~~
./search
./search tui ja
./search web en
~~~

Web UIはhttp://127.0.0.1:8787です。jaまたはenを使えます。ターミナルでは、rで不変の検証タスクを実行し、nで次のタスク、1–5でページ切替、lで言語切替を行えます。:run 3、:lang ja、:quitも利用できます。ModelとPathは方策が実行した実際のプログラムと候補を表示し、Proofは返却された証拠を独立に検証します。参照解答は方策に与えません。

WordsはJapanや日本を解決し、ローカルで探索可能なグラフ事実だけを表示します。公開Wikidataによる名前解決は外部コンテキストとして表示され、ローカル証拠には使いません。Q/P IDはオフラインでも使えます。学習中の表示には./watch（英語）または./watch ja（日本語）を使います。metrics.jsonlを読むだけです。

## 範囲と限界

報告したレイテンシーは単一Jetson・固定テストでの評価値です。反復実行の信頼区間、外部SOTA比較、自然言語から関係プログラムを推定する評価は含みません。オンラインのレイテンシーコストモデルは検証誤差が大きかったため最終報酬から外し、最終結果にはより安全な命令数ペナルティを使っています。自然言語質問の解釈とローカルLLMによる回答生成は、次の統合段階であり、ここでの主張ではありません。

## 詳細資料

- [Architecture](docs/ARCHITECTURE.md)
- [グラフ形式](docs/GRAPH_FORMAT.md)
- [Semantic lane契約](docs/SEMANTIC.md)
- [運用Runbook](docs/RUNBOOK.md)
- [第三者データとライセンス](THIRD_PARTY.md)
