# Semantic レーン

英語版: [SEMANTIC.md](SEMANTIC.md)

NEUROSEEK には、記号的な CSR グラフレーンと semantic entity-vector レーンがあります。semantic の結果は候補ジャンプにすぎません。後段のグラフ `VERIFY` が事実の証明を担います。

## artifact の契約

不変の semantic ディレクトリには `entity_ids.u32`、行優先の `embeddings.f16`、`semantic_manifest.json` を含めます。manifest はグラフ entity 数、次元、正規化、ソース、ハッシュ、全コンパクト graph entity ID を覆うかどうかを記録します。`AlignedEmbeddings` は既定で、不完全、破損、サイズ不正、順序不正なデータを拒否します。

事前学習 embedding は、明示的なコンパクト entity-ID 対応を使って変換します。完全 artifact を受け入れるのは、ID が正確に `0..graph_entity_count-1` の場合だけです。ベクトル数が一致するだけでは整列の証拠になりません。

## バックエンドの選択

このリポジトリには現在 cuVS/CAGRA バックエンドはありません。`CudaExactAnnBackend` が本番用フォールバックです。検証済みのカスタム CUDA 内積 C ABI を境界付きバッチで呼び、決定的な host top-k merge を行います。報告名は `cuda_exact` で、統計には host 側の merge を明示します。CUDA を読み込めない場合は明示的なエラーであり、NumPy へフォールバックしません。

`NumpyExactBackend` はテストとオフライン開発専用です。バックエンド名に `test_only` を含め、GPU の証拠と取り違えられないようにします。

## 境界付き TransE フォールバック

`train_bounded_transe(graph, output, TransEConfig(...))` は、mmap CSR entity の決定的な誘導部分集合でコンパクトな TransE 形式モデルを学習します。Jetson の 8 GB を守るため、疎 PyTorch embedding、境界付き step、境界付き entity 数を使います。生成 artifact は意図的に partial と記録され、呼び出し側は `allow_partial=True` を渡す必要があります。trial の semantic jump には使えますが、全 Wikidata5M entity を embedding したという主張には使えません。source manifest はシード、有効な部分集合 edge を含んだ実 step、最終観測 loss、device を記録します。

full モードでは、すべてのコンパクト entity ID に対する sparse CUDA TransE を使います。最初のバッチの前に step-zero 復旧チェックポイントを書き、定期的な進捗チェックポイントをアトミックに公開します。実 Jetson コンテナプローブでは、4,594,149 × 64 FP32 entity table の GPU 確保と CPU export に成功しました。測定した GPU 確保量は 1,176,502,272 bytes、CPU snapshot は 1,176,102,144 bytes です。境界付きプローブ記録 `artifacts/semantic_checkpoint_memory_probe.json` は確保／export テストであり、full 学習完了の主張ではありません。

`artifacts/full_semantic_resume_probe.json` には別の CUDA 障害注入の受け入れテストを記録します。テスト専用 CLI の明示的な中断で step-zero チェックポイントが書かれ、通常の次回 full build がそれを読み、complete store を公開し、公開後にだけ進捗チェックポイントを削除しました。

`artifacts/full_semantic_sigterm_probe.json` は運用上の signal 経路を記録します。SIGTERM は安全停止を要求し、次の full-TransE ループ境界がアトミックな進捗チェックポイントを書き、プロセスは通常の signal 由来 status で終了します。続く通常の full 起動が再開します。

グラフコンパイル後の再現可能なコマンドライン入口は次です。

```bash
python3 scripts/build_semantic.py --graph data/processed --output data/processed/semantic_bounded
```

このコマンドは明示的な既定値（64 dimensions、100,000 entities、1,000 steps）で境界付き CUDA TransE フォールバックを実行し、処理済みグラフ manifest hash を embedding source metadata に書き、再利用時に hash を検証し、暗黙の置換を拒否します。この派生キャッシュを明示的に再生成する場合だけ `--replace` を使います。

full-run semantic レーンを有効にする前に、完全に整列した事前学習 artifact を用意するか、明示的に完全被覆を生成・検証してください。境界付きフォールバックを完全な Wikidata5M semantic coverage として示してはいけません。

`./up.sh --trial` は `semantic_bounded` を透過的に build/reuse し、partial coverage を run manifest に記録します。一方、`./up.sh` はそうしません。`data/processed/semantic_full` を要求し、コンパクト ID が厳密に `0..graph_entity_count-1` でない manifest をすべて拒否します。これにより、完全 semantic asset の欠落を研究課題を黙って変えるのではなく、見えるリリースゲートにします。

最初の full build は sparse-SGD TransE を使い、10,000 optimizer step ごとに `data/processed/.semantic_full.training.ckpt` へアトミックな trainable-table チェックポイントを書きます。繰り返した `./up.sh` は同じ graph/configuration チェックポイントを再開します。互換性がない、または破損したチェックポイントは混在 artifact を作らず明示的に失敗します。チェックポイントは完全な FP16 artifact と manifest がアトミックに公開された後だけ削除します。
