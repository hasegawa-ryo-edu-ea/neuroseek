# Web UI デザイン QA

英語版: [design-qa.md](design-qa.md)

## 確認事項

- [P1] この実行環境では、ブラウザで描画した画面の視覚比較を行えません。
  場所: WebUI 全体。
  根拠: 元の視覚基準は `/home/nvidia/.codex/generated_images/019ffc16-82e8-7022-9ba9-fa51cb2b8f52/exec-32e36ab6-e80c-4c29-bf26-45011fb42dfd.png` です。このセッションには `http://127.0.0.1:8787` 用のブラウザまたはスクリーンショット取得ツールがありません。
  影響: タイポグラフィ、レスポンシブレイアウト、canvas animation、操作感を元画像と視覚的に比較できません。
  対応: Jetson desktop または SSH port forwarding 上で `./search web ja` を開き、1472×1027 の desktop 状態をキャプチャしてから、元画像に対する視覚 QA を再実行します。

## 未解決の点

- 実装では、元画像にあった未検証の GPU/cache/Wikipedia に関する主張を意図的に削除しました。WebUI は実際のローカル graph edge と不変 checkpoint の実行結果を表示します。

## 実装チェックリスト

- [x] 元の視覚ターゲットを特定。
- [x] 日英対応の操作、ナビゲーション、クエリ入力、relation filter、policy action を実装。
- [x] アニメーションする graph canvas は、作りものの telemetry ではなく返されたローカル graph edge を表現。
- [x] CPU/read-only と loopback-only のサービス境界を適用。
- [x] 実際の Compose service を通して HTTP API と主要操作を機能確認。
- [ ] ブラウザで描画結果をキャプチャし、視覚比較する。

## 次の仕上げ

- ブラウザキャプチャが得られたら、発表ディスプレイの native resolution で行密度と node label の読みやすさを確認します。

視覚基準のパス: `/home/nvidia/.codex/generated_images/019ffc16-82e8-7022-9ba9-fa51cb2b8f52/exec-32e36ab6-e80c-4c29-bf26-45011fb42dfd.png` と `/home/nvidia/jetson-inference/Final_Project/tmp/スクリーンショット 2026-08-14 072433.png`。

実装スクリーンショット: 利用不可。この環境にはブラウザスクリーンショットツールがありません。

Viewport: 未取得。

状態: 初期 WebUI、API 確認済み query（`日本` / `Q17`）、relation filter（`P36`）。

全体比較の根拠: キャプチャ前のため保留。

注目領域比較の根拠: キャプチャ前のため保留。

機能確認済みの主要操作: launcher output、loopback health endpoint、日本語と英語の entity resolution、local graph view、P36 filter、保持済み policy の実行。

console error: JavaScript 構文は `node --check` で確認済みです。ブラウザ console は使えません。

比較履歴: 2557×1377 の提供画像は、既存の描画基準として確認しました。第 1 回では、見える drag/zoom/reset 操作、node selection、段階的な edge/node の表示、evidence/candidate/result の時間差表示を加えました。第 2 回では、pointer 開始位置と scene coordinate の取り違えを修正し、pan 範囲を制限し、graph viewport を広げ、最大 18 本の実ローカル edge を放射状に描き、node/edge/relation/snapshot/view の統計を表示しました。第 3 回では、選択 branch のローカル graph に基づく遅延二段目展開、別の緑色 graph layer、明示的な `SECOND HOP · LOCAL ONLY` legend を追加しました。第 4 回では、不変 policy を CPU で実行する `POLICY PATH` 操作を加え、実際の proof path と代替 frontier word を表示するようにしました。proof path、relation、frontier candidate の日本語ラベルで機能確認しています。ピクセル単位の比較には、現在のブラウザキャプチャがなお必要です。

最終結果: ブラウザキャプチャがないため保留。
