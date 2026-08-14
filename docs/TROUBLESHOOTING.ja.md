# トラブルシューティング

英語版: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

`./up.sh --doctor` は学習を開始せずに、Docker／NVIDIA runtime の不足を報告します。このホストでは、現在のユーザーが Docker group にいない場合に passwordless `sudo docker` を意図的に使います。ランチャーはその状態を検出します。

CUDA の事前検査に失敗したら、`artifacts/host_probe.txt` を確認し、`sudo docker compose -f compose.yaml run --rm trainer python3 -c 'import torch; print(torch.cuda.is_available())'` を実行してください。回避策として JetPack を変更したり x86 CUDA コンテナを入れたりしないでください。

run が停止した場合は `./status.sh`、`runs/<run-id>/crash_reports/` を確認してから `./up.sh` を再実行します。有効な `latest.ckpt` は自動的に再開されます。最新 checkpoint が破損していれば、ローダーは以前の周期 checkpoint に戻ります。

コンパイラは、明示的な `--force` なしに既存の処理済み出力を拒否します。学習中に不変グラフデータを書き換えることはありません。

## CUDA は見えるが custom CUDA parity に失敗する

`neuroseek-cuda-parity` が未対応 PTX または toolchain を報告した場合、学習を開始しないでください。framework イメージは Jetson GPU を列挙できても、ホスト driver が対応しない新しい CUDA userspace を含むことがあります。NEUROSEEK は R36.3/CUDA 12.2 ホスト向けに L4T ML イメージを R36.2/CUDA 12.2 世代へ固定し、`./up.sh` は本番学習器を作る前に native parity を実行します。

## Docker bridge が iptables raw-table エラーを報告する

このホストには、最近の Docker isolation が必要とする Docker bridge raw-table rule がない場合があります。NEUROSEEK の build/trainer/TUI コンテナは `network_mode: host` を使います。ネットワークリスナーを開かず、Docker daemon、kernel、Jetson network 設定を変えずにこの問題を避けます。`./up.sh --doctor` で確認してください。
