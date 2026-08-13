# Troubleshooting

`./up.sh --doctor` reports missing Docker/NVIDIA runtime prerequisites without starting training. This host deliberately uses passwordless `sudo docker` when the current user is not in the Docker group; the launcher detects that case.

If CUDA preflight fails, inspect `artifacts/host_probe.txt` and run `sudo docker compose -f compose.yaml run --rm trainer python3 -c 'import torch; print(torch.cuda.is_available())'`. Do not change JetPack or install an x86 CUDA container to work around it.

If a run stops, use `./status.sh`, inspect `runs/<run-id>/crash_reports/`, and rerun `./up.sh`. A valid `latest.ckpt` resumes automatically; checkpoint loader falls back to a prior periodic checkpoint when the latest is corrupt.

The compiler refuses an existing processed output unless invoked with its explicit `--force` flag. It never rewrites immutable graph data during training.

## CUDA is visible but custom CUDA parity fails

Do not start training if `neuroseek-cuda-parity` reports an unsupported PTX or
toolchain. A framework image can enumerate the Jetson GPU while still carrying
a CUDA userspace newer than the host driver supports. NEUROSEEK pins its L4T
ML image to the R36.2/CUDA 12.2 generation for the R36.3/CUDA 12.2 host, and
`./up.sh` runs native parity before creating a production trainer.

## Docker bridge reports an iptables raw-table error

This host may lack the Docker bridge raw-table rule required by recent Docker
isolation. NEUROSEEK uses `network_mode: host` for its build/trainer/TUI
containers; it opens no network listener, and this avoids changing the Docker
daemon, kernel, or Jetson network configuration. Verify with `./up.sh --doctor`.
