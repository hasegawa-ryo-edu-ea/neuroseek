# The host is Jetson Linux R36.3 with CUDA 12.2.  A newer generic framework
# image (CUDA 12.4) let PyTorch enumerate Orin but its custom PTX failed at
# runtime against the host driver.  NVIDIA's L4T ML R36.2 image is the matching
# JetPack 6 / CUDA 12.2 generation.  The exact arm64 manifest was inspected on
# this host; it is pinned so a later tag change cannot alter a 50-hour run.
ARG BASE_IMAGE=nvcr.io/nvidia/l4t-ml@sha256:0b71d2e7784c392080a4edf76d3ee81772fba6655bc3b4fc5390850c2909bda8
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace/python \
    CARGO_HOME=/opt/cargo \
    RUSTUP_HOME=/opt/rustup \
    PATH=/opt/cargo/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential=12.9ubuntu3 \
      ca-certificates=20260601~22.04.1 \
      clang=1:14.0-55~exp2 \
      cmake=3.22.1-1ubuntu1.22.04.2 \
      curl=7.81.0-1ubuntu1.25 \
      git=1:2.34.1-1ubuntu1.17 \
      ninja-build=1.10.1-1 \
      pkg-config=0.29.2-1ubuntu3 \
      python3-dev=3.10.6-1~22.04.1 \
      python3-pip=22.0.2+dfsg-1ubuntu0.7 \
      rsync=3.2.7-0ubuntu0.22.04.7 && \
    rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir tomli==2.0.1

# Rust is used by the VM, CLI, and TUI.  Pinning this compiler avoids an
# otherwise invisible `stable` upgrade changing native code generation.
RUN curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs | \
      sh -s -- -y --profile minimal --default-toolchain 1.82.0

WORKDIR /workspace

# Install the dashboard outside the bind mount.  The dashboard is a read-only
# JSONL reader, so attaching it can never control or terminate the trainer.
COPY Cargo.toml Cargo.lock ./
COPY rust ./rust
RUN cargo install --locked --path rust/tui --root /usr/local && \
    cargo install --locked --path rust/cli --root /usr/local && \
    rm -rf /workspace/target

# Compile CUDA for Orin's SM87 in the Jetson container.  This is intentionally
# a build failure if the selected image lacks a compatible CUDA developer toolchain.
COPY cuda ./cuda
RUN cmake -S cuda -B /opt/neuroseek/cuda-build -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /opt/neuroseek/cuda-build --parallel 2 && \
    install -Dm755 /opt/neuroseek/cuda-build/libneuroseek_cuda.so /opt/neuroseek/lib/libneuroseek_cuda.so && \
    install -Dm755 /opt/neuroseek/cuda-build/neuroseek_cuda_parity /usr/local/bin/neuroseek-cuda-parity && \
    install -Dm755 /opt/neuroseek/cuda-build/neuroseek_cuda_bench /usr/local/bin/neuroseek-cuda-bench

# Source is bind-mounted by compose so long-running checkpoints, data, and
# generated native artifacts are never stored solely in a container layer.
CMD ["bash", "-lc", "exec python3 -m neuroseek.training.trainer --config /workspace/config/full.toml --run-dir /workspace/runs/current --resume"]
