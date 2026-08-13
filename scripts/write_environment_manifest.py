#!/usr/bin/env python3
"""Record observed, resolved host/toolchain facts; never invent versions."""
from __future__ import annotations
import json, platform, shutil, subprocess, time
from pathlib import Path

def command(*args: str) -> str | None:
    if not shutil.which(args[0]): return None
    result=subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return result.stdout.strip() if result.returncode == 0 else None

Path("artifacts").mkdir(exist_ok=True)
payload={
    "captured_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
    "machine":platform.machine(),
    "python":platform.python_version(),
    "docker":command("docker","--version"),
    "cuda_nvcc":command("nvcc","--version"),
    "rust":command("rustc","--version"),
    "container":{
        "base":"nvcr.io/nvidia/l4t-ml@sha256:0b71d2e7784c392080a4edf76d3ee81772fba6655bc3b4fc5390850c2909bda8",
        "tag":"neuroseek:jetson-r36.3-l4tml-r36.2",
        "rust_toolchain":"1.82.0",
        "python_runtime":{"tomli":"2.0.1","numpy":"1.26.1","torch":"2.1.0","torch_cuda":"12.2"},
        "apt_direct_packages":{
            "build-essential":"12.9ubuntu3","ca-certificates":"20260601~22.04.1","clang":"1:14.0-55~exp2",
            "cmake":"3.22.1-1ubuntu1.22.04.2","curl":"7.81.0-1ubuntu1.25","git":"1:2.34.1-1ubuntu1.17",
            "ninja-build":"1.10.1-1","pkg-config":"0.29.2-1ubuntu3","python3-dev":"3.10.6-1~22.04.1",
            "python3-pip":"22.0.2+dfsg-1ubuntu0.7","rsync":"3.2.7-0ubuntu0.22.04.7"
        }
    }
}
Path("artifacts/environment_manifest.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
