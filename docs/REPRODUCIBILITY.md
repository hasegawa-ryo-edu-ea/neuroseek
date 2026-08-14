# Reproducibility

日本語版: [REPRODUCIBILITY.ja.md](REPRODUCIBILITY.ja.md)

The host probe and resolved dependency manifest live in `artifacts/`. Dataset downloads are recorded with source URL, timestamp, byte size, and SHA-256. Processing is immutable under `data/processed/`; runs, logs, and checkpoints live under `runs/`.

Every run captures its configuration, device selection, dataset manifest, and checkpoint state. The container is pinned to a Jetson/L4T-compatible image selected after probing this host; it must pass a CUDA smoke command before a training service starts.

The build image is pinned by digest in `Dockerfile`.  Direct APT build dependencies
are pinned to the versions actually installed in the validated image, Python-only
dependencies are pinned, and Rust application dependencies are locked in
`Cargo.lock`.  `artifacts/environment_manifest.json` records this resolved set
and is regenerated with `python3 scripts/write_environment_manifest.py` after a
host/toolchain change.  Rebuilding later can still fail if the configured APT
mirror no longer retains one of those exact packages; that is intentional rather
than silently substituting a newer compiler or native library.
