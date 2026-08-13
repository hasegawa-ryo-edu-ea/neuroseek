"""Crash-safe checkpoint lifecycle; never replace the only usable checkpoint in place."""
from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

import torch


REQUIRED_FIELDS = frozenset({"format", "global_step", "phase", "best_metric", "elapsed_seconds", "model", "optimizer"})


class CheckpointError(RuntimeError):
    """Raised when a requested resume has checkpoint files but none is valid."""


def save_atomic(state: dict[str, Any], checkpoints: Path, name: str) -> Path:
    checkpoints.mkdir(parents=True, exist_ok=True)
    target = checkpoints / name
    temporary = checkpoints / f".{name}.{os.getpid()}.{time.time_ns()}.tmp"
    try:
        torch.save(state, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(checkpoints, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return target
    finally:
        temporary.unlink(missing_ok=True)


def publish_latest(source: Path, checkpoints: Path) -> Path:
    target = checkpoints / "latest.ckpt"
    temporary = checkpoints / ".latest.ckpt.tmp"
    shutil.copy2(source, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    directory_fd = os.open(checkpoints, os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return target


def prune_periodic(checkpoints: Path, keep: int) -> list[Path]:
    """Prune superseded periodic checkpoints only after durable publication."""
    if keep <= 0:
        raise ValueError("checkpoint retention must keep at least one periodic generation")
    periodic = sorted(checkpoints.glob("periodic-*.ckpt"))
    removed: list[Path] = []
    for old in periodic[:-keep]:
        old.unlink()
        removed.append(old)
    if removed:
        directory_fd = os.open(checkpoints, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    return removed


def _validate_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload is not a mapping")
    missing = REQUIRED_FIELDS.difference(payload)
    if missing:
        raise ValueError(f"checkpoint is missing fields: {sorted(missing)}")
    if int(payload["format"]) != 1:
        raise ValueError(f"unsupported checkpoint format: {payload['format']}")
    if int(payload["global_step"]) < 0 or float(payload["elapsed_seconds"]) < 0:
        raise ValueError("checkpoint has invalid counters")
    if not isinstance(payload["model"], dict) or not isinstance(payload["optimizer"], dict):
        raise ValueError("checkpoint has invalid model or optimizer state")
    return payload


def load_latest(checkpoints: Path, device: torch.device, *, expected_provenance: dict[str, str] | None = None) -> dict[str, Any] | None:
    """Load newest valid checkpoint, falling back from corrupt publications.

    `latest.ckpt` is a convenience copy, never the sole recovery point.  Each
    periodic generation is independently validated before it can be resumed.
    The chosen source is included only as private metadata for audit events.
    """
    _ = device  # API compatibility; checkpoints are deliberately staged on CPU.
    candidates = [checkpoints / "latest.ckpt"] + sorted(checkpoints.glob("periodic-*.ckpt"), reverse=True)
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            # RNG snapshots are CPU ByteTensors.  Mapping an entire checkpoint
            # to CUDA corrupts that contract and also creates an avoidable GPU
            # allocation spike during recovery.  Module/optimizer loading moves
            # parameter state to its owning device afterwards.
            payload = _validate_payload(torch.load(candidate, map_location="cpu"))
            if expected_provenance is not None and payload.get("provenance") != expected_provenance:
                raise ValueError("checkpoint provenance differs from current immutable run inputs")
            payload["_checkpoint_source"] = candidate.name
            return payload
        except Exception:
            continue
    return None


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """Load one explicit, validated checkpoint for an auditable derived run."""
    _ = device
    if not path.is_file():
        raise CheckpointError(f"explicit checkpoint is absent: {path}")
    payload = _validate_payload(torch.load(path, map_location="cpu"))
    payload["_checkpoint_source"] = str(path)
    return payload


def has_checkpoint_candidates(checkpoints: Path) -> bool:
    """Whether a resume was requested for an existing checkpoint generation."""
    return any(checkpoints.glob("*.ckpt"))
