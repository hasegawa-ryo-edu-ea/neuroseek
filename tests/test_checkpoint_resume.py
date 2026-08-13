"""Checkpoint recovery tests use an explicitly named smoke run only.

They deliberately do not need Wikidata5M, so a failed resume contract is
caught before an expensive full-data run is permitted.
"""
from __future__ import annotations

import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "python"))

from neuroseek.training.checkpoint import load_latest, publish_latest, save_atomic
from neuroseek.training.trainer import _checkpoint, _write_crash_bundle
from neuroseek.models.policy import NavigatorPolicy
from neuroseek.telemetry.events import EventWriter


def _payload(step: int, elapsed: float) -> dict:
    return {
        "format": 1,
        "global_step": step,
        "phase": "graph_ppo",
        "best_metric": 2.5,
        "elapsed_seconds": elapsed,
        "model": {"weight": torch.tensor([float(step)])},
        "optimizer": {"state": {}, "param_groups": []},
    }


def test_corrupt_latest_falls_back_to_newest_valid_periodic(tmp_path: Path):
    checkpoints = tmp_path / "checkpoints"
    older = save_atomic(_payload(12, 3.0), checkpoints, "periodic-000000012.ckpt")
    newest = save_atomic(_payload(20, 5.0), checkpoints, "periodic-000000020.ckpt")
    publish_latest(newest, checkpoints)
    (checkpoints / "latest.ckpt").write_bytes(b"not a torch checkpoint")

    restored = load_latest(checkpoints, torch.device("cpu"))
    assert restored is not None
    assert restored["global_step"] == 20
    assert restored["elapsed_seconds"] == 5.0
    assert restored["_checkpoint_source"] == "periodic-000000020.ckpt"
    # If both newest copies are damaged, the older atomic generation survives.
    newest.write_bytes(b"corrupt too")
    restored = load_latest(checkpoints, torch.device("cpu"))
    assert restored is not None and restored["global_step"] == 12
    assert restored["_checkpoint_source"] == older.name


def test_checkpoint_preserves_elapsed_optimizer_and_rng_state(tmp_path: Path):
    model = NavigatorPolicy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    # Materialize optimizer state, then establish the exact state to restore.
    loss = sum(value.square().sum() for value in model.parameters())
    loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
    random.seed(91); np.random.seed(92); torch.manual_seed(93)
    expected_python = random.getstate()
    expected_numpy = np.random.get_state()
    expected_torch = torch.get_rng_state().clone()
    path = _checkpoint(model, optimizer, tmp_path, "behavior_cloning", 31, 0.75, {"run": {}}, 12.25)
    assert path.name == "periodic-000000031.ckpt"

    # Deliberately alter all streams before the resume load.
    random.random(); np.random.random(); torch.rand(3)
    restored = load_latest(tmp_path / "checkpoints", torch.device("cpu"))
    assert restored is not None
    assert restored["elapsed_seconds"] == pytest.approx(12.25)
    assert restored["global_step"] == 31
    random.setstate(restored["rng_python"])
    np.random.set_state(restored["rng_numpy"])
    torch.set_rng_state(restored["rng_torch"])
    assert random.getstate() == expected_python
    assert np.array_equal(np.random.get_state()[1], expected_numpy[1])
    assert torch.equal(torch.get_rng_state(), expected_torch)
    assert restored["optimizer"]["state"]  # not just model weights


def test_checkpoint_preserves_task_stream_state(tmp_path: Path):
    model = NavigatorPolicy(); optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    class Generator:
        def state_dict(self):
            return {"format": 1, "cursor": 19, "rng_state": "opaque-test-state"}
    _checkpoint(model, optimizer, tmp_path, "graph_ppo", 19, 0.1, {"run": {}}, 2.0,
                generator=Generator(), phase_state={"curriculum": "distractor"})
    restored = load_latest(tmp_path / "checkpoints", torch.device("cpu"))
    assert restored is not None
    assert restored["task_generator"]["cursor"] == 19
    assert restored["phase_state"]["curriculum"] == "distractor"


def test_checkpoint_provenance_rejects_cross_run_payload(tmp_path: Path):
    model = NavigatorPolicy(); optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    expected = {"mode": "trial", "run_id": "run-a", "graph_manifest_sha256": "graph-a"}
    _checkpoint(model, optimizer, tmp_path, "graph_ppo", 21, 0.1, {"run": {}}, 2.0,
                provenance=expected)
    assert load_latest(tmp_path / "checkpoints", torch.device("cpu"), expected_provenance=expected) is not None
    assert load_latest(tmp_path / "checkpoints", torch.device("cpu"),
                       expected_provenance={**expected, "run_id": "run-b"}) is None


def test_sigterm_writes_checkpoint_then_resume_continues(tmp_path: Path):
    """The real entrypoint checkpoints on SIGTERM; this is smoke-only by design."""
    run_dir = tmp_path / "run"
    config = tmp_path / "smoke.toml"
    config.write_text(
        "[run]\nmode = 'smoke'\nseed = 44\nbudget_seconds = 30\n"
        "[training]\nbatch_size = 2\nlr = 0.0003\n"
        "[checkpoint]\nseconds = 0.05\n",
        encoding="utf-8",
    )
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "python")}
    command = [sys.executable, "-m", "neuroseek.training.trainer", "--config", str(config), "--run-dir", str(run_dir)]
    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and not (run_dir / "metrics.jsonl").exists():
            time.sleep(0.05)
        assert (run_dir / "metrics.jsonl").exists(), "trainer did not reach its event loop"
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=30)
    finally:
        if process.poll() is None:
            process.kill(); process.communicate()
    assert process.returncode == 0, stderr or stdout
    first = load_latest(run_dir / "checkpoints", torch.device("cpu"))
    assert first is not None and first["global_step"] > 0
    first_step = int(first["global_step"])
    # A completed smoke budget should not be required to prove continuation:
    # resume should restore and perform at least one more minibatch.
    resumed_config = tmp_path / "resume.toml"
    # Keep this bounded even on a slow development host while leaving room for
    # a resumed minibatch after the restored elapsed counter.
    resume_budget = float(first["elapsed_seconds"]) + 2.0
    resumed_config.write_text(config.read_text().replace("budget_seconds = 30", f"budget_seconds = {resume_budget}"), encoding="utf-8")
    resumed = subprocess.run(command[:4] + [str(resumed_config)] + command[5:] + ["--resume"], cwd=ROOT, env=environment,
                             text=True, capture_output=True, timeout=45)
    assert resumed.returncode == 0, resumed.stderr
    second = load_latest(run_dir / "checkpoints", torch.device("cpu"))
    assert second is not None and int(second["global_step"]) > first_step
    events = [json.loads(row) for row in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert any(event["category"] == "TrainingEvent" and float(event["elapsed_seconds"]) >= 0.0 for event in events)
    assert any(event["category"] == "RecoveredFromCheckpoint" and event["global_step"] == first_step for event in events)


def test_crash_bundle_contains_observed_context(tmp_path: Path):
    events = EventWriter(tmp_path)
    events.emit("TrainingEvent", global_step=7, reward=1.25)
    try:
        raise RuntimeError("intentional test failure")
    except RuntimeError as error:
        path = _write_crash_bundle(tmp_path, events, step=7, phase="test", config={"run": {"mode": "smoke"}}, error=error)
    report = json.loads(path.read_text())
    assert report["global_step"] == 7
    assert report["recent_events"][-1]["category"] == "TrainingEvent"
    assert "intentional test failure" in report["traceback"]
    assert "free_bytes" in report["disk"]
