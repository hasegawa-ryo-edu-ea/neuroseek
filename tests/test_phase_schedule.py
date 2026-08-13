from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "python"))

from neuroseek.training.trainer import _minimum_free_disk_bytes, _phase_for_elapsed, _validate_phase_schedule


def test_wall_clock_schedule_uses_ordered_boundaries() -> None:
    config = {"phase": [{"name": "bc", "seconds": 2}, {"name": "ppo", "seconds": 3}]}
    assert _phase_for_elapsed(config, 0.0, 5) == "bc"
    assert _phase_for_elapsed(config, 1.99, 5) == "bc"
    assert _phase_for_elapsed(config, 2.0, 5) == "ppo"
    assert _phase_for_elapsed(config, 999.0, 5) == "ppo"
    _validate_phase_schedule(config, "full", 5)


def test_full_schedule_rejects_missing_or_mismatched_budget() -> None:
    with pytest.raises(ValueError, match="requires"):
        _validate_phase_schedule({}, "full", 5)
    with pytest.raises(ValueError, match="does not equal"):
        _validate_phase_schedule({"phase": [{"name": "bc", "seconds": 1}]}, "full", 5)


def test_full_runtime_disk_reserve_is_explicit_and_smoke_is_unconstrained() -> None:
    assert _minimum_free_disk_bytes({}, "full") == 24 * 1024**3
    assert _minimum_free_disk_bytes({}, "smoke") == 0
    assert _minimum_free_disk_bytes({"storage": {"minimum_free_gib": 3.5}}, "full") == int(3.5 * 1024**3)
    with pytest.raises(ValueError, match="non-negative"):
        _minimum_free_disk_bytes({"storage": {"minimum_free_gib": -1}}, "full")
