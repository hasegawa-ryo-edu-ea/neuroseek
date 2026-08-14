import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "python"))

from neuroseek.cost_model import CostModelError, load_records, train_cost_model
from neuroseek.cost_model.model import CostModel, load_model
from neuroseek.training.trainer import _latency_model_status


def _measured_records(path: Path) -> None:
    # Explicit test fixture, not a production benchmark.  Its shape follows
    # cuda_bench output plus VM operation records with observed latency.
    events = []
    for rows in (128, 256, 512, 1024, 2048, 4096):
        events.append({"operation": "exact_scores", "rows": rows, "dims": 64,
                       "mean_ms": 0.05 + rows * 0.001})
    for edges in (10, 20, 40, 80, 160, 320):
        events.append({"operation": "EXPAND_REL", "frontier_size": 4, "edge_count": edges,
                       "latency_ms": 0.1 + edges * 0.002})
    path.write_text("\n".join(json.dumps(row) for row in events) + "\n")


def test_train_save_load_and_predict_measured_jsonl(tmp_path):
    source = tmp_path / "hardware.jsonl"; _measured_records(source)
    records = load_records([source])
    model = train_cost_model(records, ridge=1e-4)
    assert model.training_records + model.validation_records == len(records)
    assert model.metrics["mae_ms"] >= 0.0
    destination = tmp_path / "model.json"
    result = subprocess.run([sys.executable, str(ROOT / "scripts/bench_cost_model.py"), "--input", str(source),
                             "--output", str(destination)], text=True, capture_output=True, check=True)
    assert json.loads(result.stdout)["records"] == 12
    loaded = load_model(destination)
    assert loaded.known_operation("exact_scores")
    assert loaded.predict("exact_scores", rows=4096, dims=64) >= 0.0


def test_rejects_unobserved_or_invalid_latency(tmp_path):
    source = tmp_path / "bad.jsonl"
    source.write_text('{"operation":"ANN","rows":8}\n')
    with pytest.raises(CostModelError, match="observed latency"):
        load_records([source])
    source.write_text('{"operation":"ANN","latency_ms":-1}\n')
    with pytest.raises(CostModelError, match="non-negative"):
        load_records([source])


def test_skip_invalid_only_accepts_real_latency_rows(tmp_path):
    source = tmp_path / "mixed.jsonl"
    source.write_text('{"category":"TrainingEvent","reward":1}\n{"operation":"ANN","latency_ms":1.5,"ann_k":8}\n{"operation":"ANN","latency_ms":2.0,"ann_k":16}\n{"operation":"ANN","latency_ms":2.5,"ann_k":32}\n')
    assert len(load_records([source], strict=False)) == 3


def test_latency_reward_rejects_nonmonotonic_or_unvalidated_surrogate():
    # A negative candidate-count coefficient would falsely reward larger graph
    # expansions as lower latency.  It must never reach the PPO reward path.
    operations = ("graph_expand",)
    weights = [1.0, 0.0] + [0.0] * 12
    weights[3] = -0.1  # candidate_count is the second numeric feature.
    unsafe = CostModel(operations, tuple(weights), 1e-3, 20, 5, {"mape_percent": 10.0})
    assert _latency_model_status(unsafe) == "rejected_nonmonotonic_candidate_cost"
    unvalidated = CostModel(operations, tuple([1.0, 0.0] + [0.0] * 12), 1e-3, 20, 5, {"mape_percent": 80.0})
    assert _latency_model_status(unvalidated) == "rejected_validation_error"
