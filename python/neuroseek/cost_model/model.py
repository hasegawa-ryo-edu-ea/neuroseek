"""Regression and durable artifact format for NEUROSEEK hardware cost models."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


FORMAT_VERSION = 1
NUMERIC_FEATURES = (
    "frontier_size", "candidate_count", "edge_count", "nodes_touched",
    "edges_touched", "rows", "dims", "top_k", "ann_k", "bytes_read",
    "batch_size", "relation_selectivity",
)
LATENCY_KEYS = ("latency_ms", "mean_ms", "wall_time_ms", "duration_ms")
OPERATION_KEYS = ("operation", "operator", "operator_sample", "op")


class CostModelError(ValueError):
    """A measured-data or artifact invariant was violated."""


@dataclass(frozen=True)
class OperationRecord:
    """One measured operation, normalized from a JSONL event.

    ``latency_ms`` must have been observed by the caller.  ``features`` is a
    compact context dictionary; absent values are represented as zero after a
    log1p transform, not imputed from a fictional measurement.
    """

    operation: str
    latency_ms: float
    features: dict[str, float]
    source: str = ""
    line: int = 0


def _positive_finite(value: object, field: str) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise CostModelError(f"{field} is not numeric") from exc
    if not math.isfinite(converted) or converted < 0:
        raise CostModelError(f"{field} must be a finite non-negative number")
    return converted


def _first(record: Mapping[str, Any], keys: Sequence[str]) -> object | None:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def normalize_record(record: Mapping[str, Any], *, source: str = "", line: int = 0) -> OperationRecord:
    """Normalize a CUDA/VM JSON event, rejecting incomplete observations."""
    operation = _first(record, OPERATION_KEYS)
    if not isinstance(operation, str) or not operation.strip():
        raise CostModelError("record has no operation")
    latency = _first(record, LATENCY_KEYS)
    if latency is None:
        raise CostModelError("record has no observed latency_ms/mean_ms")
    features: dict[str, float] = {}
    aliases = {
        "frontier_size": ("frontier_size", "frontier_len"),
        "candidate_count": ("candidate_count", "candidates"),
        "edge_count": ("edge_count", "edges_examined"),
        "nodes_touched": ("nodes_touched", "nodes_visited"),
        "edges_touched": ("edges_touched", "edges_examined"),
        "rows": ("rows",), "dims": ("dims", "embedding_dim"),
        "top_k": ("top_k", "k"), "ann_k": ("ann_k",),
        "bytes_read": ("bytes_read", "bytes_estimated_read"),
        "batch_size": ("batch_size",), "relation_selectivity": ("relation_selectivity",),
    }
    for name, keys in aliases.items():
        value = _first(record, keys)
        if value is not None:
            features[name] = _positive_finite(value, name)
    return OperationRecord(operation=operation.strip(), latency_ms=_positive_finite(latency, "latency_ms"),
                           features=features, source=source, line=line)


def load_records(paths: Iterable[Path], *, strict: bool = True) -> list[OperationRecord]:
    """Read newline-delimited measured records without silently fabricating gaps.

    Blank lines are always ignored.  With ``strict=False`` malformed or
    latency-free events are skipped so a general runtime metrics file can be
    used.  Strict mode is the recommended benchmark mode because it catches a
    malformed collection job instead of quietly training on a partial file.
    """
    result: list[OperationRecord] = []
    errors: list[str] = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as stream:
            for line_number, raw in enumerate(stream, 1):
                if not raw.strip():
                    continue
                try:
                    item = json.loads(raw)
                    if not isinstance(item, dict):
                        raise CostModelError("JSONL event is not an object")
                    result.append(normalize_record(item, source=str(path), line=line_number))
                except (json.JSONDecodeError, CostModelError) as exc:
                    message = f"{path}:{line_number}: {exc}"
                    if strict:
                        raise CostModelError(message) from exc
                    errors.append(message)
    if not result:
        detail = f"; first rejected record: {errors[0]}" if errors else ""
        raise CostModelError("no measured latency records found" + detail)
    return result


def _row(record: OperationRecord, operations: Sequence[str]) -> np.ndarray:
    # Intercept + categorical operation + log1p context.  log1p makes the
    # frontier/edge scales tractable while preserving zero as a real value.
    values = [1.0]
    values.extend(1.0 if record.operation == operation else 0.0 for operation in operations)
    values.extend(math.log1p(record.features.get(name, 0.0)) for name in NUMERIC_FEATURES)
    return np.asarray(values, dtype=np.float64)


def _split(records: Sequence[OperationRecord], validation_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < validation_fraction < 1.0:
        raise CostModelError("validation_fraction must be between 0 and 1")
    if len(records) < 3:
        # A meaningful held-out error cannot exist under three observations.
        raise CostModelError("at least three measured records are required")
    validation = []
    for index, record in enumerate(records):
        key = f"{record.operation}|{record.source}|{record.line}|{index}".encode()
        validation.append(int(hashlib.sha256(key).hexdigest()[:8], 16) / 0xFFFFFFFF < validation_fraction)
    indices = np.arange(len(records))
    valid = indices[np.asarray(validation, dtype=bool)]
    train = indices[~np.asarray(validation, dtype=bool)]
    # Deterministic repair for tiny input sets/hash collisions.
    if not len(valid): valid, train = indices[-1:], indices[:-1]
    if not len(train): train, valid = indices[:-1], indices[-1:]
    return train, valid


@dataclass(frozen=True)
class CostModel:
    operations: tuple[str, ...]
    weights: tuple[float, ...]
    ridge: float
    training_records: int
    validation_records: int
    metrics: dict[str, float]

    def predict(self, operation: str, **features: float) -> float:
        if operation not in self.operations:
            # Unknown operators retain the learned intercept/context terms;
            # callers can see this by checking ``known_operation`` separately.
            operation = ""
        record = OperationRecord(operation=operation, latency_ms=0.0,
                                 features={name: _positive_finite(value, name) for name, value in features.items() if name in NUMERIC_FEATURES})
        value = float(_row(record, self.operations) @ np.asarray(self.weights, dtype=np.float64))
        return max(0.0, value)

    def known_operation(self, operation: str) -> bool:
        return operation in self.operations

    def to_dict(self) -> dict[str, Any]:
        return {"format": FORMAT_VERSION, "operations": list(self.operations), "numeric_features": list(NUMERIC_FEATURES),
                "weights": list(self.weights), "ridge": self.ridge, "training_records": self.training_records,
                "validation_records": self.validation_records, "metrics": self.metrics}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CostModel":
        if raw.get("format") != FORMAT_VERSION or tuple(raw.get("numeric_features", ())) != NUMERIC_FEATURES:
            raise CostModelError("unsupported cost model artifact")
        operations = tuple(str(x) for x in raw["operations"])
        weights = tuple(float(x) for x in raw["weights"])
        expected = 1 + len(operations) + len(NUMERIC_FEATURES)
        if len(weights) != expected or not all(math.isfinite(x) for x in weights):
            raise CostModelError("invalid coefficient vector")
        return cls(operations, weights, float(raw["ridge"]), int(raw["training_records"]),
                   int(raw["validation_records"]), {str(k): float(v) for k, v in raw["metrics"].items()})


def train_cost_model(records: Sequence[OperationRecord], *, ridge: float = 1e-3,
                     validation_fraction: float = 0.2) -> CostModel:
    """Fit ridge regression and return measured hold-out error statistics."""
    if ridge < 0 or not math.isfinite(ridge):
        raise CostModelError("ridge must be finite and non-negative")
    operations = tuple(sorted({item.operation for item in records}))
    train, valid = _split(records, validation_fraction)
    x = np.stack([_row(item, operations) for item in records])
    y = np.asarray([item.latency_ms for item in records], dtype=np.float64)
    penalty = np.eye(x.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0  # never regularize intercept
    try:
        weights = np.linalg.solve(x[train].T @ x[train] + penalty, x[train].T @ y[train])
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(x[train].T @ x[train] + penalty) @ (x[train].T @ y[train])
    predicted = np.maximum(0.0, x[valid] @ weights)
    actual = y[valid]
    errors = predicted - actual
    metrics = {
        "mae_ms": float(np.mean(np.abs(errors))),
        "rmse_ms": float(np.sqrt(np.mean(errors ** 2))),
        "mape_percent": float(np.mean(np.abs(errors) / np.maximum(actual, 1e-6)) * 100.0),
        "mean_observed_latency_ms": float(np.mean(actual)),
    }
    return CostModel(operations, tuple(float(value) for value in weights), ridge, len(train), len(valid), metrics)


def save_model(model: CostModel, path: Path, *, records: Sequence[OperationRecord]) -> None:
    """Atomically persist model plus the source-record provenance, never samples."""
    payload = model.to_dict()
    payload["sources"] = sorted({record.source for record in records})
    payload["record_count"] = len(records)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)


def load_model(path: Path) -> CostModel:
    return CostModel.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
