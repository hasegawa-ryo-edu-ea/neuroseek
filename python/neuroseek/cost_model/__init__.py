"""Small, auditable hardware-cost predictor trained from measured JSONL.

The model is deliberately a regularized linear regression rather than a large
network: it is cheap to retrain on a Jetson, serializes as JSON, and makes its
feature transformations inspectable.  It predicts latency only from records
that actually contain an observed latency; it never invents benchmark data.
"""

from .model import CostModel, CostModelError, OperationRecord, load_records, train_cost_model

__all__ = ["CostModel", "CostModelError", "OperationRecord", "load_records", "train_cost_model"]
