"""Runtime telemetry that preserves absence rather than manufacturing values."""

from .jetson import JetsonTelemetryCollector, ThermalDecision, ThermalSafetyPolicy, parse_tegrastats, snapshot

__all__ = ["JetsonTelemetryCollector", "ThermalDecision", "ThermalSafetyPolicy", "parse_tegrastats", "snapshot"]
