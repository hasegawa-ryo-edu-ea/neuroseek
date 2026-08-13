"""Honest Jetson telemetry and thermal-safety decisions.

This module never synthesizes a sensor reading.  A field is absent when its
source is unavailable, which matters on containers where not all of Jetson's
sysfs trees are mounted.  ``tegrastats`` is the preferred source for board
power, GR3D utilization, and named temperatures; sysfs remains useful when
the command is unavailable.

The policy below reports *requested* actions only.  It deliberately does not
change nvpmodel, clocks, fans, or kernel thermal controls.  The trainer owns
the safe point at which it checkpoints or pauses.
"""
from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


_RAM = re.compile(r"\bRAM\s+(\d+)\s*/\s*(\d+)MB\b", re.IGNORECASE)
_SWAP = re.compile(r"\bSWAP\s+(\d+)\s*/\s*(\d+)MB\b", re.IGNORECASE)
_GR3D = re.compile(r"\bGR3D_FREQ\s+(\d+)%", re.IGNORECASE)
_TEMPERATURE = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)@(-?\d+(?:\.\d+)?)C\b", re.IGNORECASE)
_POWER = re.compile(r"\b(VDD_[A-Za-z0-9_]+)\s+(\d+)mW\s*/\s*(\d+)mW\b", re.IGNORECASE)
_CPU = re.compile(r"\bCPU\s*\[([^]]*)\]", re.IGNORECASE)
_CPU_UTIL = re.compile(r"(\d+)%@")


def parse_tegrastats(line: str) -> dict[str, Any]:
    """Parse one real tegrastats line without inventing missing fields.

    The first power number is instantaneous and the second is the utility's
    rolling average.  Both are preserved with unambiguous names.
    """
    observed: dict[str, Any] = {"telemetry_source": "tegrastats"}
    ram = _RAM.search(line)
    if ram:
        used, total = (int(item) for item in ram.groups())
        observed.update(ram_used_gib=used / 1024.0, ram_total_gib=total / 1024.0)
    swap = _SWAP.search(line)
    if swap:
        used, total = (int(item) for item in swap.groups())
        observed.update(swap_used_gib=used / 1024.0, swap_total_gib=total / 1024.0)
    gr3d = _GR3D.search(line)
    if gr3d:
        observed["gpu_utilization_pct"] = float(gr3d.group(1))
    cpu = _CPU.search(line)
    if cpu:
        utilizations = [int(value) for value in _CPU_UTIL.findall(cpu.group(1))]
        if utilizations:
            observed["cpu_utilization_pct"] = sum(utilizations) / len(utilizations)
    temperatures = {name.lower(): float(value) for name, value in _TEMPERATURE.findall(line)}
    if temperatures:
        observed["thermal_sensors_c"] = temperatures
        observed["temperature_c"] = max(temperatures.values())
    rails: dict[str, dict[str, float]] = {}
    for rail, instant, average in _POWER.findall(line):
        rails[rail.lower()] = {"instant_mw": float(instant), "average_mw": float(average)}
    if rails:
        observed["power_rails_mw"] = rails
        # VDD_IN is the board-input rail exported by tegrastats, not a made-up
        # sum of other rails (which can overlap electrically).
        vdd_in = rails.get("vdd_in")
        if vdd_in:
            observed["board_power_mw"] = vdd_in["instant_mw"]
            observed["board_power_average_mw"] = vdd_in["average_mw"]
    return observed


def _read_meminfo(proc_root: Path) -> dict[str, float]:
    try:
        fields = {
            key: value
            for line in (proc_root / "meminfo").read_text(encoding="utf-8").splitlines()
            if ":" in line
            for key, value in [line.split(":", 1)]
        }
        total = float(fields["MemTotal"].split()[0])
        available = float(fields["MemAvailable"].split()[0])
        return {"ram_used_gib": (total - available) / 1024**2, "ram_total_gib": total / 1024**2}
    except (OSError, KeyError, ValueError, IndexError):
        return {}


def read_sysfs_thermal(sys_root: Path = Path("/sys")) -> dict[str, Any]:
    """Read all usable thermal zones from sysfs, retaining their real names."""
    sensors: dict[str, float] = {}
    root = sys_root / "class" / "thermal"
    try:
        zones = sorted(root.glob("thermal_zone*"))
    except OSError:
        zones = []
    for zone in zones:
        try:
            name = (zone / "type").read_text(encoding="utf-8").strip().lower()
            raw = float((zone / "temp").read_text(encoding="utf-8").strip())
        # A few kernel virtual files have been observed to raise TypeError
        # through Python's buffered decoder while their backing driver is
        # changing state.  It is not a measurement and must simply be absent.
        except (OSError, UnicodeError, TypeError, ValueError):
            continue
        # Linux thermal zones conventionally expose millidegrees.  Some
        # drivers expose degrees, so only scale values that prove the unit.
        value = raw / 1000.0 if abs(raw) >= 1000.0 else raw
        if name:
            # Multiple zones can share a type.  Preserve all without silently
            # overwriting one another.
            key, suffix = name, 2
            while key in sensors:
                key = f"{name}_{suffix}"
                suffix += 1
            sensors[key] = value
    if not sensors:
        return {}
    return {
        "telemetry_source": "sysfs",
        "thermal_sensors_c": sensors,
        "temperature_c": max(sensors.values()),
    }


def read_sysfs_power(sys_root: Path = Path("/sys")) -> dict[str, Any]:
    """Read explicitly named hwmon power sensors, if this image exposes them.

    hwmon ``power*_input`` values are micro-watts by the Linux hwmon ABI.  We
    intentionally do not scan arbitrary ``in*_input`` voltages/currents or
    multiply them, because that would produce a fabricated board-power value.
    """
    rails: dict[str, float] = {}
    try:
        devices = sorted((sys_root / "class" / "hwmon").glob("hwmon*"))
    except OSError:
        devices = []
    for device in devices:
        try:
            label = (device / "name").read_text(encoding="utf-8").strip().lower()
        except OSError:
            label = device.name
        for field in sorted(device.glob("power*_input")):
            try:
                microwatts = float(field.read_text(encoding="utf-8").strip())
            except (OSError, UnicodeError, TypeError, ValueError):
                continue
            rails[f"{label}:{field.stem}"] = microwatts / 1000.0
    return {"sysfs_power_rails_mw": rails} if rails else {}


def sample_tegrastats(
    command: Sequence[str] = ("tegrastats",), *, interval_ms: int = 250, timeout_seconds: float = 1.5
) -> dict[str, Any]:
    """Capture one tegrastats sample, always killing the helper afterward."""
    if interval_ms <= 0 or timeout_seconds <= 0:
        raise ValueError("tegrastats interval and timeout must be positive")
    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(
            [*command, "--interval", str(interval_ms)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, _stderr = process.communicate(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, _stderr = process.communicate()
    except (OSError, subprocess.SubprocessError):
        return {}
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.communicate()
    for line in reversed(stdout.splitlines()):
        parsed = parse_tegrastats(line)
        if len(parsed) > 1:
            return parsed
    return {}


class JetsonTelemetryCollector:
    """Rate-limited real telemetry collector suitable for a training loop."""

    def __init__(
        self,
        *,
        tegrastats_interval_seconds: float = 10.0,
        tegrastats_command: Sequence[str] = ("tegrastats",),
        sys_root: Path = Path("/sys"),
        proc_root: Path = Path("/proc"),
        clock: Any = time.monotonic,
    ) -> None:
        if tegrastats_interval_seconds <= 0:
            raise ValueError("tegrastats_interval_seconds must be positive")
        self.interval_seconds = float(tegrastats_interval_seconds)
        self.command = tuple(tegrastats_command)
        self.sys_root = sys_root
        self.proc_root = proc_root
        self.clock = clock
        self._last_tegrastats_at = float("-inf")
        self._last_tegrastats: dict[str, Any] = {}

    def sample(self) -> dict[str, Any]:
        now = float(self.clock())
        observed = {**_read_meminfo(self.proc_root), **read_sysfs_thermal(self.sys_root), **read_sysfs_power(self.sys_root)}
        if now - self._last_tegrastats_at >= self.interval_seconds:
            fresh = sample_tegrastats(self.command)
            self._last_tegrastats_at = now
            if fresh:
                self._last_tegrastats = fresh
        if self._last_tegrastats:
            # tegrastats has the richest Jetson-specific values.  Mark cached
            # data explicitly instead of presenting it as a fresh measurement.
            observed = {**observed, **self._last_tegrastats}
            observed["tegrastats_age_seconds"] = max(0.0, now - self._last_tegrastats_at)
        return observed


@dataclass(frozen=True)
class ThermalDecision:
    level: str
    action: str
    reason: str | None
    temperature_c: float | None


@dataclass(frozen=True)
class ThermalSafetyPolicy:
    """Configured operational thresholds, not claimed Jetson hardware limits."""

    warning_temperature_c: float
    critical_temperature_c: float
    critical_action: str = "checkpoint"

    def __post_init__(self) -> None:
        if self.warning_temperature_c <= 0 or self.critical_temperature_c <= self.warning_temperature_c:
            raise ValueError("critical_temperature_c must be greater than a positive warning_temperature_c")
        if self.critical_action not in {"warn", "checkpoint", "checkpoint_stop", "pause"}:
            raise ValueError("critical_action must be warn, checkpoint, checkpoint_stop, or pause")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "ThermalSafetyPolicy | None":
        section = config.get("telemetry", {})
        if not section or not bool(section.get("enabled", False)):
            return None
        return cls(
            warning_temperature_c=float(section["warning_temperature_c"]),
            critical_temperature_c=float(section["critical_temperature_c"]),
            critical_action=str(section.get("critical_action", "checkpoint")),
        )

    def evaluate(self, observed: Mapping[str, Any]) -> ThermalDecision:
        raw = observed.get("temperature_c")
        if not isinstance(raw, (int, float)):
            return ThermalDecision("unavailable", "none", None, None)
        temperature = float(raw)
        if temperature >= self.critical_temperature_c:
            return ThermalDecision("critical", self.critical_action, "critical_temperature", temperature)
        if temperature >= self.warning_temperature_c:
            return ThermalDecision("warning", "warn", "warning_temperature", temperature)
        return ThermalDecision("normal", "none", None, temperature)


def snapshot() -> dict[str, Any]:
    """Fast legacy snapshot: sysfs and memory only, no process spawn per step."""
    return {**_read_meminfo(Path("/proc")), **read_sysfs_thermal(), **read_sysfs_power()}
