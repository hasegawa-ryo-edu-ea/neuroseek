from __future__ import annotations

from pathlib import Path

import pytest

from neuroseek.telemetry.jetson import JetsonTelemetryCollector, ThermalSafetyPolicy, parse_tegrastats, read_sysfs_power, read_sysfs_thermal
from neuroseek.telemetry.events import EventWriter


TEGRASAMPLE = """08-10-2026 21:39:51 RAM 2513/7620MB (lfb 1x4MB) SWAP 25/3810MB (cached 0MB) CPU [11%@729,7%@729,8%@729,4%@729,3%@729,14%@729] GR3D_FREQ 0% cpu@51.218C soc2@51.093C soc0@50.125C gpu@51.593C tj@51.718C soc1@50.343C VDD_IN 4744mW/6179mW VDD_CPU_GPU_CV 1273mW/2369mW VDD_SOC 1234mW/1392mW"""


def test_parse_tegrastats_preserves_real_named_measurements() -> None:
    observed = parse_tegrastats(TEGRASAMPLE)
    assert observed["telemetry_source"] == "tegrastats"
    assert observed["ram_used_gib"] == pytest.approx(2513 / 1024)
    assert observed["gpu_utilization_pct"] == 0.0
    assert observed["thermal_sensors_c"]["gpu"] == pytest.approx(51.593)
    assert observed["temperature_c"] == pytest.approx(51.718)
    assert observed["board_power_mw"] == 4744.0
    assert observed["board_power_average_mw"] == 6179.0
    assert observed["power_rails_mw"]["vdd_soc"]["instant_mw"] == 1234.0


def test_sysfs_sources_are_scaled_and_do_not_invent_board_power(tmp_path: Path) -> None:
    thermal = tmp_path / "class/thermal/thermal_zone0"
    thermal.mkdir(parents=True)
    (thermal / "type").write_text("GPU-therm\n")
    (thermal / "temp").write_text("51250\n")
    duplicate = tmp_path / "class/thermal/thermal_zone1"
    duplicate.mkdir(parents=True)
    (duplicate / "type").write_text("GPU-therm\n")
    (duplicate / "temp").write_text("49\n")
    hwmon = tmp_path / "class/hwmon/hwmon0"
    hwmon.mkdir(parents=True)
    (hwmon / "name").write_text("ina3221\n")
    (hwmon / "power1_input").write_text("4744000\n")
    observed = read_sysfs_thermal(tmp_path)
    assert observed["thermal_sensors_c"] == {"gpu-therm": 51.25, "gpu-therm_2": 49.0}
    assert observed["temperature_c"] == 51.25
    power = read_sysfs_power(tmp_path)
    assert power == {"sysfs_power_rails_mw": {"ina3221:power1_input": 4744.0}}
    assert "board_power_mw" not in power


def test_collector_rate_limits_tegrastats_and_marks_age(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    proc = tmp_path / "proc"; proc.mkdir()
    (proc / "meminfo").write_text("MemTotal: 1048576 kB\nMemAvailable: 524288 kB\n")
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr("neuroseek.telemetry.jetson.sample_tegrastats", lambda command: calls.append(tuple(command)) or parse_tegrastats(TEGRASAMPLE))
    time_values = iter([10.0, 12.0, 21.0])
    collector = JetsonTelemetryCollector(tegrastats_interval_seconds=10, tegrastats_command=("fake-tegrastats",), sys_root=tmp_path, proc_root=proc, clock=lambda: next(time_values))
    first = collector.sample(); second = collector.sample(); third = collector.sample()
    assert len(calls) == 2
    assert first["tegrastats_age_seconds"] == 0.0
    assert second["tegrastats_age_seconds"] == 2.0
    assert third["tegrastats_age_seconds"] == 0.0
    assert second["ram_total_gib"] == pytest.approx(7620 / 1024)  # richer tegrastats source wins


def test_thermal_policy_requests_but_never_performs_actions() -> None:
    policy = ThermalSafetyPolicy.from_config({"telemetry": {"enabled": True, "warning_temperature_c": 80, "critical_temperature_c": 85, "critical_action": "checkpoint"}})
    assert policy is not None
    assert policy.evaluate({}).level == "unavailable"
    assert policy.evaluate({"temperature_c": 79.9}).action == "none"
    assert policy.evaluate({"temperature_c": 80}).action == "warn"
    critical = policy.evaluate({"temperature_c": 85})
    assert (critical.level, critical.action, critical.reason) == ("critical", "checkpoint", "critical_temperature")
    with pytest.raises(ValueError, match="greater"):
        ThermalSafetyPolicy(85, 80)
    assert ThermalSafetyPolicy(80, 85, "checkpoint_stop").evaluate({"temperature_c": 85}).action == "checkpoint_stop"
    with pytest.raises(ValueError, match="checkpoint_stop"):
        ThermalSafetyPolicy(80, 85, "throttle")


def test_hardware_events_have_a_dedicated_durable_stream(tmp_path: Path) -> None:
    writer = EventWriter(tmp_path)
    writer.emit("TrainingEvent", reward=1.0)
    writer.emit("HardwareEvent", cuda_score_latency_ms=1.25)
    assert "HardwareEvent" in (tmp_path / "metrics.jsonl").read_text()
    hardware = (tmp_path / "hardware.jsonl").read_text()
    assert "HardwareEvent" in hardware
    assert "TrainingEvent" not in hardware


def test_event_writer_batches_barriers_but_syncs_at_safe_point(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[int] = []
    monkeypatch.setattr("neuroseek.telemetry.events.os.fsync", lambda descriptor: calls.append(descriptor))
    writer = EventWriter(tmp_path, fsync_interval_seconds=60.0)
    writer.emit("TrainingEvent", global_step=1)
    writer.emit("TrainingEvent", global_step=2)
    # Every event is flushed/readable immediately, but a 50-hour run must not
    # issue one storage barrier per minibatch.
    assert len(calls) == 1
    writer.sync()
    assert len(calls) == 2
    assert "global_step" in (tmp_path / "metrics.jsonl").read_text()
