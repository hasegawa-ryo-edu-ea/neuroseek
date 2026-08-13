"""Durable, append-only runtime events. The dashboard is deliberately only a reader."""
from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Any


class EventWriter:
    def __init__(self, run_dir: Path, *, fsync_interval_seconds: float = 1.0) -> None:
        if fsync_interval_seconds <= 0.0:
            raise ValueError("fsync_interval_seconds must be positive")
        self.path = run_dir / "metrics.jsonl"
        self.hardware_path = run_dir / "hardware.jsonl"
        self.fsync_interval_seconds = float(fsync_interval_seconds)
        self._last_fsync = float("-inf")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # This in-memory ring does not replace the durable JSONL.  It provides
        # the crash bundle with context even when an exception happens before
        # the next tail reader can observe the last fsynced event.
        self.recent: deque[dict[str, Any]] = deque(maxlen=64)

    def _sync_due(self, now: float) -> bool:
        if now - self._last_fsync < self.fsync_interval_seconds:
            return False
        self._last_fsync = now
        return True

    @staticmethod
    def _append(path: Path, line: str, *, sync: bool) -> None:
        with path.open("a", encoding="utf-8") as out:
            out.write(line)
            out.flush()
            if sync:
                os.fsync(out.fileno())

    def sync(self) -> None:
        """Force all currently emitted streams to stable storage at a safe point."""
        for path in (self.path, self.hardware_path):
            if not path.is_file():
                continue
            with path.open("a", encoding="utf-8") as out:
                out.flush()
                os.fsync(out.fileno())
        self._last_fsync = time.monotonic()

    def emit(self, category: str, **values: Any) -> dict[str, Any]:
        event = {"ts_unix": time.time(), "category": category, **values}
        line = json.dumps(event, sort_keys=True, allow_nan=False) + "\n"
        sync = self._sync_due(time.monotonic())
        self._append(self.path, line, sync=sync)
        # Hardware observations are also kept in a narrow durable stream for
        # the cost model.  This avoids retaining an unbounded raw series in a
        # checkpoint while leaving training metrics and the TUI independent.
        if category == "HardwareEvent":
            self._append(self.hardware_path, line, sync=sync)
        self.recent.append(event)
        return event
