from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from poisonedskills.io import write_json


@dataclass
class RunTracker:
    run_dir: Path

    @classmethod
    def create(
        cls,
        output_root: str | Path = "outputs/runs",
        name: str = "run",
        config: dict[str, Any] | None = None,
    ) -> "RunTracker":
        ts = time.strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)
        run_dir = Path(output_root) / f"{ts}_{safe_name}"
        run_dir.mkdir(parents=True, exist_ok=False)
        tracker = cls(run_dir=run_dir)
        write_json(run_dir / "config.json", config or {})
        tracker.log_event("run_started", {"name": name})
        return tracker

    def log_event(self, event: str, payload: dict[str, Any] | None = None) -> None:
        row = {"time": time.time(), "event": event, "payload": payload or {}}
        with (self.run_dir / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def write_metrics(self, metrics: dict[str, Any]) -> None:
        write_json(self.run_dir / "metrics.json", metrics)
        self.log_event("metrics_written", metrics)
