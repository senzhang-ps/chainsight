"""轻量级结构化性能记录，供独立仿真入口使用。"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class PerformanceTelemetry:
    """以单调时钟记录阶段、日期和模块耗时，并输出 JSON。"""

    def __init__(self, implementation: str, mode: str, schema: str, run_id: str | None = None):
        self.implementation = implementation
        self.mode = mode
        self.schema = schema
        self.run_id = run_id
        self.started_at = datetime.now(timezone.utc).isoformat()
        self._started = time.perf_counter()
        self.events: list[dict[str, Any]] = []

    @staticmethod
    def start() -> float:
        return time.perf_counter()

    def record(
        self,
        stage: str,
        started: float,
        *,
        simulation_date: str | None = None,
        module: str | None = None,
        rows: int | None = None,
        **extra: Any,
    ) -> None:
        event = {
            "stage": stage,
            "simulation_date": simulation_date,
            "module": module,
            "rows": rows,
            "duration_seconds": time.perf_counter() - started,
        }
        event.update(extra)
        self.events.append(event)

    def report(self) -> dict[str, Any]:
        by_stage: dict[str, float] = defaultdict(float)
        by_day: dict[str, float] = defaultdict(float)
        by_module: dict[str, float] = defaultdict(float)
        for event in self.events:
            duration = float(event["duration_seconds"])
            by_stage[event["stage"]] += duration
            if event.get("simulation_date"):
                by_day[str(event["simulation_date"])] += duration
            if event.get("module"):
                by_module[str(event["module"])] += duration
        return {
            "implementation": self.implementation,
            "mode": self.mode,
            "schema": self.schema,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "total_seconds": time.perf_counter() - self._started,
            "by_stage_seconds": dict(sorted(by_stage.items())),
            "by_day_seconds": dict(sorted(by_day.items())),
            "by_module_seconds": dict(sorted(by_module.items())),
            "events": self.events,
        }

    def write(self, path: str | Path) -> Path:
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.report(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return target
