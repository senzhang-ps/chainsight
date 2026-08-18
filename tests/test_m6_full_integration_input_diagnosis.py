"""完整集成 M6 输入追踪：仅定位与 input 历史运行的首次 M6 差异。"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import MethodType
from typing import Any

import pandas as pd
import pytest

from pgsql_db.db_connection import DatabaseConnection
from pgsql_db.settings import get_database_config
from src.modules.logistics_execution.integration_refactor import ModuleSix
from src.models.module import OUTPUT_REGISTRY
from tests.compare_utils import compare_dataframes_by_key, dataframe_difference_details


PROJECT_ROOT = Path(__file__).parent.parent.resolve()
TEST_ENTRY_DIR = PROJECT_ROOT / "test"
if str(TEST_ENTRY_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_ENTRY_DIR))
from test_integration import run_integrated_simulation


CONFIG_PATH = Path(os.environ.get("M6_INPUT_DIAG_CONFIG_PATH", str(PROJECT_ROOT / "config" / "OC_Paste_S1_20251224_repare.xlsx")))
DB_SCHEMA = os.environ.get("M6_INPUT_DIAG_SCHEMA", "input")
RUN_ID = os.environ.get("M6_INPUT_DIAG_RUN_ID", "db_OC_Paste_S1_20251224_repare_20260814_132746")
START_DATE, END_DATE = "2025-12-15", "2025-12-19"
REPORT_BASE = PROJECT_ROOT / "outputs" / "m6_full_integration_input_diagnosis"
META = {"run_id", "sim_date", "config_name", "db_write_time", "file_date"}
M6_OUTPUTS = ("delivery_plan", "vehicle_log", "truck_usage", "unsatisfied_log", "validation_log", "bypass_log")


def _db() -> DatabaseConnection:
    cfg = get_database_config()
    return DatabaseConnection(host=cfg["host"], port=cfg["port"], database=cfg["database"], user=cfg["user"], password=cfg["password"], schema=DB_SCHEMA, auto_create_schema=False)


def _fingerprint(frame: pd.DataFrame | None) -> dict[str, Any]:
    frame = pd.DataFrame() if frame is None else pd.DataFrame(frame).copy()
    columns = sorted(map(str, frame.columns))
    work = frame.copy(); work.columns = list(map(str, work.columns)); work = work.reindex(columns=columns)
    if work.empty:
        digest = hashlib.sha256(b"").hexdigest()
    else:
        work = work.fillna("<NA>").astype(str).sort_values(columns, kind="mergesort")
        digest = hashlib.sha256(pd.util.hash_pandas_object(work, index=False).values.tobytes()).hexdigest()
    return {"rows": len(frame), "columns": list(frame.columns), "sha256": digest}


def _read_history(db: DatabaseConnection, table: str, day: str) -> pd.DataFrame:
    columns = [row[0] for row in db.execute_query("SELECT column_name FROM information_schema.columns WHERE table_schema=%s AND table_name=%s ORDER BY ordinal_position", (DB_SCHEMA, table))]
    rows = db.execute_query(f'SELECT * FROM "{DB_SCHEMA}"."{table}" WHERE run_id=%s AND sim_date::date=%s::date', (RUN_ID, day))
    return pd.DataFrame(rows, columns=columns).drop(columns=list(META), errors="ignore")


def test_m6_full_integration_input_diagnosis() -> None:
    if os.environ.get("RUN_M6_FULL_INPUT_DIAG") != "1":
        pytest.skip("设置 RUN_M6_FULL_INPUT_DIAG=1 后运行完整 M6 输入定位")
    if not CONFIG_PATH.exists():
        pytest.skip(f"找不到配置: {CONFIG_PATH}")
    report_dir = REPORT_BASE / time.strftime("run_%Y%m%d_%H%M%S")
    report_dir.mkdir(parents=True, exist_ok=False)
    captures: dict[str, dict] = {}
    original_run = ModuleSix.run

    def traced_run(module: ModuleSix):
        day = pd.Timestamp(module.simulation_date).strftime("%Y-%m-%d")
        inputs = module.load_daily_inputs(pd.Timestamp(module.simulation_date))
        run_params, prepared = module.prepare_daily_data(inputs, pd.Timestamp(module.simulation_date))

        captures[day] = {
            "static": {name: _fingerprint(value) for name, value in module._backend.static.items()},
            "inputs": {name: _fingerprint(value) for name, value in inputs.items()},
            "prepared": {name: _fingerprint(prepared[name]) for name in ("dp", "truck_con", "lead_time", "delay_dist", "bypass_rules")},
            "prepared_dp": prepared["dp"].copy(),
        }
        results = module.execute_daily_flow(run_params, prepared)
        module._result = module.finalise_result(run_params, results, prepared["validation_log"])
        module._backend.result = module._result

    ModuleSix.run = traced_run
    try:
        current = run_integrated_simulation(str(CONFIG_PATH), START_DATE, END_DATE, str(report_dir / "scratch"), engine="pandas")
    finally:
        ModuleSix.run = original_run

    db = _db(); db.connect()
    try:
        historical = {
            day.strftime("%Y-%m-%d"): {name: _read_history(db, table, day.strftime("%Y-%m-%d")) for name, table in OUTPUT_REGISTRY["module6"].items()}
            for day in pd.date_range(START_DATE, END_DATE, freq="D")
        }
    finally:
        db.close()
    comparisons, details = [], []
    for offset, day in enumerate(pd.date_range(START_DATE, END_DATE, freq="D")):
        text = day.strftime("%Y-%m-%d")
        for name in M6_OUTPUTS:
            left, right = current["results"]["module6"][offset].get(name, pd.DataFrame()), historical[text].get(name, pd.DataFrame())
            keys = ["ori_deployment_uid", "vehicle_uid"] if name == "delivery_plan" else None
            comparison = compare_dataframes_by_key(left, right, label=f"{text}:M6:{name}", key_columns=keys)
            comparisons.append({"date": text, "output": name, "current": _fingerprint(left), "historical": _fingerprint(right), "comparison": comparison})
            if name == "delivery_plan":
                detail = dataframe_difference_details(left, right, key_columns=keys)
                detail.insert(0, "date", text); details.append(detail)
                captures[text]["current_delivery_plan"] = left.to_dict(orient="records")
                captures[text]["historical_delivery_plan"] = right.to_dict(orient="records")
    if details:
        pd.concat(details, ignore_index=True).to_csv(report_dir / "delivery_plan_differences.csv", index=False, encoding="utf-8-sig")
    report = {"config_path": str(CONFIG_PATH), "schema": DB_SCHEMA, "historical_run_id": RUN_ID, "captures": captures, "comparisons": comparisons}
    report_path = report_dir / "m6_full_integration_input_diagnosis.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[M6 full input diagnosis] report: {report_path}", flush=True)
    assert len(comparisons) == 5 * len(M6_OUTPUTS)