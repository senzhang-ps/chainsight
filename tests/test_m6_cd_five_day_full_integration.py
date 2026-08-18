"""完整五日 C/D 集成诊断：配置路径 × 即时重算上游输入。

C 为 DB 配置序列化后的临时 Excel，D 为运行时 Excel。两条路径都执行完整
M1→M4→M5→M6→M3 调度；本报告只比较 M6 输出和 M6 写回后的状态。
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import pytest

from src.core.main_integration.config_loader import load_configuration_from_dict
from src.core.run.db_config import _load_config_from_database
from tests.compare_utils import compare_dataframes_by_key, dataframe_difference_details
from tests.test_full_integration_db_config_diagnosis import _db, _write_db_config_excel


PROJECT_ROOT = Path(__file__).parent.parent.resolve()
TEST_ENTRY_DIR = PROJECT_ROOT / "test"
if str(TEST_ENTRY_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_ENTRY_DIR))
from test_integration import run_integrated_simulation


CONFIG_NAME = "OC_Paste_S1_20251224_repare"
START_DATE, END_DATE = "2025-12-15", "2025-12-19"
RUNTIME_EXCEL = PROJECT_ROOT / "input" / "OC" / "OC_Paste_S1_20251224_extension" / "OC_Paste_S1_20251224_repare.xlsx"
REPORT_BASE = PROJECT_ROOT / "outputs" / "m6_cd_five_day_full_integration"
M6_OUTPUTS = ("delivery_plan", "vehicle_log", "truck_usage", "unsatisfied_log", "validation_log", "bypass_log")
M6_STATE_VIEWS = ("open_deployment", "planning_intransit", "delivery_shipment_log", "unrestricted_inventory")


def _compare(day: str, scope: str, name: str, left: pd.DataFrame, right: pd.DataFrame) -> dict:
    comparison = (
        {
            "label": f"{day}:M6:{scope}:{name}:C_db_vs_D_file",
            "left_rows": 0, "right_rows": 0, "key_columns": [],
            "left_only_keys": 0, "right_only_keys": 0, "matched_rows": 0,
            "column_differences": {}, "precision_differences": {},
            "schema": {"left_only_columns": [], "right_only_columns": []}, "reason": "both_empty",
        }
        if left.empty and right.empty
        else compare_dataframes_by_key(left, right, label=f"{day}:M6:{scope}:{name}:C_db_vs_D_file")
    )
    return {"date": day, "scope": scope, "name": name, "c_db_rows": len(left), "d_file_rows": len(right), "comparison": comparison}


def test_m6_cd_five_day_full_integration() -> None:
    if os.environ.get("RUN_M6_CD_FIVE_DAY_FULL_DIAG") != "1":
        pytest.skip("设置 RUN_M6_CD_FIVE_DAY_FULL_DIAG=1 后运行完整五日 M6 C/D 诊断")
    if not RUNTIME_EXCEL.exists():
        pytest.skip(f"找不到运行时 Excel: {RUNTIME_EXCEL}")

    report_dir = REPORT_BASE / time.strftime("run_%Y%m%d_%H%M%S")
    report_dir.mkdir(parents=True, exist_ok=False)
    db = _db()
    db.connect()
    try:
        raw_config = _load_config_from_database(db, CONFIG_NAME)
    finally:
        db.close()
    if not raw_config:
        pytest.skip(f"找不到 DB 配置: {CONFIG_NAME}")

    with tempfile.TemporaryDirectory(prefix="chainsight_m6_cd_") as temp_dir:
        db_excel = Path(temp_dir) / f"{CONFIG_NAME}_from_db.xlsx"
        _write_db_config_excel(load_configuration_from_dict(raw_config, CONFIG_NAME), db_excel)
        c_result = run_integrated_simulation(str(db_excel), START_DATE, END_DATE, str(report_dir / "C_db_config"), engine="pandas")
        d_result = run_integrated_simulation(str(RUNTIME_EXCEL), START_DATE, END_DATE, str(report_dir / "D_file_config"), engine="pandas")

    comparisons, details = [], []
    for offset, day in enumerate(pd.date_range(START_DATE, END_DATE, freq="D")):
        date_text = day.strftime("%Y-%m-%d")
        c_m6 = c_result["results"]["module6"][offset]
        d_m6 = d_result["results"]["module6"][offset]
        c_views = c_result["context_snapshots"][offset]["views"]
        d_views = d_result["context_snapshots"][offset]["views"]
        for scope, names, c_data, d_data in (
            ("output", M6_OUTPUTS, c_m6, d_m6),
            ("post_m6_state", M6_STATE_VIEWS, c_views, d_views),
        ):
            for name in names:
                left, right = c_data.get(name, pd.DataFrame()), d_data.get(name, pd.DataFrame())
                comparisons.append(_compare(date_text, scope, name, left, right))
                detail = pd.DataFrame() if left.empty and right.empty else dataframe_difference_details(left, right)
                detail.insert(0, "name", name); detail.insert(0, "scope", scope); detail.insert(0, "date", date_text)
                details.append(detail)

    pd.DataFrame([{
        "date": item["date"], "scope": item["scope"], "name": item["name"],
        "c_db_rows": item["c_db_rows"], "d_file_rows": item["d_file_rows"],
    } for item in comparisons]).to_csv(report_dir / "cd_m6_five_day_row_counts.csv", index=False, encoding="utf-8-sig")
    pd.concat(details, ignore_index=True).to_csv(report_dir / "cd_m6_five_day_differences.csv", index=False, encoding="utf-8-sig")
    report_path = report_dir / "cd_m6_five_day_full_integration.json"
    report_path.write_text(json.dumps({
        "policy": "C: DB 配置→临时 Excel→完整 M1→M4→M5→M6→M3；D: 运行时 Excel→同一完整链路；不写数据库；仅比较 M6",
        "date_range": [START_DATE, END_DATE], "engine": "pandas", "comparisons": comparisons,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[M6 full C/D five day] report: {report_path}", flush=True)
    assert c_result["dates_processed"] == d_result["dates_processed"] == 5
    assert len(comparisons) == 5 * (len(M6_OUTPUTS) + len(M6_STATE_VIEWS))
    assert report_path.exists()