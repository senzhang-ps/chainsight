"""用数据库配置驱动完整集成的只读诊断，不注入任何历史模块输出。"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from pgsql_db.db_connection import DatabaseConnection
from pgsql_db.settings import get_database_config
from src.core.main_integration.config_loader import load_configuration_from_dict
from src.core.run.db_config import _load_config_from_database
from src.models.module import OUTPUT_REGISTRY
from tests.compare_utils import compare_dataframes_by_key


PROJECT_ROOT = Path(__file__).parent.parent.resolve()
TEST_ENTRY_DIR = PROJECT_ROOT / "test"
if str(TEST_ENTRY_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_ENTRY_DIR))
from test_integration import run_integrated_simulation


DB_SCHEMA = os.environ.get("DB_CONFIG_INTEGRATION_SCHEMA", "input")
CONFIG_NAME = os.environ.get("DB_CONFIG_INTEGRATION_NAME", "OC_Paste_S1_20251224_repare")
HISTORICAL_RUN_ID = os.environ.get("DB_CONFIG_INTEGRATION_HISTORICAL_RUN_ID", "db_OC_Paste_S1_20251224_repare_20260814_132746")
START_DATE = os.environ.get("DB_CONFIG_INTEGRATION_START_DATE", "2025-12-15")
END_DATE = os.environ.get("DB_CONFIG_INTEGRATION_END_DATE", "2025-12-19")
META_COLUMNS = ["run_id", "sim_date", "config_name", "db_write_time", "file_date"]


def _db() -> DatabaseConnection:
    cfg = get_database_config()
    return DatabaseConnection(host=cfg["host"], port=cfg["port"], database=cfg["database"],
                              user=cfg["user"], password=cfg["password"], schema=DB_SCHEMA,
                              auto_create_schema=False)


def _read_historical(db: DatabaseConnection, table: str, day: pd.Timestamp) -> pd.DataFrame:
    columns = [row[0] for row in db.execute_query(
        "SELECT column_name FROM information_schema.columns WHERE table_schema=%s AND table_name=%s ORDER BY ordinal_position",
        (DB_SCHEMA, table),
    )]
    if not columns:
        return pd.DataFrame()
    qualified = f'"{DB_SCHEMA}"."{table}"'
    rows = db.execute_query(
        f"SELECT {', '.join(columns)} FROM {qualified} WHERE run_id=%s AND sim_date::date=%s::date",
        (HISTORICAL_RUN_ID, day.strftime("%Y-%m-%d")),
    )
    return pd.DataFrame(rows, columns=columns).drop(columns=META_COLUMNS, errors="ignore")


def _write_db_config_excel(config: dict[str, pd.DataFrame], path: Path) -> None:
    """仅将 DB 已加载配置序列化为临时输入；不会写回 DB。"""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, frame in config.items():
            # Excel sheet 名最长 31 字符；当前映射表名均在该约束内。
            pd.DataFrame(frame).to_excel(writer, sheet_name=str(sheet_name)[:31], index=False)


def _row_multiset_comparison(left: pd.DataFrame, right: pd.DataFrame) -> dict:
    """以全部共有业务字段比较无序行多重集，避免重复键的任意配对误报。"""
    left, right = pd.DataFrame(left).copy(), pd.DataFrame(right).copy()
    columns = sorted(set(left.columns) & set(right.columns))

    def canonical(frame: pd.DataFrame) -> Counter:
        if not columns:
            return Counter()
        values = frame.loc[:, columns].copy()
        for column in columns:
            if "date" in column.lower():
                parsed = pd.to_datetime(values[column], errors="coerce")
                values[column] = parsed.dt.strftime("%Y-%m-%d").fillna("<NA>")
            else:
                numeric = pd.to_numeric(values[column], errors="coerce")
                if numeric.notna().sum() == values[column].notna().sum() and not numeric.empty:
                    values[column] = numeric.map(lambda value: format(float(value), ".12g"))
                else:
                    values[column] = values[column].astype("string").fillna("<NA>")
        return Counter(map(tuple, values.itertuples(index=False, name=None)))

    left_rows, right_rows = canonical(left), canonical(right)
    return {
        "columns": columns,
        "left_only_rows": sum((left_rows - right_rows).values()),
        "right_only_rows": sum((right_rows - left_rows).values()),
        "equal": left_rows == right_rows,
    }


def test_full_integration_with_database_configuration() -> None:
    if os.environ.get("RUN_DB_CONFIG_INTEGRATION_DIAG") != "1":
        pytest.skip("设置 RUN_DB_CONFIG_INTEGRATION_DIAG=1 后运行数据库配置完整集成诊断")

    report_dir = PROJECT_ROOT / "outputs" / "full_integration_db_config_diagnosis" / time.strftime("run_%Y%m%d_%H%M%S")
    report_dir.mkdir(parents=True, exist_ok=False)
    db = _db(); db.connect()
    try:
        raw_config = _load_config_from_database(db, CONFIG_NAME)
        if not raw_config:
            pytest.skip(f"schema={DB_SCHEMA} 中找不到配置 {CONFIG_NAME}")
        db_config = load_configuration_from_dict(raw_config, CONFIG_NAME)
        expected = {
            day.strftime("%Y-%m-%d"): {
                module: {name: _read_historical(db, table, day) for name, table in tables.items()}
                for module, tables in OUTPUT_REGISTRY.items()
            }
            for day in pd.date_range(START_DATE, END_DATE, freq="D")
        }
    finally:
        db.close()

    with tempfile.TemporaryDirectory(prefix="chainsight_db_config_") as temp_dir:
        config_path = Path(temp_dir) / f"{CONFIG_NAME}_from_db.xlsx"
        _write_db_config_excel(db_config, config_path)
        current = run_integrated_simulation(
            config_path=str(config_path), start_date=START_DATE, end_date=END_DATE,
            output_base_dir=str(report_dir / "scratch"), engine="pandas",
        )

    comparisons = []
    m6_day19_current = m6_day19_historical = None
    for offset, day in enumerate(pd.date_range(START_DATE, END_DATE, freq="D")):
        date_text = day.strftime("%Y-%m-%d")
        for module, table_map in OUTPUT_REGISTRY.items():
            result = current["results"][module][offset]
            for output_name in table_map:
                actual = result.get(output_name, pd.DataFrame())
                historical = expected[date_text][module][output_name]
                key_columns = (
                    ["ori_deployment_uid", "vehicle_uid"]
                    if module == "module6" and output_name == "delivery_plan"
                    else None
                )
                comparison = compare_dataframes_by_key(
                    actual, historical,
                    label=f"{date_text}:{module}:{output_name}:db_config_integrated_vs_history",
                    key_columns=key_columns,
                )
                comparisons.append({
                    "date": date_text, "module": module, "output": output_name,
                    "current_rows": len(actual), "historical_rows": len(historical), "comparison": comparison,
                })
                if date_text == END_DATE and module == "module6" and output_name == "delivery_plan":
                    m6_day19_current, m6_day19_historical = actual.copy(), historical.copy()

    m6_delivery_multiset = _row_multiset_comparison(m6_day19_current, m6_day19_historical)
    m6_day19_current.to_csv(report_dir / "m6_day19_current_delivery_plan.csv", index=False, encoding="utf-8-sig")
    m6_day19_historical.to_csv(report_dir / "m6_day19_historical_delivery_plan.csv", index=False, encoding="utf-8-sig")

    report = {
        "policy": "数据库配置只读加载→临时 Excel→完整 M1→M4→M5→M6→M3；未注入历史 M1/M4/M6，未写数据库",
        "database_schema": DB_SCHEMA, "config_name": CONFIG_NAME, "historical_run_id": HISTORICAL_RUN_ID,
        "date_range": [START_DATE, END_DATE], "engine": "pandas", "comparisons": comparisons,
        "m5_deployment_row_counts": [
            item for item in comparisons if item["module"] == "module5" and item["output"] == "deployment_plan"
        ],
        "m6_day19_delivery_plan_full_row_multiset": m6_delivery_multiset,
    }
    report_path = report_dir / "full_integration_db_config_diagnosis.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[DB config full integration] report: {report_path}", flush=True)
    assert current["dates_processed"] == len(pd.date_range(START_DATE, END_DATE, freq="D"))
    assert report_path.exists()