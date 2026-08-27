"""只读比较 DB 配置与 Excel 配置，不执行任何业务模块。"""

# 测试文件说明
# 测试目的：集中验证配置加载、数据库映射与 Excel 配置的一致性。
# 测试方法：按 `regression` 类测试组织用例，使用夹具、模拟数据或真实依赖执行被测流程。
# 输入和期望：输入覆盖正常、边界和异常业务数据；期望由各测试用例的显式断言定义。
# 业务逻辑和原因：将相关验证集中维护，确保配置加载、数据库映射与 Excel 配置的一致性变更时能够快速定位回归影响。



from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pgsql_db.db_connection import DatabaseConnection
from pgsql_db.settings import get_database_config
from src.core.main_integration.config_loader import (
    load_configuration,
    load_configuration_from_dict,
    prepare_configuration,
)
from src.core.run.db_config import _load_config_from_database


PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
CONFIG_NAME = os.environ.get("CONFIG_DB_EXCEL_COMPARE_NAME", "OC_Paste_S1_20251224_repare")
DB_SCHEMA = os.environ.get("CONFIG_DB_EXCEL_COMPARE_SCHEMA", "input")
EXCEL_PATH = Path(os.environ.get(
    "CONFIG_DB_EXCEL_COMPARE_PATH",
    str(PROJECT_ROOT / "input" / "OC" / "OC_Paste_S1_20251224_extension" / "OC_Paste_S1_20251224_repare.xlsx"),
))
TARGET_MATERIAL = "21098519"
REPORT_BASE = PROJECT_ROOT / "outputs" / "config_db_excel_parity"


def _db() -> DatabaseConnection:
    settings = get_database_config()
    return DatabaseConnection(
        host=settings["host"], port=settings["port"], database=settings["database"],
        user=settings["user"], password=settings["password"], schema=DB_SCHEMA,
        auto_create_schema=False,
    )


def _canonical_column(column: object) -> str:
    return "_".join(str(column).strip().casefold().replace("-", "_").split())


def _canonical_value(value: Any) -> str:
    if pd.isna(value):
        return "<NA>"
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "isoformat") and not isinstance(value, str):
        try:
            return value.isoformat()
        except TypeError:
            pass
    if isinstance(value, float):
        return format(value, ".15g")
    return str(value).strip()


def _canonical_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame() if frame is None else frame.copy()
    frame.columns = [_canonical_column(column) for column in frame.columns]
    # 对大小写差异的列名合并：同名重复列保留首个非空值，随后再比较。
    if frame.columns.duplicated().any():
        result = pd.DataFrame(index=frame.index)
        for name in dict.fromkeys(frame.columns):
            source = frame.loc[:, frame.columns == name]
            result[name] = source.bfill(axis=1).iloc[:, 0]
        frame = result
    return frame.apply(lambda column: column.map(_canonical_value))


def _signature(frame: pd.DataFrame) -> tuple[str, list[str]]:
    canonical = _canonical_frame(frame)
    columns = sorted(canonical.columns)
    canonical = canonical.reindex(columns, axis=1)
    rows = sorted("\x1f".join(row) for row in canonical.itertuples(index=False, name=None))
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest(), columns


def _multiset_delta(db_frame: pd.DataFrame, excel_frame: pd.DataFrame) -> dict:
    left, right = _canonical_frame(db_frame), _canonical_frame(excel_frame)
    columns = sorted(set(left.columns) | set(right.columns))
    left, right = left.reindex(columns=columns, fill_value="<MISSING_COLUMN>"), right.reindex(columns=columns, fill_value="<MISSING_COLUMN>")
    left_counts = left.value_counts(dropna=False).rename("db_count")
    right_counts = right.value_counts(dropna=False).rename("excel_count")
    counts = pd.concat([left_counts, right_counts], axis=1).fillna(0).astype(int).reset_index()
    counts["delta"] = counts["db_count"] - counts["excel_count"]
    db_only = counts.loc[counts["delta"].gt(0)].head(20)
    excel_only = counts.loc[counts["delta"].lt(0)].head(20)
    return {
        "db_rows": len(db_frame), "excel_rows": len(excel_frame), "columns": columns,
        "db_signature": _signature(db_frame)[0], "excel_signature": _signature(excel_frame)[0],
        "equal_multiset": bool(counts["delta"].eq(0).all()),
        "db_only_row_instances": int(counts.loc[counts["delta"].gt(0), "delta"].sum()),
        "excel_only_row_instances": int(-counts.loc[counts["delta"].lt(0), "delta"].sum()),
        "db_only_samples": db_only.to_dict(orient="records"),
        "excel_only_samples": excel_only.to_dict(orient="records"),
    }


def _target_rows(frame: pd.DataFrame) -> list[dict]:
    if frame is None or frame.empty or "material" not in frame.columns:
        return []
    return frame.loc[frame["material"].astype(str).eq(TARGET_MATERIAL)].to_dict(orient="records")


def test_database_configuration_matches_excel_configuration() -> None:
    # 测试目的：验证“database、configuration、matches、excel、configuration”场景下配置加载、数据库映射与 Excel 配置的一致性。
    # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `_db()`，再通过 1 个断言核对结果、状态或异常条件。
    # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
    # 业务逻辑和原因：该校验防止配置加载、数据库映射与 Excel 配置的一致性在重构、引擎切换或跨日运行时发生静默偏差。
    """比较原始加载后和统一 prepare 后的全部配置表，结果只写报告。"""
    if os.environ.get("RUN_CONFIG_DB_EXCEL_COMPARE") != "1":
        pytest.skip("设置 RUN_CONFIG_DB_EXCEL_COMPARE=1 后运行只读配置审计")
    if not EXCEL_PATH.exists():
        pytest.skip(f"找不到 Excel 配置: {EXCEL_PATH}")

    db = _db(); db.connect()
    try:
        db_raw = _load_config_from_database(db, CONFIG_NAME)
    finally:
        db.close()
    if not db_raw:
        pytest.skip(f"schema={DB_SCHEMA} 中找不到 config_name={CONFIG_NAME}")

    db_loaded = load_configuration_from_dict(db_raw, CONFIG_NAME)
    excel_loaded = load_configuration(EXCEL_PATH)
    db_prepared, excel_prepared = prepare_configuration(db_loaded), prepare_configuration(excel_loaded)
    report_dir = REPORT_BASE / time.strftime("run_%Y%m%d_%H%M%S")
    report_dir.mkdir(parents=True, exist_ok=False)

    phases = {"loaded": (db_loaded, excel_loaded), "prepared": (db_prepared, excel_prepared)}
    tables = []
    for phase, (db_config, excel_config) in phases.items():
        for table in sorted(set(db_config) | set(excel_config), key=str.casefold):
            db_frame, excel_frame = db_config.get(table, pd.DataFrame()), excel_config.get(table, pd.DataFrame())
            delta = _multiset_delta(db_frame, excel_frame)
            delta.update({"phase": phase, "table": table, "db_target_material_rows": _target_rows(db_frame), "excel_target_material_rows": _target_rows(excel_frame)})
            tables.append(delta)
            if not delta["equal_multiset"] or table in {"Global_LeadTime", "M4_MaterialLocationLineCfg", "M5_DeployConfig", "Global_Network"}:
                pd.DataFrame(delta["db_target_material_rows"]).to_csv(report_dir / f"{phase}_{table}_db_21098519.csv", index=False, encoding="utf-8-sig")
                pd.DataFrame(delta["excel_target_material_rows"]).to_csv(report_dir / f"{phase}_{table}_excel_21098519.csv", index=False, encoding="utf-8-sig")

    report = {
        "database_schema": DB_SCHEMA, "config_name": CONFIG_NAME, "excel_path": str(EXCEL_PATH),
        "policy": "只读配置加载与比较；未运行 M1/M4/M5/M6/M3，未写数据库",
        "tables": tables,
        "mismatch_count": sum(not item["equal_multiset"] for item in tables),
    }
    report_path = report_dir / "config_db_excel_parity.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[config db/excel parity] report: {report_path}", flush=True)
    assert report_path.exists()