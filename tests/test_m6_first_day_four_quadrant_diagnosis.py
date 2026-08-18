"""首日 M6 四象限定位：配置生命周期 × M5 部署计划来源。

不运行 M3，不向数据库写入当前结果。历史 M5 部署计划从 ``input`` schema
固定 run 读取；即时输入严格按首日 M1→M4→M5 链路生成。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import pandas as pd
import pytest

from src.core.orchestrator import Orch
from src.modules import module1
from src.modules.deployment_planning.integration_refactor import ModuleFive
from src.modules.logistics_execution.integration_refactor import ModuleSix
from src.modules.production_planning.integration_refactor import ModuleFour
from src.modules.state_context import StateContext
from tests import test_m5_controlled_db_inputs as controlled
from tests import test_m5_two_way_compare as helpers


PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DAY = pd.Timestamp("2025-12-15")
DAY_TEXT = DAY.strftime("%Y-%m-%d")
RUNTIME_EXCEL = PROJECT_ROOT / "input" / "OC" / "OC_Paste_S1_20251224_extension" / "OC_Paste_S1_20251224_repare.xlsx"
REPORT_BASE = PROJECT_ROOT / "outputs" / "m6_first_day_four_quadrant_diagnosis"
M6_OUTPUTS = ("delivery_plan", "vehicle_log", "truck_usage", "unsatisfied_log", "validation_log", "bypass_log")


def _frame_signature(frame: pd.DataFrame) -> dict:
    frame = pd.DataFrame() if frame is None else frame.copy()
    columns = sorted(str(column) for column in frame.columns)
    canonical = frame.copy()
    canonical.columns = [str(column) for column in canonical.columns]
    canonical = canonical.reindex(columns=columns).fillna("<NA>").astype(str)
    digest = hashlib.sha256(
        pd.util.hash_pandas_object(canonical.sort_values(columns, kind="mergesort"), index=False).values.tobytes()
        if not canonical.empty else b""
    ).hexdigest()
    return {
        "rows": len(frame), "columns": list(frame.columns), "sha256": digest,
        "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
    }


def _build_file_context(report_dir: Path) -> tuple[Orch, StateContext]:
    if not RUNTIME_EXCEL.exists():
        pytest.skip(f"找不到运行时 Excel: {RUNTIME_EXCEL}")
    orch = Orch(DAY_TEXT, DAY_TEXT, config_path=str(RUNTIME_EXCEL),
                output_path=str(report_dir / "file_config"), engine="pandas",
                skip_dq=True, enable_persistence=False)
    context = StateContext(DAY_TEXT, orch=orch)
    context.initialize(orch.all_config)
    return orch, context


def _build_db_context(report_dir: Path) -> tuple[Orch, StateContext]:
    db = controlled._db()
    db.connect()
    try:
        config = helpers._load_prepared_config(db)
    finally:
        db.close()
    return helpers._new_context(config, f"{report_dir.name}_db_config", "pandas")


def _load_historical_deployment() -> pd.DataFrame:
    """读取原样历史表，避免 PostgreSQL 混合大小写列名被未引用 SELECT 改写。"""
    db = controlled._db()
    db.connect()
    try:
        rows = db.execute_query(
            'SELECT * FROM "input"."module5_output_deploymentplan" '
            "WHERE run_id = %s AND sim_date::date = %s::date",
            (controlled.RUN_ID, DAY_TEXT),
        )
        columns = [row[0] for row in db.execute_query(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position",
            (controlled.DB_SCHEMA, "module5_output_deploymentplan"),
        )]
    finally:
        db.close()
    frame = pd.DataFrame(rows, columns=columns).drop(columns=controlled.META, errors="ignore")
    if frame.empty:
        pytest.skip(f"历史 M5 部署计划为空: run_id={controlled.RUN_ID}, date={DAY_TEXT}")
    if "deployed_qty_invCon" not in frame and "deployed_qty" in frame:
        frame["deployed_qty_invCon"] = frame["deployed_qty"]
    if "deployed_qty" not in frame and "deployed_qty_invCon" in frame:
        frame["deployed_qty"] = frame["deployed_qty_invCon"]
    if "planned_deployment_date" not in frame and "date" in frame:
        frame = frame.rename(columns={"date": "planned_deployment_date"})
    required = {"material", "sending", "receiving", "planned_deployment_date", "deployed_qty_invCon", "deployed_qty", "demand_element"}
    missing = required - set(frame.columns)
    if missing:
        pytest.skip(f"历史 M5 部署计划缺少 M6 回放字段: {sorted(missing)}")
    return frame.loc[:, sorted(required)].copy()


def _run_current_m5(orch: Orch, context: StateContext) -> pd.DataFrame:
    """精确复用完整集成的首日 M1→M4→M5 写回顺序。"""
    context.day_start(DAY_TEXT)
    m1 = module1.ModuleOne(DAY_TEXT, orchestrator=context, orch=orch)
    m1.prepare()
    m1.run()
    m1_result = m1.output()
    context.apply_shipments(m1_result["shipment_df"], DAY_TEXT)
    context.apply_deployment_demand_inputs(
        m1_result["supply_demand_df"],
        m1_result.get("all_orders_for_next_day", m1_result["orders_df"]),
        DAY_TEXT,
    )
    m4 = ModuleFour(DAY_TEXT, DAY_TEXT, orchestrator=context, orch=orch)
    m4.prepare()
    m4.module3_result = context.get_previous_m3_result(DAY_TEXT)
    m4.previous_line_states_override = context.get_previous_line_state(DAY_TEXT)
    m4.allocated_capacity_override = context.get_all_previous_allocated_capacity(DAY_TEXT)
    m4.run()
    m4_result = m4.output()
    context.apply_line_state(m4_result.get("current_line_states", {}), DAY_TEXT)
    context.apply_allocated_capacity(m4_result.get("current_allocated_capacity", {}), DAY_TEXT)
    context.apply_production(m4_result["production_df"], DAY_TEXT)
    m5 = ModuleFive(DAY, DAY, state_context=context, orch=orch, verbose=False)
    m5.prepare()
    m5.run()
    return m5.output()["deployment_plan"]


def _run_cell(name: str, config_source: str, input_source: str, report_dir: Path, historical: pd.DataFrame) -> dict:
    if config_source == "file":
        orch, context = _build_file_context(report_dir)
    else:
        orch, context = _build_db_context(report_dir)
    if input_source == "history":
        context.day_start(DAY_TEXT)
        deployment = historical.copy()
    else:
        deployment = _run_current_m5(orch, context)
    context.apply_deployment(deployment, DAY_TEXT)
    input_views = {
        "OpenDeployment": _frame_signature(context.get_open_deployment_view(DAY_TEXT)),
        "Inventory": _frame_signature(context.get_unrestricted_inventory_view(DAY_TEXT)),
    }
    m6 = ModuleSix(DAY, state_context=context, orch=orch, random_seed=42)
    m6.prepare()
    static_config = {
        name: _frame_signature(frame)
        for name, frame in m6._backend.static.items()
    }
    m6.run()
    result = m6.output()
    context.apply_delivery(result["delivery_plan"], DAY_TEXT)
    return {
        "cell": name, "config_source": config_source, "input_source": input_source,
        "m5_deployment": _frame_signature(deployment), "input_views": input_views,
        "m6_static_config": static_config,
        "m6_outputs": {output: _frame_signature(result.get(output, pd.DataFrame())) for output in M6_OUTPUTS},
        "post_delivery_views": {
            "OpenDeployment": _frame_signature(context.get_open_deployment_view(DAY_TEXT)),
            "InTransit": _frame_signature(context.get_planning_intransit_view(DAY_TEXT)),
            "DeliveryShipment": _frame_signature(context.get_delivery_shipment_log_view(DAY_TEXT)),
        },
    }


def _safe_run_cell(*args, **kwargs) -> dict:
    try:
        return _run_cell(*args, **kwargs)
    except Exception as error:  # noqa: BLE001 - 诊断报告需保留原始异常
        return {"cell": args[0], "config_source": args[1], "input_source": args[2],
                "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()}


def test_m6_first_day_four_quadrant_diagnosis() -> None:
    if os.environ.get("RUN_M6_FOUR_QUADRANT_DIAG") != "1":
        pytest.skip("设置 RUN_M6_FOUR_QUADRANT_DIAG=1 后运行首日四象限定位")
    report_dir = REPORT_BASE / time.strftime("run_%Y%m%d_%H%M%S")
    report_dir.mkdir(parents=True, exist_ok=False)
    historical = _load_historical_deployment()
    cells = [
        _safe_run_cell("A_db_config_history_m5", "db", "history", report_dir, historical),
        _safe_run_cell("B_file_config_history_m5", "file", "history", report_dir, historical),
        _safe_run_cell("C_db_config_current_m5", "db", "current", report_dir, historical),
        _safe_run_cell("D_file_config_current_m5", "file", "current", report_dir, historical),
    ]
    report_path = report_dir / "m6_first_day_four_quadrant_diagnosis.json"
    report_path.write_text(json.dumps({
        "date": DAY_TEXT,
        "historical_m5_run_id": controlled.RUN_ID,
        "policy": "首日截至 M6；M3 未运行；DB 只读；不修改生产代码",
        "cells": cells,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[M6 four quadrant] report: {report_path}", flush=True)
    assert len(cells) == 4
    assert report_path.exists()