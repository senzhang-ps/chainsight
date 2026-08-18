"""首日 M5 四象限定位：配置生命周期 × M1/M4 输入来源。

只执行至 M5：不运行 M6/M3，不修改生产代码，不向 DB 写入任何数据。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import pytest

from src.core.orchestrator import Orch
from src.modules import module1
from src.modules.deployment_planning.integration_refactor import ModuleFive
from src.modules.production_planning.integration_refactor import ModuleFour
from src.modules.state_context import StateContext
from tests import test_m5_controlled_db_inputs as controlled
from tests import test_m5_two_way_compare as helpers


PROJECT_ROOT = Path(__file__).parent.parent.resolve()
TEST_ENTRY_DIR = PROJECT_ROOT / "test"
if str(TEST_ENTRY_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_ENTRY_DIR))

DAY = pd.Timestamp("2025-12-15")
DAY_TEXT = DAY.strftime("%Y-%m-%d")
RUNTIME_EXCEL = PROJECT_ROOT / "input" / "OC" / "OC_Paste_S1_20251224_extension" / "OC_Paste_S1_20251224_repare.xlsx"
REPORT_BASE = PROJECT_ROOT / "outputs" / "m5_first_day_four_quadrant_diagnosis"
TARGET_MATERIAL = "21098519"
TARGET_NODES = {"0386", "386", "A668", "A672"}


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
    return {"rows": len(frame), "columns": list(frame.columns), "sha256": digest,
            "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()}}


def _target_rows(frame: pd.DataFrame) -> list[dict]:
    frame = pd.DataFrame() if frame is None else frame.copy()
    if frame.empty:
        return []
    mask = pd.Series(False, index=frame.index)
    if "material" in frame:
        mask |= frame["material"].astype(str).eq(TARGET_MATERIAL)
    for column in ("location", "sending", "receiving", "node", "upstream"):
        if column in frame:
            mask &= frame[column].astype(str).isin(TARGET_NODES) if "material" not in frame else pd.Series(True, index=frame.index)
    if "material" in frame:
        mask = frame["material"].astype(str).eq(TARGET_MATERIAL)
    return frame.loc[mask].to_dict(orient="records")


def _build_file_context(report_dir: Path) -> tuple[Orch, StateContext]:
    if not RUNTIME_EXCEL.exists():
        pytest.skip(f"找不到运行时 Excel: {RUNTIME_EXCEL}")
    orch = Orch(DAY_TEXT, DAY_TEXT, config_path=str(RUNTIME_EXCEL),
                output_path=str(report_dir / "file_config"), engine="pandas",
                skip_dq=True, enable_persistence=False)
    context = StateContext(DAY_TEXT, orch=orch); context.initialize(orch.all_config)
    return orch, context


def _build_db_context(report_dir: Path) -> tuple[Orch, StateContext, dict]:
    db = controlled._db(); db.connect()
    try:
        config = helpers._load_prepared_config(db)
        history = controlled._history(db)
    finally:
        db.close()
    orch, context = helpers._new_context(config, f"{report_dir.name}_db_config", "pandas")
    return orch, context, history


def _run_current_m1_m4(orch: Orch, context: StateContext) -> tuple[dict, dict]:
    """精确复用完整集成首日 M1→M4 以及 StateContext 写回顺序。"""
    context.day_start(DAY_TEXT)
    m1 = module1.ModuleOne(DAY_TEXT, orchestrator=context, orch=orch)
    m1.prepare(); m1.run(); m1_result = m1.output()
    context.apply_shipments(m1_result["shipment_df"], DAY_TEXT)
    context.apply_deployment_demand_inputs(
        m1_result["supply_demand_df"],
        m1_result.get("all_orders_for_next_day", m1_result["orders_df"]), DAY_TEXT,
    )
    m4 = ModuleFour(DAY_TEXT, DAY_TEXT, orchestrator=context, orch=orch)
    m4.prepare(); m4.module3_result = context.get_previous_m3_result(DAY_TEXT)
    m4.previous_line_states_override = context.get_previous_line_state(DAY_TEXT)
    m4.allocated_capacity_override = context.get_all_previous_allocated_capacity(DAY_TEXT)
    m4.run(); m4_result = m4.output()
    context.apply_line_state(m4_result.get("current_line_states", {}), DAY_TEXT)
    context.apply_allocated_capacity(m4_result.get("current_allocated_capacity", {}), DAY_TEXT)
    context.apply_production(m4_result["production_df"], DAY_TEXT)
    return m1_result, m4_result


def _run_cell(name: str, config_source: str, input_source: str, report_dir: Path) -> dict:
    if config_source == "file":
        orch, context = _build_file_context(report_dir)
        history = None
    else:
        orch, context, history = _build_db_context(report_dir)
    if input_source == "history":
        if history is None:
            db = controlled._db(); db.connect()
            try:
                history = controlled._history(db)
            finally:
                db.close()
        m1_result, m4_result = helpers._replay_inputs(context, DAY, history)
    else:
        m1_result, m4_result = _run_current_m1_m4(orch, context)

    m5 = ModuleFive(DAY, DAY, state_context=context, orch=orch, verbose=False)
    m5.prepare()
    daily_inputs = m5.load_daily_inputs(DAY)
    active = m5.build_active_network(daily_inputs, DAY)
    routes = m5.build_route_parameters(active, daily_inputs)
    horizon = m5.build_node_horizon(active, DAY, routes)
    direct = m5.build_direct_demand(active, DAY, daily_inputs, routes)
    m5.run(); result = m5.output()
    return {
        "cell": name, "config_source": config_source, "input_source": input_source,
        "m5_deployment_rows": len(result["deployment_plan"]),
        "m1": {name: _frame_signature(m1_result.get(name, pd.DataFrame())) for name in ("supply_demand_df", "orders_df", "all_orders_for_next_day", "shipment_df")},
        "m4_production": _frame_signature(m4_result.get("production_df", pd.DataFrame())),
        "views": {name: _frame_signature(getattr(context, getter)(DAY_TEXT)) for name, getter in {
            "SupplyDemandLog": "get_deployment_supply_demand_view", "OrderLog": "get_deployment_order_log_view",
            "TodayShipment": "get_shipment_log_view", "Inventory": "get_beginning_inventory_view",
            "Production": "get_deployment_production_view"}.items()},
        "m5_material_location": _frame_signature(m5.static["MaterialLocation"]),
        "m5_material_location_target": _target_rows(m5.static["MaterialLocation"]),
        "routes_target": _target_rows(routes), "horizon_target": _target_rows(horizon),
        "direct_target": _target_rows(direct),
        "deployment_target": _target_rows(result["deployment_plan"]),
        "config_m4_material_location": _frame_signature(orch.all_config.get("M4_MaterialLocationLineCfg", pd.DataFrame())),
        "config_m4_material_location_target": _target_rows(orch.all_config.get("M4_MaterialLocationLineCfg", pd.DataFrame())),
    }


def _safe_run_cell(name: str, config_source: str, input_source: str, report_dir: Path) -> dict:
    """诊断必须记录单一路径失败，而非使后续象限无法执行。"""
    try:
        return _run_cell(name, config_source, input_source, report_dir)
    except Exception as error:  # noqa: BLE001 - 诊断报告需保留原始异常
        return {
            "cell": name,
            "config_source": config_source,
            "input_source": input_source,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }


def test_m5_first_day_four_quadrant_diagnosis() -> None:
    if os.environ.get("RUN_M5_FOUR_QUADRANT_DIAG") != "1":
        pytest.skip("设置 RUN_M5_FOUR_QUADRANT_DIAG=1 后运行首日四象限定位")
    report_dir = REPORT_BASE / time.strftime("run_%Y%m%d_%H%M%S")
    report_dir.mkdir(parents=True, exist_ok=False)
    cells = [
        _safe_run_cell("A_db_config_history_inputs", "db", "history", report_dir),
        _safe_run_cell("B_file_config_history_inputs", "file", "history", report_dir),
        _safe_run_cell("C_db_config_current_inputs", "db", "current", report_dir),
        _safe_run_cell("D_file_config_current_inputs", "file", "current", report_dir),
    ]
    report = {"date": DAY_TEXT, "policy": "首日截至 M5；M6/M3 未运行；DB 只读；不修改生产代码", "cells": cells}
    report_path = report_dir / "m5_first_day_four_quadrant_diagnosis.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[M5 four quadrant] report: {report_path}", flush=True)
    assert len(cells) == 4
    assert report_path.exists()