"""M3 五日四象限定位：配置生命周期 × M1/M4/M5/M6 上游来源。

历史象限逐日回放固定 run 的 M1/M4/M5/M6 输出；实时象限逐日执行
M1→M4→M5→M6→M3。两种实时配置路径都由 StateContext 保存 M3 输出，
使下一日 M4 严格读取前一日 M3 ``net_demand_df``。

M5 PlanningFacts 只在内存中保存一天，历史 DB 没有该表；故历史象限仍即时运行
M5 构建事实，但实际写回 StateContext 的 M5/M6 业务输出始终来自固定历史 run。
不向数据库写入当前结果。
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
from src.modules.mrp_planning.integration_refactor import ModuleThree
from src.modules.production_planning.integration_refactor import ModuleFour
from src.modules.state_context import StateContext
from tests import test_m5_controlled_db_inputs as controlled
from tests import test_m5_two_way_compare as helpers


PROJECT_ROOT = Path(__file__).parent.parent.resolve()
START_DATE, END_DATE = "2025-12-15", "2025-12-19"
DAYS = pd.date_range(START_DATE, END_DATE, freq="D")
RUNTIME_EXCEL = PROJECT_ROOT / "input" / "OC" / "OC_Paste_S1_20251224_extension" / "OC_Paste_S1_20251224_repare.xlsx"
REPORT_BASE = PROJECT_ROOT / "outputs" / "m3_full_chain_four_quadrant_diagnosis"
M5_PLAN_TABLE = "module5_output_deploymentplan"


def _frame_signature(frame: pd.DataFrame | None) -> dict:
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
    orch = Orch(
        START_DATE, END_DATE, config_path=str(RUNTIME_EXCEL),
        output_path=str(report_dir / "file_config"), engine="pandas",
        skip_dq=True, enable_persistence=False,
    )
    context = StateContext(START_DATE, orch=orch)
    context.initialize(orch.all_config)
    return orch, context


def _load_history() -> tuple[dict, dict[str, pd.DataFrame]]:
    db = controlled._db()
    db.connect()
    try:
        config = helpers._load_prepared_config(db)
        history = controlled._history(db)
        m5_plans = {
            day.strftime("%Y-%m-%d"): controlled._load(db, M5_PLAN_TABLE, day)
            for day in DAYS
        }
    finally:
        db.close()
    if all(plan.empty for plan in m5_plans.values()):
        pytest.skip(f"历史 M5 部署计划为空: run_id={controlled.RUN_ID}")
    return history, m5_plans


def _build_db_context(report_dir: Path, config: dict) -> tuple[Orch, StateContext]:
    return helpers._new_context(config, f"{report_dir.name}_db_config", "pandas")


def _run_current_m1_m4(
    m1: module1.ModuleOne, orch: Orch, context: StateContext, day: pd.Timestamp,
) -> tuple[dict, dict]:
    """真实链路：M4 从 Context 读取前一日由 M3 写回的净需求。"""
    day_text = day.strftime("%Y-%m-%d")
    context.day_start(day_text)
    m1.simulation_date = day
    m1.run()
    m1_result = m1.output()
    context.apply_shipments(m1_result["shipment_df"], day_text)
    context.apply_deployment_demand_inputs(
        m1_result["supply_demand_df"],
        m1_result.get("all_orders_for_next_day", m1_result["orders_df"]),
        day_text,
    )
    m4 = ModuleFour(day_text, START_DATE, orchestrator=context, orch=orch)
    m4.prepare()
    m4.module3_result = context.get_previous_m3_result(day_text)
    m4.previous_line_states_override = context.get_previous_line_state(day_text)
    m4.allocated_capacity_override = context.get_all_previous_allocated_capacity(day_text)
    m4.run(); m4_result = m4.output()
    context.apply_line_state(m4_result.get("current_line_states", {}), day_text)
    context.apply_allocated_capacity(m4_result.get("current_allocated_capacity", {}), day_text)
    context.apply_production(m4_result["production_df"], day_text)
    return m1_result, m4_result


def _build_m5_facts(orch: Orch, context: StateContext, day: pd.Timestamp) -> tuple[dict, dict]:
    m5 = ModuleFive(day, START_DATE, state_context=context, orch=orch, verbose=False)
    m5.prepare(); m5.run()
    return m5.output(), context.get_planning_facts(day.strftime("%Y-%m-%d"))


def _run_current_m6(orch: Orch, context: StateContext, day: pd.Timestamp) -> dict:
    m6 = ModuleSix(day, state_context=context, orch=orch, random_seed=42)
    m6.prepare(); m6.run()
    return m6.output()


def _run_m3(orch: Orch, context: StateContext, day: pd.Timestamp) -> tuple[dict, dict]:
    m3 = ModuleThree(day, START_DATE, state_context=context, orch=orch, verbose=False)
    m3.prepare(); m3.run()
    return m3.output(), m3._backend.static


def _m3_input_views(context: StateContext, day: pd.Timestamp) -> dict[str, dict]:
    day_text = day.strftime("%Y-%m-%d")
    return {
        name: _frame_signature(getattr(context, getter)(day_text))
        for name, getter in {
            "BeginningInventory": "get_beginning_inventory_view",
            "InTransit": "get_planning_intransit_view",
            "DeliveryGR": "get_delivery_gr_view",
            "AllProduction": "get_all_production_view",
            "OpenDeployment": "get_m3_open_deployment_view",
            "DeliveryShipment": "get_delivery_shipment_log_view",
        }.items()
    }


def _apply_historical_m5_m6(
    context: StateContext, day: pd.Timestamp, m5_plan: pd.DataFrame, m6_delivery: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """历史 M5/M6 是上游事实；只用当日即时 M5 生成 M3 所需 PlanningFacts。"""
    deployment = m5_plan.copy()
    legacy_qty = "deployed_qty_invcon"
    if "deployed_qty_invCon" not in deployment and legacy_qty in deployment:
        deployment = deployment.rename(columns={legacy_qty: "deployed_qty_invCon"})
    elif "deployed_qty_invCon" in deployment and legacy_qty in deployment:
        deployment = deployment.drop(columns=[legacy_qty])
    if "deployed_qty_invCon" not in deployment and "deployed_qty" in deployment:
        deployment["deployed_qty_invCon"] = deployment["deployed_qty"]
    if "deployed_qty" not in deployment and "deployed_qty_invCon" in deployment:
        deployment["deployed_qty"] = deployment["deployed_qty_invCon"]
    context.apply_deployment(deployment, day.strftime("%Y-%m-%d"))
    delivery = m6_delivery.copy()
    context.apply_delivery(delivery, day.strftime("%Y-%m-%d"))
    return deployment, delivery


def _run_cell(
    name: str,
    config_source: str,
    input_source: str,
    report_dir: Path,
    db_config: dict,
    history: dict,
    historical_m5: dict[str, pd.DataFrame],
) -> dict:
    if config_source == "file":
        orch, context = _build_file_context(report_dir)
    else:
        orch, context = _build_db_context(report_dir, db_config)

    # 与 ``simulation_db`` 一致：M1 在整个仿真期只实例化、prepare 一次，
    # 其累计订单状态必须跨日保留，不能在每一天重新创建实例。
    current_m1 = None
    if input_source == "current":
        current_m1 = module1.ModuleOne(DAYS[0], orchestrator=context, orch=orch)
        current_m1.prepare()

    days = []
    for day in DAYS:
        day_text = day.strftime("%Y-%m-%d")
        if input_source == "history":
            m1_result, m4_result = helpers._replay_inputs(context, day, history)
        else:
            m1_result, m4_result = _run_current_m1_m4(current_m1, orch, context, day)

        m5_derived, planning_facts = _build_m5_facts(orch, context, day)
        if input_source == "history":
            m5_applied, m6_applied = _apply_historical_m5_m6(
                context, day, historical_m5[day_text], history[day_text]["delivery_plan"],
            )
        else:
            m5_applied = m5_derived["deployment_plan"]
            context.apply_deployment(m5_applied, day_text)
            m6_result = _run_current_m6(orch, context, day)
            m6_applied = m6_result["delivery_plan"]
            context.apply_delivery(m6_applied, day_text)

        input_views = _m3_input_views(context, day)
        m3_result, m3_static = _run_m3(orch, context, day)
        previous_m3_for_next_m4 = context.get_previous_m3_result(day + pd.Timedelta(days=1))["net_demand_df"]
        context.day_end(day_text)
        days.append({
            "date": day_text,
            "m1": {item: _frame_signature(m1_result.get(item)) for item in (
                "supply_demand_df", "orders_df", "all_orders_for_next_day", "shipment_df",
            )},
            "m4_production": _frame_signature(m4_result.get("production_df")),
            "m5_derived_for_facts": _frame_signature(m5_derived.get("deployment_plan")),
            "m5_applied": _frame_signature(m5_applied),
            "m6_applied": _frame_signature(m6_applied),
            "m3_static_config": {item: _frame_signature(frame) for item, frame in m3_static.items()},
            "planning_facts": {
                "version": planning_facts["version"],
                "active_network": _frame_signature(planning_facts["active_network"]),
                "routes": _frame_signature(planning_facts["routes"]),
                "node_horizon": _frame_signature(planning_facts["node_horizon"]),
                "direct_demand": _frame_signature(planning_facts["direct_demand"]),
                "layers": planning_facts["layers"], "layer_map_size": len(planning_facts["layer_map"]),
            },
            "m3_input_views": input_views,
            "m3_net_demand": _frame_signature(m3_result.get("net_demand_df")),
            "m3_saved_for_next_m4": _frame_signature(previous_m3_for_next_m4),
        })
    return {"cell": name, "config_source": config_source, "input_source": input_source, "days": days}


def _safe_run_cell(*args, **kwargs) -> dict:
    try:
        return _run_cell(*args, **kwargs)
    except Exception as error:  # noqa: BLE001 - 诊断报告需保留原始异常
        return {
            "cell": args[0], "config_source": args[1], "input_source": args[2],
            "error_type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc(),
        }


def test_m3_full_chain_four_quadrant_diagnosis() -> None:
    if os.environ.get("RUN_M3_FOUR_QUADRANT_DIAG") != "1":
        pytest.skip("设置 RUN_M3_FOUR_QUADRANT_DIAG=1 后运行 M3 五日完整链路四象限定位")
    report_dir = REPORT_BASE / time.strftime("run_%Y%m%d_%H%M%S")
    report_dir.mkdir(parents=True, exist_ok=False)
    history, historical_m5 = _load_history()
    db = controlled._db(); db.connect()
    try:
        db_config = helpers._load_prepared_config(db)
    finally:
        db.close()
    cells = [
        _safe_run_cell("A_db_config_history_m1_m4_m5_m6", "db", "history", report_dir, db_config, history, historical_m5),
        _safe_run_cell("B_file_config_history_m1_m4_m5_m6", "file", "history", report_dir, db_config, history, historical_m5),
        _safe_run_cell("C_db_config_current_full_chain", "db", "current", report_dir, db_config, history, historical_m5),
        _safe_run_cell("D_file_config_current_full_chain", "file", "current", report_dir, db_config, history, historical_m5),
    ]
    report_path = report_dir / "m3_full_chain_four_quadrant_diagnosis.json"
    report_path.write_text(json.dumps({
        "date_range": [START_DATE, END_DATE],
        "historical_run_id": controlled.RUN_ID,
        "policy": (
            "A/B 回放历史 M1/M4/M5/M6；C/D 逐日即时运行 M1→M4→M5→M6→M3；"
            "C/D 前一日 M3 输出经 StateContext 输入下一日 M4；M5 PlanningFacts 每日即时构建；DB 只读"
        ),
        "cells": cells,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[M3 full-chain four quadrant] report: {report_path}", flush=True)
    assert len(cells) == 4
    assert report_path.exists()