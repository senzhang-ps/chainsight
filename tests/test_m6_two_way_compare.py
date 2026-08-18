"""OC 真实五日输入下 legacy、pandas、Polars M6 的严格回放。

历史 M5 DeploymentPlan 是 M6 的上游业务事实。两条独立 StateContext 以同一
初始库存、同一每日 M5 跨节点调拨回放，并各自通过 ``apply_delivery`` 推进状态。
数据库读取不纳入算法计时；测试同时比较六张 M6 输出及 M6 影响的 Context 状态。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

from pgsql_db.db_connection import DatabaseConnection
from pgsql_db.settings import get_database_config
from src.core.main_integration.config_loader import load_configuration_from_dict, prepare_configuration
from src.core.orchestrator import Orch
from src.core.run.db_config import _load_config_from_database
from src.modules.logistics_execution.integration_refactor import ModuleSix
from src.modules.logistics_execution.main import run_daily_physical_flow
from src.modules.state_context import StateContext

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
CONFIG_NAME = "OC_Paste_S1_20251224_repare"
HISTORICAL_RUN_ID = "db_OC_Paste_S1_20251224_repare_20260803_105300"
START_DATE = "2025-12-15"
END_DATE = "2025-12-19"
TABLE = "module5_output_deploymentplan"
REPORT_DIR = PROJECT_ROOT / "outputs" / "m6_three_way_compare"
REPORT_PATH = REPORT_DIR / "m6_legacy_pandas_real_parity.json"
M6_FRAME_KEYS = (
    "delivery_plan", "vehicle_log", "truck_usage", "unsatisfied_log", "validation_log", "bypass_log",
)
TIMED_SCOPE = "仅 ModuleSix.run()/legacy run_daily_physical_flow()；不含数据库读取、配置准备和 StateContext 写回"


def _db() -> DatabaseConnection:
    settings = get_database_config()
    return DatabaseConnection(
        host=settings["host"], port=settings["port"], database=settings["database"],
        user=settings["user"], password=settings["password"],
        schema=settings.get("default_schema", "public"), auto_create_schema=False,
    )


def _load_config(db: DatabaseConnection) -> dict[str, pd.DataFrame]:
    raw = _load_config_from_database(db, CONFIG_NAME)
    if not raw:
        pytest.skip(f"找不到真实配置 {CONFIG_NAME}")
    return prepare_configuration(load_configuration_from_dict(raw, CONFIG_NAME))


def _load_plans(db: DatabaseConnection) -> dict[str, pd.DataFrame]:
    columns = [row[0] for row in db.execute_query(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = 'public' AND table_name = %s ORDER BY ordinal_position", (TABLE,)
    )]
    if not columns:
        pytest.skip(f"找不到历史 M5 表 {TABLE}")
    history: dict[str, pd.DataFrame] = {}
    for day in pd.date_range(START_DATE, END_DATE, freq="D"):
        rows = db.execute_query(
            f"SELECT * FROM public.{TABLE} "
            "WHERE run_id = %s AND sim_date::date = %s::date",
            (HISTORICAL_RUN_ID, day.strftime("%Y-%m-%d")),
        )
        frame = pd.DataFrame(rows, columns=columns).drop(
            columns=["run_id", "sim_date", "config_name", "db_write_time"], errors="ignore",
        )
        inv_constraint_col = "deployed_qty_invCon" if "deployed_qty_invCon" in frame else "deployed_qty"
        deployed_qty_col = "deployed_qty" if "deployed_qty" in frame else inv_constraint_col
        if not frame.empty and inv_constraint_col in frame:
            frame = frame.copy()
            frame["deployed_qty_invCon"] = pd.to_numeric(frame[inv_constraint_col], errors="coerce").fillna(0)
            frame["deployed_qty"] = pd.to_numeric(frame[deployed_qty_col], errors="coerce").fillna(0)
            frame = frame.loc[(frame["deployed_qty_invCon"] > 0) & (frame["sending"] != frame["receiving"])].copy()
            frame = frame.rename(columns={"date": "planned_deployment_date"})
            frame = frame.loc[:, [
                "material", "sending", "receiving", "planned_deployment_date", "deployed_qty_invCon", "deployed_qty", "demand_element",
            ]]
        history[day.strftime("%Y-%m-%d")] = frame
    if all(frame.empty for frame in history.values()):
        pytest.skip("历史 M5 五日中没有可执行的跨节点调拨")
    return history


def _new_context(config: dict[str, pd.DataFrame], name: str) -> tuple[Orch, StateContext]:
    orch = Orch(START_DATE, END_DATE, config_dict={key: value.copy() for key, value in config.items()},
                output_path=str(REPORT_DIR / name), engine="pandas", skip_dq=True,
                enable_persistence=False)
    context = StateContext(START_DATE, orch=orch)
    context.initialize(orch.all_config)
    return orch, context


def _state_snapshot(context: StateContext, day: str) -> dict[str, pd.DataFrame]:
    return {
        "open_deployment": context.get_open_deployment_view(day),
        "planning_intransit": context.get_planning_intransit_view(day),
        "unrestricted_inventory": context.get_unrestricted_inventory_view(day),
        "delivery_shipment_log": context.get_delivery_shipment_log_view(day),
    }


def _assert_same(left: pd.DataFrame, right: pd.DataFrame, name: str) -> None:
    left, right = left.copy(), right.copy()
    common = sorted(set(left.columns) & set(right.columns))
    left, right = left.reindex(columns=common), right.reindex(columns=common)
    sort_columns = [column for column in common if column not in {"WFR", "VFR", "truck_load_pct"}]
    if sort_columns and not left.empty:
        left = left.sort_values(sort_columns, kind="mergesort", na_position="last")
    if sort_columns and not right.empty:
        right = right.sort_values(sort_columns, kind="mergesort", na_position="last")
    pdt.assert_frame_equal(left.reset_index(drop=True), right.reset_index(drop=True), check_dtype=False, check_like=False, obj=name)


@pytest.fixture(scope="module")
def real_replay():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    db = _db()
    try:
        db.connect()
        config, plans = _load_config(db), _load_plans(db)
    finally:
        db.close()

    legacy_orch, legacy_context = _new_context(config, "legacy")
    pandas_orch, pandas_context = _new_context(config, "pandas")
    polars_orch, polars_context = _new_context(config, "polars")
    module = ModuleSix(START_DATE, state_context=pandas_context, orch=pandas_orch, random_seed=42)
    polars_module = ModuleSix(START_DATE, state_context=polars_context, orch=polars_orch, random_seed=42)
    module.prepare()
    polars_module.prepare()
    days: list[dict] = []
    for day in pd.date_range(START_DATE, END_DATE, freq="D"):
        text = day.strftime("%Y-%m-%d")
        for context in (legacy_context, pandas_context, polars_context):
            context.day_start(text)
            context.apply_deployment(plans[text], text)
        started = time.perf_counter()
        legacy = run_daily_physical_flow(config, legacy_context, day, str(REPORT_DIR), random_seed=42, skip_file_output=True)
        legacy_seconds = time.perf_counter() - started
        module.simulation_date = day
        started = time.perf_counter()
        module.run()
        pandas = module.output()
        pandas_seconds = time.perf_counter() - started
        polars_module.simulation_date = day
        started = time.perf_counter()
        polars_module.run()
        polars = polars_module.output()
        polars_seconds = time.perf_counter() - started
        for key in M6_FRAME_KEYS:
            _assert_same(legacy[key], pandas[key], f"{text}:{key}")
            _assert_same(pandas[key], polars[key], f"{text}:pandas_vs_polars:{key}")
        legacy_context.apply_delivery(legacy["delivery_plan"], text)
        pandas_context.apply_delivery(pandas["delivery_plan"], text)
        polars_context.apply_delivery(polars["delivery_plan"], text)
        legacy_context.day_end(text)
        pandas_context.day_end(text)
        polars_context.day_end(text)
        legacy_state, pandas_state = _state_snapshot(legacy_context, text), _state_snapshot(pandas_context, text)
        polars_state = _state_snapshot(polars_context, text)
        for key in legacy_state:
            _assert_same(legacy_state[key], pandas_state[key], f"{text}:state:{key}")
            _assert_same(pandas_state[key], polars_state[key], f"{text}:pandas_vs_polars:state:{key}")
        days.append({"date": text, "legacy_seconds": legacy_seconds, "pandas_seconds": pandas_seconds, "polars_seconds": polars_seconds,
                     "delivery_rows": len(pandas["delivery_plan"]), "vehicle_rows": len(pandas["vehicle_log"])})
    totals = {
        engine: sum(row[f"{engine}_seconds"] for row in days)
        for engine in ("legacy", "pandas", "polars")
    }
    summary = {
        "timed_scope": TIMED_SCOPE,
        "total_seconds": totals,
        "average_seconds_per_day": {engine: total / len(days) for engine, total in totals.items()},
        "relative_to_legacy": {
            engine: totals[engine] / totals["legacy"] if totals["legacy"] else None
            for engine in ("pandas", "polars")
        },
    }
    REPORT_PATH.write_text(json.dumps({
        "config_name": CONFIG_NAME, "historical_run_id": HISTORICAL_RUN_ID,
        "start_date": START_DATE, "end_date": END_DATE, "strict_parity": True,
        "performance": summary, "days": days,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return days


def test_m6_legacy_pandas_polars_replay_five_real_business_days(real_replay):
    assert len(real_replay) == 5
    assert REPORT_PATH.exists()


def test_m6_real_replay_records_comparable_execution_timing(real_replay):
    assert all(day["legacy_seconds"] >= 0 and day["pandas_seconds"] >= 0 and day["polars_seconds"] >= 0 for day in real_replay)
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert report["performance"]["timed_scope"] == TIMED_SCOPE
    assert all(value > 0 for value in report["performance"]["total_seconds"].values())
