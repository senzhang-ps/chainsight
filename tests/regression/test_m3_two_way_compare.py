"""真实数据库数据下 legacy / refactor pandas / refactor Polars M3 严格对比。"""

# 测试文件说明
# 测试目的：集中验证净需求计算及跨日计划事实传递的一致性。
# 测试方法：按 `regression` 类测试组织用例，使用夹具、模拟数据或真实依赖执行被测流程。
# 输入和期望：输入覆盖正常、边界和异常业务数据；期望由各测试用例的显式断言定义。
# 业务逻辑和原因：将相关验证集中维护，确保净需求计算及跨日计划事实传递的一致性变更时能够快速定位回归影响。



from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

from pgsql_db.db_connection import DatabaseConnection
from pgsql_db.settings import get_database_config
from src.core.main_integration.config_loader import load_configuration_from_dict, prepare_configuration
from src.core.orchestrator import Orch
from src.core.run.db_config import _load_config_from_database
from src.modules.deployment_planning.integration_refactor import ModuleFive
from src.modules.mrp_planning.integration import _calculate_net_demand, _load_static_configs
from src.modules.mrp_planning.integration_refactor import ModuleThree
from src.modules.state_context import StateContext
from tests.helpers.compare_utils import compare_dataframes_by_key, dataframe_difference_details


PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
CONFIG_NAME = "OC_Paste_S1_20251224_repare"
HISTORICAL_RUN_ID = "db_OC_Paste_S1_20251224_repare_20260803_105300"
DB_SCHEMA = "public"
START_DATE = "2025-12-15"
END_DATE = "2025-12-19"
REPORT_DIR = PROJECT_ROOT / "outputs" / "m3_two_way_compare"
M1_TABLES = {
    "all_orders_for_next_day": "module1_output_orderlog",
    "shipment_df": "module1_output_shipmentlog",
    "supply_demand_df": "module1_output_supplydemandlog",
}
M4_PRODUCTION_TABLE = "module4_output_productionplan"
M5_PLAN_TABLE = "module5_output_deploymentplan"
M6_DELIVERY_TABLE = "module6_output_deliveryplan"
M3_KEYS = ["material", "location", "requirement_date", "demand_element", "layer"]


def _progress(message: str) -> None:
    try:
        sys.__stdout__.write(f"{time.strftime('%H:%M:%S')} [M3 compare] {message}\n")
        sys.__stdout__.flush()
    except OSError:
        pass


def _db() -> DatabaseConnection:
    config = get_database_config()
    return DatabaseConnection(
        host=config["host"], port=config["port"], database=config["database"],
        user=config["user"], password=config["password"],
        schema=config.get("default_schema", "public"), auto_create_schema=False,
    )


def _table_columns(db: DatabaseConnection, table: str) -> list[str]:
    rows = db.execute_query(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position",
        (DB_SCHEMA, table),
    )
    return [row[0] for row in rows]


def _load_daily(db: DatabaseConnection, table: str, day: pd.Timestamp) -> pd.DataFrame:
    columns = _table_columns(db, table)
    if not columns:
        pytest.skip(f"找不到历史表 {DB_SCHEMA}.{table}")
    qualified = f'"{DB_SCHEMA}"."{table}"'
    rows = db.execute_query(
        f"SELECT {', '.join(columns)} FROM {qualified} WHERE run_id = %s AND sim_date::date = %s::date",
        (HISTORICAL_RUN_ID, day.strftime("%Y-%m-%d")),
    )
    return pd.DataFrame(rows, columns=columns).drop(
        columns=["run_id", "sim_date", "config_name", "db_write_time"], errors="ignore"
    )


def _load_inputs(db: DatabaseConnection) -> dict[str, dict[str, pd.DataFrame]]:
    result = {}
    for day in pd.date_range(START_DATE, END_DATE, freq="D"):
        key = day.strftime("%Y-%m-%d")
        result[key] = {name: _load_daily(db, table, day) for name, table in M1_TABLES.items()}
        result[key]["production"] = _load_daily(db, M4_PRODUCTION_TABLE, day)
        result[key]["deployment"] = _load_daily(db, M5_PLAN_TABLE, day)
        result[key]["delivery"] = _load_daily(db, M6_DELIVERY_TABLE, day)
    return result


def _prepared_config(db: DatabaseConnection) -> dict[str, pd.DataFrame]:
    raw = _load_config_from_database(db, CONFIG_NAME)
    if not raw:
        pytest.skip(f"找不到配置 {CONFIG_NAME}")
    return prepare_configuration(load_configuration_from_dict(raw, CONFIG_NAME))


def _new_context(config: dict[str, pd.DataFrame], engine: str) -> tuple[Orch, StateContext]:
    orch = Orch(
        start_date=START_DATE, end_date=END_DATE,
        config_dict={name: value.copy() for name, value in config.items()},
        output_path=str(REPORT_DIR / f"scratch_{engine}"), engine=engine, skip_dq=True,
        enable_persistence=False,
    )
    ctx = StateContext(START_DATE, orch=orch)
    ctx.initialize(orch.all_config)
    return orch, ctx


def _prepare_day(ctx: StateContext, day: pd.Timestamp, history: dict[str, pd.DataFrame]) -> None:
    text = day.strftime("%Y-%m-%d")
    ctx.day_start(text)
    if not history["shipment_df"].empty:
        ctx.apply_shipments(history["shipment_df"], text)
    ctx.apply_deployment_demand_inputs(
        history["supply_demand_df"], history["all_orders_for_next_day"], text,
    )
    if not history["production"].empty:
        ctx.apply_production(history["production"], text)
    if not history["deployment"].empty:
        deployment = history["deployment"].copy()
        if "deployed_qty" not in deployment and "deployed_qty_invCon" in deployment:
            deployment["deployed_qty"] = deployment["deployed_qty_invCon"]
        ctx.apply_deployment(deployment, text)
    if not history["delivery"].empty:
        ctx.apply_delivery(history["delivery"], text)


def _state_size(ctx: StateContext) -> dict[str, int]:
    """返回会影响 M5 账本/需求计算的跨日状态规模，用于性能归因。"""
    return {
        "open_deployment": len(ctx.open_deployment),
        "in_transit": len(ctx.in_transit),
        "production_backlog": len(ctx.production_plan_backlog),
        "inventory_nodes": len(ctx.unrestricted_inventory),
    }


def _plan_size(profile: list[dict] | None) -> dict[str, int]:
    """汇总 M5 各层的输入/缺口行数，避免只看到总耗时而无法定位增长来源。"""
    profile = profile or []
    return {
        "direct_rows": sum(int(item.get("direct_rows", 0)) for item in profile),
        "gap_rows": sum(int(item.get("gap_rows", 0)) for item in profile),
        "demand_rows": sum(int(item.get("demand_rows", 0)) for item in profile),
        "shortage_rows": sum(int(item.get("shortage_rows", 0)) for item in profile),
    }


def _legacy_day(configs: dict, ctx: StateContext, day: pd.Timestamp, history: dict[str, pd.DataFrame]) -> pd.DataFrame:
    supply = history["supply_demand_df"]
    orders = history["all_orders_for_next_day"]
    shipment = history["shipment_df"]
    text = day.strftime("%Y-%m-%d")
    return _calculate_net_demand(
        day,
        {"supply_demand_df": supply, "order_df": orders, "shipment_df": shipment},
        {
            "beginning_inventory_df": ctx.get_beginning_inventory_view(text),
            "in_transit_df": ctx.get_planning_intransit_view(text),
            "delivery_gr_df": ctx.get_delivery_gr_view(text),
            "all_production_df": ctx.get_all_production_view(text),
            "open_deployment_df": ctx.get_open_deployment_view(text),
            "delivery_shipment_df": ctx.get_delivery_shipment_log_view(text),
        },
        configs,
    )


def _run_refactor(config: dict[str, pd.DataFrame], history: dict[str, dict[str, pd.DataFrame]], engine: str) -> dict:
    orch, ctx = _new_context(config, engine)
    m5 = ModuleFive(START_DATE, START_DATE, state_context=ctx, orch=orch)
    m3 = ModuleThree(START_DATE, START_DATE, state_context=ctx, orch=orch)
    m5.prepare()
    m3.prepare()
    days, m5_seconds, m3_seconds, total_seconds = [], [], [], []
    for day in pd.date_range(START_DATE, END_DATE, freq="D"):
        text = day.strftime("%Y-%m-%d")
        _progress(f"{engine} {text}: 准备日状态")
        _prepare_day(ctx, day, history[text])
        state_size = _state_size(ctx)
        m5.simulation_date = day
        m3.simulation_date = day
        started = time.perf_counter()
        m5.run()
        m5_elapsed = time.perf_counter() - started
        m3_started = time.perf_counter()
        m3.run()
        m3_elapsed = time.perf_counter() - m3_started
        elapsed = time.perf_counter() - started
        m5_seconds.append(m5_elapsed)
        m3_seconds.append(m3_elapsed)
        total_seconds.append(elapsed)
        plan_size = _plan_size(getattr(m5._backend, "last_plan_layer_profile", None))
        _progress(
            f"{engine} {text}: M5 共享事实 {m5_elapsed:.2f}s，"
            f"M3 增量 {m3_elapsed:.2f}s，总计 {elapsed:.2f}s；"
            f"状态 open={state_size['open_deployment']:,}/backlog={state_size['production_backlog']:,}，"
            f"M5 demand={plan_size['demand_rows']:,}"
        )
        days.append({
            "date": text,
            "result": m3.output()["net_demand_df"].copy(),
            "elapsed_seconds": elapsed,
            "state_size_before_m5": state_size,
            "m5_plan_size": plan_size,
        })
    return {
        "days": days,
        "seconds": m3_seconds,
        "total_seconds": sum(m3_seconds),
        "m5_shared_fact_seconds": m5_seconds,
        "m5_m3_total_seconds": total_seconds,
        "m5_m3_total": sum(total_seconds),
    }


def _run_legacy(config: dict[str, pd.DataFrame], history: dict[str, dict[str, pd.DataFrame]]) -> dict:
    _, ctx = _new_context(config, "pandas")
    configs = _load_static_configs(config, skip_normalize=True)
    days, seconds = [], []
    for day in pd.date_range(START_DATE, END_DATE, freq="D"):
        text = day.strftime("%Y-%m-%d")
        _progress(f"legacy {text}: 准备日状态")
        entries = history[text]
        _prepare_day(ctx, day, entries)
        started = time.perf_counter()
        result = _legacy_day(configs, ctx, day, entries)
        elapsed = time.perf_counter() - started
        seconds.append(elapsed)
        _progress(f"legacy {text}: M3 完成 {elapsed:.2f}s，输出 {len(result):,} 行")
        days.append({"date": text, "result": result, "elapsed_seconds": elapsed})
    return {"days": days, "seconds": seconds, "total_seconds": sum(seconds)}


def _compare(left: pd.DataFrame, right: pd.DataFrame, label: str) -> dict:
    return compare_dataframes_by_key(left, right, key_columns=M3_KEYS, label=label)


@pytest.fixture(scope="module")
def comparison_data():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    db = _db()
    try:
        db.connect()
        config = _prepared_config(db)
        history = _load_inputs(db)
    finally:
        db.close()
    _progress("开始 legacy M3 五日运行")
    legacy = _run_legacy(config, history)
    _progress("开始 refactor pandas M3 五日运行")
    pandas_run = _run_refactor(config, history, "pandas")
    _progress("开始 refactor Polars M3 五日运行")
    polars_run = _run_refactor(config, history, "polars")
    comparisons = []
    details = []
    for legacy_day, pandas_day, polars_day in zip(legacy["days"], pandas_run["days"], polars_run["days"]):
        date = legacy_day["date"]
        for name, left, right in (
            ("legacy_vs_pandas", legacy_day["result"], pandas_day["result"]),
            ("pandas_vs_polars", pandas_day["result"], polars_day["result"]),
        ):
            comparison = _compare(left, right, f"{date}:{name}")
            comparisons.append({"date": date, "comparison": name, **comparison})
            detail = dataframe_difference_details(left, right, key_columns=M3_KEYS)
            if not detail.empty:
                detail.insert(0, "date", date)
                detail.insert(1, "comparison", name)
                details.append(detail)
    detail_path = REPORT_DIR / "m3_differences.csv"
    pd.concat(details, ignore_index=True).to_csv(detail_path, index=False) if details else pd.DataFrame().to_csv(detail_path, index=False)
    report = {
        "config_name": CONFIG_NAME, "historical_run_id": HISTORICAL_RUN_ID,
        "date_range": [START_DATE, END_DATE],
        "performance": {
            name: {
                "m3_total_seconds": run["total_seconds"],
                "m3_per_day_seconds": run["seconds"],
                **({
                    "m5_shared_fact_per_day_seconds": run["m5_shared_fact_seconds"],
                    "m5_m3_total_seconds": run["m5_m3_total"],
                    "m5_m3_total_per_day_seconds": run["m5_m3_total_seconds"],
                    "daily_m5_state_size": [day["state_size_before_m5"] for day in run["days"]],
                    "daily_m5_plan_size": [day["m5_plan_size"] for day in run["days"]],
                } if "m5_shared_fact_seconds" in run else {}),
            }
            for name, run in (("legacy", legacy), ("pandas", pandas_run), ("polars", polars_run))
        },
        "comparisons": comparisons,
        "difference_csv": detail_path.name,
    }
    report_path = REPORT_DIR / "m3_three_way_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {"legacy": legacy, "pandas": pandas_run, "polars": polars_run, "report": report, "report_path": report_path}


def test_m3_real_data_three_way_strict_parity(comparison_data):
    # 测试目的：验证“m3、real、data、three、way、strict、parity”场景下净需求计算及跨日计划事实传递的一致性。
    # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `exists()`，再通过 6 个断言核对结果、状态或异常条件。
    # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
    # 业务逻辑和原因：该校验防止净需求计算及跨日计划事实传递的一致性在重构、引擎切换或跨日运行时发生静默偏差。
    assert comparison_data["report_path"].exists()
    assert len(comparison_data["report"]["comparisons"]) == 10
    for item in comparison_data["report"]["comparisons"]:
        assert item["left_only_keys"] == 0, item
        assert item["right_only_keys"] == 0, item
        assert not item["column_differences"], item
        assert not item["precision_differences"], item


def test_m3_real_data_performance_report(comparison_data):
    # 测试目的：验证“m3、real、data、performance、report”场景下净需求计算及跨日计划事实传递的一致性。
    # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `all()`，再通过 2 个断言核对结果、状态或异常条件。
    # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
    # 业务逻辑和原因：该校验防止净需求计算及跨日计划事实传递的一致性在重构、引擎切换或跨日运行时发生静默偏差。
    performance = comparison_data["report"]["performance"]
    assert set(performance) == {"legacy", "pandas", "polars"}
    assert all(len(value["m3_per_day_seconds"]) == 5 for value in performance.values())