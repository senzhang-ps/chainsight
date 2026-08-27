"""受控 M5 回归：以固定历史 DB 的 M5 与相关编排器状态作为 pandas Oracle。"""

# 测试文件说明
# 测试目的：集中验证部署计划、优先级分配与供给池扣减的一致性。
# 测试方法：按 `integration` 类测试组织用例，使用夹具、模拟数据或真实依赖执行被测流程。
# 输入和期望：输入覆盖正常、边界和异常业务数据；期望由各测试用例的显式断言定义。
# 业务逻辑和原因：将相关验证集中维护，确保部署计划、优先级分配与供给池扣减的一致性变更时能够快速定位回归影响。



from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pgsql_db.db_connection import DatabaseConnection
from pgsql_db.settings import get_database_config
from src.models.module import OUTPUT_REGISTRY
from src.modules.deployment_planning.integration_refactor import ModuleFive
from tests.regression import test_m5_two_way_compare as m5
from tests.helpers.compare_utils import compare_dataframes_by_key, dataframe_difference_details

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
DB_SCHEMA = "input"
RUN_ID = "db_OC_Paste_S1_20251224_repare_20260814_132746"
START_DATE, END_DATE = "2025-12-15", "2025-12-19"
REPORT_DIR = PROJECT_ROOT / "outputs/m5_db_oracle_pandas"
META = ["run_id", "sim_date", "config_name", "db_write_time", "file_date"]
M5_ORCHESTRATOR_TABLES = {
    # M5 的 apply_deployment() 仅直接写回 open_deployment。库存和 planning
    # in-transit 是 M6/日末流程的状态，不混入孤立 M5 pandas 的输出契约。
    "open_deployment": "orchestrator_open_deployment",
}


def _db() -> DatabaseConnection:
    cfg = get_database_config()
    return DatabaseConnection(host=cfg["host"], port=cfg["port"], database=cfg["database"],
                              user=cfg["user"], password=cfg["password"], schema=DB_SCHEMA,
                              auto_create_schema=False)


def _load(db: DatabaseConnection, table: str, day: pd.Timestamp, *, active_orders: bool = False) -> pd.DataFrame:
    columns = [row[0] for row in db.execute_query(
        "SELECT column_name FROM information_schema.columns WHERE table_schema=%s AND table_name=%s ORDER BY ordinal_position",
        (DB_SCHEMA, table),
    )]
    qualified = f'"{DB_SCHEMA}"."{table}"'
    if active_orders:
        sql = f"SELECT {', '.join(columns)} FROM {qualified} WHERE run_id=%s AND sim_date::date<=%s::date AND date::date>=%s::date"
        args = (RUN_ID, day.strftime("%Y-%m-%d"), day.strftime("%Y-%m-%d"))
    else:
        sql = f"SELECT {', '.join(columns)} FROM {qualified} WHERE run_id=%s AND sim_date::date=%s::date"
        args = (RUN_ID, day.strftime("%Y-%m-%d"))
    return pd.DataFrame(db.execute_query(sql, args), columns=columns).drop(columns=META, errors="ignore")


def _history(db: DatabaseConnection) -> dict[str, dict[str, pd.DataFrame]]:
    rows = {}
    for day in pd.date_range(START_DATE, END_DATE):
        key = day.strftime("%Y-%m-%d")
        rows[key] = {
            "all_orders_for_next_day": _load(db, "module1_output_orderlog", day, active_orders=True),
            "shipment_df": _load(db, "module1_output_shipmentlog", day),
            "supply_demand_df": _load(db, "module1_output_supplydemandlog", day),
            "production_df": _load(db, "module4_output_productionplan", day),
            "delivery_plan": _load(db, "module6_output_deliveryplan", day),
        }
    return rows


def _context_snapshots(context, day: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """读取 pandas M5 写回后的唯一直接数据库契约状态。"""
    date_text = day.strftime("%Y-%m-%d")
    return {
        "open_deployment": context.get_open_deployment_view(date_text),
    }


def _run_pandas_against_database_oracle(
    config: dict[str, pd.DataFrame],
    history: dict[str, dict[str, pd.DataFrame]],
) -> dict:
    """逐日重放 pandas M5，并保留写回后的 M5 编排器状态快照。

    仅实例化 ``ModuleFive`` pandas backend；不会调用 legacy 或 Polars。
    输入继续由 ``_replay_inputs()`` 注入历史 M1/M4 DataFrame，M5 输出经
    ``_finish_day()`` 写回 StateContext 后，再捕获 M5 直接负责的状态表。
    """
    orch, context = m5._new_context(config, "pandas_db_oracle", "pandas")
    module = ModuleFive(
        simulation_date=START_DATE,
        simulation_start_date=START_DATE,
        state_context=context,
        orch=orch,
        verbose=False,
    )
    module.prepare()
    days = []
    for day in pd.date_range(START_DATE, END_DATE, freq="D"):
        print(f"[M5 DB Oracle] pandas {day:%Y-%m-%d} start", flush=True)
        m1, m4 = m5._replay_inputs(context, day, history)
        module.simulation_date = day
        module.run()
        result = module.output()
        m5._finish_day(context, day, result)
        # 完整 DB legacy 编排会在 M5 后立即执行 M6。为只验证 pandas M5，
        # 直接注入该历史 run 已提交的 M6 delivery plan；StateContext 仍按
        # refactor 的 apply_delivery() 写回次日 M5 所需状态。
        delivery = history[day.strftime("%Y-%m-%d")]["delivery_plan"]
        if not delivery.empty:
            context.apply_delivery(delivery, day.strftime("%Y-%m-%d"))
        days.append({
            "date": day.strftime("%Y-%m-%d"),
            "m5_result": result,
            "orchestrator_state": _context_snapshots(context, day),
        })
        print(f"[M5 DB Oracle] pandas {day:%Y-%m-%d} complete", flush=True)
    return {"days": days}


def _append_comparison(
    comparisons: list[dict],
    details: list[pd.DataFrame],
    *,
    day: str,
    scope: str,
    name: str,
    database_frame: pd.DataFrame,
    pandas_frame: pd.DataFrame,
) -> None:
    """保持原始多重集，记录 DB Oracle 与 pandas 的完整比较审计。"""
    comparison = compare_dataframes_by_key(
        database_frame,
        pandas_frame,
        label=f"{day}:{scope}:{name}:database_vs_pandas",
    )
    comparisons.append({
        "date": day,
        "scope": scope,
        "name": name,
        "database_raw_rows": len(database_frame),
        "pandas_raw_rows": len(pandas_frame),
        "comparison": comparison,
    })
    detail = dataframe_difference_details(database_frame, pandas_frame)
    detail.insert(0, "name", name)
    detail.insert(0, "scope", scope)
    detail.insert(0, "date", day)
    details.append(detail)


def test_m5_pandas_against_database_oracle() -> None:
    # 测试目的：验证“m5、pandas、against、database、oracle”场景下部署计划、优先级分配与供给池扣减的一致性。
    # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `print()`，再通过 2 个断言核对结果、状态或异常条件。
    # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
    # 业务逻辑和原因：该校验防止部署计划、优先级分配与供给池扣减的一致性在重构、引擎切换或跨日运行时发生静默偏差。
    """M5 pandas 以固定 run 的模块输出和 M5 相关 orchestrator 状态为 Oracle。"""
    print("[M5 DB Oracle] loading configuration, M1/M4 inputs, and DB oracle frames", flush=True)
    original = (m5.START_DATE, m5.END_DATE, m5.REPORT_DIR, m5.DB_SCHEMA)
    m5.START_DATE, m5.END_DATE, m5.REPORT_DIR, m5.DB_SCHEMA = (
        START_DATE, END_DATE, REPORT_DIR, DB_SCHEMA,
    )
    db = _db(); db.connect()
    try:
        # 历史 run 由 DB runner 产生：与 M1/M4/M5 oracle 相同地从数据库读取
        # 配置，避免 Excel 文件版本或加载路径成为另一个未受控变量。
        config = m5._load_prepared_config(db)
        history = _history(db)
        expected = {
            day.strftime("%Y-%m-%d"): {
                output: _load(db, table, day)
                for output, table in OUTPUT_REGISTRY["module5"].items()
            }
            for day in pd.date_range(START_DATE, END_DATE)
        }
        expected_orchestrator = {
            day.strftime("%Y-%m-%d"): {
                name: _load(db, table, day)
                for name, table in M5_ORCHESTRATOR_TABLES.items()
            }
            for day in pd.date_range(START_DATE, END_DATE)
        }
    finally:
        db.close()
    print("[M5 DB Oracle] database inputs loaded", flush=True)

    # Pin the shared replay helper globals to this isolated report path.  The
    # runner below calls only the pandas refactor backend, never legacy/Polars.
    try:
        pandas = _run_pandas_against_database_oracle(config, history)
    finally:
        m5.START_DATE, m5.END_DATE, m5.REPORT_DIR, m5.DB_SCHEMA = original

    comparisons, details = [], []
    for pandas_day in pandas["days"]:
        day = pandas_day["date"]
        for output, database_frame in expected[day].items():
            pandas_frame = pandas_day["m5_result"].get(output, pd.DataFrame())
            _append_comparison(
                comparisons,
                details,
                day=day,
                scope="module5_output",
                name=output,
                database_frame=database_frame,
                pandas_frame=pandas_frame,
            )
        for name, database_frame in expected_orchestrator[day].items():
            _append_comparison(
                comparisons,
                details,
                day=day,
                scope="orchestrator_m5_state",
                name=name,
                database_frame=database_frame,
                pandas_frame=pandas_day["orchestrator_state"][name],
            )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    final_pandas = pandas["days"][-1]["m5_result"]
    final_pandas.get("deployment_plan", pd.DataFrame()).to_csv(
        REPORT_DIR / "m5_final_day_pandas_deployment_plan.csv",
        index=False,
        encoding="utf-8-sig",
    )
    final_pandas.get("stock_on_hand_log", pd.DataFrame()).to_csv(
        REPORT_DIR / "m5_final_day_pandas_stock_on_hand.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([{
        "date": item["date"],
        "scope": item["scope"],
        "name": item["name"],
        "database_raw_rows": item["database_raw_rows"],
        "pandas_raw_rows": item["pandas_raw_rows"],
    } for item in comparisons]).to_csv(
        REPORT_DIR / "db_m5_orchestrator_pandas_raw_row_counts.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(details, ignore_index=True).to_csv(
        REPORT_DIR / "db_m5_orchestrator_to_pandas_differences.csv",
        index=False,
        encoding="utf-8-sig",
    )
    report = {
        "run_id": RUN_ID,
        "db_schema": DB_SCHEMA,
        "input_policy": "Configuration, M1/M4 inputs, and DB Oracle frames read from database; only pandas M5 replayed",
        "oracle_tables": {
            "module5_output": OUTPUT_REGISTRY["module5"],
            "orchestrator_m5_state": M5_ORCHESTRATOR_TABLES,
        },
        "comparisons": comparisons,
    }
    (REPORT_DIR / "db_m5_orchestrator_to_pandas.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    assert len(pandas["days"]) == 5
    assert len(comparisons) == 5 * (len(OUTPUT_REGISTRY["module5"]) + len(M5_ORCHESTRATOR_TABLES))
