"""真实历史数据下 Decoupling M1 pandas / polars 五日一致性与性能对比。

历史业务基线固定从 PostgreSQL ``public`` schema 的指定 ``run_id`` 读取；
当前重构 pandas 与 polars 后端各自在独立内存状态上回放。主项目 legacy M1
已与重构逻辑分叉，不参与测试。

运行：
    conda run -n work pytest tests/regression/test_m1_two_way_compare.py -s -q
"""

# 测试文件说明
# 测试目的：集中验证订单生成、库存消耗与发运数据的一致性。
# 测试方法：按 `regression` 类测试组织用例，使用夹具、模拟数据或真实依赖执行被测流程。
# 输入和期望：输入覆盖正常、边界和异常业务数据；期望由各测试用例的显式断言定义。
# 业务逻辑和原因：将相关验证集中维护，确保订单生成、库存消耗与发运数据的一致性变更时能够快速定位回归影响。



from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

from pgsql_db.db_connection import DatabaseConnection
from pgsql_db.settings import get_database_config
from src.models.module import OUTPUT_REGISTRY
from src.core.main_integration.config_loader import (
    load_configuration_from_dict,
    prepare_configuration,
)
from src.core.orchestrator import Orch
from src.core.run.db_config import _load_config_from_database
from src.modules.demand_planning.integration_refactor import ModuleOne
from src.modules.state_context import StateContext
from tests.helpers.compare_utils import compare_dataframes_by_key


PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
CONFIG_NAME = os.getenv("M1_COMPARE_CONFIG_NAME", "OC_Paste_S1_20251224_repare")
START_DATE = os.getenv("M1_COMPARE_START_DATE", "2025-12-15")
END_DATE = os.getenv("M1_COMPARE_END_DATE", "2025-12-19")
SIMULATION_END_DATE = os.getenv("M1_COMPARE_SIMULATION_END_DATE", END_DATE)
HISTORICAL_RUN_ID = os.getenv(
    "M1_COMPARE_HISTORICAL_RUN_ID",
    "db_OC_Paste_S1_20251224_repare_20260803_105300",
)
DB_SCHEMA = os.getenv("M1_COMPARE_DB_SCHEMA", "public")
CONFIG_SCHEMA = os.getenv("M1_COMPARE_CONFIG_SCHEMA", "public")
REPORT_DIR = PROJECT_ROOT / "outputs" / "m1_two_way_compare"
PROGRESS_PATH = REPORT_DIR / "m1_two_way_compare.progress.log"
M1_FRAME_KEYS = (
    "orders_df",
    "shipment_df",
    "cut_df",
    "supply_demand_df",
    "summary_df",
)
M1_BUSINESS_COLUMNS = {
    "orders_df": ["simulation_date", "date", "material", "location", "demand_type", "advance_days", "quantity"],
    "shipment_df": ["date", "material", "location", "demand_type", "quantity"],
    "cut_df": ["date", "material", "location", "quantity"],
    "supply_demand_df": ["date", "material", "location", "demand_element", "quantity"],
    "summary_df": ["Total_Orders", "Total_Shipments", "Total_Cuts", "Total_SupplyDemand"],
}


def _progress(message: str) -> None:
    """绕过项目 logger 对 stdout 的重定向，持续输出长任务进度。"""
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [M1 compare] {message}\n"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with PROGRESS_PATH.open("a", encoding="utf-8") as progress_file:
        progress_file.write(line)
    sys.__stdout__.write(line)
    sys.__stdout__.flush()


def _db(schema: str = DB_SCHEMA) -> DatabaseConnection:
    config = get_database_config()
    return DatabaseConnection(
        host=config["host"],
        port=config["port"],
        database=config["database"],
        user=config["user"],
        password=config["password"],
        schema=schema,
        auto_create_schema=False,
    )


def _load_prepared_config(db: DatabaseConnection) -> dict[str, pd.DataFrame]:
    raw = _load_config_from_database(db, CONFIG_NAME)
    if not raw:
        pytest.skip(f"数据库中找不到配置: {CONFIG_NAME}")
    return prepare_configuration(load_configuration_from_dict(raw, CONFIG_NAME))


def _load_historical_daily_output(
    db: DatabaseConnection, table_name: str, day: pd.Timestamp,
) -> pd.DataFrame:
    """从指定 run 只读加载一张 M1 输出表。"""
    columns = [row[0] for row in db.execute_query(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position",
        (DB_SCHEMA, table_name),
    )]
    if not columns:
        return pd.DataFrame()
    if table_name == OUTPUT_REGISTRY["module1"]["orders_df"]:
        # M1 当前日的有效订单包含此前创建、但尚未到期的订单；必须镜像
        # merge_with_history(): simulation_date <= today AND date >= today。
        rows = db.execute_query(
            f'SELECT {", ".join(columns)} FROM "{DB_SCHEMA}"."{table_name}" '
            "WHERE run_id = %s AND sim_date::date <= %s::date AND date::date >= %s::date",
            (HISTORICAL_RUN_ID, day.strftime("%Y-%m-%d"), day.strftime("%Y-%m-%d")),
        )
    else:
        rows = db.execute_query(
            f'SELECT {", ".join(columns)} FROM "{DB_SCHEMA}"."{table_name}" '
            "WHERE run_id = %s AND sim_date::date = %s::date",
            (HISTORICAL_RUN_ID, day.strftime("%Y-%m-%d")),
        )
    return pd.DataFrame(rows, columns=columns).drop(
        columns=["run_id", "sim_date", "config_name", "db_write_time", "file_date"],
        errors="ignore",
    )


def _load_historical_m1(db: DatabaseConnection) -> dict:
    """按日加载固定历史 run 的 M1 原始业务输出。"""
    return {"days": [{
        "date": day.strftime("%Y-%m-%d"),
        "m1_result": {
            output_key: _load_historical_daily_output(db, table_name, day)
            for output_key, table_name in OUTPUT_REGISTRY["module1"].items()
        },
    } for day in pd.date_range(START_DATE, END_DATE, freq="D")]}


def _new_context(config: dict[str, pd.DataFrame], *, engine: str) -> tuple[Orch, StateContext]:
    """为指定重构后端创建独立 Orch / StateContext，避免共享可变库存。"""
    orch = Orch(
        start_date=START_DATE,
        end_date=SIMULATION_END_DATE,
        config_dict={name: frame.copy() for name, frame in config.items()},
        output_path=str(REPORT_DIR / f"refactor_{engine}_scratch"),
        engine=engine,
        skip_dq=True,
        enable_persistence=False,
    )
    context = StateContext(simulation_date=START_DATE, orch=orch)
    context.initialize(orch.all_config)
    return orch, context


def _run_refactor(config: dict[str, pd.DataFrame], *, engine: str) -> dict:
    """在独立 StateContext 上运行指定 M1 后端；准备阶段不计入计算耗时。"""
    orch, context = _new_context(config, engine=engine)
    module = ModuleOne(
        simulation_date=pd.Timestamp(START_DATE),
        orchestrator=context,
        orch=orch,
    )
    module.prepare()

    days: list[dict] = []
    elapsed: list[float] = []
    for day in pd.date_range(START_DATE, END_DATE, freq="D"):
        day_key = day.strftime("%Y-%m-%d")
        _progress(f"refactor {engine} 计时: {day_key} 开始")
        context.day_start(day_key)
        module.simulation_date = day
        tick = time.perf_counter()
        module.run()
        result = module.output()
        seconds = time.perf_counter() - tick
        shipments = result.get("shipment_df", pd.DataFrame())
        if not shipments.empty:
            context.apply_shipments(shipments, day_key)
        context.day_end(day_key)
        elapsed.append(seconds)
        days.append({"date": day_key, "m1_result": result, "elapsed_seconds": seconds})
        _progress(f"refactor {engine} 计时: {day_key} 完成，{seconds:.2f}s")
    return {"days": days, "seconds": elapsed, "total_seconds": sum(elapsed)}


def _business_frame(frame: pd.DataFrame, output_name: str) -> pd.DataFrame:
    """仅保留模块间传递的业务字段，过滤不影响调度的零数量占位行。"""
    frame = pd.DataFrame() if frame is None else frame.copy()
    if output_name == "summary_df":
        canonical_summary_columns = {
            "total_orders": "Total_Orders",
            "total_shipments": "Total_Shipments",
            "total_cuts": "Total_Cuts",
            "total_supplydemand": "Total_SupplyDemand",
        }
        frame = frame.rename(columns={
            source: target
            for source, target in canonical_summary_columns.items()
            if source in frame.columns and target not in frame.columns
        })
    columns = [column for column in M1_BUSINESS_COLUMNS[output_name] if column in frame.columns]
    frame = frame.loc[:, columns]
    if output_name in {"orders_df", "shipment_df", "cut_df"} and "quantity" in frame.columns:
        frame = frame[pd.to_numeric(frame["quantity"], errors="coerce").fillna(0) != 0]
    if output_name == "summary_df" and not frame.empty:
        frame.insert(0, "summary_key", "daily")
    return frame


def _compare_day(left: dict, right: dict, label: str) -> dict:
    return {
        name: compare_dataframes_by_key(
            _business_frame(left["m1_result"].get(name), name),
            _business_frame(right["m1_result"].get(name), name),
            label=f"{label}:{name}",
            key_columns=["summary_key"] if name == "summary_df" else None,
        )
        for name in M1_FRAME_KEYS
    }


def _performance_summary(run: dict) -> dict:
    """记录本次真实对比运行的实际耗时，不重复执行算法采样。"""
    total = run["total_seconds"]
    return {
        "runs": [total],
        "min_seconds": total,
        "median_seconds": total,
        "mean_seconds": total,
        "per_day_mean_seconds": total / len(pd.date_range(START_DATE, END_DATE, freq="D")),
    }


def _write_report(report: dict) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "m1_two_way_compare.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = [
        "# M1 历史 DB / 重构 pandas / polars 对比报告",
        "",
        f"- 配置: `{CONFIG_NAME}`",
        f"- 范围: {START_DATE} 至 {END_DATE}",
        f"- 历史基线: PostgreSQL `{DB_SCHEMA}` schema，run_id `{HISTORICAL_RUN_ID}`。",
        "- 输入: PostgreSQL 中真实配置数据（含初始库存）；M1 不消费历史模块业务输出。",
        "- 当前运行: Decoupling refactor pandas 与 polars；均为内存运行，不写数据库。",
        "- 性能: 两个后端各执行一次完整五日回放，计时不含数据库读取、配置初始化与 prepare。",
        "",
        "## 性能",
        "",
        "| 实现 | min(s) | median(s) | mean(s) | 每日 mean(s) | 相对 pandas |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, value in report["performance"].items():
        lines.append(
            f"| {name} | {value['min_seconds']:.4f} | {value['median_seconds']:.4f} | "
            f"{value['mean_seconds']:.4f} | {value['per_day_mean_seconds']:.4f} | "
            f"{value['relative_to_pandas']:.2f}x |"
        )
    lines.extend([
        "",
        "## 差异说明",
        "",
        "业务差异仅记录在 JSON；不作为测试失败条件。每个仿真日分别比较历史 DB、"
        "pandas 与 polars 的 OrderLog、ShipmentLog、CutLog、SupplyDemandLog 和 Summary。",
    ])
    markdown_path = REPORT_DIR / "m1_two_way_compare.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return markdown_path


@pytest.fixture(scope="module")
def comparison_data():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text("", encoding="utf-8")
    db = _db()
    config_db = _db(CONFIG_SCHEMA)
    try:
        _progress("开始从数据库加载重构 M1 所需的真实配置数据")
        config_db.connect()
        config = _load_prepared_config(config_db)
        db.connect()
        historical = _load_historical_m1(db)
    finally:
        db.close()
        config_db.close()
    _progress("数据库配置加载完成")
    _progress("开始 Decoupling pandas M1 唯一一次真实对比运行并计时")
    refactor = _run_refactor(config, engine="pandas")
    _progress("开始 Decoupling polars M1 唯一一次真实对比运行并计时")
    polars = _run_refactor(config, engine="polars")

    performance = {
        "refactor_pandas": _performance_summary(refactor),
        "refactor_polars": _performance_summary(polars),
    }
    pandas_mean = performance["refactor_pandas"]["mean_seconds"]
    for summary in performance.values():
        summary["relative_to_pandas"] = summary["mean_seconds"] / pandas_mean
    report = {
        "config_name": CONFIG_NAME,
        "historical_run_id": HISTORICAL_RUN_ID,
        "database_schema": DB_SCHEMA,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "performance": performance,
        "comparisons": {
            "historical_db_vs_refactor_pandas": [
                {"date": left["date"], "comparison": _compare_day(left, right, "historical_vs_pandas")}
                for left, right in zip(historical["days"], refactor["days"])
            ],
            "historical_db_vs_refactor_polars": [
                {"date": left["date"], "comparison": _compare_day(left, right, "historical_vs_polars")}
                for left, right in zip(historical["days"], polars["days"])
            ],
            "refactor_pandas_vs_polars": [
                {"date": left["date"], "comparison": _compare_day(left, right, "pandas_vs_polars")}
                for left, right in zip(refactor["days"], polars["days"])
            ],
        },
    }
    report_path = _write_report(report)
    _progress(f"报告已写入: {report_path}")
    return {
        "historical": historical,
        "refactor": refactor,
        "polars": polars,
        "report": report,
        "report_path": report_path,
    }


def test_m1_two_way_compare_replays_configured_business_days(comparison_data):
    # 测试目的：验证“m1、two、way、compare、replays、configured、business、days”场景下订单生成、库存消耗与发运数据的一致性。
    # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `len()`，再通过 3 个断言核对结果、状态或异常条件。
    # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
    # 业务逻辑和原因：该校验防止订单生成、库存消耗与发运数据的一致性在重构、引擎切换或跨日运行时发生静默偏差。
    """两个实现均使用数据库配置完成指定日期范围的 M1 回放。"""
    expected_days = len(pd.date_range(START_DATE, END_DATE, freq="D"))
    assert len(comparison_data["refactor"]["days"]) == expected_days
    assert len(comparison_data["polars"]["days"]) == expected_days
    assert comparison_data["report_path"].exists()
    print(f"\nM1 真实配置数据对比报告: {comparison_data['report_path']}")


def test_m1_two_way_compare_records_actual_execution_timing(comparison_data):
    # 测试目的：验证“m1、two、way、compare、records、actual、execution、timing”场景下订单生成、库存消耗与发运数据的一致性。
    # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `all()`，再通过 2 个断言核对结果、状态或异常条件。
    # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
    # 业务逻辑和原因：该校验防止订单生成、库存消耗与发运数据的一致性在重构、引擎切换或跨日运行时发生静默偏差。
    """报告记录实际对比运行的单次端到端耗时，不重复执行计算。"""
    performance = comparison_data["report"]["performance"]
    assert set(performance) == {"refactor_pandas", "refactor_polars"}
    assert all(len(summary["runs"]) == 1 for summary in performance.values())
