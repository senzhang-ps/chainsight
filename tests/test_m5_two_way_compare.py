"""真实数据库业务数据下的旧 M5 与重构 pandas M5 五日对比。

与 ``test_m4_three_way_compare.py`` 的原则一致：从固定 ``run_id`` 读取真实的
M1 / M4 历史输出；两套独立 ``StateContext`` 在相同输入上逐日回放。业务差异
仅写入报告，不使测试失败，供重构阶段逐项收敛。

运行：
    conda run -n work pytest tests/test_m5_two_way_compare.py -s -q
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

from pgsql_db.db_connection import DatabaseConnection
from pgsql_db.settings import get_database_config
from src.core.main_integration.config_loader import (
    load_configuration_from_dict,
    prepare_configuration,
)
from src.core.orchestrator import Orch
from src.core.run.db_config import _load_config_from_database
from src.modules.deployment_planning.integration_refactor import ModuleFive
from src.modules.deployment_planning.main import run_daily_deployment_planning
from src.modules.state_context import StateContext
from tests.compare_utils import compare_dataframes_by_key


PROJECT_ROOT = Path(__file__).parent.parent.resolve()
CONFIG_NAME = "OC_Paste_S1_20251224_repare"
HISTORICAL_RUN_ID = "db_OC_Paste_S1_20251224_repare_20260803_105300"
DB_SCHEMA = "public"
START_DATE = "2025-12-15"
END_DATE = "2025-12-19"
REPORT_DIR = PROJECT_ROOT / "outputs" / "m5_two_way_compare"
PROGRESS_PATH = REPORT_DIR / "m5_two_way_compare.progress.log"
M1_TABLES = {
    "all_orders_for_next_day": "module1_output_orderlog",
    "shipment_df": "module1_output_shipmentlog",
    "supply_demand_df": "module1_output_supplydemandlog",
}
M4_PRODUCTION_TABLE = "module4_output_productionplan"
M5_FRAME_KEYS = (
    "deployment_plan",
    "unfulfilled_log",
    "stock_on_hand_log",
    "validation_log",
)


def _progress(message: str) -> None:
    """绕过项目 logger 对 stdout 的重定向，持续输出长任务进度。"""
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [M5 compare] {message}\n"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with PROGRESS_PATH.open("a", encoding="utf-8") as progress_file:
        progress_file.write(line)
    sys.__stdout__.write(line)
    sys.__stdout__.flush()


def _db() -> DatabaseConnection:
    config = get_database_config()
    return DatabaseConnection(
        host=config["host"],
        port=config["port"],
        database=config["database"],
        user=config["user"],
        password=config["password"],
        schema=config.get("default_schema", "public"),
        auto_create_schema=False,
    )


def _qualified(table: str) -> str:
    return f'"{DB_SCHEMA}"."{table}"'


def _table_columns(db: DatabaseConnection, table: str) -> list[str]:
    rows = db.execute_query(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position",
        (DB_SCHEMA, table),
    )
    return [row[0] for row in rows]


def _load_daily_table(
    db: DatabaseConnection,
    table: str,
    day: pd.Timestamp,
) -> pd.DataFrame:
    """按固定历史运行和仿真日读取业务表，并剥离写库元数据。"""
    columns = _table_columns(db, table)
    if not columns:
        pytest.skip(f"找不到历史表: {DB_SCHEMA}.{table}")
    rows = db.execute_query(
        f"SELECT {', '.join(columns)} FROM {_qualified(table)} "
        "WHERE run_id = %s AND sim_date::date = %s::date",
        (HISTORICAL_RUN_ID, day.strftime("%Y-%m-%d")),
    )
    return pd.DataFrame(rows, columns=columns).drop(
        columns=["run_id", "sim_date", "config_name", "db_write_time"],
        errors="ignore",
    )


def _load_history(db: DatabaseConnection) -> dict[str, dict[str, pd.DataFrame]]:
    """预读五天 M1/M4 历史输出；运行计时不包含数据库读取。"""
    history: dict[str, dict[str, pd.DataFrame]] = {}
    for day in pd.date_range(START_DATE, END_DATE, freq="D"):
        day_key = day.strftime("%Y-%m-%d")
        m1 = {name: _load_daily_table(db, table, day) for name, table in M1_TABLES.items()}
        history[day_key] = {
            **m1,
            "production_df": _load_daily_table(db, M4_PRODUCTION_TABLE, day),
        }
    if all(frame.empty for values in history.values() for frame in values.values()):
        pytest.skip(f"run_id={HISTORICAL_RUN_ID} 没有可用的 M1/M4 历史业务数据")
    return history


def _load_prepared_config(db: DatabaseConnection) -> dict[str, pd.DataFrame]:
    raw = _load_config_from_database(db, CONFIG_NAME)
    if not raw:
        pytest.skip(f"数据库 schema={DB_SCHEMA} 中找不到配置: {CONFIG_NAME}")
    return prepare_configuration(load_configuration_from_dict(raw, CONFIG_NAME))


def _m1_result(history: dict[str, pd.DataFrame]) -> dict:
    """构造 M5 所消费的 M1 内存结果。"""
    return {
        "all_orders_for_next_day": history["all_orders_for_next_day"].copy(),
        "shipment_df": history["shipment_df"].copy(),
        "supply_demand_df": history["supply_demand_df"].copy(),
    }


def _m4_result(history: dict[str, pd.DataFrame]) -> dict:
    production = history["production_df"].copy()
    if not production.empty and "quantity" not in production and "produced_qty" in production:
        production["quantity"] = production["produced_qty"]
    return {"production_df": production}


def _new_context(config: dict[str, pd.DataFrame], output_name: str) -> tuple[Orch, StateContext]:
    """为一条回放链路构建独立的 Orch / StateContext，禁止两端共享可变状态。"""
    orch = Orch(
        start_date=START_DATE,
        end_date=END_DATE,
        config_dict={name: frame.copy() for name, frame in config.items()},
        output_path=str(REPORT_DIR / output_name),
        engine="pandas",
        skip_dq=True,
    )
    context = StateContext(simulation_date=START_DATE, orch=orch)
    context.initialize(orch.all_config)
    return orch, context


def _replay_inputs(context: StateContext, day: pd.Timestamp, history: dict[str, pd.DataFrame]) -> tuple[dict, dict]:
    """按真实历史 M1/M4 结果驱动当天 Context 状态。"""
    day_key = day.strftime("%Y-%m-%d")
    context.day_start(day_key)
    m1 = _m1_result(history[day_key])
    m4 = _m4_result(history[day_key])
    if not m1["shipment_df"].empty:
        context.apply_shipments(m1["shipment_df"], day_key)
    context.apply_deployment_demand_inputs(
        m1["supply_demand_df"], m1["all_orders_for_next_day"], day_key
    )
    if not m4["production_df"].empty:
        context.apply_production(m4["production_df"], day_key)
    return m1, m4


def _finish_day(context: StateContext, day: pd.Timestamp, result: dict) -> None:
    deployment = result.get("deployment_plan", pd.DataFrame())
    if not deployment.empty:
        context.apply_deployment(deployment, day.strftime("%Y-%m-%d"))
    context.day_end(day.strftime("%Y-%m-%d"))


def _run_legacy(
    config: dict[str, pd.DataFrame],
    history: dict[str, dict[str, pd.DataFrame]],
    *,
    timed: bool,
) -> dict:
    """使用真实数据库输入逐日运行旧 M5；状态仅由旧输出推进。"""
    _, context = _new_context(config, "legacy_scratch")
    days, elapsed = [], []
    for day in pd.date_range(START_DATE, END_DATE, freq="D"):
        _progress(f"legacy {'计时' if timed else '预热'}: {day:%Y-%m-%d} 开始")
        m1, m4 = _replay_inputs(context, day, history)
        tick = time.perf_counter() if timed else None
        result = run_daily_deployment_planning(
            config_dict=config,
            orchestrator=context,
            current_date=day.strftime("%Y-%m-%d"),
            skip_file_output=True,
            module1_result=m1,
            module4_result=m4,
        )
        seconds = time.perf_counter() - tick if timed else None
        _finish_day(context, day, result)
        days.append({"date": day.strftime("%Y-%m-%d"), "m5_result": result, "elapsed_seconds": seconds})
        if seconds is not None:
            elapsed.append(seconds)
        _progress(
            f"legacy {'计时' if timed else '预热'}: {day:%Y-%m-%d} 完成"
            + (f"，{seconds:.2f}s" if seconds is not None else "")
        )
    return {"days": days, "seconds": elapsed, "total_seconds": sum(elapsed)}


def _run_refactor(
    config: dict[str, pd.DataFrame],
    history: dict[str, dict[str, pd.DataFrame]],
    *,
    timed: bool,
) -> dict:
    """使用同一真实输入逐日运行独立 pandas M5；状态仅由新输出推进。"""
    orch, context = _new_context(config, "refactor_scratch")
    module = ModuleFive(
        simulation_date=START_DATE,
        simulation_start_date=START_DATE,
        state_context=context,
        orch=orch,
    )
    module.prepare()
    days, elapsed = [], []
    for day in pd.date_range(START_DATE, END_DATE, freq="D"):
        _progress(f"refactor {'计时' if timed else '预热'}: {day:%Y-%m-%d} 开始")
        m1, m4 = _replay_inputs(context, day, history)
        module.simulation_date = day
        tick = time.perf_counter() if timed else None
        module.run()
        result = module.output()
        seconds = time.perf_counter() - tick if timed else None
        _finish_day(context, day, result)
        days.append({"date": day.strftime("%Y-%m-%d"), "m5_result": result, "elapsed_seconds": seconds})
        if seconds is not None:
            elapsed.append(seconds)
        _progress(
            f"refactor {'计时' if timed else '预热'}: {day:%Y-%m-%d} 完成"
            + (f"，{seconds:.2f}s" if seconds is not None else "")
        )
    return {"days": days, "seconds": elapsed, "total_seconds": sum(elapsed)}


def _compare_day(left: dict, right: dict) -> dict:
    return {
        name: compare_dataframes_by_key(
            left["m5_result"].get(name, pd.DataFrame()),
            right["m5_result"].get(name, pd.DataFrame()),
            label=f"legacy_vs_refactor:{name}",
        )
        for name in M5_FRAME_KEYS
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
    json_path = REPORT_DIR / "m5_two_way_compare.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = [
        "# M5 旧实现与 pandas 重构对比报告",
        "",
        f"- 历史 run_id: `{HISTORICAL_RUN_ID}`",
        f"- 配置: `{CONFIG_NAME}`",
        f"- 范围: {START_DATE} 至 {END_DATE}",
        "- 输入: PostgreSQL 中真实 M1/M4 历史业务输出；旧/新 Context 独立逐日回放。",
        "- 性能: 每个实现仅执行一次完整五日对比，记录该次实际端到端运行耗时。",
        "",
        "## 性能",
        "",
        "| 实现 | min(s) | median(s) | mean(s) | 每日 mean(s) | 相对旧 M5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, value in report["performance"].items():
        lines.append(
            f"| {name} | {value['min_seconds']:.4f} | {value['median_seconds']:.4f} | "
            f"{value['mean_seconds']:.4f} | {value['per_day_mean_seconds']:.4f} | "
            f"{value['relative_to_legacy']:.2f}x |"
        )
    lines.extend([
        "",
        "## 差异说明",
        "",
        "业务差异仅记录在 JSON；不作为测试失败条件。每个仿真日分别比较 DeploymentPlan、"
        "UnfulfilledLog、StockOnHandLog 和 Validation。",
    ])
    markdown_path = REPORT_DIR / "m5_two_way_compare.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return markdown_path


@pytest.fixture(scope="module")
def comparison_data():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text("", encoding="utf-8")
    db = _db()
    try:
        _progress("开始加载数据库配置与五日真实 M1/M4 历史业务数据")
        db.connect()
        config = _load_prepared_config(db)
        history = _load_history(db)
    finally:
        db.close()
    _progress("数据库输入加载完成")

    _progress("开始旧 M5 唯一一次真实对比运行并计时")
    legacy = _run_legacy(config, history, timed=True)
    _progress("开始重构 M5 唯一一次真实对比运行并计时")
    refactor = _run_refactor(config, history, timed=True)
    performance = {
        "legacy": _performance_summary(legacy),
        "refactor_pandas": _performance_summary(refactor),
    }
    legacy_mean = performance["legacy"]["mean_seconds"]
    for summary in performance.values():
        summary["relative_to_legacy"] = summary["mean_seconds"] / legacy_mean
    report = {
        "run_id": HISTORICAL_RUN_ID,
        "config_name": CONFIG_NAME,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "performance": performance,
        "comparisons": [
            {"date": left["date"], "comparison": _compare_day(left, right)}
            for left, right in zip(legacy["days"], refactor["days"])
        ],
    }
    report_path = _write_report(report)
    _progress(f"报告已写入: {report_path}")
    return {"legacy": legacy, "refactor": refactor, "report": report, "report_path": report_path}


def test_m5_two_way_compare_replays_five_real_business_days(comparison_data):
    """真实数据库输入的旧/新两条链路均完成固定五天。"""
    assert len(comparison_data["legacy"]["days"]) == 5
    assert len(comparison_data["refactor"]["days"]) == 5
    assert comparison_data["report_path"].exists()
    print(f"\nM5 真实业务数据对比报告: {comparison_data['report_path']}")


def test_m5_two_way_compare_records_actual_execution_timing(comparison_data):
    """报告记录实际对比运行的单次端到端耗时，不重复执行计算。"""
    performance = comparison_data["report"]["performance"]
    assert set(performance) == {"legacy", "refactor_pandas"}
    assert all(len(summary["runs"]) == 1 for summary in performance.values())
