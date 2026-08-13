"""真实数据库业务数据下的旧 M5、重构 pandas/Polars M5 五日对比。

与 ``test_m4_three_way_compare.py`` 的原则一致：从固定 ``run_id`` 读取真实的
M1 / M4 历史输出；两套独立 ``StateContext`` 在相同输入上逐日回放。业务差异
仅写入报告，不使测试失败，供重构阶段逐项收敛。

运行：
    conda run -n work pytest tests/test_m5_two_way_compare.py -s -q
"""
from __future__ import annotations

import json
import os
import pickle
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
from tests.compare_utils import compare_dataframes_by_key, dataframe_difference_details


PROJECT_ROOT = Path(__file__).parent.parent.resolve()
CONFIG_NAME = "OC_Paste_S1_20251224_repare"
HISTORICAL_RUN_ID = "db_OC_Paste_S1_20251224_repare_20260803_105300"
DB_SCHEMA = "public"
START_DATE = "2025-12-15"
END_DATE = "2025-12-19"
REPORT_DIR = PROJECT_ROOT / "outputs" / "m5_three_way_compare"
PROGRESS_PATH = REPORT_DIR / "m5_three_way_compare.progress.log"
PANDAS_BASELINE_CACHE_PATH = REPORT_DIR / "m5_refactor_pandas_baseline.pickle"
PANDAS_BASELINE_CACHE_VERSION = 2
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

M5_STATE_VIEW_GETTERS = {
    "SupplyDemandLog": "get_deployment_supply_demand_view",
    "OrderLog": "get_deployment_order_log_view",
    "TodayShipment": "get_shipment_log_view",
    "Inventory": "get_beginning_inventory_view",
    "InTransit": "get_planning_intransit_view",
    "DeliveryGR": "get_delivery_gr_view",
    "OpenDeployment": "get_open_deployment_view",
    "Production": "get_deployment_production_view",
    "ReceivingSpace": "get_space_quota_view",
}


class _M5StateViewSnapshot:
    """只读日初 Context 快照，供单日 Polars 与 pandas oracle 对齐。"""

    def __init__(self, views: dict[str, pd.DataFrame]):
        self._views = views

    def _get(self, name: str) -> pd.DataFrame:
        return self._views.get(name, pd.DataFrame()).copy(deep=True)

    def get_deployment_supply_demand_view(self, date): return self._get("SupplyDemandLog")
    def get_deployment_order_log_view(self, date): return self._get("OrderLog")
    def get_shipment_log_view(self, date): return self._get("TodayShipment")
    def get_beginning_inventory_view(self, date): return self._get("Inventory")
    def get_planning_intransit_view(self, date): return self._get("InTransit")
    def get_delivery_gr_view(self, date): return self._get("DeliveryGR")
    def get_open_deployment_view(self, date): return self._get("OpenDeployment")
    def get_deployment_production_view(self, date): return self._get("Production")
    def get_space_quota_view(self, date): return self._get("ReceivingSpace")


def _progress(message: str) -> None:
    """绕过项目 logger 对 stdout 的重定向，持续输出长任务进度。"""
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [M5 compare] {message}\n"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with PROGRESS_PATH.open("a", encoding="utf-8") as progress_file:
        progress_file.write(line)
    try:
        sys.__stdout__.write(line)
        sys.__stdout__.flush()
    except OSError:
        # pytest/IDE 捕获输出时，原始 Windows stdout 句柄可能已失效。
        pass


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


def _new_context(
    config: dict[str, pd.DataFrame], output_name: str, engine: str = "pandas"
) -> tuple[Orch, StateContext]:
    """为一条回放链路构建独立的 Orch / StateContext，禁止两端共享可变状态。"""
    orch = Orch(
        start_date=START_DATE,
        end_date=END_DATE,
        config_dict={name: frame.copy() for name, frame in config.items()},
        output_path=str(REPORT_DIR / output_name),
        engine=engine,
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


def _capture_m5_state_views(context: StateContext, day: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """捕获 M5 当日读取的全部原始 Context view，作为不可变基线输入。"""
    date_text = day.strftime("%Y-%m-%d")
    return {
        name: getattr(context, getter)(date_text).copy(deep=True)
        for name, getter in M5_STATE_VIEW_GETTERS.items()
    }


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
    engine: str,
) -> dict:
    """使用同一真实输入逐日运行独立 M5；状态仅由新输出推进。"""
    orch, context = _new_context(config, f"refactor_{engine}_scratch", engine)
    module = ModuleFive(
        simulation_date=START_DATE,
        simulation_start_date=START_DATE,
        state_context=context,
        orch=orch,
        verbose=True,
    )
    module.prepare()
    days, elapsed = [], []
    for day in pd.date_range(START_DATE, END_DATE, freq="D"):
        _progress(f"refactor {engine} {'计时' if timed else '预热'}: {day:%Y-%m-%d} 开始")
        m1, m4 = _replay_inputs(context, day, history)
        state_views = _capture_m5_state_views(context, day)
        module.simulation_date = day
        tick = time.perf_counter() if timed else None
        module.run()
        result = module.output()
        seconds = time.perf_counter() - tick if timed else None
        step_seconds = {
            name: round(value, 6)
            for name, value in module.spends.items()
            if name != "prepare"
        }
        daily_input_seconds = {
            name: round(value, 6)
            for name, value in getattr(module._backend, "last_daily_input_profile", {}).items()
        }
        row_counts = {
            name: len(frame)
            for name, frame in result.items()
            if isinstance(frame, pd.DataFrame)
        }
        day_key = day.strftime("%Y-%m-%d")
        open_deployment = context.get_open_deployment_view(day_key)
        in_transit = context.get_planning_intransit_view(day_key)
        _finish_day(context, day, result)
        days.append({
            "date": day.strftime("%Y-%m-%d"),
            "m5_result": result,
            "state_views": state_views,
            "elapsed_seconds": seconds,
            "step_seconds": step_seconds,
            "daily_input_seconds": daily_input_seconds,
            "plan_layer_profile": getattr(module._backend, "last_plan_layer_profile", []),
            "direct_demand_diagnostics": getattr(module._backend, "last_direct_demand_diagnostics", []),
            "row_counts": row_counts,
            "state_rows": {
                "open_deployment": len(open_deployment),
                "in_transit": len(in_transit),
            },
        })
        if seconds is not None:
            elapsed.append(seconds)
        _progress(
            f"refactor {engine} {'计时' if timed else '预热'}: {day:%Y-%m-%d} 完成"
            + (f"，{seconds:.2f}s" if seconds is not None else "")
        )
        _progress(
            f"refactor {engine} 明细: steps={step_seconds}; "
            f"output_rows={row_counts}; state_rows={{'open_deployment': {len(open_deployment)}, "
            f"'in_transit': {len(in_transit)}}}"
        )
        _progress(f"refactor {engine} daily_inputs 明细: {daily_input_seconds}")
    return {"days": days, "seconds": elapsed, "total_seconds": sum(elapsed)}


def _run_polars_against_pandas_state_baseline(
    config: dict[str, pd.DataFrame], pandas_run: dict, *, timed: bool
) -> dict:
    """对每个日期使用 pandas 缓存的日初 Context，隔离 Polars 单日语义差异。"""
    orch, _ = _new_context(config, "refactor_polars_cached_state", "polars")
    days, elapsed = [], []
    for baseline_day in pandas_run["days"]:
        day = pd.Timestamp(baseline_day["date"])
        module = ModuleFive(
            simulation_date=day,
            simulation_start_date=START_DATE,
            state_context=_M5StateViewSnapshot(baseline_day["state_views"]),
            orch=orch,
            verbose=True,
        )
        module.prepare()
        _progress(f"refactor polars 缓存 Context 计时: {day:%Y-%m-%d} 开始")
        tick = time.perf_counter() if timed else None
        module.run()
        result = module.output()
        seconds = time.perf_counter() - tick if timed else None
        days.append({
            "date": baseline_day["date"],
            "m5_result": result,
            "elapsed_seconds": seconds,
            "step_seconds": {name: round(value, 6) for name, value in module.spends.items() if name != "prepare"},
            "daily_input_seconds": {name: round(value, 6) for name, value in getattr(module._backend, "last_daily_input_profile", {}).items()},
            "plan_layer_profile": getattr(module._backend, "last_plan_layer_profile", []),
            "direct_demand_diagnostics": getattr(module._backend, "last_direct_demand_diagnostics", []),
            "row_counts": {name: len(frame) for name, frame in result.items() if isinstance(frame, pd.DataFrame)},
        })
        if seconds is not None:
            elapsed.append(seconds)
        _progress(
            f"refactor polars 缓存 Context 计时: {day:%Y-%m-%d} 完成"
            + (f"，{seconds:.2f}s" if seconds is not None else "")
        )
    return {"days": days, "seconds": elapsed, "total_seconds": sum(elapsed)}


def _to_pandas(frame) -> pd.DataFrame:
    """将诊断中间表统一到公开 pandas 比较边界。"""
    return frame.to_pandas() if hasattr(frame, "to_pandas") else frame.copy()


def _run_refactor_steps_against_snapshot(
    config: dict[str, pd.DataFrame], baseline_day: dict, engine: str
) -> dict[str, pd.DataFrame]:
    """显式执行单日稳定门面步骤，返回可与另一引擎逐项比较的中间表。"""
    day = pd.Timestamp(baseline_day["date"])
    orch, _ = _new_context(config, f"refactor_{engine}_step_diagnostic", engine)
    module = ModuleFive(
        simulation_date=day,
        simulation_start_date=START_DATE,
        state_context=_M5StateViewSnapshot(baseline_day["state_views"]),
        orch=orch,
        verbose=True,
    )
    module.prepare()
    inputs = module.load_daily_inputs(day)
    active = module.build_active_network(inputs, day)
    priority, validation = module.validate_config(inputs, active)
    routes = module.build_route_parameters(active, inputs)
    available, pools, projected, transit, shipment = module.build_supply_ledger(inputs, day)
    plan, direct, unfulfilled = module.build_layer_plan(
        day, inputs, active, routes, priority, available, pools
    )
    push = module.build_push_plan(plan, direct, active, routes, inputs, available, projected, day)
    appended_plan = module.append_plan(plan, push)
    constrained_plan, space_unfulfilled = module.apply_space_constraints(
        appended_plan, inputs["ReceivingSpace"], priority
    )
    final_unfulfilled = module.append_unfulfilled(unfulfilled, space_unfulfilled)
    result = module.finalise_result(
        constrained_plan, final_unfulfilled, available, transit, shipment, validation, day
    )
    return {
        "inputs.OpenDeployment": _to_pandas(inputs["OpenDeployment"]),
        "active_network": _to_pandas(active),
        "route_parameters": _to_pandas(routes),
        "supply.available": _to_pandas(available),
        "supply.pools": _to_pandas(pools),
        "supply.projected": _to_pandas(projected),
        "layer.direct": _to_pandas(direct),
        "layer.plan": _to_pandas(plan),
        "layer.unfulfilled": _to_pandas(unfulfilled),
        "push": _to_pandas(push),
        "space.plan": _to_pandas(constrained_plan),
        "space.unfulfilled": _to_pandas(space_unfulfilled),
        "result.deployment_plan": result["deployment_plan"],
        "result.unfulfilled_log": result["unfulfilled_log"],
        "result.stock_on_hand_log": result["stock_on_hand_log"],
        "result.validation_log": result["validation_log"],
    }


def _compare_day(left: dict, right: dict, label: str) -> dict:
    return {
        name: compare_dataframes_by_key(
            left["m5_result"].get(name, pd.DataFrame()),
            right["m5_result"].get(name, pd.DataFrame()),
            label=f"{label}:{name}",
        )
        for name in M5_FRAME_KEYS
    }


def _comparison_with_details(left_run: dict, right_run: dict, label: str) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    """比较两条已完成的回放链路，并按输出表汇总完整差异明细。"""
    days: list[dict] = []
    details_by_frame = {name: [] for name in M5_FRAME_KEYS}
    for left_day, right_day in zip(left_run["days"], right_run["days"]):
        date_text = left_day["date"]
        comparison = _compare_day(left_day, right_day, label)
        days.append({"date": date_text, "comparison": comparison})
        for name in M5_FRAME_KEYS:
            detail = dataframe_difference_details(
                left_day["m5_result"].get(name, pd.DataFrame()),
                right_day["m5_result"].get(name, pd.DataFrame()),
            )
            detail.insert(0, "output_table", name)
            detail.insert(0, "simulation_date", date_text)
            detail.insert(0, "comparison", label)
            details_by_frame[name].append(detail)
    return days, {
        name: pd.concat(frames, ignore_index=True)
        for name, frames in details_by_frame.items()
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


def _load_or_build_pandas_baseline(
    config: dict[str, pd.DataFrame],
    history: dict[str, dict[str, pd.DataFrame]],
) -> tuple[dict, bool]:
    """读取稳定 pandas oracle；仅在显式刷新或缓存缺失时重新回放五日。"""
    refresh = os.environ.get("M5_REFRESH_PANDAS_BASELINE") == "1"
    if PANDAS_BASELINE_CACHE_PATH.exists() and not refresh:
        with PANDAS_BASELINE_CACHE_PATH.open("rb") as cache_file:
            cached = pickle.load(cache_file)
        if cached.get("version") == PANDAS_BASELINE_CACHE_VERSION:
            _progress(f"pandas 基线缓存命中: {PANDAS_BASELINE_CACHE_PATH}")
            return cached["run"], True

    _progress("pandas 基线缓存未命中或已请求刷新：执行一次真实五日回放")
    run = _run_refactor(config, history, timed=True, engine="pandas")
    with PANDAS_BASELINE_CACHE_PATH.open("wb") as cache_file:
        pickle.dump(
            {
                "version": PANDAS_BASELINE_CACHE_VERSION,
                "run_id": HISTORICAL_RUN_ID,
                "config_name": CONFIG_NAME,
                "start_date": START_DATE,
                "end_date": END_DATE,
                "run": run,
            },
            cache_file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    _progress(f"pandas 基线缓存已写入: {PANDAS_BASELINE_CACHE_PATH}")
    return run, False


def _write_report(report: dict, details: dict[str, dict[str, pd.DataFrame]]) -> tuple[Path, dict[str, Path]]:
    """写入本轮汇总与指定两组比较的八份跨五日完整差异 CSV。"""
    run_dir = REPORT_DIR / time.strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    detail_paths: dict[str, Path] = {}
    for comparison, frames in details.items():
        for frame_name, frame in frames.items():
            path = run_dir / f"{comparison}_{frame_name}_differences.csv"
            frame.to_csv(path, index=False, encoding="utf-8-sig")
            detail_paths[f"{comparison}:{frame_name}"] = path

    report["detail_files"] = {
        name: str(path.relative_to(run_dir)).replace("\\", "/")
        for name, path in detail_paths.items()
    }
    json_path = run_dir / "m5_three_way_compare.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = [
        "# M5 单轮真实数据对比报告",
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
        "## 差异明细",
        "",
        "仅记录 legacy vs pandas、pandas vs Polars；不生成 legacy vs Polars。每组比较的"
        "DeploymentPlan、UnfulfilledLog、StockOnHandLog、Validation 各汇总为一份跨五日 CSV。",
        "本轮差异的业务性质以各 CSV 为准；若均为零值记录的输出/空值表示口径差异，"
        "则不影响非零业务结果。",
        "",
        "| 比较 | 输出表 | 差异记录数 | 明细 CSV |",
        "|---|---|---:|---|",
    ])
    for comparison, frames in details.items():
        for frame_name, frame in frames.items():
            path = detail_paths[f"{comparison}:{frame_name}"].name
            lines.append(f"| {comparison} | {frame_name} | {len(frame)} | {path} |")
    markdown_path = run_dir / "m5_three_way_compare.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return markdown_path, detail_paths


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
    _progress("开始重构 pandas M5 唯一一次真实对比运行并计时")
    refactor_pandas = _run_refactor(config, history, timed=True, engine="pandas")
    _progress("开始重构 Polars M5 唯一一次真实对比运行并计时")
    refactor_polars = _run_refactor(config, history, timed=True, engine="polars")
    performance = {
        "legacy": _performance_summary(legacy),
        "refactor_pandas": _performance_summary(refactor_pandas),
        "refactor_polars": _performance_summary(refactor_polars),
    }
    legacy_mean = performance["legacy"]["mean_seconds"]
    for summary in performance.values():
        summary["relative_to_legacy"] = summary["mean_seconds"] / legacy_mean
    legacy_pandas_days, legacy_pandas_details = _comparison_with_details(
        legacy, refactor_pandas, "legacy_vs_refactor_pandas"
    )
    pandas_polars_days, pandas_polars_details = _comparison_with_details(
        refactor_pandas, refactor_polars, "refactor_pandas_vs_polars"
    )
    report = {
        "run_id": HISTORICAL_RUN_ID,
        "config_name": CONFIG_NAME,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "performance": performance,
        "comparisons": {
            "legacy_vs_refactor_pandas": legacy_pandas_days,
            "refactor_pandas_vs_polars": pandas_polars_days,
        },
    }
    report_path, detail_paths = _write_report(
        report,
        {
            "legacy_vs_refactor_pandas": legacy_pandas_details,
            "refactor_pandas_vs_polars": pandas_polars_details,
        },
    )
    _progress(f"报告已写入: {report_path}")
    return {
        "legacy": legacy,
        "refactor_pandas": refactor_pandas,
        "refactor_polars": refactor_polars,
        "report": report,
        "report_path": report_path,
        "detail_paths": detail_paths,
    }


def test_m5_three_way_compare_replays_five_real_business_days(comparison_data):
    """真实数据库输入的旧、pandas、Polars 三条链路均完成固定五天。"""
    assert len(comparison_data["legacy"]["days"]) == 5
    assert len(comparison_data["refactor_pandas"]["days"]) == 5
    assert len(comparison_data["refactor_polars"]["days"]) == 5
    assert comparison_data["report_path"].exists()
    assert len(comparison_data["detail_paths"]) == 8
    assert all(path.exists() for path in comparison_data["detail_paths"].values())
    assert set(comparison_data["report"]["comparisons"]) == {
        "legacy_vs_refactor_pandas",
        "refactor_pandas_vs_polars",
    }
    assert all(len(days) == 5 for days in comparison_data["report"]["comparisons"].values())
    print(f"\nM5 真实业务数据对比报告: {comparison_data['report_path']}")


def test_m5_three_way_compare_records_actual_execution_timing(comparison_data):
    """报告记录实际对比运行的单次端到端耗时，不重复执行计算。"""
    performance = comparison_data["report"]["performance"]
    assert set(performance) == {"legacy", "refactor_pandas", "refactor_polars"}
    assert all(len(summary["runs"]) == 1 for summary in performance.values())


def _test_m5_real_data_performance_only(engine: str) -> None:
    """独立测量指定重构引擎的真实五日 M5，避免执行其他链路。"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    db = _db()
    try:
        _progress(f"{engine} 单独性能测试：开始加载数据库配置与历史 M1/M4 数据")
        db.connect()
        config = _load_prepared_config(db)
        history = _load_history(db)
    finally:
        db.close()

    _progress(f"{engine} 单独性能测试：开始五日运行并计时")
    run = _run_refactor(config, history, timed=True, engine=engine)
    summary = _performance_summary(run)
    result_path = REPORT_DIR / f"m5_{engine}_real_performance.json"
    result_path.write_text(
        json.dumps(
            {
                "engine": engine,
                "start_date": START_DATE,
                "end_date": END_DATE,
                "performance": summary,
                "per_day_seconds": run["seconds"],
                "days": [{
                    "date": day["date"],
                    "step_seconds": day["step_seconds"],
                    "daily_input_seconds": day["daily_input_seconds"],
                    "row_counts": day["row_counts"],
                    "state_rows": day["state_rows"],
                } for day in run["days"]],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _progress(f"{engine} 单独性能测试完成：{summary['total_seconds'] if 'total_seconds' in summary else summary['mean_seconds']:.2f}s；结果已写入 {result_path}")
    assert len(run["days"]) == 5
    assert len(run["seconds"]) == 5


def test_m5_polars_real_data_performance_only():
    """独立测量真实五日 Polars M5，避免先执行 legacy/pandas 链路。"""
    _test_m5_real_data_performance_only("polars")


def test_m5_pandas_real_data_performance_only():
    """独立测量真实五日 pandas M5，供逐日性能诊断使用。"""
    _test_m5_real_data_performance_only("pandas")


def test_m5_pandas_polars_real_data_difference_diagnostic():
    """写出真实五日双引擎差异，供逐步骤收敛到严格 parity。"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    db = _db()
    try:
        _progress("pandas/Polars 差异诊断：开始加载数据库配置与历史 M1/M4 数据")
        db.connect()
        config = _load_prepared_config(db)
        history = _load_history(db)
    finally:
        db.close()

    pandas_run, pandas_cache_hit = _load_or_build_pandas_baseline(config, history)
    polars_run = _run_polars_against_pandas_state_baseline(
        config, pandas_run, timed=True
    )
    differences = [
        {
            "date": pandas_day["date"],
            "comparison": _compare_day(pandas_day, polars_day, "refactor_pandas_vs_polars"),
            "pandas_rows": pandas_day["row_counts"],
            "polars_rows": polars_day["row_counts"],
            "pandas_plan_layers": pandas_day["plan_layer_profile"],
            "polars_plan_layers": polars_day["plan_layer_profile"],
            "polars_direct_demand_diagnostics": polars_day["direct_demand_diagnostics"],
        }
        for pandas_day, polars_day in zip(pandas_run["days"], polars_run["days"])
    ]
    result_path = REPORT_DIR / "m5_pandas_polars_real_differences.json"
    result_path.write_text(
        json.dumps(
            {
                "start_date": START_DATE,
                "end_date": END_DATE,
                "pandas_baseline_cache_hit": pandas_cache_hit,
                "comparison_mode": "isolated_polars_against_cached_pandas_day_start_state",
                "pandas_total_seconds": pandas_run["total_seconds"],
                "polars_total_seconds": polars_run["total_seconds"],
                "days": differences,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    _progress(f"pandas/Polars 差异诊断完成，结果已写入 {result_path}")
    assert len(differences) == 5
    for day in differences:
        for name, comparison in day["comparison"].items():
            assert comparison["left_only_keys"] == 0, (day["date"], name, "pandas_only")
            assert comparison["right_only_keys"] == 0, (day["date"], name, "polars_only")
            assert not comparison["column_differences"], (day["date"], name, "value")
            assert not comparison["precision_differences"], (day["date"], name, "precision")


def test_m5_pandas_polars_real_data_state_propagation_parity():
    """连续五日状态推进后，Polars 最终输出仍须与缓存 pandas oracle 严格一致。"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    db = _db()
    try:
        _progress("连续状态 parity：开始加载真实配置与历史输入")
        db.connect()
        config = _load_prepared_config(db)
        history = _load_history(db)
    finally:
        db.close()

    pandas_run, pandas_cache_hit = _load_or_build_pandas_baseline(config, history)
    polars_run = _run_refactor(config, history, timed=True, engine="polars")
    differences = [
        {
            "date": pandas_day["date"],
            "comparison": _compare_day(
                pandas_day, polars_day, "refactor_pandas_vs_polars_state_propagation"
            ),
        }
        for pandas_day, polars_day in zip(pandas_run["days"], polars_run["days"])
    ]
    result_path = REPORT_DIR / "m5_pandas_polars_state_propagation_differences.json"
    result_path.write_text(
        json.dumps(
            {
                "start_date": START_DATE,
                "end_date": END_DATE,
                "pandas_baseline_cache_hit": pandas_cache_hit,
                "comparison_mode": "consecutive_pandas_and_polars_state_propagation",
                "pandas_total_seconds": pandas_run["total_seconds"],
                "polars_total_seconds": polars_run["total_seconds"],
                "days": differences,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    _progress(f"连续状态 parity 完成，结果已写入 {result_path}")
    for day in differences:
        for name, comparison in day["comparison"].items():
            assert comparison["left_only_keys"] == 0, (day["date"], name, "pandas_only")
            assert comparison["right_only_keys"] == 0, (day["date"], name, "polars_only")
            assert not comparison["column_differences"], (day["date"], name, "value")
            assert not comparison["precision_differences"], (day["date"], name, "precision")


def test_m5_pandas_polars_first_divergent_step_diagnostic():
    """在首个不一致日逐门面比较，避免完整五日回放后才定位差异。"""
    db = _db()
    try:
        _progress("逐步骤差异诊断：开始加载真实配置")
        db.connect()
        config = _load_prepared_config(db)
        history = _load_history(db)
    finally:
        db.close()

    pandas_run, _ = _load_or_build_pandas_baseline(config, history)
    baseline_day = next(day for day in pandas_run["days"] if day["date"] == "2025-12-16")
    pandas_steps = _run_refactor_steps_against_snapshot(config, baseline_day, "pandas")
    polars_steps = _run_refactor_steps_against_snapshot(config, baseline_day, "polars")
    differences = {
        name: compare_dataframes_by_key(
            pandas_steps[name], polars_steps[name],
            label=f"2025-12-16:{name}",
        )
        for name in pandas_steps
    }
    result_path = REPORT_DIR / "m5_pandas_polars_first_divergent_step.json"
    result_path.write_text(
        json.dumps(
            {
                "date": "2025-12-16",
                "comparison_mode": "isolated_pandas_and_polars_against_cached_pandas_day_start_state",
                "steps": differences,
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    _progress(f"逐步骤差异诊断完成，结果已写入 {result_path}")
    assert set(differences) == set(pandas_steps)
