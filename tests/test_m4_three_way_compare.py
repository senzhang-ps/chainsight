"""旧 M4、重构 M4 pandas 与重构 M4 polars 的五天对比测试。

历史 M3 是唯一动态输入，严格按 ``run_id + sim_date`` 查询。旧/新 M4 在仿真日
D 都消费 D-1 日的 M3；首日使用空净需求。测试只写报告，不因业务输出差异失败。

运行：
    pytest tests/test_m4_three_way_compare.py -s -q
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

from pgsql_db.settings import get_database_config
from pgsql_db.db_connection import DatabaseConnection
from src.core.main_integration.config_loader import (
    load_configuration_from_dict,
    prepare_configuration,
)
from src.core.main_integration.production_runner import (
    run_daily_production_planning_integrated,
)
from src.core.main_integration.runtime_state import DbRuntimeState
from src.core.run.db_config import _load_config_from_database
from src.core.orchestrator import Orch
from src.modules.production_planning.integration_refactor import ModuleFour
from src.modules.production_planning.plan_builder import build_unconstrained_plan_for_single_day
from src.modules.production_planning.utils import cast_identifiers_to_str
from src.utils.date_helpers import is_offset_review_day
from src.utils.normalization import normalize_material

from tests.compare_utils import compare_dataframes_by_key, compare_mappings


PROJECT_ROOT = Path(__file__).parent.parent.resolve()
CONFIG_NAME = "OC_Paste_S1_20251224_repare"
M3_RUN_ID = "db_OC_Paste_S1_20251224_repare_20260803_105300"
DB_SCHEMA = "public"
START_DATE = "2025-12-15"
END_DATE = "2025-12-19"
M3_TABLE = "module3_output_netdemand"
REPORT_DIR = PROJECT_ROOT / "outputs" / "m4_three_way_compare"
PROGRESS_PATH = REPORT_DIR / "m4_three_way_compare.progress.log"
M4_FRAME_KEYS = (
    "production_df",
    "exceed_log",
    "issues_df",
    "changeover_log",
    "unconstrained_plan",
)
M4_MAPPING_KEYS = ("current_line_states", "current_allocated_capacity")
LEGACY_STATE_POLICY = "historical_capacity_merged"
REFACTOR_STATE_POLICY = "historical_capacity_merged"


def _progress(message: str) -> None:
    """绕过项目 logger 对 stdout 的重定向，持续输出长任务进度。"""
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [M4 compare] {message}\n"
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
        # cfg_* 与本次历史 M3 都位于默认 public schema；M3 查询仍在
        # _load_m3_by_sim_date 中显式限定 schema，确保 run_id + sim_date 筛选。
        schema=config.get("default_schema", "public"),
        auto_create_schema=False,
    )


def _qualified(schema: str, table: str) -> str:
    return f'"{schema}"."{table}"'


def _load_m3_by_sim_date(db: DatabaseConnection) -> dict[str, pd.DataFrame]:
    """逐日按 ``run_id + sim_date`` 拉取 M3，防止跨运行、跨日期混入。"""
    table = _qualified(DB_SCHEMA, M3_TABLE)
    columns = [row[0] for row in db.execute_query(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position",
        (DB_SCHEMA, M3_TABLE),
    )]
    if not columns:
        pytest.skip(f"找不到历史 M3 表: {DB_SCHEMA}.{M3_TABLE}")

    output: dict[str, pd.DataFrame] = {}
    for date in pd.date_range(START_DATE, END_DATE, freq="D"):
        date_str = date.strftime("%Y-%m-%d")
        rows = db.execute_query(
            f"SELECT {', '.join(columns)} FROM {table} "
            "WHERE run_id = %s AND sim_date::date = %s::date",
            (M3_RUN_ID, date_str),
        )
        frame = pd.DataFrame(rows, columns=columns)
        output[date_str] = frame.drop(
            columns=["run_id", "sim_date", "config_name", "db_write_time"],
            errors="ignore",
        )
    missing = [
        date.strftime("%Y-%m-%d")
        for date in pd.date_range(START_DATE, END_DATE, freq="D")[1:]
        if output[(date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")].empty
    ]
    if missing:
        pytest.skip(
            f"run_id={M3_RUN_ID} 缺少 M4 所需的前一日 M3: {', '.join(missing)}"
        )
    return output


def _load_prepared_config(db: DatabaseConnection) -> dict[str, pd.DataFrame]:
    raw = _load_config_from_database(db, CONFIG_NAME)
    if not raw:
        pytest.skip(f"数据库 schema={DB_SCHEMA} 中找不到配置: {CONFIG_NAME}")
    return prepare_configuration(load_configuration_from_dict(raw, CONFIG_NAME))


def _m3_input(m3_by_date: dict[str, pd.DataFrame], day: pd.Timestamp) -> dict | None:
    if day.normalize() == pd.Timestamp(START_DATE):
        return None
    previous = (day - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return {"net_demand_df": m3_by_date[previous].copy()}


def _run_legacy(config: dict[str, pd.DataFrame], m3_by_date: dict[str, pd.DataFrame], *, timed: bool) -> dict:
    """直接运行旧 M4，复刻数据库入口的跨日产能历史合并。"""
    days, elapsed = [], []
    start = pd.Timestamp(START_DATE)
    state = DbRuntimeState()
    for day in pd.date_range(START_DATE, END_DATE, freq="D"):
        _progress(f"legacy {'计时' if timed else '预热'}: {day:%Y-%m-%d} 开始")
        tick = time.perf_counter() if timed else None
        result = run_daily_production_planning_integrated(
            config_dict=config,
            module3_output_dir="",
            simulation_date=day,
            simulation_start=start,
            output_dir="",
            skip_file_output=True,
            module3_result=_m3_input(m3_by_date, day),
            previous_line_states_override=state.load_line_state(day),
            allocated_capacity_override=state.load_all_previous_capacity(day),
            skip_state_file_output=True,
        )
        seconds = time.perf_counter() - tick if timed else None
        line_states = result.get("current_line_states", {})
        capacity = result.get("current_allocated_capacity", {})
        state.save_line_state(day, line_states)
        state.save_allocated_capacity(day, capacity)
        days.append({"date": day.strftime("%Y-%m-%d"), "m4_result": result,
                     "production_df": result.get("production_df", pd.DataFrame()),
                     "current_line_states": line_states,
                     "current_allocated_capacity": capacity,
                     "elapsed_seconds": seconds})
        if seconds is not None:
            elapsed.append(seconds)
        _progress(
            f"legacy {'计时' if timed else '预热'}: {day:%Y-%m-%d} 完成"
            + (f"，{seconds:.2f}s" if seconds is not None else "")
        )
    return {"days": days, "seconds": elapsed, "total_seconds": sum(elapsed)}


def _run_refactor(
    config: dict[str, pd.DataFrame],
    m3_by_date: dict[str, pd.DataFrame],
    *,
    engine: str,
    timed: bool,
) -> dict:
    """运行重构 M4。历史 M3 已预读，计时内不含数据库查询或配置初始化。"""
    # DB 配置保留 Excel 的 ``MCT`` 列名；重构 backend 的 schema 使用 ``mct``。
    # 仅在测试副本做兼容转换，不能改变旧 M4 使用的原始配置。
    refactor_config = {name: frame.copy() for name, frame in config.items()}
    mlcfg = refactor_config.get("M4_MaterialLocationLineCfg")
    if mlcfg is not None and "MCT" in mlcfg.columns and "mct" not in mlcfg.columns:
        refactor_config["M4_MaterialLocationLineCfg"] = mlcfg.rename(columns={"MCT": "mct"})
    orch = Orch(
        start_date=START_DATE,
        end_date=END_DATE,
        config_dict=refactor_config,
        output_path=str(REPORT_DIR / "scratch"),
        engine=engine,
        skip_dq=True,
    )
    m4 = ModuleFour(
        simulation_date=START_DATE,
        simulation_start_date=START_DATE,
        orchestrator=orch,
        orch=orch,
    )
    m4.prepare()

    line_states_by_date: dict[str, dict] = {}
    allocated_by_date: dict[str, dict] = {}
    days, elapsed = [], []
    for day in pd.date_range(START_DATE, END_DATE, freq="D"):
        day_key = day.strftime("%Y-%m-%d")
        previous_key = (day - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        previous_capacity: dict = {}
        for capacity_date in sorted(allocated_by_date):
            if capacity_date >= day_key:
                break
            for key, value in allocated_by_date[capacity_date].items():
                previous_capacity[key] = previous_capacity.get(key, 0.0) + value
        m4.simulation_date = day
        m4.module3_result = _m3_input(m3_by_date, day)
        m4.previous_line_states_override = line_states_by_date.get(previous_key, {})
        m4.allocated_capacity_override = previous_capacity
        _progress(f"refactor {engine} {'计时' if timed else '预热'}: {day:%Y-%m-%d} 开始")
        tick = time.perf_counter() if timed else None
        m4.run()
        result = m4.output()
        seconds = time.perf_counter() - tick if timed else None
        line_states = result.get("current_line_states", {})
        capacity = result.get("current_allocated_capacity", {})
        line_states_by_date[day_key] = line_states
        allocated_by_date[day_key] = capacity
        days.append({"date": day_key, "m4_result": result,
                     "production_df": result.get("production_df", pd.DataFrame()),
                     "current_line_states": line_states,
                     "current_allocated_capacity": capacity,
                     "elapsed_seconds": seconds})
        if seconds is not None:
            elapsed.append(seconds)
        _progress(
            f"refactor {engine} {'计时' if timed else '预热'}: {day:%Y-%m-%d} 完成"
            + (f"，{seconds:.2f}s" if seconds is not None else "")
        )
    return {"days": days, "seconds": elapsed, "total_seconds": sum(elapsed)}


def _compare_day(left: dict, right: dict, label: str) -> dict:
    comparison = {
        "production_df": compare_dataframes_by_key(
            left["production_df"], right["production_df"], label=f"{label}:production_df"),
        "current_line_states": compare_mappings(
            left["current_line_states"], right["current_line_states"],
            label=f"{label}:current_line_states"),
        "current_allocated_capacity": compare_mappings(
            left["current_allocated_capacity"], right["current_allocated_capacity"],
            label=f"{label}:current_allocated_capacity"),
        "m4_result": {},
    }
    for key in M4_FRAME_KEYS:
        comparison["m4_result"][key] = compare_dataframes_by_key(
            left["m4_result"].get(key, pd.DataFrame()),
            right["m4_result"].get(key, pd.DataFrame()),
            label=f"{label}:m4_result.{key}")
    for key in M4_MAPPING_KEYS:
        comparison["m4_result"][key] = compare_mappings(
            left["m4_result"].get(key, {}), right["m4_result"].get(key, {}),
            label=f"{label}:m4_result.{key}")
    return comparison


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


def _frame_records(frame: pd.DataFrame) -> list[dict]:
    """转换为可写入 JSON 的记录，同时保留日期与 Pandas nullable 类型。"""
    if frame is None or frame.empty:
        return []
    return json.loads(frame.to_json(orient="records", date_format="iso", default_handler=str))


def _legacy_net_demand(m3_by_date: dict[str, pd.DataFrame], day: pd.Timestamp) -> pd.DataFrame:
    """复刻旧 M4 进入计划构建器前的净需求预处理。"""
    source = _m3_input(m3_by_date, day)
    if source is None:
        return pd.DataFrame()
    net_demand = source["net_demand_df"].copy()
    if "layer" in net_demand.columns:
        net_demand = net_demand[net_demand["layer"] == 0].copy()
    if "quantity" in net_demand.columns:
        net_demand["quantity"] = net_demand["quantity"].abs()
    for column in ("material", "location"):
        if column in net_demand.columns:
            net_demand[column] = net_demand[column].astype(str)
    if "material" in net_demand.columns:
        net_demand["material"] = net_demand["material"].map(normalize_material).astype("string")
    if "requirement_date" in net_demand.columns:
        net_demand["requirement_date"] = pd.to_datetime(net_demand["requirement_date"])
    return net_demand


def _write_first_row_difference_trace(
    config: dict[str, pd.DataFrame],
    m3_by_date: dict[str, pd.DataFrame],
    legacy: dict,
    refactor_pandas: dict,
    report: dict,
) -> Path | None:
    """追踪第一个旧端/重构端生产行集合差异，判断差异首次出现的位置。"""
    comparisons = report["comparisons"]["legacy_vs_refactor_pandas"]
    selected = next(
        (
            item for item in comparisons
            if item["comparison"]["production_df"].get("right_only_samples")
            or item["comparison"]["production_df"].get("left_only_samples")
        ),
        None,
    )
    if selected is None:
        return None

    production_comparison = selected["comparison"]["production_df"]
    side = "right_only_samples" if production_comparison.get("right_only_samples") else "left_only_samples"
    target = production_comparison[side][0]
    day = pd.Timestamp(selected["date"])
    material, location, line = (
        str(target["material"]), str(target["location"]), str(target["line"]),
    )

    legacy_net_demand = _legacy_net_demand(m3_by_date, day)
    config_rows = cast_identifiers_to_str(
        config["M4_MaterialLocationLineCfg"].copy(), ["material", "location"],
    )
    config_rows = config_rows[
        (config_rows["material"] == material)
        & (config_rows["location"] == location)
    ].copy()
    config_rows["is_review_day"] = config_rows.apply(
        lambda row: is_offset_review_day(
            day, pd.Timestamp(START_DATE), int(row["lsk"]), int(row["day"]),
        ),
        axis=1,
    )
    issues: list[dict] = []
    legacy_unconstrained = build_unconstrained_plan_for_single_day(
        legacy_net_demand,
        config["M4_MaterialLocationLineCfg"],
        day,
        pd.Timestamp(START_DATE),
        issues,
    )

    day_index = next(
        index for index, item in enumerate(legacy["days"])
        if item["date"] == selected["date"]
    )
    legacy_production = legacy["days"][day_index]["production_df"]
    refactor_day = refactor_pandas["days"][day_index]["m4_result"]
    refactor_unconstrained = refactor_day["unconstrained_plan"]
    refactor_production = refactor_day["production_df"]

    def filter_plan(frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame()
        matches = (frame["material"].astype(str) == material) & (frame["location"].astype(str) == location)
        if "line" in frame.columns:
            matches &= frame["line"].astype(str) == line
        return frame[matches].copy()

    def filter_line(frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame()
        return frame[frame["line"].astype(str) == line].copy()

    legacy_line_unconstrained = filter_line(legacy_unconstrained)
    refactor_line_unconstrained = filter_line(refactor_unconstrained)
    unconstrained_keys = [
        "material", "location", "line", "planned_date", "simulation_date",
    ]
    unconstrained_differences = legacy_line_unconstrained.merge(
        refactor_line_unconstrained,
        on=unconstrained_keys,
        how="outer",
        suffixes=("__legacy", "__refactor"),
        indicator=True,
    )
    unconstrained_differences = unconstrained_differences[
        (unconstrained_differences["_merge"] != "both")
        | (
            unconstrained_differences["uncon_planned_qty__legacy"]
            != unconstrained_differences["uncon_planned_qty__refactor"]
        )
    ]

    trace = {
        "date": selected["date"],
        "difference_side": side,
        "target_production_key": target,
        "interpretation": (
            "若 legacy_unconstrained 与 refactor_unconstrained 已不同，差异在无约束计划构建阶段；"
            "若二者相同而 production 不同，差异首次出现于产能分配或换产阶段。"
        ),
        "legacy_net_demand_for_material": _frame_records(
            legacy_net_demand[
                (legacy_net_demand["material"].astype(str) == material)
                & (legacy_net_demand["location"].astype(str) == location)
            ]
        ),
        "matching_mlcfg_rows": _frame_records(config_rows),
        "legacy_unconstrained": _frame_records(filter_plan(legacy_unconstrained)),
        "refactor_unconstrained": _frame_records(filter_plan(refactor_unconstrained)),
        "legacy_unconstrained_same_line": _frame_records(legacy_line_unconstrained),
        "refactor_unconstrained_same_line": _frame_records(refactor_line_unconstrained),
        "unconstrained_same_line_differences": _frame_records(unconstrained_differences),
        "legacy_production": _frame_records(filter_plan(legacy_production)),
        "refactor_production": _frame_records(filter_plan(refactor_production)),
        "legacy_production_same_line": _frame_records(filter_line(legacy_production)),
        "refactor_production_same_line": _frame_records(filter_line(refactor_production)),
        "legacy_plan_builder_issues": issues,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    trace_path = REPORT_DIR / "m4_first_row_difference_trace.json"
    trace_path.write_text(json.dumps(trace, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return trace_path


def _write_report(report: dict) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORT_DIR / "m4_three_way_compare.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = ["# M4 三方对比报告", "", f"- M3 run_id: `{M3_RUN_ID}`", f"- 范围: {START_DATE} 至 {END_DATE}", f"- 旧 M4 跨日状态: {LEGACY_STATE_POLICY}", f"- 重构 M4 跨日状态: {REFACTOR_STATE_POLICY}", "- 输入: PostgreSQL 中真实 M3 历史业务输出；运行计时不包含数据库读取或配置初始化。", "- 性能: 每个实现仅执行一次完整五日对比，记录该次实际端到端计算耗时。", "", "## 性能", "", "| 实现 | min(s) | median(s) | mean(s) | 每日 mean(s) | 相对旧 M4 |", "|---|---:|---:|---:|---:|---:|"]
    for name, value in report["performance"].items():
        lines.append(f"| {name} | {value['min_seconds']:.4f} | {value['median_seconds']:.4f} | {value['mean_seconds']:.4f} | {value['per_day_mean_seconds']:.4f} | {value['relative_to_legacy']:.2f}x |")
    lines.extend(["", "## 差异说明", "", "业务差异仅报告；绝对差 $\\le 10^{-6}$ 或零附近微小差异归类为精度问题。完整明细见同目录 JSON。"])
    markdown_path = REPORT_DIR / "m4_three_way_compare.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return markdown_path


@pytest.fixture(scope="module")
def comparison_data():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text("", encoding="utf-8")
    db = _db()
    try:
        _progress("开始加载数据库配置与五日真实 M3 历史业务数据")
        db.connect()
        config = _load_prepared_config(db)
        m3_by_date = _load_m3_by_sim_date(db)
    finally:
        db.close()
    _progress("数据库输入加载完成")

    _progress("开始旧 M4 唯一一次真实对比运行并计时")
    legacy = _run_legacy(config, m3_by_date, timed=True)
    _progress("开始重构 M4 pandas 唯一一次真实对比运行并计时")
    pandas_result = _run_refactor(config, m3_by_date, engine="pandas", timed=True)
    _progress("开始重构 M4 polars 唯一一次真实对比运行并计时")
    polars_result = _run_refactor(config, m3_by_date, engine="polars", timed=True)
    performance = {
        "legacy": _performance_summary(legacy),
        "refactor_pandas": _performance_summary(pandas_result),
        "refactor_polars": _performance_summary(polars_result),
    }
    legacy_mean = performance["legacy"]["mean_seconds"]
    for summary in performance.values():
        summary["relative_to_legacy"] = summary["mean_seconds"] / legacy_mean
    report = {
        "run_id": M3_RUN_ID,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "state_policies": {
            "legacy": LEGACY_STATE_POLICY,
            "refactor": REFACTOR_STATE_POLICY,
        },
        "performance": performance,
        "comparisons": {
            "legacy_vs_refactor_pandas": [
                {"date": left["date"], "comparison": _compare_day(left, right, "legacy_vs_pandas")}
                for left, right in zip(legacy["days"], pandas_result["days"])
            ],
            "refactor_pandas_vs_polars": [
                {"date": left["date"], "comparison": _compare_day(left, right, "pandas_vs_polars")}
                for left, right in zip(pandas_result["days"], polars_result["days"])
            ],
        },
    }
    trace_path = _write_first_row_difference_trace(config, m3_by_date, legacy, pandas_result, report)
    if trace_path is not None:
        report["first_row_difference_trace"] = str(trace_path)
    report_path = _write_report(report)
    _progress(f"报告已写入: {report_path}")
    return {"legacy": legacy, "pandas": pandas_result, "polars": polars_result,
            "report": report, "report_path": report_path}


def test_m4_three_way_compare_covers_five_days(comparison_data):
    """校验三个实现均完成固定五天；业务差异仅由报告承载。"""
    for name in ("legacy", "pandas", "polars"):
        assert len(comparison_data[name]["days"]) == 5, name
    assert comparison_data["report_path"].exists()
    print(f"\nM4 三方差异与性能报告: {comparison_data['report_path']}")


def test_m4_three_way_compare_records_actual_execution_timing(comparison_data):
    """报告记录实际对比运行的单次端到端耗时，不重复执行计算。"""
    performance = comparison_data["report"]["performance"]
    assert set(performance) == {"legacy", "refactor_pandas", "refactor_polars"}
    assert all(len(summary["runs"]) == 1 for summary in performance.values())
