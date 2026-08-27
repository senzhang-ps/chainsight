"""冻结 chainsight-main legacy M1 输出，验证 refactor M4→M5→M6→M3。

不执行 refactor M1；从指定 legacy 历史 run 提取 M1 日度结果并注入完整集成调度，
再逐日将 refactor 下游输出与同一 legacy run 的 M4–M3 持久化输出比较。

运行：
    $env:RUN_FIXED_LEGACY_M1_DOWNSTREAM='1'
    conda run --no-capture-output -n work pytest tests/regression/test_fixed_legacy_m1_downstream_parity.py -s -q
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
from src.modules.deployment_planning import main as legacy_m5_main
from tests.helpers.compare_utils import compare_dataframes_by_key, dataframe_difference_details


PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
TEST_ENTRY_DIR = PROJECT_ROOT / "test"
if str(TEST_ENTRY_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_ENTRY_DIR))
from test.test_integration import run_integrated_simulation

START_DATE = os.getenv("FIXED_LEGACY_M1_START_DATE", "2025-12-15")
END_DATE = os.getenv("FIXED_LEGACY_M1_END_DATE", "2025-12-25")
LEGACY_DATABASE = os.getenv("FIXED_LEGACY_M1_DATABASE", "test_bc")
LEGACY_SCHEMA = os.getenv("FIXED_LEGACY_M1_SCHEMA", "input")
LEGACY_RUN_ID = os.getenv(
    "FIXED_LEGACY_M1_RUN_ID",
    "db_OC_Paste_S1_20251224_repare_20260826_130441",
)
REFACTOR_ENGINE = os.getenv("FIXED_LEGACY_M1_REFACTOR_ENGINE", "pandas")
CONFIG_PATH = Path(os.getenv(
    "FIXED_LEGACY_M1_CONFIG_PATH",
    str(PROJECT_ROOT / "input" / "OC" / "OC_Paste_S1_20251224_extension" / "OC_Paste_S1_20251224_repare.xlsx"),
))
REPORT_BASE = PROJECT_ROOT / "outputs" / "fixed_legacy_m1_downstream_parity"
META_COLUMNS = ["run_id", "sim_date", "config_name", "db_write_time", "file_date"]
DOWNSTREAM_MODULES = ("module4", "module5", "module6", "module3")
M5_TRACE_MATERIAL = "21307836"
M5_TRACE_DAY = "2025-12-24"
NON_BUSINESS_OUTPUTS = {("module5", "validation_log")}
M3_STATE_TABLES = {
    "unrestricted_inventory": "orchestrator_unrestricted_inventory",
    "planning_intransit": "orchestrator_planning_intransit",
    "delivery_gr": "orchestrator_delivery_gr",
    "production_gr": "orchestrator_production_gr",
    "production_plan_backlog": "orchestrator_production_plan_backlog",
    "open_deployment": "orchestrator_open_deployment",
    "shipment_log": "orchestrator_shipment_log",
    "delivery_shipment_log": "orchestrator_delivery_shipment_log",
}
M5_CONTEXT_TABLES = {
    "unrestricted_inventory": "orchestrator_unrestricted_inventory",
    "planning_intransit": "orchestrator_planning_intransit",
    "open_deployment": "orchestrator_open_deployment",
    "space_quota": "orchestrator_space_quota",
}


def _db() -> DatabaseConnection:
    settings = get_database_config()
    return DatabaseConnection(
        host=os.getenv("FIXED_LEGACY_M1_HOST", settings["host"]),
        port=int(os.getenv("FIXED_LEGACY_M1_PORT", str(settings["port"]))),
        database=LEGACY_DATABASE,
        user=settings["user"], password=settings["password"], schema=LEGACY_SCHEMA,
        auto_create_schema=False,
    )


def _read(db: DatabaseConnection, table: str, day: pd.Timestamp, *, active_orders: bool = False) -> pd.DataFrame:
    columns = [row[0] for row in db.execute_query(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema=%s AND table_name=%s ORDER BY ordinal_position",
        (LEGACY_SCHEMA, table),
    )]
    if not columns:
        pytest.skip(f"缺少 legacy 历史表: {LEGACY_SCHEMA}.{table}")
    qualified = f'"{LEGACY_SCHEMA}"."{table}"'
    if active_orders:
        sql = (
            f"SELECT {', '.join(columns)} FROM {qualified} "
            "WHERE run_id=%s AND sim_date::date<=%s::date AND date::date>=%s::date"
        )
        args = (LEGACY_RUN_ID, day.strftime("%Y-%m-%d"), day.strftime("%Y-%m-%d"))
    else:
        sql = f"SELECT {', '.join(columns)} FROM {qualified} WHERE run_id=%s AND sim_date::date=%s::date"
        args = (LEGACY_RUN_ID, day.strftime("%Y-%m-%d"))
    return pd.DataFrame(db.execute_query(sql, args), columns=columns).drop(columns=META_COLUMNS, errors="ignore")


def _load_history() -> tuple[dict[str, dict], dict[str, dict[str, dict[str, pd.DataFrame]]]]:
    """读取冻结 M1 合同和同一 run 的 legacy 下游 Oracle。"""
    db = _db()
    db.connect()
    try:
        m1_history: dict[str, dict] = {}
        downstream_oracle: dict[str, dict[str, dict[str, pd.DataFrame]]] = {}
        for day in pd.date_range(START_DATE, END_DATE, freq="D"):
            date_text = day.strftime("%Y-%m-%d")
            m1_tables = OUTPUT_REGISTRY["module1"]
            m1_history[date_text] = {
                "orders_df": _read(db, m1_tables["orders_df"], day),
                "all_orders_for_next_day": _read(db, m1_tables["orders_df"], day, active_orders=True),
                "shipment_df": _read(db, m1_tables["shipment_df"], day),
                "cut_df": _read(db, m1_tables["cut_df"], day),
                "supply_demand_df": _read(db, m1_tables["supply_demand_df"], day),
                "summary_df": _read(db, m1_tables["summary_df"], day),
            }
            downstream_oracle[date_text] = {
                module_id: {
                    output: _read(db, table, day)
                    for output, table in OUTPUT_REGISTRY[module_id].items()
                }
                for module_id in DOWNSTREAM_MODULES
            }
    finally:
        db.close()

    missing = [date for date, result in m1_history.items() if result["supply_demand_df"].empty]
    if missing:
        pytest.skip(f"legacy M1 历史输入缺少 SupplyDemandLog: {', '.join(missing)}")
    return m1_history, downstream_oracle


def _compare(current: pd.DataFrame, legacy: pd.DataFrame, module: str, output: str, label: str) -> dict:
    current = pd.DataFrame(current).drop(columns=["order_id"], errors="ignore")
    legacy = pd.DataFrame(legacy).drop(columns=["order_id"], errors="ignore")
    # Polars 返回 ``date``，legacy 表读取为 ``Timestamp``。业务比较只关心
    # 自然日，先统一所有日期字段以避免同日的类型/时间分量伪差异。
    for frame in (current, legacy):
        for column in frame.columns:
            if "date" in str(column).casefold():
                parsed = pd.to_datetime(frame[column], errors="coerce")
                if parsed.notna().any():
                    frame[column] = parsed.dt.normalize()
    # 动态输出表在没有业务行时可能没有可共同关联的列；两侧同为空是
    # 一致的模块行为，不应被比较器的“无共有业务键”诊断误判为失败。
    if current.empty and legacy.empty:
        return {
            "label": label,
            "left_rows": 0,
            "right_rows": 0,
            "key_columns": [],
            "left_only_keys": 0,
            "right_only_keys": 0,
            "matched_rows": 0,
            "column_differences": {},
            "precision_differences": {},
            "schema": {
                "left_only_columns": sorted(set(current.columns) - set(legacy.columns)),
                "right_only_columns": sorted(set(legacy.columns) - set(current.columns)),
            },
            "both_empty": True,
        }
    key_columns = ["ori_deployment_uid", "vehicle_uid"] if (module, output) == ("module6", "delivery_plan") else None
    return compare_dataframes_by_key(
        current,
        legacy,
        label=label,
        key_columns=key_columns,
    )


def _consistent(comparison: dict) -> bool:
    return not (
        comparison.get("error")
        or comparison.get("left_only_keys")
        or comparison.get("right_only_keys")
        or comparison.get("column_differences")
        or comparison.get("precision_differences")
    )


def _m3_difference_keys(details: list[pd.DataFrame]) -> list[dict]:
    """提取 M3 的首层业务键，供状态输入审计使用。"""
    keys: list[dict] = []
    for detail in details:
        rows = detail.loc[
            (detail["module"] == "module3")
            & (detail["output"] == "net_demand_df")
            & detail["difference_type"].isin(["left_only_row", "right_only_row", "value_difference"])
        ]
        for row in rows.itertuples(index=False):
            try:
                key = json.loads(row.key_values)
            except (TypeError, json.JSONDecodeError):
                continue
            if {"simulation_date", "material", "location"}.issubset(key):
                keys.append(key)
    return list({json.dumps(key, sort_keys=True): key for key in keys}.values())


def _filtered_state(frame: pd.DataFrame, *, material: str, location: str) -> pd.DataFrame:
    """按 M3 供给视图的物料/节点语义筛选并聚合数量。"""
    frame = pd.DataFrame(frame).copy()
    if frame.empty or "material" not in frame.columns:
        return pd.DataFrame([{"rows": 0, "quantity_sum": 0.0}])
    rows = frame.loc[frame["material"].astype(str).eq(str(material))].copy()
    location_columns = [column for column in ("location", "receiving", "sending") if column in rows]
    if location_columns:
        mask = pd.Series(False, index=rows.index)
        for column in location_columns:
            mask |= rows[column].astype(str).eq(str(location))
        rows = rows.loc[mask]
    quantity_column = next(
        (column for column in ("quantity", "deployed_qty", "produced_qty", "con_planned_qty") if column in rows),
        None,
    )
    quantity = 0.0 if quantity_column is None else float(
        pd.to_numeric(rows[quantity_column], errors="coerce").fillna(0).sum()
    )
    return pd.DataFrame([{"rows": len(rows), "quantity_sum": quantity}])


def _legacy_all_production(db: DatabaseConnection, day: pd.Timestamp) -> pd.DataFrame:
    """按 legacy ``get_all_production_view()`` 口径重建 M3 的生产输入。"""
    production_gr = _read(db, M3_STATE_TABLES["production_gr"], day)
    backlog = _read(db, M3_STATE_TABLES["production_plan_backlog"], day)
    columns = ["material", "location", "available_date", "quantity"]
    if not production_gr.empty:
        today = production_gr.rename(columns={"date": "available_date"}).reindex(columns=columns)
    else:
        today = pd.DataFrame(columns=columns)
    if not backlog.empty and "available_date" in backlog:
        future = backlog.copy()
        future["available_date"] = pd.to_datetime(future["available_date"], errors="coerce").dt.normalize()
        future = future.loc[future["available_date"].ge(day.normalize())].reindex(columns=columns)
    else:
        future = pd.DataFrame(columns=columns)
    result = pd.concat([today, future], ignore_index=True)
    if result.empty:
        return pd.DataFrame(columns=columns)
    result["quantity"] = pd.to_numeric(result["quantity"], errors="coerce").fillna(0)
    return result.groupby(["material", "location", "available_date"], as_index=False)["quantity"].sum()


def _m3_state_diagnosis(
    current_snapshots: list[dict],
    difference_keys: list[dict],
    refactor_all_production: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """对比 M3 首差键在 refactor / legacy 的输入状态聚合。"""
    db = _db()
    db.connect()
    records: list[dict] = []
    try:
        snapshots = {
            pd.Timestamp(snapshot["simulation_date"]).strftime("%Y-%m-%d"): snapshot["views"]
            for snapshot in current_snapshots
        }
        for key in difference_keys:
            date_text = str(key["simulation_date"])
            material, location = str(key["material"]), str(key["location"])
            for view_name, table_name in M3_STATE_TABLES.items():
                legacy = _read(db, table_name, pd.Timestamp(date_text))
                refactor = snapshots.get(date_text, {}).get(view_name, pd.DataFrame())
                left = _filtered_state(refactor, material=material, location=location).iloc[0]
                right = _filtered_state(legacy, material=material, location=location).iloc[0]
                records.append({
                    "simulation_date": date_text,
                    "material": material,
                    "location": location,
                    "view": view_name,
                    "refactor_rows": int(left["rows"]),
                    "legacy_rows": int(right["rows"]),
                    "refactor_quantity_sum": float(left["quantity_sum"]),
                    "legacy_quantity_sum": float(right["quantity_sum"]),
                    "quantity_delta_refactor_minus_legacy": float(left["quantity_sum"] - right["quantity_sum"]),
                })
            legacy_production = _legacy_all_production(db, pd.Timestamp(date_text))
            refactor_production = refactor_all_production.get(date_text, pd.DataFrame())
            left = _filtered_state(refactor_production, material=material, location=location).iloc[0]
            right = _filtered_state(legacy_production, material=material, location=location).iloc[0]
            records.append({
                "simulation_date": date_text,
                "material": material,
                "location": location,
                "view": "all_production_for_m3",
                "refactor_rows": int(left["rows"]),
                "legacy_rows": int(right["rows"]),
                "refactor_quantity_sum": float(left["quantity_sum"]),
                "legacy_quantity_sum": float(right["quantity_sum"]),
                "quantity_delta_refactor_minus_legacy": float(left["quantity_sum"] - right["quantity_sum"]),
            })
    finally:
        db.close()
    return pd.DataFrame(records)


def _m3_production_detail_diagnosis(
    refactor_all_production: dict[str, pd.DataFrame],
    refactor_backlog: dict[str, pd.DataFrame],
    refactor_m4_production: dict[str, pd.DataFrame],
    difference_keys: list[dict],
) -> pd.DataFrame:
    """保留生产输入原始行，避免聚合相同却日期过滤不同而无法定位。"""
    db = _db()
    db.connect()
    records: list[dict] = []
    try:
        for key in difference_keys:
            date_text = str(key["simulation_date"])
            day = pd.Timestamp(date_text)
            material, location = str(key["material"]), str(key["location"])
            legacy_backlog = _read(db, M3_STATE_TABLES["production_plan_backlog"], day)
            legacy_all_production = _legacy_all_production(db, day)
            frames = {
                "refactor_all_production": refactor_all_production.get(date_text, pd.DataFrame()),
                "refactor_backlog": refactor_backlog.get(date_text, pd.DataFrame()),
                "refactor_m4_production": refactor_m4_production.get(date_text, pd.DataFrame()),
                "legacy_all_production": legacy_all_production,
                "legacy_backlog": legacy_backlog,
            }
            for source, frame in frames.items():
                rows = pd.DataFrame(frame)
                if "material" in rows:
                    rows = rows.loc[rows["material"].astype(str).eq(material)]
                if "location" in rows:
                    rows = rows.loc[rows["location"].astype(str).eq(location)]
                records.append({
                    "simulation_date": date_text,
                    "material": material,
                    "location": location,
                    "source": source,
                    "rows": rows.to_json(orient="records", date_format="iso"),
                })
    finally:
        db.close()
    return pd.DataFrame(records)


def _m3_production_timeline(
    refactor_all_production: dict[str, pd.DataFrame],
    refactor_backlog: dict[str, pd.DataFrame],
    difference_keys: list[dict],
) -> pd.DataFrame:
    """首差物料按全仿真日记录 M3 生产账本，找出状态首次偏离日。"""
    targets = {(str(key["material"]), str(key["location"])) for key in difference_keys}
    if not targets:
        return pd.DataFrame()
    db = _db()
    db.connect()
    records: list[dict] = []
    try:
        for day in pd.date_range(START_DATE, END_DATE, freq="D"):
            date_text = day.strftime("%Y-%m-%d")
            legacy_all_production = _legacy_all_production(db, day)
            legacy_backlog = _read(db, M3_STATE_TABLES["production_plan_backlog"], day)
            for material, location in sorted(targets):
                for source, frame in {
                    "refactor_all_production": refactor_all_production.get(date_text, pd.DataFrame()),
                    "refactor_backlog": refactor_backlog.get(date_text, pd.DataFrame()),
                    "legacy_all_production": legacy_all_production,
                    "legacy_backlog": legacy_backlog,
                }.items():
                    summary = _filtered_state(frame, material=material, location=location).iloc[0]
                    records.append({
                        "simulation_date": date_text,
                        "material": material,
                        "location": location,
                        "source": source,
                        "rows": int(summary["rows"]),
                        "quantity_sum": float(summary["quantity_sum"]),
                    })
    finally:
        db.close()
    return pd.DataFrame(records)


def _m3_planning_facts_diagnosis(
    planning_facts: dict[str, dict],
    difference_keys: list[dict],
) -> pd.DataFrame:
    """输出每个 M3 首差键的 M5 共享需求和 horizon 事实。"""
    records: list[dict] = []
    for key in difference_keys:
        date_text = str(key["simulation_date"])
        material, location = str(key["material"]), str(key["location"])
        facts = planning_facts.get(date_text, {})
        direct = pd.DataFrame(facts.get("direct_demand", pd.DataFrame()))
        horizon = pd.DataFrame(facts.get("node_horizon", pd.DataFrame()))
        direct_rows = direct.loc[
            direct.get("material", pd.Series("", index=direct.index)).astype(str).eq(material)
            & direct.get("node", pd.Series("", index=direct.index)).astype(str).eq(location)
        ]
        horizon_rows = horizon.loc[
            horizon.get("material", pd.Series("", index=horizon.index)).astype(str).eq(material)
            & horizon.get("node", pd.Series("", index=horizon.index)).astype(str).eq(location)
        ]
        records.append({
            "simulation_date": date_text,
            "material": material,
            "location": location,
            "direct_demand_rows": direct_rows.to_json(orient="records", date_format="iso"),
            "node_horizon_rows": horizon_rows.to_json(orient="records", date_format="iso"),
        })
    return pd.DataFrame(records)


def _m3_propagation_diagnosis(
    diagnostics_by_date: dict[str, list[dict]], difference_keys: list[dict],
) -> pd.DataFrame:
    """保留首差物料全部节点的 refactor M3 shortage→parent 传播链。"""
    targets = {(str(key["simulation_date"]), str(key["material"])) for key in difference_keys}
    frames: list[pd.DataFrame] = []
    for (date_text, material) in sorted(targets):
        diagnostics = diagnostics_by_date.get(date_text, [])
        frame = pd.DataFrame(diagnostics)
        if frame.empty:
            continue
        frame = frame.loc[frame["material"].astype(str).eq(material)].copy()
        frame.insert(0, "simulation_date", date_text)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def test_refactor_downstream_matches_legacy_with_fixed_legacy_m1() -> None:
    # 测试目的：验证“refactor、downstream、matches、legacy、with、fixed、legacy、m1”场景下订单生成、库存消耗与发运数据的一致性。
    # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `mkdir()`，再通过 2 个断言核对结果、状态或异常条件。
    # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
    # 业务逻辑和原因：该校验防止订单生成、库存消耗与发运数据的一致性在重构、引擎切换或跨日运行时发生静默偏差。
    if os.getenv("RUN_FIXED_LEGACY_M1_DOWNSTREAM") != "1":
        pytest.skip("设置 RUN_FIXED_LEGACY_M1_DOWNSTREAM=1 后运行真实 legacy M1 固定下游验证")
    if not CONFIG_PATH.exists():
        pytest.skip(f"找不到 refactor 运行配置: {CONFIG_PATH}")

    report_dir = REPORT_BASE / time.strftime("run_%Y%m%d_%H%M%S")
    report_dir.mkdir(parents=True, exist_ok=False)
    m1_history, oracle = _load_history()
    observed_planning_facts: dict[str, dict] = {}
    observed_m3_diagnostics: dict[str, list[dict]] = {}
    observed_m3_all_production: dict[str, pd.DataFrame] = {}
    observed_m3_backlog: dict[str, pd.DataFrame] = {}
    observed_m4_production: dict[str, pd.DataFrame] = {}
    observed_m5_plan_trace: dict[str, pd.DataFrame] = {}
    same_state_legacy_m5: dict[str, object] = {}

    def legacy_m1_provider(day: pd.Timestamp) -> dict:
        return m1_history[day.strftime("%Y-%m-%d")]

    def observe_module_input(module_id: str, day: pd.Timestamp, module: object, context) -> None:
        if module_id == "module5":
            module.capture_plan_trace = True
            if day.strftime("%Y-%m-%d") == M5_TRACE_DAY:
                trace_rows: list[dict] = []
                original_allocate = legacy_m5_main._allocate_pipeline_sources

                def traced_allocate(demand_rows, adjusted_qtys, loc, mat, *args, **kwargs):
                    if str(mat) == M5_TRACE_MATERIAL and str(loc) == "A668":
                        trace_rows.extend({
                            "phase": "before", "location": str(loc), "row": dict(item),
                            "adjusted_qty": int(adjusted_qtys.get(index, item.get("demand_qty", 0))),
                        } for index, item in enumerate(demand_rows))
                    result = original_allocate(demand_rows, adjusted_qtys, loc, mat, *args, **kwargs)
                    if str(mat) == M5_TRACE_MATERIAL and str(loc) == "A668":
                        trace_rows.extend({
                            "phase": "after", "location": str(loc), "row": dict(item),
                            "adjusted_qty": int(adjusted_qtys.get(index, item.get("demand_qty", 0))),
                        } for index, item in enumerate(demand_rows))
                    return result

                legacy_m5_main._allocate_pipeline_sources = traced_allocate
                try:
                    same_state_legacy_m5["result"] = legacy_m5_main.run_daily_deployment_planning(
                        config_dict=module.orchestrator.all_config,
                        orchestrator=context,
                        current_date=day.strftime("%Y-%m-%d"),
                        skip_file_output=True,
                        module1_result=m1_history[day.strftime("%Y-%m-%d")],
                        module4_result=observed_m4_production.get(day.strftime("%Y-%m-%d"), {}),
                    )
                    same_state_legacy_m5["trace"] = trace_rows
                finally:
                    legacy_m5_main._allocate_pipeline_sources = original_allocate
            return
        if module_id != "module3":
            return
        facts = context.get_planning_facts(day)
        observed_planning_facts[day.strftime("%Y-%m-%d")] = {
            name: value.copy(deep=True) if isinstance(value, pd.DataFrame) else value
            for name, value in facts.items()
        }
        production_view = module._backend.supply_views(context, day)["all_production"]
        observed_m3_all_production[day.strftime("%Y-%m-%d")] = (
            production_view.to_pandas()
            if hasattr(production_view, "to_pandas")
            else production_view.copy(deep=True)
        )
        observed_m3_backlog[day.strftime("%Y-%m-%d")] = pd.DataFrame(
            context.production_plan_backlog
        ).copy(deep=True)

    def observe_module_result(module_id: str, day: pd.Timestamp, module: object, result: dict, context) -> None:
        if module_id == "module4":
            observed_m4_production[day.strftime("%Y-%m-%d")] = result.get(
                "production_df", pd.DataFrame()
            ).copy(deep=True)
            return
        if module_id == "module5":
            frames = getattr(module._backend, "last_plan_layer_demands", [])
            trace = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
            if not trace.empty and "material" in trace:
                trace = trace.loc[trace["material"].astype(str).eq(M5_TRACE_MATERIAL)].copy()
            observed_m5_plan_trace[day.strftime("%Y-%m-%d")] = trace
            module.capture_plan_trace = False
            return
        if module_id != "module3":
            return
        diagnostics = getattr(getattr(module, "_backend", None), "last_layer_diagnostics", [])
        observed_m3_diagnostics[day.strftime("%Y-%m-%d")] = [dict(row) for row in diagnostics]

    current = run_integrated_simulation(
        config_path=str(CONFIG_PATH),
        start_date=START_DATE,
        end_date=END_DATE,
        output_base_dir=str(report_dir / "scratch"),
        engine=REFACTOR_ENGINE,
        skip_dq=True,
        enable_persistence=False,
        module_result_providers={"module1": legacy_m1_provider},
        module_input_observer=observe_module_input,
        module_result_observer=observe_module_result,
    )

    comparisons: list[dict] = []
    context_comparisons: list[dict] = []
    details: list[pd.DataFrame] = []
    db = _db()
    db.connect()
    try:
        for offset, day in enumerate(pd.date_range(START_DATE, END_DATE, freq="D")):
            date_text = day.strftime("%Y-%m-%d")
            current_context = current["context_snapshots"][offset]["views"]
            for view_name, table_name in M5_CONTEXT_TABLES.items():
                actual = current_context.get(view_name, pd.DataFrame())
                expected = _read(db, table_name, day)
                comparison = _compare(actual, expected, "context", view_name, f"{date_text}:context:{view_name}")
                context_comparisons.append({
                    "date": date_text,
                    "view": view_name,
                    "refactor_rows": len(actual),
                    "legacy_rows": len(expected),
                    "consistent": _consistent(comparison),
                    "comparison": comparison,
                })
    finally:
        db.close()
    for offset, day in enumerate(pd.date_range(START_DATE, END_DATE, freq="D")):
        date_text = day.strftime("%Y-%m-%d")
        for module_id in DOWNSTREAM_MODULES:
            for output in OUTPUT_REGISTRY[module_id]:
                actual = current["results"][module_id][offset].get(output, pd.DataFrame())
                expected = oracle[date_text][module_id][output]
                comparison = _compare(actual, expected, module_id, output, f"{date_text}:{module_id}:{output}")
                is_consistent = _consistent(comparison)
                comparisons.append({
                    "date": date_text,
                    "module": module_id,
                    "output": output,
                    "refactor_rows": len(actual),
                    "legacy_rows": len(expected),
                    "consistent": is_consistent,
                    "comparison": comparison,
                })
                if not is_consistent:
                    detail = dataframe_difference_details(
                        actual.drop(columns=["order_id"], errors="ignore"),
                        expected.drop(columns=["order_id"], errors="ignore"),
                        key_columns=comparison.get("key_columns") or None,
                    )
                    detail.insert(0, "output", output)
                    detail.insert(0, "module", module_id)
                    detail.insert(0, "date", date_text)
                    details.append(detail)

    mismatches = [
        item for item in comparisons
        if not item["consistent"]
        and (item["module"], item["output"]) not in NON_BUSINESS_OUTPUTS
    ]
    if details:
        pd.concat(details, ignore_index=True).to_csv(
            report_dir / "downstream_differences.csv", index=False, encoding="utf-8-sig"
        )
    m3_diagnostic = _m3_state_diagnosis(
        current["context_snapshots"], _m3_difference_keys(details), observed_m3_all_production,
    )
    if not m3_diagnostic.empty:
        m3_diagnostic.to_csv(
            report_dir / "m3_difference_input_state.csv", index=False, encoding="utf-8-sig"
        )
    m3_production_details = _m3_production_detail_diagnosis(
        observed_m3_all_production, observed_m3_backlog, observed_m4_production,
        _m3_difference_keys(details),
    )
    if not m3_production_details.empty:
        m3_production_details.to_csv(
            report_dir / "m3_difference_production_detail.csv", index=False, encoding="utf-8-sig"
        )
    m3_production_timeline = _m3_production_timeline(
        observed_m3_all_production, observed_m3_backlog, _m3_difference_keys(details),
    )
    if not m3_production_timeline.empty:
        m3_production_timeline.to_csv(
            report_dir / "m3_difference_production_timeline.csv", index=False, encoding="utf-8-sig"
        )
    m3_planning_facts = _m3_planning_facts_diagnosis(
        observed_planning_facts, _m3_difference_keys(details),
    )
    if not m3_planning_facts.empty:
        m3_planning_facts.to_csv(
            report_dir / "m3_difference_planning_facts.csv", index=False, encoding="utf-8-sig"
        )
    m3_propagation = _m3_propagation_diagnosis(
        observed_m3_diagnostics, _m3_difference_keys(details),
    )
    if not m3_propagation.empty:
        m3_propagation.to_csv(
            report_dir / "m3_difference_propagation.csv", index=False, encoding="utf-8-sig"
        )
    m5_trace = observed_m5_plan_trace.get("2025-12-24", pd.DataFrame())
    if not m5_trace.empty:
        m5_trace.to_csv(
            report_dir / "m5_20251224_21307836_plan_trace.csv", index=False, encoding="utf-8-sig"
        )
    same_state_legacy_plan = pd.DataFrame(
        same_state_legacy_m5.get("result", {}).get("deployment_plan", pd.DataFrame())
    )
    if not same_state_legacy_plan.empty:
        same_state_legacy_plan.loc[
            same_state_legacy_plan.get("material", pd.Series("", index=same_state_legacy_plan.index)).astype(str).eq(M5_TRACE_MATERIAL)
        ].to_csv(
            report_dir / "m5_20251224_same_state_legacy_plan.csv", index=False, encoding="utf-8-sig"
        )
    same_state_legacy_trace = pd.DataFrame(same_state_legacy_m5.get("trace", []))
    if not same_state_legacy_trace.empty:
        same_state_legacy_trace.to_csv(
            report_dir / "m5_20251224_same_state_legacy_pipeline_trace.csv", index=False, encoding="utf-8-sig"
        )
    report_path = report_dir / "fixed_legacy_m1_downstream_parity.json"
    report_path.write_text(json.dumps({
        "policy": (
            "冻结 chainsight-main legacy M1 历史结果；refactor 不实例化或执行 M1，"
            "只运行 M4→M5→M6→M3，并与同一 legacy run 的下游输出比较；"
            "M5 validation_log 仅为实现诊断合同，不纳入业务等价性验收"
        ),
        "legacy_schema": LEGACY_SCHEMA,
        "legacy_database": LEGACY_DATABASE,
        "legacy_run_id": LEGACY_RUN_ID,
        "refactor_config_path": str(CONFIG_PATH),
        "date_range": [START_DATE, END_DATE],
        "refactor_engine": REFACTOR_ENGINE,
        "database_writes": False,
        "fixed_m1_rows": {
            date: {name: len(frame) for name, frame in result.items() if isinstance(frame, pd.DataFrame)}
            for date, result in m1_history.items()
        },
        "comparison_count": len(comparisons),
        "mismatch_count": len(mismatches),
        "ignored_non_business_outputs": sorted(
            f"{module}.{output}" for module, output in NON_BUSINESS_OUTPUTS
        ),
        "context_comparison_count": len(context_comparisons),
        "context_mismatch_count": sum(not item["consistent"] for item in context_comparisons),
        "m3_difference_input_state_rows": len(m3_diagnostic),
        "m3_difference_production_detail_rows": len(m3_production_details),
        "m3_difference_production_timeline_rows": len(m3_production_timeline),
        "m3_difference_planning_facts_rows": len(m3_planning_facts),
        "m3_difference_propagation_rows": len(m3_propagation),
        "m5_trace_material": M5_TRACE_MATERIAL,
        "m5_trace_rows": len(m5_trace),
        "m5_same_state_legacy_plan_rows": len(same_state_legacy_plan),
        "m5_same_state_legacy_pipeline_trace_rows": len(same_state_legacy_trace),
        "comparisons": comparisons,
        "context_comparisons": context_comparisons,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[fixed legacy M1 downstream parity] report: {report_path}", flush=True)

    assert current["dates_processed"] == len(pd.date_range(START_DATE, END_DATE, freq="D"))
    assert not mismatches, f"固定 legacy M1 后下游存在差异；报告: {report_path}"
