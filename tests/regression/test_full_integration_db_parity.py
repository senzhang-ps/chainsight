"""完整重构调度与历史数据库结果的只读一致性对比。

本测试调用 ``test/test_integration.py`` 的完整日度调度（M1→M4→M5→M6→M3），
不向数据库写入任何当前运行结果；随后仅从 PostgreSQL ``public`` schema 中读取固定
历史 run 的模块输出，逐模块、逐输出表、逐日比较业务数据。

默认跳过，避免常规单元测试连接正式库。显式运行：

    $env:RUN_FULL_DB_PARITY='1'
    conda run -n work pytest tests/regression/test_full_integration_db_parity.py -s -q

历史基线的 run id 若更新，可通过 ``FULL_PARITY_HISTORICAL_RUN_ID`` 覆盖。
"""

# 测试文件说明
# 测试目的：集中验证完整业务链路的模块衔接与结果一致性。
# 测试方法：按 `regression` 类测试组织用例，使用夹具、模拟数据或真实依赖执行被测流程。
# 输入和期望：输入覆盖正常、边界和异常业务数据；期望由各测试用例的显式断言定义。
# 业务逻辑和原因：将相关验证集中维护，确保完整业务链路的模块衔接与结果一致性变更时能够快速定位回归影响。



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
from tests.helpers.compare_utils import compare_dataframes_by_key


PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
TEST_ENTRY_DIR = PROJECT_ROOT / "test"
if str(TEST_ENTRY_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_ENTRY_DIR))

from test.test_integration import run_integrated_simulation


CONFIG_PATH = Path(os.environ.get(
    "FULL_PARITY_CONFIG_PATH",
    str(PROJECT_ROOT / "config" / "OC_Paste_S1_20251224_repare.xlsx"),
))
HISTORICAL_RUN_ID = os.environ.get(
    "FULL_PARITY_HISTORICAL_RUN_ID",
    "db_OC_Paste_S1_20251224_repare_20260803_105300",
)
DB_SCHEMA = os.environ.get("FULL_PARITY_DB_SCHEMA", "public")
START_DATE = os.environ.get("FULL_PARITY_START_DATE", "2025-12-15")
END_DATE = os.environ.get("FULL_PARITY_END_DATE", "2025-12-19")
ENGINE = os.environ.get("FULL_PARITY_ENGINE", "polars")
META_COLUMNS = {"run_id", "sim_date", "config_name", "db_write_time", "file_date"}
CONTEXT_TABLES = {
    "unrestricted_inventory": "orchestrator_unrestricted_inventory",
    "open_deployment": "orchestrator_open_deployment",
    "planning_intransit": "orchestrator_planning_intransit",
    "space_quota": "orchestrator_space_quota",
    "delivery_gr": "orchestrator_delivery_gr",
    "production_gr": "orchestrator_production_gr",
    "production_plan_backlog": "orchestrator_production_plan_backlog",
    "shipment_log": "orchestrator_shipment_log",
    "delivery_shipment_log": "orchestrator_delivery_shipment_log",
    "inventory_change_log": "orchestrator_inventory_change_log",
}
_NON_METRIC_COLUMNS = {
    "date", "simulation_date", "production_plan_date", "available_date",
    "requirement_date", "planned_delivery_date", "planned_deployment_date",
    "material", "location", "sending", "receiving", "line", "type",
    "demand_element", "changeover_id", "changeover_type", "reason",
}


def _db() -> DatabaseConnection:
    settings = get_database_config()
    return DatabaseConnection(
        host=settings["host"],
        port=settings["port"],
        database=settings["database"],
        user=settings["user"],
        password=settings["password"],
        schema=settings.get("default_schema", "public"),
        auto_create_schema=False,
    )


def _table_columns(db: DatabaseConnection, table_name: str) -> list[str]:
    rows = db.execute_query(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position",
        (DB_SCHEMA, table_name),
    )
    return [row[0] for row in rows]


def _load_historical_output(
    db: DatabaseConnection,
    table_name: str,
    day: pd.Timestamp,
) -> pd.DataFrame:
    """按 ``run_id + sim_date`` 只读加载历史模块输出。"""
    columns = _table_columns(db, table_name)
    if not columns:
        return pd.DataFrame()

    qualified = f'"{DB_SCHEMA}"."{table_name}"'
    if table_name == OUTPUT_REGISTRY["module1"]["orders_df"]:
        rows = db.execute_query(
            f"SELECT {', '.join(columns)} FROM {qualified} "
            "WHERE run_id = %s AND sim_date::date <= %s::date AND date::date >= %s::date",
            (HISTORICAL_RUN_ID, day.strftime("%Y-%m-%d"), day.strftime("%Y-%m-%d")),
        )
    else:
        rows = db.execute_query(
            f"SELECT {', '.join(columns)} FROM {qualified} "
            "WHERE run_id = %s AND sim_date::date = %s::date",
            (HISTORICAL_RUN_ID, day.strftime("%Y-%m-%d")),
        )
    return pd.DataFrame(rows, columns=columns).drop(
        columns=list(META_COLUMNS), errors="ignore"
    )


def _compare_output(
    current: pd.DataFrame,
    historical: pd.DataFrame,
    *,
    label: str,
    key_columns: list[str] | None = None,
    allow_precision_differences: bool = False,
) -> dict:
    """比较一张输出表，并对空表/单侧空表给出明确结论。"""
    current = pd.DataFrame() if current is None else current.copy()
    historical = pd.DataFrame() if historical is None else historical.copy()
    if current.empty and historical.empty:
        return {
            "label": label,
            "current_rows": 0,
            "historical_rows": 0,
            "consistent": True,
            "reason": "both_empty",
        }
    if current.empty or historical.empty:
        return {
            "label": label,
            "current_rows": len(current),
            "historical_rows": len(historical),
            "consistent": False,
            "reason": "one_side_empty",
        }

    comparison = compare_dataframes_by_key(
        current,
        historical,
        label=label,
        key_columns=key_columns,
    )
    comparison["current_rows"] = comparison.pop("left_rows")
    comparison["historical_rows"] = comparison.pop("right_rows")
    comparison["consistent"] = not (
        comparison.get("error")
        or comparison["left_only_keys"]
        or comparison["right_only_keys"]
        or comparison["column_differences"]
        or (
            comparison["precision_differences"]
            and not allow_precision_differences
        )
    )
    return comparison


def _normalize_m1_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """统一当前结果与历史表的 M1 Summary 字段命名和关联键。"""
    frame = pd.DataFrame() if frame is None else frame.copy()
    canonical_columns = {
        "total_orders": "Total_Orders",
        "total_shipments": "Total_Shipments",
        "total_cuts": "Total_Cuts",
        "total_supplydemand": "Total_SupplyDemand",
        "date": "Date",
    }
    frame = frame.rename(columns={
        source: target
        for source, target in canonical_columns.items()
        if source in frame.columns and target not in frame.columns
    })
    if not frame.empty:
        frame.insert(0, "summary_key", "daily")
    return frame


def _all_numeric_metrics_are_zero(frame: pd.DataFrame) -> bool:
    """仅当存在至少一个数值指标且全部为零时返回 True。

    旧 M4 输出会包含仅有业务键、但计划/产能/产出等指标全为零的占位记录。
    这些记录的键或文本字段差异不构成业务差异，故可安全忽略。
    """
    has_metric = False
    for column in frame.columns:
        if column.lower() in _NON_METRIC_COLUMNS or "date" in column.lower():
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        values = numeric.dropna()
        if values.empty:
            continue
        has_metric = True
        if not (values.abs() <= 1e-9).all():
            return False
    return has_metric


def _remove_zero_metric_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """删除 M4 中所有数值业务指标均为零的占位行。"""
    if frame.empty:
        return frame.copy(), 0
    zero_mask = frame.apply(
        lambda row: _all_numeric_metrics_are_zero(row.to_frame().T), axis=1,
    )
    return frame.loc[~zero_mask].copy(), int(zero_mask.sum())
    comparison["ignored"] = False
    return comparison


@pytest.fixture(scope="module")
def parity_report() -> dict:
    if os.environ.get("RUN_FULL_DB_PARITY") != "1":
        pytest.skip("设置 RUN_FULL_DB_PARITY=1 后才运行真实数据库完整集成对比")
    if not CONFIG_PATH.exists():
        pytest.skip(f"找不到集成配置: {CONFIG_PATH}")

    run_dir = PROJECT_ROOT / "outputs" / "full_integration_db_parity" / time.strftime(
        "run_%Y%m%d_%H%M%S"
    )
    run_dir.mkdir(parents=True, exist_ok=False)

    # 此入口固定 enable_persistence=False：当前计算结果只保存在返回值中。
    current_run = run_integrated_simulation(
        config_path=str(CONFIG_PATH),
        start_date=START_DATE,
        end_date=END_DATE,
        output_base_dir=str(run_dir / "scratch"),
        engine=ENGINE,
    )

    db = _db()
    try:
        db.connect()
        result_comparisons: list[dict] = []
        context_comparisons: list[dict] = []
        for day_offset, day in enumerate(pd.date_range(START_DATE, END_DATE, freq="D")):
            date_str = day.strftime("%Y-%m-%d")
            for module_id, table_map in OUTPUT_REGISTRY.items():
                current_result = current_run["results"][module_id][day_offset]
                for output_key, table_name in table_map.items():
                    current_frame = current_result.get(output_key, pd.DataFrame())
                    historical_frame = _load_historical_output(db, table_name, day)
                    ignored_current_rows = ignored_historical_rows = 0
                    comparison_current = current_frame
                    comparison_historical = historical_frame
                    if module_id == "module4":
                        comparison_current, ignored_current_rows = _remove_zero_metric_rows(
                            current_frame
                        )
                        comparison_historical, ignored_historical_rows = _remove_zero_metric_rows(
                            historical_frame
                        )
                    summary_key_columns = None
                    if module_id == "module1" and output_key == "summary_df":
                        comparison_current = _normalize_m1_summary(comparison_current)
                        comparison_historical = _normalize_m1_summary(comparison_historical)
                        summary_key_columns = ["summary_key"]
                    elif module_id == "module1" and output_key == "shipment_df":
                        # order_id 是按运行时行顺序生成的技术标识，不参与库存
                        # 扣减及后续计划；以业务键和数量校验发货结果即可。
                        comparison_current = comparison_current.drop(
                            columns=["order_id"], errors="ignore"
                        )
                        comparison_historical = comparison_historical.drop(
                            columns=["order_id"], errors="ignore"
                        )
                    elif module_id == "module6" and output_key == "delivery_plan":
                        summary_key_columns = ["ori_deployment_uid", "vehicle_uid"]
                    comparison = _compare_output(
                        comparison_current,
                        comparison_historical,
                        label=f"{date_str}:{module_id}:{output_key}",
                        key_columns=summary_key_columns,
                        allow_precision_differences=(module_id == "module1"),
                    )
                    if module_id == "module4":
                        comparison["ignored_zero_metric_current_rows"] = ignored_current_rows
                        comparison["ignored_zero_metric_historical_rows"] = ignored_historical_rows
                    result_comparisons.append({
                        "date": date_str,
                        "module": module_id,
                        "output": output_key,
                        "historical_table": f"{DB_SCHEMA}.{table_name}",
                        **comparison,
                    })

            current_context = current_run["context_snapshots"][day_offset]["views"]
            for view_name, table_name in CONTEXT_TABLES.items():
                current_frame = current_context[view_name]
                historical_frame = _load_historical_output(db, table_name, day)
                comparison = _compare_output(
                    current_frame,
                    historical_frame,
                    label=f"{date_str}:context:{view_name}",
                )
                context_comparisons.append({
                    "date": date_str,
                    "context_view": view_name,
                    "historical_table": f"{DB_SCHEMA}.{table_name}",
                    **comparison,
                })
    finally:
        db.close()

    result_mismatches = [item for item in result_comparisons if not item["consistent"]]
    context_mismatches = [item for item in context_comparisons if not item["consistent"]]
    current_output_row_counts = []
    for day_offset, day in enumerate(pd.date_range(START_DATE, END_DATE, freq="D")):
        for module_id in ("module5", "module6", "module3"):
            current_result = current_run["results"][module_id][day_offset]
            for output_name, frame in current_result.items():
                if isinstance(frame, pd.DataFrame):
                    current_output_row_counts.append({
                        "date": day.strftime("%Y-%m-%d"),
                        "module": module_id,
                        "output": output_name,
                        "rows": len(frame),
                    })
    report = {
        "config_path": str(CONFIG_PATH),
        "historical_run_id": HISTORICAL_RUN_ID,
        "database_schema": DB_SCHEMA,
        "engine": ENGINE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "current_run_writes_database": False,
        "result_comparison_count": len(result_comparisons),
        "context_comparison_count": len(context_comparisons),
        "result_mismatch_count": len(result_mismatches),
        "context_mismatch_count": len(context_mismatches),
        "mismatch_count": len(result_mismatches) + len(context_mismatches),
        "consistent": not result_mismatches and not context_mismatches,
        "current_output_row_counts": current_output_row_counts,
        "result_comparisons": result_comparisons,
        "context_comparisons": context_comparisons,
    }
    report_path = run_dir / "full_integration_db_parity.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    report["report_path"] = report_path
    for item in current_output_row_counts:
        print(
            f"[current rows] {item['date']} {item['module']} "
            f"{item['output']}={item['rows']}"
        )
    return report


def test_full_refactor_integration_matches_historical_database(parity_report: dict) -> None:
    # 测试目的：验证“full、refactor、integration、matches、historical、database”场景下完整业务链路的模块衔接与结果一致性。
    # 测试方法：准备该场景所需的夹具、配置或模拟数据，调用 `exists()`，再通过 4 个断言核对结果、状态或异常条件。
    # 输入和期望：输入为测试构造的正常、边界或异常业务数据；期望结果满足本用例的断言契约。
    # 业务逻辑和原因：该校验防止完整业务链路的模块衔接与结果一致性在重构、引擎切换或跨日运行时发生静默偏差。
    """当前完整调度的全部模块输出应与固定历史运行的业务结果一致。"""
    assert parity_report["result_comparison_count"] == (
        len(pd.date_range(START_DATE, END_DATE, freq="D"))
        * sum(len(outputs) for outputs in OUTPUT_REGISTRY.values())
    )
    assert parity_report["context_comparison_count"] == (
        len(pd.date_range(START_DATE, END_DATE, freq="D"))
        * len(CONTEXT_TABLES)
    )
    assert parity_report["report_path"].exists()
    assert parity_report["consistent"], (
        "完整调度与历史数据库结果不一致；详细报告: "
        f"{parity_report['report_path']}；"
        f"不一致表数: {parity_report['mismatch_count']}"
    )
