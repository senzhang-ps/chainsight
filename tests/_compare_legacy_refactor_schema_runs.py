"""对比 input legacy 与 test refactor 的完整落库结果（只读诊断）。

比较范围：
- ``module*`` ↔ 同名 ``module*``；
- legacy ``orchestrator_*`` ↔ refactor ``viewcontext_*``（同后缀）；
- ``summary_*`` ↔ 同名 ``summary_*``。

脚本会自动确认两个 schema 各自唯一的 run_id，并产出总行数、逐日行数及
全量行级/字段级差异。``order_id`` 等顺序生成技术列不会参与关联、重复行配对
或差异判定。

运行：
    conda run --no-capture-output -n work python tests/_compare_legacy_refactor_schema_runs.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pgsql_db.settings import resolve_database_config
from src.core.db.pgsql.db import DB
from tests.compare_utils import (
    DIFFERENCE_DETAIL_COLUMNS,
    compare_dataframes_as_multiset,
    compare_dataframes_by_key,
    dataframe_difference_details,
)

DEFAULT_LEGACY_SCHEMA = "input"
DEFAULT_REFACTOR_SCHEMA = "test"
META_COLUMNS = {"run_id", "config_name", "db_write_time", "file_date"}
# 仅排除明确表示生成顺序的技术列；任何业务 ID 均保留。
SEQUENCE_COLUMNS = {
    "order_id", "orderid", "row_number", "row_num", "row_index",
    "sequence", "sequence_no", "sequence_number", "serial_no", "serial_number",
    "unnamed: 0",
}
NON_METRIC_COLUMNS = {
    "sim_date", "simulation_date", "date", "order_date", "delivery_date",
    "ship_date", "available_date", "requirement_date", "production_plan_date",
    "planned_delivery_date", "planned_deployment_date", "created_date",
    "material", "location", "sending", "receiving", "line", "type", "reason",
    "demand_element", "changeover_id", "changeover_type", "capacity_key",
}
METRIC_NAME_HINTS = (
    "qty", "quantity", "amount", "volume", "capacity", "inventory", "stock",
    "demand", "supply", "shipment", "delivery", "production", "cut", "hours",
    "count", "utilization", "rate", "value", "cost", "weight",
)
SUMMARY_TABLE_RENAMES = {
    "summary_output_fullcapacityexceed": "summary_full_exceed_capacity_report",
    "summary_output_fullchangeoverlog": "summary_full_changeover_report",
    "summary_output_fulldeliveryplan": "summary_full_delivery_plan_report",
    "summary_output_fulldeploymentplan": "summary_full_deployment_plan_report",
    "summary_output_fullproductionplan": "summary_full_production_plan_report",
    "summary_output_fulltruckusage": "summary_full_truck_usage_report",
    "summary_output_ordershipmentcutsummary": "summary_full_order_shipment_cut_report",
}
OUT_OF_SCOPE_TABLES = {
    # legacy M4/M6 日志在本次基线为空，不能提供有效的业务对比。
    "module4_output_validation": "legacy 空表，不纳入差异审计",
    "module6_output_bypassrulehitlog": "legacy 空表，不纳入差异审计",
    "module6_output_unsatisfiedmdqlog": "legacy 空表，不纳入差异审计",
    "module6_output_validationlog": "legacy 空表，不纳入差异审计",
    # 旧编排状态与已比较的模块输出重复，避免重复计入覆盖缺口。
    "orchestrator_shipment_log": "与 module1_output_shipmentlog 重复",
    "orchestrator_delivery_shipment_log": "与 module6_output_deliveryplan 语义重复",
    "orchestrator_open_deployment_pastdue_cleanup": "legacy 空表，不纳入差异审计",
    # refactor 新增的 M4 续跑内部状态，无 legacy 对等表。
    "viewcontext_m4_allocated_capacity": "refactor 新增 M4 内部状态，无 legacy 对等表",
    "viewcontext_m4_line_states": "refactor 新增 M4 内部状态，无 legacy 对等表",
    # 日志带运行时 timestamp，且 legacy/rewrite 的保留/清理策略不同；不属于
    # 计划、库存、调拨的业务事实，单独审计而不纳入业务一致性结论。
    "orchestrator_daily_logs": "运行审计日志，时间戳与保留策略不同，不纳入业务一致性",
    "viewcontext_daily_logs": "运行审计日志，时间戳与保留策略不同，不纳入业务一致性",
    # 新版 StateContext 新增的历史库存报表，legacy 没有语义对等报表。
    "summary_historical_inventory_record": "refactor 新增历史库存报表，无 legacy 对等表",
}

# M6/状态表中同一路线可存在多笔并发明细。默认业务键只能关联到路线层级，
# 会把不同车辆/部署单交叉配对并制造数量差异；这些表必须优先使用稳定 UID。
PAIR_KEY_COLUMNS = {
    ("module6_output_deliveryplan", "module6_output_deliveryplan"): (
        "sim_date", "ori_deployment_uid", "vehicle_uid",
    ),
    ("orchestrator_open_deployment", "viewcontext_open_deployment"): (
        "sim_date", "ori_deployment_uid",
    ),
    ("orchestrator_planning_intransit", "viewcontext_planning_intransit"): (
        "sim_date", "transit_uid",
    ),
    ("summary_output_fulldeliveryplan", "summary_full_delivery_plan_report"): (
        "ori_deployment_uid", "vehicle_uid",
    ),
    ("summary_output_fulldeploymentplan", "summary_full_deployment_plan_report"): (
        "sim_date", "date", "material", "sending", "receiving",
        "demand_element", "planned_delivery_date", "orig_location",
        "demand_qty", "planned_qty", "deployed_qty_invcon",
        "deploy_qty_with_plan_order", "deploy_from_in_transit",
        "deploy_from_open_deployment_inbound", "deploy_from_future_production",
        "leadtime", "is_cross_node", "deployed_qty", "quota",
        "deployed_qty_invcon_push",
    ),
    ("summary_output_ordershipmentcutsummary", "summary_full_order_shipment_cut_report"): (
        "simulation_date", "date", "material", "location",
    ),
}

PAIR_EXCLUDED_COLUMNS = {
    # legacy OrderShipmentCut Summary 的 sim_date 为 NULL；refactor 写入时
    # 使用运行元数据日期。该列不是报表业务事实，业务快照日期在
    # simulation_date 中维护。
    ("summary_output_ordershipmentcutsummary", "summary_full_order_shipment_cut_report"): {
        "sim_date",
        "simulation_date",
    },
}

MULTISET_PAIRS = {
    ("summary_output_fulldeploymentplan", "summary_full_deployment_plan_report"),
}

# DeploymentPlan 没有稳定的明细顺序列。同一完整业务行的重复是运行输出
# 结构的一部分，不代表可区分的业务事实；先以全部共同业务字段去重，再比较。
FULL_ROW_DEDUP_PAIRS = {
    ("module5_output_deploymentplan", "module5_output_deploymentplan"),
}


def _db(schema: str) -> DB:
    cfg = resolve_database_config()
    db = DB(
        host=cfg["host"],
        port=cfg["port"],
        database=cfg["database"],
        user=cfg["user"],
        password=cfg["password"],
        schema=schema,
        auto_create_schema=False,
    )
    db.connect()
    return db


def _target_tables(db: DB, schema: str, prefix: str) -> list[str]:
    rows = db.execute_query(
        "SELECT DISTINCT table_name FROM information_schema.columns "
        "WHERE table_schema = %s AND column_name = 'run_id' "
        "AND table_name LIKE %s ORDER BY table_name",
        (schema, f"{prefix}%"),
    )
    return [str(row[0]) for row in rows]


def _run_ids(db: DB, schema: str, prefixes: tuple[str, ...]) -> list[str]:
    # 不能把表名以参数绑定到 FROM；先从元数据取得表名后逐表读取，保持只读。
    values: set[str] = set()
    tables: set[str] = set()
    for prefix in prefixes:
        tables.update(_target_tables(db, schema, prefix))
    for table in sorted(tables):
        rows = db.execute_query(
            f"SELECT DISTINCT run_id FROM {db.qualified_name(table)} "
            "WHERE run_id IS NOT NULL ORDER BY run_id"
        )
        values.update(str(row[0]) for row in rows)
    return sorted(values)


def _table_columns(db: DB, schema: str, table: str) -> list[str]:
    rows = db.execute_query(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position",
        (schema, table),
    )
    return [str(row[0]) for row in rows]


def _read_table(db: DB, table: str, run_id: str) -> pd.DataFrame:
    columns = _table_columns(db, db.schema, table)
    if not columns:
        return pd.DataFrame()
    rows = db.execute_query(
        f"SELECT * FROM {db.qualified_name(table)} WHERE run_id = %s",
        (run_id,),
    )
    return pd.DataFrame(rows, columns=columns)


def _clean_for_business_comparison(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """移除运行元数据和顺序技术列，保留业务 ID 与 ``sim_date``。"""
    frame = frame.drop(columns=list(META_COLUMNS), errors="ignore").copy()
    excluded = sorted(
        column for column in frame.columns
        if str(column).casefold() in SEQUENCE_COLUMNS
    )
    return frame, excluded


def _pair_key_columns(pair: dict[str, str], left: pd.DataFrame, right: pd.DataFrame) -> list[str] | None:
    """返回可用于该表对的稳定 UID 键；缺列时安全回退通用选择器。"""
    configured = PAIR_KEY_COLUMNS.get((pair["legacy_table"], pair["refactor_table"]))
    if configured and all(column in left.columns and column in right.columns for column in configured):
        return list(configured)
    return None


def _pair_excluded_columns(pair: dict[str, str]) -> set[str]:
    """返回表对专属的非业务列排除项。"""
    return PAIR_EXCLUDED_COLUMNS.get(
        (pair["legacy_table"], pair["refactor_table"]), set()
    )


def _deduplicate_full_business_rows(
    pair: dict[str, str], left: pd.DataFrame, right: pd.DataFrame,
    excluded_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """对指定表对按全部共有业务列去重，保留无序明细的比较语义。"""
    if (pair["legacy_table"], pair["refactor_table"]) not in FULL_ROW_DEDUP_PAIRS:
        return left, right
    excluded = {str(column).casefold() for column in excluded_columns}
    columns = [
        column for column in sorted(set(left.columns) & set(right.columns))
        if str(column).casefold() not in excluded
    ]
    if not columns:
        return left, right
    return (
        left.drop_duplicates(subset=columns, keep="first").copy(),
        right.drop_duplicates(subset=columns, keep="first").copy(),
    )


def _remove_all_zero_metric_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """删除至少含一个指标且所有数值指标均为零的占位行。

    数量差异中，这类行不代表业务差异。日期、地点、物料、需求元素和各种
    标识列不是指标；其余包含常见数值指标名称的列会按数值转换判定。为防止
    纯业务键记录被误删，必须至少识别出一个非空数值指标。
    """
    if frame.empty:
        return frame.copy(), 0

    metric_columns = [
        column for column in frame.columns
        if str(column).casefold() not in NON_METRIC_COLUMNS
        and not str(column).casefold().endswith("_id")
        and any(hint in str(column).casefold() for hint in METRIC_NAME_HINTS)
    ]
    if not metric_columns:
        return frame.copy(), 0

    numeric_metrics = frame[metric_columns].apply(pd.to_numeric, errors="coerce")
    has_metric = numeric_metrics.notna().any(axis=1)
    all_zero = numeric_metrics.fillna(0).abs().le(1e-9).all(axis=1)
    zero_row = has_metric & all_zero
    return frame.loc[~zero_row].copy(), int(zero_row.sum())


def _daily_counts(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    date_column = next((column for column in ("sim_date", "simulation_date", "date") if column in frame.columns), None)
    if date_column is None:
        return [{"date_column": "", "date": "<all>", "rows": int(len(frame))}]
    normalized = pd.to_datetime(frame[date_column], errors="coerce").dt.strftime("%Y-%m-%d")
    normalized = normalized.fillna("<NULL>")
    counts = normalized.value_counts(dropna=False).sort_index()
    return [
        {"date_column": date_column, "date": str(date), "rows": int(rows)}
        for date, rows in counts.items()
    ]


def _run_event(db: DB, run_id: str) -> dict[str, Any] | None:
    try:
        rows = db.execute_query(
            f"SELECT runid, status, progress_date, started_at, finished_at "
            f"FROM {db.qualified_name('orch_run_event')} WHERE runid = %s",
            (run_id,),
        )
    except Exception:
        return None
    if not rows:
        return None
    return dict(zip(("run_id", "status", "progress_date", "started_at", "finished_at"), rows[0]))


def _module_pair_key(table: str) -> str:
    """移除模块表前缀，但保留模块号以避免不同模块的日志误配。"""
    match = re.fullmatch(r"(?:module|m)(\d+)(?:_output)?_(.+)", table.casefold())
    return f"module{match.group(1)}:{match.group(2)}" if match else table.casefold()


def _state_pair_key(table: str) -> str:
    """将 legacy orchestrator 与 refactor viewcontext 归一到同一状态键。"""
    normalized = table.casefold()
    for prefix in ("orchestrator_", "viewcontext_"):
        if normalized.startswith(prefix):
            return normalized.removeprefix(prefix)
    return normalized


def _summary_pair_key(table: str) -> str:
    """移除 Summary 表名前缀；保留其余业务语义作为一对一匹配键。"""
    normalized = table.casefold()
    for prefix in ("summary_output_", "summary_"):
        if normalized.startswith(prefix):
            return normalized.removeprefix(prefix)
    return normalized


def _index_pair_keys(tables: set[str], key_builder) -> tuple[dict[str, str], dict[str, list[str]]]:
    """构造规范化键索引，并显式保留同键多表这一不安全配对情况。"""
    grouped: dict[str, list[str]] = {}
    for table in tables:
        grouped.setdefault(key_builder(table), []).append(table)
    unique = {key: names[0] for key, names in grouped.items() if len(names) == 1}
    ambiguous = {key: sorted(names) for key, names in grouped.items() if len(names) > 1}
    return unique, ambiguous


def _pair_tables(
    legacy_db: DB,
    refactor_db: DB,
    legacy_schema: str,
    refactor_schema: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    legacy_modules = set(_target_tables(legacy_db, legacy_schema, "module"))
    refactor_modules = set(_target_tables(refactor_db, refactor_schema, "module"))
    legacy_orchestrator = set(_target_tables(legacy_db, legacy_schema, "orchestrator_"))
    refactor_viewcontext = set(_target_tables(refactor_db, refactor_schema, "viewcontext_"))
    legacy_summary = set(_target_tables(legacy_db, legacy_schema, "summary"))
    refactor_summary = set(_target_tables(refactor_db, refactor_schema, "summary"))

    excluded = [
        {"table": table, "reason": reason}
        for table, reason in OUT_OF_SCOPE_TABLES.items()
    ]
    legacy_modules -= set(OUT_OF_SCOPE_TABLES)
    refactor_modules -= set(OUT_OF_SCOPE_TABLES)
    legacy_orchestrator -= set(OUT_OF_SCOPE_TABLES)
    refactor_viewcontext -= set(OUT_OF_SCOPE_TABLES)
    legacy_summary -= set(OUT_OF_SCOPE_TABLES)
    refactor_summary -= set(OUT_OF_SCOPE_TABLES)

    pairs: list[dict[str, str]] = []
    gaps: list[dict[str, str]] = []

    def pair_by_stripped_prefix(category: str, left: set[str], right: set[str], key_builder) -> None:
        left_by_key, left_ambiguous = _index_pair_keys(left, key_builder)
        right_by_key, right_ambiguous = _index_pair_keys(right, key_builder)
        for key in sorted(left_by_key.keys() & right_by_key.keys()):
            pairs.append({
                "category": category,
                "pairing_key": key,
                "legacy_table": left_by_key[key],
                "refactor_table": right_by_key[key],
            })
        for key in sorted(left_by_key.keys() - right_by_key.keys()):
            gaps.append({"category": category, "side": "legacy_only_table", "table": left_by_key[key], "pairing_key": key})
        for key in sorted(right_by_key.keys() - left_by_key.keys()):
            gaps.append({"category": category, "side": "refactor_only_table", "table": right_by_key[key], "pairing_key": key})
        for key, tables in left_ambiguous.items():
            gaps.append({"category": category, "side": "legacy_ambiguous_pairing_key", "table": ", ".join(tables), "pairing_key": key})
        for key, tables in right_ambiguous.items():
            gaps.append({"category": category, "side": "refactor_ambiguous_pairing_key", "table": ", ".join(tables), "pairing_key": key})

    pair_by_stripped_prefix("module", legacy_modules, refactor_modules, _module_pair_key)

    # legacy Summary 使用 ``summary_output_*``，重构后使用语义化
    # ``summary_full_*_report`` 命名；显式映射后再处理保留的同名表。
    mapped_legacy_summary = set(SUMMARY_TABLE_RENAMES)
    mapped_refactor_summary = set(SUMMARY_TABLE_RENAMES.values())
    for legacy_table, refactor_table in SUMMARY_TABLE_RENAMES.items():
        if legacy_table in legacy_summary and refactor_table in refactor_summary:
            pairs.append({
                "category": "summary",
                "pairing_key": _summary_pair_key(refactor_table),
                "legacy_table": legacy_table,
                "refactor_table": refactor_table,
            })
        elif legacy_table in legacy_summary:
            gaps.append({"category": "summary", "side": "legacy_only_table", "table": legacy_table})
        elif refactor_table in refactor_summary:
            gaps.append({"category": "summary", "side": "refactor_only_table", "table": refactor_table})
    pair_by_stripped_prefix(
        "summary",
        legacy_summary - mapped_legacy_summary,
        refactor_summary - mapped_refactor_summary,
        _summary_pair_key,
    )

    pair_by_stripped_prefix(
        "orchestrator_to_viewcontext",
        legacy_orchestrator,
        refactor_viewcontext,
        _state_pair_key,
    )
    return sorted(pairs, key=lambda item: (item["category"], item["legacy_table"])), gaps, excluded


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    comparisons = report["comparisons"]
    different = [item for item in comparisons if not item["business_consistent"]]
    lines = [
        "# Legacy / Refactor 数据库差异审计",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- Legacy：`{report['legacy_schema']}` / `{report['legacy_run_id']}`",
        f"- Refactor：`{report['refactor_schema']}` / `{report['refactor_run_id']}`",
        f"- 已配对表：{len(comparisons)}；业务存在差异：{len(different)}；表覆盖缺口：{len(report['table_gaps'])}；按范围剔除：{len(report['excluded_tables'])}",
        "- `order_id`、`orderid` 和明确的行序字段已排除，不参与关联、重复行配对或字段差异。",
        "- 仅包含零数值指标的占位行不计入业务数量差异；原始行数仍完整保留在报告中。",
        "",
        "## 表级结果",
        "",
        "| 类别 | Legacy 表 | Refactor 表 | Legacy 原始行数 | Refactor 原始行数 | Legacy 忽略全零 | Refactor 忽略全零 | 匹配行 | Legacy 独有 | Refactor 独有 | 业务差异列数 | 结论 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in comparisons:
        comparison = item["comparison"]
        lines.append(
            f"| {item['category']} | {item['legacy_table']} | {item['refactor_table']} | "
            f"{item['legacy_raw_rows']} | {item['refactor_raw_rows']} | "
            f"{item['ignored_legacy_all_zero_metric_rows']} | "
            f"{item['ignored_refactor_all_zero_metric_rows']} | "
            f"{comparison.get('matched_rows', 0)} | "
            f"{comparison.get('left_only_keys', 0)} | {comparison.get('right_only_keys', 0)} | "
            f"{len(comparison.get('column_differences', {}))} | "
            f"{'一致' if item['business_consistent'] else '存在差异'} |"
        )
    if report["table_gaps"]:
        lines.extend(["", "## 表覆盖缺口", "", "| 类别 | 情况 | 表 |", "|---|---|---|"])
        lines.extend(
            f"| {item['category']} | {item['side']} | {item['table']} |"
            for item in report["table_gaps"]
        )
    if report["excluded_tables"]:
        lines.extend(["", "## 已剔除表", "", "| 表 | 原因 |", "|---|---|"])
        lines.extend(
            f"| {item['table']} | {item['reason']} |"
            for item in report["excluded_tables"]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="审计 Legacy 与 Refactor schema 的完整落库结果")
    parser.add_argument("--legacy-schema", default=DEFAULT_LEGACY_SCHEMA)
    parser.add_argument("--refactor-schema", default=DEFAULT_REFACTOR_SCHEMA)
    args = parser.parse_args()
    legacy_schema, refactor_schema = args.legacy_schema, args.refactor_schema
    legacy_db, refactor_db = _db(legacy_schema), _db(refactor_schema)
    try:
        legacy_run_ids = _run_ids(legacy_db, legacy_schema, ("module", "orchestrator_", "summary"))
        refactor_run_ids = _run_ids(refactor_db, refactor_schema, ("module", "viewcontext_", "summary"))
        if len(legacy_run_ids) != 1 or len(refactor_run_ids) != 1:
            raise RuntimeError(
                "要求每个 schema 只有一个待比较 run_id；"
                f"{legacy_schema}={legacy_run_ids}，{refactor_schema}={refactor_run_ids}"
            )
        legacy_run_id, refactor_run_id = legacy_run_ids[0], refactor_run_ids[0]
        pairs, gaps, excluded_tables = _pair_tables(legacy_db, refactor_db, legacy_schema, refactor_schema)
        output_dir = PROJECT_ROOT / "outputs" / "legacy_refactor_db_compare" / datetime.now().strftime("run_%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=False)

        comparisons: list[dict[str, Any]] = []
        count_rows: list[dict[str, Any]] = []
        details: list[pd.DataFrame] = []
        for pair in pairs:
            legacy_raw = _read_table(legacy_db, pair["legacy_table"], legacy_run_id)
            refactor_raw = _read_table(refactor_db, pair["refactor_table"], refactor_run_id)
            legacy, legacy_sequence_columns = _clean_for_business_comparison(legacy_raw)
            refactor, refactor_sequence_columns = _clean_for_business_comparison(refactor_raw)
            legacy, ignored_legacy_zero_rows = _remove_all_zero_metric_rows(legacy)
            refactor, ignored_refactor_zero_rows = _remove_all_zero_metric_rows(refactor)
            excluded_columns = sorted(
                set(legacy_sequence_columns)
                | set(refactor_sequence_columns)
                | _pair_excluded_columns(pair)
            )
            legacy, refactor = _deduplicate_full_business_rows(
                pair, legacy, refactor, excluded_columns
            )
            key_columns = _pair_key_columns(pair, legacy, refactor)
            if (pair["legacy_table"], pair["refactor_table"]) in MULTISET_PAIRS:
                comparison = compare_dataframes_as_multiset(
                    legacy,
                    refactor,
                    label=f"{pair['legacy_table']} -> {pair['refactor_table']}",
                    excluded_columns=excluded_columns,
                )
            else:
                comparison = compare_dataframes_by_key(
                    legacy,
                    refactor,
                    label=f"{pair['legacy_table']} -> {pair['refactor_table']}",
                    key_columns=key_columns,
                    excluded_columns=excluded_columns,
                )
            business_consistent = not (
                comparison.get("error")
                or comparison["left_only_keys"]
                or comparison["right_only_keys"]
                or comparison["column_differences"]
            )
            item = {
                **pair,
                "legacy_raw_rows": int(len(legacy_raw)),
                "refactor_raw_rows": int(len(refactor_raw)),
                "legacy_business_rows": int(len(legacy)),
                "refactor_business_rows": int(len(refactor)),
                "legacy_columns": legacy_raw.columns.tolist(),
                "refactor_columns": refactor_raw.columns.tolist(),
                "excluded_sequence_columns": excluded_columns,
                "ignored_legacy_all_zero_metric_rows": ignored_legacy_zero_rows,
                "ignored_refactor_all_zero_metric_rows": ignored_refactor_zero_rows,
                "business_consistent": business_consistent,
                "comparison": comparison,
            }
            comparisons.append(item)
            for side, frame in (("legacy", legacy_raw), ("refactor", refactor_raw)):
                for daily in _daily_counts(frame):
                    count_rows.append({**pair, "side": side, **daily})

            if os.environ.get("SKIP_AUDIT_DIFFERENCE_DETAILS") != "1":
                table_details = dataframe_difference_details(
                    legacy,
                    refactor,
                    key_columns=key_columns,
                    excluded_columns=excluded_columns,
                )
                table_details.insert(0, "refactor_table", pair["refactor_table"])
                table_details.insert(0, "legacy_table", pair["legacy_table"])
                table_details.insert(0, "category", pair["category"])
                details.append(table_details)

        report = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "legacy_schema": legacy_schema,
            "refactor_schema": refactor_schema,
            "legacy_run_id": legacy_run_id,
            "refactor_run_id": refactor_run_id,
            "legacy_run_event": _run_event(legacy_db, legacy_run_id),
            "refactor_run_event": _run_event(refactor_db, refactor_run_id),
            "sequence_columns_excluded_by_policy": sorted(SEQUENCE_COLUMNS),
            "all_zero_metric_row_policy": (
                "Rows with at least one recognized numeric metric and all such metrics equal to zero "
                "are excluded from business row-count and detail differences; raw counts are retained."
            ),
            "pair_count": len(comparisons),
            "business_difference_table_count": sum(not item["business_consistent"] for item in comparisons),
            "table_gaps": gaps,
            "excluded_tables": excluded_tables,
            "comparisons": comparisons,
        }
        (output_dir / "comparison_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        pd.DataFrame(count_rows).to_csv(output_dir / "row_counts_by_date.csv", index=False, encoding="utf-8-sig")
        detail_columns = ["category", "legacy_table", "refactor_table", *DIFFERENCE_DETAIL_COLUMNS]
        all_details = pd.concat(details, ignore_index=True) if details else pd.DataFrame(columns=detail_columns)
        all_details.to_csv(output_dir / "difference_details.csv", index=False, encoding="utf-8-sig")
        _write_markdown(output_dir / "summary.md", report)

        print(f"报告目录: {output_dir}")
        print(f"Legacy run_id ({legacy_schema}): {legacy_run_id}")
        print(f"Refactor run_id ({refactor_schema}): {refactor_run_id}")
        print(f"已配对表: {len(comparisons)}；存在业务差异: {report['business_difference_table_count']}；覆盖缺口: {len(gaps)}")
        for item in comparisons:
            result = item["comparison"]
            marker = "OK" if item["business_consistent"] else "DIFF"
            print(
                f"[{marker}] {item['legacy_table']} -> {item['refactor_table']}: "
                f"legacy={item['legacy_raw_rows']}, refactor={item['refactor_raw_rows']}, "
                f"matched={result.get('matched_rows', 0)}, "
                f"legacy_only={result.get('left_only_keys', 0)}, refactor_only={result.get('right_only_keys', 0)}, "
                f"value_columns={len(result.get('column_differences', {}))}"
            )
        # 诊断结果有差异是预期输入，仍以成功返回，方便稳定生成审计产物。
        return 0
    finally:
        legacy_db.close()
        refactor_db.close()


if __name__ == "__main__":
    raise SystemExit(main())
