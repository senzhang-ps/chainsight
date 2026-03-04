#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
oc_3way_content_compare.py

三版本（Dev/Src/DB）逐日逐Sheet内容级对比工具。

对比维度:
  1. 行数是否一致
  2. 列名是否一致
  3. 数值内容是否一致（按key列merge后逐列比较）

输出:
  - JSON格式的全量对比结果 (tools/oc_3way_content_results.json)
  - 控制台摘要

Usage:
    python tools/oc_3way_content_compare.py [--days 1-10] [--modules module1]
"""

from __future__ import annotations
import json, sys, time, warnings, argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import numpy as np
import psycopg2

warnings.filterwarnings("ignore")

# ─── Configuration ───────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
DEV_DIR = PROJECT_ROOT / "outputs" / "run_20260127_142402"
SRC_DIR = PROJECT_ROOT / "outputs" / "OC_Paste_S1_20251224" / "run_20260209_222302"
DB_CONN_PARAMS = dict(host="localhost", port=5432, dbname="test_db",
                      user="postgres", password="123456", client_encoding="utf8")

START_DATE = datetime(2025, 12, 15)
NUM_DAYS = 76

DB_RUN_ID = "OC_Paste_S1_20251224_20260227_052227"
DB_META_COLS = {"sim_date", "run_id", "config_name", "db_write_time"}

# ─── Sheet/Table mapping ────────────────────────────────────────────
# Each entry: (module_dir, file_pattern, sheet_name, db_table, db_date_col, xlsx_key_cols, compare_cols_or_None)
# xlsx_key_cols: columns used as merge key for content comparison
# compare_cols: if None, compare all non-key columns; if list, compare only those

SHEET_MAPPINGS = [
    # Module1
    {
        "module": "module1",
        "file_pattern": "module1_output_{date}.xlsx",
        "sheet": "OrderLog",
        "db_table": "module1_output_orderlog",
        "db_date_col": "sim_date",
        "key_cols": ["date", "material", "location", "demand_type"],
        "compare_cols": ["quantity", "simulation_date", "advance_days"],
        "db_col_rename": {},
    },
    {
        "module": "module1",
        "file_pattern": "module1_output_{date}.xlsx",
        "sheet": "ShipmentLog",
        "db_table": "module1_output_shipmentlog",
        "db_date_col": "sim_date",
        "key_cols": ["date", "material", "location", "demand_type", "order_id"],
        "compare_cols": ["quantity"],
        "db_col_rename": {},
    },
    {
        "module": "module1",
        "file_pattern": "module1_output_{date}.xlsx",
        "sheet": "CutLog",
        "db_table": "module1_output_cutlog",
        "db_date_col": "sim_date",
        "key_cols": ["date", "material", "location"],
        "compare_cols": ["quantity"],
        "db_col_rename": {},
    },
    {
        "module": "module1",
        "file_pattern": "module1_output_{date}.xlsx",
        "sheet": "Summary",
        "db_table": "module1_output_summary",
        "db_date_col": "sim_date",
        "key_cols": ["date"],
        "compare_cols": ["total_orders", "total_shipments", "total_cuts", "total_supplydemand"],
        "db_col_rename": {},
        "xlsx_col_rename": {"Total_Orders": "total_orders", "Total_Shipments": "total_shipments",
                            "Total_Cuts": "total_cuts", "Total_SupplyDemand": "total_supplydemand",
                            "Date": "date"},
    },
    # Module3
    {
        "module": "module3",
        "file_pattern": "Module3Output_{date}.xlsx",
        "sheet": "NetDemand",
        "db_table": "module3_output_netdemand",
        "db_date_col": "sim_date",
        "key_cols": ["material", "location", "requirement_date", "demand_element", "layer"],
        "compare_cols": ["quantity", "simulation_date", "horizon_days"],
        "db_col_rename": {},
    },
    # Module4
    {
        "module": "module4",
        "file_pattern": "Module4Output_{date}.xlsx",
        "sheet": "ProductionPlan",
        "db_table": "module4_output_productionplan",
        "db_date_col": "sim_date",
        "key_cols": ["material", "location", "line", "simulation_date", "production_plan_date"],
        "compare_cols": ["available_date", "uncon_planned_qty", "con_planned_qty", "produced_qty",
                         "changeover_id", "changeover_time", "changeover_time_remaining",
                         "is_first_changeover_day"],
        "db_col_rename": {},
    },
    {
        "module": "module4",
        "file_pattern": "Module4Output_{date}.xlsx",
        "sheet": "CapacityExceed",
        "db_table": "module4_output_capacityexceed",
        "db_date_col": "sim_date",
        "key_cols": ["material", "location", "line", "simulation_date", "exceed_type"],
        "compare_cols": ["exceed_qty"],
        "db_col_rename": {},
    },
    {
        "module": "module4",
        "file_pattern": "Module4Output_{date}.xlsx",
        "sheet": "ChangeoverLog",
        "db_table": "module4_output_changeoverlog",
        "db_date_col": "sim_date",
        "key_cols": ["date", "location", "line", "changeover_type"],
        "compare_cols": ["count", "time", "cost", "mu_loss"],
        "db_col_rename": {},
    },
    # Module5
    {
        "module": "module5",
        "file_pattern": "Module5Output_{date}.xlsx",
        "sheet": "DeploymentPlan",
        "db_table": "module5_output_deploymentplan",
        "db_date_col": "sim_date",
        "key_cols": ["date", "material", "sending", "receiving", "demand_element"],
        "compare_cols": ["demand_qty", "planned_qty", "deployed_qty_invCon",
                         "deploy_qty_with_plan_order", "deploy_from_in_transit",
                         "deploy_from_open_deployment_inbound", "deploy_from_future_production",
                         "planned_delivery_date", "orig_location", "leadtime", "is_cross_node",
                         "deployed_qty_invCon_push", "deployed_qty", "quota"],
        "db_col_rename": {"deployed_qty_invcon": "deployed_qty_invCon",
                          "deployed_qty_invcon_push": "deployed_qty_invCon_push",
                          "wfr": "WFR", "vfr": "VFR"},
    },
    {
        "module": "module5",
        "file_pattern": "Module5Output_{date}.xlsx",
        "sheet": "UnfulfilledLog",
        "db_table": "module5_output_unfulfilledlog",
        "db_date_col": "sim_date",
        "key_cols": ["date", "sending", "receiving", "demand_element"],
        "compare_cols": ["demand_qty", "unfulfilled_qty", "reason"],
        "db_col_rename": {},
    },
    {
        "module": "module5",
        "file_pattern": "Module5Output_{date}.xlsx",
        "sheet": "StockOnHandLog",
        "db_table": "module5_output_stockonhandlog",
        "db_date_col": "sim_date",
        "key_cols": ["material", "location", "date"],
        "compare_cols": ["beginning_soh", "production", "in_transit", "delivery_gr",
                         "today_shipment", "deployed_qty", "ending_soh"],
        "db_col_rename": {},
    },
    {
        "module": "module5",
        "file_pattern": "Module5Output_{date}.xlsx",
        "sheet": "Validation",
        "db_table": "module5_output_validation",
        "db_date_col": "sim_date",
        "key_cols": [],
        "compare_cols": ["no", "issue"],
        "db_col_rename": {},
        "xlsx_col_rename": {"No": "no", "Issue": "issue"},
    },
    # Module6
    {
        "module": "module6",
        "file_pattern": "Module6Output_{date}.xlsx",
        "sheet": "DeliveryPlan",
        "db_table": "module6_output_deliveryplan",
        "db_date_col": "sim_date",
        "key_cols": ["vehicle_uid", "ori_deployment_uid", "material", "sending", "receiving"],
        "compare_cols": ["planned_deployment_date", "actual_ship_date", "actual_delivery_date",
                         "delivery_qty", "truck_type", "truck_load_pct", "WFR", "VFR"],
        "db_col_rename": {"wfr": "WFR", "vfr": "VFR"},
    },
    {
        "module": "module6",
        "file_pattern": "Module6Output_{date}.xlsx",
        "sheet": "VehicleLog",
        "db_table": "module6_output_vehiclelog",
        "db_date_col": "sim_date",
        "key_cols": ["date", "sending", "receiving", "truck_type", "vehicle_uid"],
        "compare_cols": ["vehicle_no", "total_units", "total_weight", "total_volume",
                         "WFR", "VFR", "trigger"],
        "db_col_rename": {"wfr": "WFR", "vfr": "VFR"},
    },
    {
        "module": "module6",
        "file_pattern": "Module6Output_{date}.xlsx",
        "sheet": "TruckUsageLog",
        "db_table": "module6_output_truckusagelog",
        "db_date_col": "sim_date",
        "key_cols": ["date", "sending", "receiving", "truck_type"],
        "compare_cols": ["truck_used"],
        "db_col_rename": {},
    },
]


# ─── Helper functions ────────────────────────────────────────────────

def sim_date_str(day_num: int) -> str:
    """Return YYYYMMDD for 1-indexed simulation day."""
    return (START_DATE + timedelta(days=day_num - 1)).strftime("%Y%m%d")


def sim_date_iso(day_num: int) -> str:
    """Return YYYY-MM-DD for 1-indexed simulation day."""
    return (START_DATE + timedelta(days=day_num - 1)).strftime("%Y-%m-%d")


def get_daily_order_count(filepath: Path, date_str: str) -> int | None:
    """从 OrderLog 中统计当日新增订单数 (simulation_date = 当天的行数).
    
    Dev/Src 的 Summary 中 total_orders 是累积值，需要转换为当日值才能与 DB 对比。
    Args:
        filepath: xlsx 文件路径 (module1_output_YYYYMMDD.xlsx)
        date_str: YYYY-MM-DD 格式的仿真日期
    Returns:
        当日新增订单行数, 或 None 如果文件不存在
    """
    if not filepath.exists():
        return None
    try:
        df = pd.read_excel(filepath, sheet_name="OrderLog", engine="openpyxl")
        if "simulation_date" not in df.columns:
            return None
        target = pd.Timestamp(date_str)
        return int((df["simulation_date"] == target).sum())
    except Exception:
        return None


def load_xlsx_sheet(filepath: Path, sheet_name: str,
                    col_rename: dict | None = None,
                    filter_date: str | None = None) -> pd.DataFrame | None:
    """Load a single sheet from an xlsx file. Returns None if not found/empty.

    If filter_date (YYYY-MM-DD) is provided and sheet_name == 'OrderLog',
    filter to only rows where simulation_date == filter_date.
    This is needed because OrderLog in xlsx is a cumulative snapshot.
    """
    if not filepath.exists():
        return None
    try:
        df = pd.read_excel(filepath, sheet_name=sheet_name, engine="openpyxl")
        if sheet_name == "OrderLog" and filter_date is not None and "simulation_date" in df.columns:
            target = pd.Timestamp(filter_date)
            df = df[df["simulation_date"] == target].copy()
        if col_rename:
            df = df.rename(columns=col_rename)
        return df
    except Exception:
        return None


def load_db_data(conn, table: str, date_col: str, date_val: str,
                 col_rename: dict, run_id: str = DB_RUN_ID) -> pd.DataFrame | None:
    """Load data from DB for a specific simulation date, filtered by run_id."""
    try:
        # date_val is YYYYMMDD, DB stores as date or string
        query = f"SELECT * FROM {table} WHERE {date_col} = %s AND run_id = %s"
        df = pd.read_sql(query, conn, params=[date_val, run_id])
        # Drop DB metadata columns
        drop_cols = [c for c in df.columns if c in DB_META_COLS]
        df = df.drop(columns=drop_cols, errors="ignore")
        # Rename DB columns to match xlsx
        if col_rename:
            df = df.rename(columns=col_rename)
        return df
    except Exception as e:
        return None


def _normalize_date_str(s: str) -> str:
    """Normalize date-like strings: strip trailing ' 00:00:00', whitespace, etc.
    Converts '2025-12-16 00:00:00' -> '2025-12-16' so xlsx and DB dates compare equal."""
    s = s.strip()
    if s.endswith(" 00:00:00"):
        s = s[:-9]
    # Also handle Timestamp repr like 'Timestamp(...)' or 'NaT'
    if s == "NaT" or s == "nan" or s == "None":
        return ""
    return s


def _normalize_location_str(s: str) -> str:
    """Normalize location codes: strip leading zeros from purely numeric strings.
    '0386' -> '386', but 'A668' stays 'A668'."""
    s = s.strip()
    if s.isdigit():
        return str(int(s))
    return s


def _normalize_bool_str(s: str) -> str:
    """Normalize boolean-like strings: 'True'/'true'/'TRUE'/1 -> 'True', 'False'/'false'/0 -> 'False'."""
    s = s.strip().lower()
    if s in ("true", "1", "1.0"):
        return "True"
    if s in ("false", "0", "0.0"):
        return "False"
    if s in ("nan", "none", "nat", "<na>", ""):
        return ""
    return s


# Column names that represent location codes (may have leading-zero differences)
LOCATION_LIKE_NAMES = {"location", "sending", "receiving", "orig_location"}
# Column names that represent boolean values
BOOL_LIKE_NAMES = {"is_cross_node", "is_first_changeover_day"}


def normalize_df(df: pd.DataFrame, key_cols: list[str], compare_cols: list[str]) -> pd.DataFrame:
    """Normalize a DataFrame for comparison: ensure consistent types for merge."""
    all_cols = [c for c in key_cols + compare_cols if c in df.columns]
    df = df[all_cols].copy()

    # Detect date-like column names for special normalization
    DATE_LIKE_NAMES = {"date", "simulation_date", "requirement_date", "production_plan_date",
                       "available_date", "planned_delivery_date", "planned_deployment_date",
                       "actual_ship_date", "actual_delivery_date"}

    for col in df.columns:
        is_date_col = col.lower() in DATE_LIKE_NAMES
        is_location_col = col.lower() in LOCATION_LIKE_NAMES
        is_bool_col = col.lower() in BOOL_LIKE_NAMES

        if col in key_cols:
            # Key columns must be strings for consistent merging
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].dt.strftime("%Y-%m-%d").fillna("")
            else:
                df[col] = df[col].fillna("").astype(str).str.strip()
                # Normalize null representations: None/nan/NaN/NaT -> ""
                df[col] = df[col].replace({"nan": "", "None": "", "NaT": "", "NaN": "", "<NA>": ""})
                if is_date_col:
                    df[col] = df[col].apply(_normalize_date_str)
                if is_location_col:
                    df[col] = df[col].apply(_normalize_location_str)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y-%m-%d").fillna("")
        elif is_bool_col:
            # Normalize booleans to consistent string representation
            df[col] = df[col].astype(str).str.strip().apply(_normalize_bool_str)
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = df[col].astype(str).str.strip()
            if is_date_col:
                df[col] = df[col].apply(_normalize_date_str)
            if is_location_col:
                df[col] = df[col].apply(_normalize_location_str)

    return df


def compare_two_dfs(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    key_cols: list[str],
    compare_cols: list[str],
    label1: str = "A",
    label2: str = "B",
    tolerance: float = 1e-6,
) -> dict:
    """
    Compare two DataFrames on content level.
    Returns a dict with comparison summary.
    """
    result = {
        "rows_1": len(df1),
        "rows_2": len(df2),
        "row_match": len(df1) == len(df2),
        "content_match": False,
        "diff_count": 0,
        "diff_details": [],  # list of {key, col, val_1, val_2}
    }

    if df1.empty and df2.empty:
        result["content_match"] = True
        return result

    if df1.empty or df2.empty:
        result["diff_count"] = max(len(df1), len(df2))
        return result

    # Normalize column names that exist in both
    avail_compare = [c for c in compare_cols if c in df1.columns and c in df2.columns]

    if not key_cols:
        # No key columns — compare sorted DataFrames directly
        avail_all = avail_compare
        if not avail_all:
            result["content_match"] = result["row_match"]
            return result

        df1_s = df1[avail_all].sort_values(by=avail_all).reset_index(drop=True)
        df2_s = df2[avail_all].sort_values(by=avail_all).reset_index(drop=True)

        if len(df1_s) != len(df2_s):
            result["diff_count"] = abs(len(df1_s) - len(df2_s))
            return result

        diff_count = 0
        for col in avail_all:
            if pd.api.types.is_numeric_dtype(df1_s[col]):
                mask = ~np.isclose(
                    df1_s[col].fillna(0).values,
                    df2_s[col].fillna(0).values,
                    atol=tolerance, equal_nan=True
                )
            else:
                mask = df1_s[col].fillna("").values != df2_s[col].fillna("").values
            ndiff = int(mask.sum())
            diff_count += ndiff

        result["diff_count"] = diff_count
        result["content_match"] = diff_count == 0
        return result

    # With key columns — merge and compare
    avail_keys = [c for c in key_cols if c in df1.columns and c in df2.columns]
    if not avail_keys:
        # Can't merge, fallback to sorted comparison
        result["content_match"] = result["row_match"]
        return result

    # Handle duplicate keys by adding a counter
    # IMPORTANT: Sort both DataFrames by ALL available columns before cumcount
    # so that _dup_idx aligns even when source row ordering differs
    all_sort_cols = avail_keys + [c for c in avail_compare if c in df1.columns and c in df2.columns]
    df1_c = df1.sort_values(by=[c for c in all_sort_cols if c in df1.columns]).reset_index(drop=True).copy()
    df2_c = df2.sort_values(by=[c for c in all_sort_cols if c in df2.columns]).reset_index(drop=True).copy()
    df1_c["_dup_idx"] = df1_c.groupby(avail_keys).cumcount()
    df2_c["_dup_idx"] = df2_c.groupby(avail_keys).cumcount()
    merge_keys = avail_keys + ["_dup_idx"]

    merged = df1_c.merge(df2_c, on=merge_keys, how="outer",
                         suffixes=("_1", "_2"), indicator=True)

    # Rows only in one side
    only_1 = int((merged["_merge"] == "left_only").sum())
    only_2 = int((merged["_merge"] == "right_only").sum())
    both = merged[merged["_merge"] == "both"]

    diff_count = only_1 + only_2
    diff_details = []

    # Compare values in matched rows
    for col in avail_compare:
        c1 = f"{col}_1"
        c2 = f"{col}_2"
        if c1 not in both.columns or c2 not in both.columns:
            continue

        try:
            v1 = pd.to_numeric(both[c1], errors="coerce")
            v2 = pd.to_numeric(both[c2], errors="coerce")
            both_numeric = v1.notna().any() or v2.notna().any()
        except Exception:
            both_numeric = False

        if both_numeric:
            try:
                mask = ~np.isclose(
                    v1.fillna(0).astype(float).values,
                    v2.fillna(0).astype(float).values,
                    atol=tolerance, equal_nan=True,
                )
            except Exception:
                mask = both[c1].fillna("").astype(str).values != both[c2].fillna("").astype(str).values
        else:
            # Normalize date-like strings before comparison, and also normalize null values
            s1 = both[c1].fillna("").astype(str).apply(_normalize_date_str).values
            s2 = both[c2].fillna("").astype(str).apply(_normalize_date_str).values
            # Also normalize common null representations
            null_vals = {"nan", "None", "NaT", "NaN", "<NA>", ""}
            s1 = np.array(["" if v in null_vals else v for v in s1])
            s2 = np.array(["" if v in null_vals else v for v in s2])
            mask = s1 != s2

        ndiff = int(mask.sum())
        diff_count += ndiff

        if ndiff > 0 and ndiff <= 10:
            # Record first few diffs for detail
            diff_rows = both[mask]
            for _, row in diff_rows.head(5).iterrows():
                key_vals = {k: str(row[k]) for k in avail_keys if k in row.index}
                diff_details.append({
                    "key": key_vals,
                    "col": col,
                    f"val_{label1}": str(row[c1]),
                    f"val_{label2}": str(row[c2]),
                })
        elif ndiff > 10:
            diff_details.append({
                "col": col,
                "ndiff": ndiff,
                "note": f"{ndiff} rows differ in column '{col}'"
            })

    result["diff_count"] = diff_count
    result["content_match"] = diff_count == 0
    result["only_in_1"] = only_1
    result["only_in_2"] = only_2
    if diff_details:
        result["diff_details"] = diff_details[:20]  # Cap detail size

    return result


# ─── Main comparison loop ────────────────────────────────────────────

def run_comparison(day_range: range, module_filter: str | None = None,
                   verbose: bool = True) -> dict:
    """Run 3-way comparison for specified days and modules."""

    # Connect to DB
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONN_PARAMS)
        if verbose:
            print("DB connected", flush=True)
    except Exception as e:
        print(f"DB connection failed: {e}", flush=True)

    results: dict[str, Any] = {
        "run_time": datetime.now().isoformat(),
        "day_range": f"{min(day_range)}-{max(day_range)}",
        "comparisons": {},
    }

    total_sheets = 0
    total_pass_dev_src = 0
    total_pass_dev_db = 0
    total_pass_src_db = 0

    for mapping in SHEET_MAPPINGS:
        mod = mapping["module"]
        sheet = mapping["sheet"]
        comp_key = f"{mod}/{sheet}"

        if module_filter and mod != module_filter:
            continue

        if verbose:
            sys.stdout.write(f"\n=== {comp_key} ===\n")
            sys.stdout.flush()

        daily_results = []

        for day_num in day_range:
            ds = sim_date_str(day_num)
            ds_iso = sim_date_iso(day_num)

            # Load Dev xlsx
            xlsx_rename = mapping.get("xlsx_col_rename")
            dev_file = DEV_DIR / mod / mapping["file_pattern"].format(date=ds)
            dev_df = load_xlsx_sheet(dev_file, sheet, col_rename=xlsx_rename, filter_date=ds_iso)

            # Load Src xlsx
            src_file = SRC_DIR / mod / mapping["file_pattern"].format(date=ds)
            src_df = load_xlsx_sheet(src_file, sheet, col_rename=xlsx_rename, filter_date=ds_iso)

            # Load DB
            db_df = None
            if conn is not None:
                db_df = load_db_data(conn, mapping["db_table"], mapping["db_date_col"],
                                     ds, mapping["db_col_rename"])

            # Note: module1/Summary total_orders is cumulative in both xlsx and DB.
            # No conversion needed — compare raw values directly.

            # Normalize
            key_cols = mapping["key_cols"]
            compare_cols = mapping["compare_cols"]

            dev_norm = normalize_df(dev_df, key_cols, compare_cols) if dev_df is not None else pd.DataFrame()
            src_norm = normalize_df(src_df, key_cols, compare_cols) if src_df is not None else pd.DataFrame()
            db_norm = normalize_df(db_df, key_cols, compare_cols) if db_df is not None else pd.DataFrame()

            # Compare Dev vs Src
            cmp_dev_src = compare_two_dfs(dev_norm, src_norm, key_cols, compare_cols,
                                          label1="Dev", label2="Src")

            # Compare Dev vs DB
            cmp_dev_db = compare_two_dfs(dev_norm, db_norm, key_cols, compare_cols,
                                         label1="Dev", label2="DB")

            # Compare Src vs DB
            cmp_src_db = compare_two_dfs(src_norm, db_norm, key_cols, compare_cols,
                                         label1="Src", label2="DB")

            day_result = {
                "day": day_num,
                "date": ds_iso,
                "dev_rows": len(dev_norm),
                "src_rows": len(src_norm),
                "db_rows": len(db_norm),
                "dev_vs_src": {
                    "row_match": cmp_dev_src["row_match"],
                    "content_match": cmp_dev_src["content_match"],
                    "diff_count": cmp_dev_src["diff_count"],
                },
                "dev_vs_db": {
                    "row_match": cmp_dev_db["row_match"],
                    "content_match": cmp_dev_db["content_match"],
                    "diff_count": cmp_dev_db["diff_count"],
                },
                "src_vs_db": {
                    "row_match": cmp_src_db["row_match"],
                    "content_match": cmp_src_db["content_match"],
                    "diff_count": cmp_src_db["diff_count"],
                },
            }

            # Include diff details if any
            if cmp_dev_src.get("diff_details"):
                day_result["dev_vs_src"]["diff_details"] = cmp_dev_src["diff_details"]
            if cmp_dev_db.get("diff_details"):
                day_result["dev_vs_db"]["diff_details"] = cmp_dev_db["diff_details"]
            if cmp_src_db.get("diff_details"):
                day_result["src_vs_db"]["diff_details"] = cmp_src_db["diff_details"]
            if "only_in_1" in cmp_dev_src:
                day_result["dev_vs_src"]["only_in_dev"] = cmp_dev_src["only_in_1"]
                day_result["dev_vs_src"]["only_in_src"] = cmp_dev_src["only_in_2"]
            if "only_in_1" in cmp_dev_db:
                day_result["dev_vs_db"]["only_in_dev"] = cmp_dev_db["only_in_1"]
                day_result["dev_vs_db"]["only_in_db"] = cmp_dev_db["only_in_2"]
            if "only_in_1" in cmp_src_db:
                day_result["src_vs_db"]["only_in_src"] = cmp_src_db["only_in_1"]
                day_result["src_vs_db"]["only_in_db"] = cmp_src_db["only_in_2"]

            daily_results.append(day_result)

            if verbose:
                ds_match = "OK" if cmp_dev_src["content_match"] else f"DIFF({cmp_dev_src['diff_count']})"
                db_match = "OK" if cmp_dev_db["content_match"] else f"DIFF({cmp_dev_db['diff_count']})"
                sb_match = "OK" if cmp_src_db["content_match"] else f"DIFF({cmp_src_db['diff_count']})"
                sys.stdout.write(f"  Day{day_num:02d} {ds_iso}: Dev={len(dev_norm):>6} Src={len(src_norm):>6} DB={len(db_norm):>6}  "
                                 f"Dev/Src={ds_match:<12} Dev/DB={db_match:<12} Src/DB={sb_match}\n")
                sys.stdout.flush()

        # Compute totals for this sheet
        all_dev_src_match = all(d["dev_vs_src"]["content_match"] for d in daily_results)
        all_dev_db_match = all(d["dev_vs_db"]["content_match"] for d in daily_results)
        all_src_db_match = all(d["src_vs_db"]["content_match"] for d in daily_results)
        total_dev_rows = sum(d["dev_rows"] for d in daily_results)
        total_src_rows = sum(d["src_rows"] for d in daily_results)
        total_db_rows = sum(d["db_rows"] for d in daily_results)
        total_dev_src_diffs = sum(d["dev_vs_src"]["diff_count"] for d in daily_results)
        total_dev_db_diffs = sum(d["dev_vs_db"]["diff_count"] for d in daily_results)
        total_src_db_diffs = sum(d["src_vs_db"]["diff_count"] for d in daily_results)

        results["comparisons"][comp_key] = {
            "total_dev_rows": total_dev_rows,
            "total_src_rows": total_src_rows,
            "total_db_rows": total_db_rows,
            "dev_vs_src_all_match": all_dev_src_match,
            "dev_vs_db_all_match": all_dev_db_match,
            "src_vs_db_all_match": all_src_db_match,
            "total_dev_src_diffs": total_dev_src_diffs,
            "total_dev_db_diffs": total_dev_db_diffs,
            "total_src_db_diffs": total_src_db_diffs,
            "daily": daily_results,
        }

        total_sheets += 1
        if all_dev_src_match:
            total_pass_dev_src += 1
        if all_dev_db_match:
            total_pass_dev_db += 1
        if all_src_db_match:
            total_pass_src_db += 1

        if verbose:
            status_ds = "PASS" if all_dev_src_match else f"FAIL(diffs={total_dev_src_diffs})"
            status_db = "PASS" if all_dev_db_match else f"FAIL(diffs={total_dev_db_diffs})"
            status_sb = "PASS" if all_src_db_match else f"FAIL(diffs={total_src_db_diffs})"
            print(f"  TOTAL: Dev={total_dev_rows:,} Src={total_src_rows:,} DB={total_db_rows:,}"
                  f"  Dev/Src={status_ds}  Dev/DB={status_db}  Src/DB={status_sb}", flush=True)

    if conn:
        conn.close()

    results["summary"] = {
        "total_sheets_compared": total_sheets,
        "dev_vs_src_pass": total_pass_dev_src,
        "dev_vs_db_pass": total_pass_dev_db,
        "src_vs_db_pass": total_pass_src_db,
        "dev_vs_src_fail": total_sheets - total_pass_dev_src,
        "dev_vs_db_fail": total_sheets - total_pass_dev_db,
        "src_vs_db_fail": total_sheets - total_pass_src_db,
    }

    return results


# ─── Summary printer ─────────────────────────────────────────────────

def print_summary(results: dict) -> None:
    """Print a clean summary table."""
    print("\n" + "=" * 130)
    print("3-WAY COMPARISON SUMMARY (OC_Paste_S1_20251224)")
    print("=" * 130)
    print(f"{'Sheet':<35} {'Dev Rows':>10} {'Src Rows':>10} {'DB Rows':>10} {'Dev/Src':>12} {'Dev/DB':>12} {'Src/DB':>12}")
    print("-" * 130)

    for comp_key, data in results["comparisons"].items():
        ds_status = "PASS" if data["dev_vs_src_all_match"] else f"FAIL({data['total_dev_src_diffs']})"
        db_status = "PASS" if data["dev_vs_db_all_match"] else f"FAIL({data['total_dev_db_diffs']})"
        sb_status = "PASS" if data["src_vs_db_all_match"] else f"FAIL({data['total_src_db_diffs']})"
        print(f"{comp_key:<35} {data['total_dev_rows']:>10,} {data['total_src_rows']:>10,} "
              f"{data['total_db_rows']:>10,} {ds_status:>12} {db_status:>12} {sb_status:>12}")

    s = results["summary"]
    print("-" * 130)
    print(f"Total: {s['total_sheets_compared']} sheets | "
          f"Dev/Src: {s['dev_vs_src_pass']} PASS, {s['dev_vs_src_fail']} FAIL | "
          f"Dev/DB: {s['dev_vs_db_pass']} PASS, {s['dev_vs_db_fail']} FAIL | "
          f"Src/DB: {s['src_vs_db_pass']} PASS, {s['src_vs_db_fail']} FAIL")
    print("=" * 130)


# ─── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="3-way content comparison")
    parser.add_argument("--days", type=str, default=None,
                        help="Day range, e.g. '1-10' or '1-76' (default: all 76)")
    parser.add_argument("--module", type=str, default=None,
                        help="Filter to single module, e.g. 'module1'")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: tools/oc_3way_content_results.json)")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    args = parser.parse_args()

    if args.days:
        parts = args.days.split("-")
        day_range = range(int(parts[0]), int(parts[1]) + 1)
    else:
        day_range = range(1, NUM_DAYS + 1)

    t0 = time.time()
    results = run_comparison(day_range, module_filter=args.module, verbose=not args.quiet)
    elapsed = time.time() - t0
    results["elapsed_seconds"] = round(elapsed, 1)

    print_summary(results)

    # Save results
    out_path = Path(args.output) if args.output else PROJECT_ROOT / "tools" / "oc_3way_content_results_full.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
