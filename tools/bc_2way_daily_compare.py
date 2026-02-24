#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bc_2way_daily_compare.py

Dev vs Src 逐日逐Sheet内容级对比工具 — BC_S5 版本。
与 bc_3way_content_compare.py 保持相同的报告维度，但仅对比 Dev vs Src（无DB）。
采样模式：每10天取样 + 最后一天，减少运行时间。

对比维度:
  1. 行数是否一致
  2. 列名是否一致（取交集对比）
  3. 数值内容是否一致（频率/多集对比，无排序敏感）

输出:
  - JSON格式的全量对比结果 (tools/bc_2way_daily_results.json)
  - 控制台摘要（与3-way报告同格式）

Usage:
    python tools/bc_2way_daily_compare.py [--days 1-87] [--module module1] [--every 10]
"""
from __future__ import annotations
import json, sys, time, warnings, argparse
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter
from typing import Any

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ─── Configuration ───────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DEV_DIR = PROJECT_ROOT / "ChainSight_Dev" / "BC_S5" / "run_20260211_181635"
SRC_DIR = PROJECT_ROOT / "outputs" / "BC_S5" / "run_20260211_145215"

START_DATE = datetime(2025, 10, 5)
NUM_DAYS = 87
SAMPLE_EVERY = 10  # default: every 10 days

# ─── Sheet/Table mapping (same as 3-way) ────────────────────────────
SHEET_MAPPINGS = [
    # Module1
    {"module": "module1", "file_pattern": "module1_output_{date}.xlsx",
     "sheet": "OrderLog",
     "key_cols": ["date", "material", "location", "demand_type"],
     "compare_cols": ["quantity", "simulation_date", "advance_days"],
     "xlsx_col_rename": {}},
    {"module": "module1", "file_pattern": "module1_output_{date}.xlsx",
     "sheet": "ShipmentLog",
     "key_cols": ["date", "material", "location", "demand_type", "order_id"],
     "compare_cols": ["quantity"],
     "xlsx_col_rename": {}},
    {"module": "module1", "file_pattern": "module1_output_{date}.xlsx",
     "sheet": "CutLog",
     "key_cols": ["date", "material", "location"],
     "compare_cols": ["quantity"],
     "xlsx_col_rename": {}},
    {"module": "module1", "file_pattern": "module1_output_{date}.xlsx",
     "sheet": "SupplyDemandLog",
     "key_cols": ["date", "material", "location", "demand_element"],
     "compare_cols": ["quantity"],
     "xlsx_col_rename": {}},
    {"module": "module1", "file_pattern": "module1_output_{date}.xlsx",
     "sheet": "Summary",
     "key_cols": ["date"],
     "compare_cols": ["total_orders", "total_shipments", "total_cuts", "total_supplydemand"],
     "xlsx_col_rename": {"Total_Orders": "total_orders", "Total_Shipments": "total_shipments",
                          "Total_Cuts": "total_cuts", "Total_SupplyDemand": "total_supplydemand",
                          "Date": "date"}},
    # Module3
    {"module": "module3", "file_pattern": "Module3Output_{date}.xlsx",
     "sheet": "NetDemand",
     "key_cols": ["material", "location", "requirement_date", "demand_element", "layer"],
     "compare_cols": ["quantity", "simulation_date", "horizon_days"],
     "xlsx_col_rename": {}},
    # Module4
    {"module": "module4", "file_pattern": "Module4Output_{date}.xlsx",
     "sheet": "ProductionPlan",
     "key_cols": ["material", "location", "line", "simulation_date", "production_plan_date"],
     "compare_cols": ["available_date", "uncon_planned_qty", "con_planned_qty", "produced_qty",
                      "changeover_id", "changeover_time", "changeover_time_remaining",
                      "is_first_changeover_day"],
     "xlsx_col_rename": {}},
    {"module": "module4", "file_pattern": "Module4Output_{date}.xlsx",
     "sheet": "CapacityExceed",
     "key_cols": ["material", "location", "line", "simulation_date", "exceed_type"],
     "compare_cols": ["exceed_qty"],
     "xlsx_col_rename": {}},
    {"module": "module4", "file_pattern": "Module4Output_{date}.xlsx",
     "sheet": "Validation",
     "key_cols": [],
     "compare_cols": [],
     "xlsx_col_rename": {}},
    {"module": "module4", "file_pattern": "Module4Output_{date}.xlsx",
     "sheet": "ChangeoverLog",
     "key_cols": ["date", "location", "line", "changeover_type"],
     "compare_cols": ["count", "time", "cost", "mu_loss"],
     "xlsx_col_rename": {}},
    # Module5
    {"module": "module5", "file_pattern": "Module5Output_{date}.xlsx",
     "sheet": "DeploymentPlan",
     "key_cols": ["date", "material", "sending", "receiving", "demand_element"],
     "compare_cols": ["demand_qty", "planned_qty", "deployed_qty_invCon",
                      "deploy_qty_with_plan_order", "deploy_from_in_transit",
                      "deploy_from_open_deployment_inbound", "deploy_from_future_production",
                      "planned_delivery_date", "orig_location", "leadtime", "is_cross_node",
                      "deployed_qty_invCon_push", "deployed_qty", "quota"],
     "xlsx_col_rename": {}},
    {"module": "module5", "file_pattern": "Module5Output_{date}.xlsx",
     "sheet": "UnfulfilledLog",
     "key_cols": ["date", "sending", "receiving", "demand_element"],
     "compare_cols": ["demand_qty", "unfulfilled_qty", "reason"],
     "xlsx_col_rename": {}},
    {"module": "module5", "file_pattern": "Module5Output_{date}.xlsx",
     "sheet": "StockOnHandLog",
     "key_cols": ["material", "location", "date"],
     "compare_cols": ["beginning_soh", "production", "in_transit", "delivery_gr",
                      "today_shipment", "deployed_qty", "ending_soh"],
     "xlsx_col_rename": {}},
    {"module": "module5", "file_pattern": "Module5Output_{date}.xlsx",
     "sheet": "Validation",
     "key_cols": [],
     "compare_cols": ["no", "issue"],
     "xlsx_col_rename": {"No": "no", "Issue": "issue"}},
    # Module6
    {"module": "module6", "file_pattern": "Module6Output_{date}.xlsx",
     "sheet": "DeliveryPlan",
     "key_cols": ["vehicle_uid", "ori_deployment_uid", "material", "sending", "receiving"],
     "compare_cols": ["planned_deployment_date", "actual_ship_date", "actual_delivery_date",
                      "delivery_qty", "truck_type", "truck_load_pct", "WFR", "VFR"],
     "xlsx_col_rename": {}},
    {"module": "module6", "file_pattern": "Module6Output_{date}.xlsx",
     "sheet": "VehicleLog",
     "key_cols": ["date", "sending", "receiving", "truck_type", "vehicle_uid"],
     "compare_cols": ["vehicle_no", "total_units", "total_weight", "total_volume",
                      "WFR", "VFR", "trigger"],
     "xlsx_col_rename": {}},
    {"module": "module6", "file_pattern": "Module6Output_{date}.xlsx",
     "sheet": "TruckUsageLog",
     "key_cols": ["date", "sending", "receiving", "truck_type"],
     "compare_cols": ["truck_used"],
     "xlsx_col_rename": {}},
]


# ─── Helper functions ────────────────────────────────────────────────

def sim_date_str(day_num: int) -> str:
    return (START_DATE + timedelta(days=day_num - 1)).strftime("%Y%m%d")

def sim_date_iso(day_num: int) -> str:
    return (START_DATE + timedelta(days=day_num - 1)).strftime("%Y-%m-%d")

def sample_days(total: int, every: int) -> list[int]:
    """Generate sampled day numbers: 1, 1+every, 1+2*every, ..., last day."""
    days = list(range(1, total + 1, every))
    if total not in days:
        days.append(total)
    return days


def load_xlsx_sheet(filepath: Path, sheet_name: str,
                    col_rename: dict | None = None) -> pd.DataFrame | None:
    if not filepath.exists():
        return None
    try:
        df = pd.read_excel(filepath, sheet_name=sheet_name, engine="openpyxl")
        if col_rename:
            df = df.rename(columns=col_rename)
        return df
    except Exception:
        return None


def normalize_value(val):
    """Normalize a value for frequency comparison."""
    if pd.isna(val):
        return ""
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    if isinstance(val, (float, np.floating)):
        # Remove trailing .0
        if val == int(val):
            return str(int(val))
        return str(round(float(val), 6))
    s = str(val).strip()
    # Normalize date strings: "2025-10-15 00:00:00" -> "2025-10-15"
    if s.endswith(" 00:00:00"):
        s = s[:-9]
    if s in ("NaT", "nan", "None", "NaN"):
        return ""
    # Try numeric normalization
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return str(round(f, 6))
    except (ValueError, OverflowError):
        pass
    return s


def df_to_counter(df: pd.DataFrame, cols: list[str]) -> Counter:
    """Convert selected columns of a DataFrame to a Counter of normalized row tuples."""
    avail = [c for c in cols if c in df.columns]
    if not avail:
        avail = list(df.columns)
    sub = df[avail]
    counter = Counter()
    for _, row in sub.iterrows():
        key = tuple(normalize_value(v) for v in row)
        counter[key] += 1
    return counter


def compare_sheets(df_dev: pd.DataFrame | None, df_src: pd.DataFrame | None,
                   key_cols: list[str], compare_cols: list[str]) -> dict:
    """
    Compare Dev vs Src DataFrames using frequency-based approach.
    Returns result dict matching 3-way report format.
    """
    dev = df_dev if df_dev is not None else pd.DataFrame()
    src = df_src if df_src is not None else pd.DataFrame()

    result = {
        "row_match": len(dev) == len(src),
        "content_match": False,
        "diff_count": 0,
    }

    if dev.empty and src.empty:
        result["content_match"] = True
        return result

    if dev.empty or src.empty:
        result["diff_count"] = max(len(dev), len(src))
        return result

    # Determine columns to compare
    all_cols = key_cols + compare_cols if compare_cols else key_cols
    if not all_cols:
        all_cols = list(set(dev.columns) & set(src.columns))
    avail = [c for c in all_cols if c in dev.columns and c in src.columns]

    if not avail:
        # Fallback: use all common columns
        avail = sorted(set(dev.columns) & set(src.columns))
    if not avail:
        result["content_match"] = result["row_match"]
        return result

    c_dev = df_to_counter(dev, avail)
    c_src = df_to_counter(src, avail)

    if c_dev == c_src:
        result["content_match"] = True
        result["diff_count"] = 0
        return result

    # Compute diffs
    only_dev = c_dev - c_src
    only_src = c_src - c_dev
    n_only_dev = sum(only_dev.values())
    n_only_src = sum(only_src.values())
    result["diff_count"] = n_only_dev + n_only_src
    result["content_match"] = False
    result["only_in_dev"] = n_only_dev
    result["only_in_src"] = n_only_src

    return result


# ─── Main comparison loop ────────────────────────────────────────────

def run_comparison(day_list: list[int], module_filter: str | None = None,
                   verbose: bool = True) -> dict:
    results: dict[str, Any] = {
        "run_time": datetime.now().isoformat(),
        "day_range": f"{min(day_list)}-{max(day_list)}",
        "sample_days": day_list,
        "total_sample_days": len(day_list),
        "comparisons": {},
    }

    total_sheets = 0
    total_pass = 0

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
        xlsx_rename = mapping.get("xlsx_col_rename") or None

        for day_num in day_list:
            ds = sim_date_str(day_num)
            ds_iso = sim_date_iso(day_num)

            dev_file = DEV_DIR / mod / mapping["file_pattern"].format(date=ds)
            src_file = SRC_DIR / mod / mapping["file_pattern"].format(date=ds)

            dev_df = load_xlsx_sheet(dev_file, sheet, col_rename=xlsx_rename)
            src_df = load_xlsx_sheet(src_file, sheet, col_rename=xlsx_rename)

            cmp = compare_sheets(dev_df, src_df,
                                 mapping["key_cols"], mapping["compare_cols"])

            day_result = {
                "day": day_num,
                "date": ds_iso,
                "dev_rows": len(dev_df) if dev_df is not None else 0,
                "src_rows": len(src_df) if src_df is not None else 0,
                "dev_vs_src": {
                    "row_match": cmp["row_match"],
                    "content_match": cmp["content_match"],
                    "diff_count": cmp["diff_count"],
                },
            }
            if "only_in_dev" in cmp:
                day_result["dev_vs_src"]["only_in_dev"] = cmp["only_in_dev"]
                day_result["dev_vs_src"]["only_in_src"] = cmp["only_in_src"]

            daily_results.append(day_result)

            if verbose:
                ds_match = "OK" if cmp["content_match"] else f"DIFF({cmp['diff_count']})"
                sys.stdout.write(f"  Day{day_num:02d} {ds_iso}: "
                                 f"Dev={day_result['dev_rows']:>6} "
                                 f"Src={day_result['src_rows']:>6}  "
                                 f"Dev/Src={ds_match}\n")
                sys.stdout.flush()

        # Sheet-level totals
        all_match = all(d["dev_vs_src"]["content_match"] for d in daily_results)
        total_dev_rows = sum(d["dev_rows"] for d in daily_results)
        total_src_rows = sum(d["src_rows"] for d in daily_results)
        total_diffs = sum(d["dev_vs_src"]["diff_count"] for d in daily_results)

        results["comparisons"][comp_key] = {
            "total_dev_rows": total_dev_rows,
            "total_src_rows": total_src_rows,
            "dev_vs_src_all_match": all_match,
            "total_dev_src_diffs": total_diffs,
            "daily": daily_results,
        }

        total_sheets += 1
        if all_match:
            total_pass += 1

        if verbose:
            status = "PASS" if all_match else f"FAIL(diffs={total_diffs})"
            print(f"  TOTAL: Dev={total_dev_rows:,} Src={total_src_rows:,}  "
                  f"Dev/Src={status}", flush=True)

    results["summary"] = {
        "total_sheets_compared": total_sheets,
        "dev_vs_src_pass": total_pass,
        "dev_vs_src_fail": total_sheets - total_pass,
    }

    return results


# ─── Summary printer (same format as 3-way) ─────────────────────────

def print_summary(results: dict) -> None:
    print("\n" + "=" * 100)
    print("BC COMPARISON SUMMARY (Dev vs Src) — BC_S5")
    print(f"Sample: {results['total_sample_days']} days out of {NUM_DAYS} "
          f"(every {SAMPLE_EVERY} days)")
    print("=" * 100)
    print(f"{'Sheet':<35} {'Dev Rows':>10} {'Src Rows':>10} {'Dev/Src':>12}")
    print("-" * 100)

    for comp_key, data in results["comparisons"].items():
        ds_status = "PASS" if data["dev_vs_src_all_match"] else \
            f"FAIL({data['total_dev_src_diffs']})"
        print(f"{comp_key:<35} {data['total_dev_rows']:>10,} "
              f"{data['total_src_rows']:>10,} {ds_status:>12}")

    s = results["summary"]
    print("-" * 100)
    print(f"Total: {s['total_sheets_compared']} sheets | "
          f"Dev/Src: {s['dev_vs_src_pass']} PASS, {s['dev_vs_src_fail']} FAIL")
    print("=" * 100)


# ─── CLI ─────────────────────────────────────────────────────────────

def main():
    global SAMPLE_EVERY
    parser = argparse.ArgumentParser(description="BC_S5 Dev vs Src daily comparison")
    parser.add_argument("--days", type=str, default=None,
                        help="Day range, e.g. '1-87' (default: all 87)")
    parser.add_argument("--module", type=str, default=None,
                        help="Filter to single module, e.g. 'module1'")
    parser.add_argument("--every", type=int, default=10,
                        help="Sample interval (default: 10)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    SAMPLE_EVERY = args.every

    if args.days:
        parts = args.days.split("-")
        total = int(parts[1])
        start = int(parts[0])
        all_days = list(range(start, total + 1))
        day_list = [d for d in all_days if (d - start) % args.every == 0]
        if total not in day_list:
            day_list.append(total)
    else:
        day_list = sample_days(NUM_DAYS, args.every)

    t0 = time.time()
    print(f"Comparing Dev vs Src: sampling {len(day_list)} days "
          f"(every {args.every}) from {NUM_DAYS} total")
    print(f"Sample days: {day_list}")
    print(f"Dev: {DEV_DIR}")
    print(f"Src: {SRC_DIR}")

    results = run_comparison(day_list, module_filter=args.module,
                             verbose=not args.quiet)
    elapsed = time.time() - t0
    results["elapsed_seconds"] = round(elapsed, 1)

    print_summary(results)

    out_path = Path(args.output) if args.output else \
        PROJECT_ROOT / "tools" / "bc_2way_daily_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
