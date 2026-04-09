# -*- coding: utf-8 -*-
"""
regression_compare.py - ChainSight 回归对比工具

用法:
    python tools/regression_compare.py <baseline_run_dir> <target_run_dir> [--tolerance 1e-6]

比较两次模拟运行结果，按 DataFrame 语义对比（排序、类型归一化后），
只比较业务结果字段，忽略日志时间戳、运行耗时等非业务字段。

覆盖范围:
    - Module1 日度输出 (OrderLog / ShipmentLog / CutLog / SupplyDemandLog)
    - Module3 日度输出 (NetDemand)
    - Module4 日度输出 (ProductionPlan / CapacityExceed / Validation / ChangeoverLog)
    - Module5 日度输出 (DeploymentPlan / UnfulfilledLog / StockOnHandLog)
    - Module6 日度输出 (DeliveryPlan / VehicleLog / TruckUsageLog)
    - Orchestrator 日度状态 CSV
    - Summary 汇总报告
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 不参与对比的列（非业务字段）
IGNORE_COLUMNS = {
    "run_id", "run_timestamp", "execution_time", "log_timestamp",
    "created_at", "updated_at", "elapsed_ms", "timestamp",
}

# 各模块日度输出的子目录名与文件前缀
MODULE_DIRS = {
    "module1": ("module1", "module1_output_"),
    "module3": ("module3", "Module3Output_"),
    "module4": ("module4", "Module4Output_"),
    "module5": ("module5", "Module5Output_"),
    "module6": ("module6", "Module6Output_"),
}

# Summary 报告文件 → 主要数值列（用于汇总对比）
SUMMARY_FILES = [
    "full_order_shipment_cut_report.xlsx",
    "full_production_plan_report.xlsx",
    "full_exceed_capacity_report.xlsx",
    "full_changeover_report.xlsx",
    "full_deployment_plan_report.xlsx",
    "full_delivery_plan_report.xlsx",
    "full_truck_usage_report.xlsx",
    "historical_inventory_record.csv",
]


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _norm_col(col: str) -> str:
    """列名统一小写去空格"""
    return str(col).strip().lower().replace(" ", "_")


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """对 DataFrame 做归一化处理，使其可比较"""
    df = df.copy()
    # 列名小写
    df.columns = [_norm_col(c) for c in df.columns]
    # 去掉非业务列
    drop_cols = [c for c in df.columns if c in IGNORE_COLUMNS]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")
    # 标识符列统一为 str
    for col in ("material", "location", "sending", "receiving", "sourcing",
                "from_location", "to_location", "line", "delegate_line"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()
    # 日期列统一为 str
    for col in df.columns:
        if "date" in col:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")
    # 数值列保留精度
    for col in df.select_dtypes(include=[np.floating]).columns:
        df[col] = df[col].round(8)
    # 排序：按所有列排序以消除行序差异
    sort_cols = list(df.columns)
    try:
        df = df.sort_values(sort_cols, ignore_index=True)
    except TypeError:
        pass  # 混合类型无法排序时跳过
    return df.reset_index(drop=True)


def _read_file(path: str) -> Dict[str, pd.DataFrame]:
    """
    读取 xlsx/csv/xls 文件，返回 {sheet_name: DataFrame}。
    CSV 文件返回 {"Sheet1": df}。
    """
    result = {}
    path_str = str(path)
    if path_str.endswith(".csv"):
        try:
            df = pd.read_csv(path, dtype=str if _is_id_heavy(path) else None)
            result["Sheet1"] = df
        except Exception as e:
            print(f"    [WARN] 读取失败: {path} -> {e}")
    elif path_str.endswith((".xlsx", ".xls")):
        try:
            xls = pd.ExcelFile(path)
            for sheet in xls.sheet_names:
                result[sheet] = pd.read_excel(xls, sheet_name=sheet)
        except Exception as e:
            print(f"    [WARN] 读取失败: {path} -> {e}")
    return result


def _is_id_heavy(path) -> bool:
    """判断是否是以标识符为主的文件（orchestrator 状态 CSV）"""
    name = Path(path).stem.lower()
    return any(k in name for k in (
        "unrestricted_inventory", "open_deployment", "planning_intransit",
        "space_quota", "production_plan_backlog", "delivery_gr",
        "production_gr", "shipments", "deployments",
        "historical_inventory", "beginning_inventory", "ending_inventory",
    ))


def compare_dataframes(
    df_old: pd.DataFrame,
    df_new: pd.DataFrame,
    label: str,
    tolerance: float = 1e-6,
) -> Tuple[bool, List[str]]:
    """
    比较两个 DataFrame，返回 (是否一致, 差异描述列表)。
    """
    messages = []
    old = _normalize_df(df_old)
    new = _normalize_df(df_new)

    # 列集合对比
    old_cols = set(old.columns)
    new_cols = set(new.columns)
    if old_cols != new_cols:
        only_old = old_cols - new_cols
        only_new = new_cols - old_cols
        if only_old:
            messages.append(f"  仅 baseline 有列: {only_old}")
        if only_new:
            messages.append(f"  仅 target 有列: {only_new}")
        common = sorted(old_cols & new_cols)
        old = old[common]
        new = new[common]

    # 行数对比
    if len(old) != len(new):
        messages.append(f"  行数不同: baseline={len(old)}, target={len(new)}")

    # 逐列值对比（取共同行数）
    min_rows = min(len(old), len(new))
    if min_rows > 0:
        old_cmp = old.head(min_rows).reset_index(drop=True)
        new_cmp = new.head(min_rows).reset_index(drop=True)

        diff_cols = []
        for col in old_cmp.columns:
            try:
                if old_cmp[col].dtype in (np.float64, np.float32):
                    if not np.allclose(
                        old_cmp[col].fillna(0).values,
                        new_cmp[col].fillna(0).values,
                        atol=tolerance, rtol=0, equal_nan=True,
                    ):
                        diff_cols.append(col)
                else:
                    if not old_cmp[col].fillna("").astype(str).equals(
                        new_cmp[col].fillna("").astype(str)
                    ):
                        diff_cols.append(col)
            except Exception:
                diff_cols.append(col)

        if diff_cols:
            messages.append(f"  值差异列 ({len(diff_cols)}): {diff_cols[:10]}")
            # 展示前 3 行差异样本
            for col in diff_cols[:3]:
                mask = old_cmp[col].fillna("").astype(str) != new_cmp[col].fillna("").astype(str)
                if mask.any():
                    idx = mask.idxmax()
                    messages.append(
                        f"    {col}[{idx}]: baseline={old_cmp[col].iloc[idx]!r} "
                        f"-> target={new_cmp[col].iloc[idx]!r}"
                    )

    is_match = len(messages) == 0
    return is_match, messages


# ---------------------------------------------------------------------------
# 对比逻辑
# ---------------------------------------------------------------------------

def compare_module_daily(
    baseline_dir: str,
    target_dir: str,
    module_name: str,
    subdir: str,
    prefix: str,
    tolerance: float,
) -> Tuple[int, int, List[str]]:
    """对比某模块的日度输出文件，返回 (pass_count, fail_count, messages)"""
    base_mod = os.path.join(baseline_dir, subdir)
    tgt_mod = os.path.join(target_dir, subdir)

    if not os.path.isdir(base_mod):
        return 0, 0, [f"  [SKIP] baseline 目录不存在: {base_mod}"]
    if not os.path.isdir(tgt_mod):
        return 0, 0, [f"  [SKIP] target 目录不存在: {tgt_mod}"]

    # 收集 baseline 中的日度 xlsx 文件
    base_files = sorted([
        f for f in os.listdir(base_mod)
        if f.startswith(prefix) and f.endswith(".xlsx")
    ])
    tgt_files = sorted([
        f for f in os.listdir(tgt_mod)
        if f.startswith(prefix) and f.endswith(".xlsx")
    ])

    if not base_files:
        return 0, 0, [f"  [SKIP] baseline 无 {prefix}*.xlsx 文件"]

    passed = 0
    failed = 0
    msgs = []

    # 取交集日期
    base_set = set(base_files)
    tgt_set = set(tgt_files)
    only_base = base_set - tgt_set
    only_tgt = tgt_set - base_set
    if only_base:
        msgs.append(f"  仅 baseline 有 ({len(only_base)}): {sorted(only_base)[:5]}")
    if only_tgt:
        msgs.append(f"  仅 target 有 ({len(only_tgt)}): {sorted(only_tgt)[:5]}")

    common_files = sorted(base_set & tgt_set)
    for fname in common_files:
        sheets_old = _read_file(os.path.join(base_mod, fname))
        sheets_new = _read_file(os.path.join(tgt_mod, fname))

        all_sheets = sorted(set(sheets_old.keys()) | set(sheets_new.keys()))
        file_ok = True
        for sheet in all_sheets:
            label = f"{module_name}/{fname}/{sheet}"
            if sheet not in sheets_old:
                msgs.append(f"  [DIFF] {label}: 仅 target 有此 sheet")
                file_ok = False
                continue
            if sheet not in sheets_new:
                msgs.append(f"  [DIFF] {label}: 仅 baseline 有此 sheet")
                file_ok = False
                continue

            ok, diff_msgs = compare_dataframes(
                sheets_old[sheet], sheets_new[sheet], label, tolerance
            )
            if not ok:
                msgs.append(f"  [DIFF] {label}:")
                msgs.extend(diff_msgs)
                file_ok = False

        if file_ok:
            passed += 1
        else:
            failed += 1

    return passed, failed, msgs


def compare_orchestrator(
    baseline_dir: str,
    target_dir: str,
    tolerance: float,
) -> Tuple[int, int, List[str]]:
    """对比 orchestrator/ 日度状态 CSV"""
    base_orch = os.path.join(baseline_dir, "orchestrator")
    tgt_orch = os.path.join(target_dir, "orchestrator")

    if not os.path.isdir(base_orch):
        return 0, 0, ["  [SKIP] baseline orchestrator/ 不存在"]
    if not os.path.isdir(tgt_orch):
        return 0, 0, ["  [SKIP] target orchestrator/ 不存在"]

    base_csvs = sorted([f for f in os.listdir(base_orch) if f.endswith(".csv")])
    tgt_csvs = sorted([f for f in os.listdir(tgt_orch) if f.endswith(".csv")])

    base_set = set(base_csvs)
    tgt_set = set(tgt_csvs)
    msgs = []
    only_base = base_set - tgt_set
    only_tgt = tgt_set - base_set
    if only_base:
        msgs.append(f"  仅 baseline 有 ({len(only_base)}): {sorted(only_base)[:5]}")
    if only_tgt:
        msgs.append(f"  仅 target 有 ({len(only_tgt)}): {sorted(only_tgt)[:5]}")

    passed = 0
    failed = 0
    for fname in sorted(base_set & tgt_set):
        sheets_old = _read_file(os.path.join(base_orch, fname))
        sheets_new = _read_file(os.path.join(tgt_orch, fname))
        label = f"orchestrator/{fname}"
        df_old = sheets_old.get("Sheet1", pd.DataFrame())
        df_new = sheets_new.get("Sheet1", pd.DataFrame())
        ok, diff_msgs = compare_dataframes(df_old, df_new, label, tolerance)
        if ok:
            passed += 1
        else:
            failed += 1
            msgs.append(f"  [DIFF] {label}:")
            msgs.extend(diff_msgs)

    return passed, failed, msgs


def compare_summary(
    baseline_dir: str,
    target_dir: str,
    tolerance: float,
) -> Tuple[int, int, List[str]]:
    """对比 summary/ 汇总报告"""
    base_sum = os.path.join(baseline_dir, "summary")
    tgt_sum = os.path.join(target_dir, "summary")

    if not os.path.isdir(base_sum):
        return 0, 0, ["  [SKIP] baseline summary/ 不存在"]
    if not os.path.isdir(tgt_sum):
        return 0, 0, ["  [SKIP] target summary/ 不存在"]

    passed = 0
    failed = 0
    msgs = []

    for fname in SUMMARY_FILES:
        base_path = os.path.join(base_sum, fname)
        tgt_path = os.path.join(tgt_sum, fname)

        if not os.path.exists(base_path):
            continue
        if not os.path.exists(tgt_path):
            msgs.append(f"  [DIFF] summary/{fname}: 仅 baseline 有")
            failed += 1
            continue

        sheets_old = _read_file(base_path)
        sheets_new = _read_file(tgt_path)

        all_sheets = sorted(set(sheets_old.keys()) | set(sheets_new.keys()))
        file_ok = True
        for sheet in all_sheets:
            label = f"summary/{fname}/{sheet}"
            if sheet not in sheets_old:
                msgs.append(f"  [DIFF] {label}: 仅 target 有此 sheet")
                file_ok = False
                continue
            if sheet not in sheets_new:
                msgs.append(f"  [DIFF] {label}: 仅 baseline 有此 sheet")
                file_ok = False
                continue
            ok, diff_msgs = compare_dataframes(
                sheets_old[sheet], sheets_new[sheet], label, tolerance
            )
            if not ok:
                msgs.append(f"  [DIFF] {label}:")
                msgs.extend(diff_msgs)
                file_ok = False

        if file_ok:
            passed += 1
        else:
            failed += 1

    return passed, failed, msgs


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def run_comparison(baseline_dir: str, target_dir: str, tolerance: float = 1e-6) -> bool:
    """执行完整回归对比，返回是否全部通过"""
    print("=" * 78)
    print("ChainSight 回归对比")
    print("=" * 78)
    print(f"  Baseline: {baseline_dir}")
    print(f"  Target  : {target_dir}")
    print(f"  Tolerance: {tolerance}")
    print()

    total_pass = 0
    total_fail = 0
    all_msgs: List[str] = []

    # --- 模块日度输出 ---
    for mod_name, (subdir, prefix) in MODULE_DIRS.items():
        print(f"[{mod_name}] 日度输出对比 ...")
        p, f, msgs = compare_module_daily(
            baseline_dir, target_dir, mod_name, subdir, prefix, tolerance
        )
        total_pass += p
        total_fail += f
        if msgs:
            all_msgs.append(f"\n--- {mod_name} ---")
            all_msgs.extend(msgs)
        status = "PASS" if f == 0 and p > 0 else ("SKIP" if p == 0 and f == 0 else "FAIL")
        print(f"  {status}: {p} passed, {f} failed")

    # --- Orchestrator ---
    print(f"[orchestrator] 状态 CSV 对比 ...")
    p, f, msgs = compare_orchestrator(baseline_dir, target_dir, tolerance)
    total_pass += p
    total_fail += f
    if msgs:
        all_msgs.append("\n--- orchestrator ---")
        all_msgs.extend(msgs)
    status = "PASS" if f == 0 and p > 0 else ("SKIP" if p == 0 and f == 0 else "FAIL")
    print(f"  {status}: {p} passed, {f} failed")

    # --- Summary ---
    print(f"[summary] 汇总报告对比 ...")
    p, f, msgs = compare_summary(baseline_dir, target_dir, tolerance)
    total_pass += p
    total_fail += f
    if msgs:
        all_msgs.append("\n--- summary ---")
        all_msgs.extend(msgs)
    status = "PASS" if f == 0 and p > 0 else ("SKIP" if p == 0 and f == 0 else "FAIL")
    print(f"  {status}: {p} passed, {f} failed")

    # --- 汇总 ---
    print()
    print("=" * 78)
    all_pass = total_fail == 0 and total_pass > 0
    if all_pass:
        print(f"RESULT: ALL PASSED ({total_pass} checks)")
    else:
        print(f"RESULT: FAILED ({total_fail} failures, {total_pass} passed)")
    print("=" * 78)

    if all_msgs:
        print("\n详细差异:\n")
        for m in all_msgs:
            print(m)

    return all_pass


def main():
    parser = argparse.ArgumentParser(
        description="ChainSight 回归对比工具 - 比较两次运行结果是否一致",
    )
    parser.add_argument("baseline", help="基线运行目录 (如 outputs/BC_S5/run_20260101_120000)")
    parser.add_argument("target", help="目标运行目录 (如 outputs/BC_S5/run_20260102_120000)")
    parser.add_argument(
        "--tolerance", "-t", type=float, default=1e-6,
        help="浮点数比较容差 (默认 1e-6)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.baseline):
        print(f"ERROR: baseline 目录不存在: {args.baseline}", file=sys.stderr)
        sys.exit(2)
    if not os.path.isdir(args.target):
        print(f"ERROR: target 目录不存在: {args.target}", file=sys.stderr)
        sys.exit(2)

    success = run_comparison(args.baseline, args.target, args.tolerance)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
