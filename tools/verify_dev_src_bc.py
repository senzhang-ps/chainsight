#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_dev_src_bc.py - 验证 BC 场景 Dev 与 Src 两个本地文件版本的数据一致性
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
DEV_DIR = PROJECT_ROOT / "ChainSight_Dev" / "BC_S5" / "run_20260211_181635"
SRC_DIR = PROJECT_ROOT / "outputs" / "BC_S5" / "run_20260211_145215"

START_DATE = datetime(2025, 10, 5)
NUM_DAYS = 87

# 要比对的模块和文件 (BC 场景实际的 Sheet 名称)
COMPARE_ITEMS = [
    ("module1", "module1_output_{date}.xlsx", ["OrderLog", "ShipmentLog", "CutLog", "SupplyDemandLog", "Summary"]),
    ("module3", "Module3Output_{date}.xlsx", ["NetDemand"]),
    ("module4", "Module4Output_{date}.xlsx", ["ProductionPlan", "CapacityExceed", "ChangeoverLog"]),
    ("module5", "Module5Output_{date}.xlsx", ["DeploymentPlan", "UnfulfilledLog", "StockOnHandLog", "Validation"]),
]


def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, name: str) -> dict:
    """比对两个DataFrame"""
    result = {"name": name, "match": True, "issues": []}
    
    # 行数
    if len(df1) != len(df2):
        result["match"] = False
        result["issues"].append(f"行数不同: Dev={len(df1)}, Src={len(df2)}")
    
    # 列名
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    if cols1 != cols2:
        result["match"] = False
        only_in_dev = cols1 - cols2
        only_in_src = cols2 - cols1
        if only_in_dev:
            result["issues"].append(f"仅Dev有列: {only_in_dev}")
        if only_in_src:
            result["issues"].append(f"仅Src有列: {only_in_src}")
    
    # 数值比对（只比对共有列）
    common_cols = list(cols1 & cols2)
    if len(df1) == len(df2) and common_cols:
        df1_sorted = df1[common_cols].sort_values(by=common_cols).reset_index(drop=True)
        df2_sorted = df2[common_cols].sort_values(by=common_cols).reset_index(drop=True)
        
        for col in common_cols:
            try:
                if df1_sorted[col].dtype in ['float64', 'int64'] and df2_sorted[col].dtype in ['float64', 'int64']:
                    diff = (df1_sorted[col] - df2_sorted[col]).abs()
                    mismatch_count = (diff > 1e-6).sum()
                    if mismatch_count > 0:
                        result["match"] = False
                        result["issues"].append(f"列 {col}: {mismatch_count} 行数值差异")
                else:
                    mismatch = (df1_sorted[col].astype(str) != df2_sorted[col].astype(str)).sum()
                    if mismatch > 0:
                        result["match"] = False
                        result["issues"].append(f"列 {col}: {mismatch} 行内容差异")
            except Exception as e:
                result["issues"].append(f"列 {col} 比对异常: {e}")
    
    return result


def main():
    print("=" * 60)
    print("BC 场景 Dev vs Src 数据一致性验证")
    print("=" * 60)
    print(f"Dev 路径: {DEV_DIR}")
    print(f"Src 路径: {SRC_DIR}")
    print()
    
    total_sheets = 0
    pass_count = 0
    fail_count = 0
    failures = []
    
    # 抽查几天（第1天、中间、最后一天）
    sample_days = [0, 10, 40, 86]  # day indices
    
    for day_idx in sample_days:
        current_date = START_DATE + timedelta(days=day_idx)
        date_str = current_date.strftime("%Y%m%d")
        print(f"\n--- Day {day_idx + 1} ({date_str}) ---")
        
        for module, pattern, sheets in COMPARE_ITEMS:
            filename = pattern.format(date=date_str)
            dev_file = DEV_DIR / module / filename
            src_file = SRC_DIR / module / filename
            
            if not dev_file.exists():
                print(f"  [SKIP] {module}/{filename} - Dev文件不存在")
                continue
            if not src_file.exists():
                print(f"  [SKIP] {module}/{filename} - Src文件不存在")
                continue
            
            for sheet in sheets:
                total_sheets += 1
                try:
                    df_dev = pd.read_excel(dev_file, sheet_name=sheet)
                    df_src = pd.read_excel(src_file, sheet_name=sheet)
                    
                    result = compare_dataframes(df_dev, df_src, f"{module}/{sheet}")
                    
                    if result["match"]:
                        print(f"  [PASS] {module}/{sheet}: {len(df_dev)} rows")
                        pass_count += 1
                    else:
                        print(f"  [FAIL] {module}/{sheet}: {result['issues']}")
                        fail_count += 1
                        failures.append({
                            "day": day_idx + 1,
                            "date": date_str,
                            "module": module,
                            "sheet": sheet,
                            "issues": result["issues"]
                        })
                except Exception as e:
                    print(f"  [ERROR] {module}/{sheet}: {e}")
                    fail_count += 1
    
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print(f"检查Sheet总数: {total_sheets}")
    print(f"通过: {pass_count}")
    print(f"失败: {fail_count}")
    
    if failures:
        print("\n失败详情:")
        for f in failures:
            print(f"  Day {f['day']} ({f['date']}) {f['module']}/{f['sheet']}: {f['issues']}")
    
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
