#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_dev_src_bc_detailed.py - BC 场景 Dev 与 Src 详细数值比对
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


def compare_dataframes_detailed(df1: pd.DataFrame, df2: pd.DataFrame, name: str, date_str: str) -> dict:
    """详细比对两个DataFrame，返回差异信息"""
    result = {
        "name": name,
        "date": date_str,
        "match": True,
        "row_count_dev": len(df1),
        "row_count_src": len(df2),
        "issues": [],
        "value_diffs": []
    }
    
    # 行数检查
    if len(df1) != len(df2):
        result["match"] = False
        result["issues"].append(f"行数不同: Dev={len(df1)}, Src={len(df2)}")
        return result
    
    if len(df1) == 0:
        return result
    
    # 列名检查
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
    common_cols = sorted(list(cols1 & cols2))
    
    # 尝试排序以对齐数据
    try:
        df1_sorted = df1[common_cols].sort_values(by=common_cols).reset_index(drop=True)
        df2_sorted = df2[common_cols].sort_values(by=common_cols).reset_index(drop=True)
    except:
        df1_sorted = df1[common_cols].reset_index(drop=True)
        df2_sorted = df2[common_cols].reset_index(drop=True)
    
    for col in common_cols:
        try:
            col1 = df1_sorted[col]
            col2 = df2_sorted[col]
            
            # 数值列
            if pd.api.types.is_numeric_dtype(col1) and pd.api.types.is_numeric_dtype(col2):
                # 处理 NaN
                nan_mask = col1.isna() | col2.isna()
                both_nan = col1.isna() & col2.isna()
                
                # 只有一边是 NaN
                one_nan = nan_mask & ~both_nan
                if one_nan.sum() > 0:
                    result["match"] = False
                    result["issues"].append(f"列 {col}: {one_nan.sum()} 行 NaN 不一致")
                
                # 数值差异
                valid_mask = ~nan_mask
                if valid_mask.sum() > 0:
                    diff = (col1[valid_mask] - col2[valid_mask]).abs()
                    mismatch_mask = diff > 1e-6
                    mismatch_count = mismatch_mask.sum()
                    
                    if mismatch_count > 0:
                        result["match"] = False
                        max_diff = diff[mismatch_mask].max()
                        result["issues"].append(f"列 {col}: {mismatch_count} 行数值差异 (最大差异: {max_diff:.6f})")
                        
                        # 记录前5个差异
                        diff_indices = diff[mismatch_mask].head(5).index.tolist()
                        for idx in diff_indices:
                            result["value_diffs"].append({
                                "col": col,
                                "row": idx,
                                "dev_val": float(col1.iloc[idx]) if not pd.isna(col1.iloc[idx]) else None,
                                "src_val": float(col2.iloc[idx]) if not pd.isna(col2.iloc[idx]) else None,
                                "diff": float(diff.iloc[idx])
                            })
            else:
                # 非数值列
                mismatch = (col1.astype(str) != col2.astype(str)).sum()
                if mismatch > 0:
                    result["match"] = False
                    result["issues"].append(f"列 {col}: {mismatch} 行内容差异")
        except Exception as e:
            result["issues"].append(f"列 {col} 比对异常: {e}")
    
    return result


def main():
    print("=" * 70)
    print("BC 场景 Dev vs Src 详细数值比对")
    print("=" * 70)
    print(f"Dev 路径: {DEV_DIR}")
    print(f"Src 路径: {SRC_DIR}")
    print(f"比对天数: {NUM_DAYS} 天")
    print()
    
    total_sheets = 0
    pass_count = 0
    fail_count = 0
    all_failures = []
    
    # 比对所有87天
    for day_idx in range(NUM_DAYS):
        current_date = START_DATE + timedelta(days=day_idx)
        date_str = current_date.strftime("%Y%m%d")
        
        day_pass = 0
        day_fail = 0
        
        for module, pattern, sheets in COMPARE_ITEMS:
            filename = pattern.format(date=date_str)
            dev_file = DEV_DIR / module / filename
            src_file = SRC_DIR / module / filename
            
            if not dev_file.exists() or not src_file.exists():
                continue
            
            for sheet in sheets:
                total_sheets += 1
                try:
                    df_dev = pd.read_excel(dev_file, sheet_name=sheet)
                    df_src = pd.read_excel(src_file, sheet_name=sheet)
                    
                    result = compare_dataframes_detailed(df_dev, df_src, f"{module}/{sheet}", date_str)
                    
                    if result["match"]:
                        pass_count += 1
                        day_pass += 1
                    else:
                        fail_count += 1
                        day_fail += 1
                        all_failures.append(result)
                        
                except Exception as e:
                    fail_count += 1
                    day_fail += 1
                    all_failures.append({
                        "name": f"{module}/{sheet}",
                        "date": date_str,
                        "issues": [f"读取异常: {e}"]
                    })
        
        # 每天输出一行状态
        status = "PASS" if day_fail == 0 else f"FAIL ({day_fail})"
        print(f"Day {day_idx + 1:2d} ({date_str}): {status}")
    
    print()
    print("=" * 70)
    print("总结")
    print("=" * 70)
    print(f"检查 Sheet 总数: {total_sheets}")
    print(f"通过: {pass_count}")
    print(f"失败: {fail_count}")
    
    if all_failures:
        print()
        print("=" * 70)
        print(f"失败详情 (共 {len(all_failures)} 项)")
        print("=" * 70)
        for f in all_failures[:20]:  # 只显示前20个
            print(f"\n{f['date']} - {f['name']}:")
            for issue in f.get("issues", []):
                print(f"  - {issue}")
            for vd in f.get("value_diffs", [])[:3]:
                print(f"    行{vd['row']}: {vd['col']} Dev={vd['dev_val']} Src={vd['src_val']} diff={vd['diff']:.6f}")
        
        if len(all_failures) > 20:
            print(f"\n... 还有 {len(all_failures) - 20} 项失败未显示")
    
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
