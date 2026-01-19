#!/usr/bin/env python3
"""
比较code_vo和src的输出结果
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# 输出目录
CODE_VO_OUTPUT = r"c:\Users\25936\Desktop\Code\chainsight\test_outputs\BC_S5\run_20260119_110937"
SRC_OUTPUT = r"c:\Users\25936\Desktop\Code\chainsight\outputs\BC_S5_src\run_20260119_113028"

def compare_csv_files(file1, file2, file_name):
    """比较两个CSV文件"""
    try:
        df1 = pd.read_csv(file1, dtype=str)
        df2 = pd.read_csv(file2, dtype=str)
        
        # 检查列名
        cols1 = set(df1.columns)
        cols2 = set(df2.columns)
        if cols1 != cols2:
            only_in_1 = cols1 - cols2
            only_in_2 = cols2 - cols1
            print(f"  列差异: code_vo特有={only_in_1}, src特有={only_in_2}")
            return False
        
        # 检查行数
        if len(df1) != len(df2):
            print(f"  行数差异: code_vo={len(df1)}, src={len(df2)}")
            return False
        
        if len(df1) == 0:
            return True
        
        # 排序以便比较
        sort_cols = [c for c in ['material', 'location', 'date', 'deployment_uid', 'transit_uid'] if c in df1.columns]
        if sort_cols:
            df1 = df1.sort_values(sort_cols).reset_index(drop=True)
            df2 = df2.sort_values(sort_cols).reset_index(drop=True)
        
        # 逐列比较
        diff_cols = []
        for col in df1.columns:
            try:
                # 尝试数值比较
                v1 = pd.to_numeric(df1[col], errors='coerce')
                v2 = pd.to_numeric(df2[col], errors='coerce')
                
                if not v1.isna().all() and not v2.isna().all():
                    diff = np.abs(v1.fillna(0) - v2.fillna(0))
                    max_diff = diff.max()
                    if max_diff > 0.01:
                        diff_cols.append((col, max_diff))
                else:
                    # 字符串比较
                    if not df1[col].fillna('').equals(df2[col].fillna('')):
                        diff_count = (df1[col].fillna('') != df2[col].fillna('')).sum()
                        diff_cols.append((col, f"{diff_count} rows"))
            except Exception as e:
                if not df1[col].fillna('').equals(df2[col].fillna('')):
                    diff_cols.append((col, str(e)))
        
        if diff_cols:
            print(f"  列值差异: {diff_cols}")
            return False
        
        return True
    
    except Exception as e:
        print(f"  比较错误: {e}")
        return False


def main():
    code_vo_orch = Path(CODE_VO_OUTPUT) / "orchestrator"
    src_orch = Path(SRC_OUTPUT) / "orchestrator"
    
    if not code_vo_orch.exists():
        print(f"code_vo输出目录不存在: {code_vo_orch}")
        return
    
    if not src_orch.exists():
        print(f"src输出目录不存在: {src_orch}")
        return
    
    print("=" * 80)
    print("📊 Orchestrator输出比较")
    print("=" * 80)
    
    # 获取所有CSV文件
    files1 = set(f.name for f in code_vo_orch.glob("*.csv"))
    files2 = set(f.name for f in src_orch.glob("*.csv"))
    
    all_files = sorted(files1 | files2)
    
    matched = []
    mismatched = []
    
    for file_name in all_files:
        file1 = code_vo_orch / file_name
        file2 = src_orch / file_name
        
        if file_name not in files1:
            mismatched.append((file_name, "仅存在于src"))
            continue
        if file_name not in files2:
            mismatched.append((file_name, "仅存在于code_vo"))
            continue
        
        print(f"\n比较: {file_name}")
        if compare_csv_files(file1, file2, file_name):
            matched.append(file_name)
            print(f"  ✅ 匹配")
        else:
            mismatched.append((file_name, "内容不匹配"))
    
    print("\n" + "=" * 80)
    print("📊 比较汇总")
    print("=" * 80)
    print(f"✅ 匹配文件数: {len(matched)}")
    print(f"❌ 不匹配文件数: {len(mismatched)}")
    
    if mismatched:
        print("\n不匹配的文件:")
        for name, reason in mismatched:
            print(f"  ❌ {name}: {reason}")


if __name__ == "__main__":
    main()
