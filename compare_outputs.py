#!/usr/bin/env python3
"""
比较 code_vo 和 src 重构代码的输出结果差异

用途：
1. 运行原始代码生成基准输出
2. 运行重构代码生成对比输出
3. 逐文件比较输出差异
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import shutil

# 配置
CONFIG_FILE = r"c:\Users\25936\Desktop\Code\chainsight\test_files\BC_S5.xlsx"
START_DATE = "2026-01-01"
END_DATE = "2026-01-03"  # 先测试3天

# 输出目录
CODE_VO_OUTPUT = r"c:\Users\25936\Desktop\Code\chainsight\test_outputs\code_vo"
SRC_OUTPUT = r"c:\Users\25936\Desktop\Code\chainsight\test_outputs\src"

def clean_and_create_dir(dir_path):
    """清理并创建目录"""
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)

def compare_csv_files(file1, file2, tolerance=0.001):
    """比较两个CSV文件的内容"""
    try:
        df1 = pd.read_csv(file1, dtype=str)
        df2 = pd.read_csv(file2, dtype=str)
        
        # 检查列名
        if set(df1.columns) != set(df2.columns):
            return {
                'match': False,
                'error': f'列名不同: {set(df1.columns)} vs {set(df2.columns)}'
            }
        
        # 检查行数
        if len(df1) != len(df2):
            return {
                'match': False,
                'error': f'行数不同: {len(df1)} vs {len(df2)}'
            }
        
        # 对齐列顺序
        df2 = df2[df1.columns]
        
        # 逐列比较
        diff_cols = []
        for col in df1.columns:
            try:
                # 尝试数值比较
                v1 = pd.to_numeric(df1[col], errors='coerce')
                v2 = pd.to_numeric(df2[col], errors='coerce')
                
                # 如果两列都是数值，比较数值差异
                if not v1.isna().all() and not v2.isna().all():
                    diff = np.abs(v1.fillna(0) - v2.fillna(0))
                    if diff.max() > tolerance:
                        diff_cols.append(col)
                else:
                    # 字符串比较
                    if not df1[col].fillna('').equals(df2[col].fillna('')):
                        diff_cols.append(col)
            except Exception as e:
                # 字符串比较
                if not df1[col].fillna('').equals(df2[col].fillna('')):
                    diff_cols.append(col)
        
        if diff_cols:
            return {
                'match': False,
                'error': f'以下列存在差异: {diff_cols}'
            }
        
        return {'match': True, 'error': None}
    
    except Exception as e:
        return {'match': False, 'error': str(e)}


def compare_output_directories(dir1, dir2):
    """比较两个输出目录的所有文件"""
    results = {}
    
    # 获取所有CSV文件
    files1 = set()
    files2 = set()
    
    for root, dirs, files in os.walk(dir1):
        for f in files:
            if f.endswith('.csv'):
                rel_path = os.path.relpath(os.path.join(root, f), dir1)
                files1.add(rel_path)
    
    for root, dirs, files in os.walk(dir2):
        for f in files:
            if f.endswith('.csv'):
                rel_path = os.path.relpath(os.path.join(root, f), dir2)
                files2.add(rel_path)
    
    all_files = files1 | files2
    
    for rel_path in sorted(all_files):
        file1 = os.path.join(dir1, rel_path)
        file2 = os.path.join(dir2, rel_path)
        
        if rel_path not in files1:
            results[rel_path] = {'match': False, 'error': '仅存在于src输出'}
        elif rel_path not in files2:
            results[rel_path] = {'match': False, 'error': '仅存在于code_vo输出'}
        else:
            results[rel_path] = compare_csv_files(file1, file2)
    
    return results


def print_comparison_report(results):
    """打印比较报告"""
    print("\n" + "=" * 80)
    print("📊 输出比较报告")
    print("=" * 80)
    
    matched = []
    mismatched = []
    
    for file_path, result in results.items():
        if result['match']:
            matched.append(file_path)
        else:
            mismatched.append((file_path, result['error']))
    
    print(f"\n✅ 匹配文件数: {len(matched)}")
    print(f"❌ 不匹配文件数: {len(mismatched)}")
    
    if mismatched:
        print("\n不匹配的文件:")
        for file_path, error in mismatched:
            print(f"  ❌ {file_path}")
            print(f"     错误: {error}")
    
    if matched:
        print("\n匹配的文件:")
        for file_path in matched[:10]:  # 只显示前10个
            print(f"  ✅ {file_path}")
        if len(matched) > 10:
            print(f"  ... 还有 {len(matched) - 10} 个文件")
    
    return len(mismatched) == 0


if __name__ == "__main__":
    print("=" * 80)
    print("ChainSight 代码比较测试")
    print("=" * 80)
    
    # 比较已有输出
    if os.path.exists(CODE_VO_OUTPUT) and os.path.exists(SRC_OUTPUT):
        results = compare_output_directories(
            os.path.join(CODE_VO_OUTPUT, "orchestrator"),
            os.path.join(SRC_OUTPUT, "orchestrator")
        )
        success = print_comparison_report(results)
        sys.exit(0 if success else 1)
    else:
        print("请先分别运行 code_vo 和 src 代码生成输出")
        print(f"code_vo 输出目录: {CODE_VO_OUTPUT}")
        print(f"src 输出目录: {SRC_OUTPUT}")
