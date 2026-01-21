#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
详细数据比较脚本 - 比较两个输出目录的所有文件内容
"""

import os
import sys
import pandas as pd
from pathlib import Path


def load_file(filepath):
    """加载CSV或Excel文件"""
    if not os.path.exists(filepath):
        return None
    
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == '.csv':
            return pd.read_csv(filepath)
        elif ext in ['.xlsx', '.xls']:
            return pd.read_excel(filepath)
        else:
            return None
    except Exception as e:
        print(f"    ⚠️ 加载文件失败 {filepath}: {e}")
        return None


def compare_dataframes(df1, df2, name):
    """详细比较两个DataFrame"""
    differences = []
    
    # 检查形状
    if df1.shape != df2.shape:
        differences.append(f"形状不同: {df1.shape} vs {df2.shape}")
        return differences
    
    # 检查列名
    if list(df1.columns) != list(df2.columns):
        differences.append(f"列名不同: {list(df1.columns)[:5]} vs {list(df2.columns)[:5]}")
        return differences
    
    # 逐列比较
    for col in df1.columns:
        try:
            # 尝试数值比较
            v1 = pd.to_numeric(df1[col], errors='coerce')
            v2 = pd.to_numeric(df2[col], errors='coerce')
            
            if v1.notna().any() and v2.notna().any():
                # 数值列
                diff = (v1 - v2).abs()
                max_diff = diff.max()
                if max_diff > 1e-6:
                    diff_count = (diff > 1e-6).sum()
                    differences.append(f"列 '{col}': {diff_count} 个值有差异 (最大差: {max_diff:.6f})")
            else:
                # 字符串列
                if not df1[col].equals(df2[col]):
                    diff_mask = df1[col] != df2[col]
                    diff_count = diff_mask.sum()
                    differences.append(f"列 '{col}': {diff_count} 个值不同")
        except Exception as e:
            differences.append(f"列 '{col}' 比较失败: {e}")
    
    return differences


def compare_excel_files(file1, file2, filename):
    """比较Excel文件的所有Sheet"""
    results = {}
    
    try:
        xl1 = pd.ExcelFile(file1)
        xl2 = pd.ExcelFile(file2)
        
        sheets1 = set(xl1.sheet_names)
        sheets2 = set(xl2.sheet_names)
        
        if sheets1 != sheets2:
            results['_sheets'] = f"Sheet不同: {sheets1} vs {sheets2}"
            return results
        
        for sheet in sheets1:
            df1 = xl1.parse(sheet)
            df2 = xl2.parse(sheet)
            
            diffs = compare_dataframes(df1, df2, f"{filename}[{sheet}]")
            if diffs:
                results[sheet] = diffs
            else:
                results[sheet] = "✅ 完全一致"
        
    except Exception as e:
        results['_error'] = str(e)
    
    return results


def compare_directories(base_dir, target_dir):
    """比较两个目录的所有输出"""
    base_path = Path(base_dir)
    target_path = Path(target_dir)
    
    print("=" * 70)
    print("详细数据比较")
    print("=" * 70)
    print(f"基准目录: {base_path}")
    print(f"目标目录: {target_path}")
    print("=" * 70)
    
    # 遍历所有子目录
    subdirs = ['module1', 'module3', 'module4', 'module5', 'module6', 'orchestrator', 'summary']
    
    total_files = 0
    matched_files = 0
    
    for subdir in subdirs:
        sub_base = base_path / subdir
        sub_target = target_path / subdir
        
        if not sub_base.exists() or not sub_target.exists():
            continue
        
        print(f"\n📁 {subdir.upper()}:")
        
        # 获取文件列表
        base_files = set(f.name for f in sub_base.iterdir() if f.is_file())
        target_files = set(f.name for f in sub_target.iterdir() if f.is_file())
        
        common_files = base_files & target_files
        
        for fname in sorted(common_files):
            if not (fname.endswith('.csv') or fname.endswith('.xlsx')):
                continue
            
            f1 = sub_base / fname
            f2 = sub_target / fname
            total_files += 1
            
            if fname.endswith('.csv'):
                df1 = load_file(f1)
                df2 = load_file(f2)
                
                if df1 is None or df2 is None:
                    print(f"  ⚠️ {fname}: 加载失败")
                    continue
                
                diffs = compare_dataframes(df1, df2, fname)
                if diffs:
                    print(f"  ❌ {fname}:")
                    for d in diffs[:3]:
                        print(f"     - {d}")
                else:
                    print(f"  ✅ {fname}: 完全一致")
                    matched_files += 1
            
            elif fname.endswith('.xlsx'):
                results = compare_excel_files(f1, f2, fname)
                
                has_diff = False
                for sheet, result in results.items():
                    if isinstance(result, list) and result:
                        has_diff = True
                        print(f"  ❌ {fname}[{sheet}]:")
                        for d in result[:2]:
                            print(f"     - {d}")
                    elif isinstance(result, str) and not result.startswith("✅"):
                        has_diff = True
                        print(f"  ❌ {fname}: {result}")
                
                if not has_diff:
                    print(f"  ✅ {fname}: 完全一致")
                    matched_files += 1
    
    print("\n" + "=" * 70)
    print(f"📊 总计: {matched_files}/{total_files} 文件完全一致")
    print("=" * 70)


def main():
    if len(sys.argv) < 3:
        print("用法: python compare_outputs_detail.py <基准目录> <目标目录>")
        print("示例: python compare_outputs_detail.py test_files/BC_S5/run_xxx outputs/BC_S5/run_yyy")
        sys.exit(1)
    
    base_dir = sys.argv[1]
    target_dir = sys.argv[2]
    
    compare_directories(base_dir, target_dir)


if __name__ == "__main__":
    main()
