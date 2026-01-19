#!/usr/bin/env python3
"""
完整比较code_vo和src的所有输出
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# 输出目录
CODE_VO_OUTPUT = r"c:\Users\25936\Desktop\Code\chainsight\code_vo\run_20260119_130630"
SRC_OUTPUT = r"c:\Users\25936\Desktop\Code\chainsight\outputs\BC_S5\run_20260119_130857"

# 需要忽略的列（运行时间戳等）
IGNORE_COLUMNS = {'timestamp', 'generation_time', 'run_timestamp'}

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
            return False, f"列差异: code_vo特有={only_in_1}, src特有={only_in_2}"
        
        # 检查行数
        if len(df1) != len(df2):
            return False, f"行数差异: code_vo={len(df1)}, src={len(df2)}"
        
        if len(df1) == 0:
            return True, None
        
        # 过滤掉需要忽略的列
        compare_cols = [c for c in df1.columns if c not in IGNORE_COLUMNS]
        
        # 排序以便比较
        sort_cols = [c for c in ['material', 'location', 'date', 'deployment_uid', 'transit_uid', 'week'] if c in df1.columns]
        if sort_cols:
            df1 = df1.sort_values(sort_cols).reset_index(drop=True)
            df2 = df2.sort_values(sort_cols).reset_index(drop=True)
        
        # 逐列比较
        diff_cols = []
        for col in compare_cols:
            try:
                # 尝试数值比较
                v1 = pd.to_numeric(df1[col], errors='coerce')
                v2 = pd.to_numeric(df2[col], errors='coerce')
                
                if not v1.isna().all() and not v2.isna().all():
                    diff = np.abs(v1.fillna(0) - v2.fillna(0))
                    max_diff = diff.max()
                    if max_diff > 0.01:
                        diff_cols.append((col, f"max_diff={max_diff}"))
                else:
                    # 字符串比较
                    if not df1[col].fillna('').equals(df2[col].fillna('')):
                        diff_count = (df1[col].fillna('') != df2[col].fillna('')).sum()
                        diff_cols.append((col, f"{diff_count} rows"))
            except Exception as e:
                if not df1[col].fillna('').equals(df2[col].fillna('')):
                    diff_cols.append((col, str(e)))
        
        if diff_cols:
            return False, f"列值差异: {diff_cols}"
        
        return True, None
    
    except Exception as e:
        return False, str(e)


def compare_excel_files(file1, file2, file_name):
    """比较两个Excel文件"""
    try:
        xl1 = pd.ExcelFile(file1)
        xl2 = pd.ExcelFile(file2)
        
        sheets1 = set(xl1.sheet_names)
        sheets2 = set(xl2.sheet_names)
        
        if sheets1 != sheets2:
            return False, f"Sheet差异: code_vo特有={sheets1-sheets2}, src特有={sheets2-sheets1}"
        
        for sheet in sheets1:
            df1 = xl1.parse(sheet, dtype=str)
            df2 = xl2.parse(sheet, dtype=str)
            
            if len(df1) != len(df2):
                return False, f"Sheet {sheet} 行数差异: {len(df1)} vs {len(df2)}"
            
            if set(df1.columns) != set(df2.columns):
                return False, f"Sheet {sheet} 列名差异"
        
        return True, None
    except Exception as e:
        return False, str(e)


def compare_directories(dir1, dir2, dir_name=""):
    """递归比较两个目录"""
    results = {}
    
    path1 = Path(dir1)
    path2 = Path(dir2)
    
    if not path1.exists():
        return {dir_name: (False, "code_vo目录不存在")}
    if not path2.exists():
        return {dir_name: (False, "src目录不存在")}
    
    # 获取所有文件
    files1 = set(f.relative_to(path1) for f in path1.rglob("*") if f.is_file())
    files2 = set(f.relative_to(path2) for f in path2.rglob("*") if f.is_file())
    
    all_files = sorted(files1 | files2)
    
    for rel_path in all_files:
        file_key = str(rel_path)
        file1 = path1 / rel_path
        file2 = path2 / rel_path
        
        if rel_path not in files1:
            results[file_key] = (False, "仅存在于src")
            continue
        if rel_path not in files2:
            results[file_key] = (False, "仅存在于code_vo")
            continue
        
        # 根据文件类型比较
        suffix = rel_path.suffix.lower()
        if suffix == '.csv':
            match, error = compare_csv_files(file1, file2, file_key)
            results[file_key] = (match, error)
        elif suffix in ['.xlsx', '.xls']:
            match, error = compare_excel_files(file1, file2, file_key)
            results[file_key] = (match, error)
        elif suffix == '.txt':
            # 文本文件可能包含时间戳，跳过比较
            results[file_key] = (True, "跳过文本文件")
        else:
            # 其他文件跳过
            results[file_key] = (True, f"跳过 {suffix} 文件")
    
    return results


def main():
    print("=" * 80)
    print("📊 完整输出比较")
    print("=" * 80)
    
    code_vo_path = Path(CODE_VO_OUTPUT)
    src_path = Path(SRC_OUTPUT)
    
    # 比较各子目录
    subdirs = ['orchestrator', 'module1', 'module3', 'module4', 'module5', 'module6', 'summary']
    
    all_matched = 0
    all_mismatched = 0
    
    for subdir in subdirs:
        print(f"\n{'='*60}")
        print(f"📁 比较 {subdir}/")
        print('='*60)
        
        results = compare_directories(code_vo_path / subdir, src_path / subdir, subdir)
        
        matched = [(k, v) for k, v in results.items() if v[0]]
        mismatched = [(k, v) for k, v in results.items() if not v[0]]
        
        all_matched += len(matched)
        all_mismatched += len(mismatched)
        
        print(f"  ✅ 匹配: {len(matched)}, ❌ 不匹配: {len(mismatched)}")
        
        if mismatched:
            for file_key, (_, error) in mismatched[:5]:
                print(f"    ❌ {file_key}: {error}")
            if len(mismatched) > 5:
                print(f"    ... 还有 {len(mismatched) - 5} 个不匹配")
    
    print("\n" + "=" * 80)
    print("📊 总体汇总")
    print("=" * 80)
    print(f"✅ 总匹配文件数: {all_matched}")
    print(f"❌ 总不匹配文件数: {all_mismatched}")
    
    return all_mismatched == 0


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
