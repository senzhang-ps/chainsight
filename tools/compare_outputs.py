#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ChainSight 输出数据对比工具

对比 Dev版本、重构本地版、数据库版 的输出数据一致性。

运行方法:
    python tools/compare_outputs.py
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import pandas as pd
import numpy as np

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent


def find_latest_run_dir(base_path: Path) -> Optional[Path]:
    """找到最新的运行目录"""
    if not base_path.exists():
        return None
    
    run_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('run_')]
    if not run_dirs:
        return None
    
    # 按名称排序（包含时间戳），取最新的
    run_dirs.sort(key=lambda x: x.name, reverse=True)
    return run_dirs[0]


def load_excel_files(directory: Path) -> Dict[str, pd.DataFrame]:
    """加载目录下所有Excel文件"""
    files = {}
    if not directory.exists():
        return files
    
    for f in directory.glob('*.xlsx'):
        try:
            df = pd.read_excel(f, engine='openpyxl')
            files[f.name] = df
        except Exception as e:
            print(f"  ⚠️ 加载失败 {f.name}: {e}")
    
    return files


def compare_dataframes(
    df1: pd.DataFrame, 
    df2: pd.DataFrame, 
    name: str,
    key_columns: Optional[List[str]] = None
) -> Dict:
    """比较两个DataFrame"""
    result = {
        'name': name,
        'match': False,
        'df1_rows': len(df1),
        'df2_rows': len(df2),
        'df1_cols': list(df1.columns),
        'df2_cols': list(df2.columns),
        'differences': []
    }
    
    # 检查行数
    if len(df1) != len(df2):
        result['differences'].append(f"行数不同: {len(df1)} vs {len(df2)}")
    
    # 检查列名
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    
    if cols1 != cols2:
        only_in_1 = cols1 - cols2
        only_in_2 = cols2 - cols1
        if only_in_1:
            result['differences'].append(f"仅在版本1存在的列: {only_in_1}")
        if only_in_2:
            result['differences'].append(f"仅在版本2存在的列: {only_in_2}")
    
    # 比较共同列的数据
    common_cols = list(cols1 & cols2)
    if not common_cols:
        result['differences'].append("没有共同的列")
        return result
    
    # 如果行数相同，比较数据内容
    if len(df1) == len(df2) and len(df1) > 0:
        # 尝试按键排序后比较
        if key_columns:
            valid_keys = [k for k in key_columns if k in common_cols]
            if valid_keys:
                try:
                    df1 = df1.sort_values(valid_keys).reset_index(drop=True)
                    df2 = df2.sort_values(valid_keys).reset_index(drop=True)
                except Exception:
                    pass
        
        # 比较数值列
        for col in common_cols:
            try:
                if df1[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                    # 数值列：比较近似相等
                    diff = np.abs(df1[col].fillna(0) - df2[col].fillna(0))
                    max_diff = diff.max()
                    if max_diff > 0.01:  # 允许小误差
                        result['differences'].append(
                            f"列 '{col}' 数值差异: 最大差异={max_diff:.4f}"
                        )
                else:
                    # 非数值列：直接比较
                    if not df1[col].astype(str).equals(df2[col].astype(str)):
                        mismatch_count = (df1[col].astype(str) != df2[col].astype(str)).sum()
                        result['differences'].append(
                            f"列 '{col}' 有 {mismatch_count} 行不匹配"
                        )
            except Exception as e:
                result['differences'].append(f"列 '{col}' 比较出错: {e}")
    
    # 判断是否完全匹配
    result['match'] = len(result['differences']) == 0
    
    return result


def compare_module_outputs(
    dir1: Path, 
    dir2: Path, 
    module_name: str,
    version1_name: str = "版本1",
    version2_name: str = "版本2"
) -> List[Dict]:
    """比较两个模块目录的输出"""
    results = []
    
    files1 = load_excel_files(dir1)
    files2 = load_excel_files(dir2)
    
    all_files = set(files1.keys()) | set(files2.keys())
    
    for fname in sorted(all_files):
        if fname in files1 and fname in files2:
            # 确定比较键
            key_cols = ['material', 'location', 'date', 'simulation_date']
            result = compare_dataframes(files1[fname], files2[fname], fname, key_cols)
            results.append(result)
        elif fname in files1:
            results.append({
                'name': fname,
                'match': False,
                'df1_rows': len(files1[fname]),
                'df2_rows': 0,
                'differences': [f"文件仅在{version1_name}存在"]
            })
        else:
            results.append({
                'name': fname,
                'match': False,
                'df1_rows': 0,
                'df2_rows': len(files2[fname]),
                'differences': [f"文件仅在{version2_name}存在"]
            })
    
    return results


def print_comparison_report(
    results: Dict[str, List[Dict]],
    version1_name: str,
    version2_name: str
):
    """打印对比报告"""
    print("\n" + "=" * 80)
    print(f"ChainSight 输出数据对比报告")
    print(f"对比: {version1_name} vs {version2_name}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    total_files = 0
    matched_files = 0
    
    for module, module_results in results.items():
        if not module_results:
            continue
        
        print(f"\n📁 {module}")
        print("-" * 60)
        
        for r in module_results:
            total_files += 1
            status = "✅" if r['match'] else "❌"
            if r['match']:
                matched_files += 1
            
            print(f"  {status} {r['name']}")
            print(f"      行数: {r.get('df1_rows', 'N/A')} vs {r.get('df2_rows', 'N/A')}")
            
            if r.get('differences'):
                for diff in r['differences'][:3]:  # 最多显示3条差异
                    print(f"      ⚠️ {diff}")
                if len(r.get('differences', [])) > 3:
                    print(f"      ... 还有 {len(r['differences']) - 3} 条差异")
    
    # 总结
    print("\n" + "=" * 80)
    print("📊 总结")
    print("=" * 80)
    match_rate = (matched_files / total_files * 100) if total_files > 0 else 0
    print(f"  总文件数: {total_files}")
    print(f"  匹配文件: {matched_files}")
    print(f"  不匹配:   {total_files - matched_files}")
    print(f"  一致率:   {match_rate:.1f}%")
    
    if match_rate == 100:
        print("\n🎉 所有输出完全一致！")
    elif match_rate >= 95:
        print("\n✅ 输出高度一致，存在少量差异")
    elif match_rate >= 80:
        print("\n⚠️ 输出基本一致，需要关注差异")
    else:
        print("\n❌ 输出存在较大差异，需要详细检查")
    
    print("=" * 80)
    
    return total_files, matched_files


def main():
    print("ChainSight 输出数据对比工具")
    print("=" * 60)
    
    # 定义输出目录
    # Dev版本 (5天仿真)
    dev_base = ROOT_DIR / "test_files" / "BC_S5"
    dev_run = find_latest_run_dir(dev_base)
    
    # 重构本地版 (3天仿真)
    refactored_base = ROOT_DIR / "outputs" / "BC_S5"
    refactored_run = find_latest_run_dir(refactored_base)
    
    # Dev原始目录 (3天仿真)
    dev_orig_base = ROOT_DIR / "ChainSight_Dev" / "BC_S5"
    dev_orig_run = find_latest_run_dir(dev_orig_base)
    
    print(f"\n发现的运行目录:")
    print(f"  Dev版(5天):      {dev_run}")
    print(f"  Dev版(3天):      {dev_orig_run}")
    print(f"  重构版(3天):     {refactored_run}")
    
    # 模块列表
    modules = ['module1', 'module3', 'module4', 'module5', 'module6', 'orchestrator', 'summary']
    
    # ==================== 对比1: Dev原始版 vs 重构版 (3天) ====================
    if dev_orig_run and refactored_run:
        print("\n\n" + "=" * 80)
        print("对比 1: ChainSight_Dev vs src重构版 (3天仿真)")
        print("=" * 80)
        
        results = {}
        for module in modules:
            dir1 = dev_orig_run / module
            dir2 = refactored_run / module
            
            if dir1.exists() or dir2.exists():
                results[module] = compare_module_outputs(
                    dir1, dir2, module,
                    "Dev版", "重构版"
                )
        
        print_comparison_report(results, "ChainSight_Dev (3天)", "src重构版 (3天)")
    
    # ==================== 详细数据统计 ====================
    print("\n\n" + "=" * 80)
    print("详细数据统计")
    print("=" * 80)
    
    for name, run_dir in [
        ("Dev版(5天)", dev_run),
        ("Dev版(3天)", dev_orig_run),
        ("重构版(3天)", refactored_run)
    ]:
        if not run_dir or not run_dir.exists():
            continue
        
        print(f"\n📁 {name}: {run_dir.name}")
        print("-" * 60)
        
        for module in modules:
            module_dir = run_dir / module
            if not module_dir.exists():
                continue
            
            files = load_excel_files(module_dir)
            if files:
                total_rows = sum(len(df) for df in files.values())
                print(f"  {module}:")
                for fname, df in sorted(files.items()):
                    print(f"    - {fname}: {len(df)} 行, {len(df.columns)} 列")


if __name__ == '__main__':
    main()
