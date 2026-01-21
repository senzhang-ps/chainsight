#!/usr/bin/env python3
"""
完整比较code_vo和src的所有输出
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# 默认输出目录
CODE_VO_OUTPUT = r"c:\Users\25936\Desktop\Code\chainsight\test_files\BC_S5\run_20260119_195544"
SRC_OUTPUT = r"c:\Users\25936\Desktop\Code\chainsight\outputs\BC_S5\run_20260119_205030"

# 需要忽略的列（运行时间戳等）
IGNORE_COLUMNS = {'timestamp', 'generation_time', 'run_timestamp'}

# 各Sheet/文件类型的排序键
SORT_KEYS = {
    'DeploymentPlan': ['date', 'material', 'sending', 'receiving', 
                      'demand_element', 'demand_qty', 'planned_qty', 'deployed_qty',
                      'planned_delivery_date', 'orig_location', 'leadtime'],
    'UnfulfilledLog': ['date', 'sending', 'receiving', 'demand_element', 
                      'demand_qty', 'unfulfilled_qty', 'reason'],
    'StockOnHandLog': ['date', 'material', 'location', 'beginning_soh', 'ending_soh'],
    'Validation': ['No'],
    'inventory_change_log': ['date', 'material', 'location'],
    # Orchestrator文件的排序键
    'order_log': ['date', 'order_number', 'material', 'location'],
    'production_log': ['date', 'prod_version', 'material', 'location'],
    'delivery_log': ['date', 'delivery_no', 'material'],
    'wip_cov': ['date', 'material', 'location'],
    # Summary文件的Sheet排序键
    'FullDeploymentPlan': ['date', 'material', 'sending', 'receiving', 
                           'demand_element', 'demand_qty', 'planned_qty', 'deployed_qty',
                           'planned_delivery_date', 'orig_location', 'leadtime'],
}

def _sort_df(df, sort_keys):
    """按可用的排序键排序DataFrame"""
    available_keys = [k for k in sort_keys if k in df.columns]
    if available_keys:
        return df.sort_values(available_keys, ignore_index=True)
    return df

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
            print(f"  ❌ {file_name}: 列差异")
            if only_in_1:
                print(f"     仅在基准版: {only_in_1}")
            if only_in_2:
                print(f"     仅在优化版: {only_in_2}")
            return False
        
        # 检查行数
        if len(df1) != len(df2):
            print(f"  ❌ {file_name}: 行数差异 (基准={len(df1)}, 优化={len(df2)})")
            return False
        
        # 排序后比较（确保顺序一致）
        # 优先根据文件名匹配排序键
        sort_keys = None
        base_name = os.path.splitext(file_name)[0]  # 去掉扩展名
        # 去掉日期后缀（如 _20251006）
        for suffix in ['_20251006', '_20251007', '_20251008', '_20251009', '_20251010']:
            base_name = base_name.replace(suffix, '')
        
        if base_name in SORT_KEYS:
            sort_keys = SORT_KEYS[base_name]
        else:
            # 回退：按文件名部分匹配
            for sheet_name, keys in SORT_KEYS.items():
                if sheet_name in base_name or sheet_name in file_name:
                    sort_keys = keys
                    break
        
        if sort_keys:
            df1 = _sort_df(df1, sort_keys)
            df2 = _sort_df(df2, sort_keys)
        
        # 逐列比较
        different_cols = []
        for col in df1.columns:
            if col in IGNORE_COLUMNS:
                continue
            if not df1[col].equals(df2[col]):
                different_cols.append(col)
        
        if different_cols:
            print(f"  ❌ {file_name}: 数值差异 (列: {different_cols[:5]}{'...' if len(different_cols) > 5 else ''})")
            return False
        
        print(f"  ✅ {file_name}: 完全一致")
        return True
        
    except Exception as e:
        print(f"  ⚠️ {file_name}: 比较失败 - {e}")
        return False


def compare_excel_files(file1, file2, file_name):
    """比较两个Excel文件"""
    try:
        xl1 = pd.ExcelFile(file1)
        xl2 = pd.ExcelFile(file2)
        
        sheets1 = set(xl1.sheet_names)
        sheets2 = set(xl2.sheet_names)
        
        if sheets1 != sheets2:
            print(f"  ❌ {file_name}: Sheet差异")
            return False
        
        all_match = True
        for sheet in sheets1:
            df1 = xl1.parse(sheet, dtype=str)
            df2 = xl2.parse(sheet, dtype=str)
            
            # 按排序键排序
            sort_keys = SORT_KEYS.get(sheet, [])
            df1 = _sort_df(df1, sort_keys)
            df2 = _sort_df(df2, sort_keys)
            
            if len(df1) != len(df2):
                print(f"  ❌ {file_name}[{sheet}]: 行数差异 (基准={len(df1)}, 优化={len(df2)})")
                all_match = False
                continue
            
            if list(df1.columns) != list(df2.columns):
                print(f"  ❌ {file_name}[{sheet}]: 列名差异")
                all_match = False
                continue
            
            # 逐列比较
            for col in df1.columns:
                if col in IGNORE_COLUMNS:
                    continue
                if not df1[col].equals(df2[col]):
                    print(f"  ❌ {file_name}[{sheet}]: 列 '{col}' 数值差异")
                    all_match = False
                    break
        
        if all_match:
            print(f"  ✅ {file_name}: 完全一致")
        return all_match
        
    except Exception as e:
        print(f"  ⚠️ {file_name}: 比较失败 - {e}")
        return False


def compare_directories(dir1, dir2, subdir_name):
    """比较两个目录下的所有文件"""
    path1 = Path(dir1) / subdir_name
    path2 = Path(dir2) / subdir_name
    
    if not path1.exists():
        print(f"  ⚠️ 基准目录不存在: {path1}")
        return 0, 0
    if not path2.exists():
        print(f"  ⚠️ 优化目录不存在: {path2}")
        return 0, 0
    
    files1 = set(f.name for f in path1.iterdir() if f.is_file())
    files2 = set(f.name for f in path2.iterdir() if f.is_file())
    
    common_files = files1 & files2
    only_in_1 = files1 - files2
    only_in_2 = files2 - files1
    
    if only_in_1:
        print(f"  ⚠️ 仅在基准版: {only_in_1}")
    if only_in_2:
        print(f"  ⚠️ 仅在优化版: {only_in_2}")
    
    matched = 0
    total = len(common_files)
    
    for fname in sorted(common_files):
        f1 = path1 / fname
        f2 = path2 / fname
        
        if fname.endswith('.csv'):
            if compare_csv_files(f1, f2, fname):
                matched += 1
        elif fname.endswith('.xlsx'):
            if compare_excel_files(f1, f2, fname):
                matched += 1
        else:
            # 其他文件跳过
            total -= 1
    
    return matched, total


def main():
    """主函数"""
    base_dir = sys.argv[1] if len(sys.argv) > 1 else CODE_VO_OUTPUT
    target_dir = sys.argv[2] if len(sys.argv) > 2 else SRC_OUTPUT
    
    print("=" * 70)
    print("ChainSight 输出比较")
    print("=" * 70)
    print(f"基准目录: {base_dir}")
    print(f"优化目录: {target_dir}")
    print("=" * 70)
    
    total_matched = 0
    total_files = 0
    
    # 比较各模块输出
    modules = ['module1', 'module3', 'module4', 'module5', 'module6', 'orchestrator', 'summary']
    
    for module in modules:
        print(f"\n📁 {module.upper()}:")
        matched, total = compare_directories(base_dir, target_dir, module)
        total_matched += matched
        total_files += total
        if total > 0:
            print(f"   小计: {matched}/{total} 文件匹配")
    
    print("\n" + "=" * 70)
    print(f"📊 总计: {total_matched}/{total_files} 文件匹配 ({100*total_matched/total_files:.1f}%)" if total_files > 0 else "无文件比较")
    print("=" * 70)


if __name__ == "__main__":
    main()
