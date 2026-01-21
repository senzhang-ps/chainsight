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
        
        # 逐列比较（忽略时间戳类列）
        ignore_cols = {'timestamp', 'generation_time'}
        different_cols = []
        for col in df1.columns:
            if col in ignore_cols:
                continue
            if not df1[col].equals(df2[col]):
                different_cols.append(col)
        
        if different_cols:
            print(f"  数值差异列: {different_cols[:5]}{'...' if len(different_cols) > 5 else ''}")
            return False
        
        return True
        
    except Exception as e:
        print(f"  比较失败: {e}")
        return False


def compare_orchestrator_outputs(base_dir, target_dir):
    """比较orchestrator输出"""
    base_path = Path(base_dir) / 'orchestrator'
    target_path = Path(target_dir) / 'orchestrator'
    
    print("\n" + "=" * 60)
    print("Orchestrator 输出比较")
    print("=" * 60)
    
    if not base_path.exists():
        print(f"基准目录不存在: {base_path}")
        return
    if not target_path.exists():
        print(f"目标目录不存在: {target_path}")
        return
    
    # 获取所有CSV文件
    base_files = set(f.name for f in base_path.glob('*.csv'))
    target_files = set(f.name for f in target_path.glob('*.csv'))
    
    common_files = base_files & target_files
    only_in_base = base_files - target_files
    only_in_target = target_files - base_files
    
    if only_in_base:
        print(f"仅在基准版: {only_in_base}")
    if only_in_target:
        print(f"仅在优化版: {only_in_target}")
    
    matched = 0
    total = len(common_files)
    
    for fname in sorted(common_files):
        f1 = base_path / fname
        f2 = target_path / fname
        
        result = compare_csv_files(f1, f2, fname)
        if result:
            print(f"✅ {fname}: 完全一致")
            matched += 1
        else:
            print(f"❌ {fname}: 存在差异")
    
    print(f"\n小计: {matched}/{total} 文件匹配")
    return matched, total


def compare_module_outputs(base_dir, target_dir, module_name):
    """比较模块输出"""
    base_path = Path(base_dir) / module_name
    target_path = Path(target_dir) / module_name
    
    print(f"\n" + "=" * 60)
    print(f"{module_name.upper()} 输出比较")
    print("=" * 60)
    
    if not base_path.exists():
        print(f"基准目录不存在: {base_path}")
        return 0, 0
    if not target_path.exists():
        print(f"目标目录不存在: {target_path}")
        return 0, 0
    
    # 获取所有文件
    base_files = set(f.name for f in base_path.iterdir() if f.is_file())
    target_files = set(f.name for f in target_path.iterdir() if f.is_file())
    
    common_files = base_files & target_files
    
    matched = 0
    total = 0
    
    for fname in sorted(common_files):
        if not (fname.endswith('.csv') or fname.endswith('.xlsx')):
            continue
        
        total += 1
        f1 = base_path / fname
        f2 = target_path / fname
        
        if fname.endswith('.csv'):
            result = compare_csv_files(f1, f2, fname)
        else:
            # Excel文件比较
            try:
                xl1 = pd.ExcelFile(f1)
                xl2 = pd.ExcelFile(f2)
                
                if set(xl1.sheet_names) != set(xl2.sheet_names):
                    print(f"❌ {fname}: Sheet名称差异")
                    continue
                
                all_match = True
                for sheet in xl1.sheet_names:
                    df1 = xl1.parse(sheet)
                    df2 = xl2.parse(sheet)
                    
                    if len(df1) != len(df2):
                        print(f"❌ {fname}[{sheet}]: 行数差异")
                        all_match = False
                        break
                    
                    if list(df1.columns) != list(df2.columns):
                        print(f"❌ {fname}[{sheet}]: 列名差异")
                        all_match = False
                        break
                
                result = all_match
            except Exception as e:
                print(f"❌ {fname}: 比较失败 - {e}")
                result = False
        
        if result:
            print(f"✅ {fname}: 完全一致")
            matched += 1
        else:
            print(f"❌ {fname}: 存在差异")
    
    print(f"\n小计: {matched}/{total} 文件匹配")
    return matched, total


def main():
    """主函数"""
    import sys
    
    base_dir = sys.argv[1] if len(sys.argv) > 1 else CODE_VO_OUTPUT
    target_dir = sys.argv[2] if len(sys.argv) > 2 else SRC_OUTPUT
    
    print("=" * 60)
    print("ChainSight Orchestrator 输出比较")
    print("=" * 60)
    print(f"基准目录: {base_dir}")
    print(f"目标目录: {target_dir}")
    
    total_matched = 0
    total_files = 0
    
    # 比较Orchestrator
    m, t = compare_orchestrator_outputs(base_dir, target_dir)
    total_matched += m
    total_files += t
    
    # 比较各模块
    for module in ['module1', 'module3', 'module4', 'module5', 'module6', 'summary']:
        m, t = compare_module_outputs(base_dir, target_dir, module)
        total_matched += m
        total_files += t
    
    print("\n" + "=" * 60)
    print(f"📊 总计: {total_matched}/{total_files} 文件匹配")
    print("=" * 60)


if __name__ == "__main__":
    main()
