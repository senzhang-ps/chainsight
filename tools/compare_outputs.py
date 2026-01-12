# -*- coding: utf-8 -*-
"""
输出目录对比工具

对比两个输出目录中的数据差异，生成Excel差异报告。
"""

import os
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional


def find_latest_run(output_dir: Path) -> Optional[Path]:
    """查找最新的运行目录"""
    run_dirs = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
    if not run_dirs:
        return None
    return max(run_dirs, key=lambda x: x.name)


def get_excel_files(run_dir: Path) -> Dict[str, Path]:
    """获取运行目录下所有Excel文件"""
    excel_files = {}
    for module_dir in run_dir.iterdir():
        if module_dir.is_dir():
            for f in module_dir.glob('*.xlsx'):
                # 使用相对路径作为key
                rel_path = f.relative_to(run_dir)
                excel_files[str(rel_path)] = f
    # 添加summary目录
    summary_dir = run_dir / 'summary'
    if summary_dir.exists():
        for f in summary_dir.glob('*.xlsx'):
            rel_path = f.relative_to(run_dir)
            excel_files[str(rel_path)] = f
    return excel_files


def compare_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, 
                       file_name: str, sheet_name: str) -> List[Dict]:
    """比较两个DataFrame，返回差异列表"""
    differences = []
    
    # 统一列名为字符串
    df1.columns = df1.columns.astype(str)
    df2.columns = df2.columns.astype(str)
    
    # 检查列差异
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    
    only_in_1 = cols1 - cols2
    only_in_2 = cols2 - cols1
    
    if only_in_1:
        differences.append({
            '文件名': file_name,
            'Sheet名': sheet_name,
            '差异类型': '列仅在BC_S5中存在',
            '差异详情': ', '.join(sorted(only_in_1)),
            'BC_S5值': str(list(only_in_1)),
            'BC_S50值': '-'
        })
    
    if only_in_2:
        differences.append({
            '文件名': file_name,
            'Sheet名': sheet_name,
            '差异类型': '列仅在BC_S50中存在',
            '差异详情': ', '.join(sorted(only_in_2)),
            'BC_S5值': '-',
            'BC_S50值': str(list(only_in_2))
        })
    
    # 检查行数差异
    if len(df1) != len(df2):
        differences.append({
            '文件名': file_name,
            'Sheet名': sheet_name,
            '差异类型': '行数不同',
            '差异详情': f'BC_S5有{len(df1)}行, BC_S50有{len(df2)}行',
            'BC_S5值': str(len(df1)),
            'BC_S50值': str(len(df2))
        })
    
    # 仅比较共同列
    common_cols = list(cols1 & cols2)
    if not common_cols:
        return differences
    
    # 对齐数据进行比较
    df1_common = df1[common_cols].copy()
    df2_common = df2[common_cols].copy()
    
    # 尝试找到关键列进行匹配
    key_candidates = ['material', 'location', 'sending', 'receiving', 'date', 'line']
    key_cols = [c for c in key_candidates if c in common_cols]
    
    if key_cols and len(df1) > 0 and len(df2) > 0:
        # 将所有key列转为字符串以便比较
        for col in key_cols:
            df1_common[col] = df1_common[col].astype(str).str.strip()
            df2_common[col] = df2_common[col].astype(str).str.strip()
        
        # 创建组合键
        df1_common['_key'] = df1_common[key_cols].apply(lambda x: '|'.join(x), axis=1)
        df2_common['_key'] = df2_common[key_cols].apply(lambda x: '|'.join(x), axis=1)
        
        keys1 = set(df1_common['_key'])
        keys2 = set(df2_common['_key'])
        
        only_in_df1 = keys1 - keys2
        only_in_df2 = keys2 - keys1
        
        if only_in_df1:
            sample = list(only_in_df1)[:5]
            differences.append({
                '文件名': file_name,
                'Sheet名': sheet_name,
                '差异类型': f'记录仅在BC_S5中存在 (共{len(only_in_df1)}条)',
                '差异详情': f'Key列: {key_cols}, 示例: {sample}',
                'BC_S5值': str(len(only_in_df1)),
                'BC_S50值': '0'
            })
        
        if only_in_df2:
            sample = list(only_in_df2)[:5]
            differences.append({
                '文件名': file_name,
                'Sheet名': sheet_name,
                '差异类型': f'记录仅在BC_S50中存在 (共{len(only_in_df2)}条)',
                '差异详情': f'Key列: {key_cols}, 示例: {sample}',
                'BC_S5值': '0',
                'BC_S50值': str(len(only_in_df2))
            })
        
        # 比较共同记录的值差异
        common_keys = keys1 & keys2
        if common_keys:
            df1_merged = df1_common[df1_common['_key'].isin(common_keys)].set_index('_key')
            df2_merged = df2_common[df2_common['_key'].isin(common_keys)].set_index('_key')
            
            # 对齐索引
            common_index = df1_merged.index.intersection(df2_merged.index)
            df1_aligned = df1_merged.loc[common_index]
            df2_aligned = df2_merged.loc[common_index]
            
            # 比较数值列
            value_cols = [c for c in common_cols if c not in key_cols]
            value_diffs = []
            
            for col in value_cols:
                if col in df1_aligned.columns and col in df2_aligned.columns:
                    try:
                        # 转换为可比较的类型
                        s1 = pd.to_numeric(df1_aligned[col], errors='coerce')
                        s2 = pd.to_numeric(df2_aligned[col], errors='coerce')
                        
                        # 计算差异
                        diff_mask = (s1.fillna(-999999) != s2.fillna(-999999))
                        diff_count = diff_mask.sum()
                        
                        if diff_count > 0:
                            value_diffs.append(f"{col}({diff_count}处)")
                    except:
                        pass
            
            if value_diffs:
                differences.append({
                    '文件名': file_name,
                    'Sheet名': sheet_name,
                    '差异类型': '数值差异',
                    '差异详情': ', '.join(value_diffs[:10]) + ('...' if len(value_diffs) > 10 else ''),
                    'BC_S5值': '-',
                    'BC_S50值': '-'
                })
    
    return differences


def compare_excel_files(file1: Path, file2: Path) -> List[Dict]:
    """比较两个Excel文件"""
    differences = []
    file_name = file1.name
    
    try:
        xl1 = pd.ExcelFile(file1)
        xl2 = pd.ExcelFile(file2)
        
        sheets1 = set(xl1.sheet_names)
        sheets2 = set(xl2.sheet_names)
        
        # 检查sheet差异
        only_in_1 = sheets1 - sheets2
        only_in_2 = sheets2 - sheets1
        
        if only_in_1:
            differences.append({
                '文件名': file_name,
                'Sheet名': '-',
                '差异类型': 'Sheet仅在BC_S5中存在',
                '差异详情': ', '.join(sorted(only_in_1)),
                'BC_S5值': str(list(only_in_1)),
                'BC_S50值': '-'
            })
        
        if only_in_2:
            differences.append({
                '文件名': file_name,
                'Sheet名': '-',
                '差异类型': 'Sheet仅在BC_S50中存在',
                '差异详情': ', '.join(sorted(only_in_2)),
                'BC_S5值': '-',
                'BC_S50值': str(list(only_in_2))
            })
        
        # 比较共同sheet
        common_sheets = sheets1 & sheets2
        for sheet in common_sheets:
            try:
                df1 = xl1.parse(sheet)
                df2 = xl2.parse(sheet)
                
                sheet_diffs = compare_dataframes(df1, df2, file_name, sheet)
                differences.extend(sheet_diffs)
            except Exception as e:
                differences.append({
                    '文件名': file_name,
                    'Sheet名': sheet,
                    '差异类型': '比较错误',
                    '差异详情': str(e),
                    'BC_S5值': '-',
                    'BC_S50值': '-'
                })
        
        xl1.close()
        xl2.close()
        
    except Exception as e:
        differences.append({
            '文件名': file_name,
            'Sheet名': '-',
            '差异类型': '文件读取错误',
            '差异详情': str(e),
            'BC_S5值': '-',
            'BC_S50值': '-'
        })
    
    return differences


def main():
    """主函数"""
    # 设置路径
    project_root = Path(__file__).parent.parent
    outputs_dir = project_root / 'outputs'
    
    dir1 = outputs_dir / 'BC_S5'
    dir2 = outputs_dir / 'BC_S50'
    
    if not dir1.exists() or not dir2.exists():
        print(f"❌ 输出目录不存在: {dir1} 或 {dir2}")
        sys.exit(1)
    
    # 查找最新运行目录
    run1 = find_latest_run(dir1)
    run2 = find_latest_run(dir2)
    
    if not run1 or not run2:
        print("❌ 未找到运行目录")
        sys.exit(1)
    
    print(f"📂 BC_S5 运行目录: {run1.name}")
    print(f"📂 BC_S50 运行目录: {run2.name}")
    
    # 获取所有Excel文件
    files1 = get_excel_files(run1)
    files2 = get_excel_files(run2)
    
    print(f"\n📊 BC_S5 Excel文件数: {len(files1)}")
    print(f"📊 BC_S50 Excel文件数: {len(files2)}")
    
    all_differences = []
    
    # 检查文件差异
    only_in_1 = set(files1.keys()) - set(files2.keys())
    only_in_2 = set(files2.keys()) - set(files1.keys())
    
    for f in only_in_1:
        all_differences.append({
            '文件名': f,
            'Sheet名': '-',
            '差异类型': '文件仅在BC_S5中存在',
            '差异详情': str(files1[f]),
            'BC_S5值': '存在',
            'BC_S50值': '不存在'
        })
    
    for f in only_in_2:
        all_differences.append({
            '文件名': f,
            'Sheet名': '-',
            '差异类型': '文件仅在BC_S50中存在',
            '差异详情': str(files2[f]),
            'BC_S5值': '不存在',
            'BC_S50值': '存在'
        })
    
    # 比较共同文件
    common_files = set(files1.keys()) & set(files2.keys())
    print(f"\n🔍 开始比较 {len(common_files)} 个共同文件...")
    
    for i, rel_path in enumerate(sorted(common_files), 1):
        print(f"  [{i}/{len(common_files)}] 比较: {rel_path}")
        diffs = compare_excel_files(files1[rel_path], files2[rel_path])
        all_differences.extend(diffs)
    
    # 生成报告
    if all_differences:
        df_report = pd.DataFrame(all_differences)
        
        # 按文件名和Sheet名排序
        df_report = df_report.sort_values(['文件名', 'Sheet名', '差异类型'])
        
        # 输出Excel
        output_file = outputs_dir / f'差异对比报告_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # 汇总sheet
            df_report.to_excel(writer, sheet_name='差异汇总', index=False)
            
            # 按差异类型分组
            for diff_type in df_report['差异类型'].unique():
                sheet_name = diff_type[:31]  # Excel sheet名最长31字符
                df_type = df_report[df_report['差异类型'] == diff_type]
                df_type.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"\n✅ 差异报告已生成: {output_file}")
        print(f"📊 共发现 {len(all_differences)} 处差异")
        
        # 打印差异类型统计
        print("\n📋 差异类型统计:")
        type_counts = df_report['差异类型'].value_counts()
        for t, c in type_counts.items():
            print(f"   - {t}: {c} 处")
    else:
        print("\n✅ 两个输出目录完全相同，无差异!")


if __name__ == '__main__':
    main()
