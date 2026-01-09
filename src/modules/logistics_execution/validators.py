# -*- coding: utf-8 -*-
"""
数据验证模块

提供 Module6 的数据验证和报告生成功能，包括：
- 数据去重检查
- 验证报告生成
- 部署计划验证
- 卡车配置验证

Typical usage example:
    df = check_and_deduplicate(df, 'material', 'MaterialMD', log)
    generate_validation_report(validation_log, 'output.xlsx')
"""

import os
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd


def check_and_deduplicate(
    df: pd.DataFrame,
    key_column: str,
    sheet_name: str,
    validation_log: List[Dict]
) -> pd.DataFrame:
    """
    检查并移除 DataFrame 中的重复数据。
    
    基于指定的键列检查重复，保留第一条记录，
    并将重复信息记录到验证日志。
    
    Args:
        df: 要检查的 DataFrame
        key_column: 用于检查重复的列名
        sheet_name: 配置表名称（用于日志）
        validation_log: 验证日志列表
        
    Returns:
        去重后的 DataFrame
    """
    if df.empty:
        return df
    
    # 检查是否有重复
    has_duplicates = df[key_column].duplicated().any()
    if not has_duplicates:
        return df
    
    # 分析重复数据
    dup_info = _analyze_duplicates(df, key_column)
    
    # 只有真正有重复时才记录和打印
    if dup_info['unique_count'] > 0:
        _log_duplicate_warning(sheet_name, key_column, dup_info, validation_log)
        print(f"  ⚠️  发现{sheet_name}中有 {dup_info['unique_count']} 个重复的"
              f"{key_column}（共 {dup_info['total_count']} 条记录），将去重保留第一条")
    
    return df.drop_duplicates(subset=[key_column], keep='first')


def _analyze_duplicates(
    df: pd.DataFrame,
    key_column: str
) -> Dict[str, int]:
    """
    分析重复数据的统计信息。
    
    Args:
        df: DataFrame
        key_column: 键列名
        
    Returns:
        包含重复统计的字典
    """
    dup_mask = df.duplicated(subset=[key_column], keep=False)
    return {
        'total_count': dup_mask.sum(),
        'unique_count': df.loc[dup_mask, key_column].nunique()
    }


def _log_duplicate_warning(
    sheet_name: str,
    key_column: str,
    dup_info: Dict[str, int],
    validation_log: List[Dict]
) -> None:
    """
    记录重复数据警告到验证日志。
    
    Args:
        sheet_name: 表名
        key_column: 键列名
        dup_info: 重复统计信息
        validation_log: 验证日志列表
    """
    removed_count = dup_info['total_count'] - dup_info['unique_count']
    
    validation_log.append({
        'sheet': sheet_name,
        'row': '',
        'issue': f'Found {dup_info["unique_count"]} duplicate {key_column} values '
                 f'in {sheet_name} ({dup_info["total_count"]} total duplicates). '
                 f'Keeping first occurrence of each {key_column}.',
        'severity': 'WARNING',
        'impact': f'Data Deduplication - {removed_count} duplicate records removed',
        f'duplicate_{key_column}': dup_info['unique_count']
    })


def generate_validation_report(
    validation_log: List[Dict],
    output_file: str
) -> None:
    """
    生成验证报告文件。
    
    Args:
        validation_log: 验证日志列表
        output_file: 输出文件路径
    """
    validation_file = _get_validation_file_path(output_file)
    errors, warnings = _categorize_issues(validation_log)
    
    with open(validation_file, 'w', encoding='utf-8') as f:
        _write_report_header(f, validation_log, errors, warnings)
        _write_errors_section(f, errors)
        _write_warnings_section(f, warnings)
        _write_success_section(f, errors, warnings)
        _write_recommendations(f, errors, warnings)


def _get_validation_file_path(output_file: str) -> str:
    """
    生成验证报告文件路径。
    
    Args:
        output_file: 原始输出文件路径
        
    Returns:
        验证报告文件路径
    """
    output_dir = os.path.dirname(output_file)
    base_name = os.path.splitext(os.path.basename(output_file))[0]
    return os.path.join(output_dir, f"{base_name}_validation.txt")


def _categorize_issues(
    validation_log: List[Dict]
) -> tuple:
    """
    按严重程度分类验证问题。
    
    Args:
        validation_log: 验证日志列表
        
    Returns:
        (errors, warnings) 元组
    """
    errors = [log for log in validation_log if log.get('severity') == 'ERROR']
    warnings = [log for log in validation_log if log.get('severity') != 'ERROR']
    return errors, warnings


def _write_report_header(
    f,
    validation_log: List[Dict],
    errors: List[Dict],
    warnings: List[Dict]
) -> None:
    """
    写入报告头部信息。
    
    Args:
        f: 文件对象
        validation_log: 验证日志
        errors: 错误列表
        warnings: 警告列表
    """
    f.write("=" * 80 + "\n")
    f.write("MODULE6 VALIDATION REPORT\n")
    f.write("=" * 80 + "\n")
    f.write(f"Generated Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Total Issues: {len(validation_log)}\n")
    f.write(f"Errors: {len(errors)}\n")
    f.write(f"Warnings: {len(warnings)}\n")
    f.write("\n")


def _write_errors_section(f, errors: List[Dict]) -> None:
    """
    写入错误部分。
    
    Args:
        f: 文件对象
        errors: 错误列表
    """
    if not errors:
        return
    
    f.write("❌ CRITICAL ERRORS FOUND\n")
    f.write("Following issues may cause data loss or incorrect processing:\n")
    f.write("-" * 60 + "\n")
    
    for i, error in enumerate(errors, 1):
        _write_error_detail(f, i, error)


def _write_error_detail(f, index: int, error: Dict) -> None:
    """
    写入单条错误详情。
    
    Args:
        f: 文件对象
        index: 错误序号
        error: 错误信息字典
    """
    f.write(f"{index}. {error.get('issue', 'Unknown error')}\n")
    
    detail_keys = ['impact', 'missing_element', 'affected_records', 'route_breakdown']
    for key in detail_keys:
        if key in error:
            label = key.replace('_', ' ').title()
            f.write(f"   {label}: {error[key]}\n")
    
    f.write("\n")


def _write_warnings_section(f, warnings: List[Dict]) -> None:
    """
    写入警告部分。
    
    Args:
        f: 文件对象
        warnings: 警告列表
    """
    if not warnings:
        return
    
    f.write("⚠️  WARNINGS\n")
    f.write("Following issues should be reviewed but may not block processing:\n")
    f.write("-" * 60 + "\n")
    
    for i, warning in enumerate(warnings, 1):
        f.write(f"{i}. {warning.get('issue', 'Unknown warning')}\n")
        f.write(f"   Sheet: {warning.get('sheet', 'Unknown')}\n")
        f.write("\n")


def _write_success_section(
    f,
    errors: List[Dict],
    warnings: List[Dict]
) -> None:
    """
    写入成功部分（无错误和警告时）。
    
    Args:
        f: 文件对象
        errors: 错误列表
        warnings: 警告列表
    """
    if errors or warnings:
        return
    
    f.write("✅ ALL VALIDATIONS PASSED\n")
    f.write("No configuration issues detected.\n")
    f.write("All demand_element types are properly configured.\n")
    f.write("All material metadata is available.\n")
    f.write("All truck configurations are valid.\n")


def _write_recommendations(
    f,
    errors: List[Dict],
    warnings: List[Dict]
) -> None:
    """
    写入建议部分。
    
    Args:
        f: 文件对象
        errors: 错误列表
        warnings: 警告列表
    """
    f.write("\n")
    f.write("=" * 80 + "\n")
    f.write("RECOMMENDATIONS:\n")
    f.write("=" * 80 + "\n")
    
    if errors:
        _write_error_recommendations(f)
    elif warnings:
        _write_warning_recommendations(f)
    else:
        _write_success_recommendations(f)
    
    f.write("\n")
    f.write("For detailed information, check the ValidationLog sheet.\n")
    f.write("=" * 80 + "\n")


def _write_error_recommendations(f) -> None:
    """写入错误情况的建议。"""
    f.write("1. Fix all ERROR-level issues before proceeding\n")
    f.write("2. Add missing demand_element configurations to Global_DemandPriority\n")
    f.write("3. Ensure all material metadata is defined in M6_MaterialMD\n")


def _write_warning_recommendations(f) -> None:
    """写入警告情况的建议。"""
    f.write("1. Review WARNING-level issues for optimization opportunities\n")
    f.write("2. Check truck capacity and configuration alignment\n")


def _write_success_recommendations(f) -> None:
    """写入无问题情况的建议。"""
    f.write("1. Configuration is optimal for current deployment plans\n")
    f.write("2. Monitor validation reports in future runs\n")
    f.write("3. Consider adding bypass rules if delivery performance is suboptimal\n")


def validate_deployment_plan(
    dp: pd.DataFrame,
    validation_log: List[Dict]
) -> pd.DataFrame:
    """
    验证部署计划并补充缺失列。
    
    Args:
        dp: 部署计划 DataFrame
        validation_log: 验证日志列表
        
    Returns:
        验证并补充后的 DataFrame
    """
    if dp.empty:
        return dp
    
    required_cols = {
        'material': 'UNKNOWN',
        'sending': 'UNKNOWN_SENDING',
        'receiving': 'UNKNOWN_RECEIVING',
        'demand_element': 'DEFAULT',
        'planned_deployment_date': pd.Timestamp.now(),
        'deployed_qty': 0
    }
    
    missing_cols = [col for col in required_cols if col not in dp.columns]
    if missing_cols:
        print(f"  ⚠️  DeploymentPlan缺失列: {missing_cols}")
        for col in missing_cols:
            dp[col] = required_cols[col]
    
    return dp


def validate_truck_config(
    truck_con: pd.DataFrame,
    validation_log: List[Dict]
) -> pd.DataFrame:
    """
    验证卡车配置并补充缺失列。
    
    Args:
        truck_con: 卡车配置 DataFrame
        validation_log: 验证日志列表
        
    Returns:
        验证并补充后的 DataFrame
    """
    if truck_con.empty:
        return truck_con
    
    required_cols = ['sending', 'receiving', 'truck_type', 'WFR', 'VFR']
    missing_cols = [col for col in required_cols if col not in truck_con.columns]
    
    if not missing_cols:
        return truck_con
    
    print(f"  ⚠️  TruckReleaseCon缺失列: {missing_cols}")
    
    for col in missing_cols:
        if col in ['sending', 'receiving', 'truck_type']:
            truck_con[col] = 'UNKNOWN'
        elif col in ['WFR', 'VFR']:
            truck_con[col] = 0.0
    
    return truck_con


def validate_priority_mapping(
    dp: pd.DataFrame,
    prio_map: Dict[str, int],
    validation_log: List[Dict]
) -> pd.DataFrame:
    """
    验证优先级映射并过滤缺失配置的记录。
    
    Args:
        dp: 部署计划 DataFrame
        prio_map: 优先级映射字典
        validation_log: 验证日志列表
        
    Returns:
        过滤后的 DataFrame
    """
    missing_prio = dp[~dp['demand_element'].isin(prio_map.keys())]
    
    if missing_prio.empty:
        return dp
    
    missing_elements = missing_prio['demand_element'].unique()
    _log_missing_priority(missing_prio, missing_elements, validation_log)
    
    print(f"  ⚠️  发现 {len(missing_elements)} 个缺失的demand_element配置，"
          f"将过滤 {len(missing_prio)} 条记录")
    
    return dp[dp['demand_element'].isin(prio_map.keys())]


def _log_missing_priority(
    missing_prio: pd.DataFrame,
    missing_elements: Any,
    validation_log: List[Dict]
) -> None:
    """
    记录缺失的优先级配置。
    
    Args:
        missing_prio: 缺失优先级的记录
        missing_elements: 缺失的元素列表
        validation_log: 验证日志列表
    """
    for val in missing_elements:
        missing_records = missing_prio[missing_prio['demand_element'] == val]
        route_info = _get_route_info(missing_records)
        
        validation_log.append({
            'sheet': 'Global_DemandPriority',
            'row': '',
            'issue': f'Missing priority configuration for demand_element "{val}" '
                     f'(affects {len(missing_records)} records: {route_info}). '
                     f'Records will be filtered out and not processed.',
            'severity': 'ERROR',
            'impact': f'Data Loss - {len(missing_records)} deployment plans excluded',
            'missing_element': val,
            'affected_records': len(missing_records),
            'route_breakdown': route_info
        })


def _get_route_info(records: pd.DataFrame) -> str:
    """
    获取路线类型统计信息。
    
    Args:
        records: 记录 DataFrame
        
    Returns:
        路线统计字符串
    """
    if 'sending' not in records.columns or 'receiving' not in records.columns:
        return 'N/A (columns missing)'
    
    route_stats = records.apply(
        lambda row: 'self_loop' if row['sending'] == row['receiving'] else 'cross_node',
        axis=1
    ).value_counts()
    
    return ', '.join([f"{k}: {v}" for k, v in route_stats.items()])


def validate_threshold_config(
    truck_con: pd.DataFrame,
    validation_log: List[Dict]
) -> None:
    """
    验证阈值配置（WFR/VFR > 1.0 的告警）。
    
    Args:
        truck_con: 卡车配置 DataFrame
        validation_log: 验证日志列表
    """
    if truck_con.empty:
        return
    
    if 'WFR' not in truck_con.columns or 'VFR' not in truck_con.columns:
        return
    
    bad_threshold = truck_con[
        (truck_con['WFR'] > 1.0) | (truck_con['VFR'] > 1.0)
    ]
    
    for row in bad_threshold.itertuples(index=False):
        validation_log.append({
            'sheet': 'M6_TruckReleaseCon',
            'row': '',
            'issue': f"Threshold > 1.0 for route {row.sending}->{row.receiving} "
                     f"type {row.truck_type} (WFR={row.WFR}, VFR={row.VFR}). "
                     f"Will never trigger by threshold; bypass only.",
            'severity': 'WARNING',
            'impact': 'Configuration Issue - Route can only be triggered by bypass rules',
            'route': f"{row.sending}->{row.receiving}",
            'truck_type': row.truck_type,
            'wfr_threshold': row.WFR,
            'vfr_threshold': row.VFR
        })


def validate_truck_specs(
    truck_con: pd.DataFrame,
    spec_map: Dict[str, Any],
    validation_log: List[Dict]
) -> None:
    """
    验证车型规格是否完整。
    
    Args:
        truck_con: 卡车配置 DataFrame
        spec_map: 车型规格映射
        validation_log: 验证日志列表
    """
    if truck_con.empty:
        return
    
    missing_specs = set(truck_con['truck_type'].unique()) - set(spec_map.keys())
    
    for val in missing_specs:
        affected_routes = truck_con[truck_con['truck_type'] == val]
        
        validation_log.append({
            'sheet': 'M6_TruckTypeSpecs',
            'row': '',
            'issue': f'Missing truck type specification for "{val}" '
                     f'(affects {len(affected_routes)} route configurations). '
                     f'Routes using this truck type will be skipped.',
            'severity': 'ERROR',
            'impact': f'Data Loss - {len(affected_routes)} route configurations unavailable',
            'missing_truck_type': val,
            'affected_routes': len(affected_routes)
        })
