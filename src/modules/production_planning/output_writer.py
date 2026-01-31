"""
输出管理模块

负责生成和写出Module4的输出文件。
支持DuckDB内存模式，可跳过磁盘IO直接写入内存表。
"""

import os
from typing import List, Dict, Any, Optional

import pandas as pd

from .constants import (
    PLAN_COLUMNS,
    EXCEED_COLUMNS,
    VALIDATION_COLUMNS,
    CHANGEOVER_LOG_COLUMNS,
)
from .utils import ensure_dataframe_columns

# 延迟导入内存存储模块（避免循环导入）
_memory_store_imported = False
_is_memory_mode_enabled = None
_write_module4_output = None


def _ensure_memory_store_imported():
    """延迟导入内存存储模块"""
    global _memory_store_imported, _is_memory_mode_enabled, _write_module4_output
    if not _memory_store_imported:
        try:
            from src.utils.memory_data_store import (
                is_memory_mode_enabled,
                write_module4_output,
            )
            _is_memory_mode_enabled = is_memory_mode_enabled
            _write_module4_output = write_module4_output
        except ImportError:
            _is_memory_mode_enabled = lambda: False
            _write_module4_output = lambda *args, **kwargs: False
        _memory_store_imported = True


def write_output(
    plan: pd.DataFrame,
    exc: pd.DataFrame,
    issues: List[Dict[str, Any]],
    changeover_log: pd.DataFrame,
    out_path: str,
    simulation_date: Optional[pd.Timestamp] = None,
    skip_file_output: bool = False
) -> str:
    """写出每日或汇总输出文件。

    Args:
        plan: 生产计划DataFrame
        exc: 超额记录DataFrame
        issues: 校验问题列表
        changeover_log: 换产日志DataFrame
        out_path: 基础输出路径
        simulation_date: 仿真日期（提供则写每日版本）
        skip_file_output: 是否跳过文件输出（使用DuckDB内存模式时为True）

    Returns:
        str: 实际写出的文件路径（或内存模式下的虚拟路径）
    """
    plan = ensure_dataframe_columns(plan, PLAN_COLUMNS)
    exc = ensure_dataframe_columns(exc, EXCEED_COLUMNS)
    changeover_log = ensure_dataframe_columns(
        changeover_log, CHANGEOVER_LOG_COLUMNS
    )

    issues_df = _prepare_issues_df(issues)

    final_path = _get_output_path(out_path, simulation_date)
    
    # 尝试写入DuckDB内存存储
    if simulation_date is not None:
        _ensure_memory_store_imported()
        if _is_memory_mode_enabled and _is_memory_mode_enabled():
            date_str = simulation_date.strftime('%Y%m%d')
            _write_module4_output(
                date_str=date_str,
                production_plan=plan,
                capacity_exceed=exc,
                validation=issues_df,
                changeover_log=changeover_log
            )
            # 如果跳过文件输出，直接返回
            if skip_file_output:
                return final_path

    # 写入Excel文件（默认行为或fallback）
    _write_excel_file(final_path, plan, exc, issues_df, changeover_log)

    return final_path


def _prepare_issues_df(issues: List[Dict[str, Any]]) -> pd.DataFrame:
    """准备校验问题DataFrame。

    Args:
        issues: 问题列表

    Returns:
        pd.DataFrame: 问题DataFrame
    """
    issues_df = pd.DataFrame(issues)
    return ensure_dataframe_columns(issues_df, VALIDATION_COLUMNS)


def _get_output_path(
    out_path: str,
    simulation_date: Optional[pd.Timestamp]
) -> str:
    """获取输出文件路径。

    Args:
        out_path: 基础路径
        simulation_date: 仿真日期

    Returns:
        str: 最终路径
    """
    if simulation_date is None:
        return out_path

    date_str = simulation_date.strftime('%Y%m%d')
    out_dir = os.path.dirname(out_path)
    base_name = os.path.splitext(os.path.basename(out_path))[0]

    return os.path.join(out_dir, f"{base_name}_{date_str}.xlsx")


def _write_excel_file(
    file_path: str,
    plan: pd.DataFrame,
    exc: pd.DataFrame,
    issues: pd.DataFrame,
    changeover_log: pd.DataFrame
) -> None:
    """写出Excel文件。

    Args:
        file_path: 文件路径
        plan: 生产计划
        exc: 超额记录
        issues: 校验问题
        changeover_log: 换产日志
    """
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        plan.to_excel(writer, sheet_name='ProductionPlan', index=False)
        exc.to_excel(writer, sheet_name='CapacityExceed', index=False)
        issues.to_excel(writer, sheet_name='Validation', index=False)
        changeover_log.to_excel(
            writer, sheet_name='ChangeoverLog', index=False
        )


def generate_consolidated_output(
    daily_output_files: List[str],
    output_path: str
) -> None:
    """合并多个每日输出生成汇总文件。

    Args:
        daily_output_files: 每日输出文件列表
        output_path: 汇总输出路径
    """
    if not daily_output_files:
        print("警告: 没有每日输出文件可合并")
        return

    all_data = _collect_all_daily_data(daily_output_files)

    consolidated = _consolidate_data(all_data)

    write_output(
        consolidated['plans'],
        consolidated['exceeds'],
        consolidated['issues'],
        consolidated['changeovers'],
        output_path
    )


def _collect_all_daily_data(
    daily_files: List[str]
) -> Dict[str, List[pd.DataFrame]]:
    """收集所有每日数据。

    Args:
        daily_files: 文件列表

    Returns:
        Dict[str, List[pd.DataFrame]]: 按类型分组的数据
    """
    all_data = {
        'plans': [],
        'exceeds': [],
        'issues': [],
        'changeovers': [],
    }

    for file_path in daily_files:
        if not os.path.exists(file_path):
            print(f"警告: 每日输出文件不存在: {file_path}")
            continue

        daily_data = _read_daily_file(file_path)
        _merge_daily_data(all_data, daily_data)

    return all_data


def _read_daily_file(
    file_path: str
) -> Dict[str, Optional[pd.DataFrame]]:
    """读取每日文件。

    Args:
        file_path: 文件路径

    Returns:
        Dict[str, Optional[pd.DataFrame]]: 读取的数据
    """
    try:
        xl = pd.ExcelFile(file_path)
        return {
            'plan': _read_sheet(xl, 'ProductionPlan'),
            'exceed': _read_sheet(xl, 'CapacityExceed'),
            'issues': _read_sheet(xl, 'Validation'),
            'changeover': _read_sheet(xl, 'ChangeoverLog'),
        }
    except Exception as e:
        print(f"读取每日文件出错 {file_path}: {e}")
        return {'plan': None, 'exceed': None, 'issues': None, 'changeover': None}


def _read_sheet(
    xl: pd.ExcelFile,
    sheet_name: str
) -> Optional[pd.DataFrame]:
    """读取工作表。

    Args:
        xl: Excel文件对象
        sheet_name: 工作表名

    Returns:
        Optional[pd.DataFrame]: 数据或None
    """
    if sheet_name not in xl.sheet_names:
        return None

    df = xl.parse(sheet_name)
    return df if not df.empty else None


def _merge_daily_data(
    all_data: Dict[str, List[pd.DataFrame]],
    daily_data: Dict[str, Optional[pd.DataFrame]]
) -> None:
    """合并每日数据到总数据。

    Args:
        all_data: 总数据（会被修改）
        daily_data: 每日数据
    """
    mapping = {
        'plan': 'plans',
        'exceed': 'exceeds',
        'issues': 'issues',
        'changeover': 'changeovers',
    }

    for daily_key, all_key in mapping.items():
        df = daily_data.get(daily_key)
        if df is not None:
            all_data[all_key].append(df)


def _consolidate_data(
    all_data: Dict[str, List[pd.DataFrame]]
) -> Dict[str, Any]:
    """合并所有数据。

    Args:
        all_data: 按类型分组的数据

    Returns:
        Dict[str, Any]: 合并后的数据
    """
    plans = (
        pd.concat(all_data['plans'], ignore_index=True)
        if all_data['plans'] else pd.DataFrame()
    )
    exceeds = (
        pd.concat(all_data['exceeds'], ignore_index=True)
        if all_data['exceeds'] else pd.DataFrame()
    )
    issues = (
        pd.concat(all_data['issues'], ignore_index=True).drop_duplicates()
        if all_data['issues'] else pd.DataFrame()
    )
    changeovers = (
        pd.concat(all_data['changeovers'], ignore_index=True)
        if all_data['changeovers'] else pd.DataFrame()
    )

    issues_list = (
        issues.to_dict('records') if not issues.empty else []
    )

    return {
        'plans': plans,
        'exceeds': exceeds,
        'issues': issues_list,
        'changeovers': changeovers,
    }
