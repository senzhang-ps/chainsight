"""
工具函数模块

提供Module4中使用的通用工具函数。
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Any

from src.utils.date_helpers import (
    compute_planning_window as compute_shared_planning_window,
    is_offset_review_day,
)
from src.utils.normalization import (
    cast_identifier_columns,
    normalize_location_preserve_non_numeric,
)

from .constants import IDENTIFIER_COLS


def normalize_location(location_str: str) -> str:
    """标准化地点字符串。

    将纯数字地点左补零至4位，非数字地点保持原样。

    参数：
        location_str: 地点字符串（如 "386"/"0386"/"A888"）

    返回：
        str: 标准化后的地点字符串

    示例：
        >>> normalize_location("386")
        '0386'
        >>> normalize_location("A888")
        'A888'
    """
    return normalize_location_preserve_non_numeric(location_str)


def cast_identifiers_to_str(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """将标识符列转换为字符串类型并标准化地点。

    参数：
        df: 待处理的DataFrame
        cols: 要转换的列名列表，默认使用IDENTIFIER_COLS

    返回：
        pd.DataFrame: 处理后的DataFrame副本
    """
    cols = cols or IDENTIFIER_COLS
    return cast_identifier_columns(
        df,
        cols=cols,
        normalized_location_cols=("location",),
    )


def validate_merge_keys(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    keys: List[str]
) -> None:
    """校验两个DataFrame合并键的dtype一致性。

    参数：
        df1: 第一个DataFrame
        df2: 第二个DataFrame
        keys: 合并键列表

    异常：
        TypeError: 当合并键dtype不一致时
    """
    for key in keys:
        if key in df1.columns and key in df2.columns:
            if df1[key].dtype != df2[key].dtype:
                raise TypeError(
                    f"合并键 '{key}' 的dtype不匹配: "
                    f"{df1[key].dtype} vs {df2[key].dtype}"
                )


def compute_planning_window(
    simulation_date: pd.Timestamp,
    ptf: int,
    lsk: int
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """计算计划窗口的起止日期。

    根据计划冻结期(PTF)和批量周期键(LSK)计算计划窗口。

    参数：
        simulation_date: 当前仿真日期（审查日）
        ptf: 计划冻结期（天）
        lsk: 批量周期键（规划视窗天数）

    返回：
        Tuple[pd.Timestamp, pd.Timestamp]: (窗口起始日, 窗口结束日)

    示例：
        >>> compute_planning_window(pd.Timestamp('2024-01-01'), 2, 7)
        (Timestamp('2024-01-03'), Timestamp('2024-01-09'))
    """
    return compute_shared_planning_window(simulation_date, ptf, lsk)


def is_review_day(
    simulation_date: pd.Timestamp,
    simulation_start: pd.Timestamp,
    lsk: int,
    day: int
) -> bool:
    """判断是否为物料的审查日。

    基于LSK周期和首次审查偏移判断当前日期是否为审查日。

    参数：
        simulation_date: 当前仿真日期
        simulation_start: 仿真起始日期
        lsk: 审查间隔天数
        day: 首次审查相对起始的偏移天数

    返回：
        bool: 若是审查日返回True
    """
    return is_offset_review_day(simulation_date, simulation_start, lsk, day)


def dedup_issues(issues: List[dict]) -> List[dict]:
    """去重校验问题记录。

    参数：
        issues: 问题记录列表

    返回：
        List[dict]: 去重后的问题列表
    """
    if not issues:
        return issues

    df = pd.DataFrame(issues)
    df = df.drop_duplicates()
    return df.to_dict(orient='records')


def round_up_to_batch(
    quantity: float,
    min_batch: int,
    rounding_volume: int
) -> int:
    """按最小批量和舍入量向上取整。

    参数：
        quantity: 原始数量
        min_batch: 最小批量
        rounding_volume: 舍入量

    返回：
        int: 取整后的数量
    """
    base = max(quantity, min_batch)

    if base % rounding_volume == 0:
        return int(base)

    return int(np.ceil(base / rounding_volume) * rounding_volume)


def safe_float_conversion(value: Any) -> float:
    """安全地将值转换为float类型。

    处理numpy数值类型，确保JSON序列化兼容。

    参数：
        value: 待转换的值

    返回：
        float: 转换后的浮点数
    """
    if isinstance(value, (np.integer, np.int64)):
        return float(value)
    if isinstance(value, np.floating):
        return float(value)
    return float(value) if value else 0.0


def ensure_dataframe_columns(
    df: Optional[pd.DataFrame],
    columns: List[str]
) -> pd.DataFrame:
    """确保DataFrame包含指定列。

    如果DataFrame为空或缺少列，则创建/添加空列。

    参数：
        df: 待处理的DataFrame
        columns: 必需的列名列表

    返回：
        pd.DataFrame: 包含所有指定列的DataFrame
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)

    for col in columns:
        if col not in df.columns:
            df[col] = pd.Series(dtype='object')

    return df[columns]
