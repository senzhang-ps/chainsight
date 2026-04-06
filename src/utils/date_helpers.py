# -*- coding: utf-8 -*-
"""
统一的日期辅助函数模块

提供日期格式转换、日期列处理等通用功能。
与 time_manager.py 互补：time_manager 管理仿真时间状态，
本模块提供纯函数工具。
"""

from typing import List, Optional, Union

import pandas as pd


def convert_date_column(df: pd.DataFrame, column: str) -> None:
    """将 DataFrame 中的指定列转换为 datetime 类型（原地修改）。

    参数：
        df: 待处理的 DataFrame
        column: 日期列名
    """
    if column in df.columns:
        df[column] = pd.to_datetime(df[column], errors='coerce')


def convert_date_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """将 DataFrame 中的多个列转换为 datetime 类型。

    参数：
        df: 待处理的 DataFrame
        columns: 日期列名列表

    返回：
        处理后的 DataFrame
    """
    for col in columns:
        convert_date_column(df, col)
    return df


def ensure_date_format(df: pd.DataFrame, column: str = 'planned_deployment_date') -> None:
    """确保 DataFrame 中的日期列为 datetime 类型（原地修改）。

    参数：
        df: 待处理的 DataFrame
        column: 日期列名
    """
    if column in df.columns and not df.empty:
        if not pd.api.types.is_datetime64_any_dtype(df[column]):
            df[column] = pd.to_datetime(df[column], errors='coerce')


def format_date(date: Union[str, pd.Timestamp], fmt: str = '%Y-%m-%d') -> str:
    """将日期格式化为字符串。

    参数：
        date: 日期对象或字符串
        fmt: 输出格式

    返回：
        格式化后的日期字符串
    """
    if isinstance(date, str):
        date = pd.to_datetime(date)
    return date.strftime(fmt)


def format_date_for_file(date: Union[str, pd.Timestamp]) -> str:
    """将日期格式化为文件名友好的格式 (YYYYMMDD)。

    参数：
        date: 日期对象或字符串

    返回：
        YYYYMMDD 格式的字符串
    """
    return format_date(date, '%Y%m%d')


def parse_date(date_str: str) -> pd.Timestamp:
    """解析日期字符串为 Timestamp。

    参数：
        date_str: 日期字符串

    返回：
        pd.Timestamp
    """
    return pd.to_datetime(date_str)
