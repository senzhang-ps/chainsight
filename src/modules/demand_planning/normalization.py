"""Module 1 标识符规范化函数。

本模块提供物料、地点等标识符的规范化处理功能，
确保整个系统中键值的一致性。

主要函数：
- normalize_location: 规范化地点标识
- normalize_material: 规范化物料标识
- normalize_identifiers: 批量规范化DataFrame标识列
"""

from typing import Any

import pandas as pd


def normalize_location(location_str: Any) -> str:
    """规范化地点标识为4位零填充字符串。

    参数:
        location_str: 地点标识（可为int、str或None）。

    返回:
        4位零填充的字符串。None/NaN返回空字符串。

    示例:
        >>> normalize_location(7)
        '0007'
        >>> normalize_location('12')
        '0012'
    """
    if location_str is None or pd.isna(location_str):
        return ""
    try:
        return str(int(location_str)).zfill(4)
    except (ValueError, TypeError):
        return str(location_str).zfill(4)


def normalize_material(material_str: Any) -> str:
    """规范化物料标识为字符串格式。

    作用：统一 material 字段格式，与code_v0保持一致。
    注意：直接转换为字符串，不做额外处理，以确保与code_v0输出一致。

    参数:
        material_str: 物料标识（可为任意类型）。

    返回:
        物料的字符串表示。None/NaN返回空字符串。
    """
    if material_str is None or pd.isna(material_str):
        return ""
    return str(material_str)


def normalize_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """规范化DataFrame中的标识列以确保键一致性。

    使用向量化操作提升性能。
    
    规范化以下列（如存在）:
        - material: 转为字符串，移除.0后缀，NaN填充空字符串
        - location/dps_location: 纯数字零填充为4位
        - sending/receiving/sourcing: 纯数字零填充为4位

    参数:
        df: 需要规范化的输入DataFrame。

    返回:
        标识列已规范化的DataFrame。
    """
    if df.empty:
        return df

    df = df.copy()
    
    # 向量化处理 material 列
    if 'material' in df.columns:
        df['material'] = df['material'].astype(str)
        df['material'] = df['material'].replace(['nan', 'None', '<NA>', 'NaN'], '')
        # 移除数字的 .0 后缀
        df['material'] = df['material'].str.replace(r'\.0$', '', regex=True)
    
    # 向量化处理 location 类列
    location_cols = ['location', 'dps_location', 'sending', 'receiving', 'sourcing']
    for col in location_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(['nan', 'None', '<NA>', 'NaN'], '')
            # 识别纯数字的行并补齐4位
            is_numeric = df[col].str.match(r'^\d+$', na=False)
            df.loc[is_numeric, col] = df.loc[is_numeric, col].str.zfill(4)

    return df
