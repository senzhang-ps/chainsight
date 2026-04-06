# -*- coding: utf-8 -*-
"""
统一的标识符规范化模块

提供物料、地点等标识符的规范化处理功能，
确保整个系统中键值的一致性。

本模块为唯一的规范化实现，各模块通过 re-export 使用。

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
        非数字地点（如 A888）原样返回。

    示例:
        >>> normalize_location(7)
        '0007'
        >>> normalize_location('12')
        '0012'
        >>> normalize_location('A888')
        'A888'
    """
    if location_str is None or pd.isna(location_str):
        return ""
    location_str = str(location_str).strip()
    try:
        if location_str.isdigit():
            return str(int(location_str)).zfill(4)
        return location_str
    except (ValueError, TypeError):
        return str(location_str)


def normalize_material(material_str: Any) -> str:
    """规范化物料标识为字符串格式。

    数值型物料编码转为整数字符串以移除 .0 后缀。
    空值/None/NaN 返回空字符串。

    参数:
        material_str: 物料标识（可为任意类型）。

    返回:
        物料的字符串表示。
    """
    if material_str is None or material_str == '' or str(
        material_str).lower() in ['nan', 'none', '<na>']:
        return ""
    try:
        if isinstance(material_str, (int, float)) or str(
            material_str).replace('.', '').replace('-', '').isdigit():
            return str(int(float(material_str)))
        return str(material_str).strip()
    except (ValueError, TypeError):
        return str(material_str).strip()


def normalize_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """规范化DataFrame中的标识列以确保键一致性。

    使用向量化操作提升性能。

    规范化以下列（如存在）:
        - material/from_material/to_material: 转为字符串，移除.0后缀
        - location/dps_location/sending/receiving/sourcing: 纯数字零填充为4位
        - line/delegate_line/changeover_id: 仅转为字符串

    参数:
        df: 需要规范化的输入DataFrame。

    返回:
        标识列已规范化的DataFrame（副本）。
    """
    if df.empty:
        return df

    df = df.copy()

    # 向量化处理 material 类列
    material_cols = ['material', 'from_material', 'to_material']
    for col in material_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].replace(['nan', 'None', '<NA>', 'NaN', 'none'], '')
            # 移除数字的 .0 后缀
            df[col] = df[col].str.replace(r'\.0$', '', regex=True)

    # 向量化处理 location 类列
    location_cols = ['location', 'dps_location', 'sending', 'receiving', 'sourcing']
    for col in location_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(['nan', 'None', '<NA>', 'NaN', 'none'], '')
            # 识别纯数字的行并补齐4位
            is_numeric = df[col].str.match(r'^\d+$', na=False)
            df.loc[is_numeric, col] = df.loc[is_numeric, col].str.zfill(4)

    # 字符串转换列（仅类型转换，无格式化）
    str_only_cols = ['line', 'delegate_line', 'changeover_id']
    for col in str_only_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].replace(['nan', 'None', '<NA>', 'NaN', 'none'], '')

    return df
