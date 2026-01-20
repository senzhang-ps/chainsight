# -*- coding: utf-8 -*-
"""
标识符规范化模块

提供物料编码和位置编码的标准化函数，确保数据一致性。
"""
import pandas as pd

from .constants import IDENTIFIER_COLUMNS, LOCATION_COLUMNS


def normalize_location(location_str) -> str:
    """
    规范化地点编码：补齐为4位数字字符串。

    作用：统一 location/sending/receiving/sourcing 字段格式，避免匹配失败。

    Args:
        location_str: 地点编码，可以是字符串、数字或None

    Returns:
        str: 规范化后的4位地点编码字符串
    """
    if location_str is None or pd.isna(location_str):
        return ""
    try:
        return str(int(location_str)).zfill(4)
    except (ValueError, TypeError):
        return str(location_str).zfill(4)


def normalize_material(material_str) -> str:
    """
    规范化物料编码为字符串。

    作用：统一 material 字段格式，与code_v0保持一致。
    注意：直接转换为字符串，不做额外处理，以确保与code_v0输出一致。

    Args:
        material_str: 物料编码，可以是字符串、数字或None

    Returns:
        str: 规范化后的物料编码字符串
    """
    if material_str is None or pd.isna(material_str):
        return ""
    return str(material_str)


def normalize_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    标识字段统一为字符串（并格式化地点字段）。

    作用：确保配置与日志中的标识符可一致匹配。
    
    使用向量化操作提升性能。

    Args:
        df: 需要规范化的DataFrame

    Returns:
        pd.DataFrame: 规范化后的DataFrame
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
    for col in LOCATION_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(['nan', 'None', '<NA>', 'NaN'], '')
            # 识别纯数字的行并补齐4位
            is_numeric = df[col].str.match(r'^\d+$', na=False)
            df.loc[is_numeric, col] = df.loc[is_numeric, col].str.zfill(4)
    
    # 其他标识符列
    other_cols = [c for c in IDENTIFIER_COLUMNS if c not in LOCATION_COLUMNS and c != 'material']
    for col in other_cols:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)

    return df
