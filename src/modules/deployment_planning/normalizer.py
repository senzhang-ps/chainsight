# -*- coding: utf-8 -*-
"""
标识符规范化模块

提供物料编码和位置编码的标准化函数，确保数据一致性。
"""
import pandas as pd

from src.utils.normalization_common import (
    normalize_identifiers_vectorized,
    normalize_location_zero_fill_any,
    normalize_material_basic,
)

from .constants import IDENTIFIER_COLUMNS, LOCATION_COLUMNS


def normalize_location(location_str) -> str:
    """
    规范化地点编码：补齐为4位数字字符串。

    作用：统一 location/sending/receiving/sourcing 字段格式，避免匹配失败。

    参数：
        location_str: 地点编码，可以是字符串、数字或None

    返回：
        str: 规范化后的4位地点编码字符串
    """
    return normalize_location_zero_fill_any(location_str)


def normalize_material(material_str) -> str:
    """
    规范化物料编码为字符串。

    作用：统一 material 字段格式，与code_v0保持一致。
    注意：直接转换为字符串，不做额外处理，以确保与code_v0输出一致。

    参数：
        material_str: 物料编码，可以是字符串、数字或None

    返回：
        str: 规范化后的物料编码字符串
    """
    return normalize_material_basic(material_str)


def normalize_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    标识字段统一为字符串（并格式化地点字段）。

    作用：确保配置与日志中的标识符可一致匹配。
    
    使用向量化操作提升性能。

    参数：
        df: 需要规范化的DataFrame

    返回：
        pd.DataFrame: 规范化后的DataFrame
    """
    other_cols = [
        c for c in IDENTIFIER_COLUMNS if c not in LOCATION_COLUMNS and c != "material"
    ]
    return normalize_identifiers_vectorized(
        df,
        material_cols=("material",),
        location_cols=tuple(LOCATION_COLUMNS),
        other_identifier_cols=tuple(other_cols),
    )
