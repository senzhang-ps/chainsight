# normalize.py
# 标识符规范化函数集
#
# 将 material / location / sending / receiving 等标识符列
# 转换为统一的字符串格式，确保跨模块数据一致性。

import pandas as pd

from src.utils.normalization_common import (
    normalize_identifiers_vectorized,
    normalize_location_preserve_non_numeric,
    normalize_material_numeric_preserve_text,
)


def _normalize_material(material_str) -> str:
    """规范化物料字符串——移除数值物料的 .0 后缀。

    Args:
        material_str: 原始物料标识（int/float/str/None）

    Returns:
        规范化后的字符串
    """
    return normalize_material_numeric_preserve_text(material_str)


def _normalize_location(location_str) -> str:
    """规范化地点字符串：纯数字补齐到 4 位。

    Args:
        location_str: 原始地点标识

    Returns:
        规范化后的字符串
    """
    return normalize_location_preserve_non_numeric(location_str)


def _normalize_sending(sending_str) -> str:
    """规范化发货地字符串：纯数字补齐到 4 位。

    Args:
        sending_str: 原始发货地标识

    Returns:
        规范化后的字符串
    """
    return normalize_location_preserve_non_numeric(sending_str)


def _normalize_receiving(receiving_str) -> str:
    """规范化收货地字符串：纯数字补齐到 4 位。

    Args:
        receiving_str: 原始收货地标识

    Returns:
        规范化后的字符串
    """
    return normalize_location_preserve_non_numeric(receiving_str)


def _normalize_identifiers(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """将标识符列规范化为字符串并按规则格式化。

    使用向量化操作提升性能，与 Dev 版本行为一致：
    - material 列：移除数值物料的 .0 后缀
    - location/sending/receiving/sourcing 列：纯数字补齐到 4 位

    Args:
        df: 待规范化的 DataFrame

    Returns:
        规范化后的 DataFrame 副本
    """
    return normalize_identifiers_vectorized(
        df,
        material_cols=("material",),
        location_cols=("location", "sending", "receiving", "sourcing"),
    )
