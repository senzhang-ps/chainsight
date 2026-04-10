# -*- coding: utf-8 -*-
"""
标识符规范化模块

提供物料编码和位置编码的标准化函数，确保数据一致性。

注意：所有实现已统一到 src/utils/normalization.py，
本文件作为兼容导入层保留。
"""

# 统一实现来源
from src.utils.normalization import (
    normalize_location,
    normalize_material,
    normalize_identifiers,
    LOCATION_COLUMNS,
    ALL_IDENTIFIER_COLUMNS as IDENTIFIER_COLUMNS,
)

__all__ = [
    'normalize_location',
    'normalize_material',
    'normalize_identifiers',
    'LOCATION_COLUMNS',
    'IDENTIFIER_COLUMNS',
]
