"""Module1 标识符规范化函数。

本模块提供物料、地点等标识符的规范化处理功能，
确保整个系统中键值的一致性。

主要函数：
- normalize_location: 规范化地点标识
- normalize_material: 规范化物料标识
- normalize_identifiers: 批量规范化DataFrame标识列

注意：所有实现已统一到 src/utils/normalization.py，
本文件作为兼容导入层保留。
"""

# 统一实现来源
from src.utils.normalization import (
    normalize_location,
    normalize_material,
    normalize_identifiers,
)

__all__ = [
    'normalize_location',
    'normalize_material',
    'normalize_identifiers',
]
