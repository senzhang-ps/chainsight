# -*- coding: utf-8 -*-
"""标识符规范化模块 — 薄代理层。

所有实现已统一至 src.utils.normalization，本模块仅 re-export 以保持向后兼容。
"""

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
