"""normalize.py — 薄代理层

所有实现已统一至 src.utils.normalization，
本模块仅 re-export 私有名称以保持向后兼容。
"""

from src.utils.normalization import (
    normalize_location as _normalize_location,
    normalize_material as _normalize_material,
    normalize_identifiers as _normalize_identifiers,
)

_normalize_sending = _normalize_location
_normalize_receiving = _normalize_location
