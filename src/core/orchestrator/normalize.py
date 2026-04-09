# normalize.py
# 标识符规范化函数集
#
# 将 material / location / sending / receiving 等标识符列
# 转换为统一的字符串格式，确保跨模块数据一致性。
#
# 注意：所有实现已统一到 src/utils/normalization.py，
# 本文件作为兼容导入层保留，使用下划线前缀别名以维持原有接口。

from src.utils.normalization import (
    normalize_material as _normalize_material,
    normalize_location as _normalize_location,
    normalize_location as _normalize_sending,
    normalize_location as _normalize_receiving,
    normalize_identifiers as _normalize_identifiers,
)

__all__ = [
    '_normalize_material',
    '_normalize_location',
    '_normalize_sending',
    '_normalize_receiving',
    '_normalize_identifiers',
]
