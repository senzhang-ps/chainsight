"""配置表输入数据质量检测公共接口。"""

from .checker import (
    ConfigInputDataQualityChecker,
    ConfigTableQualityRules,
    DataQualityError,
)

__all__ = [
    "ConfigInputDataQualityChecker",
    "ConfigTableQualityRules",
    "DataQualityError",
]
