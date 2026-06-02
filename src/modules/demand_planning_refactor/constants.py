"""Module1 常量定义与默认配置。

本模块定义Module1使用的全局常量和默认配置参数。

常量分类：
- 性能优化参数：并行计算开关与并发度
- 业务参数：最大AO提前天数
- 日志参数：错误日志路径
"""

import os
from typing import Optional

# 导入统一的 CPU 配置
from src.utils.resource_config import MAX_WORKERS
from src.utils.defaults import M1_DEFAULT_MAX_ADVANCE_DAYS

# ----------- 性能优化参数 -----------

DEFAULT_MAX_ADVANCE_DAYS: int = M1_DEFAULT_MAX_ADVANCE_DAYS


# 并行计算开关（默认关闭以确保与旧版输出一致）
DEFAULT_USE_PARALLEL_AO_CONSUME: bool = True
DEFAULT_USE_PARALLEL_FILE_LOAD: bool = True
DEFAULT_USE_PARALLEL_NORMAL_CONSUME: Optional[bool] = None

# 使用优化版消耗（向量化+字典索引）
DEFAULT_USE_OPTIMIZED_CONSUME: bool = True

# 并发工作进程/线程数（动态配置：使用 90% CPU 核心）
DEFAULT_PARALLEL_MAX_WORKERS: int = MAX_WORKERS

# ----------- 日志参数 -----------

# 异常日志文件路径（默认不启用）
DEFAULT_ERROR_LOG_PATH: Optional[str] = None


def append_error_log(message: str) -> None:
    """追加错误消息到日志文件。

    参数:
        message: 需要记录的错误消息。

    说明:
        - 默认不写入磁盘。
        - 设置 `DEFAULT_ERROR_LOG_PATH` 为文件路径以启用日志。
        - 写入失败时静默处理，避免影响主流程。
    """
    try:
        path = DEFAULT_ERROR_LOG_PATH
        if not path:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(str(message).rstrip('\n') + '\n')
    except Exception:
        pass
