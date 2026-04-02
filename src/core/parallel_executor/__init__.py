"""
parallel_executor package

并行执行框架 - Phase 5 性能优化

提供：
- ParallelTaskResult: 任务执行结果数据模型
- ParallelExecutor: 并行执行管理器
- run_parallel_modules: 便捷的并行执行函数

使用：
    from .parallel_executor import (
        ParallelExecutor, run_parallel_modules
    )
"""

from .models import ParallelTaskResult
from .parallel_executor_main import ParallelExecutor
from .convenience import run_parallel_modules

__all__ = [
    'ParallelTaskResult',
    'ParallelExecutor',
    'run_parallel_modules',
]
