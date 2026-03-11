"""
models.py

并行任务执行结果数据模型

定义：
- ParallelTaskResult: 任务执行结果的数据类
- 包含状态、结果、错误信息和计时
"""

from dataclasses import dataclass
from typing import Any, Optional
from datetime import datetime


@dataclass
class ParallelTaskResult:
    """并行任务执行结果"""
    task_name: str
    status: str  # 'success' | 'error' | 'timeout'
    result: Any = None
    error: Optional[Exception] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def elapsed_time(self) -> float:
        """返回执行耗时（秒）"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    def __str__(self) -> str:
        """格式化结果字符串"""
        if self.status == 'success':
            return (f"✅ {self.task_name}: 成功 "
                    f"({self.elapsed_time:.2f}s)")
        elif self.status == 'error':
            return (f"❌ {self.task_name}: 失败 - "
                    f"{str(self.error)}")
        else:
            return (f"⏱️ {self.task_name}: {self.status}")
