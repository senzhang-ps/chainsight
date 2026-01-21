# -*- coding: utf-8 -*-
"""
CPU 配置模块

动态获取CPU核心数，worker数量等于运行环境的CPU核心数。
确保在不同环境中都能充分利用CPU资源。
"""

import os
from typing import Optional

# 获取CPU核心数
CPU_COUNT = os.cpu_count() or 4

# worker数量等于CPU核心数
MAX_WORKERS = CPU_COUNT


def get_optimal_workers(task_count: Optional[int] = None) -> int:
    """
    获取最优的worker数量。
    
    根据CPU核心数和任务数量，返回最优的并行worker数。
    worker数量等于CPU核心数，同时不超过任务数量。
    
    Args:
        task_count: 任务数量，如果为None则返回MAX_WORKERS
        
    Returns:
        int: 最优worker数量
    """
    if task_count is None:
        return MAX_WORKERS
    return min(MAX_WORKERS, max(1, task_count))


def get_cpu_info() -> dict:
    """
    获取CPU配置信息。
    
    Returns:
        dict: 包含CPU核心数和最大worker数的字典
    """
    return {
        'cpu_count': CPU_COUNT,
        'max_workers': MAX_WORKERS,
        'utilization': 1.0
    }


# 模块初始化时打印配置信息（可选）
if __name__ == '__main__':
    info = get_cpu_info()
    print(f"CPU cores: {info['cpu_count']}")
    print(f"Max workers (100% utilization): {info['max_workers']}")
