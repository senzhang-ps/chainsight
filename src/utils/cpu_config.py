# -*- coding: utf-8 -*-
"""
CPU 配置模块

动态获取CPU核心数，worker数量为运行环境CPU核心数的90%。
确保在不同环境中都能充分利用CPU资源，同时保留10%给系统其他进程。

重要：所有配置值都是动态获取的，每次调用函数时重新检测系统资源。

注意: 此模块已更新为使用90%资源利用率。
      更完整的资源配置请参考 resource_config.py
"""

import os
from typing import Optional

# 资源利用率 (90%)
RESOURCE_UTILIZATION = 0.9


def get_cpu_count() -> int:
    """
    动态获取CPU核心数。
    
    Returns:
        int: CPU核心数，至少为4
    """
    return os.cpu_count() or 4


def get_max_workers() -> int:
    """
    动态获取最大worker数 (90% CPU)。
    
    Returns:
        int: 最大worker数
    """
    return max(1, int(get_cpu_count() * RESOURCE_UTILIZATION))


def get_optimal_workers(task_count: Optional[int] = None) -> int:
    """
    动态获取最优的worker数量。
    
    根据CPU核心数和任务数量，返回最优的并行worker数。
    worker数量为CPU核心数的90%，同时不超过任务数量。
    
    Args:
        task_count: 任务数量，如果为None则返回MAX_WORKERS
        
    Returns:
        int: 最优worker数量
    """
    max_workers = get_max_workers()
    if task_count is None:
        return max_workers
    return min(max_workers, max(1, task_count))


def get_cpu_info() -> dict:
    """
    动态获取CPU配置信息。
    
    Returns:
        dict: 包含CPU核心数和最大worker数的字典 (每次调用重新检测)
    """
    cpu_count = get_cpu_count()
    max_workers = get_max_workers()
    return {
        'cpu_count': cpu_count,
        'max_workers': max_workers,
        'utilization': RESOURCE_UTILIZATION
    }


# ============================================================================
# 兼容性别名 (通过__getattr__实现动态获取)
# ============================================================================

def __getattr__(name: str):
    """
    模块级别的动态属性访问。
    
    允许使用 CPU_COUNT, MAX_WORKERS 等常量名，
    但实际上每次访问都会动态计算。
    """
    if name == 'CPU_COUNT':
        return get_cpu_count()
    elif name == 'MAX_WORKERS':
        return get_max_workers()
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# 模块初始化时打印配置信息（可选）
if __name__ == '__main__':
    info = get_cpu_info()
    print(f"CPU cores: {info['cpu_count']}")
    print(f"Max workers ({int(info['utilization']*100)}% utilization): {info['max_workers']}")
