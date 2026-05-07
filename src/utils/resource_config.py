# -*- coding: utf-8 -*-
"""
系统资源配置模块

动态获取运行环境的CPU和内存信息，统一配置资源使用率。
默认使用90%的系统资源，确保系统稳定性同时最大化性能。

重要：所有配置值都是动态获取的，每次调用函数时重新检测系统资源。
      这确保了在容器化环境、资源限制变化时能正确获取当前可用资源。

用法:
    from src.utils.resource_config import (
        get_optimal_threads,
        get_optimal_memory,
        get_resource_config,
        get_duckdb_config,
        RESOURCE_UTILIZATION,
    )
"""

import os
from typing import Optional

from src.utils.defaults import RESOURCE_UTILIZATION

# 尝试导入psutil获取更准确的内存信息
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


# ============================================================================
# 核心配置参数 (从 defaults.yaml 加载)
# ============================================================================



# ============================================================================
# 动态CPU配置
# ============================================================================

def get_cpu_count() -> int:
    """
    动态获取CPU核心数。
    
    Returns:
        int: CPU核心数，至少为1
    """
    return os.cpu_count() or 4


def get_optimal_threads(task_count: Optional[int] = None) -> int:
    """
    动态获取最优的线程数。
    
    根据当前CPU核心数(90%)和任务数量，返回最优的并行线程数。
    
    Args:
        task_count: 任务数量，如果为None则返回90% CPU线程数
        
    Returns:
        int: 最优线程数
    """
    cpu_count = get_cpu_count()
    optimal = max(1, int(cpu_count * RESOURCE_UTILIZATION))
    
    if task_count is None:
        return optimal
    return min(optimal, max(1, task_count))


def get_optimal_workers(task_count: Optional[int] = None) -> int:
    """
    动态获取最优的worker数量。
    
    根据当前CPU核心数(90%)和任务数量，返回最优的并行worker数。
    
    Args:
        task_count: 任务数量，如果为None则返回90% CPU worker数
        
    Returns:
        int: 最优worker数量
    """
    return get_optimal_threads(task_count)


# ============================================================================
# 动态内存配置
# ============================================================================

def get_total_memory_gb() -> float:
    """
    动态获取系统总内存 (GB)。
    
    Returns:
        float: 总内存GB数，如果无法获取则返回8GB作为默认值
    """
    if PSUTIL_AVAILABLE:
        try:
            mem = psutil.virtual_memory()
            return mem.total / (1024 ** 3)  # 转换为GB
        except Exception:
            pass
    
    # 回退方案：尝试从环境变量获取或使用默认值
    return float(os.environ.get('SYSTEM_MEMORY_GB', 8))


def get_available_memory_gb() -> float:
    """
    动态获取系统当前可用内存 (GB)。
    
    Returns:
        float: 可用内存GB数
    """
    if PSUTIL_AVAILABLE:
        try:
            mem = psutil.virtual_memory()
            return mem.available / (1024 ** 3)
        except Exception:
            pass
    
    # 回退：使用总内存的50%作为估计
    return get_total_memory_gb() * 0.5


def get_optimal_memory_gb() -> int:
    """
    动态获取最优内存配置 (GB整数)。
    
    Returns:
        int: 可分配的内存GB数 (总内存的90%)
    """
    total = get_total_memory_gb()
    return max(1, int(total * RESOURCE_UTILIZATION))


def get_optimal_memory() -> str:
    """
    动态获取最优的内存配置字符串。
    
    Returns:
        str: 内存配置字符串，如 "28GB"
    """
    return f"{get_optimal_memory_gb()}GB"


def get_optimal_memory_bytes() -> int:
    """
    动态获取最优的内存配置 (bytes)。
    
    Returns:
        int: 内存字节数
    """
    return get_optimal_memory_gb() * (1024 ** 3)


# ============================================================================
# 综合配置接口
# ============================================================================

def get_resource_config() -> dict:
    """
    动态获取完整的资源配置信息。
    
    Returns:
        dict: 资源配置字典 (所有值都是实时获取的)
    """
    cpu_count = get_cpu_count()
    total_memory = get_total_memory_gb()
    optimal_threads = get_optimal_threads()
    optimal_memory_gb = get_optimal_memory_gb()
    
    return {
        'utilization': RESOURCE_UTILIZATION,
        'cpu_count': cpu_count,
        'optimal_threads': optimal_threads,
        'max_workers': optimal_threads,
        'total_memory_gb': total_memory,
        'available_memory_gb': get_available_memory_gb(),
        'optimal_memory_gb': optimal_memory_gb,
        'optimal_memory_str': f"{optimal_memory_gb}GB",
        'psutil_available': PSUTIL_AVAILABLE,
    }


def print_resource_config():
    """打印当前资源配置信息。"""
    config = get_resource_config()


# ============================================================================
# DuckDB专用配置
# ============================================================================

def get_duckdb_config() -> dict:
    """
    动态获取DuckDB优化配置。
    
    每次调用都会重新检测系统资源，确保配置是最新的。
    
    Returns:
        dict: DuckDB配置参数
    """
    return {
        'memory_limit': get_optimal_memory(),
        'threads': get_optimal_threads(),
    }


# ============================================================================
# 兼容性别名 (通过__getattr__实现动态获取)
# ============================================================================

def __getattr__(name: str):
    """
    模块级别的动态属性访问。
    
    允许使用 CPU_COUNT, OPTIMAL_THREADS 等常量名，
    但实际上每次访问都会动态计算。
    """
    if name == 'CPU_COUNT':
        return get_cpu_count()
    elif name == 'OPTIMAL_THREADS':
        return get_optimal_threads()
    elif name == 'MAX_WORKERS':
        return get_optimal_threads()
    elif name == 'TOTAL_MEMORY_GB':
        return get_total_memory_gb()
    elif name == 'OPTIMAL_MEMORY_GB':
        return get_optimal_memory_gb()
    elif name == 'OPTIMAL_MEMORY_STR':
        return get_optimal_memory()
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


# 模块初始化时打印配置信息（仅在直接运行时）
if __name__ == '__main__':
    print("=== 动态资源配置测试 ===")
    print_resource_config()
    print(f"\nDuckDB配置: {get_duckdb_config()}")
    
    # 测试动态属性
    print(f"\n动态属性测试:")
    print(f"  CPU_COUNT: {CPU_COUNT}")
    print(f"  OPTIMAL_THREADS: {OPTIMAL_THREADS}")
    print(f"  TOTAL_MEMORY_GB: {TOTAL_MEMORY_GB}")
    print(f"  OPTIMAL_MEMORY_STR: {OPTIMAL_MEMORY_STR}")
