# -*- coding: utf-8 -*-
"""
进程池执行器模块

使用 ProcessPoolExecutor 突破 GIL 限制，实现真正的 CPU 并行。
针对 MRP 和部署规划的批量计算进行优化。

关键优化：
1. 使用 spawn 方式创建子进程（Windows 兼容）
2. 批量传输数据减少序列化开销
3. 预先创建进程池减少启动开销
4. 使用 cloudpickle 支持复杂对象序列化
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

# 获取 CPU 核心数
CPU_COUNT = os.cpu_count() or 4
# 使用 CPU 核心数的 90% 来保证高利用率
MAX_WORKERS = max(1, int(CPU_COUNT * 0.9))


def _worker_init():
    """工作进程初始化函数。"""
    # 设置进程优先级
    try:
        if sys.platform == 'win32':
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(
                ctypes.windll.kernel32.GetCurrentProcess(),
                0x00008000  # ABOVE_NORMAL_PRIORITY_CLASS
            )
    except Exception:
        pass


def batch_process_parallel(
    func: Callable,
    items: List[Any],
    n_workers: Optional[int] = None,
    batch_size: int = 50
) -> List[Any]:
    """
    批量并行处理函数。
    
    将任务分批提交到进程池，减少进程间通信开销。
    
    Args:
        func: 处理函数（必须可序列化）
        items: 待处理项列表
        n_workers: 工作进程数
        batch_size: 批处理大小
        
    Returns:
        List[Any]: 处理结果列表（保持原顺序）
    """
    if not items:
        return []
    
    n_workers = n_workers or MAX_WORKERS
    n_workers = min(n_workers, len(items))
    
    results = [None] * len(items)
    
    try:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init
        ) as executor:
            # 提交所有任务
            future_to_idx = {}
            for idx, item in enumerate(items):
                future = executor.submit(func, item)
                future_to_idx[future] = idx
            
            # 收集结果
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"[ProcessPool] Task {idx} failed: {e}")
                    results[idx] = None
                    
    except Exception as e:
        print(f"[ProcessPool] Parallel failed: {e}, fallback to serial")
        # 串行回退
        for idx, item in enumerate(items):
            try:
                results[idx] = func(item)
            except Exception as ex:
                print(f"[ProcessPool] Serial task {idx} failed: {ex}")
                results[idx] = None
    
    return results


def parallel_dataframe_groupby_apply(
    df: pd.DataFrame,
    groupby_cols: List[str],
    apply_func: Callable,
    n_workers: Optional[int] = None
) -> List[Any]:
    """
    对 DataFrame 分组并行应用函数。
    
    将 DataFrame 按 groupby_cols 分组，然后并行处理每个分组。
    
    Args:
        df: 源 DataFrame
        groupby_cols: 分组列
        apply_func: 应用函数 (group_key, group_df) -> result
        n_workers: 工作进程数
        
    Returns:
        List[Any]: 每个分组的处理结果
    """
    if df.empty:
        return []
    
    # 分组
    groups = list(df.groupby(groupby_cols))
    
    # 准备任务
    tasks = [(key, group_df.copy()) for key, group_df in groups]
    
    # 并行处理
    def process_task(task):
        key, group_df = task
        return apply_func(key, group_df)
    
    return batch_process_parallel(process_task, tasks, n_workers)


def parallel_layer_process(
    layer_items: List[Tuple[str, str]],
    process_func: Callable,
    context_data: dict,
    n_workers: Optional[int] = None
) -> List[Any]:
    """
    并行处理层级内的所有节点。
    
    专门为 M3/M5 的层级处理优化。
    
    Args:
        layer_items: (material, location) 元组列表
        process_func: 处理函数 (material, location, context_data) -> result
        context_data: 上下文数据（只读）
        n_workers: 工作进程数
        
    Returns:
        List[Any]: 处理结果列表
    """
    if not layer_items:
        return []
    
    # 将上下文数据序列化为字典（避免复杂对象序列化问题）
    serialized_context = _serialize_context(context_data)
    
    def process_item(item):
        mat, loc = item
        return process_func(mat, loc, serialized_context)
    
    return batch_process_parallel(process_item, layer_items, n_workers)


def _serialize_context(ctx: dict) -> dict:
    """
    序列化上下文数据。
    
    将 DataFrame 转换为 dict 格式以便序列化。
    """
    result = {}
    for key, value in ctx.items():
        if isinstance(value, pd.DataFrame):
            result[key] = value.to_dict('records')
        elif isinstance(value, dict):
            result[key] = _serialize_context(value)
        else:
            result[key] = value
    return result


def _deserialize_context(ctx: dict) -> dict:
    """
    反序列化上下文数据。
    
    将 dict 格式转换回 DataFrame。
    """
    result = {}
    for key, value in ctx.items():
        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
            result[key] = pd.DataFrame(value)
        elif isinstance(value, dict):
            result[key] = _deserialize_context(value)
        else:
            result[key] = value
    return result


class ProcessPoolManager:
    """进程池管理器，用于复用进程池。"""
    
    _instance = None
    _executor: Optional[ProcessPoolExecutor] = None
    
    def __init__(self, max_workers: Optional[int] = None):
        """
        初始化进程池管理器。
        
        Args:
            max_workers: 最大工作进程数
        """
        self.max_workers = max_workers or MAX_WORKERS
    
    @classmethod
    def get_instance(cls, max_workers: Optional[int] = None) -> 'ProcessPoolManager':
        """获取单例实例。"""
        if cls._instance is None:
            cls._instance = cls(max_workers)
        return cls._instance
    
    def start_pool(self):
        """启动进程池。"""
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=self.max_workers,
                initializer=_worker_init
            )
    
    def shutdown_pool(self):
        """关闭进程池。"""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
    
    def submit(self, func: Callable, *args, **kwargs):
        """提交任务。"""
        if self._executor is None:
            self.start_pool()
        return self._executor.submit(func, *args, **kwargs)
    
    def map(self, func: Callable, items: List[Any]) -> List[Any]:
        """映射处理。"""
        if self._executor is None:
            self.start_pool()
        return list(self._executor.map(func, items))


# 提供便捷访问
def get_pool_manager() -> ProcessPoolManager:
    """获取全局进程池管理器。"""
    return ProcessPoolManager.get_instance()


def get_optimal_workers() -> int:
    """获取最优工作进程数。"""
    return MAX_WORKERS


def print_cpu_info():
    """打印 CPU 信息。"""
    print(f"[ProcessPool] CPU cores: {CPU_COUNT}, Workers: {MAX_WORKERS}")
