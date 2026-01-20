# -*- coding: utf-8 -*-
"""
多进程执行器模块

使用 ProcessPoolExecutor 突破 GIL 限制，实现真正的 CPU 并行。
"""

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Any
import pandas as pd

# 设置进程启动方法（Windows 下必须用 spawn）
if os.name == 'nt':
    mp.set_start_method('spawn', force=True)


# 获取 CPU 核心数
CPU_COUNT = os.cpu_count() or 4
# 使用 90% 的 CPU 核心数来保证高利用率
DEFAULT_WORKERS = max(1, int(CPU_COUNT * 0.9))


class MultiProcessExecutor:
    """多进程执行器，用于 CPU 密集型任务的并行处理。"""
    
    _instance = None
    _executor = None
    
    def __init__(self, max_workers: Optional[int] = None):
        """
        初始化多进程执行器。
        
        Args:
            max_workers: 最大工作进程数，默认使用 90% CPU 核心
        """
        self.max_workers = max_workers or DEFAULT_WORKERS
    
    @classmethod
    def get_instance(cls, max_workers: Optional[int] = None) -> 'MultiProcessExecutor':
        """获取单例实例。"""
        if cls._instance is None:
            cls._instance = cls(max_workers)
        return cls._instance
    
    def map_parallel(
        self,
        func: Callable,
        items: List[Any],
        chunk_size: int = 10
    ) -> List[Any]:
        """
        并行处理列表项。
        
        Args:
            func: 处理函数
            items: 待处理项列表
            chunk_size: 每批处理的项数
            
        Returns:
            List[Any]: 处理结果列表
        """
        if not items:
            return []
        
        results = []
        n_workers = min(self.max_workers, len(items))
        
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(func, item): idx 
                          for idx, item in enumerate(items)}
                
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        result = future.result()
                        results.append((idx, result))
                    except Exception as e:
                        print(f"[MultiProcess] Task {idx} failed: {e}")
                        results.append((idx, None))
        except Exception as e:
            print(f"[MultiProcess] Parallel execution failed: {e}")
            # 回退到串行执行
            for idx, item in enumerate(items):
                try:
                    result = func(item)
                    results.append((idx, result))
                except Exception as e:
                    results.append((idx, None))
        
        # 按原始顺序排序并返回结果
        results.sort(key=lambda x: x[0])
        return [r[1] for r in results]


def parallel_dataframe_apply(
    df: pd.DataFrame,
    func: Callable,
    axis: int = 1,
    n_workers: Optional[int] = None
) -> pd.DataFrame:
    """
    并行应用函数到 DataFrame 的每行或每列。
    
    Args:
        df: 源 DataFrame
        func: 应用函数
        axis: 0 表示按列, 1 表示按行
        n_workers: 工作进程数
        
    Returns:
        pd.DataFrame: 结果 DataFrame
    """
    if df.empty:
        return df
    
    n_workers = n_workers or DEFAULT_WORKERS
    n_partitions = min(n_workers, len(df))
    
    # 分割 DataFrame
    chunks = _split_dataframe(df, n_partitions)
    
    # 并行处理
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_apply_to_chunk, chunk, func, axis) 
                  for chunk in chunks]
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"[parallel_apply] Chunk failed: {e}")
    
    # 合并结果
    if results:
        return pd.concat(results, ignore_index=True)
    return df


def _split_dataframe(df: pd.DataFrame, n: int) -> List[pd.DataFrame]:
    """将 DataFrame 分割为 n 个块。"""
    chunk_size = max(1, len(df) // n)
    return [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]


def _apply_to_chunk(chunk: pd.DataFrame, func: Callable, axis: int) -> pd.DataFrame:
    """对 DataFrame 块应用函数。"""
    return chunk.apply(func, axis=axis)


def get_optimal_workers(task_count: int, max_workers: Optional[int] = None) -> int:
    """
    根据任务数量获取最优工作进程数。
    
    Args:
        task_count: 任务数量
        max_workers: 最大工作进程数
        
    Returns:
        int: 推荐的工作进程数
    """
    max_w = max_workers or DEFAULT_WORKERS
    # 确保不超过任务数量和最大工作进程数
    return min(task_count, max_w, CPU_COUNT)


# 全局执行器实例
_executor: Optional[MultiProcessExecutor] = None


def get_executor() -> MultiProcessExecutor:
    """获取全局多进程执行器实例。"""
    global _executor
    if _executor is None:
        _executor = MultiProcessExecutor.get_instance()
    return _executor
