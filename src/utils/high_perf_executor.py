# -*- coding: utf-8 -*-
"""
高性能并行执行器 v3.0

使用 multiprocessing 和 concurrent.futures 实现真正的 CPU 并行。
针对 ChainSight 的批量计算优化，目标是 95% CPU 利用率。

关键特性:
1. 进程池复用减少创建开销
2. 数据分区减少序列化开销
3. 异步提交最大化并发
4. 自适应批量大小
"""

import os
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Tuple, Any, Iterator
from functools import partial
import pandas as pd
import numpy as np

# CPU 配置
CPU_COUNT = os.cpu_count() or 4
# 使用 95% 的 CPU 核心（目标）
OPTIMAL_WORKERS = max(1, int(CPU_COUNT * 0.95))


class HighPerformanceExecutor:
    """高性能执行器，支持进程/线程混合并行。"""
    
    _instance = None
    _process_pool: Optional[ProcessPoolExecutor] = None
    _thread_pool: Optional[ThreadPoolExecutor] = None
    
    def __init__(self, n_workers: Optional[int] = None):
        """初始化执行器。"""
        self.n_workers = n_workers or OPTIMAL_WORKERS
        self._stats = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'total_time': 0.0
        }
    
    @classmethod
    def get_instance(cls, n_workers: Optional[int] = None) -> 'HighPerformanceExecutor':
        """获取单例实例。"""
        if cls._instance is None:
            cls._instance = cls(n_workers)
        return cls._instance
    
    def get_process_pool(self) -> ProcessPoolExecutor:
        """获取或创建进程池。"""
        if self._process_pool is None:
            self.__class__._process_pool = ProcessPoolExecutor(
                max_workers=self.n_workers,
                mp_context=mp.get_context('spawn')  # Windows 兼容
            )
        return self._process_pool
    
    def get_thread_pool(self) -> ThreadPoolExecutor:
        """获取或创建线程池。"""
        if self._thread_pool is None:
            # 线程池使用更多 workers（IO 密集型）
            self.__class__._thread_pool = ThreadPoolExecutor(
                max_workers=self.n_workers * 4
            )
        return self._thread_pool
    
    def shutdown(self):
        """关闭所有池。"""
        if self._process_pool is not None:
            self._process_pool.shutdown(wait=True)
            self.__class__._process_pool = None
        if self._thread_pool is not None:
            self._thread_pool.shutdown(wait=True)
            self.__class__._thread_pool = None
    
    def parallel_map_threads(
        self,
        func: Callable,
        items: List[Any],
        timeout: Optional[float] = None
    ) -> List[Any]:
        """
        使用线程池并行映射（适合 IO 密集型）。
        
        Args:
            func: 处理函数
            items: 项列表
            timeout: 超时时间
            
        Returns:
            处理结果列表
        """
        if not items:
            return []
        
        pool = self.get_thread_pool()
        results = [None] * len(items)
        
        futures = {
            pool.submit(func, item): idx
            for idx, item in enumerate(items)
        }
        
        for future in as_completed(futures, timeout=timeout):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"[Executor] Thread task {idx} failed: {e}")
                results[idx] = None
        
        return results
    
    def parallel_map_processes(
        self,
        func: Callable,
        items: List[Any],
        timeout: Optional[float] = None
    ) -> List[Any]:
        """
        使用进程池并行映射（适合 CPU 密集型）。
        
        注意：func 和 items 必须可序列化。
        
        Args:
            func: 处理函数
            items: 项列表
            timeout: 超时时间
            
        Returns:
            处理结果列表
        """
        if not items:
            return []
        
        # 对于小任务数，直接串行
        if len(items) < 4:
            return [func(item) for item in items]
        
        pool = self.get_process_pool()
        results = [None] * len(items)
        
        try:
            futures = {
                pool.submit(func, item): idx
                for idx, item in enumerate(items)
            }
            
            for future in as_completed(futures, timeout=timeout):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"[Executor] Process task {idx} failed: {e}")
                    results[idx] = None
                    
        except Exception as e:
            print(f"[Executor] Process pool failed: {e}, falling back to serial")
            results = [func(item) for item in items]
        
        return results
    
    def batch_process(
        self,
        func: Callable,
        items: List[Any],
        batch_size: int = 100,
        use_processes: bool = False
    ) -> List[Any]:
        """
        分批并行处理（减少内存压力）。
        
        Args:
            func: 处理函数
            items: 项列表
            batch_size: 批大小
            use_processes: 是否使用进程
            
        Returns:
            处理结果列表
        """
        if not items:
            return []
        
        all_results = []
        n_batches = (len(items) + batch_size - 1) // batch_size
        
        for i in range(n_batches):
            batch = items[i * batch_size:(i + 1) * batch_size]
            if use_processes:
                batch_results = self.parallel_map_processes(func, batch)
            else:
                batch_results = self.parallel_map_threads(func, batch)
            all_results.extend(batch_results)
        
        return all_results
    
    def chunked_dataframe_apply(
        self,
        df: pd.DataFrame,
        func: Callable[[pd.DataFrame], pd.DataFrame],
        n_chunks: Optional[int] = None
    ) -> pd.DataFrame:
        """
        将 DataFrame 分块并行处理。
        
        Args:
            df: 源 DataFrame
            func: 处理函数 (chunk_df) -> result_df
            n_chunks: 分块数
            
        Returns:
            合并后的 DataFrame
        """
        if df.empty:
            return df
        
        n_chunks = n_chunks or self.n_workers
        n_chunks = min(n_chunks, len(df))
        
        if n_chunks <= 1:
            return func(df)
        
        # 分块
        chunk_size = len(df) // n_chunks
        chunks = [
            df.iloc[i * chunk_size:(i + 1) * chunk_size].copy()
            for i in range(n_chunks - 1)
        ]
        chunks.append(df.iloc[(n_chunks - 1) * chunk_size:].copy())
        
        # 并行处理
        results = self.parallel_map_threads(func, chunks)
        
        # 合并结果
        valid_results = [r for r in results if r is not None and not r.empty]
        if not valid_results:
            return pd.DataFrame()
        
        return pd.concat(valid_results, ignore_index=True)


# 全局执行器实例
_executor: Optional[HighPerformanceExecutor] = None


def get_executor() -> HighPerformanceExecutor:
    """获取全局执行器。"""
    global _executor
    if _executor is None:
        _executor = HighPerformanceExecutor.get_instance()
    return _executor


def parallel_process_layers(
    layer_data: Dict[int, List[Tuple[str, str]]],
    process_func: Callable,
    ctx: dict
) -> List[Any]:
    """
    并行处理多个层级的节点。
    
    注意：层级之间必须串行（存在依赖），但层内节点可以并行。
    
    Args:
        layer_data: {layer_num: [(material, location), ...]}
        process_func: 处理函数 (material, location, ctx) -> result
        ctx: 上下文数据
        
    Returns:
        所有层的处理结果
    """
    executor = get_executor()
    all_results = []
    
    # 按层级串行处理（层内并行）
    for layer_num in sorted(layer_data.keys(), reverse=True):
        nodes = layer_data[layer_num]
        
        # 构建任务
        def process_node(node):
            mat, loc = node
            return process_func(mat, loc, ctx)
        
        # 层内并行
        layer_results = executor.parallel_map_threads(process_node, nodes)
        all_results.extend(layer_results)
    
    return all_results


def optimize_cpu_bound_loop(
    items: List[Any],
    func: Callable,
    use_multiprocessing: bool = False
) -> List[Any]:
    """
    优化 CPU 密集型循环。
    
    Args:
        items: 待处理项
        func: 处理函数
        use_multiprocessing: 是否使用多进程
        
    Returns:
        处理结果
    """
    executor = get_executor()
    
    if use_multiprocessing:
        return executor.parallel_map_processes(func, items)
    else:
        return executor.parallel_map_threads(func, items)


def get_optimal_workers() -> int:
    """获取最优工作数。"""
    return OPTIMAL_WORKERS


def print_executor_stats():
    """打印执行器统计。"""
    print(f"[Executor] CPU cores: {CPU_COUNT}, Workers: {OPTIMAL_WORKERS}")
    if _executor:
        print(f"[Executor] Stats: {_executor._stats}")
