# -*- coding: utf-8 -*-
"""
并行优化器模块

使用 ProcessPoolExecutor + DuckDB 实现高 CPU 利用率的并行计算。
目标：达到 90% CPU 利用率。

策略:
1. 使用 ProcessPoolExecutor 突破 GIL 限制
2. 批量预计算减少重复 DataFrame 过滤
3. 利用 DuckDB C++ 引擎进行高效数据处理

优化历史:
- v1.0: 基础实现
- v1.1: 动态CPU配置，使用90%CPU资源
"""
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Callable, Any, Optional
import pandas as pd
import numpy as np

# 使用统一的CPU配置
from .cpu_config import MAX_WORKERS, get_optimal_workers


def parallel_batch_process(
    items: List[Any],
    process_func: Callable,
    max_workers: int = MAX_WORKERS,
    use_process: bool = False,
    chunk_size: int = 10
) -> List[Any]:
    """
    并行批量处理项目。
    
    Args:
        items: 要处理的项目列表
        process_func: 处理函数
        max_workers: 最大工作进程/线程数
        use_process: 是否使用进程池（突破GIL）
        chunk_size: 每批处理的项目数
        
    Returns:
        处理结果列表
    """
    if not items:
        return []
    
    # 对于小数据量，直接串行处理
    if len(items) <= chunk_size:
        return [process_func(item) for item in items]
    
    results = []
    Executor = ProcessPoolExecutor if use_process else ThreadPoolExecutor
    
    with Executor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_func, item): i 
                   for i, item in enumerate(items)}
        
        # 按提交顺序收集结果
        result_map = {}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result_map[idx] = future.result()
            except Exception as e:
                print(f"Error processing item {idx}: {e}")
                result_map[idx] = None
        
        results = [result_map[i] for i in range(len(items))]
    
    return results


def batch_dataframe_filter(
    df: pd.DataFrame,
    key_pairs: List[Tuple[str, str]],
    key_cols: Tuple[str, str] = ('material', 'location')
) -> Dict[Tuple[str, str], pd.DataFrame]:
    """
    批量过滤 DataFrame，一次性构建所有键的索引。
    
    比逐个过滤快 10-100x。
    
    Args:
        df: 源 DataFrame
        key_pairs: (key1, key2) 对列表
        key_cols: 键列名 (col1, col2)
        
    Returns:
        dict: (key1, key2) -> 过滤后的 DataFrame
    """
    if df.empty or not key_pairs:
        return {pair: pd.DataFrame() for pair in key_pairs}
    
    col1, col2 = key_cols
    
    # 转换为字符串确保匹配
    df = df.copy()
    df[col1] = df[col1].astype(str)
    df[col2] = df[col2].astype(str)
    
    # 构建复合键
    df['_key'] = df[col1] + '|' + df[col2]
    
    # 一次性分组
    grouped = df.groupby('_key', sort=False)
    
    # 构建结果
    result = {}
    for k1, k2 in key_pairs:
        key = f"{k1}|{k2}"
        if key in grouped.groups:
            result[(k1, k2)] = grouped.get_group(key).drop(columns=['_key'])
        else:
            result[(k1, k2)] = pd.DataFrame()
    
    return result


def vectorized_demand_aggregation(
    demands: List[dict],
    group_keys: List[str] = ['material', 'location', 'requirement_date']
) -> pd.DataFrame:
    """
    向量化需求聚合。
    
    Args:
        demands: 需求字典列表
        group_keys: 分组键
        
    Returns:
        聚合后的 DataFrame
    """
    if not demands:
        return pd.DataFrame()
    
    df = pd.DataFrame(demands)
    
    # 聚合数值列
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    agg_dict = {col: 'sum' for col in numeric_cols if col not in group_keys}
    
    # 非数值列取第一个
    non_numeric = [col for col in df.columns if col not in group_keys and col not in agg_dict]
    for col in non_numeric:
        agg_dict[col] = 'first'
    
    if not agg_dict:
        return df
    
    return df.groupby(group_keys, as_index=False).agg(agg_dict)


class ParallelBatchProcessor:
    """并行批量处理器，用于高效处理大量节点。"""
    
    def __init__(self, max_workers: int = MAX_WORKERS, use_multiprocess: bool = False):
        """
        初始化处理器。
        
        Args:
            max_workers: 最大工作线程/进程数
            use_multiprocess: 是否使用多进程
        """
        self.max_workers = max_workers
        self.use_multiprocess = use_multiprocess
        
    def process_nodes_parallel(
        self,
        nodes: List[Tuple[str, str]],  # (material, location)
        process_func: Callable[[str, str], Any],
        batch_size: int = 50
    ) -> Dict[Tuple[str, str], Any]:
        """
        并行处理节点。
        
        Args:
            nodes: 节点列表
            process_func: 处理函数 (material, location) -> result
            batch_size: 批处理大小
            
        Returns:
            dict: (material, location) -> result
        """
        if not nodes:
            return {}
        
        # 小数据量串行处理
        if len(nodes) <= batch_size:
            return {node: process_func(node[0], node[1]) for node in nodes}
        
        results = {}
        Executor = ProcessPoolExecutor if self.use_multiprocess else ThreadPoolExecutor
        
        with Executor(max_workers=self.max_workers) as executor:
            futures = {}
            for mat, loc in nodes:
                future = executor.submit(process_func, mat, loc)
                futures[future] = (mat, loc)
            
            for future in as_completed(futures):
                node = futures[future]
                try:
                    results[node] = future.result()
                except Exception as e:
                    print(f"Error processing node {node}: {e}")
                    results[node] = None
        
        return results


def precompute_all_indices(
    supply_demand_df: pd.DataFrame,
    safety_stock_df: pd.DataFrame,
    order_df: pd.DataFrame,
    inventory_df: pd.DataFrame
) -> Dict[str, Dict[Tuple[str, str], pd.DataFrame]]:
    """
    预计算所有数据的 (material, location) 索引。
    
    一次性构建，避免重复过滤。
    
    Args:
        supply_demand_df: 供需日志
        safety_stock_df: 安全库存
        order_df: 订单日志
        inventory_df: 库存
        
    Returns:
        dict: {数据类型: {(material, location): DataFrame}}
    """
    indices = {}
    
    # 供需日志索引
    if not supply_demand_df.empty:
        indices['supply_demand'] = _build_ml_index(supply_demand_df)
    
    # 安全库存索引
    if not safety_stock_df.empty:
        indices['safety_stock'] = _build_ml_index(safety_stock_df)
    
    # 订单索引
    if not order_df.empty:
        indices['order'] = _build_ml_index(order_df)
    
    # 库存索引
    if not inventory_df.empty:
        indices['inventory'] = _build_ml_index(inventory_df)
    
    return indices


def _build_ml_index(df: pd.DataFrame) -> Dict[Tuple[str, str], pd.DataFrame]:
    """构建 (material, location) 索引。"""
    if df.empty or 'material' not in df.columns or 'location' not in df.columns:
        return {}
    
    df = df.copy()
    df['material'] = df['material'].astype(str)
    df['location'] = df['location'].astype(str)
    
    result = {}
    for (mat, loc), group in df.groupby(['material', 'location'], sort=False):
        result[(mat, loc)] = group.copy()
    
    return result


# 全局预热缓存
_PRECOMPUTED_CACHE = {}


def warmup_cache(config_data: Dict[str, pd.DataFrame]):
    """
    预热缓存，在仿真开始前一次性准备好所有静态数据索引。
    
    Args:
        config_data: 配置数据字典
    """
    global _PRECOMPUTED_CACHE
    
    # 构建网络索引
    if 'Global_Network' in config_data:
        network = config_data['Global_Network']
        if not network.empty:
            _PRECOMPUTED_CACHE['network_index'] = _build_ml_index(network)
    
    # 构建安全库存索引
    if 'M3_SafetyStock' in config_data:
        ss = config_data['M3_SafetyStock']
        if not ss.empty:
            _PRECOMPUTED_CACHE['safety_stock_index'] = _build_ml_index(ss)
    
    print(f"✅ 预热缓存完成: {list(_PRECOMPUTED_CACHE.keys())}")


def get_cached_index(name: str) -> Optional[Dict[Tuple[str, str], pd.DataFrame]]:
    """获取预热的缓存索引。"""
    return _PRECOMPUTED_CACHE.get(name)


def clear_cache():
    """清除预热缓存。"""
    global _PRECOMPUTED_CACHE
    _PRECOMPUTED_CACHE = {}
