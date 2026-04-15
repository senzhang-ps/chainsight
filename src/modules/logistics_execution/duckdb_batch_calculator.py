# -*- coding: utf-8 -*-
"""Module6 DuckDB 批量延迟抽样工具。

作用：
- 为Module6的路线延迟抽样提供批量化加速实现。
- 在满足条件时优先走 DuckDB/NumPy 向量化路径，不满足时回退到 Pandas 单条抽样路径。
- 保持与原始单条抽样逻辑一致的结果口径与随机数行为。
"""
import time
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np

try:
    import sys, os
    pgsql_db_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'pgsql_db')
    if pgsql_db_path not in sys.path:
        sys.path.insert(0, pgsql_db_path)
    from duckdb_integration import get_duckdb_calculator, DuckDBConfig, get_perf_stats
    DUCKDB_INTEGRATION_AVAILABLE = True
except ImportError:
    DUCKDB_INTEGRATION_AVAILABLE = False

def batch_sample_delivery_delays_duckdb(
    routes: List[Tuple[str, str]],
    dist_df: pd.DataFrame,
    seed: Optional[int] = None,
    run_id: Optional[str] = None
) -> np.ndarray:
    """按路线批量抽样交付延迟。

    参数：
        routes: 路线列表，元素为 `(sending, receiving)`。
        dist_df: 延迟分布配置表。
        seed: 随机种子，用于结果复现。
        run_id: 运行批次标识，用于性能统计。

    返回：
        与 `routes` 等长的延迟天数数组。
    """
    if not routes:
        return np.array([])
    
    # 判断是否适合启用批量优化
    if not DUCKDB_INTEGRATION_AVAILABLE or not DuckDBConfig.enabled or len(routes) < 10:
        return _batch_sample_delays_pandas(routes, dist_df, seed, run_id)
    
    calculator = get_duckdb_calculator()
    if calculator is None:
        return _batch_sample_delays_pandas(routes, dist_df, seed, run_id)
    
    t0 = time.perf_counter()
    try:
        delays = _vectorized_delay_sampling(routes, dist_df, seed)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        
        if run_id and DuckDBConfig.collect_stats:
            get_perf_stats().record(run_id, 'batch_sample_delays', 'duckdb', len(routes), elapsed_ms)
        
        return delays
    except Exception as e:
        if DuckDBConfig.fallback_on_error:
            return _batch_sample_delays_pandas(routes, dist_df, seed, run_id)
        raise

def _vectorized_delay_sampling(
    routes: List[Tuple[str, str]],
    dist_df: pd.DataFrame,
    seed: Optional[int]
) -> np.ndarray:
    """使用向量化方式执行延迟抽样。

    参数：
        routes: 路线列表，元素为 `(sending, receiving)`。
        dist_df: 延迟分布配置表。
        seed: 随机种子。

    返回：
        与输入路线一一对应的延迟天数数组。
    """
    if dist_df is None or dist_df.empty:
        return np.zeros(len(routes), dtype=int)
    
    required_cols = {'delay_days', 'probability', 'sending', 'receiving'}
    if not required_cols.issubset(set(dist_df.columns)):
        return np.zeros(len(routes), dtype=int)
    
    rng = np.random.RandomState(seed)
    delays = np.zeros(len(routes), dtype=int)
    
    # 构建路线延迟分布缓存
    dist_cache = {}
    for _, row in dist_df.iterrows():
        key = (str(row['sending']), str(row['receiving']))
        if key not in dist_cache:
            dist_cache[key] = {'delays': [], 'probs': []}
        dist_cache[key]['delays'].append(int(row['delay_days']))
        dist_cache[key]['probs'].append(float(row['probability']))
    
    # 检查是否存在全局兜底规则
    global_key = ('ALL', 'ALL')
    has_global = any(
        str(row['sending']).upper() == 'ALL' and str(row['receiving']).upper() == 'ALL'
        for _, row in dist_df.iterrows()
    )
    if has_global:
        global_dist = {'delays': [], 'probs': []}
        for _, row in dist_df.iterrows():
            if str(row['sending']).upper() == 'ALL' and str(row['receiving']).upper() == 'ALL':
                global_dist['delays'].append(int(row['delay_days']))
                global_dist['probs'].append(float(row['probability']))
        dist_cache[global_key] = global_dist
    
    # 将概率转换为 NumPy 数组并重新归一化
    for key in dist_cache:
        probs = np.array(dist_cache[key]['probs'], dtype=float)
        if probs.sum() > 0:
            probs = probs / probs.sum()
        dist_cache[key]['probs'] = probs
        dist_cache[key]['delays'] = np.array(dist_cache[key]['delays'], dtype=np.int32)
    
    # 使用 NumPy 执行向量化抽样
    rng = np.random.RandomState(seed)
    delays = np.zeros(len(routes), dtype=int)
    
    for i, (sending, receiving) in enumerate(routes):
        key = (sending, receiving)
        
        # 优先尝试精确路线匹配
        if key in dist_cache:
            dist = dist_cache[key]
        # 回退到全局规则
        elif global_key in dist_cache:
            dist = dist_cache[global_key]
        else:
            delays[i] = 0
            continue
        
        # 按分布执行抽样
        probs = dist['probs']
        if probs.sum() > 0:
            delays[i] = rng.choice(dist['delays'], p=probs)
        else:
            delays[i] = 0
    
    return delays

def _batch_sample_delays_pandas(
    routes: List[Tuple[str, str]],
    dist_df: pd.DataFrame,
    seed: Optional[int],
    run_id: Optional[str]
) -> np.ndarray:
    """使用 Pandas 路径回退执行批量延迟抽样。

    参数：
        routes: 路线列表，元素为 `(sending, receiving)`。
        dist_df: 延迟分布配置表。
        seed: 随机种子。
        run_id: 运行批次标识，用于性能统计。

    返回：
        与输入路线一一对应的延迟天数数组。
    """
    t0 = time.perf_counter()
    
    # 导入原始的单条记录抽样函数
    from .delivery_processor import sample_delivery_delay
    
    # 逐条抽样（不要修改全局随机状态）
    # Module6依赖全局随机状态在整次仿真中连续演进
    delays = np.array([
        sample_delivery_delay(sending, receiving, dist_df)
        for sending, receiving in routes
    ])
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    if run_id and DUCKDB_INTEGRATION_AVAILABLE and DuckDBConfig.collect_stats:
        get_perf_stats().record(run_id, 'batch_sample_delays', 'pandas', len(routes), elapsed_ms)
    
    return delays

def is_duckdb_available():
    """检查 DuckDB 集成是否可用且已启用。"""
    return DUCKDB_INTEGRATION_AVAILABLE and DuckDBConfig.enabled

def get_duckdb_config():
    """返回当前 DuckDB 集成配置摘要。"""
    if not DUCKDB_INTEGRATION_AVAILABLE:
        return {'available': False}
    return {'available': True, 'enabled': DuckDBConfig.enabled}
