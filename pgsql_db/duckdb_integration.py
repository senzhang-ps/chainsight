# -*- coding: utf-8 -*-
"""
DuckDB 集成桥接模块

将 high_performance_engine.py 中的 DuckDB 计算函数实际接入到:
1. Module3 净需求计算
2. Module5 优先级分配
3. Module5 MOQ/RV处理

此模块提供向后兼容的包装器，可以在不修改原有代码逻辑的情况下，
通过配置开关启用 DuckDB 优化计算。
"""

import time
import functools
from typing import Dict, List, Optional, Tuple, Callable, Any
from contextlib import contextmanager

import numpy as np
import pandas as pd

# 导入高性能引擎
try:
    from .high_performance_engine import (
        DuckDBCalculator,
        HighPerformanceEngine,
        create_high_performance_engine,
    )
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

# 导入动态资源配置
try:
    from src.utils.resource_config import (
        get_optimal_memory,
        get_optimal_threads,
    )
    _RESOURCE_CONFIG_AVAILABLE = True
except ImportError:
    _RESOURCE_CONFIG_AVAILABLE = False
    def get_optimal_memory(): return "4GB"
    def get_optimal_threads(n=None): return 4 if n is None else min(4, n)


# ============================================================================
# 全局配置
# ============================================================================

class DuckDBConfig:
    """DuckDB 集成配置 (所有资源配置都是动态获取的)"""
    
    # 是否启用 DuckDB 优化
    enabled: bool = True
    
    # 是否记录性能统计
    collect_stats: bool = True
    
    # 自动回退到 Pandas（出错时）
    fallback_on_error: bool = True
    
    @classmethod
    def get_memory_limit(cls) -> str:
        """动态获取内存限制 (系统90%内存)"""
        return get_optimal_memory()
    
    @classmethod
    def get_threads(cls) -> int:
        """动态获取线程数 (系统90% CPU)"""
        return get_optimal_threads()
    
    # 兼容性属性 (动态获取)
    @property
    def memory_limit(self) -> str:
        return get_optimal_memory()
    
    @property
    def threads(self) -> int:
        return get_optimal_threads()
    
    # 最小数据量阈值（低于此值使用 Pandas 更快）
    # 优化：降低阈值以利用DuckDB的SQL优化，特别是批量计算场景
    min_rows_threshold: int = 100


# 全局引擎实例（懒加载）
_engine_instance: Optional[DuckDBCalculator] = None


def get_duckdb_calculator() -> Optional[DuckDBCalculator]:
    """获取全局 DuckDB 计算器实例（懒加载）"""
    global _engine_instance
    
    if not DUCKDB_AVAILABLE or not DuckDBConfig.enabled:
        return None
    
    if _engine_instance is None:
        _engine_instance = DuckDBCalculator(
            memory_limit=DuckDBConfig.memory_limit,
            threads=DuckDBConfig.threads
        )
    
    return _engine_instance


def close_duckdb_calculator():
    """关闭全局 DuckDB 计算器"""
    global _engine_instance
    if _engine_instance is not None:
        _engine_instance.close()
        _engine_instance = None


# ============================================================================
# 性能统计收集器
# ============================================================================

class PerformanceStats:
    """性能统计收集器"""
    
    def __init__(self):
        self._stats: Dict[str, Dict[str, Any]] = {}
        self._current_run: Dict[str, List[Dict]] = {}
    
    def start_run(self, run_id: str):
        """开始一次运行"""
        self._current_run[run_id] = []
    
    def record(self, run_id: str, operation: str, method: str, 
               rows: int, elapsed_ms: float, success: bool = True):
        """记录一次操作"""
        if run_id not in self._current_run:
            self._current_run[run_id] = []
        
        self._current_run[run_id].append({
            'operation': operation,
            'method': method,  # 'duckdb' or 'pandas'
            'rows': rows,
            'elapsed_ms': elapsed_ms,
            'success': success,
            'timestamp': time.time()
        })
    
    def end_run(self, run_id: str) -> Dict[str, Any]:
        """结束运行并返回统计"""
        if run_id not in self._current_run:
            return {}
        
        records = self._current_run.pop(run_id)
        
        # 按操作和方法分组统计
        stats = {}
        for r in records:
            key = f"{r['operation']}_{r['method']}"
            if key not in stats:
                stats[key] = {
                    'count': 0,
                    'total_rows': 0,
                    'total_ms': 0.0,
                    'errors': 0
                }
            stats[key]['count'] += 1
            stats[key]['total_rows'] += r['rows']
            stats[key]['total_ms'] += r['elapsed_ms']
            if not r['success']:
                stats[key]['errors'] += 1
        
        self._stats[run_id] = stats
        return stats
    
    def get_comparison(self, run_id: str) -> Dict[str, Any]:
        """获取 DuckDB vs Pandas 对比"""
        if run_id not in self._stats:
            return {}
        
        stats = self._stats[run_id]
        comparison = {}
        
        operations = set(k.rsplit('_', 1)[0] for k in stats.keys())
        for op in operations:
            duck_key = f"{op}_duckdb"
            pandas_key = f"{op}_pandas"
            
            duck_stats = stats.get(duck_key, {})
            pandas_stats = stats.get(pandas_key, {})
            
            comparison[op] = {
                'duckdb': duck_stats,
                'pandas': pandas_stats,
                'speedup': (
                    pandas_stats.get('total_ms', 0) / max(duck_stats.get('total_ms', 1), 0.001)
                    if duck_stats.get('total_ms', 0) > 0 else None
                )
            }
        
        return comparison


# 全局统计实例
_perf_stats = PerformanceStats()


def get_perf_stats() -> PerformanceStats:
    """获取全局性能统计实例"""
    return _perf_stats


# ============================================================================
# Module3 净需求计算 - DuckDB 集成
# ============================================================================

def calculate_net_demand_duckdb(
    material: str,
    location: str,
    date: pd.Timestamp,
    beginning_inventory: float,
    in_transit: float,
    delivery_gr: float,
    production: float,
    shipment: float,
    open_deployment_out: float,
    open_deployment_in: float,
    ao_demand: float,
    forecast_demand: float,
    safety_stock: float,
    downstream_ao_gap: float = 0.0,
    downstream_fc_gap: float = 0.0,
    downstream_ss_gap: float = 0.0,
) -> Tuple[float, float, float]:
    """
    计算单个节点的净需求（DuckDB 优化版本）
    
    这是一个向量化友好的接口，可以批量处理多个节点。
    """
    # 计算总供给
    total_supply = (
        beginning_inventory + in_transit + delivery_gr + 
        production + open_deployment_in - 
        shipment - open_deployment_out
    )
    
    # 总需求 = 本地需求 + 下游缺口
    total_ao = ao_demand + downstream_ao_gap
    total_fc = forecast_demand + downstream_fc_gap
    total_ss = safety_stock + downstream_ss_gap
    
    # 按优先级计算缺口（AO > Forecast > Safety Stock）
    remaining = total_supply
    
    # AO gap
    if total_ao > 0:
        ao_gap = max(0, total_ao - remaining)
        remaining = max(0, remaining - total_ao)
    else:
        ao_gap = 0
    
    # Forecast gap
    if total_fc > 0:
        fc_gap = max(0, total_fc - remaining)
        remaining = max(0, remaining - total_fc)
    else:
        fc_gap = 0
    
    # Safety Stock gap
    if total_ss > 0:
        ss_gap = max(0, total_ss - remaining)
    else:
        ss_gap = 0
    
    return ao_gap, fc_gap, ss_gap


def calculate_net_demand_batch_duckdb(
    nodes_df: pd.DataFrame,
    run_id: Optional[str] = None
) -> pd.DataFrame:
    """
    批量计算净需求（DuckDB SQL 向量化实现）
    
    Args:
        nodes_df: 节点数据，包含以下列:
            - material, location
            - beginning_inventory, in_transit, delivery_gr
            - production, shipment
            - open_deployment_out, open_deployment_in
            - ao_demand, forecast_demand, safety_stock
            - downstream_ao_gap, downstream_fc_gap, downstream_ss_gap
        run_id: 运行ID（用于性能统计）
    
    Returns:
        包含 ao_gap, fc_gap, ss_gap 的 DataFrame
    """
    calculator = get_duckdb_calculator()
    
    if calculator is None or len(nodes_df) < DuckDBConfig.min_rows_threshold:
        # 使用 Pandas 向量化计算
        return _calculate_net_demand_pandas(nodes_df, run_id)
    
    t0 = time.perf_counter()
    
    try:
        # 注册数据到 DuckDB
        calculator.conn.register('nodes', nodes_df)
        
        result = calculator.conn.execute("""
            WITH supply_calc AS (
                SELECT 
                    material,
                    location,
                    COALESCE(beginning_inventory, 0) + 
                    COALESCE(in_transit, 0) + 
                    COALESCE(delivery_gr, 0) + 
                    COALESCE(production, 0) + 
                    COALESCE(open_deployment_in, 0) - 
                    COALESCE(shipment, 0) - 
                    COALESCE(open_deployment_out, 0) as total_supply,
                    COALESCE(ao_demand, 0) + COALESCE(downstream_ao_gap, 0) as total_ao,
                    COALESCE(forecast_demand, 0) + COALESCE(downstream_fc_gap, 0) as total_fc,
                    COALESCE(safety_stock, 0) + COALESCE(downstream_ss_gap, 0) as total_ss
                FROM nodes
            ),
            gap_calc AS (
                SELECT 
                    material,
                    location,
                    total_supply,
                    total_ao,
                    total_fc,
                    total_ss,
                    -- AO gap
                    GREATEST(0, total_ao - total_supply) as ao_gap,
                    -- Remaining after AO
                    GREATEST(0, total_supply - total_ao) as remaining_after_ao
                FROM supply_calc
            )
            SELECT 
                material,
                location,
                ao_gap,
                GREATEST(0, total_fc - remaining_after_ao) as fc_gap,
                GREATEST(0, total_ss - GREATEST(0, remaining_after_ao - total_fc)) as ss_gap
            FROM gap_calc
        """).fetchdf()
        
        calculator.conn.unregister('nodes')
        
        elapsed_ms = (time.perf_counter() - t0) * 1000
        
        if run_id and DuckDBConfig.collect_stats:
            _perf_stats.record(run_id, 'net_demand', 'duckdb', 
                             len(nodes_df), elapsed_ms)
        
        return result
        
    except Exception as e:
        if DuckDBConfig.fallback_on_error:
            print(f"[DuckDB] 净需求计算出错，回退到Pandas: {e}")
            return _calculate_net_demand_pandas(nodes_df, run_id)
        raise


def _calculate_net_demand_pandas(
    nodes_df: pd.DataFrame,
    run_id: Optional[str] = None
) -> pd.DataFrame:
    """Pandas 向量化实现（作为回退方案）"""
    t0 = time.perf_counter()
    
    df = nodes_df.copy()
    
    # 填充默认值
    for col in ['beginning_inventory', 'in_transit', 'delivery_gr', 
                'production', 'shipment', 'open_deployment_out', 
                'open_deployment_in', 'ao_demand', 'forecast_demand', 
                'safety_stock', 'downstream_ao_gap', 'downstream_fc_gap',
                'downstream_ss_gap']:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0)
    
    # 计算总供给
    df['total_supply'] = (
        df['beginning_inventory'] + df['in_transit'] + df['delivery_gr'] +
        df['production'] + df['open_deployment_in'] -
        df['shipment'] - df['open_deployment_out']
    )
    
    # 计算总需求
    df['total_ao'] = df['ao_demand'] + df['downstream_ao_gap']
    df['total_fc'] = df['forecast_demand'] + df['downstream_fc_gap']
    df['total_ss'] = df['safety_stock'] + df['downstream_ss_gap']
    
    # 计算缺口
    df['ao_gap'] = np.maximum(0, df['total_ao'] - df['total_supply'])
    df['remaining'] = np.maximum(0, df['total_supply'] - df['total_ao'])
    df['fc_gap'] = np.maximum(0, df['total_fc'] - df['remaining'])
    df['remaining2'] = np.maximum(0, df['remaining'] - df['total_fc'])
    df['ss_gap'] = np.maximum(0, df['total_ss'] - df['remaining2'])
    
    result = df[['material', 'location', 'ao_gap', 'fc_gap', 'ss_gap']].copy()
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    if run_id and DuckDBConfig.collect_stats:
        _perf_stats.record(run_id, 'net_demand', 'pandas', 
                         len(nodes_df), elapsed_ms)
    
    return result


# ============================================================================
# Module5 MOQ/RV 处理 - DuckDB 集成
# ============================================================================

def apply_moq_rv_batch_duckdb(
    demand_df: pd.DataFrame,
    config_df: pd.DataFrame,
    run_id: Optional[str] = None
) -> pd.DataFrame:
    """
    批量应用 MOQ/RV 约束（DuckDB SQL 向量化实现）
    
    Args:
        demand_df: 需求数据，包含:
            - material, sending, receiving
            - quantity (原始需求量)
        config_df: MOQ/RV 配置，包含:
            - material, sending
            - moq, rv
        run_id: 运行ID（用于性能统计）
    
    Returns:
        包含 adjusted_qty 的 DataFrame
    """
    calculator = get_duckdb_calculator()
    
    if calculator is None or len(demand_df) < DuckDBConfig.min_rows_threshold:
        return _apply_moq_rv_pandas(demand_df, config_df, run_id)
    
    t0 = time.perf_counter()
    
    try:
        calculator.conn.register('demand', demand_df)
        calculator.conn.register('config', config_df)
        
        result = calculator.conn.execute("""
            SELECT 
                d.*,
                COALESCE(c.moq, 1) as moq,
                COALESCE(c.rv, 1) as rv,
                CASE 
                    -- 数量为0或负数
                    WHEN d.quantity <= 0 THEN 0
                    -- 自循环不应用MOQ/RV
                    WHEN d.sending = d.receiving THEN CAST(d.quantity AS INTEGER)
                    -- 小于MOQ时使用MOQ
                    WHEN d.quantity < COALESCE(c.moq, 1) THEN COALESCE(c.moq, 1)
                    -- 否则向上取整到RV的倍数
                    ELSE CAST(
                        CEIL(d.quantity::DOUBLE / COALESCE(c.rv, 1)) 
                        * COALESCE(c.rv, 1) AS INTEGER
                    )
                END as adjusted_qty
            FROM demand d
            LEFT JOIN config c 
                ON d.material = c.material AND d.sending = c.sending
        """).fetchdf()
        
        calculator.conn.unregister('demand')
        calculator.conn.unregister('config')
        
        elapsed_ms = (time.perf_counter() - t0) * 1000
        
        if run_id and DuckDBConfig.collect_stats:
            _perf_stats.record(run_id, 'moq_rv', 'duckdb', 
                             len(demand_df), elapsed_ms)
        
        return result
        
    except Exception as e:
        if DuckDBConfig.fallback_on_error:
            print(f"[DuckDB] MOQ/RV计算出错，回退到Pandas: {e}")
            return _apply_moq_rv_pandas(demand_df, config_df, run_id)
        raise


def _apply_moq_rv_pandas(
    demand_df: pd.DataFrame,
    config_df: pd.DataFrame,
    run_id: Optional[str] = None
) -> pd.DataFrame:
    """Pandas 向量化实现（作为回退方案）"""
    t0 = time.perf_counter()
    
    df = demand_df.copy()
    
    # 合并配置
    if not config_df.empty:
        df = df.merge(
            config_df[['material', 'sending', 'moq', 'rv']],
            on=['material', 'sending'],
            how='left'
        )
    else:
        df['moq'] = 1
        df['rv'] = 1
    
    df['moq'] = df['moq'].fillna(1).astype(int)
    df['rv'] = df['rv'].fillna(1).astype(int)
    
    # 向量化计算调整后数量
    is_self_loop = df['sending'] == df['receiving']
    qty_positive = df['quantity'] > 0
    below_moq = df['quantity'] < df['moq']
    
    df['adjusted_qty'] = np.where(
        ~qty_positive, 0,
        np.where(
            is_self_loop, df['quantity'].astype(int),
            np.where(
                below_moq, df['moq'],
                (np.ceil(df['quantity'] / df['rv']) * df['rv']).astype(int)
            )
        )
    )
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    if run_id and DuckDBConfig.collect_stats:
        _perf_stats.record(run_id, 'moq_rv', 'pandas', 
                         len(demand_df), elapsed_ms)
    
    return df


# ============================================================================
# Module5 优先级分配 - DuckDB 集成
# ============================================================================

def priority_allocation_batch_duckdb(
    demand_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    priority_df: pd.DataFrame,
    run_id: Optional[str] = None
) -> pd.DataFrame:
    """
    批量优先级分配（DuckDB SQL 向量化实现）
    
    Args:
        demand_df: 需求数据，包含:
            - material, sending, receiving
            - demand_element (需求类型)
            - demand_qty (需求量)
        inventory_df: 库存数据，包含:
            - material, location
            - qty (可用库存)
        priority_df: 优先级配置，包含:
            - demand_element
            - priority (数字越小优先级越高)
        run_id: 运行ID（用于性能统计）
    
    Returns:
        包含 allocated_qty, unmet_qty 的 DataFrame
    """
    calculator = get_duckdb_calculator()
    
    if calculator is None or len(demand_df) < DuckDBConfig.min_rows_threshold:
        return _priority_allocation_pandas(demand_df, inventory_df, priority_df, run_id)
    
    t0 = time.perf_counter()
    
    try:
        calculator.conn.register('demand', demand_df)
        calculator.conn.register('inventory', inventory_df)
        calculator.conn.register('priority', priority_df)
        
        result = calculator.conn.execute("""
            WITH ranked AS (
                SELECT 
                    d.*,
                    COALESCE(p.priority, 999) as priority_rank,
                    ROW_NUMBER() OVER (
                        PARTITION BY d.material, d.sending
                        ORDER BY COALESCE(p.priority, 999), d.demand_qty DESC
                    ) as alloc_order
                FROM demand d
                LEFT JOIN priority p ON d.demand_element = p.demand_element
            ),
            with_inv AS (
                SELECT 
                    r.*,
                    COALESCE(i.qty, 0) as available_qty
                FROM ranked r
                LEFT JOIN inventory i 
                    ON r.material = i.material AND r.sending = i.location
            ),
            cumulative AS (
                SELECT 
                    *,
                    SUM(demand_qty) OVER (
                        PARTITION BY material, sending
                        ORDER BY alloc_order
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) as cum_demand
                FROM with_inv
            )
            SELECT 
                material, sending, receiving, demand_element,
                demand_qty, priority_rank, available_qty, cum_demand,
                GREATEST(0, LEAST(
                    demand_qty,
                    available_qty - (cum_demand - demand_qty)
                ))::INTEGER as allocated_qty,
                (demand_qty - GREATEST(0, LEAST(
                    demand_qty,
                    available_qty - (cum_demand - demand_qty)
                )))::INTEGER as unmet_qty
            FROM cumulative
            ORDER BY material, sending, alloc_order
        """).fetchdf()
        
        for tbl in ['demand', 'inventory', 'priority']:
            calculator.conn.unregister(tbl)
        
        elapsed_ms = (time.perf_counter() - t0) * 1000
        
        if run_id and DuckDBConfig.collect_stats:
            _perf_stats.record(run_id, 'priority_allocation', 'duckdb', 
                             len(demand_df), elapsed_ms)
        
        return result
        
    except Exception as e:
        if DuckDBConfig.fallback_on_error:
            print(f"[DuckDB] 优先级分配出错，回退到Pandas: {e}")
            return _priority_allocation_pandas(demand_df, inventory_df, priority_df, run_id)
        raise


def _priority_allocation_pandas(
    demand_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    priority_df: pd.DataFrame,
    run_id: Optional[str] = None
) -> pd.DataFrame:
    """Pandas 向量化实现（作为回退方案）"""
    t0 = time.perf_counter()
    
    df = demand_df.copy()
    
    # 合并优先级
    if not priority_df.empty:
        df = df.merge(
            priority_df[['demand_element', 'priority']],
            on='demand_element',
            how='left'
        )
    df['priority_rank'] = df.get('priority', 999).fillna(999)
    
    # 合并库存
    if not inventory_df.empty:
        inv = inventory_df.groupby(['material', 'location'])['qty'].sum().reset_index()
        df = df.merge(
            inv.rename(columns={'location': 'sending', 'qty': 'available_qty'}),
            on=['material', 'sending'],
            how='left'
        )
    df['available_qty'] = df.get('available_qty', 0).fillna(0)
    
    # 排序并计算累积需求
    df = df.sort_values(
        ['material', 'sending', 'priority_rank', 'demand_qty'],
        ascending=[True, True, True, False]
    )
    df['cum_demand'] = df.groupby(['material', 'sending'])['demand_qty'].cumsum()
    
    # 计算分配量
    df['allocated_qty'] = np.maximum(0, np.minimum(
        df['demand_qty'],
        df['available_qty'] - (df['cum_demand'] - df['demand_qty'])
    )).astype(int)
    df['unmet_qty'] = (df['demand_qty'] - df['allocated_qty']).astype(int)
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    if run_id and DuckDBConfig.collect_stats:
        _perf_stats.record(run_id, 'priority_allocation', 'pandas', 
                         len(demand_df), elapsed_ms)
    
    return df


# ============================================================================
# 便捷装饰器 - 自动选择最优实现
# ============================================================================

def with_duckdb_fallback(operation_name: str):
    """
    装饰器：自动选择 DuckDB 或 Pandas 实现
    
    Usage:
        @with_duckdb_fallback('net_demand')
        def calculate_net_demand(...):
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            run_id = kwargs.pop('run_id', None)
            use_duckdb = kwargs.pop('use_duckdb', DuckDBConfig.enabled)
            
            t0 = time.perf_counter()
            method = 'pandas'
            
            if use_duckdb and DUCKDB_AVAILABLE:
                try:
                    result = func(*args, **kwargs, _use_duckdb=True)
                    method = 'duckdb'
                except Exception as e:
                    if DuckDBConfig.fallback_on_error:
                        result = func(*args, **kwargs, _use_duckdb=False)
                    else:
                        raise
            else:
                result = func(*args, **kwargs, _use_duckdb=False)
            
            elapsed_ms = (time.perf_counter() - t0) * 1000
            
            if run_id and DuckDBConfig.collect_stats:
                rows = len(result) if isinstance(result, pd.DataFrame) else 1
                _perf_stats.record(run_id, operation_name, method, rows, elapsed_ms)
            
            return result
        
        return wrapper
    return decorator


# ============================================================================
# 性能对比测试接口
# ============================================================================

@contextmanager
def performance_comparison(name: str = "default"):
    """
    性能对比上下文管理器
    
    Usage:
        with performance_comparison("my_test") as run_id:
            # 运行一些计算
            result = calculate_something(run_id=run_id)
        
        comparison = get_perf_stats().get_comparison(run_id)
    """
    run_id = f"{name}_{int(time.time()*1000)}"
    _perf_stats.start_run(run_id)
    
    try:
        yield run_id
    finally:
        stats = _perf_stats.end_run(run_id)
        print(f"\n📊 性能统计 [{name}]:")
        for op, data in stats.items():
            avg_ms = data['total_ms'] / max(data['count'], 1)
            print(f"  {op}: {data['count']}次, 总计{data['total_ms']:.1f}ms, "
                  f"平均{avg_ms:.2f}ms, {data['total_rows']:,}行")


def run_ab_comparison(
    func_a: Callable,
    func_b: Callable,
    test_data: Any,
    iterations: int = 5,
    warmup: int = 1
) -> Dict[str, Any]:
    """
    运行 A/B 性能对比测试
    
    Args:
        func_a: 方案A（通常是 DuckDB）
        func_b: 方案B（通常是 Pandas）
        test_data: 测试数据
        iterations: 迭代次数
        warmup: 预热次数
    
    Returns:
        对比结果
    """
    results = {'a': [], 'b': []}
    
    # 预热
    for _ in range(warmup):
        func_a(test_data)
        func_b(test_data)
    
    # 正式测试
    for i in range(iterations):
        # 方案A
        t0 = time.perf_counter()
        result_a = func_a(test_data)
        results['a'].append((time.perf_counter() - t0) * 1000)
        
        # 方案B
        t0 = time.perf_counter()
        result_b = func_b(test_data)
        results['b'].append((time.perf_counter() - t0) * 1000)
    
    # 统计
    import statistics
    
    return {
        'a': {
            'mean_ms': statistics.mean(results['a']),
            'std_ms': statistics.stdev(results['a']) if len(results['a']) > 1 else 0,
            'min_ms': min(results['a']),
            'max_ms': max(results['a']),
        },
        'b': {
            'mean_ms': statistics.mean(results['b']),
            'std_ms': statistics.stdev(results['b']) if len(results['b']) > 1 else 0,
            'min_ms': min(results['b']),
            'max_ms': max(results['b']),
        },
        'speedup': statistics.mean(results['b']) / max(statistics.mean(results['a']), 0.001)
    }


# ============================================================================
# 模块清理
# ============================================================================

import atexit

@atexit.register
def _cleanup():
    """程序退出时清理资源"""
    close_duckdb_calculator()
