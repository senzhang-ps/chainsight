# -*- coding: utf-8 -*-
"""
Module5 DuckDB 批量计算优化模块

提供使用 DuckDB 向量化 SQL 进行批量处理的优化实现:
1. MOQ/RV 批量应用
2. 优先级分配批量计算
"""

import time
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np

# 尝试导入 DuckDB 集成模块
try:
    import sys
    import os
    # 添加 pgsql_db 到路径
    pgsql_db_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'pgsql_db')
    if pgsql_db_path not in sys.path:
        sys.path.insert(0, pgsql_db_path)
    
    from duckdb_integration import (
        get_duckdb_calculator,
        DuckDBConfig,
        get_perf_stats,
    )
    DUCKDB_INTEGRATION_AVAILABLE = True
except ImportError:
    DUCKDB_INTEGRATION_AVAILABLE = False


# ============================================================================
# MOQ/RV 批量应用 - DuckDB 优化版
# ============================================================================

def apply_moq_rv_batch_duckdb(
    demand_rows: List[dict],
    moq_rv_config: pd.DataFrame,
    location: str,
    run_id: Optional[str] = None,
) -> Dict[int, int]:
    """
    使用 DuckDB 批量应用 MOQ/RV 约束
    
    这是 deployment_planning/allocation.py 中 apply_grouped_moq_rv 的优化版本。
    
    参数：
        demand_rows: 需求行列表，每行包含:
            - material: 物料
            - from_location / receiving: 接收位置
            - demand_qty: 需求量
            - moq: 最小订货量（可选）
            - rv: 重订量（可选）
        moq_rv_config: MOQ/RV 配置 DataFrame，包含:
            - material, sending, moq, rv
        location: 当前发送位置（sending）
        run_id: 运行ID（用于性能统计）
    
    返回：
        {index: adjusted_qty} 字典
    """
    if not demand_rows:
        return {}
    
    if not DUCKDB_INTEGRATION_AVAILABLE or not DuckDBConfig.enabled:
        # 回退到原始实现
        from .allocation import apply_grouped_moq_rv
        return apply_grouped_moq_rv(demand_rows, location)
    
    calculator = get_duckdb_calculator()
    if calculator is None or len(demand_rows) < DuckDBConfig.min_rows_threshold:
        from .allocation import apply_grouped_moq_rv
        return apply_grouped_moq_rv(demand_rows, location)
    
    t0 = time.perf_counter()
    
    try:
        # 构建需求 DataFrame
        demand_df = pd.DataFrame([
            {
                'idx': i,
                'material': str(d['material']),
                'receiving': str(d.get('from_location', d.get('receiving', location))),
                'demand_qty': max(0, int(d.get('demand_qty', 0) or 0)),
                'row_moq': int(d.get('moq', 0) or 0),
                'row_rv': int(d.get('rv', 0) or 0),
            }
            for i, d in enumerate(demand_rows)
        ])
        demand_df['sending'] = location
        demand_df['is_cross_node'] = demand_df['sending'] != demand_df['receiving']
        
        # 准备配置
        if moq_rv_config is not None and not moq_rv_config.empty:
            config_df = moq_rv_config[['material', 'sending', 'moq', 'rv']].copy()
            config_df['material'] = config_df['material'].astype(str)
            config_df['sending'] = config_df['sending'].astype(str)
        else:
            config_df = pd.DataFrame({'material': [], 'sending': [], 'moq': [], 'rv': []})
        
        # 注册到 DuckDB
        calculator.conn.register('demand', demand_df)
        calculator.conn.register('config', config_df)
        
        # 执行计算
        # 步骤1：按路径分组计算组级 MOQ/RV 调整
        result = calculator.conn.execute("""
            WITH route_groups AS (
                -- 按路径分组
                SELECT 
                    material,
                    sending,
                    receiving,
                    -- 组级MOQ/RV取各行最大值
                    MAX(GREATEST(row_moq, COALESCE(c.moq, 1))) as group_moq,
                    MAX(GREATEST(row_rv, COALESCE(c.rv, 1))) as group_rv,
                    SUM(demand_qty) as group_total_qty,
                    BOOL_AND(is_cross_node) as is_cross_node,
                    ARRAY_AGG(idx ORDER BY idx) as indices,
                    ARRAY_AGG(demand_qty ORDER BY idx) as qtys
                FROM demand d
                LEFT JOIN config c ON d.material = c.material AND d.sending = c.sending
                GROUP BY d.material, d.sending, d.receiving
            ),
            adjusted_groups AS (
                -- 计算组级调整后总量
                SELECT 
                    *,
                    CASE 
                        WHEN group_total_qty <= 0 THEN 0
                        WHEN NOT is_cross_node THEN group_total_qty
                        WHEN group_total_qty < group_moq THEN group_moq
                        ELSE CAST(CEIL(group_total_qty::DOUBLE / group_rv) * group_rv AS INTEGER)
                    END as adjusted_total
                FROM route_groups
            )
            SELECT * FROM adjusted_groups
        """).fetchdf()
        
        calculator.conn.unregister('demand')
        calculator.conn.unregister('config')
        
        # 分配调整量到各行（最大余数法）
        adjusted_qtys = {}
        
        for _, row in result.iterrows():
            indices = row['indices']
            qtys = row['qtys']
            total_qty = int(row['group_total_qty'])
            adjusted_total = int(row['adjusted_total'])
            
            if total_qty <= 0:
                # 总量为0时，第一个得到adjusted_total，其余为0
                if adjusted_total > 0 and indices:
                    adjusted_qtys[indices[0]] = adjusted_total
                    for idx in indices[1:]:
                        adjusted_qtys[idx] = 0
                else:
                    for idx in indices:
                        adjusted_qtys[idx] = 0
                continue
            
            # 最大余数法分配
            r = adjusted_total / float(total_qty)
            floors = []
            for i, (idx, qty) in enumerate(zip(indices, qtys)):
                exact = qty * r
                floor_val = int(np.floor(exact))
                remainder = float(exact - floor_val)
                floors.append((idx, floor_val, remainder, qty, i))
            
            total_floor = sum(x[1] for x in floors)
            remaining = max(0, adjusted_total - total_floor)
            
            # 按余数降序排序
            floors.sort(key=lambda x: (-x[2], -x[3], x[4]))
            
            # 分配floor值
            for idx, floor_val, _, _, _ in floors:
                adjusted_qtys[idx] = floor_val
            
            # 分配剩余量
            for k in range(min(remaining, len(floors))):
                idx = floors[k][0]
                adjusted_qtys[idx] += 1
        
        elapsed_ms = (time.perf_counter() - t0) * 1000
        
        if run_id and DuckDBConfig.collect_stats:
            get_perf_stats().record(run_id, 'moq_rv_batch', 'duckdb', 
                                   len(demand_rows), elapsed_ms)
        
        return adjusted_qtys
        
    except Exception as e:
        print(f"[M5-DuckDB] MOQ/RV计算出错，回退到Pandas: {e}")
        if DuckDBConfig.fallback_on_error:
            from .allocation import apply_grouped_moq_rv
            return apply_grouped_moq_rv(demand_rows, location)
        raise


# ============================================================================
# 优先级分配 - DuckDB 优化版
# ============================================================================

def apply_priority_allocation_duckdb(
    demand_rows: List[dict],
    adjusted_qtys: Dict[int, int],
    current_stock: int,
    demand_priority_map: Dict[str, int],
    run_id: Optional[str] = None,
) -> int:
    """
    使用 DuckDB 进行优先级分配
    
    这是 deployment_planning/allocation.py 中 apply_priority_allocation_vectorized 的优化版本。
    
    参数：
        demand_rows: 需求行列表
        adjusted_qtys: 调整后的需求量字典 {index: adjusted_qty}
        current_stock: 当前可用库存
        demand_priority_map: 需求类型优先级映射 {demand_element: priority}
        run_id: 运行ID（用于性能统计）
    
    返回：
        剩余库存量
    """
    if not demand_rows:
        return current_stock
    
    if not DUCKDB_INTEGRATION_AVAILABLE or not DuckDBConfig.enabled:
        from .allocation import apply_priority_allocation_vectorized
        return apply_priority_allocation_vectorized(
            demand_rows, adjusted_qtys, current_stock, demand_priority_map
        )
    
    calculator = get_duckdb_calculator()
    if calculator is None or len(demand_rows) < DuckDBConfig.min_rows_threshold:
        from .allocation import apply_priority_allocation_vectorized
        return apply_priority_allocation_vectorized(
            demand_rows, adjusted_qtys, current_stock, demand_priority_map
        )
    
    t0 = time.perf_counter()
    
    try:
        # 构建 DataFrame
        demand_df = pd.DataFrame([
            {
                'idx': i,
                'demand_element': str(d.get('demand_element', '')),
                'demand_qty': int(d.get('demand_qty', 0) or 0),
                'adjusted_qty': int(adjusted_qtys.get(i, d.get('demand_qty', 0))),
            }
            for i, d in enumerate(demand_rows)
        ])
        
        # 构建优先级 DataFrame
        priority_df = pd.DataFrame([
            {'demand_element': str(k), 'priority': int(v)}
            for k, v in demand_priority_map.items()
        ])
        
        # 注册到 DuckDB
        calculator.conn.register('demand', demand_df)
        calculator.conn.register('priority', priority_df)
        
        # 执行优先级分配计算
        result = calculator.conn.execute(f"""
            WITH ranked AS (
                SELECT 
                    d.idx,
                    d.demand_element,
                    d.demand_qty,
                    d.adjusted_qty,
                    COALESCE(p.priority, 99) as priority_rank,
                    ROW_NUMBER() OVER (ORDER BY COALESCE(p.priority, 99), d.adjusted_qty DESC) as alloc_order
                FROM demand d
                LEFT JOIN priority p ON d.demand_element = p.demand_element
            ),
            cumulative AS (
                SELECT 
                    *,
                    SUM(adjusted_qty) OVER (
                        ORDER BY alloc_order
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) as cum_demand,
                    {current_stock} as available_stock
                FROM ranked
            )
            SELECT 
                idx,
                priority_rank,
                adjusted_qty,
                cum_demand,
                GREATEST(0, LEAST(
                    adjusted_qty,
                    available_stock - (cum_demand - adjusted_qty)
                ))::INTEGER as allocated_qty
            FROM cumulative
            ORDER BY idx
        """).fetchdf()
        
        calculator.conn.unregister('demand')
        calculator.conn.unregister('priority')
        
        # 写回 demand_rows
        total_allocated = 0
        for _, row in result.iterrows():
            idx = int(row['idx'])
            allocated = int(row['allocated_qty'])
            demand_rows[idx]['deployed_qty_invCon'] = allocated
            total_allocated += allocated
        
        remaining_stock = max(0, current_stock - total_allocated)
        
        elapsed_ms = (time.perf_counter() - t0) * 1000
        
        if run_id and DuckDBConfig.collect_stats:
            get_perf_stats().record(run_id, 'priority_allocation', 'duckdb', 
                                   len(demand_rows), elapsed_ms)
        
        return remaining_stock
        
    except Exception as e:
        print(f"[M5-DuckDB] 优先级分配出错，回退到Pandas: {e}")
        if DuckDBConfig.fallback_on_error:
            from .allocation import apply_priority_allocation_vectorized
            return apply_priority_allocation_vectorized(
                demand_rows, adjusted_qtys, current_stock, demand_priority_map
            )
        raise


# ============================================================================
# 批量库存分配 - DuckDB 优化版
# ============================================================================

def batch_inventory_allocation_duckdb(
    nodes_data: List[Dict[str, Any]],
    run_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    批量进行多节点库存分配（DuckDB 向量化实现）
    
    当有多个节点需要进行库存分配时，使用此函数可以显著提升性能。
    
    参数：
        nodes_data: 节点数据列表，每个元素包含:
            - material: 物料
            - location: 位置
            - demand_rows: 需求行列表
            - current_stock: 当前库存
            - demand_priority_map: 优先级映射
            - moq_rv_config: MOQ/RV 配置（可选）
        run_id: 运行ID
    
    返回：
        分配结果列表
    """
    if not nodes_data:
        return []
    
    if not DUCKDB_INTEGRATION_AVAILABLE or not DuckDBConfig.enabled:
        return _batch_allocation_pandas(nodes_data, run_id)
    
    calculator = get_duckdb_calculator()
    if calculator is None or len(nodes_data) < DuckDBConfig.min_rows_threshold:
        return _batch_allocation_pandas(nodes_data, run_id)
    
    t0 = time.perf_counter()
    
    try:
        # 展平所有需求行
        all_demands = []
        for node_idx, node in enumerate(nodes_data):
            for row_idx, row in enumerate(node.get('demand_rows', [])):
                all_demands.append({
                    'node_idx': node_idx,
                    'row_idx': row_idx,
                    'material': str(node['material']),
                    'location': str(node['location']),
                    'demand_element': str(row.get('demand_element', '')),
                    'demand_qty': int(row.get('demand_qty', 0) or 0),
                    'receiving': str(row.get('from_location', row.get('receiving', node['location']))),
                    'moq': int(row.get('moq', 1) or 1),
                    'rv': int(row.get('rv', 1) or 1),
                })
        
        if not all_demands:
            return nodes_data
        
        demands_df = pd.DataFrame(all_demands)
        
        # 节点库存
        inventory_df = pd.DataFrame([
            {
                'node_idx': i,
                'material': str(n['material']),
                'location': str(n['location']),
                'current_stock': int(n.get('current_stock', 0)),
            }
            for i, n in enumerate(nodes_data)
        ])
        
        # 收集所有优先级映射
        all_priorities = {}
        for node in nodes_data:
            all_priorities.update(node.get('demand_priority_map', {}))
        
        priority_df = pd.DataFrame([
            {'demand_element': str(k), 'priority': int(v)}
            for k, v in all_priorities.items()
        ])
        
        # 注册到 DuckDB
        calculator.conn.register('demands', demands_df)
        calculator.conn.register('inventory', inventory_df)
        calculator.conn.register('priority', priority_df)
        
        # 执行批量分配
        result = calculator.conn.execute("""
            WITH with_priority AS (
                SELECT 
                    d.*,
                    COALESCE(p.priority, 99) as priority_rank,
                    i.current_stock
                FROM demands d
                LEFT JOIN priority p ON d.demand_element = p.demand_element
                LEFT JOIN inventory i ON d.node_idx = i.node_idx
            ),
            ranked AS (
                SELECT 
                    *,
                    ROW_NUMBER() OVER (
                        PARTITION BY node_idx
                        ORDER BY priority_rank, demand_qty DESC
                    ) as alloc_order
                FROM with_priority
            ),
            with_moq_rv AS (
                SELECT 
                    *,
                    -- 简化的MOQ/RV调整（单行级别）
                    CASE 
                        WHEN demand_qty <= 0 THEN 0
                        WHEN location = receiving THEN demand_qty
                        WHEN demand_qty < moq THEN moq
                        ELSE CAST(CEIL(demand_qty::DOUBLE / rv) * rv AS INTEGER)
                    END as adjusted_qty
                FROM ranked
            ),
            cumulative AS (
                SELECT 
                    *,
                    SUM(adjusted_qty) OVER (
                        PARTITION BY node_idx
                        ORDER BY alloc_order
                        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                    ) as cum_demand
                FROM with_moq_rv
            )
            SELECT 
                node_idx,
                row_idx,
                priority_rank,
                demand_qty,
                adjusted_qty,
                cum_demand,
                current_stock,
                GREATEST(0, LEAST(
                    adjusted_qty,
                    current_stock - (cum_demand - adjusted_qty)
                ))::INTEGER as allocated_qty
            FROM cumulative
            ORDER BY node_idx, row_idx
        """).fetchdf()
        
        # 清理
        calculator.conn.unregister('demands')
        calculator.conn.unregister('inventory')
        calculator.conn.unregister('priority')
        
        # 写回结果
        for _, row in result.iterrows():
            node_idx = int(row['node_idx'])
            row_idx = int(row['row_idx'])
            allocated = int(row['allocated_qty'])
            adjusted = int(row['adjusted_qty'])
            
            if node_idx < len(nodes_data) and row_idx < len(nodes_data[node_idx].get('demand_rows', [])):
                nodes_data[node_idx]['demand_rows'][row_idx]['deployed_qty_invCon'] = allocated
                nodes_data[node_idx]['demand_rows'][row_idx]['adjusted_qty'] = adjusted
        
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"[M5-DuckDB] 批量分配 {len(nodes_data)} 节点, {len(all_demands)} 需求行: {elapsed_ms:.1f}ms")
        
        if run_id and DuckDBConfig.collect_stats:
            get_perf_stats().record(run_id, 'batch_allocation', 'duckdb', 
                                   len(all_demands), elapsed_ms)
        
        return nodes_data
        
    except Exception as e:
        print(f"[M5-DuckDB] 批量分配出错，回退到Pandas: {e}")
        if DuckDBConfig.fallback_on_error:
            return _batch_allocation_pandas(nodes_data, run_id)
        raise


def _batch_allocation_pandas(
    nodes_data: List[Dict[str, Any]],
    run_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Pandas 批量分配（回退方案）"""
    from .allocation import apply_grouped_moq_rv, apply_priority_allocation_vectorized
    
    t0 = time.perf_counter()
    
    for node in nodes_data:
        demand_rows = node.get('demand_rows', [])
        if not demand_rows:
            continue
        
        location = node['location']
        current_stock = node.get('current_stock', 0)
        priority_map = node.get('demand_priority_map', {})
        
        # 应用 MOQ/RV
        adjusted_qtys = apply_grouped_moq_rv(demand_rows, location)
        
        # 应用优先级分配
        apply_priority_allocation_vectorized(
            demand_rows, adjusted_qtys, current_stock, priority_map
        )
    
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    if run_id and DUCKDB_INTEGRATION_AVAILABLE and DuckDBConfig.collect_stats:
        total_rows = sum(len(n.get('demand_rows', [])) for n in nodes_data)
        get_perf_stats().record(run_id, 'batch_allocation', 'pandas', 
                               total_rows, elapsed_ms)
    
    return nodes_data


# ============================================================================
# 工具函数
# ============================================================================

def is_duckdb_available() -> bool:
    """检查 DuckDB 集成是否可用"""
    return DUCKDB_INTEGRATION_AVAILABLE and DuckDBConfig.enabled


def set_duckdb_enabled(enabled: bool):
    """设置 DuckDB 是否启用"""
    if DUCKDB_INTEGRATION_AVAILABLE:
        DuckDBConfig.enabled = enabled


def get_duckdb_config() -> Dict[str, Any]:
    """获取当前 DuckDB 配置"""
    if not DUCKDB_INTEGRATION_AVAILABLE:
        return {'available': False}
    
    return {
        'available': True,
        'enabled': DuckDBConfig.enabled,
        'memory_limit': DuckDBConfig.memory_limit,
        'threads': DuckDBConfig.threads,
        'min_rows_threshold': DuckDBConfig.min_rows_threshold,
        'fallback_on_error': DuckDBConfig.fallback_on_error,
    }
