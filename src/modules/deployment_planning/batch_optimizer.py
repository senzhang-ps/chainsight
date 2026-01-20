# -*- coding: utf-8 -*-
"""
M5 批量优化器模块

使用 DuckDB 对 M5 (Module5) 的层级处理进行批量预过滤优化。
核心思想：在每层开始处理前，使用 DuckDB 的 C++ 引擎一次性过滤所有节点的数据，
避免在 Python 中逐个节点进行 DataFrame 过滤。
"""

import time
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
from ..utils.duckdb_accelerator import get_accelerator


def batch_prefilter_layer_data(
    all_pairs: Set[Tuple[str, str]],
    sim_date: pd.Timestamp,
    horizon_days: int,
    supply_demand_log: pd.DataFrame,
    safety_stock: pd.DataFrame,
    order_log: pd.DataFrame,
) -> Dict[str, Dict[Tuple[str, str], pd.DataFrame]]:
    """
    批量预过滤层内所有节点的数据。
    
    使用 DuckDB 一次性过滤所有 (material, location) 对的数据，
    减少 Python 层面的 DataFrame 过滤操作。
    
    Args:
        all_pairs: (material, location) 对集合
        sim_date: 仿真日期
        horizon_days: 展望天数
        supply_demand_log: 供需日志 DataFrame
        safety_stock: 安全库存 DataFrame
        order_log: 订单日志 DataFrame
        
    Returns:
        dict: {
            'sdl': {(mat, loc): df, ...},
            'ss': {(mat, loc): df, ...},
            'order': {(mat, loc): df, ...}
        }
    """
    t_start = time.perf_counter()
    
    accel = get_accelerator()
    pairs_list = list(all_pairs)
    horizon_end = sim_date + pd.Timedelta(days=horizon_days)
    
    result = {
        'sdl': {},
        'ss': {},
        'order': {}
    }
    
    if not pairs_list:
        return result
    
    # 1. 过滤 SupplyDemandLog
    if not supply_demand_log.empty:
        # 首先按日期范围过滤
        sdl_filtered = supply_demand_log[
            (supply_demand_log['date'] >= sim_date) &
            (supply_demand_log['date'] <= horizon_end)
        ]
        if not sdl_filtered.empty:
            result['sdl'] = accel.batch_filter_all_pairs(
                sdl_filtered, pairs_list, 'material', 'location'
            )
    
    # 2. 过滤 SafetyStock (仅 horizon_end 当天)
    if not safety_stock.empty:
        ss_filtered = safety_stock[
            safety_stock['date'] == horizon_end
        ]
        if not ss_filtered.empty:
            result['ss'] = accel.batch_filter_all_pairs(
                ss_filtered, pairs_list, 'material', 'location'
            )
    
    # 3. 过滤 OrderLog
    if not order_log.empty:
        order_filtered = order_log[
            (order_log['date'] >= sim_date) &
            (order_log['date'] <= horizon_end)
        ]
        if not order_filtered.empty:
            result['order'] = accel.batch_filter_all_pairs(
                order_filtered, pairs_list, 'material', 'location'
            )
    
    elapsed = time.perf_counter() - t_start
    # print(f"[M5 Batch] Prefiltered {len(pairs_list)} pairs in {elapsed:.3f}s")
    
    return result


def batch_build_indices_for_layer(
    all_pairs: Set[Tuple[str, str]],
    supply_demand_log: pd.DataFrame,
    safety_stock: pd.DataFrame,
    order_log: pd.DataFrame,
    deploy_config: pd.DataFrame,
) -> Dict[str, Dict[Tuple[str, str], pd.DataFrame]]:
    """
    批量为层内所有节点构建索引。
    
    在层处理开始前，预先构建所有节点的数据索引，
    后续处理时直接从索引中获取数据，避免重复过滤。
    
    Args:
        all_pairs: (material, location) 对集合
        supply_demand_log: 供需日志 DataFrame
        safety_stock: 安全库存 DataFrame
        order_log: 订单日志 DataFrame
        deploy_config: 部署配置 DataFrame
        
    Returns:
        dict: 各数据源的索引字典
    """
    accel = get_accelerator()
    pairs_list = list(all_pairs)
    
    result = {
        'sdl_index': {},
        'ss_index': {},
        'order_index': {},
        'deploy_config_index': {}
    }
    
    if not pairs_list:
        return result
    
    # 批量构建索引
    if not supply_demand_log.empty:
        result['sdl_index'] = accel.batch_filter_all_pairs(
            supply_demand_log, pairs_list, 'material', 'location'
        )
    
    if not safety_stock.empty:
        result['ss_index'] = accel.batch_filter_all_pairs(
            safety_stock, pairs_list, 'material', 'location'
        )
    
    if not order_log.empty:
        result['order_index'] = accel.batch_filter_all_pairs(
            order_log, pairs_list, 'material', 'location'
        )
    
    # DeployConfig 使用 (material, sending) 作为键
    if not deploy_config.empty:
        deploy_pairs = [
            (mat, loc) for mat, loc in pairs_list
        ]
        result['deploy_config_index'] = accel.batch_filter_all_pairs(
            deploy_config, deploy_pairs, 'material', 'sending'
        )
    
    return result
