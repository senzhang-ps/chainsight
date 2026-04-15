# -*- coding: utf-8 -*-
"""
需求收集模块 - 向量化版本

使用 numpy 和 pandas 向量化操作替代逐节点循环，大幅提升性能。

关键优化:
1. 批量处理所有节点，而非逐个循环
2. 使用 numpy 广播和 pandas merge 替代嵌套循环
3. 预计算所有节点的 horizon 和 upstream
4. 一次性过滤所有数据，而非每节点单独过滤

性能预期: 10s -> 1-2s (80-90% 提升)
"""
import time
from datetime import timedelta
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd

from .constants import DEFAULT_MOQ, DEFAULT_RV


def collect_demands_batch_vectorized(
    pairs: Set[Tuple[str, str]],
    sim_date: pd.Timestamp,
    config: dict,
    up_gap_buffer: dict,
    ptf_lsk_cache: Optional[dict] = None,
    lead_time_cache: Optional[dict] = None,
    active_network_cache: Optional[dict] = None
) -> Dict[Tuple[str, str], List[dict]]:
    """
    批量收集所有节点的需求 - 向量化版本。
    
    使用向量化操作替代逐节点循环，大幅提升性能。
    
    参数：
        pairs: (material, location) 对的集合
        sim_date: 仿真日期
        config: 配置字典
        up_gap_buffer: 上游缺口缓冲区
        ptf_lsk_cache: PTF/LSK缓存
        lead_time_cache: LeadTime缓存
        active_network_cache: Network缓存
        
    返回：
        dict: (material, location) -> 需求行列表
    """
    if not pairs:
        return {}
    
    t_start = time.perf_counter()
    
    # 1. 构建节点 DataFrame
    pairs_list = list(pairs)
    nodes_df = pd.DataFrame(pairs_list, columns=['material', 'location'])
    nodes_df['material'] = nodes_df['material'].astype(str)
    nodes_df['location'] = nodes_df['location'].astype(str)
    
    # 2. 批量获取 upstream 和 horizon
    nodes_df = _batch_get_upstream_horizon(
        nodes_df, sim_date, config,
        ptf_lsk_cache, lead_time_cache, active_network_cache
    )
    
    # 3. 批量收集 SDL 需求
    sdl_demands = _batch_collect_sdl_demands(
        nodes_df, sim_date, config
    )
    
    # 4. 批量收集安全库存需求
    ss_demands = _batch_collect_safety_stock_demands(
        nodes_df, sim_date, config
    )
    
    # 5. 批量收集订单需求
    order_demands = _batch_collect_order_demands(
        nodes_df, sim_date, config
    )
    
    # 6. 收集 GAP 传递需求（仍需逐节点，因为依赖 up_gap_buffer）
    gap_demands = _collect_gap_demands_batch(
        nodes_df, sim_date, config, up_gap_buffer
    )
    
    # 7. 合并所有需求
    result = _merge_demands(
        pairs_list, sdl_demands, ss_demands, order_demands, gap_demands
    )
    
    elapsed = time.perf_counter() - t_start
    
    return result


def _batch_get_upstream_horizon(
    nodes_df: pd.DataFrame,
    sim_date: pd.Timestamp,
    config: dict,
    ptf_lsk_cache: Optional[dict],
    lead_time_cache: Optional[dict],
    active_network_cache: Optional[dict]
) -> pd.DataFrame:
    """
    批量获取所有节点的 upstream 和 horizon（优化版）。
    使用纯向量化操作替代 df.apply()。
    """
    network_df = config.get('Network', pd.DataFrame())
    leadtime_df = config.get('LeadTime', pd.DataFrame())
    m4_mlcfg_df = config.get('M4_MaterialLocationLineCfg', pd.DataFrame())
    
    # 默认值
    nodes_df['upstream'] = ''
    nodes_df['horizon'] = 7  # 默认 horizon
    nodes_df['leadtime'] = 0
    
    if network_df.empty:
        nodes_df['horizon_end'] = sim_date + timedelta(days=7)
        return nodes_df
    
    # 准备 network 数据
    network_active = network_df.copy()
    network_active['material'] = network_active['material'].astype(str)
    network_active['location'] = network_active['location'].astype(str)
    
    # 过滤有效期
    if 'eff_from' in network_active.columns:
        network_active = network_active[
            (network_active['eff_from'] <= sim_date) &
            (network_active['eff_to'] >= sim_date)
        ]
    
    # 合并获取 upstream
    merged = nodes_df.merge(
        network_active[['material', 'location', 'sourcing']],
        on=['material', 'location'],
        how='left'
    )
    nodes_df['upstream'] = merged['sourcing'].fillna('').astype(str)
    
    # 优化：使用向量化计算horizon（替代df.apply）
    if lead_time_cache:
        # 构建查找键
        n = len(nodes_df)
        upstreams = nodes_df['upstream'].values
        locations = nodes_df['location'].values
        
        # 使用列表推导代替apply（更快）
        horizons = np.array([
            _get_horizon_from_cache(upstreams[i], locations[i], lead_time_cache)
            for i in range(n)
        ], dtype=np.int64)
        
        nodes_df['horizon'] = horizons
    else:
        # 无缓存时使用默认值
        nodes_df['horizon'] = 7
    
    # 优化：向量化计算leadtime（替代apply）
    # 有upstream时使用horizon，否则为0
    has_upstream = (nodes_df['upstream'] != '') & (nodes_df['upstream'].notna())
    nodes_df['leadtime'] = np.where(has_upstream, nodes_df['horizon'], 0)
    
    # 优化：向量化计算horizon_end
    horizon_days = nodes_df['horizon'].values
    nodes_df['horizon_end'] = pd.to_datetime([
        sim_date + timedelta(days=int(h)) for h in horizon_days
    ])
    
    return nodes_df


def _get_horizon_from_cache(upstream: str, location: str, lead_time_cache: dict) -> int:
    """从缓存获取horizon值（辅助函数）。"""
    if not upstream or str(upstream).strip() == '':
        return 7  # 根节点默认
    key = (str(upstream), str(location))
    if key in lead_time_cache:
        vals = lead_time_cache[key]
        if isinstance(vals, tuple) and len(vals) >= 3:
            pdt, gr, mct = vals[0], vals[1], vals[2]
            horizon = max(int(mct), int(pdt) + int(gr))
            return max(1, horizon)
        elif isinstance(vals, (int, float)):
            return max(1, int(vals))
    return 7


def _batch_collect_sdl_demands(
    nodes_df: pd.DataFrame,
    sim_date: pd.Timestamp,
    config: dict
) -> Dict[Tuple[str, str], List[dict]]:
    """
    批量收集 SDL 需求。
    """
    sdl_df = config.get('SupplyDemandLog', pd.DataFrame())
    
    if sdl_df.empty:
        return {}
    
    # 准备 SDL 数据
    sdl = sdl_df.copy()
    sdl['material'] = sdl['material'].astype(str)
    sdl['location'] = sdl['location'].astype(str)
    sdl['date'] = pd.to_datetime(sdl['date'])
    
    # 与节点合并
    merged = sdl.merge(
        nodes_df[['material', 'location', 'upstream', 'leadtime', 'horizon_end']],
        on=['material', 'location'],
        how='inner'
    )
    
    if merged.empty:
        return {}
    
    # 过滤日期窗口
    merged = merged[
        (merged['date'] >= sim_date) &
        (merged['date'] <= merged['horizon_end'])
    ]
    
    if merged.empty:
        return {}
    
    # 构建结果
    result = {}
    for (mat, loc), group in merged.groupby(['material', 'location']):
        rows = []
        upstream = group['upstream'].iloc[0]
        leadtime = int(group['leadtime'].iloc[0])
        
        for _, row in group.iterrows():
            rows.append({
                'material': mat,
                'location': loc,
                'sending': upstream if upstream and str(upstream).strip() else None,
                'receiving': loc,
                'demand_element': row['demand_element'],
                'demand_qty': int(row['quantity']),
                'planned_qty': int(row['quantity']),
                'moq': DEFAULT_MOQ,
                'rv': DEFAULT_RV,
                'leadtime': leadtime,
                'requirement_date': row['date'],
                'plan_deploy_date': sim_date,
                'orig_location': loc
            })
        result[(mat, loc)] = rows
    
    return result


def _batch_collect_safety_stock_demands(
    nodes_df: pd.DataFrame,
    sim_date: pd.Timestamp,
    config: dict
) -> Dict[Tuple[str, str], List[dict]]:
    """
    批量收集安全库存需求（仅 horizon_end 当天）。
    """
    ss_df = config.get('SafetyStock', pd.DataFrame())
    
    if ss_df.empty:
        return {}
    
    # 准备安全库存数据
    ss = ss_df.copy()
    ss['material'] = ss['material'].astype(str)
    ss['location'] = ss['location'].astype(str)
    ss['date'] = pd.to_datetime(ss['date'])
    
    # 与节点合并
    merged = ss.merge(
        nodes_df[['material', 'location', 'upstream', 'leadtime', 'horizon_end']],
        on=['material', 'location'],
        how='inner'
    )
    
    if merged.empty:
        return {}
    
    # 过滤：仅取 horizon_end 当天
    merged = merged[merged['date'] == merged['horizon_end']]
    
    if merged.empty:
        return {}
    
    # 构建结果
    result = {}
    for (mat, loc), group in merged.groupby(['material', 'location']):
        upstream = group['upstream'].iloc[0]
        leadtime = int(group['leadtime'].iloc[0])
        horizon_end = group['horizon_end'].iloc[0]
        
        ss_qty = int(pd.to_numeric(
            group['safety_stock_qty'], errors='coerce'
        ).fillna(0).sum())
        
        if ss_qty > 0:
            result[(mat, loc)] = [{
                'material': mat,
                'location': loc,
                'sending': upstream if upstream and str(upstream).strip() else None,
                'receiving': loc,
                'demand_element': 'safety',
                'demand_qty': ss_qty,
                'planned_qty': ss_qty,
                'moq': DEFAULT_MOQ,
                'rv': DEFAULT_RV,
                'leadtime': leadtime,
                'requirement_date': horizon_end,
                'plan_deploy_date': sim_date,
                'orig_location': loc
            }]
    
    return result


def _batch_collect_order_demands(
    nodes_df: pd.DataFrame,
    sim_date: pd.Timestamp,
    config: dict
) -> Dict[Tuple[str, str], List[dict]]:
    """
    批量收集订单需求。
    """
    order_df = config.get('OrderLog', pd.DataFrame())
    
    if order_df.empty:
        return {}
    
    # 准备订单数据
    orders = order_df.copy()
    orders['material'] = orders['material'].astype(str)
    orders['location'] = orders['location'].astype(str)
    orders['date'] = pd.to_datetime(orders['date'])
    
    # 与节点合并
    merged = orders.merge(
        nodes_df[['material', 'location', 'upstream', 'leadtime', 'horizon_end']],
        on=['material', 'location'],
        how='inner'
    )
    
    if merged.empty:
        return {}
    
    # 过滤日期窗口
    merged = merged[
        (merged['date'] >= sim_date) &
        (merged['date'] <= merged['horizon_end'])
    ]
    
    if merged.empty:
        return {}
    
    # 构建结果
    result = {}
    for (mat, loc), group in merged.groupby(['material', 'location']):
        rows = []
        upstream = group['upstream'].iloc[0]
        leadtime = int(group['leadtime'].iloc[0])
        
        for _, row in group.iterrows():
            rows.append({
                'material': mat,
                'location': loc,
                'sending': upstream if upstream and str(upstream).strip() else None,
                'receiving': loc,
                'demand_element': row.get('demand_type', 'AO'),
                'demand_qty': int(row['quantity']),
                'planned_qty': int(row['quantity']),
                'moq': DEFAULT_MOQ,
                'rv': DEFAULT_RV,
                'leadtime': leadtime,
                'requirement_date': row['date'],
                'plan_deploy_date': sim_date,
                'orig_location': loc
            })
        result[(mat, loc)] = rows
    
    return result


def _collect_gap_demands_batch(
    nodes_df: pd.DataFrame,
    sim_date: pd.Timestamp,
    config: dict,
    up_gap_buffer: dict
) -> Dict[Tuple[str, str], List[dict]]:
    """
    批量收集 GAP 传递需求。
    
    注意：这部分仍需要逐节点处理，因为依赖 up_gap_buffer 的结构。
    """
    if not up_gap_buffer:
        return {}
    
    result = {}
    deploy_cfg = config.get('DeployConfig', pd.DataFrame())
    
    # 创建节点信息字典用于快速查找
    node_info = {}
    for _, row in nodes_df.iterrows():
        node_info[(row['material'], row['location'])] = {
            'upstream': row['upstream'],
            'leadtime': int(row['leadtime']),
            'horizon_end': row['horizon_end']
        }
    
    for (mat, loc) in up_gap_buffer.keys():
        if (mat, loc) not in node_info:
            continue
        
        info = node_info[(mat, loc)]
        horizon_end = info['horizon_end']
        leadtime = info['leadtime']
        upstream = info['upstream']
        
        rows = []
        for gap in up_gap_buffer[(mat, loc)]:
            req_dt = pd.to_datetime(gap.get('requirement_date', sim_date))
            if not (req_dt >= sim_date and req_dt <= horizon_end):
                continue
            
            rows.append({
                'material': mat,
                'location': gap.get('location', loc),
                'receiving': gap.get('receiving', gap.get('location', loc)),
                'orig_location': gap.get('orig_location', gap.get('location', loc)),
                'sending': upstream if upstream and str(upstream).strip() else None,
                'demand_element': gap['demand_element'],
                'demand_qty': int(gap['planned_qty']),
                'planned_qty': int(gap['planned_qty']),
                'moq': DEFAULT_MOQ,
                'rv': DEFAULT_RV,
                'leadtime': leadtime,
                'requirement_date': req_dt,
                'plan_deploy_date': sim_date,
                'from_location': gap.get('from_location', None),
            })
        
        if rows:
            result[(mat, loc)] = rows
    
    return result


def _merge_demands(
    pairs_list: List[Tuple[str, str]],
    sdl_demands: Dict[Tuple[str, str], List[dict]],
    ss_demands: Dict[Tuple[str, str], List[dict]],
    order_demands: Dict[Tuple[str, str], List[dict]],
    gap_demands: Dict[Tuple[str, str], List[dict]]
) -> Dict[Tuple[str, str], List[dict]]:
    """
    合并所有需求来源。
    """
    result = {}
    
    for pair in pairs_list:
        demands = []
        
        if pair in sdl_demands:
            demands.extend(sdl_demands[pair])
        if pair in ss_demands:
            demands.extend(ss_demands[pair])
        if pair in order_demands:
            demands.extend(order_demands[pair])
        if pair in gap_demands:
            demands.extend(gap_demands[pair])
        
        result[pair] = demands
    
    return result
