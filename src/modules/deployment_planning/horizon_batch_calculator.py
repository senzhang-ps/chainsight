# -*- coding: utf-8 -*-
"""
批量Horizon计算器

为所有(material, location)对批量计算horizon，避免重复的单节点查找。
与原始逻辑100%一致，仅通过预计算提升性能。

优化策略:
- 预构建Network查找索引
- 批量计算所有节点的(upstream, location_type, horizon, leadtime_for_row)
- 将结果存入缓存供demand_collector使用
"""
from datetime import timedelta
from typing import Dict, List, Optional, Tuple, Set
import pandas as pd

from .cache_utils import (
    get_ptf_lsk,
    get_active_network,
    get_sending_location_type,
    determine_lead_time,
    DEFAULT_LEAD_TIME
)
from .constants import DEFAULT_PTF, DEFAULT_LSK


def build_horizon_cache(
    all_pairs: Set[Tuple[str, str]],
    sim_date: pd.Timestamp,
    network_df: pd.DataFrame,
    leadtime_df: pd.DataFrame,
    m4_mlcfg_df: Optional[pd.DataFrame],
    ptf_lsk_cache: Dict,
    lead_time_cache: Dict,
    active_network_cache: Dict,
    location_layer_map: Dict
) -> Dict[Tuple[str, str], dict]:
    """
    批量计算所有(material, location)对的horizon相关参数。
    
    返回的缓存结构:
    {
        (material, location): {
            'upstream': str or None,
            'location_type': str,
            'horizon': int,
            'leadtime_for_row': int,
            'horizon_end': pd.Timestamp
        }
    }
    
    Args:
        all_pairs: 所有(material, location)对的集合
        sim_date: 仿真日期
        network_df: Network配置
        leadtime_df: LeadTime配置
        m4_mlcfg_df: M4配置(用于PTF/LSK)
        ptf_lsk_cache: PTF/LSK缓存
        lead_time_cache: LeadTime缓存
        active_network_cache: 活动Network缓存
        location_layer_map: 位置层级映射
    
    Returns:
        dict: (material, location) -> horizon参数字典
    """
    horizon_cache = {}
    
    # 预构建 Network 快速查找索引
    # 键：(material, location) -> network_row
    network_index = _build_network_index(network_df, sim_date)
    
    # 排序确保遍历顺序一致
    sorted_pairs = sorted(all_pairs)
    for mat, loc in sorted_pairs:
        mat_str = str(mat)
        loc_str = str(loc)
        
        # 获取上游
        upstream = _get_upstream_from_index(
            network_index, mat_str, loc_str
        )
        
        # 计算horizon
        if upstream and str(upstream).strip():
            # 有上游：计算DC/Plant类型的lead time
            sending_location_type = _infer_location_type(
                material=mat_str,
                sending=str(upstream),
                sim_date=sim_date,
                network_index=network_index,
                location_layer_map=location_layer_map
            )
            
            horizon, err = determine_lead_time(
                sending=str(upstream),
                receiving=loc_str,
                location_type=str(sending_location_type),
                lead_time_df=leadtime_df,
                m4_mlcfg_df=m4_mlcfg_df,
                material=mat_str,
                lead_time_cache=lead_time_cache,
                ptf_lsk_cache=ptf_lsk_cache
            )
            
            if err:
                horizon = 1
            
            leadtime_for_row = int(horizon)
        else:
            # 顶层(无上游)：按Plant公式计算
            ptf, lsk_val = get_ptf_lsk(
                material=mat_str,
                site=loc_str,
                m4_mlcfg_df=m4_mlcfg_df,
                cache=ptf_lsk_cache
            )
            
            # 从lead_time_cache获取PDT/GR/MCT
            base_values = lead_time_cache.get((loc_str, loc_str))
            if base_values:
                pdt, gr, mct = base_values
            else:
                # 回退到DataFrame查找
                df_loc = leadtime_df[leadtime_df['sending'] == loc_str]
                mct = int(pd.to_numeric(
                    df_loc.get('MCT', 0), errors='coerce'
                ).fillna(0).max()) if not df_loc.empty else 0
                pdt = int(pd.to_numeric(
                    df_loc.get('PDT', 0), errors='coerce'
                ).fillna(0).max()) if not df_loc.empty else 0
                gr = int(pd.to_numeric(
                    df_loc.get('GR', 0), errors='coerce'
                ).fillna(0).max()) if not df_loc.empty else 0
            
            base_lt = max(mct, pdt + gr)
            horizon = max(1, int(base_lt + int(ptf) + int(lsk_val) - 1))
            leadtime_for_row = 0
        
        horizon_end = sim_date + timedelta(days=int(horizon))
        
        horizon_cache[(mat_str, loc_str)] = {
            'upstream': upstream,
            'horizon': int(horizon),
            'leadtime_for_row': int(leadtime_for_row),
            'horizon_end': horizon_end
        }
    
    return horizon_cache


def _build_network_index(
    network_df: pd.DataFrame,
    sim_date: pd.Timestamp
) -> Dict[Tuple[str, str], dict]:
    """
    构建Network快速查找索引（按sim_date过滤后）。
    
    Args:
        network_df: Network DataFrame
        sim_date: 仿真日期
    
    Returns:
        dict: (material, location) -> row dict
    """
    if network_df.empty:
        return {}
    
    # 预过滤有效日期范围内的记录
    active_mask = (
        (network_df['eff_from'] <= sim_date) &
        (network_df['eff_to'] >= sim_date)
    )
    active_network = network_df[active_mask]
    
    index = {}
    for row in active_network.itertuples():
        mat = str(row.material)
        loc = str(row.location)
        key = (mat, loc)
        
        # 只保留第一个匹配（与原始逻辑一致）
        if key not in index:
            index[key] = {
                'sourcing': getattr(row, 'sourcing', None),
                'location_type': getattr(row, 'location_type', 'DC')
            }
    
    return index


def _get_upstream_from_index(
    network_index: Dict[Tuple[str, str], dict],
    material: str,
    location: str
) -> Optional[str]:
    """
    从预构建索引获取上游sourcing。
    
    Args:
        network_index: Network索引
        material: 物料编码
        location: 位置编码
    
    Returns:
        str or None: 上游位置
    """
    row = network_index.get((material, location))
    if row:
        sourcing = row.get('sourcing')
        if sourcing and pd.notna(sourcing) and str(sourcing).strip():
            return str(sourcing)
    return None


def _infer_location_type(
    material: str,
    sending: str,
    sim_date: pd.Timestamp,
    network_index: Dict[Tuple[str, str], dict],
    location_layer_map: dict
) -> str:
    """
    推断发送端位置类型（与原始get_sending_location_type一致）。
    
    Args:
        material: 物料编码
        sending: 发送端编码
        sim_date: 仿真日期（未使用但保持接口一致）
        network_index: Network索引
        location_layer_map: 位置层级映射
    
    Returns:
        str: 'Plant' 或 'DC'
    """
    if not sending or pd.isna(sending) or str(sending).strip() == "":
        return 'DC'
    
    # 从索引查找
    row = network_index.get((material, sending))
    if row:
        return str(row.get('location_type', 'DC') or 'DC')
    
    # 未维护但被识别为根节点 → Plant
    # Use (material, location) tuple key to match baseline
    if location_layer_map.get((str(material), str(sending)), None) == 0:
        return 'Plant'
    
    return 'DC'
