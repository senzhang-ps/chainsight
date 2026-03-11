# -*- coding: utf-8 -*-
"""
车辆容量管理模块

提供 Module6 的车辆容量计划管理功能，包括：
- 容量计划标准化（支持逐日和区间两种格式）
- 车辆容量查询
- 容量分配计算

典型用法示例:
    cap_daily = normalize_capacity_plan(truck_cap, sim_start, sim_end)
    capacity = get_truck_capacity(cap_map, date, route, truck_type)
"""

from typing import Any, Dict, Optional, Tuple

import pandas as pd


def normalize_capacity_plan(
    truck_cap_df: pd.DataFrame,
    sim_start: pd.Timestamp,
    sim_end: pd.Timestamp
) -> pd.DataFrame:
    """
    标准化容量计划为日粒度。
    
    兼容两种输入格式：
    - 逐日格式（date 列）
    - 区间格式（eff_from, eff_to 列）
    
    展开并聚合为日粒度，重叠区间会求和。
    
    参数：
        truck_cap_df: 容量计划 DataFrame
        sim_start: 仿真开始日期
        sim_end: 仿真结束日期
        
    返回：
        标准化后的日粒度容量计划 DataFrame
    """
    df = truck_cap_df.copy()
    parts = []
    
    daily_part = _process_daily_capacity(df, sim_start, sim_end)
    if daily_part is not None:
        parts.append(daily_part)
    
    range_part = _process_range_capacity(df, sim_start, sim_end)
    if range_part is not None:
        parts.append(range_part)
    
    if not parts:
        return _empty_capacity_dataframe()
    
    return _aggregate_capacity(pd.concat(parts, ignore_index=True))


def _process_daily_capacity(
    df: pd.DataFrame,
    sim_start: pd.Timestamp,
    sim_end: pd.Timestamp
) -> Optional[pd.DataFrame]:
    """
    处理逐日格式的容量数据。
    
    参数：
        df: 原始容量 DataFrame
        sim_start: 仿真开始日期
        sim_end: 仿真结束日期
        
    返回：
        处理后的日粒度容量数据，如果无数据则返回 None
    """
    if 'date' not in df.columns:
        return None
    
    daily = df[['date', 'sending', 'receiving', 'truck_type', 'truck_number']].copy()
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily[(daily['date'] >= sim_start) & (daily['date'] <= sim_end)]
    
    return daily if not daily.empty else None


def _process_range_capacity(
    df: pd.DataFrame,
    sim_start: pd.Timestamp,
    sim_end: pd.Timestamp
) -> Optional[pd.DataFrame]:
    """
    处理区间格式的容量数据。
    
    参数：
        df: 原始容量 DataFrame
        sim_start: 仿真开始日期
        sim_end: 仿真结束日期
        
    返回：
        展开后的日粒度容量数据，如果无数据则返回 None
    """
    required_cols = {'eff_from', 'eff_to'}
    if not required_cols.issubset(set(df.columns)):
        return None
    
    range_df = _prepare_range_data(df, sim_start, sim_end)
    if range_df.empty:
        return None
    
    return _expand_ranges_to_daily(range_df)


def _prepare_range_data(
    df: pd.DataFrame,
    sim_start: pd.Timestamp,
    sim_end: pd.Timestamp
) -> pd.DataFrame:
    """
    准备区间数据并裁剪到仿真范围。
    
    参数：
        df: 原始容量 DataFrame
        sim_start: 仿真开始日期
        sim_end: 仿真结束日期
        
    返回：
        裁剪后的区间数据
    """
    cols = ['eff_from', 'eff_to', 'sending', 'receiving', 'truck_type', 'truck_number']
    range_df = df[cols].copy()
    
    range_df['eff_from'] = pd.to_datetime(range_df['eff_from'])
    range_df['eff_to'] = pd.to_datetime(range_df['eff_to'])
    
    range_df['from_clip'] = range_df['eff_from'].clip(lower=sim_start)
    range_df['to_clip'] = range_df['eff_to'].clip(upper=sim_end)
    
    return range_df[range_df['from_clip'] <= range_df['to_clip']]


def _expand_ranges_to_daily(range_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    将区间数据展开为逐日数据。
    
    参数：
        range_df: 裁剪后的区间数据
        
    返回：
        展开后的日粒度数据
    """
    if range_df.empty:
        return None
    
    # 使用向量化方式展开日期区间
    df = range_df.copy()
    df['date'] = df.apply(
        lambda r: pd.date_range(r['from_clip'], r['to_clip'], freq='D').tolist(),
        axis=1
    )
    exploded = df.explode('date', ignore_index=True)
    
    return exploded[['date', 'sending', 'receiving', 'truck_type', 'truck_number']]


def _aggregate_capacity(cap_daily: pd.DataFrame) -> pd.DataFrame:
    """
    聚合日粒度容量（相同键的车辆数量求和）。
    
    参数：
        cap_daily: 日粒度容量数据
        
    返回：
        聚合后的容量数据
    """
    return cap_daily.groupby(
        ['date', 'sending', 'receiving', 'truck_type'],
        as_index=False
    )['truck_number'].sum()


def _empty_capacity_dataframe() -> pd.DataFrame:
    """
    创建空的容量 DataFrame。
    
    返回：
        具有正确列结构的空 DataFrame
    """
    return pd.DataFrame(
        columns=['date', 'sending', 'receiving', 'truck_type', 'truck_number']
    )


def get_truck_capacity(
    cap_map: Dict[Tuple, int],
    date: pd.Timestamp,
    sending: str,
    receiving: str,
    truck_type: str,
    default_capacity: int = 99
) -> int:
    """
    获取指定日期和路线的车辆容量。
    
    参数：
        cap_map: 容量映射字典
        date: 日期
        sending: 发送地点
        receiving: 接收地点
        truck_type: 车型
        default_capacity: 默认容量（未配置时使用）
        
    返回：
        可用车辆数量
    """
    key = (date, sending, receiving, truck_type)
    return int(cap_map.get(key, default_capacity))


def build_capacity_map(
    cap_daily: pd.DataFrame
) -> Dict[Tuple, int]:
    """
    构建容量查询映射。
    
    参数：
        cap_daily: 日粒度容量 DataFrame
        
    返回：
        容量映射字典 {(date, sending, receiving, truck_type): truck_number}
    """
    if cap_daily.empty:
        return {}
    
    return cap_daily.set_index(
        ['date', 'sending', 'receiving', 'truck_type']
    )['truck_number'].to_dict()


def get_truck_spec(
    spec_map: Dict[str, Dict[str, Any]],
    truck_type: str
) -> Optional[Dict[str, Any]]:
    """
    获取车型规格。
    
    参数：
        spec_map: 车型规格映射
        truck_type: 车型名称
        
    返回：
        车型规格字典，不存在则返回 None
    """
    return spec_map.get(truck_type)


def get_truck_config(
    truck_con: pd.DataFrame,
    sending: str,
    receiving: str
) -> pd.DataFrame:
    """
    获取指定路线的卡车配置。
    
    参数：
        truck_con: 卡车配置 DataFrame
        sending: 发送地点
        receiving: 接收地点
        
    返回：
        该路线的卡车配置子集
    """
    return truck_con[
        (truck_con['sending'] == sending) &
        (truck_con['receiving'] == receiving)
    ]


def get_optimal_truck_sequence(truck_cfgs: pd.DataFrame) -> list:
    """
    获取最优车型序列。
    
    优先使用标记为 optimal_type='Y' 的车型，
    其余车型按原顺序排列。
    
    参数：
        truck_cfgs: 卡车配置 DataFrame
        
    返回：
        按优先级排序的车型列表
    """
    if truck_cfgs.empty:
        return []
    
    all_types = truck_cfgs['truck_type'].tolist()
    
    # `optimal_type` 列可能不存在（如数据库模式下全空值列被 `dropna` 移除）
    if 'optimal_type' in truck_cfgs.columns:
        optimal_types = truck_cfgs[
            truck_cfgs['optimal_type'] == 'Y'
        ]['truck_type'].tolist()
        non_optimal = [t for t in all_types if t not in optimal_types]
        return optimal_types + non_optimal
    
    return all_types
