# -*- coding: utf-8 -*-
"""
需求收集模块

提供节点需求收集功能，包括SDL、安全库存、订单和GAP传递。
"""
from datetime import timedelta
from typing import Dict, List, Optional

import pandas as pd

from .cache_utils import (
    get_active_network,
    get_from_index,
    get_ptf_lsk,
    get_sending_location_type,
    determine_lead_time
)
from .constants import DEFAULT_MOQ, DEFAULT_RV


def _lookup_moq_rv(
    deploy_cfg: pd.DataFrame,
    material: str,
    sending: str,
    receiving: Optional[str]
) -> tuple:
    """
    从DeployConfig查找(material, sending, receiving)的MOQ/RV。

    优先级：
    1. 按三键(material, sending, receiving)精确匹配
    2. 回退按(material, sending)匹配
    3. 默认返回(1, 1)

    参数：
        deploy_cfg: DeployConfig DataFrame
        material: 物料编码
        sending: 发送端编码
        receiving: 接收端编码

    返回：
        tuple: (moq, rv)
    """
    try:
        # 尝试三键匹配
        if 'receiving' in deploy_cfg.columns and receiving is not None:
            rows = deploy_cfg[
                (deploy_cfg['material'] == str(material)) &
                (deploy_cfg['sending'] == str(sending)) &
                (deploy_cfg['receiving'] == str(receiving))
            ]
            if not rows.empty:
                moq = int(pd.to_numeric(
                    rows.iloc[0].get('moq', 1), errors='coerce'
                ) or 1)
                rv = int(pd.to_numeric(
                    rows.iloc[0].get('rv', 1), errors='coerce'
                ) or 1)
                return max(0, moq), max(0, rv)

        # 回退到两键匹配
        rows2 = deploy_cfg[
            (deploy_cfg['material'] == str(material)) &
            (deploy_cfg['sending'] == str(sending))
        ]
        if not rows2.empty:
            moq = int(pd.to_numeric(
                rows2.iloc[0].get('moq', 1), errors='coerce'
            ) or 1)
            rv = int(pd.to_numeric(
                rows2.iloc[0].get('rv', 1), errors='coerce'
            ) or 1)
            return max(0, moq), max(0, rv)
    except Exception:
        pass

    return DEFAULT_MOQ, DEFAULT_RV


def _collect_sdl_demands(
    supply_demand_log: pd.DataFrame,
    material: str,
    location: str,
    upstream: Optional[str],
    sim_date: pd.Timestamp,
    horizon_end: pd.Timestamp,
    leadtime_for_row: int,
    sdl_index: Optional[Dict[tuple, pd.DataFrame]] = None
) -> List[dict]:
    """
    收集SupplyDemandLog需求。

    参数：
        supply_demand_log: SDL DataFrame
        material: 物料编码
        location: 位置编码
        upstream: 上游位置
        sim_date: 仿真日期
        horizon_end: 窗口结束日期
        leadtime_for_row: 行级lead time
        sdl_index: SDL预建索引（可选）

    返回：
        list: 需求行列表
    """
    demand_rows = []

    # 使用预建索引或回退到过滤
    if sdl_index is not None:
        sdl = get_from_index(sdl_index, (str(material), str(location)))
        if sdl.empty:
            return demand_rows
        sdl = sdl.copy()
    else:
        # 回退到逐行过滤
        sdl_mask = (
            (supply_demand_log['material'] == material) &
            (supply_demand_log['location'] == location)
        )
        sdl = supply_demand_log[sdl_mask].copy()

    if sdl.empty:
        return demand_rows

    sdl['requirement_date'] = pd.to_datetime(sdl['date'])

    # 筛选窗口内数据（forecast和其他类型统一处理）
    mask = (
        (sdl['requirement_date'] >= sim_date) &
        (sdl['requirement_date'] <= horizon_end)
    )
    combined = sdl[mask]

    if combined.empty:
        return demand_rows

    # 使用向量化方式构建结果（避免itertuples循环）
    leadtime_val = leadtime_for_row if upstream else 0
    
    # 向量化：直接构建 DataFrame 然后转换为 records
    result_df = pd.DataFrame({
        'material': material,
        'location': location,
        'sending': upstream,
        'receiving': location,
        'demand_element': combined['demand_element'].values,
        'demand_qty': combined['quantity'].astype(int).values,
        'planned_qty': combined['quantity'].astype(int).values,
        'moq': DEFAULT_MOQ,
        'rv': DEFAULT_RV,
        'leadtime': leadtime_val,
        'requirement_date': combined['requirement_date'].values,
        'plan_deploy_date': sim_date,
        'orig_location': location
    })
    
    return result_df.to_dict('records')


def _collect_safety_stock_demands(
    safety_stock: pd.DataFrame,
    material: str,
    location: str,
    upstream: Optional[str],
    sim_date: pd.Timestamp,
    horizon_end: pd.Timestamp,
    leadtime_for_row: int,
    ss_index: Optional[Dict[tuple, pd.DataFrame]] = None
) -> List[dict]:
    """
    收集安全库存需求（仅horizon_end当天）。

    参数：
        safety_stock: `SafetyStock` DataFrame
        material: 物料编码
        location: 位置编码
        upstream: 上游位置
        sim_date: 仿真日期
        horizon_end: 窗口结束日期
        leadtime_for_row: 行级lead time
        ss_index: SafetyStock预建索引（可选）

    返回：
        list: 需求行列表
    """
    demand_rows = []

    # 使用预建索引或回退到过滤
    if ss_index is not None:
        ss = get_from_index(ss_index, (str(material), str(location)))
        if ss.empty:
            return demand_rows
        ss = ss.copy()
    else:
        ss_mask = (
            (safety_stock['material'] == material) &
            (safety_stock['location'] == location)
        )
        ss = safety_stock[ss_mask].copy()

    if ss.empty:
        return demand_rows

    ss['date'] = pd.to_datetime(ss['date'])
    ss_end = ss[ss['date'] == horizon_end]

    if ss_end.empty:
        return demand_rows

    ss_qty = int(pd.to_numeric(
        ss_end['safety_stock_qty'], errors='coerce'
    ).fillna(0).sum())

    if ss_qty > 0:
        demand_rows.append({
            'material': material,
            'location': location,
            'sending': upstream,
            'receiving': location,
            'demand_element': 'safety',
            'demand_qty': ss_qty,
            'planned_qty': ss_qty,
            'moq': DEFAULT_MOQ,
            'rv': DEFAULT_RV,
            'leadtime': leadtime_for_row if upstream else 0,
            'requirement_date': horizon_end,
            'plan_deploy_date': sim_date,
            'orig_location': location
        })

    return demand_rows


def _collect_order_demands(
    order_df: pd.DataFrame,
    material: str,
    location: str,
    upstream: Optional[str],
    sim_date: pd.Timestamp,
    horizon_end: pd.Timestamp,
    leadtime_for_row: int,
    order_index: Optional[Dict[tuple, pd.DataFrame]] = None
) -> List[dict]:
    """
    收集订单需求（AO/normal）。

    参数：
        order_df: `OrderLog` DataFrame
        material: 物料编码
        location: 位置编码
        upstream: 上游位置
        sim_date: 仿真日期
        horizon_end: 窗口结束日期
        leadtime_for_row: 行级lead time
        order_index: OrderLog预建索引（可选）

    返回：
        list: 需求行列表
    """
    demand_rows = []

    if order_df.empty and order_index is None:
        return demand_rows

    # 使用预建索引或回退到过滤
    if order_index is not None:
        orders = get_from_index(order_index, (str(material), str(location)))
        if orders.empty:
            return demand_rows
        orders = orders.copy()
    else:
        if order_df.empty:
            return demand_rows
        orders_mask = (
            (order_df['material'] == material) &
            (order_df['location'] == location)
        )
        orders = order_df[orders_mask].copy()

    if orders.empty:
        return demand_rows

    orders['requirement_date'] = pd.to_datetime(orders['date'])
    orders['demand_element'] = orders['demand_type']

    # 筛选窗口内订单
    mask = (
        (orders['requirement_date'] >= sim_date) &
        (orders['requirement_date'] <= horizon_end)
    )
    orders = orders[mask]

    if orders.empty:
        return demand_rows

    # 使用向量化方式（避免itertuples循环开销）
    leadtime_val = leadtime_for_row if upstream else 0
    
    # 向量化：直接构建 DataFrame 然后转换为 records
    result_df = pd.DataFrame({
        'material': material,
        'location': location,
        'sending': upstream,
        'receiving': location,
        'demand_element': orders['demand_element'].astype(str).values,
        'demand_qty': orders['quantity'].astype(int).values,
        'planned_qty': orders['quantity'].astype(int).values,
        'moq': DEFAULT_MOQ,
        'rv': DEFAULT_RV,
        'leadtime': leadtime_val,
        'requirement_date': orders['requirement_date'].values,
        'plan_deploy_date': sim_date,
        'orig_location': location
    })
    
    return result_df.to_dict('records')


def _collect_gap_demands(
    up_gap_buffer: dict,
    deploy_cfg: pd.DataFrame,
    material: str,
    location: str,
    upstream: Optional[str],
    sim_date: pd.Timestamp,
    horizon_end: pd.Timestamp,
    leadtime_for_row: int
) -> List[dict]:
    """
    收集GAP传递需求（上游下发的净需求）。

    参数：
        up_gap_buffer: 上游缺口缓冲区
        deploy_cfg: DeployConfig DataFrame
        material: 物料编码
        location: 位置编码
        upstream: 上游位置
        sim_date: 仿真日期
        horizon_end: 窗口结束日期
        leadtime_for_row: 行级lead time

    返回：
        list: 需求行列表
    """
    demand_rows = []

    if up_gap_buffer is None:
        return demand_rows

    if (material, location) not in up_gap_buffer:
        return demand_rows

    for gap in up_gap_buffer[(material, location)]:
        req_dt = pd.to_datetime(gap.get('requirement_date', sim_date))
        if not (req_dt >= sim_date and req_dt <= horizon_end):
            continue

        recv_for_cfg = str(gap.get(
            'from_location', gap.get('location', location)
        ))
        row_moq, row_rv = _lookup_moq_rv(
            deploy_cfg, material=str(material),
            sending=str(location), receiving=recv_for_cfg
        )

        demand_rows.append({
            'material': material,
            'location': gap.get('location', location),
            'receiving': gap.get('receiving', gap.get('location', location)),
            'orig_location': gap.get(
                'orig_location', gap.get('location', location)
            ),
            'sending': upstream,
            'demand_element': gap['demand_element'],
            'demand_qty': int(gap['planned_qty']),
            'planned_qty': int(gap['planned_qty']),
            'moq': int(row_moq),
            'rv': int(row_rv),
            'leadtime': leadtime_for_row if upstream else 0,
            'requirement_date': req_dt,
            'plan_deploy_date': sim_date,
            'from_location': gap.get('from_location', None),
        })

    return demand_rows


def collect_node_demands(
    material: str,
    location: str,
    sim_date: pd.Timestamp,
    config: dict,
    up_gap_buffer: dict,
    ptf_lsk_cache: Optional[dict] = None,
    lead_time_cache: Optional[dict] = None,
    active_network_cache: Optional[dict] = None,
    sdl_index: Optional[Dict[tuple, pd.DataFrame]] = None,
    ss_index: Optional[Dict[tuple, pd.DataFrame]] = None,
    order_index: Optional[Dict[tuple, pd.DataFrame]] = None,
    deploy_config_index: Optional[Dict[tuple, pd.DataFrame]] = None
) -> List[dict]:
    """
    收集节点在当天窗口内的需求。

    需求来源：
    - `SupplyDemandLog`（forecast/others）
    - `SafetyStock`（仅horizon_end当天）
    - `OrderLog`（AO/normal）
    - 上游up_gap_buffer（净需求）

    参数：
        material: 物料编码
        location: 位置编码
        sim_date: 仿真日期
        config: 配置字典
        up_gap_buffer: 上游缺口缓冲区
        ptf_lsk_cache: PTF/LSK缓存
        lead_time_cache: LeadTime缓存
        active_network_cache: Network缓存
        sdl_index: SDL预建索引
        ss_index: SafetyStock预建索引
        order_index: OrderLog预建索引
        deploy_config_index: DeployConfig预建索引

    返回：
        list: 需求行列表
    """
    supply_demand_log = config['SupplyDemandLog']
    safety_stock = config['SafetyStock']
    deploy_cfg = config['DeployConfig']
    network = config['Network']
    leadtime_df = config['LeadTime']

    # 读取LSK/Day - 使用索引优化
    if deploy_config_index is not None:
        param_df = get_from_index(
            deploy_config_index, (str(material), str(location))
        )
        if not param_df.empty:
            lsk = param_df.iloc[0].get('lsk', 1)
        else:
            lsk = 1
    else:
        param_row = deploy_cfg[
            (deploy_cfg['material'] == material) &
            (deploy_cfg['sending'] == location)
        ]
        if not param_row.empty:
            lsk = param_row.iloc[0].get('lsk', 1)
        else:
            lsk = 1

    # 获取上游
    network_row = get_active_network(
        network, material, location, sim_date,
        cache=active_network_cache
    )
    upstream = (
        network_row.iloc[0]['sourcing']
        if not network_row.empty else None
    )

    # 计算horizon
    if upstream and str(upstream).strip():
        sending_location_type = get_sending_location_type(
            material=str(material),
            sending=str(upstream),
            sim_date=sim_date,
            network_df=network,
            location_layer_map=config.get('LocationLayerMap', {})
        )
        horizon, err = determine_lead_time(
            sending=str(upstream),
            receiving=str(location),
            location_type=str(sending_location_type),
            lead_time_df=leadtime_df,
            m4_mlcfg_df=config.get('M4_MaterialLocationLineCfg', pd.DataFrame()),
            material=str(material),
            lead_time_cache=lead_time_cache,
            ptf_lsk_cache=ptf_lsk_cache
        )
        if err:
            horizon = 1
        leadtime_for_row = int(horizon)
    else:
        # 顶层：按Plant公式计算
        ptf, lsk_val = get_ptf_lsk(
            material=str(material),
            site=str(location),
            m4_mlcfg_df=config.get('M4_MaterialLocationLineCfg', pd.DataFrame()),
            cache=ptf_lsk_cache
        )
        df_loc = leadtime_df[leadtime_df['sending'] == str(location)]
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

    # 收集各类需求
    demand_rows = []

    # 1. SDL需求（使用索引）
    demand_rows.extend(_collect_sdl_demands(
        supply_demand_log, material, location, upstream,
        sim_date, horizon_end, leadtime_for_row,
        sdl_index=sdl_index
    ))

    # 2. 安全库存需求（使用索引）
    demand_rows.extend(_collect_safety_stock_demands(
        safety_stock, material, location, upstream,
        sim_date, horizon_end, leadtime_for_row,
        ss_index=ss_index
    ))

    # 3. 订单需求（使用索引）
    order_df = config.get('OrderLog', pd.DataFrame())
    demand_rows.extend(_collect_order_demands(
        order_df, material, location, upstream,
        sim_date, horizon_end, leadtime_for_row,
        order_index=order_index
    ))

    # 4. GAP传递需求
    demand_rows.extend(_collect_gap_demands(
        up_gap_buffer, deploy_cfg, material, location, upstream,
        sim_date, horizon_end, leadtime_for_row
    ))

    return demand_rows


def collect_node_demands_fast(
    material: str,
    location: str,
    sim_date: pd.Timestamp,
    config: dict,
    up_gap_buffer: dict,
    horizon_cache: dict,
    sdl_index: Optional[Dict[tuple, pd.DataFrame]] = None,
    ss_index: Optional[Dict[tuple, pd.DataFrame]] = None,
    order_index: Optional[Dict[tuple, pd.DataFrame]] = None
) -> List[dict]:
    """
    使用预计算horizon缓存的快速需求收集。
    
    与collect_node_demands逻辑完全一致，但使用预计算的horizon参数。
    
    参数：
        material: 物料编码
        location: 位置编码
        sim_date: 仿真日期
        config: 配置字典
        up_gap_buffer: 上游缺口缓冲区
        horizon_cache: 预计算的horizon缓存
        sdl_index: SDL预建索引
        ss_index: SafetyStock预建索引
        order_index: OrderLog预建索引
    
    返回：
        list: 需求行列表
    """
    mat_str = str(material)
    loc_str = str(location)
    
    # 从预计算缓存获取horizon参数
    cache_entry = horizon_cache.get((mat_str, loc_str))
    if cache_entry is None:
        # 回退到原始方法
        return collect_node_demands(
            material, location, sim_date, config, up_gap_buffer,
            sdl_index=sdl_index, ss_index=ss_index, order_index=order_index
        )
    
    upstream = cache_entry['upstream']
    horizon_end = cache_entry['horizon_end']
    leadtime_for_row = cache_entry['leadtime_for_row']
    
    supply_demand_log = config['SupplyDemandLog']
    safety_stock = config['SafetyStock']
    deploy_cfg = config['DeployConfig']
    
    # 收集各类需求
    demand_rows = []

    # 1. SDL需求
    demand_rows.extend(_collect_sdl_demands(
        supply_demand_log, mat_str, loc_str, upstream,
        sim_date, horizon_end, leadtime_for_row,
        sdl_index=sdl_index
    ))

    # 2. 安全库存需求
    demand_rows.extend(_collect_safety_stock_demands(
        safety_stock, mat_str, loc_str, upstream,
        sim_date, horizon_end, leadtime_for_row,
        ss_index=ss_index
    ))

    # 3. 订单需求
    order_df = config.get('OrderLog', pd.DataFrame())
    demand_rows.extend(_collect_order_demands(
        order_df, mat_str, loc_str, upstream,
        sim_date, horizon_end, leadtime_for_row,
        order_index=order_index
    ))

    # 4. GAP传递需求
    demand_rows.extend(_collect_gap_demands(
        up_gap_buffer, deploy_cfg, mat_str, loc_str, upstream,
        sim_date, horizon_end, leadtime_for_row
    ))

    return demand_rows
