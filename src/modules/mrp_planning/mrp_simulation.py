"""
Module3 MRP模拟核心逻辑模块。

负责MRP每日模拟的主流程控制。
"""

import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .layer_assignment import assign_location_layers
from .node_processor import NodeProcessor
from .utils import build_ptf_lsk_cache, normalize_identifiers


def run_mrp_layered_simulation_daily(
    sim_date: pd.Timestamp,
    daily_supply_demand_df: pd.DataFrame,
    daily_order_df: pd.DataFrame,
    daily_shipment_df: pd.DataFrame,
    safety_stock_df: pd.DataFrame,
    beginning_inventory_df: pd.DataFrame,
    in_transit_df: pd.DataFrame,
    delivery_gr_df: pd.DataFrame,
    all_production_df: pd.DataFrame,
    open_deployment_df: pd.DataFrame,
    network_df: pd.DataFrame,
    lead_time_df: pd.DataFrame,
    m4_mlcfg_df: Optional[pd.DataFrame] = None,
    delivery_shipment_df: Optional[pd.DataFrame] = None,
    deploy_config_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    运行单日MRP模拟。

    Args:
        sim_date: 模拟日期
        daily_supply_demand_df: 当日供需数据
        daily_order_df: 当日订单数据
        daily_shipment_df: 当日发货数据
        safety_stock_df: 安全库存数据
        beginning_inventory_df: 期初库存数据
        in_transit_df: 在途数据
        delivery_gr_df: 收货数据
        all_production_df: 生产计划数据
        open_deployment_df: 开放调拨数据
        network_df: 网络配置数据
        lead_time_df: 提前期数据
        m4_mlcfg_df: M4配置
        delivery_shipment_df: 发运记录
        deploy_config_df: MOQ/RV配置

    Returns:
        pd.DataFrame: 当日净需求记录
    """
    t_start = time.perf_counter()

    if network_df.empty:
        print(f"Warning: Empty network config for {sim_date}")
        return _empty_net_demand_df()

    # 初始化上下文
    ctx = _init_simulation_context(
        sim_date, network_df, m4_mlcfg_df
    )
    if ctx['active_network'].empty:
        return _empty_net_demand_df()

    # 准备生产数据
    future_prod_df = _prepare_production_df(all_production_df)

    # 按层级处理
    all_records = []
    downstream_gaps = defaultdict(lambda: {'AO': 0.0, 'FC': 0.0, 'SS': 0.0})

    for layer in ctx['all_layers']:
        layer_records, downstream_gaps = _process_layer(
            layer=layer,
            ctx=ctx,
            downstream_gaps=downstream_gaps,
            sim_date=sim_date,
            daily_supply_demand_df=daily_supply_demand_df,
            daily_order_df=daily_order_df,
            daily_shipment_df=daily_shipment_df,
            safety_stock_df=safety_stock_df,
            beginning_inventory_df=beginning_inventory_df,
            in_transit_df=in_transit_df,
            delivery_gr_df=delivery_gr_df,
            future_production_df=future_prod_df,
            open_deployment_df=open_deployment_df,
            lead_time_df=lead_time_df,
            m4_mlcfg_df=m4_mlcfg_df,
            delivery_shipment_df=delivery_shipment_df,
            deploy_config_df=deploy_config_df,
        )
        all_records.extend(layer_records)

    # 生成结果
    result_df = _build_result_df(all_records)

    elapsed = time.perf_counter() - t_start
    print(f"[M3] mrp_simulation: {elapsed:.3f}s, records={len(result_df)}")
    return result_df


def _empty_net_demand_df() -> pd.DataFrame:
    """返回空的净需求DataFrame。"""
    return pd.DataFrame({
        'material': [], 'location': [],
        'requirement_date': [], 'quantity': [],
        'demand_element': [], 'layer': []
    })


def _init_simulation_context(
    sim_date: pd.Timestamp,
    network_df: pd.DataFrame,
    m4_mlcfg_df: Optional[pd.DataFrame]
) -> dict:
    """初始化模拟上下文。"""
    network_df = normalize_identifiers(network_df)

    active_network = network_df[
        (network_df['eff_from'] <= sim_date) &
        (network_df['eff_to'] >= sim_date)
    ]

    location_layer_df = assign_location_layers(active_network)
    location_layer = dict(
        zip(location_layer_df['location'], location_layer_df['layer'])
    )
    all_layers = sorted(set(location_layer.values()), reverse=True)

    ptf_lsk_cache = {}
    if m4_mlcfg_df is not None and not m4_mlcfg_df.empty:
        ptf_lsk_cache = build_ptf_lsk_cache(m4_mlcfg_df)

    material_locations = _build_material_locations(
        active_network, location_layer, network_df
    )

    return {
        'active_network': active_network,
        'location_layer_df': location_layer_df,
        'location_layer': location_layer,
        'all_layers': all_layers,
        'ptf_lsk_cache': ptf_lsk_cache,
        'material_locations': material_locations,
    }


def _build_material_locations(
    active_network: pd.DataFrame,
    location_layer: dict,
    network_df: pd.DataFrame
) -> pd.DataFrame:
    """构建material-location组合。"""
    all_locations = set(location_layer.keys())
    all_materials = set(network_df['material'].unique())

    extended = []
    for row in active_network.itertuples():
        extended.append({
            'material': str(row.material),
            'location': str(row.location)
        })

    for location in all_locations:
        for material in all_materials:
            exists = any(
                ml['material'] == material and ml['location'] == location
                for ml in extended
            )
            if not exists:
                extended.append({
                    'material': str(material),
                    'location': str(location)
                })

    df = pd.DataFrame(extended).drop_duplicates()
    return normalize_identifiers(df)


def _prepare_production_df(all_production_df: pd.DataFrame) -> pd.DataFrame:
    """准备生产数据。"""
    if all_production_df.empty:
        return pd.DataFrame()
    if 'available_date' not in all_production_df.columns:
        return pd.DataFrame()

    df = all_production_df.copy()
    df['available_date'] = pd.to_datetime(df['available_date'])

    for col in ['produced_qty', 'uncon_planned_qty', 'quantity']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df


def _process_layer(
    layer: int,
    ctx: dict,
    downstream_gaps: dict,
    sim_date: pd.Timestamp,
    **data_dfs
) -> Tuple[list, dict]:
    """处理单个层级。"""
    parent_accum = defaultdict(lambda: {'AO': 0.0, 'FC': 0.0, 'SS': 0.0})

    layer_locs = [
        loc for loc, lyr in ctx['location_layer'].items()
        if lyr == layer
    ]
    layer_mask = ctx['material_locations']['location'].isin(layer_locs)
    layer_nodes = ctx['material_locations'][layer_mask]

    all_records = []
    records_lock = threading.Lock()
    parent_lock = threading.Lock()

    processor = NodeProcessor(
        ctx=ctx,
        downstream_gaps=downstream_gaps,
        current_layer=layer,
        sim_date=sim_date,
        **data_dfs
    )

    try:
        n_workers = min(32, max(1, len(layer_nodes)))
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(processor.process, ml): ml
                for ml in layer_nodes.itertuples()
            }

            for fut in as_completed(futures):
                try:
                    records, parent_key, parent_gaps = fut.result()
                    with records_lock:
                        all_records.extend(records)
                    if parent_key:
                        with parent_lock:
                            for k in ['AO', 'FC', 'SS']:
                                parent_accum[parent_key][k] += parent_gaps[k]
                except Exception as e:
                    print(f"[M3] task failed on layer {layer}: {e}")

    except Exception as e:
        print(f"[M3] parallel failed for layer {layer}: {e}")

    return all_records, parent_accum


def _build_result_df(all_records: list) -> pd.DataFrame:
    """构建结果DataFrame。"""
    if not all_records:
        return _empty_net_demand_df()

    df = pd.DataFrame(all_records)
    df = normalize_identifiers(df)

    group_cols = [
        'material', 'location', 'requirement_date',
        'demand_element', 'layer'
    ]
    df = df.groupby(group_cols, as_index=False).agg({
        'quantity': 'sum',
        'simulation_date': 'first',
        'horizon_days': 'first',
    })

    return df.reset_index(drop=True)
