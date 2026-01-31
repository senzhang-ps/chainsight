"""
Module3 MRP模拟核心逻辑模块。

负责MRP每日模拟的主流程控制。

优化历史:
- v1.0: 基础实现
- v2.0: 添加 ThreadPoolExecutor 并行处理
- v2.1: 添加 DataIndexer 预索引优化，将 O(n*m) 过滤降为 O(1) 查找
- v2.2: 动态CPU配置，使用90%CPU资源
- v2.3: 添加DuckDB批量计算选项，优化大层级节点处理
"""

import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data_indexer import DataIndexer, create_simulation_indexer
from ...utils.cpu_config import get_optimal_workers
from .layer_assignment import assign_location_layers
from .node_processor import NodeProcessor
from .utils import (
    build_ptf_lsk_cache,
    normalize_identifiers,
    apply_moq_rv,
    lookup_moq_rv_three_keys,
    apportion_largest_remainder,
)
from .constants import (
    USE_DUCKDB_BATCH_CALCULATION,
    BATCH_CALCULATION_THRESHOLD,
    DEMAND_ELEMENT_AO,
    DEMAND_ELEMENT_FORECAST,
    DEMAND_ELEMENT_SAFETY,
)


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
    
    # 创建数据索引器（预处理优化）
    t_index = time.perf_counter()
    data_indexer = create_simulation_indexer(
        beginning_inventory_df=beginning_inventory_df,
        in_transit_df=in_transit_df,
        delivery_gr_df=delivery_gr_df,
        future_production_df=future_prod_df,
        today_shipment_df=daily_shipment_df,
        open_deployment_df=open_deployment_df,
        supply_demand_df=daily_supply_demand_df,
        safety_stock_df=safety_stock_df,
        order_df=daily_order_df,
        delivery_shipment_df=delivery_shipment_df,
    )
    ctx['data_indexer'] = data_indexer
    # print(f"[M3] DataIndexer built in {time.perf_counter() - t_index:.3f}s")

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
    
    # Build location_layer_map as {(material, location): layer} - matching baseline
    location_layer_map: dict[tuple[str, str], int] = {}
    for row in location_layer_df.itertuples(index=False):
        mat = getattr(row, 'material', '')  # type: ignore[attr-defined]
        loc = getattr(row, 'location', '')  # type: ignore[attr-defined]
        lyr = getattr(row, 'layer', 0)  # type: ignore[attr-defined]
        location_layer_map[(str(mat), str(loc))] = int(lyr)
    
    all_layers = sorted(location_layer_df['layer'].unique(), reverse=True) if not location_layer_df.empty else []

    ptf_lsk_cache = {}
    if m4_mlcfg_df is not None and not m4_mlcfg_df.empty:
        ptf_lsk_cache = build_ptf_lsk_cache(m4_mlcfg_df)

    return {
        'active_network': active_network,
        'location_layer_df': location_layer_df,
        'location_layer_map': location_layer_map,
        'all_layers': all_layers,
        'ptf_lsk_cache': ptf_lsk_cache,
    }


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
    """处理单个层级（支持批量和并行两种模式）。"""
    parent_accum = defaultdict(lambda: {'AO': 0.0, 'FC': 0.0, 'SS': 0.0})

    # Build layer nodes using original logic:
    # 1. Base nodes from location_layer_df at this layer
    # 2. Gap nodes from downstream_gaps at this layer
    location_layer_df = ctx['location_layer_df']
    location_layer_map = ctx['location_layer_map']
    
    base_nodes_df = location_layer_df[location_layer_df['layer'] == layer][['material', 'location']]
    base_pairs = {
        (str(row.material), str(row.location))
        for row in base_nodes_df.itertuples(index=False)
    }
    
    gap_pairs = {
        (str(mat), str(loc))
        for (mat, loc) in downstream_gaps.keys()
        if location_layer_map.get((str(mat), str(loc)), None) == layer
    }
    
    all_pairs = base_pairs | gap_pairs
    if not all_pairs:
        return [], parent_accum
    
    layer_nodes = (
        pd.DataFrame(list(all_pairs), columns=['material', 'location'])
        .sort_values(['material', 'location'])
        .reset_index(drop=True)
    )
    
    num_nodes = len(layer_nodes)
    
    # 根据节点数量选择处理策略
    if USE_DUCKDB_BATCH_CALCULATION and num_nodes >= BATCH_CALCULATION_THRESHOLD:
        # 使用批量向量化处理
        return _process_layer_batch(
            layer, ctx, downstream_gaps, sim_date, layer_nodes, **data_dfs
        )
    else:
        # 使用原有的并行处理
        return _process_layer_parallel(
            layer, ctx, downstream_gaps, sim_date, layer_nodes, **data_dfs
        )


def _process_layer_parallel(
    layer: int,
    ctx: dict,
    downstream_gaps: dict,
    sim_date: pd.Timestamp,
    layer_nodes: pd.DataFrame,
    **data_dfs
) -> Tuple[list, dict]:
    """并行处理单个层级（原有逻辑）。"""
    parent_accum = defaultdict(lambda: {'AO': 0.0, 'FC': 0.0, 'SS': 0.0})

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
        # 动态获取worker数量（使用90% CPU资源）
        n_workers = get_optimal_workers(len(layer_nodes))
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


def _process_layer_batch(
    layer: int,
    ctx: dict,
    downstream_gaps: dict,
    sim_date: pd.Timestamp,
    layer_nodes: pd.DataFrame,
    **data_dfs
) -> Tuple[list, dict]:
    """批量向量化处理单个层级（优化版本）。"""
    t0 = time.perf_counter()
    parent_accum = defaultdict(lambda: {'AO': 0.0, 'FC': 0.0, 'SS': 0.0})
    
    try:
        # 尝试使用 DuckDB 批量计算
        from .duckdb_batch_calculator import batch_calculate_net_demand_duckdb
        
        # 准备节点列表
        nodes = [(str(row.material), str(row.location)) for row in layer_nodes.itertuples()]
        
        # 获取所有节点的 horizon
        horizons = {}
        for material, location in nodes:
            horizons[(material, location)] = _get_node_horizon(
                ctx, material, location, sim_date, data_dfs
            )
        
        # 调用批量计算
        gaps = batch_calculate_net_demand_duckdb(
            nodes=nodes,
            sim_date=sim_date,
            beginning_inventory_df=data_dfs.get('beginning_inventory_df', pd.DataFrame()),
            in_transit_df=data_dfs.get('in_transit_df', pd.DataFrame()),
            delivery_gr_df=data_dfs.get('delivery_gr_df', pd.DataFrame()),
            future_production_df=data_dfs.get('future_production_df', pd.DataFrame()),
            today_shipment_df=data_dfs.get('daily_shipment_df', pd.DataFrame()),
            open_deployment_df=data_dfs.get('open_deployment_df', pd.DataFrame()),
            supply_demand_df=data_dfs.get('daily_supply_demand_df', pd.DataFrame()),
            safety_stock_df=data_dfs.get('safety_stock_df', pd.DataFrame()),
            order_df=data_dfs.get('daily_order_df'),
            downstream_gaps=downstream_gaps,
            horizons=horizons,
            delivery_shipment_df=data_dfs.get('delivery_shipment_df'),
        )
        
        # 构建记录和父节点累积
        all_records = []
        req_date = sim_date + pd.Timedelta(days=1)
        
        for (material, location), (ao_gap, fc_gap, ss_gap) in gaps.items():
            horizon = horizons.get((material, location), 1)
            
            # 构建记录（仅在gap > 0时创建）
            if ao_gap > 0:
                all_records.append({
                    'material': material,
                    'location': location,
                    'requirement_date': req_date,
                    'quantity': -ao_gap,
                    'demand_element': DEMAND_ELEMENT_AO,
                    'layer': layer,
                    'simulation_date': sim_date,
                    'horizon_days': horizon,
                })
            if fc_gap > 0:
                all_records.append({
                    'material': material,
                    'location': location,
                    'requirement_date': req_date,
                    'quantity': -fc_gap,
                    'demand_element': DEMAND_ELEMENT_FORECAST,
                    'layer': layer,
                    'simulation_date': sim_date,
                    'horizon_days': horizon,
                })
            if ss_gap > 0:
                all_records.append({
                    'material': material,
                    'location': location,
                    'requirement_date': req_date,
                    'quantity': -ss_gap,
                    'demand_element': DEMAND_ELEMENT_SAFETY,
                    'layer': layer,
                    'simulation_date': sim_date,
                    'horizon_days': horizon,
                })
            
            # 计算父节点缺口（应用MOQ/RV和最大余数法分配，与NodeProcessor保持一致）
            upstream = _get_upstream(ctx, material, location)
            if upstream:
                parent_key = (material, upstream)
                components = [
                    ('AO', max(0.0, ao_gap)),
                    ('FC', max(0.0, fc_gap)),
                    ('SS', max(0.0, ss_gap)),
                ]
                total_gap = sum(v for _, v in components)
                if total_gap > 0:
                    # 查找MOQ/RV配置
                    moq, rv = lookup_moq_rv_three_keys(
                        data_dfs.get('deploy_config_df'),
                        material, upstream, location
                    )
                    # 应用MOQ/RV计算目标值
                    target = apply_moq_rv(total_gap, moq, rv, is_cross_node=True)
                    if target > 0:
                        # 使用最大余数法分配到各类型
                        base_vals = [v for _, v in components]
                        apportion = apportion_largest_remainder(base_vals, target)
                        for (de, _), q in zip(components, apportion):
                            parent_accum[parent_key][de] += float(q)
        
        elapsed = time.perf_counter() - t0
        print(f"[M3] batch layer {layer}: {len(nodes)} nodes in {elapsed:.3f}s")
        
        return all_records, parent_accum
        
    except Exception as e:
        # 批量处理失败，回退到并行处理
        print(f"[M3] batch processing failed, fallback to parallel: {e}")
        return _process_layer_parallel(
            layer, ctx, downstream_gaps, sim_date, layer_nodes, **data_dfs
        )


def _get_node_horizon(ctx: dict, material: str, location: str, 
                      sim_date: pd.Timestamp, data_dfs: dict) -> int:
    """获取节点的horizon值。"""
    from .lead_time import (
        compute_root_horizon,
        determine_lead_time,
        infer_sending_location_type,
    )
    
    network_candidates = ctx['active_network'][
        (ctx['active_network']['material'] == material) &
        (ctx['active_network']['location'] == location)
    ]
    
    if not network_candidates.empty:
        row = network_candidates.iloc[0]
        upstream = row['sourcing']
        
        if pd.isna(upstream) or str(upstream).strip() == '':
            # 根节点 - use (material, location) key
            if ctx['location_layer_map'].get((str(material), str(location)), -1) == 0:
                return compute_root_horizon(
                    material, location,
                    data_dfs.get('lead_time_df', pd.DataFrame()),
                    data_dfs.get('m4_mlcfg_df'),
                    ctx['ptf_lsk_cache']
                )
            return 1
        
        location_type = infer_sending_location_type(
            ctx['active_network'],
            ctx['location_layer_map'],
            str(upstream), material, sim_date
        )
        horizon, _ = determine_lead_time(
            str(upstream), location, location_type,
            data_dfs.get('lead_time_df', pd.DataFrame()),
            data_dfs.get('m4_mlcfg_df'),
            material, ctx['ptf_lsk_cache']
        )
        return max(1, horizon)
    
    # 节点不在网络中：检查是否为根节点（层级 0）
    # 处理未在网络配置中显式出现的根节点物料
    if ctx['location_layer_map'].get((str(material), str(location)), -1) == 0:
        return compute_root_horizon(
            material, location,
            data_dfs.get('lead_time_df', pd.DataFrame()),
            data_dfs.get('m4_mlcfg_df'),
            ctx['ptf_lsk_cache']
        )
    
    return 1


def _get_upstream(ctx: dict, material: str, location: str) -> Optional[str]:
    """获取节点的上游位置。"""
    network_candidates = ctx['active_network'][
        (ctx['active_network']['material'] == material) &
        (ctx['active_network']['location'] == location)
    ]
    
    if not network_candidates.empty:
        upstream = network_candidates.iloc[0]['sourcing']
        if pd.notna(upstream) and str(upstream).strip():
            return str(upstream)
    
    return None


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
