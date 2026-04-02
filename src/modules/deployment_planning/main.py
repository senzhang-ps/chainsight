# -*- coding: utf-8 -*-
"""
Module5 主流程模块

提供多层级部署规划的主入口函数。

优化历史:
- v3.0: 添加向量化需求收集优化 (demand_collector_vectorized)
- v3.1: 动态CPU配置，使用90%CPU资源
"""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ...utils.cpu_config import get_optimal_workers

from .allocation import (
    apply_grouped_moq_rv,
    apply_priority_allocation_vectorized,
    apply_receiving_space_quota
)
from .cache_utils import (
    assign_location_layers,
    build_active_network_cache,
    build_deploy_config_index,
    build_lead_time_cache,
    build_order_log_index,
    build_ptf_lsk_cache,
    build_safety_stock_index,
    build_sdl_index,
    determine_lead_time,
    get_active_network,
    get_ptf_lsk,
    get_sending_location_type,
    get_upstream
)
from .constants import USE_VECTORIZED_DEMAND_COLLECTION, USE_MULTIPROCESS_DEMAND_COLLECTION, USE_HORIZON_CACHE
from .data_loader import load_config, load_integrated_config
from .demand_collector import collect_node_demands, collect_node_demands_fast
from .demand_collector_vectorized import collect_demands_batch_vectorized
from .horizon_batch_calculator import build_horizon_cache
from .multiprocess_optimizer import process_layer_multiprocess
from .inventory import (
    build_delivery_gr_dict,
    build_intransit_dicts,
    build_open_deployment_dict,
    build_open_deployment_inbound,
    build_production_dicts,
    build_shipment_dict,
    calculate_available_inventory,
    calculate_projected_inventory
)
from .push_allocation import push_softpush_allocation
from .validation import log_outputs, validate_config_before_run


def _validate_deployment_shipment_constraint(
    deployment_plan_df: pd.DataFrame,
    config: Dict,
    orchestrator: Optional[object],
    sim_date: pd.Timestamp,
    validation_log: List[Dict]
) -> None:
    """
    验证部署计划约束：deployed_qty 不超过 shipment_qty。
    
    参数：
        deployment_plan_df: 部署计划 DataFrame
        config: 配置字典
        orchestrator: Orchestrator 实例
        sim_date: 仿真日期
        validation_log: 验证日志
    """
    if deployment_plan_df.empty:
        return
    
    # 计算各地点的部署量
    deployed_qty_by_location = deployment_plan_df.groupby('sending')['deployed_qty'].sum()
    total_deployed_qty = deployed_qty_by_location.sum()
    
    # 计算各地点的订单（shipment）量
    shipment_log = config.get('ShipmentLog', pd.DataFrame())
    if shipment_log.empty:
        return
    
    # 过滤当日shipment
    shipment_log['date'] = pd.to_datetime(shipment_log['date'])
    today_shipment = shipment_log[shipment_log['date'] == sim_date]
    
    if today_shipment.empty:
        return
    
    shipment_qty_by_location = today_shipment.groupby('location')['quantity'].sum()
    total_shipment_qty = shipment_qty_by_location.sum()
    
    # 对比
    if total_deployed_qty > total_shipment_qty * 1.01:  # 允许1%的浮点数误差
        print(f"\n⚠️  [Module5] 约束警告: 部署量 > 订单量")
        print(f"    订单量: {total_shipment_qty:.0f}")
        print(f"    部署量: {total_deployed_qty:.0f}")
        print(f"    超出: {total_deployed_qty - total_shipment_qty:.0f}")
        print(f"    可能原因:")
        print(f"    1. MOQ/RV 调整导致部署量增加")
        print(f"    2. Push/SoftPush 分配产生了额外的部署")
        print(f"    3. 订单去重不当导致重复处理")
        
        validation_log.append({
            'sheet': 'Module5_Constraint',
            'row': '',
            'issue': f'Deployment quantity ({total_deployed_qty:.0f}) exceeds shipment quantity ({total_shipment_qty:.0f})',
            'severity': 'WARNING',
            'impact': f'Constraint Check - {total_deployed_qty - total_shipment_qty:.0f} units over',
            'shipment_qty': total_shipment_qty,
            'deployed_qty': total_deployed_qty
        })


def _initialize_soh_dict(
    config: dict,
    inventory_log: pd.DataFrame,
    actual_sim_start: pd.Timestamp
) -> dict:
    """
    初始化库存字典。

    参数：
        config: 配置字典
        inventory_log: 库存日志DataFrame
        actual_sim_start: 仿真开始日期

    返回：
        dict: (material, location) -> 库存量
    """
    ol_df = config.get('OrderLog', pd.DataFrame())
    mats_from_ol = (
        set(ol_df['material'].unique())
        if 'material' in ol_df.columns and not ol_df.empty else set()
    )
    locs_from_ol = (
        set(ol_df['location'].unique())
        if 'location' in ol_df.columns and not ol_df.empty else set()
    )

    all_mats = set()
    sdl = config['SupplyDemandLog']
    ss = config['SafetyStock']
    if 'material' in sdl.columns and not sdl.empty:
        all_mats |= set(sdl['material'].unique())
    if 'material' in ss.columns and not ss.empty:
        all_mats |= set(ss['material'].unique())
    all_mats |= mats_from_ol

    all_locs = set()
    if 'location' in sdl.columns and not sdl.empty:
        all_locs |= set(sdl['location'].unique())
    if 'location' in ss.columns and not ss.empty:
        all_locs |= set(ss['location'].unique())
    all_locs |= locs_from_ol

    inv_df = inventory_log[inventory_log['date'] == actual_sim_start]
    if inv_df.empty:
        print(f"[WARN] No inventory records found for sim_start: {actual_sim_start}")

    # 检查重复
    duplicates = inv_df.duplicated(subset=['material', 'location'], keep=False)
    if duplicates.any():
        dup_rows = inv_df[duplicates]
        raise ValueError(
            f"InventoryLog contains duplicate (material, location) "
            f"on sim_start:\n{dup_rows[['material', 'location', 'date']]}"
        )

    soh_dict = {(mat, loc): 0 for mat in all_mats for loc in all_locs}

    for row in inv_df.itertuples():
        soh_dict[(row.material, row.location)] = int(row.quantity)

    return soh_dict


def _process_layer_demands(
    layer: int,
    all_pairs: set,
    sim_date: pd.Timestamp,
    config: dict,
    up_gap_buffer: dict,
    ptf_lsk_cache: dict,
    lead_time_cache: dict,
    active_network_cache: dict,
    sdl_index: Optional[dict] = None,
    ss_index: Optional[dict] = None,
    order_index: Optional[dict] = None,
    deploy_config_index: Optional[dict] = None
) -> Dict[tuple, list]:
    """
    并行收集层内所有节点的需求。

    参数：
        layer: 层级
        all_pairs: (material, location)对集合
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
        dict: (material, location) -> 需求行列表
    """
    node_demands_map: Dict[tuple, list] = {}

    if not all_pairs:
        return node_demands_map

    # 优先使用向量化版本（如果启用）
    if USE_VECTORIZED_DEMAND_COLLECTION:
        try:
            return collect_demands_batch_vectorized(
                pairs=all_pairs,
                sim_date=sim_date,
                config=config,
                up_gap_buffer=up_gap_buffer,
                ptf_lsk_cache=ptf_lsk_cache,
                lead_time_cache=lead_time_cache,
                active_network_cache=active_network_cache
            )
        except Exception as e:
            print(f"  ⚠️  向量化收集需求失败，回退线程池: {e}")
    
    # 其次使用多进程版本（如果启用）
    if USE_MULTIPROCESS_DEMAND_COLLECTION:
        try:
            return process_layer_multiprocess(
                all_pairs=all_pairs,
                sim_date=sim_date,
                config=config,
                up_gap_buffer=up_gap_buffer,
                ptf_lsk_cache=ptf_lsk_cache,
                lead_time_cache=lead_time_cache,
                active_network_cache=active_network_cache,
                sdl_index=sdl_index,
                ss_index=ss_index,
                order_index=order_index,
                deploy_config_index=deploy_config_index
            )
        except Exception as e:
            print(f"  ⚠️  多进程收集需求失败，回退线程池: {e}")

    # 回退到 ThreadPoolExecutor 版本
    # 使用 horizon 预计算缓存（如果启用）
    horizon_cache = None
    if USE_HORIZON_CACHE:
        try:
            horizon_cache = build_horizon_cache(
                all_pairs=all_pairs,
                sim_date=sim_date,
                network_df=config['Network'],
                leadtime_df=config['LeadTime'],
                m4_mlcfg_df=config.get('M4_MaterialLocationLineCfg', pd.DataFrame()),
                ptf_lsk_cache=ptf_lsk_cache,
                lead_time_cache=lead_time_cache,
                active_network_cache=active_network_cache,
                location_layer_map=config.get('LocationLayerMap', {})
            )
        except Exception as e:
            print(f"  ⚠️  构建horizon缓存失败，回退原始方法: {e}")
            horizon_cache = None

    try:
        # 排序all_pairs确保遍历顺序一致
        sorted_pairs = sorted(all_pairs)
        
        # 动态获取worker数量（使用90% CPU资源）
        n_workers = get_optimal_workers(len(sorted_pairs))
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            if horizon_cache:
                # 使用快速版本
                futures = {
                    ex.submit(
                        collect_node_demands_fast,
                        mat, loc, sim_date, config, up_gap_buffer,
                        horizon_cache,
                        sdl_index, ss_index, order_index
                    ): (mat, loc)
                    for (mat, loc) in sorted_pairs
                }
            else:
                # 原始版本
                futures = {
                    ex.submit(
                        collect_node_demands,
                        mat, loc, sim_date, config, up_gap_buffer,
                        ptf_lsk_cache, lead_time_cache, active_network_cache,
                        sdl_index, ss_index, order_index, deploy_config_index
                    ): (mat, loc)
                    for (mat, loc) in sorted_pairs
                }
            for fut in as_completed(futures):
                key = futures[fut]
                try:
                    node_demands_map[key] = fut.result()
                except Exception as e:
                    print(f"  ⚠️  并行收集需求失败: {key} -> {e}")
                    node_demands_map[key] = []
    except Exception as e:
        print(f"  ⚠️  并行收集需求初始化失败，回退串行: {e}")

    return node_demands_map


def _allocate_pipeline_sources(
    demand_rows: List[dict],
    adjusted_qtys: Dict[int, int],
    loc: str,
    mat: str,
    demand_priority_map: Dict[str, int],
    future_intransit: dict,
    open_deployment_inbound: dict,
    future_production: dict
) -> None:
    """
    用pipeline supply覆盖剩余gap（向量化优化版）。

    参数：
        demand_rows: 需求行列表（会被修改）
        adjusted_qtys: 调整后的数量
        loc: 位置编码
        mat: 物料编码
        demand_priority_map: 优先级映射
        future_intransit: 未来在途
        open_deployment_inbound: 开放调拨入库
        future_production: 未来生产
    """
    if not demand_rows:
        return

    n = len(demand_rows)
    
    # 优化：使用numpy数组代替DataFrame操作
    # 提取receiving信息
    receivings = np.array([
        r.get('from_location', r.get('receiving', loc))
        for r in demand_rows
    ])
    is_self = receivings == loc
    
    # 早期退出：如果没有self行则返回
    self_mask = is_self
    if not np.any(self_mask):
        return
    
    # 提取demand_element并计算priority
    demand_elements = np.array([d['demand_element'] for d in demand_rows])
    priorities = np.array([demand_priority_map.get(de, 99) for de in demand_elements], dtype=np.int64)
    
    # 计算demand_qty数组
    demand_qtys = np.array([int(d.get('demand_qty', 0)) for d in demand_rows], dtype=np.int64)
    
    # 优化：向量化计算adjusted_qty（避免低效的lambda）
    adjusted_arr = np.array([
        int(adjusted_qtys.get(i, demand_qtys[i])) for i in range(n)
    ], dtype=np.int64)
    
    # 提取已分配量
    allocated_invcon = np.array([
        int(r.get('deployed_qty_invCon', 0) or 0) for r in demand_rows
    ], dtype=np.int64)
    plan_order_cover = np.array([
        int(r.get('deploy_qty_with_plan_order', 0) or 0) for r in demand_rows
    ], dtype=np.int64)
    
    # 计算raw_gap
    raw_gap = adjusted_arr - allocated_invcon - plan_order_cover
    
    # 初始化分配数组
    alloc_intrans = np.zeros(n, dtype=np.int64)
    alloc_odi = np.zeros(n, dtype=np.int64)
    alloc_future = np.zeros(n, dtype=np.int64)

    node_key = (mat, loc)
    pool_in_transit = int(future_intransit.get(node_key, 0) or 0)
    pool_odi = int(open_deployment_inbound.get(node_key, 0) or 0)
    pool_future_production = int(future_production.get(node_key, 0) or 0)

    def _alloc_source_numpy(raw_gap_arr, is_self_arr, priorities_arr, pool, alloc_arr):
        """使用纯numpy操作分配供给"""
        if pool <= 0:
            return raw_gap_arr, 0
        
        # 找出self行且gap>0的索引
        valid_mask = is_self_arr & (raw_gap_arr > 0)
        if not np.any(valid_mask):
            return raw_gap_arr, 0
        
        valid_idxs = np.where(valid_mask)[0]
        valid_gaps = raw_gap_arr[valid_idxs]
        valid_priorities = priorities_arr[valid_idxs]
        
        # 按优先级排序
        sort_order = np.argsort(valid_priorities)
        sorted_idxs = valid_idxs[sort_order]
        sorted_gaps = valid_gaps[sort_order]
        
        total_gap = float(sorted_gaps.sum())
        if total_gap <= 0:
            return raw_gap_arr, 0
        
        weights = sorted_gaps.astype(float) / total_gap
        shares = np.floor(pool * weights).astype(np.int64)
        shares = np.minimum(shares, sorted_gaps)
        
        # 写入分配结果
        alloc_arr[sorted_idxs] = shares
        raw_gap_arr[sorted_idxs] = sorted_gaps - shares
        
        return raw_gap_arr, int(shares.sum())

    # 分配各来源
    raw_gap, used_intrans = _alloc_source_numpy(
        raw_gap, self_mask, priorities, pool_in_transit, alloc_intrans
    )
    raw_gap, used_odi = _alloc_source_numpy(
        raw_gap, self_mask, priorities, pool_odi, alloc_odi
    )
    raw_gap, used_future = _alloc_source_numpy(
        raw_gap, self_mask, priorities, pool_future_production, alloc_future
    )

    # 更新plan_order_cover
    plan_order_cover = plan_order_cover + alloc_intrans + alloc_odi + alloc_future

    # 优化：直接写回self行（避免itertuples）
    for i in range(n):
        if self_mask[i]:
            demand_rows[i]['deploy_qty_with_plan_order'] = int(plan_order_cover[i])
            demand_rows[i]['deploy_from_in_transit'] = int(alloc_intrans[i])
            demand_rows[i]['deploy_from_open_deployment_inbound'] = int(alloc_odi[i])
            demand_rows[i]['deploy_from_future_production'] = int(alloc_future[i])


def _process_gaps_and_create_plans(
    demand_rows: List[dict],
    adjusted_qtys: Dict[int, int],
    mat: str,
    loc: str,
    sim_date: pd.Timestamp,
    config: dict,
    demand_priority_map: Dict[str, int],
    active_network_cache: dict,
    lead_time_cache: dict,
    ptf_lsk_cache: dict,
    deployment_plan_rows: List[dict],
    unfulfilled_rows: List[dict],
    up_gap_next: dict
) -> None:
    """
    处理GAP和生成调拨计划。

    参数：
        demand_rows: 需求行列表
        adjusted_qtys: 调整后的数量
        mat: 物料编码
        loc: 位置编码
        sim_date: 仿真日期
        config: 配置字典
        demand_priority_map: 优先级映射
        active_network_cache: Network缓存
        lead_time_cache: LeadTime缓存
        ptf_lsk_cache: PTF/LSK缓存
        deployment_plan_rows: 计划行列表（会被修改）
        unfulfilled_rows: 未满足行列表（会被修改）
        up_gap_next: 上游缺口（会被修改）
    """
    if not demand_rows:
        return

    network = config['Network']

    df_gap = pd.DataFrame(demand_rows).copy()
    df_gap['idx'] = np.arange(len(demand_rows))
    df_gap['receiving'] = [
        r.get('from_location', r.get('receiving', loc))
        for r in demand_rows
    ]
    df_gap['is_self'] = df_gap['receiving'] == loc
    df_gap['priority'] = df_gap['demand_element'].map(
        lambda x: demand_priority_map.get(x, 99)
    )

    # 优化：向量化生成调整后数量
    df_gap['adjusted_qty'] = df_gap['idx'].map(
        lambda i: int(adjusted_qtys.get(int(i), int(df_gap.at[int(i), 'demand_qty'])))
    )

    df_gap['allocated_invcon'] = [
        int(r.get('deployed_qty_invCon', 0) or 0)
        for r in demand_rows
    ]
    df_gap['plan_order_cover'] = [
        int(r.get('deploy_qty_with_plan_order', 0) or 0)
        for r in demand_rows
    ]
    df_gap['gap_qty'] = (
        df_gap['adjusted_qty'] -
        df_gap['allocated_invcon'] -
        df_gap['plan_order_cover']
    )

    df_gap_pos = df_gap[df_gap['gap_qty'] > 0]
    # 与 ChainSight_Dev/module5.py 保持一致：不对 df_gap_pos 进行排序
    up_loc = get_upstream(
        loc, mat, network, sim_date,
        active_network_cache=active_network_cache
    )

    if not df_gap_pos.empty:
        # 未满足记录
        for row in df_gap_pos.itertuples(index=False):
            unfulfilled_rows.append({
                'date': row.plan_deploy_date,
                'sending': loc,
                'receiving': row.receiving,
                'demand_qty': row.demand_qty,
                'demand_element': row.demand_element,
                'unfulfilled_qty': int(row.gap_qty),
                'reason': "supply shortage"
            })

        # 上游缺口传递
        if up_loc:
            for row in df_gap_pos.itertuples(index=False):
                new_demand_element = f"net demand for {row.demand_element}"
                req_dt = (
                    row.requirement_date
                    if hasattr(row, 'requirement_date') and pd.notna(row.requirement_date)
                    else row.plan_deploy_date
                )
                up_gap_next.setdefault((mat, up_loc), []).append({
                    'demand_element': new_demand_element,
                    'planned_qty': int(row.gap_qty),
                    'leadtime': int(row.leadtime),
                    'requirement_date': req_dt,
                    'location': up_loc,
                    'from_location': loc,
                    'orig_location': (
                        row.orig_location
                        if hasattr(row, 'orig_location')
                        else row.location
                    )
                })

    # 预计算lead time
    sending_location_type = get_sending_location_type(
        material=str(mat),
        sending=str(loc),
        sim_date=sim_date,
        network_df=network,
        location_layer_map=config.get('LocationLayerMap', {})
    )
    ptf_val, lsk_val = get_ptf_lsk(
        material=str(mat),
        site=str(loc),
        m4_mlcfg_df=config.get('M4_MaterialLocationLineCfg', pd.DataFrame()),
        cache=ptf_lsk_cache
    )

    unique_receivings = set()
    for d in demand_rows:
        rcv = d.get('from_location', d.get('receiving', loc))
        if rcv != loc:
            unique_receivings.add(str(rcv))

    lt_map: Dict[tuple, int] = {}
    if unique_receivings:
        for rcv in unique_receivings:
            if lead_time_cache is not None:
                base_vals = lead_time_cache.get((str(loc), str(rcv)))
                if base_vals is None:
                    pdt, gr, mct = 0, 0, 0
                else:
                    pdt, gr, mct = base_vals
            else:
                row_lt = config['LeadTime'][
                    (config['LeadTime']['sending'] == str(loc)) &
                    (config['LeadTime']['receiving'] == str(rcv))
                ]
                pdt = int(pd.to_numeric(
                    row_lt['PDT'], errors='coerce'
                ).fillna(0).iloc[0]) if not row_lt.empty else 0
                gr = int(pd.to_numeric(
                    row_lt['GR'], errors='coerce'
                ).fillna(0).iloc[0]) if not row_lt.empty else 0
                mct = int(pd.to_numeric(
                    row_lt['MCT'], errors='coerce'
                ).fillna(0).iloc[0]) if not row_lt.empty else 0

            if str(sending_location_type).lower() == 'plant':
                base_lt = max(int(mct), int(pdt) + int(gr))
                leadtime_val = max(1, int(base_lt + int(ptf_val) + int(lsk_val) - 1))
            else:
                leadtime_val = max(1, int(int(pdt) + int(gr)))

            lt_map[(str(mat), str(loc), str(rcv))] = int(leadtime_val)

    # 生成计划行
    for i, d in enumerate(demand_rows):
        receiving = d.get('from_location', d.get('receiving', loc))
        is_cross_node = (loc != receiving)
        actual_planned_qty = adjusted_qtys.get(i, d['demand_qty'])

        if loc == receiving:
            planned_delivery_date = d['plan_deploy_date']
            leadtime_for_row = 0
        else:
            planned_delivery_date = d.get(
                'requirement_date', d['plan_deploy_date']
            )
            leadtime_for_row = int(lt_map.get(
                (str(mat), str(loc), str(receiving)), 1
            ))

        plan_row = {
            'date': d['plan_deploy_date'],
            'material': mat,
            'sending': loc,
            'receiving': receiving,
            'demand_qty': d['demand_qty'],
            'demand_element': d['demand_element'],
            'planned_qty': actual_planned_qty,
            'deployed_qty_invCon': d['deployed_qty_invCon'],
            'deploy_qty_with_plan_order': d.get('deploy_qty_with_plan_order', 0),
            'deploy_from_in_transit': d.get('deploy_from_in_transit', 0),
            'deploy_from_open_deployment_inbound': d.get(
                'deploy_from_open_deployment_inbound', 0
            ),
            'deploy_from_future_production': d.get(
                'deploy_from_future_production', 0
            ),
            'planned_delivery_date': planned_delivery_date,
            'orig_location': d.get('orig_location', d['location']),
            'leadtime': leadtime_for_row,
            'is_cross_node': is_cross_node,
        }

        deployment_plan_rows.append(plan_row)


def _update_soh_dict(
    soh_dict: dict,
    deployment_plan_rows: List[dict],
    sim_date: pd.Timestamp,
    beginning_inventory: dict,
    today_production_gr: dict,
    today_intransit: dict,
    delivery_gr: dict,
    today_shipment: dict,
    stock_on_hand_log: List[dict]
) -> None:
    """
    更新库存字典为下一日的期初库存。

    参数：
        soh_dict: 库存字典（会被修改）
        deployment_plan_rows: 计划行列表
        sim_date: 仿真日期
        beginning_inventory: 期初库存
        today_production_gr: 当日生产
        today_intransit: 当日在途
        delivery_gr: 当日收货
        today_shipment: 当日发货
        stock_on_hand_log: 库存日志（会被修改）
    """
    deployed_dict = {}
    df = pd.DataFrame(deployment_plan_rows)

    if not df.empty:
        today_rows = df[df['date'] == sim_date].copy()
        if not today_rows.empty:
            # 只统计 sending != receiving 的 deployed_qty_invCon
            today_rows['deploy_qty'] = today_rows.apply(
                lambda r: r['deployed_qty_invCon'] if r['sending'] != r['receiving'] else 0,
                axis=1
            )
            deployed_dict = (
                today_rows.groupby(['material', 'sending'])['deploy_qty']
                .sum().to_dict()
            )

    all_keys = set(
        list(beginning_inventory.keys()) +
        list(today_production_gr.keys()) +
        list(today_intransit.keys()) +
        list(deployed_dict.keys()) +
        list(today_shipment.keys()) +
        list(delivery_gr.keys())
    )

    for (mat, loc) in all_keys:
        beginning_soh = beginning_inventory.get((mat, loc), 0)
        prod = today_production_gr.get((mat, loc), 0)
        intrans = today_intransit.get((mat, loc), 0)
        deliv_gr = delivery_gr.get((mat, loc), 0)
        deployed = deployed_dict.get((mat, loc), 0)
        shipped = today_shipment.get((mat, loc), 0)

        end_soh = beginning_soh + prod + intrans + deliv_gr - shipped - deployed
        soh_dict[(mat, loc)] = end_soh

        stock_on_hand_log.append({
            'material': mat,
            'location': loc,
            'date': sim_date,
            'beginning_soh': beginning_soh,
            'production': prod,
            'in_transit': intrans,
            'delivery_gr': deliv_gr,
            'today_shipment': shipped,
            'deployed_qty': deployed,
            'ending_soh': end_soh
        })


def main(
    input_path: str = None,
    output_path: str = None,
    sim_start: str = None,
    sim_end: str = None,
    config_dict: dict = None,
    module1_output_dir: str = None,
    module4_output_path: str = None,
    orchestrator: object = None,
    current_date: str = None,
    skip_file_output: bool = False,
    module1_result: dict = None,
    module4_result: dict = None
) -> dict:
    """
    Module5 主入口：多层级部署规划。

    支持两种运行模式：
    - 独立模式：使用Excel文件
    - 集成模式：使用各模块/Orchestrator视图

    参数：
        input_path: 输入Excel路径（独立模式）
        output_path: 输出Excel路径
        sim_start: 仿真开始日期（独立模式）
        sim_end: 仿真结束日期（独立模式）
        config_dict: 配置字典（集成模式）
        module1_output_dir: Module1输出目录
        module4_output_path: Module4输出文件路径（当module4_result为None时使用）
        orchestrator: Orchestrator实例
        current_date: 当前日期（集成模式）
        skip_file_output: 是否跳过文件输出
        module1_result: Module1运行结果（内存数据）
        module4_result: Module4运行结果（内存数据），优先使用此参数获取生产计划

    返回：
        dict: 运行结果，包含deployment_plan等
    """
    # 判断运行模式
    if config_dict is not None:
        current_date_obj = (
            pd.to_datetime(current_date) if current_date else None
        )
        config = load_integrated_config(
            config_dict, module1_output_dir, module4_output_path,
            orchestrator, current_date_obj,
            module1_result=module1_result,
            module4_result=module4_result
        )
        sim_dates = (
            [current_date_obj] if current_date_obj
            else pd.date_range(sim_start, sim_end, freq='D')
        )

        if output_path is None:
            output_path = (
                f"./Module5Output_{current_date_obj.strftime('%Y%m%d')}.xlsx"
                if current_date_obj else "./Module5Output.xlsx"
            )
    else:
        config = load_config(input_path)
        sim_dates = pd.date_range(sim_start, sim_end, freq='D')

    # 校验配置
    validation_log = list(config.get('ValidationLog', []))
    validate_config_before_run(config, validation_log)

    network = config['Network']
    inventory_log = config['InventoryLog']
    production_plan = config['ProductionPlan']
    in_transit = config['InTransit']
    demand_priority = config['DemandPriority']
    receiving_space = config['ReceivingSpace']

    # 构建层级映射 - per-material, per-location
    network_layers = assign_location_layers(network)
    location_to_layer: Dict[tuple, int] = {}
    for row in network_layers.itertuples(index=False):
        mat = getattr(row, 'material', '')  # type: ignore[attr-defined]
        loc = getattr(row, 'location', '')  # type: ignore[attr-defined]
        lyr = getattr(row, 'layer', 0)  # type: ignore[attr-defined]
        location_to_layer[(str(mat), str(loc))] = int(lyr)
    layer_list = sorted(set(location_to_layer.values()), reverse=True)
    demand_priority_map = dict(zip(
        demand_priority['demand_element'], demand_priority['priority']
    ))
    config['LocationLayerMap'] = location_to_layer

    # 构建缓存
    ptf_lsk_cache = build_ptf_lsk_cache(
        config.get('M4_MaterialLocationLineCfg', pd.DataFrame())
    )
    lead_time_cache = build_lead_time_cache(
        config.get('LeadTime', pd.DataFrame())
    )
    active_network_cache = build_active_network_cache(network)

    # 构建DataFrame索引（优化重复过滤）
    sdl_index = build_sdl_index(config['SupplyDemandLog'])
    ss_index = build_safety_stock_index(config['SafetyStock'])
    order_index = build_order_log_index(config.get('OrderLog', pd.DataFrame()))
    deploy_config_index = build_deploy_config_index(config['DeployConfig'])

    print(
        f"✅缓存已初始化: PTF/LSK={len(ptf_lsk_cache)} | "
        f"LeadTime={len(lead_time_cache)} | Network={len(active_network_cache)}"
    )
    print(
        f"✅索引已构建: SDL={len(sdl_index)} | SS={len(ss_index)} | "
        f"Order={len(order_index)} | DeployCfg={len(deploy_config_index)}"
    )

    # 初始化库存
    actual_sim_start = (
        sim_dates[0] if hasattr(sim_dates, '__getitem__')
        else pd.to_datetime(sim_start)
    )
    soh_dict = _initialize_soh_dict(config, inventory_log, actual_sim_start)

    deployment_plan_rows = []
    unfulfilled_rows = []
    stock_on_hand_log = []
    up_gap_buffer = {}
    
    # 构建 shipment 索引用于约束检查
    shipment_log_df = config.get('ShipmentLog', pd.DataFrame())
    shipment_qty_index = {}  # (material, location, date) -> quantity
    if not shipment_log_df.empty:
        shipment_log_df = shipment_log_df.copy()
        shipment_log_df['date'] = pd.to_datetime(shipment_log_df['date'])
        for _, row in shipment_log_df.iterrows():
            key = (str(row['material']), str(row['location']), row['date'])
            shipment_qty_index[key] = shipment_qty_index.get(key, 0) + int(row.get('quantity', 0))

    for sim_date in sim_dates:
        day_start = time.perf_counter()
        beginning_inventory = soh_dict.copy()
        
        # 计算当日 shipment 总量上限（用于约束部署量不超过订单量）
        today_shipment_qty_total = sum(
            qty for (mat, loc, dt), qty in shipment_qty_index.items()
            if dt == sim_date
        )

        # 构建生产字典
        today_production_gr, future_production = build_production_dicts(
            production_plan, sim_date
        )

        # 构建在途字典
        today_intransit, future_intransit = build_intransit_dicts(
            in_transit, sim_date
        )

        # 构建其他字典
        delivery_gr_data = config.get('DeliveryGR', pd.DataFrame())
        today_shipment_data = config.get('TodayShipment', pd.DataFrame())
        open_deployment_data = config.get('OpenDeployment', pd.DataFrame())

        delivery_gr = build_delivery_gr_dict(delivery_gr_data, sim_date)
        today_shipment = build_shipment_dict(today_shipment_data, sim_date)
        open_deployment = build_open_deployment_dict(open_deployment_data)
        open_deployment_inbound = build_open_deployment_inbound(
            open_deployment_data
        )

        # 计算库存
        projected_soh = calculate_projected_inventory(
            beginning_inventory=beginning_inventory,
            in_transit=today_intransit,
            delivery_gr=delivery_gr,
            today_production_gr=today_production_gr,
            future_production=future_production,
            today_shipment=today_shipment,
            open_deployment=open_deployment
        )

        dynamic_soh = calculate_available_inventory(
            beginning_inventory=beginning_inventory,
            delivery_gr=delivery_gr,
            today_production_gr=today_production_gr,
            today_shipment=today_shipment,
            open_deployment=open_deployment,
            open_deployment_inbound=open_deployment_inbound
        )

        demand_collect_total_start = time.perf_counter()
        demand_collect_only_elapsed = 0.0
        up_gap_next = {}
        global_node_demands_map: Dict[tuple, list] = {}

        # 预计算materials_union
        sdl = config['SupplyDemandLog']
        materials_union = set(
            sdl['material'].unique()
        ) if 'material' in sdl.columns and not sdl.empty else set()
        if 'OrderLog' in config and not config['OrderLog'].empty:
            materials_union |= set(config['OrderLog']['material'].unique())
        if not config['SafetyStock'].empty:
            materials_union |= set(config['SafetyStock']['material'].unique())

        # 按层级处理
        for layer in layer_list:
            # 计算当前层级的pairs - 与基准版本一致
            # `location_to_layer` 的键为 `(material, location)` 元组
            base_pairs = set(
                (mat, loc)
                for (mat, loc), lyr in location_to_layer.items()
                if lyr == layer
            )
            # 补充 gap 缓冲中的节点
            gap_pairs = set(
                (mat, loc)
                for (mat, loc) in up_gap_buffer
                if location_to_layer.get((mat, loc), None) == layer
            )
            all_pairs = base_pairs | gap_pairs

            # 并行收集需求
            layer_collect_start = time.perf_counter()
            node_demands_map = _process_layer_demands(
                layer, all_pairs, sim_date, config, up_gap_buffer,
                ptf_lsk_cache, lead_time_cache, active_network_cache,
                sdl_index, ss_index, order_index, deploy_config_index
            )
            demand_collect_only_elapsed += (
                time.perf_counter() - layer_collect_start
            )

            for k, v in node_demands_map.items():
                global_node_demands_map[k] = v

            # 处理每个节点 - 与源码保持一致，不排序
            for mat, loc in all_pairs:
                node_key = (mat, loc)
                current_stock = dynamic_soh.get(node_key, 0)

                demand_rows = node_demands_map.get((mat, loc))
                if demand_rows is None:
                    demand_rows = collect_node_demands(
                        mat, loc, sim_date, config, up_gap_buffer,
                        ptf_lsk_cache=ptf_lsk_cache,
                        lead_time_cache=lead_time_cache,
                        active_network_cache=active_network_cache,
                        sdl_index=sdl_index,
                        ss_index=ss_index,
                        order_index=order_index,
                        deploy_config_index=deploy_config_index
                    )

                if not demand_rows:
                    continue

                # 初始化planned_qty
                for d in demand_rows:
                    d['planned_qty'] = d['demand_qty']

                # 获取当前(material, location)的订单量上限
                shipment_qty_limit = shipment_qty_index.get(
                    (str(mat), str(loc), sim_date), None
                )
                
                # 分组MOQ/RV（传入订单量上限约束）
                adjusted_qtys = apply_grouped_moq_rv(
                    demand_rows, loc, 
                    shipment_qty_limit=shipment_qty_limit
                )

                # 优先级分配
                current_stock = apply_priority_allocation_vectorized(
                    demand_rows=demand_rows,
                    adjusted_qtys=adjusted_qtys,
                    current_stock=current_stock,
                    demand_priority_map=demand_priority_map
                )

                # 初始化pipeline字段
                for d in demand_rows:
                    d.setdefault('deploy_qty_with_plan_order', 0)
                    d.setdefault('deploy_from_in_transit', 0)
                    d.setdefault('deploy_from_open_deployment_inbound', 0)
                    d.setdefault('deploy_from_future_production', 0)

                # Pipeline 供给分配
                _allocate_pipeline_sources(
                    demand_rows, adjusted_qtys, loc, mat,
                    demand_priority_map, future_intransit,
                    open_deployment_inbound, future_production
                )

                # 处理GAP和生成计划
                _process_gaps_and_create_plans(
                    demand_rows, adjusted_qtys, mat, loc, sim_date,
                    config, demand_priority_map, active_network_cache,
                    lead_time_cache, ptf_lsk_cache, deployment_plan_rows,
                    unfulfilled_rows, up_gap_next
                )

            up_gap_buffer = up_gap_next.copy()

        print(
            f"[M5] Demand collection only 用时: "
            f"{demand_collect_only_elapsed:.3f}s"
        )
        print(
            f"[M5] Demand collection+allocation 总用时: "
            f"{time.perf_counter()-demand_collect_total_start:.3f}s"
        )

        # Push/Soft-push 分配
        dynamic_soh_for_push = dynamic_soh.copy()
        plan_push = push_softpush_allocation(
            deployment_plan_rows, config, dynamic_soh_for_push, sim_date,
            ptf_lsk_cache=ptf_lsk_cache,
            lead_time_cache=lead_time_cache,
            projected_soh=projected_soh,
            node_demands_map=global_node_demands_map
        )

        if plan_push:
            deployment_plan_rows.extend(plan_push)

        # 更新库存
        _update_soh_dict(
            soh_dict, deployment_plan_rows, sim_date,
            beginning_inventory, today_production_gr, today_intransit,
            delivery_gr, today_shipment, stock_on_hand_log
        )

    # 应用接收空间配额
    deployment_plan_rows_df, unfulfilled_space = apply_receiving_space_quota(
        deployment_plan_rows, receiving_space, sim_date, demand_priority_map
    )
    unfulfilled_all = pd.DataFrame(unfulfilled_rows + unfulfilled_space)
    
    # 与code_v0保持一致：不对unfulfilled_all排序

    # 🔧 验证约束：deployed_qty 不超过 shipment_qty
    _validate_deployment_shipment_constraint(
        deployment_plan_rows_df, config, orchestrator, 
        actual_sim_start, validation_log
    )

    outputs = {
        'DeploymentPlan': deployment_plan_rows_df,
        'UnfulfilledLog': unfulfilled_all,
        'StockOnHandLog': pd.DataFrame(stock_on_hand_log),
        'Validation': pd.DataFrame(validation_log),
    }

    if not skip_file_output:
        log_outputs(output_path, outputs)

    print(f"[M5] Full day total 用时: {time.perf_counter()-day_start:.3f}s")

    return {
        'deployment_plan': deployment_plan_rows_df,
        'unfulfilled_log': unfulfilled_all,
        'stock_on_hand_log': pd.DataFrame(stock_on_hand_log),
        'validation_log': pd.DataFrame(validation_log),
        'statistics': {
            'deployment_count': len(deployment_plan_rows_df),
            'unfulfilled_count': len(unfulfilled_all),
            'processed_dates': (
                len(sim_dates) if isinstance(sim_dates, list) else 1
            )
        }
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Module 5: Multi-echelon Deployment Planning'
    )
    parser.add_argument('--input', required=True, help='Input config excel path')
    parser.add_argument('--output', required=True, help='Output excel path')
    parser.add_argument(
        '--sim_start', required=True,
        help='Simulation start date, YYYY-MM-DD'
    )
    parser.add_argument(
        '--sim_end', required=True,
        help='Simulation end date, YYYY-MM-DD'
    )
    args = parser.parse_args()

    main(args.input, args.output, args.sim_start, args.sim_end)
