# -*- coding: utf-8 -*-
"""
Push/Soft-Push分配模块

提供push和soft-push补货分配功能。
"""
import time
from datetime import timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .cache_utils import (
    determine_lead_time,
    get_ptf_lsk,
    get_sending_location_type
)
from .constants import DEFAULT_PUSH_LEVELS
from .demand_collector import collect_node_demands


def _calculate_receiving_ss_data(
    mat: str,
    sending: str,
    recs: list,
    sim_date: pd.Timestamp,
    config: dict,
    lead_time_cache: Optional[dict],
    ptf_lsk_cache: Optional[dict]
) -> List[dict]:
    """
    计算接收端安全库存数据。

    Args:
        mat: 物料编码
        sending: 发送端编码
        recs: 接收端列表
        sim_date: 仿真日期
        config: 配置字典
        lead_time_cache: LeadTime缓存
        ptf_lsk_cache: PTF/LSK缓存

    Returns:
        list: 接收端安全库存数据列表
    """
    safety = config['SafetyStock']
    lt_df = config['LeadTime']
    net = config['Network']

    receiving_ss_data = []

    for rec in recs:
        # 计算lead time
        sending_location_type = get_sending_location_type(
            material=str(mat),
            sending=str(sending),
            sim_date=sim_date,
            network_df=net,
            location_layer_map=config.get('LocationLayerMap', {})
        )
        leadtime, err = determine_lead_time(
            sending=str(sending),
            receiving=str(rec),
            location_type=str(sending_location_type),
            lead_time_df=lt_df,
            m4_mlcfg_df=config.get('M4_MaterialLocationLineCfg', pd.DataFrame()),
            material=str(mat),
            lead_time_cache=lead_time_cache,
            ptf_lsk_cache=ptf_lsk_cache
        )
        if err:
            leadtime = 1

        target_date = sim_date + timedelta(days=int(leadtime))

        # 获取安全库存
        ss_rec = safety[
            (safety['material'] == mat) &
            (safety['location'] == rec)
        ]
        if not ss_rec.empty:
            ss_rec = ss_rec[pd.to_datetime(ss_rec['date']) == target_date]

        ss_qty = (
            int(ss_rec['safety_stock_qty'].sum())
            if not ss_rec.empty else 0
        )

        receiving_ss_data.append({
            'receiving': rec,
            'ss_qty': ss_qty,
            'leadtime': int(leadtime),
            'planned_delivery_date': target_date
        })

    return receiving_ss_data


def _calculate_commitments(
    mat: str,
    receiving_ss_data: List[dict],
    sim_date: pd.Timestamp,
    config: dict,
    node_demands_map: Optional[Dict],
    ptf_lsk_cache: Optional[dict],
    lead_time_cache: Optional[dict]
) -> Dict[str, float]:
    """
    计算接收端的承诺消耗。

    Args:
        mat: 物料编码
        receiving_ss_data: 接收端安全库存数据
        sim_date: 仿真日期
        config: 配置字典
        node_demands_map: 节点需求映射
        ptf_lsk_cache: PTF/LSK缓存
        lead_time_cache: LeadTime缓存

    Returns:
        dict: 接收端 -> 承诺消耗量
    """
    commitments_map: Dict[str, float] = {}

    for x in receiving_ss_data:
        rec = str(x['receiving'])
        target_date = pd.to_datetime(x['planned_delivery_date'])

        # 优先使用已收集的节点需求
        if node_demands_map is not None:
            rows = node_demands_map.get((mat, rec), [])
        else:
            try:
                rows = collect_node_demands(
                    material=mat,
                    location=rec,
                    sim_date=sim_date,
                    config=config,
                    up_gap_buffer=None,
                    ptf_lsk_cache=ptf_lsk_cache,
                    lead_time_cache=lead_time_cache,
                    active_network_cache=None
                )
            except Exception:
                rows = []

        commit_qty = 0.0
        for r in rows:
            de = str(r.get('demand_element', '')).lower()
            rq = int(r.get('planned_qty', r.get('demand_qty', 0)) or 0)
            req_dt = pd.to_datetime(r.get('requirement_date', sim_date))

            # 仅计未来窗口内的AO/normal/forecast
            if de in ['ao', 'normal', 'forecast']:
                if (req_dt > sim_date) and (req_dt <= target_date):
                    commit_qty += rq
            # 计入窗口末日的安全库存目标
            elif de == 'safety':
                if req_dt == target_date:
                    commit_qty += rq

        commitments_map[rec] = commit_qty

    return commitments_map


def _select_push_level(
    receiving_ss_data: List[dict],
    pi_map: Dict[str, float],
    available_soh: float,
    push_levels: List[float]
) -> float:
    """
    选择最高可行挡位。

    Args:
        receiving_ss_data: 接收端安全库存数据
        pi_map: 接收端库存基线映射
        available_soh: 可用库存
        push_levels: 挡位列表

    Returns:
        float: 选中的挡位
    """
    feasible_level = None

    for level in push_levels:
        need_sum = 0.0
        for x in receiving_ss_data:
            ssq = float(x['ss_qty'] or 0)
            if ssq <= 0:
                continue
            pi = pi_map.get(x['receiving'], 0.0)
            need_sum += max(0.0, level * ssq - pi)

        if need_sum <= float(available_soh) + 1e-9:
            feasible_level = level
        else:
            break

    if feasible_level is None:
        feasible_level = push_levels[0]

    return feasible_level


def _allocate_push_quantities(
    receiving_ss_data: List[dict],
    pi_map: Dict[str, float],
    feasible_level: float,
    available_soh: float
) -> List[tuple]:
    """
    按比例分配push数量。

    Args:
        receiving_ss_data: 接收端安全库存数据
        pi_map: 接收端库存基线映射
        feasible_level: 选中的挡位
        available_soh: 可用库存

    Returns:
        list: (接收端数据, 分配数量) 元组列表
    """
    needs = []
    for x in receiving_ss_data:
        ssq = float(x['ss_qty'] or 0)
        if ssq <= 0:
            needs.append((x, 0.0))
            continue
        pi = pi_map.get(x['receiving'], 0.0)
        needs.append((x, max(0.0, feasible_level * ssq - pi)))

    total_need = sum(n for _, n in needs)
    allocated = []

    if total_need > 0:
        for x, need in needs:
            share = (
                (available_soh * need / total_need)
                if total_need > 0 else 0.0
            )
            q = int(np.floor(share))
            allocated.append((x, q))

    return allocated


def push_softpush_allocation(
    deployment_plan_rows: List[dict],
    config: dict,
    dynamic_soh: dict,
    sim_date: pd.Timestamp,
    ptf_lsk_cache: Optional[dict] = None,
    lead_time_cache: Optional[dict] = None,
    projected_soh: Optional[dict] = None,
    node_demands_map: Optional[Dict] = None
) -> List[dict]:
    """
    执行push/soft-push补货分配。

    在当日所有非push需求已满足的前提下，使用剩余可用库存执行补货分配。
    采用"挡位（bucket）+ 比例兜底"的方法。

    Args:
        deployment_plan_rows: 当日已生成的分配计划
        config: 配置字典
        dynamic_soh: 当日真实可用库存
        sim_date: 仿真日期
        ptf_lsk_cache: PTF/LSK缓存
        lead_time_cache: LeadTime缓存
        projected_soh: 接收端当日预测库存
        node_demands_map: 节点需求映射

    Returns:
        list: push/soft-push计划行列表
    """
    t0 = time.perf_counter()
    plan_rows_push = []

    pushpull = config['PushPullModel']
    safety = config['SafetyStock']
    deploy_cfg = config['DeployConfig']
    net = config['Network']

    # 统计当日已分配的库存
    allocated_inventory: Dict[tuple, int] = {}
    for r in deployment_plan_rows:
        if 'push' in str(r.get('demand_element', '')).lower():
            continue
        if pd.to_datetime(r.get('date')) != sim_date:
            continue

        mat = r.get('material')
        snd = r.get('sending')
        if mat is None or snd is None:
            continue

        key = (mat, snd)
        qty = int(r.get('deployed_qty_invCon', 0) or 0)
        if qty > 0:
            allocated_inventory[key] = allocated_inventory.get(key, 0) + qty

    # 收集需要处理的组合（使用set，与code_v0行为一致）
    group_keys = {
        (r['material'], r['sending'])
        for r in deployment_plan_rows
        if r.get('material') and r.get('sending')
    }

    # 获取push levels
    push_levels = config.get('M5_PushLevels', DEFAULT_PUSH_LEVELS)
    try:
        push_levels = sorted([float(l) for l in push_levels])
    except Exception:
        push_levels = DEFAULT_PUSH_LEVELS

    # 逐组处理
    for mat, sending in group_keys:
        # 检查是否有未满足的非push需求
        pending_gap = any(
            (
                r.get('material') == mat and
                r.get('sending') == sending and
                'push' not in str(r.get('demand_element', '')).lower() and
                pd.to_datetime(r.get('date')) == sim_date and
                int(r.get('deployed_qty_invCon', 0) or 0) <
                int(r.get('planned_qty', 0) or 0)
            )
            for r in deployment_plan_rows
        )
        if pending_gap:
            continue

        # 检查push/soft push配置
        row_pp = pushpull[
            (pushpull['material'] == mat) &
            (pushpull['sending'] == sending)
        ]
        if row_pp.empty:
            continue

        model = str(row_pp.iloc[0]['model']).strip().lower()
        if model not in ['push', 'soft push']:
            continue

        # 计算剩余库存
        total_soh = int(dynamic_soh.get((mat, sending), 0) or 0)
        already_allocated = int(allocated_inventory.get((mat, sending), 0) or 0)
        soh = max(0, total_soh - already_allocated)
        if soh <= 0:
            continue

        # 读取LSK/Day
        row_cfg = deploy_cfg[
            (deploy_cfg['material'] == mat) &
            (deploy_cfg['sending'] == sending)
        ]
        lsk = int(row_cfg.iloc[0]['lsk']) if not row_cfg.empty else 1

        # soft-push需先保留本节点当日safety
        sending_ss = 0
        if model == 'soft push':
            ss_self = safety[
                (safety['material'] == mat) &
                (safety['location'] == sending)
            ]
            if not ss_self.empty:
                ss_self = ss_self[
                    pd.to_datetime(ss_self['date']) == sim_date
                ]
            if not ss_self.empty:
                sending_ss = int(ss_self['safety_stock_qty'].sum())

        available_soh = (
            soh if model == 'push'
            else max(0, soh - sending_ss)
        )
        if available_soh <= 0:
            continue

        # 找下游receiving
        recs = net[
            (net['material'] == mat) &
            (net['sourcing'] == sending)
        ]['location'].dropna().unique().tolist()
        if not recs:
            continue

        # 计算接收端安全库存数据
        receiving_ss_data = _calculate_receiving_ss_data(
            mat, sending, recs, sim_date, config,
            lead_time_cache, ptf_lsk_cache
        )

        total_ss = sum(x['ss_qty'] for x in receiving_ss_data)
        if total_ss <= 0:
            continue

        # 构建库存基线
        if projected_soh is not None:
            pi_map = {
                x['receiving']: float(
                    projected_soh.get((mat, x['receiving']), 0) or 0
                )
                for x in receiving_ss_data
            }
        else:
            pi_map = {
                x['receiving']: float(
                    dynamic_soh.get((mat, x['receiving']), 0) or 0
                )
                for x in receiving_ss_data
            }

        # 计算承诺消耗
        commitments_map = _calculate_commitments(
            mat, receiving_ss_data, sim_date, config,
            node_demands_map, ptf_lsk_cache, lead_time_cache
        )

        # 扣减承诺消耗
        for rec, commit in commitments_map.items():
            pi_map[rec] = float(max(
                0.0, (pi_map.get(rec, 0.0) or 0.0) - float(commit or 0.0)
            ))

        # 选择挡位
        feasible_level = _select_push_level(
            receiving_ss_data, pi_map, available_soh, push_levels
        )

        # 分配数量
        allocated = _allocate_push_quantities(
            receiving_ss_data, pi_map, feasible_level, available_soh
        )

        # 生成计划行
        for x, qty in allocated:
            if qty <= 0:
                continue

            plan = {
                'date': sim_date,
                'material': mat,
                'sending': sending,
                'receiving': x['receiving'],
                'demand_qty': 0,
                'demand_element': (
                    'push replenishment' if model == 'push'
                    else 'soft push replenishment'
                ),
                'planned_qty': int(qty),
                # 注意: 移除deployed_qty_invCon_push以与Dev版保持一致
                'deployed_qty_invCon': int(qty),
                'planned_delivery_date': x['planned_delivery_date'],
                'orig_location': x['receiving'],
                'leadtime': int(x['leadtime']),
                'is_cross_node': True
            }
            plan_rows_push.append(plan)

    elapsed = time.perf_counter() - t0
    print(
        f"[M5] push_softpush_allocation 用时: {elapsed:.3f}s，"
        f"生成行数: {len(plan_rows_push)}"
    )

    return plan_rows_push
