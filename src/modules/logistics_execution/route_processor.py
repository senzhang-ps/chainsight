# -*- coding: utf-8 -*-
"""
物流执行模块 - 路线处理与车辆装载
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .capacity_manager import (
    get_optimal_truck_sequence,
    get_truck_capacity,
    get_truck_config,
    get_truck_spec,
)
from .delivery_processor import (
    calculate_actual_delivery_date,
    calculate_lead_time,
    create_bypass_record,
    create_delivery_record,
    create_unsatisfied_record,
    sample_delivery_delay,
    should_bypass_mdq,
)
from .expression_evaluator import SafeExpressionEvaluator
from .inventory_manager import (
    calculate_inventory_limit,
    update_inventory_after_load,
)
from .vehicle_packer import (
    VehiclePacker,
    create_vehicle_log_entry,
    determine_trigger_cause,
    get_representative_context,
)

# DuckDB 批量优化
try:
    from .duckdb_batch_calculator import (
        batch_sample_delivery_delays_duckdb,
        is_duckdb_available as m6_is_duckdb_available,
    )
    M6_DUCKDB_AVAILABLE = True
except ImportError:
    M6_DUCKDB_AVAILABLE = False


def process_routes(
    sim_date: pd.Timestamp,
    cross_node_sorted: pd.DataFrame,
    agg_status: Dict[str, Dict],
    prepared_data: Dict[str, Any],
    run_params: Dict[str, Any],
    evaluator: SafeExpressionEvaluator,
    available_inventory: Dict[Tuple[str, str], float],
    inventory_check_enabled: bool,
    results: Dict[str, List]
) -> None:
    """
    处理所有路线。

    参数：
        sim_date: 仿真日期
        cross_node_sorted: 排序后的跨节点需求
        agg_status: 聚合状态
        prepared_data: 准备好的数据
        run_params: 运行参数
        evaluator: 表达式解析器
        available_inventory: 可用库存
        inventory_check_enabled: 是否启用库存检查
        results: 结果收集器
    """
    # 使用与 code_v0 一致的路线处理顺序：按全局排序逐行遍历并缓存路线
    processed_routes: set[Tuple[str, str]] = set()
    for row in cross_node_sorted.itertuples(index=False):
        route_key = (row.sending, row.receiving)
        if route_key in processed_routes:
            continue

        processed_routes.add(route_key)
        route_demands = cross_node_sorted[
            (cross_node_sorted['sending'] == row.sending) &
            (cross_node_sorted['receiving'] == row.receiving)
        ].copy()

        _process_single_route(
            sim_date, route_key, route_demands, agg_status,
            prepared_data, run_params, evaluator, available_inventory,
            inventory_check_enabled, results
        )


def _process_single_route(
    sim_date: pd.Timestamp,
    route_key: Tuple[str, str],
    route_demands: pd.DataFrame,
    agg_status: Dict[str, Dict],
    prepared_data: Dict[str, Any],
    run_params: Dict[str, Any],
    evaluator: SafeExpressionEvaluator,
    available_inventory: Dict[Tuple[str, str], float],
    inventory_check_enabled: bool,
    results: Dict[str, List]
) -> None:
    """
    处理单条路线。

    参数：
        各种参数
    """
    sending, receiving = route_key
    truck_cfgs = get_truck_config(prepared_data['truck_con'], sending, receiving)

    if truck_cfgs.empty:
        return

    type_seq = get_optimal_truck_sequence(truck_cfgs)
    remaining_demands = route_demands.copy()
    route_mdq = truck_cfgs['MDQ'].min() if not truck_cfgs.empty else np.nan

    # 尝试每种车型
    for truck_type in type_seq:
        if remaining_demands.empty:
            break

        remaining_demands = _process_truck_type(
            sim_date, sending, receiving, truck_type,
            truck_cfgs, remaining_demands, agg_status,
            prepared_data, run_params, evaluator,
            available_inventory, inventory_check_enabled, results
        )

    # 处理剩余未发出的需求
    _handle_remaining_demands(
        route_demands, agg_status, sending, receiving,
        sim_date, run_params['max_wait_days'], route_mdq,
        results['unsat_log']
    )


def _process_truck_type(
    sim_date: pd.Timestamp,
    sending: str,
    receiving: str,
    truck_type: str,
    truck_cfgs: pd.DataFrame,
    remaining_demands: pd.DataFrame,
    agg_status: Dict[str, Dict],
    prepared_data: Dict[str, Any],
    run_params: Dict[str, Any],
    evaluator: SafeExpressionEvaluator,
    available_inventory: Dict[Tuple[str, str], float],
    inventory_check_enabled: bool,
    results: Dict[str, List]
) -> pd.DataFrame:
    """
    处理单种车型的装载。

    参数：
        各种参数

    返回：
        剩余未处理的需求 DataFrame
    """
    # 获取车辆数量
    n_truck_total = get_truck_capacity(
        prepared_data['cap_map'], sim_date, sending, receiving, truck_type
    )

    # 获取车型配置
    conf = truck_cfgs[truck_cfgs['truck_type'] == truck_type].iloc[0]
    spec = get_truck_spec(prepared_data['spec_map'], truck_type)

    if not spec:
        return remaining_demands

    # 配置参数
    wfr_th, vfr_th = float(conf['WFR']), float(conf['VFR'])
    mdq = float(conf['MDQ']) if pd.notna(conf['MDQ']) else 0.0
    cap_w = float(spec['capacity_qty_in_weight'])
    cap_v = float(spec['capacity_qty_in_volume'])

    used = 0
    while used < n_truck_total and not remaining_demands.empty:
        # 创建装载器
        packer = VehiclePacker(cap_weight=cap_w, cap_volume=cap_v)

        # 第一轮装载
        remaining_demands = _first_pass_loading(
            packer, remaining_demands, available_inventory,
            inventory_check_enabled
        )

        # 获取代表性上下文
        repr_type, repr_wait = get_representative_context(packer.load_records)

        # 构建上下文
        wfr, vfr = packer.get_load_ratios()
        context = _build_context(
            sending, receiving, truck_type, repr_type,
            repr_wait, packer.current_units, mdq
        )

        # 检查旁路规则
        bypass, rule_id = should_bypass_mdq(
            context, prepared_data['bypass_rules'], evaluator
        )

        # 确定触发原因
        max_wait_in_load = max(
            (r['demand_row']['waiting_days'] for r in packer.load_records),
            default=0
        )
        trigger_cause = determine_trigger_cause(
            packer.has_load(), wfr, vfr, wfr_th, vfr_th,
            bypass, max_wait_in_load, run_params['max_wait_days']
        )

        if trigger_cause:
            # 触发后再次装载（贴近 1.0）
            remaining_demands = _second_pass_loading(
                packer, remaining_demands, available_inventory,
                inventory_check_enabled
            )

            # 生成发运记录
            vehicle_no = used + 1
            _generate_shipment_records(
                sim_date, sending, receiving, truck_type, vehicle_no,
                packer, trigger_cause, rule_id, bypass, context,
                prepared_data, agg_status, available_inventory,
                inventory_check_enabled, results, remaining_demands,
                random_seed=run_params.get('random_seed')
            )

            # 更新剩余需求
            remaining_demands = remaining_demands[
                remaining_demands['deployed_qty'] > 0
            ].copy()
            used += 1
        else:
            break

    return remaining_demands


def _first_pass_loading(
    packer: VehiclePacker,
    remaining_demands: pd.DataFrame,
    available_inventory: Dict[Tuple[str, str], float],
    inventory_check_enabled: bool
) -> pd.DataFrame:
    """
    第一轮装载（尽量装入但不超容量）。

    参数：
        packer: 装载器
        remaining_demands: 剩余需求
        available_inventory: 可用库存
        inventory_check_enabled: 是否启用库存检查

    返回：
        更新后的剩余需求
    """
    for row_tuple in remaining_demands.itertuples():
        if packer.is_full():
            break

        inv_limit = None
        if inventory_check_enabled:
            inv_limit = calculate_inventory_limit(
                available_inventory,
                row_tuple.material,
                row_tuple.sending,
                packer.get_material_loaded(row_tuple.material)
            )

        demand_row = remaining_demands.loc[row_tuple.Index]
        packer.add_demand(row_tuple.Index, demand_row, inv_limit)

    return remaining_demands


def _second_pass_loading(
    packer: VehiclePacker,
    remaining_demands: pd.DataFrame,
    available_inventory: Dict[Tuple[str, str], float],
    inventory_check_enabled: bool
) -> pd.DataFrame:
    """
    第二轮装载（触发后贴近 1.0）。

    参数：
        packer: 装载器
        remaining_demands: 剩余需求
        available_inventory: 可用库存
        inventory_check_enabled: 是否启用库存检查

    返回：
        更新后的剩余需求
    """
    taken = packer.get_loaded_indices()

    for row_tuple in remaining_demands.itertuples():
        if row_tuple.Index in taken:
            continue

        if packer.is_full():
            break

        inv_limit = None
        if inventory_check_enabled:
            inv_limit = calculate_inventory_limit(
                available_inventory,
                row_tuple.material,
                row_tuple.sending,
                packer.get_material_loaded(row_tuple.material)
            )

        demand_row = remaining_demands.loc[row_tuple.Index]
        packer.add_demand(row_tuple.Index, demand_row, inv_limit)

    return remaining_demands


def _build_context(
    sending: str,
    receiving: str,
    truck_type: str,
    demand_element: Optional[str],
    waiting_days: int,
    qty_units: float,
    mdq: float
) -> Dict[str, Any]:
    """构建评估上下文。"""
    return {
        'sending': sending,
        'receiving': receiving,
        'truck_type': truck_type,
        'demand_element': demand_element,
        'waiting_days': waiting_days,
        'deployed_qty_ratio': (qty_units / mdq) if mdq > 0 else 0.0,
        'exception_MDQ': 1 if mdq == 0 else 0
    }


def _generate_shipment_records(
    sim_date: pd.Timestamp,
    sending: str,
    receiving: str,
    truck_type: str,
    vehicle_no: int,
    packer: VehiclePacker,
    trigger_cause: str,
    rule_id: Optional[str],
    bypass: bool,
    context: Dict[str, Any],
    prepared_data: Dict[str, Any],
    agg_status: Dict[str, Dict],
    available_inventory: Dict[Tuple[str, str], float],
    inventory_check_enabled: bool,
    results: Dict[str, List],
    remaining_demands: pd.DataFrame,
    random_seed: Optional[int] = None
) -> None:
    """
    生成发运记录。

    参数：
        各种参数
        random_seed: 随机种子，用于批量延迟采样的可复现性
    """
    wfr, vfr = packer.get_load_ratios()

    # 生成车辆日志
    vehicle_log_entry = create_vehicle_log_entry(
        sim_date, sending, receiving, truck_type, vehicle_no,
        packer, trigger_cause
    )
    results['vehicle_log'].append(vehicle_log_entry)
    vehicle_uid = vehicle_log_entry['vehicle_uid']

    # 批量采样延迟（DuckDB优化）
    if M6_DUCKDB_AVAILABLE and m6_is_duckdb_available() and len(packer.load_records) >= 10:
        try:
            routes = [(sending, receiving)] * len(packer.load_records)
            delays = batch_sample_delivery_delays_duckdb(
                routes, prepared_data['delay_dist'], seed=random_seed
            )
        except Exception as e:
            print(f"[M6] Batch delay sampling failed, using single-record mode: {e}")
            delays = None
    else:
        delays = None

    # 生成发货明细
    for i, rec in enumerate(packer.load_records):
        sub = rec['demand_row']
        uid = sub['ori_deployment_uid']

        # 计算交货时间
        try:
            lt_info = calculate_lead_time(
                prepared_data['lead_time'], sending, receiving
            )
        except ValueError as e:
            raise ValueError(f"缺少路线 {sending}->{receiving} 的 LeadTime 行") from e

        # 使用批量采样结果或单个采样
        if delays is not None:
            delay = int(delays[i])
        else:
            delay = sample_delivery_delay(
                sending, receiving, prepared_data['delay_dist']
            )

        ship_date = sim_date
        eta = calculate_actual_delivery_date(
            ship_date, lt_info['OTD'], lt_info['GR'], delay
        )

        # 创建发货记录
        delivery_record = create_delivery_record(
            vehicle_uid, uid, sub, rec['load_qty'],
            ship_date, eta, truck_type, wfr, vfr
        )
        results['delivery_plan'].append(delivery_record)

        # 更新库存
        if inventory_check_enabled:
            available_inventory = update_inventory_after_load(
                available_inventory, sub['material'], sending, rec['load_qty']
            )

        # 更新聚合状态
        agg_status[uid]['qty'] = max(0, agg_status[uid]['qty'] - rec['load_qty'])
        remaining_demands.at[rec['idx'], 'deployed_qty'] = max(
            0, sub['deployed_qty'] - rec['load_qty']
        )

    # 旁路规则命中记录
    if trigger_cause == 'bypass':
        for rec in packer.load_records:
            bypass_record = create_bypass_record(
                rec['demand_row']['ori_deployment_uid'],
                rule_id, sim_date, context, vehicle_uid
            )
            results['bypass_log'].append(bypass_record)


def _handle_remaining_demands(
    route_demands: pd.DataFrame,
    agg_status: Dict[str, Dict],
    sending: str,
    receiving: str,
    sim_date: pd.Timestamp,
    max_wait_days: int,
    route_mdq: float,
    unsat_log: List[Dict]
) -> None:
    """
    处理剩余未发出的需求。

    参数：
        各种参数
    """
    route_remaining = route_demands[route_demands['deployed_qty'] > 0]

    # 优化：使用 itertuples() 替代 iterrows()
    for row in route_remaining.itertuples(index=False):
        uid = row.ori_deployment_uid

        if agg_status.get(uid, {}).get('qty', 0) <= 0:
            continue

        waiting_days = row.waiting_days

        if waiting_days > max_wait_days:
            unsat_record = create_unsatisfied_record(
                uid, row._asdict(), sending, receiving, sim_date,
                waiting_days, agg_status[uid]['qty'], route_mdq
            )
            unsat_log.append(unsat_record)
            agg_status[uid]['qty'] = 0
