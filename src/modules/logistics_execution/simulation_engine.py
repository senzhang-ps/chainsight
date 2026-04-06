# -*- coding: utf-8 -*-
"""
物流执行模块 - 仿真主循环
"""

from typing import Any, Dict, List, Tuple

import pandas as pd

from .constants import ALLOWED_EXPRESSION_VARS
from .expression_evaluator import SafeExpressionEvaluator
from .inventory_manager import calculate_physical_inventory
from .route_processor import process_routes


def run_simulation_loop(
    run_params: Dict[str, Any],
    prepared_data: Dict[str, Any]
) -> Dict[str, List]:
    """
    运行仿真主循环。

    参数：
        run_params: 运行参数
        prepared_data: 准备好的数据

    返回：
        仿真结果字典
    """
    # 初始化状态
    agg_status = _init_aggregation_status(prepared_data['dp_dict'])
    evaluator = SafeExpressionEvaluator(ALLOWED_EXPRESSION_VARS)

    # 结果收集器
    results = {
        'delivery_plan': [],
        'vehicle_log': [],
        'unsat_log': [],
        'bypass_log': []
    }

    # 库存检查设置
    inventory_check_enabled = (
        run_params['is_integrated'] and
        run_params['orchestrator'] is not None
    )

    # 日期循环
    for sim_date in run_params['sim_dates']:
        available_inventory = {}
        if inventory_check_enabled:
            available_inventory = calculate_physical_inventory(
                run_params['orchestrator'], sim_date
            )

        # 处理当日需求
        _process_daily_demands(
            sim_date, agg_status, prepared_data, run_params,
            evaluator, available_inventory, inventory_check_enabled,
            results
        )

    return results


def _init_aggregation_status(
    dp_dict: Dict[str, Dict]
) -> Dict[str, Dict[str, Any]]:
    """
    初始化聚合状态。

    参数：
        dp_dict: 部署计划字典

    返回：
        聚合状态字典
    """
    return {
        uid: {
            'qty': row['deployed_qty'],
            'waiting': 1,
            'planned': row['planned_deployment_date']
        }
        for uid, row in dp_dict.items()
    }


def _process_daily_demands(
    sim_date: pd.Timestamp,
    agg_status: Dict[str, Dict],
    prepared_data: Dict[str, Any],
    run_params: Dict[str, Any],
    evaluator: SafeExpressionEvaluator,
    available_inventory: Dict[Tuple[str, str], float],
    inventory_check_enabled: bool,
    results: Dict[str, List]
) -> None:
    """
    处理单日需求。

    参数：
        sim_date: 仿真日期
        agg_status: 聚合状态
        prepared_data: 准备好的数据
        run_params: 运行参数
        evaluator: 表达式解析器
        available_inventory: 可用库存
        inventory_check_enabled: 是否启用库存检查
        results: 结果收集器
    """
    # 收集待处理需求
    pending_rows = _collect_pending_demands(
        sim_date, agg_status, prepared_data['dp_dict']
    )

    if not pending_rows:
        return

    pendf = pd.DataFrame(pending_rows)

    # 过滤跨节点路线
    cross_node_df = pendf[pendf['sending'] != pendf['receiving']].copy()
    if cross_node_df.empty:
        return

    # 按优先级排序
    cross_node_sorted = cross_node_df.sort_values([
        'priority', 'planned_deployment_date', 'sending', 'receiving'
    ]).reset_index(drop=True)

    # 按路线处理
    process_routes(
        sim_date, cross_node_sorted, agg_status, prepared_data,
        run_params, evaluator, available_inventory,
        inventory_check_enabled, results
    )


def _collect_pending_demands(
    sim_date: pd.Timestamp,
    agg_status: Dict[str, Dict],
    dp_dict: Dict[str, Dict]
) -> List[Dict]:
    """
    收集待处理的需求。

    参数：
        sim_date: 仿真日期
        agg_status: 聚合状态
        dp_dict: 部署计划字典

    返回：
        待处理需求列表
    """
    pending_rows = []

    # 保持原始迭代顺序（与 ChainSightMVPOri 版本保持一致）
    # 注意：不使用 sorted()，因为原始版本使用字典的默认迭代顺序
    for uid, st in agg_status.items():
        if st['qty'] <= 0:
            continue

        planned_date = pd.to_datetime(st['planned'])
        waiting_days = (sim_date - planned_date).days + 1

        full = dp_dict[uid]
        pending_rows.append({
            'ori_deployment_uid': uid,
            'material': full['material'],
            'sending': full['sending'],
            'receiving': full['receiving'],
            'planned_deployment_date': planned_date,
            'deployed_qty': st['qty'],
            'demand_element': full['demand_element'],
            'demand_unit_to_weight': full['demand_unit_to_weight'],
            'demand_unit_to_volume': full['demand_unit_to_volume'],
            'priority': full['priority'],
            'waiting_days': waiting_days,
        })

    return pending_rows
