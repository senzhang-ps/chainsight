# -*- coding: utf-8 -*-
"""
Module 6 - Physical Flow Management Module (物流管理模块)

提供供应链的物流发运管理功能，包括：
- 车辆装载优化
- 发运计划生成
- MDQ (最小发货量) 规则处理
- 延迟采样和交货时间计算

Integration Mode Support:
- Standalone Mode: Excel file input/output (legacy)
- Integrated Mode: Config dict + Orchestrator integration

Data Sources (Integrated):
- OpenDeployment: orchestrator.get_open_deployment(current_date)
- M6_ Configs: M6_TruckReleaseCon, M6_TruckCapacityPlan, etc.
- Global_ Configs: Global_DemandPriority, Global_LeadTime

Execution Pattern: Daily processing following Module4/5 pattern
Module Execution Order: Module1 → Module4 → Module5 → Module6 → Module3

Typical usage example:
    # 独立模式
    result = run_physical_flow_module(
        input_excel='input.xlsx',
        simulation_start='2025-01-01',
        simulation_end='2025-01-31',
        output_excel='output.xlsx'
    )
    
    # 集成模式
    result = run_physical_flow_module(
        config_dict=config_dict,
        orchestrator=orchestrator,
        current_date='2025-01-15',
        output_path='output.xlsx'
    )
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 从子模块导入
from .logistics_execution.config_loader import (
    load_integrated_config,
    load_standalone_config,
)
from .logistics_execution.capacity_manager import (
    build_capacity_map,
    get_optimal_truck_sequence,
    get_truck_capacity,
    get_truck_config,
    get_truck_spec,
    normalize_capacity_plan,
)
from .logistics_execution.delivery_processor import (
    calculate_actual_delivery_date,
    calculate_lead_time,
    create_bypass_record,
    create_delivery_record,
    create_unsatisfied_record,
    sample_delivery_delay,
    should_bypass_mdq,
)
from .logistics_execution.expression_evaluator import SafeExpressionEvaluator
from .logistics_execution.inventory_manager import (
    calculate_inventory_limit,
    calculate_physical_inventory,
    update_inventory_after_load,
)
from .logistics_execution.validators import (
    check_and_deduplicate,
    generate_validation_report,
    validate_deployment_plan,
    validate_priority_mapping,
    validate_threshold_config,
    validate_truck_config,
    validate_truck_specs,
)
from .logistics_execution.vehicle_packer import (
    VehiclePacker,
    create_vehicle_log_entry,
    determine_trigger_cause,
    get_representative_context,
)


# ============= 常量定义 =============
ALLOWED_EXPRESSION_VARS: List[str] = [
    'waiting_days', 'deployed_qty_ratio', 'exception_MDQ',
    'sending', 'receiving', 'truck_type', 'demand_element'
]

OUTPUT_COLUMNS = {
    'vehicle_log': [
        'date', 'sending', 'receiving', 'truck_type', 'vehicle_no',
        'vehicle_uid', 'total_units', 'total_weight', 'total_volume',
        'WFR', 'VFR', 'trigger'
    ],
    'usage': ['date', 'sending', 'receiving', 'truck_type', 'truck_used']
}


def run_daily_physical_flow(
    config_dict: Dict[str, Any],
    orchestrator: object,
    current_date: pd.Timestamp,
    output_dir: str,
    max_wait_days: int = 30,
    random_seed: Optional[int] = None,
    skip_file_output: bool = False
) -> Dict[str, Any]:
    """
    每日物流执行函数，处理当日的部署计划。
    
    Args:
        config_dict: 配置数据字典
        orchestrator: Orchestrator 实例
        current_date: 当前仿真日期
        output_dir: 输出目录
        max_wait_days: 最大等待天数
        random_seed: 随机种子
        skip_file_output: 是否跳过写入 Excel 文件
        
    Returns:
        包含输出结果的字典
    """
    daily_output_file = f"{output_dir}/Module6Output_{current_date.strftime('%Y%m%d')}.xlsx"
    
    result = main(
        config_dict=config_dict,
        orchestrator=orchestrator,
        current_date=current_date.strftime('%Y-%m-%d'),
        output_path=daily_output_file,
        max_wait_days=max_wait_days,
        random_seed=random_seed,
        skip_file_output=skip_file_output
    )
    
    # 完整转发所有输出，确保数据库写入时能获取所有表数据
    return {
        'daily_output_file': daily_output_file,
        'delivery_plan': result.get('delivery_plan', pd.DataFrame()),
        'vehicle_log': result.get('vehicle_log', pd.DataFrame()),
        'truck_usage': result.get('truck_usage', pd.DataFrame()),
        'unsatisfied_log': result.get('unsatisfied_log', pd.DataFrame()),
        'validation_log': result.get('validation_log', pd.DataFrame()),
        'bypass_log': result.get('bypass_log', pd.DataFrame()),
        'statistics': result.get('statistics', {})
    }


def run_physical_flow_module(
    # Standalone mode parameters
    input_excel: Optional[str] = None,
    simulation_start: Optional[str] = None,
    simulation_end: Optional[str] = None,
    output_excel: Optional[str] = None,
    # Integrated mode parameters
    config_dict: Optional[Dict[str, Any]] = None,
    orchestrator: Optional[object] = None,
    current_date: Optional[str] = None,
    output_path: Optional[str] = None,
    # Common parameters
    max_wait_days: int = 30,
    random_seed: Optional[int] = None,
    skip_file_output: bool = False
) -> Dict[str, Any]:
    """
    物流模块主入口函数。
    
    支持两种运行模式：
    - 独立模式：使用 Excel 文件输入输出
    - 集成模式：与 Orchestrator 集成运行
    
    Args:
        input_excel: 独立模式的输入 Excel 文件路径
        simulation_start: 独立模式的仿真开始日期
        simulation_end: 独立模式的仿真结束日期
        output_excel: 独立模式的输出 Excel 文件路径
        config_dict: 集成模式的配置字典
        orchestrator: 集成模式的 Orchestrator 实例
        current_date: 集成模式的当前日期
        output_path: 集成模式的输出路径
        max_wait_days: 最大等待天数
        random_seed: 随机种子
        skip_file_output: 是否跳过文件输出
        
    Returns:
        包含处理结果的字典
    """
    # 初始化运行参数
    run_params = _initialize_run_params(
        input_excel, simulation_start, simulation_end, output_excel,
        config_dict, orchestrator, current_date, output_path,
        max_wait_days, random_seed
    )
    
    # 数据准备和验证
    prepared_data = _prepare_data(run_params)
    
    # 执行主仿真循环
    results = _run_simulation_loop(run_params, prepared_data)
    
    # 生成输出
    return _generate_outputs(
        run_params, results, prepared_data['validation_log'],
        skip_file_output
    )


def _initialize_run_params(
    input_excel: Optional[str],
    simulation_start: Optional[str],
    simulation_end: Optional[str],
    output_excel: Optional[str],
    config_dict: Optional[Dict[str, Any]],
    orchestrator: Optional[object],
    current_date: Optional[str],
    output_path: Optional[str],
    max_wait_days: int,
    random_seed: Optional[int]
) -> Dict[str, Any]:
    """
    初始化运行参数。
    
    Args:
        各种输入参数
        
    Returns:
        运行参数字典
    """
    params = {
        'max_wait_days': max_wait_days,
        'random_seed': random_seed,
        'is_integrated': config_dict is not None
    }
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    if params['is_integrated']:
        params.update(_init_integrated_params(
            config_dict, orchestrator, current_date, output_path
        ))
    else:
        params.update(_init_standalone_params(
            input_excel, simulation_start, simulation_end,
            output_excel, max_wait_days
        ))
    
    return params


def _init_integrated_params(
    config_dict: Dict[str, Any],
    orchestrator: object,
    current_date: str,
    output_path: str
) -> Dict[str, Any]:
    """
    初始化集成模式参数。
    
    Args:
        config_dict: 配置字典
        orchestrator: Orchestrator 实例
        current_date: 当前日期字符串
        output_path: 输出路径
        
    Returns:
        集成模式参数字典
    """
    sim_date = pd.to_datetime(current_date)
    config = load_integrated_config(config_dict, orchestrator, sim_date)
    
    return {
        'config': config,
        'orchestrator': orchestrator,
        'sim_dates': pd.DatetimeIndex([sim_date]),
        'sim_start': sim_date,
        'sim_end': sim_date,
        'output_file': output_path
    }


def _init_standalone_params(
    input_excel: str,
    simulation_start: str,
    simulation_end: str,
    output_excel: str,
    max_wait_days: int
) -> Dict[str, Any]:
    """
    初始化独立模式参数。
    
    Args:
        input_excel: 输入文件路径
        simulation_start: 开始日期
        simulation_end: 结束日期
        output_excel: 输出文件路径
        max_wait_days: 最大等待天数
        
    Returns:
        独立模式参数字典
    """
    config = load_standalone_config(input_excel)
    sim_start = pd.to_datetime(simulation_start)
    sim_end = pd.to_datetime(simulation_end)
    
    dp = config['DeploymentPlan']
    if not dp.empty:
        sim_dates = pd.date_range(
            max(sim_start, dp['planned_deployment_date'].min()),
            min(sim_end, dp['planned_deployment_date'].max() + 
                pd.Timedelta(days=max_wait_days))
        )
    else:
        sim_dates = pd.date_range(sim_start, sim_end)
    
    return {
        'config': config,
        'orchestrator': None,
        'sim_dates': sim_dates,
        'sim_start': sim_start,
        'sim_end': sim_end,
        'output_file': output_excel
    }


def _prepare_data(run_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    准备和验证数据。
    
    Args:
        run_params: 运行参数
        
    Returns:
        准备好的数据字典
    """
    config = run_params['config']
    validation_log = list(config.get('ValidationLog', []))
    
    # 获取配置数据
    dp = config['DeploymentPlan']
    truck_con = config['TruckReleaseCon']
    truck_specs = config['TruckTypeSpecs']
    material_md = config['MaterialMD']
    demand_prio = config['DemandPriority']
    
    # 验证数据
    dp = validate_deployment_plan(dp, validation_log)
    truck_con = validate_truck_config(truck_con, validation_log)
    
    # 去重处理
    # 🔧 修复：确保demand_prio没有重复的demand_element，与code_vo保持一致
    demand_prio = check_and_deduplicate(
        demand_prio, 'demand_element', 'Global_DemandPriority', validation_log
    )
    material_md = check_and_deduplicate(
        material_md, 'material', 'M6_MaterialMD', validation_log
    )
    truck_specs = check_and_deduplicate(
        truck_specs, 'truck_type', 'M6_TruckTypeSpecs', validation_log
    )
    
    # 构建映射
    prio_map = _build_priority_map(demand_prio)
    mat_map = _build_material_map(material_md)
    spec_map = _build_spec_map(truck_specs)
    
    # 优先级验证和过滤
    dp = validate_priority_mapping(dp, prio_map, validation_log)
    
    # 物料元数据处理
    dp = _process_material_metadata(dp, mat_map, validation_log)
    
    # 验证卡车配置
    validate_threshold_config(truck_con, validation_log)
    validate_truck_specs(truck_con, spec_map, validation_log)
    
    # 准备部署计划
    dp = _prepare_deployment_plan(dp, prio_map)
    
    # 检查并处理 UID 重复
    dp, validation_log = _handle_uid_duplicates(dp, validation_log)
    
    # 构建容量映射
    cap_daily = normalize_capacity_plan(
        config['TruckCapacityPlan'].copy(),
        run_params['sim_start'],
        run_params['sim_end']
    )
    cap_map = build_capacity_map(cap_daily)
    
    return {
        'dp': dp,
        'dp_dict': dp.set_index('ori_deployment_uid').to_dict('index'),
        'truck_con': truck_con,
        'lead_time': config['LeadTime'],
        'delay_dist': config['DeliveryDelayDistribution'],
        'bypass_rules': config['MDQBypassRules'],
        'prio_map': prio_map,
        'spec_map': spec_map,
        'cap_map': cap_map,
        'validation_log': validation_log
    }


def _filter_empty_demand_element(
    demand_prio: pd.DataFrame,
    validation_log: List[Dict]
) -> pd.DataFrame:
    """
    过滤 demand_element 为空的记录。
    
    demand_element 字段允许重复，但不允许为空。
    
    Args:
        demand_prio: DemandPriority DataFrame
        validation_log: 验证日志
        
    Returns:
        过滤后的 DataFrame
    """
    if demand_prio.empty:
        return demand_prio
    
    # 检查空值
    empty_mask = demand_prio['demand_element'].isna() | (demand_prio['demand_element'] == '')
    empty_count = empty_mask.sum()
    
    if empty_count > 0:
        print(f"  ⚠️  Global_DemandPriority中有 {empty_count} 条demand_element为空的记录，已过滤")
        validation_log.append({
            'sheet': 'Global_DemandPriority',
            'row': '',
            'issue': f'Found {empty_count} records with empty demand_element. '
                     f'Records have been filtered out.',
            'severity': 'WARNING',
            'impact': f'Data Filtering - {empty_count} records removed'
        })
        return demand_prio[~empty_mask].copy()
    
    return demand_prio


def _build_priority_map(demand_prio: pd.DataFrame) -> Dict[str, int]:
    """
    构建优先级映射。
    
    假设已经通过check_and_deduplicate进行了去重处理，直接构建映射。
    """
    if demand_prio.empty:
        return {}
    return demand_prio.set_index('demand_element')['priority'].to_dict()


def _build_material_map(material_md: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """构建物料元数据映射。"""
    if material_md.empty:
        return {}
    return material_md.set_index('material')[
        ['demand_unit_to_weight', 'demand_unit_to_volume']
    ].to_dict('index')


def _build_spec_map(truck_specs: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """构建车型规格映射。"""
    if truck_specs.empty:
        return {}
    return truck_specs.set_index('truck_type').to_dict('index')


def _process_material_metadata(
    dp: pd.DataFrame,
    mat_map: Dict[str, Dict[str, float]],
    validation_log: List[Dict]
) -> pd.DataFrame:
    """
    处理物料元数据。
    
    Args:
        dp: 部署计划 DataFrame
        mat_map: 物料映射
        validation_log: 验证日志
        
    Returns:
        处理后的部署计划
    """
    if dp.empty:
        return dp
    
    missing_mat = dp[~dp['material'].isin(mat_map.keys())]
    
    if not missing_mat.empty:
        _log_missing_materials(missing_mat, validation_log)
        dp['demand_unit_to_weight'] = dp['material'].map(
            lambda x: mat_map.get(x, {}).get('demand_unit_to_weight', 1.0)
        )
        dp['demand_unit_to_volume'] = dp['material'].map(
            lambda x: mat_map.get(x, {}).get('demand_unit_to_volume', 1.0)
        )
    else:
        # 使用 merge 时避免产生重复行
        mat_df = pd.DataFrame([
            {'material': m, **v} for m, v in mat_map.items()
        ])
        dp = dp.merge(mat_df, on='material', how='left')
        dp['demand_unit_to_weight'] = dp['demand_unit_to_weight'].fillna(1.0)
        dp['demand_unit_to_volume'] = dp['demand_unit_to_volume'].fillna(1.0)
    
    return dp


def _log_missing_materials(
    missing_mat: pd.DataFrame,
    validation_log: List[Dict]
) -> None:
    """记录缺失的物料元数据。"""
    missing_materials = missing_mat['material'].unique()
    print(f"  ⚠️  发现 {len(missing_materials)} 个缺失的material配置，将使用默认值")
    
    for val in missing_materials:
        missing_records = missing_mat[missing_mat['material'] == val]
        validation_log.append({
            'sheet': 'M6_MaterialMD',
            'row': '',
            'issue': f'Missing material metadata for "{val}" '
                     f'(affects {len(missing_records)} records). '
                     f'Default unit conversion factors (1.0) will be used.',
            'severity': 'WARNING',
            'impact': f'Default Values Used - {len(missing_records)} records',
            'missing_material': val,
            'affected_records': len(missing_records)
        })


def _prepare_deployment_plan(
    dp: pd.DataFrame,
    prio_map: Dict[str, int]
) -> pd.DataFrame:
    """
    准备部署计划数据。
    
    Args:
        dp: 部署计划 DataFrame
        prio_map: 优先级映射
        
    Returns:
        准备好的部署计划
    """
    if dp.empty:
        return dp
    
    dp['planned_deployment_date'] = pd.to_datetime(dp['planned_deployment_date'])
    
    # 为保证在相同配置和随机种子下UID可复现，先对关键字段做稳定排序（与code_vo保持一致）
    sort_cols = [
        col for col in ['planned_deployment_date', 'sending', 'receiving', 'material', 'demand_element']
        if col in dp.columns
    ]
    if sort_cols:
        dp = dp.sort_values(by=sort_cols, kind='mergesort')
    dp = dp.reset_index(drop=True)
    
    # 生成或保留 UID
    if 'ori_deployment_uid' not in dp.columns or dp['ori_deployment_uid'].isnull().any():
        dp['ori_deployment_uid'] = [f'UID{i:06d}' for i in dp.index]
    
    dp['priority'] = dp['demand_element'].map(prio_map)
    dp['waiting_days'] = 0
    dp['simulation_date'] = dp['planned_deployment_date']
    
    # 添加路线类型标记
    dp['route_type_debug'] = np.where(
        dp['sending'] == dp['receiving'], 'self_loop', 'cross_node'
    )
    
    return dp


def _handle_uid_duplicates(
    dp: pd.DataFrame,
    validation_log: List[Dict]
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    处理 UID 重复问题。
    
    Args:
        dp: 部署计划 DataFrame
        validation_log: 验证日志
        
    Returns:
        (处理后的 DataFrame, 更新后的验证日志)
    """
    if dp.empty or not dp['ori_deployment_uid'].duplicated().any():
        return dp, validation_log
    
    dup_mask = dp.duplicated(subset=['ori_deployment_uid'], keep=False)
    dup_count = dup_mask.sum()
    dup_uid_count = dp.loc[dup_mask, 'ori_deployment_uid'].nunique()
    
    print(f"  ⚠️  发现 {dup_uid_count} 个重复的ori_deployment_uid"
          f"（共 {dup_count} 条记录）")
    
    validation_log.append({
        'sheet': 'DeploymentPlan',
        'row': '',
        'issue': f'Found {dup_uid_count} duplicate ori_deployment_uid values. '
                 f'Deduplicating by keeping first occurrence.',
        'severity': 'ERROR',
        'impact': f'Data Deduplication - {dup_count - dup_uid_count} removed',
        'duplicate_uids': dup_uid_count
    })
    
    dp = dp.drop_duplicates(subset=['ori_deployment_uid'], keep='first')
    print(f"  ✅ 去重后保留: {len(dp)} 条记录")
    
    return dp, validation_log


def _run_simulation_loop(
    run_params: Dict[str, Any],
    prepared_data: Dict[str, Any]
) -> Dict[str, List]:
    """
    运行仿真主循环。
    
    Args:
        run_params: 运行参数
        prepared_data: 准备好的数据
        
    Returns:
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
    
    Args:
        dp_dict: 部署计划字典
        
    Returns:
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
    
    Args:
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
    _process_routes(
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
    
    Args:
        sim_date: 仿真日期
        agg_status: 聚合状态
        dp_dict: 部署计划字典
        
    Returns:
        待处理需求列表
    """
    pending_rows = []
    
    # 使用sorted()确保确定性迭代顺序（与code_vo保持一致）
    for uid, st in sorted(agg_status.items()):
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


def _process_routes(
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
    
    Args:
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
    
    Args:
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
    
    Args:
        各种参数
        
    Returns:
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
                inventory_check_enabled, results, remaining_demands
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
    
    Args:
        packer: 装载器
        remaining_demands: 剩余需求
        available_inventory: 可用库存
        inventory_check_enabled: 是否启用库存检查
        
    Returns:
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
    
    Args:
        packer: 装载器
        remaining_demands: 剩余需求
        available_inventory: 可用库存
        inventory_check_enabled: 是否启用库存检查
        
    Returns:
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
    remaining_demands: pd.DataFrame
) -> None:
    """
    生成发运记录。
    
    Args:
        各种参数
    """
    wfr, vfr = packer.get_load_ratios()
    
    # 生成车辆日志
    vehicle_log_entry = create_vehicle_log_entry(
        sim_date, sending, receiving, truck_type, vehicle_no,
        packer, trigger_cause
    )
    results['vehicle_log'].append(vehicle_log_entry)
    vehicle_uid = vehicle_log_entry['vehicle_uid']
    
    # 生成发货明细
    for rec in packer.load_records:
        sub = rec['demand_row']
        uid = sub['ori_deployment_uid']
        
        # 计算交货时间
        try:
            lt_info = calculate_lead_time(
                prepared_data['lead_time'], sending, receiving
            )
        except ValueError as e:
            raise ValueError(f"缺少路线 {sending}->{receiving} 的 LeadTime 行") from e
        
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
    
    Args:
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


def _enforce_shipment_constraint(
    delivery_plan_df: pd.DataFrame,
    orchestrator: Optional[object],
    validation_log: List[Dict]
) -> pd.DataFrame:
    """
    强制约束出货量不超过订单量，如果超出则按比例裁剪。
    
    Args:
        delivery_plan_df: 出货计划 DataFrame
        orchestrator: Orchestrator 实例
        validation_log: 验证日志
        
    Returns:
        裁剪后的出货计划 DataFrame
    """
    if delivery_plan_df.empty or orchestrator is None:
        return delivery_plan_df
    
    # 获取日期列
    if 'sim_date' in delivery_plan_df.columns:
        date_col = 'sim_date'
    elif 'date' in delivery_plan_df.columns:
        date_col = 'date'
    else:
        return delivery_plan_df
    
    # 按物料和发货地点分组，比对订单量
    result_df = delivery_plan_df.copy()
    
    for date_val in result_df[date_col].unique():
        date_str = str(date_val).split()[0]
        
        # 获取当日订单数据
        shipment_data = orchestrator.get_shipment_log_view(date_str)
        if shipment_data.empty:
            continue
        
        # 按(material, location)计算订单量
        shipment_qty_by_ml = shipment_data.groupby(
            ['material', 'location']
        )['quantity'].sum().to_dict()
        
        # 按(material, sending)计算当前出货量
        day_mask = result_df[date_col] == date_val
        day_deliveries = result_df[day_mask]
        
        if day_deliveries.empty:
            continue
        
        for (mat, sending), group in day_deliveries.groupby(['material', 'sending']):
            # 获取该(material, location)的订单量
            order_qty = shipment_qty_by_ml.get((mat, sending), 0)
            delivery_qty = group['delivery_qty'].sum()
            
            if delivery_qty > order_qty and delivery_qty > 0:
                # 按比例裁剪
                ratio = order_qty / delivery_qty if delivery_qty > 0 else 0
                
                print(f"🔧 [Module6] 强制裁剪: {mat}@{sending}")
                print(f"    订单量: {order_qty:.0f}, 出货量: {delivery_qty:.0f}")
                print(f"    裁剪比例: {ratio:.2%}")
                
                # 裁剪每一行的 delivery_qty
                for idx in group.index:
                    old_qty = result_df.at[idx, 'delivery_qty']
                    new_qty = int(old_qty * ratio)
                    result_df.at[idx, 'delivery_qty'] = new_qty
                
                validation_log.append({
                    'sheet': 'Module6_Enforcement',
                    'row': f'{mat}@{sending}',
                    'issue': f'Delivery quantity trimmed from {delivery_qty:.0f} to {order_qty:.0f}',
                    'severity': 'INFO',
                    'impact': f'Enforced shipment constraint - reduced by {delivery_qty - order_qty:.0f} units',
                    'shipment_qty': order_qty,
                    'original_delivery_qty': delivery_qty
                })
    
    return result_df


def _validate_shipment_delivery_constraint(
    delivery_plan_df: pd.DataFrame,
    orchestrator: Optional[object],
    validation_log: List[Dict]
) -> Tuple[bool, int, int]:
    """
    验证出货量不超过订单量约束。
    
    Args:
        delivery_plan_df: 出货计划 DataFrame
        orchestrator: Orchestrator 实例
        validation_log: 验证日志
        
    Returns:
        (是否通过验证, 订单量, 出货量)
    """
    if delivery_plan_df.empty or orchestrator is None:
        return True, 0, 0
    
    # 检查 sim_date 列是否存在，如果不存在则使用 date 列（本地模式下）
    if 'sim_date' not in delivery_plan_df.columns:
        if 'date' not in delivery_plan_df.columns:
            # 两个列都不存在，无法验证
            return True, 0, 0
        # 使用 date 列作为替代
        date_col = 'date'
    else:
        date_col = 'sim_date'
    
    # 计算各日期各地点的出货量
    delivery_qty_by_date_loc = delivery_plan_df.groupby(date_col)['delivery_qty'].sum()
    total_delivery_qty = delivery_qty_by_date_loc.sum()
    
    # 计算各日期的订单量
    shipment_logs = []
    for date_val in delivery_plan_df[date_col].unique():
        # 从日期值中提取日期部分（处理 sim_date 和 date 格式）
        date_str = str(date_val).split()[0]
        shipment_data = orchestrator.get_shipment_log_view(date_str)
        if not shipment_data.empty:
            shipment_logs.append(shipment_data)
    
    if shipment_logs:
        all_shipments = pd.concat(shipment_logs, ignore_index=True)
        total_shipment_qty = all_shipments['quantity'].astype(float).sum()
    else:
        total_shipment_qty = 0
    
    # 验证
    if total_delivery_qty > total_shipment_qty:
        print(f"\n⚠️  约束违反: 出货量 > 订单量")
        print(f"    订单量: {total_shipment_qty:.0f}")
        print(f"    出货量: {total_delivery_qty:.0f}")
        print(f"    超出: {total_delivery_qty - total_shipment_qty:.0f}")
        
        validation_log.append({
            'sheet': 'Module6_Constraint',
            'row': '',
            'issue': f'Delivery quantity ({total_delivery_qty:.0f}) exceeds shipment quantity ({total_shipment_qty:.0f})',
            'severity': 'ERROR',
            'impact': f'Constraint Violation - {total_delivery_qty - total_shipment_qty:.0f} units over limit',
            'shipment_qty': total_shipment_qty,
            'delivery_qty': total_delivery_qty
        })
        return False, int(total_shipment_qty), int(total_delivery_qty)
    
    return True, int(total_shipment_qty), int(total_delivery_qty)


def _generate_outputs(
    run_params: Dict[str, Any],
    results: Dict[str, List],
    validation_log: List[Dict],
    skip_file_output: bool
) -> Dict[str, Any]:
    """
    生成输出结果。
    
    Args:
        run_params: 运行参数
        results: 仿真结果
        validation_log: 验证日志
        skip_file_output: 是否跳过文件输出
        
    Returns:
        输出结果字典
    """
    # 构建 DataFrame
    delivery_plan_df = pd.DataFrame(results['delivery_plan'])
    
    # 🔧 强制约束: 出货量 <= 订单量
    # 如果发现超出，则按比例裁剪
    delivery_plan_df = _enforce_shipment_constraint(
        delivery_plan_df,
        run_params.get('orchestrator'),
        validation_log
    )
    
    # 验证约束: 出货量 <= 订单量
    constraint_passed, shipment_qty, delivery_qty = _validate_shipment_delivery_constraint(
        delivery_plan_df, 
        run_params.get('orchestrator'),
        validation_log
    )
    
    if not constraint_passed:
        print(f"⚠️  发现约束违反: delivery_qty ({delivery_qty}) > shipment_qty ({shipment_qty})")
        print(f"   这可能由以下原因引起:")
        print(f"   1. Module5部署计划多重生成导致deployed_qty超出shipment数量")
        print(f"   2. 订单去重不当导致同一订单被多次处理")
        print(f"   3. Module6装载优化产生了超额分配")
    
    vehicle_df = _build_vehicle_df(results['vehicle_log'])
    usage_df = _build_usage_df(vehicle_df)
    unsat_df = pd.DataFrame(results['unsat_log'])
    validation_df = pd.DataFrame(validation_log)
    bypass_df = pd.DataFrame(results['bypass_log'])
    
    # 写入文件
    if not skip_file_output:
        _write_excel_output(
            run_params['output_file'], delivery_plan_df, vehicle_df,
            usage_df, unsat_df, validation_df, bypass_df
        )
        generate_validation_report(validation_log, run_params['output_file'])
    
    # 统计信息
    statistics = {
        'delivery_count': len(results['delivery_plan']),
        'vehicle_count': usage_df['truck_used'].sum() if not usage_df.empty else 0,
        'unsatisfied_count': len(results['unsat_log']),
        'bypass_count': len(results['bypass_log'])
    }
    
    return {
        'delivery_plan': delivery_plan_df,
        'vehicle_log': vehicle_df,
        'truck_usage': usage_df,
        'unsatisfied_log': unsat_df,
        'validation_log': validation_df,
        'bypass_log': bypass_df,
        'statistics': statistics
    }


def _build_vehicle_df(vehicle_log: List[Dict]) -> pd.DataFrame:
    """构建车辆日志 DataFrame。"""
    if vehicle_log:
        return pd.DataFrame(vehicle_log)
    return pd.DataFrame(columns=OUTPUT_COLUMNS['vehicle_log'])


def _build_usage_df(vehicle_df: pd.DataFrame) -> pd.DataFrame:
    """构建使用统计 DataFrame。"""
    if vehicle_df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS['usage'])
    
    return vehicle_df.groupby(
        ['date', 'sending', 'receiving', 'truck_type'],
        as_index=False
    ).agg(truck_used=('vehicle_uid', 'nunique'))


def _write_excel_output(
    output_file: str,
    delivery_plan_df: pd.DataFrame,
    vehicle_df: pd.DataFrame,
    usage_df: pd.DataFrame,
    unsat_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    bypass_df: pd.DataFrame
) -> None:
    """写入 Excel 输出文件。"""
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        delivery_plan_df.to_excel(writer, sheet_name='DeliveryPlan', index=False)
        vehicle_df.to_excel(writer, sheet_name='VehicleLog', index=False)
        usage_df.to_excel(writer, sheet_name='TruckUsageLog', index=False)
        unsat_df.to_excel(writer, sheet_name='UnsatisfiedMDQLog', index=False)
        validation_df.to_excel(writer, sheet_name='ValidationLog', index=False)
        bypass_df.to_excel(writer, sheet_name='BypassRuleHitLog', index=False)


# 主函数别名（保持与 Module4/5 一致）
main = run_physical_flow_module


# ======================== Example ========================
if __name__ == "__main__":
    # Standalone mode example
    run_physical_flow_module(
        input_excel='Module_6_1_1/config_SC.xlsx',
        simulation_start='2025-08-01',
        simulation_end='2025-08-03',
        output_excel='Module_6_1_1/output_SC.xlsx',
        random_seed=42
    )
