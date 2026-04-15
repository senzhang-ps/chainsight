# -*- coding: utf-8 -*-
"""
main.py - Module6 物流执行主入口

提供 run_daily_physical_flow / run_physical_flow_module 等入口函数，
以及初始化、数据准备等辅助逻辑。
从 module6.py 等价迁入，函数签名和行为不变。
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config_loader import load_integrated_config, load_standalone_config
from .capacity_manager import build_capacity_map, normalize_capacity_plan
from .validators import (
    check_and_deduplicate,
    validate_deployment_plan,
    validate_priority_mapping,
    validate_threshold_config,
    validate_truck_config,
    validate_truck_specs,
)
from .simulation import run_simulation_loop
from .output_writer import generate_outputs
from ...utils.defaults import M6_MAX_WAIT_DAYS


# ---------------------------------------------------------------------------
# 入口函数
# ---------------------------------------------------------------------------

def run_daily_physical_flow(
    config_dict: Dict[str, Any],
    orchestrator: object,
    current_date: pd.Timestamp,
    output_dir: str,
    max_wait_days: int = M6_MAX_WAIT_DAYS,
    random_seed: Optional[int] = None,
    skip_file_output: bool = False
) -> Dict[str, Any]:
    """
    每日物流执行函数，处理当日的部署计划。

    参数：
        config_dict: 配置数据字典
        orchestrator: Orchestrator 实例
        current_date: 当前仿真日期
        output_dir: 输出目录
        max_wait_days: 最大等待天数
        random_seed: 随机种子
        skip_file_output: 是否跳过写入 Excel 文件

    返回：
        包含输出结果的字典
    """
    daily_output_file = f"{output_dir}/Module6Output_{current_date.strftime('%Y%m%d')}.xlsx"

    result = run_physical_flow_module(
        config_dict=config_dict,
        orchestrator=orchestrator,
        current_date=current_date.strftime('%Y-%m-%d'),
        output_path=daily_output_file,
        max_wait_days=max_wait_days,
        random_seed=random_seed,
        skip_file_output=skip_file_output
    )

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
    # 独立模式参数
    input_excel: Optional[str] = None,
    simulation_start: Optional[str] = None,
    simulation_end: Optional[str] = None,
    output_excel: Optional[str] = None,
    # 集成模式参数
    config_dict: Optional[Dict[str, Any]] = None,
    orchestrator: Optional[object] = None,
    current_date: Optional[str] = None,
    output_path: Optional[str] = None,
    # 通用参数
    max_wait_days: int = M6_MAX_WAIT_DAYS,
    random_seed: Optional[int] = None,
    skip_file_output: bool = False
) -> Dict[str, Any]:
    """
    物流模块主入口函数。

    支持两种运行模式：
    - 独立模式：使用 Excel 文件输入输出
    - 集成模式：与 Orchestrator 集成运行
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
    results = run_simulation_loop(run_params, prepared_data)

    # 生成输出
    return generate_outputs(
        run_params, results, prepared_data['validation_log'],
        skip_file_output
    )


# ---------------------------------------------------------------------------
# 初始化
# ---------------------------------------------------------------------------

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
    """初始化运行参数。"""
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
    """初始化集成模式参数。"""
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
    """初始化独立模式参数。"""
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


# ---------------------------------------------------------------------------
# 数据准备
# ---------------------------------------------------------------------------

def _prepare_data(run_params: Dict[str, Any]) -> Dict[str, Any]:
    """准备和验证数据。"""
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


def _build_priority_map(demand_prio: pd.DataFrame) -> Dict[str, int]:
    """构建优先级映射。"""
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
    """处理物料元数据。"""
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
    """准备部署计划数据。"""
    if dp.empty:
        return dp

    dp['planned_deployment_date'] = pd.to_datetime(dp['planned_deployment_date'])

    # 稳定排序以保证 UID 可复现
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
    """处理 UID 重复问题。"""
    if dp.empty or not dp['ori_deployment_uid'].duplicated().any():
        return dp, validation_log

    dup_mask = dp.duplicated(subset=['ori_deployment_uid'], keep=False)
    dup_count = dup_mask.sum()
    dup_uid_count = dp.loc[dup_mask, 'ori_deployment_uid'].nunique()


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

    return dp, validation_log


# 主函数别名
main = run_physical_flow_module
