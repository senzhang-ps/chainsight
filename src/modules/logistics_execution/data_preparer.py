# -*- coding: utf-8 -*-
"""
物流执行模块 - 数据准备与验证
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .capacity_manager import build_capacity_map, normalize_capacity_plan
from .validators import (
    check_and_deduplicate,
    validate_deployment_plan,
    validate_priority_mapping,
    validate_threshold_config,
    validate_truck_config,
    validate_truck_specs,
)


def prepare_data(run_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    准备和验证数据。

    参数：
        run_params: 运行参数

    返回：
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


def filter_empty_demand_element(
    demand_prio: pd.DataFrame,
    validation_log: List[Dict]
) -> pd.DataFrame:
    """
    过滤 demand_element 为空的记录。

    demand_element 字段允许重复，但不允许为空。

    参数：
        demand_prio: DemandPriority DataFrame
        validation_log: 验证日志

    返回：
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

    参数：
        dp: 部署计划 DataFrame
        mat_map: 物料映射
        validation_log: 验证日志

    返回：
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

    参数：
        dp: 部署计划 DataFrame
        prio_map: 优先级映射

    返回：
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

    参数：
        dp: 部署计划 DataFrame
        validation_log: 验证日志

    返回：
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
