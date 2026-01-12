# -*- coding: utf-8 -*-
"""
验证模块

提供配置校验和输出功能。
"""
from typing import Dict, List

import pandas as pd

from .normalizer import normalize_identifiers
from .constants import (
    DEFAULT_AO_PRIORITY,
    DEFAULT_NORMAL_PRIORITY,
    DEFAULT_OTHER_PRIORITY
)


def validate_config_before_run(config: dict, validation_log: list) -> list:
    """
    运行前配置校验与自动补充。

    校验项：
    - Network不允许同一(material, location)多个sourcing
    - 缺失LeadTime/PushPullModel的必要行会记录校验项
    - DemandPriority自动补齐AO=1、normal=2、其他=9

    Args:
        config: 配置字典
        validation_log: 校验日志列表

    Returns:
        list: 更新后的校验日志
    """
    deploy_cfg = config['DeployConfig']
    leadtime_df = config['LeadTime']
    pushpull = config['PushPullModel']
    demand_priority = config['DemandPriority']
    network = config['Network']

    # 校验Network是否有multiple sourcing
    multi_sourcing = (
        network.groupby(['material', 'location'])['sourcing']
        .nunique().reset_index()
    )
    multi_sourcing = multi_sourcing[multi_sourcing['sourcing'] > 1]

    for row in multi_sourcing.itertuples(index=False):
        validation_log.append({
            'No': len(validation_log) + 1,
            'Issue': (
                f"Network配置不合法: material={row.material}, "
                f"location={row.location} 有多个sourcing"
            )
        })

    # 校验leadtime - 使用merge检查缺失
    if not network.empty:
        network_check = network[['sourcing', 'location', 'material']].drop_duplicates()
        lt_keys = leadtime_df[['sending', 'receiving']].drop_duplicates()
        lt_keys = lt_keys.rename(columns={'sending': 'sourcing', 'receiving': 'location'})
        missing_lt = network_check.merge(
            lt_keys, on=['sourcing', 'location'], how='left', indicator=True
        )
        missing_lt = missing_lt[missing_lt['_merge'] == 'left_only']
        for row in missing_lt.itertuples(index=False):
            validation_log.append({
                'No': len(validation_log) + 1,
                'Issue': (
                    f"Missing leadtime for {row.sourcing}->"
                    f"{row.location} ({row.material})"
                )
            })

    # 校验pushpull - 使用merge检查缺失
    if not deploy_cfg.empty:
        cfg_check = deploy_cfg[['material', 'sending']].drop_duplicates()
        pp_keys = pushpull[['material', 'sending']].drop_duplicates()
        missing_pp = cfg_check.merge(
            pp_keys, on=['material', 'sending'], how='left', indicator=True
        )
        missing_pp = missing_pp[missing_pp['_merge'] == 'left_only']
        for row in missing_pp.itertuples(index=False):
            validation_log.append({
                'No': len(validation_log) + 1,
                'Issue': f"Missing PushPullModel for {row.material}/{row.sending}"
            })

    # 校验/补充DemandPriority
    dp = demand_priority.copy()

    # 收集所有需求类型
    sdl_types = (
        set(config['SupplyDemandLog']['demand_element'].unique())
        if not config['SupplyDemandLog'].empty else set()
    )
    ol = config.get('OrderLog', pd.DataFrame())
    ol_types = (
        set(ol['demand_type'].unique())
        if ('demand_type' in ol.columns and not ol.empty) else set()
    )

    needed = sdl_types | ol_types

    # 补充缺失的优先级
    new_rows = []
    existing_elements = (
        set(dp['demand_element'].unique())
        if not dp.empty and 'demand_element' in dp.columns else set()
    )

    for elem in needed:
        if elem in existing_elements:
            continue

        if elem == 'AO':
            default_p = DEFAULT_AO_PRIORITY
        elif elem == 'normal':
            default_p = DEFAULT_NORMAL_PRIORITY
        else:
            default_p = DEFAULT_OTHER_PRIORITY

        new_rows.append({'demand_element': elem, 'priority': default_p})
        validation_log.append({
            'No': len(validation_log) + 1,
            'Issue': f'Auto add DemandPriority for {elem}={default_p}'
        })
        existing_elements.add(elem)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        dp = pd.concat([dp, new_df], ignore_index=True)

    config['DemandPriority'] = dp
    return validation_log


def log_outputs(output_path: str, outputs: Dict[str, pd.DataFrame]) -> None:
    """
    将结果表写入Excel。

    Args:
        output_path: 输出文件路径
        outputs: 输出表字典
    """
    # 定义各表的排序键，确保输出顺序一致性
    sort_keys = {
        'DeploymentPlan': ['date', 'material', 'sending', 'receiving', 'demand_element', 
                          'planned_delivery_date', 'demand_qty'],
        'UnfulfilledLog': ['date', 'material', 'location', 'demand_element'],
        'StockOnHandLog': ['date', 'material', 'location'],
        'Validation': ['No']
    }
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for sheet, df in outputs.items():
            if df.empty:
                pd.DataFrame(columns=df.columns).to_excel(
                    writer, sheet_name=sheet, index=False
                )
            else:
                normalized_df = normalize_identifiers(df)
                # 应用排序以确保输出顺序一致
                if sheet in sort_keys:
                    available_keys = [k for k in sort_keys[sheet] if k in normalized_df.columns]
                    if available_keys:
                        normalized_df = normalized_df.sort_values(available_keys).reset_index(drop=True)
                normalized_df.to_excel(writer, sheet_name=sheet, index=False)
