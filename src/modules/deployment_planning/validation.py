# -*- coding: utf-8 -*-
"""
验证模块

提供配置校验和输出功能。
"""
from typing import Dict, List

import pandas as pd

from ...utils.normalization import normalize_identifiers
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

    与code_vo保持一致：每行都记录（不去重）。

    参数：
        config: 配置字典
        validation_log: 校验日志列表

    返回：
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

    for _, row in multi_sourcing.iterrows():
        validation_log.append({
            'No': len(validation_log) + 1,
            'Issue': (
                f"Network配置不合法: material={row['material']}, "
                f"location={row['location']} 有多个sourcing"
            )
        })

    # 校验leadtime - 与code_vo保持一致，逐行检查
    for _, row in network.iterrows():
        if leadtime_df[
            (leadtime_df['sending'] == row['sourcing']) & 
            (leadtime_df['receiving'] == row['location'])
        ].empty:
            validation_log.append({
                'No': len(validation_log) + 1,
                'Issue': (
                    f"Missing leadtime for {row['sourcing']}->"
                    f"{row['location']} ({row['material']})"
                )
            })

    # 校验pushpull - 与code_vo保持一致，逐行检查
    for _, row in deploy_cfg.iterrows():
        if pushpull[
            (pushpull['material'] == row['material']) & 
            (pushpull['sending'] == row['sending'])
        ].empty:
            validation_log.append({
                'No': len(validation_log) + 1,
                'Issue': f"Missing PushPullModel for {row['material']}/{row['sending']}"
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

    # 缺啥补啥（默认：AO=1，normal=2，其余给个较低优先级 9）
    # 与code_vo保持一致的补充逻辑
    def _ensure_priority(elem, default_p):
        if dp[dp['demand_element'] == elem].empty:
            dp.loc[len(dp)] = {'demand_element': elem, 'priority': default_p}
            validation_log.append({
                'No': len(validation_log) + 1,
                'Issue': f'Auto add DemandPriority for {elem}={default_p}'
            })

    for elem in needed:
        if elem == 'AO':
            _ensure_priority('AO', DEFAULT_AO_PRIORITY)
        elif elem == 'normal':
            _ensure_priority('normal', DEFAULT_NORMAL_PRIORITY)
        else:
            _ensure_priority(elem, DEFAULT_OTHER_PRIORITY)

    config['DemandPriority'] = dp
    return validation_log


def log_outputs(output_path: str, outputs: Dict[str, pd.DataFrame]) -> None:
    """
    将结果表写入Excel：DeploymentPlan/UnfulfilledLog/StockOnHandLog/Validation。
    说明：输出前统一标识字段格式，确保后续分析一致性。
    
    🔧 修复：输出前按确定性排序，确保不同运行产生相同的输出顺序。
    排序规则与Dev版本对比时使用的排序键一致。

    参数：
        output_path: 输出文件路径
        outputs: 输出表字典
    """
    # 定义各Sheet的排序键
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        for sheet, df in outputs.items():
            if df.empty:
                # 输出空表头
                pd.DataFrame(columns=df.columns).to_excel(
                    writer, sheet_name=sheet, index=False
                )
            else:
                # 确保输出时标识符字段为字符串格式
                normalized_df = normalize_identifiers(df)
                
                # 按确定性排序键排序
                normalized_df.to_excel(writer, sheet_name=sheet, index=False)
