# -*- coding: utf-8 -*-
"""
Module6 - 实物流执行管理模块 (物流管理模块)

提供供应链的物流发运管理功能，包括：
- 车辆装载优化
- 发运计划生成
- MDQ (最小发货量) 规则处理
- 延迟采样和交货时间计算

集成模式支持:
- 独立模式: Excel 文件输入输出（旧版流程）
- 集成模式: 配置字典 + 编排器集成

典型用法示例:
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

from typing import Any, Dict, Optional

import pandas as pd

from .data_preparer import prepare_data
from .initializer import initialize_run_params
from .output_builder import generate_outputs
from .simulation_engine import run_simulation_loop


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
    max_wait_days: int = 30,
    random_seed: Optional[int] = None,
    skip_file_output: bool = False
) -> Dict[str, Any]:
    """
    物流模块主入口函数。

    支持两种运行模式：
    - 独立模式：使用 Excel 文件输入输出
    - 集成模式：与 Orchestrator 集成运行

    参数：
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

    返回：
        包含处理结果的字典
    """
    # 初始化运行参数
    run_params = initialize_run_params(
        input_excel, simulation_start, simulation_end, output_excel,
        config_dict, orchestrator, current_date, output_path,
        max_wait_days, random_seed
    )

    # 数据准备和验证
    prepared_data = prepare_data(run_params)

    # 执行主仿真循环
    results = run_simulation_loop(run_params, prepared_data)

    # 生成输出
    return generate_outputs(
        run_params, results, prepared_data['validation_log'],
        skip_file_output
    )


# 主函数别名（保持与 Module4/5 一致）
main = run_physical_flow_module
