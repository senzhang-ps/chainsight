# -*- coding: utf-8 -*-
"""
物流执行模块 - 运行参数初始化
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .config_loader import load_integrated_config, load_standalone_config


def initialize_run_params(
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

    参数：
        各种输入参数

    返回：
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

    参数：
        config_dict: 配置字典
        orchestrator: Orchestrator 实例
        current_date: 当前日期字符串
        output_path: 输出路径

    返回：
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

    参数：
        input_excel: 输入文件路径
        simulation_start: 开始日期
        simulation_end: 结束日期
        output_excel: 输出文件路径
        max_wait_days: 最大等待天数

    返回：
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
