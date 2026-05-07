"""Module4 集成模式入口（对齐其他模块的 `moduleN.run_*` 调用风格）。

此文件仅提供薄包装，将 `module4.run_daily_production_planning_integrated(...)`
委托给 `src.core.main_integration.production_runner.run_daily_production_planning_integrated`。
实际逻辑仍保留在 production_runner.py 中，便于和其余 main_integration 层协作。

为避免 `src.modules.production_planning` ↔ `src.core.main_integration.production_runner`
在模块加载期的循环导入（production_runner 顶部会 `from ...modules import module4`），
这里在函数体内执行延迟导入。
"""

from typing import Any, Dict, Optional

import pandas as pd


def run_daily_production_planning_integrated(
    config_dict: dict,
    module3_output_dir: str,
    simulation_date: pd.Timestamp,
    simulation_start: pd.Timestamp,
    output_dir: str,
    skip_file_output: bool = False,
    module3_result: Optional[Dict[str, Any]] = None,
    previous_line_states_override: Optional[dict] = None,
    allocated_capacity_override: Optional[dict] = None,
    skip_state_file_output: bool = False,
) -> Dict[str, Any]:
    """集成模式运行 Module4 生产计划。

    签名与返回结构与 production_runner.run_daily_production_planning_integrated 保持一致。
    """
    from src.core.main_integration.production_runner import run_daily_production_planning_integrated as _impl

    return _impl(
        config_dict=config_dict,
        module3_output_dir=module3_output_dir,
        simulation_date=simulation_date,
        simulation_start=simulation_start,
        output_dir=output_dir,
        skip_file_output=skip_file_output,
        module3_result=module3_result,
        previous_line_states_override=previous_line_states_override,
        allocated_capacity_override=allocated_capacity_override,
        skip_state_file_output=skip_state_file_output,
    )
