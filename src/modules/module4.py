"""
module4.py

整体目的：
- 模块4负责工业级 APS 生产计划的日度执行逻辑：读取净需求，依据产线配置、
  产能与换产矩阵进行无约束计划与集中产能分配，跟踪跨天换产连续性，
  并输出生产与校验日志。

功能点：
- 配置加载与校验：读取 M4 相关配置并进行必要的验证与类型标准化
- 净需求读取：按日从 Module3 输出读取净需求数据，筛选层级与日期
- 产能分配与换产：支持换产矩阵与定义，进行最优序列与集中分配
- 跨天连续性：保存/恢复产线状态与已分配产能
- 输出与汇总：每日输出生产计划、超额、校验与换产日志

使用方法：
- 日度模式：调用 `run_daily_production_planning(...)` 处理单日生成每日输出
- 集成模式：由 `main_integration.py` 直接调用内部函数

注意：
本模块已按照Python代码规范重构，核心逻辑位于 production_planning 子包中。
本文件作为兼容层，保持原有接口不变。
"""

# =============================================================================
# 标准库导入
# =============================================================================

import os
import argparse
from typing import Optional, List, Dict, Any, Tuple
from datetime import timedelta

import pandas as pd
import numpy as np


# =============================================================================
# 从重构子模块导入（保持向后兼容）
# =============================================================================

# 常量
from .production_planning.constants import (
    IDENTIFIER_COLS,
    DEFAULT_CHANGEOVER_TIME,
    PLAN_COLUMNS,
    EXCEED_COLUMNS,
    VALIDATION_COLUMNS,
    CHANGEOVER_LOG_COLUMNS,
    UNCONSTRAINED_PLAN_COLUMNS,
    REQUIRED_CONFIG_SHEETS,
    SHEET_KEY_MAPPING,
)

# 工具函数（使用原始命名以保持兼容）
from .production_planning.utils import (
    normalize_location as _normalize_location,
    cast_identifiers_to_str as _cast_identifiers_to_str,
    validate_merge_keys as _validate_merge_keys,
    compute_planning_window,
    is_review_day,
    dedup_issues,
    round_up_to_batch,
    safe_float_conversion,
    ensure_dataframe_columns,
)

# 状态管理
from .production_planning.state_manager import (
    get_or_init_simulation_start,
    save_line_state,
    load_line_state,
    save_allocated_capacity,
    load_allocated_capacity,
    load_all_previous_capacity,
)

# 配置加载
from .production_planning.config_loader import (
    load_config,
    validate_config,
)

# 需求加载
from .production_planning.demand_loader import (
    load_daily_net_demand,
)

# 计划构建
from .production_planning.plan_builder import (
    build_unconstrained_plan_for_single_day,
    optimal_changeover_sequence,
)

# 产能分配
from .production_planning.capacity_allocator import (
    centralized_capacity_allocation_with_changeover,
    extract_allocated_capacity_from_plan,
    validate_capacity_allocation,
    extract_line_states_from_plan,
    calculate_changeover_metrics,
    simulate_production,
    _analyze_end_of_day_changeover,
)

# 输出管理
from .production_planning.output_writer import (
    write_output,
    generate_consolidated_output,
)

# 主函数
from .production_planning.main import (
    run_daily_production_planning,
    main,
    DailyProductionPlanner,
)


# =============================================================================
# 向后兼容：保留原始函数名（别名）
# =============================================================================

def analyze_end_of_day_changeover_state(
    plan_df: pd.DataFrame,
    cap_df: pd.DataFrame,
    co_def: Dict[Tuple, float],
    simulation_date: pd.Timestamp,
    rate_map: Dict[Tuple, float]
) -> Dict[str, Any]:
    """分析日末换产状态（向后兼容函数）。

    通过重建分配逻辑，检测即便未产生生产记录也可能已启动但未完成的换产，
    并推断剩余时间。

    Args:
        plan_df: 当日生产计划
        cap_df: 产能数据
        co_def: 换产定义字典
        simulation_date: 当前仿真日期
        rate_map: 产率映射，用于计算生产时间

    Returns:
        Dict[str, Any]: 产线换产状态字典
    """
    return _analyze_end_of_day_changeover(
        plan_df, cap_df, co_def, simulation_date, rate_map
    )


# =============================================================================
# 类型定义（从types模块导出以保持兼容）
# =============================================================================

from .production_planning.types import (
    LineState,
    ChangeoverInfo,
    PlanRecord,
    ExceedRecord,
    ValidationIssue,
)


# =============================================================================
# 导出所有公共接口
# =============================================================================

__all__ = [
    # 常量
    'IDENTIFIER_COLS',
    'DEFAULT_CHANGEOVER_TIME',
    'PLAN_COLUMNS',
    'EXCEED_COLUMNS',
    'VALIDATION_COLUMNS',
    'CHANGEOVER_LOG_COLUMNS',
    'UNCONSTRAINED_PLAN_COLUMNS',
    'REQUIRED_CONFIG_SHEETS',
    'SHEET_KEY_MAPPING',
    # 工具函数
    '_normalize_location',
    '_cast_identifiers_to_str',
    '_validate_merge_keys',
    'compute_planning_window',
    'is_review_day',
    'dedup_issues',
    'round_up_to_batch',
    'safe_float_conversion',
    'ensure_dataframe_columns',
    # 状态管理
    'get_or_init_simulation_start',
    'save_line_state',
    'load_line_state',
    'save_allocated_capacity',
    'load_allocated_capacity',
    'load_all_previous_capacity',
    # 配置
    'load_config',
    'validate_config',
    # 需求
    'load_daily_net_demand',
    # 计划构建
    'build_unconstrained_plan_for_single_day',
    'optimal_changeover_sequence',
    # 产能分配
    'centralized_capacity_allocation_with_changeover',
    'extract_allocated_capacity_from_plan',
    'validate_capacity_allocation',
    'extract_line_states_from_plan',
    'analyze_end_of_day_changeover_state',
    'calculate_changeover_metrics',
    'simulate_production',
    # 输出
    'write_output',
    'generate_consolidated_output',
    # 主函数
    'run_daily_production_planning',
    'main',
    'DailyProductionPlanner',
    # 类型
    'LineState',
    'ChangeoverInfo',
    'PlanRecord',
    'ExceedRecord',
    'ValidationIssue',
]


if __name__ == '__main__':
    main()
