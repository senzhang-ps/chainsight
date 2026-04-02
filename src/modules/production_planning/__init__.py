"""
Production Planning Module (生产计划模块)

该包包含生产计划的所有子模块，按照Python代码规范拆分为以下子模块:
- constants: 常量定义
- types: 类型定义和数据类
- utils: 通用工具函数
- state_manager: 状态持久化管理
- config_loader: 配置加载和校验
- demand_loader: 净需求数据加载
- plan_builder: 无约束计划构建
- capacity_allocator: 产能分配和换产处理
- output_writer: 输出文件生成

使用方法:
    from src.modules.production_planning import run_daily_production_planning
"""

from .constants import (
    IDENTIFIER_COLS,
    DEFAULT_CHANGEOVER_TIME,
    PLAN_COLUMNS,
    EXCEED_COLUMNS,
    VALIDATION_COLUMNS,
    CHANGEOVER_LOG_COLUMNS,
)
from .types import LineState, ChangeoverInfo, PlanRecord, ExceedRecord
from .utils import (
    normalize_location,
    cast_identifiers_to_str,
    validate_merge_keys,
    compute_planning_window,
    is_review_day,
    dedup_issues,
)
from .state_manager import (
    get_or_init_simulation_start,
    save_line_state,
    load_line_state,
    save_allocated_capacity,
    load_allocated_capacity,
    load_all_previous_capacity,
)
from .config_loader import load_config, validate_config
from .demand_loader import load_daily_net_demand
from .plan_builder import (
    build_unconstrained_plan_for_single_day,
    optimal_changeover_sequence,
)
from .capacity_allocator import (
    centralized_capacity_allocation_with_changeover,
    extract_allocated_capacity_from_plan,
    validate_capacity_allocation,
    extract_line_states_from_plan,
    calculate_changeover_metrics,
    simulate_production,
    _analyze_end_of_day_changeover,
)
from .output_writer import write_output, generate_consolidated_output
from .main import run_daily_production_planning, main

__all__ = [
    # 常量
    'IDENTIFIER_COLS',
    'DEFAULT_CHANGEOVER_TIME',
    'PLAN_COLUMNS',
    'EXCEED_COLUMNS',
    'VALIDATION_COLUMNS',
    'CHANGEOVER_LOG_COLUMNS',
    # 类型
    'LineState',
    'ChangeoverInfo',
    'PlanRecord',
    'ExceedRecord',
    # 工具函数
    'normalize_location',
    'cast_identifiers_to_str',
    'validate_merge_keys',
    'compute_planning_window',
    'is_review_day',
    'dedup_issues',
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
    'calculate_changeover_metrics',
    'simulate_production',
    # 输出
    'write_output',
    'generate_consolidated_output',
    # 主入口
    'run_daily_production_planning',
    'main',
]
