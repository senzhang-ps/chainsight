"""Production planning package public API.

This package contains the refactored Module4 public API.
"""

from .constants import (
    CHANGEOVER_LOG_COLUMNS,
    DEFAULT_CHANGEOVER_TIME,
    EXCEED_COLUMNS,
    IDENTIFIER_COLS,
    PLAN_COLUMNS,
    REQUIRED_CONFIG_SHEETS,
    SHEET_KEY_MAPPING,
    UNCONSTRAINED_PLAN_COLUMNS,
    VALIDATION_COLUMNS,
)
from .types import (
    ChangeoverInfo,
    ExceedRecord,
    LineState,
    PlanRecord,
    ValidationIssue,
)
from .utils import (
    cast_identifiers_to_str,
    dedup_issues,
    ensure_dataframe_columns,
    round_up_to_batch,
    safe_float_conversion,
    validate_merge_keys,
)
from .state_manager import (
    get_or_init_simulation_start,
    load_all_previous_capacity,
    load_allocated_capacity,
    load_line_state,
    save_allocated_capacity,
    save_line_state,
)
from .config_loader import load_config, validate_config
from .demand_loader import load_daily_net_demand
from .plan_builder import (
    build_unconstrained_plan_for_single_day,
    optimal_changeover_sequence,
)
from .capacity_allocator import (
    _analyze_end_of_day_changeover,
    calculate_changeover_metrics,
    centralized_capacity_allocation_with_changeover,
    extract_allocated_capacity_from_plan,
    extract_line_states_from_plan,
    simulate_production,
    validate_capacity_allocation,
)
from .output_writer import generate_consolidated_output, write_output
from .main import DailyProductionPlanner, run_daily_production_planning
from .integration import run_daily_production_planning_integrated


__all__ = [
    "IDENTIFIER_COLS",
    "DEFAULT_CHANGEOVER_TIME",
    "PLAN_COLUMNS",
    "EXCEED_COLUMNS",
    "VALIDATION_COLUMNS",
    "CHANGEOVER_LOG_COLUMNS",
    "UNCONSTRAINED_PLAN_COLUMNS",
    "REQUIRED_CONFIG_SHEETS",
    "SHEET_KEY_MAPPING",
    "LineState",
    "ChangeoverInfo",
    "PlanRecord",
    "ExceedRecord",
    "ValidationIssue",
    "cast_identifiers_to_str",
    "validate_merge_keys",
    "dedup_issues",
    "round_up_to_batch",
    "safe_float_conversion",
    "ensure_dataframe_columns",
    "get_or_init_simulation_start",
    "save_line_state",
    "load_line_state",
    "save_allocated_capacity",
    "load_allocated_capacity",
    "load_all_previous_capacity",
    "load_config",
    "validate_config",
    "load_daily_net_demand",
    "build_unconstrained_plan_for_single_day",
    "optimal_changeover_sequence",
    "centralized_capacity_allocation_with_changeover",
    "extract_allocated_capacity_from_plan",
    "validate_capacity_allocation",
    "extract_line_states_from_plan",
    "_analyze_end_of_day_changeover",
    "calculate_changeover_metrics",
    "simulate_production",
    "write_output",
    "generate_consolidated_output",
    "run_daily_production_planning",
    "run_daily_production_planning_integrated",
    "DailyProductionPlanner",
]
