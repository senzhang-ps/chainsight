"""
常量定义模块

定义Module4中使用的所有常量，包括列名、默认值等。

共享参数从 config.yaml 读取。
"""

from typing import List
from src.config import get_module_config


def _module():
    return get_module_config('production_planning')


# 标识符列名列表
IDENTIFIER_COLS: List[str] = [
    'material',
    'location',
    'line',
    'delegate_line',
    'from_material',
    'to_material',
]

# 默认换产时间（从配置读取）
DEFAULT_CHANGEOVER_TIME: float = _module().get('changeover_time_default', 24.0)

# 生产计划表列名
PLAN_COLUMNS: List[str] = [
    'material',
    'location',
    'line',
    'simulation_date',
    'production_plan_date',
    'available_date',
    'uncon_planned_qty',
    'con_planned_qty',
    'produced_qty',
    'changeover_id',
    'changeover_time',
    'changeover_time_remaining',
    'is_first_changeover_day',
]

# 产能超额表列名
EXCEED_COLUMNS: List[str] = [
    'material',
    'location',
    'line',
    'simulation_date',
    'exceed_type',
    'exceed_qty',
]

# 校验结果表列名
VALIDATION_COLUMNS: List[str] = [
    'type',
    'location',
    'line',
    'production_plan_date',
    'simulation_date',
    'previously_allocated_hours',
    'currently_allocated_hours',
    'total_allocated_hours',
    'message',
    'issue',
    'sheet',
    'row',
]

# 换产日志表列名
CHANGEOVER_LOG_COLUMNS: List[str] = [
    'date',
    'location',
    'line',
    'changeover_type',
    'count',
    'time',
    'cost',
    'mu_loss',
]

# 无约束计划表列名
UNCONSTRAINED_PLAN_COLUMNS: List[str] = [
    'material',
    'location',
    'line',
    'planned_date',
    'uncon_planned_qty',
    'simulation_date',
    'original_quantity',
]

# 必需的配置工作表名称
REQUIRED_CONFIG_SHEETS: List[str] = [
    'M4_MaterialLocationLineCfg',
    'M4_LineCapacity',
    'M4_ChangeoverMatrix',
    'M4_ChangeoverDefinition',
    'M4_ProductionReliability',
    'Global_DemandPriority',
]

# 配置工作表到内部键的映射
SHEET_KEY_MAPPING: dict = {
    'M4_MaterialLocationLineCfg': 'MaterialLocationLineCfg',
    'M4_LineCapacity': 'LineCapacity',
    'M4_ChangeoverMatrix': 'ChangeoverMatrix',
    'M4_ChangeoverDefinition': 'ChangeoverDefinition',
    'M4_ProductionReliability': 'ProductionReliability',
    'Global_DemandPriority': 'NetDemandTypePriority',
}
