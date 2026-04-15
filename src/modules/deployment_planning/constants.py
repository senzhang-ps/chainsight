# -*- coding: utf-8 -*-
"""
常量定义模块

定义Module 5使用的各类常量，包括默认配置值、列名、优先级等。

优化历史:
- v3.0: 添加 USE_VECTORIZED_DEMAND_COLLECTION 开关
- v3.1: 添加 USE_DUCKDB_ACCELERATION 开关，提升并行度
"""
from typing import List

# 使用统一的 CPU 配置
from src.utils.resource_config import CPU_COUNT, MAX_WORKERS

# 性能优化开关
USE_VECTORIZED_DEMAND_COLLECTION: bool = False  # 暂时关闭，需要修复horizon计算问题
USE_MULTIPROCESS_DEMAND_COLLECTION: bool = False  # 暂时关闭，pickle问题
USE_HORIZON_CACHE: bool = True  # 启用horizon预计算缓存优化
USE_DUCKDB_ACCELERATION: bool = True  # 启用DuckDB加速

# 并行工作线程数 - 使用统一的90%配置
DEFAULT_MAX_WORKERS: int = MAX_WORKERS

# 默认仿真日期范围
DEFAULT_SIM_START: str = '2025-01-01'
DEFAULT_SIM_END: str = '2025-12-31'

# 默认优先级配置
DEFAULT_AO_PRIORITY: int = 1
DEFAULT_NORMAL_PRIORITY: int = 2
DEFAULT_OTHER_PRIORITY: int = 9

# 默认业务参数 —— 统一来源
from src.utils.defaults import (
    DEFAULT_MOQ,
    DEFAULT_RV,
    DEFAULT_PTF,
    DEFAULT_LSK,
    DEFAULT_LEAD_TIME,
    DEFAULT_PUSH_LEVELS,
)

# 标识符列名
IDENTIFIER_COLUMNS: List[str] = [
    'material',
    'location',
    'sending',
    'receiving',
    'sourcing'
]

# 位置相关列名
LOCATION_COLUMNS: List[str] = [
    'location',
    'sending',
    'receiving',
    'sourcing'
]

# 日期字段配置
DATE_FIELDS_MAP: dict = {
    'SupplyDemandLog': ['date'],
    'ProductionPlan': ['available_date'],
    'InventoryLog': ['date'],
    'InTransit': ['available_date'],
    'SafetyStock': ['date'],
    'ReceivingSpace': ['date'],
    'Network': ['eff_from', 'eff_to'],
    'OrderLog': ['date', 'simulation_date'],
}

# 必需的工作表
REQUIRED_SHEETS: List[str] = [
    'SupplyDemandLog',
    'ProductionPlan',
    'InventoryLog',
    'InTransit',
    'SafetyStock',
    'Network',
    'PushPullModel',
    'ReceivingSpace',
    'LeadTime',
    'DemandPriority',
    'DeployConfig'
]

# ``SupplyDemandLog`` 必需列
SDL_REQUIRED_COLUMNS: List[str] = [
    'date',
    'material',
    'location',
    'demand_element',
    'quantity'
]
