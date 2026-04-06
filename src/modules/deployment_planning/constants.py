# -*- coding: utf-8 -*-
"""
常量定义模块

定义Module 5使用的各类常量，包括默认配置值、列名、优先级等。

共享参数从 config.yaml 读取。
"""
from typing import List

# 使用统一的 CPU 配置
from src.utils.cpu_config import CPU_COUNT, MAX_WORKERS
from src.config import get_module_config, get_shared_config


def _shared():
    return get_shared_config()


def _module():
    return get_module_config('deployment_planning')


# 性能优化开关
USE_VECTORIZED_DEMAND_COLLECTION: bool = False  # 暂时关闭，需要修复horizon计算问题
USE_MULTIPROCESS_DEMAND_COLLECTION: bool = False  # 暂时关闭，pickle问题
USE_HORIZON_CACHE: bool = True  # 启用horizon预计算缓存优化
USE_DUCKDB_ACCELERATION: bool = _module().get('use_duckdb_acceleration', True)

# 并行工作线程数 - 使用统一的90%配置
DEFAULT_MAX_WORKERS: int = MAX_WORKERS

# 默认仿真日期范围
DEFAULT_SIM_START: str = '2025-01-01'
DEFAULT_SIM_END: str = '2025-12-31'

# 默认优先级配置
DEFAULT_AO_PRIORITY: int = 1
DEFAULT_NORMAL_PRIORITY: int = 2
DEFAULT_OTHER_PRIORITY: int = 9

# 默认MOQ/RV值（从共享配置读取）
DEFAULT_MOQ: int = _shared().get('default_moq', 1)
DEFAULT_RV: int = _shared().get('default_rv', 1)

# 默认PTF/LSK值（从共享配置读取）
DEFAULT_PTF: int = _shared().get('default_ptf', 0)
DEFAULT_LSK: int = _shared().get('default_lsk', 1)

# 默认lead time（从模块配置读取）
DEFAULT_LEAD_TIME: int = _module().get('default_lead_time', 1)

# 默认push levels（从模块配置读取）
DEFAULT_PUSH_LEVELS: List[float] = _module().get('push_levels', [1.2, 1.5, 2.0, 2.5, 3.0])

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
