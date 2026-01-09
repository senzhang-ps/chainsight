"""
module1.py

整体目的：
- 模块1负责需求预测与订单生成的日度执行逻辑：加载并处理需求预测配置、
  将周度预测拆分为日度预测、生成AO（提前订单）与Normal（普通订单）、
  计算发货与缺货、生成供需日志。

功能点：
- 配置加载与校验：读取 M1 相关配置并进行必要的验证与类型标准化
- DPS地点拆分：按DPS配置进行地点拆分
- 供应选择：应用供应选择调整到周度预测
- 预测拆分：将周度预测拆分为日度预测（整数分配）
- 订单生成：生成AO和Normal订单
- 发货计算：基于库存生成发货与缺货
- 供需日志：生成集成模式供需日志

使用方法：
- 日度模式：调用 `run_daily_order_generation(...)` 处理单日生成每日输出
- 集成模式：由 `main_integration.py` 直接调用内部函数

注意：
本模块已按照Python代码规范重构，核心逻辑位于 demand_planning 子包中。
本文件作为兼容层，保持原有接口不变。
"""

# =============================================================================
# 标准库导入
# =============================================================================

import os
from typing import Optional, Dict, Any, Tuple, List

import pandas as pd
import numpy as np


# =============================================================================
# 从重构子模块导入（保持向后兼容）
# =============================================================================

# 常量
from .demand_planning.constants import (
    DEFAULT_MAX_ADVANCE_DAYS,
    DEFAULT_USE_PARALLEL_AO_CONSUME,
    DEFAULT_USE_PARALLEL_FILE_LOAD,
    DEFAULT_PARALLEL_MAX_WORKERS,
    DEFAULT_ERROR_LOG_PATH,
    DEFAULT_USE_PARALLEL_NORMAL_CONSUME,
    append_error_log as _append_error_log,
)

# 工具函数（使用原始命名以保持兼容）
from .demand_planning.normalization import (
    normalize_location as _normalize_location,
    normalize_material as _normalize_material,
    normalize_identifiers as _normalize_identifiers,
)

# 同时导出公开名称
from .demand_planning.normalization import (
    normalize_location,
    normalize_material,
    normalize_identifiers,
)

# 配置加载
from .demand_planning.config import (
    load_config,
    validate_m1_config as _validate_m1_config,
)

# DPS与供应选择
from .demand_planning.dps import (
    apply_dps,
    apply_supply_choice,
)

# 预测拆分
from .demand_planning.forecast import (
    expand_forecast_to_days_integer_split,
    prepare_daily_forecasts as _prepare_daily_forecasts,
)

# 订单生成
from .demand_planning.order import (
    generate_daily_orders,
    generate_quantity_with_percent_error,
    consume_forecast_ao_logic,
    consume_forecast_normal_logic,
)

# 订单消耗
from .demand_planning.consume import (
    consume_orders as _consume_orders,
    consume_ao_orders_serial as _consume_ao_orders_serial,
    consume_normal_orders_serial as _consume_normal_orders_serial,
)

# 发货计算
from .demand_planning.shipment import (
    simulate_shipment_for_single_day,
    generate_shipment_with_inventory_check,
)

# 集成模式
from .demand_planning.integration import (
    run_daily_order_generation,
    generate_supply_demand_log_for_integration,
)

# IO工具
from .demand_planning.io_utils import (
    load_previous_orders as _load_previous_orders,
    save_module1_output_with_supply_demand,
)


# =============================================================================
# 导出所有公开接口
# =============================================================================

__all__ = [
    # 常量
    'DEFAULT_MAX_ADVANCE_DAYS',
    'DEFAULT_USE_PARALLEL_AO_CONSUME',
    'DEFAULT_USE_PARALLEL_FILE_LOAD',
    'DEFAULT_PARALLEL_MAX_WORKERS',
    'DEFAULT_ERROR_LOG_PATH',
    'DEFAULT_USE_PARALLEL_NORMAL_CONSUME',
    # 规范化（公开名称）
    'normalize_location',
    'normalize_material',
    'normalize_identifiers',
    # 规范化（私有名称，向后兼容）
    '_normalize_location',
    '_normalize_material',
    '_normalize_identifiers',
    '_append_error_log',
    # 配置
    'load_config',
    '_validate_m1_config',
    # DPS
    'apply_dps',
    'apply_supply_choice',
    # 预测
    'expand_forecast_to_days_integer_split',
    '_prepare_daily_forecasts',
    # 订单
    'generate_daily_orders',
    'generate_quantity_with_percent_error',
    'consume_forecast_ao_logic',
    'consume_forecast_normal_logic',
    # 消耗
    '_consume_orders',
    '_consume_ao_orders_serial',
    '_consume_normal_orders_serial',
    # 发货
    'simulate_shipment_for_single_day',
    'generate_shipment_with_inventory_check',
    # 集成
    'run_daily_order_generation',
    'generate_supply_demand_log_for_integration',
    # IO
    '_load_previous_orders',
    'save_module1_output_with_supply_demand',
]
