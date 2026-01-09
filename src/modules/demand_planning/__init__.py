"""Demand Planning Module (需求规划模块)

本模块负责需求预测与订单生成：
- 加载并处理需求预测配置
- 将周度预测拆分为日度预测
- 生成AO（提前订单）与Normal（普通订单）
- 计算发货与缺货
- 生成供需日志

子模块结构：
- constants: 常量定义与默认配置
- normalization: 标识符规范化函数
- config: 配置加载与校验
- dps: DPS地点拆分与供应选择
- forecast: 预测拆分与处理
- order: 订单生成与消耗
- shipment: 发货与缺货计算
- integration: 集成模式主入口

主要导出：
    - run_daily_order_generation: 集成模式主入口
    - load_config: 加载Excel配置文件
    - apply_dps: DPS地点拆分
    - apply_supply_choice: 供应选择调整
    - expand_forecast_to_days_integer_split: 周度转日度预测
    - generate_daily_orders: 生成单日订单
"""

from .constants import (
    DEFAULT_MAX_ADVANCE_DAYS,
    DEFAULT_USE_PARALLEL_AO_CONSUME,
    DEFAULT_USE_PARALLEL_FILE_LOAD,
    DEFAULT_PARALLEL_MAX_WORKERS,
    DEFAULT_ERROR_LOG_PATH,
    DEFAULT_USE_PARALLEL_NORMAL_CONSUME,
    append_error_log,
)
from .normalization import (
    normalize_location,
    normalize_material,
    normalize_identifiers,
)
from .config import load_config, validate_m1_config
from .dps import apply_dps, apply_supply_choice
from .forecast import expand_forecast_to_days_integer_split, prepare_daily_forecasts
from .order import (
    generate_daily_orders,
    generate_quantity_with_percent_error,
    consume_forecast_ao_logic,
    consume_forecast_normal_logic,
)
from .consume import (
    consume_orders,
    consume_ao_orders_serial,
    consume_normal_orders_serial,
)
from .shipment import (
    simulate_shipment_for_single_day,
    generate_shipment_with_inventory_check,
)
from .integration import (
    run_daily_order_generation,
    generate_supply_demand_log_for_integration,
)
from .io_utils import (
    load_previous_orders,
    save_module1_output_with_supply_demand,
)

# 向后兼容：保留私有函数名
_normalize_location = normalize_location
_normalize_material = normalize_material
_normalize_identifiers = normalize_identifiers
_append_error_log = append_error_log
_validate_m1_config = validate_m1_config
_prepare_daily_forecasts = prepare_daily_forecasts

__all__ = [
    # 常量
    'DEFAULT_MAX_ADVANCE_DAYS',
    'DEFAULT_USE_PARALLEL_AO_CONSUME',
    'DEFAULT_USE_PARALLEL_FILE_LOAD',
    'DEFAULT_PARALLEL_MAX_WORKERS',
    'DEFAULT_ERROR_LOG_PATH',
    'DEFAULT_USE_PARALLEL_NORMAL_CONSUME',
    'append_error_log',
    # 规范化（公开名称）
    'normalize_location',
    'normalize_material',
    'normalize_identifiers',
    # 规范化（旧私有名称，向后兼容）
    '_normalize_location',
    '_normalize_material',
    '_normalize_identifiers',
    '_append_error_log',
    # 配置
    'load_config',
    'validate_m1_config',
    '_validate_m1_config',
    # DPS
    'apply_dps',
    'apply_supply_choice',
    # 预测
    'expand_forecast_to_days_integer_split',
    'prepare_daily_forecasts',
    '_prepare_daily_forecasts',
    # 订单
    'generate_daily_orders',
    'generate_quantity_with_percent_error',
    'consume_forecast_ao_logic',
    'consume_forecast_normal_logic',
    # 消耗
    'consume_orders',
    'consume_ao_orders_serial',
    'consume_normal_orders_serial',
    # 发货
    'simulate_shipment_for_single_day',
    'generate_shipment_with_inventory_check',
    # 集成
    'run_daily_order_generation',
    'generate_supply_demand_log_for_integration',
    # IO
    'load_previous_orders',
    'save_module1_output_with_supply_demand',
]
