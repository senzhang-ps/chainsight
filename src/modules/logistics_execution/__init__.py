# -*- coding: utf-8 -*-
"""
Logistics Execution Module (物流执行模块)

提供供应链的物流发运管理功能，包含以下子模块：
- expression_evaluator: 安全表达式解析器
- config_loader: 配置加载器（独立/集成模式）
- validators: 数据验证工具
- capacity_manager: 车辆容量管理
- delivery_processor: 发货处理器
- inventory_manager: 库存管理
- vehicle_packer: 车辆装载优化
"""

from .expression_evaluator import SafeExpressionEvaluator
from .config_loader import (
    load_standalone_config,
    load_integrated_config
)
from .validators import (
    check_and_deduplicate,
    generate_validation_report,
    validate_deployment_plan,
    validate_truck_config
)
from .capacity_manager import (
    normalize_capacity_plan,
    get_truck_capacity
)
from .delivery_processor import (
    sample_delivery_delay,
    should_bypass_mdq,
    calculate_lead_time
)
from .inventory_manager import (
    calculate_physical_inventory,
    update_inventory_after_load
)
from .main import (
    run_daily_physical_flow,
    run_physical_flow_module,
    main,
)
from .vehicle_packer import (
    VehiclePacker,
    create_load_record,
    calculate_load_ratios
)

__all__ = [
    'SafeExpressionEvaluator',
    'load_standalone_config',
    'load_integrated_config',
    'check_and_deduplicate',
    'generate_validation_report',
    'validate_deployment_plan',
    'validate_truck_config',
    'normalize_capacity_plan',
    'get_truck_capacity',
    'sample_delivery_delay',
    'should_bypass_mdq',
    'calculate_lead_time',
    'calculate_physical_inventory',
    'update_inventory_after_load',
    'run_daily_physical_flow',
    'run_physical_flow_module',
    'main',
    'VehiclePacker',
    'create_load_record',
    'calculate_load_ratios',
]
