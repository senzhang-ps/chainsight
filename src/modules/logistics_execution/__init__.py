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
- main: 主入口函数
- simulation: 仿真循环与路线处理
- output_writer: 输出组装与文件写入
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
from .vehicle_packer import (
    VehiclePacker,
    create_vehicle_log_entry,
    determine_trigger_cause,
    get_representative_context,
)

# 主入口函数
from .main import (
    run_daily_physical_flow,
    run_physical_flow_module,
)

__all__ = [
    # 入口函数
    'run_daily_physical_flow',
    'run_physical_flow_module',
    # 子模块导出
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
    'VehiclePacker',
    'create_vehicle_log_entry',
    'determine_trigger_cause',
    'get_representative_context',
]
