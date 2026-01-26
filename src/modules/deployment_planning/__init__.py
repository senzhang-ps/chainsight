# -*- coding: utf-8 -*-
"""
Deployment Planning Module (部署规划模块)

多层级部署规划模块，在给定网络、需求、库存与产运数据下，
按优先级与约束生成跨节点调拨计划。

运行模式：
- 独立模式：通过 --input/--output/--sim_start/--sim_end 读写Excel
- 集成模式：通过 config_dict + orchestrator + current_date 直接读取各模块输出

关键配置表：
- DeployConfig：按 (material, sending) 维护 moq/rv/lsk/day
- PushPullModel：按 (material, sending) 维护 model ∈ {push, soft push, pull}
- LeadTime：按 (sending, receiving) 维护 PDT/GR/MCT
- SafetyStock：按 (material, location, date) 维护安全库存目标量
- DemandPriority：维护 demand_element -> priority
- ReceivingSpace：按 (receiving, date) 维护 max_qty
- Network：按 (material, location) 维护 sourcing
"""
# 规范化函数
from .normalizer import (
    normalize_identifiers,
    normalize_location,
    normalize_material,
)

# 数据加载函数
from .data_loader import (
    load_config,
    load_integrated_config,
    load_module1_daily_shipment,
    load_module1_daily_orders,
    load_orchestrator_delivery_gr,
    load_orchestrator_open_deployment,
    clear_static_config_cache,
    get_static_config_cache_status,
)

# 库存计算函数
from .inventory import (
    build_open_deployment_inbound,
    calculate_projected_inventory,
    calculate_available_inventory,
)

# 缓存工具函数
from .cache_utils import (
    build_ptf_lsk_cache,
    build_lead_time_cache,
    build_active_network_cache,
    get_ptf_lsk,
    get_upstream,
    get_active_network,
    assign_location_layers,
    determine_lead_time,
    get_sending_location_type,
)

# 需求收集函数
from .demand_collector import collect_node_demands

# 分配函数
from .allocation import (
    apply_moq_rv,
    apply_grouped_moq_rv,
    apply_priority_allocation_vectorized,
    apply_receiving_space_quota,
)

# 推送分配函数
from .push_allocation import push_softpush_allocation

# 验证与输出函数
from .validation import validate_config_before_run, log_outputs

# 主函数
from .main import main

__all__ = [
    # 规范化函数
    'normalize_identifiers',
    'normalize_location',
    'normalize_material',
    # 数据加载函数
    'load_config',
    'load_integrated_config',
    'load_module1_daily_shipment',
    'load_module1_daily_orders',
    'load_orchestrator_delivery_gr',
    'load_orchestrator_open_deployment',
    'clear_static_config_cache',
    'get_static_config_cache_status',
    # 库存计算函数
    'build_open_deployment_inbound',
    'calculate_projected_inventory',
    'calculate_available_inventory',
    # 缓存工具函数
    'build_ptf_lsk_cache',
    'build_lead_time_cache',
    'build_active_network_cache',
    'get_ptf_lsk',
    'get_upstream',
    'get_active_network',
    'assign_location_layers',
    'determine_lead_time',
    'get_sending_location_type',
    # 需求收集函数
    'collect_node_demands',
    # 分配函数
    'apply_moq_rv',
    'apply_grouped_moq_rv',
    'apply_priority_allocation_vectorized',
    'apply_receiving_space_quota',
    # 推送分配函数
    'push_softpush_allocation',
    # 验证与输出函数
    'validate_config_before_run',
    'log_outputs',
    # 主函数
    'main',
]
