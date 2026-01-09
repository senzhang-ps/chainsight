# =============================================================
# Module 5 多层级部署规划（说明与配置指引）
#
# 用途：在给定网络、需求、库存与产运数据下，按优先级与约束生成跨节点调拨计划。
# 运行模式：
# - 独立模式：通过 `--input/--output/--sim_start/--sim_end` 读写Excel。
# - 集成模式：通过 `config_dict + orchestrator + current_date` 直接读取各模块输出。
#
# 本文件为重构后的入口模块，实际实现已拆分至 deployment_planning 子包。
# 所有公开接口均从子包导入，保持对外接口不变。
#
# 关键配置表（列名以实际表为准）：
# - `DeployConfig`：按 (material, sending) 维护 `moq`(最小订货量) 与 `rv`(重订量)。
# - `PushPullModel`：按 (material, sending) 维护 `model` ∈ {push, soft push, pull}。
# - `LeadTime`：按 (sending, receiving) 维护 `PDT/GR/MCT`。
# - `M4_MaterialLocationLineCfg`：按 (material, location) 维护 `PTF/LSK`。
# - `SafetyStock`：按 (material, location, date) 维护安全库存目标量。
# - `DemandPriority`：维护 `demand_element -> priority`。
# - `ReceivingSpace`：按 (receiving, date) 维护 `max_qty`（收货空间上限）。
# - `Network`：按 (material, location) 维护 `sourcing`（上游）。
# - `OrderLog`（集成模式自动从Module1日输出提取）：AO/normal订单参与分配。
# =============================================================
"""
Module 5: 多层级部署规划模块。

本模块提供跨节点调拨计划生成功能，支持独立运行和集成运行两种模式。
重构后的实现已拆分至 deployment_planning 子包，本文件作为对外接口入口。
"""

# 从重构后的子模块导入所有公开接口
from .deployment_planning import (
    # 规范化函数
    normalize_location,
    normalize_material,
    normalize_identifiers,

    # 数据加载函数
    load_config,
    load_integrated_config,
    load_module1_daily_shipment,
    load_module1_daily_orders,
    load_orchestrator_delivery_gr,
    load_orchestrator_open_deployment,

    # 库存计算函数
    build_open_deployment_inbound,
    calculate_projected_inventory,
    calculate_available_inventory,

    # 缓存工具函数
    build_ptf_lsk_cache,
    build_lead_time_cache,
    build_active_network_cache,
    get_ptf_lsk,
    get_active_network,
    get_upstream,
    determine_lead_time,
    assign_location_layers,
    get_sending_location_type,

    # 需求收集函数
    collect_node_demands,

    # 分配函数
    apply_moq_rv,
    apply_grouped_moq_rv,
    apply_priority_allocation_vectorized,
    apply_receiving_space_quota,

    # 推送分配函数
    push_softpush_allocation,

    # 验证与输出函数
    validate_config_before_run,
    log_outputs,

    # 主函数
    main,
)

# 为保持向后兼容，提供私有函数别名
_normalize_location = normalize_location
_normalize_material = normalize_material
_normalize_identifiers = normalize_identifiers

# 导出列表
__all__ = [
    # 规范化函数
    'normalize_location',
    'normalize_material',
    'normalize_identifiers',
    # 数据加载函数
    'load_config',
    'load_integrated_config',
    'load_module1_daily_shipment',
    'load_module1_daily_orders',
    'load_orchestrator_delivery_gr',
    'load_orchestrator_open_deployment',
    # 库存计算函数
    'build_open_deployment_inbound',
    'calculate_projected_inventory',
    'calculate_available_inventory',
    # 缓存工具函数
    'build_ptf_lsk_cache',
    'build_lead_time_cache',
    'build_active_network_cache',
    'get_ptf_lsk',
    'get_active_network',
    'get_upstream',
    'determine_lead_time',
    'assign_location_layers',
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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Module 5: Multi-echelon Deployment Planning'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Input config excel path'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output excel path'
    )
    parser.add_argument(
        '--sim_start',
        required=True,
        help='Simulation start date, YYYY-MM-DD'
    )
    parser.add_argument(
        '--sim_end',
        required=True,
        help='Simulation end date, YYYY-MM-DD'
    )
    args = parser.parse_args()
    main(args.input, args.output, args.sim_start, args.sim_end)
