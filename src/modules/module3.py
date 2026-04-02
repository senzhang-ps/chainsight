"""
Module3: MRP层级模拟模块。

该模块负责供应链网络中的物料需求计划(MRP)计算，
包括净需求计算、层级传递和MOQ/RV应用等核心功能。

主要功能:
    - 计算每日净需求(Net Demand)
    - 按层级传递需求缺口
    - 应用MOQ/RV约束进行需求调整
    - 与Module1/Module5协同工作

重构说明:
    本模块已重构为多个子模块，详见mrp_planning包。
    此文件作为向后兼容的入口，导出所有公开API。

版本: 2.0.0
"""

# 从重构后的子模块导入所有公开API
from .mrp_planning import (
    # 常量
    DEFAULT_MOQ,
    DEFAULT_RV,
    DEFAULT_HORIZON,
    # 工具函数
    apply_moq_rv,
    normalize_location,
    normalize_material,
    normalize_identifiers,
    apportion_largest_remainder,
    lookup_moq_rv_three_keys,
    build_ptf_lsk_cache,
    get_ptf_lsk,
    # 配置加载
    load_config,
    load_module1_daily_outputs,
    load_excel_with_sheets,
    # 提前期计算
    compute_root_horizon,
    determine_lead_time,
    infer_sending_location_type,
    # 核心功能
    calculate_daily_net_demand,
    run_mrp_layered_simulation_daily,
    assign_location_layers,
    run_integrated_mode,
)

# 为向后兼容保留的别名
_normalize_location = normalize_location
_normalize_material = normalize_material
_normalize_identifiers = normalize_identifiers
_lookup_moq_rv_three_keys = lookup_moq_rv_three_keys
_apportion_largest_remainder = apportion_largest_remainder
_build_ptf_lsk_cache_m3 = build_ptf_lsk_cache
_get_ptf_lsk = get_ptf_lsk
_compute_root_horizon = compute_root_horizon

__all__ = [
    # 常量
    'DEFAULT_MOQ',
    'DEFAULT_RV',
    'DEFAULT_HORIZON',
    # 核心函数
    'apply_moq_rv',
    'load_config',
    'load_module1_daily_outputs',
    'load_excel_with_sheets',
    'assign_location_layers',
    'infer_sending_location_type',
    'determine_lead_time',
    'calculate_daily_net_demand',
    'run_mrp_layered_simulation_daily',
    'run_integrated_mode',
    # 工具函数
    'normalize_location',
    'normalize_material',
    'normalize_identifiers',
    'apportion_largest_remainder',
    'lookup_moq_rv_three_keys',
    'build_ptf_lsk_cache',
    'get_ptf_lsk',
    'compute_root_horizon',
    # 向后兼容别名
    '_normalize_location',
    '_normalize_material',
    '_normalize_identifiers',
    '_lookup_moq_rv_three_keys',
    '_apportion_largest_remainder',
    '_build_ptf_lsk_cache_m3',
    '_get_ptf_lsk',
    '_compute_root_horizon',
]
