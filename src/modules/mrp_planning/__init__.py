"""
MRP Planning Module (MRP计划模块)

物料需求计划(MRP)层级模拟模块，包含以下子模块:
    - constants: 常量定义
    - utils: 通用工具函数
    - config_loader: 配置加载
    - lead_time: 提前期计算
    - layer_assignment: 层级分配
    - node_processor: 节点处理器
    - net_demand: 净需求计算
    - mrp_simulation: MRP模拟核心逻辑
    - integration: 集成模式入口

作者: ChainSight Team
版本: 2.0.0
"""

from .constants import DEFAULT_MOQ, DEFAULT_RV, DEFAULT_HORIZON
from .utils import (
    apply_moq_rv,
    normalize_location,
    normalize_material,
    normalize_identifiers,
    apportion_largest_remainder,
    lookup_moq_rv_three_keys,
    build_ptf_lsk_cache,
    get_ptf_lsk,
)
from .config_loader import (
    load_config,
    load_module1_daily_outputs,
    load_excel_with_sheets,
)
from .lead_time import (
    compute_root_horizon,
    determine_lead_time,
    infer_sending_location_type,
)
from .layer_assignment import assign_location_layers
from .net_demand import calculate_daily_net_demand
from .mrp_simulation import run_mrp_layered_simulation_daily
from .integration import run_integrated_mode

__all__ = [
    # 常量
    'DEFAULT_MOQ',
    'DEFAULT_RV',
    'DEFAULT_HORIZON',
    # 工具函数
    'apply_moq_rv',
    'normalize_location',
    'normalize_material',
    'normalize_identifiers',
    'apportion_largest_remainder',
    'lookup_moq_rv_three_keys',
    'build_ptf_lsk_cache',
    'get_ptf_lsk',
    # 配置加载
    'load_config',
    'load_module1_daily_outputs',
    'load_excel_with_sheets',
    # 提前期计算
    'compute_root_horizon',
    'determine_lead_time',
    'infer_sending_location_type',
    # 层级分配
    'assign_location_layers',
    # 核心功能
    'calculate_daily_net_demand',
    'run_mrp_layered_simulation_daily',
    'run_integrated_mode',
]

# 向后兼容别名
_normalize_location = normalize_location
_normalize_material = normalize_material
_normalize_identifiers = normalize_identifiers
