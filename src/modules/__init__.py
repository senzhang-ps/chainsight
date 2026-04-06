"""
业务模块 - 供应链计划各独立模块
"""

from . import demand_planning
from . import mrp_planning
from . import production_planning
from . import deployment_planning
from . import logistics_execution

# 向后兼容别名
module1 = demand_planning
module3 = mrp_planning
module4 = production_planning
module5 = deployment_planning
module6 = logistics_execution

__all__ = [
    'demand_planning',
    'mrp_planning',
    'production_planning',
    'deployment_planning',
    'logistics_execution',
    # 向后兼容
    'module1',
    'module3',
    'module4',
    'module5',
    'module6',
]
