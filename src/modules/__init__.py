"""
业务模块 - 供应链计划各独立模块

子包映射：
  module1 → demand_planning     (需求规划)
  module3 → mrp_planning        (MRP计划)
  module4 → production_planning (生产计划)
  module5 → deployment_planning (部署规划)
  module6 → logistics_execution (物流执行)
"""

from . import demand_planning as module1
from . import mrp_planning as module3
from . import production_planning as module4
from . import deployment_planning as module5
from . import logistics_execution as module6

__all__ = [
    'module1',
    'module3',
    'module4',
    'module5',
    'module6',
]
