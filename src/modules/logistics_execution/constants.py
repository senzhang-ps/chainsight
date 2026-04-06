# -*- coding: utf-8 -*-
"""
物流执行模块常量定义
"""

from typing import List

ALLOWED_EXPRESSION_VARS: List[str] = [
    'waiting_days', 'deployed_qty_ratio', 'exception_MDQ',
    'sending', 'receiving', 'truck_type', 'demand_element'
]
