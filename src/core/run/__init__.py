"""
供应链计划系统的生产集成运行器

ChainSight运行器包，提供命令行接口用于执行供应链仿真。
支持本地文件模式和数据库模式两种运行方式。
"""
from __future__ import annotations

from .run_main import main

# 从子模块导出公共函数供外部使用
from .db_config import _load_config_from_database
from .output_dir import get_or_init_simulation_start

__all__ = [
    'main',
    '_load_config_from_database',
    'get_or_init_simulation_start',
]
