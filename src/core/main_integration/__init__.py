"""
main_integration 包

供应链集成仿真编排模块。

导出当前 main_integration 包的公共 API。
"""

import sys
from pathlib import Path

# 将父级目录加入路径以支持导入
sys.path.insert(0, str(Path(__file__).parent.parent))

# 核心仿真函数
from .simulation_file import run_integrated_simulation
from .simulation_db import run_integrated_simulation_from_dict

# 配置加载
from .config_loader import load_configuration, load_configuration_from_dict

# 断点续跑函数
from .resume import check_resume_capability

# CLI 入口
from .cli import main

__all__ = [
    "run_integrated_simulation",
    "run_integrated_simulation_from_dict",
    "load_configuration",
    "load_configuration_from_dict",
    "check_resume_capability",
    "main",
]
