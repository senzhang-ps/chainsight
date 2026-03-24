"""
main_integration 包

供应链集成仿真编排模块。

重新导出所有公共 API，保持向后兼容性。
"""

import sys
from pathlib import Path

# 将父级目录加入路径以支持导入
sys.path.insert(0, str(Path(__file__).parent.parent))

# 核心仿真函数
from .simulation_file import run_integrated_simulation
from .simulation_db import run_integrated_simulation_from_dict

# 配置加载
from .config_loader import load_configuration, load_configuration_from_dict, load_csv_overrides

# 断点续跑函数
from .resume import check_resume_capability

# 规范化函数（供其他模块使用）
from .normalize import (
    _normalize_identifiers,
    _normalize_material,
    _normalize_location,
    _normalize_sending,
    _normalize_receiving
)

# CLI 入口
from .cli import main

# 向后兼容性重新导出 SummaryReportGenerator
from ...services.summary_report_generator import SummaryReportGenerator

__all__ = [
    "run_integrated_simulation",
    "run_integrated_simulation_from_dict",
    "load_configuration",
    "load_configuration_from_dict",
    "load_csv_overrides",
    "check_resume_capability",
    "_normalize_identifiers",
    "_normalize_material",
    "_normalize_location",
    "_normalize_sending",
    "_normalize_receiving",
    "SummaryReportGenerator",
    "main",
]
