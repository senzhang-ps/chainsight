# -*- coding: utf-8 -*-
"""
defaults.py - 跨模块共享默认参数的唯一真源

从 config/defaults.yaml 加载参数，以 Python 常量形式暴露给各模块。
各模块通过 ``from src.utils.defaults import DEFAULT_MOQ`` 等方式使用，
修改默认值只需编辑 YAML 文件，无需改动 Python 代码。
"""

from pathlib import Path
from typing import Any, Dict, List

import yaml

# ---------------------------------------------------------------------------
# 加载 YAML
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "defaults.yaml"


def _load_yaml() -> Dict[str, Any]:
    """读取 defaults.yaml 并返回原始字典。"""
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


_cfg = _load_yaml()
_shared: Dict[str, Any] = _cfg.get("shared", {})
_prod: Dict[str, Any] = _cfg.get("production_planning", {})
_deploy: Dict[str, Any] = _cfg.get("deployment_planning", {})

# ---------------------------------------------------------------------------
# 跨模块共享默认值
# ---------------------------------------------------------------------------

#: 最小订单量 —— mrp_planning, deployment_planning 共用
DEFAULT_MOQ: int = int(_shared["default_moq"])

#: 四舍五入值 —— mrp_planning, deployment_planning 共用
DEFAULT_RV: int = int(_shared["default_rv"])

#: 规划时间栅栏 —— mrp_planning, deployment_planning 共用
DEFAULT_PTF: int = int(_shared["default_ptf"])

#: 批次规模键 —— mrp_planning, deployment_planning 共用
DEFAULT_LSK: int = int(_shared["default_lsk"])

#: 默认提前期天数 —— deployment_planning 使用
DEFAULT_LEAD_TIME: int = int(_shared["default_lead_time"])

#: MRP 默认时间窗口 —— mrp_planning 使用
DEFAULT_HORIZON: int = int(_shared["default_horizon"])

# ---------------------------------------------------------------------------
# 模块专属默认值
# ---------------------------------------------------------------------------

#: 默认换产时间（小时） —— production_planning 使用
DEFAULT_CHANGEOVER_TIME: float = float(_prod["default_changeover_time"])

#: 默认推送层级 —— deployment_planning 使用
DEFAULT_PUSH_LEVELS: List[float] = [float(x) for x in _deploy["default_push_levels"]]
