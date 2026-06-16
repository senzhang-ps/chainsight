"""orch 运行事件表与 DQ 明细表的建表 DDL —— 迁移版。

原手写 DDL 已迁入 ``src/models/orch.py`` 的 SA declarative 声明；
本文件只留薄封装，委托 ``src.models.migrate(db)`` 统一建表。

函数名/签名/建表行为（含 drop 旧 cfg_dq_check_result）不变，
向后兼容所有调用方。
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .db import DB

logger = logging.getLogger(__name__)


# ── 表名（保留原常量，向后兼容） ────────────────────────────────────────────

RUN_EVENT_TABLE = "orch_run_event"
DQ_DETAIL_TABLE = "orch_dq_detail"


def ensure_run_event_tables(db: "DB") -> None:
    """确保 orch_run_event / orch_dq_detail 存在（幂等）。

    委托 ``src.models.migrate(db)`` —— Django ``python manage.py migrate`` 风格，
    从 orch 模型 metadata 自动编译 CREATE TABLE IF NOT EXISTS。

    同时清理已废弃的旧版 ``cfg_dq_check_result`` 表（DROP IF EXISTS，幂等）。
    """
    from src.models import migrate as _migrate
    _migrate(db)


def drop_legacy_dq_table(db: "DB") -> None:
    """删除旧版 cfg_dq_check_result 表（显式调用一次即可）。"""
    db.execute("DROP TABLE IF EXISTS cfg_dq_check_result;")
