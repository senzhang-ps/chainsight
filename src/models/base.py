"""模型基类层 —— SA declarative 基类 + 类型映射 + 分组 mixin。

贴近 Django 的开发体验：
- 所有模型共享一个 ``Base``（SA declarative 基类），migrate 只遍历 Base.metadata
- ``CfgBase`` / ``OrchBase`` / ``ModuleBase`` 是 **marker mixin**（不是 SA mapped class），
  用于按组分类表，便于分组查询和逻辑分组
- ``map_db_type`` 把 schema 中的 db_type 映射到 SA 列类型（供参考）

运行时 DB 路径仍是 psycopg（Phase 2 才换 SA Engine）；本模块仅承担「声明 + 建表」。
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    Integer,
    Numeric,
    Text,
)
from sqlalchemy.orm import DeclarativeBase


# ── SA declarative 基类（所有模型共享一个 Base.metadata） ──────────────────


class Base(DeclarativeBase):
    """SA declarative 基类 — 所有模型组的共享根。

    migrate() 遍历 Base.metadata.sorted_tables 建 CREATE TABLE IF NOT EXISTS。
    """


# ── 分组 marker mixin（不继承 DeclarativeBase，不是 SA mapped class） ──────


class CfgBase:
    """Marker mixin — 标识配置表（cfg_*）。不是 SA mapped class。"""


class OrchBase:
    """Marker mixin — 标识运行事件表（orch_*）。不是 SA mapped class。"""


class ModuleBase:
    """Marker mixin — 标识输出表（module*_output_*）。
    不是 SA mapped class。输出表无固定列，通常不声明为 SA 类。
    """


class ViewContextBase:
    """Marker mixin — 标识 viewcontext 输出表（viewcontext_*）。不是 SA mapped class。"""


class ModuleOutputBase:
    """Marker mixin — 标识模块输出表（module*_output_*），有固定列的 SA 模型。不是 SA mapped class。"""


class ResumeBase:
    """Marker mixin — 标识断点续跑快照表（resume_*）。不是 SA mapped class。"""


# ── db_type → SA 类型映射 ──────────────────────────────────────────────────

_DB_TYPE_MAP: dict[str, Any] = {
    "str": Text,
    "text": Text,
    "bigint": BigInteger,
    "integer": Integer,
    "int": Integer,
    "double precision": Float,
    "float": Float,
    "real": Float,
    "numeric": Numeric,
    "timestamp without time zone": DateTime,
    "timestamp": DateTime,
    "boolean": Text,
    "bool": Text,
}


def map_db_type(db_type: str | None) -> Any:
    """把 schema 中的 ``db_type`` 字符串映射到 SQLAlchemy 列类型。

    全部未知时回退 ``Text``，保持与运行时全 TEXT 写入一致。
    """
    if db_type:
        key = db_type.strip().lower()
        if key in _DB_TYPE_MAP:
            return _DB_TYPE_MAP[key]
    return Text


__all__ = [
    "Base",
    "CfgBase",
    "OrchBase",
    "ModuleBase",
    "ViewContextBase",
    "ModuleOutputBase",
    "ResumeBase",
    "map_db_type",
]
