"""ChainSight DB 层 — SQLAlchemy Engine + Session 管理。

Phase 2a 最小化范围：只提供 Engine 创建和 Session 工厂，
保留 psycopg3 原生连接用于 COPY 批量写入。

用法：
    from src.db.engine import create_db_engine
    from src.db.session import Session, get_session

    engine = create_db_engine(db_config)
    with get_session(engine) as session:
        # SA ORM 操作
        ...
"""
from __future__ import annotations

from .engine import create_db_engine
from .session import Session, get_session

__all__ = [
    "create_db_engine",
    "Session",
    "get_session",
]