"""SQLAlchemy Session 工厂。

提供 Session 上下文管理器，支持事务回滚。
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session as SASession, sessionmaker

logger = logging.getLogger(__name__)

# 全局 Session 工厂（需先调用 create_db_engine 创建 Engine）
Session: sessionmaker | None = None


def init_session(engine: Engine) -> sessionmaker:
    """初始化全局 Session 工厂。

    Args:
        engine: SA Engine 实例。

    Returns:
        sessionmaker 实例。
    """
    global Session
    Session = sessionmaker(bind=engine)
    logger.info("Session 工厂已初始化")
    return Session


@contextmanager
def get_session(engine: Engine | None = None) -> Iterator[SASession]:
    """获取 SA Session 上下文管理器。

    自动处理 commit/rollback/close。用 contextmanager 包装，支持 ``with`` 语法。

    用法：
        with get_session(engine) as session:
            session.add(OrchRunEvent(...))
            session.commit()

    Args:
        engine: SA Engine 实例。如果为 None，使用全局 Session 工厂。

    Yields:
        SA Session 实例。
    """
    if engine is not None:
        session = SASession(bind=engine)
    elif Session is not None:
        session = Session()
    else:
        raise RuntimeError(
            "Session 工厂未初始化，请先调用 init_session(engine) 或传入 engine 参数"
        )

    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@contextmanager
def get_raw_connection(engine: Engine) -> Iterator:
    """获取 psycopg3 原生连接（用于 COPY 批量写入）。

    从 SA Engine 获取底层 psycopg3 连接，保留 COPY 协议性能。

    用法：
        with get_raw_connection(engine) as conn:
            with conn.cursor() as cur:
                with cur.copy("COPY ... FROM STDIN") as copy:
                    copy.write_rows(rows)

    Args:
        engine: SA Engine 实例。

    Yields:
        psycopg3 原生连接。
    """
    with engine.connect() as sa_conn:
        raw_conn = sa_conn.connection  # 获取底层 psycopg3 连接
        yield raw_conn


__all__ = [
    "Session",
    "init_session",
    "get_session",
    "get_raw_connection",
]