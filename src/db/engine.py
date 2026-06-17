"""SQLAlchemy Engine 创建。

从 defaults.yaml 读数据库配置，创建 SA Engine。
连接池大小可配，默认 pool_size=5, max_overflow=10。
"""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import URL, create_engine
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

# 默认连接池配置
DEFAULT_POOL_SIZE = 5
DEFAULT_MAX_OVERFLOW = 10


def create_db_engine(
    config: dict[str, Any] | None = None,
    *,
    host: str | None = None,
    port: int | None = None,
    database: str | None = None,
    user: str | None = None,
    password: str | None = None,
    pool_size: int = DEFAULT_POOL_SIZE,
    max_overflow: int = DEFAULT_MAX_OVERFLOW,
    echo: bool = False,
) -> Engine:
    """创建 SQLAlchemy Engine。

    两种调用方式：
    1. 从 defaults.yaml 全量配置 dict（推荐）：
         create_db_engine(config["database"])
    2. 显式传参：
         create_db_engine(host="localhost", port=5432, database="test_db", ...)

    Args:
        config: 完整配置 dict（优先级低于显式参数）。
        host: 数据库主机地址。
        port: 数据库端口。
        database: 数据库名。
        user: 用户名。
        password: 密码。
        pool_size: 连接池大小，默认 5。
        max_overflow: 连接池溢出上限，默认 10。
        echo: 是否打印 SQL 日志，默认 False。

    Returns:
        SQLAlchemy Engine 实例。
    """
    # 从 config dict 中提取未显式指定的参数
    if config:
        if host is None:
            host = config.get("host", "localhost")
        if port is None:
            port = int(config.get("port", 5432))
        if database is None:
            database = config.get("database", "test_db")
        if user is None:
            user = config.get("user", "postgres")
        if password is None:
            password = config.get("password", "")

    # 确保所有参数都有默认值
    host = host or "localhost"
    port = port or 5432
    database = database or "test_db"
    user = user or "postgres"
    password = password or ""

    url = URL.create(
        "postgresql+psycopg",
        username=user,
        password=password,
        host=host,
        port=port,
        database=database,
    )

    engine = create_engine(
        url,
        pool_size=pool_size,
        max_overflow=max_overflow,
        echo=echo,
        # psycopg3 连接参数
        connect_args={
            "autocommit": True,
            "connect_timeout": 30,
            "client_encoding": "UTF8",
        },
    )

    logger.info(
        "DB Engine 已创建: postgresql://%s@%s:%s/%s (pool=%s+%s)",
        user, host, port, database, pool_size, max_overflow,
    )
    return engine


__all__ = [
    "create_db_engine",
    "DEFAULT_POOL_SIZE",
    "DEFAULT_MAX_OVERFLOW",
]