"""ChainSight 模型包 —— 表结构的单一权威来源。

Django 风格：
- 声明式模型（cfg.py / orch.py）继承自 Base + 分组 mixin（CfgBase / OrchBase）
- ``migrate(db)`` 自动从 Base.metadata 建 CREATE TABLE IF NOT EXISTS
- 直接 import 模型类读 ``__tablename__`` / ``__table__.columns``
- 不引入 Django 框架，只借用其声明式 ORM 思路

用法：
    from src.models import migrate
    migrate(db)   # 建表（等同 Django python manage.py migrate）

    from src.models.cfg import GlobalNetwork, CONFIG_TABLE_REGISTRY
    from src.models.orch import OrchRunEvent
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sqlalchemy.schema import CreateTable
from sqlalchemy.dialects import postgresql

if TYPE_CHECKING:
    from ..core.db.pgsql.db import DB

logger = logging.getLogger(__name__)


def migrate(db: "DB") -> None:
    """Django ``python manage.py migrate`` 风格：从 Base.metadata 自动建表。

    遍历 Base.metadata.sorted_tables，对有列的表编译
    CREATE TABLE IF NOT EXISTS 并经 ``db.execute`` 执行；
    无列的表跳过（输出表由 write_df 运行时动态建）。
    """
    from .base import Base

    # 导入具体模型以触发 metadata 注册
    from . import cfg as _cfg  # noqa: F401
    from . import orch as _orch  # noqa: F401

    dialect = postgresql.dialect()

    for table in Base.metadata.sorted_tables:
        if not table.columns:
            logger.debug(f"migrate: 跳过无列表 {table.name}")
            continue
        # 表已存在则跳过（建表幂等；避免每次启动都刷一遍「已建表」误导日志）
        if db.table_exists(table.name):
            logger.debug(f"migrate: 表已存在，跳过 {table.name}")
            continue
        ddl_str = str(
            CreateTable(table, if_not_exists=True).compile(dialect=dialect)
        )
        try:
            db.execute(ddl_str)
            logger.info(f"migrate: 已建表 {table.name}")
        except Exception as e:
            logger.warning(f"migrate: 建表 {table.name} 失败: {e}")

    # 保留清理：drop 旧 cfg_dq_check_result
    try:
        db.execute("DROP TABLE IF EXISTS cfg_dq_check_result")
        logger.debug("migrate: 已清理旧表 cfg_dq_check_result")
    except Exception as e:
        logger.warning(f"migrate: 清理旧表失败: {e}")


__all__ = [
    "migrate",
]
