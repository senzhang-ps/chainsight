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
    from . import resume as _resume  # noqa: F401
    from . import viewcontext as _viewcontext  # noqa: F401
    from . import module as _module  # noqa: F401

    dialect = postgresql.dialect()

    for table in Base.metadata.sorted_tables:
        # P1 schema 隔离：编译前把每张表限定到 db.schema，使 SA 发出
        # ``CREATE TABLE IF NOT EXISTS "schema"."table"``（不依赖 search_path）。
        table.schema = db.schema
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

    # ── 表结构演进（幂等 ALTER；CREATE TABLE IF NOT EXISTS 不会改已存在的表）──
    _evolve_orch_run_event(db)
    _evolve_module5_deploymentplan(db)
    _ensure_resume_indexes(db)

    # 保留清理：drop 旧 cfg_dq_check_result
    try:
        db.execute(f"DROP TABLE IF EXISTS {db.qualified_name('cfg_dq_check_result')}")
        logger.debug("migrate: 已清理旧表 cfg_dq_check_result")
    except Exception as e:
        logger.warning(f"migrate: 清理旧表失败: {e}")


def _evolve_orch_run_event(db: "DB") -> None:
    """orch_run_event 列演进（幂等）。

    新增 status / current_date / total_days；删除 errors / warnings / hard_blocks。
    CREATE TABLE IF NOT EXISTS 不会改已存在的表，故用 ALTER 兜底旧库。
    """
    if not db.table_exists("orch_run_event"):
        return
    tbl = db.qualified_name("orch_run_event")
    for ddl in (
        f"ALTER TABLE {tbl} ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'running'",
        f"ALTER TABLE {tbl} ADD COLUMN IF NOT EXISTS progress_date TEXT",
        f"ALTER TABLE {tbl} ADD COLUMN IF NOT EXISTS total_days INTEGER",
        f"ALTER TABLE {tbl} DROP COLUMN IF EXISTS errors",
        f"ALTER TABLE {tbl} DROP COLUMN IF EXISTS warnings",
        f"ALTER TABLE {tbl} DROP COLUMN IF EXISTS hard_blocks",
    ):
        try:
            db.execute(ddl)
        except Exception as e:
            logger.warning(f"migrate: orch_run_event 演进失败 ({ddl}): {e}")


def _evolve_module5_deploymentplan(db: "DB") -> None:
    """统一 M5 部署计划的库存约束量列名为小写物理列。"""
    table_name = "module5_output_deploymentplan"
    if not db.table_exists(table_name):
        return

    try:
        db.execute(
            "ALTER TABLE {table} RENAME COLUMN \"deployed_qty_invCon\" "
            "TO deployed_qty_invcon".format(table=db.qualified_name(table_name))
        )
        logger.info("migrate: 已统一 module5 部署计划列 deployed_qty_invcon")
    except Exception as e:
        # 正常幂等场景（旧列已改名或从未存在）均会进入这里，不影响后续运行。
        logger.debug(f"migrate: module5 部署计划列演进跳过: {e}")


def _ensure_resume_indexes(db: "DB") -> None:
    """为断点续跑的定位查询创建幂等索引。

    状态和 M1 快照恢复均按 ``(run_id, sim_date)`` 精确过滤。ViewContext
    中部分表由运行时动态创建，故先批量确认表存在再建索引；不存在的早期状态表
    保持可选、不会阻断迁移。
    """
    from .resume import M1_SNAPSHOT_REGISTRY
    from .viewcontext import VIEWCONTEXT_REGISTRY

    resume_tables = (
        set(VIEWCONTEXT_REGISTRY.values())
        | set(M1_SNAPSHOT_REGISTRY.values())
        | {'module3_output_netdemand'}
    )
    existing_tables = db.existing_tables(list(resume_tables | {'orch_run_event'}))

    for table_name in sorted(resume_tables & existing_tables):
        index_name = f"idx_{table_name}_run_sim"
        try:
            db.execute(
                f'CREATE INDEX IF NOT EXISTS "{index_name}" '
                f'ON {db.qualified_name(table_name)} (run_id, sim_date)'
            )
        except Exception as e:
            logger.warning(
                "migrate: 创建 resume 索引 %s 失败: %s", index_name, e
            )

    if 'orch_run_event' not in existing_tables:
        return

    event_table = db.qualified_name('orch_run_event')
    event_indexes = (
        (
            'idx_orch_run_event_unfinished_resume',
            "(config_name, started_at DESC) "
            "WHERE status <> 'finished' AND progress_date IS NOT NULL",
        ),
        (
            'idx_orch_run_event_dq_cache',
            "(config_name, started_at DESC) WHERE dq_status = 'passed'",
        ),
    )
    for index_name, definition in event_indexes:
        try:
            db.execute(
                f'CREATE INDEX IF NOT EXISTS "{index_name}" '
                f'ON {event_table} {definition}'
            )
        except Exception as e:
            logger.warning(
                "migrate: 创建运行事件索引 %s 失败: %s", index_name, e
            )


__all__ = [
    "migrate",
]
