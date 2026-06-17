"""orch 模型组 —— 运行事件表与 DQ 明细表的权威声明。

两张表以 ``runid`` 为组织主键，严格对照 run_event_tables.py 现有 DDL
（全 TEXT 列，dq_status DEFAULT 'running'，orch_dq_detail 复合主键 (runid, sheet_name))。

migrate() 会从 Base.metadata 编译 CREATE TABLE IF NOT EXISTS 并建表，
取代原 ensure_run_event_tables 的手写 DDL。

用法（Django 风格）：

    from src.models.orch import OrchRunEvent, OrchDqDetail
    from src.models import migrate
    migrate(db)  # 自动建 orch_run_event / orch_dq_detail
"""
from __future__ import annotations

from sqlalchemy import Column, Integer, PrimaryKeyConstraint, Text

from .base import Base, OrchBase


class OrchRunEvent(Base, OrchBase):
    """每次 orch 运行的元数据注册。

    两套独立状态：
    - ``dq_status``：DQ 检测状态机 running → cached / passed / blocked / skipped
    - ``status``：orch 任务执行状态 running / finished（用于断点续跑判断）

    续跑三件套：``status``（是否完成）+ ``progress_date``（断点日期）+ ``total_days``（首次记录的分母）。
    ``finished_at`` 仅在 ``status='finished'``（orch 真正结束）时写入，DQ 完成不写。
    """
    __tablename__ = "orch_run_event"

    runid = Column(Text, primary_key=True)
    config_name = Column(Text)
    config_hash = Column(Text)
    dq_status = Column(Text, nullable=False, server_default="running")
    status = Column(Text, nullable=False, server_default="running")
    progress_date = Column(Text)
    total_days = Column(Integer)
    started_at = Column(Text)
    finished_at = Column(Text)
    db_write_time = Column(Text)


class OrchDqDetail(Base, OrchBase):
    """一次 run 内某个 sheet 的 DQ 检测明细。"""
    __tablename__ = "orch_dq_detail"

    runid = Column(Text, nullable=False)
    config_name = Column(Text)
    sheet_name = Column(Text, nullable=False)
    status = Column(Text)
    errors = Column(Text)
    warnings = Column(Text)
    hard_blocks = Column(Text)
    issues_json = Column(Text)
    db_write_time = Column(Text)

    __table_args__ = (
        PrimaryKeyConstraint("runid", "sheet_name"),
    )


__all__ = [
    "OrchRunEvent",
    "OrchDqDetail",
]
