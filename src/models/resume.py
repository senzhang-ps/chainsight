"""resume 模型组 —— 断点续跑快照表的 SA declarative 声明。

m1 prepare 后的 4 个属性（order_df / daily_detail / daily_detail_sc / order_cal）
每项一张独立表，严格对照线上 DDL（列名 / 类型完全一致）。

migrate() 会从 Base.metadata 编译 CREATE TABLE IF NOT EXISTS 并建表，
取代原 write_df 运行时「全 TEXT 动态建表」——现在表结构是声明式的，
write_df 写入时按库表真实列类型（float8/int/text）做原生类型转换。

注册表用法（Django 风格）：

    from src.models.resume import M1_SNAPSHOT_REGISTRY
    table = M1_SNAPSHOT_REGISTRY['order_df']   # "resume_m1_order_df"
    ResumeM1OrderDF.__table__.columns           # 取 SA 列
"""
from __future__ import annotations

from sqlalchemy import Column, Float, Integer, Text

from .base import Base, ResumeBase


class ResumeM1DailyDetail(Base, ResumeBase):
    """m1 daily_detail 快照 —— 每日明细（含 ao / dps / 误差 / 抢救率）。"""
    __tablename__ = "resume_m1_daily_detail"

    run_id = Column(Text)
    sim_date = Column(Text)
    week = Column(Integer)
    material = Column(Text)
    location = Column(Text)
    week_start = Column(Text)
    month = Column(Integer)
    dps_percent = Column(Float)
    quantity_percentage = Column(Float)
    quantity_total = Column(Float)
    order_type = Column(Text)
    ao_percent = Column(Float)
    split_quantity = Column(Float)
    error_std_percent = Column(Float)
    abs_std = Column(Float)
    cov_quantity_raw = Column(Float)
    rescue_rate = Column(Float)
    simulation_date = Column(Text)
    order_day_flag = Column(Integer)
    date = Column(Text)
    flag_count = Column(Integer)
    quantity = Column(Float)
    remainder = Column(Integer)

    # 线上表无 DB 级 PK；ORM 需显式指认主键（mapper-only，不发 DB 约束，
    # 编译出的 CREATE TABLE 与原 DDL 完全一致——无 PRIMARY KEY）。
    __mapper_args__ = {
        "primary_key": [run_id, sim_date, material, location, date, order_type]
    }


class ResumeM1DailyDetailSc(Base, ResumeBase):
    """m1 daily_detail_sc 快照 —— supply-choice 调整后的每日明细。"""
    __tablename__ = "resume_m1_daily_detail_sc"

    run_id = Column(Text)
    sim_date = Column(Text)
    week = Column(Integer)
    material = Column(Text)
    location = Column(Text)
    week_start = Column(Text)
    month = Column(Integer)
    dps_percent = Column(Float)
    quantity_percentage = Column(Float)
    adjust_quantity = Column(Float)
    simulation_date = Column(Text)
    order_day_flag = Column(Integer)
    date = Column(Text)
    flag_count = Column(Integer)
    quantity = Column(Float)
    remainder = Column(Integer)

    # 线上表无 DB 级 PK；ORM 需显式指认主键（mapper-only，不发 DB 约束，
    # 编译出的 CREATE TABLE 与原 DDL 完全一致——无 PRIMARY KEY）。
    __mapper_args__ = {"primary_key": [run_id, sim_date, material, location, date]}


class ResumeM1OrderDf(Base, ResumeBase):
    """m1 order_df 快照 —— 订单明细（含 demand_type / advance_days / percent）。"""
    __tablename__ = "resume_m1_order_df"

    run_id = Column(Text)
    sim_date = Column(Text)
    week = Column(Integer)
    material = Column(Text)
    location = Column(Text)
    week_start = Column(Text)
    month = Column(Integer)
    dps_percent = Column(Float)
    quantity_percentage = Column(Float)
    quantity_total = Column(Float)
    demand_type = Column(Text)
    ao_percent = Column(Float)
    split_quantity = Column(Float)
    error_std_percent = Column(Float)
    abs_std = Column(Float)
    cov_quantity_raw = Column(Float)
    rescue_rate = Column(Float)
    simulation_date = Column(Text)
    order_day_flag = Column(Integer)
    date = Column(Text)
    flag_count = Column(Integer)
    quantity = Column(Float)
    remainder = Column(Integer)
    advance_days = Column(Integer)
    percent = Column(Float)

    # 线上表无 DB 级 PK；ORM 需显式指认主键（mapper-only，不发 DB 约束，
    # 编译出的 CREATE TABLE 与原 DDL 完全一致——无 PRIMARY KEY）。
    __mapper_args__ = {
        "primary_key": [run_id, sim_date, material, location, date, demand_type]
    }


class ResumeM1OrderCal(Base, ResumeBase):
    """m1 order_cal 快照 —— 订单日历（date / order_day_flag）。"""
    __tablename__ = "resume_m1_order_cal"

    run_id = Column(Text)
    sim_date = Column(Text)
    date = Column(Text)
    order_day_flag = Column(Integer)

    # 线上表无 DB 级 PK；ORM 需显式指认主键（mapper-only，不发 DB 约束，
    # 编译出的 CREATE TABLE 与原 DDL 完全一致——无 PRIMARY KEY）。
    __mapper_args__ = {"primary_key": [run_id, sim_date, date]}


# ── m1 快照表注册 ──────────────────────────────────────────
# attr → 表名（保持原 dict[str, str] 契约；persistence_manager 按 .items() / .values() 消费）。
# 表结构已声明为 SA 类，migrate() 自动建表，write_df 按真实列类型写入。

M1_SNAPSHOT_REGISTRY: dict[str, str] = {
    'order_df':          ResumeM1OrderDf.__tablename__,
    'daily_detail':      ResumeM1DailyDetail.__tablename__,
    'daily_detail_sc':   ResumeM1DailyDetailSc.__tablename__,
    'order_cal':         ResumeM1OrderCal.__tablename__,
}


# attr → SA 模型类（需要直接拿 Column / Table 时用）。
M1_SNAPSHOT_MODEL_REGISTRY: dict[str, type] = {
    'order_df':          ResumeM1OrderDf,
    'daily_detail':      ResumeM1DailyDetail,
    'daily_detail_sc':   ResumeM1DailyDetailSc,
    'order_cal':         ResumeM1OrderCal,
}


def get_all_m1_snapshot_tables() -> list[str]:
    """获取所有 m1 快照表名列表。"""
    return list(M1_SNAPSHOT_REGISTRY.values())


__all__ = [
    "M1_SNAPSHOT_REGISTRY",
    "M1_SNAPSHOT_MODEL_REGISTRY",
    "get_all_m1_snapshot_tables",
    "ResumeM1DailyDetail",
    "ResumeM1DailyDetailSc",
    "ResumeM1OrderDf",
    "ResumeM1OrderCal",
]
