"""viewcontext 模型组 —— viewcontext + summary 输出表的注册表。

与 module.py 同理：输出表无固定列，只注册表名；migrate() 自动跳过无列的表，
输出表仍由 write_df 运行时建——完全保留现有行为。

已提供 DDL 的 3 张 viewcontext 表有 SA declarative 声明（ViewContextBase），
migrate() 会统一建表；其余表仍走 write_df 动态建。

注册表用法（Django 风格）：

    from src.models.viewcontext import VIEWCONTEXT_REGISTRY, SUMMARY_REGISTRY
"""
from __future__ import annotations

from sqlalchemy import Column, DateTime, Float, Integer, Table, Text

from .base import Base, SummaryBase, ViewContextBase


# ════════════════════════════════════════════════════════════════════
# SA declarative 模型（有固定列的表）
# ════════════════════════════════════════════════════════════════════

class ViewContextDailyLogs(Base, ViewContextBase):
    """viewcontext 每日日志。"""
    __tablename__ = "viewcontext_daily_logs"

    run_id = Column(Text)
    sim_date = Column(Text)
    config_name = Column(Text)
    db_write_time = Column(DateTime)
    timestamp = Column(DateTime)
    date = Column(Text)
    event_type = Column(Text)
    message = Column(Text)

    __mapper_args__ = {
        "primary_key": [run_id, sim_date, timestamp, event_type]
    }


class ViewContextInventoryChangeLog(Base, ViewContextBase):
    """viewcontext 库存变更日志。"""
    __tablename__ = "viewcontext_inventory_change_log"

    run_id = Column(Text)
    sim_date = Column(Text)
    config_name = Column(Text)
    db_write_time = Column(DateTime)
    date = Column(Text)
    material = Column(Text)
    location = Column(Text)
    beginning_inventory = Column(Float)
    production_gr = Column(Float)
    delivery_gr = Column(Float)
    shipment = Column(Float)
    delivery_ship = Column(Float)
    ending_inventory = Column(Float)
    calculated_ending = Column(Float)
    balance_diff = Column(Float)

    __mapper_args__ = {
        "primary_key": [run_id, sim_date, material, location, date]
    }


class ViewContextUnrestrictedInventory(Base, ViewContextBase):
    """viewcontext 非限制库存。"""
    __tablename__ = "viewcontext_unrestricted_inventory"

    run_id = Column(Text)
    sim_date = Column(Text)
    config_name = Column(Text)
    db_write_time = Column(DateTime)
    date = Column(Text)
    material = Column(Text)
    location = Column(Text)
    quantity = Column(Float)

    __mapper_args__ = {
        "primary_key": [run_id, sim_date, material, location, date]
    }


class ViewContextSpaceQuota(Base, ViewContextBase):
    """viewcontext 空间配额。"""
    __tablename__ = "viewcontext_space_quota"

    run_id = Column(Text)
    sim_date = Column(Text)
    config_name = Column(Text)
    db_write_time = Column(DateTime)
    receiving = Column(Text)
    date = Column(Text)
    max_qty = Column(Float)

    __mapper_args__ = {
        "primary_key": [run_id, sim_date, receiving, date]
    }


class ViewContextM4LineStates(Base, ViewContextBase):
    """viewcontext M4 产线状态（换产连续性）。"""
    __tablename__ = "viewcontext_m4_line_states"

    run_id = Column(Text)
    sim_date = Column(Text)
    config_name = Column(Text)
    db_write_time = Column(DateTime)
    line = Column(Text)
    last_material = Column(Text)
    last_location = Column(Text)
    last_activity = Column(Text)
    remaining_time = Column(Float)
    changeover_id = Column(Text)
    to_material = Column(Text)

    __mapper_args__ = {
        "primary_key": [run_id, sim_date, line]
    }


class ViewContextM4AllocatedCapacity(Base, ViewContextBase):
    """viewcontext M4 已分配产能（防重复分配）。"""
    __tablename__ = "viewcontext_m4_allocated_capacity"

    run_id = Column(Text)
    sim_date = Column(Text)
    config_name = Column(Text)
    db_write_time = Column(DateTime)
    capacity_key = Column(Text)
    allocated_hours = Column(Float)

    __mapper_args__ = {
        "primary_key": [run_id, sim_date, capacity_key]
    }


# ════════════════════════════════════════════════════════════════════
# 注册表（保留向后兼容）
# ════════════════════════════════════════════════════════════════════


# ── viewcontext 输出表 ──────────────────────────────────────────────────

VIEWCONTEXT_REGISTRY: dict[str, str] = {
    "unrestricted_inventory": "viewcontext_unrestricted_inventory",
    "open_deployment": "viewcontext_open_deployment",
    "planning_intransit": "viewcontext_planning_intransit",
    "space_quota": "viewcontext_space_quota",
    "delivery_gr": "viewcontext_delivery_gr",
    "production_gr": "viewcontext_production_gr",
    "production_plan_backlog": "viewcontext_production_plan_backlog",
    "shipment_log": "viewcontext_shipment_log",
    "delivery_shipment_log": "viewcontext_delivery_shipment_log",
    "inventory_change_log": "viewcontext_inventory_change_log",
    "daily_logs": "viewcontext_daily_logs",
    "open_deployment_pastdue_cleanup": "viewcontext_open_deployment_pastdue_cleanup",
    "m4_line_states": "viewcontext_m4_line_states",
    "m4_allocated_capacity": "viewcontext_m4_allocated_capacity",
}

# ── summary 输出表 ──────────────────────────────────────────────────────

SUMMARY_REGISTRY: dict[str, str] = {
    "historical_inventory_record": "summary_historical_inventory_record",
    "full_order_shipment_cut_report": "summary_full_order_shipment_cut_report",
    "full_production_plan_report": "summary_full_production_plan_report",
    "full_changeover_report": "summary_full_changeover_report",
    "full_deployment_plan_report": "summary_full_deployment_plan_report",
    "full_delivery_plan_report": "summary_full_delivery_plan_report",
    "full_truck_usage_report": "summary_full_truck_usage_report",
    "full_exceed_capacity_report": "summary_full_exceed_capacity_report",
}

# Summary 的业务列随模块输出扩展；为避免把它们错误固化在迁移中，仅声明
# 所有 Summary 共用的运行元数据列。PersistenceManager 会在首次写入某个
# 业务列前以幂等 ALTER 补齐该列，从而既保证 migrate 可预建八张表，也保留
# DataFrame 输出的完整列集。不要为这些表配置 run_id/sim_date 主键：一份
# Summary 会包含多行业务记录，主键由其业务列决定且各表不相同。
for _summary_table_name in SUMMARY_REGISTRY.values():
    if _summary_table_name not in Base.metadata.tables:
        Table(
            _summary_table_name,
            Base.metadata,
            Column('run_id', Text),
            Column('sim_date', Text),
            Column('config_name', Text),
            Column('db_write_time', DateTime),
            info={'marker': SummaryBase},
        )


def get_viewcontext_tables() -> list[str]:
    """获取所有 viewcontext 表名列表。"""
    return list(VIEWCONTEXT_REGISTRY.values())


def get_summary_tables() -> list[str]:
    """获取所有 summary 表名列表。"""
    return list(SUMMARY_REGISTRY.values())


__all__ = [
    "VIEWCONTEXT_REGISTRY",
    "SUMMARY_REGISTRY",
    "get_viewcontext_tables",
    "get_summary_tables",
    "ViewContextDailyLogs",
    "ViewContextInventoryChangeLog",
    "ViewContextUnrestrictedInventory",
    "ViewContextSpaceQuota",
    "ViewContextM4LineStates",
    "ViewContextM4AllocatedCapacity",
]
