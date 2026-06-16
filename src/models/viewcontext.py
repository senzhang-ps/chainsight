"""viewcontext 模型组 —— viewcontext + summary 输出表的注册表。

与 module.py 同理：输出表无固定列，只注册表名；migrate() 自动跳过无列的表，
输出表仍由 write_df 运行时建——完全保留现有行为。

注册表用法（Django 风格）：

    from src.models.viewcontext import VIEWCONTEXT_REGISTRY, SUMMARY_REGISTRY
"""
from __future__ import annotations


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
]
