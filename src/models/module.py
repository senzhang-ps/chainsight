"""module 模型组 —— 模块输出表的注册表。

输出表无固定列（运行时按 DataFrame 动态建表全 TEXT），故只注册表名；
migrate() 自动跳过无列的表，输出表仍由 write_df 运行时建——完全保留现有行为。

注册表用法（Django 风格）：

    from src.models.module import OUTPUT_REGISTRY, get_output_table_name
    table = get_output_table_name("module1", "orders_df")   # "module1_output_orderlog"
"""
from __future__ import annotations

from sqlalchemy import Table, Column, Text as SAText, MetaData

from .base import ModuleBase

# ── module1 输出表 ──────────────────────────────────────────────────────

_MODULE1_OUTPUTS: dict[str, str] = {
    "orders_df": "module1_output_orderlog",
    "shipment_df": "module1_output_shipmentlog",
    "cut_df": "module1_output_cutlog",
    "supply_demand_df": "module1_output_supplydemandlog",
    "summary_df": "module1_output_summary",
}

# ── module3 输出表 ──────────────────────────────────────────────────────

_MODULE3_OUTPUTS: dict[str, str] = {
    "net_demand_df": "module3_output_netdemand",
}

# ── module4 输出表 ──────────────────────────────────────────────────────

_MODULE4_OUTPUTS: dict[str, str] = {
    "production_df": "module4_output_productionplan",
    "exceed_log": "module4_output_capacityexceed",
    "issues_df": "module4_output_validation",
    "changeover_log": "module4_output_changeoverlog",
}

# ── module5 输出表 ──────────────────────────────────────────────────────

_MODULE5_OUTPUTS: dict[str, str] = {
    "deployment_plan": "module5_output_deploymentplan",
    "unfulfilled_log": "module5_output_unfulfilledlog",
    "stock_on_hand_log": "module5_output_stockonhandlog",
    "validation_log": "module5_output_validation",
}

# ── module6 输出表 ──────────────────────────────────────────────────────

_MODULE6_OUTPUTS: dict[str, str] = {
    "delivery_plan": "module6_output_deliveryplan",
    "vehicle_log": "module6_output_vehiclelog",
    "truck_usage": "module6_output_truckusagelog",
    "unsatisfied_log": "module6_output_unsatisfiedmdqlog",
    "validation_log": "module6_output_validationlog",
    "bypass_log": "module6_output_bypassrulehitlog",
}

# ── 全量注册表 ──────────────────────────────────────────────────────────

OUTPUT_REGISTRY: dict[str, dict[str, str]] = {
    "module1": _MODULE1_OUTPUTS,
    "module3": _MODULE3_OUTPUTS,
    "module4": _MODULE4_OUTPUTS,
    "module5": _MODULE5_OUTPUTS,
    "module6": _MODULE6_OUTPUTS,
}


def get_output_table_name(module: str, output_key: str) -> str | None:
    """获取模块输出表的数据库表名。"""
    return OUTPUT_REGISTRY.get(module, {}).get(output_key)


def get_all_output_tables() -> list[str]:
    """获取所有输出表名列表。"""
    tables: list[str] = []
    for module_tables in OUTPUT_REGISTRY.values():
        tables.extend(module_tables.values())
    return tables


__all__ = [
    "OUTPUT_REGISTRY",
    "get_output_table_name",
    "get_all_output_tables",
]
