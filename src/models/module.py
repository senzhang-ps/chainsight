"""module 模型组 —— 模块输出表的注册表。

输出表无固定列（运行时按 DataFrame 动态建表全 TEXT），故只注册表名；
migrate() 自动跳过无列的表，输出表仍由 write_df 运行时建——完全保留现有行为。

已提供 DDL 的 5 张 module1 输出表有 SA declarative 声明（ModuleOutputBase），
migrate() 会统一建表；其余表仍走 write_df 动态建。

注册表用法（Django 风格）：

    from src.models.module import OUTPUT_REGISTRY, get_output_table_name
    table = get_output_table_name("module1", "orders_df")   # "module1_output_orderlog"
"""
from __future__ import annotations

from sqlalchemy import Column, DateTime, Float, Integer, Text

from .base import Base, ModuleOutputBase


# ════════════════════════════════════════════════════════════════════
# SA declarative 模型（有固定列的表 — module1）
# ════════════════════════════════════════════════════════════════════

class Module1OutputOrderlog(Base, ModuleOutputBase):
    """module1 订单日志。"""
    __tablename__ = "module1_output_orderlog"

    run_id = Column(Text)
    sim_date = Column(Text)
    week = Column(Integer)
    material = Column(Text)
    location = Column(Text)
    week_start = Column(Text)
    month = Column(Integer)
    dps_percent = Column(Text)
    quantity_percentage = Column(Float)
    quantity_total = Column(Float)
    demand_type = Column(Text)
    ao_percent = Column(Float)
    split_quantity = Column(Float)
    error_std_percent = Column(Float)
    abs_std = Column(Float)
    cov_quantity_raw = Column(Float)
    rescue_rate = Column(Float)
    cov_quantity = Column(Float)
    simulation_date = Column(Text)
    date = Column(Text)
    order_day_flag = Column(Integer)
    flag_count = Column(Integer)
    quantity = Column(Float)
    remainder = Column(Integer)
    advance_days = Column(Integer)
    percent = Column(Float)
    config_name = Column(Text)
    db_write_time = Column(DateTime)

    __mapper_args__ = {
        "primary_key": [run_id, sim_date, material, location, date, demand_type]
    }


class Module1OutputShipmentlog(Base, ModuleOutputBase):
    """module1 发货日志。"""
    __tablename__ = "module1_output_shipmentlog"

    run_id = Column(Text)
    sim_date = Column(Text)
    config_name = Column(Text)
    db_write_time = Column(DateTime)
    date = Column(Text)
    material = Column(Text)
    location = Column(Text)
    quantity = Column(Float)
    demand_type = Column(Text)
    order_id = Column(Text)

    __mapper_args__ = {
        "primary_key": [run_id, sim_date, material, location, date, order_id]
    }


class Module1OutputCutlog(Base, ModuleOutputBase):
    """module1 截单日志。"""
    __tablename__ = "module1_output_cutlog"

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


class Module1OutputSupplydemandlog(Base, ModuleOutputBase):
    """module1 供需日志。"""
    __tablename__ = "module1_output_supplydemandlog"

    run_id = Column(Text)
    sim_date = Column(Text)
    config_name = Column(Text)
    db_write_time = Column(DateTime)
    date = Column(Text)
    material = Column(Text)
    location = Column(Text)
    quantity = Column(Float)
    demand_element = Column(Text)

    __mapper_args__ = {
        "primary_key": [run_id, sim_date, material, location, date, demand_element]
    }


class Module1OutputSummary(Base, ModuleOutputBase):
    """module1 汇总。"""
    __tablename__ = "module1_output_summary"

    run_id = Column(Text)
    sim_date = Column(Text)
    config_name = Column(Text)
    db_write_time = Column(DateTime)
    total_orders = Column(Float)
    total_shipments = Column(Float)
    total_cuts = Column(Float)
    total_supplydemand = Column(Float)
    date = Column(Text)

    __mapper_args__ = {
        "primary_key": [run_id, sim_date, date]
    }


# ════════════════════════════════════════════════════════════════════
# 注册表（保留向后兼容）
# ════════════════════════════════════════════════════════════════════

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
    "Module1OutputOrderlog",
    "Module1OutputShipmentlog",
    "Module1OutputCutlog",
    "Module1OutputSupplydemandlog",
    "Module1OutputSummary",
]
