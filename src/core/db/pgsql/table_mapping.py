"""表名映射配置（唯一来源）。

定义 Excel 配置表与数据库表之间的对应关系。
所有模块的映射统一在此维护，不再在 db_config.py / config_loader.py 中重复定义。
"""
from __future__ import annotations

from typing import Dict, List

# ==================== 配置表映射 ====================
# Excel 工作表名称 -> 数据库表名（不含 cfg_ 前缀）

CONFIG_TABLE_MAPPING: Dict[str, str] = {
    # 全局配置
    "Config Guide": "config_guide",
    "Global_Network": "global_network",
    "Global_Network_old": "global_network_old",
    "Global_LeadTime": "global_leadtime",
    "Global_DemandPriority": "global_demandpriority",
    "Global_seed": "global_seed",
    "Global_SpaceCapacity": "global_spacecapacity",

    # Module1 配置
    "M1_DemandForecast": "m1_demandforecast",
    "M1_ForecastError": "m1_forecasterror",
    "M1_InitialInventory": "m1_initialinventory",
    "M1_InitialInventory_30D": "m1_initialinventory_30d",
    "M1_OrderCalendar": "m1_ordercalendar",
    "M1_AOConfig": "m1_aoconfig",
    "M1_DPSConfig": "m1_dpsconfig",
    "M1_SupplyChoiceConfig": "m1_supplychoiceconfig",

    # Module3 配置
    "M3_SafetyStock": "m3_safetystock",

    # Module4 配置
    "M4_MaterialLocationLineCfg": "m4_materiallocationlinecfg",
    "M4_LineCapacity": "m4_linecapacity",
    "M4_ChangeoverMatrix": "m4_changeovermatrix",
    "M4_ChangeoverDefinition": "m4_changeoverdefinition",
    "M4_ProductionReliability": "m4_productionreliability",

    # Module5 配置
    "M5_DeployConfig": "m5_deployconfig",
    "M5_PushPullModel": "m5_pushpullmodel",
    "deploy_config_with_moq_rv": "deploy_config_with_moq_rv",

    # Module6 配置
    "M6_MaterialMD": "m6_materialmd",
    "M6_TruckTypeSpecs": "m6_trucktypespecs",
    "M6_TruckReleaseCon": "m6_truckreleasecon",
    "M6_DeliveryDelayDistribution": "m6_deliverydelaydistribution",
    "M6_MDQBypassRules": "m6_mdqbypassrules",
    "M6_TruckCapacityPlan": "m6_truckcapacityplan",

    # 验证配置
    "COValidation": "covalidation",
    "MaterialValidation": "material_validation",
    "LaneValidation": "lane_validation",
    "LocationValidation": "location_validation",
    "MatLocValidation": "matloc_validation",

    # SIT 设计
    "SIT Design": "sit_design",

    # 汇总表
    "material_location_summary": "material_location_summary",
    "safety_stock_summary": "safety_stock_summary",
    "Sheet1": "sheet1",
}


# ==================== 输出表映射 ====================
# 模块输出 -> 数据库表名

OUTPUT_TABLE_MAPPING: Dict[str, Dict[str, str]] = {
    "module1": {
        "orders_df": "module1_output_orderlog",
        "shipment_df": "module1_output_shipmentlog",
        "cut_df": "module1_output_cutlog",
        "supply_demand_df": "module1_output_supplydemandlog",
        "summary_df": "module1_output_summary",
    },
    "module3": {
        "net_demand_df": "module3_output_netdemand",
    },
    "module4": {
        "production_df": "module4_output_productionplan",
        "exceed_log": "module4_output_capacityexceed",
        "issues_df": "module4_output_validation",
        "changeover_log": "module4_output_changeoverlog",
    },
    "module5": {
        "deployment_plan": "module5_output_deploymentplan",
        "unfulfilled_log": "module5_output_unfulfilledlog",
        "stock_on_hand_log": "module5_output_stockonhandlog",
        "validation_log": "module5_output_validation",
    },
    "module6": {
        "delivery_plan": "module6_output_deliveryplan",
        "vehicle_log": "module6_output_vehiclelog",
        "truck_usage": "module6_output_truckusagelog",
        "unsatisfied_log": "module6_output_unsatisfiedmdqlog",
        "validation_log": "module6_output_validationlog",
        "bypass_log": "module6_output_bypassrulehitlog",
    },
    "viewcontext": {
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
    },
    "summary": {
        "historical_inventory_record": "summary_historical_inventory_record",
        "full_order_shipment_cut_report": "summary_full_order_shipment_cut_report",
        "full_production_plan_report": "summary_full_production_plan_report",
        "full_changeover_report": "summary_full_changeover_report",
        "full_deployment_plan_report": "summary_full_deployment_plan_report",
        "full_delivery_plan_report": "summary_full_delivery_plan_report",
        "full_truck_usage_report": "summary_full_truck_usage_report",
        "full_exceed_capacity_report": "summary_full_exceed_capacity_report",
    },
}


# ==================== 必需的配置表 ====================

REQUIRED_CONFIG_TABLES: List[str] = [
    "global_network",
    "global_leadtime",
    "m1_demandforecast",
    "m1_initialinventory",
    "m3_safetystock",
]

OPTIONAL_CONFIG_TABLES: List[str] = [
    "global_seed",
    "global_spacecapacity",
    "m1_forecasterror",
    "m1_ordercalendar",
    "m1_aoconfig",
    "m1_dpsconfig",
    "m4_productionreliability",
    "m6_mdqbypassrules",
]


# ==================== 工具函数 ====================

def get_config_table_name(sheet_name: str) -> str:
    """获取配置表的完整数据库表名（带 cfg_ 前缀）。

    采用"同结构同表"规则，通过 config_name 字段区分不同配置。
    """
    if sheet_name in CONFIG_TABLE_MAPPING:
        base_name = CONFIG_TABLE_MAPPING[sheet_name]
    else:
        base_name = sheet_name.lower().replace(" ", "_").replace("-", "_")
    return f"cfg_{base_name}"


def get_output_table_name(module: str, output_key: str) -> str | None:
    """获取模块输出表的数据库表名。"""
    module_tables = OUTPUT_TABLE_MAPPING.get(module, {})
    return module_tables.get(output_key)


def get_all_output_tables() -> List[str]:
    """获取所有输出表名列表。"""
    tables = []
    for module_tables in OUTPUT_TABLE_MAPPING.values():
        tables.extend(module_tables.values())
    return tables
