"""
数据库表映射配置
定义Excel配置表与数据库表、输出表之间的对应关系
"""

from typing import Dict, List

# ==================== 配置表映射 ====================
# Excel 工作表名称 -> 数据库表名（不含前缀）

CONFIG_TABLE_MAPPING = {
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
# 模块输出文件 -> 数据库表名

OUTPUT_TABLE_MAPPING = {
    # Module1 输出
    "module1": {
        "orders_df": "module1_output_orderlog",
        "shipment_df": "module1_output_shipmentlog",
        "cut_df": "module1_output_cutlog",
        "supply_demand_df": "module1_output_supplydemandlog",
        "summary_df": "module1_output_summary",
    },
    
    # Module3 输出
    "module3": {
        "net_demand_df": "module3_output_netdemand",
    },
    
    # Module4 输出
    "module4": {
        "production_df": "module4_output_productionplan",
        "exceed_log": "module4_output_capacityexceed",
        "issues_df": "module4_output_validation",
        "changeover_log": "module4_output_changeoverlog",
    },
    
    # Module5 输出
    "module5": {
        "deployment_plan": "module5_output_deploymentplan",
        "unfulfilled_log": "module5_output_unfulfilledlog",
        "stock_on_hand_log": "module5_output_stockonhandlog",
        "validation_log": "module5_output_validation",
    },
    
    # Module6 输出
    "module6": {
        "delivery_plan": "module6_output_deliveryplan",
        "vehicle_log": "module6_output_vehiclelog",
        "truck_usage": "module6_output_truckusagelog",
        "unsatisfied_log": "module6_output_unsatisfiedmdqlog",
        "validation_log": "module6_output_validationlog",
        "bypass_log": "module6_output_bypassrulehitlog",
    },
    
    # 编排器输出
    "orchestrator": {
        "unrestricted_inventory_*.csv": "orchestrator_unrestricted_inventory",
        "open_deployment_*.csv": "orchestrator_open_deployment",
        "planning_intransit_*.csv": "orchestrator_planning_intransit",
        "space_quota_*.csv": "orchestrator_space_quota",
        "delivery_gr_*.csv": "orchestrator_delivery_gr",
        "production_gr_*.csv": "orchestrator_production_gr",
        "shipment_log_*.csv": "orchestrator_shipment_log",
        "delivery_shipment_log_*.csv": "orchestrator_delivery_shipment_log",
        "inventory_change_log_*.csv": "orchestrator_inventory_change_log",
        "daily_logs_*.csv": "orchestrator_daily_logs",
    },
    
    # 汇总输出
    "summary": {
        "historical_inventory_record.csv": "summary_historical_inventory_record",
        "full_order_shipment_cut_report.xlsx": "summary_full_order_shipment_cut_report",
        "full_production_plan_report.xlsx": "summary_full_production_plan_report",
        "full_changeover_report.xlsx": "summary_full_changeover_report",
        "full_deployment_plan_report.xlsx": "summary_full_deployment_plan_report",
        "full_delivery_plan_report.xlsx": "summary_full_delivery_plan_report",
        "full_truck_usage_report.xlsx": "summary_full_truck_usage_report",
        "full_exceed_capacity_report.xlsx": "summary_full_exceed_capacity_report",
    },
}


# ==================== 必需的配置表 ====================
# 运行仿真必须存在的配置表

REQUIRED_CONFIG_TABLES = [
    "global_network",
    "global_leadtime",
    "m1_demandforecast",
    "m1_initialinventory",
    "m3_safetystock",
]


# ==================== 可选的配置表 ====================
# 可选的配置表（不存在时使用默认值）

OPTIONAL_CONFIG_TABLES = [
    "global_seed",
    "global_spacecapacity",
    "m1_forecasterror",
    "m1_ordercalendar",
    "m1_aoconfig",
    "m1_dpsconfig",
    "m4_productionreliability",
    "m6_mdqbypassrules",
]


def get_config_table_name(sheet_name: str, prefix: str = None) -> str:
    """
    获取配置表的完整数据库表名
    
    注意：采用"同结构同表"规则，所有配置使用统一表名，通过 config_name 字段区分不同配置。
    不再为每个配置创建独立的表（如 bc_s5_xxx, bc_s9_xxx）。
    
    参数：
        sheet_name: Excel Sheet名称
        prefix: 配置前缀（已废弃，保留参数兼容性但不再使用）
    
    返回：
        str: 统一的数据库表名（带 cfg_ 前缀）
    """
    # 查找映射
    if sheet_name in CONFIG_TABLE_MAPPING:
        base_name = CONFIG_TABLE_MAPPING[sheet_name]
    else:
        # 默认转换
        base_name = sheet_name.lower().replace(" ", "_").replace("-", "_")
    
    # 统一配置表使用 cfg_ 前缀，通过 config_name 字段区分不同配置
    return f"cfg_{base_name}"


def get_output_table_name(module: str, file_pattern: str) -> str:
    """
    获取输出表的数据库表名
    
    参数：
        module: 模块名称
        file_pattern: 文件模式
    
    返回：
        str: 数据库表名，如果没找到则返回None
    """
    if module in OUTPUT_TABLE_MAPPING:
        for pattern, table_name in OUTPUT_TABLE_MAPPING[module].items():
            # 简单匹配（忽略日期部分）
            pattern_base = pattern.replace("*", "").replace(".xlsx", "").replace(".csv", "")
            file_base = file_pattern.replace(".xlsx", "").replace(".csv", "")
            if pattern_base.lower() in file_base.lower():
                return table_name
    return None


def get_all_output_tables() -> List[str]:
    """
    获取所有输出表名列表
    
    返回：
        List[str]: 所有输出表名
    """
    tables = []
    for module_tables in OUTPUT_TABLE_MAPPING.values():
        tables.extend(module_tables.values())
    return tables


def print_table_mapping():
    """打印表映射关系"""
    print("\n" + "=" * 70)
    print("📋 配置表映射关系")
    print("=" * 70)
    print(f"{'Excel Sheet':<40} {'数据库表名':<30}")
    print("-" * 70)
    for sheet, table in CONFIG_TABLE_MAPPING.items():
        print(f"{sheet:<40} {table:<30}")
    
    print("\n" + "=" * 70)
    print("📋 输出表映射关系")
    print("=" * 70)
    for module, tables in OUTPUT_TABLE_MAPPING.items():
        print(f"\n📁 {module}:")
        for pattern, table in tables.items():
            print(f"   {pattern:<45} → {table}")
