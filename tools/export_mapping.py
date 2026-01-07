
import pandas as pd

# Data from pgsql_db/table_mapping.py

CONFIG_TABLE_MAPPING = {
    # Global配置
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
    
    # SIT设计
    "SIT Design": "sit_design",
    
    # 汇总表
    "material_location_summary": "material_location_summary",
    "safety_stock_summary": "safety_stock_summary",
    "Sheet1": "sheet1",
}

OUTPUT_TABLE_MAPPING = {
    # Module1 输出
    "module1": {
        "orders_df": "module1_output_orderlog",
        "shipment_df": "module1_output_shipmentlog",
        "cut_df": "module1_output_cutlog",
        "supply_demand_df": "module1_output_supplydemandlog",
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
        "changeover_log": "module4_output_changeover",
    },
    
    # Module5 输出
    "module5": {
        "deployment_plan": "module5_output_deploymentplan",
        "stock_on_hand_log": "module5_output_stockonhandlog",
        "unfulfilled_log": "module5_output_unfulfilledlog",
        "validation_log": "module5_output_validation",
    },
    
    # Module6 输出
    "module6": {
        "delivery_plan": "module6_output_deliveryplan",
        "truck_usage": "module6_output_truckusagelog",
        "vehicle_log": "module6_output_vehiclelog",
        "validation_log": "module6_output_validationlog",
        "unsatisfied_log": "module6_output_unsatisfiedlog",
        "bypass_log": "module6_output_bypasslog",
    },
    
    # Orchestrator 输出
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
    
    # Summary 输出
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

# Generate rows
rows = []

# Config
for sheet, table in CONFIG_TABLE_MAPPING.items():
    rows.append({
        'Type': 'Input (Config)',
        'Module': 'Global/Input',
        'Excel Sheet Name / Output Key': sheet,
        'Database Table Name': table,
        'Description': 'Excel configuration sheet'
    })

# Output
for module, mapping in OUTPUT_TABLE_MAPPING.items():
    for key, table in mapping.items():
        rows.append({
            'Type': 'Output',
            'Module': module.upper(),
            'Excel Sheet Name / Output Key': key,
            'Database Table Name': table,
            'Description': f'Simulation output for {module}'
        })

df = pd.DataFrame(rows)

# Save to Excel
df.to_excel('database_table_mapping.xlsx', index=False)
print("database_table_mapping.xlsx created successfully.")
