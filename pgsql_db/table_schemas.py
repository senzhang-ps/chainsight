"""
表结构定义模块
定义Module6等模块的输出表结构，确保空表也有正确的列名

注意：这些列定义需要与src/modules/module6.py中的_build_*函数保持一致
"""

# Module6 输出表的列定义 - 与module6.py中_write_excel_output保持一致
MODULE6_OUTPUT_SCHEMAS = {
    # DeliveryPlan - 空表不带列名（与基线一致）
    'DeliveryPlan': None,  
    # VehicleLog - 空表带列名
    'VehicleLog': [
        'date', 'sending', 'receiving', 'truck_type', 'vehicle_no', 'vehicle_uid',
        'total_units', 'total_weight', 'total_volume', 'WFR', 'VFR', 'trigger'
    ],
    # TruckUsageLog - 空表带列名
    'TruckUsageLog': [
        'date', 'sending', 'receiving', 'truck_type', 'truck_used'
    ],
    # UnsatisfiedMDQLog - 空表不带列名（与基线一致）
    'UnsatisfiedMDQLog': None,
    # ValidationLog - 空表不带列名（与基线一致）  
    'ValidationLog': None,
    # BypassRuleHitLog - 空表不带列名（与基线一致）
    'BypassRuleHitLog': None
}

# 添加其他模块的表结构定义（供参考，实际模块有自己的空表处理逻辑）
MODULE1_OUTPUT_SCHEMAS = {
    'OrderLog': None,  # 按实际数据列
    'ShipmentLog': None,
    'CutLog': None,
    'SupplyDemandLog': None
}

MODULE3_OUTPUT_SCHEMAS = {
    'NetDemand': None
}

MODULE4_OUTPUT_SCHEMAS = {
    'ProductionPlan': None,
    'CapacityExceed': None,
    'ChangeoverLog': None
}

MODULE5_OUTPUT_SCHEMAS = {
    'DeploymentPlan': None,
    'StockOnHandLog': None,
    'UnfulfilledLog': None
}

def get_columns(module_name: str, sheet_name: str) -> list:
    """获取指定模块和sheet的列名列表
    
    Args:
        module_name: 模块名称 (module1, module3, module4, module5, module6)
        sheet_name: Excel sheet名称
        
    Returns:
        list: 列名列表，如果未定义则返回None
    """
    schemas = {
        'module1': MODULE1_OUTPUT_SCHEMAS,
        'module3': MODULE3_OUTPUT_SCHEMAS,
        'module4': MODULE4_OUTPUT_SCHEMAS,
        'module5': MODULE5_OUTPUT_SCHEMAS,
        'module6': MODULE6_OUTPUT_SCHEMAS
    }
    
    module_schemas = schemas.get(module_name.lower())
    if module_schemas:
        return module_schemas.get(sheet_name)
    return None


def get_all_module_tables() -> dict:
    """获取所有模块的表定义
    
    Returns:
        dict: {module_name: {table_name: [columns]}}
    """
    return {
        'module1': MODULE1_OUTPUT_SCHEMAS,
        'module3': MODULE3_OUTPUT_SCHEMAS,
        'module4': MODULE4_OUTPUT_SCHEMAS,
        'module5': MODULE5_OUTPUT_SCHEMAS,
        'module6': MODULE6_OUTPUT_SCHEMAS
    }
