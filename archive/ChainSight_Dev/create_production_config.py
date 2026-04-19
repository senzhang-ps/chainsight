#!/usr/bin/env python3
"""
根据E2E测试配置生成生产环境Excel配置文件
基于用户的配置表结构和E2E测试的成功配置
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def create_production_config():
    """创建生产环境配置Excel文件"""
    print("🏭 创建生产环境配置文件...")
    
    config_file = "production_config.xlsx"
    
    # 基于E2E测试的成功配置，但使用用户修改后的配置表名称
    config_data = {
        # ========== 全局配置 ==========
        # 注意：E2E测试没有Global_Seed，所以移除以保持一致
        
        # 初始库存 (减少库存，创造供需不平衡)
        'M1_InitialInventory': pd.DataFrame([
            {'material': 'MAT_A', 'location': 'PLANT_001', 'quantity': 200},  # 从1000减少到200
            {'material': 'MAT_A', 'location': 'DC_001', 'quantity': 50},      # 从500减少到50
            {'material': 'MAT_A', 'location': 'DC_002', 'quantity': 30},      # 从300减少到30
            {'material': 'MAT_B', 'location': 'PLANT_001', 'quantity': 150},  # 从800减少到150
            {'material': 'MAT_B', 'location': 'DC_001', 'quantity': 20},      # 从200减少到20
            {'material': 'MAT_B', 'location': 'DC_002', 'quantity': 15}       # 从100减少到15
        ]),
        
        # 空间容量
        'Global_SpaceCapacity': pd.DataFrame([
            {'location': 'PLANT_001', 'eff_from': '2024-01-01', 'eff_to': '2024-12-31', 'capacity': 2000},
            {'location': 'DC_001', 'eff_from': '2024-01-01', 'eff_to': '2024-12-31', 'capacity': 1500},
            {'location': 'DC_002', 'eff_from': '2024-01-01', 'eff_to': '2024-12-31', 'capacity': 1000}
        ]),
        
        # 网络配置 (3层网络: Plant → DC → Customer)
        'Global_Network': pd.DataFrame([
            {'material': 'MAT_A', 'location': 'DC_001', 'sourcing': 'PLANT_001', 'location_type': 'DC', 'eff_from': '2024-01-01', 'eff_to': '2024-12-31'},
            {'material': 'MAT_A', 'location': 'DC_002', 'sourcing': 'PLANT_001', 'location_type': 'DC', 'eff_from': '2024-01-01', 'eff_to': '2024-12-31'},
            {'material': 'MAT_B', 'location': 'DC_001', 'sourcing': 'PLANT_001', 'location_type': 'DC', 'eff_from': '2024-01-01', 'eff_to': '2024-12-31'},
            {'material': 'MAT_A', 'location': 'PLANT_001', 'sourcing': None, 'location_type': 'plant', 'eff_from': '2024-01-01', 'eff_to': '2024-12-31'},
            {'material': 'MAT_B', 'location': 'PLANT_001', 'sourcing': None, 'location_type': 'plant', 'eff_from': '2024-01-01', 'eff_to': '2024-12-31'}
        ]),
        
        # 运输时间配置
        'Global_LeadTime': pd.DataFrame([
            {'sending': 'PLANT_001', 'receiving': 'DC_001', 'PDT': 1, 'GR': 1, 'MCT': 1},
            {'sending': 'PLANT_001', 'receiving': 'DC_002', 'PDT': 2, 'GR': 1, 'MCT': 1},
            {'sending': 'DC_001', 'receiving': 'DC_002', 'PDT': 1, 'GR': 1, 'MCT': 1}
        ]),
        
        # 需求优先级
        'Global_DemandPriority': pd.DataFrame([
            {'demand_element': 'normal', 'priority': 1},
            {'demand_element': 'AO', 'priority': 2},
            {'demand_element': 'customer', 'priority': 1},
            {'demand_element': 'safety', 'priority': 2},
            {'demand_element': 'replenishment', 'priority': 3},
            {'demand_element': 'net demand for customer', 'priority': 1},
            {'demand_element': 'net demand for safety', 'priority': 2}
        ]),
        
        # ========== Module1 配置 (按照设计要求，使用正确的表名和周度数据格式) ==========
        
        # 需求预测 (增加需求，创造更多活动)
        'M1_DemandForecast': pd.DataFrame([
            {'week': 1, 'material': 'MAT_A', 'location': 'DC_001', 'quantity': 300},  # 从110增加到300
            {'week': 1, 'material': 'MAT_A', 'location': 'DC_002', 'quantity': 250},  # 从90增加到250
            {'week': 1, 'material': 'MAT_B', 'location': 'DC_001', 'quantity': 200},  # 从65增加到200
            {'week': 2, 'material': 'MAT_A', 'location': 'DC_001', 'quantity': 350},  # 从125增加到350
            {'week': 2, 'material': 'MAT_A', 'location': 'DC_002', 'quantity': 300},  # 从95增加到300
            {'week': 2, 'material': 'MAT_B', 'location': 'DC_001', 'quantity': 250},  # 从75增加到250
            {'week': 3, 'material': 'MAT_A', 'location': 'DC_001', 'quantity': 400},  # 从115增加到400
            {'week': 3, 'material': 'MAT_A', 'location': 'DC_002', 'quantity': 350},  # 从90增加到350
            {'week': 3, 'material': 'MAT_B', 'location': 'DC_001', 'quantity': 300}   # 从70增加到300
        ]),
        
        # 预测误差配置 (包含normal和AO类型)
        'M1_ForecastError': pd.DataFrame([
            {'material': 'MAT_A', 'location': 'DC_001', 'order_type': 'normal', 'error_std_percent': 0.05},
            {'material': 'MAT_A', 'location': 'DC_002', 'order_type': 'normal', 'error_std_percent': 0.05},
            {'material': 'MAT_B', 'location': 'DC_001', 'order_type': 'normal', 'error_std_percent': 0.05},
            {'material': 'MAT_A', 'location': 'DC_001', 'order_type': 'AO', 'error_std_percent': 0.03},
            {'material': 'MAT_A', 'location': 'DC_002', 'order_type': 'AO', 'error_std_percent': 0.03},
            {'material': 'MAT_B', 'location': 'DC_001', 'order_type': 'AO', 'error_std_percent': 0.03}
        ]),
        
        # 订单日历
        'M1_OrderCalendar': pd.DataFrame([
            {'date': '2024-01-01', 'order_day_flag': 1},
            {'date': '2024-01-02', 'order_day_flag': 1},
            {'date': '2024-01-03', 'order_day_flag': 1},
            {'date': '2024-01-04', 'order_day_flag': 1},
            {'date': '2024-01-05', 'order_day_flag': 1}
        ]),
        
        # AO配置 (高级订单)
        'M1_AOConfig': pd.DataFrame([
            {'material': 'MAT_A', 'location': 'DC_001', 'advance_days': 3, 'ao_percent': 0.15},
            {'material': 'MAT_A', 'location': 'DC_002', 'advance_days': 3, 'ao_percent': 0.12},
            {'material': 'MAT_B', 'location': 'DC_001', 'advance_days': 2, 'ao_percent': 0.10}
        ]),
        
        # DPS配置 (需求分割配置) - 增加更多跨地点活动
        'M1_DPSConfig': pd.DataFrame([
            {'material': 'MAT_A', 'location': 'DC_001', 'dps_location': 'DC_002', 'dps_percent': 0.30},  # 从20%增加到30%
            {'material': 'MAT_A', 'location': 'DC_001', 'dps_location': 'PLANT_001', 'dps_percent': 0.20},  # 新增：部分需求转回工厂
            {'material': 'MAT_B', 'location': 'DC_001', 'dps_location': 'DC_002', 'dps_percent': 0.25}   # 新增：MAT_B也做DPS分割
        ]),
        
        # 供应选择配置 (供应调整) - 增加更多调整
        'M1_SupplyChoiceConfig': pd.DataFrame([
            {'material': 'MAT_A', 'location': 'DC_001', 'week': 1, 'adjust_quantity': 50},   # 从20增加到50
            {'material': 'MAT_A', 'location': 'DC_002', 'week': 1, 'adjust_quantity': 30},   # 新增：DC_002也做调整
            {'material': 'MAT_B', 'location': 'DC_001', 'week': 2, 'adjust_quantity': -20},  # 从-10增加到-20
            {'material': 'MAT_B', 'location': 'DC_002', 'week': 2, 'adjust_quantity': 25}    # 新增：DC_002也做调整
        ]),
        
        # ========== Module3 配置 ==========
        
        # Module3 Enhanced Configuration (新增安全库存需求)
        'M3_SafetyStock': pd.DataFrame([
            {'material': 'MAT_A', 'location': 'PLANT_001', 'date': '2024-01-01', 'safety_stock_qty': 100},  # 从50增加到100
            {'material': 'MAT_A', 'location': 'DC_001', 'date': '2024-01-01', 'safety_stock_qty': 80},   # 从30增加到80
            {'material': 'MAT_A', 'location': 'DC_002', 'date': '2024-01-01', 'safety_stock_qty': 60},   # 从25增加到60
            {'material': 'MAT_B', 'location': 'PLANT_001', 'date': '2024-01-01', 'safety_stock_qty': 80},  # 从40增加到80
            {'material': 'MAT_B', 'location': 'DC_001', 'date': '2024-01-01', 'safety_stock_qty': 50},   # 从20增加到50
            {'material': 'MAT_B', 'location': 'DC_002', 'date': '2024-01-01', 'safety_stock_qty': 40}    # 新增：DC_002的安全库存
        ]),
        
        # ========== Module4 配置 ==========
        
        # Module4 Enhanced Configuration (新增生产约束)
        'M4_MaterialLocationLineCfg': pd.DataFrame([
            {'material': 'MAT_A', 'location': 'PLANT_001', 'delegate_line': 'LINE_A', 'prd_rate': 80, 'min_batch': 20, 'rv': 10, 'lsk': 1, 'ptf': 0, 'day': 1, 'MCT': 1},
            {'material': 'MAT_B', 'location': 'PLANT_001', 'delegate_line': 'LINE_B', 'prd_rate': 60, 'min_batch': 15, 'rv': 8, 'lsk': 1, 'ptf': 0, 'day': 1, 'MCT': 1}
        ]),
        
        'M4_LineCapacity': pd.DataFrame([
            {'location': 'PLANT_001', 'line': 'LINE_A', 'date': '2024-01-01', 'capacity': 60.0},
            {'location': 'PLANT_001', 'line': 'LINE_A', 'date': '2024-01-02', 'capacity': 70.0},
            {'location': 'PLANT_001', 'line': 'LINE_A', 'date': '2024-01-03', 'capacity': 65.0},
            {'location': 'PLANT_001', 'line': 'LINE_A', 'date': '2024-01-04', 'capacity': 75.0},
            {'location': 'PLANT_001', 'line': 'LINE_A', 'date': '2024-01-05', 'capacity': 80.0},
            {'location': 'PLANT_001', 'line': 'LINE_B', 'date': '2024-01-01', 'capacity': 40.0},
            {'location': 'PLANT_001', 'line': 'LINE_B', 'date': '2024-01-02', 'capacity': 45.0},
            {'location': 'PLANT_001', 'line': 'LINE_B', 'date': '2024-01-03', 'capacity': 50.0},
            {'location': 'PLANT_001', 'line': 'LINE_B', 'date': '2024-01-04', 'capacity': 55.0},
            {'location': 'PLANT_001', 'line': 'LINE_B', 'date': '2024-01-05', 'capacity': 60.0}
        ]),
        
        # 换产矩阵
        'M4_ChangeoverMatrix': pd.DataFrame([
            {'from_material': 'MAT_A', 'to_material': 'MAT_B', 'changeover_id': 'CO_AB'},
            {'from_material': 'MAT_B', 'to_material': 'MAT_A', 'changeover_id': 'CO_BA'}
        ]),
        
        # 换产定义
        'M4_ChangeoverDefinition': pd.DataFrame([
            {'changeover_id': 'CO_AB', 'line': 'LINE_001', 'time': 1.0, 'cost': 100, 'mu_loss': 10},
            {'changeover_id': 'CO_BA', 'line': 'LINE_001', 'time': 1.5, 'cost': 150, 'mu_loss': 15},
            {'changeover_id': 'CO_AB', 'line': 'LINE_002', 'time': 0.8, 'cost': 80, 'mu_loss': 8},
            {'changeover_id': 'CO_BA', 'line': 'LINE_002', 'time': 1.2, 'cost': 120, 'mu_loss': 12}
        ]),
        
        # 生产可靠性
        'M4_ProductionReliability': pd.DataFrame([
            {'location': 'PLANT_001', 'line': 'LINE_001', 'pr': 0.95},
            {'location': 'PLANT_001', 'line': 'LINE_002', 'pr': 0.92}
        ]),
        
        # ========== Module5 配置 ==========
        
        # Module5 Configuration
        'M5_PushPullModel': pd.DataFrame([
            {'material': 'MAT_A', 'sending': 'PLANT_001', 'model': 'push'},
            {'material': 'MAT_B', 'sending': 'PLANT_001', 'model': 'push'}
        ]),
        
        'M5_DeployConfig': pd.DataFrame([
            {'material': 'MAT_A', 'sending': 'PLANT_001', 'receiving': 'DC_001', 'moq': 50, 'rv': 25, 'lsk': 7, 'day': 1},
            {'material': 'MAT_A', 'sending': 'PLANT_001', 'receiving': 'DC_002', 'moq': 50, 'rv': 25, 'lsk': 7, 'day': 1},
            {'material': 'MAT_B', 'sending': 'PLANT_001', 'receiving': 'DC_001', 'moq': 40, 'rv': 20, 'lsk': 7, 'day': 1},
            {'material': 'MAT_A', 'sending': 'DC_001', 'receiving': 'DC_002', 'moq': 30, 'rv': 15, 'lsk': 3, 'day': 1},  # 新增：DC间调拨
            {'material': 'MAT_B', 'sending': 'DC_001', 'receiving': 'DC_002', 'moq': 25, 'rv': 12, 'lsk': 3, 'day': 1}   # 新增：DC间调拨
        ]),
        
        # ========== Module6 配置 ==========
        
        # Module6 Configuration
        'M6_TruckReleaseCon': pd.DataFrame([
            {'sending': 'PLANT_001', 'receiving': 'DC_001', 'truck_type': 'LARGE', 'optimal_type': 'Y', 'WFR': 0.8, 'VFR': 0.8, 'MDQ': 200},
            {'sending': 'PLANT_001', 'receiving': 'DC_002', 'truck_type': 'MEDIUM', 'optimal_type': 'Y', 'WFR': 0.7, 'VFR': 0.7, 'MDQ': 150},
            {'sending': 'DC_001', 'receiving': 'DC_002', 'truck_type': 'SMALL', 'optimal_type': 'Y', 'WFR': 0.9, 'VFR': 0.9, 'MDQ': 100},
            {'sending': 'DC_002', 'receiving': 'DC_001', 'truck_type': 'SMALL', 'optimal_type': 'Y', 'WFR': 0.9, 'VFR': 0.9, 'MDQ': 100},
            {'sending': 'DC_001', 'receiving': 'DC_001', 'truck_type': 'SMALL', 'optimal_type': 'Y', 'WFR': 0.9, 'VFR': 0.9, 'MDQ': 50},
            {'sending': 'DC_002', 'receiving': 'DC_002', 'truck_type': 'SMALL', 'optimal_type': 'Y', 'WFR': 0.9, 'VFR': 0.9, 'MDQ': 50},
            {'sending': 'PLANT_001', 'receiving': 'PLANT_001', 'truck_type': 'LARGE', 'optimal_type': 'Y', 'WFR': 0.9, 'VFR': 0.9, 'MDQ': 200}
        ]),
        
        'M6_TruckCapacityPlan': pd.DataFrame([
            {'date': '2024-01-01', 'sending': 'PLANT_001', 'receiving': 'DC_001', 'truck_type': 'LARGE', 'truck_number': 2},
            {'date': '2024-01-01', 'sending': 'PLANT_001', 'receiving': 'DC_002', 'truck_type': 'MEDIUM', 'truck_number': 2},
            {'date': '2024-01-01', 'sending': 'DC_001', 'receiving': 'DC_002', 'truck_type': 'SMALL', 'truck_number': 3},
            {'date': '2024-01-01', 'sending': 'DC_002', 'receiving': 'DC_001', 'truck_type': 'SMALL', 'truck_number': 3},
            {'date': '2024-01-01', 'sending': 'DC_001', 'receiving': 'DC_001', 'truck_type': 'SMALL', 'truck_number': 5},
            {'date': '2024-01-01', 'sending': 'DC_002', 'receiving': 'DC_002', 'truck_type': 'SMALL', 'truck_number': 5},
            {'date': '2024-01-01', 'sending': 'PLANT_001', 'receiving': 'PLANT_001', 'truck_type': 'LARGE', 'truck_number': 2},
            {'date': '2024-01-02', 'sending': 'PLANT_001', 'receiving': 'DC_001', 'truck_type': 'LARGE', 'truck_number': 2},
            {'date': '2024-01-02', 'sending': 'PLANT_001', 'receiving': 'DC_002', 'truck_type': 'MEDIUM', 'truck_number': 2},
            {'date': '2024-01-02', 'sending': 'DC_001', 'receiving': 'DC_002', 'truck_type': 'SMALL', 'truck_number': 3},
            {'date': '2024-01-02', 'sending': 'DC_002', 'receiving': 'DC_001', 'truck_type': 'SMALL', 'truck_number': 3},
            {'date': '2024-01-02', 'sending': 'DC_001', 'receiving': 'DC_001', 'truck_type': 'SMALL', 'truck_number': 5},
            {'date': '2024-01-02', 'sending': 'DC_002', 'receiving': 'DC_002', 'truck_type': 'SMALL', 'truck_number': 5},
            {'date': '2024-01-02', 'sending': 'PLANT_001', 'receiving': 'PLANT_001', 'truck_type': 'LARGE', 'truck_number': 2},
            {'date': '2024-01-03', 'sending': 'PLANT_001', 'receiving': 'DC_001', 'truck_type': 'LARGE', 'truck_number': 2},
            {'date': '2024-01-03', 'sending': 'PLANT_001', 'receiving': 'DC_002', 'truck_type': 'MEDIUM', 'truck_number': 2},
            {'date': '2024-01-03', 'sending': 'DC_001', 'receiving': 'DC_002', 'truck_type': 'SMALL', 'truck_number': 3},
            {'date': '2024-01-03', 'sending': 'DC_002', 'receiving': 'DC_001', 'truck_type': 'SMALL', 'truck_number': 3},
            {'date': '2024-01-03', 'sending': 'DC_001', 'receiving': 'DC_001', 'truck_type': 'SMALL', 'truck_number': 5},
            {'date': '2024-01-03', 'sending': 'DC_002', 'receiving': 'DC_002', 'truck_type': 'SMALL', 'truck_number': 5},
            {'date': '2024-01-03', 'sending': 'PLANT_001', 'receiving': 'PLANT_001', 'truck_type': 'LARGE', 'truck_number': 2},
            {'date': '2024-01-04', 'sending': 'PLANT_001', 'receiving': 'DC_001', 'truck_type': 'LARGE', 'truck_number': 2},
            {'date': '2024-01-04', 'sending': 'PLANT_001', 'receiving': 'DC_002', 'truck_type': 'MEDIUM', 'truck_number': 2},
            {'date': '2024-01-04', 'sending': 'DC_001', 'receiving': 'DC_002', 'truck_type': 'SMALL', 'truck_number': 3},
            {'date': '2024-01-04', 'sending': 'DC_002', 'receiving': 'DC_001', 'truck_type': 'SMALL', 'truck_number': 3},
            {'date': '2024-01-04', 'sending': 'DC_001', 'receiving': 'DC_001', 'truck_type': 'SMALL', 'truck_number': 5},
            {'date': '2024-01-04', 'sending': 'DC_002', 'receiving': 'DC_002', 'truck_type': 'SMALL', 'truck_number': 5},
            {'date': '2024-01-04', 'sending': 'PLANT_001', 'receiving': 'PLANT_001', 'truck_type': 'LARGE', 'truck_number': 2},
            {'date': '2024-01-05', 'sending': 'PLANT_001', 'receiving': 'DC_001', 'truck_type': 'LARGE', 'truck_number': 2},
            {'date': '2024-01-05', 'sending': 'PLANT_001', 'receiving': 'DC_002', 'truck_type': 'MEDIUM', 'truck_number': 2},
            {'date': '2024-01-05', 'sending': 'DC_001', 'receiving': 'DC_002', 'truck_type': 'SMALL', 'truck_number': 3},
            {'date': '2024-01-05', 'sending': 'DC_002', 'receiving': 'DC_001', 'truck_type': 'SMALL', 'truck_number': 3},
            {'date': '2024-01-05', 'sending': 'DC_001', 'receiving': 'DC_001', 'truck_type': 'SMALL', 'truck_number': 5},
            {'date': '2024-01-05', 'sending': 'DC_002', 'receiving': 'DC_002', 'truck_type': 'SMALL', 'truck_number': 5},
            {'date': '2024-01-05', 'sending': 'PLANT_001', 'receiving': 'PLANT_001', 'truck_type': 'LARGE', 'truck_number': 2}
        ]),
        
        # 卡车类型规格
        'M6_TruckTypeSpecs': pd.DataFrame([
            {'truck_type': 'TYPE_A', 'capacity_qty_in_weight': 1000, 'capacity_qty_in_volume': 2000},
            {'truck_type': 'TYPE_B', 'capacity_qty_in_weight': 800, 'capacity_qty_in_volume': 1600},
            {'truck_type': 'TYPE_LOCAL', 'capacity_qty_in_weight': 500, 'capacity_qty_in_volume': 1000}
        ]),
        
        # 物料主数据
        'M6_MaterialMD': pd.DataFrame([
            {'material': 'MAT_A', 'weight': 1.0, 'volume': 2.0, 'priority': 1},
            {'material': 'MAT_B', 'weight': 0.8, 'volume': 1.5, 'priority': 2}
        ]),
        
        # 配送延迟分布
        'M6_DeliveryDelayDistribution': pd.DataFrame([
            {'material': 'MAT_A', 'location': 'DC_001', 'delay_type': 'normal', 'mean_delay': 0.5, 'std_delay': 0.2},
            {'material': 'MAT_A', 'location': 'DC_002', 'delay_type': 'normal', 'mean_delay': 1.0, 'std_delay': 0.3},
            {'material': 'MAT_B', 'location': 'DC_001', 'delay_type': 'normal', 'mean_delay': 0.3, 'std_delay': 0.1},
            {'material': 'MAT_B', 'location': 'DC_002', 'delay_type': 'normal', 'mean_delay': 0.8, 'std_delay': 0.2},
            {'material': 'MAT_A', 'location': 'PLANT_001', 'delay_type': 'normal', 'mean_delay': 0.0, 'std_delay': 0.0},
            {'material': 'MAT_B', 'location': 'PLANT_001', 'delay_type': 'normal', 'mean_delay': 0.0, 'std_delay': 0.0},
            {'material': 'MAT_A', 'location': 'DC_001', 'delay_type': 'AO', 'mean_delay': 0.2, 'std_delay': 0.1}
        ]),
        
        # MDQ绕过规则
        'M6_MDQBypassRules': pd.DataFrame([
            {'material': 'MAT_A', 'location': 'DC_001', 'bypass_threshold': 100, 'bypass_action': 'skip'}
        ])
    }
    
    # 保存到Excel
    with pd.ExcelWriter(config_file, engine='openpyxl') as writer:
        for sheet_name, df in config_data.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  ✅ 创建配置表: {sheet_name} ({len(df)} 行)")
    
    print(f"\n🎉 生产配置文件已创建: {config_file}")
    print(f"📋 包含 {len(config_data)} 个配置表")
    
    # 验证配置数据
    print("✅ 配置验证:")
    print(f"  📊 M1_DemandForecast: {len(config_data['M1_DemandForecast'])} 行 (周度数据)")
    print(f"  📊 M1_ForecastError: {len(config_data['M1_ForecastError'])} 行 (包含AO类型)")
    print(f"  📊 M1_OrderCalendar: {len(config_data['M1_OrderCalendar'])} 行")
    print(f"  📊 M1_AOConfig: {len(config_data['M1_AOConfig'])} 行")
    print(f"  📊 M1_DPSConfig: {len(config_data['M1_DPSConfig'])} 行")
    print(f"  📊 M1_SupplyChoiceConfig: {len(config_data['M1_SupplyChoiceConfig'])} 行")
    print(f"  📊 M3_SafetyStock: {len(config_data['M3_SafetyStock'])} 行")
    print(f"  📊 M4_MaterialLocationLineCfg: {len(config_data['M4_MaterialLocationLineCfg'])} 行")
    print(f"  📊 M5_DeployConfig: {len(config_data['M5_DeployConfig'])} 行 (包含DC间调拨)")
    print(f"  📊 M6_TruckReleaseCon: {len(config_data['M6_TruckReleaseCon'])} 行")
    print(f"  📊 M6_TruckCapacityPlan: {len(config_data['M6_TruckCapacityPlan'])} 行")
    
    return config_file

if __name__ == "__main__":
    config_file = create_production_config()
    print(f"\n✅ 完成！可以使用以下命令测试:")
    print(f"python main_integration.py -c {config_file} -s 2024-01-01 -e 2024-01-05")
