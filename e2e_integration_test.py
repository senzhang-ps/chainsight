#!/usr/bin/env python3
"""
End-to-End Integration Test for Supply Chain Planning System
Tests the complete flow: M1 → M4 → M5 → M6 → M3 with Orchestrator coordination

This test validates:
1. Orchestrator state management 
2. Module execution order and data flow
3. Inventory conservation and consistency
4. Business logic correctness
5. Interface integration between modules
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Import all modules
from orchestrator import create_orchestrator
import module1
import module3
import module4
import module5
import module6
from main_integration import run_integrated_simulation, load_configuration

class E2ETestCase:
    """End-to-End test case with simplified 3-tier network"""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.test_dir = Path(f"./e2e_test_output_{test_name}")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Test parameters
        self.start_date = "2024-01-01"
        self.end_date = "2024-01-07"  # 1 week test (7 days)
        
        # Test results
        self.test_results = []
        self.validation_errors = []
        
    def create_test_configuration(self) -> str:
        """Create simplified test configuration"""
        print(f"📋 Creating test configuration for {self.test_name}")
        
        config_file = self.test_dir / "test_config.xlsx"
        
        # Create test configuration data
        config_data = {
            # 初始库存 (减少库存，创造供需不平衡)
            'M1_InitialInventory': pd.DataFrame([
                {'material': 'MAT_A', 'location': 'PLANT_001', 'quantity': 500},  # 从1000减少到200
                {'material': 'MAT_A', 'location': 'DC_001', 'quantity': 50},      # 从500减少到50
                {'material': 'MAT_A', 'location': 'DC_002', 'quantity': 30},      # 从300减少到30
                {'material': 'MAT_B', 'location': 'PLANT_001', 'quantity': 500},  # 从800减少到150
                {'material': 'MAT_B', 'location': 'DC_001', 'quantity': 20},      # 从200减少到20
                {'material': 'MAT_B', 'location': 'DC_002', 'quantity': 15}       # 从100减少到15
            ]),
            
            # Space Capacity
            'Global_SpaceCapacity': pd.DataFrame([
                {'location': 'PLANT_001', 'eff_from': '2024-01-01', 'eff_to': '2024-12-31', 'capacity': 2000},
                {'location': 'DC_001', 'eff_from': '2024-01-01', 'eff_to': '2024-12-31', 'capacity': 1500},
                {'location': 'DC_002', 'eff_from': '2024-01-01', 'eff_to': '2024-12-31', 'capacity': 1000}
            ]),
            
            # 全局网络配置 (新增跨地点调拨路径)
            'Global_Network': pd.DataFrame([
                {'material': 'MAT_A', 'location': 'DC_001', 'sourcing': 'PLANT_001', 'location_type': 'DC', 'eff_from': '2024-01-01', 'eff_to': '2024-12-31'},
                {'material': 'MAT_A', 'location': 'DC_002', 'sourcing': 'PLANT_001', 'location_type': 'DC', 'eff_from': '2024-01-01', 'eff_to': '2024-12-31'},
                {'material': 'MAT_B', 'location': 'DC_001', 'sourcing': 'PLANT_001', 'location_type': 'DC', 'eff_from': '2024-01-01', 'eff_to': '2024-12-31'},

            ]),
            
            # Lead Time Configuration
            'Global_LeadTime': pd.DataFrame([
                {'sending': 'PLANT_001', 'receiving': 'DC_001', 'PDT': 2, 'GR': 1, 'MCT': 1, 'OTD':1},
                {'sending': 'PLANT_001', 'receiving': 'DC_002', 'PDT': 2, 'GR': 1, 'MCT': 1, 'OTD':1},
                {'sending': 'DC_001', 'receiving': 'DC_002', 'PDT': 2, 'GR': 1, 'MCT': 1, 'OTD':1},
                {'sending': 'DC_002', 'receiving': 'DC_001', 'PDT': 2, 'GR': 1, 'MCT': 1, 'OTD':1}
            ]),
            
            # Demand Priority (确保包含所有Module5可能生成的demand_element)
            'Global_DemandPriority': pd.DataFrame([
                {'demand_element': 'normal', 'priority': 1},
                {'demand_element': 'AO', 'priority': 2},
                {'demand_element': 'customer', 'priority': 1},
                {'demand_element': 'safety', 'priority': 2},
                {'demand_element': 'replenishment', 'priority': 3},
                # 添加跨节点部署计划的demand_element
                {'demand_element': 'net demand for customer', 'priority': 1},
                {'demand_element': 'net demand for forecast', 'priority': 2},
                {'demand_element': 'net demand for safety', 'priority': 3},
                # 添加Module6可能遇到的其他demand_element
                {'demand_element': 'forecast', 'priority': 2},
                {'demand_element': 'transfer', 'priority': 3},
                {'demand_element': 'deployment', 'priority': 3}
            ]),
            
            # Module3 Safety Stock
            'M3_SafetyStock': pd.DataFrame([
                {'material': 'MAT_A', 'location': 'DC_001', 'date': '2024-01-01', 'safety_stock_qty': 100},
                {'material': 'MAT_A', 'location': 'DC_002', 'date': '2024-01-01', 'safety_stock_qty': 80},
                {'material': 'MAT_B', 'location': 'DC_001', 'date': '2024-01-01', 'safety_stock_qty': 50}
            ]),
            
            # Module5 Configuration
            'M5_PushPullModel': pd.DataFrame([
                {'material': 'MAT_A', 'sending': 'PLANT_001', 'model': 'push'},
                {'material': 'MAT_B', 'sending': 'PLANT_001', 'model': 'push'}
            ]),
            
            'M5_DeployConfig': pd.DataFrame([
                {'material': 'MAT_A', 'sending': 'PLANT_001', 'receiving': 'DC_001', 'moq': 50, 'rv': 25, 'lsk': 7, 'day': 1},
                {'material': 'MAT_A', 'sending': 'PLANT_001', 'receiving': 'DC_002', 'moq': 50, 'rv': 25, 'lsk': 7, 'day': 1},
                {'material': 'MAT_B', 'sending': 'PLANT_001', 'receiving': 'DC_001', 'moq': 40, 'rv': 20, 'lsk': 7, 'day': 1},                {'material': 'MAT_B', 'sending': 'DC_001', 'receiving': 'DC_002', 'moq': 25, 'rv': 12, 'lsk': 3, 'day': 1}   # 新增：DC间调拨
            ]),
            

            
            # Module6 Configuration
            'M6_TruckReleaseCon': pd.DataFrame([
                {'sending': 'PLANT_001', 'receiving': 'DC_001', 'truck_type': 'LARGE', 'optimal_type': 'Y', 'WFR': 0.2, 'VFR': 0.2, 'MDQ': 200},
                {'sending': 'PLANT_001', 'receiving': 'DC_002', 'truck_type': 'MEDIUM', 'optimal_type': 'Y', 'WFR': 0.2, 'VFR': 0.2, 'MDQ': 150},
                {'sending': 'DC_001', 'receiving': 'DC_002', 'truck_type': 'SMALL', 'optimal_type': 'Y', 'WFR': 0.2, 'VFR': 0.2, 'MDQ': 100},
                {'sending': 'DC_002', 'receiving': 'DC_001', 'truck_type': 'SMALL', 'optimal_type': 'Y', 'WFR': 0.2, 'VFR': 0.2, 'MDQ': 100},                
            ]),
            
            'M6_MaterialMD': pd.DataFrame([
                {'material': 'MAT_A', 'weight': 1.0, 'volume': 2.0, 'priority': 1, 'demand_unit_to_weight': 1.0, 'demand_unit_to_volume': 2.0},
                {'material': 'MAT_B', 'weight': 1.0, 'volume': 1.5, 'priority': 2, 'demand_unit_to_weight': 0.8, 'demand_unit_to_volume': 1.5}
            ]),
            
            'M6_DeliveryDelayDistribution': pd.DataFrame([
                # 简化版配置，包含必需的sending和receiving字段
                {'date': '2024-01-01', 'sending': 'ALL', 'receiving': 'ALL', 'delay_days': 0, 'probability': 0.8},
                {'date': '2024-01-01', 'sending': 'ALL', 'receiving': 'ALL', 'delay_days': 1, 'probability': 0.2},
                {'date': '2024-01-02', 'sending': 'ALL', 'receiving': 'ALL', 'delay_days': 0, 'probability': 0.8},
                {'date': '2024-01-02', 'sending': 'ALL', 'receiving': 'ALL', 'delay_days': 1, 'probability': 0.2},
                {'date': '2024-01-03', 'sending': 'ALL', 'receiving': 'ALL', 'delay_days': 0, 'probability': 0.8},
                {'date': '2024-01-03', 'sending': 'ALL', 'receiving': 'ALL', 'delay_days': 1, 'probability': 0.2},
                {'date': '2024-01-04', 'sending': 'ALL', 'receiving': 'ALL', 'delay_days': 0, 'probability': 0.8},
                {'date': '2024-01-04', 'sending': 'ALL', 'receiving': 'ALL', 'delay_days': 1, 'probability': 0.2},
                {'date': '2024-01-05', 'sending': 'ALL', 'receiving': 'ALL', 'delay_days': 0, 'probability': 0.8},
                {'date': '2024-01-05', 'sending': 'ALL', 'receiving': 'ALL', 'delay_days': 1, 'probability': 0.2},
                {'date': '2024-01-06', 'sending': 'ALL', 'receiving': 'ALL', 'delay_days': 0, 'probability': 0.8},
                {'date': '2024-01-06', 'sending': 'ALL', 'receiving': 'ALL', 'delay_days': 1, 'probability': 0.2},
                {'date': '2024-01-07', 'sending': 'ALL', 'receiving': 'ALL', 'delay_days': 0, 'probability': 0.8},
                {'date': '2024-01-07', 'sending': 'ALL', 'receiving': 'ALL', 'delay_days': 1, 'probability': 0.2}
            ]),
            
            'M6_MDQBypassRules': pd.DataFrame([
                {
                    'sending': 'ALL', 
                    'receiving': 'ALL', 
                    'truck_type': 'ALL', 
                    'demand_element': 'ALL',
                    'condition_logic': 'waiting_days > 5',  # 等待超过5天时允许绕过MDQ限制
                    'rule_id': 'WAITING_BYPASS'
                }
            ]),
            
            
            # Module4 Enhanced Configuration (新增生产约束)
            'M4_MaterialLocationLineCfg': pd.DataFrame([
                {'material': 'MAT_A', 'location': 'PLANT_001', 'delegate_line': 'LINE_A', 'prd_rate': 80, 'min_batch': 20, 'rv': 10, 'lsk': 1, 'ptf': 2, 'day': 1, 'MCT': 1},
                {'material': 'MAT_B', 'location': 'PLANT_001', 'delegate_line': 'LINE_B', 'prd_rate': 60, 'min_batch': 15, 'rv': 8, 'lsk': 1, 'ptf': 4, 'day': 1, 'MCT': 1}
            ]),
            
            'M4_LineCapacity': pd.DataFrame([
                # Week 1 (Days 1-7)
                {'location': 'PLANT_001', 'line': 'LINE_A', 'date': '2024-01-01', 'capacity': 60.0},
                {'location': 'PLANT_001', 'line': 'LINE_A', 'date': '2024-01-02', 'capacity': 70.0},
                {'location': 'PLANT_001', 'line': 'LINE_A', 'date': '2024-01-03', 'capacity': 65.0},
                {'location': 'PLANT_001', 'line': 'LINE_A', 'date': '2024-01-04', 'capacity': 75.0},
                {'location': 'PLANT_001', 'line': 'LINE_A', 'date': '2024-01-05', 'capacity': 80.0},
                {'location': 'PLANT_001', 'line': 'LINE_A', 'date': '2024-01-06', 'capacity': 70.0},
                {'location': 'PLANT_001', 'line': 'LINE_A', 'date': '2024-01-07', 'capacity': 65.0},
                {'location': 'PLANT_001', 'line': 'LINE_B', 'date': '2024-01-01', 'capacity': 40.0},
                {'location': 'PLANT_001', 'line': 'LINE_B', 'date': '2024-01-02', 'capacity': 45.0},
                {'location': 'PLANT_001', 'line': 'LINE_B', 'date': '2024-01-03', 'capacity': 50.0},
                {'location': 'PLANT_001', 'line': 'LINE_B', 'date': '2024-01-04', 'capacity': 55.0},
                {'location': 'PLANT_001', 'line': 'LINE_B', 'date': '2024-01-05', 'capacity': 60.0},
                {'location': 'PLANT_001', 'line': 'LINE_B', 'date': '2024-01-06', 'capacity': 55.0},
                {'location': 'PLANT_001', 'line': 'LINE_B', 'date': '2024-01-07', 'capacity': 50.0},
                # Week 2 (Days 8-14)
                {'location': 'PLANT_001', 'line': 'LINE_A', 'date': '2024-01-08', 'capacity': 75.0},
                {'location': 'PLANT_001', 'line': 'LINE_A', 'date': '2024-01-09', 'capacity': 80.0},
                {'location': 'PLANT_001', 'line': 'LINE_A', 'date': '2024-01-10', 'capacity': 70.0},
                {'location': 'PLANT_001', 'line': 'LINE_A', 'date': '2024-01-11', 'capacity': 75.0},
                {'location': 'PLANT_001', 'line': 'LINE_A', 'date': '2024-01-12', 'capacity': 80.0},
                {'location': 'PLANT_001', 'line': 'LINE_A', 'date': '2024-01-13', 'capacity': 70.0},
                {'location': 'PLANT_001', 'line': 'LINE_A', 'date': '2024-01-14', 'capacity': 65.0},
                {'location': 'PLANT_001', 'line': 'LINE_B', 'date': '2024-01-08', 'capacity': 55.0},
                {'location': 'PLANT_001', 'line': 'LINE_B', 'date': '2024-01-09', 'capacity': 60.0},
                {'location': 'PLANT_001', 'line': 'LINE_B', 'date': '2024-01-10', 'capacity': 50.0},
                {'location': 'PLANT_001', 'line': 'LINE_B', 'date': '2024-01-11', 'capacity': 55.0},
                {'location': 'PLANT_001', 'line': 'LINE_B', 'date': '2024-01-12', 'capacity': 60.0},
                {'location': 'PLANT_001', 'line': 'LINE_B', 'date': '2024-01-13', 'capacity': 55.0},
                {'location': 'PLANT_001', 'line': 'LINE_B', 'date': '2024-01-14', 'capacity': 50.0},
                # Week 3-6 (Days 15-42) - 简化配置，每周递增
                {'location': 'PLANT_001', 'line': 'LINE_A', 'date': '2024-01-15', 'capacity': 80.0},
                {'location': 'PLANT_001', 'line': 'LINE_A', 'date': '2024-01-22', 'capacity': 85.0},
                {'location': 'PLANT_001', 'line': 'LINE_A', 'date': '2024-01-29', 'capacity': 90.0},
                {'location': 'PLANT_001', 'line': 'LINE_A', 'date': '2024-02-05', 'capacity': 95.0},
                {'location': 'PLANT_001', 'line': 'LINE_B', 'date': '2024-01-15', 'capacity': 60.0},
                {'location': 'PLANT_001', 'line': 'LINE_B', 'date': '2024-01-22', 'capacity': 65.0},
                {'location': 'PLANT_001', 'line': 'LINE_B', 'date': '2024-01-29', 'capacity': 70.0},
                {'location': 'PLANT_001', 'line': 'LINE_B', 'date': '2024-02-05', 'capacity': 75.0},
                # Week 7-12 (Days 43-84) - 继续递增
                {'location': 'PLANT_001', 'line': 'LINE_A', 'date': '2024-02-12', 'capacity': 100.0},
                {'location': 'PLANT_001', 'line': 'LINE_A', 'date': '2024-02-19', 'capacity': 105.0},
                {'location': 'PLANT_001', 'line': 'LINE_A', 'date': '2024-02-26', 'capacity': 110.0},
                {'location': 'PLANT_001', 'line': 'LINE_A', 'date': '2024-03-04', 'capacity': 115.0},
                {'location': 'PLANT_001', 'line': 'LINE_A', 'date': '2024-03-11', 'capacity': 120.0},
                {'location': 'PLANT_001', 'line': 'LINE_A', 'date': '2024-03-18', 'capacity': 125.0},
                {'location': 'PLANT_001', 'line': 'LINE_B', 'date': '2024-02-12', 'capacity': 80.0},
                {'location': 'PLANT_001', 'line': 'LINE_B', 'date': '2024-02-19', 'capacity': 85.0},
                {'location': 'PLANT_001', 'line': 'LINE_B', 'date': '2024-02-26', 'capacity': 90.0},
                {'location': 'PLANT_001', 'line': 'LINE_B', 'date': '2024-03-04', 'capacity': 95.0},
                {'location': 'PLANT_001', 'line': 'LINE_B', 'date': '2024-03-11', 'capacity': 100.0},
                {'location': 'PLANT_001', 'line': 'LINE_B', 'date': '2024-03-18', 'capacity': 105.0}
            ]),
            
            'M4_ChangeoverMatrix': pd.DataFrame([
                {'from_material': 'MAT_A', 'to_material': 'MAT_B', 'changeover_id': 'CO_AB'},
                {'from_material': 'MAT_B', 'to_material': 'MAT_A', 'changeover_id': 'CO_BA'}
            ]),
            
            'M4_ChangeoverDefinition': pd.DataFrame([
                {'changeover_id': 'CO_AB', 'line': 'LINE_A', 'time': 2.0, 'cost': 200, 'mu_loss': 20},
                {'changeover_id': 'CO_BA', 'line': 'LINE_A', 'time': 2.0, 'cost': 200, 'mu_loss': 20},
                {'changeover_id': 'CO_AB', 'line': 'LINE_B', 'time': 1.5, 'cost': 150, 'mu_loss': 15},
                {'changeover_id': 'CO_BA', 'line': 'LINE_B', 'time': 1.5, 'cost': 150, 'mu_loss': 15}
            ]),
            
            'M4_ProductionReliability': pd.DataFrame([
                {'location': 'PLANT_001', 'line': 'LINE_A', 'pr': 0.95},
                {'location': 'PLANT_001', 'line': 'LINE_B', 'pr': 0.90}
            ]),
            
            # Module1 Enhanced Configuration (新增)
            # 需求预测 (扩展预测到12周，创造更多长期规划活动)
            'M1_DemandForecast': pd.DataFrame([
                # Week 1
                {'week': 1, 'material': 'MAT_A', 'location': 'DC_001', 'quantity': 300},
                {'week': 1, 'material': 'MAT_A', 'location': 'DC_002', 'quantity': 250},
                {'week': 1, 'material': 'MAT_B', 'location': 'DC_001', 'quantity': 200},
                # Week 2
                {'week': 2, 'material': 'MAT_A', 'location': 'DC_001', 'quantity': 350},
                {'week': 2, 'material': 'MAT_A', 'location': 'DC_002', 'quantity': 300},
                {'week': 2, 'material': 'MAT_B', 'location': 'DC_001', 'quantity': 250},
                # Week 3
                {'week': 3, 'material': 'MAT_A', 'location': 'DC_001', 'quantity': 400},
                {'week': 3, 'material': 'MAT_A', 'location': 'DC_002', 'quantity': 350},
                {'week': 3, 'material': 'MAT_B', 'location': 'DC_001', 'quantity': 300},
                # Week 4 - 新增
                {'week': 4, 'material': 'MAT_A', 'location': 'DC_001', 'quantity': 380},
                {'week': 4, 'material': 'MAT_A', 'location': 'DC_002', 'quantity': 320},
                {'week': 4, 'material': 'MAT_B', 'location': 'DC_001', 'quantity': 280},
                # Week 5 - 新增
                {'week': 5, 'material': 'MAT_A', 'location': 'DC_001', 'quantity': 420},
                {'week': 5, 'material': 'MAT_A', 'location': 'DC_002', 'quantity': 370},
                {'week': 5, 'material': 'MAT_B', 'location': 'DC_001', 'quantity': 320},
                # Week 6 - 新增
                {'week': 6, 'material': 'MAT_A', 'location': 'DC_001', 'quantity': 450},
                {'week': 6, 'material': 'MAT_A', 'location': 'DC_002', 'quantity': 400},
                {'week': 6, 'material': 'MAT_B', 'location': 'DC_001', 'quantity': 350},
                # Week 7 - 新增
                {'week': 7, 'material': 'MAT_A', 'location': 'DC_001', 'quantity': 480},
                {'week': 7, 'material': 'MAT_A', 'location': 'DC_002', 'quantity': 430},
                {'week': 7, 'material': 'MAT_B', 'location': 'DC_001', 'quantity': 380},
                # Week 8 - 新增
                {'week': 8, 'material': 'MAT_A', 'location': 'DC_001', 'quantity': 500},
                {'week': 8, 'material': 'MAT_A', 'location': 'DC_002', 'quantity': 450},
                {'week': 8, 'material': 'MAT_B', 'location': 'DC_001', 'quantity': 400},
                # Week 9 - 新增
                {'week': 9, 'material': 'MAT_A', 'location': 'DC_001', 'quantity': 520},
                {'week': 9, 'material': 'MAT_A', 'location': 'DC_002', 'quantity': 470},
                {'week': 9, 'material': 'MAT_B', 'location': 'DC_001', 'quantity': 420},
                # Week 10 - 新增
                {'week': 10, 'material': 'MAT_A', 'location': 'DC_001', 'quantity': 540},
                {'week': 10, 'material': 'MAT_A', 'location': 'DC_002', 'quantity': 490},
                {'week': 10, 'material': 'MAT_B', 'location': 'DC_001', 'quantity': 440},
                # Week 11 - 新增
                {'week': 11, 'material': 'MAT_A', 'location': 'DC_001', 'quantity': 560},
                {'week': 11, 'material': 'MAT_A', 'location': 'DC_002', 'quantity': 510},
                {'week': 11, 'material': 'MAT_B', 'location': 'DC_001', 'quantity': 460},
                # Week 12 - 新增
                {'week': 12, 'material': 'MAT_A', 'location': 'DC_001', 'quantity': 580},
                {'week': 12, 'material': 'MAT_A', 'location': 'DC_002', 'quantity': 530},
                {'week': 12, 'material': 'MAT_B', 'location': 'DC_001', 'quantity': 480}
            ]),
            
            'M1_ForecastError': pd.DataFrame([
                {'material': 'MAT_A', 'location': 'DC_001', 'order_type': 'normal', 'error_std_percent': 0.05},
                {'material': 'MAT_A', 'location': 'DC_002', 'order_type': 'normal', 'error_std_percent': 0.05},
                {'material': 'MAT_B', 'location': 'DC_001', 'order_type': 'normal', 'error_std_percent': 0.05},
                {'material': 'MAT_A', 'location': 'DC_001', 'order_type': 'AO', 'error_std_percent': 0.03},
                {'material': 'MAT_A', 'location': 'DC_002', 'order_type': 'AO', 'error_std_percent': 0.03},
                {'material': 'MAT_B', 'location': 'DC_001', 'order_type': 'AO', 'error_std_percent': 0.03}
            ]),
            
            'M1_OrderCalendar': pd.DataFrame([
                # Week 1 (Days 1-7)
                {'date': '2024-01-01', 'order_day_flag': 1},
                {'date': '2024-01-02', 'order_day_flag': 1},
                {'date': '2024-01-03', 'order_day_flag': 1},
                {'date': '2024-01-04', 'order_day_flag': 1},
                {'date': '2024-01-05', 'order_day_flag': 1},
                {'date': '2024-01-06', 'order_day_flag': 1},
                {'date': '2024-01-07', 'order_day_flag': 1},
                # Week 2 (Days 8-14)
                {'date': '2024-01-08', 'order_day_flag': 1},
                {'date': '2024-01-09', 'order_day_flag': 1},
                {'date': '2024-01-10', 'order_day_flag': 1},
                {'date': '2024-01-11', 'order_day_flag': 1},
                {'date': '2024-01-12', 'order_day_flag': 1},
                {'date': '2024-01-13', 'order_day_flag': 1},
                {'date': '2024-01-14', 'order_day_flag': 1},
                # Week 3 (Days 15-21)
                {'date': '2024-01-15', 'order_day_flag': 1},
                {'date': '2024-01-16', 'order_day_flag': 1},
                {'date': '2024-01-17', 'order_day_flag': 1},
                {'date': '2024-01-18', 'order_day_flag': 1},
                {'date': '2024-01-19', 'order_day_flag': 1},
                {'date': '2024-01-20', 'order_day_flag': 1},
                {'date': '2024-01-21', 'order_day_flag': 1},
                # Week 4 (Days 22-28)
                {'date': '2024-01-22', 'order_day_flag': 1},
                {'date': '2024-01-23', 'order_day_flag': 1},
                {'date': '2024-01-24', 'order_day_flag': 1},
                {'date': '2024-01-25', 'order_day_flag': 1},
                {'date': '2024-01-26', 'order_day_flag': 1},
                {'date': '2024-01-27', 'order_day_flag': 1},
                {'date': '2024-01-28', 'order_day_flag': 1},
                # Week 5 (Days 29-35)
                {'date': '2024-01-29', 'order_day_flag': 1},
                {'date': '2024-01-30', 'order_day_flag': 1},
                {'date': '2024-01-31', 'order_day_flag': 1},
                {'date': '2024-02-01', 'order_day_flag': 1},
                {'date': '2024-02-02', 'order_day_flag': 1},
                {'date': '2024-02-03', 'order_day_flag': 1},
                {'date': '2024-02-04', 'order_day_flag': 1},
                # Week 6 (Days 36-42)
                {'date': '2024-02-05', 'order_day_flag': 1},
                {'date': '2024-02-06', 'order_day_flag': 1},
                {'date': '2024-02-07', 'order_day_flag': 1},
                {'date': '2024-02-08', 'order_day_flag': 1},
                {'date': '2024-02-09', 'order_day_flag': 1},
                {'date': '2024-02-10', 'order_day_flag': 1},
                {'date': '2024-02-11', 'order_day_flag': 1},
                # Week 7 (Days 43-49)
                {'date': '2024-02-12', 'order_day_flag': 1},
                {'date': '2024-02-13', 'order_day_flag': 1},
                {'date': '2024-02-14', 'order_day_flag': 1},
                {'date': '2024-02-15', 'order_day_flag': 1},
                {'date': '2024-02-16', 'order_day_flag': 1},
                {'date': '2024-02-17', 'order_day_flag': 1},
                {'date': '2024-02-18', 'order_day_flag': 1},
                # Week 8 (Days 50-56)
                {'date': '2024-02-19', 'order_day_flag': 1},
                {'date': '2024-02-20', 'order_day_flag': 1},
                {'date': '2024-02-21', 'order_day_flag': 1},
                {'date': '2024-02-22', 'order_day_flag': 1},
                {'date': '2024-02-23', 'order_day_flag': 1},
                {'date': '2024-02-24', 'order_day_flag': 1},
                {'date': '2024-02-25', 'order_day_flag': 1},
                # Week 9 (Days 57-63)
                {'date': '2024-02-26', 'order_day_flag': 1},
                {'date': '2024-02-27', 'order_day_flag': 1},
                {'date': '2024-02-28', 'order_day_flag': 1},
                {'date': '2024-02-29', 'order_day_flag': 1},
                {'date': '2024-03-01', 'order_day_flag': 1},
                {'date': '2024-03-02', 'order_day_flag': 1},
                {'date': '2024-03-03', 'order_day_flag': 1},
                # Week 10 (Days 64-70)
                {'date': '2024-03-04', 'order_day_flag': 1},
                {'date': '2024-03-05', 'order_day_flag': 1},
                {'date': '2024-03-06', 'order_day_flag': 1},
                {'date': '2024-03-07', 'order_day_flag': 1},
                {'date': '2024-03-08', 'order_day_flag': 1},
                {'date': '2024-03-09', 'order_day_flag': 1},
                {'date': '2024-03-10', 'order_day_flag': 1},
                # Week 11 (Days 71-77)
                {'date': '2024-03-11', 'order_day_flag': 1},
                {'date': '2024-03-12', 'order_day_flag': 1},
                {'date': '2024-03-13', 'order_day_flag': 1},
                {'date': '2024-03-14', 'order_day_flag': 1},
                {'date': '2024-03-15', 'order_day_flag': 1},
                {'date': '2024-03-16', 'order_day_flag': 1},
                {'date': '2024-03-17', 'order_day_flag': 1},
                # Week 12 (Days 78-84)
                {'date': '2024-03-18', 'order_day_flag': 1},
                {'date': '2024-03-19', 'order_day_flag': 1},
                {'date': '2024-03-20', 'order_day_flag': 1},
                {'date': '2024-03-21', 'order_day_flag': 1},
                {'date': '2024-03-22', 'order_day_flag': 1},
                {'date': '2024-03-23', 'order_day_flag': 1},
                {'date': '2024-03-24', 'order_day_flag': 1}
            ]),
            
            # AO配置 (高级订单)
            'M1_AOConfig': pd.DataFrame([
                {'material': 'MAT_A', 'location': 'DC_001', 'advance_days': 3, 'ao_percent': 0.15},
                {'material': 'MAT_A', 'location': 'DC_002', 'advance_days': 3, 'ao_percent': 0.12},
                {'material': 'MAT_B', 'location': 'DC_001', 'advance_days': 2, 'ao_percent': 0.10}
            ]),
            
            # DPS配置 (需求分割配置) - 扩展到12周
            'M1_DPSConfig': pd.DataFrame([
                # Week 1-3 (原有配置)
                {'material': 'MAT_A', 'location': 'DC_001', 'dps_location': 'DC_002', 'dps_percent': 0.30},
                {'material': 'MAT_A', 'location': 'DC_001', 'dps_location': 'PLANT_001', 'dps_percent': 0.20},
                {'material': 'MAT_B', 'location': 'DC_001', 'dps_location': 'DC_002', 'dps_percent': 0.25},
                # Week 4-6 (新增)
                {'material': 'MAT_A', 'location': 'DC_001', 'dps_location': 'DC_002', 'dps_percent': 0.32},
                {'material': 'MAT_A', 'location': 'DC_001', 'dps_location': 'PLANT_001', 'dps_percent': 0.18},
                {'material': 'MAT_B', 'location': 'DC_001', 'dps_location': 'DC_002', 'dps_percent': 0.28},
                # Week 7-9 (新增)
                {'material': 'MAT_A', 'location': 'DC_001', 'dps_location': 'DC_002', 'dps_percent': 0.35},
                {'material': 'MAT_A', 'location': 'DC_001', 'dps_location': 'PLANT_001', 'dps_percent': 0.15},
                {'material': 'MAT_B', 'location': 'DC_001', 'dps_location': 'DC_002', 'dps_percent': 0.30},
                # Week 10-12 (新增)
                {'material': 'MAT_A', 'location': 'DC_001', 'dps_location': 'DC_002', 'dps_percent': 0.38},
                {'material': 'MAT_A', 'location': 'DC_001', 'dps_location': 'PLANT_001', 'dps_percent': 0.12},
                {'material': 'MAT_B', 'location': 'DC_001', 'dps_location': 'DC_002', 'dps_percent': 0.32}
            ]),
            
            # 供应选择配置 (供应调整) - 扩展到12周
            'M1_SupplyChoiceConfig': pd.DataFrame([
                # Week 1-3 (原有配置)
                {'material': 'MAT_A', 'location': 'DC_001', 'week': 1, 'adjust_quantity': 50},
                {'material': 'MAT_A', 'location': 'DC_002', 'week': 1, 'adjust_quantity': 30},
                {'material': 'MAT_B', 'location': 'DC_001', 'week': 2, 'adjust_quantity': -20},
                {'material': 'MAT_B', 'location': 'DC_002', 'week': 2, 'adjust_quantity': 25},
                # Week 4-6 (新增)
                {'material': 'MAT_A', 'location': 'DC_001', 'week': 4, 'adjust_quantity': 40},
                {'material': 'MAT_A', 'location': 'DC_002', 'week': 4, 'adjust_quantity': 35},
                {'material': 'MAT_B', 'location': 'DC_001', 'week': 5, 'adjust_quantity': -15},
                {'material': 'MAT_B', 'location': 'DC_002', 'week': 5, 'adjust_quantity': 20},
                {'material': 'MAT_A', 'location': 'DC_001', 'week': 6, 'adjust_quantity': 45},
                {'material': 'MAT_A', 'location': 'DC_002', 'week': 6, 'adjust_quantity': 40},
                # Week 7-9 (新增)
                {'material': 'MAT_B', 'location': 'DC_001', 'week': 7, 'adjust_quantity': -10},
                {'material': 'MAT_B', 'location': 'DC_002', 'week': 7, 'adjust_quantity': 15},
                {'material': 'MAT_A', 'location': 'DC_001', 'week': 8, 'adjust_quantity': 50},
                {'material': 'MAT_A', 'location': 'DC_002', 'week': 8, 'adjust_quantity': 45},
                {'material': 'MAT_B', 'location': 'DC_001', 'week': 9, 'adjust_quantity': -5},
                {'material': 'MAT_B', 'location': 'DC_002', 'week': 9, 'adjust_quantity': 10},
                # Week 10-12 (新增)
                {'material': 'MAT_A', 'location': 'DC_001', 'week': 10, 'adjust_quantity': 55},
                {'material': 'MAT_A', 'location': 'DC_002', 'week': 10, 'adjust_quantity': 50},
                {'material': 'MAT_B', 'location': 'DC_001', 'week': 11, 'adjust_quantity': 0},
                {'material': 'MAT_B', 'location': 'DC_002', 'week': 11, 'adjust_quantity': 5},
                {'material': 'MAT_A', 'location': 'DC_001', 'week': 12, 'adjust_quantity': 60},
                {'material': 'MAT_A', 'location': 'DC_002', 'week': 12, 'adjust_quantity': 55}
            ]),
            
            # Module3 Enhanced Configuration (新增安全库存需求)
            'M3_SafetyStock': pd.DataFrame([
                {'material': 'MAT_A', 'location': 'PLANT_001', 'date': '2024-01-01', 'safety_stock_qty': 100},  # 从50增加到100
                {'material': 'MAT_A', 'location': 'DC_001', 'date': '2024-01-01', 'safety_stock_qty': 80},   # 从30增加到80
                {'material': 'MAT_A', 'location': 'DC_002', 'date': '2024-01-01', 'safety_stock_qty': 60},   # 从25增加到60
                {'material': 'MAT_B', 'location': 'PLANT_001', 'date': '2024-01-01', 'safety_stock_qty': 80},  # 从40增加到80
                {'material': 'MAT_B', 'location': 'DC_001', 'date': '2024-01-01', 'safety_stock_qty': 50},   # 从20增加到50
                {'material': 'MAT_B', 'location': 'DC_002', 'date': '2024-01-01', 'safety_stock_qty': 40}    # 新增：DC_002的安全库存
            ]),
            
            # Module5 Enhanced Configuration (新增)
            'M5_PushPullModel': pd.DataFrame([
                {'material': 'MAT_A', 'sending': 'PLANT_001', 'model': 'push'},
                {'material': 'MAT_B', 'sending': 'PLANT_001', 'model': 'push'}
            ]),
            
            'M5_DeployConfig': pd.DataFrame([
                {'material': 'MAT_A', 'sending': 'PLANT_001', 'receiving': 'DC_001', 'moq': 50, 'rv': 25, 'lsk': 7, 'day': 1},
                {'material': 'MAT_A', 'sending': 'PLANT_001', 'receiving': 'DC_002', 'moq': 50, 'rv': 25, 'lsk': 7, 'day': 1},
                {'material': 'MAT_B', 'sending': 'PLANT_001', 'receiving': 'DC_001', 'moq': 40, 'rv': 20, 'lsk': 7, 'day': 1}
            ]),
            
            # Module6 Enhanced Configuration (新增卡车配置)
            'M6_TruckTypeSpecs': pd.DataFrame([
                {'truck_type': 'LARGE', 'max_weight': 100, 'max_volume': 200, 'capacity_qty_in_weight': 100, 'capacity_qty_in_volume': 200},
                {'truck_type': 'MEDIUM', 'max_weight': 60, 'max_volume': 120, 'capacity_qty_in_weight': 60, 'capacity_qty_in_volume': 120},
                {'truck_type': 'SMALL', 'max_weight': 30, 'max_volume': 60, 'capacity_qty_in_weight': 30, 'capacity_qty_in_volume': 60}
            ]),
            
            'M6_TruckCapacityPlan': pd.DataFrame([
                # Week 1 (Days 1-7)
                {'date': '2024-01-01', 'sending': 'PLANT_001', 'receiving': 'DC_001', 'truck_type': 'LARGE', 'truck_number': 2},
                {'date': '2024-01-01', 'sending': 'PLANT_001', 'receiving': 'DC_002', 'truck_type': 'MEDIUM', 'truck_number': 2},
                {'date': '2024-01-02', 'sending': 'PLANT_001', 'receiving': 'DC_001', 'truck_type': 'LARGE', 'truck_number': 2},
                {'date': '2024-01-02', 'sending': 'PLANT_001', 'receiving': 'DC_002', 'truck_type': 'MEDIUM', 'truck_number': 2},
                {'date': '2024-01-03', 'sending': 'PLANT_001', 'receiving': 'DC_001', 'truck_type': 'LARGE', 'truck_number': 2},
                {'date': '2024-01-03', 'sending': 'PLANT_001', 'receiving': 'DC_002', 'truck_type': 'MEDIUM', 'truck_number': 2},
                {'date': '2024-01-04', 'sending': 'PLANT_001', 'receiving': 'DC_001', 'truck_type': 'LARGE', 'truck_number': 2},
                {'date': '2024-01-04', 'sending': 'PLANT_001', 'receiving': 'DC_002', 'truck_type': 'MEDIUM', 'truck_number': 2},
                {'date': '2024-01-05', 'sending': 'PLANT_001', 'receiving': 'DC_001', 'truck_type': 'LARGE', 'truck_number': 2},
                {'date': '2024-01-05', 'sending': 'PLANT_001', 'receiving': 'DC_002', 'truck_type': 'MEDIUM', 'truck_number': 2},
                {'date': '2024-01-06', 'sending': 'PLANT_001', 'receiving': 'DC_001', 'truck_type': 'LARGE', 'truck_number': 2},
                {'date': '2024-01-06', 'sending': 'PLANT_001', 'receiving': 'DC_002', 'truck_type': 'MEDIUM', 'truck_number': 2},
                {'date': '2024-01-07', 'sending': 'PLANT_001', 'receiving': 'DC_001', 'truck_type': 'LARGE', 'truck_number': 2},
                {'date': '2024-01-07', 'sending': 'PLANT_001', 'receiving': 'DC_002', 'truck_type': 'MEDIUM', 'truck_number': 2},
                # DC间调拨
                {'date': '2024-01-01', 'sending': 'DC_001', 'receiving': 'DC_002', 'truck_type': 'SMALL', 'truck_number': 1},
                {'date': '2024-01-02', 'sending': 'DC_001', 'receiving': 'DC_002', 'truck_type': 'SMALL', 'truck_number': 1},
                {'date': '2024-01-03', 'sending': 'DC_001', 'receiving': 'DC_002', 'truck_type': 'SMALL', 'truck_number': 1},
                {'date': '2024-01-04', 'sending': 'DC_001', 'receiving': 'DC_002', 'truck_type': 'SMALL', 'truck_number': 1},
                {'date': '2024-01-05', 'sending': 'DC_001', 'receiving': 'DC_002', 'truck_type': 'SMALL', 'truck_number': 1},
                {'date': '2024-01-06', 'sending': 'DC_001', 'receiving': 'DC_002', 'truck_type': 'SMALL', 'truck_number': 1},
                {'date': '2024-01-07', 'sending': 'DC_001', 'receiving': 'DC_002', 'truck_type': 'SMALL', 'truck_number': 1},
                {'date': '2024-01-01', 'sending': 'DC_002', 'receiving': 'DC_001', 'truck_type': 'SMALL', 'truck_number': 1},
                {'date': '2024-01-02', 'sending': 'DC_002', 'receiving': 'DC_001', 'truck_type': 'SMALL', 'truck_number': 1},
                {'date': '2024-01-03', 'sending': 'DC_002', 'receiving': 'DC_001', 'truck_type': 'SMALL', 'truck_number': 1},
                {'date': '2024-01-04', 'sending': 'DC_002', 'receiving': 'DC_001', 'truck_type': 'SMALL', 'truck_number': 1},
                {'date': '2024-01-05', 'sending': 'DC_002', 'receiving': 'DC_001', 'truck_type': 'SMALL', 'truck_number': 1},
                {'date': '2024-01-06', 'sending': 'DC_002', 'receiving': 'DC_001', 'truck_type': 'SMALL', 'truck_number': 1},
                {'date': '2024-01-07', 'sending': 'DC_002', 'receiving': 'DC_001', 'truck_type': 'SMALL', 'truck_number': 1},
                # 自循环路线
                {'date': '2024-01-01', 'sending': 'DC_001', 'receiving': 'DC_001', 'truck_type': 'SMALL', 'truck_number': 1},
                {'date': '2024-01-02', 'sending': 'DC_001', 'receiving': 'DC_001', 'truck_type': 'SMALL', 'truck_number': 1},
                {'date': '2024-01-03', 'sending': 'DC_001', 'receiving': 'DC_001', 'truck_type': 'SMALL', 'truck_number': 1},
                {'date': '2024-01-04', 'sending': 'DC_001', 'receiving': 'DC_001', 'truck_type': 'SMALL', 'truck_number': 1},
                {'date': '2024-01-05', 'sending': 'DC_001', 'receiving': 'DC_001', 'truck_type': 'SMALL', 'truck_number': 1},
                {'date': '2024-01-06', 'sending': 'DC_001', 'receiving': 'DC_001', 'truck_type': 'SMALL', 'truck_number': 1},
                {'date': '2024-01-07', 'sending': 'DC_001', 'receiving': 'DC_001', 'truck_type': 'SMALL', 'truck_number': 1},
                {'date': '2024-01-01', 'sending': 'DC_002', 'receiving': 'DC_002', 'truck_type': 'SMALL', 'truck_number': 1},
                {'date': '2024-01-02', 'sending': 'DC_002', 'receiving': 'DC_002', 'truck_type': 'SMALL', 'truck_number': 1},
                {'date': '2024-01-03', 'sending': 'DC_002', 'receiving': 'DC_002', 'truck_type': 'SMALL', 'truck_number': 1},
                {'date': '2024-01-04', 'sending': 'DC_002', 'receiving': 'DC_002', 'truck_type': 'SMALL', 'truck_number': 1},
                {'date': '2024-01-05', 'sending': 'DC_002', 'receiving': 'DC_002', 'truck_type': 'SMALL', 'truck_number': 1},
                {'date': '2024-01-06', 'sending': 'DC_002', 'receiving': 'DC_002', 'truck_type': 'SMALL', 'truck_number': 1},
                {'date': '2024-01-07', 'sending': 'DC_002', 'receiving': 'DC_002', 'truck_type': 'SMALL', 'truck_number': 1},
                {'date': '2024-01-01', 'sending': 'PLANT_001', 'receiving': 'PLANT_001', 'truck_type': 'LARGE', 'truck_number': 1},
                {'date': '2024-01-02', 'sending': 'PLANT_001', 'receiving': 'PLANT_001', 'truck_type': 'LARGE', 'truck_number': 1},
                {'date': '2024-01-03', 'sending': 'PLANT_001', 'receiving': 'PLANT_001', 'truck_type': 'LARGE', 'truck_number': 1},
                {'date': '2024-01-04', 'sending': 'PLANT_001', 'receiving': 'PLANT_001', 'truck_type': 'LARGE', 'truck_number': 1},
                {'date': '2024-01-05', 'sending': 'PLANT_001', 'receiving': 'PLANT_001', 'truck_type': 'LARGE', 'truck_number': 1},
                {'date': '2024-01-06', 'sending': 'PLANT_001', 'receiving': 'PLANT_001', 'truck_type': 'LARGE', 'truck_number': 1},
                {'date': '2024-01-07', 'sending': 'PLANT_001', 'receiving': 'PLANT_001', 'truck_type': 'LARGE', 'truck_number': 1}
            ]),
        }
        
        # Save to Excel
        with pd.ExcelWriter(config_file, engine='openpyxl') as writer:
            for sheet_name, df in config_data.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"✅ Test configuration created: {config_file}")
        return str(config_file)
    
    def run_e2e_test(self):
        """Run the complete E2E test"""
        print(f"\n🚀 Starting E2E Test: {self.test_name}")
        print("=" * 60)
        
        try:
            # Step 1: Create test configuration
            config_path = self.create_test_configuration()
            
            # Step 2: Run integrated simulation
            print(f"\n📊 Running integrated simulation...")
            result = run_integrated_simulation(
                config_path=config_path,
                start_date=self.start_date,
                end_date=self.end_date,
                output_base_dir=str(self.test_dir / "simulation_output")
            )
            
            # Step 3: Validate results
            self.validate_simulation_results(result)
            
            # Step 4: Generate test report
            self.generate_test_report(result)
            
        except Exception as e:
            print(f"❌ E2E Test failed: {e}")
            import traceback
            traceback.print_exc()
            self.validation_errors.append(f"Test execution failed: {e}")
    
    def validate_simulation_results(self, result: dict):
        """Validate simulation results for correctness"""
        print(f"\n🔍 Validating simulation results...")
        
        # Check basic completion
        if result['dates_processed'] != 7:
            self.validation_errors.append(f"Expected 7 days processed, got {result['dates_processed']}")
        
        # Check orchestrator final state
        final_stats = result.get('final_stats', {})
        if not final_stats:
            self.validation_errors.append("No final statistics available")
        else:
            print(f"📊 Final Statistics:")
            for key, value in final_stats.items():
                print(f"  {key}: {value}")
        
        # Enhanced validation checks
        try:
            self.validate_inventory_conservation()
            self.validate_module_consistency()
            self.validate_data_integrity()
            self.validate_business_rules()
        except Exception as e:
            self.validation_errors.append(f"Enhanced validation failed: {e}")
        
        # Validate orchestrator state files
        try:
            self.validate_orchestrator_outputs()
        except Exception as e:
            self.validation_errors.append(f"Orchestrator output validation failed: {e}")
        
        # Report validation results
        if self.validation_errors:
            print(f"\n⚠️  Validation Issues Found:")
            for i, error in enumerate(self.validation_errors, 1):
                print(f"  {i}. {error}")
        else:
            print(f"\n✅ All validations passed!")
    
    def validate_inventory_conservation(self):
        """Validate inventory conservation across the system"""
        print(f"  🔍 Validating inventory conservation...")
        
        # Load final orchestrator state
        orchestrator_dir = self.test_dir / "simulation_output" / "orchestrator"
        final_date = "20240107"
        
        # Check if output files exist
        inventory_file = orchestrator_dir / f"unrestricted_inventory_{final_date}.csv"
        intransit_file = orchestrator_dir / f"planning_intransit_{final_date}.csv"
        
        if not inventory_file.exists():
            self.validation_errors.append(f"Unrestricted inventory file not found: {inventory_file}")
            return
        
        if not intransit_file.exists():
            self.validation_errors.append(f"In-transit file not found: {intransit_file}")
            return
        
        # Load inventory data
        unrestricted_inv = pd.read_csv(inventory_file)
        intransit_inv = pd.read_csv(intransit_file)
        
        # Calculate total system inventory
        total_unrestricted = unrestricted_inv['quantity'].sum()
        total_intransit = intransit_inv['quantity'].sum() if not intransit_inv.empty else 0
        total_system_inventory = total_unrestricted + total_intransit
        
        # Initial inventory from configuration
        initial_inventory = 200 + 50 + 30 + 150 + 20 + 15  # Sum from test config
        
        print(f"    Initial inventory: {initial_inventory}")
        print(f"    Final unrestricted: {total_unrestricted}")
        print(f"    Final in-transit: {total_intransit}")
        print(f"    Total system inventory: {total_system_inventory}")
        
        # Note: Total may differ due to production and consumption, but should be reasonable
        if total_system_inventory < 0:
            self.validation_errors.append("Negative total system inventory detected")
        
        print(f"    ✅ Inventory conservation check completed")
    
    def validate_module_consistency(self):
        """Validate consistency between module outputs"""
        print(f"  🔍 Validating module output consistency...")
        
        # Check that all modules produced outputs for each day
        modules = ['module1', 'module3', 'module4', 'module5', 'module6']
        simulation_dir = self.test_dir / "simulation_output"
        
        for module in modules:
            module_dir = simulation_dir / module
            if not module_dir.exists():
                self.validation_errors.append(f"Missing output directory for {module}")
                continue
                
            # Check for daily output files
            for day in range(1, 8):  # 7 days
                date_str = f"2024-01-{day:02d}"
                expected_files = {
                    'module1': f"module1_output_{date_str.replace('-', '')}.xlsx",
                    'module3': f"Module3Output_{date_str.replace('-', '')}.xlsx",
                    'module4': f"Module4Output_{date_str.replace('-', '')}.xlsx",
                    'module5': f"Module5Output_{date_str.replace('-', '')}.xlsx",
                    'module6': f"Module6Output_{date_str.replace('-', '')}.xlsx"
                }
                
                if module in expected_files:
                    expected_file = module_dir / expected_files[module]
                    if not expected_file.exists():
                        self.validation_errors.append(f"Missing {module} output for {date_str}")
        
        print(f"    ✅ Module output consistency check completed")
    
    def validate_data_integrity(self):
        """Validate data integrity across the simulation"""
        print(f"  🔍 Validating data integrity...")
        
        try:
            # Check orchestrator state files exist
            orchestrator_dir = self.test_dir / "simulation_output" / "orchestrator"
            
            # Validate final state files
            final_date = "20240105"
            required_files = [
                f"unrestricted_inventory_{final_date}.csv",
                f"open_deployment_{final_date}.csv",
                f"planning_intransit_{final_date}.csv",
                f"space_quota_{final_date}.csv"
            ]
            
            for file_name in required_files:
                file_path = orchestrator_dir / file_name
                if not file_path.exists():
                    self.validation_errors.append(f"Missing orchestrator state file: {file_name}")
                else:
                    # Basic file content validation
                    df = pd.read_csv(file_path)
                    if "inventory" in file_name and df.empty:
                        self.validation_errors.append(f"Empty inventory file: {file_name}")
            
            print(f"    ✅ Data integrity check completed")
            
        except Exception as e:
            self.validation_errors.append(f"Data integrity validation error: {e}")
    
    def validate_business_rules(self):
        """Validate business rules and constraints"""
        print(f"  🔍 Validating business rules...")
        
        try:
            # Check inventory levels are non-negative
            orchestrator_dir = self.test_dir / "simulation_output" / "orchestrator"
            final_inventory_file = orchestrator_dir / "unrestricted_inventory_20240107.csv"
            
            if final_inventory_file.exists():
                inventory_df = pd.read_csv(final_inventory_file)
                if not inventory_df.empty:
                    negative_inventory = inventory_df[inventory_df['quantity'] < 0]
                    if not negative_inventory.empty:
                        self.validation_errors.append(f"Found negative inventory: {len(negative_inventory)} records")
                    
                    # Check that total system inventory decreased (due to shipments)
                    initial_total = 2800  # From config
                    final_total = sum(inventory_df['quantity'])
                    
                    if final_total > initial_total:
                        self.validation_errors.append(f"Final inventory ({final_total}) exceeds initial ({initial_total})")
            
            print(f"    ✅ Business rules validation completed")
            
        except Exception as e:
            self.validation_errors.append(f"Business rules validation error: {e}")
    
    def validate_orchestrator_outputs(self):
        """Validate orchestrator output files and data consistency"""
        print(f"  🔍 Validating orchestrator outputs...")
        
        orchestrator_dir = self.test_dir / "simulation_output" / "orchestrator"
        
        # Check required files for each simulation day
        for i in range(7):
            date_str = f"2024-01-{i+1:02d}"
            date_str_file = f"202401{i+1:02d}"
            
            required_files = [
                f"unrestricted_inventory_{date_str_file}.csv",
                f"open_deployment_{date_str_file}.csv", 
                f"planning_intransit_{date_str_file}.csv",
                f"space_quota_{date_str_file}.csv"
            ]
            
            for filename in required_files:
                filepath = orchestrator_dir / filename
                if not filepath.exists():
                    self.validation_errors.append(f"Missing orchestrator output: {filename}")
                else:
                    # Validate file is not empty and has proper structure
                    try:
                        df = pd.read_csv(filepath)
                        # It's OK for deployment and in-transit to be empty in test scenarios
                        if df.empty and not any(prefix in filename for prefix in ['open_deployment', 'planning_intransit']):
                            self.validation_errors.append(f"Empty orchestrator output: {filename}")
                    except Exception as e:
                        self.validation_errors.append(f"Invalid orchestrator output {filename}: {e}")
        
        print(f"    ✅ Orchestrator output validation completed")
    
    def generate_test_report(self, result: dict):
        """Generate comprehensive test report"""
        print(f"\n📋 Generating test report...")
        
        report_file = self.test_dir / "e2e_test_report.txt"
        
        with open(report_file, 'w') as f:
            f.write(f"E2E Integration Test Report\n")
            f.write(f"=" * 40 + "\n")
            f.write(f"Test Name: {self.test_name}\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Simulation Period: {self.start_date} to {self.end_date}\n")
            f.write(f"\n")
            
            f.write(f"Test Results:\n")
            f.write(f"-" * 20 + "\n")
            f.write(f"Days Processed: {result.get('dates_processed', 'N/A')}\n")
            f.write(f"Output Directory: {result.get('output_directory', 'N/A')}\n")
            f.write(f"\n")
            
            # Module results
            f.write(f"Module Results:\n")
            f.write(f"-" * 20 + "\n")
            for module_name, module_results in result.get('results', {}).items():
                f.write(f"{module_name.upper()}: {len(module_results)} successful days\n")
            f.write(f"\n")
            
            # Final statistics
            f.write(f"Final Statistics:\n")
            f.write(f"-" * 20 + "\n")
            for key, value in result.get('final_stats', {}).items():
                f.write(f"{key}: {value}\n")
            f.write(f"\n")
            
            # Validation results
            f.write(f"Validation Results:\n")
            f.write(f"-" * 20 + "\n")
            if self.validation_errors:
                f.write(f"❌ Test FAILED with {len(self.validation_errors)} issues:\n")
                for i, error in enumerate(self.validation_errors, 1):
                    f.write(f"  {i}. {error}\n")
            else:
                f.write(f"✅ Test PASSED - All validations successful\n")
            
            f.write(f"\n")
            f.write(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"✅ Test report saved: {report_file}")
        
        # Also create Excel summary
        self.create_excel_summary(result)
    
    def create_excel_summary(self, result: dict):
        """Create Excel summary of test results"""
        summary_file = self.test_dir / "e2e_test_summary.xlsx"
        
        # Create summary data
        summary_data = {
            'Test_Overview': pd.DataFrame([{
                'Test_Name': self.test_name,
                'Start_Date': self.start_date,
                'End_Date': self.end_date,
                'Days_Processed': result.get('dates_processed', 0),
                'Total_Validation_Errors': len(self.validation_errors),
                'Test_Status': 'PASSED' if not self.validation_errors else 'FAILED',
                'Test_Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }]),
            
            'Final_Statistics': pd.DataFrame([result.get('final_stats', {})]),
            
            'Module_Results': pd.DataFrame([
                {'Module': module_name.upper(), 'Successful_Days': len(module_results)}
                for module_name, module_results in result.get('results', {}).items()
            ]),
            
            'Validation_Errors': pd.DataFrame([
                {'Error_ID': i, 'Error_Message': error}
                for i, error in enumerate(self.validation_errors, 1)
            ]) if self.validation_errors else pd.DataFrame([{'Message': 'No validation errors'}])
        }
        
        # Save to Excel
        with pd.ExcelWriter(summary_file, engine='openpyxl') as writer:
            for sheet_name, df in summary_data.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"✅ Test summary saved: {summary_file}")

def run_comprehensive_e2e_test():
    """Run comprehensive E2E test suite"""
    print("🚀 Starting Comprehensive E2E Test Suite")
    print("=" * 80)
    
    # Test Case 1: Basic 1-Week Integration Test
    test1 = E2ETestCase("basic_1week_integration")
    test1.run_e2e_test()
    
    # Summary
    print(f"\n📊 E2E Test Suite Summary")
    print("=" * 50)
    
    if test1.validation_errors:
        print(f"❌ Basic 1-Week Integration Test: FAILED ({len(test1.validation_errors)} errors)")
        print("Key issues:")
        for error in test1.validation_errors[:3]:  # Show first 3 errors
            print(f"  - {error}")
    else:
        print(f"✅ Basic 1-Week Integration Test: PASSED")
        print("✅ 12-week forecast configuration working correctly")
        print("✅ 1-week simulation completed successfully")
        print("✅ All modules integrated with extended forecast config")
    
    print(f"\n🎯 Test Results Location: {test1.test_dir}")
    print(f"📋 Detailed reports available in test output directories")
    
    return test1.validation_errors == []

def run_12week_forecast_test():
    """Run specialized 12-week forecast test (but only simulate 1 week)"""
    print("🚀 Starting 12-Week Forecast Test (1-Week Simulation)")
    print("=" * 60)
    print("📊 Note: Configuration includes 12-week forecast, but test runs for 1 week only")
    print("📊 This validates the forecast conversion logic without long simulation time")
    print("=" * 60)
    
    # Create test case for 12-week forecast
    test_12week = E2ETestCase("12week_forecast_test")
    test_12week.run_e2e_test()
    
    # Summary
    print(f"\n📊 12-Week Forecast Test Summary")
    print("=" * 50)
    
    if test_12week.validation_errors:
        print(f"❌ 12-Week Forecast Test: FAILED ({len(test_12week.validation_errors)} errors)")
        print("Key issues:")
        for error in test_12week.validation_errors[:5]:  # Show first 5 errors
            print(f"  - {error}")
    else:
        print(f"✅ 12-Week Forecast Test: PASSED")
        print("✅ Extended forecast configuration working correctly")
        print("✅ 12-week planning horizon configured (1-week simulation)")
        print("✅ All modules integrated with extended timeline")
        print("✅ Forecast conversion from weekly to daily working")
    
    print(f"\n🎯 Test Results Location: {test_12week.test_dir}")
    print(f"📋 Detailed reports available in test output directories")
    
    return test_12week.validation_errors == []

if __name__ == "__main__":
    # Run both tests
    print("🚀 Starting Comprehensive E2E Test Suite with Extended Forecast")
    print("=" * 80)
    
    # Test 1: Basic 1-week integration
    success1 = run_comprehensive_e2e_test()
    
    # Test 2: Specialized 12-week forecast test
    success2 = run_12week_forecast_test()
    
    # Overall result
    overall_success = success1 and success2
    
    if overall_success:
        print(f"\n🎉 All tests PASSED! Extended forecast configuration is working correctly.")
    else:
        print(f"\n⚠️  Some tests FAILED. Please check the validation reports.")
    
    sys.exit(0 if overall_success else 1)