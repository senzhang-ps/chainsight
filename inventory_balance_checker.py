# inventory_balance_checker.py
# 库存平衡检查器 - 验证库存守恒原理
# 期初库存 + 入库（生产+收货） - 出库（发货+部署） = 期末库存

import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from validation_manager import ValidationManager

def _normalize_location(location_str: str) -> str:
    """标准化location为4位补0格式"""
    try:
        if pd.isna(location_str):
            return ""
        return str(int(float(location_str))).zfill(4)
    except (ValueError, TypeError):
        location_str = str(location_str).strip()
        if location_str.isdigit():
            return location_str.zfill(4)
        return location_str

def _normalize_inventory_keys(inventory_dict: Dict) -> Dict:
    """标准化库存字典的keys中的location"""
    normalized_dict = {}
    for (material, location), quantity in inventory_dict.items():
        normalized_location = _normalize_location(location)
        normalized_key = (str(material), normalized_location)
        normalized_dict[normalized_key] = quantity
    return normalized_dict

class InventoryBalanceChecker:
    """库存平衡检查器"""
    
    def __init__(self, validation_manager: ValidationManager, orchestrator):
        """
        初始化库存平衡检查器
        
        Args:
            validation_manager: 验证管理器
            orchestrator: Orchestrator 实例
        """
        self.vm = validation_manager
        self.orchestrator = orchestrator
        self.balance_log = []
        self.tolerance = 0.01  # 允许的数值误差
    
    def check_daily_balance(self, date: str) -> bool:
        """
        检查指定日期的库存平衡
        
        Args:
            date: 检查日期 (YYYY-MM-DD)
            
        Returns:
            bool: 平衡检查是否通过
        """
        try:
            self.vm.add_info("InventoryBalance", "DailyCheck", f"Checking inventory balance for {date}")
            
            date_obj = pd.to_datetime(date)
            
            # 获取期初库存：应该是当天模块运行之前的库存状态
            if date_obj <= pd.to_datetime('2024-01-01'):  # 仿真的第一天
                # 第一天的期初库存应该是初始库存配置
                beginning_inventory = self._get_initial_inventory()
            else:
                # 其他日子的期初库存：使用保存的期初库存记录
                prev_date = date_obj - pd.Timedelta(days=1)
                prev_date_str = prev_date.strftime('%Y-%m-%d')
                # 获取当天的期初库存记录
                beginning_inventory = self._get_beginning_inventory_by_date(date)  # 使用专门的期初库存方法
            
            # 获取当日各项库存变动
            production_receipts = self._get_production_receipts(date)
            delivery_receipts = self._get_delivery_receipts(date)
            shipments = self._get_shipments(date)
            delivery_plans = self._get_delivery_plans_from_module6(date)  # 从Module6输出读取实际delivery plan
            
            # 获取期末库存：当天模块运行完后的库存状态（从状态文件读取）
            ending_inventory = self._get_inventory_by_date(date)
            
            # 执行平衡检查
            balance_passed = self._validate_inventory_balance(
                date, beginning_inventory, production_receipts, 
                delivery_receipts, shipments, delivery_plans, ending_inventory
            )
            
            return balance_passed
            
        except Exception as e:
            self.vm.add_error("InventoryBalance", "CheckError", 
                            f"Failed to check balance for {date}: {str(e)}")
            return False
    
    def check_period_balance(self, start_date: str, end_date: str) -> bool:
        """
        检查整个期间的库存平衡
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            bool: 整个期间的平衡检查是否通过
        """
        date_range = pd.date_range(start_date, end_date, freq='D')
        all_passed = True
        
        for date in date_range:
            date_str = date.strftime('%Y-%m-%d')
            daily_passed = self.check_daily_balance(date_str)
            if not daily_passed:
                all_passed = False
        
        # 生成期间汇总报告
        self._generate_period_summary(start_date, end_date)
        
        return all_passed
    
    def _get_inventory_by_date(self, date: str) -> Dict[Tuple[str, str], float]:
        """
        获取指定日期的当前库存状态（期末库存）
        保持原有逻辑，不影响其他模块调用
        
        Args:
            date: 日期字符串
            
        Returns:
            Dict: 库存字典 {(material, location): quantity}
        """
        try:
            # 优先从期末库存记录获取
            if hasattr(self.orchestrator, 'daily_ending_inventory') and date in self.orchestrator.daily_ending_inventory:
                ending_inventory = self.orchestrator.daily_ending_inventory[date]
                inventory_dict = {}
                for (material, location), quantity in ending_inventory.items():
                    normalized_location = _normalize_location(location)
                    inventory_dict[(material, normalized_location)] = float(quantity)
                print(f"    🔍 从期末库存记录获取 [{date}]: {len(inventory_dict)} 项")
                return inventory_dict
            
            # 如果没有期末库存记录，使用当前库存状态
            inventory_df = self.orchestrator.get_unrestricted_inventory_view(date)
            
            inventory_dict = {}
            for _, row in inventory_df.iterrows():
                key = (row['material'], row['location'])
                normalized_location = _normalize_location(row['location'])
                inventory_dict[key] = float(row['quantity'])
            
            print(f"    🔍 从当前库存状态获取 [{date}]: {len(inventory_dict)} 项")
            return inventory_dict
            
        except Exception as e:
            self.vm.add_warning("InventoryBalance", "DataAccess", 
                              f"Failed to get inventory for {date}: {str(e)}")
            return {}
    
    def _get_beginning_inventory_by_date(self, date: str) -> Dict[Tuple[str, str], float]:
        """
        获取指定日期的期初库存状态
        专门用于库存平衡检查的期初库存
        
        Args:
            date: 日期字符串
            
        Returns:
            Dict: 期初库存字典 {(material, location): quantity}
        """
        try:
            # 优先从orchestrator的期初库存记录获取
            if hasattr(self.orchestrator, 'daily_beginning_inventory') and date in self.orchestrator.daily_beginning_inventory:
                beginning_inventory = self.orchestrator.daily_beginning_inventory[date]
                inventory_dict = {}
                for (material, location), quantity in beginning_inventory.items():
                    normalized_location = _normalize_location(location)
                    inventory_dict[(material, normalized_location)] = float(quantity)
                print(f"    🔍 从期初库存记录获取 [{date}]: {len(inventory_dict)} 项")
                return inventory_dict
            
            # 如果没有期初库存记录，回退到初始库存或前一天的库存
            date_obj = pd.to_datetime(date)
            if date_obj <= pd.to_datetime('2024-01-01'):
                # 第一天使用初始库存
                return self._get_initial_inventory()
            else:
                # 其他天使用当前库存（作为fallback）
                self.vm.add_warning("InventoryBalance", "DataAccess", 
                                  f"No beginning inventory record for {date}, using current inventory as fallback")
                return self._get_inventory_by_date(date)
            
        except Exception as e:
            self.vm.add_error("InventoryBalance", "BeginningInventoryError", 
                              f"Failed to get beginning inventory for {date}: {str(e)}")
            return {}
    
    def _get_production_receipts(self, date: str) -> Dict[Tuple[str, str], float]:
        """
        获取指定日期的生产入库
        
        Args:
            date: 日期字符串
            
        Returns:
            Dict: 生产入库字典
        """
        try:
            date_obj = pd.to_datetime(date).normalize()
            receipts_dict = {}
            
            # 方法1：从Orchestrator实例获取
            production_gr_df = self.orchestrator.get_production_gr_view(date)
            
            for _, row in production_gr_df.iterrows():
                key = (row['material'], row['location'])
                normalized_location = _normalize_location(row['location'])
                receipts_dict[key] = receipts_dict.get(key, 0) + float(row['quantity'])
            
            # 如果从Orchestrator实例获取不到数据，记录警告
            if not receipts_dict:
                self.vm.add_warning("InventoryBalance", "DataAccess", 
                                  f"No production GR data available from orchestrator for {date}. Using zero production receipts.")
            
            return receipts_dict
            
        except Exception as e:
            self.vm.add_warning("InventoryBalance", "DataAccess", 
                              f"Failed to get production receipts for {date}: {str(e)}")
            return {}
    
    def _get_delivery_receipts(self, date: str) -> Dict[Tuple[str, str], float]:
        """
        获取指定日期的交付入库
        
        Args:
            date: 日期字符串
            
        Returns:
            Dict: 交付入库字典
        """
        try:
            date_obj = pd.to_datetime(date).normalize()
            receipts_dict = {}
            
            # 方法1：从Orchestrator实例获取
            delivery_gr_df = self.orchestrator.get_delivery_gr_view(date)
            
            for _, row in delivery_gr_df.iterrows():
                key = (row['material'], row['receiving'])
                normalized_location = _normalize_location(row['receiving'])
                receipts_dict[key] = receipts_dict.get(key, 0) + float(row['quantity'])
            
            # 如果从Orchestrator实例获取不到数据，记录警告
            if not receipts_dict:
                self.vm.add_warning("InventoryBalance", "DataAccess", 
                                  f"No delivery GR data available from orchestrator for {date}. Using zero delivery receipts.")
            
            return receipts_dict
            
        except Exception as e:
            self.vm.add_warning("InventoryBalance", "DataAccess", 
                              f"Failed to get delivery receipts for {date}: {str(e)}")
            return {}
    
    def _get_initial_inventory(self) -> Dict[Tuple[str, str], float]:
        """
        获取初始库存配置（从orchestrator获取）
        
        Returns:
            Dict: 初始库存字典 {(material, location): quantity}
        """
        try:
            # 优先从orchestrator的initial_inventory获取真正的初始状态
            if hasattr(self.orchestrator, 'initial_inventory') and self.orchestrator.initial_inventory:
                initial_dict = {}
                for (material, location), quantity in self.orchestrator.initial_inventory.items():
                    normalized_location = _normalize_location(location)
                    initial_dict[(material, normalized_location)] = float(quantity)
                print(f"    🔍 从初始库存配置获取: {len(initial_dict)} 项")
                return initial_dict
            # 回退到当前库存（兼容旧版本）
            elif hasattr(self.orchestrator, 'unrestricted_inventory'):
                initial_dict = {}
                for (material, location), quantity in self.orchestrator.unrestricted_inventory.items():
                    normalized_location = _normalize_location(location)
                    initial_dict[(material, normalized_location)] = float(quantity)
                self.vm.add_warning("InventoryBalance", "DataAccess", 
                                  "Using current inventory as initial inventory (no initial_inventory found)")
                return initial_dict
            else:
                self.vm.add_warning("InventoryBalance", "DataAccess", 
                                  "Orchestrator has no unrestricted_inventory attribute")
                return {}
            
        except Exception as e:
            self.vm.add_warning("InventoryBalance", "DataAccess", 
                              f"Failed to get initial inventory from orchestrator: {str(e)}")
            return {}
    
    def _get_inventory_from_state_file(self, date: str) -> Dict[Tuple[str, str], float]:
        """
        从Orchestrator获取指定日期的库存（不再从文件读取）
        
        Args:
            date: 日期字符串
            
        Returns:
            Dict: 库存字典 {(material, location): quantity}
        """
        # 直接使用_get_inventory_by_date方法
        return self._get_inventory_by_date(date)
    
    def _get_delivery_plans_from_module6(self, date: str) -> Dict[Tuple[str, str], float]:
        """
        从orchestrator内存获取指定日期的delivery plan出库记录
        修复：直接从orchestrator内存数据获取，而不是从文件读取
        
        Args:
            date: 日期字符串
            
        Returns:
            Dict: 发运出库字典 {(material, sending_location): total_delivery_quantity}
        """
        try:
            # 直接从orchestrator内存获取发运出库数据
            if hasattr(self.orchestrator, 'get_delivery_shipments_by_date'):
                delivery_plans_dict = self.orchestrator.get_delivery_shipments_by_date(date)
                print(f"    📊 从orchestrator内存获取发运出库 [{date}]: {len(delivery_plans_dict)} 项")
                return delivery_plans_dict
            
            # 备选方案：从delivery_shipment_log直接读取
            elif hasattr(self.orchestrator, 'delivery_shipment_log'):
                date_obj = pd.to_datetime(date).normalize()
                delivery_plans_dict = {}
                
                for record in self.orchestrator.delivery_shipment_log:
                    if pd.to_datetime(record['date']).normalize() == date_obj:
                        key = (record['material'], record['sending'])
                        delivery_plans_dict[key] = delivery_plans_dict.get(key, 0) + record['quantity']
                
                print(f"    📊 从orchestrator内存获取发运出库 [{date}]: {len(delivery_plans_dict)} 项")
                return delivery_plans_dict
            
            # 最后的备选方案：从in_transit获取
            else:
                date_obj = pd.to_datetime(date).normalize()
                delivery_plans_dict = {}
                
                in_transit_data = getattr(self.orchestrator, 'in_transit', {})
                for transit_uid, transit_record in in_transit_data.items():
                    actual_ship_date = pd.to_datetime(transit_record.get('actual_ship_date'))
                    if actual_ship_date.normalize() == date_obj:
                        key = (transit_record['material'], transit_record['sending'])
                        delivery_plans_dict[key] = delivery_plans_dict.get(key, 0) + float(transit_record['quantity'])
                
                print(f"    📊 从orchestrator in_transit获取发运出库 [{date}]: {len(delivery_plans_dict)} 项")
                return delivery_plans_dict
            
        except Exception as e:
            self.vm.add_error("InventoryBalance", "DataAccess", 
                              f"Failed to get delivery plans from orchestrator memory for {date}: {str(e)}")
            return {}
    
    def _get_shipments(self, date: str) -> Dict[Tuple[str, str], float]:
        """
        获取指定日期的发货出库
        
        Args:
            date: 日期字符串
            
        Returns:
            Dict: 发货出库字典
        """
        try:
            date_obj = pd.to_datetime(date).normalize()
            shipments_dict = {}
            
            # 方法1：从 Orchestrator 的发货日志中获取
            shipment_log = getattr(self.orchestrator, 'shipment_log', [])
            
            for shipment in shipment_log:
                # 修复：正确比较日期，确保都是datetime格式
                shipment_date = pd.to_datetime(shipment.get('date')).normalize()
                if shipment_date == date_obj:
                    key = (shipment['material'], shipment['location'])
                    normalized_location = _normalize_location(shipment['location'])
                    shipments_dict[key] = shipments_dict.get(key, 0) + float(shipment['quantity'])
            
            # 如果从Orchestrator实例获取不到数据，记录警告
            if not shipments_dict:
                self.vm.add_warning("InventoryBalance", "DataAccess", 
                                  f"No shipment data available from orchestrator for {date}. Using zero shipments.")
            
            return shipments_dict
            
        except Exception as e:
            self.vm.add_warning("InventoryBalance", "DataAccess", 
                              f"Failed to get shipments for {date}: {str(e)}")
            return {}
    
    def _get_delivery_plans(self, date: str) -> Dict[Tuple[str, str], float]:
        """
        获取指定日期的实际发运出库（delivery plan执行后的出库）
        注意：delivery GR是按actual_delivery_date记录的，表示当天交付到达的物料
        这里需要获取的是从发送地点发出的物料（出库）
        
        Args:
            date: 日期字符串
            
        Returns:
            Dict: 发运出库字典 {(material, sending_location): quantity}
        """
        try:
            date_obj = pd.to_datetime(date).normalize()
            delivery_plans_dict = {}
            
            # 从 Orchestrator 的 delivery_gr 中获取数据
            # delivery_gr 记录的是当日交付到达的物料（入库）
            # 但我们需要的是发出时的出库记录
            
            # 方法1：直接从 Orchestrator 的 delivery_gr 获取
            try:
                delivery_gr_df = self.orchestrator.get_delivery_gr_view(date)
                
                # delivery_gr 记录的是交付到达，但我们需要的是发出出库
                # 需要通过 in_transit 或者其他方式获取发出记录
                
                # 从 in_transit 中查找当日发出的记录（actual_ship_date == date）
                in_transit_data = getattr(self.orchestrator, 'in_transit', {})
                
                for transit_uid, transit_record in in_transit_data.items():
                    actual_ship_date = pd.to_datetime(transit_record.get('actual_ship_date'))
                    if actual_ship_date.normalize() == date_obj:
                        material = transit_record['material']
                        sending = transit_record['sending']
                        quantity = float(transit_record['quantity'])
                        
                        key = (material, sending)
                        delivery_plans_dict[key] = delivery_plans_dict.get(key, 0) + quantity
                        
            except Exception as orchestrator_error:
                self.vm.add_warning("InventoryBalance", "DataAccess", 
                                  f"Failed to get delivery data from orchestrator for {date}: {orchestrator_error}")
            
            # 如果从Orchestrator内存获取不到数据，记录警告
            if not delivery_plans_dict:
                self.vm.add_warning("InventoryBalance", "DataAccess", 
                                  f"No delivery plan data available from orchestrator memory for {date}. Using zero delivery plans.")
            
            return delivery_plans_dict
            
        except Exception as e:
            self.vm.add_warning("InventoryBalance", "DataAccess", 
                              f"Failed to get delivery plans for {date}: {str(e)}")
            return {}
    
    def _validate_inventory_balance(self, date: str, beginning: Dict, production: Dict, 
                                  delivery: Dict, shipments: Dict, delivery_plans: Dict, 
                                  ending: Dict) -> bool:
        """
        验证库存平衡：期初库存 + 入库（生产+收货） - 出库（shipment+delivery plan） = 期末库存
        注意：需要减去delivery plan出库，因为Module6运行完后的unrestricted_inventory已经扣减了delivery plan
        
        Args:
            date: 日期
            beginning: 期初库存
            production: 生产入库
            delivery: 交付入库
            shipments: 发货出库
            delivery_plans: 实际执行的发运计划出库
            ending: 期末库存
            
        Returns:
            bool: 平衡是否正确
        """
        # 获取所有涉及的物料-地点组合
        all_keys = set()
        for d in [beginning, production, delivery, shipments, delivery_plans, ending]:
            # 标准化每个字典的keys
            normalized_d = _normalize_inventory_keys(d)
            all_keys.update(normalized_d.keys())

        # 同时标准化所有字典
        beginning = _normalize_inventory_keys(beginning)
        production = _normalize_inventory_keys(production)
        delivery = _normalize_inventory_keys(delivery)
        shipments = _normalize_inventory_keys(shipments)
        delivery_plans = _normalize_inventory_keys(delivery_plans)
        ending = _normalize_inventory_keys(ending)
        
        balance_passed = True
        imbalances = []
        
        # 输出详细的调试信息
        print(f"\n=== 库存平衡检查详情 [{date}] ===")
        print(f"总共检查 {len(all_keys)} 个物料-地点组合")
        
        # 输出各类数据的汇总统计
        print(f"\n--- 数据来源汇总 ---")
        print(f"期初库存条目数: {len(beginning)}")
        print(f"生产入库条目数: {len(production)}")
        print(f"交付入库条目数: {len(delivery)}")
        print(f"发货出库条目数: {len(shipments)}")
        print(f"发运计划出库条目数: {len(delivery_plans)}")
        print(f"期末库存条目数: {len(ending)}")
        
        # 输出从Orchestrator读取的期末库存详情
        print(f"\n--- 从Orchestrator读取的期末库存 ---")
        for key, qty in sorted(ending.items()):
            material, location = key
            print(f"{material}@{location}: {qty:.2f}")
        
        for key in all_keys:
            material, location = key
            
            # 计算理论期末库存（系统库存公式）
            begin_qty = beginning.get(key, 0)
            prod_in = production.get(key, 0)
            del_in = delivery.get(key, 0)
            ship_out = shipments.get(key, 0)
            delivery_out = delivery_plans.get(key, 0)  # 使用delivery plan出库
            actual_end = ending.get(key, 0)
            
            # 系统库存平衡公式：期初 + 生产GR + 交付GR - 发货 - delivery plan出库 = 期末
            calculated_end = begin_qty + prod_in + del_in - ship_out - delivery_out
            
            # 应用与Orchestrator相同的负库存重置逻辑
            # 当计算结果为负数时，系统自动重置为0（允许超量发货的业务逻辑）
            if calculated_end < 0:
                calculated_end = 0  # 重置负库存为0，与Orchestrator保持一致
                
            balance_diff = actual_end - calculated_end
            
            # 记录平衡信息
            balance_record = {
                'date': date,
                'material': material,
                'location': location,
                'beginning_inventory': begin_qty,
                'production_receipts': prod_in,
                'delivery_receipts': del_in,
                'shipments': ship_out,
                'delivery_plans': delivery_out,  # 计入系统库存公式
                'calculated_ending': calculated_end,
                'actual_ending': actual_end,
                'balance_difference': balance_diff
            }
            
            self.balance_log.append(balance_record)
            
            # 输出详细的计算过程（只针对有差异的项目）
            if abs(balance_diff) > self.tolerance:
                print(f"\n--- 库存不平衡详情: {material}@{location} ---")
                print(f"期初库存: {begin_qty:.2f}")
                print(f"生产入库: {prod_in:.2f}")
                print(f"交付入库: {del_in:.2f}")
                print(f"发货出库: {ship_out:.2f}")
                print(f"发运计划出库: {delivery_out:.2f}")
                print(f"计算期末: {begin_qty:.2f} + {prod_in:.2f} + {del_in:.2f} - {ship_out:.2f} - {delivery_out:.2f} = {calculated_end:.2f}")
                print(f"实际期末: {actual_end:.2f}")
                print(f"差异: {balance_diff:.2f}")
                
                imbalances.append({
                    'material': material,
                    'location': location,
                    'difference': balance_diff,
                    'calculated': calculated_end,
                    'actual': actual_end
                })
                balance_passed = False
            
            # 输出与Orchestrator的一一对比（总是输出，不管是否有差异）
            self._output_detailed_comparison(date, key, beginning, production, delivery, shipments, delivery_plans, ending)
        
        # 记录验证结果
        if balance_passed:
            self.vm.add_info("InventoryBalance", "BalanceCheck", 
                           f"[{date}] Inventory balance check passed for all {len(all_keys)} items")
        else:
            for imbalance in imbalances:
                self.vm.add_error("InventoryBalance", "Imbalance", 
                                f"[{date}] {imbalance['material']}@{imbalance['location']}: "
                                f"calculated={imbalance['calculated']:.2f}, "
                                f"actual={imbalance['actual']:.2f}, "
                                f"diff={imbalance['difference']:.2f} "
                                f"(formula: begin + production + delivery - shipment - delivery_plan)")
        
        return balance_passed
    
    def _generate_period_summary(self, start_date: str, end_date: str):
        """生成期间汇总报告"""
        if not self.balance_log:
            return
        
        balance_df = pd.DataFrame(self.balance_log)
        
        # 统计不平衡的条目
        imbalanced_items = balance_df[abs(balance_df['balance_difference']) > self.tolerance]
        
        if not imbalanced_items.empty:
            self.vm.add_error("InventoryBalance", "PeriodSummary", 
                            f"Period {start_date} to {end_date}: "
                            f"{len(imbalanced_items)} inventory imbalances detected")
        else:
            self.vm.add_info("InventoryBalance", "PeriodSummary", 
                           f"Period {start_date} to {end_date}: "
                           f"All {len(balance_df)} inventory transactions balanced")
    
    def get_balance_summary(self) -> pd.DataFrame:
        """
        获取平衡检查汇总
        
        Returns:
            pd.DataFrame: 平衡检查汇总数据
        """
        if not self.balance_log:
            return pd.DataFrame()
        
        return pd.DataFrame(self.balance_log)
    
    def check_negative_inventory(self, date: str) -> bool:
        """
        检查负库存
        
        Args:
            date: 检查日期
            
        Returns:
            bool: 是否存在负库存
        """
        try:
            inventory_dict = self._get_inventory_by_date(date)
            
            negative_items = [(k, v) for k, v in inventory_dict.items() if v < 0]
            
            if negative_items:
                for (material, location), qty in negative_items:
                    self.vm.add_error("InventoryBalance", "NegativeInventory", 
                                    f"[{date}] Negative inventory: {material}@{location} = {qty}")
                return False
            else:
                self.vm.add_info("InventoryBalance", "NegativeCheck", 
                               f"[{date}] No negative inventory detected")
                return True
                
        except Exception as e:
            self.vm.add_error("InventoryBalance", "NegativeCheckError", 
                            f"Failed to check negative inventory for {date}: {str(e)}")
            return False
    
    def validate_inventory_consistency(self, start_date: str, end_date: str) -> bool:
        """
        验证库存一致性（包括平衡检查和负库存检查）
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            bool: 一致性检查是否通过
        """
        self.vm.add_info("InventoryBalance", "ConsistencyCheck", 
                       f"Starting inventory consistency validation from {start_date} to {end_date}")
        
        # 检查期间平衡
        balance_passed = self.check_period_balance(start_date, end_date)
        
        # 检查负库存
        date_range = pd.date_range(start_date, end_date, freq='D')
        negative_check_passed = True
        
        for date in date_range:
            date_str = date.strftime('%Y-%m-%d')
            if not self.check_negative_inventory(date_str):
                negative_check_passed = False
        
        overall_passed = balance_passed and negative_check_passed
        
        if overall_passed:
            self.vm.add_info("InventoryBalance", "ConsistencyResult", 
                           "Inventory consistency validation passed")
        else:
            self.vm.add_error("InventoryBalance", "ConsistencyResult", 
                            "Inventory consistency validation failed")
        
        return overall_passed
    
    def _output_detailed_comparison(self, date: str, key: Tuple[str, str], 
                                   beginning: Dict, production: Dict, delivery: Dict, 
                                   shipments: Dict, delivery_plans: Dict, ending: Dict):
        """
        输出inventory_balance_checker与orchestrator的详细一一对比
        
        Args:
            date: 日期
            key: (material, location)
            beginning, production, delivery, shipments, delivery_plans, ending: 各项数据字典
        """
        material, location = key
        
        # 只对主要库存位置输出详细对比
        if key == ('MAT_B', 'DC_001') and date == '2024-01-01':
            print(f"\n🔍 === 详细一一对比 [{date}] {material}@{location} ===")
            
            # 获取库存平衡检查的数据
            checker_begin = beginning.get(key, 0)
            checker_prod = production.get(key, 0)
            checker_del = delivery.get(key, 0)
            checker_ship = shipments.get(key, 0)
            checker_delivery = delivery_plans.get(key, 0)
            checker_end = ending.get(key, 0)
            
            # 获取orchestrator的实际记录
            date_obj = pd.to_datetime(date).normalize()
            
            # Orchestrator期初库存
            orch_begin = 0
            if hasattr(self.orchestrator, 'daily_beginning_inventory') and date in self.orchestrator.daily_beginning_inventory:
                orch_begin = self.orchestrator.daily_beginning_inventory[date].get(key, 0)
            
            # Orchestrator生产GR
            orch_prod = 0
            if hasattr(self.orchestrator, 'production_gr'):
                orch_prod = sum(gr['quantity'] for gr in self.orchestrator.production_gr 
                               if pd.to_datetime(gr['date']).normalize() == date_obj and 
                                  gr['material'] == material and gr['location'] == location)
            
            # Orchestrator交付GR
            orch_del = 0
            if hasattr(self.orchestrator, 'delivery_gr'):
                orch_del = sum(gr['quantity'] for gr in self.orchestrator.delivery_gr 
                              if pd.to_datetime(gr['date']).normalize() == date_obj and 
                                 gr['material'] == material and gr['receiving'] == location)
            
            # Orchestrator发货
            orch_ship = 0
            if hasattr(self.orchestrator, 'shipment_log'):
                orch_ship = sum(ship['quantity'] for ship in self.orchestrator.shipment_log 
                               if pd.to_datetime(ship['date']).normalize() == date_obj and 
                                  ship['material'] == material and ship['location'] == location)
            
            # Orchestrator发运
            orch_delivery = 0
            if hasattr(self.orchestrator, 'in_transit'):
                for uid, transit in self.orchestrator.in_transit.items():
                    if (pd.to_datetime(transit['actual_ship_date']).normalize() == date_obj and 
                        transit['material'] == material and transit['sending'] == location):
                        orch_delivery += transit['quantity']
            
            # Orchestrator期末库存
            orch_end = 0
            if hasattr(self.orchestrator, 'daily_ending_inventory') and date in self.orchestrator.daily_ending_inventory:
                orch_end = self.orchestrator.daily_ending_inventory[date].get(key, 0)
            
            # 输出对比表格
            print("📋 数据项目          | 库存检查器 | Orchestrator | 差异")
            print("=" * 55)
            print(f"期初库存            | {checker_begin:10.2f} | {orch_begin:12.2f} | {checker_begin-orch_begin:4.2f}")
            print(f"生产入库            | {checker_prod:10.2f} | {orch_prod:12.2f} | {checker_prod-orch_prod:4.2f}")
            print(f"交付入库            | {checker_del:10.2f} | {orch_del:12.2f} | {checker_del-orch_del:4.2f}")
            print(f"发货出库            | {checker_ship:10.2f} | {orch_ship:12.2f} | {checker_ship-orch_ship:4.2f}")
            print(f"发运出库            | {checker_delivery:10.2f} | {orch_delivery:12.2f} | {checker_delivery-orch_delivery:4.2f}")
            print(f"期末库存            | {checker_end:10.2f} | {orch_end:12.2f} | {checker_end-orch_end:4.2f}")
            print("=" * 55)
            
            # 计算期望值对比
            checker_expected = checker_begin + checker_prod + checker_del - checker_ship - checker_delivery
            orch_expected = orch_begin + orch_prod + orch_del - orch_ship - orch_delivery
            
            print(f"计算期望            | {checker_expected:10.2f} | {orch_expected:12.2f} | {checker_expected-orch_expected:4.2f}")
            
            # 判断是否一致
            total_diff = abs(checker_begin-orch_begin) + abs(checker_prod-orch_prod) + abs(checker_del-orch_del) + abs(checker_ship-orch_ship) + abs(checker_delivery-orch_delivery) + abs(checker_end-orch_end)
            
            if total_diff < 0.01:
                print("✅ 数据完全一致")
            else:
                print(f"❌ 数据不一致，总差异: {total_diff:.2f}")
                
            print()