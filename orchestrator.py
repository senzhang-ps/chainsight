# orchestrator.py
# Unified State Management and Coordination Hub for Supply Chain Planning System
#
# Execution Order: M1 → M4 → M5 → M6 → M3
# 
# Core Responsibilities:
# 1. Physical inventory tracking (unrestricted inventory)
# 2. Open deployment management (deployment plans awaiting shipment)
# 3. In-transit inventory tracking (shipped but not yet delivered)
# 4. Production GR tracking (production receipts)
# 5. Delivery GR tracking (delivery receipts)
# 6. Space capacity management
# 7. State persistence and audit logging

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import os
from datetime import datetime

@dataclass
class DeploymentUID:
    """Unique identifier for deployment tracking"""
    material: str
    sending: str
    receiving: str
    planned_deploy_date: str  # YYYY-MM-DD format
    demand_element: str
    sequence: int  # Auto-incrementing sequence for uniqueness
    
    def to_string(self) -> str:
        """Convert to string representation for tracking"""
        return f"{self.material}|{self.sending}|{self.receiving}|{self.planned_deploy_date}|{self.demand_element}|{self.sequence:06d}"
    
    @classmethod
    def from_string(cls, uid_str: str) -> 'DeploymentUID':
        """Parse from string representation"""
        parts = uid_str.split('|')
        return cls(
            material=parts[0],
            sending=parts[1], 
            receiving=parts[2],
            planned_deploy_date=parts[3],
            demand_element=parts[4],
            sequence=int(parts[5])
        )

class Orchestrator:
    """
    Central state management and coordination hub for supply chain planning
    """
    
    def __init__(self, start_date: str, output_dir: str = "./orchestrator_output"):
        """
        Initialize orchestrator
        
        Args:
            start_date: Simulation start date (YYYY-MM-DD)
            output_dir: Directory for persistent storage
        """
        self.start_date = pd.to_datetime(start_date).normalize()
        self.current_date = self.start_date
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Core state management
        self.unrestricted_inventory: Dict[Tuple[str, str], int] = {}  # (material, location) -> quantity
        self.open_deployment: Dict[str, Dict] = {}  # uid -> deployment record
        self.in_transit: Dict[str, Dict] = {}  # uid -> in-transit record
        self.production_gr: List[Dict] = []  # Daily production receipts
        self.delivery_gr: List[Dict] = []  # Daily delivery receipts
        self.shipment_log: List[Dict] = []  # Daily shipments
        
        # Space capacity configuration
        self.space_capacity: pd.DataFrame = pd.DataFrame()
        
        # UID sequence counter
        self.uid_sequence = 0
        
        # Daily logs for audit
        self.daily_logs: List[Dict] = []
        
        # 期初和期末库存存储
        self.daily_beginning_inventory: Dict[str, Dict[Tuple[str, str], int]] = {}  # date -> {(material, location): quantity}
        self.daily_ending_inventory: Dict[str, Dict[Tuple[str, str], int]] = {}  # date -> {(material, location): quantity}
        
        # 初始库存配置存储
        self.initial_inventory: Dict[Tuple[str, str], int] = {}  # (material, location) -> quantity
        
        # 🆕 新增：发运出库日志  
        self.delivery_shipment_log: List[Dict] = []  # Daily delivery shipments from Module6
        
        print(f"✅ Orchestrator initialized for simulation starting {start_date}")
    
    def initialize_inventory(self, initial_inventory_df: pd.DataFrame):
        """
        Initialize physical inventory from M1_InitialInventory configuration
        
        Args:
            initial_inventory_df: DataFrame with columns [material, location, quantity]
        """
        self.unrestricted_inventory.clear()
        self.initial_inventory.clear()
        
        for _, row in initial_inventory_df.iterrows():
            key = (str(row['material']), str(row['location']))
            quantity = int(row['quantity'])
            self.unrestricted_inventory[key] = quantity
            self.initial_inventory[key] = quantity  # 保存初始库存副本
        
        print(f"✅ Initialized inventory with {len(initial_inventory_df)} records")
        self._log_event("INIT_INVENTORY", f"Initialized {len(initial_inventory_df)} inventory records")
    
    def set_space_capacity(self, space_capacity_df: pd.DataFrame):
        """
        Set space capacity configuration from Global_SpaceCapacity
        
        Args:
            space_capacity_df: DataFrame with columns [location, eff_from, eff_to, capacity]
        """
        self.space_capacity = space_capacity_df.copy()
        self.space_capacity['eff_from'] = pd.to_datetime(self.space_capacity['eff_from'])
        self.space_capacity['eff_to'] = pd.to_datetime(self.space_capacity['eff_to'])
        
        print(f"✅ Set space capacity configuration with {len(space_capacity_df)} records")
        self._log_event("SET_SPACE_CAPACITY", f"Configured {len(space_capacity_df)} space capacity records")
    
    def get_unrestricted_inventory_view(self, date: str) -> pd.DataFrame:
        """
        Get unrestricted inventory view for specified date
        
        Args:
            date: Date in YYYY-MM-DD format
            
        Returns:
            DataFrame with columns [date, material, location, quantity]
        """
        date_obj = pd.to_datetime(date).normalize()
        
        records = []
        for (material, location), quantity in self.unrestricted_inventory.items():
            records.append({
                'date': date_obj,
                'material': material,
                'location': location,
                'quantity': quantity
            })
        
        df = pd.DataFrame(records)
        if df.empty:
            df = pd.DataFrame(columns=['date', 'material', 'location', 'quantity'])
        
        return df
    
    def get_planning_intransit_view(self, date: str) -> pd.DataFrame:
        """
        Get planning in-transit view for specified date
        
        Args:
            date: Date in YYYY-MM-DD format
            
        Returns:
            DataFrame with columns [date, available_date, material, receiving, quantity]
        """
        date_obj = pd.to_datetime(date).normalize()
        
        records = []
        for uid, transit_record in self.in_transit.items():
            records.append({
                'date': date_obj,
                'available_date': pd.to_datetime(transit_record['actual_delivery_date']),
                'material': transit_record['material'],
                'receiving': transit_record['receiving'],
                'quantity': transit_record['quantity']
            })
        
        df = pd.DataFrame(records)
        if df.empty:
            df = pd.DataFrame(columns=['date', 'available_date', 'material', 'receiving', 'quantity'])
        
        return df
    
    def get_open_deployment(self, current_date: pd.Timestamp) -> pd.DataFrame:
        """
        Get open deployment view for specified date (Module6 interface)
        
        Args:
            current_date: Date as pandas Timestamp
            
        Returns:
            DataFrame with columns [material, sending, receiving, planned_deployment_date, 
                                   deployed_qty, demand_element, ori_deployment_uid]
        """
        return self.get_open_deployment_view(current_date.strftime('%Y-%m-%d'))
    
    def process_delivery_plan(self, delivery_plan_df: pd.DataFrame, simulation_date: pd.Timestamp):
        """
        Process delivery plan from Module6 (interface for Module6)
        
        Args:
            delivery_plan_df: DataFrame with delivery plans
            simulation_date: Current simulation date
        """
        self.process_module6_delivery(delivery_plan_df, simulation_date.strftime('%Y-%m-%d'))
    
    def get_open_deployment_view(self, date: str) -> pd.DataFrame:
        """
        Get open deployment view for specified date
        
        Args:
            date: Date in YYYY-MM-DD format
            
        Returns:
            DataFrame with columns [material, sending, receiving, planned_deployment_date, 
                                   deployed_qty, demand_element, ori_deployment_uid]
        """
        records = []
        for uid, deployment_record in self.open_deployment.items():
            records.append({
                'material': deployment_record['material'],
                'sending': deployment_record['sending'],
                'receiving': deployment_record['receiving'],
                'planned_deployment_date': pd.to_datetime(deployment_record['planned_deployment_date']),
                'deployed_qty': deployment_record['deployed_qty'],
                'demand_element': deployment_record['demand_element'],
                'ori_deployment_uid': uid
            })
        
        df = pd.DataFrame(records)
        if df.empty:
            df = pd.DataFrame(columns=['material', 'sending', 'receiving', 'planned_deployment_date',
                                     'deployed_qty', 'demand_element', 'ori_deployment_uid'])
        
        return df
    
    def get_space_quota_view(self, date: str) -> pd.DataFrame:
        """
        Calculate available space quota for specified date
        Formula: capacity - unrestricted_inventory (at beginning of simulation date)
        
        Args:
            date: Date in YYYY-MM-DD format
            
        Returns:
            DataFrame with columns [receiving, date, max_qty]
        """
        date_obj = pd.to_datetime(date).normalize()
        
        # Get effective space capacity for the date
        # Check if space_capacity is empty or not configured
        if self.space_capacity.empty or 'eff_from' not in self.space_capacity.columns:
            # Return empty DataFrame with correct structure
            return pd.DataFrame(columns=['receiving', 'date', 'max_qty'])
            
        effective_capacity = self.space_capacity[
            (self.space_capacity['eff_from'] <= date_obj) &
            (self.space_capacity['eff_to'] >= date_obj)
        ]
        
        records = []
        for _, capacity_row in effective_capacity.iterrows():
            location = capacity_row['location']
            capacity = capacity_row['capacity']
            
            # Calculate total unrestricted inventory at this location
            location_inventory = sum([
                qty for (material, loc), qty in self.unrestricted_inventory.items()
                if loc == location
            ])
            
            # Available quota = capacity - current inventory
            max_qty = max(0, capacity - location_inventory)
            
            records.append({
                'receiving': location,
                'date': date_obj,
                'max_qty': max_qty
            })
        
        df = pd.DataFrame(records)
        if df.empty:
            df = pd.DataFrame(columns=['receiving', 'date', 'max_qty'])
        
        return df
    
    def get_production_gr_view(self, date: str) -> pd.DataFrame:
        """
        Get production GR records for specified date
        
        Args:
            date: Date in YYYY-MM-DD format
            
        Returns:
            DataFrame with columns [date, material, location, quantity]
        """
        date_obj = pd.to_datetime(date).normalize()
        
        records = [record for record in self.production_gr 
                  if pd.to_datetime(record['date']).normalize() == date_obj]
        
        df = pd.DataFrame(records)
        if df.empty:
            df = pd.DataFrame(columns=['date', 'material', 'location', 'quantity'])
        
        return df
    
    def get_delivery_gr_view(self, date: str) -> pd.DataFrame:
        """
        Get delivery GR records for specified date
        
        Args:
            date: Date in YYYY-MM-DD format
            
        Returns:
            DataFrame with columns [date, material, receiving, quantity]
        """
        date_obj = pd.to_datetime(date).normalize()
        
        records = [record for record in self.delivery_gr 
                  if pd.to_datetime(record['date']).normalize() == date_obj]
        
        df = pd.DataFrame(records)
        if df.empty:
            df = pd.DataFrame(columns=['date', 'material', 'receiving', 'quantity'])
        
        return df
    
    def get_shipment_log_view(self, date: str) -> pd.DataFrame:
        """
        Get shipment log records for specified date
        
        Args:
            date: Date in YYYY-MM-DD format
            
        Returns:
            DataFrame with columns [date, material, location, quantity]
        """
        date_obj = pd.to_datetime(date).normalize()
        
        records = [record for record in self.shipment_log 
                  if pd.to_datetime(record['date']).normalize() == date_obj]
        
        df = pd.DataFrame(records)
        if df.empty:
            df = pd.DataFrame(columns=['date', 'material', 'location', 'quantity'])
        
        return df
    
    def process_module1_shipments(self, shipment_df: pd.DataFrame, date: str):
        """
        Process Module1 shipment data for the specified date
        
        Args:
            shipment_df: DataFrame with columns [date, material, location, quantity]
            date: Simulation date in YYYY-MM-DD format
        """
        date_obj = pd.to_datetime(date).normalize()
        
        # Filter shipments for current date
        daily_shipments = shipment_df[
            pd.to_datetime(shipment_df['date']).dt.normalize() == date_obj
        ]
        
        # Update unrestricted inventory
        for _, row in daily_shipments.iterrows():
            key = (str(row['material']), str(row['location']))
            if key in self.unrestricted_inventory:
                self.unrestricted_inventory[key] = max(0, self.unrestricted_inventory[key] - int(row['quantity']))
            
            # Log shipment
            self.shipment_log.append({
                'date': date_obj,
                'material': row['material'],
                'location': row['location'],
                'quantity': int(row['quantity']),
                'type': 'customer_shipment'
            })
        
        if len(daily_shipments) > 0:
            print(f"✅ Processed {len(daily_shipments)} M1 shipments for {date}")
            self._log_event("M1_SHIPMENTS", f"Processed {len(daily_shipments)} shipments")
    
    def process_module4_production(self, production_df: pd.DataFrame, date: str):
        """
        Process Module4 production data for the specified date
        
        Args:
            production_df: DataFrame with columns [available_date, material, location, produced_qty]
            date: Simulation date in YYYY-MM-DD format
        """
        date_obj = pd.to_datetime(date).normalize()
        
        # Filter production for current date (available_date = inventory receipt date)
        daily_production = production_df[
            pd.to_datetime(production_df['available_date']).dt.normalize() == date_obj
        ]
        
        # Update unrestricted inventory and log production GR
        for _, row in daily_production.iterrows():
            key = (str(row['material']), str(row['location']))
            quantity = int(row['produced_qty'])
            
            self.unrestricted_inventory[key] = self.unrestricted_inventory.get(key, 0) + quantity
            
            # Log production GR
            self.production_gr.append({
                'date': date_obj,
                'material': row['material'],
                'location': row['location'],
                'quantity': quantity
            })
        
        if len(daily_production) > 0:
            print(f"✅ Processed {len(daily_production)} M4 production receipts for {date}")
            self._log_event("M4_PRODUCTION", f"Processed {len(daily_production)} production receipts")
    
    def process_module5_deployment(self, deployment_df: pd.DataFrame, date: str):
        """
        Process Module5 deployment plans and update open deployment
        
        Args:
            deployment_df: DataFrame with columns [material, sending, receiving, planned_deployment_date,
                                                 deployed_qty, demand_element]
            date: Simulation date in YYYY-MM-DD format
        """
        date_obj = pd.to_datetime(date).normalize()
        
        print(f"    🔍 Orchestrator正在处理Module5部署计划: {len(deployment_df)} 条")
        if len(deployment_df) > 0:
            print(f"    📈 部署计划deployed_qty统计: {deployment_df['deployed_qty'].describe()}")
        
        # Add new deployment plans to open deployment
        for i, row in deployment_df.iterrows():
            # Generate unique UID
            self.uid_sequence += 1
            uid_obj = DeploymentUID(
                material=str(row['material']),
                sending=str(row['sending']),
                receiving=str(row['receiving']),
                planned_deploy_date=pd.to_datetime(row['planned_deployment_date']).strftime('%Y-%m-%d'),
                demand_element=str(row['demand_element']),
                sequence=self.uid_sequence
            )
            uid = uid_obj.to_string()
            
            original_qty = row['deployed_qty']
            converted_qty = self._safe_convert_to_int(row['deployed_qty'])
            
            if i < 3:  # 只显示前3条记录的详细信息
                print(f"      记录{i+1}: original_qty={original_qty} (类型: {type(original_qty)}), converted_qty={converted_qty}")
            
            self.open_deployment[uid] = {
                'material': str(row['material']),
                'sending': str(row['sending']),
                'receiving': str(row['receiving']),
                'planned_deployment_date': pd.to_datetime(row['planned_deployment_date']).strftime('%Y-%m-%d'),
                'deployed_qty': converted_qty,
                'demand_element': str(row['demand_element']),
                'creation_date': date_obj.strftime('%Y-%m-%d')
            }
        
        if len(deployment_df) > 0:
            print(f"✅ Added {len(deployment_df)} M5 deployment plans to open deployment for {date}")
            # 检查存储后的数量
            stored_qtys = [v['deployed_qty'] for v in self.open_deployment.values()]
            non_zero_qtys = [q for q in stored_qtys if q > 0]
            print(f"    🔍 存储后的数量统计: 总数={len(stored_qtys)}, 非零数量={len(non_zero_qtys)}")
            self._log_event("M5_DEPLOYMENT", f"Added {len(deployment_df)} deployment plans")
    
    def process_module6_delivery(self, delivery_df: pd.DataFrame, date: str):
        """
        Process Module6 delivery plans and update states
        
        Args:
            delivery_df: DataFrame with columns [ori_deployment_uid, material, sending, receiving,
                                               actual_ship_date, actual_delivery_date, delivery_qty]
            date: Simulation date in YYYY-MM-DD format
        """
        date_obj = pd.to_datetime(date).normalize()
        
        # Process each delivery record
        for _, row in delivery_df.iterrows():
            uid = str(row['ori_deployment_uid'])
            material = str(row['material'])
            sending = str(row['sending'])
            receiving = str(row['receiving'])
            ship_date = pd.to_datetime(row['actual_ship_date'])
            delivery_date = pd.to_datetime(row['actual_delivery_date'])
            quantity = self._safe_convert_to_int(row['delivery_qty'])
            
            # 只处理当天发运的货物（actual_ship_date == 当前仿真日期）
            if ship_date.normalize() != date_obj:
                continue
            
            # Reduce open deployment quantity
            if uid in self.open_deployment:
                self.open_deployment[uid]['deployed_qty'] -= quantity
                if self.open_deployment[uid]['deployed_qty'] <= 0:
                    del self.open_deployment[uid]
            
            # Reduce unrestricted inventory at sending location
            sending_key = (material, sending)
            if sending_key in self.unrestricted_inventory:
                self.unrestricted_inventory[sending_key] = max(0, 
                    self.unrestricted_inventory[sending_key] - quantity)
            
            # 🆕 记录发运出库日志
            self.delivery_shipment_log.append({
                'date': date_obj,
                'material': material,
                'sending': sending,
                'receiving': receiving,
                'quantity': quantity,
                'ori_deployment_uid': uid,
                'actual_ship_date': ship_date.strftime('%Y-%m-%d'),
                'actual_delivery_date': delivery_date.strftime('%Y-%m-%d'),
                'type': 'delivery_shipment'
            })
            
            # 判断处理逻辑：基于delivery_date是否为未来日期
            if delivery_date.normalize() > date_obj:
                # Create in-transit record for future delivery
                transit_uid = f"{uid}_transit_{date_obj.strftime('%Y%m%d')}"
                self.in_transit[transit_uid] = {
                    'material': material,
                    'sending': sending,
                    'receiving': receiving,
                    'actual_ship_date': ship_date.strftime('%Y-%m-%d'),
                    'actual_delivery_date': delivery_date.strftime('%Y-%m-%d'),
                    'quantity': quantity,
                    'ori_deployment_uid': uid
                }
            elif delivery_date.normalize() == date_obj:
                # Delivery is today, create delivery GR and update inventory immediately
                receiving_key = (material, receiving)
                self.unrestricted_inventory[receiving_key] = (
                    self.unrestricted_inventory.get(receiving_key, 0) + quantity)
                
                # Log delivery GR (with deduplication check)
                gr_record = {
                    'date': date_obj,
                    'material': material,
                    'receiving': receiving,
                    'quantity': quantity,
                    'ori_deployment_uid': uid
                }
                
                # Check for duplicates based on key fields
                existing_key = (date_obj, material, receiving, uid)
                if not any(
                    (record['date'], record['material'], record['receiving'], record['ori_deployment_uid']) == existing_key
                    for record in self.delivery_gr
                ):
                    self.delivery_gr.append(gr_record)
            # 如果delivery_date < date_obj，这是历史数据，应该已经处理过，跳过
        
        if len(delivery_df) > 0:
            print(f"✅ Processed {len(delivery_df)} M6 delivery plans for {date}")
            self._log_event("M6_DELIVERY", f"Processed {len(delivery_df)} delivery plans")
    
    def run_daily_processing(self, date: str,
                           shipment_df: Optional[pd.DataFrame] = None,
                           production_df: Optional[pd.DataFrame] = None,
                           deployment_df: Optional[pd.DataFrame] = None,
                           delivery_df: Optional[pd.DataFrame] = None):
        """
        Execute daily processing in correct order: M1 → M4 → M5 → M6
        
        Args:
            date: Simulation date in YYYY-MM-DD format
            shipment_df: Module1 shipment data
            production_df: Module4 production data
            deployment_df: Module5 deployment data
            delivery_df: Module6 delivery data
        """
        self.current_date = pd.to_datetime(date).normalize()
        
        print(f"\n📅 Processing date: {date}")
        
        # Check for delivery arrivals at start of day
        self._process_delivery_arrivals(date)
        
        # M1: Process shipments
        if shipment_df is not None and not shipment_df.empty:
            self.process_module1_shipments(shipment_df, date)
        
        # M4: Process production
        if production_df is not None and not production_df.empty:
            self.process_module4_production(production_df, date)
        
        # M5: Process deployments
        if deployment_df is not None and not deployment_df.empty:
            self.process_module5_deployment(deployment_df, date)
        
        # M6: Process deliveries
        if delivery_df is not None and not delivery_df.empty:
            self.process_module6_delivery(delivery_df, date)
        
        # Save daily state
        self.save_daily_state(date)
        
        print(f"✅ Completed daily processing for {date}")
    
    def _process_delivery_arrivals(self, date: str):
        """
        Process delivery arrivals for in-transit items that arrive today
        """
        date_obj = pd.to_datetime(date).normalize()
        
        completed_transits = []
        for transit_uid, transit_record in self.in_transit.items():
            if pd.to_datetime(transit_record['actual_delivery_date']).normalize() == date_obj:
                # Add to receiving location inventory
                receiving_key = (transit_record['material'], transit_record['receiving'])
                self.unrestricted_inventory[receiving_key] = (
                    self.unrestricted_inventory.get(receiving_key, 0) + transit_record['quantity'])
                
                # Log delivery GR (with improved deduplication check)
                gr_record = {
                    'date': date_obj,
                    'material': transit_record['material'],
                    'receiving': transit_record['receiving'],
                    'quantity': transit_record['quantity'],
                    'ori_deployment_uid': transit_record['ori_deployment_uid'],
                    'actual_ship_date': transit_record['actual_ship_date']  # 新增字段
                }
                
                # 改进的重复检查：包含发运日期
                existing_key = (date_obj, transit_record['material'], transit_record['receiving'], 
                              transit_record['ori_deployment_uid'], transit_record['actual_ship_date'])
                if not any(
                    (record['date'], record['material'], record['receiving'], 
                     record['ori_deployment_uid'], record.get('actual_ship_date')) == existing_key
                    for record in self.delivery_gr
                ):
                    self.delivery_gr.append(gr_record)
                
                completed_transits.append(transit_uid)
        
        for transit_uid in completed_transits:
            del self.in_transit[transit_uid]
        
        if completed_transits:
            print(f"✅ Processed {len(completed_transits)} delivery arrivals for {date}")
            self._log_event("DELIVERY_ARRIVALS", f"Processed {len(completed_transits)} delivery arrivals")
    
    def _safe_convert_to_int(self, value):
        """Safely convert pandas Series or scalar to integer"""
        try:
            # 如果是pandas Series，取第一个值
            if hasattr(value, 'iloc') and len(value) > 0:
                value = value.iloc[0]  # 从Series中取第一个值
            elif hasattr(value, 'item'):
                value = value.item()  # Convert Series to scalar using item()
            elif isinstance(value, pd.Series):
                # 处理特殊情况的Series
                if len(value) == 1:
                    value = value.iloc[0]
                elif len(value) > 1:
                    # 如果Series有多个值，取第一个并发出警告
                    print(f"    ⚠️  Series有多个值，取第一个: {value.iloc[0]}")
                    value = value.iloc[0]
                else:
                    # 空Series
                    return 0
            
            # 处理None或NaN
            if value is None or pd.isna(value):
                return 0
            
            # 转换为int
            return int(float(value))
            
        except (ValueError, TypeError, IndexError, AttributeError) as e:
            print(f"    ⚠️  数值转换错误: {value} (类型: {type(value)}) -> {e}")
            return 0
    
    def save_daily_state(self, date: str):
        """
        Save daily state to persistent storage
        
        Args:
            date: Date in YYYY-MM-DD format
        """
        date_str = pd.to_datetime(date).strftime('%Y%m%d')
        
        # Save unrestricted inventory view
        unrestricted_df = self.get_unrestricted_inventory_view(date)
        unrestricted_df.to_csv(self.output_dir / f"unrestricted_inventory_{date_str}.csv", index=False)
        
        # Save open deployment view
        open_deployment_df = self.get_open_deployment_view(date)
        open_deployment_df.to_csv(self.output_dir / f"open_deployment_{date_str}.csv", index=False)
        
        # Save in-transit view
        intransit_df = self.get_planning_intransit_view(date)
        intransit_df.to_csv(self.output_dir / f"planning_intransit_{date_str}.csv", index=False)
        
        # Save space quota view
        space_quota_df = self.get_space_quota_view(date)
        space_quota_df.to_csv(self.output_dir / f"space_quota_{date_str}.csv", index=False)
        
        # Save daily delivery GR
        delivery_gr_df = self.get_delivery_gr_view(date)
        delivery_gr_df.to_csv(self.output_dir / f"delivery_gr_{date_str}.csv", index=False)
        
        # Save daily production GR  
        production_gr_df = self.get_production_gr_view(date)
        production_gr_df.to_csv(self.output_dir / f"production_gr_{date_str}.csv", index=False)
        
        # Save daily shipment log
        date_obj = pd.to_datetime(date).normalize()
        daily_shipments = [record for record in self.shipment_log 
                          if pd.to_datetime(record['date']).normalize() == date_obj]
        shipment_df = pd.DataFrame(daily_shipments)
        if shipment_df.empty:
            shipment_df = pd.DataFrame(columns=['date', 'material', 'location', 'quantity'])
        shipment_df.to_csv(self.output_dir / f"shipment_log_{date_str}.csv", index=False)
        
        # 🆕 保存发运出库日志
        daily_delivery_shipments = [record for record in self.delivery_shipment_log 
                                   if pd.to_datetime(record['date']).normalize() == date_obj]
        delivery_shipment_df = pd.DataFrame(daily_delivery_shipments)
        if delivery_shipment_df.empty:
            delivery_shipment_df = pd.DataFrame(columns=['date', 'material', 'sending', 'receiving', 'quantity', 
                                                       'ori_deployment_uid', 'actual_ship_date', 'actual_delivery_date', 'type'])
        delivery_shipment_df.to_csv(self.output_dir / f"delivery_shipment_log_{date_str}.csv", index=False)
        
        # 🆕 生成库存变动日志
        inventory_change_df = self.generate_inventory_change_log(date)
        inventory_change_df.to_csv(self.output_dir / f"inventory_change_log_{date_str}.csv", index=False)
        print(f"  📊 已生成库存变动日志: {len(inventory_change_df)} 条记录")
        
        # Save daily logs
        if self.daily_logs:
            logs_df = pd.DataFrame(self.daily_logs)
            logs_df.to_csv(self.output_dir / f"daily_logs_{date_str}.csv", index=False)
    
    def _log_event(self, event_type: str, message: str):
        """
        Log orchestrator events for audit trail
        
        Args:
            event_type: Type of event
            message: Event message
        """
        self.daily_logs.append({
            'timestamp': datetime.now().isoformat(),
            'date': self.current_date.strftime('%Y-%m-%d'),
            'event_type': event_type,
            'message': message
        })
    
    def get_summary_statistics(self, date: str) -> Dict:
        """
        Get summary statistics for specified date
        
        Args:
            date: Date in YYYY-MM-DD format
            
        Returns:
            Dictionary with summary statistics
        """
        return {
            'date': date,
            'total_inventory_items': len(self.unrestricted_inventory),
            'total_inventory_quantity': sum(self.unrestricted_inventory.values()),
            'open_deployment_count': len(self.open_deployment),
            'in_transit_count': len(self.in_transit),
            'production_gr_count': len([r for r in self.production_gr 
                                      if r['date'] == pd.to_datetime(date).normalize()]),
            'delivery_gr_count': len([r for r in self.delivery_gr 
                                    if r['date'] == pd.to_datetime(date).normalize()]),
            'shipment_count': len([r for r in self.shipment_log 
                                 if r['date'] == pd.to_datetime(date).normalize()])
        }
    
    def save_beginning_inventory(self, date: str):
        """
        保存指定日期的期初库存状态（在任何库存变动之前调用）
        
        Args:
            date: 日期字符串 (YYYY-MM-DD)
        """
        self.daily_beginning_inventory[date] = self.unrestricted_inventory.copy()
        print(f"  💾 已保存 {date} 期初库存: {len(self.unrestricted_inventory)} 项")
    
    def save_ending_inventory(self, date: str):
        """
        保存指定日期的期末库存状态（在所有模块运行完成后调用）
        
        Args:
            date: 日期字符串 (YYYY-MM-DD)
        """
        self.daily_ending_inventory[date] = self.unrestricted_inventory.copy()
        print(f"  💾 已保存 {date} 期末库存: {len(self.unrestricted_inventory)} 项")
    
    def get_beginning_inventory_view(self, date: str) -> pd.DataFrame:
        """
        获取指定日期的期初库存视图
        
        Args:
            date: 日期字符串 (YYYY-MM-DD)
            
        Returns:
            DataFrame with columns [date, material, location, quantity]
        """
        date_obj = pd.to_datetime(date).normalize()
        
        if date in self.daily_beginning_inventory:
            beginning_inventory = self.daily_beginning_inventory[date]
        else:
            # 如果没有记录，使用当前库存（通常是第一天的情况）
            beginning_inventory = self.unrestricted_inventory
        
        records = []
        for (material, location), quantity in beginning_inventory.items():
            records.append({
                'date': date_obj,
                'material': material,
                'location': location,
                'quantity': quantity
            })
        
        df = pd.DataFrame(records)
        if df.empty:
            df = pd.DataFrame(columns=['date', 'material', 'location', 'quantity'])
        
        return df
    
    def generate_inventory_change_log(self, date: str) -> pd.DataFrame:
        """
        生成指定日期的库存变动日志
        记录每个物料-地点的完整库存变动：期初、入库、出库、期末
        
        Args:
            date: 日期字符串 (YYYY-MM-DD)
            
        Returns:
            DataFrame: 库存变动日志
        """
        date_obj = pd.to_datetime(date).normalize()
        
        # 获取所有涉及的物料-地点组合
        all_keys = set()
        
        # 从期初和期末库存获取
        if date in self.daily_beginning_inventory:
            all_keys.update(self.daily_beginning_inventory[date].keys())
        if date in self.daily_ending_inventory:
            all_keys.update(self.daily_ending_inventory[date].keys())
        
        # 从各种变动记录获取
        for record in self.production_gr:
            if pd.to_datetime(record['date']).normalize() == date_obj:
                all_keys.add((record['material'], record['location']))
        
        for record in self.delivery_gr:
            if pd.to_datetime(record['date']).normalize() == date_obj:
                all_keys.add((record['material'], record['receiving']))
        
        for record in self.shipment_log:
            if pd.to_datetime(record['date']).normalize() == date_obj:
                all_keys.add((record['material'], record['location']))
        
        # 🔧 修复：直接从内存的delivery_shipment_log获取发运出库数据
        delivery_ship_data = {}
        for record in self.delivery_shipment_log:
            if pd.to_datetime(record['date']).normalize() == date_obj:
                material = record['material']
                sending = record['sending']
                quantity = float(record['quantity'])
                
                key = (material, sending)
                delivery_ship_data[key] = delivery_ship_data.get(key, 0) + quantity
                all_keys.add(key)
        
        print(f"  📊 从内存获取发运出库 [{date}]: {len(delivery_ship_data)} 项")
        
        change_log = []
        
        for material, location in all_keys:
            # 期初库存
            beginning_qty = 0
            if date in self.daily_beginning_inventory:
                beginning_qty = self.daily_beginning_inventory[date].get((material, location), 0)
            
            # 生产入库
            production_qty = sum(
                record['quantity'] for record in self.production_gr
                if (pd.to_datetime(record['date']).normalize() == date_obj and 
                    record['material'] == material and record['location'] == location)
            )
            
            # 交付入库
            delivery_qty = sum(
                record['quantity'] for record in self.delivery_gr
                if (pd.to_datetime(record['date']).normalize() == date_obj and 
                    record['material'] == material and record['receiving'] == location)
            )
            
            # 发货出库
            shipment_qty = sum(
                record['quantity'] for record in self.shipment_log
                if (pd.to_datetime(record['date']).normalize() == date_obj and 
                    record['material'] == material and record['location'] == location)
            )
            
            # 发运出库（从内存获取）
            delivery_ship_qty = delivery_ship_data.get((material, location), 0)
            
            # 期末库存
            ending_qty = 0
            if date in self.daily_ending_inventory:
                ending_qty = self.daily_ending_inventory[date].get((material, location), 0)
            
            # 只记录有变动的记录
            if (beginning_qty != 0 or production_qty != 0 or delivery_qty != 0 or 
                shipment_qty != 0 or delivery_ship_qty != 0 or ending_qty != 0):
                
                # 应用负库存重置逻辑
                calculated_ending = beginning_qty + production_qty + delivery_qty - shipment_qty - delivery_ship_qty
                if calculated_ending < 0:
                    calculated_ending = 0
                
                change_log.append({
                    'date': date_obj,
                    'material': material,
                    'location': location,
                    'beginning_inventory': beginning_qty,
                    'production_gr': production_qty,
                    'delivery_gr': delivery_qty,
                    'shipment': shipment_qty,
                    'delivery_ship': delivery_ship_qty,
                    'ending_inventory': ending_qty,
                    'calculated_ending': calculated_ending,
                    'balance_diff': ending_qty - calculated_ending
                })
        
        df = pd.DataFrame(change_log)
        if df.empty:
            df = pd.DataFrame(columns=[
                'date', 'material', 'location', 'beginning_inventory', 'production_gr', 
                'delivery_gr', 'shipment', 'delivery_ship', 'ending_inventory', 
                'calculated_ending', 'balance_diff'
            ])
        
        return df
    
    def output_daily_inventory_summary(self, date: str):
        """
        输出指定日期的详细库存变动记录，用于与库存平衡检查对照
        
        Args:
            date: 日期字符串 (YYYY-MM-DD)
        """
        print(f"\n📊 === Orchestrator每日库存变动详情 [{date}] ===")
        
        # 获取期初期末库存
        beginning_inv = self.daily_beginning_inventory.get(date, {})
        ending_inv = self.daily_ending_inventory.get(date, {})
        
        date_obj = pd.to_datetime(date).normalize()
        
        # 获取当日各项变动
        production_gr = [gr for gr in self.production_gr if pd.to_datetime(gr['date']).normalize() == date_obj]
        delivery_gr = [gr for gr in self.delivery_gr if pd.to_datetime(gr['date']).normalize() == date_obj]
        shipments = [ship for ship in self.shipment_log if pd.to_datetime(ship['date']).normalize() == date_obj]
        
        # In-transit（当日发运的）
        in_transit_today = []
        for uid, transit in self.in_transit.items():
            if pd.to_datetime(transit['actual_ship_date']).normalize() == date_obj:
                in_transit_today.append(transit)
        
        # 统计汇总
        print(f"期初库存条目: {len(beginning_inv)}")
        print(f"生产入库条目: {len(production_gr)}")
        print(f"交付入库条目: {len(delivery_gr)}")
        print(f"发货出库条目: {len(shipments)}")
        print(f"发运计划出库条目: {len(in_transit_today)}")
        print(f"期末库存条目: {len(ending_inv)}")
        
        # 重点分析MAT_B@DC_001
        key = ('MAT_B', 'DC_001')
        material, location = key
        
        begin_qty = beginning_inv.get(key, 0)
        end_qty = ending_inv.get(key, 0)
        
        print(f"\n=== 重点分析: {material}@{location} ===")
        print(f"期初库存: {begin_qty}")
        
        # Production GR
        prod_qty = sum(gr['quantity'] for gr in production_gr 
                      if gr['material'] == material and gr['location'] == location)
        print(f"生产入库: +{prod_qty}")
        
        # Delivery GR
        del_qty = sum(gr['quantity'] for gr in delivery_gr 
                     if gr['material'] == material and gr['receiving'] == location)
        print(f"交付入库: +{del_qty}")
        
        # Shipment
        ship_qty = sum(ship['quantity'] for ship in shipments 
                      if ship['material'] == material and ship['location'] == location)
        print(f"发货出库: -{ship_qty}")
        
        # In-transit
        transit_qty = sum(transit['quantity'] for transit in in_transit_today 
                         if transit['material'] == material and transit['sending'] == location)
        print(f"发运出库: -{transit_qty}")
        
        print(f"期末库存: {end_qty}")
        
        # 计算期望值
        expected = begin_qty + prod_qty + del_qty - ship_qty - transit_qty
        print(f"计算期望: {begin_qty} + {prod_qty} + {del_qty} - {ship_qty} - {transit_qty} = {expected}")
        
        if expected != end_qty:
            print(f"⚠️  差异: 期望{expected}, 实际{end_qty}, 差异{end_qty - expected}")
        else:
            print(f"✅ 一致")

# Convenience functions for module integration
def create_orchestrator(start_date: str, output_dir: str = "./orchestrator_output") -> Orchestrator:
    """
    Create and initialize orchestrator instance
    
    Args:
        start_date: Simulation start date (YYYY-MM-DD)
        output_dir: Output directory for persistent storage
        
    Returns:
        Orchestrator instance
    """
    return Orchestrator(start_date, output_dir)

# Example usage and testing
if __name__ == "__main__":
    # Example initialization
    orchestrator = create_orchestrator("2024-01-01")
    
    # Example initial inventory
    initial_inventory = pd.DataFrame([
        {'material': 'MAT_A', 'location': 'PLANT_001', 'quantity': 1000},
        {'material': 'MAT_B', 'location': 'DC_001', 'quantity': 500}
    ])
    orchestrator.initialize_inventory(initial_inventory)
    
    # Example space capacity
    space_capacity = pd.DataFrame([
        {'location': 'DC_001', 'eff_from': '2024-01-01', 'eff_to': '2024-12-31', 'capacity': 2000}
    ])
    orchestrator.set_space_capacity(space_capacity)
    
    # Test views
    inventory_view = orchestrator.get_unrestricted_inventory_view("2024-01-01")
    space_quota = orchestrator.get_space_quota_view("2024-01-01")
    
    print("\n📊 Initial State:")
    print(f"Inventory items: {len(inventory_view)}")
    print(f"Space quota available: {space_quota['max_qty'].sum() if not space_quota.empty else 0}")
    
    stats = orchestrator.get_summary_statistics("2024-01-01")
    print(f"Summary: {stats}")