# orchestrator.py（编排器）
# 供应链计划系统的一体化状态管理与协调枢纽
#
# 执行顺序：M1 → M4 → M5 → M6 → M3
# 
# 核心职责：
# 1. 实物库存跟踪（非限制库存）
# 2. 开放调拨管理（等待发运的调拨计划）
# 3. 在途库存跟踪（已发运但尚未交付）
# 4. 生产 GR 跟踪（生产入库）
# 5. 交付 GR 跟踪（收货入库）
# 6. 空间容量管理
# 7. 状态持久化与审计日志

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import os
from datetime import datetime

# 在文件开头添加字符串格式化函数
def _normalize_material(material_str) -> str:
    """规范化物料字符串以确保一致格式——移除数值物料的 .0 后缀"""
    # 处理 None 和 pandas 的 NA
    if material_str is None or pd.isna(material_str):
        return ""

    try:
        # 如果是数字（int或float），转换为整数字符串以移除多余的.0
        if isinstance(material_str, (int, float)) or str(material_str).replace('.', '').replace('-', '').isdigit():
            return str(int(float(material_str)))
        else:
            # 非数字material，直接返回字符串
            return str(material_str)
    except (ValueError, TypeError):
        # 如果转换失败，直接返回字符串
        return str(material_str)

def _normalize_location(location_str) -> str:
    """规范化地点字符串：若为数字则补齐到 4 位"""
    if pd.isna(location_str) or location_str is None:
        return ""
    
    location_str = str(location_str).strip()
    
    try:
        # 检查是否为纯数字字符串
        if location_str.isdigit():
            return str(int(location_str)).zfill(4)
        else:
            # 非数字location（如A888），直接返回字符串，不做padding
            return location_str
    except (ValueError, TypeError):
        return str(location_str)

def _normalize_sending(sending_str) -> str:
    """规范化发货地字符串：若为数字则补齐到 4 位"""
    if pd.isna(sending_str) or sending_str is None:
        return ""
    
    sending_str = str(sending_str).strip()
    
    try:
        # 检查是否为纯数字字符串
        if sending_str.isdigit():
            return str(int(sending_str)).zfill(4)
        else:
            # 非数字sending（如A888），直接返回字符串，不做padding
            return sending_str
    except (ValueError, TypeError):
        return str(sending_str)

def _normalize_receiving(receiving_str) -> str:
    """规范化收货地字符串：若为数字则补齐到 4 位"""
    if pd.isna(receiving_str) or receiving_str is None:
        return ""
    
    receiving_str = str(receiving_str).strip()
    
    try:
        # 检查是否为纯数字字符串
        if receiving_str.isdigit():
            return str(int(receiving_str)).zfill(4)
        else:
            # 非数字receiving（如A888），直接返回字符串，不做padding
            return receiving_str
    except (ValueError, TypeError):
        return str(receiving_str)

def _normalize_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    将标识符列规范化为字符串并按规则格式化。

    使用向量化操作提升性能，与 Dev 版本 orchestrator._normalize_identifiers 行为一致：
    - material 列：调用 _normalize_material 移除数值物料的 .0 后缀
    - location/sending/receiving/sourcing 列：纯数字补齐到 4 位
    """
    if df.empty:
        return df
    
    # 定义需要字符串转换的标识符列
    identifier_cols = ['material', 'location', 'sending', 'receiving', 'sourcing']
    
    df = df.copy()
    
    # 向量化处理 material 列 —— 与 Dev 版本一致，移除数值物料的 .0 后缀
    if 'material' in df.columns:
        df['material'] = df['material'].astype(str)
        df['material'] = df['material'].replace(['nan', 'None', '<NA>', 'NaN'], '')
        # 移除数值物料尾部的 .0（例如 "80813644.0" → "80813644"）
        df['material'] = df['material'].str.replace(r'\.0$', '', regex=True)
    
    # 向量化处理 location 类列（location, sending, receiving, sourcing）
    location_cols = ['location', 'sending', 'receiving', 'sourcing']
    for col in location_cols:
        if col in df.columns:
            # 转换为字符串并去除空白
            df[col] = df[col].astype(str).str.strip()
            # 处理 NA/None
            df[col] = df[col].replace(['nan', 'None', '<NA>', 'NaN'], '')
            # 识别纯数字的行并补齐4位
            is_numeric = df[col].str.match(r'^\d+$', na=False)
            df.loc[is_numeric, col] = df.loc[is_numeric, col].str.zfill(4)
    
    return df

@dataclass
class DeploymentUID:
    """用于部署跟踪的唯一标识符"""
    material: str
    sending: str
    receiving: str
    planned_deploy_date: str  # YYYY-MM-DD format
    demand_element: str
    sequence: int  # Auto-incrementing sequence for uniqueness
    
    def to_string(self) -> str:
        """转换为字符串表示以便跟踪"""
        return f"{self.material}|{self.sending}|{self.receiving}|{self.planned_deploy_date}|{self.demand_element}|{self.sequence:06d}"
    
    @classmethod
    def from_string(cls, uid_str: str) -> 'DeploymentUID':
        """从字符串表示解析"""
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
    供应链计划的中心状态管理与协调枢纽
    """
    
    def __init__(self, start_date: str, output_dir: str = "./orchestrator_output"):
        """
        初始化编排器

        Args:
            start_date: 仿真开始日期（YYYY-MM-DD）
            output_dir: 持久化存储目录
        """
        self.start_date = pd.to_datetime(start_date).normalize()
        self.current_date = self.start_date
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 核心状态管理
        self.unrestricted_inventory: Dict[Tuple[str, str], int] = {}  # (material, location) -> quantity
        self.open_deployment: Dict[str, Dict] = {}  # uid -> deployment record
        self.in_transit: Dict[str, Dict] = {}  # uid -> in-transit record
        self.production_gr: List[Dict] = []  # Daily production receipts
        self.delivery_gr: List[Dict] = []  # Daily delivery receipts
        self.shipment_log: List[Dict] = []  # Daily shipments
        self.production_plan_backlog: List[Dict] = []  # 存所有已确认生产(含未来)，供 M3 查询
        # 空间容量配置
        self.space_capacity: pd.DataFrame = pd.DataFrame()
        
        # 🚀 阶段 6：按日期索引实现 O(1) 查询（替代 O(n) 列表扫描）
        self.production_gr_by_date: Dict[str, List[Dict]] = {}  # date_str -> records
        self.delivery_gr_by_date: Dict[str, List[Dict]] = {}  # date_str -> records
        self.shipment_log_by_date: Dict[str, List[Dict]] = {}  # date_str -> records
        self.delivery_shipment_log_by_date: Dict[str, List[Dict]] = {}  # date_str -> records
        
        # UID 序列计数器
        self.uid_sequence = 0
        # 过期清理的全局宽限天数（可运行时修改）
        self.cleanup_grace_days: int = 100

        # 用于审计的每日日志
        self.daily_logs: List[Dict] = []
        
        # 期初和期末库存存储
        self.daily_beginning_inventory: Dict[str, Dict[Tuple[str, str], int]] = {}  # date -> {(material, location): quantity}
        self.daily_ending_inventory: Dict[str, Dict[Tuple[str, str], int]] = {}  # date -> {(material, location): quantity}
        
        # 初始库存配置存储
        self.initial_inventory: Dict[Tuple[str, str], int] = {}  # (material, location) -> quantity
        
        # 🆕 新增：发运出库日志  
        self.delivery_shipment_log: List[Dict] = []  # Daily delivery shipments from Module6

        # 记录当天是否应经完成过一次清理
        self._last_cleanup_date: Optional[pd.Timestamp] = None
        
        print(f"✅ Orchestrator initialized for simulation starting {start_date}")
    
    def initialize_inventory(self, initial_inventory_df: pd.DataFrame):
        """
        从 M1_InitialInventory 配置初始化实物库存

        Args:
            initial_inventory_df: 含列 [material, location, quantity] 的 DataFrame
        """
        self.unrestricted_inventory.clear()
        self.initial_inventory.clear()
        
        # 确保标识符字段为字符串格式
        normalized_df = _normalize_identifiers(initial_inventory_df)
        
        # 性能优化：使用 itertuples 替代 iterrows
        for row in normalized_df.itertuples():
            key = (row.material, row.location)
            quantity = int(row.quantity)
            self.unrestricted_inventory[key] = quantity
            self.initial_inventory[key] = quantity  # 保存初始库存副本
        
        # print(f"✅ Initialized inventory with {len(normalized_df)} records")
        self._log_event("INIT_INVENTORY", f"Initialized {len(normalized_df)} inventory records")
    
    def set_space_capacity(self, space_capacity_df: pd.DataFrame):
        """
        从 Global_SpaceCapacity 配置设置空间容量

        Args:
            space_capacity_df: 含列 [location, eff_from, eff_to, capacity] 的 DataFrame
        """
        # 确保标识符字段为字符串格式
        self.space_capacity = _normalize_identifiers(space_capacity_df.copy())
        self.space_capacity["eff_from"] = pd.to_datetime(
            self.space_capacity["eff_from"].astype(str),
            format="%Y-%m-%d",
            errors="coerce",
        )
        self.space_capacity["eff_to"] = pd.to_datetime(
            self.space_capacity["eff_to"].astype(str),
            format="%Y-%m-%d",
            errors="coerce",
        )
        
        # print(f"✅ Set space capacity configuration with {len(space_capacity_df)} records")
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
                'material': _normalize_material(material),  # 添加格式化
                'location': _normalize_location(location),  # 添加格式化
                'quantity': quantity
            })
        
        df = pd.DataFrame(records)
        if df.empty:
            df = pd.DataFrame(columns=['date', 'material', 'location', 'quantity'])
        
        return df
    
    def get_current_unrestricted_inventory(self) -> Dict[Tuple[str, str], int]:
        """
        Get current unrestricted inventory as a dictionary
        
        Returns:
            Dict: {(material, location): quantity} with normalized keys
        """
        normalized_inventory = {}
        for (material, location), quantity in self.unrestricted_inventory.items():
            normalized_key = (_normalize_material(material), _normalize_location(location))
            normalized_inventory[normalized_key] = quantity
        return normalized_inventory
    
    def get_planning_intransit_view(self, date: str) -> pd.DataFrame:
        """
        Get planning in-transit view for specified date
        
        Args:
            date: Date in YYYY-MM-DD format
            
        Returns:
            DataFrame with all fields needed for restoration:
            [transit_uid, date, material, sending, receiving, actual_ship_date, 
             actual_delivery_date, quantity, ori_deployment_uid, vehicle_uid]
        """
        date_obj = pd.to_datetime(date).normalize()
        
        records = []
        for uid, transit_record in self.in_transit.items():
            records.append({
                'transit_uid': uid,  # Add UID for restoration
                'date': date_obj,
                'material': _normalize_material(transit_record['material']), # 添加格式化
                'sending': transit_record.get('sending', ''),  # Add sending
                'receiving': transit_record['receiving'],
                'actual_ship_date': transit_record.get('actual_ship_date', ''),  # Add ship date
                'actual_delivery_date': transit_record['actual_delivery_date'],
                'quantity': transit_record['quantity'],
                'ori_deployment_uid': transit_record.get('ori_deployment_uid', ''),  # Add original UID
                'vehicle_uid': transit_record.get('vehicle_uid', '')  # Add vehicle UID
            })
        
        df = pd.DataFrame(records)
        if df.empty:
            df = pd.DataFrame(columns=[
                'transit_uid', 'date', 'material', 'sending', 'receiving', 
                'actual_ship_date', 'actual_delivery_date', 'quantity', 
                'ori_deployment_uid', 'vehicle_uid'
            ])
        
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
        获取指定日期的开放调拨视图
        注意：本函数不再触发过期清理；清理只在 run_daily_processing() 开头执行一次。
        返回列: [material, sending, receiving, planned_deployment_date, deployed_qty, demand_element, ori_deployment_uid]
        """
        records = []
        for uid, deployment_record in self.open_deployment.items():
            records.append({
                'material': _normalize_material(deployment_record['material']),
                'sending': _normalize_sending(deployment_record['sending']),
                'receiving': _normalize_receiving(deployment_record['receiving']),
                'planned_deployment_date': pd.to_datetime(deployment_record['planned_deployment_date']),
                'deployed_qty': deployment_record['deployed_qty'],
                'demand_element': deployment_record['demand_element'],
                'ori_deployment_uid': uid
            })
        df = pd.DataFrame(records)
        if df.empty:
            df = pd.DataFrame(columns=[
                'material', 'sending', 'receiving', 'planned_deployment_date',
                'deployed_qty', 'demand_element', 'ori_deployment_uid'
            ])
        return df

    
    def get_space_quota_view(self, date: str) -> pd.DataFrame:
        """
        计算指定日期的可用空间额度
        公式：capacity - unrestricted_inventory（仿真日开始时）

        Args:
            date: 日期（YYYY-MM-DD）

        Returns:
            含列 [receiving, date, max_qty] 的 DataFrame
        """
        date_obj = pd.to_datetime(date).normalize()
        
        # 获取指定日期的有效空间容量
        # 检查 space_capacity 是否为空或未配置
        if self.space_capacity.empty or 'eff_from' not in self.space_capacity.columns:
            # 返回结构正确的空 DataFrame
            return pd.DataFrame(columns=['receiving', 'date', 'max_qty'])
            
        effective_capacity = self.space_capacity[
            (self.space_capacity['eff_from'] <= date_obj) &
            (self.space_capacity['eff_to'] >= date_obj)
        ]
        
        records = []
        # 性能优化：使用 itertuples 替代 iterrows
        for capacity_row in effective_capacity.itertuples():
            location = capacity_row.location
            capacity = capacity_row.capacity
            
            # 计算该地点的非限制库存总量
            location_inventory = sum([
                qty for (material, loc), qty in self.unrestricted_inventory.items()
                if loc == location
            ])
            
            # 可用额度 = 容量 - 当前库存
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
    
    def get_production_plan_backlog_view(self, date: str) -> pd.DataFrame:
        """
        获取生产计划 backlog（含未来生产），用于持久化

        Args:
            date: 参考日期（YYYY-MM-DD）

        Returns:
            含列 [material, location, available_date, quantity] 的 DataFrame
        """
        if not self.production_plan_backlog:
            return pd.DataFrame(columns=['material', 'location', 'available_date', 'quantity'])
        
        backlog_df = pd.DataFrame(self.production_plan_backlog)
        if backlog_df.empty:
            return pd.DataFrame(columns=['material', 'location', 'available_date', 'quantity'])
        
        # 确保所有必需列存在
        for col in ['material', 'location', 'available_date', 'quantity']:
            if col not in backlog_df.columns:
                backlog_df[col] = ''
        
        return backlog_df[['material', 'location', 'available_date', 'quantity']]
    
    def get_all_production_view(self, date: str) -> pd.DataFrame:
        date_obj = pd.to_datetime(date).normalize()

        # 当日 GR -> 统一为 available_date 字段
        today_gr = self.get_production_gr_view(date)
        if not today_gr.empty:
            today_gr = today_gr.rename(columns={'date':'available_date'})[['material','location','available_date','quantity']]
        else:
            today_gr = pd.DataFrame(columns=['material','location','available_date','quantity'])

        # backlog 中的未来计划（含当天及以后）
        future = pd.DataFrame(self.production_plan_backlog)
        if not future.empty:
            future['available_date'] = pd.to_datetime(future['available_date']).dt.normalize()
            future = future[future['available_date'] >= date_obj][['material','location','available_date','quantity']]
        else:
            future = pd.DataFrame(columns=['material','location','available_date','quantity'])

        # 🔧 FIX: Filter empty DataFrames before concat to avoid FutureWarning
        dfs_to_concat = [df for df in [today_gr, future] if not df.empty]
        if dfs_to_concat:
            out = pd.concat(dfs_to_concat, ignore_index=True)
        else:
            return pd.DataFrame(columns=['material','location','available_date','quantity'])
        if out.empty:
            return out
        out = out.groupby(['material','location','available_date'], as_index=False).agg({'quantity':'sum'})
        out['quantity'] = out['quantity'].astype(int)
        return out

    def get_production_gr_view(self, date: str) -> pd.DataFrame:
        """
        获取指定日期的生产 GR 记录

        Args:
            date: 日期（YYYY-MM-DD）

        Returns:
            含列 [date, material, location, quantity] 的 DataFrame
        """
        # 🚀 阶段 6：用 O(1) 索引查询替代 O(n) 列表扫描
        date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
        records = self.production_gr_by_date.get(date_str, [])
        
        df = pd.DataFrame(records)
        if df.empty:
            df = pd.DataFrame(columns=['date', 'material', 'location', 'quantity'])
        
        return df
    
    def get_delivery_gr_view(self, date: str) -> pd.DataFrame:
        """
        获取指定日期的交付 GR 记录

        Args:
            date: 日期（YYYY-MM-DD）

        Returns:
            含列 [date, material, receiving, quantity, ori_deployment_uid, vehicle_uid] 的 DataFrame
        """
        # 🚀 阶段 6：用 O(1) 索引查询替代 O(n) 列表扫描
        date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
        records = self.delivery_gr_by_date.get(date_str, [])
        
        df = pd.DataFrame(records)
        if df.empty:
            df = pd.DataFrame(columns=['date', 'material', 'receiving', 'quantity', 'ori_deployment_uid', 'vehicle_uid'])
        
        return df
    
    def get_shipment_log_view(self, date: str) -> pd.DataFrame:
        """
        获取指定日期的发货日志记录

        Args:
            date: 日期（YYYY-MM-DD）

        Returns:
            含列 [date, material, location, quantity] 的 DataFrame
        """
        # 🚀 阶段 6：用 O(1) 索引查询替代 O(n) 列表扫描
        date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
        records = self.shipment_log_by_date.get(date_str, [])
        
        df = pd.DataFrame(records)
        if df.empty:
            df = pd.DataFrame(columns=['date', 'material', 'location', 'quantity'])
        
        return df
    
    def get_delivery_shipment_log_view(self, date: str) -> pd.DataFrame:
        # 🚀 阶段 6：用 O(1) 索引查询替代 O(n) 列表扫描
        date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
        rows = self.delivery_shipment_log_by_date.get(date_str, [])
        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=['date','material','sending','receiving','quantity','ori_deployment_uid','actual_ship_date','actual_delivery_date','type'])
        return df

    def process_module1_shipments(self, shipment_df: pd.DataFrame, date: str):
        """
        处理指定日期的 Module1 发货数据

        Args:
            shipment_df: 含列 [date, material, location, quantity] 的 DataFrame
            date: 仿真日期（YYYY-MM-DD）
        """
        date_obj = pd.to_datetime(date).normalize()
        
        # 筛选当日发货记录
        daily_shipments = shipment_df[
            pd.to_datetime(shipment_df['date']).dt.normalize() == date_obj
        ]
        
        # 更新非限制库存
        # 性能优化：使用 itertuples 替代 iterrows
        for row in daily_shipments.itertuples():
            # 🔧 使用标准化函数确保数据一致性
            key = (_normalize_material(row.material), _normalize_location(row.location))
            if key in self.unrestricted_inventory:
                self.unrestricted_inventory[key] = max(0, self.unrestricted_inventory[key] - int(row.quantity))
            
            # 记录发货日志
            record = {
                'date': date_obj,
                'material': _normalize_material(row.material), # 添加格式化
                'location': _normalize_location(row.location), # 添加格式化
                'quantity': int(row.quantity),
                'type': 'customer_shipment'
            }
            self.shipment_log.append(record)
            # 🚀 阶段 6：加入索引以便 O(1) 查询
            date_str = date_obj.strftime('%Y-%m-%d')
            if date_str not in self.shipment_log_by_date:
                self.shipment_log_by_date[date_str] = []
            self.shipment_log_by_date[date_str].append(record)
        
        if len(daily_shipments) > 0:
            print(f"✅ Processed {len(daily_shipments)} M1 shipments for {date}")
            self._log_event("M1_SHIPMENTS", f"Processed {len(daily_shipments)} shipments")
    
    def process_module4_production(self, production_df: pd.DataFrame, date: str):
        """
        处理指定日期的 Module4 生产数据

        Args:
            production_df: 含列 [available_date, material, location, produced_qty] 的 DataFrame
            date: 仿真日期（YYYY-MM-DD）
        """
        date_obj = pd.to_datetime(date).normalize()
        # === A) 缓存当日GR的生产计划到 backlog 中，供 M3 查询未来生产计划使用 ===
        if production_df is not None and not production_df.empty:
            tmp = production_df.copy()
            # 标准列名：available_date / quantity
            if 'available_date' in tmp.columns:
                tmp['available_date'] = pd.to_datetime(tmp['available_date']).dt.normalize()
            if 'quantity' not in tmp.columns and 'produced_qty' in tmp.columns:
                tmp = tmp.rename(columns={'produced_qty': 'quantity'})
            keep = ['material', 'location', 'available_date', 'quantity']
            tmp = tmp[keep].copy()
            tmp['material'] = tmp['material'].astype(str)
            # 标准化location格式（兼容数字和字母数字混合）
            tmp['location'] = tmp['location'].apply(_normalize_location)
            tmp['quantity'] = tmp['quantity'].fillna(0).astype(int)

            # 新增：用于精确去重的维度
            tmp['simulation_date'] = date_obj
            if 'production_plan_date' in production_df.columns:
                tmp['production_plan_date'] = pd.to_datetime(production_df['production_plan_date']).dt.normalize()
            else:
                # 若未提供生产日期，则回退为可用日期（同日生产与可用）
                tmp['production_plan_date'] = tmp['available_date']
            # 追加到 backlog（两阶段：先5维去重，再3维汇总）
            existing_df = pd.DataFrame(self.production_plan_backlog) if self.production_plan_backlog else pd.DataFrame()
            # 确保旧记录具备新字段
            for col in ['simulation_date', 'production_plan_date']:
                if col not in existing_df.columns:
                    existing_df[col] = pd.NaT

            combined = pd.concat([existing_df, tmp], ignore_index=True)

            # 第一阶段：按 material, location, simulation_date, production_plan_date, available_date 去重
            combined = combined.drop_duplicates(
                subset=['material', 'location', 'simulation_date', 'production_plan_date', 'available_date'],
                keep='first'
            )

            # 第二阶段：按 material, location, available_date 汇总数量
            aggregated = combined.groupby(['material', 'location', 'available_date'], as_index=False).agg({'quantity': 'sum'})
            aggregated['quantity'] = aggregated['quantity'].fillna(0).astype(int)

            self.production_plan_backlog = aggregated.to_dict('records')

        # === B) 原有逻辑：只对“今天到货”的进行 GR 入库 ===
        # 筛选当日生产记录（available_date = 入库日期）
        daily_production = production_df[
            pd.to_datetime(production_df['available_date']).dt.normalize() == date_obj
        ]
        
        # 更新非限制库存并记录生产 GR
        # 性能优化：使用 itertuples 替代 iterrows
        for row in daily_production.itertuples():
            # 🔧 修复：使用标准化的location格式，确保与其他地方一致
            key = (_normalize_material(row.material), _normalize_location(row.location))
            quantity = int(row.produced_qty)
            
            self.unrestricted_inventory[key] = self.unrestricted_inventory.get(key, 0) + quantity
            
            # 记录生产 GR
            record = {
                'date': date_obj,
                'material': _normalize_material(row.material), # 添加格式化
                'location': _normalize_location(row.location), # 添加格式化
                'quantity': quantity
            }
            self.production_gr.append(record)
            # 🚀 阶段 6：加入索引以便 O(1) 查询
            date_str = date_obj.strftime('%Y-%m-%d')
            if date_str not in self.production_gr_by_date:
                self.production_gr_by_date[date_str] = []
            self.production_gr_by_date[date_str].append(record)
        
        if len(daily_production) > 0:
            print(f"✅ Processed {len(daily_production)} M4 production receipts for {date}")
            self._log_event("M4_PRODUCTION", f"Processed {len(daily_production)} production receipts")
    
    def process_module5_deployment(self, deployment_df: pd.DataFrame, date: str):
        """
        处理 Module5 部署计划并更新开放调拨

        Args:
            deployment_df: 含列 [material, sending, receiving, planned_deployment_date,
                                                 deployed_qty, demand_element]
            date: 仿真日期（YYYY-MM-DD）
        """
        date_obj = pd.to_datetime(date).normalize()
        
        # print(f"    🔍 Orchestrator正在处理Module5部署计划: {len(deployment_df)} 条")
        # if len(deployment_df) > 0:
        #     print(f"    📈 部署计划deployed_qty统计: {deployment_df['deployed_qty'].describe()}")
        # 为保证在相同配置和随机种子下 ori_deployment_uid 可复现，
        # 在生成 UID 之前对部署计划做一次稳定排序
        sort_cols = [
            col for col in ['material', 'sending', 'receiving', 'planned_deployment_date', 'demand_element', 'deployed_qty']
            if col in deployment_df.columns
        ]
        if sort_cols:
            deployment_df = deployment_df.sort_values(by=sort_cols, kind='mergesort')
        # 将新的部署计划加入开放调拨
        # 性能优化：使用 itertuples 替代 iterrows
        for row in deployment_df.itertuples():
            # 生成唯一 UID
            self.uid_sequence += 1
            uid_obj = DeploymentUID(
                material=str(row.material),
                sending=str(row.sending),
                receiving=str(row.receiving),
                planned_deploy_date=pd.to_datetime(row.planned_deployment_date).strftime('%Y-%m-%d'),
                demand_element=str(row.demand_element),
                sequence=self.uid_sequence
            )
            uid = uid_obj.to_string()
            
            original_qty = row.deployed_qty
            converted_qty = self._safe_convert_to_int(row.deployed_qty)
            
            # if i < 3:  # 只显示前3条记录的详细信息
                # print(f"      记录{i+1}: original_qty={original_qty} (类型: {type(original_qty)}), converted_qty={converted_qty}")
            
            self.open_deployment[uid] = {
                'material': _normalize_material(row.material), # 添加格式化
                'sending': _normalize_sending(row.sending), # 添加格式化
                'receiving': _normalize_receiving(row.receiving), # 添加格式化
                'planned_deployment_date': pd.to_datetime(row.planned_deployment_date).strftime('%Y-%m-%d'),
                'deployed_qty': converted_qty,
                'demand_element': str(row.demand_element),
                'creation_date': date_obj.strftime('%Y-%m-%d')
            }
        
        if len(deployment_df) > 0:
            print(f"✅ Added {len(deployment_df)} M5 deployment plans to open deployment for {date}")
            # 检查存储后的数量
            stored_qtys = [v['deployed_qty'] for v in self.open_deployment.values()]
            non_zero_qtys = [q for q in stored_qtys if q > 0]
            # print(f"    🔍 存储后的数量统计: 总数={len(stored_qtys)}, 非零数量={len(non_zero_qtys)}")
            self._log_event("M5_DEPLOYMENT", f"Added {len(deployment_df)} deployment plans")
    
    def process_module6_delivery(self, delivery_df: pd.DataFrame, date: str):
        """
        处理 Module6 交付计划并更新状态

        Args:
            delivery_df: 含列 [ori_deployment_uid, material, sending, receiving,
                                               actual_ship_date, actual_delivery_date, delivery_qty]
            date: 仿真日期（YYYY-MM-DD）
        """
        date_obj = pd.to_datetime(date).normalize()
        print(f"[M6->Orch] incoming rows: {len(delivery_df)}; date={date}")
        
        # 添加调试信息：显示输入数据的详细信息
        # if not delivery_df.empty:
        #     # print(f"  📊 M6输入数据预览:")
        #     for idx, row in delivery_df.head(3).iterrows():
        #         # print(f"    Row {idx}: {row['material']}@{row['sending']}->{row['receiving']}, ship:{row['actual_ship_date']}, delivery:{row['actual_delivery_date']}, qty:{row['delivery_qty']}")
        
        # 处理每条交付记录
        # 性能优化：使用 itertuples 替代 iterrows
        for row in delivery_df.itertuples():
            uid = str(row.ori_deployment_uid)
            vehicle_uid = str(row.vehicle_uid)
            material = str(row.material)
            sending = str(row.sending)
            receiving = str(row.receiving)
            
            # 添加调试信息：显示原始和标准化后的标识符
            normalized_material = _normalize_material(material)
            normalized_receiving = _normalize_receiving(receiving)
            # if material == '80813644' and receiving in ['C816', 'C810']:
                # print(f"      🔍 标识符标准化: 原始material='{material}' -> '{normalized_material}', 原始receiving='{receiving}' -> '{normalized_receiving}'")
            ship_date = pd.to_datetime(row.actual_ship_date)
            delivery_date = pd.to_datetime(row.actual_delivery_date)
            quantity = self._safe_convert_to_int(row.delivery_qty)
            
            # 只处理当天发运的货物（actual_ship_date == 当前仿真日期）
            if ship_date.normalize() != date_obj:
                # print(f"    ⏭️  跳过非当天发运: {material}@{sending}->{receiving}, ship_date:{ship_date.date()}, current:{date_obj.date()}")
                continue
            
            # print(f"    ✅ 处理当天发运: {material}@{sending}->{receiving}, ship:{ship_date.date()}, delivery:{delivery_date.date()}, qty:{quantity}")
            
            # 减少开放调拨数量
            if uid in self.open_deployment:
                self.open_deployment[uid]['deployed_qty'] -= quantity
                if self.open_deployment[uid]['deployed_qty'] <= 0:
                    del self.open_deployment[uid]
            
            # 减少发货地非限制库存
            # 🔧 使用标准化函数确保数据一致性
            sending_key = (_normalize_material(material), _normalize_location(sending))
            if sending_key in self.unrestricted_inventory:
                self.unrestricted_inventory[sending_key] = max(0, 
                    self.unrestricted_inventory[sending_key] - quantity)
            
            # 🆕 记录发运出库日志
            shipment_record = {
                'date': date_obj,
                'material': _normalize_material(material), # 添加格式化
                'sending': _normalize_sending(sending), # 添加格式化
                'receiving': _normalize_receiving(receiving), # 添加格式化
                'quantity': quantity,
                'ori_deployment_uid': uid,
                'actual_ship_date': ship_date.strftime('%Y-%m-%d'),
                'actual_delivery_date': delivery_date.strftime('%Y-%m-%d'),
                'type': 'delivery_shipment'
            }
            self.delivery_shipment_log.append(shipment_record)
            # 🚀 阶段 6：加入索引以便 O(1) 查询
            date_str = date_obj.strftime('%Y-%m-%d')
            if date_str not in self.delivery_shipment_log_by_date:
                self.delivery_shipment_log_by_date[date_str] = []
            self.delivery_shipment_log_by_date[date_str].append(shipment_record)
            
            # 判断处理逻辑：基于delivery_date是否为未来日期
            if delivery_date.normalize() > date_obj:
                # 为未来交付创建在途记录
                # 使用 vehicle_uid 确保同一 ori_deployment_uid 的多车记录唯一
                transit_uid = f"{uid}_transit_{vehicle_uid}"
                self.in_transit[transit_uid] = {
                    'material': _normalize_material(material), # 添加格式化
                    'sending': _normalize_sending(sending), # 添加格式化
                    'receiving': _normalize_receiving(receiving), # 添加格式化
                    'actual_ship_date': ship_date.strftime('%Y-%m-%d'),
                    'actual_delivery_date': delivery_date.strftime('%Y-%m-%d'),
                    'quantity': quantity,
                    'ori_deployment_uid': uid,
                    'vehicle_uid': vehicle_uid
                }
            elif delivery_date.normalize() == date_obj:
                # 当天交付：创建 delivery GR 并立即更新库存
                # print(f"      📦 同天到达，创建delivery GR: {material}@{receiving}, qty:{quantity}, uid:{uid}")
                receiving_key = (material, receiving)
                self.unrestricted_inventory[receiving_key] = (
                    self.unrestricted_inventory.get(receiving_key, 0) + quantity)
                
                # 记录 delivery GR（含去重检查）
                gr_record = {
                    'date': date_obj,
                    'material': _normalize_material(material), # 添加格式化
                    'receiving': _normalize_receiving(receiving), # 添加格式化
                    'quantity': quantity,
                    'ori_deployment_uid': uid,
                    'vehicle_uid': vehicle_uid  # 使用vehicle_uid来区分同一deployment的不同车辆
                }
                
                # 基于关键字段检查重复
                # 修复：使用ori_deployment_uid + vehicle_uid作为唯一键，完美支持多车情况
                existing_key = (date_obj, material, receiving, uid, vehicle_uid)
                is_duplicate = any(
                    (record['date'], record['material'], record['receiving'], 
                     record['ori_deployment_uid'], record['vehicle_uid']) == existing_key
                    for record in self.delivery_gr
                )
                
                if not is_duplicate:
                    self.delivery_gr.append(gr_record)
                    # 🚀 阶段 6：加入索引以便 O(1) 查询
                    date_str = date_obj.strftime('%Y-%m-%d')
                    if date_str not in self.delivery_gr_by_date:
                        self.delivery_gr_by_date[date_str] = []
                    self.delivery_gr_by_date[date_str].append(gr_record)
                    # print(f"        ✅ 已添加delivery GR记录: {material}@{receiving}={quantity}")
                    # 特别追踪80813644@C816
            #         if material == '80813644' and receiving == 'C816':
            #             # print(f"        🎯 特别追踪80813644@C816: 当前delivery_gr总数={len(self.delivery_gr)}")
            #     else:
            #         # print(f"        ⚠️  跳过重复的delivery GR记录: {material}@{receiving}={quantity}, uid:{uid}")
            # else:
            #     # 如果delivery_date < date_obj，这是历史数据，应该已经处理过，跳过
            #     # print(f"      ⏭️  跳过历史数据: delivery_date={delivery_date.date()}, current={date_obj.date()}")
        
        if len(delivery_df) > 0:
            print(f"✅ Processed {len(delivery_df)} M6 delivery plans for {date}")
            self._log_event("M6_DELIVERY", f"Processed {len(delivery_df)} delivery plans")
    
    def run_daily_processing(self, date: str,
                            shipment_df: Optional[pd.DataFrame] = None,
                            production_df: Optional[pd.DataFrame] = None,
                            deployment_df: Optional[pd.DataFrame] = None,
                            delivery_df: Optional[pd.DataFrame] = None,
                            grace_days: Optional[int] = None):
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
        # ✅ 仅在每日跑批开头清理一次；grace_days 未传则使用全局 self.cleanup_grace_days
        normalized_date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
        g = self.cleanup_grace_days if grace_days is None else int(grace_days)
        if self._last_cleanup_date != normalized_date_str:
            self.cleanup_past_due_open_deployments(date, grace_days=g, write_audit=True)
            self._last_cleanup_date = normalized_date_str

        # 在每日开始时检查到货
        self._process_delivery_arrivals(date)
        
        # M1：处理发货
        if shipment_df is not None and not shipment_df.empty:
            self.process_module1_shipments(shipment_df, date)
        
        # M4：处理生产
        if production_df is not None and not production_df.empty:
            self.process_module4_production(production_df, date)
        
        # M5：处理部署
        if deployment_df is not None and not deployment_df.empty:
            self.process_module5_deployment(deployment_df, date)
        
        # M6：处理交付
        if delivery_df is not None and not delivery_df.empty:
            self.process_module6_delivery(delivery_df, date)
        
        # 保存每日状态
        self.save_daily_state(date)
        
        print(f"✅ Completed daily processing for {date}")
    
    def _process_delivery_arrivals(self, date: str):
        """
        处理当天到达的在途交付
        """
        date_obj = pd.to_datetime(date).normalize()
        
        completed_transits = []
        for transit_uid, transit_record in self.in_transit.items():
            if pd.to_datetime(transit_record['actual_delivery_date']).normalize() == date_obj:
                # 增加收货地库存
                receiving_key = (transit_record['material'], transit_record['receiving'])
                self.unrestricted_inventory[receiving_key] = (
                    self.unrestricted_inventory.get(receiving_key, 0) + transit_record['quantity'])
                
                # 记录 delivery GR（改进的去重检查）
                gr_record = {
                    'date': date_obj,
                    'material': _normalize_material(transit_record['material']), # 添加格式化
                    'receiving': _normalize_receiving(transit_record['receiving']), # 添加格式化
                    'quantity': transit_record['quantity'],
                    'ori_deployment_uid': transit_record['ori_deployment_uid'],
                    'vehicle_uid': transit_record['vehicle_uid'],
                    'actual_ship_date': transit_record['actual_ship_date']  # 新增字段
                }
                
                # 改进的重复检查：使用ori_deployment_uid + vehicle_uid作为唯一键
                existing_key = (date_obj, transit_record['material'], transit_record['receiving'], 
                              transit_record['ori_deployment_uid'], transit_record['vehicle_uid'])
                if not any(
                    (record['date'], record['material'], record['receiving'], 
                     record['ori_deployment_uid'], record['vehicle_uid']) == existing_key
                    for record in self.delivery_gr
                ):
                    self.delivery_gr.append(gr_record)
                    # 🚀 阶段 6：加入索引以便 O(1) 查询
                    date_str = date_obj.strftime('%Y-%m-%d')
                    if date_str not in self.delivery_gr_by_date:
                        self.delivery_gr_by_date[date_str] = []
                    self.delivery_gr_by_date[date_str].append(gr_record)
                
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
    def set_past_due_cleanup_grace_days(self, days: int):
        """
        设置 open deployment 过期清理的全局宽限天数（默认0）。
        之后每次 get_open_deployment_view() 都会按该值清理并落审计文件。
        """
        try:
            self.cleanup_grace_days = max(0, int(days))
        except Exception:
            self.cleanup_grace_days = 0

    def cleanup_past_due_open_deployments(self, date: str, grace_days: int = 0, write_audit: bool = True) -> pd.DataFrame:
        """
        清理过期的 open deployment，并输出审计文件
        规则：planned_deployment_date < (date - grace_days) 的记录会被清理

        Args:
            date: 当前仿真日期 YYYY-MM-DD
            grace_days: 宽限天数（允许延迟不清理）
            write_audit: 是否写入审计CSV

        Returns:
            DataFrame: 被清理掉的记录明细（用于链路追溯）
                    列: [cleanup_date, grace_days, ori_deployment_uid, material, sending, receiving,
                        planned_deployment_date, remaining_qty, demand_element, creation_date, reason]
        """
        cleanup_date = pd.to_datetime(date).normalize()
        threshold_date = cleanup_date - pd.Timedelta(days=int(grace_days))

        removed = []
        # 注意：遍历时不要直接修改字典，先收集再删除
        to_delete = []

        for uid, rec in self.open_deployment.items():
            pdd = pd.to_datetime(rec.get('planned_deployment_date')).normalize()
            remaining_qty = int(rec.get('deployed_qty', 0))
            # 只清理：计划日早于阈值（严格小于）
            if pdd < threshold_date:
                to_delete.append(uid)
                removed.append({
                    'cleanup_date': cleanup_date,
                    'grace_days': int(grace_days),
                    'ori_deployment_uid': uid,
                    'material': _normalize_material(rec.get('material')),
                    'sending': _normalize_sending(rec.get('sending')),
                    'receiving': _normalize_receiving(rec.get('receiving')),
                    'planned_deployment_date': pdd,
                    'remaining_qty': remaining_qty,
                    'demand_element': rec.get('demand_element', ''),
                    'creation_date': rec.get('creation_date', ''),
                    'reason': f"past_due>{int(grace_days)}d"
                })

        # 真正删除
        for uid in to_delete:
            del self.open_deployment[uid]

        # 生成审计DF（即使为空也输出表头，便于留痕）
        cleanup_df = pd.DataFrame(removed)
        if cleanup_df.empty:
            cleanup_df = pd.DataFrame(columns=[
                'cleanup_date', 'grace_days', 'ori_deployment_uid', 'material', 'sending', 'receiving',
                'planned_deployment_date', 'remaining_qty', 'demand_element', 'creation_date', 'reason'
            ])

        # 写审计CSV
        if write_audit:
            date_str = cleanup_date.strftime('%Y%m%d')
            out_path = self.output_dir / f"open_deployment_pastdue_cleanup_{date_str}.csv"
            _normalize_identifiers(cleanup_df).to_csv(out_path, index=False)

        # 记录日志
        self._log_event(
            "OPEN_DEPLOYMENT_CLEANUP",
            f"Removed {len(to_delete)} past-due open deployments (grace_days={grace_days})"
        )

        return cleanup_df
    
    def save_daily_state(self, date: str):
        """
        将每日状态保存到持久化存储

        Args:
            date: 日期（YYYY-MM-DD）
        """
        date_str = pd.to_datetime(date).strftime('%Y%m%d')
        
        # 保存非限制库存视图
        unrestricted_df = self.get_unrestricted_inventory_view(date)
        _normalize_identifiers(unrestricted_df).to_csv(self.output_dir / f"unrestricted_inventory_{date_str}.csv", index=False)
        
        # 保存开放调拨视图
        open_deployment_df = self.get_open_deployment_view(date)
        _normalize_identifiers(open_deployment_df).to_csv(self.output_dir / f"open_deployment_{date_str}.csv", index=False)
        
        # 保存在途视图
        intransit_df = self.get_planning_intransit_view(date)
        _normalize_identifiers(intransit_df).to_csv(self.output_dir / f"planning_intransit_{date_str}.csv", index=False)
        
        # 保存空间额度视图
        space_quota_df = self.get_space_quota_view(date)
        _normalize_identifiers(space_quota_df).to_csv(self.output_dir / f"space_quota_{date_str}.csv", index=False)
        
        # 保存生产计划 backlog（含未来生产）
        production_backlog_df = self.get_production_plan_backlog_view(date)
        _normalize_identifiers(production_backlog_df).to_csv(self.output_dir / f"production_plan_backlog_{date_str}.csv", index=False)
        
        # 保存每日交付 GR
        delivery_gr_df = self.get_delivery_gr_view(date)
        _normalize_identifiers(delivery_gr_df).to_csv(self.output_dir / f"delivery_gr_{date_str}.csv", index=False)
        
        # 保存每日生产 GR
        production_gr_df = self.get_production_gr_view(date)
        _normalize_identifiers(production_gr_df).to_csv(self.output_dir / f"production_gr_{date_str}.csv", index=False)
        
        # 保存每日发货日志
        # 🚀 阶段 6：用 O(1) 索引查询替代 O(n) 列表扫描
        date_key = pd.to_datetime(date).strftime('%Y-%m-%d')
        daily_shipments = self.shipment_log_by_date.get(date_key, [])
        shipment_df = pd.DataFrame(daily_shipments)
        if shipment_df.empty:
            shipment_df = pd.DataFrame(columns=['date', 'material', 'location', 'quantity'])
        _normalize_identifiers(shipment_df).to_csv(self.output_dir / f"shipment_log_{date_str}.csv", index=False)
        
        # 🆕 保存发运出库日志
        # 🚀 阶段 6：用 O(1) 索引查询替代 O(n) 列表扫描
        daily_delivery_shipments = self.delivery_shipment_log_by_date.get(date_key, [])
        delivery_shipment_df = pd.DataFrame(daily_delivery_shipments)
        if delivery_shipment_df.empty:
            delivery_shipment_df = pd.DataFrame(columns=['date', 'material', 'sending', 'receiving', 'quantity', 
                                                       'ori_deployment_uid', 'actual_ship_date', 'actual_delivery_date', 'type'])
        _normalize_identifiers(delivery_shipment_df).to_csv(self.output_dir / f"delivery_shipment_log_{date_str}.csv", index=False)
        
        # 🆕 生成库存变动日志
        inventory_change_df = self.generate_inventory_change_log(date)
        _normalize_identifiers(inventory_change_df).to_csv(self.output_dir / f"inventory_change_log_{date_str}.csv", index=False)
        # print(f"  📊 已生成库存变动日志: {len(inventory_change_df)} 条记录")
        
        # 保存每日日志（改为无论是否有事件都输出文件，含表头）
        logs_file = self.output_dir / f"daily_logs_{date_str}.csv"
        if self.daily_logs:
            logs_df = pd.DataFrame(self.daily_logs)
        else:
            # 保证列头一致
            logs_df = pd.DataFrame(columns=['timestamp', 'date', 'event_type', 'message'])
        logs_df.to_csv(logs_file, index=False)
    
    def _log_event(self, event_type: str, message: str):
        """
        记录编排器事件用于审计追踪

        Args:
            event_type: 事件类型
            message: 事件消息
        """
        self.daily_logs.append({
            'timestamp': datetime.now().isoformat(),
            'date': self.current_date.strftime('%Y-%m-%d'),
            'event_type': event_type,
            'message': message
        })
    
    def get_summary_statistics(self, date: str) -> Dict:
        """
        获取指定日期的汇总统计

        Args:
            date: 日期（YYYY-MM-DD）

        Returns:
            汇总统计字典
        """
        # 🚀 阶段 6：用 O(1) 索引查询替代 O(n) 列表扫描
        date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
        return {
            'date': date,
            'total_inventory_items': len(self.unrestricted_inventory),
            'total_inventory_quantity': sum(self.unrestricted_inventory.values()),
            'open_deployment_count': len(self.open_deployment),
            'in_transit_count': len(self.in_transit),
            'production_gr_count': len(self.production_gr_by_date.get(date_str, [])),
            'delivery_gr_count': len(self.delivery_gr_by_date.get(date_str, [])),
            'shipment_count': len(self.shipment_log_by_date.get(date_str, []))
        }
    
    def save_beginning_inventory(self, date: str):
        """
        保存指定日期的期初库存状态（在任何库存变动之前调用）
        
        Args:
            date: 日期字符串 (YYYY-MM-DD)
        """
        self.daily_beginning_inventory[date] = self.unrestricted_inventory.copy()
        print(f"💾已保存 {date} 期初库存: {len(self.unrestricted_inventory)} 项")
    
    def save_ending_inventory(self, date: str):
        """
        保存指定日期的期末库存状态（在所有模块运行完成后调用）
        
        Args:
            date: 日期字符串 (YYYY-MM-DD)
        """
        self.daily_ending_inventory[date] = self.unrestricted_inventory.copy()
        print(f"💾已保存 {date} 期末库存: {len(self.unrestricted_inventory)} 项")
    
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
                'material': _normalize_material(material),  # 添加格式化
                'location': _normalize_location(location),  # 添加格式化
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
        # 🚀 阶段 6：用 O(1) 索引查询替代 O(n) 列表扫描
        date_str = date_obj.strftime('%Y-%m-%d')
        
        for record in self.production_gr_by_date.get(date_str, []):
            all_keys.add((record['material'], record['location']))
        
        for record in self.delivery_gr_by_date.get(date_str, []):
            all_keys.add((record['material'], record['receiving']))
        
        for record in self.shipment_log_by_date.get(date_str, []):
            all_keys.add((record['material'], record['location']))
        
        # 🔧 修复：直接从内存的delivery_shipment_log获取发运出库数据
        delivery_ship_data = {}
        for record in self.delivery_shipment_log_by_date.get(date_str, []):
            material = record['material']
            sending = record['sending']
            quantity = float(record['quantity'])
            
            key = (material, sending)
            delivery_ship_data[key] = delivery_ship_data.get(key, 0) + quantity
            all_keys.add(key)
        
        print(f"  📊 从内存获取发运出库 [{date}]: {len(delivery_ship_data)} 项")
        
        # 调试信息：显示delivery_gr中相关记录的详细信息
        # print(f"  📊 当前delivery_gr中共有 {len(self.delivery_gr)} 条记录")
        # relevant_gr_records = [
        #     record for record in self.delivery_gr
        #     if (pd.to_datetime(record['date']).normalize() == date_obj and 
        #         record['material'] == '80813644' and record['receiving'] in ['C816', 'C810'])
        # ]
        # if relevant_gr_records:
        #     print(f"  🔍 找到 {len(relevant_gr_records)} 条80813644的delivery_gr记录:")
        #     for i, rec in enumerate(relevant_gr_records):
        #         print(f"    记录{i+1}: material='{rec['material']}', receiving='{rec['receiving']}', qty={rec['quantity']}, uid={rec.get('ori_deployment_uid', 'N/A')}")
        
        change_log = []
        
        for material, location in all_keys:
            # 期初库存
            beginning_qty = 0
            if date in self.daily_beginning_inventory:
                beginning_qty = self.daily_beginning_inventory[date].get((material, location), 0)
            
            # 生产入库
            # 🚀 阶段 6：用 O(1) 索引查询替代 O(n) 列表扫描
            production_qty = sum(
                record['quantity'] for record in self.production_gr_by_date.get(date_str, [])
                if record['material'] == material and record['location'] == location
            )
            
            # 交付入库
            delivery_qty = sum(
                record['quantity'] for record in self.delivery_gr_by_date.get(date_str, [])
                if record['material'] == material and record['receiving'] == location
            )
            
            # 调试信息：显示delivery_gr匹配情况
            # if material == '80813644' and location in ['C816', 'C810']:
            #     matching_records = [
            #         record for record in self.delivery_gr
            #         if (pd.to_datetime(record['date']).normalize() == date_obj and 
            #             record['material'] == material and record['receiving'] == location)
            #     ]
            #     print(f"  🔍 调试 {material}@{location}: 找到 {len(matching_records)} 条delivery_gr记录, 总量={delivery_qty}")
            #     for i, rec in enumerate(matching_records):
            #         print(f"    记录{i+1}: uid={rec.get('ori_deployment_uid', 'N/A')}, qty={rec['quantity']}, date={rec['date']}")
            
            # 发货出库
            # 🚀 阶段 6：用 O(1) 索引查询替代 O(n) 列表扫描
            shipment_qty = sum(
                record['quantity'] for record in self.shipment_log_by_date.get(date_str, [])
                if record['material'] == material and record['location'] == location
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
        # print(f"\n📊 === Orchestrator每日库存变动详情 [{date}] ===")
        
        # 获取期初期末库存
        beginning_inv = self.daily_beginning_inventory.get(date, {})
        ending_inv = self.daily_ending_inventory.get(date, {})
        
        # 🚀 阶段 6：用 O(1) 索引查询替代 O(n) 列表扫描
        date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
        
        # 获取当日各项变动
        production_gr = self.production_gr_by_date.get(date_str, [])
        delivery_gr = self.delivery_gr_by_date.get(date_str, [])
        shipments = self.shipment_log_by_date.get(date_str, [])
        
        # M6 发运（当日实际发运）
        m6_ship_df = self.get_delivery_shipment_log_view(date)
        m6_ship_count = len(m6_ship_df)
        m6_ship_qty_total = int(m6_ship_df['quantity'].sum()) if not m6_ship_df.empty else 0
        
        # 统计汇总
        print(f"期初库存条目: {len(beginning_inv)}")
        print(f"生产入库条目: {len(production_gr)}")
        print(f"交付入库条目: {len(delivery_gr)}")
        print(f"发货出库条目: {len(shipments)}")
        print(f"发运出库条目(M6): {m6_ship_count}，数量合计: {m6_ship_qty_total}")
        print(f"期末库存条目: {len(ending_inv)}")
        
        # 重点分析MAT_B@DC_001
        # key = ('MAT_B', 'DC_001')
        # material, location = key
        
        # begin_qty = beginning_inv.get(key, 0)
        # end_qty = ending_inv.get(key, 0)
        
        # print(f"\n=== 重点分析: {material}@{location} ===")
        # print(f"期初库存: {begin_qty}")
        
        # # Production GR
        # prod_qty = sum(gr['quantity'] for gr in production_gr 
        #               if gr['material'] == material and gr['location'] == location)
        # print(f"生产入库: +{prod_qty}")
        
        # # Delivery GR
        # del_qty = sum(gr['quantity'] for gr in delivery_gr 
        #              if gr['material'] == material and gr['receiving'] == location)
        # print(f"交付入库: +{del_qty}")
        
        # # Shipment
        # ship_qty = sum(ship['quantity'] for ship in shipments 
        #               if ship['material'] == material and ship['location'] == location)
        # print(f"发货出库: -{ship_qty}")
        
        # # 发运出库按 M6 发运日志统计
        # transit_qty = 0
        # if not m6_ship_df.empty:
        #     mask = (m6_ship_df['material'] == material) & (m6_ship_df['sending'] == location)
        #     transit_qty = int(m6_ship_df.loc[mask, 'quantity'].sum())
        # print(f"发运出库(M6): -{transit_qty}")
        
        # print(f"期末库存: {end_qty}")
        
        # # 计算期望值
        # expected = begin_qty + prod_qty + del_qty - ship_qty - transit_qty
        # print(f"计算期望: {begin_qty} + {prod_qty} + {del_qty} - {ship_qty} - {transit_qty} = {expected}")
        
        # if expected != end_qty:
        #     print(f"⚠️  差异: 期望{expected}, 实际{end_qty}, 差异{end_qty - expected}")
        # else:
        #     print(f"✅ 一致")

# 用于模块集成的便捷函数
def create_orchestrator(start_date: str, output_dir: str = "./orchestrator_output") -> Orchestrator:
    """
    创建并初始化编排器实例

    Args:
        start_date: 仿真开始日期（YYYY-MM-DD）
        output_dir: 持久化存储输出目录

    Returns:
        Orchestrator 实例
    """
    return Orchestrator(start_date, output_dir)

# 示例用法与测试
if __name__ == "__main__":
    # 示例初始化
    orchestrator = create_orchestrator("2024-01-01")
    
    # 示例期初库存
    initial_inventory = pd.DataFrame([
        {'material': 'MAT_A', 'location': 'PLANT_001', 'quantity': 1000},
        {'material': 'MAT_B', 'location': 'DC_001', 'quantity': 500}
    ])
    orchestrator.initialize_inventory(initial_inventory)
    
    # 示例空间容量
    space_capacity = pd.DataFrame([
        {'location': 'DC_001', 'eff_from': '2024-01-01', 'eff_to': '2024-12-31', 'capacity': 2000}
    ])
    orchestrator.set_space_capacity(space_capacity)
    
    # 视图测试
    inventory_view = orchestrator.get_unrestricted_inventory_view("2024-01-01")
    space_quota = orchestrator.get_space_quota_view("2024-01-01")
    
    # print("\n📊 Initial State:")
    # print(f"Inventory items: {len(inventory_view)}")
    # print(f"Space quota available: {space_quota['max_qty'].sum() if not space_quota.empty else 0}")
    
    # stats = orchestrator.get_summary_statistics("2024-01-01")
    # print(f"Summary: {stats}")