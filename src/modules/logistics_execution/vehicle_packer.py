# -*- coding: utf-8 -*-
"""
车辆装载优化模块

提供 Module6 的车辆装载优化功能，包括：
- 车辆装载器类
- 装载记录创建
- 装载比例计算

Typical usage example:
    packer = VehiclePacker(cap_weight=1000, cap_volume=100)
    packer.add_demand(demand_row, available_qty, inventory_limit)
    wfr, vfr = packer.get_load_ratios()
"""

from dataclasses import dataclass, field
from math import floor
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class LoadRecord:
    """装载记录数据类。"""
    idx: int
    load_qty: int
    demand_row: pd.Series


@dataclass
class VehiclePacker:
    """
    车辆装载优化器类。
    
    负责单辆车的装载优化，确保不超过容量限制，
    并支持库存检查和部分装载。
    
    Attributes:
        cap_weight: 车辆重量容量
        cap_volume: 车辆体积容量
        current_weight: 当前已装载重量
        current_volume: 当前已装载体积
        current_units: 当前已装载单位数
        load_records: 装载记录列表
        material_loaded: 各物料已装载数量
    """
    cap_weight: float
    cap_volume: float
    current_weight: float = 0.0
    current_volume: float = 0.0
    current_units: float = 0.0
    load_records: List[Dict] = field(default_factory=list)
    material_loaded: Dict[str, float] = field(default_factory=dict)
    
    def add_demand(
        self,
        idx: int,
        demand_row: pd.Series,
        inventory_limit: Optional[float] = None
    ) -> int:
        """
        尝试将需求添加到车辆。
        
        Args:
            idx: 需求行索引
            demand_row: 需求行数据
            inventory_limit: 库存限制（可选）
            
        Returns:
            实际装载数量
        """
        qty_pending = float(demand_row['deployed_qty'])
        if qty_pending <= 0:
            return 0
        
        uw = float(demand_row['demand_unit_to_weight'])
        uv = float(demand_row['demand_unit_to_volume'])
        material = demand_row['material']
        
        addable = self._calculate_addable_qty(
            qty_pending, uw, uv, material, inventory_limit
        )
        
        if addable <= 0:
            return 0
        
        self._update_load_state(addable, uw, uv, material)
        self.load_records.append({
            'idx': idx,
            'load_qty': addable,
            'demand_row': demand_row
        })
        
        return addable
    
    def _calculate_addable_qty(
        self,
        qty_pending: float,
        unit_weight: float,
        unit_volume: float,
        material: str,
        inventory_limit: Optional[float]
    ) -> int:
        """
        计算可装载数量。
        
        Args:
            qty_pending: 待装载数量
            unit_weight: 单位重量
            unit_volume: 单位体积
            material: 物料
            inventory_limit: 库存限制
            
        Returns:
            可装载数量
        """
        cap_w_rem = max(0.0, self.cap_weight - self.current_weight)
        cap_v_rem = max(0.0, self.cap_volume - self.current_volume)
        
        limits = [qty_pending]
        
        if unit_weight > 0:
            limits.append(floor(cap_w_rem / unit_weight))
        if unit_volume > 0:
            limits.append(floor(cap_v_rem / unit_volume))
        if inventory_limit is not None:
            limits.append(inventory_limit)
        
        return int(max(0, min(limits)))
    
    def _update_load_state(
        self,
        qty: int,
        unit_weight: float,
        unit_volume: float,
        material: str
    ) -> None:
        """
        更新装载状态。
        
        Args:
            qty: 装载数量
            unit_weight: 单位重量
            unit_volume: 单位体积
            material: 物料
        """
        self.current_units += qty
        self.current_weight += qty * unit_weight
        self.current_volume += qty * unit_volume
        self.material_loaded[material] = (
            self.material_loaded.get(material, 0) + qty
        )
    
    def get_load_ratios(self) -> Tuple[float, float]:
        """
        获取装载比例。
        
        Returns:
            (重量填充率, 体积填充率) 元组
        """
        wfr = (self.current_weight / self.cap_weight) if self.cap_weight > 0 else 0.0
        vfr = (self.current_volume / self.cap_volume) if self.cap_volume > 0 else 0.0
        return wfr, vfr
    
    def is_full(self) -> bool:
        """
        检查车辆是否已满。
        
        Returns:
            是否已满（任一维度达到容量）
        """
        return (self.current_weight >= self.cap_weight or 
                self.current_volume >= self.cap_volume)
    
    def has_load(self) -> bool:
        """
        检查是否有装载内容。
        
        Returns:
            是否有装载
        """
        return len(self.load_records) > 0
    
    def get_loaded_indices(self) -> set:
        """
        获取已装载的需求索引集合。
        
        Returns:
            索引集合
        """
        return {r['idx'] for r in self.load_records}
    
    def get_material_loaded(self, material: str) -> float:
        """
        获取指定物料已装载数量。
        
        Args:
            material: 物料名称
            
        Returns:
            已装载数量
        """
        return self.material_loaded.get(material, 0)
    
    def reset(self) -> None:
        """重置装载状态。"""
        self.current_weight = 0.0
        self.current_volume = 0.0
        self.current_units = 0.0
        self.load_records = []
        self.material_loaded = {}


def create_load_record(
    idx: int,
    load_qty: int,
    demand_row: pd.Series
) -> Dict[str, Any]:
    """
    创建装载记录字典。
    
    Args:
        idx: 需求行索引
        load_qty: 装载数量
        demand_row: 需求行数据
        
    Returns:
        装载记录字典
    """
    return {
        'idx': idx,
        'load_qty': load_qty,
        'demand_row': demand_row
    }


def calculate_load_ratios(
    weight_sum: float,
    volume_sum: float,
    cap_weight: float,
    cap_volume: float
) -> Tuple[float, float]:
    """
    计算装载比例。
    
    Args:
        weight_sum: 总重量
        volume_sum: 总体积
        cap_weight: 重量容量
        cap_volume: 体积容量
        
    Returns:
        (重量填充率, 体积填充率) 元组
    """
    wfr = (weight_sum / cap_weight) if cap_weight > 0 else 0.0
    vfr = (volume_sum / cap_volume) if cap_volume > 0 else 0.0
    return wfr, vfr


def calculate_addable_quantity(
    qty_pending: float,
    unit_weight: float,
    unit_volume: float,
    cap_weight_remaining: float,
    cap_volume_remaining: float,
    inventory_limit: Optional[float] = None
) -> int:
    """
    计算可添加的数量。
    
    Args:
        qty_pending: 待装载数量
        unit_weight: 单位重量
        unit_volume: 单位体积
        cap_weight_remaining: 剩余重量容量
        cap_volume_remaining: 剩余体积容量
        inventory_limit: 库存限制（可选）
        
    Returns:
        可装载数量
    """
    limits = [qty_pending]
    
    if unit_weight > 0:
        limits.append(floor(cap_weight_remaining / unit_weight))
    if unit_volume > 0:
        limits.append(floor(cap_volume_remaining / unit_volume))
    if inventory_limit is not None:
        limits.append(inventory_limit)
    
    return int(max(0, min(limits)))


def create_vehicle_log_entry(
    sim_date: pd.Timestamp,
    sending: str,
    receiving: str,
    truck_type: str,
    vehicle_no: int,
    packer: VehiclePacker,
    trigger_cause: str
) -> Dict[str, Any]:
    """
    创建车辆日志条目。
    
    Args:
        sim_date: 仿真日期
        sending: 发送地点
        receiving: 接收地点
        truck_type: 车型
        vehicle_no: 车辆序号
        packer: 装载器实例
        trigger_cause: 触发原因
        
    Returns:
        车辆日志字典
    """
    wfr, vfr = packer.get_load_ratios()
    vehicle_uid = f"{sim_date:%Y%m%d}-{sending}-{receiving}-{truck_type}-#{vehicle_no}"
    
    return {
        'date': sim_date,
        'sending': sending,
        'receiving': receiving,
        'truck_type': truck_type,
        'vehicle_no': vehicle_no,
        'vehicle_uid': vehicle_uid,
        'total_units': int(packer.current_units),
        'total_weight': packer.current_weight,
        'total_volume': packer.current_volume,
        'WFR': min(wfr, 1.0),
        'VFR': min(vfr, 1.0),
        'trigger': trigger_cause
    }


def get_representative_context(
    load_records: List[Dict]
) -> Tuple[Optional[str], int]:
    """
    获取代表性上下文（优先级最高的需求）。
    
    Args:
        load_records: 装载记录列表
        
    Returns:
        (需求类型, 等待天数) 元组
    """
    if not load_records:
        return None, 0
    
    highest_record = min(
        load_records,
        key=lambda r: (
            r['demand_row']['priority'],
            r['demand_row']['planned_deployment_date']
        )
    )
    
    return (
        highest_record['demand_row']['demand_element'],
        highest_record['demand_row']['waiting_days']
    )


def determine_trigger_cause(
    has_load: bool,
    wfr: float,
    vfr: float,
    wfr_threshold: float,
    vfr_threshold: float,
    bypass: bool,
    max_wait_in_load: int,
    max_wait_days: int
) -> Optional[str]:
    """
    确定发运触发原因。
    
    Args:
        has_load: 是否有装载
        wfr: 重量填充率
        vfr: 体积填充率
        wfr_threshold: 重量阈值
        vfr_threshold: 体积阈值
        bypass: 是否命中旁路规则
        max_wait_in_load: 当前装载中的最大等待天数
        max_wait_days: 最大等待天数限制
        
    Returns:
        触发原因字符串，未触发返回 None
    """
    if not has_load:
        return None
    
    if wfr >= wfr_threshold or vfr >= vfr_threshold:
        return 'threshold'
    
    if bypass:
        return 'bypass'
    
    if max_wait_in_load >= max_wait_days:
        return 'force_wait_timeout'
    
    return None
