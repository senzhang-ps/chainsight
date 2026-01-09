# -*- coding: utf-8 -*-
"""
库存管理模块

提供 Module6 的库存管理功能，包括：
- 实物库存计算
- 装载后库存更新
- 库存可用性检查

Typical usage example:
    inventory = calculate_physical_inventory(orchestrator, current_date)
    inventory = update_inventory_after_load(inventory, material, location, qty)
"""

from typing import Any, Dict, Tuple

import pandas as pd


def calculate_physical_inventory(
    orchestrator: object,
    current_date: pd.Timestamp
) -> Dict[Tuple[str, str], float]:
    """
    获取指定日期的实物库存。
    
    直接使用 Orchestrator 在 M1, M4, M5 执行后的最新实物库存状态。
    
    Args:
        orchestrator: Orchestrator 实例
        current_date: 当前日期
        
    Returns:
        实物库存字典 {(material, location): physical_quantity}
    """
    try:
        physical_inventory = _extract_inventory_from_orchestrator(orchestrator)
        _check_inventory_duplicates(orchestrator.unrestricted_inventory)
        _log_inventory_statistics(physical_inventory)
        return physical_inventory
    except Exception as e:
        print(f"  ⚠️  获取实物库存失败: {e}")
        return {}


def _extract_inventory_from_orchestrator(
    orchestrator: object
) -> Dict[Tuple[str, str], float]:
    """
    从 Orchestrator 提取库存数据。
    
    Args:
        orchestrator: Orchestrator 实例
        
    Returns:
        库存字典
    """
    physical_inventory = {}
    
    for key, qty in orchestrator.unrestricted_inventory.items():
        material, location = key
        physical_inventory[key] = float(qty)
    
    return physical_inventory


def _check_inventory_duplicates(
    inventory: Dict[Tuple[str, str], Any]
) -> None:
    """
    检查库存数据中的重复键。
    
    Args:
        inventory: 库存字典
    """
    location_counts = {}
    
    for key, qty in inventory.items():
        material, location = key
        location_key = f"{material}@{location}"
        
        if location_key in location_counts:
            location_counts[location_key] += 1
            print(f"    ⚠️  发现重复键: {location_key} "
                  f"(第{location_counts[location_key]}次)")
        else:
            location_counts[location_key] = 1
    
    duplicates = {k: v for k, v in location_counts.items() if v > 1}
    if duplicates:
        print(f"    🚨 重复的material-location组合: {len(duplicates)} 个")


def _log_inventory_statistics(
    physical_inventory: Dict[Tuple[str, str], float]
) -> None:
    """
    记录库存统计信息。
    
    Args:
        physical_inventory: 库存字典
    """
    if not physical_inventory:
        return
    
    total_items = sum(1 for qty in physical_inventory.values() if qty > 0)
    positive_qty = sum(qty for qty in physical_inventory.values() if qty > 0)
    
    # 调试信息（可根据需要启用）
    # print(f"  📊 实物库存统计: {len(physical_inventory)} 个SKU-地点组合")
    # print(f"  ✅ 有库存SKU: {total_items}/{len(physical_inventory)}, 总量: {positive_qty:.1f}")


def update_inventory_after_load(
    inventory: Dict[Tuple[str, str], float],
    material: str,
    location: str,
    load_qty: float
) -> Dict[Tuple[str, str], float]:
    """
    装载后更新可用库存。
    
    Args:
        inventory: 当前库存字典
        material: 物料
        location: 地点
        load_qty: 装载数量
        
    Returns:
        更新后的库存字典
    """
    inv_key = (material, location)
    
    if inv_key in inventory:
        inventory[inv_key] -= load_qty
        inventory[inv_key] = max(0, inventory[inv_key])
    
    return inventory


def get_available_inventory(
    inventory: Dict[Tuple[str, str], float],
    material: str,
    location: str
) -> float:
    """
    获取可用库存数量。
    
    Args:
        inventory: 库存字典
        material: 物料
        location: 地点
        
    Returns:
        可用库存数量
    """
    return inventory.get((material, location), 0)


def calculate_inventory_limit(
    inventory: Dict[Tuple[str, str], float],
    material: str,
    location: str,
    already_loaded: float
) -> float:
    """
    计算库存限制（考虑已装载量）。
    
    Args:
        inventory: 库存字典
        material: 物料
        location: 地点
        already_loaded: 该物料已装载的数量
        
    Returns:
        可用于装载的剩余数量
    """
    available = get_available_inventory(inventory, material, location)
    return max(0, available - already_loaded)


def has_sufficient_inventory(
    inventory: Dict[Tuple[str, str], float],
    material: str,
    location: str,
    required_qty: float
) -> bool:
    """
    检查是否有足够的库存。
    
    Args:
        inventory: 库存字典
        material: 物料
        location: 地点
        required_qty: 需求数量
        
    Returns:
        是否有足够库存
    """
    available = get_available_inventory(inventory, material, location)
    return available >= required_qty


def batch_update_inventory(
    inventory: Dict[Tuple[str, str], float],
    load_records: list
) -> Dict[Tuple[str, str], float]:
    """
    批量更新库存。
    
    Args:
        inventory: 当前库存字典
        load_records: 装载记录列表，每条包含 material, sending, load_qty
        
    Returns:
        更新后的库存字典
    """
    for record in load_records:
        material = record['demand_row']['material']
        location = record['demand_row']['sending']
        load_qty = record['load_qty']
        
        inventory = update_inventory_after_load(
            inventory, material, location, load_qty
        )
    
    return inventory


def get_inventory_summary(
    inventory: Dict[Tuple[str, str], float]
) -> Dict[str, Any]:
    """
    获取库存摘要信息。
    
    Args:
        inventory: 库存字典
        
    Returns:
        包含统计信息的字典
    """
    if not inventory:
        return {
            'total_items': 0,
            'items_with_stock': 0,
            'total_quantity': 0,
            'locations': 0,
            'materials': 0
        }
    
    locations = set()
    materials = set()
    
    for (material, location), qty in inventory.items():
        materials.add(material)
        locations.add(location)
    
    return {
        'total_items': len(inventory),
        'items_with_stock': sum(1 for qty in inventory.values() if qty > 0),
        'total_quantity': sum(max(0, qty) for qty in inventory.values()),
        'locations': len(locations),
        'materials': len(materials)
    }
