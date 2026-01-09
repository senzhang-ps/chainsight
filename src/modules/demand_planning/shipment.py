"""Module 1 发货与缺货计算。

本模块提供发货和缺货的计算功能。

主要函数：
- simulate_shipment_for_single_day: 计算单日发货与缺货
- generate_shipment_with_inventory_check: 基于库存生成发货
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .normalization import (
    normalize_identifiers,
    normalize_location,
    normalize_material,
)


def simulate_shipment_for_single_day(
    simulation_date: pd.Timestamp,
    order_log: pd.DataFrame,
    current_inventory: Dict[Tuple[str, str], int],
    material_list: list,
    location_list: list,
    production_plan: Optional[pd.DataFrame] = None,
    delivery_plan: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[Tuple[str, str], int]]:
    """计算单日的发货（shipment）与缺货（cut）。

    处理逻辑:
        1. 当日订单在ML粒度聚合为qty_ordered
        2. 与库存合并得到qty_avail
        3. 发货量为二者最小值，cut为差值

    参数:
        simulation_date: 仿真日期。
        order_log: 订单日志DataFrame。
        current_inventory: 当前库存字典{(material, location): quantity}。
        material_list: 物料列表（用于参考）。
        location_list: 地点列表（用于参考）。
        production_plan: 当天生产计划（未使用）。
        delivery_plan: 当天调运计划（未使用）。

    返回:
        (shipment_df, cut_df, current_inventory)元组。
    """
    # 构建库存DataFrame
    inv_df = pd.DataFrame([
        {'material': k[0], 'location': k[1], 'qty_avail': v}
        for k, v in current_inventory.items()
    ])
    if inv_df.empty:
        inv_df = pd.DataFrame(columns=['material', 'location', 'qty_avail'])

    # 过滤当日订单
    todays_orders = order_log[order_log['date'] == simulation_date] \
        if not order_log.empty else pd.DataFrame(columns=order_log.columns)

    # 聚合订单
    if not todays_orders.empty:
        ord_g = todays_orders.groupby(
            ['material', 'location'], as_index=False
        )['quantity'].sum()
        ord_g = ord_g.rename(columns={'quantity': 'qty_ordered'})
    else:
        ord_g = pd.DataFrame(columns=['material', 'location', 'qty_ordered'])

    # 合并订单和库存
    merged = ord_g.merge(inv_df, on=['material', 'location'], how='left')
    merged['qty_avail'] = merged['qty_avail'].fillna(0).astype(int)
    merged['qty_ordered'] = merged['qty_ordered'].fillna(0).astype(int)

    # 计算发货和缺货
    merged['shipped'] = np.minimum(
        merged['qty_ordered'], merged['qty_avail']
    ).astype(int)
    merged['cut'] = (merged['qty_ordered'] - merged['shipped']).astype(int)

    # 构建输出DataFrame
    shipment_df = pd.DataFrame({
        'date': simulation_date,
        'material': merged['material'].astype(str),
        'location': merged['location'].astype(str),
        'quantity': merged['shipped'].astype(int)
    })
    cut_df = pd.DataFrame({
        'date': simulation_date,
        'material': merged['material'].astype(str),
        'location': merged['location'].astype(str),
        'quantity': merged['cut'].astype(int)
    })

    # 规范化
    shipment_df = normalize_identifiers(shipment_df)
    cut_df = normalize_identifiers(cut_df)

    return shipment_df, cut_df, current_inventory


def generate_shipment_with_inventory_check(
    orders_df: pd.DataFrame,
    simulation_date: pd.Timestamp,
    orchestrator: Optional[Any],
    demand_forecast: Optional[pd.DataFrame] = None,
    forecast_error: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """基于真实可用库存生成当日发货与缺货。

    参数:
        orders_df: 订单DataFrame。
        simulation_date: 仿真日期。
        orchestrator: 编排器对象。
        demand_forecast: 需求预测（未使用）。
        forecast_error: 预测误差（未使用）。

    返回:
        (shipment_df, cut_df)元组。
    """
    if orders_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # 当日到期订单
    today_orders = orders_df[
        pd.to_datetime(orders_df['date']) == simulation_date.normalize()
    ].copy()
    if today_orders.empty:
        return pd.DataFrame(), pd.DataFrame()

    # 规范化物料类型
    today_orders['material'] = today_orders['material'].astype(str)

    # 构建可用库存
    current_inventory = _build_available_inventory_from_orchestrator(
        orchestrator, simulation_date
    )

    # 规范化列表
    materials = [
        normalize_material(m)
        for m in today_orders['material'].unique().tolist()
    ]
    locations = [
        normalize_location(loc)
        for loc in today_orders['location'].unique().tolist()
    ]

    # 规范化订单日志
    order_log = today_orders.copy()
    order_log['material'] = order_log['material'].apply(normalize_material)
    order_log['location'] = order_log['location'].apply(normalize_location)

    # 计算发货
    shipment_df, cut_df, _ = simulate_shipment_for_single_day(
        simulation_date=simulation_date,
        order_log=order_log,
        current_inventory=current_inventory,
        material_list=materials,
        location_list=locations,
        production_plan=None,
        delivery_plan=None
    )

    # 添加订单ID
    if not shipment_df.empty:
        shipment_df['demand_type'] = 'customer'
        date_str = simulation_date.strftime('%Y%m%d')
        shipment_df['order_id'] = (
            'ORD_' + date_str + '_' + shipment_df.index.astype(str)
        )

    return shipment_df, cut_df


def _build_available_inventory_from_orchestrator(
    orchestrator: Any,
    simulation_date: pd.Timestamp
) -> Dict[Tuple[str, str], int]:
    """从编排器视图构建可用库存字典。

    可用库存 = 期初库存 + 当日生产GR + 当日调运GR。

    参数:
        orchestrator: 编排器对象。
        simulation_date: 仿真日期。

    返回:
        (material, location)到可用数量的字典。
    """
    date_str = simulation_date.strftime('%Y-%m-%d')

    # 获取各部分库存
    beg_df = orchestrator.get_beginning_inventory_view(date_str)
    prod_df = orchestrator.get_production_gr_view(date_str)
    delv_df = orchestrator.get_delivery_gr_view(date_str)

    # 转换为ML粒度
    beg = _to_ml_granularity(beg_df, 'location')
    prod = _to_ml_granularity(prod_df, 'location')
    delv = _to_ml_granularity(delv_df, 'receiving')

    # 合并
    inv_df = beg.merge(
        prod, on=['material', 'location'],
        how='outer', suffixes=('_beg', '_prod')
    )
    inv_df = inv_df.merge(delv, on=['material', 'location'], how='outer')

    # 数值转换
    for col in ['quantity_beg', 'quantity_prod', 'quantity']:
        if col in inv_df.columns:
            inv_df[col] = pd.to_numeric(inv_df[col], errors='coerce')

    inv_df[['quantity_beg', 'quantity_prod', 'quantity']] = \
        inv_df[['quantity_beg', 'quantity_prod', 'quantity']].fillna(0)

    # 计算总量
    qty = (
        inv_df.get('quantity_beg', 0) +
        inv_df.get('quantity_prod', 0) +
        inv_df.get('quantity', 0)
    )
    inv_df['qty'] = pd.to_numeric(qty, errors='coerce').fillna(0).astype(int)
    inv_df = inv_df[['material', 'location', 'qty']]

    return {
        (r.material, r.location): int(r.qty)
        for r in inv_df.itertuples(index=False)
    }


def _to_ml_granularity(
    df: Optional[pd.DataFrame],
    loc_col: str
) -> pd.DataFrame:
    """转换DataFrame到物料-地点粒度。

    参数:
        df: 输入DataFrame。
        loc_col: 地点列名。

    返回:
        ML粒度的DataFrame。
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=['material', 'location', 'quantity'])

    o = df[['material', loc_col, 'quantity']].copy()
    o['material'] = o['material'].astype(str)
    o['location'] = o[loc_col].astype(str).str.zfill(4)

    return o.groupby(['material', 'location'], as_index=False)['quantity'].sum()
