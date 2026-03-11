# -*- coding: utf-8 -*-
"""
库存计算模块

提供库存相关的计算功能，包括预测库存、可用库存和开放调拨处理。
"""
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def build_open_deployment_inbound(
    open_deployment_df: pd.DataFrame
) -> Dict[Tuple[str, str], int]:
    """
    构造开放调拨的接收端视图（pipeline inbound）。

    过滤sending != receiving且数量>0的记录，按(material, receiving)汇总。

    参数：
        open_deployment_df: 开放调拨DataFrame

    返回：
        dict: (material, receiving) -> sum(quantity)
    """
    if open_deployment_df is None or open_deployment_df.empty:
        return {}

    df = open_deployment_df.copy()

    # 统一数量列名
    if 'quantity' not in df.columns:
        if 'deployed_qty' in df.columns:
            df = df.rename(columns={'deployed_qty': 'quantity'})
        elif 'planned_qty' in df.columns:
            df = df.rename(columns={'planned_qty': 'quantity'})
        else:
            return {}

    # 过滤：数量>0，且非自循环
    df['quantity'] = pd.to_numeric(
        df['quantity'], errors='coerce'
    ).fillna(0).astype(int)
    df = df[(df['quantity'] > 0) & (df['sending'] != df['receiving'])]

    # 聚合
    g = df.groupby(['material', 'receiving'])['quantity'].sum().reset_index()

    # 构建字典
    inbound = {
        (row.material, row.receiving): int(row.quantity)
        for row in g.itertuples(index=False)
    }
    return inbound


def calculate_projected_inventory(
    beginning_inventory: dict,
    in_transit: dict,
    delivery_gr: dict,
    today_production_gr: dict,
    future_production: dict,
    today_shipment: dict,
    open_deployment: dict
) -> dict:
    """
    计算预测库存（用于缺口判断与规划）。

    公式：beginning + in_transit + delivery_gr + today_production
          + future_production - today_shipment - open_deployment

    参数：
        beginning_inventory: 期初库存
        in_transit: 在途库存
        delivery_gr: 当日收货
        today_production_gr: 当日生产
        future_production: 未来生产
        today_shipment: 当日发货
        open_deployment: 开放调拨

    返回：
        dict: (material, location) -> 预测库存量
    """
    all_keys = set()
    for d in [beginning_inventory, in_transit, delivery_gr,
              today_production_gr, future_production,
              today_shipment, open_deployment]:
        all_keys.update(d.keys())

    projected_inventory = {}
    for key in all_keys:
        projected_inventory[key] = (
            beginning_inventory.get(key, 0) +
            in_transit.get(key, 0) +
            delivery_gr.get(key, 0) +
            today_production_gr.get(key, 0) +
            future_production.get(key, 0) -
            today_shipment.get(key, 0) -
            open_deployment.get(key, 0)
        )

    return projected_inventory


def calculate_available_inventory(
    beginning_inventory: dict,
    delivery_gr: dict,
    today_production_gr: dict,
    today_shipment: dict,
    open_deployment: dict,
    open_deployment_inbound: dict
) -> dict:
    """
    计算当日真实可用库存（用于实际分配）。

    公式：beginning + delivery_gr + today_production_gr - open_deployment

    参数：
        beginning_inventory: 期初库存
        delivery_gr: 当日收货
        today_production_gr: 当日生产
        today_shipment: 当日发货（未使用，保留接口兼容性）
        open_deployment: 开放调拨
        open_deployment_inbound: 开放调拨入库（未使用，保留接口兼容性）

    返回：
        dict: (material, location) -> 可用库存量
    """
    all_keys = set()
    for d in [beginning_inventory, delivery_gr, today_production_gr,
              today_shipment, open_deployment]:
        all_keys.update(d.keys())

    soh = {}
    for key in all_keys:
        soh[key] = (
            beginning_inventory.get(key, 0) +
            delivery_gr.get(key, 0) +
            today_production_gr.get(key, 0) -
            open_deployment.get(key, 0)
        )
    return soh


def get_qty_from_row(row, col_names: list) -> int:
    """
    从行中获取第一个有效的数量值。

    参数：
        row: DataFrame行（namedtuple）
        col_names: 按优先级排列的列名列表

    返回：
        int: 找到的数量值，未找到返回0
    """
    for col in col_names:
        val = getattr(row, col, None)
        if val is not None and not pd.isna(val):
            return int(val)
    return 0


def build_production_dicts(
    production_plan: pd.DataFrame,
    sim_date: pd.Timestamp
) -> Tuple[dict, dict]:
    """
    构建当日和未来生产字典。

    参数：
        production_plan: 生产计划DataFrame
        sim_date: 仿真日期

    返回：
        tuple: (today_production_gr, future_production) 两个字典
    """
    today_production_gr = {}
    future_production = {}

    if production_plan.empty:
        return today_production_gr, future_production

    # 当日生产
    today_prod = production_plan[
        production_plan['available_date'] == sim_date
    ]
    for row in today_prod.itertuples():
        k = (row.material, row.location)
        qty = get_qty_from_row(
            row, ['produced_qty', 'planned_qty', 'quantity']
        )
        today_production_gr[k] = today_production_gr.get(k, 0) + qty

    # 未来生产
    future_prod = production_plan[
        production_plan['available_date'] > sim_date
    ]
    for row in future_prod.itertuples():
        k = (row.material, row.location)
        qty = get_qty_from_row(
            row, ['uncon_planned_qty', 'produced_qty', 'planned_qty', 'quantity']
        )
        future_production[k] = future_production.get(k, 0) + qty

    return today_production_gr, future_production


def build_intransit_dicts(
    in_transit: pd.DataFrame,
    sim_date: pd.Timestamp
) -> Tuple[dict, dict]:
    """
    构建当日和未来在途字典。

    参数：
        in_transit: 在途库存DataFrame
        sim_date: 仿真日期

    返回：
        tuple: (today_intransit, future_intransit) 两个字典
    """
    today_intransit = {}
    future_intransit = {}

    if in_transit.empty:
        return today_intransit, future_intransit

    # 确定日期列
    date_col = None
    if 'actual_delivery_date' in in_transit.columns:
        date_col = 'actual_delivery_date'
    elif 'available_date' in in_transit.columns:
        date_col = 'available_date'

    if date_col is None:
        return today_intransit, future_intransit

    # 当日在途
    mask_today = (
        pd.to_datetime(in_transit[date_col]).dt.normalize() ==
        sim_date.normalize()
    )
    for row in in_transit[mask_today].itertuples():
        k = (row.material, row.receiving)
        today_intransit[k] = today_intransit.get(k, 0) + int(row.quantity)

    # 未来在途
    mask_future = (
        pd.to_datetime(in_transit[date_col]).dt.normalize() >
        sim_date.normalize()
    )
    for row in in_transit[mask_future].itertuples():
        k = (row.material, row.receiving)
        future_intransit[k] = future_intransit.get(k, 0) + int(row.quantity)

    return today_intransit, future_intransit


def build_delivery_gr_dict(
    delivery_gr_data: pd.DataFrame,
    sim_date: pd.Timestamp
) -> dict:
    """
    构建当日收货字典。

    参数：
        delivery_gr_data: 收货数据DataFrame
        sim_date: 仿真日期

    返回：
        dict: (material, receiving) -> 收货量
    """
    delivery_gr = {}

    if delivery_gr_data.empty:
        return delivery_gr

    if 'date' in delivery_gr_data.columns:
        filtered = delivery_gr_data[
            pd.to_datetime(delivery_gr_data['date']).dt.normalize() ==
            sim_date.normalize()
        ]
    else:
        filtered = delivery_gr_data

    for row in filtered.itertuples():
        k = (row.material, row.receiving)
        delivery_gr[k] = delivery_gr.get(k, 0) + int(row.quantity)

    return delivery_gr


def build_shipment_dict(
    shipment_data: pd.DataFrame,
    sim_date: pd.Timestamp
) -> dict:
    """
    构建当日发货字典。

    参数：
        shipment_data: 发货数据DataFrame
        sim_date: 仿真日期

    返回：
        dict: (material, location) -> 发货量
    """
    today_shipment = {}

    if shipment_data.empty:
        return today_shipment

    if 'date' in shipment_data.columns:
        filtered = shipment_data[
            pd.to_datetime(shipment_data['date']) == sim_date
        ]
    else:
        filtered = shipment_data

    for row in filtered.itertuples():
        k = (row.material, row.location)
        today_shipment[k] = today_shipment.get(k, 0) + int(row.quantity)

    return today_shipment


def build_open_deployment_dict(open_deployment_data: pd.DataFrame) -> dict:
    """
    构建开放调拨发送端字典（排除自循环）。

    参数：
        open_deployment_data: 开放调拨DataFrame

    返回：
        dict: (material, sending) -> 调拨量
    """
    open_deployment = {}

    if open_deployment_data.empty:
        return open_deployment

    for row in open_deployment_data.itertuples():
        # 只计算非自循环
        if row.sending != row.receiving:
            k = (row.material, row.sending)
            open_deployment[k] = open_deployment.get(k, 0) + int(row.quantity)

    return open_deployment
