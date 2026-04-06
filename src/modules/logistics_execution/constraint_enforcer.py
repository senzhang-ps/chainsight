# -*- coding: utf-8 -*-
"""
物流执行模块 - 约束执行与验证
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def enforce_shipment_constraint(
    delivery_plan_df: pd.DataFrame,
    orchestrator: Optional[object],
    validation_log: List[Dict]
) -> pd.DataFrame:
    """
    强制约束出货量不超过订单量，如果超出则按比例裁剪。

    参数：
        delivery_plan_df: 出货计划 DataFrame
        orchestrator: Orchestrator 实例
        validation_log: 验证日志

    返回：
        裁剪后的出货计划 DataFrame
    """
    if delivery_plan_df.empty or orchestrator is None:
        return delivery_plan_df

    # 获取日期列
    if 'sim_date' in delivery_plan_df.columns:
        date_col = 'sim_date'
    elif 'date' in delivery_plan_df.columns:
        date_col = 'date'
    else:
        return delivery_plan_df

    # 按物料和发货地点分组，比对订单量
    result_df = delivery_plan_df.copy()

    for date_val in result_df[date_col].unique():
        date_str = str(date_val).split()[0]

        # 获取当日订单数据
        shipment_data = orchestrator.get_shipment_log_view(date_str)
        if shipment_data.empty:
            continue

        # 按(material, location)计算订单量
        shipment_qty_by_ml = shipment_data.groupby(
            ['material', 'location']
        )['quantity'].sum().to_dict()

        # 按(material, sending)计算当前出货量
        day_mask = result_df[date_col] == date_val
        day_deliveries = result_df[day_mask]

        if day_deliveries.empty:
            continue

        for (mat, sending), group in day_deliveries.groupby(['material', 'sending']):
            # 获取该(material, location)的订单量
            order_qty = shipment_qty_by_ml.get((mat, sending), 0)
            delivery_qty = group['delivery_qty'].sum()

            if delivery_qty > order_qty and delivery_qty > 0:
                # 按比例裁剪
                ratio = order_qty / delivery_qty if delivery_qty > 0 else 0

                print(f"🔧 [Module6] 强制裁剪: {mat}@{sending}")
                print(f"    订单量: {order_qty:.0f}, 出货量: {delivery_qty:.0f}")
                print(f"    裁剪比例: {ratio:.2%}")

                # 裁剪每一行的 delivery_qty
                for idx in group.index:
                    old_qty = result_df.at[idx, 'delivery_qty']
                    new_qty = int(old_qty * ratio)
                    result_df.at[idx, 'delivery_qty'] = new_qty

                validation_log.append({
                    'sheet': 'Module6_Enforcement',
                    'row': f'{mat}@{sending}',
                    'issue': f'Delivery quantity trimmed from {delivery_qty:.0f} to {order_qty:.0f}',
                    'severity': 'INFO',
                    'impact': f'Enforced shipment constraint - reduced by {delivery_qty - order_qty:.0f} units',
                    'shipment_qty': order_qty,
                    'original_delivery_qty': delivery_qty
                })

    return result_df


def validate_shipment_delivery_constraint(
    delivery_plan_df: pd.DataFrame,
    orchestrator: Optional[object],
    validation_log: List[Dict]
) -> Tuple[bool, int, int]:
    """
    验证出货量不超过订单量约束。

    参数：
        delivery_plan_df: 出货计划 DataFrame
        orchestrator: Orchestrator 实例
        validation_log: 验证日志

    返回：
        (是否通过验证, 订单量, 出货量)
    """
    if delivery_plan_df.empty or orchestrator is None:
        return True, 0, 0

    # 检查 sim_date 列是否存在，如果不存在则使用 date 列（本地模式下）
    if 'sim_date' not in delivery_plan_df.columns:
        if 'date' not in delivery_plan_df.columns:
            # 两个列都不存在，无法验证
            return True, 0, 0
        # 使用 date 列作为替代
        date_col = 'date'
    else:
        date_col = 'sim_date'

    # 计算各日期各地点的出货量
    delivery_qty_by_date_loc = delivery_plan_df.groupby(date_col)['delivery_qty'].sum()
    total_delivery_qty = delivery_qty_by_date_loc.sum()

    # 计算各日期的订单量
    shipment_logs = []
    for date_val in delivery_plan_df[date_col].unique():
        # 从日期值中提取日期部分（处理 sim_date 和 date 格式）
        date_str = str(date_val).split()[0]
        shipment_data = orchestrator.get_shipment_log_view(date_str)
        if not shipment_data.empty:
            shipment_logs.append(shipment_data)

    if shipment_logs:
        all_shipments = pd.concat(shipment_logs, ignore_index=True)
        total_shipment_qty = all_shipments['quantity'].astype(float).sum()
    else:
        total_shipment_qty = 0

    # 验证
    if total_delivery_qty > total_shipment_qty:
        print(f"\n⚠️  约束违反: 出货量 > 订单量")
        print(f"    订单量: {total_shipment_qty:.0f}")
        print(f"    出货量: {total_delivery_qty:.0f}")
        print(f"    超出: {total_delivery_qty - total_shipment_qty:.0f}")

        validation_log.append({
            'sheet': 'Module6_Constraint',
            'row': '',
            'issue': f'Delivery quantity ({total_delivery_qty:.0f}) exceeds shipment quantity ({total_shipment_qty:.0f})',
            'severity': 'ERROR',
            'impact': f'Constraint Violation - {total_delivery_qty - total_shipment_qty:.0f} units over limit',
            'shipment_qty': total_shipment_qty,
            'delivery_qty': total_delivery_qty
        })
        return False, int(total_shipment_qty), int(total_delivery_qty)

    return True, int(total_shipment_qty), int(total_delivery_qty)
