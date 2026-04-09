# -*- coding: utf-8 -*-
"""
output_writer.py - Module6 输出组装与文件写入

包含输出 DataFrame 构建、约束验证、Excel 写入等逻辑。
从 module6.py 等价迁入，函数签名和行为不变。
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .validators import generate_validation_report


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
            order_qty = shipment_qty_by_ml.get((mat, sending), 0)
            delivery_qty = group['delivery_qty'].sum()

            if delivery_qty > order_qty and delivery_qty > 0:
                ratio = order_qty / delivery_qty if delivery_qty > 0 else 0

                print(f"  [Module6] enforce trim: {mat}@{sending}")
                print(f"    order_qty: {order_qty:.0f}, delivery_qty: {delivery_qty:.0f}")
                print(f"    ratio: {ratio:.2%}")

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

    返回：
        (是否通过验证, 订单量, 出货量)
    """
    if delivery_plan_df.empty or orchestrator is None:
        return True, 0, 0

    if 'sim_date' not in delivery_plan_df.columns:
        if 'date' not in delivery_plan_df.columns:
            return True, 0, 0
        date_col = 'date'
    else:
        date_col = 'sim_date'

    delivery_qty_by_date_loc = delivery_plan_df.groupby(date_col)['delivery_qty'].sum()
    total_delivery_qty = delivery_qty_by_date_loc.sum()

    shipment_logs = []
    for date_val in delivery_plan_df[date_col].unique():
        date_str = str(date_val).split()[0]
        shipment_data = orchestrator.get_shipment_log_view(date_str)
        if not shipment_data.empty:
            shipment_logs.append(shipment_data)

    if shipment_logs:
        all_shipments = pd.concat(shipment_logs, ignore_index=True)
        total_shipment_qty = all_shipments['quantity'].astype(float).sum()
    else:
        total_shipment_qty = 0

    if total_delivery_qty > total_shipment_qty:
        print(f"\n  constraint violation: delivery_qty > shipment_qty")
        print(f"    shipment_qty: {total_shipment_qty:.0f}")
        print(f"    delivery_qty: {total_delivery_qty:.0f}")
        print(f"    over: {total_delivery_qty - total_shipment_qty:.0f}")

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


def generate_outputs(
    run_params: Dict[str, Any],
    results: Dict[str, List],
    validation_log: List[Dict],
    skip_file_output: bool
) -> Dict[str, Any]:
    """
    生成输出结果。

    参数：
        run_params: 运行参数
        results: 仿真结果
        validation_log: 验证日志
        skip_file_output: 是否跳过文件输出

    返回：
        输出结果字典
    """
    # 构建 DataFrame
    delivery_plan_df = build_delivery_plan_df(results['delivery_plan'])

    # 验证约束: 出货量 <= 订单量
    constraint_passed, shipment_qty, delivery_qty = validate_shipment_delivery_constraint(
        delivery_plan_df,
        run_params.get('orchestrator'),
        validation_log
    )

    if not constraint_passed:
        print(f"  constraint violation: delivery_qty ({delivery_qty}) > shipment_qty ({shipment_qty})")

    vehicle_df = build_vehicle_df(results['vehicle_log'])
    usage_df = build_usage_df(vehicle_df)
    unsat_df = build_unsat_df(results['unsat_log'])
    validation_df = build_validation_df(validation_log)
    bypass_df = build_bypass_df(results['bypass_log'])

    # 写入文件
    if not skip_file_output:
        write_excel_output(
            run_params['output_file'], delivery_plan_df, vehicle_df,
            usage_df, unsat_df, validation_df, bypass_df
        )
        generate_validation_report(validation_log, run_params['output_file'])

    # 统计信息
    statistics = {
        'delivery_count': len(results['delivery_plan']),
        'vehicle_count': usage_df['truck_used'].sum() if not usage_df.empty else 0,
        'unsatisfied_count': len(results['unsat_log']),
        'bypass_count': len(results['bypass_log'])
    }

    return {
        'delivery_plan': delivery_plan_df,
        'vehicle_log': vehicle_df,
        'truck_usage': usage_df,
        'unsatisfied_log': unsat_df,
        'validation_log': validation_df,
        'bypass_log': bypass_df,
        'statistics': statistics
    }


def build_delivery_plan_df(delivery_plan: List[Dict]) -> pd.DataFrame:
    """构建交付计划 DataFrame。"""
    if delivery_plan:
        return pd.DataFrame(delivery_plan)
    return pd.DataFrame()


def build_vehicle_df(vehicle_log: List[Dict]) -> pd.DataFrame:
    """构建车辆日志 DataFrame。"""
    if vehicle_log:
        return pd.DataFrame(vehicle_log)
    return pd.DataFrame(columns=[
        'date', 'sending', 'receiving', 'truck_type', 'vehicle_no', 'vehicle_uid',
        'total_units', 'total_weight', 'total_volume', 'WFR', 'VFR', 'trigger'
    ])


def build_usage_df(vehicle_df: pd.DataFrame) -> pd.DataFrame:
    """构建使用统计 DataFrame。"""
    if vehicle_df.empty:
        return pd.DataFrame(columns=['date', 'sending', 'receiving', 'truck_type', 'truck_used'])

    return vehicle_df.groupby(
        ['date', 'sending', 'receiving', 'truck_type'],
        as_index=False
    ).agg(truck_used=('vehicle_uid', 'nunique'))


def build_unsat_df(unsat_log: List[Dict]) -> pd.DataFrame:
    """构建未满足MDQ日志 DataFrame。"""
    if unsat_log:
        return pd.DataFrame(unsat_log)
    return pd.DataFrame()


def build_validation_df(validation_log: List[Dict]) -> pd.DataFrame:
    """构建验证日志 DataFrame。"""
    if validation_log:
        return pd.DataFrame(validation_log)
    return pd.DataFrame()


def build_bypass_df(bypass_log: List[Dict]) -> pd.DataFrame:
    """构建绕过规则命中日志 DataFrame。"""
    if bypass_log:
        return pd.DataFrame(bypass_log)
    return pd.DataFrame()


def write_excel_output(
    output_file: str,
    delivery_plan_df: pd.DataFrame,
    vehicle_df: pd.DataFrame,
    usage_df: pd.DataFrame,
    unsat_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    bypass_df: pd.DataFrame
) -> None:
    """写入 Excel 输出文件。"""
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        delivery_plan_df.to_excel(writer, sheet_name='DeliveryPlan', index=False)
        vehicle_df.to_excel(writer, sheet_name='VehicleLog', index=False)
        usage_df.to_excel(writer, sheet_name='TruckUsageLog', index=False)
        unsat_df.to_excel(writer, sheet_name='UnsatisfiedMDQLog', index=False)
        validation_df.to_excel(writer, sheet_name='ValidationLog', index=False)
        bypass_df.to_excel(writer, sheet_name='BypassRuleHitLog', index=False)
