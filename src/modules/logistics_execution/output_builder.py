# -*- coding: utf-8 -*-
"""
物流执行模块 - 输出生成
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from .constraint_enforcer import validate_shipment_delivery_constraint
from .validators import generate_validation_report


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
    # 构建 DataFrame - 空列表时也要带列名
    delivery_plan_df = _build_delivery_plan_df(results['delivery_plan'])

    # 验证约束: 出货量 <= 订单量
    constraint_passed, shipment_qty, delivery_qty = validate_shipment_delivery_constraint(
        delivery_plan_df,
        run_params.get('orchestrator'),
        validation_log
    )

    if not constraint_passed:
        print(f"⚠️  发现约束违反: delivery_qty ({delivery_qty}) > shipment_qty ({shipment_qty})")
        print(f"   这可能由以下原因引起:")
        print(f"   1. Module5部署计划多重生成导致deployed_qty超出shipment数量")
        print(f"   2. 订单去重不当导致同一订单被多次处理")
        print(f"   3. Module6装载优化产生了超额分配")

    vehicle_df = _build_vehicle_df(results['vehicle_log'])
    usage_df = _build_usage_df(vehicle_df)
    unsat_df = _build_unsat_df(results['unsat_log'])
    validation_df = _build_validation_df(validation_log)
    bypass_df = _build_bypass_df(results['bypass_log'])

    # 写入文件
    if not skip_file_output:
        _write_excel_output(
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


def _build_delivery_plan_df(delivery_plan: List[Dict]) -> pd.DataFrame:
    """构建交付计划 DataFrame。"""
    if delivery_plan:
        return pd.DataFrame(delivery_plan)
    # 与基线保持一致：空表不带列名
    return pd.DataFrame()


def _build_vehicle_df(vehicle_log: List[Dict]) -> pd.DataFrame:
    """构建车辆日志 DataFrame。"""
    if vehicle_log:
        return pd.DataFrame(vehicle_log)
    # 与基线保持一致：空表带列名
    return pd.DataFrame(columns=[
        'date', 'sending', 'receiving', 'truck_type', 'vehicle_no', 'vehicle_uid',
        'total_units', 'total_weight', 'total_volume', 'WFR', 'VFR', 'trigger'
    ])


def _build_usage_df(vehicle_df: pd.DataFrame) -> pd.DataFrame:
    """构建使用统计 DataFrame。"""
    if vehicle_df.empty:
        # 与基线保持一致：空表带列名
        return pd.DataFrame(columns=['date', 'sending', 'receiving', 'truck_type', 'truck_used'])

    return vehicle_df.groupby(
        ['date', 'sending', 'receiving', 'truck_type'],
        as_index=False
    ).agg(truck_used=('vehicle_uid', 'nunique'))


def _build_unsat_df(unsat_log: List[Dict]) -> pd.DataFrame:
    """构建未满足MDQ日志 DataFrame。"""
    if unsat_log:
        return pd.DataFrame(unsat_log)
    # 与基线保持一致：空表不带列名
    return pd.DataFrame()


def _build_validation_df(validation_log: List[Dict]) -> pd.DataFrame:
    """构建验证日志 DataFrame。"""
    if validation_log:
        return pd.DataFrame(validation_log)
    # 与基线保持一致：空表不带列名
    return pd.DataFrame()


def _build_bypass_df(bypass_log: List[Dict]) -> pd.DataFrame:
    """构建绕过规则命中日志 DataFrame。"""
    if bypass_log:
        return pd.DataFrame(bypass_log)
    # 与基线保持一致：空表不带列名
    return pd.DataFrame()


def _write_excel_output(
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
