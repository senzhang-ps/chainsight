"""Module1 集成模式主入口。

本模块提供集成模式的主要功能。

主要函数：
- run_daily_order_generation: 集成模式主入口
- generate_supply_demand_log_for_integration: 生成供需日志
"""

import os
from typing import Any, Optional

import numpy as np
import pandas as pd

from .constants import DEFAULT_MAX_ADVANCE_DAYS
from ...utils.defaults import M1_FUTURE_CUTOFF_DAYS
from ...utils.normalization import normalize_identifiers
from .shipment import generate_shipment_with_inventory_check
from .io_utils import (
    load_previous_orders,
    save_module1_output_with_supply_demand,
)


def run_daily_order_generation(
    config_dict: dict,
    simulation_date: pd.Timestamp,
    output_dir: str,
    orchestrator: object = None,
    skip_file_output: bool = False,
    previous_orders_df: Optional[pd.DataFrame] = None
) -> dict:
    """集成模式主入口：生成指定日期的订单与发货。"""
    try:
        # 1) 校验配置
        configs = _validate_config(config_dict)
        demand_forecast, forecast_error, order_calendar, ao_config, dps_config, dps_sc_config = configs
        demand_forecast['quantity']/=1.19828

        # 2) 准备AO汇总
        ao_config, ao_config_summary = _prepare_ao_summary(ao_config)

        # 3) DPS拆分 + 周预测准备
        demand_forecast_total, demand_forecast_total_sc, order_calendar = _apply_dps_and_supply_choice(
            demand_forecast, dps_config, dps_sc_config, orchestrator, order_calendar
        )

        # 4) 日度拆分
        demand_forecast_detail = _distribute_to_daily(demand_forecast_total, order_calendar)
        demand_forecast_detail_sc = _distribute_to_daily(demand_forecast_total_sc, order_calendar)


        # 5) AO拆分 + 误差应用
        df_with_error = _split_by_ao_and_apply_error(
            demand_forecast_total, ao_config_summary, forecast_error
        )

        # 6) advance_days拆分 + 构建订单
        ao_detail = _split_ao_by_advance_days(
            df_with_error, ao_config, simulation_date, order_calendar
        )
        order_df = _build_order_df(ao_detail, simulation_date)

        # 7) 合并历史订单
        all_orders_df = _merge_with_history(
            output_dir, simulation_date, order_df, ao_config,
            previous_orders_df=previous_orders_df
        )
        value = order_calendar.loc[order_calendar['date'] == simulation_date, 'order_day_flag'].item()
        orchestrator.shipment_valid = int(value)

        # 8) 生成发货
        shipment_df, cut_df = _generate_shipments(
            all_orders_df, simulation_date, orchestrator, demand_forecast_detail
        )

        # 9) 生成供需日志
        consumed_supply = _apply_orders_consumption(demand_forecast_detail_sc, order_df)
        supply_demand_df = generate_supply_demand_log_for_integration(
            demand_forecast_detail_sc, consumed_supply, simulation_date
        )

        # 10) 保存输出
        output_file = _save_output(
            all_orders_df, shipment_df, cut_df, supply_demand_df,
            output_dir, simulation_date, skip_file_output
        )

        # 11) 生成Summary
        summary_df = _build_summary_df(all_orders_df, shipment_df, cut_df, supply_demand_df)

        return {
            'orders_df': all_orders_df,
            'shipment_df': shipment_df,
            'cut_df': cut_df,
            'supply_demand_df': supply_demand_df,
            'summary_df': summary_df,
            'output_file': output_file,
            'all_orders_for_next_day': all_orders_df
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return _empty_result()


# ---- 步骤函数 ----


def _prepare_ao_summary(ao_config: pd.DataFrame) -> tuple:
    """聚合ao_config，构造AO/normal比例汇总表。

    Returns:
        (ao_config, ao_config_summary): 处理后的ao_config（含percent列）和汇总比例表
    """
    ao_config = ao_config.copy()
    ao_config.drop_duplicates(['material', 'location', 'advance_days', 'ao_percent'], inplace=True)

    ao_config_summary = ao_config.groupby(
        ['material', 'location']
    ).agg(ao_percent_sum=('ao_percent', 'sum')).reset_index()

    ao_config['order_type'] = 'AO'
    ao_config.rename(columns={'ao_percent': 'percent'}, inplace=True)

    ao_config_summary['ao_type'] = 'AO'
    ao_config_summary['normal_percent'] = 1 - ao_config_summary['ao_percent_sum'].clip(0, 1)
    ao_config_summary['normal_type'] = 'normal'

    ao_config_summary = pd.concat([
        ao_config_summary[['material', 'location', 'ao_type', 'ao_percent_sum']].rename(
            columns={'ao_type': 'order_type', 'ao_percent_sum': 'ao_percent'}),
        ao_config_summary[['material', 'location', 'normal_type', 'normal_percent']].rename(
            columns={'normal_type': 'order_type', 'normal_percent': 'ao_percent'})
    ])

    return ao_config, ao_config_summary


def _apply_dps_and_supply_choice(
    demand_forecast: pd.DataFrame,
    dps_config: pd.DataFrame,
    dps_sc_config: pd.DataFrame,
    orchestrator: Any,
    order_calendar: pd.DataFrame
) -> tuple:
    """DPS拆分、周预测聚合、flag_count计算、supply choice合并。

    同时为order_calendar添加week_start列。

    Returns:
        (demand_forecast_total, demand_forecast_total_sc, order_calendar):
            基础预测总量、含SC调整量的预测总量、带有week_start的日历
    """
    # DPS反向拆分
    dps_config = dps_config.copy()
    dps_config['reverse_dps_percent'] = 1 - dps_config['dps_percent']
    dps_config = pd.concat([
        dps_config[['material', 'location', 'reverse_dps_percent']].rename(
            columns={'reverse_dps_percent': 'dps_percent'}),
        dps_config[['material', 'dps_location', 'dps_percent']].rename(
            columns={'dps_location': 'location'})
    ])

    # 按周聚合需求
    demand_forecast = demand_forecast.groupby(['week', 'material', 'location'])['quantity'].sum().reset_index()
    demand_forecast = demand_forecast.sort_values(['material', 'location', 'week'])
    demand_forecast['week_start'] = demand_forecast['week'].map(
        lambda x: orchestrator.start_date + pd.Timedelta(days=(x - 1) * 7)
    )

    # 关联order_calendar统计每周有效配货天数
    week_starts = demand_forecast['week_start'].sort_values().drop_duplicates().reset_index(drop=True)
    bins = week_starts.tolist() + [week_starts.iloc[-1] + pd.Timedelta(days=7)]
    order_calendar['week_start'] = pd.cut(
        order_calendar['date'], bins=bins, labels=week_starts, right=False
    )
    flag_summary = order_calendar.groupby('week_start')['order_day_flag'].sum().reset_index(name='flag_count')
    flag_summary['week_start'] = pd.to_datetime(flag_summary['week_start'])

    demand_forecast = demand_forecast.merge(flag_summary, on='week_start', how='left')
    demand_forecast.dropna(inplace=True)

    # DPS拆分
    demand_forecast_split_by_dps = pd.merge(
        demand_forecast, dps_config, on=['material', 'location'], how='left'
    )
    demand_forecast_split_by_dps = demand_forecast_split_by_dps.fillna(1)
    demand_forecast_split_by_dps['quantity_percentage'] = (
        demand_forecast_split_by_dps['quantity'] * demand_forecast_split_by_dps['dps_percent']
    )

    # 基础版本（不含SC调整量）
    demand_forecast_total = demand_forecast_split_by_dps.copy()
    demand_forecast_total['quantity_total'] = demand_forecast_total['quantity_percentage']

    # Supply choice合并版本
    if not dps_sc_config.empty:
        demand_forecast_total_sc = pd.merge(
            demand_forecast_split_by_dps, dps_sc_config,
            on=['week', 'material', 'quantity'], how='left'
        )
        demand_forecast_total_sc['quantity_total'] = (
            demand_forecast_total_sc['quantity_percentage'] + demand_forecast_total_sc['adjust_quantity']
        )
    else:
        demand_forecast_total_sc = demand_forecast_total.copy()

    # 转换order_calendar的week_start为datetime
    order_calendar['week_start'] = pd.to_datetime(order_calendar['week_start'].astype(object))

    return demand_forecast_total, demand_forecast_total_sc, order_calendar


def _distribute_to_daily(
    demand_forecast_total: pd.DataFrame,
    order_calendar: pd.DataFrame
) -> pd.DataFrame:
    """将周预测按order_calendar拆分到每日。

    base_qty = quantity_total // flag_count，余数分配到前几天。
    """
    detail = pd.merge(
        demand_forecast_total[['week', 'location', 'material', 'quantity_total', 'week_start', 'flag_count']],
        order_calendar, on='week_start', how='left'
    )
    detail['base_qty'] = np.where(
        (detail['order_day_flag'] == 1) & (detail['flag_count'] > 0),
        (detail['quantity_total'] // detail['flag_count']).astype(int),
        0
    )
    detail['remainder'] = (detail['quantity_total'] % detail['flag_count']).astype(int)
    detail['base_qty'] = detail.apply(
        lambda x: x.base_qty + 1 if (x.date - x.week_start) < pd.Timedelta(days=x.remainder)
                else x.base_qty,
        axis=1
    )
    detail.drop('quantity_total', axis=1, inplace=True)
    detail = detail.rename(columns={'base_qty': 'quantity'})
    return detail


def _split_by_ao_and_apply_error(
    demand_forecast_total: pd.DataFrame,
    ao_config_summary: pd.DataFrame,
    forecast_error: pd.DataFrame
) -> pd.DataFrame:
    """按AO/normal比例拆分quantity_total，并应用正态分布误差生成cov_quantity。"""
    # AO拆分
    df = pd.merge(demand_forecast_total, ao_config_summary, on=['material', 'location'], how='left')
    df['order_type'] = df['order_type'].fillna('normal')
    df['ao_percent'] = df['ao_percent'].fillna(1)
    df['split_quantity'] = df['ao_percent'] * df['quantity_total']

    # 应用误差
    df = pd.merge(df, forecast_error, on=['material', 'location', 'order_type'], how='left')
    df['error_std_percent'] = df['error_std_percent'].fillna(0)
    df['abs_std'] = df['split_quantity'] * df['error_std_percent']
    df['cov_quantity'] = np.maximum(
        0, np.round(np.random.normal(df['split_quantity'], df['abs_std']))
    ).astype(int)

    return df


def _split_ao_by_advance_days(
    df_with_error: pd.DataFrame,
    ao_config: pd.DataFrame,
    simulation_date: pd.Timestamp,
    order_calendar: pd.DataFrame
) -> pd.DataFrame:
    """筛选当周数据，按advance_days进一步拆分AO订单。"""
    target_date = (
        order_calendar[order_calendar['date'] == simulation_date]['week_start']
        .dt.strftime('%Y-%m-%d').item()
    )
    df_filtered = df_with_error[df_with_error['week_start'] == target_date]

    result = pd.merge(df_filtered, ao_config, on=['material', 'location', 'order_type'], how='left')
    result['advance_days'] = result['advance_days'].fillna(0)
    result['percent'] = result['percent'].fillna(result['ao_percent'])
    result['cov_quantity_ao_detail'] = (
        result['cov_quantity'] * result['percent'] / result['ao_percent']
    )

    return result


def _build_order_df(
    ao_detail: pd.DataFrame,
    simulation_date: pd.Timestamp
) -> pd.DataFrame:
    """汇总生成最终订单DataFrame。"""
    ao_detail['daily_quantity'] = ao_detail['cov_quantity_ao_detail'] / ao_detail['flag_count']
    ao_detail['date'] = (
        pd.to_datetime(simulation_date)
        + pd.to_timedelta(ao_detail['advance_days'], unit='D')
    )

    order_df = (
        ao_detail
        .groupby(['date', 'material', 'location', 'order_type', 'advance_days'])['daily_quantity']
        .sum()
        .reset_index()
        .rename(columns={'daily_quantity': 'quantity', 'order_type': 'demand_type'})
    )
    order_df['simulation_date'] = simulation_date
    # order_df['quantity'] = order_df['quantity'] / bias_rate
    # order_df['quantity'] = np.floor(order_df['quantity'] / 1.2 + 0.5).astype(int)

    return order_df


# ---- 辅助函数 ----


def _empty_result() -> dict:
    """返回空结果字典。"""
    return {
        'orders_df': pd.DataFrame(),
        'shipment_df': pd.DataFrame(),
        'cut_df': pd.DataFrame(),
        'supply_demand_df': pd.DataFrame(),
        'summary_df': pd.DataFrame(),
        'output_file': None
    }


def _build_summary_df(
    orders_df: pd.DataFrame,
    shipment_df: pd.DataFrame,
    cut_df: pd.DataFrame,
    supply_demand_df: pd.DataFrame
) -> pd.DataFrame:
    """构建汇总DataFrame（供数据库模式使用）。"""
    date_val = orders_df['date'].iloc[0] if not orders_df.empty else None
    return pd.DataFrame([{
        'Total_Orders': len(orders_df),
        'Total_Shipments': len(shipment_df),
        'Total_Cuts': len(cut_df),
        'Total_SupplyDemand': len(supply_demand_df),
        'Date': date_val
    }])


def _validate_config(config_dict: dict) -> tuple:
    """校验M1配置。"""
    demand_forecast = config_dict.get('M1_DemandForecast', pd.DataFrame())
    forecast_error = config_dict.get('M1_ForecastError', pd.DataFrame())
    order_calendar = config_dict.get('M1_OrderCalendar', pd.DataFrame())
    ao_config = config_dict.get('M1_AOConfig', pd.DataFrame())
    dps_config = config_dict.get('M1_DPSConfig', pd.DataFrame())
    dps_sc_config = config_dict.get('M1_SupplyChoiceConfig', pd.DataFrame())

    if demand_forecast.empty:
        raise ValueError("缺少必需的配置数据：M1_DemandForecast")
    if order_calendar.empty:
        raise ValueError("缺少必需的配置数据：M1_OrderCalendar")
    if ao_config.empty:
        raise ValueError("缺少必需的配置数据：M1_AOConfig")
    if forecast_error.empty:
        raise ValueError("缺少必需的配置数据：M1_ForecastError")

    order_calendar['date'] = pd.to_datetime(order_calendar['date'])

    if 'quantity' in demand_forecast.columns:
        demand_forecast['quantity'] = demand_forecast['quantity'].clip(lower=0)

    demand_forecast = normalize_identifiers(demand_forecast)
    forecast_error = normalize_identifiers(forecast_error)
    ao_config = normalize_identifiers(ao_config)

    return demand_forecast, forecast_error, order_calendar, ao_config, dps_config, dps_sc_config


def _merge_with_history(
    output_dir: str,
    simulation_date: pd.Timestamp,
    today_orders_df: pd.DataFrame,
    ao_config: pd.DataFrame,
    previous_orders_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """合并历史订单与当日订单。"""
    max_advance = _get_max_advance_days(ao_config)

    if previous_orders_df is not None and not previous_orders_df.empty:
        previous_orders = previous_orders_df.copy()
    else:
        previous_orders = load_previous_orders(output_dir, simulation_date, max_advance)

    previous_orders = _filter_future_orders(previous_orders, simulation_date)
    previous_orders = _deduplicate_orders(previous_orders)

    if today_orders_df is not None and not today_orders_df.empty:
        orders_df = pd.concat([previous_orders, today_orders_df], ignore_index=True)
    else:
        orders_df = previous_orders.copy()

    return _normalize_orders(orders_df)


def _get_max_advance_days(ao_config: pd.DataFrame) -> int:
    """获取最大提前天数。"""
    if not ao_config.empty and 'advance_days' in ao_config.columns:
        max_val = ao_config['advance_days'].max(skipna=True)
        return int(max_val) if pd.notna(max_val) else DEFAULT_MAX_ADVANCE_DAYS
    return DEFAULT_MAX_ADVANCE_DAYS


def _filter_future_orders(orders: pd.DataFrame, sim_date: pd.Timestamp) -> pd.DataFrame:
    """过滤未来订单。"""
    if not orders.empty and 'date' in orders.columns:
        orders['date'] = pd.to_datetime(orders['date'])
        return orders[orders['date'] >= sim_date].copy()
    return orders


def _deduplicate_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """订单去重。"""
    if orders.empty:
        return orders
    dedup_keys = [
        c for c in [
            'date', 'material', 'location', 'demand_type',
            'simulation_date', 'advance_days', 'quantity'
        ]
        if c in orders.columns
    ]
    if dedup_keys:
        return orders.drop_duplicates(subset=dedup_keys)
    return orders


def _normalize_orders(orders_df: pd.DataFrame) -> pd.DataFrame:
    """规范化订单。"""
    if orders_df.empty:
        return orders_df
    if 'quantity' in orders_df.columns:
        orders_df['quantity'] = orders_df['quantity'].astype(int)
    if 'simulation_date' not in orders_df.columns:
        orders_df['simulation_date'] = orders_df['date']
    return normalize_identifiers(orders_df)


def _generate_shipments(
    orders_df: pd.DataFrame,
    simulation_date: pd.Timestamp,
    orchestrator: Any,
    daily_for_orders: pd.DataFrame
) -> tuple:
    """生成发货与缺货。"""
    if orchestrator is None or orchestrator.shipment_valid == 0:
        return pd.DataFrame(), pd.DataFrame()

    shipment_df, cut_df = generate_shipment_with_inventory_check(
        orders_df, simulation_date, orchestrator, daily_for_orders, None
    )
    return shipment_df, cut_df


def _save_output(
    orders_df: pd.DataFrame,
    shipment_df: pd.DataFrame,
    cut_df: pd.DataFrame,
    supply_demand_df: pd.DataFrame,
    output_dir: str,
    simulation_date: pd.Timestamp,
    skip_file_output: bool
) -> Optional[str]:
    """保存输出文件。"""
    if skip_file_output:
        return None

    output_file = os.path.join(
        output_dir,
        f"module1_output_{simulation_date.strftime('%Y%m%d')}.xlsx"
    )
    save_module1_output_with_supply_demand(
        orders_df, shipment_df, supply_demand_df, output_file, cut_df
    )
    return output_file


def _apply_orders_consumption(
    forecast_df: pd.DataFrame,
    orders_df: pd.DataFrame
) -> pd.DataFrame:
    """应用订单消耗到预测。"""
    if forecast_df is None or forecast_df.empty:
        return pd.DataFrame(columns=['material', 'location', 'date', 'quantity'])

    base = forecast_df.groupby(
        ['material', 'location', 'date'], as_index=False
    )['quantity'].sum()
    consumed = base.copy()

    if orders_df is None or orders_df.empty:
        return consumed

    consumed = normalize_identifiers(consumed)
    orders_df = normalize_identifiers(orders_df.copy())
    offsets = [0, -1, -2, 1, 2, 3]

    idx_map = {}
    for idx, row in enumerate(consumed.itertuples()):
        key = (row.material, row.location, row.date)
        idx_map[key] = idx

    quantities = consumed['quantity'].values.copy().astype(float)

    ao_orders = orders_df[orders_df['demand_type'] == 'AO'].copy()
    if not ao_orders.empty:
        ao_orders = ao_orders.sort_values(
            by=['date', 'advance_days', 'quantity', 'simulation_date']
        )
        _apply_fast_consumption(ao_orders, quantities, idx_map, offsets)

    normal_orders = orders_df[orders_df['demand_type'] == 'normal'].copy()
    if not normal_orders.empty:
        normal_orders = normal_orders.sort_values(
            by=['date', 'quantity', 'simulation_date']
        )
        _apply_fast_consumption(normal_orders, quantities, idx_map, offsets)

    consumed['quantity'] = quantities.astype(int)

    return normalize_identifiers(consumed)


def _apply_fast_consumption(
    orders: pd.DataFrame,
    quantities: np.ndarray,
    idx_map: dict,
    offsets: list
) -> None:
    """快速应用订单消耗（直接修改quantities数组）。"""
    for r in orders.itertuples():
        if r.quantity <= 0:
            continue

        mat = r.material
        loc = r.location
        order_date = pd.to_datetime(r.date)
        remaining = int(r.quantity)

        for offset in offsets:
            if remaining <= 0:
                break
            target_date = order_date + pd.Timedelta(days=offset)
            key = (mat, loc, target_date)

            if key in idx_map:
                idx = idx_map[key]
                avail = int(quantities[idx])
                take = min(avail, remaining)
                quantities[idx] = avail - take
                remaining -= take


def generate_supply_demand_log_for_integration(
    demand_forecast: pd.DataFrame,
    consumed_forecast: pd.DataFrame,
    simulation_date: pd.Timestamp
) -> pd.DataFrame:
    """为集成模式生成供需日志。"""
    empty_cols = ['date', 'material', 'location', 'quantity', 'demand_element']

    if consumed_forecast.empty or 'date' not in consumed_forecast.columns:
        return pd.DataFrame(columns=empty_cols)

    future_cutoff = simulation_date + pd.Timedelta(days=M1_FUTURE_CUTOFF_DAYS)

    future_demand = consumed_forecast[
        (pd.to_datetime(consumed_forecast['date']) > simulation_date) &
        (pd.to_datetime(consumed_forecast['date']) <= future_cutoff)
    ].copy()

    if future_demand.empty:
        return pd.DataFrame(columns=empty_cols)

    future_demand['demand_element'] = 'forecast'
    supply_demand_log = future_demand[empty_cols].copy()

    return normalize_identifiers(supply_demand_log)
