"""Module1 订单生成与管理。

本模块提供订单生成相关功能。

主要函数：
- generate_daily_orders: 生成单日订单
- generate_quantity_with_percent_error: 基于误差生成数量
- consume_forecast_ao_logic: AO预测消耗逻辑
- consume_forecast_normal_logic: Normal预测消耗逻辑
"""

import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

from ...utils.normalization import normalize_identifiers
from .consume import consume_orders


def generate_daily_orders(
    sim_date: pd.Timestamp,
    original_forecast: pd.DataFrame,
    current_forecast: pd.DataFrame,
    ao_config: pd.DataFrame,
    order_calendar: pd.DataFrame,
    forecast_error: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """生成单日订单（含AO与Normal），并消耗预测。

    处理逻辑:
        1. 检查是否为订单日
        2. 计算7天窗口平均需求
        3. 生成AO订单和Normal订单
        4. 执行订单消耗

    参数:
        sim_date: 仿真日期。
        original_forecast: 原始预测视图。
        current_forecast: 当前预测视图。
        ao_config: AO配置DataFrame。
        order_calendar: 订单日历DataFrame。
        forecast_error: 预测误差DataFrame。

    返回:
        (orders_df, consumed_forecast)元组。
    """
    # 检查是否为订单日
    is_order_day = not order_calendar[
        order_calendar['date'] == sim_date
    ].empty
    if not is_order_day:
        return pd.DataFrame(), current_forecast

    t0 = time.perf_counter()

    # 预测视图按键聚合
    current_forecast = current_forecast.groupby(
        ['material', 'location', 'date'], as_index=False
    )['quantity'].sum()
    consumed_forecast = current_forecast.copy()

    # 计算平均需求
    ml_avg_demand = _compute_ml_avg_demand(original_forecast, sim_date)
    if ml_avg_demand.empty:
        return pd.DataFrame(), consumed_forecast

    elapsed = time.perf_counter() - t0

    # 生成AO和Normal订单
    t1 = time.perf_counter()
    ao_orders_df = _generate_ao_orders(
        ml_avg_demand, ao_config, forecast_error, sim_date
    )
    t2 = time.perf_counter()
    normal_orders_df = _generate_normal_orders(
        ml_avg_demand, ao_config, forecast_error, sim_date
    )

    # 合并订单
    orders_df = pd.concat([ao_orders_df, normal_orders_df], ignore_index=True)
    if not orders_df.empty:
        orders_df = _aggregate_orders(orders_df)

    ao_time = t2 - t1
    normal_time = time.perf_counter() - t2
    total_time = time.perf_counter() - t0

    # 执行消耗
    consumed_forecast = consume_orders(orders_df, consumed_forecast)

    return orders_df, consumed_forecast


def _compute_ml_avg_demand(
    original_forecast: pd.DataFrame,
    sim_date: pd.Timestamp,
    window_days: int = 7
) -> pd.DataFrame:
    """计算物料-地点粒度的平均日需求。

    参数:
        original_forecast: 原始预测DataFrame。
        sim_date: 仿真日期。
        window_days: 窗口天数。

    返回:
        包含[material, location, avg_daily_demand]的DataFrame。
    """
    end_date = sim_date + pd.Timedelta(days=window_days)
    windowed = original_forecast[
        (original_forecast['date'] >= sim_date) &
        (original_forecast['date'] < end_date)
    ].copy()

    if not windowed.empty:
        ml_avg = windowed.groupby(
            ['material', 'location'], as_index=False
        )['quantity'].mean()
        ml_avg.columns = ['material', 'location', 'avg_daily_demand']
        # 确保确定性排序，使np.random.normal()结果与输入数据顺序无关
        ml_avg = ml_avg.sort_values(['material', 'location']).reset_index(drop=True)
        return ml_avg

    # 回退至1天窗口
    short_end = sim_date + pd.Timedelta(days=1)
    short_windowed = original_forecast[
        (original_forecast['date'] >= sim_date) &
        (original_forecast['date'] < short_end)
    ].copy()

    if not short_windowed.empty:
        ml_avg = short_windowed.groupby(
            ['material', 'location'], as_index=False
        )['quantity'].mean()
        ml_avg.columns = ['material', 'location', 'avg_daily_demand']
        # 确保确定性排序，使np.random.normal()结果与输入数据顺序无关
        ml_avg = ml_avg.sort_values(['material', 'location']).reset_index(drop=True)
        return ml_avg

    return pd.DataFrame(columns=['material', 'location', 'avg_daily_demand'])


def _generate_ao_orders(
    ml_avg_demand: pd.DataFrame,
    ao_config: pd.DataFrame,
    forecast_error: pd.DataFrame,
    sim_date: pd.Timestamp
) -> pd.DataFrame:
    """生成AO订单。

    参数:
        ml_avg_demand: 物料-地点平均需求DataFrame。
        ao_config: AO配置DataFrame。
        forecast_error: 预测误差DataFrame。
        sim_date: 仿真日期。

    返回:
        AO订单DataFrame。
    """
    empty_cols = [
        'date', 'material', 'location', 'demand_type',
        'quantity', 'simulation_date', 'advance_days'
    ]
    if ao_config.empty or ml_avg_demand.empty:
        return pd.DataFrame(columns=empty_cols)

    # AO 配置去重
    ao_cols = ['material', 'location', 'advance_days', 'ao_percent']
    ao_cfg = ao_config[ao_cols].drop_duplicates()
    # 🔧 标准化AO配置中的标识符以确保merge键类型一致
    ao_cfg = normalize_identifiers(ao_cfg)

    # 合并平均需求
    ao_lines = ml_avg_demand.merge(
        ao_cfg, on=['material', 'location'], how='left'
    )
    ao_lines = ao_lines.dropna(subset=['ao_percent'])

    if ao_lines.empty:
        return pd.DataFrame(columns=empty_cols)

    # 计算AO日均需求
    ao_lines['ao_daily_avg'] = (
        ao_lines['avg_daily_demand'] * ao_lines['ao_percent']
    )

    # 获取AO误差配置
    # 🔧 标准化forecast_error中的标识符以确保merge键类型一致
    fe_normalized = normalize_identifiers(forecast_error.copy())
    fe = fe_normalized.groupby(
        ['material', 'location', 'order_type'], as_index=False
    )['error_std_percent'].max()
    fe_ao = fe[fe['order_type'] == 'AO'][
        ['material', 'location', 'error_std_percent']
    ]
    ao_e = ao_lines.merge(fe_ao, on=['material', 'location'], how='left')


    # 向量化生成数量
    ao_abs_std = ao_e['ao_daily_avg'] * ao_e['error_std_percent'].fillna(0)
    ao_qty = np.maximum(
        0, np.round(np.random.normal(ao_e['ao_daily_avg'], ao_abs_std))
    ).astype(int)
    ao_dates = sim_date + pd.to_timedelta(
        ao_e['advance_days'].astype(int), unit='D'
    )

    return pd.DataFrame({
        'date': ao_dates,
        'material': ao_e['material'].astype(str),
        'location': ao_e['location'].astype(str),
        'demand_type': 'AO',
        'quantity': ao_qty,
        'simulation_date': sim_date,
        'advance_days': ao_e['advance_days'].astype(int)
    })


def _generate_normal_orders(
    ml_avg_demand: pd.DataFrame,
    ao_config: pd.DataFrame,
    forecast_error: pd.DataFrame,
    sim_date: pd.Timestamp
) -> pd.DataFrame:
    """生成Normal订单。

    参数:
        ml_avg_demand: 物料-地点平均需求DataFrame。
        ao_config: AO配置DataFrame。
        forecast_error: 预测误差DataFrame。
        sim_date: 仿真日期。

    返回:
        Normal订单DataFrame。
    """
    empty_cols = [
        'date', 'material', 'location', 'demand_type',
        'quantity', 'simulation_date', 'advance_days'
    ]
    if ml_avg_demand.empty:
        return pd.DataFrame(columns=empty_cols)

    # 计算总AO百分比
    ao_cols = ['material', 'location', 'advance_days', 'ao_percent']
    if not ao_config.empty:
        ao_cfg = ao_config[ao_cols].drop_duplicates()
        # 🔧 标准化AO配置中的标识符以确保merge键类型一致
        ao_cfg = normalize_identifiers(ao_cfg)
    else:
        ao_cfg = pd.DataFrame(columns=ao_cols)

    total_ao = ao_cfg.groupby(
        ['material', 'location'], as_index=False
    )['ao_percent'].sum()

    # 计算Normal日均需求
    normal = ml_avg_demand.merge(
        total_ao, on=['material', 'location'], how='left'
    )
    normal['ao_percent'] = normal['ao_percent'].fillna(0).clip(0, 1)
    normal['normal_daily_avg'] = (
        normal['avg_daily_demand'] * (1 - normal['ao_percent'])
    )
    normal = normal[normal['normal_daily_avg'] > 0]

    if normal.empty:
        return pd.DataFrame(columns=empty_cols)

    # 获取Normal误差配置
    # 🔧 标准化forecast_error中的标识符以确保merge键类型一致
    fe_normalized = normalize_identifiers(forecast_error.copy())
    fe = fe_normalized.groupby(
        ['material', 'location', 'order_type'], as_index=False
    )['error_std_percent'].max()
    fe_n = fe[fe['order_type'] == 'normal'][
        ['material', 'location', 'error_std_percent']
    ]
    n_e = normal.merge(fe_n, on=['material', 'location'], how='left')


    # 向量化生成数量
    n_abs_std = n_e['normal_daily_avg'] * n_e['error_std_percent'].fillna(0)
    normal_qty = np.maximum(
        0, np.round(np.random.normal(n_e['normal_daily_avg'], n_abs_std))
    ).astype(int)

    return pd.DataFrame({
        'date': pd.Series([sim_date] * len(n_e)),
        'material': n_e['material'].astype(str),
        'location': n_e['location'].astype(str),
        'demand_type': 'normal',
        'quantity': normal_qty,
        'simulation_date': pd.Series([sim_date] * len(n_e)),
        'advance_days': 0
    })


def _aggregate_orders(orders_df: pd.DataFrame) -> pd.DataFrame:
    """聚合订单并规范化。

    参数:
        orders_df: 订单DataFrame。

    返回:
        聚合后的订单DataFrame。
    """
    group_cols = [
        'date', 'material', 'location', 'demand_type',
        'simulation_date', 'advance_days'
    ]
    orders_df = orders_df.groupby(
        group_cols, as_index=False
    )['quantity'].sum()
    orders_df['quantity'] = orders_df['quantity'].astype(int)
    return normalize_identifiers(orders_df)


def generate_quantity_with_percent_error(
    mean_qty: float,
    material: str,
    location: str,
    order_type: str,
    forecast_error: pd.DataFrame
) -> int:
    """基于百分比误差生成订单数量。

    使用截断正态分布（下界0）生成数量。

    参数:
        mean_qty: 生成的平均数量。
        material: 物料标识。
        location: 地点标识。
        order_type: 订单类型。
        forecast_error: 误差配置DataFrame。

    返回:
        生成的非负整数数量。
    """
    # 查找误差配置
    mask = (
        (forecast_error['material'] == material) &
        (forecast_error['location'] == location) &
        (forecast_error['order_type'] == order_type)
    )
    error_config = forecast_error[mask]

    if error_config.empty:
        # 回退到旧格式
        return _fallback_quantity_generation(
            mean_qty, material, location, forecast_error
        )

    # 使用百分比误差
    error_percent = _get_error_percent(error_config)
    abs_std = mean_qty * error_percent

    if abs_std <= 0:
        return max(0, int(round(mean_qty)))

    # 生成截断正态分布
    a = (0 - mean_qty) / abs_std
    value = truncnorm.rvs(a, np.inf, loc=mean_qty, scale=abs_std)

    return max(0, int(round(value)))


def _fallback_quantity_generation(
    mean_qty: float,
    material: str,
    location: str,
    forecast_error: pd.DataFrame
) -> int:
    """回退到旧格式的数量生成。

    参数:
        mean_qty: 平均数量。
        material: 物料标识。
        location: 地点标识。
        forecast_error: 误差配置。

    返回:
        生成的数量。
    """
    mask_old = (
        (forecast_error['material'] == material) &
        (forecast_error['location'] == location)
    )
    error_config_old = forecast_error[mask_old]

    if not error_config_old.empty and 'error_std' in error_config_old.columns:
        error_std = float(error_config_old['error_std'].iloc[0])
        if error_std > 0:
            error = np.random.normal(0, error_std)
            return max(0, int(round(mean_qty + error)))

    return max(0, int(round(mean_qty)))


def _get_error_percent(error_config: pd.DataFrame) -> float:
    """从配置中提取误差百分比。

    参数:
        error_config: 误差配置DataFrame。

    返回:
        误差百分比值。
    """
    if 'error_std_percent' in error_config.columns:
        return float(error_config['error_std_percent'].iloc[0])
    return 0.0


def consume_forecast_ao_logic(
    forecast_df: pd.DataFrame,
    material: str,
    location: str,
    order_date: pd.Timestamp,
    consume_qty: int
) -> pd.DataFrame:
    """应用AO预测消耗逻辑（固定窗口）。

    消耗窗口: [order_date, order_date-1, order_date-2,
              order_date+1, order_date+2, order_date+3]

    参数:
        forecast_df: 预测DataFrame。
        material: 物料标识。
        location: 地点标识。
        order_date: 订单日期。
        consume_qty: 要消耗的数量。

    返回:
        消耗后的预测DataFrame。
    """
    if consume_qty <= 0:
        return forecast_df

    offsets = [0, -1, -2, 1, 2, 3]
    consumption_dates = [
        order_date + pd.Timedelta(days=offset) for offset in offsets
    ]

    result_forecast = forecast_df.copy()
    remaining = consume_qty

    for date in consumption_dates:
        if remaining <= 0:
            break

        mask = (
            (result_forecast['material'] == material) &
            (result_forecast['location'] == location) &
            (result_forecast['date'] == date)
        )
        matching_rows = result_forecast[mask]

        if not matching_rows.empty:
            idx = matching_rows.index[0]
            available = int(result_forecast.at[idx, 'quantity'])
            actual = min(available, remaining)
            result_forecast.at[idx, 'quantity'] = max(0, available - actual)
            remaining -= actual

    return result_forecast


def consume_forecast_normal_logic(
    forecast_df: pd.DataFrame,
    material: str,
    location: str,
    order_date: pd.Timestamp,
    consume_qty: int
) -> pd.DataFrame:
    """应用Normal预测消耗逻辑（仅当天）。

    参数:
        forecast_df: 预测DataFrame。
        material: 物料标识。
        location: 地点标识。
        order_date: 订单日期。
        consume_qty: 要消耗的数量。

    返回:
        消耗后的预测DataFrame。
    """
    if consume_qty <= 0:
        return forecast_df

    result_forecast = forecast_df.copy()

    mask = (
        (result_forecast['material'] == material) &
        (result_forecast['location'] == location) &
        (result_forecast['date'] == order_date)
    )
    matching_rows = result_forecast[mask]

    if not matching_rows.empty:
        idx = matching_rows.index[0]
        available = int(result_forecast.at[idx, 'quantity'])
        actual = min(available, consume_qty)
        result_forecast.at[idx, 'quantity'] = max(0, available - actual)

    return result_forecast
