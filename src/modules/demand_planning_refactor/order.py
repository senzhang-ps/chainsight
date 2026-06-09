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
import hashlib

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
    if _uses_weekly_order_generation(original_forecast):
        return _generate_daily_orders_from_weekly(
            sim_date,
            original_forecast,
            current_forecast,
            ao_config,
            order_calendar,
            forecast_error,
        )

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


def _uses_weekly_order_generation(forecast: pd.DataFrame) -> bool:
    """Return whether the forecast is a weekly order baseline."""
    if forecast is None or forecast.empty:
        return False
    return 'week' in forecast.columns and 'date' not in forecast.columns


def _generate_daily_orders_from_weekly(
    sim_date: pd.Timestamp,
    weekly_forecast: pd.DataFrame,
    current_forecast: pd.DataFrame,
    ao_config: pd.DataFrame,
    order_calendar: pd.DataFrame,
    forecast_error: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """按周生成订单量，再按有效订单日拆分并返回当前仿真日的订单。"""
    consumed_forecast = _prepare_consumed_forecast(current_forecast)
    sim_date = pd.to_datetime(sim_date).normalize()

    weekly_orders = generate_weekly_orders(
        weekly_forecast, ao_config, forecast_error
    )
    orders_df = split_weekly_orders_to_daily(
        weekly_orders,
        sim_date,
        order_calendar,
        current_forecast,
    )

    if not orders_df.empty:
        orders_df = _aggregate_orders(orders_df)

    consumed_forecast = consume_orders(orders_df, consumed_forecast)
    return orders_df, consumed_forecast


def generate_weekly_orders(
    weekly_forecast: pd.DataFrame,
    ao_config: pd.DataFrame,
    forecast_error: pd.DataFrame,
) -> pd.DataFrame:
    """Generate weekly AO / normal orders reconciled to a sampled total."""
    weekly_demand = _build_weekly_ml_demand(weekly_forecast)
    if weekly_demand.empty:
        return _empty_weekly_order_frame()

    base_seed = int(weekly_forecast.attrs.get('random_seed', 0) or 0)
    weekly_order_means = _split_weekly_demand_by_ao_means(
        weekly_demand, ao_config
    )
    weekly_total_orders = _generate_weekly_total_orders(
        weekly_demand, forecast_error, base_seed
    )
    provisional_orders = _generate_weekly_component_orders(
        weekly_order_means, forecast_error, base_seed
    )
    orders = _reconcile_weekly_component_orders_to_total(
        provisional_orders, weekly_total_orders
    )
    if orders.empty:
        return _empty_weekly_order_frame()

    return orders.sort_values(
        ['week', 'material', 'location', 'demand_type', 'advance_days']
    ).reset_index(drop=True)


def split_weekly_orders_to_daily(
    weekly_orders: pd.DataFrame,
    simulation_date: pd.Timestamp,
    order_calendar: pd.DataFrame,
    reference_daily_forecast: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """将周级订单拆到有效订单日，并返回指定 simulation_date 的日级订单。"""
    if weekly_orders is None or weekly_orders.empty:
        return _empty_order_frame()

    simulation_date = pd.to_datetime(simulation_date).normalize()
    calendar_dates = _normalized_calendar_dates(order_calendar)
    if simulation_date not in set(calendar_dates):
        return _empty_order_frame()

    week_start_map = _build_week_start_map(
        weekly_orders, calendar_dates, reference_daily_forecast
    )
    rows = []

    for order in weekly_orders.itertuples(index=False):
        week = int(order.week)
        week_start = week_start_map[week]
        week_end = week_start + pd.Timedelta(days=7)
        valid_order_dates = [
            d for d in calendar_dates if week_start <= d < week_end
        ]
        if simulation_date not in valid_order_dates:
            continue

        split_quantities = _split_integer_quantity_by_days(
            int(order.quantity), len(valid_order_dates)
        )
        split_idx = valid_order_dates.index(simulation_date)
        split_qty = int(split_quantities[split_idx])
        if split_qty <= 0:
            continue

        advance_days = int(order.advance_days)
        demand_date = simulation_date + pd.Timedelta(days=advance_days)
        rows.append({
            'date': demand_date,
            'material': str(order.material),
            'location': str(order.location),
            'demand_type': str(order.demand_type),
            'quantity': split_qty,
            'simulation_date': simulation_date,
            'advance_days': advance_days,
        })

    if not rows:
        return _empty_order_frame()

    return normalize_identifiers(pd.DataFrame(rows))


def _build_weekly_ml_demand(weekly_forecast: pd.DataFrame) -> pd.DataFrame:
    """按 material / location / week 聚合周需求，并稳定排序。"""
    if weekly_forecast is None or weekly_forecast.empty:
        return pd.DataFrame(columns=['material', 'location', 'week', 'quantity'])

    required = ['material', 'location', 'week', 'quantity']
    missing = [c for c in required if c not in weekly_forecast.columns]
    if missing:
        return pd.DataFrame(columns=required)

    df = normalize_identifiers(weekly_forecast.copy())
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0).clip(lower=0)
    df['week'] = pd.to_numeric(df['week'], errors='coerce').fillna(0).astype(int)
    weekly = df.groupby(
        ['material', 'location', 'week'], as_index=False
    )['quantity'].sum()
    weekly['quantity'] = weekly['quantity'].astype(int)
    return weekly.sort_values(['week', 'material', 'location']).reset_index(drop=True)


def _generate_weekly_total_orders(
    weekly_demand: pd.DataFrame,
    forecast_error: pd.DataFrame,
    base_seed: int,
) -> pd.DataFrame:
    """基于 weekly forecast 和 forecast error(CoV)生成整周订单总量。"""
    if weekly_demand.empty:
        return pd.DataFrame(columns=['material', 'location', 'week', 'quantity'])

    cov = _forecast_error_for_weekly_total(forecast_error)
    total = weekly_demand.merge(cov, on=['material', 'location'], how='left')
    mean_qty = total['quantity']
    std_qty = mean_qty * total['error_std_percent'].fillna(0)
    keys = [
        ('weekly_total', r.material, r.location, int(r.week))
        for r in total.itertuples(index=False)
    ]

    return pd.DataFrame({
        'material': total['material'].astype(str),
        'location': total['location'].astype(str),
        'week': total['week'].astype(int),
        'quantity': _stable_normal_quantities(mean_qty, std_qty, keys, base_seed),
    })


def _split_weekly_demand_by_ao_means(
    weekly_demand: pd.DataFrame,
    ao_config: pd.DataFrame,
) -> pd.DataFrame:
    empty = _empty_weekly_order_mean_frame()
    if weekly_demand.empty:
        return empty

    ao_cols = ['material', 'location', 'advance_days', 'ao_percent']
    if ao_config is not None and not ao_config.empty:
        ao_cfg = ao_config[ao_cols].copy()
        ao_cfg['advance_days'] = pd.to_numeric(
            ao_cfg['advance_days'], errors='coerce'
        ).fillna(0).astype(int)
        ao_cfg['ao_percent'] = pd.to_numeric(
            ao_cfg['ao_percent'], errors='coerce'
        ).fillna(0).clip(lower=0)
        ao_cfg = normalize_identifiers(ao_cfg).drop_duplicates(subset=ao_cols)
    else:
        ao_cfg = pd.DataFrame(columns=ao_cols)

    rows = []
    for demand in weekly_demand.itertuples(index=False):
        weekly_qty = float(demand.quantity)
        if weekly_qty <= 0:
            continue

        ao_lines = ao_cfg[
            (ao_cfg['material'] == str(demand.material)) &
            (ao_cfg['location'] == str(demand.location))
        ].sort_values(['advance_days']).reset_index(drop=True)

        ao_weight_sum = float(ao_lines['ao_percent'].astype(float).sum())
        ao_total_share = min(ao_weight_sum, 1.0)
        normal_weight = 1.0 - ao_total_share

        ao_total_mean = weekly_qty * ao_total_share
        if ao_total_mean > 0 and ao_weight_sum > 0:
            for line in ao_lines.itertuples(index=False):
                share_within_ao = float(line.ao_percent) / ao_weight_sum
                mean_qty = ao_total_mean * share_within_ao
                if mean_qty <= 0:
                    continue
                rows.append({
                    'material': str(demand.material),
                    'location': str(demand.location),
                    'week': int(demand.week),
                    'demand_type': 'AO',
                    'advance_days': int(line.advance_days),
                    'mean_quantity': mean_qty,
                })

        normal_mean = weekly_qty * normal_weight
        if normal_mean > 0:
            rows.append({
                'material': str(demand.material),
                'location': str(demand.location),
                'week': int(demand.week),
                'demand_type': 'normal',
                'advance_days': 0,
                'mean_quantity': normal_mean,
            })

    if not rows:
        return empty
    return normalize_identifiers(pd.DataFrame(rows)[empty.columns])


def _generate_weekly_component_orders(
    weekly_order_means: pd.DataFrame,
    forecast_error: pd.DataFrame,
    base_seed: int,
) -> pd.DataFrame:
    empty = _empty_weekly_order_sample_frame()
    if weekly_order_means.empty:
        return empty

    component_cov = _forecast_error_for_weekly_components(forecast_error)
    components = weekly_order_means.copy()
    components['_order_type'] = components['demand_type'].astype(str).str.lower()
    components = components.merge(
        component_cov, on=['material', 'location', '_order_type'], how='left'
    )

    mean_qty = pd.to_numeric(
        components['mean_quantity'], errors='coerce'
    ).fillna(0).clip(lower=0)
    std_qty = mean_qty * components['error_std_percent'].fillna(0)
    keys = [
        (
            'weekly_component',
            r.demand_type,
            r.material,
            r.location,
            int(r.week),
            int(r.advance_days),
        )
        for r in components.itertuples(index=False)
    ]

    result = components[[
        'material', 'location', 'week', 'demand_type',
        'advance_days', 'mean_quantity'
    ]].copy()
    result['quantity'] = _stable_normal_quantities(
        mean_qty, std_qty, keys, base_seed
    )
    return normalize_identifiers(result[empty.columns])


def _reconcile_weekly_component_orders_to_total(
    provisional_orders: pd.DataFrame,
    weekly_total_orders: pd.DataFrame,
) -> pd.DataFrame:
    empty = _empty_weekly_order_frame()
    if weekly_total_orders.empty:
        return empty

    rows = []
    for total in weekly_total_orders.itertuples(index=False):
        total_qty = int(total.quantity)
        if total_qty <= 0:
            continue

        components = provisional_orders[
            (provisional_orders['material'] == str(total.material)) &
            (provisional_orders['location'] == str(total.location)) &
            (provisional_orders['week'].astype(int) == int(total.week))
        ].copy()

        if components.empty:
            rows.append({
                'material': str(total.material),
                'location': str(total.location),
                'week': int(total.week),
                'demand_type': 'normal',
                'advance_days': 0,
                'quantity': total_qty,
            })
            continue

        weights = pd.to_numeric(
            components['quantity'], errors='coerce'
        ).fillna(0).clip(lower=0).astype(float).tolist()
        if sum(weights) <= 0:
            weights = pd.to_numeric(
                components['mean_quantity'], errors='coerce'
            ).fillna(0).clip(lower=0).astype(float).tolist()

        quantities = _allocate_integer_quantity_by_weights(total_qty, weights)
        for qty, component in zip(quantities, components.itertuples(index=False)):
            qty = int(qty)
            if qty <= 0:
                continue
            rows.append({
                'material': str(component.material),
                'location': str(component.location),
                'week': int(component.week),
                'demand_type': str(component.demand_type),
                'advance_days': int(component.advance_days),
                'quantity': qty,
            })

    if not rows:
        return empty
    return normalize_identifiers(pd.DataFrame(rows)[empty.columns])


def _forecast_error_for_weekly_total(forecast_error: pd.DataFrame) -> pd.DataFrame:
    if forecast_error is None or forecast_error.empty:
        return pd.DataFrame(columns=['material', 'location', 'error_std_percent'])

    fe = normalize_identifiers(forecast_error.copy())
    if 'order_type' not in fe.columns:
        return fe.groupby(
            ['material', 'location'], as_index=False
        )['error_std_percent'].max()

    fe['_order_type'] = fe['order_type'].astype(str).str.lower()
    fe['error_std_percent'] = pd.to_numeric(
        fe['error_std_percent'], errors='coerce'
    ).fillna(0)

    rows = []
    for (material, location), group in fe.groupby(['material', 'location']):
        weekly_total = group[group['_order_type'].isin(['total', 'weekly'])]
        if not weekly_total.empty:
            source = weekly_total
        else:
            normal = group[group['_order_type'] == 'normal']
            source = normal if not normal.empty else group
        rows.append({
            'material': material,
            'location': location,
            'error_std_percent': float(source['error_std_percent'].max()),
        })

    return pd.DataFrame(rows, columns=['material', 'location', 'error_std_percent'])


def _forecast_error_for_weekly_components(forecast_error: pd.DataFrame) -> pd.DataFrame:
    columns = ['material', 'location', '_order_type', 'error_std_percent']
    if forecast_error is None or forecast_error.empty:
        return pd.DataFrame(columns=columns)

    required = {'material', 'location', 'order_type', 'error_std_percent'}
    if not required.issubset(forecast_error.columns):
        return pd.DataFrame(columns=columns)

    fe = normalize_identifiers(forecast_error.copy())
    fe['_order_type'] = fe['order_type'].astype(str).str.lower()
    fe = fe[fe['_order_type'].isin(['ao', 'normal'])]
    if fe.empty:
        return pd.DataFrame(columns=columns)

    return fe.groupby(
        ['material', 'location', '_order_type'], as_index=False
    )['error_std_percent'].max()


def _allocate_integer_quantity_by_weights(quantity: int, weights: list[float]) -> list[int]:
    if not weights:
        return [max(0, int(quantity))]

    quantity = max(0, int(quantity))
    clean_weights = [max(0.0, float(w)) for w in weights]
    total_weight = sum(clean_weights)
    if total_weight <= 0:
        result = [0 for _ in clean_weights]
        result[-1] = quantity
        return result

    raw = [quantity * w / total_weight for w in clean_weights]
    floors = [int(np.floor(v)) for v in raw]
    remainder = quantity - sum(floors)
    order = sorted(
        range(len(raw)),
        key=lambda idx: (raw[idx] - floors[idx], -idx),
        reverse=True,
    )
    for idx in order[:remainder]:
        floors[idx] += 1
    return floors


def _stable_normal_quantities(
    means: pd.Series,
    stds: pd.Series,
    keys: list,
    base_seed: int,
) -> np.ndarray:
    quantities = []
    for mean, std, key in zip(means, stds, keys):
        mean = float(mean)
        std = float(std) if pd.notna(std) else 0.0
        if std <= 0:
            quantities.append(max(0, int(round(mean))))
            continue
        rng = np.random.default_rng(_stable_seed(base_seed, key))
        quantities.append(max(0, int(round(rng.normal(mean, std)))))
    return np.asarray(quantities, dtype=int)


def _stable_seed(base_seed: int, key: tuple) -> int:
    raw = "|".join([str(base_seed), *[str(v) for v in key]])
    digest = hashlib.sha256(raw.encode('utf-8')).hexdigest()
    return int(digest[:16], 16) % (2 ** 32)


def _build_week_start_map(
    weekly_orders: pd.DataFrame,
    calendar_dates: list[pd.Timestamp],
    reference_daily_forecast: Optional[pd.DataFrame],
) -> dict[int, pd.Timestamp]:
    if reference_daily_forecast is not None and not reference_daily_forecast.empty:
        if {'week', 'date'}.issubset(reference_daily_forecast.columns):
            ref = reference_daily_forecast.copy()
            ref['date'] = pd.to_datetime(ref['date']).dt.normalize()
            return {
                int(r.week): pd.Timestamp(r.date)
                for r in ref.groupby('week', as_index=False)['date'].min().itertuples(index=False)
            }

    weeks = sorted(pd.to_numeric(weekly_orders['week'], errors='coerce').dropna().astype(int).unique())
    if not weeks:
        return {}

    first_calendar_date = min(calendar_dates) if calendar_dates else pd.Timestamp.today().normalize()
    min_week = min(weeks)
    base_start = first_calendar_date - pd.Timedelta(days=(min_week - 1) * 7)
    return {
        week: base_start + pd.Timedelta(days=(week - min_week) * 7)
        for week in weeks
    }


def _split_integer_quantity_by_days(quantity: int, days: int) -> list[int]:
    if days <= 0:
        return []
    quantity = max(0, int(quantity))
    base = quantity // days
    remainder = quantity % days
    return [base + (1 if i < remainder else 0) for i in range(days)]


def _normalized_calendar_dates(order_calendar: pd.DataFrame) -> list[pd.Timestamp]:
    if order_calendar is None or order_calendar.empty or 'date' not in order_calendar.columns:
        return []
    dates = pd.to_datetime(order_calendar['date']).dt.normalize().dropna().drop_duplicates()
    return sorted(pd.Timestamp(d) for d in dates)

def _prepare_consumed_forecast(current_forecast: pd.DataFrame) -> pd.DataFrame:
    if current_forecast is None or current_forecast.empty:
        return pd.DataFrame(columns=['material', 'location', 'date', 'quantity'])
    if not {'material', 'location', 'date', 'quantity'}.issubset(current_forecast.columns):
        return current_forecast.copy()
    return current_forecast.groupby(
        ['material', 'location', 'date'], as_index=False
    )['quantity'].sum()


def _empty_order_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        'date', 'material', 'location', 'demand_type',
        'quantity', 'simulation_date', 'advance_days'
    ])


def _empty_weekly_order_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        'material', 'location', 'week', 'demand_type',
        'advance_days', 'quantity'
    ])


def _empty_weekly_order_mean_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        'material', 'location', 'week', 'demand_type',
        'advance_days', 'mean_quantity'
    ])


def _empty_weekly_order_sample_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        'material', 'location', 'week', 'demand_type',
        'advance_days', 'mean_quantity', 'quantity'
    ])


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
