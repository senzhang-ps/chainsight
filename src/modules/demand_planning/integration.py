"""Module1 集成模式主入口。

本模块提供集成模式的主要功能。

主要函数：
- run_daily_order_generation: 集成模式主入口
- generate_supply_demand_log_for_integration: 生成供需日志
"""

import os
import time
from typing import Any, Optional

import numpy as np
import pandas as pd

from .constants import DEFAULT_MAX_ADVANCE_DAYS
from ...utils.defaults import M1_FUTURE_CUTOFF_DAYS
from ...utils.normalization import normalize_identifiers
from .dps import apply_dps, apply_supply_choice
from .forecast import expand_forecast_to_days_integer_split
from .order import generate_daily_orders
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
    """集成模式主入口：生成指定日期的订单与发货。

    参数:
        config_dict: 配置字典。
        simulation_date: 仿真日期。
        output_dir: 输出目录。
        orchestrator: 编排器对象。
        skip_file_output: 是否跳过文件输出。
        previous_orders_df: 历史订单数据（可选，用于DB模式跳过文件读取）。

    返回:
        包含订单、发货、缺货、供需日志的字典。
    """
    try:
        # 1) 校验配置
        configs = _validate_config(config_dict)
        demand_forecast, forecast_error, order_calendar, ao_config = configs

        # 2) 准备日度预测
        daily_for_orders, daily_for_supply = _prepare_forecasts(
            config_dict, demand_forecast, orchestrator
        )

        # 3) 生成当日订单
        t0 = time.perf_counter()
        today_orders_df, _ = generate_daily_orders(
            simulation_date, daily_for_orders, daily_for_orders,
            ao_config, order_calendar, forecast_error
        )
        elapsed = time.perf_counter() - t0

        # 4) 合并历史订单（用于内部计算，如发货检查）
        all_orders_df = _merge_with_history(
            output_dir, simulation_date, today_orders_df, ao_config,
            previous_orders_df=previous_orders_df
        )

        # 5) 生成发货（使用累积订单）
        shipment_df, cut_df = _generate_shipments(
            all_orders_df, simulation_date, orchestrator, daily_for_orders
        )

        # 6) 生成供需日志
        t3 = time.perf_counter()
        consumed_supply = _apply_orders_consumption(daily_for_supply, today_orders_df)
        supply_demand_df = generate_supply_demand_log_for_integration(
            daily_for_supply, consumed_supply, simulation_date
        )
        elapsed = time.perf_counter() - t3

        # 7) 保存输出（使用累积订单，与Dev版本兼容）
        output_file = _save_output(
            all_orders_df, shipment_df, cut_df, supply_demand_df,
            output_dir, simulation_date, skip_file_output
        )

        # 8) 生成Summary（供数据库模式使用，与Excel保持一致：使用累计订单）
        summary_df = _build_summary_df(all_orders_df, shipment_df, cut_df, supply_demand_df)

        # `orders_df`：累积订单（`all_orders_df`），与本地 Excel 输出保持一致
        # 本地 _save_output 使用 all_orders_df 写入 `OrderLog` sheet，
        # 数据库模式也应写入累积订单，确保数据库与本地结果完全一致。
        return {
            'orders_df': all_orders_df,  # ✅ 累积订单，与本地 Excel `OrderLog` 一致
            'shipment_df': shipment_df,
            'cut_df': cut_df,
            'supply_demand_df': supply_demand_df,
            'summary_df': summary_df,
            'output_file': output_file,
            'all_orders_for_next_day': all_orders_df  # 保留累积订单供下一天使用
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return _empty_result()


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
    """构建汇总DataFrame（供数据库模式使用）。

    参数:
        orders_df: 订单数据。
        shipment_df: 发货数据。
        cut_df: 缺货数据。
        supply_demand_df: 供需日志数据。

    返回:
        汇总DataFrame。
    """
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

    if demand_forecast.empty:
        raise ValueError("缺少必需的配置数据：M1_DemandForecast")
    if order_calendar.empty:
        raise ValueError("缺少必需的配置数据：M1_OrderCalendar")
    if ao_config.empty:
        raise ValueError("缺少必需的配置数据：M1_AOConfig")
    if forecast_error.empty:
        raise ValueError("缺少必需的配置数据：M1_ForecastError")

    order_calendar['date'] = pd.to_datetime(order_calendar['date'])

    # 🔧 将 DemandForecast 中的负值 quantity 修正为 0
    if 'quantity' in demand_forecast.columns:
        demand_forecast['quantity'] = demand_forecast['quantity'].clip(lower=0)

    # 🔧 确保所有配置的material列为string类型，避免merge时类型不匹配
    demand_forecast = normalize_identifiers(demand_forecast)
    forecast_error = normalize_identifiers(forecast_error)
    ao_config = normalize_identifiers(ao_config)
    
    return demand_forecast, forecast_error, order_calendar, ao_config


def _prepare_forecasts(
    config_dict: dict,
    demand_forecast: pd.DataFrame,
    orchestrator: Any
) -> tuple:
    """准备日度预测。"""
    if orchestrator is None or not hasattr(orchestrator, 'start_date'):
        raise ValueError("orchestrator.start_date 必须提供")

    if 'week' not in demand_forecast.columns:
        return demand_forecast.copy(), demand_forecast.copy()

    dps_config = config_dict.get('M1_DPSConfig', pd.DataFrame())
    supply_choice = config_dict.get('M1_SupplyChoiceConfig', pd.DataFrame())

    dps_cfg = dps_config if dps_config is not None else pd.DataFrame()
    demand_dps = apply_dps(demand_forecast, dps_cfg)

    sc_cfg = supply_choice if supply_choice is not None else pd.DataFrame()
    demand_dps_sc = apply_supply_choice(demand_dps, sc_cfg)

    sim_start = pd.to_datetime(orchestrator.start_date).normalize()

    max_week_dps = _get_max_week(demand_dps)
    daily_for_orders = expand_forecast_to_days_integer_split(
        demand_dps, sim_start, max_week_dps
    )

    max_week_sc = _get_max_week(demand_dps_sc)
    daily_for_supply = expand_forecast_to_days_integer_split(
        demand_dps_sc, sim_start, max_week_sc
    )

    return daily_for_orders, daily_for_supply


def _get_max_week(df: pd.DataFrame) -> int:
    """获取最大周数。"""
    if df.empty or 'week' not in df.columns:
        return 1
    return int(df['week'].max())


def _merge_with_history(
    output_dir: str,
    simulation_date: pd.Timestamp,
    today_orders_df: pd.DataFrame,
    ao_config: pd.DataFrame,
    previous_orders_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """合并历史订单与当日订单。
    
    参数:
        output_dir: 输出目录。
        simulation_date: 仿真日期。
        today_orders_df: 当日订单。
        ao_config: AO配置。
        previous_orders_df: 历史订单数据（可选，用于DB模式跳过文件读取）。
    """
    max_advance = _get_max_advance_days(ao_config)

    t1 = time.perf_counter()
    
    # 🦆 如果提供了内存中的历史订单数据（DB模式），则直接使用，跳过文件读取
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
    if orchestrator is None:
        return pd.DataFrame(), pd.DataFrame()

    t2 = time.perf_counter()
    shipment_df, cut_df = generate_shipment_with_inventory_check(
        orders_df, simulation_date, orchestrator, daily_for_orders, None
    )
    elapsed = time.perf_counter() - t2
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
    """应用订单消耗到预测（优化版：使用字典索引替代DataFrame遍历）。"""
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

    # 构建 (material, location, date) -> row_index 的字典映射
    # 同时构建可变的 quantities 数组用于快速更新
    idx_map = {}
    for idx, row in enumerate(consumed.itertuples()):
        key = (row.material, row.location, row.date)
        idx_map[key] = idx
    
    quantities = consumed['quantity'].values.copy().astype(float)

    # AO 订单消耗
    ao_orders = orders_df[orders_df['demand_type'] == 'AO'].copy()
    if not ao_orders.empty:
        ao_orders = ao_orders.sort_values(
            by=['date', 'advance_days', 'quantity', 'simulation_date']
        )
        _apply_fast_consumption(ao_orders, quantities, idx_map, offsets)

    # 普通订单消耗
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
    """快速应用订单消耗（直接修改quantities数组）。
    
    使用字典索引实现O(1)查找，避免DataFrame mask操作的O(n)开销。
    """
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
    """为集成模式生成供需日志。

    参数:
        demand_forecast: 原始需求预测（保留以兼容）。
        consumed_forecast: 消耗后的预测。
        simulation_date: 仿真日期。

    返回:
        供需日志DataFrame。
    """
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
