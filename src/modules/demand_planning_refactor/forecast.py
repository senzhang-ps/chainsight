"""Module1 预测拆分与处理。

本模块提供周度预测到日度预测的转换功能。

主要函数：
- expand_forecast_to_days_integer_split: 周度转日度预测
- prepare_daily_forecasts: 准备订单和供需日志的日度预测
"""

import time
from typing import Any, Optional, Tuple

import pandas as pd

from ...utils.normalization import normalize_identifiers
from .dps import apply_dps, apply_supply_choice


def expand_forecast_to_days_integer_split(
    demand_weekly: pd.DataFrame,
    start_date: pd.Timestamp,
    num_weeks: int,
    simulation_end_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """将周度预测拆分为日度预测（整数分配）。

    使用整数除法将周度数量均匀分配到7天。
    余数分配给前N天，其中N = quantity % 7。

    参数:
        demand_weekly: 周度预测DataFrame，包含[material, location, week, quantity]。
        start_date: 仿真开始日期。
        num_weeks: 周数（仅作参考）。
        simulation_end_date: 可选结束日期用于过滤。

    返回:
        日度预测DataFrame，包含列:
        [date, material, location, week, demand_type, quantity, original_quantity]。
    """
    if demand_weekly.empty:
        return pd.DataFrame(columns=[
            'date', 'material', 'location', 'week',
            'demand_type', 'quantity', 'original_quantity'
        ])

    start_date = pd.to_datetime(start_date)
    demand_weekly = demand_weekly.copy()

    # 计算周起始日期
    demand_weekly['week_start'] = start_date + pd.to_timedelta(
        (demand_weekly['week'] - 1) * 7, unit='D'
    )

    # 计算每日基础数量和余数
    demand_weekly['base_qty'] = (demand_weekly['quantity'] // 7).astype(int)
    demand_weekly['remainder'] = (demand_weekly['quantity'] % 7).astype(int)

    # 向量化生成7天数据
    t0 = time.perf_counter()
    days = []
    for day_offset in range(7):
        day_df = demand_weekly.copy()
        day_df['date'] = day_df['week_start'] + pd.Timedelta(days=day_offset)
        # 前remainder天多分配1个单位
        extra = (day_offset < day_df['remainder']).astype(int)
        day_df['quantity'] = day_df['base_qty'] + extra
        days.append(
            day_df[['date', 'material', 'location', 'week', 'quantity']]
        )

    result_df = pd.concat(days, ignore_index=True)

    # 过滤结束日期
    if simulation_end_date is not None:
        end_ts = pd.to_datetime(simulation_end_date)
        result_df = result_df[result_df['date'] <= end_ts]

    result_df['demand_type'] = 'normal'
    result_df['original_quantity'] = result_df['quantity']
    result_df['quantity'] = result_df['quantity'].astype(int)

    elapsed = time.perf_counter() - t0

    return normalize_identifiers(result_df)


def prepare_daily_forecasts(
    config_dict: dict,
    demand_forecast: pd.DataFrame,
    orchestrator: Any
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """准备订单和供需日志的日度预测。

    参数:
        config_dict: 配置字典。
        demand_forecast: 需求预测DataFrame。
        orchestrator: 编排器对象。

    返回:
        (daily_forecast_for_orders, daily_forecast_for_supply)元组。
    """
    dps_config = config_dict.get('M1_DPSConfig', pd.DataFrame())
    supply_choice = config_dict.get('M1_SupplyChoiceConfig', pd.DataFrame())

    if 'week' not in demand_forecast.columns:
        return demand_forecast.copy(), demand_forecast.copy()

    # DPS 处理后基线（用于订单）
    dps_cfg = dps_config if dps_config is not None else pd.DataFrame()
    demand_dps = apply_dps(demand_forecast, dps_cfg)

    # DPS 处理+SupplyChoice基线（用于供需日志）
    sc_cfg = supply_choice if supply_choice is not None else pd.DataFrame()
    demand_dps_sc = apply_supply_choice(demand_dps, sc_cfg)

    # 起始日期来自orchestrator
    sim_start = pd.to_datetime(orchestrator.start_date).normalize()

    # 生成日度基线
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
    """获取DataFrame中的最大周数。

    参数:
        df: 包含week列的DataFrame。

    返回:
        最大周数，空DataFrame返回1。
    """
    if df.empty or 'week' not in df.columns:
        return 1
    return int(df['week'].max())
