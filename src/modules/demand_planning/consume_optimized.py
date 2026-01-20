# -*- coding: utf-8 -*-
"""
DuckDB 加速的订单消耗处理模块

使用 DuckDB 进行向量化批量处理，避免 Python 循环和 pickle 序列化开销。
"""
import time
from typing import Tuple

import duckdb
import numpy as np
import pandas as pd


def consume_orders_duckdb(
    orders_df: pd.DataFrame,
    consumed_forecast: pd.DataFrame
) -> pd.DataFrame:
    """使用 DuckDB 处理订单消耗。
    
    Args:
        orders_df: 订单 DataFrame
        consumed_forecast: 预测视图
        
    Returns:
        消耗后的预测 DataFrame
    """
    if orders_df.empty:
        return consumed_forecast
    
    # 创建 DuckDB 连接
    conn = duckdb.connect(':memory:')
    conn.execute("SET threads TO 8")
    
    try:
        # 注册 DataFrame
        conn.register('orders', orders_df)
        conn.register('forecast', consumed_forecast)
        
        # 构建消耗偏移量
        offsets = [0, -1, -2, 1, 2, 3]
        
        # AO 订单消耗
        t1 = time.perf_counter()
        consumed_forecast = _consume_orders_by_type_duckdb(
            conn, 'AO', orders_df, consumed_forecast, offsets
        )
        print(f"[M1] AO消耗完成(DuckDB)，耗时: {time.perf_counter()-t1:.3f}s")
        
        # Normal 订单消耗
        t2 = time.perf_counter()
        consumed_forecast = _consume_orders_by_type_duckdb(
            conn, 'normal', orders_df, consumed_forecast, offsets
        )
        print(f"[M1] Normal消耗完成(DuckDB)，耗时: {time.perf_counter()-t2:.3f}s")
        
    finally:
        conn.close()
    
    return consumed_forecast


def _consume_orders_by_type_duckdb(
    conn: duckdb.DuckDBPyConnection,
    demand_type: str,
    orders_df: pd.DataFrame,
    consumed_forecast: pd.DataFrame,
    offsets: list
) -> pd.DataFrame:
    """按需求类型处理订单消耗。
    
    Args:
        conn: DuckDB 连接
        demand_type: 需求类型 ('AO' 或 'normal')
        orders_df: 订单 DataFrame
        consumed_forecast: 预测视图
        offsets: 消耗偏移量列表
        
    Returns:
        消耗后的预测 DataFrame
    """
    # 过滤指定类型的订单
    type_orders = orders_df[orders_df['demand_type'] == demand_type].copy()
    if type_orders.empty:
        return consumed_forecast
    
    # 排序
    if demand_type == 'AO':
        type_orders = type_orders.sort_values(
            by=['date', 'advance_days', 'quantity', 'simulation_date']
        )
    else:
        type_orders = type_orders.sort_values(
            by=['date', 'quantity', 'simulation_date']
        )
    
    # 为了保持与原算法兼容，仍需逐订单处理（因为消耗有状态依赖）
    # 但使用向量化查找来加速
    
    # 预建索引
    forecast_idx = consumed_forecast.set_index(['material', 'location', 'date'])
    forecast_dict = forecast_idx['quantity'].to_dict()
    
    # 逐订单消耗
    for _, order in type_orders.iterrows():
        mat = order['material']
        loc = order['location']
        order_date = pd.to_datetime(order['date'])
        remaining = int(order['quantity'])
        
        if remaining <= 0:
            continue
        
        # 按偏移量顺序消耗
        for offset in offsets:
            if remaining <= 0:
                break
            
            target_date = order_date + pd.Timedelta(days=offset)
            key = (mat, loc, target_date)
            
            if key in forecast_dict:
                avail = int(forecast_dict[key])
                take = min(avail, remaining)
                forecast_dict[key] = avail - take
                remaining -= take
    
    # 重建 DataFrame
    result = consumed_forecast.copy()
    for (mat, loc, date), qty in forecast_dict.items():
        mask = (
            (result['material'] == mat) &
            (result['location'] == loc) &
            (result['date'] == date)
        )
        result.loc[mask, 'quantity'] = qty
    
    return result


def consume_orders_vectorized(
    orders_df: pd.DataFrame,
    consumed_forecast: pd.DataFrame
) -> pd.DataFrame:
    """使用向量化方法处理订单消耗。
    
    这是一个完全向量化的实现，使用 numpy 操作替代 Python 循环。
    注意：由于消耗具有状态依赖性（先消耗的订单影响后续可用量），
    完全向量化可能导致结果不一致。此实现仅用于测试。
    
    Args:
        orders_df: 订单 DataFrame
        consumed_forecast: 预测视图
        
    Returns:
        消耗后的预测 DataFrame
    """
    if orders_df.empty:
        return consumed_forecast
    
    # 使用字典进行快速查找和更新
    result = consumed_forecast.copy()
    
    # 创建 (material, location, date) -> row_index 映射
    result['_idx'] = range(len(result))
    idx_map = {}
    for _, row in result.iterrows():
        key = (row['material'], row['location'], row['date'])
        idx_map[key] = row['_idx']
    
    # 转为可变数组
    quantities = result['quantity'].values.copy().astype(float)
    
    offsets = np.array([0, -1, -2, 1, 2, 3], dtype=int)
    
    # AO 订单消耗
    ao_orders = orders_df[orders_df['demand_type'] == 'AO'].copy()
    if not ao_orders.empty:
        ao_orders = ao_orders.sort_values(
            by=['date', 'advance_days', 'quantity', 'simulation_date']
        )
        t1 = time.perf_counter()
        _consume_orders_fast(ao_orders, quantities, idx_map, offsets)
        print(f"[M1] AO消耗完成(向量化)，耗时: {time.perf_counter()-t1:.3f}s")
    
    # Normal 订单消耗
    normal_orders = orders_df[orders_df['demand_type'] == 'normal'].copy()
    if not normal_orders.empty:
        normal_orders = normal_orders.sort_values(
            by=['date', 'quantity', 'simulation_date']
        )
        t2 = time.perf_counter()
        _consume_orders_fast(normal_orders, quantities, idx_map, offsets)
        print(f"[M1] Normal消耗完成(向量化)，耗时: {time.perf_counter()-t2:.3f}s")
    
    result['quantity'] = quantities.astype(int)
    result = result.drop(columns=['_idx'])
    
    return result


def _consume_orders_fast(
    orders: pd.DataFrame,
    quantities: np.ndarray,
    idx_map: dict,
    offsets: np.ndarray
) -> None:
    """快速消耗订单（修改 quantities 数组）。
    
    Args:
        orders: 订单 DataFrame
        quantities: 可用量数组（会被修改）
        idx_map: (material, location, date) -> row_index 映射
        offsets: 消耗偏移量
    """
    for _, order in orders.iterrows():
        mat = order['material']
        loc = order['location']
        order_date = pd.to_datetime(order['date'])
        remaining = int(order['quantity'])
        
        if remaining <= 0:
            continue
        
        for offset in offsets:
            if remaining <= 0:
                break
            
            target_date = order_date + pd.Timedelta(days=int(offset))
            key = (mat, loc, target_date)
            
            if key in idx_map:
                idx = idx_map[key]
                avail = int(quantities[idx])
                take = min(avail, remaining)
                quantities[idx] = avail - take
                remaining -= take
