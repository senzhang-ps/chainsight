"""Module1 订单消耗处理。

本模块提供订单消耗预测的功能。

主要函数：
- consume_orders: 处理订单消耗（AO优先，然后Normal）
- consume_ao_orders_serial: 串行处理AO订单消耗
- consume_normal_orders_serial: 串行处理Normal订单消耗
"""

import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from .constants import (
    DEFAULT_USE_PARALLEL_AO_CONSUME,
    DEFAULT_USE_PARALLEL_NORMAL_CONSUME,
    DEFAULT_PARALLEL_MAX_WORKERS,
    DEFAULT_USE_OPTIMIZED_CONSUME,
    append_error_log,
)
from ...utils.numeric_safe import safe_int_series


# 消耗窗口偏移量
CONSUME_OFFSETS = np.array([0, -1, -2, 1, 2, 3], dtype=int)


def consume_orders(
    orders_df: pd.DataFrame,
    consumed_forecast: pd.DataFrame
) -> pd.DataFrame:
    """处理订单消耗（AO优先，然后Normal）。

    参数:
        orders_df: 订单DataFrame。
        consumed_forecast: 当前预测视图。

    返回:
        消耗后的预测DataFrame。
    """
    # 使用优化版消耗（向量化+字典索引）
    if DEFAULT_USE_OPTIMIZED_CONSUME:
        from .consume_optimized import consume_orders_vectorized
        return consume_orders_vectorized(orders_df, consumed_forecast)
    
    use_parallel = DEFAULT_USE_PARALLEL_AO_CONSUME
    max_workers = DEFAULT_PARALLEL_MAX_WORKERS

    # AO 订单消耗
    ao_consume = orders_df[orders_df['demand_type'] == 'AO'].copy()
    if not ao_consume.empty:
        ao_consume = ao_consume.sort_values(
            by=['date', 'advance_days', 'quantity', 'simulation_date']
        )
        t3 = time.perf_counter()
        if use_parallel:
            consumed_forecast = _consume_ao_orders_parallel(
                ao_consume, consumed_forecast, CONSUME_OFFSETS, max_workers
            )
        else:
            consumed_forecast = consume_ao_orders_serial(
                ao_consume, consumed_forecast, CONSUME_OFFSETS
            )

    # 普通订单消耗
    normal_consume = orders_df[orders_df['demand_type'] == 'normal'].copy()
    inherit_flag = DEFAULT_USE_PARALLEL_NORMAL_CONSUME
    use_parallel_normal = (
        inherit_flag if inherit_flag is not None else use_parallel
    )

    if not normal_consume.empty:
        normal_consume = normal_consume.sort_values(
            by=['date', 'quantity', 'simulation_date']
        )
        t4 = time.perf_counter()
        if use_parallel_normal:
            consumed_forecast = _consume_normal_orders_parallel(
                normal_consume, consumed_forecast, CONSUME_OFFSETS, max_workers
            )
        else:
            consumed_forecast = consume_normal_orders_serial(
                normal_consume, consumed_forecast, CONSUME_OFFSETS
            )

    return consumed_forecast


def consume_ao_orders_serial(
    ao_consume: pd.DataFrame,
    consumed_forecast: pd.DataFrame,
    offsets: np.ndarray
) -> pd.DataFrame:
    """串行处理AO订单消耗。

    参数:
        ao_consume: AO订单DataFrame。
        consumed_forecast: 当前预测视图。
        offsets: 消耗窗口偏移数组。

    返回:
        消耗后的预测DataFrame。
    """
    for r in ao_consume.itertuples():
        if r.quantity <= 0:
            continue
        consumed_forecast = _consume_single_order(
            consumed_forecast, r, offsets
        )
    return consumed_forecast


def consume_normal_orders_serial(
    normal_consume: pd.DataFrame,
    consumed_forecast: pd.DataFrame,
    offsets: np.ndarray
) -> pd.DataFrame:
    """串行处理Normal订单消耗。

    参数:
        normal_consume: Normal订单DataFrame。
        consumed_forecast: 当前预测视图。
        offsets: 消耗窗口偏移数组。

    返回:
        消耗后的预测DataFrame。
    """
    for r in normal_consume.itertuples():
        if r.quantity <= 0:
            continue
        consumed_forecast = _consume_single_order(
            consumed_forecast, r, offsets
        )
    return consumed_forecast


def _consume_single_order(
    consumed_forecast: pd.DataFrame,
    order_row: Any,
    offsets: np.ndarray
) -> pd.DataFrame:
    """消耗单个订单。

    参数:
        consumed_forecast: 预测视图。
        order_row: 订单行（namedtuple）。
        offsets: 消耗偏移量。

    返回:
        更新后的预测视图。
    """
    target_dates = pd.to_datetime(order_row.date) + pd.to_timedelta(
        offsets, unit='D'
    )
    ml_mask = (
        (consumed_forecast['material'] == order_row.material) &
        (consumed_forecast['location'] == order_row.location)
    )
    window_mask = ml_mask & consumed_forecast['date'].isin(target_dates)
    window = consumed_forecast.loc[window_mask, ['date', 'quantity']].copy()
    remaining = int(order_row.quantity)

    for od in offsets:
        if remaining <= 0:
            break
        d = pd.to_datetime(order_row.date) + pd.to_timedelta(int(od), unit='D')
        idxs = window.index[window['date'] == d]
        if len(idxs) == 0:
            continue
        idx = idxs[0]
        avail = int(window.at[idx, 'quantity'])
        take = min(avail, remaining)
        window.at[idx, 'quantity'] = avail - take
        remaining -= take

    # 优化：使用 itertuples() 替代 iterrows()
    for w in window.itertuples():
        consumed_forecast.loc[
            ml_mask & (consumed_forecast['date'] == w.date),
            'quantity'
        ] = int(w.quantity)

    return consumed_forecast


def _consume_ao_orders_parallel(
    ao_consume: pd.DataFrame,
    consumed_forecast: pd.DataFrame,
    offsets: np.ndarray,
    max_workers: Optional[int]
) -> pd.DataFrame:
    """并行处理AO订单消耗。

    参数:
        ao_consume: AO订单DataFrame。
        consumed_forecast: 当前预测视图。
        offsets: 消耗窗口偏移数组。
        max_workers: 并行工作进程数。

    返回:
        消耗后的预测DataFrame。
    """
    tasks = _build_parallel_tasks(
        ao_consume, consumed_forecast, offsets,
        ['date', 'quantity', 'advance_days', 'simulation_date']
    )

    patches = _execute_parallel_consume(
        tasks, _consume_ao_for_ml_worker, max_workers, 'AO'
    )

    return _apply_patches(consumed_forecast, patches)


def _consume_normal_orders_parallel(
    normal_consume: pd.DataFrame,
    consumed_forecast: pd.DataFrame,
    offsets: np.ndarray,
    max_workers: Optional[int]
) -> pd.DataFrame:
    """并行处理Normal订单消耗。

    参数:
        normal_consume: Normal订单DataFrame。
        consumed_forecast: 当前预测视图。
        offsets: 消耗窗口偏移数组。
        max_workers: 并行工作进程数。

    返回:
        消耗后的预测DataFrame。
    """
    tasks = _build_parallel_tasks(
        normal_consume, consumed_forecast, offsets,
        ['date', 'quantity', 'simulation_date']
    )

    patches = _execute_parallel_consume(
        tasks, _consume_normal_for_ml_worker, max_workers, 'Normal'
    )

    return _apply_patches(consumed_forecast, patches)


def _build_parallel_tasks(
    orders: pd.DataFrame,
    consumed_forecast: pd.DataFrame,
    offsets: np.ndarray,
    order_cols: List[str]
) -> List[Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, str, str]]:
    """构建并行任务列表。

    参数:
        orders: 订单DataFrame。
        consumed_forecast: 预测视图。
        offsets: 消耗偏移量。
        order_cols: 订单需要的列。

    返回:
        任务列表。
    """
    tasks: List[Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, str, str]] = []

    for (mat, loc), grp in orders.groupby(['material', 'location']):
        ml_mask = (
            (consumed_forecast['material'] == mat) &
            (consumed_forecast['location'] == loc)
        )
        ml_forecast = consumed_forecast.loc[
            ml_mask, ['date', 'quantity']
        ].copy()
        if ml_forecast.empty:
            continue

        sample = (grp[order_cols], ml_forecast, offsets, mat, loc)
        try:
            pickle.dumps(sample)
        except Exception as e:
            # 不能静默丢弃该 (material, location) 的消耗任务——否则结果会少算且无人知晓。
            raise RuntimeError(
                f"并行消耗任务无法序列化 (material={mat!r}, location={loc!r})：{e}"
            ) from e
        tasks.append(sample)

    return tasks


def _execute_parallel_consume(
    tasks: List[Tuple],
    worker_func,
    max_workers: Optional[int],
    order_type: str
) -> List[pd.DataFrame]:
    """执行并行消耗任务。

    参数:
        tasks: 任务列表。
        worker_func: 工作函数。
        max_workers: 最大工作进程数。
        order_type: 订单类型（用于日志）。

    返回:
        补丁DataFrame列表。
    """
    patches: List[pd.DataFrame] = []

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(worker_func, t) for t in tasks]
            for f in as_completed(futures):
                try:
                    res = f.result()
                except Exception as e:
                    # 子任务失败意味着这部分消耗没算——不能吞掉，否则结果静默偏差。
                    append_error_log(f'[{order_type}并行] 子任务异常: {e}')
                    raise
                if res is not None and not res.empty:
                    patches.append(res)
    except Exception as e:
        append_error_log(f'[{order_type}并行] 执行器失败: {e}')
        raise

    return patches


def _apply_patches(
    consumed_forecast: pd.DataFrame,
    patches: List[pd.DataFrame]
) -> pd.DataFrame:
    """应用消耗补丁到预测视图。

    参数:
        consumed_forecast: 原预测视图。
        patches: 补丁列表。

    返回:
        更新后的预测视图。
    """
    if not patches:
        return consumed_forecast

    patch_df = pd.concat(patches, ignore_index=True)
    key_cols = ['material', 'location', 'date']
    cf = consumed_forecast.merge(patch_df, on=key_cols, how='left')
    cf['new_quantity'] = pd.to_numeric(cf['new_quantity'], errors='coerce')
    cf['quantity'] = np.where(
        cf['new_quantity'].notna(),
        safe_int_series(np.maximum(0, cf['new_quantity'].fillna(0)),
                        context='module1._apply_patches.new_quantity'),
        safe_int_series(cf['quantity'], context='module1._apply_patches.quantity'),
    )
    return cf[['material', 'location', 'date', 'quantity']]


def _consume_ao_for_ml_worker(
    args: Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, str, str]
) -> pd.DataFrame:
    """处理单个物料-地点对的AO订单消耗。

    参数:
        args: (ml_orders, ml_forecast, offsets, material, location)。

    返回:
        该物料-地点消耗后预测数量的DataFrame。
    """
    ml_orders, ml_forecast, offsets_local, mat, loc = args
    empty_cols = ['material', 'location', 'date', 'new_quantity']

    if ml_orders.empty or ml_forecast.empty:
        return pd.DataFrame(columns=empty_cols)

    ml_forecast = ml_forecast.copy()
    for r in ml_orders.itertuples():
        if r.quantity <= 0:
            continue
        remaining = int(r.quantity)
        for od in offsets_local:
            if remaining <= 0:
                break
            d = pd.to_datetime(r.date) + pd.to_timedelta(int(od), unit='D')
            idxs = ml_forecast.index[ml_forecast['date'] == d]
            if len(idxs) == 0:
                continue
            idx = idxs[0]
            avail = int(ml_forecast.at[idx, 'quantity'])
            take = min(avail, remaining)
            ml_forecast.at[idx, 'quantity'] = avail - take
            remaining -= take

    out = ml_forecast[['date', 'quantity']].copy()
    out['material'] = mat
    out['location'] = loc
    out = out.rename(columns={'quantity': 'new_quantity'})
    return out[empty_cols]


def _consume_normal_for_ml_worker(
    args: Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, str, str]
) -> pd.DataFrame:
    """处理单个物料-地点对的Normal订单消耗。

    参数:
        args: (ml_orders, ml_forecast, offsets, material, location)。

    返回:
        该物料-地点消耗后预测数量的DataFrame。
    """
    ml_orders, ml_forecast, offsets_local, mat, loc = args
    empty_cols = ['material', 'location', 'date', 'new_quantity']

    if ml_orders.empty or ml_forecast.empty:
        return pd.DataFrame(columns=empty_cols)

    ml_forecast = ml_forecast.copy()
    for r in ml_orders.itertuples():
        if r.quantity <= 0:
            continue
        remaining = int(r.quantity)
        for od in offsets_local:
            if remaining <= 0:
                break
            d = pd.to_datetime(r.date) + pd.to_timedelta(int(od), unit='D')
            idxs = ml_forecast.index[ml_forecast['date'] == d]
            if len(idxs) == 0:
                continue
            idx = idxs[0]
            avail = int(ml_forecast.at[idx, 'quantity'])
            take = min(avail, remaining)
            ml_forecast.at[idx, 'quantity'] = avail - take
            remaining -= take

    out = ml_forecast[['date', 'quantity']].copy()
    out['material'] = mat
    out['location'] = loc
    out = out.rename(columns={'quantity': 'new_quantity'})
    return out[empty_cols]
