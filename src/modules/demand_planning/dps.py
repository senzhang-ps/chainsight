"""Module1 DPS地点拆分与供应选择。

本模块提供DPS配置处理和供应选择调整功能。

主要函数：
- apply_dps: DPS地点拆分
- apply_supply_choice: 供应选择调整
"""

import time

import numpy as np
import pandas as pd

from ...utils.normalization import normalize_identifiers
from ...utils.numeric_safe import safe_int_series


def apply_dps(df: pd.DataFrame, dps_cfg: pd.DataFrame) -> pd.DataFrame:
    """按DPS配置进行地点拆分。

    处理逻辑:
        1. 在物料-地点-周粒度聚合输入
        2. 按百分比分割为"保留量"和"拆分量"
        3. 拆分量的地点改为dps_location
        4. 输出重新聚合为整数

    特殊处理:
        - 缺失dps_percent视为0，不拆分
        - 使用向量化运算提高性能

    参数:
        df: 周度预测DataFrame，包含[material, location, week, quantity]。
        dps_cfg: DPS配置，包含[material, location, dps_location, dps_percent]。

    返回:
        DPS拆分后的DataFrame。
    """
    if dps_cfg.empty:
        return df.copy()

    t0 = time.perf_counter()

    # 聚合到物料-地点-周粒度
    df_g = df.groupby(
        ['material', 'location', 'week'], as_index=False
    )['quantity'].sum()

    # 合并DPS配置
    cols = ['material', 'location', 'dps_location', 'dps_percent']
    m = df_g.merge(dps_cfg[cols], on=['material', 'location'], how='left')
    m['dps_percent'] = m['dps_percent'].fillna(0.0)

    # 计算拆分量和保留量
    m['split_qty'] = safe_int_series(
        np.round(m['quantity'] * m['dps_percent']), context='module1.apply_dps.split_qty'
    )
    m['remain_qty'] = safe_int_series(
        m['quantity'] - m['split_qty'], context='module1.apply_dps.remain_qty'
    )

    # 构建保留部分和拆分部分
    remain = m[['material', 'location', 'week', 'remain_qty']].rename(
        columns={'remain_qty': 'quantity'}
    )
    split = m[['material', 'dps_location', 'week', 'split_qty']].rename(
        columns={'dps_location': 'location', 'split_qty': 'quantity'}
    )

    # 合并并聚合
    out = pd.concat([remain, split], ignore_index=True)
    out = out.groupby(
        ['material', 'location', 'week'], as_index=False
    )['quantity'].sum()
    out['quantity'] = safe_int_series(out['quantity'], context='module1.apply_dps.quantity')

    elapsed = time.perf_counter() - t0

    return normalize_identifiers(out)


def apply_supply_choice(
    df: pd.DataFrame,
    supply_cfg: pd.DataFrame
) -> pd.DataFrame:
    """应用供应选择调整到周度预测。

    将supply_cfg中的调整数量合并到预测中。

    参数:
        df: 周度预测DataFrame，包含[material, location, week, quantity]列。
        supply_cfg: 供应选择配置，包含[material, location, week, adjust_quantity]列。

    返回:
        在物料-地点-周粒度调整后的DataFrame。
    """
    if supply_cfg.empty:
        return df.copy()

    t0 = time.perf_counter()

    # 聚合输入数据
    df_g = df.groupby(
        ['material', 'location', 'week'], as_index=False
    )['quantity'].sum()
    sup_g = supply_cfg.groupby(
        ['material', 'location', 'week'], as_index=False
    )['adjust_quantity'].sum()

    # 合并并应用调整
    m = df_g.merge(sup_g, on=['material', 'location', 'week'], how='left')
    m['quantity'] = safe_int_series(
        m['quantity'] + m['adjust_quantity'].fillna(0),
        context='module1.apply_supply_choice.quantity',
    )
    out = m[['material', 'location', 'week', 'quantity']]

    elapsed = time.perf_counter() - t0

    return normalize_identifiers(out)
