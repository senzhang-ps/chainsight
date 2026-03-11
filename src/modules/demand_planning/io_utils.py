"""Module1 历史订单与文件IO处理。

本模块提供历史订单加载和文件输出功能。

主要函数：
- load_previous_orders: 加载历史订单
- save_module1_output_with_supply_demand: 保存Module1输出
"""

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd

from .constants import (
    DEFAULT_MAX_ADVANCE_DAYS,
    DEFAULT_USE_PARALLEL_FILE_LOAD,
    DEFAULT_PARALLEL_MAX_WORKERS,
    append_error_log,
)
from .normalization import normalize_identifiers


def load_previous_orders(
    m1_output_dir: str,
    current_date: pd.Timestamp,
    max_advance_days: int = DEFAULT_MAX_ADVANCE_DAYS
) -> pd.DataFrame:
    """加载近期历史订单。

    参数:
        m1_output_dir: Module1输出目录。
        current_date: 当前仿真日期。
        max_advance_days: 最大回溯天数。

    返回:
        合并的历史订单DataFrame。
    """
    try:
        if not os.path.isdir(m1_output_dir):
            return pd.DataFrame()

        pattern = re.compile(r"module1_output_(\d{8})\.xlsx$")
        earliest = current_date - pd.Timedelta(days=max_advance_days + 1)

        # 收集候选文件
        candidates = _collect_candidate_files(
            m1_output_dir, pattern, current_date, earliest
        )

        # 读取文件
        rows = _read_order_files(candidates)

        count = sum(len(r) for r in rows) if rows else 0
        print(
            f"[M1] 历史订单读取完成，文件数: {len(candidates)}，"
            f"合并条目: {count}"
        )

        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    except Exception:
        return pd.DataFrame()


def _collect_candidate_files(
    output_dir: str,
    pattern: re.Pattern,
    current_date: pd.Timestamp,
    earliest: pd.Timestamp
) -> list:
    """收集候选文件列表。

    参数:
        output_dir: 输出目录。
        pattern: 文件名匹配模式。
        current_date: 当前日期。
        earliest: 最早日期。

    返回:
        候选文件路径列表。
    """
    candidates = []
    for fname in os.listdir(output_dir):
        m = pattern.match(fname)
        if not m:
            continue
        fdate = pd.to_datetime(m.group(1))
        if fdate.normalize() >= current_date.normalize():
            continue
        if fdate.normalize() < earliest.normalize():
            continue
        candidates.append(os.path.join(output_dir, fname))
    return candidates


def _read_order_files(candidates: list) -> list:
    """读取订单文件。

    参数:
        candidates: 候选文件列表。

    返回:
        DataFrame列表。
    """
    use_parallel = DEFAULT_USE_PARALLEL_FILE_LOAD
    max_workers = DEFAULT_PARALLEL_MAX_WORKERS
    rows = []

    if use_parallel and candidates:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_read_orderlog, p) for p in candidates]
            for f in as_completed(futures):
                df = f.result()
                if df is not None and not df.empty:
                    rows.append(df)
    else:
        for p in candidates:
            df = _read_orderlog(p)
            if df is not None and not df.empty:
                rows.append(df)

    return rows


def _read_orderlog(path: str) -> Optional[pd.DataFrame]:
    """读取单个OrderLog文件。

    参数:
        path: 文件路径。

    返回:
        `OrderLog` DataFrame或None。
    """
    try:
        xl = pd.ExcelFile(path)
        if 'OrderLog' not in xl.sheet_names:
            return None
        df = xl.parse('OrderLog')
        if df is None or df.empty:
            return None
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        if 'simulation_date' in df.columns:
            df['simulation_date'] = pd.to_datetime(df['simulation_date'])
        return df
    except Exception:
        append_error_log(f"[历史文件并行] 读取失败：{path}")
        return None


def save_module1_output_with_supply_demand(
    orders_df: pd.DataFrame,
    shipment_df: pd.DataFrame,
    supply_demand_df: pd.DataFrame,
    output_file: str,
    cut_df: Optional[pd.DataFrame] = None
) -> None:
    """将Module1输出写入Excel文件。

    参数:
        orders_df: 订单DataFrame。
        shipment_df: 发货DataFrame。
        supply_demand_df: 供需日志DataFrame。
        output_file: 输出文件路径。
        cut_df: 缺货DataFrame。
    """
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            orders_out = _ensure_cols(
                orders_df,
                ['date', 'material', 'location', 'demand_type',
                 'quantity', 'simulation_date', 'advance_days']
            )
            shipment_out = _ensure_cols(
                shipment_df,
                ['date', 'material', 'location', 'quantity',
                 'demand_type', 'order_id']
            )
            cut_out = _ensure_cols(
                cut_df,
                ['date', 'material', 'location', 'quantity']
            )
            supply_out = _ensure_cols(
                supply_demand_df,
                ['date', 'material', 'location', 'quantity', 'demand_element']
            )

            normalize_identifiers(orders_out).to_excel(
                writer, sheet_name='OrderLog', index=False
            )
            normalize_identifiers(shipment_out).to_excel(
                writer, sheet_name='ShipmentLog', index=False
            )
            normalize_identifiers(cut_out).to_excel(
                writer, sheet_name='CutLog', index=False
            )
            normalize_identifiers(supply_out).to_excel(
                writer, sheet_name='SupplyDemandLog', index=False
            )

            summary = _build_summary(orders_df, shipment_df, cut_out, supply_demand_df)
            summary.to_excel(writer, sheet_name='Summary', index=False)

    except Exception as e:
        print(f"⚠️  Module1 输出保存失败: {e}")


def _build_summary(
    orders_df: pd.DataFrame,
    shipment_df: pd.DataFrame,
    cut_df: pd.DataFrame,
    supply_demand_df: pd.DataFrame
) -> pd.DataFrame:
    """构建汇总信息。

    参数:
        orders_df: 订单数据。
        shipment_df: 发货数据。
        cut_df: 缺货数据。
        supply_demand_df: 供需日志数据。

    返回:
        汇总DataFrame。
    """
    date_val = orders_df['date'].iloc[0] if not orders_df.empty else 'N/A'
    return pd.DataFrame([{
        'Total_Orders': len(orders_df),
        'Total_Shipments': len(shipment_df),
        'Total_Cuts': len(cut_df),
        'Total_SupplyDemand': len(supply_demand_df),
        'Date': date_val
    }])


def _ensure_cols(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """确保DataFrame包含指定列。

    参数:
        df: 输入DataFrame。
        cols: 需要的列列表。

    返回:
        包含所有指定列的DataFrame。
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    for c in cols:
        if c not in df.columns:
            df[c] = pd.Series(dtype='object')
    return df[cols]
