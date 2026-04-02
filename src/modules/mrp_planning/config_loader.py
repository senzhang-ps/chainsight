"""
Module3 配置加载模块。

负责从Excel文件加载配置数据和Module1输出数据。
"""

import os
import time
from typing import Dict, Optional

import pandas as pd

from .constants import SHEET_MAPPING
from .utils import normalize_identifiers


def load_config(config_path: str) -> Dict[str, pd.DataFrame]:
    """
    从Excel文件加载配置数据。

    参数：
        config_path: 配置Excel文件路径

    返回：
        Dict[str, pd.DataFrame]: 配置数据字典

    异常：
        RuntimeError: 配置文件加载失败
    """
    try:
        xl = pd.ExcelFile(config_path)
        loaded_config = {}

        for sheet_name, (key, default) in SHEET_MAPPING.items():
            loaded_config[key] = _load_sheet(xl, sheet_name, key)

        return loaded_config

    except Exception as e:
        raise RuntimeError(
            f"Failed to load module3 config from {config_path}: {e}"
        )


def _load_sheet(
    xl: pd.ExcelFile,
    sheet_name: str,
    key: str
) -> pd.DataFrame:
    """
    加载单个Excel sheet。

    参数：
        xl: Excel文件对象
        sheet_name: sheet名称
        key: 配置键名

    返回：
        pd.DataFrame: 加载的数据
    """
    if sheet_name not in xl.sheet_names:
        return pd.DataFrame()

    df = xl.parse(sheet_name)
    df = _convert_date_columns(df, key)
    df = normalize_identifiers(df)
    return df


def _convert_date_columns(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """转换DataFrame中的日期列。"""
    if key == 'safety_stock' and 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    elif key == 'network_config':
        if 'eff_from' in df.columns:
            df['eff_from'] = pd.to_datetime(df['eff_from'])
        if 'eff_to' in df.columns:
            df['eff_to'] = pd.to_datetime(df['eff_to'])
    return df


def load_module1_daily_outputs(
    module1_output_dir: str,
    simulation_date: pd.Timestamp
) -> Dict[str, pd.DataFrame]:
    """
    读取Module1当天版本的输出。

    参数：
        module1_output_dir: Module1输出目录
        simulation_date: 模拟日期

    返回：
        Dict[str, pd.DataFrame]: Module1输出数据字典
    """
    t_start = time.perf_counter()
    try:
        date_str = simulation_date.strftime('%Y%m%d')
        module1_file = _find_module1_file(module1_output_dir, date_str)

        if not module1_file:
            print(f"Warning: Module1 output not found for {date_str}.")
            return _empty_module1_data()

        data = _read_module1_file(module1_file, simulation_date)
        elapsed = time.perf_counter() - t_start
        print(f"[M3] load_module1 total: {elapsed:.3f}s")
        return data

    except Exception as e:
        print(f"Warning: Error loading Module1 outputs: {e}")
        return _empty_module1_data()


def _find_module1_file(
    output_dir: str,
    date_str: str
) -> Optional[str]:
    """查找Module1输出文件。"""
    patterns = [
        f"module1_output_{date_str}.xlsx",
        f"output_simulation_{date_str}.xlsx",
    ]
    for pattern in patterns:
        path = os.path.join(output_dir, pattern)
        if os.path.exists(path):
            return path
    return None


def _read_module1_file(
    filepath: str,
    simulation_date: pd.Timestamp
) -> Dict[str, pd.DataFrame]:
    """读取Module1输出文件内容。"""
    xl = pd.ExcelFile(filepath)

    sdl = _read_supply_demand_log(xl)
    shp = _read_shipment_log(xl, simulation_date)
    odl = _read_order_log(xl)

    return {
        'supply_demand_df': sdl,
        'shipment_df': shp,
        'order_df': odl,
    }


def _read_supply_demand_log(xl: pd.ExcelFile) -> pd.DataFrame:
    """读取SupplyDemandLog。"""
    if 'SupplyDemandLog' not in xl.sheet_names:
        return pd.DataFrame()

    sdl = xl.parse('SupplyDemandLog')
    if not sdl.empty and 'date' in sdl.columns:
        sdl['date'] = pd.to_datetime(sdl['date'])
    return normalize_identifiers(sdl)


def _read_shipment_log(
    xl: pd.ExcelFile,
    simulation_date: pd.Timestamp
) -> pd.DataFrame:
    """读取ShipmentLog，仅保留当日数据。"""
    if 'ShipmentLog' not in xl.sheet_names:
        return pd.DataFrame()

    shp = xl.parse('ShipmentLog')
    if not shp.empty and 'date' in shp.columns:
        shp['date'] = pd.to_datetime(shp['date'])
        shp = shp[shp['date'] == simulation_date].copy()
    return normalize_identifiers(shp)


def _read_order_log(xl: pd.ExcelFile) -> pd.DataFrame:
    """读取OrderLog。"""
    if 'OrderLog' not in xl.sheet_names:
        return pd.DataFrame()

    odl = xl.parse('OrderLog')
    if not odl.empty:
        if 'date' in odl.columns:
            odl['date'] = pd.to_datetime(odl['date'])
        if 'simulation_date' in odl.columns:
            odl['simulation_date'] = pd.to_datetime(odl['simulation_date'])
    return normalize_identifiers(odl)


def _empty_module1_data() -> Dict[str, pd.DataFrame]:
    """返回空的Module1数据字典。"""
    return {
        'supply_demand_df': pd.DataFrame(),
        'shipment_df': pd.DataFrame(),
        'order_df': pd.DataFrame(),
    }


def load_excel_with_sheets(filepath: str) -> Dict[str, pd.DataFrame]:
    """
    加载Excel文件的所有sheet。

    参数：
        filepath: Excel文件路径

    返回：
        Dict[str, pd.DataFrame]: sheet名称到DataFrame的映射
    """
    xl = pd.ExcelFile(filepath)
    return {str(sheet): xl.parse(sheet) for sheet in xl.sheet_names}
