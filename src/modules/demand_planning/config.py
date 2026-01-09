"""Module 1 配置加载与校验。

本模块提供Excel配置文件的加载和校验功能。

主要函数：
- load_config: 从Excel文件加载配置表
- validate_m1_config: 校验M1必需配置
"""

from typing import Any, Dict, Optional, Tuple

import pandas as pd

from .normalization import normalize_identifiers


def load_config(
    filename: str,
    sheet_mapping: Optional[Dict[str, Tuple[str, Any]]] = None
) -> Dict[str, pd.DataFrame]:
    """从Excel文件加载配置表到DataFrame字典。

    参数:
        filename: Excel配置文件路径。
        sheet_mapping: 可选的表名到(键名, 默认值)的映射。
            若为None，使用标准配置表的默认映射。

    返回:
        键名到DataFrame的字典。

    异常:
        RuntimeError: 文件加载失败时抛出。
    """
    if sheet_mapping is None:
        sheet_mapping = _get_default_sheet_mapping()

    try:
        xl = pd.ExcelFile(filename)
        loaded_sheets = {}
        for sheet_name, (key, default) in sheet_mapping.items():
            if sheet_name in xl.sheet_names:
                df = xl.parse(sheet_name)
                loaded_sheets[key] = normalize_identifiers(df)
            else:
                loaded_sheets[key] = default
        return loaded_sheets
    except Exception as e:
        raise RuntimeError(f"Failed to load config from {filename}: {e}")


def _get_default_sheet_mapping() -> Dict[str, Tuple[str, Any]]:
    """获取默认的表名映射配置。

    返回:
        表名到(键名, 默认值)的映射字典。
    """
    return {
        'DemandForecast': ('demand_forecast', None),
        'ForecastError': ('forecast_error', None),
        'OrderCalendar': ('order_calendar', None),
        'AOConfig': ('ao_config', pd.DataFrame()),
        'SupplyChoiceConfig': ('supply_choice', pd.DataFrame()),
        'InitialInventory': ('initial_inventory', None),
        'DPSConfig': ('dps_config', pd.DataFrame()),
        'ProductionPlan': ('production_plan', pd.DataFrame()),
        'DeliveryPlan': ('delivery_plan', pd.DataFrame()),
    }


def validate_m1_config(
    config_dict: dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """校验并返回M1必需配置。

    参数:
        config_dict: 配置字典。

    返回:
        (demand_forecast, forecast_error, order_calendar, ao_config)元组。

    异常:
        ValueError: 缺少必需配置时抛出。
    """
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

    return demand_forecast, forecast_error, order_calendar, ao_config
