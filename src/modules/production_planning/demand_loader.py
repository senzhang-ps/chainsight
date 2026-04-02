"""
需求加载模块

负责从Module3输出加载净需求数据。
"""

import os
from typing import List

import pandas as pd

from .constants import UNCONSTRAINED_PLAN_COLUMNS
from .utils import cast_identifiers_to_str


def load_daily_net_demand(
    module3_output_dir: str,
    simulation_date: pd.Timestamp
) -> pd.DataFrame:
    """加载前一日Module3的净需求。

    按数据流规范读取前一日输出，筛选layer=0下游需求，
    数量取绝对值，保证requirement_date为日期类型。

    参数：
        module3_output_dir: Module3每日输出目录
        simulation_date: 当前仿真日期

    返回：
        pd.DataFrame: 处理后的净需求数据
    """
    empty_columns = _get_net_demand_columns()

    try:
        net_demand_file = _get_net_demand_file_path(
            module3_output_dir,
            simulation_date
        )

        if not os.path.exists(net_demand_file):
            return pd.DataFrame(columns=empty_columns)

        return _load_and_process_net_demand(net_demand_file, empty_columns)

    except Exception as e:
        date_str = simulation_date.strftime('%Y-%m-%d')
        print(f"加载净需求数据出错 {date_str}: {e}")
        return pd.DataFrame(columns=empty_columns)


def _get_net_demand_columns() -> List[str]:
    """获取净需求数据的列名。

    返回：
        List[str]: 列名列表
    """
    return [
        'material',
        'location',
        'requirement_date',
        'quantity',
        'demand_type',
        'layer',
    ]


def _get_net_demand_file_path(
    output_dir: str,
    simulation_date: pd.Timestamp
) -> str:
    """获取净需求文件路径。

    Module4读取前一天的Module3输出。

    参数：
        output_dir: Module3输出目录
        simulation_date: 当前仿真日期

    返回：
        str: 文件路径
    """
    prev_date = simulation_date - pd.Timedelta(days=1)
    prev_date_str = prev_date.strftime('%Y%m%d')
    return os.path.join(output_dir, f"Module3Output_{prev_date_str}.xlsx")


def _load_and_process_net_demand(
    file_path: str,
    empty_columns: List[str]
) -> pd.DataFrame:
    """加载并处理净需求文件。

    参数：
        file_path: 文件路径
        empty_columns: 空DataFrame的列名

    返回：
        pd.DataFrame: 处理后的净需求数据
    """
    xl = pd.ExcelFile(file_path)

    if 'NetDemand' not in xl.sheet_names:
        print(f"警告: 文件 {file_path} 中未找到NetDemand工作表")
        return pd.DataFrame(columns=empty_columns)

    net_demand = pd.read_excel(file_path, sheet_name='NetDemand')
    net_demand = cast_identifiers_to_str(net_demand, ['material', 'location'])

    if net_demand.empty:
        return pd.DataFrame(columns=empty_columns)

    return _filter_and_normalize_demand(net_demand)


def _filter_and_normalize_demand(net_demand: pd.DataFrame) -> pd.DataFrame:
    """筛选并规范化净需求数据。

    参数：
        net_demand: 原始净需求数据

    返回：
        pd.DataFrame: 处理后的数据
    """
    layer0_demand = _filter_layer_zero(net_demand)
    layer0_demand = _normalize_quantity(layer0_demand)
    layer0_demand = _normalize_date(layer0_demand)

    return layer0_demand


def _filter_layer_zero(df: pd.DataFrame) -> pd.DataFrame:
    """筛选layer=0的记录。

    参数：
        df: 原始DataFrame

    返回：
        pd.DataFrame: 筛选后的DataFrame
    """
    if 'layer' in df.columns:
        return df[df['layer'] == 0].copy()

    print("警告: 'layer'列不存在，使用全部需求")
    return df.copy()


def _normalize_quantity(df: pd.DataFrame) -> pd.DataFrame:
    """规范化数量（取绝对值）。

    参数：
        df: DataFrame

    返回：
        pd.DataFrame: 处理后的DataFrame
    """
    if 'quantity' in df.columns:
        df['quantity'] = df['quantity'].abs()
    return df


def _normalize_date(df: pd.DataFrame) -> pd.DataFrame:
    """规范化日期列。

    参数：
        df: DataFrame

    返回：
        pd.DataFrame: 处理后的DataFrame
    """
    if 'requirement_date' in df.columns:
        df['requirement_date'] = pd.to_datetime(df['requirement_date'])
    return df
