"""
状态管理模块

负责产线状态和已分配产能的持久化管理，支持跨天连续性。
"""

import os
import json
from typing import Optional, Dict, Any

import pandas as pd


def get_or_init_simulation_start(
    output_dir: str,
    provided_start: Optional[pd.Timestamp]
) -> pd.Timestamp:
    """读取或初始化仿真开始日期（持久化）。

    首次运行时将提供的开始日期写入状态文件；
    后续运行读取持久化值，保证审查日计算一致。

    Args:
        output_dir: 模块4输出/状态目录
        provided_start: 用户提供的开始日期（首次必需）

    Returns:
        pd.Timestamp: 持久化的仿真开始日期

    Raises:
        ValueError: 当状态文件不存在且未提供开始日期时
    """
    state_file = os.path.join(output_dir, "simulation_start.txt")

    if os.path.exists(state_file):
        return _read_start_from_file(state_file)

    if provided_start is None:
        raise ValueError(
            "仿真开始日期未提供且状态文件不存在"
        )

    return _write_start_to_file(output_dir, state_file, provided_start)


def _read_start_from_file(state_file: str) -> pd.Timestamp:
    """从文件读取仿真开始日期。

    Args:
        state_file: 状态文件路径

    Returns:
        pd.Timestamp: 仿真开始日期

    Raises:
        ValueError: 读取失败时
    """
    try:
        with open(state_file, "r", encoding='utf-8') as f:
            return pd.to_datetime(f.read().strip())
    except Exception as e:
        raise ValueError(f"读取仿真开始日期失败 {state_file}: {e}")


def _write_start_to_file(
    output_dir: str,
    state_file: str,
    start_date: pd.Timestamp
) -> pd.Timestamp:
    """将仿真开始日期写入文件。

    Args:
        output_dir: 输出目录
        state_file: 状态文件路径
        start_date: 开始日期

    Returns:
        pd.Timestamp: 写入的开始日期
    """
    os.makedirs(output_dir, exist_ok=True)
    with open(state_file, "w", encoding='utf-8') as f:
        f.write(start_date.strftime("%Y-%m-%d"))
    return start_date


def save_line_state(
    output_dir: str,
    simulation_date: pd.Timestamp,
    line_states: Dict[str, Any]
) -> None:
    """保存产线状态用于跨天连续性。

    Args:
        output_dir: 输出/状态目录
        simulation_date: 当前仿真日期
        line_states: 产线状态字典
    """
    os.makedirs(output_dir, exist_ok=True)
    date_str = simulation_date.strftime('%Y%m%d')
    state_file = os.path.join(output_dir, f"line_states_{date_str}.json")

    with open(state_file, "w", encoding='utf-8') as f:
        json.dump(line_states, f, indent=2, ensure_ascii=False)


def load_line_state(
    output_dir: str,
    simulation_date: pd.Timestamp
) -> Dict[str, Any]:
    """加载前一日的产线状态。

    Args:
        output_dir: 输出/状态目录
        simulation_date: 当前仿真日期

    Returns:
        Dict[str, Any]: 前一日产线状态，不存在则返回空字典
    """
    prev_date = simulation_date - pd.Timedelta(days=1)
    date_str = prev_date.strftime('%Y%m%d')
    state_file = os.path.join(output_dir, f"line_states_{date_str}.json")

    if not os.path.exists(state_file):
        return {}

    try:
        with open(state_file, "r", encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"警告: 加载产线状态失败 {state_file}: {e}")
        return {}


def save_allocated_capacity(
    output_dir: str,
    simulation_date: pd.Timestamp,
    allocated_capacity: Dict[str, float]
) -> None:
    """保存已分配产能用于跨日跟踪。

    Args:
        output_dir: 输出/状态目录
        simulation_date: 当前仿真日期
        allocated_capacity: 已分配产能字典（小时）
    """
    os.makedirs(output_dir, exist_ok=True)
    date_str = simulation_date.strftime('%Y%m%d')
    capacity_file = os.path.join(
        output_dir,
        f"allocated_capacity_{date_str}.json"
    )

    with open(capacity_file, "w", encoding='utf-8') as f:
        json.dump(allocated_capacity, f, indent=2, ensure_ascii=False)


def load_allocated_capacity(
    output_dir: str,
    simulation_date: pd.Timestamp
) -> Dict[str, float]:
    """加载当前仿真日的已分配产能。

    Args:
        output_dir: 输出/状态目录
        simulation_date: 当前仿真日期

    Returns:
        Dict[str, float]: 已分配产能字典，不存在则返回空字典
    """
    date_str = simulation_date.strftime('%Y%m%d')
    capacity_file = os.path.join(
        output_dir,
        f"allocated_capacity_{date_str}.json"
    )

    if not os.path.exists(capacity_file):
        return {}

    try:
        with open(capacity_file, "r", encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"警告: 加载已分配产能失败 {capacity_file}: {e}")
        return {}


def load_all_previous_capacity(
    output_dir: str,
    simulation_date: pd.Timestamp
) -> Dict[str, float]:
    """汇总所有历史仿真日的已分配产能。

    Args:
        output_dir: 输出/状态目录
        simulation_date: 当前仿真日期

    Returns:
        Dict[str, float]: 合并后的历史产能分配字典
    """
    consolidated = {}

    if not os.path.exists(output_dir):
        return consolidated

    for file_name in os.listdir(output_dir):
        if not _is_capacity_file(file_name):
            continue

        file_date = _extract_date_from_filename(file_name)
        if file_date is None or file_date >= simulation_date:
            continue

        daily_capacity = _load_capacity_file(output_dir, file_name)
        consolidated = _merge_capacity(consolidated, daily_capacity)

    return consolidated


def _is_capacity_file(file_name: str) -> bool:
    """检查是否为产能文件。

    Args:
        file_name: 文件名

    Returns:
        bool: 是否为产能文件
    """
    return (
        file_name.startswith("allocated_capacity_") and
        file_name.endswith(".json")
    )


def _extract_date_from_filename(file_name: str) -> Optional[pd.Timestamp]:
    """从文件名提取日期。

    Args:
        file_name: 文件名

    Returns:
        Optional[pd.Timestamp]: 提取的日期，失败返回None
    """
    try:
        date_str = file_name.replace(
            "allocated_capacity_", ""
        ).replace(".json", "")
        return pd.to_datetime(date_str, format='%Y%m%d')
    except Exception:
        return None


def _load_capacity_file(
    output_dir: str,
    file_name: str
) -> Dict[str, float]:
    """加载单个产能文件。

    Args:
        output_dir: 目录路径
        file_name: 文件名

    Returns:
        Dict[str, float]: 产能字典
    """
    try:
        file_path = os.path.join(output_dir, file_name)
        with open(file_path, "r", encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"警告: 加载产能文件失败 {file_name}: {e}")
        return {}


def _merge_capacity(
    target: Dict[str, float],
    source: Dict[str, float]
) -> Dict[str, float]:
    """合并产能字典。

    Args:
        target: 目标字典
        source: 源字典

    Returns:
        Dict[str, float]: 合并后的字典
    """
    for key, value in source.items():
        if key not in target:
            target[key] = 0
        target[key] += value
    return target
