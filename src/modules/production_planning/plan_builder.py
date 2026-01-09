"""
计划构建模块

负责构建无约束生产计划和优化换产序列。
"""

from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np

from .constants import UNCONSTRAINED_PLAN_COLUMNS
from .utils import (
    cast_identifiers_to_str,
    validate_merge_keys,
    compute_planning_window,
    is_review_day,
    round_up_to_batch,
)


def build_unconstrained_plan_for_single_day(
    net_demand_df: pd.DataFrame,
    mlcfg: pd.DataFrame,
    simulation_date: pd.Timestamp,
    simulation_start: pd.Timestamp,
    issues: List[Dict[str, Any]]
) -> pd.DataFrame:
    """构建单日无约束生产计划。

    仅针对审查日物料，读取当日净需求，计算最小批与舍入。

    Args:
        net_demand_df: 当日净需求
        mlcfg: 物料地点产线配置
        simulation_date: 当前仿真日期
        simulation_start: 仿真起始日期
        issues: 问题收集列表

    Returns:
        pd.DataFrame: 无约束计划
    """
    if net_demand_df.empty:
        return pd.DataFrame(columns=UNCONSTRAINED_PLAN_COLUMNS)

    mlcfg = cast_identifiers_to_str(mlcfg.copy(), ['material', 'location'])
    plans = []

    # 优化：使用 itertuples() 替代 iterrows()
    for row in mlcfg.itertuples():
        plan = _build_plan_for_material(
            row, net_demand_df, mlcfg,
            simulation_date, simulation_start, issues
        )
        if plan is not None:
            plans.append(plan)

    if not plans:
        return pd.DataFrame(columns=UNCONSTRAINED_PLAN_COLUMNS)

    return pd.concat(plans, ignore_index=True)


def _build_plan_for_material(
    row,
    net_demand_df: pd.DataFrame,
    mlcfg: pd.DataFrame,
    simulation_date: pd.Timestamp,
    simulation_start: pd.Timestamp,
    issues: List[Dict[str, Any]]
) -> Optional[pd.DataFrame]:
    """为单个物料构建计划。

    Args:
        row: 配置行（namedtuple 或 Series）
        net_demand_df: 净需求数据
        mlcfg: 配置DataFrame
        simulation_date: 仿真日期
        simulation_start: 仿真起始日期
        issues: 问题列表

    Returns:
        Optional[pd.DataFrame]: 计划DataFrame或None
    """
    # 兼容 itertuples() 返回的 namedtuple
    if hasattr(row, '_fields'):
        material = row.material
        location = row.location
        lsk = int(row.lsk)
        day = int(row.day)
    else:
        material = row['material']
        location = row['location']
        lsk = int(row['lsk'])
        day = int(row['day'])

    if not is_review_day(simulation_date, simulation_start, lsk, day):
        return None

    nd_sub = _get_material_demands(
        net_demand_df, material, location
    )

    if nd_sub.empty:
        return None

    nd_sub = _merge_with_config(nd_sub, mlcfg, material, location)
    nd_sub = _filter_by_date(
        nd_sub, simulation_date, material, location, issues
    )

    if nd_sub.empty:
        return None

    return _create_plan_record(row, nd_sub, simulation_date)


def _get_material_demands(
    net_demand_df: pd.DataFrame,
    material: str,
    location: str
) -> pd.DataFrame:
    """获取物料的需求记录。

    Args:
        net_demand_df: 净需求数据
        material: 物料编码
        location: 地点

    Returns:
        pd.DataFrame: 筛选后的需求
    """
    mask = (
        (net_demand_df['material'] == material) &
        (net_demand_df['location'] == location)
    )
    return net_demand_df[mask].copy()


def _merge_with_config(
    nd_sub: pd.DataFrame,
    mlcfg: pd.DataFrame,
    material: str,
    location: str
) -> pd.DataFrame:
    """与配置合并。

    Args:
        nd_sub: 需求子集
        mlcfg: 配置DataFrame
        material: 物料编码
        location: 地点

    Returns:
        pd.DataFrame: 合并后的DataFrame
    """
    cfg_slice = mlcfg[
        (mlcfg['material'] == material) &
        (mlcfg['location'] == location)
    ]
    validate_merge_keys(nd_sub, cfg_slice, ['material', 'location'])

    return nd_sub.merge(
        cfg_slice,
        on=['material', 'location'],
        how='left'
    )


def _filter_by_date(
    nd_sub: pd.DataFrame,
    simulation_date: pd.Timestamp,
    material: str,
    location: str,
    issues: List[Dict[str, Any]]
) -> pd.DataFrame:
    """按日期筛选需求。

    Args:
        nd_sub: 需求子集
        simulation_date: 仿真日期
        material: 物料编码
        location: 地点
        issues: 问题列表

    Returns:
        pd.DataFrame: 筛选后的DataFrame
    """
    nd_sub['requirement_date'] = pd.to_datetime(
        nd_sub['requirement_date']
    ).dt.normalize()

    sim_date_normalized = pd.to_datetime(simulation_date).normalize()
    mask = nd_sub['requirement_date'] == sim_date_normalized

    _report_date_mismatches(
        nd_sub[~mask], simulation_date, issues
    )

    return nd_sub[mask]


def _report_date_mismatches(
    mismatched: pd.DataFrame,
    simulation_date: pd.Timestamp,
    issues: List[Dict[str, Any]]
) -> None:
    """报告日期不匹配的需求。

    Args:
        mismatched: 不匹配的记录
        simulation_date: 仿真日期
        issues: 问题列表
    """
    # 优化：使用 itertuples() 替代 iterrows()
    for r in mismatched.itertuples():
        try:
            req_date = pd.to_datetime(r.requirement_date).date()
            sim_date = simulation_date.date()
            issues.append({
                'sheet': 'NetDemand',
                'row': '',
                'issue': (
                    f"物料 {r.material} 在地点 {r.location} "
                    f"的需求日期 {req_date} 与仿真日期 {sim_date} "
                    f"不匹配（已排除）"
                )
            })
        except Exception:
            pass


def _create_plan_record(
    row,
    nd_sub: pd.DataFrame,
    simulation_date: pd.Timestamp
) -> pd.DataFrame:
    """创建计划记录。

    Args:
        row: 配置行（namedtuple 或 Series）
        nd_sub: 需求子集
        simulation_date: 仿真日期

    Returns:
        pd.DataFrame: 计划记录
    """
    agg_qty = nd_sub['quantity'].sum()

    # 兼容 itertuples() 返回的 namedtuple
    if hasattr(row, '_fields'):
        min_batch = int(row.min_batch)
        rv = int(row.rv)
        material = row.material
        location = row.location
        line = row.delegate_line
    else:
        min_batch = int(row['min_batch'])
        rv = int(row['rv'])
        material = row['material']
        location = row['location']
        line = row['delegate_line']

    uncon_qty = round_up_to_batch(agg_qty, min_batch, rv)

    return pd.DataFrame([{
        'material': material,
        'location': location,
        'line': line,
        'planned_date': simulation_date,
        'uncon_planned_qty': uncon_qty,
        'simulation_date': simulation_date,
        'original_quantity': agg_qty,
    }])


def optimal_changeover_sequence(
    batches: List[Dict[str, Any]],
    co_mat: pd.Series,
    co_def: Dict[tuple, float],
    line: str
) -> List[int]:
    """优化换产序列。

    首件按原始需求量最大选择；后续优先最小换产时间，
    若并列使用数量打破。

    Args:
        batches: 批次列表
        co_mat: 换产矩阵
        co_def: 换产定义
        line: 产线标识

    Returns:
        List[int]: 批次索引的执行顺序
    """
    if not batches:
        return []

    batch_indices = list(range(len(batches)))
    remaining = set(batch_indices)

    first = _select_first_batch(batches, remaining)
    sequence = [first]
    remaining.remove(first)

    current_material = batches[first]['material']

    while remaining:
        next_idx = _select_next_batch(
            batches, remaining, current_material,
            co_mat, co_def, line
        )
        sequence.append(next_idx)
        remaining.remove(next_idx)
        current_material = batches[next_idx]['material']

    return sequence


def _select_first_batch(
    batches: List[Dict[str, Any]],
    remaining: set
) -> int:
    """选择第一个批次（最大原始数量）。

    Args:
        batches: 批次列表
        remaining: 剩余批次索引集合

    Returns:
        int: 选中的批次索引
    """
    return max(
        remaining,
        key=lambda i: batches[i].get(
            'original_quantity',
            batches[i]['uncon_planned_qty']
        )
    )


def _select_next_batch(
    batches: List[Dict[str, Any]],
    remaining: set,
    current_material: str,
    co_mat: pd.Series,
    co_def: Dict[tuple, float],
    line: str
) -> int:
    """选择下一个批次。

    优先最小换产时间，同时间用数量打破。

    Args:
        batches: 批次列表
        remaining: 剩余索引集合
        current_material: 当前物料
        co_mat: 换产矩阵
        co_def: 换产定义
        line: 产线

    Returns:
        int: 选中的批次索引
    """
    min_cost = None
    candidates = []

    for idx in remaining:
        co_time = _get_changeover_time(
            current_material,
            batches[idx]['material'],
            co_mat, co_def, line
        )

        if min_cost is None or co_time < min_cost:
            min_cost = co_time
            candidates = [idx]
        elif co_time == min_cost:
            candidates.append(idx)

    if len(candidates) == 1:
        return candidates[0]

    return max(
        candidates,
        key=lambda i: batches[i].get(
            'original_quantity',
            batches[i]['uncon_planned_qty']
        )
    )


def _get_changeover_time(
    from_material: str,
    to_material: str,
    co_mat: pd.Series,
    co_def: Dict[tuple, float],
    line: str
) -> float:
    """获取换产时间。

    Args:
        from_material: 源物料
        to_material: 目标物料
        co_mat: 换产矩阵
        co_def: 换产定义
        line: 产线

    Returns:
        float: 换产时间
    """
    try:
        from_str = str(from_material)
        to_str = str(to_material)

        if (from_str, to_str) not in co_mat.index:
            return 0

        coid = co_mat.loc[(from_str, to_str)]
        if isinstance(coid, pd.Series):
            coid = str(coid.iloc[0])
        else:
            coid = str(coid)

        return co_def.get((coid, line), 0)

    except Exception:
        return 0
