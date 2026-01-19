"""
Module3 工具函数模块。

提供MOQ/RV计算、标识符规范化、分配算法等通用功能。
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .constants import (
    DEFAULT_MOQ,
    DEFAULT_RV,
    IDENTIFIER_COLUMNS,
    LOCATION_TYPE_COLUMNS,
    COL_MATERIAL,
)


def apply_moq_rv(
    qty: float,
    moq: int,
    rv: int,
    is_cross_node: bool = True
) -> int:
    """
    应用MOQ/RV约束调整补货数量。

    Args:
        qty: 需求数量
        moq: 最小订货量
        rv: 重订量
        is_cross_node: 是否为跨节点调运

    Returns:
        int: 调整后的补货数量

    Examples:
        >>> apply_moq_rv(50, 100, 20)
        100
        >>> apply_moq_rv(150, 100, 20)
        160
    """
    if qty <= 0:
        return 0

    if not is_cross_node:
        return qty  # 与code_vo保持一致，直接返回原值不强制转整数

    if qty < moq:
        return moq
    return int(np.ceil(qty / rv)) * rv


def normalize_location(location_str: Union[str, int, float, None]) -> str:
    """
    将地点标识符规范化为4位前导零字符串。

    Args:
        location_str: 地点标识符

    Returns:
        str: 规范化后的地点字符串
    """
    if location_str is None or pd.isna(location_str):
        return ""
    try:
        return str(int(location_str)).zfill(4)
    except (ValueError, TypeError):
        return str(location_str).zfill(4)


def normalize_material(material_str: Union[str, int, float, None]) -> str:
    """
    将物料标识符规范化为字符串。

    作用：统一 material 字段格式，与code_v0保持一致。
    注意：直接转换为字符串，不做额外处理，以确保与code_v0输出一致。

    Args:
        material_str: 物料标识符

    Returns:
        str: 规范化后的物料字符串
    """
    if material_str is None or pd.isna(material_str):
        return ""
    return str(material_str)


def normalize_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    规范化DataFrame中的标识符列。

    Args:
        df: 需要规范化的DataFrame

    Returns:
        pd.DataFrame: 规范化后的DataFrame副本
    """
    if df.empty:
        return df

    df = df.copy()
    for col in IDENTIFIER_COLUMNS:
        if col not in df.columns:
            continue

        df[col] = df[col].astype('string')

        if col in LOCATION_TYPE_COLUMNS:
            df[col] = df[col].apply(normalize_location)
        elif col == COL_MATERIAL:
            df[col] = df[col].apply(normalize_material)
        else:
            df[col] = df[col].fillna('').astype(str)

    return df


def lookup_moq_rv_three_keys(
    deploy_config_df: Optional[pd.DataFrame],
    material: str,
    sending: str,
    receiving: Optional[str]
) -> Tuple[int, int]:
    """
    按三键查询MOQ/RV配置。

    优先级: (material, sending, receiving) > (material, sending) > 默认值

    Args:
        deploy_config_df: 部署配置DataFrame
        material: 物料编码
        sending: 发送节点
        receiving: 接收节点

    Returns:
        Tuple[int, int]: (moq, rv) 元组
    """
    try:
        if deploy_config_df is None or deploy_config_df.empty:
            return DEFAULT_MOQ, DEFAULT_RV

        # 三键匹配
        if 'receiving' in deploy_config_df.columns and receiving:
            rows = deploy_config_df[
                (deploy_config_df['material'] == str(material)) &
                (deploy_config_df['sending'] == str(sending)) &
                (deploy_config_df['receiving'] == str(receiving))
            ]
            if not rows.empty:
                return _extract_moq_rv(rows.iloc[0])

        # 二键匹配
        rows = deploy_config_df[
            (deploy_config_df['material'] == str(material)) &
            (deploy_config_df['sending'] == str(sending))
        ]
        if not rows.empty:
            return _extract_moq_rv(rows.iloc[0])

    except Exception:
        pass

    return DEFAULT_MOQ, DEFAULT_RV


def _extract_moq_rv(row: pd.Series) -> Tuple[int, int]:
    """从行数据提取MOQ/RV值。"""
    moq = int(pd.to_numeric(row.get('moq', 1), errors='coerce') or 1)
    rv = int(pd.to_numeric(row.get('rv', 1), errors='coerce') or 1)
    return max(0, moq), max(0, rv)


def apportion_largest_remainder(
    values: List[float],
    target: int
) -> List[int]:
    """
    使用最大余数法进行保和分配。

    Args:
        values: 非负浮点数列表
        target: 目标总和

    Returns:
        List[int]: 分配结果列表
    """
    n = len(values)
    if n == 0:
        return []
    if target <= 0:
        return [0] * n

    total = float(sum(max(0.0, float(v)) for v in values))
    if total <= 0:
        out = [0] * n
        out[0] = int(target)
        return out

    ratio = float(target) / total
    floors = _compute_floors(values, ratio)

    floor_sum = int(sum(x[1] for x in floors))
    remainder_count = int(max(0, target - floor_sum))

    floors.sort(key=lambda x: (-x[2], -x[3], x[4]))

    out = [0] * n
    for idx, fval, _, _, _ in floors:
        out[idx] = int(fval)

    for k in range(min(remainder_count, n)):
        out[floors[k][0]] += 1

    return out


def _compute_floors(
    values: List[float],
    ratio: float
) -> List[Tuple[int, int, float, float, int]]:
    """计算各项的地板值和余数。"""
    floors = []
    for pos, v in enumerate(values):
        orig = max(0.0, float(v))
        exact = orig * ratio
        fval = int(np.floor(exact))
        rem = float(exact - fval)
        floors.append((pos, fval, rem, orig, pos))
    return floors


def build_ptf_lsk_cache(
    m4_mlcfg_df: Optional[pd.DataFrame]
) -> Dict[Tuple[str, str], Tuple[int, int]]:
    """
    构建PTF/LSK查询缓存。

    Args:
        m4_mlcfg_df: M4配置DataFrame

    Returns:
        Dict: (material, location) -> (ptf, lsk) 缓存
    """
    cache = {}
    if m4_mlcfg_df is None or m4_mlcfg_df.empty:
        return cache

    for row in m4_mlcfg_df.itertuples():
        material = getattr(row, 'material', None)
        location = getattr(row, 'location', None)
        if material is None or location is None:
            continue

        ptf, lsk = _extract_ptf_lsk_from_row(row)
        cache[(str(material), str(location))] = (ptf, lsk)

    return cache


def _extract_ptf_lsk_from_row(row) -> Tuple[int, int]:
    """从行数据提取PTF/LSK值。"""
    ptf = 0
    lsk = 1

    ptf_val = getattr(row, 'ptf', None) or getattr(row, 'PTF', None)
    lsk_val = getattr(row, 'lsk', None) or getattr(row, 'LSK', None)

    if ptf_val is not None and not pd.isna(ptf_val):
        ptf = int(ptf_val)
    if lsk_val is not None and not pd.isna(lsk_val):
        lsk = int(lsk_val)

    return ptf, lsk


def get_ptf_lsk(
    material: str,
    site: str,
    m4_mlcfg_df: Optional[pd.DataFrame],
    cache: Optional[Dict[Tuple[str, str], Tuple[int, int]]] = None
) -> Tuple[int, int]:
    """
    从M4配置读取PTF/LSK值。

    Args:
        material: 物料编码
        site: 地点编码
        m4_mlcfg_df: M4配置DataFrame
        cache: PTF/LSK缓存

    Returns:
        Tuple[int, int]: (ptf, lsk) 元组
    """
    if cache is not None:
        return cache.get((str(material), str(site)), (0, 1))

    ptf, lsk = 0, 1
    if m4_mlcfg_df is None or m4_mlcfg_df.empty:
        return ptf, lsk

    ml = m4_mlcfg_df[
        (m4_mlcfg_df['material'] == material) &
        (m4_mlcfg_df['location'] == site)
    ]
    if ml.empty:
        return ptf, lsk

    return _extract_ptf_lsk_from_df(ml)


def _extract_ptf_lsk_from_df(ml: pd.DataFrame) -> Tuple[int, int]:
    """从DataFrame提取PTF/LSK值。"""
    ptf, lsk = 0, 1
    row = ml.iloc[0]

    if 'ptf' in ml.columns and pd.notna(row.get('ptf')):
        ptf = int(row['ptf'])
    elif 'PTF' in ml.columns and pd.notna(row.get('PTF')):
        ptf = int(row['PTF'])

    if 'lsk' in ml.columns and pd.notna(row.get('lsk')):
        lsk = int(row['lsk'])
    elif 'LSK' in ml.columns and pd.notna(row.get('LSK')):
        lsk = int(row['LSK'])

    return ptf, lsk
