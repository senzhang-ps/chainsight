"""
PTF/LSK 缓存与查询工具（共享模块）。

PTF (Planning Time Fence) 与 LSK (Lot-Sizing Key) 是 M3 (mrp_planning) 与
M5 (deployment_planning) 共同依赖的物料-地点级别参数，原本在两个模块中
分别有几乎相同的实现。本模块作为唯一权威来源，统一了二者。

数据来源：
    M4_MaterialLocationLineCfg 表（列：material, location, ptf/PTF, lsk/LSK）

默认值来自 ``config/defaults.yaml``，由 ``src.utils.defaults`` 暴露为
``DEFAULT_PTF`` / ``DEFAULT_LSK``。
"""

from typing import Dict, Optional, Tuple

import pandas as pd

from src.utils.defaults import DEFAULT_LSK, DEFAULT_PTF


PtfLskCache = Dict[Tuple[str, str], Tuple[int, int]]


def _extract_ptf_lsk_from_row(row) -> Tuple[int, int]:
    """从 itertuples 行提取 PTF/LSK，缺失时回退默认值。"""
    ptf: int = DEFAULT_PTF
    lsk: int = DEFAULT_LSK

    ptf_val = getattr(row, 'ptf', None)
    if ptf_val is None:
        ptf_val = getattr(row, 'PTF', None)
    lsk_val = getattr(row, 'lsk', None)
    if lsk_val is None:
        lsk_val = getattr(row, 'LSK', None)

    if ptf_val is not None and not pd.isna(ptf_val):
        ptf = int(ptf_val)
    if lsk_val is not None and not pd.isna(lsk_val):
        lsk = int(lsk_val)

    return ptf, lsk


def build_ptf_lsk_cache(
    m4_mlcfg_df: Optional[pd.DataFrame],
) -> PtfLskCache:
    """构建 (material, location) -> (ptf, lsk) 缓存。

    参数：
        m4_mlcfg_df: M4_MaterialLocationLineCfg DataFrame；为 None/空时返回空缓存。

    返回：
        dict: 以 (material, location) 字符串元组为键的缓存。
    """
    cache: PtfLskCache = {}
    if m4_mlcfg_df is None or m4_mlcfg_df.empty:
        return cache

    for row in m4_mlcfg_df.itertuples():
        material = getattr(row, 'material', None)
        location = getattr(row, 'location', None)
        if material is None or location is None:
            continue
        cache[(str(material), str(location))] = _extract_ptf_lsk_from_row(row)

    return cache


def get_ptf_lsk(
    material: str,
    site: str,
    m4_mlcfg_df: Optional[pd.DataFrame],
    cache: Optional[PtfLskCache] = None,
) -> Tuple[int, int]:
    """获取指定 (material, site) 的 PTF/LSK 值。

    优先查询缓存，缓存未命中或未提供时退回 DataFrame 查询；仍未命中返回默认值。

    参数：
        material: 物料编码
        site: 地点编码
        m4_mlcfg_df: M4 配置 DataFrame
        cache: 由 :func:`build_ptf_lsk_cache` 构建的缓存

    返回：
        Tuple[int, int]: (ptf, lsk)
    """
    if cache is not None:
        return cache.get((str(material), str(site)), (DEFAULT_PTF, DEFAULT_LSK))

    ptf, lsk = DEFAULT_PTF, DEFAULT_LSK
    if m4_mlcfg_df is None or m4_mlcfg_df.empty:
        return ptf, lsk

    ml = m4_mlcfg_df[
        (m4_mlcfg_df['material'] == material)
        & (m4_mlcfg_df['location'] == site)
    ]
    if ml.empty:
        return ptf, lsk

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


__all__ = [
    "PtfLskCache",
    "build_ptf_lsk_cache",
    "get_ptf_lsk",
]
