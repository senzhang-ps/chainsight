"""
Module3 提前期计算模块。

负责提前期相关的计算和节点类型推断。
"""

from typing import Dict, Optional, Tuple

import pandas as pd

from src.utils.date_helpers import calculate_transport_lead_time

from .constants import (
    DEFAULT_HORIZON,
    LOCATION_TYPE_DC,
    LOCATION_TYPE_PLANT,
)
from .utils import get_ptf_lsk


def compute_root_horizon(
    material: str,
    location: str,
    lead_time_df: pd.DataFrame,
    m4_mlcfg_df: Optional[pd.DataFrame] = None,
    ptf_lsk_cache: Optional[Dict[Tuple[str, str], Tuple[int, int]]] = None
) -> int:
    """
    计算顶层节点的计划窗口horizon。

    公式: horizon = max(PDT+GR, MCT) + PTF + LSK - 1

    参数：
        material: 物料编码
        location: 地点编码
        lead_time_df: 提前期配置DataFrame
        m4_mlcfg_df: M4配置DataFrame
        ptf_lsk_cache: PTF/LSK缓存

    返回：
        int: 计划窗口天数
    """
    ptf, lsk = get_ptf_lsk(
        material=material,
        site=location,
        m4_mlcfg_df=m4_mlcfg_df,
        cache=ptf_lsk_cache
    )

    pdt, gr, mct = _get_lead_time_params(lead_time_df, location)
    base_lt = max(mct, pdt + gr)
    horizon = max(DEFAULT_HORIZON, int(base_lt + ptf + lsk - 1))
    return horizon


def _get_lead_time_params(
    lead_time_df: Optional[pd.DataFrame],
    location: str
) -> Tuple[int, int, int]:
    """获取PDT/GR/MCT参数。"""
    if lead_time_df is None or lead_time_df.empty:
        return 0, 0, 0

    df_loc = lead_time_df[
        lead_time_df['sending'].astype(str) == str(location)
    ]
    if df_loc.empty:
        return 0, 0, 0

    pdt = _get_max_numeric(df_loc, 'PDT')
    gr = _get_max_numeric(df_loc, 'GR')
    mct = _get_max_numeric(df_loc, 'MCT')

    return pdt, gr, mct


def _get_max_numeric(df: pd.DataFrame, col: str) -> int:
    """获取列的最大数值。"""
    return int(
        pd.to_numeric(df.get(col, 0), errors='coerce').fillna(0).max()
    )


def determine_lead_time(
    sending: str,
    receiving: str,
    location_type: str,
    lead_time_df: pd.DataFrame,
    m4_mlcfg_df: Optional[pd.DataFrame] = None,
    material: Optional[str] = None,
    ptf_lsk_cache: Optional[Dict[Tuple[str, str], Tuple[int, int]]] = None
) -> Tuple[int, str]:
    """
    确定提前期。

    参数：
        sending: 发送节点
        receiving: 接收节点
        location_type: 发送端类型
        lead_time_df: 提前期配置
        m4_mlcfg_df: M4配置
        material: 物料编码
        ptf_lsk_cache: PTF/LSK缓存

    返回：
        Tuple[int, str]: (提前期, 错误信息)
    """
    if lead_time_df.empty:
        return DEFAULT_HORIZON, 'empty_lead_time_config'

    row = lead_time_df[
        (lead_time_df['sending'] == sending) &
        (lead_time_df['receiving'] == receiving)
    ]
    if row.empty:
        return DEFAULT_HORIZON, 'lead_time_missing'

    try:
        leadtime = _calculate_lead_time(
            row.iloc[0], location_type, material,
            sending, m4_mlcfg_df, ptf_lsk_cache
        )
        return max(0, int(leadtime)), ""
    except Exception as e:
        return 0, f'lead_time_calculation_error: {e}'


def _calculate_lead_time(
    row: pd.Series,
    location_type: str,
    material: Optional[str],
    sending: str,
    m4_mlcfg_df: Optional[pd.DataFrame],
    ptf_lsk_cache: Optional[Dict]
) -> int:
    """根据节点类型计算提前期。"""
    pdt = int(row.get('PDT', 0) or 0)
    gr = int(row.get('GR', 0) or 0)
    mct = int(row.get('MCT', 0) or 0)

    ptf, lsk = 0, 1
    if str(location_type).lower() == 'plant' and material:
        ptf, lsk = get_ptf_lsk(
            material=material,
            site=sending,
            m4_mlcfg_df=m4_mlcfg_df,
            cache=ptf_lsk_cache
        )

    return calculate_transport_lead_time(
        pdt=pdt,
        gr=gr,
        mct=mct,
        location_type=location_type,
        ptf=ptf,
        lsk=lsk,
        minimum=0,
    )


def infer_sending_location_type(
    network_df: pd.DataFrame,
    location_layer_map: Dict[Tuple[str, str], int],
    sending: str,
    material: Optional[str],
    sim_date: pd.Timestamp
) -> str:
    """
    推断发送端的 location_type：
    1) 若存在 (material, location==sending) 的显式配置，直接使用其 location_type
    2) 若 sending 是根节点(layer==0)，判为 'Plant'
    3) 若 sending 只在 sourcing 列出现、从不在 location 列出现，判为 'Plant'
    4) 其他情况默认为 'DC'

    参数：
        network_df: 网络配置DataFrame
        location_layer_map: 节点层级映射 dict[(material, location): layer]
        sending: 发送节点标识
        material: 物料编码
        sim_date: 模拟日期

    返回：
        str: 'Plant' 或 'DC'
    """
    if sending is None or (isinstance(sending, float) and pd.isna(sending)) or str(sending).strip() == '':
        return LOCATION_TYPE_DC

    # ① 显式配置（同物料、有效期内）
    if material is not None and not network_df.empty:
        explicit = network_df[
            (network_df['material'] == material) &
            (network_df['location'] == sending) &
            (network_df['eff_from'] <= sim_date) &
            (network_df['eff_to'] >= sim_date)
        ]
        if not explicit.empty:
            t = explicit.iloc[0].get('location_type', None)
            if isinstance(t, str) and t.strip():
                return t

    # ② 根节点（layer==0）→ Plant
    if location_layer_map:
        mat_key = '' if material is None else str(material)
        if location_layer_map.get((mat_key, str(sending)), None) == 0:
            return LOCATION_TYPE_PLANT

    # ③ 只在 sourcing 中出现、从不在 location 中出现 → Plant
    #    （处理"源头 Plant 只维护在 sourcing 列"的常见情况）
    appears_as_sourcing = network_df['sourcing'].astype(str).eq(str(sending)).any()
    appears_as_location = network_df['location'].astype(str).eq(str(sending)).any()
    if appears_as_sourcing and not appears_as_location:
        return LOCATION_TYPE_PLANT

    # ④ 兜底
    return LOCATION_TYPE_DC
