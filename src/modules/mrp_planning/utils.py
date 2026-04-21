"""
Module3 工具函数模块。

提供MOQ/RV计算、标识符规范化、分配算法等通用功能。
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.normalization import (
    normalize_identifiers as _canonical_normalize_identifiers,
)

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

    参数：
        qty: 需求数量
        moq: 最小订货量
        rv: 重订量
        is_cross_node: 是否为跨节点调运

    返回：
        int: 调整后的补货数量

    示例：
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


def normalize_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    规范化DataFrame中的标识符列。
    
    使用向量化操作提升性能。

    参数：
        df: 需要规范化的DataFrame

    返回：
        pd.DataFrame: 规范化后的DataFrame副本
    """
    other_cols = [
        c
        for c in IDENTIFIER_COLUMNS
        if c not in LOCATION_TYPE_COLUMNS and c != COL_MATERIAL
    ]
    return _canonical_normalize_identifiers(
        df,
        material_cols=(COL_MATERIAL,),
        location_cols=tuple(LOCATION_TYPE_COLUMNS),
        other_identifier_cols=tuple(other_cols),
    )


def lookup_moq_rv_three_keys(
    deploy_config_df: Optional[pd.DataFrame],
    material: str,
    sending: str,
    receiving: Optional[str]
) -> Tuple[int, int]:
    """
    按三键查询MOQ/RV配置。

    优先级: (material, sending, receiving) > (material, sending) > 默认值

    参数：
        deploy_config_df: 部署配置DataFrame
        material: 物料编码
        sending: 发送节点
        receiving: 接收节点

    返回：
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

    参数：
        values: 非负浮点数列表
        target: 目标总和

    返回：
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
