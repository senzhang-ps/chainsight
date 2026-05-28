# -*- coding: utf-8 -*-
"""确定性输出排序工具。

并行流水线（如 module5 的 ``as_completed``）会让"等价行"在每次运行间的输入顺序
发生漂移。仅按业务优先列做稳定排序时，这些在优先列上完全相等的行会保持输入顺序，
从而导致落盘行序在多次运行间不可复现（逐行比对出现大量"仅行顺序不同"的差异）。

本工具在给定的优先列之后，追加 DataFrame 其余所有列作为次级排序键：
只有当两行在 *所有* 列上都完全相同时才可能保持输入顺序——而这种完全相同行的
相互交换不会影响任何数值或逐行比对结果。由此保证本地链路输出行序可复现。

注意：排序只改变行的顺序，不修改任何单元格的值或 dtype，因此输出精度保持不变。
"""
from __future__ import annotations

from typing import Sequence

import pandas as pd


def stable_sort_for_output(
    df: pd.DataFrame,
    preferred_cols: Sequence[str],
) -> pd.DataFrame:
    """按 ``preferred_cols`` 排序并以其余所有列兜底，返回行序确定的副本。

    Args:
        df: 待排序的输出 DataFrame。
        preferred_cols: 业务优先排序列（不存在的列会被忽略）。

    Returns:
        行序确定、值与 dtype 不变的新 DataFrame。

    实现要点：
        - 优先列保持原生（数值/日期）排序，便于阅读；其余列仅作为兜底键以消除
          并行输入顺序带来的不确定性。
        - 对含混合类型、无法直接比较的列（``sort_values`` 抛 ``TypeError``），
          回退到字符串键排序以确定行序，最终仍按原始值重排——精度不受影响。
    """
    if df.empty:
        return df

    primary = [c for c in preferred_cols if c in df.columns]
    remaining = [c for c in df.columns if c not in primary]
    sort_cols = primary + remaining
    if not sort_cols:
        return df.reset_index(drop=True)

    try:
        return df.sort_values(by=sort_cols, kind="mergesort").reset_index(drop=True)
    except TypeError:
        # 某列含不可直接比较的混合类型：用字符串表示作为排序键，仅用于确定行序，
        # 最终仍按原始值（df.loc[order]）重排，单元格的值与精度均不改变。
        ranks = {
            f"__sortkey_{idx}__": df[col].astype(str)
            for idx, col in enumerate(sort_cols)
        }
        order = (
            pd.DataFrame(ranks, index=df.index)
            .sort_values(by=list(ranks), kind="mergesort")
            .index
        )
        return df.loc[order].reset_index(drop=True)
