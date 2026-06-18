"""数据对比工具 — 重构前后 DataFrame 一致性校验。

用于验证重构后的代码输出与原始代码输出一致。
"""
from __future__ import annotations

import pandas as pd


def compare_dataframes(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    *,
    label: str = "",
    ignore_cols: list[str] | None = None,
    ignore_row_order: bool = True,
    float_tolerance: float = 1e-6,
) -> dict:
    """对比两个 DataFrame 是否一致。

    Args:
        df1: 第一个 DataFrame（如重构前）。
        df2: 第二个 DataFrame（如重构后）。
        label: 对比标签（用于日志/错误信息）。
        ignore_cols: 忽略的列名列表（如元数据列 db_write_time）。
        ignore_row_order: 是否忽略行顺序（默认 True，会排序后对比）。
        float_tolerance: 浮点数比较容差。

    Returns:
        ``{'match': bool, 'diffs': list[str], 'label': str}``
    """
    diffs: list[str] = []

    if df1 is None and df2 is None:
        return {"match": True, "diffs": [], "label": label}
    if df1 is None:
        return {"match": False, "diffs": ["df1 为 None，df2 不为 None"], "label": label}
    if df2 is None:
        return {"match": False, "diffs": ["df2 为 None，df1 不为 None"], "label": label}

    # 去除忽略列
    if ignore_cols:
        df1 = df1.drop(columns=[c for c in ignore_cols if c in df1.columns], errors="ignore")
        df2 = df2.drop(columns=[c for c in ignore_cols if c in df2.columns], errors="ignore")

    # 列名对比
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)
    if cols1 != cols2:
        only_1 = cols1 - cols2
        only_2 = cols2 - cols1
        if only_1:
            diffs.append(f"仅在 df1: {sorted(only_1)}")
        if only_2:
            diffs.append(f"仅在 df2: {sorted(only_2)}")

    # 行数对比
    if len(df1) != len(df2):
        diffs.append(f"行数不一致: df1={len(df1)}, df2={len(df2)}")

    if diffs:
        return {"match": False, "diffs": diffs, "label": label}

    # 共同列
    common_cols = sorted(cols1 & cols2)
    if not common_cols:
        diffs.append("无共同列")
        return {"match": False, "diffs": diffs, "label": label}

    # 行顺序对齐
    if ignore_row_order:
        # 按所有共同列排序
        df1 = df1.sort_values(common_cols).reset_index(drop=True)
        df2 = df2.sort_values(common_cols).reset_index(drop=True)

    # 逐列对比
    for col in common_cols:
        s1 = df1[col].reset_index(drop=True)
        s2 = df2[col].reset_index(drop=True)

        # 浮点列用容差对比
        if pd.api.types.is_float_dtype(s1) and pd.api.types.is_float_dtype(s2):
            diff_mask = (s1 - s2).abs() > float_tolerance
            # 处理 NaN：NaN != NaN，但 NaN == NaN 应该视为匹配
            both_na = s1.isna() & s2.isna()
            diff_mask = diff_mask & ~both_na
            if diff_mask.any():
                n_diff = int(diff_mask.sum())
                diffs.append(f"列 '{col}': {n_diff} 行浮点值超出容差 {float_tolerance}")
        else:
            # 非浮点列：严格相等（NaN 视为相等）
            s1_filled = s1.fillna("__NA__")
            s2_filled = s2.fillna("__NA__")
            if not s1_filled.equals(s2_filled):
                n_diff = int((s1_filled != s2_filled).sum())
                diffs.append(f"列 '{col}': {n_diff} 行值不一致")

    return {
        "match": len(diffs) == 0,
        "diffs": diffs,
        "label": label,
    }


def assert_frames_equal(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    *,
    label: str = "",
    ignore_cols: list[str] | None = None,
    ignore_row_order: bool = True,
):
    """断言两个 DataFrame 一致，不一致时抛出 AssertionError 并打印详细差异。

    Args:
        df1: 第一个 DataFrame。
        df2: 第二个 DataFrame。
        label: 对比标签。
        ignore_cols: 忽略的列名列表。
        ignore_row_order: 是否忽略行顺序。

    Raises:
        AssertionError: DataFrames 不一致。
    """
    result = compare_dataframes(
        df1, df2,
        label=label,
        ignore_cols=ignore_cols,
        ignore_row_order=ignore_row_order,
    )
    if not result["match"]:
        header = f"DataFrame 对比失败{f' [{label}]' if label else ''}:"
        details = "\n  - ".join(result["diffs"])
        raise AssertionError(f"{header}\n  - {details}")