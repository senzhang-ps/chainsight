"""数值安全转换工具。

正常计算和数据库写入前统一把 NaN、±inf、不可转换值替换为 0，
避免裸的 ``pandas.errors.IntCastingNaNError``。

真正的程序/数据库异常不应被吞掉，应继续抛出。
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger("SupplyChainSimulation")

_SAMPLE_LIMIT = 5
_INDEX_SAMPLE_LIMIT = 10


def _bad_numeric_details(values: "pd.Series") -> tuple["np.ndarray", "np.ndarray", list, list]:
    """返回 float64 数组、异常掩码、样例原值、样例位置。"""
    numeric = pd.to_numeric(values, errors="coerce")
    arr = np.asarray(numeric, dtype="float64")
    bad_mask = ~np.isfinite(arr)  # 同时覆盖 NaN、+inf、-inf
    sample_vals: list = []
    sample_idx: list = []
    if bad_mask.any():
        sample_vals = list(pd.Series(values).to_numpy()[bad_mask][:_SAMPLE_LIMIT])
        try:
            sample_idx = list(pd.Series(values).index.to_numpy()[bad_mask][:_INDEX_SAMPLE_LIMIT])
        except Exception:
            sample_idx = list(np.flatnonzero(bad_mask)[:_INDEX_SAMPLE_LIMIT])
    return arr, bad_mask, sample_vals, sample_idx


def _coerce_finite_to_default(values: "pd.Series", context: str, default: int) -> "np.ndarray":
    """转 float64 数组；NaN/±inf/不可转换值替换为 default 并记录定位信息。"""
    arr, bad_mask, sample_vals, sample_idx = _bad_numeric_details(values)
    if bad_mask.any():
        bad_count = int(bad_mask.sum())
        logger.warning(
            "[安全整数转换] %s：发现 %d 个异常值（NaN/inf/不可转换），已替换为 %d；"
            "样例原值=%s；样例位置=%s",
            context, bad_count, default, sample_vals, sample_idx,
        )
        arr = np.where(bad_mask, float(default), arr)
    return arr


def safe_int_series(
    values: Any,
    context: str,
    default: int = 0,
) -> "pd.Series":
    """把 ``values`` 安全转换为 int ``Series``。

    - NaN / ±inf / 不可转换值 → ``default``（默认 0），并记录 WARNING。
    - 截断取整（与 ``astype(int)`` 行为一致），不四舍五入。
    - 保留传入 ``Series`` 的 index；传入 list/ndarray 时使用 ``RangeIndex``。
    """
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    arr = _coerce_finite_to_default(series, context, default)
    return pd.Series(arr.astype(np.int64), index=series.index)


def safe_int_array(
    values: Any,
    context: str,
    default: int = 0,
) -> "np.ndarray":
    """``safe_int_series`` 的 ndarray 版本，返回 ``int64`` 数组。"""
    return safe_int_series(values, context, default=default).to_numpy()
