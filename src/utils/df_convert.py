"""DataFrame 引擎互转工具。

pandas ↔ polars 的统一转换入口，列级 numpy 转换、不依赖 pyarrow。

抽取自 ``_PolarsBackend._to_pl``，供后端与持久化读取边界共用，
避免在各处重复实现"pandas → polars"逻辑。
"""

from datetime import date, datetime

import pandas as pd
import polars as pl


def pandas_to_polars(df: pd.DataFrame) -> pl.DataFrame:
    """pandas DataFrame → polars DataFrame（列级 numpy，无需 pyarrow）。

    将 nullable / extension dtypes 转为 numpy-backed dtypes，避免
    ``pl.from_pandas`` 的 pyarrow 依赖；空表（无列）返回空 polars DataFrame。

    Args:
        df: pandas DataFrame（``None`` 也允许）。

    Returns:
        等价的 polars DataFrame。
    """
    if df is None or len(df.columns) == 0:
        return pl.DataFrame()
    converted = {}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            converted[col] = [value.to_pydatetime() if not pd.isna(value) else None for value in s]
            continue
        values = s.dropna()
        # Arrow extension array 的全列 ``Series.map`` 会逐元素回调 Python；
        # M1 的大日期列会在 prepare 阶段因此停滞。日期列是同质列，只抽样
        # 有效值即可判定，并保留下面原来的逐值日期转换语义。
        date_sample = values.iloc[:256]
        if not date_sample.empty and date_sample.map(
            lambda value: isinstance(value, (pd.Timestamp, datetime, date))
        ).all():
            converted[col] = [
                value.to_pydatetime() if isinstance(value, pd.Timestamp) else value if not pd.isna(value) else None
                for value in s
            ]
            continue
        if pd.api.types.is_extension_array_dtype(s.dtype):
            # PostgreSQL 读取后经模型投影会产生 pandas nullable dtype；其
            # 缺失标记 ``pd.NA`` 不能由 Polars 的 numpy object 构造器判断，
            # 必须在引擎边界归一为 Python ``None``。不能保留 numpy object
            # 数组，否则 Polars 会将整列推断为 Object，后续数值 cast 失败。
            converted[col] = [None if pd.isna(value) else value for value in s]
            continue
        try:
            # 先尝试直接转 numpy
            arr = s.to_numpy()
            if arr.dtype == object and s.isna().any():
                converted[col] = [None if pd.isna(value) else value for value in s]
                continue
            converted[col] = arr
        except Exception:
            # 降级为 str
            converted[col] = s.astype(str).to_numpy()
    return pl.DataFrame(converted)
