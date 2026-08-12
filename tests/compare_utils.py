"""数据对比工具 — 重构前后 DataFrame 一致性校验。

用于验证重构后的代码输出与原始代码输出一致。
"""
from __future__ import annotations

import numbers

import pandas as pd


DEFAULT_KEY_PRIORITY = [
    "simulation_date", "production_plan_date", "available_date", "date",
    "requirement_date", "material", "location", "line", "sending",
    "receiving", "demand_element", "changeover_id", "changeover_type",
]


def _normalize_key_series(series: pd.Series, column: str) -> pd.Series:
    """将业务关联键标准化为可稳定比较的字符串。"""
    if "date" in column.lower():
        parsed = pd.to_datetime(series, errors="coerce")
        return parsed.dt.strftime("%Y-%m-%d").fillna("<NA>")
    return series.astype("string").fillna("<NA>").str.strip()


def _select_join_keys(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    """选择两侧共有且最适合业务关联的列集合。"""
    shared = set(left.columns) & set(right.columns)
    keys = [column for column in DEFAULT_KEY_PRIORITY if column in shared]
    if keys:
        return keys
    return sorted(shared)


def compare_dataframes_by_key(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    label: str = "",
    key_columns: list[str] | None = None,
    float_tolerance: float = 1e-6,
    sample_limit: int = 5,
) -> dict:
    """按业务键关联两个 DataFrame，并输出逐列差异统计。

    当业务键不唯一时，函数为每个键组追加稳定序号，确保不会产生多对多
    笛卡尔匹配。绝对差不大于 ``float_tolerance``（含零附近数值）单列为
    ``precision_differences``，不会与业务数值差异混在一起。
    """
    left = pd.DataFrame() if left is None else left.copy()
    right = pd.DataFrame() if right is None else right.copy()
    keys = key_columns or _select_join_keys(left, right)
    keys = [column for column in keys if column in left.columns and column in right.columns]
    result = {
        "label": label,
        "left_rows": len(left),
        "right_rows": len(right),
        "key_columns": keys,
        "left_only_keys": 0,
        "right_only_keys": 0,
        "matched_rows": 0,
        "column_differences": {},
        "precision_differences": {},
        "schema": {
            "left_only_columns": sorted(set(left.columns) - set(right.columns)),
            "right_only_columns": sorted(set(right.columns) - set(left.columns)),
        },
    }
    if not keys:
        result["error"] = "无可用的共有业务关联键"
        return result
    if left.empty and right.empty:
        return result
    if left.empty:
        result["right_only_keys"] = len(right)
        return result
    if right.empty:
        result["left_only_keys"] = len(left)
        return result

    def prepare(frame: pd.DataFrame) -> pd.DataFrame:
        prepared = frame.copy()
        for column in keys:
            prepared[column] = _normalize_key_series(prepared[column], column)
        sort_columns = keys + sorted(column for column in prepared.columns if column not in keys)
        prepared = prepared.sort_values(sort_columns, kind="mergesort", na_position="last")
        prepared["__duplicate_ordinal"] = prepared.groupby(keys, dropna=False).cumcount()
        return prepared

    left = prepare(left)
    right = prepare(right)
    join_keys = keys + ["__duplicate_ordinal"]
    merged = left.merge(
        right,
        on=join_keys,
        how="outer",
        suffixes=("__left", "__right"),
        indicator=True,
    )
    result["left_only_keys"] = int((merged["_merge"] == "left_only").sum())
    result["right_only_keys"] = int((merged["_merge"] == "right_only").sum())
    matched = merged[merged["_merge"] == "both"].copy()
    result["matched_rows"] = len(matched)
    sample_columns = [column for column in merged.columns if column != "_merge"]
    result["left_only_samples"] = (
        merged.loc[merged["_merge"] == "left_only", sample_columns]
        .head(sample_limit)
        .to_dict("records")
    )
    result["right_only_samples"] = (
        merged.loc[merged["_merge"] == "right_only", sample_columns]
        .head(sample_limit)
        .to_dict("records")
    )

    common_columns = sorted((set(left.columns) & set(right.columns)) - set(join_keys))
    for column in common_columns:
        left_value = matched[f"{column}__left"]
        right_value = matched[f"{column}__right"]

        def is_numeric_series(series: pd.Series) -> bool:
            """兼容 DataFrame 中混入 Python int/float 的 object 列。"""
            if pd.api.types.is_numeric_dtype(series):
                return True
            values = series.dropna()
            return not values.empty and values.map(
                lambda value: isinstance(value, numbers.Number) and not isinstance(value, bool)
            ).all()

        numeric = (
            is_numeric_series(left_value)
            and is_numeric_series(right_value)
        )
        if numeric:
            difference = (left_value.astype(float) - right_value.astype(float)).abs()
            both_null = left_value.isna() & right_value.isna()
            precision = (difference > 0) & (difference <= float_tolerance) & ~both_null
            business = (difference > float_tolerance) & ~both_null
            if precision.any():
                result["precision_differences"][column] = {
                    "count": int(precision.sum()),
                    "max_abs_difference": float(difference[precision].max()),
                }
        else:
            if "date" in column.lower():
                left_value = _normalize_key_series(left_value, column)
                right_value = _normalize_key_series(right_value, column)
            else:
                left_value = left_value.astype("string").fillna("<NA>")
                right_value = right_value.astype("string").fillna("<NA>")
            business = left_value != right_value

        if business.any():
            samples = matched.loc[business, join_keys + [f"{column}__left", f"{column}__right"]]
            result["column_differences"][column] = {
                "count": int(business.sum()),
                "samples": samples.head(sample_limit).to_dict("records"),
            }
    return result


def flatten_mapping(mapping: dict | None) -> pd.DataFrame:
    """递归摊平嵌套状态字典，以 ``path/value`` DataFrame 统一参与对比。"""
    rows: list[dict] = []

    def visit(value, path: str) -> None:
        if isinstance(value, dict):
            for key in sorted(value, key=str):
                visit(value[key], f"{path}.{key}" if path else str(key))
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
        elif isinstance(value, numbers.Number) and not isinstance(value, bool):
            rows.append({"path": path, "value": float(value)})
        else:
            rows.append({"path": path, "value": value})

    visit(mapping or {}, "")
    return pd.DataFrame(rows, columns=["path", "value"])


def compare_mappings(
    left: dict | None,
    right: dict | None,
    *,
    label: str = "",
    float_tolerance: float = 1e-6,
) -> dict:
    """递归对比两个状态字典。"""
    return compare_dataframes_by_key(
        flatten_mapping(left),
        flatten_mapping(right),
        label=label,
        key_columns=["path"],
        float_tolerance=float_tolerance,
    )


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