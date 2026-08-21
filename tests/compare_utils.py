"""数据对比工具 — 重构前后 DataFrame 一致性校验。

用于验证重构后的代码输出与原始代码输出一致。
"""
from __future__ import annotations

import json
import numbers
from collections import Counter

import pandas as pd


DEFAULT_KEY_PRIORITY = [
    "sim_date", "simulation_date", "production_plan_date", "available_date", "date",
    "requirement_date", "material", "location", "line", "sending",
    "receiving", "demand_element", "changeover_id", "changeover_type",
]

_NUMERIC_KEY_HINTS = (
    "qty", "quantity", "amount", "volume", "capacity", "inventory", "stock",
    "demand", "supply", "shipment", "delivery", "production", "cut", "hours",
    "count", "utilization", "rate", "value", "cost", "weight", "leadtime", "deploy",
    "quota", "pct", "wfr", "vfr", "time", "loss",
)


def _normalize_key_series(series: pd.Series, column: str) -> pd.Series:
    """将业务关联键标准化为可稳定比较的字符串。"""
    if "date" in column.lower() or column.casefold() in {"week_start", "week_end"}:
        parsed = pd.to_datetime(series, errors="coerce")
        return parsed.dt.strftime("%Y-%m-%d").fillna("<NA>")
    if any(hint in column.casefold() for hint in _NUMERIC_KEY_HINTS):
        values = series.dropna()
        numeric = pd.to_numeric(values, errors="coerce")
        if not values.empty and numeric.notna().all():
            normalized = pd.to_numeric(series, errors="coerce")
            return normalized.map(
                lambda value: "<NA>" if pd.isna(value) else format(float(value), ".12g")
            )
    normalized = series.astype("string").fillna("<NA>").str.strip()
    return normalized.replace({"": "<NA>", "nan": "<NA>", "None": "<NA>", "<NA>": "<NA>"})


def _numeric_series_or_none(series: pd.Series) -> pd.Series | None:
    """返回可完整解析的数值序列；混合文本或业务标识则返回 ``None``。

    PostgreSQL 动态列可能将同一指标读回为 ``float``、``int`` 或数值字符串。
    比较业务字段时应统一为数值，不能把 $3582.0$ 与 ``"3582"`` 误判为差异。
    """
    values = series.dropna()
    if values.empty:
        return None
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        return None
    return pd.to_numeric(series, errors="coerce")


def _select_join_keys(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    """选择两侧共有且最适合业务关联的列集合。"""
    shared = set(left.columns) & set(right.columns)
    keys = [column for column in DEFAULT_KEY_PRIORITY if column in shared]
    if keys:
        return keys
    return sorted(shared)


def _drop_excluded_columns(
    frame: pd.DataFrame,
    excluded_columns: list[str] | None,
) -> pd.DataFrame:
    """删除不应参与业务关联或差异判定的技术列。"""
    if not excluded_columns:
        return frame
    excluded = {str(column).casefold() for column in excluded_columns}
    return frame.drop(
        columns=[column for column in frame.columns if str(column).casefold() in excluded],
        errors="ignore",
    )


def compare_dataframes_as_multiset(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    label: str = "",
    excluded_columns: list[str] | None = None,
) -> dict:
    """按完整规范化业务行的多重集比较，适用于无稳定唯一键的明细报表。"""
    left = _drop_excluded_columns(pd.DataFrame() if left is None else left.copy(), excluded_columns)
    right = _drop_excluded_columns(pd.DataFrame() if right is None else right.copy(), excluded_columns)
    columns = sorted(set(left.columns) & set(right.columns))
    result = {
        "label": label,
        "left_rows": len(left), "right_rows": len(right),
        "key_columns": ["<full_business_row_multiset>"],
        "excluded_columns": sorted({str(column) for column in excluded_columns or []}),
        "left_only_keys": 0, "right_only_keys": 0, "matched_rows": 0,
        "column_differences": {}, "precision_differences": {},
        "schema": {
            "left_only_columns": sorted(set(left.columns) - set(right.columns)),
            "right_only_columns": sorted(set(right.columns) - set(left.columns)),
        },
    }

    def canonical(frame: pd.DataFrame) -> Counter:
        work = frame.reindex(columns=columns).copy()
        for column in columns:
            work[column] = _normalize_key_series(work[column], column)
        return Counter(map(tuple, work.itertuples(index=False, name=None)))

    left_rows, right_rows = canonical(left), canonical(right)
    left_only, right_only = left_rows - right_rows, right_rows - left_rows
    result["left_only_keys"] = int(sum(left_only.values()))
    result["right_only_keys"] = int(sum(right_only.values()))
    result["matched_rows"] = int(sum((left_rows & right_rows).values()))
    result["left_only_samples"] = [dict(zip(columns, row)) for row in list(left_only)[:5]]
    result["right_only_samples"] = [dict(zip(columns, row)) for row in list(right_only)[:5]]
    return result


def compare_dataframes_by_key(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    label: str = "",
    key_columns: list[str] | None = None,
    excluded_columns: list[str] | None = None,
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
    left = _drop_excluded_columns(left, excluded_columns)
    right = _drop_excluded_columns(right, excluded_columns)
    keys = key_columns or _select_join_keys(left, right)
    keys = [column for column in keys if column in left.columns and column in right.columns]
    result = {
        "label": label,
        "left_rows": len(left),
        "right_rows": len(right),
        "key_columns": keys,
        "excluded_columns": sorted({str(column) for column in excluded_columns or []}),
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

        left_numeric = _numeric_series_or_none(left_value)
        right_numeric = _numeric_series_or_none(right_value)
        if left_numeric is not None and right_numeric is not None:
            difference = (left_numeric - right_numeric).abs()
            both_null = left_value.isna() & right_value.isna()
            precision = (difference > 0) & (difference <= float_tolerance) & ~both_null
            business = (difference > float_tolerance) & ~both_null
            if precision.any():
                result["precision_differences"][column] = {
                    "count": int(precision.sum()),
                    "max_abs_difference": float(difference[precision].max()),
                }
        else:
            if "date" in column.lower() or column.casefold() in {"week_start", "week_end"}:
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


DIFFERENCE_DETAIL_COLUMNS = [
    "difference_type", "column", "key_values", "duplicate_ordinal",
    "left_value", "right_value", "abs_difference", "left_row", "right_row",
]


def dataframe_difference_details(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    key_columns: list[str] | None = None,
    excluded_columns: list[str] | None = None,
    float_tolerance: float = 1e-6,
) -> pd.DataFrame:
    """返回完整的按业务键行级差异，适合直接落盘审计。

    与 :func:`compare_dataframes_by_key` 使用相同的键选择、日期归一化及重复
    键序号规则。输出包含仅存在于任一侧的整行 JSON，以及共有键下逐字段的
    业务/精度差异，避免报告仅保留少量样例而丢失可复现明细。
    """
    left = pd.DataFrame() if left is None else left.copy()
    right = pd.DataFrame() if right is None else right.copy()
    left = _drop_excluded_columns(left, excluded_columns)
    right = _drop_excluded_columns(right, excluded_columns)
    keys = key_columns or _select_join_keys(left, right)
    keys = [column for column in keys if column in left.columns and column in right.columns]
    rows: list[dict] = []

    if not keys:
        return pd.DataFrame([{
            "difference_type": "comparison_error",
            "column": "",
            "key_values": "{}",
            "duplicate_ordinal": "",
            "left_value": "",
            "right_value": "",
            "abs_difference": "",
            "left_row": "",
            "right_row": "",
        }], columns=DIFFERENCE_DETAIL_COLUMNS)

    def prepare(frame: pd.DataFrame) -> pd.DataFrame:
        prepared = frame.copy()
        for column in keys:
            prepared[column] = _normalize_key_series(prepared[column], column)
        sort_columns = keys + sorted(column for column in prepared.columns if column not in keys)
        prepared = prepared.sort_values(sort_columns, kind="mergesort", na_position="last")
        prepared["__duplicate_ordinal"] = prepared.groupby(keys, dropna=False).cumcount()
        return prepared

    left, right = prepare(left), prepare(right)
    join_keys = keys + ["__duplicate_ordinal"]
    merged = left.merge(
        right,
        on=join_keys,
        how="outer",
        suffixes=("__left", "__right"),
        indicator=True,
    )

    def as_json(record: dict) -> str:
        return json.dumps(record, ensure_ascii=False, default=str)

    def key_values(row: pd.Series) -> str:
        return as_json({column: row[column] for column in keys})

    left_columns = [column for column in left.columns if column not in join_keys]
    right_columns = [column for column in right.columns if column not in join_keys]

    def merged_value(row: pd.Series, column: str, side: str):
        suffix = f"{column}__{side}"
        return row[suffix] if suffix in row else row.get(column)

    for _, row in merged.loc[merged["_merge"] == "left_only"].iterrows():
        rows.append({
            "difference_type": "left_only_row",
            "column": "",
            "key_values": key_values(row),
            "duplicate_ordinal": row["__duplicate_ordinal"],
            "left_value": "",
            "right_value": "",
            "abs_difference": "",
            "left_row": as_json({column: merged_value(row, column, "left") for column in left_columns}),
            "right_row": "",
        })
    for _, row in merged.loc[merged["_merge"] == "right_only"].iterrows():
        rows.append({
            "difference_type": "right_only_row",
            "column": "",
            "key_values": key_values(row),
            "duplicate_ordinal": row["__duplicate_ordinal"],
            "left_value": "",
            "right_value": "",
            "abs_difference": "",
            "left_row": "",
            "right_row": as_json({column: merged_value(row, column, "right") for column in right_columns}),
        })

    matched = merged.loc[merged["_merge"] == "both"].copy()
    common_columns = sorted((set(left.columns) & set(right.columns)) - set(join_keys))
    for column in common_columns:
        left_value = matched[f"{column}__left"]
        right_value = matched[f"{column}__right"]
        left_numeric = _numeric_series_or_none(left_value)
        right_numeric = _numeric_series_or_none(right_value)
        if left_numeric is not None and right_numeric is not None:
            difference = (left_numeric - right_numeric).abs()
            both_null = left_value.isna() & right_value.isna()
            null_mismatch = left_value.isna() ^ right_value.isna()
            precision = (difference.gt(0) & difference.le(float_tolerance) & ~both_null)
            business = (difference.gt(float_tolerance) & ~both_null) | null_mismatch
        else:
            if "date" in column.lower() or column.casefold() in {"week_start", "week_end"}:
                normalized_left = _normalize_key_series(left_value, column)
                normalized_right = _normalize_key_series(right_value, column)
            else:
                normalized_left = left_value.astype("string").fillna("<NA>")
                normalized_right = right_value.astype("string").fillna("<NA>")
            precision = pd.Series(False, index=matched.index)
            business = normalized_left != normalized_right
            difference = pd.Series(pd.NA, index=matched.index, dtype="Float64")

        for index in matched.index[business | precision]:
            row = matched.loc[index]
            rows.append({
                "difference_type": "precision_difference" if bool(precision.loc[index]) else "value_difference",
                "column": column,
                "key_values": key_values(row),
                "duplicate_ordinal": row["__duplicate_ordinal"],
                "left_value": row[f"{column}__left"],
                "right_value": row[f"{column}__right"],
                "abs_difference": difference.loc[index],
                "left_row": "",
                "right_row": "",
            })

    for column in sorted(set(left.columns) - set(right.columns) - {"__duplicate_ordinal"}):
        rows.append({"difference_type": "left_only_column", "column": column, "key_values": "{}",
                     "duplicate_ordinal": "", "left_value": "", "right_value": "",
                     "abs_difference": "", "left_row": "", "right_row": ""})
    for column in sorted(set(right.columns) - set(left.columns) - {"__duplicate_ordinal"}):
        rows.append({"difference_type": "right_only_column", "column": column, "key_values": "{}",
                     "duplicate_ordinal": "", "left_value": "", "right_value": "",
                     "abs_difference": "", "left_row": "", "right_row": ""})
    return pd.DataFrame(rows, columns=DIFFERENCE_DETAIL_COLUMNS)


def _has_only_numbers(series: pd.Series) -> bool:
    """判断混合 object 列的非空值是否均为数值。"""
    values = series.dropna()
    return not values.empty and values.map(
        lambda value: isinstance(value, numbers.Number) and not isinstance(value, bool)
    ).all()


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