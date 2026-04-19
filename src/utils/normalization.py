# -*- coding: utf-8 -*-
"""
normalization.py - 标识符规范化统一实现

本模块是 material / location / sending / receiving 等标识符列规范化处理的
**唯一真源 (single source of truth)**。所有模块均从此模块导入,不应自行定义。

对外 API:
    标量层:
        normalize_material(value, *, mode, treat_missing_tokens)
        normalize_location(value, *, mode)
        normalize_sending / normalize_receiving / normalize_sourcing
            —— location 的业务语义别名(等价于 normalize_location 默认模式)

    DataFrame 层:
        normalize_identifiers(df, *, material_cols=None, location_cols=None,
                               other_identifier_cols=None, extra_columns=None)
            —— 不传显式列集时使用模块默认列集 + 可选 extra_columns;
               传入任一显式列集参数则按自定义列集规范化
        cast_identifier_columns(df, *, cols, normalized_location_cols)
            —— 仅做类型转换 + 可选 location 规范化

规范化规则:
    material  mode="numeric" (默认): 数值型 → int(float(x)) 去 .0;其他 str().strip()
    material  mode="basic":           仅 None/NaN → "";其他直接 str()
    location  mode="numeric_only" (默认): 纯数字 zfill(4);非数字原样
    location  mode="any":                   无条件 zfill(4),非数字文本也补零
"""

from typing import Any, Iterable, List, Literal, Optional, Sequence

import pandas as pd


# ---------------------------------------------------------------------------
# 标识符列名常量
# ---------------------------------------------------------------------------

#: 地点类列:纯数字补零至 4 位
LOCATION_COLUMNS: List[str] = [
    'location', 'dps_location', 'sending', 'receiving', 'sourcing',
]

#: 物料类列:去除 .0 后缀
MATERIAL_COLUMNS: List[str] = [
    'material', 'from_material', 'to_material',
]

#: 仅需 str 转换的标识符列
STRING_ONLY_COLUMNS: List[str] = [
    'line', 'delegate_line', 'changeover_id',
]

#: 全部标识符列 (superset)
ALL_IDENTIFIER_COLUMNS: List[str] = (
    MATERIAL_COLUMNS + LOCATION_COLUMNS + STRING_ONLY_COLUMNS
)

#: 字符串化后等同于缺失值的 token 集合
_MISSING_STRING_TOKENS = ["nan", "None", "<NA>", "NaN"]


# ---------------------------------------------------------------------------
# 标量函数 —— material
# ---------------------------------------------------------------------------

MaterialMode = Literal["numeric", "basic"]


def normalize_material(
    value: Any,
    *,
    mode: MaterialMode = "numeric",
    treat_missing_tokens: bool = False,
) -> str:
    """规范化物料标识符。

    Args:
        value: 原始物料标识 (int / float / str / None)
        mode:
            "numeric" (默认) —— 数值型 → 去 .0 后缀;其他 str().strip()
            "basic"          —— 仅 None/NaN → "";其他直接 str(),不 strip、不去 .0
        treat_missing_tokens: 仅在 mode="numeric" 下生效;是否把
            ``'nan' / 'none' / '<na>'`` 字符串也视为缺失(返回 "")

    Returns:
        规范化后的字符串。None / NaN → ""
    """
    if value is None or pd.isna(value):
        return ""

    if mode == "basic":
        return str(value)

    if treat_missing_tokens and (
        value == "" or str(value).lower() in ["nan", "none", "<na>"]
    ):
        return ""

    try:
        if (
            isinstance(value, (int, float))
            or str(value).replace(".", "").replace("-", "").isdigit()
        ):
            return str(int(float(value)))
        return str(value).strip()
    except (ValueError, TypeError):
        return str(value).strip()


# ---------------------------------------------------------------------------
# 标量函数 —— location
# ---------------------------------------------------------------------------

LocationMode = Literal["numeric_only", "any"]


def normalize_location(
    value: Any,
    *,
    mode: LocationMode = "numeric_only",
) -> str:
    """规范化地点标识符。

    Args:
        value: 原始地点标识
        mode:
            "numeric_only" (默认) —— 纯数字 zfill(4);非数字(如 "A888")原样保留
            "any"                 —— 无条件 zfill(4),非数字文本也补零

    Returns:
        规范化后的字符串。None / NaN → ""
    """
    if value is None or pd.isna(value):
        return ""

    if mode == "any":
        try:
            return str(int(value)).zfill(4)
        except (ValueError, TypeError):
            return str(value).zfill(4)

    text = str(value).strip()
    try:
        if text.isdigit():
            return str(int(text)).zfill(4)
        return text
    except (ValueError, TypeError):
        return str(text)


# sending / receiving / sourcing 复用 location 规则(业务语义别名)
normalize_sending = normalize_location
normalize_receiving = normalize_location
normalize_sourcing = normalize_location


# ---------------------------------------------------------------------------
# DataFrame 向量化函数
# ---------------------------------------------------------------------------

def normalize_identifiers(
    df: pd.DataFrame,
    *,
    material_cols: Optional[Iterable[str]] = None,
    location_cols: Optional[Iterable[str]] = None,
    other_identifier_cols: Optional[Iterable[str]] = None,
    extra_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """DataFrame 级标识符规范化 (单一入口)。

    两种调用形态:

    1. **默认列集 + 可选 extra_columns** (18 处旧调用的形态):
       ``normalize_identifiers(df)`` 或 ``normalize_identifiers(df, extra_columns=[...])``
       - material / from_material / to_material: 去 .0 后缀
       - location / dps_location / sending / receiving / sourcing: 纯数字补零 4 位
       - line / delegate_line / changeover_id: 仅 str 转换
       - ``extra_columns`` 内不重复的列并入 "仅 str 转换" 集合

    2. **完全显式列集** (M3 等需要自定义列集的场景):
       ``normalize_identifiers(df, material_cols=(...), location_cols=(...), other_identifier_cols=(...))``
       传入任一显式列集参数即视为进入自定义模式:未显式指定的列集按空集处理,
       ``extra_columns`` 在此模式下被忽略。

    Returns:
        标识符已规范化的 DataFrame 副本;空表原样返回。
    """
    if df.empty:
        return df

    explicit_mode = (
        material_cols is not None
        or location_cols is not None
        or other_identifier_cols is not None
    )

    if explicit_mode:
        mcols: Iterable[str] = material_cols if material_cols is not None else ()
        lcols: Iterable[str] = location_cols if location_cols is not None else ()
        ocols: Iterable[str] = (
            other_identifier_cols if other_identifier_cols is not None else ()
        )
    else:
        mcols = MATERIAL_COLUMNS
        lcols = LOCATION_COLUMNS
        other_default: List[str] = list(STRING_ONLY_COLUMNS)
        if extra_columns:
            other_default.extend(
                c for c in extra_columns if c not in ALL_IDENTIFIER_COLUMNS
            )
        ocols = other_default

    df = df.copy()

    for col in mcols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].replace(_MISSING_STRING_TOKENS, "")
            df[col] = df[col].str.replace(r"\.0$", "", regex=True)

    for col in lcols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(_MISSING_STRING_TOKENS, "")
            is_numeric = df[col].str.match(r"^\d+$", na=False)
            df.loc[is_numeric, col] = df.loc[is_numeric, col].str.zfill(4)

    for col in ocols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    return df


def cast_identifier_columns(
    df: pd.DataFrame,
    *,
    cols: Iterable[str],
    normalized_location_cols: Iterable[str] = (),
) -> pd.DataFrame:
    """将标识符列转为 string,并对指定 location 列做 numeric_only 规范化。"""
    if df is None or df.empty:
        return df

    normalized_location_cols = set(normalized_location_cols)
    df = df.copy()

    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype("string")
            if col in normalized_location_cols:
                df[col] = df[col].apply(normalize_location)

    return df
