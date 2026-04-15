# -*- coding: utf-8 -*-
"""
normalization.py - 标识符规范化统一实现

本模块是 material / location / sending / receiving 等标识符列
规范化处理的 **唯一真源 (single source of truth)**。

所有模块（demand_planning、deployment_planning、orchestrator、main_integration）
均应从此模块导入，而非自行定义。

规范化规则：
    material   : 数值型 → int(float(x)) 去除 .0 后缀；其他 → str(x).strip()
    location   : 纯数字 → zfill(4)；非数字(如 A888) → 原样保留
    sending    : 同 location 规则
    receiving  : 同 location 规则
    sourcing   : 同 location 规则
    dps_location : 同 location 规则
"""

from typing import Any, Iterable, List, Optional, Sequence

import pandas as pd


# ---------------------------------------------------------------------------
# 标识符列名常量
# ---------------------------------------------------------------------------

#: 地点类列：纯数字补零至 4 位
LOCATION_COLUMNS: List[str] = [
    'location', 'dps_location', 'sending', 'receiving', 'sourcing',
]

#: 物料类列：去除 .0 后缀
MATERIAL_COLUMNS: List[str] = [
    'material', 'from_material', 'to_material',
]

#: 仅需 str 转换的标识符列
STRING_ONLY_COLUMNS: List[str] = [
    'line', 'delegate_line', 'changeover_id',
]

#: 全部标识符列（superset）
ALL_IDENTIFIER_COLUMNS: List[str] = (
    MATERIAL_COLUMNS + LOCATION_COLUMNS + STRING_ONLY_COLUMNS
)


# ---------------------------------------------------------------------------
# 标量函数
# ---------------------------------------------------------------------------

def normalize_material(material_str: Any) -> str:
    """规范化物料编码：数值型去除 .0 后缀。

    Args:
        material_str: 原始物料标识（int / float / str / None）

    Returns:
        规范化后的字符串。None / NaN → ""
    """
    if material_str is None or pd.isna(material_str):
        return ""

    try:
        if (
            isinstance(material_str, (int, float))
            or str(material_str).replace('.', '').replace('-', '').isdigit()
        ):
            return str(int(float(material_str)))
        else:
            return str(material_str).strip()
    except (ValueError, TypeError):
        return str(material_str).strip()


def normalize_location(location_str: Any) -> str:
    """规范化地点编码：纯数字补零至 4 位，非数字原样保留。

    Args:
        location_str: 原始地点标识

    Returns:
        规范化后的字符串。None / NaN → ""
    """
    if location_str is None or pd.isna(location_str):
        return ""

    location_str = str(location_str).strip()

    try:
        if location_str.isdigit():
            return str(int(location_str)).zfill(4)
        else:
            return location_str
    except (ValueError, TypeError):
        return str(location_str)


# sending / receiving / sourcing 复用 location 规则
normalize_sending = normalize_location
normalize_receiving = normalize_location
normalize_sourcing = normalize_location


# ---------------------------------------------------------------------------
# DataFrame 向量化函数
# ---------------------------------------------------------------------------

def normalize_identifiers(
    df: pd.DataFrame,
    extra_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """将 DataFrame 中的标识符列规范化为统一字符串格式。

    使用向量化操作提升性能。处理规则：
    - material / from_material / to_material: 去除 .0 后缀
    - location / dps_location / sending / receiving / sourcing: 纯数字补零 4 位
    - line / delegate_line / changeover_id: 仅 str 转换

    Args:
        df: 待规范化的 DataFrame
        extra_columns: 额外需要 str 转换的列名（可选）

    Returns:
        标识符已规范化的 DataFrame 副本；空表原样返回
    """
    if df.empty:
        return df

    df = df.copy()

    # --- material 类列：去除 .0 后缀 ---
    for col in MATERIAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)
            df[col] = df[col].replace(['nan', 'None', '<NA>', 'NaN'], '')
            df[col] = df[col].str.replace(r'\.0$', '', regex=True)

    # --- location 类列：纯数字补零 4 位 ---
    for col in LOCATION_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str).str.strip()
            df[col] = df[col].replace(['nan', 'None', '<NA>', 'NaN'], '')
            is_numeric = df[col].str.match(r'^\d+$', na=False)
            df.loc[is_numeric, col] = df.loc[is_numeric, col].str.zfill(4)

    # --- 仅需 str 转换的列 ---
    for col in STRING_ONLY_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)

    # --- 额外列 ---
    if extra_columns:
        for col in extra_columns:
            if col in df.columns and col not in ALL_IDENTIFIER_COLUMNS:
                df[col] = df[col].fillna('').astype(str)

    return df


# ---------------------------------------------------------------------------
# normalization_common 合并函数
# ---------------------------------------------------------------------------

_MISSING_STRING_TOKENS = ["nan", "None", "<NA>", "NaN"]


def normalize_location_zero_fill_any(location_str) -> str:
    """Normalize location-like identifiers by zero-filling any non-nulls."""
    if location_str is None or pd.isna(location_str):
        return ""
    try:
        return str(int(location_str)).zfill(4)
    except (ValueError, TypeError):
        return str(location_str).zfill(4)


def normalize_location_preserve_non_numeric(location_str) -> str:
    """Normalize location-like identifiers while preserving non-numeric text."""
    if location_str is None or pd.isna(location_str):
        return ""

    location_str = str(location_str).strip()
    try:
        if location_str.isdigit():
            return str(int(location_str)).zfill(4)
        return location_str
    except (ValueError, TypeError):
        return str(location_str)


def normalize_material_basic(material_str) -> str:
    """Normalize material identifiers to string without extra cleanup."""
    if material_str is None or pd.isna(material_str):
        return ""
    return str(material_str)


def normalize_material_numeric_cleanup(
    material_str,
    *,
    strip_text: bool,
    treat_string_missing_tokens: bool,
) -> str:
    """Normalize material identifiers with caller-selected text semantics."""
    if material_str is None or pd.isna(material_str):
        return ""
    if treat_string_missing_tokens and (
        material_str == "" or str(material_str).lower() in ["nan", "none", "<na>"]
    ):
        return ""

    try:
        if (
            isinstance(material_str, (int, float))
            or str(material_str).replace(".", "").replace("-", "").isdigit()
        ):
            return str(int(float(material_str)))
        return str(material_str).strip() if strip_text else str(material_str)
    except (ValueError, TypeError):
        return str(material_str).strip() if strip_text else str(material_str)


def normalize_material_numeric_token_cleanup(material_str) -> str:
    """Normalize material identifiers with main-integration semantics."""
    return normalize_material_numeric_cleanup(
        material_str,
        strip_text=True,
        treat_string_missing_tokens=True,
    )


def normalize_material_numeric_preserve_text(material_str) -> str:
    """Normalize material identifiers with orchestrator semantics."""
    return normalize_material_numeric_cleanup(
        material_str,
        strip_text=False,
        treat_string_missing_tokens=False,
    )


def normalize_identifiers_vectorized(
    df: pd.DataFrame,
    *,
    material_cols: Iterable[str] = ("material",),
    location_cols: Iterable[str] = (),
    other_identifier_cols: Iterable[str] = (),
) -> pd.DataFrame:
    """Vectorized identifier normalization used by M1/M3/M5 and Orchestrator."""
    if df.empty:
        return df

    df = df.copy()

    for col in material_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].replace(_MISSING_STRING_TOKENS, "")
            df[col] = df[col].str.replace(r"\.0$", "", regex=True)

    for col in location_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace(_MISSING_STRING_TOKENS, "")
            is_numeric = df[col].str.match(r"^\d+$", na=False)
            df.loc[is_numeric, col] = df.loc[is_numeric, col].str.zfill(4)

    for col in other_identifier_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    return df


def normalize_identifiers_scalar(
    df: pd.DataFrame,
    *,
    identifier_cols: Iterable[str],
    location_cols: Iterable[str],
    material_cols: Iterable[str],
    passthrough_str_cols: Iterable[str] = (),
) -> pd.DataFrame:
    """Scalar normalization path used by main integration to preserve semantics."""
    if df.empty:
        return df

    df = df.copy()
    location_cols = set(location_cols)
    material_cols = set(material_cols)
    passthrough_str_cols = set(passthrough_str_cols)

    for col in identifier_cols:
        if col not in df.columns:
            continue

        df[col] = df[col].astype(str)

        if col in location_cols:
            df[col] = df[col].apply(normalize_location_preserve_non_numeric)
        elif col in material_cols:
            df[col] = df[col].apply(normalize_material_numeric_token_cleanup)
        elif col in passthrough_str_cols:
            pass
        else:
            df[col] = df[col].apply(lambda x: str(x) if pd.notna(x) else "")

    return df


def cast_identifier_columns(
    df: pd.DataFrame,
    *,
    cols: Iterable[str],
    normalized_location_cols: Iterable[str] = (),
) -> pd.DataFrame:
    """Cast identifier columns to strings while normalizing selected locations."""
    if df is None or df.empty:
        return df

    normalized_location_cols = set(normalized_location_cols)
    df = df.copy()

    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype("string")
            if col in normalized_location_cols:
                df[col] = df[col].apply(normalize_location_preserve_non_numeric)

    return df

    df = df.copy()

    # --- material 类列：去除 .0 后缀 ---
    for col in MATERIAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)
            df[col] = df[col].replace(['nan', 'None', '<NA>', 'NaN'], '')
            df[col] = df[col].str.replace(r'\.0$', '', regex=True)

    # --- location 类列：纯数字补零 4 位 ---
    for col in LOCATION_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str).str.strip()
            df[col] = df[col].replace(['nan', 'None', '<NA>', 'NaN'], '')
            is_numeric = df[col].str.match(r'^\d+$', na=False)
            df.loc[is_numeric, col] = df.loc[is_numeric, col].str.zfill(4)

    # --- 仅需 str 转换的列 ---
    for col in STRING_ONLY_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)

    # --- 额外列 ---
    if extra_columns:
        for col in extra_columns:
            if col in df.columns and col not in ALL_IDENTIFIER_COLUMNS:
                df[col] = df[col].fillna('').astype(str)

    return df
