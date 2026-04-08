"""Shared normalization helpers used by multiple runtime packages.

The goal of this module is code consolidation without behavior drift. Module-
level wrappers can pick the helper variant that matches their historical
semantics instead of changing callers directly.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

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
