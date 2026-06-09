"""Date parsing helpers for runtime configuration data."""
from __future__ import annotations

from typing import Any

import pandas as pd
from pandas.api.types import is_list_like


def parse_mixed_datetime(values: Any, context: str):
    """Parse date/date-time values while failing loudly on invalid data.

    Accepts common mixed ISO-like values such as ``YYYY-MM-DD`` and
    ``YYYY-MM-DD HH:MM:SS``. Invalid non-null values raise a ValueError with
    examples so configuration issues can be fixed at the source.
    """
    try:
        return pd.to_datetime(values, format="mixed", errors="raise")
    except Exception as exc:
        raw = _as_series(values)
        coerced = pd.to_datetime(raw, format="mixed", errors="coerce")
        bad_mask = raw.notna() & coerced.isna()
        bad_samples = (
            raw[bad_mask]
            .astype(str)
            .drop_duplicates()
            .head(5)
            .tolist()
        )
        sample_text = ", ".join(repr(sample) for sample in bad_samples)
        if not sample_text:
            sample_text = "no invalid non-null samples identified"
        raise ValueError(
            f"{context}: failed to parse datetime values; "
            f"bad samples: {sample_text}"
        ) from exc


def _as_series(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return values
    if isinstance(values, pd.Index):
        return values.to_series(index=range(len(values)))
    if is_list_like(values) and not isinstance(values, (str, bytes)):
        return pd.Series(values)
    return pd.Series([values])
