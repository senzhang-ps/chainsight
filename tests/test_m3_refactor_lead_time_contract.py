"""M3 refactor 的内部提前期列 canonicalization 契约。"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.modules.mrp_planning.backends import _PandasBackend, _PolarsBackend
from src.modules.mrp_planning.lead_time import compute_root_horizon


_SCHEMA = {
    "Global_LeadTime": {},
    "M4_MaterialLocationLineCfg": {},
}


def _owner() -> SimpleNamespace:
    return SimpleNamespace(schema=_SCHEMA)


def _config() -> dict[str, pd.DataFrame]:
    return {
        "Global_LeadTime": pd.DataFrame([{
            "sending": "0386", "receiving": "A672",
            "pdt": 10, "gr": 2, "mct": 16,
        }]),
        "M4_MaterialLocationLineCfg": pd.DataFrame([{
            "material": "MAT-1", "location": "0386", "ptf": 2, "lsk": 2,
        }]),
    }


def test_pandas_m3_canonicalises_lowercase_lead_time_internally() -> None:
    backend = _PandasBackend(_owner())
    static = backend.normalise_static_config(_config())

    assert {"PDT", "GR", "MCT"}.issubset(static["Global_LeadTime"].columns)
    assert compute_root_horizon(
        "MAT-1", "0386", static["Global_LeadTime"], static["M4_MaterialLocationLineCfg"],
    ) == 19


def test_polars_m3_canonicalises_lowercase_lead_time_internally() -> None:
    backend = _PolarsBackend(_owner())
    static = backend.normalise_static_config(_config())

    assert {"PDT", "GR", "MCT"}.issubset(static["Global_LeadTime"].columns)