"""Export material -> category mapping from Databricks ps_psc_sku_master.

Reads credentials from config/.env (DATABRICKS_HOST / DATABRICKS_HTTP_PATH /
DATABRICKS_TOKEN). Pulls category_en / category / bu_attr for the given list of
materials and returns a DataFrame. Used by build_baseline_analysis.py.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

TABLE_FQN = "cdl_ps_hana_prd.sl.ps_psc_sku_master"
QUERY_COLUMNS = ["material_num", "category_en", "category", "bu_attr"]
CHUNK_SIZE = 500
ENV_FILE = Path("config/.env")


def _chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def fetch_category_mapping(materials: list[str]) -> pd.DataFrame:
    """Query Databricks for the given materials. Raises on connection failure."""
    from dotenv import load_dotenv
    from databricks import sql as dbsql

    load_dotenv(ENV_FILE)
    host = os.environ["DATABRICKS_HOST"]
    http_path = os.environ["DATABRICKS_HTTP_PATH"]
    token = os.environ["DATABRICKS_TOKEN"]

    mats = sorted({str(m).strip() for m in materials if str(m).strip()})
    frames: list[pd.DataFrame] = []
    with dbsql.connect(server_hostname=host, http_path=http_path, access_token=token) as conn:
        for batch in _chunked(mats, CHUNK_SIZE):
            quoted = ", ".join(f"'{m}'" for m in batch)
            query = (
                f"select {', '.join(QUERY_COLUMNS)} from {TABLE_FQN} "
                f"where material_num in ({quoted})"
            )
            with conn.cursor() as cur:
                cur.execute(query)
                rows = cur.fetchall()
                frames.append(pd.DataFrame(rows, columns=QUERY_COLUMNS))

    if not frames:
        out = pd.DataFrame(columns=QUERY_COLUMNS)
    else:
        out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["material_num"])
    out = out.rename(columns={"material_num": "material"})
    out["material"] = out["material"].astype(str)
    return out


def fetch_su_factor(materials: list[str]) -> pd.DataFrame:
    """Query Databricks for su_factor_for_buom per material. Raises on failure.

    Returns df[material, su_factor] (su_factor numeric, BUOM unit factor). Used as
    the fallback source for MSU when the local SUF workbook lacks a material.
    """
    from dotenv import load_dotenv
    from databricks import sql as dbsql

    load_dotenv(ENV_FILE)
    host = os.environ["DATABRICKS_HOST"]
    http_path = os.environ["DATABRICKS_HTTP_PATH"]
    token = os.environ["DATABRICKS_TOKEN"]

    mats = sorted({str(m).strip() for m in materials if str(m).strip()})
    frames: list[pd.DataFrame] = []
    with dbsql.connect(server_hostname=host, http_path=http_path, access_token=token) as conn:
        for batch in _chunked(mats, CHUNK_SIZE):
            quoted = ", ".join(f"'{m}'" for m in batch)
            query = (
                f"select material_num, su_factor_for_buom from {TABLE_FQN} "
                f"where material_num in ({quoted})"
            )
            with conn.cursor() as cur:
                cur.execute(query)
                frames.append(pd.DataFrame(cur.fetchall(), columns=["material", "su_factor"]))

    if not frames:
        out = pd.DataFrame(columns=["material", "su_factor"])
    else:
        out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["material"])
    out["material"] = out["material"].astype(str)
    out["su_factor"] = pd.to_numeric(out["su_factor"], errors="coerce")
    return out


if __name__ == "__main__":
    # Smoke test with a tiny query.
    df = fetch_category_mapping(["21156898", "21162178"])
    print(df.to_string(index=False))
