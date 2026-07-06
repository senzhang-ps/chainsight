"""Extract v20260427 LBE demand forecast for the xq-vmr-to-production baseline scope,
extended to 2026-12-31 (ChainSight weeks 1-27, week 1 = 2026-06-29).

Scope is preserved exactly: each target config file's own (material, location) pairs
are reused. Output schema: week, material, location, quantity (fcst_qty_in_cs, Case Count).

Writes <dir>/M1_DemandForecast_v20260427.csv next to each existing config for review;
the swap to the canonical name is done separately after validation.
"""
import os
from datetime import date
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from databricks import sql as dbsql

load_dotenv(Path("config/.env"))

FCST = "cdl_ps_hana_prd.sl.ps_psc_bop_lbe_fcst"
CAL = "cdl_ps_hana_prd.ods.psdh_md_time_fdim"
VERS = 20260427
FCST_TYPE = "LBE"
WIN_START = date(2026, 6, 29)   # ChainSight week 1
WIN_END = date(2026, 12, 31)

TARGETS = [
    Path("input/xq-vmr-to-production-202606/scenarios/baseline/config/M1_DemandForecast.csv"),
    Path("workspace/xq-vmr-to-production-202606/scenarios/baseline/config/M1_DemandForecast.csv"),
]

# Scope source: the ORIGINAL (pre-v20260427, 19-week) config backup, which holds the
# true baseline (material, location) intent. The current canonical files hold the
# wrong SU-based pull and are overwritten below.
SCOPE_SRC = Path(
    "input/xq-vmr-to-production-202606/scenarios/baseline/config/M1_DemandForecast_pre20260427_w19.csv"
)

# --- build query scope from the original baseline pairs ---
scope_df = pd.read_csv(SCOPE_SRC)
scope_pairs = set(zip(scope_df["material"].astype("int64"), scope_df["location"].astype(str)))
# Every target file uses the same original baseline scope.
pair_sets = {p: scope_pairs for p in TARGETS}
materials = sorted({m for m, _ in scope_pairs})
locations = sorted({l for _, l in scope_pairs})
print(f"scope from {SCOPE_SRC.name}: {len(materials)} materials, {len(locations)} locations, {len(scope_pairs)} pairs")

mat_list = ",".join(str(m) for m in materials)
loc_list = ",".join(f"'{l}'" for l in locations)

sql = f"""
    select c.wk_start_date as wk_start_date,
           try_cast(f.material_num as bigint) as material,
           f.site_id as location,
           sum(f.fcst_qty_in_cs) as quantity
    from {FCST} f
    join {CAL} c on c.day_date = f.tp_start_date
    where f.frcst_vers_date = {VERS}
      and f.fcst_type = '{FCST_TYPE}'
      and try_cast(f.material_num as bigint) in ({mat_list})
      and f.site_id in ({loc_list})
      and f.fcst_qty_in_cs > 0
      and c.wk_start_date between date'{WIN_START}' and date'{WIN_END}'
    group by c.wk_start_date, try_cast(f.material_num as bigint), f.site_id
"""

with dbsql.connect(
    server_hostname=os.environ["DATABRICKS_HOST"],
    http_path=os.environ["DATABRICKS_HTTP_PATH"],
    access_token=os.environ["DATABRICKS_TOKEN"],
) as conn:
    with conn.cursor() as cur:
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        raw = pd.DataFrame(cur.fetchall(), columns=cols)

print(f"\nfetched {len(raw)} rows from Databricks")

# --- renumber week: week 1 = 2026-06-29 (Monday) ---
raw["wk_start_date"] = pd.to_datetime(raw["wk_start_date"])
raw["week"] = ((raw["wk_start_date"] - pd.Timestamp(WIN_START)).dt.days // 7) + 1
raw["material"] = raw["material"].astype("int64")
raw["location"] = raw["location"].astype(str)
raw["quantity"] = raw["quantity"].astype(float).round(4)

print("week range:", int(raw["week"].min()), "->", int(raw["week"].max()),
      "| distinct weeks:", raw["week"].nunique())

# --- write one output per target file, preserving that file's exact pairs ---
for p, pairs in pair_sets.items():
    sub = raw[raw[["material", "location"]].apply(tuple, axis=1).isin(pairs)]
    out = (sub[["week", "material", "location", "quantity"]]
           .sort_values(["week", "material", "location"])
           .reset_index(drop=True))
    dest = p.with_name("M1_DemandForecast_v20260427.csv")
    out.to_csv(dest, index=False)
    print(f"\n{p.parent.parts[0]}/.../{p.name}")
    print(f"  -> {dest.name}: rows={len(out)} weeks={out['week'].min()}..{out['week'].max()} "
          f"mats={out['material'].nunique()} locs={out['location'].nunique()} "
          f"pairs={out[['material','location']].drop_duplicates().shape[0]} (orig pairs={len(pairs)})")
