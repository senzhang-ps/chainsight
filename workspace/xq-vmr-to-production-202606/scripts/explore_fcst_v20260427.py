"""Explore ps_psc_bop_lbe_fcst for version 20260427 (LBE) before extracting.

Checks:
  1. Available fcst_type values for frcst_vers_date=20260427.
  2. Week (wk_start_date) coverage of the v20260427 LBE forecast (does it reach 2026-12-31?).
  3. material_num format and overlap with the existing baseline config scope.
"""
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from databricks import sql as dbsql

load_dotenv(Path("config/.env"))

FCST = "cdl_ps_hana_prd.sl.ps_psc_bop_lbe_fcst"
CAL = "cdl_ps_hana_prd.ods.psdh_md_time_fdim"
VERS = 20260427

cfg_path = Path(
    "input/xq-vmr-to-production-202606/scenarios/baseline/config/M1_DemandForecast.csv"
)
base = pd.read_csv(cfg_path)
materials = sorted({int(m) for m in base["material"].unique()})
locations = sorted({str(l) for l in base["location"].unique()})
print(f"baseline scope: {len(materials)} materials, {len(locations)} locations")


def q(cur, sql):
    cur.execute(sql)
    cols = [c[0] for c in cur.description]
    return pd.DataFrame(cur.fetchall(), columns=cols)


with dbsql.connect(
    server_hostname=os.environ["DATABRICKS_HOST"],
    http_path=os.environ["DATABRICKS_HTTP_PATH"],
    access_token=os.environ["DATABRICKS_TOKEN"],
) as conn:
    with conn.cursor() as cur:
        print("\n[1] fcst_type counts for version", VERS)
        print(q(cur, f"""
            select fcst_type, count(*) n, count(distinct material_num) mats,
                   count(distinct site_id) sites
            from {FCST}
            where frcst_vers_date = {VERS}
            group by fcst_type order by fcst_type
        """).to_string(index=False))

        print("\n[2] tp_start_date min/max for v", VERS, "LBE")
        print(q(cur, f"""
            select min(tp_start_date) min_tp, max(tp_start_date) max_tp,
                   count(distinct tp_start_date) n_tp
            from {FCST}
            where frcst_vers_date = {VERS} and fcst_type = 'LBE'
        """).to_string(index=False))

        print("\n[3] wk_start_date coverage (joined to calendar) for v", VERS, "LBE")
        print(q(cur, f"""
            select min(c.wk_start_date) min_wk, max(c.wk_start_date) max_wk,
                   count(distinct c.wk_start_date) n_weeks
            from {FCST} f
            join {CAL} c on c.day_date = f.tp_start_date
            where f.frcst_vers_date = {VERS} and f.fcst_type = 'LBE'
        """).to_string(index=False))

        print("\n[4] weeks in target window 2026-06-29..2026-12-31 (scope-filtered)")
        mat_list = ",".join(str(m) for m in materials)
        loc_list = ",".join(f"'{l}'" for l in locations)
        print(q(cur, f"""
            select c.wk_start_date,
                   count(distinct f.material_num) mats,
                   count(distinct f.site_id) sites,
                   round(sum(f.fcst_qty_in_su),1) su
            from {FCST} f
            join {CAL} c on c.day_date = f.tp_start_date
            where f.frcst_vers_date = {VERS} and f.fcst_type = 'LBE'
              and try_cast(f.material_num as bigint) in ({mat_list})
              and f.site_id in ({loc_list})
              and f.fcst_qty_in_su > 0
              and c.wk_start_date between date'2026-06-29' and date'2026-12-31'
            group by c.wk_start_date order by c.wk_start_date
        """).to_string(index=False))
