from pathlib import Path
import pandas as pd
import psycopg

ROOT = Path(r"\\wsl.localhost\Ubuntu-22.04\home\zhangs37\ai_chainsight")
CREDS_FILE = ROOT / 'db_credentials.yaml'
RESULTS = ROOT / 'workspace' / 'south-space-rccp-jason26' / 'scenarios' / 'production-cycle' / 'results'
RUN_ID = 'db_PDS2_20260527_181240'
LOCATION = 'C816'
CATEGORY = 'Fabric'


def load_creds() -> dict:
    creds = {}
    for raw in CREDS_FILE.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        k, v = line.split(':', 1)
        creds[k.strip()] = v.strip()
    creds['database'] = 'south_rccp'
    return creds

mapping = pd.read_csv(RESULTS / 'material_category_en_databricks.csv', dtype=str)
mapping['material'] = mapping['material'].astype(str).str.strip()
fabric_materials = set(mapping.loc[mapping['category_en'] == CATEGORY, 'material'])

creds = load_creds()
with psycopg.connect(
    host=creds['host'], port=int(creds['port']), dbname=creds['database'],
    user=creds['user'], password=creds['password'], connect_timeout=10,
) as conn:
    config_name = pd.read_sql_query(
        "select min(config_name) as config_name from orchestrator_unrestricted_inventory where run_id = %(run_id)s",
        conn,
        params={'run_id': RUN_ID}
    ).iloc[0,0]

    ss = pd.read_sql_query(
        """
        select s.material, s.location, s.date::date as date, s.safety_stock_qty,
               m.demand_unit_to_volume
        from cfg_m3_safetystock s
        left join cfg_m6_materialmd m
          on m.config_name = s.config_name and m.material = s.material
        where s.config_name = %(config_name)s and s.location = %(location)s
        """,
        conn,
        params={'config_name': config_name, 'location': LOCATION}
    )

ss['material'] = ss['material'].astype(str).str.strip()
ss = ss[ss['material'].isin(fabric_materials)].copy()
ss['date'] = pd.to_datetime(ss['date'])
ss['month'] = ss['date'].dt.strftime('%Y-%m')
ss['safety_stock_qty'] = pd.to_numeric(ss['safety_stock_qty'], errors='coerce').fillna(0)
ss['demand_unit_to_volume'] = pd.to_numeric(ss['demand_unit_to_volume'], errors='coerce').fillna(0)
ss['safety_stock_cbm'] = ss['safety_stock_qty'] * ss['demand_unit_to_volume']

# month peak at total Fabric C816 daily CBM level
month_peak = ss.groupby(['month','date'], as_index=False)['safety_stock_cbm'].sum().rename(columns={'safety_stock_cbm':'daily_total_cbm'})
month_peak = month_peak.sort_values(['month','daily_total_cbm','date'], ascending=[True,False,True]).drop_duplicates('month')
peak_dates = set((row.month, row.date.strftime('%Y-%m-%d')) for row in month_peak.itertuples(index=False))

ss['date_str'] = ss['date'].dt.strftime('%Y-%m-%d')
ss['month_peak_flag'] = ss.apply(lambda r: 'Y' if (r['month'], r['date_str']) in peak_dates else 'N', axis=1)

out = ss[['month','date_str','material','safety_stock_qty','demand_unit_to_volume','safety_stock_cbm','month_peak_flag']].rename(columns={'date_str':'date'})
out = out.sort_values(['month','date','material']).reset_index(drop=True)
out_path = RESULTS / 'fabric_c816_safety_stock_qty_cbm_peakflag_by_material_date.csv'
out.to_csv(out_path, index=False, encoding='utf-8-sig')
print(out.head(80).to_string(index=False))
print(f'\nout={out_path}\nrows={len(out)}')
