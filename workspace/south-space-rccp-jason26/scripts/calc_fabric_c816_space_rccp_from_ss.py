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
ss['ss_cbm'] = ss['safety_stock_qty'] * ss['demand_unit_to_volume']

# two useful views:
# 1) monthly avg safety stock space (avg day in month)
# 2) monthly peak safety stock space (max day in month)
monthly = ss.groupby(['month','date'], as_index=False)['ss_cbm'].sum().rename(columns={'ss_cbm':'daily_ss_cbm'})
summary = monthly.groupby('month', as_index=False).agg(
    avg_ss_rccp_cbm=('daily_ss_cbm','mean'),
    peak_ss_rccp_cbm=('daily_ss_cbm','max')
)
peak_dates = monthly.sort_values(['month','daily_ss_cbm','date'], ascending=[True,False,True]).drop_duplicates('month')
summary = summary.merge(peak_dates[['month','date','daily_ss_cbm']].rename(columns={'date':'peak_date','daily_ss_cbm':'peak_ss_rccp_cbm_check'}), on='month', how='left')
summary = summary.drop(columns=['peak_ss_rccp_cbm_check'])
summary['avg_ss_rccp_cbm'] = summary['avg_ss_rccp_cbm'].round(2)
summary['peak_ss_rccp_cbm'] = summary['peak_ss_rccp_cbm'].round(2)

out_detail = RESULTS / 'fabric_c816_space_rccp_from_safety_stock_daily.csv'
out_summary = RESULTS / 'fabric_c816_space_rccp_from_safety_stock_by_month.csv'
monthly.to_csv(out_detail, index=False, encoding='utf-8-sig')
summary.to_csv(out_summary, index=False, encoding='utf-8-sig')
print(summary.to_string(index=False))
print(f'\ndetail={out_detail}\nsummary={out_summary}')
