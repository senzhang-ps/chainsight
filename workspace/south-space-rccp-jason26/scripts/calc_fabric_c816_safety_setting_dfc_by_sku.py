from pathlib import Path
import calendar
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
        select material, location, date::date as date, safety_stock_qty
        from cfg_m3_safetystock
        where config_name = %(config_name)s and location = %(location)s
        """,
        conn,
        params={'config_name': config_name, 'location': LOCATION}
    )
    fc = pd.read_sql_query(
        """
        select material, location, week, quantity
        from cfg_m1_demandforecast
        where config_name = %(config_name)s and location = %(location)s
        """,
        conn,
        params={'config_name': config_name, 'location': LOCATION}
    )

ss['material'] = ss['material'].astype(str).str.strip()
fc['material'] = fc['material'].astype(str).str.strip()
ss = ss[ss['material'].isin(fabric_materials)].copy()
fc = fc[fc['material'].isin(fabric_materials)].copy()

ss['date'] = pd.to_datetime(ss['date'])
ss['month'] = ss['date'].dt.strftime('%Y-%m')
ss['safety_stock_qty'] = pd.to_numeric(ss['safety_stock_qty'], errors='coerce').fillna(0)

# monthly avg safety setting by material
ss_month = ss.groupby(['month','material'], as_index=False)['safety_stock_qty'].mean().rename(columns={'safety_stock_qty':'avg_safety_stock_qty'})

# map week to month using simulation start 2026-06-29
sim_start = pd.Timestamp('2026-06-29')
fc['week'] = pd.to_numeric(fc['week'], errors='coerce')
fc['quantity'] = pd.to_numeric(fc['quantity'], errors='coerce').fillna(0)
fc['week_start'] = sim_start + pd.to_timedelta((fc['week'] - 1) * 7, unit='D')
fc['month'] = fc['week_start'].dt.strftime('%Y-%m')
fc['days_in_month'] = fc['month'].apply(lambda x: calendar.monthrange(int(x[:4]), int(x[5:7]))[1])
fc_month = fc.groupby(['month','material'], as_index=False)['quantity'].sum().rename(columns={'quantity':'monthly_forecast_qty'})
fc_month['days_in_month'] = fc_month['month'].apply(lambda x: calendar.monthrange(int(x[:4]), int(x[5:7]))[1])
fc_month['daily_avg_forecast_qty'] = fc_month['monthly_forecast_qty'] / fc_month['days_in_month']

out = ss_month.merge(fc_month[['month','material','monthly_forecast_qty','daily_avg_forecast_qty']], on=['month','material'], how='outer')
out['avg_safety_stock_qty'] = out['avg_safety_stock_qty'].fillna(0)
out['monthly_forecast_qty'] = out['monthly_forecast_qty'].fillna(0)
out['daily_avg_forecast_qty'] = out['daily_avg_forecast_qty'].fillna(0)
out['safety_setting_dfc_days'] = out['avg_safety_stock_qty'] / out['daily_avg_forecast_qty'].replace({0: pd.NA})
out['safety_setting_dfc_days'] = out['safety_setting_dfc_days'].round(2)
out = out.sort_values(['month','safety_setting_dfc_days','material'], ascending=[True,False,True]).reset_index(drop=True)

summary = out.groupby('month', as_index=False)['safety_setting_dfc_days'].mean().rename(columns={'safety_setting_dfc_days':'avg_sku_safety_setting_dfc_days'})
summary['sku_count'] = out.groupby('month')['material'].nunique().values

out_path = RESULTS / 'fabric_c816_safety_setting_dfc_by_sku_by_month.csv'
summary_path = RESULTS / 'fabric_c816_safety_setting_dfc_by_month_summary.csv'
out.to_csv(out_path, index=False, encoding='utf-8-sig')
summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
print('SUMMARY')
print(summary.to_string(index=False))
print('\nDETAIL')
print(out.head(60).to_string(index=False))
print(f'\nout={out_path}\nsummary={summary_path}')
