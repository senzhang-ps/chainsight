from pathlib import Path
import pandas as pd
import psycopg

ROOT = Path(r"\\wsl.localhost\Ubuntu-22.04\home\zhangs37\ai_chainsight")
CREDS_FILE = ROOT / 'db_credentials.yaml'
RESULTS = ROOT / 'workspace' / 'south-space-rccp-jason26' / 'scenarios' / 'production-cycle' / 'results'
RUN_ID = 'db_PDS2_20260527_181240'
CATEGORY = 'Baby'


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
baby_materials = set(mapping.loc[mapping['category_en'] == CATEGORY, 'material'])

creds = load_creds()
with psycopg.connect(
    host=creds['host'], port=int(creds['port']), dbname=creds['database'],
    user=creds['user'], password=creds['password'], connect_timeout=10,
) as conn:
    cfg_name = pd.read_sql_query(
        "select min(config_name) as config_name from orchestrator_unrestricted_inventory where run_id = %(run_id)s",
        conn,
        params={'run_id': RUN_ID}
    ).iloc[0,0]

    prod = pd.read_sql_query(
        """
        select simulation_date::date as date, location, line, material, produced_qty
        from module4_output_productionplan
        where run_id = %(run_id)s
        """,
        conn,
        params={'run_id': RUN_ID}
    )
    chg = pd.read_sql_query(
        """
        select date::date as date, location, line, sum(time) as changeover_time_total
        from module4_output_changeoverlog
        where run_id = %(run_id)s
        group by 1,2,3
        """,
        conn,
        params={'run_id': RUN_ID}
    )
    cap = pd.read_sql_query(
        """
        select date::date as date, location, line, capacity
        from cfg_m4_linecapacity
        where config_name = %(config_name)s
        """,
        conn,
        params={'config_name': cfg_name}
    )
    rate = pd.read_sql_query(
        """
        select material, location, delegate_line as line, prd_rate
        from cfg_m4_materiallocationlinecfg
        where config_name = %(config_name)s
        """,
        conn,
        params={'config_name': cfg_name}
    )

prod['material'] = prod['material'].astype(str).str.strip()
rate['material'] = rate['material'].astype(str).str.strip()
prod = prod[prod['material'].isin(baby_materials)].copy()
rate = rate[rate['material'].isin(baby_materials)].copy()

prod['produced_qty'] = pd.to_numeric(prod['produced_qty'], errors='coerce').fillna(0)
rate['prd_rate'] = pd.to_numeric(rate['prd_rate'], errors='coerce').fillna(0)
chg['changeover_time_total'] = pd.to_numeric(chg['changeover_time_total'], errors='coerce').fillna(0)
cap['capacity'] = pd.to_numeric(cap['capacity'], errors='coerce').fillna(0)

prod = prod.merge(rate[['material','location','line','prd_rate']].drop_duplicates(), on=['material','location','line'], how='left')
prod['prd_rate'] = prod['prd_rate'].fillna(0)
prod['production_time'] = prod['produced_qty'] / prod['prd_rate'].replace({0: pd.NA})
prod['production_time'] = pd.to_numeric(prod['production_time'], errors='coerce').fillna(0)

prod_daily = prod.groupby(['date','location','line'], as_index=False)['production_time'].sum()
base = cap.merge(prod_daily, on=['date','location','line'], how='left').merge(chg, on=['date','location','line'], how='left')
base['production_time'] = base['production_time'].fillna(0)
base['changeover_time_total'] = base['changeover_time_total'].fillna(0)
base['total_used_time'] = base['production_time'] + base['changeover_time_total']
base['month'] = pd.to_datetime(base['date']).dt.strftime('%Y-%m')

out = base.groupby(['month','location','line'], as_index=False).agg(
    configured_capacity_time=('capacity','sum'),
    production_time=('production_time','sum'),
    changeover_time=('changeover_time_total','sum'),
    total_used_time=('total_used_time','sum')
)
out = out[out['total_used_time'] > 0].copy()
out['capacity_utilization'] = out['total_used_time'] / out['configured_capacity_time'].replace({0: pd.NA})
out['capacity_utilization_pct'] = (out['capacity_utilization'] * 100).round(2)
out = out.sort_values(['month','location','line']).reset_index(drop=True)

out_path = RESULTS / 'baby_by_month_by_line_capacity_utilization_timebased.csv'
out.to_csv(out_path, index=False, encoding='utf-8-sig')
print(out.to_string(index=False))
print(f'\nout={out_path}')
