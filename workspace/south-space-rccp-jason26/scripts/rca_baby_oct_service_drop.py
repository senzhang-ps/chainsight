from pathlib import Path
import pandas as pd
import psycopg

ROOT = Path(r"\\wsl.localhost\Ubuntu-22.04\home\zhangs37\ai_chainsight")
CREDS_FILE = ROOT / 'db_credentials.yaml'
RESULTS = ROOT / 'workspace' / 'south-space-rccp-jason26' / 'scenarios' / 'production-cycle' / 'results'
RUN_ID = 'db_PDS2_20260527_181240'
CATEGORY = 'Baby'
TARGET_MONTH = '2026-10'
FOCUS_LOCATIONS = ['C816', 'C866', 'D873']
UPSTREAM_LOCATIONS = ['A888', '0386', 'C816', 'C866', 'D873']


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
    orderlog = pd.read_sql_query(
        """
        select distinct date::date as date, material, location, quantity
        from module1_output_orderlog
        where run_id = %(run_id)s and location = any(%(locations)s)
        """,
        conn,
        params={'run_id': RUN_ID, 'locations': FOCUS_LOCATIONS}
    )
    shiplog = pd.read_sql_query(
        """
        select date::date as date, material, location, quantity
        from module1_output_shipmentlog
        where run_id = %(run_id)s and location = any(%(locations)s)
        """,
        conn,
        params={'run_id': RUN_ID, 'locations': FOCUS_LOCATIONS}
    )
    inv = pd.read_sql_query(
        """
        select date::date as date, material, location, quantity
        from orchestrator_unrestricted_inventory
        where run_id = %(run_id)s and location = any(%(locations)s)
        """,
        conn,
        params={'run_id': RUN_ID, 'locations': UPSTREAM_LOCATIONS}
    )
    prod = pd.read_sql_query(
        """
        select simulation_date::date as date, material, location, line, produced_qty
        from module4_output_productionplan
        where run_id = %(run_id)s and location = 'A888'
        """,
        conn,
        params={'run_id': RUN_ID}
    )
    fc = pd.read_sql_query(
        """
        with cfg as (
          select min(config_name) as config_name
          from orchestrator_unrestricted_inventory
          where run_id = %(run_id)s
        )
        select material, location, week, quantity
        from cfg_m1_demandforecast f
        join cfg c on c.config_name = f.config_name
        where location = any(%(locations)s)
        """,
        conn,
        params={'run_id': RUN_ID, 'locations': FOCUS_LOCATIONS}
    )

for df in [orderlog, shiplog, inv, prod, fc]:
    df['material'] = df['material'].astype(str).str.strip()

orderlog = orderlog[orderlog['material'].isin(baby_materials)].copy()
shiplog = shiplog[shiplog['material'].isin(baby_materials)].copy()
inv = inv[inv['material'].isin(baby_materials)].copy()
prod = prod[prod['material'].isin(baby_materials)].copy()
fc = fc[fc['material'].isin(baby_materials)].copy()

orderlog['month'] = pd.to_datetime(orderlog['date']).dt.strftime('%Y-%m')
shiplog['month'] = pd.to_datetime(shiplog['date']).dt.strftime('%Y-%m')
inv['month'] = pd.to_datetime(inv['date']).dt.strftime('%Y-%m')
prod['month'] = pd.to_datetime(prod['date']).dt.strftime('%Y-%m')

order_oct = orderlog[orderlog['month'] == TARGET_MONTH].groupby(['location','material'], as_index=False)['quantity'].sum().rename(columns={'quantity':'order_qty'})
ship_oct = shiplog[shiplog['month'] == TARGET_MONTH].groupby(['location','material'], as_index=False)['quantity'].sum().rename(columns={'quantity':'ship_qty'})
svc = order_oct.merge(ship_oct, on=['location','material'], how='left')
svc['ship_qty'] = svc['ship_qty'].fillna(0)
svc['gap_qty'] = svc['order_qty'] - svc['ship_qty']
svc['cfr'] = svc['ship_qty'] / svc['order_qty'].replace({0: pd.NA})
svc = svc.sort_values(['location','gap_qty'], ascending=[True,False]).reset_index(drop=True)

prod_oct = prod[prod['month'] == TARGET_MONTH].groupby(['material','line'], as_index=False)['produced_qty'].sum()
prod_oct_total = prod[prod['month'] == TARGET_MONTH].groupby(['material'], as_index=False)['produced_qty'].sum().rename(columns={'produced_qty':'a888_produced_qty'})

inv_oct = inv[inv['month'] == TARGET_MONTH].groupby(['location','material'], as_index=False)['quantity'].max().rename(columns={'quantity':'max_inventory_qty_oct'})
inv_wide = inv_oct.pivot(index='material', columns='location', values='max_inventory_qty_oct').reset_index().fillna(0)

fc_oct = fc.copy()
# week buckets that start in Oct or overlap Oct materially are approximated by week start mapping from sim start
sim_start = pd.Timestamp('2026-06-29')
fc_oct['week'] = pd.to_numeric(fc_oct['week'], errors='coerce')
fc_oct['week_start'] = sim_start + pd.to_timedelta((fc_oct['week'] - 1) * 7, unit='D')
fc_oct['month'] = fc_oct['week_start'].dt.strftime('%Y-%m')
fc_oct = fc_oct[fc_oct['month'] == TARGET_MONTH].groupby(['location','material'], as_index=False)['quantity'].sum().rename(columns={'quantity':'forecast_qty_oct'})

rca = svc.merge(prod_oct_total, on='material', how='left').merge(inv_wide, on='material', how='left').merge(fc_oct, on=['location','material'], how='left')
rca['a888_produced_qty'] = rca['a888_produced_qty'].fillna(0)
rca['forecast_qty_oct'] = rca['forecast_qty_oct'].fillna(0)
for col in ['A888','0386','C816','C866','D873']:
    if col not in rca.columns:
        rca[col] = 0
rca = rca.sort_values(['location','gap_qty'], ascending=[True,False]).reset_index(drop=True)

out = RESULTS / 'rca_baby_oct_service_drop.csv'
rca.to_csv(out, index=False, encoding='utf-8-sig')

summary = rca.groupby('location', as_index=False).agg(
    order_qty=('order_qty','sum'),
    ship_qty=('ship_qty','sum'),
    gap_qty=('gap_qty','sum'),
    forecast_qty_oct=('forecast_qty_oct','sum'),
    a888_produced_qty=('a888_produced_qty','sum')
)
summary['cfr_pct'] = (summary['ship_qty'] / summary['order_qty'] * 100).round(2)
print('SUMMARY')
print(summary.to_string(index=False))
for loc in FOCUS_LOCATIONS:
    print(f'\nTOP_GAP_SKUS_{loc}')
    print(rca[rca['location'] == loc].head(15).to_string(index=False))
print(f'\nout={out}')
