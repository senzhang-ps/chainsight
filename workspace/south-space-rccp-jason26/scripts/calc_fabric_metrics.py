from pathlib import Path
import pandas as pd
import psycopg

ROOT = Path(r"\\wsl.localhost\Ubuntu-22.04\home\zhangs37\ai_chainsight")
CREDS_FILE = ROOT / 'db_credentials.yaml'
RESULTS = ROOT / 'workspace' / 'south-space-rccp-jason26' / 'scenarios' / 'production-cycle' / 'results'
RUN_ID = 'db_PDS2_20260527_181240'
FOCUS_LOCATIONS = ['C816', 'C810', 'D873', 'C866', 'C867']


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
fabric_materials = set(mapping.loc[mapping['category_en'] == 'Fabric', 'material'])

creds = load_creds()
with psycopg.connect(
    host=creds['host'], port=int(creds['port']), dbname=creds['database'],
    user=creds['user'], password=creds['password'], connect_timeout=10,
) as conn:
    inv = pd.read_sql_query(
        """
        select to_char(i.date::date, 'YYYY-MM') as month,
               i.date::date as inventory_date,
               i.location,
               i.material,
               sum(i.quantity) as qty,
               sum(i.quantity * coalesce(m.demand_unit_to_volume, 0)) as cbm
        from orchestrator_unrestricted_inventory i
        left join cfg_m6_materialmd m
          on m.config_name = i.config_name and m.material = i.material
        where i.run_id = %(run_id)s and i.location = any(%(locations)s)
        group by 1,2,3,4
        """,
        conn,
        params={'run_id': RUN_ID, 'locations': FOCUS_LOCATIONS}
    )
    ordlog = pd.read_sql_query(
        """
        select distinct date::date as date, material, location, demand_type, quantity, advance_days
        from module1_output_orderlog
        where run_id = %(run_id)s and location = any(%(locations)s)
        """,
        conn,
        params={'run_id': RUN_ID, 'locations': FOCUS_LOCATIONS}
    )
    shplog = pd.read_sql_query(
        """
        select date::date as date, material, location, quantity
        from module1_output_shipmentlog
        where run_id = %(run_id)s and location = any(%(locations)s)
        """,
        conn,
        params={'run_id': RUN_ID, 'locations': FOCUS_LOCATIONS}
    )
    fc = pd.read_sql_query(
        """
        with cfg as (
          select min(config_name) as config_name
          from orchestrator_unrestricted_inventory
          where run_id = %(run_id)s
        )
        select f.location, f.material, f.week, f.quantity
        from cfg_m1_demandforecast f
        join cfg c on c.config_name = f.config_name
        where f.location = any(%(locations)s)
        """,
        conn,
        params={'run_id': RUN_ID, 'locations': FOCUS_LOCATIONS}
    )
    sim_start = pd.read_sql_query(
        "select min(date)::date as start_date from orchestrator_unrestricted_inventory where run_id = %(run_id)s",
        conn,
        params={'run_id': RUN_ID}
    ).iloc[0,0]

inv['material'] = inv['material'].astype(str).str.strip()
ordlog['material'] = ordlog['material'].astype(str).str.strip()
shplog['material'] = shplog['material'].astype(str).str.strip()
fc['material'] = fc['material'].astype(str).str.strip()

inv_f = inv[inv['material'].isin(fabric_materials)].copy()
ord_f = ordlog[ordlog['material'].isin(fabric_materials)].copy()
shp_f = shplog[shplog['material'].isin(fabric_materials)].copy()
fc_f = fc[fc['material'].isin(fabric_materials)].copy()

# by month RCCP for Fabric across focus locations
fabric_rccp = (
    inv_f.groupby(['month', 'inventory_date'], as_index=False)['cbm'].sum()
    .rename(columns={'cbm':'daily_cbm'})
)
fabric_rccp = (
    fabric_rccp.sort_values(['month','daily_cbm','inventory_date'], ascending=[True, False, True])
    .drop_duplicates(subset=['month'], keep='first')
    .rename(columns={'inventory_date':'peak_date','daily_cbm':'peak_cbm'})
    .reset_index(drop=True)
)

# by month service / CFR
ord_f['month'] = pd.to_datetime(ord_f['date']).dt.strftime('%Y-%m')
shp_f['month'] = pd.to_datetime(shp_f['date']).dt.strftime('%Y-%m')
service = ord_f.groupby('month', as_index=False)['quantity'].sum().rename(columns={'quantity':'order_qty'}) \
    .merge(shp_f.groupby('month', as_index=False)['quantity'].sum().rename(columns={'quantity':'shipment_qty'}), on='month', how='outer')
service['order_qty'] = service['order_qty'].fillna(0)
service['shipment_qty'] = service['shipment_qty'].fillna(0)
service['cfr'] = service['shipment_qty'] / service['order_qty'].replace({0: pd.NA})
service['cfr_pct'] = (service['cfr'] * 100).round(2)
service = service.sort_values('month').reset_index(drop=True)

# by month DFC on Fabric RCCP peak day
inv_qty_daily = inv_f.groupby(['month','inventory_date'], as_index=False)['qty'].sum().rename(columns={'qty':'inventory_qty'})
peak_inv = fabric_rccp[['month','peak_date']].merge(inv_qty_daily, left_on=['month','peak_date'], right_on=['month','inventory_date'], how='left').drop(columns=['inventory_date'])
fc_weekly = fc_f.groupby(['week'], as_index=False)['quantity'].sum().rename(columns={'quantity':'weekly_forecast_qty'})
peak_inv['peak_week'] = ((pd.to_datetime(peak_inv['peak_date']) - pd.to_datetime(sim_start)).dt.days // 7) + 1
peak_inv = peak_inv.merge(fc_weekly, left_on='peak_week', right_on='week', how='left').drop(columns=['week'])
peak_inv['weekly_forecast_qty'] = peak_inv['weekly_forecast_qty'].fillna(0)
peak_inv['daily_avg_forecast_qty'] = peak_inv['weekly_forecast_qty'] / 7.0
peak_inv['dfc_days'] = peak_inv['inventory_qty'] / peak_inv['daily_avg_forecast_qty'].replace({0: pd.NA})
peak_inv['dfc_days'] = peak_inv['dfc_days'].round(2)

out = fabric_rccp.merge(service[['month','shipment_qty','order_qty','cfr_pct']], on='month', how='left') \
                 .merge(peak_inv[['month','peak_date','inventory_qty','peak_week','weekly_forecast_qty','daily_avg_forecast_qty','dfc_days']], on=['month','peak_date'], how='left')

out_path = RESULTS / 'fabric_by_month_rccp_service_dfc.csv'
out.to_csv(out_path, index=False, encoding='utf-8-sig')
print(out.to_string(index=False))
print(f'\nout={out_path}')
