from pathlib import Path
import pandas as pd
import psycopg

ROOT = Path(r"\\wsl.localhost\Ubuntu-22.04\home\zhangs37\ai_chainsight")
creds = {}
for raw in (ROOT / 'db_credentials.yaml').read_text(encoding='utf-8').splitlines():
    line = raw.strip()
    if not line or line.startswith('#'):
        continue
    k, v = line.split(':', 1)
    creds[k.strip()] = v.strip()
creds['database'] = 'south_rccp'

run_id = 'db_PDS2_20260527_181240'
focus_locations = ['C816', 'C810', 'D873', 'C866', 'C867']
mapping_file = ROOT / 'workspace' / 'sdc-space-rccp-simulation-202605' / 'data' / 'ps_psc_sku_master_category_en.csv'

mapping = pd.read_csv(mapping_file, dtype=str)
if 'material' not in mapping.columns and 'material_num' in mapping.columns:
    mapping = mapping.rename(columns={'material_num': 'material'})
mapping.columns = [str(c).strip() for c in mapping.columns]
for c in mapping.columns:
    mapping[c] = mapping[c].astype(str).str.strip()
keep = [c for c in ['material', 'bu_attr'] if c in mapping.columns]
mapping = mapping[keep].drop_duplicates(subset=['material']).copy()
mapping['bu_attr'] = mapping['bu_attr'].replace({'': '(unmapped)', 'nan': '(unmapped)', 'None': '(unmapped)'}).fillna('(unmapped)')

query = """
select
    to_char(i.date::date, 'YYYY-MM') as month,
    i.location,
    i.date::date as inventory_date,
    i.material,
    round(sum(i.quantity * coalesce(m.demand_unit_to_volume, 0))::numeric, 4) as material_cbm
from orchestrator_unrestricted_inventory i
left join cfg_m6_materialmd m
  on m.config_name = i.config_name
 and m.material = i.material
where i.run_id = %(run_id)s
  and i.location = any(%(locations)s)
group by 1, 2, 3, 4
order by 1, 2, 3, 4
"""

with psycopg.connect(
    host=creds['host'], port=int(creds['port']), dbname=creds['database'],
    user=creds['user'], password=creds['password'], connect_timeout=10,
) as conn:
    detail = pd.read_sql_query(query, conn, params={'run_id': run_id, 'locations': focus_locations})

detail['material'] = detail['material'].astype(str).str.strip()
detail = detail.merge(mapping, on='material', how='left')
detail['bu_attr'] = detail['bu_attr'].fillna('(unmapped)')

# month x location x BU: first sum to daily BU CBM, then take monthly peak
by_loc_bu_daily = (
    detail.groupby(['month', 'location', 'inventory_date', 'bu_attr'], as_index=False)['material_cbm']
    .sum()
    .rename(columns={'bu_attr': 'bu', 'material_cbm': 'daily_cbm'})
)
by_loc_bu_peak = (
    by_loc_bu_daily.sort_values(['month', 'location', 'bu', 'daily_cbm', 'inventory_date'], ascending=[True, True, True, False, True])
    .drop_duplicates(subset=['month', 'location', 'bu'], keep='first')
    .rename(columns={'inventory_date': 'peak_date', 'daily_cbm': 'peak_cbm'})
    .sort_values(['month', 'location', 'peak_cbm', 'bu'], ascending=[True, True, False, True])
    .reset_index(drop=True)
)

# month x BU: sum daily across focus locations first, then take monthly peak
by_bu_daily = (
    detail.groupby(['month', 'inventory_date', 'bu_attr'], as_index=False)['material_cbm']
    .sum()
    .rename(columns={'bu_attr': 'bu', 'material_cbm': 'daily_cbm'})
)
by_bu_peak = (
    by_bu_daily.sort_values(['month', 'bu', 'daily_cbm', 'inventory_date'], ascending=[True, True, False, True])
    .drop_duplicates(subset=['month', 'bu'], keep='first')
    .rename(columns={'inventory_date': 'peak_date', 'daily_cbm': 'peak_cbm'})
    .sort_values(['month', 'peak_cbm', 'bu'], ascending=[True, False, True])
    .reset_index(drop=True)
)

out1 = ROOT / 'workspace' / 'south-space-rccp-jason26' / 'scenarios' / 'production-cycle' / 'results' / 'rccp_by_month_location_bu_focus_locations.csv'
out2 = ROOT / 'workspace' / 'south-space-rccp-jason26' / 'scenarios' / 'production-cycle' / 'results' / 'rccp_by_month_bu_focus_locations.csv'
out1.parent.mkdir(parents=True, exist_ok=True)
by_loc_bu_peak.to_csv(out1, index=False, encoding='utf-8-sig')
by_bu_peak.to_csv(out2, index=False, encoding='utf-8-sig')
print('BY_MONTH_BY_LOCATION_BY_BU')
print(by_loc_bu_peak.head(60).to_string(index=False))
print(f'\\nrows={len(by_loc_bu_peak)}\\nout={out1}')
print('\\nBY_MONTH_BY_BU')
print(by_bu_peak.to_string(index=False))
print(f'\\nrows={len(by_bu_peak)}\\nout={out2}')
