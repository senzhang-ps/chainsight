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

query = """
with daily as (
    select
        to_char(i.date::date, 'YYYY-MM') as month,
        i.date::date as peak_date,
        i.location,
        sum(i.quantity * coalesce(m.demand_unit_to_volume, 0)) as daily_cbm
    from orchestrator_unrestricted_inventory i
    left join cfg_m6_materialmd m
      on m.config_name = i.config_name
     and m.material = i.material
    where i.run_id = %(run_id)s
      and i.location = any(%(locations)s)
    group by 1, 2, 3
), ranked as (
    select month, peak_date, location, daily_cbm,
           row_number() over (
               partition by month, location
               order by daily_cbm desc, peak_date asc
           ) as row_num
    from daily
)
select month, location, peak_date, round(daily_cbm::numeric, 2) as peak_cbm
from ranked
where row_num = 1
order by month, location
"""

with psycopg.connect(
    host=creds['host'], port=int(creds['port']), dbname=creds['database'],
    user=creds['user'], password=creds['password'], connect_timeout=10,
) as conn:
    df = pd.read_sql_query(query, conn, params={'run_id': run_id, 'locations': focus_locations})

out = ROOT / 'workspace' / 'south-space-rccp-jason26' / 'scenarios' / 'production-cycle' / 'results' / 'rccp_by_month_focus_locations.csv'
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False, encoding='utf-8-sig')
print(df.to_string(index=False))
print(f'\nrows={len(df)}\nout={out}')
