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
with ord as (
    select
        to_char(o.date::date, 'YYYY-MM') as month,
        o.location as receiving,
        sum(o.quantity) as order_qty
    from (
        select distinct date, material, location, demand_type, quantity, advance_days
        from module1_output_orderlog
        where run_id = %(run_id)s
          and location = any(%(locations)s)
    ) o
    group by 1, 2
), shp as (
    select
        to_char(s.date::date, 'YYYY-MM') as month,
        s.location as receiving,
        sum(s.quantity) as shipment_qty
    from module1_output_shipmentlog s
    where s.run_id = %(run_id)s
      and s.location = any(%(locations)s)
    group by 1, 2
)
select
    coalesce(ord.month, shp.month) as month,
    coalesce(ord.receiving, shp.receiving) as location,
    round(coalesce(shp.shipment_qty, 0)::numeric, 2) as shipment_qty,
    round(coalesce(ord.order_qty, 0)::numeric, 2) as order_qty,
    round((coalesce(shp.shipment_qty, 0) / nullif(ord.order_qty, 0))::numeric, 4) as cfr
from ord
full outer join shp
  on shp.month = ord.month
 and shp.receiving = ord.receiving
order by 1, 2
"""

with psycopg.connect(
    host=creds['host'], port=int(creds['port']), dbname=creds['database'],
    user=creds['user'], password=creds['password'], connect_timeout=10,
) as conn:
    df = pd.read_sql_query(query, conn, params={'run_id': run_id, 'locations': focus_locations})

df['cfr_pct'] = (df['cfr'] * 100).round(2)
out = ROOT / 'workspace' / 'south-space-rccp-jason26' / 'scenarios' / 'production-cycle' / 'results' / 'cfr_by_month_focus_locations.csv'
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False, encoding='utf-8-sig')
print(df.to_string(index=False))
print(f'\nrows={len(df)}\nout={out}')
