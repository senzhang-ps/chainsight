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
with run_cfg as (
    select min(config_name) as config_name
    from orchestrator_unrestricted_inventory
    where run_id = %(run_id)s
), daily_inventory as (
    select
        to_char(i.date::date, 'YYYY-MM') as month,
        i.date::date as inventory_date,
        i.location,
        round(sum(i.quantity)::numeric, 4) as inventory_qty
    from orchestrator_unrestricted_inventory i
    where i.run_id = %(run_id)s
      and i.location = any(%(locations)s)
    group by 1, 2, 3
), peak_day as (
    select
        month,
        location,
        inventory_date as peak_date,
        inventory_qty as peak_inventory_qty,
        row_number() over (
            partition by month, location
            order by inventory_qty desc, inventory_date asc
        ) as row_num
    from daily_inventory
), weekly_fc as (
    select
        location,
        week,
        round(sum(quantity)::numeric, 4) as weekly_forecast_qty
    from cfg_m1_demandforecast f
    join run_cfg c on c.config_name = f.config_name
    where f.location = any(%(locations)s)
    group by 1, 2
), sim_start as (
    select min(date)::date as start_date
    from orchestrator_unrestricted_inventory
    where run_id = %(run_id)s
), peak_with_week as (
    select
        p.month,
        p.location,
        p.peak_date,
        p.peak_inventory_qty,
        s.start_date,
        ((p.peak_date - s.start_date) / 7)::int + 1 as peak_week
    from peak_day p
    cross join sim_start s
    where p.row_num = 1
), fc_join as (
    select
        p.month,
        p.location,
        p.peak_date,
        p.peak_inventory_qty,
        p.peak_week,
        coalesce(w.weekly_forecast_qty, 0) as weekly_forecast_qty
    from peak_with_week p
    left join weekly_fc w
      on w.location = p.location
     and w.week = p.peak_week
)
select
    month,
    location,
    peak_date,
    peak_inventory_qty,
    peak_week,
    weekly_forecast_qty,
    round((weekly_forecast_qty / 7.0)::numeric, 4) as daily_avg_forecast_qty,
    round((peak_inventory_qty / nullif(weekly_forecast_qty / 7.0, 0))::numeric, 2) as dfc_days
from fc_join
order by month, location
"""

with psycopg.connect(
    host=creds['host'], port=int(creds['port']), dbname=creds['database'],
    user=creds['user'], password=creds['password'], connect_timeout=10,
) as conn:
    df = pd.read_sql_query(query, conn, params={'run_id': run_id, 'locations': focus_locations})

out = ROOT / 'workspace' / 'south-space-rccp-jason26' / 'scenarios' / 'production-cycle' / 'results' / 'dfc_on_rccp_peak_day_by_month_focus_locations.csv'
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False, encoding='utf-8-sig')
print(df.to_string(index=False))
print(f'\nrows={len(df)}\nout={out}')
