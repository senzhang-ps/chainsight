"""
Extract all KPIs including Month End Inventory for the line-myc project.
Run with Windows Python.
"""
from pathlib import Path
import psycopg, json, sys, decimal, datetime

scenarios = {
    'wf1': 'db_wf1-produce-hs-order-5050-current-ss_20260512_230651',
    'wf2': 'db_wf2-produce-5050-order-hs-current-ss_20260512_230707',
    'wf3': 'db_wf3-produce-hs-order-5050-reduced-ss_20260512_230717',
    'wf4': 'db_wf4-produce-5050-order-hs-reduced-ss_20260512_230848',
}
# Each scenario uses a config_name matching its workbook name (without .xlsx)
config_names = {
    'wf1': 'wf1-produce-hs-order-5050-current-ss',
    'wf2': 'wf2-produce-5050-order-hs-current-ss',
    'wf3': 'wf3-produce-hs-order-5050-reduced-ss',
    'wf4': 'wf4-produce-5050-order-hs-reduced-ss',
}

cfg = {}
for line in Path(r'\\wsl.localhost\Ubuntu-22.04\home\zhangs37\ai_chainsight\db_credentials.yaml').read_text().splitlines():
    line = line.strip()
    if not line or line.startswith('#'): continue
    k, v = line.split(':', 1)
    cfg[k.strip()] = v.strip()

class Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, decimal.Decimal): return float(o)
        if isinstance(o, (datetime.date, datetime.datetime)): return str(o)
        return super().default(o)

results = {}

with psycopg.connect(host=cfg['host'], port=int(cfg['port']), dbname=cfg['database'],
                       user=cfg['user'], password=cfg['password'], connect_timeout=10) as conn:
    with conn.cursor() as cur:
        for sc, rid in scenarios.items():
            cn = config_names[sc]
            print(f"=== {sc} ({rid}) ===", file=sys.stderr)

            # 1. Service by month
            print(f"  1/7 Service by month...", file=sys.stderr)
            cur.execute('''
            with ord as (
              select to_char(o.date::date,'YYYY-MM') as month,
                     sum(o.quantity) as order_qty
              from module1_output_orderlog o
              where o.run_id=%(rid)s
              group by 1
            ), shp as (
              select to_char(s.date::date,'YYYY-MM') as month,
                     sum(s.quantity) as shipment_qty
              from module1_output_shipmentlog s
              where s.run_id=%(rid)s group by 1
            )
            select coalesce(o.month,s.month),
                   round(coalesce(s.shipment_qty,0)::numeric,2),
                   round(coalesce(o.order_qty,0)::numeric,2),
                   round((coalesce(s.shipment_qty,0)/nullif(o.order_qty,0))::numeric,4)
            from ord o full outer join shp s on s.month=o.month order by 1
            ''', {'rid': rid})
            results[f'{sc}_service_month'] = {
                'columns': ['month','shipment_qty','order_qty','cfr'],
                'rows': [list(r) for r in cur.fetchall()]
            }

            # 2. Service by month by location
            print(f"  2/7 Service by month by location...", file=sys.stderr)
            cur.execute('''
            with ord as (
              select to_char(o.date::date,'YYYY-MM') as month, o.location,
                     sum(o.quantity) as order_qty
              from module1_output_orderlog o
              where o.run_id=%(rid)s
              group by 1,2
            ), shp as (
              select to_char(s.date::date,'YYYY-MM') as month, s.location,
                     sum(s.quantity) as shipment_qty
              from module1_output_shipmentlog s
              where s.run_id=%(rid)s group by 1,2
            )
            select coalesce(o.month,s.month), coalesce(o.location,s.location),
                   round(coalesce(s.shipment_qty,0)::numeric,2),
                   round(coalesce(o.order_qty,0)::numeric,2),
                   round((coalesce(s.shipment_qty,0)/nullif(o.order_qty,0))::numeric,4)
            from ord o full outer join shp s on s.month=o.month and s.location=o.location
            order by 1,2
            ''', {'rid': rid})
            results[f'{sc}_service_month_loc'] = {
                'columns': ['month','location','shipment_qty','order_qty','cfr'],
                'rows': [list(r) for r in cur.fetchall()]
            }

            # 3. Space RCCP by month by location
            print(f"  3/7 Space RCCP...", file=sys.stderr)
            cur.execute('''
            with inv as (
              select i.date::date as dt, i.location, i.material,
                     i.quantity * coalesce(m.demand_unit_to_volume,0) as cbm
              from orchestrator_unrestricted_inventory i
              left join cfg_m6_materialmd m on m.config_name=%(cn)s and m.material=i.material
              where i.run_id=%(rid)s
            ), daily as (
              select to_char(dt,'YYYY-MM') as month, location, dt, sum(cbm) as total_cbm
              from inv group by 1,2,3
            )
            select month, location,
                   round(max(total_cbm)::numeric,4) as rccp_cbm,
                   (array_agg(dt order by total_cbm desc))[1] as peak_date
            from daily group by 1,2
            order by 1,2
            ''', {'rid': rid, 'cn': cn})
            results[f'{sc}_rccp_month_loc'] = {
                'columns': ['month','location','rccp_cbm','peak_date'],
                'rows': [list(r) for r in cur.fetchall()]
            }

            # 4. Month End Inventory by location
            print(f"  4/7 Month End Inventory...", file=sys.stderr)
            cur.execute('''
            with inv as (
              select i.date::date as dt, i.location, sum(i.quantity) as total_qty,
                     sum(i.quantity * coalesce(m.demand_unit_to_volume,0)) as total_cbm
              from orchestrator_unrestricted_inventory i
              left join cfg_m6_materialmd m on m.config_name=%(cn)s and m.material=i.material
              where i.run_id=%(rid)s
              group by 1,2
            ), ranked as (
              select to_char(dt,'YYYY-MM') as month, location, dt, total_qty, total_cbm,
                     row_number() over (partition by to_char(dt,'YYYY-MM'), location order by dt desc) as rn
              from inv
            )
            select month, location, dt as month_end_date,
                   round(total_qty::numeric,2) as inventory_qty,
                   round(total_cbm::numeric,4) as inventory_cbm
            from ranked where rn=1
            order by 1,2
            ''', {'rid': rid, 'cn': cn})
            results[f'{sc}_mei_loc'] = {
                'columns': ['month','location','month_end_date','inventory_qty','inventory_cbm'],
                'rows': [list(r) for r in cur.fetchall()]
            }

            # 4b. Month End Inventory network total
            print(f"  4b/7 Month End Inventory network...", file=sys.stderr)
            cur.execute('''
            with inv as (
              select i.date::date as dt, sum(i.quantity) as total_qty,
                     sum(i.quantity * coalesce(m.demand_unit_to_volume,0)) as total_cbm
              from orchestrator_unrestricted_inventory i
              left join cfg_m6_materialmd m on m.config_name=%(cn)s and m.material=i.material
              where i.run_id=%(rid)s
              group by 1
            ), ranked as (
              select to_char(dt,'YYYY-MM') as month, dt, total_qty, total_cbm,
                     row_number() over (partition by to_char(dt,'YYYY-MM') order by dt desc) as rn
              from inv
            )
            select month, dt as month_end_date,
                   round(total_qty::numeric,2) as inventory_qty,
                   round(total_cbm::numeric,4) as inventory_cbm
            from ranked where rn=1
            order by 1
            ''', {'rid': rid, 'cn': cn})
            results[f'{sc}_mei_network'] = {
                'columns': ['month','month_end_date','inventory_qty','inventory_cbm'],
                'rows': [list(r) for r in cur.fetchall()]
            }

            # 5. Changeover
            print(f"  5/7 Changeover...", file=sys.stderr)
            cur.execute('''
            with raw as (
              select to_char(date::date,'YYYY-MM') as month, location, line, changeover_type,
                     sum(count) as co_count, sum(time) as co_time
              from module4_output_changeoverlog
              where run_id=%(rid)s group by 1,2,3,4
            ), totals as (
              select month, location, line, sum(co_count) as total_count
              from raw group by 1,2,3
            )
            select r.month, r.location, r.line, r.changeover_type,
                   r.co_count, round(r.co_time::numeric,4) as co_time,
                   round((r.co_count::numeric / nullif(t.total_count,0))::numeric,4) as pct
            from raw r join totals t on t.month=r.month and t.location=r.location and t.line=r.line
            order by 1,2,3,4
            ''', {'rid': rid})
            results[f'{sc}_changeover'] = {
                'columns': ['month','location','line','changeover_type','co_count','co_time','pct'],
                'rows': [list(r) for r in cur.fetchall()]
            }

            # 6. Lane lead time diagnostics
            print(f"  6/7 Lane LT...", file=sys.stderr)
            cur.execute('''
            with ship as (
              select sending, receiving, ori_deployment_uid,
                     min(planned_deployment_date::date) as planned_dt,
                     min(actual_ship_date::date) as ship_dt,
                     min(actual_delivery_date::date) as delivery_dt,
                     sum(delivery_qty) as delivery_qty
              from module6_output_deliveryplan
              where run_id=%(rid)s and sending <> receiving
              group by 1,2,3
            ), stats as (
              select sending, receiving,
                     count(*) as uid_cnt, sum(delivery_qty) as total_qty,
                     round(avg(ship_dt - planned_dt)::numeric,4) as mean_wait_moq,
                     round(percentile_cont(0.5) within group (order by ship_dt - planned_dt)::numeric,2) as median_wait_moq,
                     round(percentile_cont(0.9) within group (order by ship_dt - planned_dt)::numeric,2) as p90_wait_moq,
                     round(avg(delivery_dt - ship_dt)::numeric,4) as mean_otd,
                     round(percentile_cont(0.5) within group (order by delivery_dt - ship_dt)::numeric,2) as median_otd,
                     round(percentile_cont(0.9) within group (order by delivery_dt - ship_dt)::numeric,2) as p90_otd,
                     round(avg(delivery_dt - planned_dt)::numeric,4) as mean_ptd,
                     round(percentile_cont(0.5) within group (order by delivery_dt - planned_dt)::numeric,2) as median_ptd,
                     round(percentile_cont(0.9) within group (order by delivery_dt - planned_dt)::numeric,2) as p90_ptd
              from ship group by 1,2
            )
            select s.sending, s.receiving, s.uid_cnt, round(s.total_qty::numeric,2),
                   s.mean_wait_moq, s.median_wait_moq, s.p90_wait_moq,
                   s.mean_otd, s.median_otd, s.p90_otd,
                   s.mean_ptd, s.median_ptd, s.p90_ptd,
                   c.pdt as cfg_pdt, c.otd as cfg_otd, c.gr as cfg_gr,
                   round((c.pdt + c.gr)::numeric,2) as cfg_total_lt
            from stats s
            left join cfg_global_leadtime c on c.config_name=%(cn)s and c.sending=s.sending and c.receiving=s.receiving
            order by s.receiving, s.sending
            ''', {'rid': rid, 'cn': cn})
            results[f'{sc}_lane_lt'] = {
                'columns': ['sending','receiving','uid_cnt','total_qty',
                             'mean_wait_moq','median_wait_moq','p90_wait_moq',
                             'mean_otd','median_otd','p90_otd',
                             'mean_ptd','median_ptd','p90_ptd',
                             'cfg_pdt','cfg_otd','cfg_gr','cfg_total_lt'],
                'rows': [list(r) for r in cur.fetchall()]
            }

            print(f"  Done {sc}.", file=sys.stderr)

        print("All queries done.", file=sys.stderr)

out = Path(r'\\wsl.localhost\Ubuntu-22.04\home\zhangs37\ai_chainsight\workspace\line-myc-network-supply-choice-whatif-202605\scripts\kpi_results.json')
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, cls=Enc, indent=2))
print(f"Results written to {out}", file=sys.stderr)
print("OK")
