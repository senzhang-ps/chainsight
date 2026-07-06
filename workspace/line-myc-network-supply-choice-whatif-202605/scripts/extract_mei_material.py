"""
Extract Month End Inventory by material x location for line-myc project.
Appends {wf}_mei_material_loc keys to kpi_results.json.
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

# Load existing results
results_path = Path(r'\\wsl.localhost\Ubuntu-22.04\home\zhangs37\ai_chainsight\workspace\line-myc-network-supply-choice-whatif-202605\scripts\kpi_results.json')
results = json.loads(results_path.read_text())

with psycopg.connect(host=cfg['host'], port=int(cfg['port']), dbname=cfg['database'],
                       user=cfg['user'], password=cfg['password'], connect_timeout=10) as conn:
    with conn.cursor() as cur:
        for sc, rid in scenarios.items():
            cn = config_names[sc]
            print(f"=== {sc}: MEI by material x location ===", file=sys.stderr)
            cur.execute('''
            WITH inv AS (
              SELECT i.date::date AS dt, i.location, i.material, i.quantity
              FROM orchestrator_unrestricted_inventory i
              WHERE i.run_id = %(rid)s
            ), ranked AS (
              SELECT to_char(dt,'YYYY-MM') AS month, location, material, dt, quantity,
                     row_number() OVER (PARTITION BY to_char(dt,'YYYY-MM'), location, material ORDER BY dt DESC) AS rn
              FROM inv
            )
            SELECT month, location, material, dt AS month_end_date, round(quantity::numeric,2) AS inventory_qty
            FROM ranked WHERE rn=1
            ORDER BY 1,2,3
            ''', {'rid': rid, 'cn': cn})
            results[f'{sc}_mei_material_loc'] = {
                'columns': ['month','location','material','month_end_date','inventory_qty'],
                'rows': [list(r) for r in cur.fetchall()]
            }
            print(f"  {len(results[f'{sc}_mei_material_loc']['rows'])} rows", file=sys.stderr)

results_path.write_text(json.dumps(results, cls=Enc, indent=2))
print(f"Updated {results_path}", file=sys.stderr)
print("OK")
