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
select distinct material
from orchestrator_unrestricted_inventory
where run_id = %(run_id)s
  and location = any(%(locations)s)
order by material
"""
with psycopg.connect(
    host=creds['host'], port=int(creds['port']), dbname=creds['database'],
    user=creds['user'], password=creds['password'], connect_timeout=10,
) as conn:
    df = pd.read_sql_query(query, conn, params={'run_id': run_id, 'locations': focus_locations})
out = ROOT / 'workspace' / 'south-space-rccp-jason26' / 'scenarios' / 'production-cycle' / 'results' / 'focus_run_materials.csv'
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False, encoding='utf-8-sig')
print(df.head(20).to_string(index=False))
print(f'rows={len(df)}\nout={out}')
