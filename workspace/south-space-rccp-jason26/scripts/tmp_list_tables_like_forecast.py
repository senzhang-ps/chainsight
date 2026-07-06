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

query = """
select table_schema, table_name
from information_schema.tables
where table_schema = 'public'
  and (
    lower(table_name) like '%forecast%'
    or lower(table_name) like '%demand%'
    or lower(table_name) like '%orderlog%'
    or lower(table_name) like '%shipmentlog%'
    or lower(table_name) like '%inventory%'
  )
order by table_name
"""

with psycopg.connect(
    host=creds['host'], port=int(creds['port']), dbname=creds['database'],
    user=creds['user'], password=creds['password'], connect_timeout=10,
) as conn:
    df = pd.read_sql_query(query, conn)
print(df.to_string(index=False))
