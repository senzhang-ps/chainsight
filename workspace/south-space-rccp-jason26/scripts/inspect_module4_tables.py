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

queries = {
    'productionplan': "select * from module4_output_productionplan limit 3",
    'changeoverlog': "select * from module4_output_changeoverlog limit 3",
}
with psycopg.connect(host=creds['host'], port=int(creds['port']), dbname=creds['database'], user=creds['user'], password=creds['password'], connect_timeout=10) as conn:
    for name, q in queries.items():
        df = pd.read_sql_query(q, conn)
        print('TABLE', name)
        print(df.columns.tolist())
        print(df.head(3).to_string())
        print()
