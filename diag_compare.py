import os, pandas as pd, psycopg

base = r'D:\PG\Code\chainsight\config\OC_Paste_S1_20251224\run_20260127_142402'

# Module5 actual filenames
m5_files = sorted(os.listdir(os.path.join(base,'module5')))[:3]
print(f'Module5 files: {m5_files}')

m6_files = sorted(os.listdir(os.path.join(base,'module6')))[:3]
print(f'Module6 files: {m6_files}')

# Try correct module5 name
xf = pd.ExcelFile(os.path.join(base,'module5',m5_files[0]))
print(f'Module5 sheets: {xf.sheet_names}')

# Module6 xlsx
m6_xlsx = [f for f in os.listdir(os.path.join(base,'module6')) if f.endswith('.xlsx')][:3]
xf = pd.ExcelFile(os.path.join(base,'module6',m6_xlsx[0]))
print(f'Module6 sheets: {xf.sheet_names}')

# Check DB tables for M4/M5/M6
conn = psycopg.connect('host=localhost port=5432 dbname=test_db user=postgres password=123456')
cur = conn.cursor()
cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND (table_name LIKE 'module4%%' OR table_name LIKE 'module5%%' OR table_name LIKE 'module6%%' OR table_name LIKE 'summary%%') ORDER BY table_name")
print('DB tables:', [r[0] for r in cur.fetchall()])

for t in ['module4_output_productionplan','module4_output_capacityexceed','module5_output_deploymentplan','module6_output_changeoverlog']:
    try:
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name=%s ORDER BY ordinal_position", (t,))
        cols = [r[0] for r in cur.fetchall()]
        print(f'{t}: {cols}')
    except:
        print(f'{t}: NOT FOUND')

# summary tables
for t in ['summary_output_order_shipment_cut','summary_output_deployment','summary_output_changeover','summary_output_exceed_capacity']:
    try:
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name=%s ORDER BY ordinal_position", (t,))
        cols = [r[0] for r in cur.fetchall()]
        print(f'{t}: {cols}')
    except:
        print(f'{t}: NOT FOUND')

conn.close()
