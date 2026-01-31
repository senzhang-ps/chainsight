# -*- coding: utf-8 -*-
"""Quick comparison of UnfulfilledLog counts between DB and Local mode"""

import pandas as pd
import psycopg
from pathlib import Path

# DB connection
conn = psycopg.connect('host=localhost port=5432 dbname=test_db user=postgres password=123456')

# Load DB unfulfilled_log for the new run
run_id = 'BC_S5_20260130_114318'
query = f"""
SELECT sim_date, COUNT(*) as count 
FROM module5_output_unfulfilledlog 
WHERE run_id = '{run_id}'
GROUP BY sim_date 
ORDER BY sim_date
"""
db_df = pd.read_sql(query, conn)
conn.close()

# Load local unfulfilled_log
local_dir = Path('outputs/BC_S5/run_20260130_113019')
local_counts = {}
dates = ['2025-10-06', '2025-10-07', '2025-10-08', '2025-10-09', '2025-10-10']
for d in dates:
    d_str = d.replace("-", "")
    f = local_dir / 'module5' / f'Module5Output_{d_str}.xlsx'
    if f.exists():
        df = pd.read_excel(f, sheet_name='UnfulfilledLog')
        local_counts[d] = len(df)

print('Date comparison (UnfulfilledLog):')
print('Date         | DB     | Local  | Match?')
print('-' * 45)
for _, row in db_df.iterrows():
    # Extract just the date part, handle different formats
    d_raw = str(row['sim_date'])
    if 'T' in d_raw:
        d = d_raw.split('T')[0]
    elif ' ' in d_raw:
        d = d_raw.split(' ')[0]
    else:
        d = d_raw[:10] if len(d_raw) >= 10 else d_raw
    
    # Convert YYYYMMDD to YYYY-MM-DD if needed
    if len(d) == 8 and d.isdigit():
        d = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
    
    db_cnt = row['count']
    local_cnt = local_counts.get(d, 'N/A')
    match = 'YES' if db_cnt == local_cnt else 'NO'
    print(f'{d} | {db_cnt:6} | {str(local_cnt):6} | {match}')
