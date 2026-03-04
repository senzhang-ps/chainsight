#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查数据库中的数据分布"""

import psycopg2

conn = psycopg2.connect(
    host='localhost', port=5432, dbname='test_db', 
    user='postgres', password='123456'
)
conn.autocommit = True
cur = conn.cursor()

tables = [
    'module1_output_orderlog', 
    'module1_output_summary', 
    'module3_output_netdemand', 
    'module4_output_productionplan',
]

print('=' * 60)
print('数据库中的 run_id 和 config_name 分布')
print('=' * 60)

for table in tables:
    try:
        cur.execute(f"""
            SELECT run_id, config_name, COUNT(*) as row_count 
            FROM {table} 
            GROUP BY run_id, config_name
            ORDER BY run_id
        """)
        rows = cur.fetchall()
        print(f"\n{table}:")
        if rows:
            for r in rows:
                print(f"  run_id={r[0]}, config={r[1]}, rows={r[2]}")
        else:
            print("  (empty)")
    except Exception as e:
        print(f"  error: {e}")

# 检查是否有 OC 相关数据
print('\n' + '=' * 60)
print('Check OC data')
print('=' * 60)

try:
    cur.execute("""
        SELECT run_id, config_name, COUNT(*) 
        FROM module1_output_orderlog 
        WHERE config_name LIKE '%OC%' OR run_id LIKE '%OC%'
        GROUP BY run_id, config_name
    """)
    oc_data = cur.fetchall()
    if oc_data:
        for r in oc_data:
            print(f"OC: run_id={r[0]}, config={r[1]}, rows={r[2]}")
    else:
        print("No OC data found")
except Exception as e:
    print(f"Error: {e}")

# 检查 BC 数据
print('\n' + '=' * 60)
print('Check BC data')
print('=' * 60)

try:
    cur.execute("""
        SELECT run_id, config_name, COUNT(*) 
        FROM module1_output_orderlog 
        WHERE config_name LIKE '%BC%' OR run_id LIKE '%BC%'
        GROUP BY run_id, config_name
    """)
    bc_data = cur.fetchall()
    if bc_data:
        for r in bc_data:
            print(f"BC: run_id={r[0]}, config={r[1]}, rows={r[2]}")
    else:
        print("No BC data found")
except Exception as e:
    print(f"Error: {e}")

conn.close()
print("\nDone")
