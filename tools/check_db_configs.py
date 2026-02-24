# -*- coding: utf-8 -*-
"""Check all cfg_ tables in DB and their config_name values."""
import psycopg2

conn = psycopg2.connect(host='localhost', port=5432, dbname='test_db',
                        user='postgres', password='123456')
cur = conn.cursor()

# Find all cfg_ tables
cur.execute("SELECT tablename FROM pg_tables WHERE schemaname='public' AND tablename LIKE 'cfg_%%' ORDER BY tablename")
tables = cur.fetchall()

print('Config tables in DB:')
for t in tables:
    tname = t[0]
    try:
        cur.execute(f"SELECT DISTINCT config_name FROM {tname} WHERE config_name IS NOT NULL")
        configs = [r[0] for r in cur.fetchall()]
    except Exception:
        conn.rollback()
        configs = ['(no config_name col)']
    try:
        cur.execute(f"SELECT count(*) FROM {tname}")
        cnt = cur.fetchone()[0]
    except Exception:
        conn.rollback()
        cnt = '?'
    print(f'  {tname}: {cnt} rows, configs={configs}')

conn.close()
