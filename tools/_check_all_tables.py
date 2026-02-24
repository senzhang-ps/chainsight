import psycopg2

conn = psycopg2.connect(host='localhost', port=5432, dbname='test_db', user='postgres', password='123456')
cur = conn.cursor()
cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name")
tables = [r[0] for r in cur.fetchall()]
print(f'Total tables: {len(tables)}')
for t in tables:
    cur.execute(f'SELECT COUNT(*) FROM "{t}"')
    cnt = cur.fetchone()[0]
    flag = ' *** HAS DATA' if cnt > 0 else ''
    print(f'  {t}: {cnt} rows{flag}')

# Also check other schemas
cur.execute("SELECT schema_name FROM information_schema.schemata")
schemas = [r[0] for r in cur.fetchall()]
print(f'\nAll schemas: {schemas}')

# Check for any table with module or output in name across all schemas
cur.execute("""
    SELECT table_schema, table_name 
    FROM information_schema.tables 
    WHERE (table_name LIKE '%module%' OR table_name LIKE '%output%' OR table_name LIKE '%sim%')
    ORDER BY table_schema, table_name
""")
results = cur.fetchall()
if results:
    print(f'\nTables matching module/output/sim:')
    for schema, tname in results:
        print(f'  {schema}.{tname}')
else:
    print('\nNo tables matching module/output/sim in any schema')

# Also check all databases
cur.execute("SELECT datname FROM pg_database WHERE datistemplate = false")
dbs = [r[0] for r in cur.fetchall()]
print(f'\nAll databases: {dbs}')

conn.close()
