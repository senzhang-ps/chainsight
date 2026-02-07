import psycopg
conn = psycopg.connect('host=localhost port=5432 dbname=test_db user=postgres password=123456')
cur = conn.cursor()
output_tables = []
cur.execute("SELECT tablename FROM pg_tables WHERE schemaname='public'")
for row in cur.fetchall():
    t = row[0]
    if t.startswith('module') or t.startswith('orchestrator') or t.startswith('summary') or t.startswith('analysis'):
        output_tables.append(t)
for t in sorted(output_tables):
    cur.execute(f'TRUNCATE TABLE "{t}"')
    print(f'truncated: {t}')
conn.commit()
conn.close()
print(f'\nTruncated {len(output_tables)} output tables')
