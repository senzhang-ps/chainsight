"""Check all tables and find module output data in DB"""
import psycopg2

conn = psycopg2.connect(host='localhost', port=5432, dbname='test_db',
                        user='postgres', password='123456')
cur = conn.cursor()

# List all tables
cur.execute("""
    SELECT table_name FROM information_schema.tables 
    WHERE table_schema='public' ORDER BY table_name
""")
tables = [r[0] for r in cur.fetchall()]
print(f"Total tables: {len(tables)}")

# Find module output tables (not cfg_*)
module_tables = [t for t in tables if not t.startswith('cfg_')]
print(f"\nNon-config tables ({len(module_tables)}):")
for t in module_tables:
    cur.execute(f'SELECT COUNT(*) FROM "{t}"')
    cnt = cur.fetchone()[0]
    # check columns
    cur.execute(f"""SELECT column_name FROM information_schema.columns 
                    WHERE table_name='{t}' ORDER BY ordinal_position""")
    cols = [r[0] for r in cur.fetchall()]
    print(f"  {t}: {cnt} rows, cols={cols[:8]}{'...' if len(cols)>8 else ''}")

# Also check if any cfg table has module output data
print(f"\nConfig tables ({len(tables) - len(module_tables)}):")
for t in tables:
    if t.startswith('cfg_'):
        continue
    # skip, already printed

# Check for run_id in any table  
print("\n--- Tables with run_id column ---")
for t in tables:
    cur.execute("""SELECT column_name FROM information_schema.columns 
                   WHERE table_name=%s AND column_name='run_id'""", (t,))
    if cur.fetchone():
        cur.execute(f'SELECT DISTINCT run_id FROM "{t}" LIMIT 5')
        run_ids = [r[0] for r in cur.fetchall()]
        print(f"  {t}: run_ids={run_ids}")

# Check for sim_date in any table
print("\n--- Tables with sim_date column ---")
for t in tables:
    cur.execute("""SELECT column_name FROM information_schema.columns 
                   WHERE table_name=%s AND column_name='sim_date'""", (t,))
    if cur.fetchone():
        try:
            cur.execute(f'SELECT COUNT(*), MIN(sim_date), MAX(sim_date) FROM "{t}"')
            r = cur.fetchone()
            print(f"  {t}: {r[0]} rows, {r[1]} ~ {r[2]}")
        except:
            conn.rollback()

conn.close()
