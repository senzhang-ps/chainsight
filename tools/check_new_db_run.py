"""Check all tables in the DB and find where run data is stored"""
import psycopg2

conn = psycopg2.connect(
    host='localhost', port=5432, dbname='test_db',
    user='postgres', password='123456'
)
cur = conn.cursor()

run_id = 'BC_S5_20260212_110001'

# List all tables
cur.execute("""
    SELECT table_name FROM information_schema.tables 
    WHERE table_schema='public' 
    ORDER BY table_name
""")
tables = [r[0] for r in cur.fetchall()]
print(f"Total tables: {len(tables)}")
for t in tables:
    print(f"  {t}")

print("\n--- Checking for run_id columns ---")
for t in tables:
    try:
        cur.execute("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name=%s AND column_name='run_id'
        """, (t,))
        if cur.fetchone():
            cur.execute(f"SELECT COUNT(*) FROM \"{t}\" WHERE run_id=%s", (run_id,))
            cnt = cur.fetchone()[0]
            print(f"  {t}: {cnt} rows with run_id={run_id}")
    except Exception as e:
        conn.rollback()
        print(f"  {t}: error - {e}")

print("\n--- Checking tables with sim_date column ---")
for t in tables:
    try:
        cur.execute("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_name=%s AND column_name='sim_date'
        """, (t,))
        if cur.fetchone():
            cur.execute(f"SELECT COUNT(*) FROM \"{t}\"")
            cnt = cur.fetchone()[0]
            if cnt > 0:
                cur.execute(f"SELECT MIN(sim_date), MAX(sim_date) FROM \"{t}\"")
                r = cur.fetchone()
                print(f"  {t}: {cnt} rows, dates {r[0]} to {r[1]}")
    except Exception as e:
        conn.rollback()
        print(f"  {t}: error - {e}")

conn.close()
