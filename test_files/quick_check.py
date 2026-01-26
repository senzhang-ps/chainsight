# 快速检查数据库表统计
import psycopg2
conn = psycopg2.connect('postgresql://postgres:123456@localhost:5432/test_db')
cur = conn.cursor()

# 检查unrestricted_inventory列名
cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'orchestrator_unrestricted_inventory'")
print("orchestrator_unrestricted_inventory columns:")
for row in cur.fetchall():
    print(f"  {row[0]}")

# 检查总量
cur.execute("SELECT * FROM orchestrator_unrestricted_inventory LIMIT 1")
print("\n示例数据:")
cols = [desc[0] for desc in cur.description]
print(f"  列: {cols}")
row = cur.fetchone()
print(f"  值: {row}")

conn.close()
