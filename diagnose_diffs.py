"""Deep-dive diagnostics for M3, M4, M6 differences"""
import os, pandas as pd, numpy as np, psycopg

DEV_BASE = r'D:\PG\Code\chainsight\config\OC_Paste_S1_20251224\run_20260127_142402'
DB_META_COLS = {'run_id', 'config_name', 'db_write_time', 'sim_date'}

conn = psycopg.connect('host=localhost port=5432 dbname=test_db user=postgres password=123456')

def load_db(table, sim_date_str):
    df = pd.read_sql(f"SELECT * FROM {table} WHERE sim_date='{sim_date_str}'", conn)
    df = df.drop(columns=[c for c in df.columns if c in DB_META_COLS], errors='ignore')
    return df

# ===== MODULE 3 deep dive (Day 1) =====
print("=" * 80)
print("MODULE 3 NetDemand - Day 1 deep dive")
print("=" * 80)

dev_m3 = pd.read_excel(os.path.join(DEV_BASE, 'module3', 'Module3Output_20251215.xlsx'), sheet_name='NetDemand')
db_m3 = load_db('module3_output_netdemand', '20251215')

print(f"Dev columns: {sorted(dev_m3.columns.tolist())}")
print(f"DB  columns: {sorted(db_m3.columns.tolist())}")
common = sorted(set(dev_m3.columns) & set(db_m3.columns))
print(f"Common cols: {common}")

# Key columns for matching: material, location, requirement_date, demand_element, layer
key_cols = ['material', 'location', 'requirement_date', 'demand_element', 'layer']
# Normalize both
for df in [dev_m3, db_m3]:
    for c in df.columns:
        df[c] = df[c].fillna('').astype(str).str.replace(' 00:00:00', '', regex=False).str.strip()
        # Remove .0 from integers
        df[c] = df[c].apply(lambda v: v[:-2] if v.endswith('.0') and v[:-2].replace('-','').isdigit() else v)

# Make key 
dev_m3['_key'] = dev_m3[key_cols].apply(lambda r: '|'.join(r), axis=1)
db_m3['_key'] = db_m3[key_cols].apply(lambda r: '|'.join(r), axis=1)

# Check how many keys match
dev_keys = set(dev_m3['_key'])
db_keys = set(db_m3['_key'])
both = dev_keys & db_keys
dev_only = dev_keys - db_keys
db_only = db_keys - dev_keys
print(f"\nKey match analysis (material+location+req_date+demand_element+layer):")
print(f"  Dev keys: {len(dev_keys)}, DB keys: {len(db_keys)}")
print(f"  Both: {len(both)}, Dev-only: {len(dev_only)}, DB-only: {len(db_only)}")

# For matching keys, compare quantity with tolerance
if both:
    dev_matched = dev_m3[dev_m3['_key'].isin(both)].set_index('_key')
    db_matched = db_m3[db_m3['_key'].isin(both)].set_index('_key')
    
    # Check for duplicate keys
    dev_dups = dev_matched.index.duplicated().sum()
    db_dups = db_matched.index.duplicated().sum()
    print(f"  Dev duplicate keys: {dev_dups}, DB duplicate keys: {db_dups}")
    
    if dev_dups == 0 and db_dups == 0:
        # Compare quantity
        dev_qty = dev_matched['quantity'].astype(float)
        db_qty = db_matched.loc[dev_qty.index, 'quantity'].astype(float)
        qty_diff = (dev_qty - db_qty).abs()
        print(f"\n  Quantity comparison (matched keys, n={len(dev_qty)}):")
        print(f"    Max abs diff: {qty_diff.max()}")
        print(f"    Mismatches (>1e-6): {(qty_diff > 1e-6).sum()}")
        print(f"    Mismatches (>1e-3): {(qty_diff > 1e-3).sum()}")
        print(f"    Mismatches (>1.0):  {(qty_diff > 1.0).sum()}")
        
        # Check horizon_days
        if 'horizon_days' in common:
            dev_h = dev_matched['horizon_days'].astype(float)
            db_h = db_matched.loc[dev_h.index, 'horizon_days'].astype(float)
            h_diff = (dev_h - db_h).abs()
            print(f"\n  horizon_days comparison:")
            print(f"    Max abs diff: {h_diff.max()}")
            print(f"    Mismatches (>0): {(h_diff > 0).sum()}")

# Show some dev-only and db-only keys
if dev_only:
    print(f"\n  Sample Dev-only keys (first 3):")
    for k in list(dev_only)[:3]:
        row = dev_m3[dev_m3['_key']==k].iloc[0]
        print(f"    {k} -> qty={row['quantity']}")
if db_only:
    print(f"\n  Sample DB-only keys (first 3):")
    for k in list(db_only)[:3]:
        row = db_m3[db_m3['_key']==k].iloc[0]
        print(f"    {k} -> qty={row['quantity']}")

# ===== MODULE 4 deep dive - location format =====
print("\n" + "=" * 80)
print("MODULE 4 ProductionPlan - Day 2 (20251216) deep dive")
print("=" * 80)

dev_m4 = pd.read_excel(os.path.join(DEV_BASE, 'module4', 'Module4Output_20251216.xlsx'), sheet_name='ProductionPlan')
db_m4 = load_db('module4_output_productionplan', '20251216')

print(f"Dev columns: {sorted(dev_m4.columns.tolist())}")
print(f"DB  columns: {sorted(db_m4.columns.tolist())}")
print(f"\nDev location unique: {sorted(dev_m4['location'].astype(str).unique())}")
print(f"DB  location unique: {sorted(db_m4['location'].astype(str).unique())}")
print(f"\nDev line unique: {sorted(dev_m4['line'].astype(str).unique())}")
print(f"DB  line unique: {sorted(db_m4['line'].astype(str).unique())}")

# Check if normalizing location makes a match
dev_m4_n = dev_m4.copy()
db_m4_n = db_m4.copy()
for c in dev_m4_n.columns:
    dev_m4_n[c] = dev_m4_n[c].fillna('').astype(str).str.replace(' 00:00:00', '', regex=False).str.strip()
    dev_m4_n[c] = dev_m4_n[c].apply(lambda v: v[:-2] if v.endswith('.0') and v[:-2].replace('-','').isdigit() else v)
for c in db_m4_n.columns:
    db_m4_n[c] = db_m4_n[c].fillna('').astype(str).str.replace(' 00:00:00', '', regex=False).str.strip()
    db_m4_n[c] = db_m4_n[c].apply(lambda v: v[:-2] if v.endswith('.0') and v[:-2].replace('-','').isdigit() else v)

# Strip leading zeros from location in DB
db_m4_n['location'] = db_m4_n['location'].str.lstrip('0')

common_m4 = sorted(set(dev_m4_n.columns) & set(db_m4_n.columns))
print(f"\nCommon cols: {common_m4}")

# Compare after location normalization
d1 = dev_m4_n[common_m4].apply(lambda r: '|'.join(r), axis=1)
d2 = db_m4_n[common_m4].apply(lambda r: '|'.join(r), axis=1)
h1 = d1.value_counts()
h2 = d2.value_counts()
all_k = set(h1.index) | set(h2.index)
only1 = sum(max(0, h1.get(k,0)-h2.get(k,0)) for k in all_k)
only2 = sum(max(0, h2.get(k,0)-h1.get(k,0)) for k in all_k)
print(f"\nAfter stripping leading zeros from DB location:")
print(f"  Dev-only: {only1}, DB-only: {only2}")

if only1 > 0:
    # Show first diff
    d1_s = set(d1)
    d2_s = set(d2)
    for x in list(d1_s - d2_s)[:2]:
        print(f"  Dev-only: {x}")
    for x in list(d2_s - d1_s)[:2]:
        print(f"  DB-only:  {x}")

# ===== MODULE 6 DeliveryPlan column check =====
print("\n" + "=" * 80)
print("MODULE 6 DeliveryPlan - Column deep dive")
print("=" * 80)

# Day 1 (empty in Dev?)
dev_m6_1 = pd.read_excel(os.path.join(DEV_BASE, 'module6', 'Module6Output_20251215.xlsx'), sheet_name='DeliveryPlan')
db_m6_1 = load_db('module6_output_deliveryplan', '20251215')
print(f"Day1 Dev rows={len(dev_m6_1)}, cols={list(dev_m6_1.columns)}")
print(f"Day1 DB  rows={len(db_m6_1)}, cols={list(db_m6_1.columns)}")

# Day 2
dev_m6_2 = pd.read_excel(os.path.join(DEV_BASE, 'module6', 'Module6Output_20251216.xlsx'), sheet_name='DeliveryPlan')
db_m6_2 = load_db('module6_output_deliveryplan', '20251216')
print(f"\nDay2 Dev rows={len(dev_m6_2)}, cols={list(dev_m6_2.columns)}")
print(f"Day2 DB  rows={len(db_m6_2)}, cols={list(db_m6_2.columns)}")
# Check if Dev columns match DB columns after rename
print(f"\nDev-only cols: {set(dev_m6_2.columns) - set(db_m6_2.columns)}")
print(f"DB-only  cols: {set(db_m6_2.columns) - set(dev_m6_2.columns)}")

conn.close()
print("\nDone!")
