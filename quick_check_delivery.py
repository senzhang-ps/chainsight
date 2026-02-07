"""Quick check: DeliveryPlan for date 20260109 with various FLOAT_ROUND_DIGITS"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import psycopg

DB_CONN_STR = 'host=localhost port=5432 dbname=test_db user=postgres password=123456'
DB_META_COLS = {'run_id', 'config_name', 'db_write_time'}
LOCATION_COLS = {'location', 'sending', 'receiving', 'sourcing', 'orig_location'}

def _round_float_str(v, digits):
    if '.' in v and v.replace('-','').replace('.','').replace('e','').replace('E','').replace('+','').isdigit():
        try:
            rounded = round(float(v), digits)
            s = str(rounded)
            if s.endswith('.0') and s[:-2].replace('-','').isdigit():
                s = s[:-2]
            return s
        except (ValueError, OverflowError):
            pass
    return v

def normalize_df(df, ref_cols, digits):
    df = df[[c for c in ref_cols if c in df.columns]].copy()
    for c in df.columns:
        df[c] = df[c].fillna('').astype(str)
        df[c] = df[c].apply(lambda v: v[:-2] if v.endswith('.0') and v[:-2].replace('-','').isdigit() else v)
        df[c] = df[c].apply(lambda v: _round_float_str(v, digits))
        df[c] = df[c].str.replace(' 00:00:00', '', regex=False)
        df[c] = df[c].str.strip()
        if c in LOCATION_COLS:
            is_numeric = df[c].str.match(r'^\d+$', na=False)
            df.loc[is_numeric, c] = df.loc[is_numeric, c].str.zfill(4)
    return df

def get_common_cols(df1, df2):
    c1 = set(df1.columns) - DB_META_COLS - {'sim_date'}
    c2 = set(df2.columns) - DB_META_COLS - {'sim_date'}
    return sorted(c1 & c2)

def row_multiset_compare(df1, df2, digits, label=''):
    common_cols = get_common_cols(df1, df2)
    d1 = normalize_df(df1, common_cols, digits)
    d2 = normalize_df(df2, common_cols, digits)
    h1 = d1.apply(lambda r: '|'.join(r.values), axis=1).value_counts()
    h2 = d2.apply(lambda r: '|'.join(r.values), axis=1).value_counts()
    all_keys = set(h1.index) | set(h2.index)
    only1 = sum(max(0, h1.get(k, 0) - h2.get(k, 0)) for k in all_keys)
    only2 = sum(max(0, h2.get(k, 0) - h1.get(k, 0)) for k in all_keys)
    ok = (only1 == 0 and only2 == 0)
    return ok, only1, only2

# Load Dev
dev_path = r'D:\PG\Code\chainsight\config\OC_Paste_S1_20251224\run_20260127_142402\module6\Module6Output_20260109.xlsx'
dev_df = pd.read_excel(dev_path, sheet_name='DeliveryPlan')

# Load DB
conn = psycopg.connect(DB_CONN_STR)
db_df = pd.read_sql("SELECT * FROM module6_output_deliveryplan WHERE sim_date='20260109'", conn)
drop = [c for c in db_df.columns if c in DB_META_COLS or c == 'sim_date']
db_df = db_df.drop(columns=drop, errors='ignore')
conn.close()

print(f"Dev rows: {len(dev_df)}, DB rows: {len(db_df)}")
print(f"Dev cols: {sorted(dev_df.columns.tolist())}")
print(f"DB cols:  {sorted(db_df.columns.tolist())}")
print()

# Test different rounding levels
for digits in [10, 8, 6, 4, 2]:
    ok, only1, only2 = row_multiset_compare(dev_df, db_df, digits)
    status = "PASS" if ok else "FAIL"
    print(f"  digits={digits:2d}: {status}  only_dev={only1} only_db={only2}")

# Investigate what columns differ at digits=8
print("\n--- Investigating columns that differ at digits=8 ---")
common_cols = get_common_cols(dev_df, db_df)
d1_8 = normalize_df(dev_df, common_cols, 8)
d2_8 = normalize_df(db_df, common_cols, 8)
# Find rows in dev not in db
h1 = d1_8.apply(lambda r: '|'.join(r.values), axis=1)
h2 = d2_8.apply(lambda r: '|'.join(r.values), axis=1)
vc1 = h1.value_counts()
vc2 = h2.value_counts()
all_keys = set(vc1.index) | set(vc2.index)
diff_keys = [k for k in all_keys if vc1.get(k, 0) != vc2.get(k, 0)]
print(f"Diff hash-rows: {len(diff_keys)}")

# Show a few diff rows: find the actual cell differences
if diff_keys:
    # Get indices of first few dev-only rows
    dev_only_mask = h1.isin(diff_keys[:3])
    db_only_mask = h2.isin(diff_keys[:3])
    print(f"\nFirst 3 dev-only hash-rows sample (from dev):")
    for idx in d1_8[dev_only_mask].head(3).index:
        print(f"  Row {idx}:")
        for c in common_cols:
            v_dev = d1_8.loc[idx, c]
            # Find corresponding raw values
            v_dev_raw = dev_df.loc[idx, c] if c in dev_df.columns else 'N/A'
            print(f"    {c}: normalized='{v_dev}' raw={v_dev_raw}")
    
    print(f"\nFirst 3 db-only hash-rows sample (from db):")
    for idx in d2_8[db_only_mask].head(3).index:
        print(f"  Row {idx}:")
        for c in common_cols:
            v_db = d2_8.loc[idx, c]
            v_db_raw = db_df.loc[idx, c] if c in db_df.columns else 'N/A'
            print(f"    {c}: normalized='{v_db}' raw={v_db_raw}")
