# -*- coding: utf-8 -*-
"""
直接验证 module5/DeploymentPlan 的 Dev vs DB 差异
绕过比对脚本，直接读取原始 Dev xlsx 和 DB 数据进行对比。
重点关注 deploy_qty_with_plan_order 和 deploy_from_future_production 两列。
"""
import pandas as pd
import psycopg2
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DEV_DIR = PROJECT_ROOT / "ChainSight_Dev" / "BC_S5" / "run_20260211_181635"
SRC_DIR = PROJECT_ROOT / "outputs" / "BC_S5" / "run_20260211_145215"
DB_CONN = dict(host="localhost", port=5432, dbname="test_db",
               user="postgres", password="123456", client_encoding="utf8")
DB_RUN_ID = "BC_S5_20260211_145215"

# 已知有差异的4天：Day12, Day17, Day22, Day82
# Day1 = 2025-10-05, so Day12 = 2025-10-16, Day17 = 2025-10-21, Day22 = 2025-10-26, Day82 = 2025-12-25
DIFF_DAYS = {
    12: "20251016",
    17: "20251021",
    22: "20251026",
    82: "20251225",
}

# 也测几个无差异的天作为对照
CONTROL_DAYS = {
    1: "20251005",
    5: "20251009",
    30: "20251103",
    50: "20251123",
}

COLS_OF_INTEREST = ["deploy_qty_with_plan_order", "deploy_from_future_production"]
KEY_COLS = ["date", "material", "sending", "receiving", "demand_element"]

conn = psycopg2.connect(**DB_CONN)

def load_dev(date_str):
    f = DEV_DIR / "module5" / f"Module5Output_{date_str}.xlsx"
    if not f.exists():
        return None
    return pd.read_excel(f, sheet_name="DeploymentPlan", engine="openpyxl")

def load_src(date_str):
    f = SRC_DIR / "module5" / f"Module5Output_{date_str}.xlsx"
    if not f.exists():
        return None
    return pd.read_excel(f, sheet_name="DeploymentPlan", engine="openpyxl")

def load_db(date_str):
    q = "SELECT * FROM module5_output_deploymentplan WHERE sim_date = %s AND run_id = %s"
    df = pd.read_sql(q, conn, params=[date_str, DB_RUN_ID])
    drop = [c for c in df.columns if c in {"sim_date", "run_id", "config_name", "db_write_time"}]
    df = df.drop(columns=drop, errors="ignore")
    # Rename DB columns to match xlsx column names
    df = df.rename(columns={"deployed_qty_invcon": "deployed_qty_invCon",
                            "deployed_qty_invcon_push": "deployed_qty_invCon_push"})
    return df

print("=" * 80)
print("直接验证 DeploymentPlan: Dev xlsx vs DB 原始数据")
print("=" * 80)

all_days = {**DIFF_DAYS, **CONTROL_DAYS}
for day_num in sorted(all_days.keys()):
    date_str = all_days[day_num]
    label = "DIFF_DAY" if day_num in DIFF_DAYS else "CONTROL"
    
    dev_df = load_dev(date_str)
    src_df = load_src(date_str)
    db_df = load_db(date_str)
    
    if dev_df is None:
        print(f"Day{day_num:02d} ({date_str}) [{label}]: Dev xlsx 不存在")
        continue
    if db_df is None or db_df.empty:
        print(f"Day{day_num:02d} ({date_str}) [{label}]: DB 数据为空")
        continue
    
    print(f"\nDay{day_num:02d} ({date_str}) [{label}]:")
    print(f"  行数: Dev={len(dev_df)}, Src={len(src_df) if src_df is not None else 'N/A'}, DB={len(db_df)}")
    
    # 检查两个关注列在各版本中的值
    for col in COLS_OF_INTEREST:
        dev_vals = dev_df[col] if col in dev_df.columns else None
        db_vals = db_df[col] if col in db_df.columns else None
        src_vals = src_df[col] if src_df is not None and col in src_df.columns else None
        
        if dev_vals is not None and db_vals is not None:
            dev_nonzero = (dev_vals.fillna(0) != 0).sum()
            db_nonzero = (db_vals.fillna(0) != 0).sum()
            src_nonzero = (src_vals.fillna(0) != 0).sum() if src_vals is not None else "N/A"
            
            print(f"  {col}:")
            print(f"    Dev 非零行数: {dev_nonzero}/{len(dev_vals)}")
            print(f"    Src 非零行数: {src_nonzero}")
            print(f"    DB  非零行数: {db_nonzero}/{len(db_vals)}")
            
            # 直接比较 Dev vs DB (按行，先排序对齐)
            # 用 key_cols merge
            avail_keys = [c for c in KEY_COLS if c in dev_df.columns and c in db_df.columns]
            
            if dev_nonzero != db_nonzero:
                # 看看具体值
                if db_nonzero > 0:
                    db_nz = db_df[db_df[col].fillna(0) != 0][[*avail_keys, col]].head(5)
                    print(f"    DB 非零样本:\n{db_nz.to_string(index=False)}")
                if dev_nonzero > 0:
                    dev_nz = dev_df[dev_df[col].fillna(0) != 0][[*avail_keys, col]].head(5)
                    print(f"    Dev 非零样本:\n{dev_nz.to_string(index=False)}")

    # 全量 merge 比较这两列
    avail_keys = [c for c in KEY_COLS if c in dev_df.columns and c in db_df.columns]
    
    # Normalize key cols for merge
    dev_m = dev_df[avail_keys + COLS_OF_INTEREST].copy()
    db_m = db_df[avail_keys + COLS_OF_INTEREST].copy()
    for k in avail_keys:
        dev_m[k] = dev_m[k].astype(str).str.strip()
        db_m[k] = db_m[k].astype(str).str.strip()
    
    # Sort and add dup index
    sort_cols = avail_keys + COLS_OF_INTEREST
    dev_m = dev_m.sort_values(sort_cols).reset_index(drop=True)
    db_m = db_m.sort_values(sort_cols).reset_index(drop=True)
    dev_m["_idx"] = dev_m.groupby(avail_keys).cumcount()
    db_m["_idx"] = db_m.groupby(avail_keys).cumcount()
    
    merged = dev_m.merge(db_m, on=avail_keys + ["_idx"], how="outer",
                         suffixes=("_dev", "_db"), indicator=True)
    
    both = merged[merged["_merge"] == "both"]
    total_diff = 0
    for col in COLS_OF_INTEREST:
        c_dev = f"{col}_dev"
        c_db = f"{col}_db"
        v_dev = both[c_dev].fillna(0).astype(float)
        v_db = both[c_db].fillna(0).astype(float)
        ndiff = (v_dev != v_db).sum()
        total_diff += ndiff
    
    only_dev = (merged["_merge"] == "left_only").sum()
    only_db = (merged["_merge"] == "right_only").sum()
    
    print(f"  Merge结果: both={len(both)}, only_dev={only_dev}, only_db={only_db}")
    print(f"  这两列的差异行数: {total_diff}")

conn.close()
print("\n" + "=" * 80)
print("验证完成")
