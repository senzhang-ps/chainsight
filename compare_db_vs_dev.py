# -*- coding: utf-8 -*-
"""
数据库版本与Dev版本输出对比验证脚本
比较数据库中的数据与Dev版本文件输出的一致性
"""
import sys
import io
import os

# Set UTF-8 encoding for stdout/stderr
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Dev版本输出路径
DEV_OUTPUT_DIR = Path(r"C:\Users\25936\Desktop\Code\chainsight\ChainSight_Dev\BC_S5\run_20260127_192732")

# 数据库连接配置
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "test_db",
    "user": "postgres",
    "password": "123456"
}

# DB run_id to compare (latest run)
DB_RUN_ID = "BC_S5_20260127_202915"


def get_db_connection():
    """获取数据库连接"""
    from pgsql_db.db_connection import DatabaseConnection
    return DatabaseConnection(**DB_CONFIG)


def query_db_table(db, table_name, run_id=None):
    """从数据库查询表数据，过滤指定run_id"""
    if run_id is None:
        run_id = DB_RUN_ID
    
    try:
        df = db.read_table(table_name)
        if df is not None and not df.empty:
            # Filter by run_id if column exists
            if 'run_id' in df.columns and run_id:
                df = df[df['run_id'] == run_id]
            return df
    except Exception as e:
        print(f"    [警告] 读取表 {table_name} 失败: {e}")
    
    return pd.DataFrame()


def load_dev_inventory(output_dir):
    """加载Dev版本的最终库存数据"""
    orch_dir = output_dir / "orchestrator"
    
    # 查找最后一天的库存文件
    inv_files = sorted(orch_dir.glob("unrestricted_inventory_*.csv"))
    if inv_files:
        last_file = inv_files[-1]
        df = pd.read_csv(last_file)
        return df
    return pd.DataFrame()


def load_dev_open_deployment(output_dir):
    """加载Dev版本的open deployment数据"""
    orch_dir = output_dir / "orchestrator"
    
    # 查找最后一天的文件 (排除pastdue_cleanup文件)
    files = sorted([f for f in orch_dir.glob("open_deployment_*.csv") 
                   if 'pastdue_cleanup' not in f.name])
    if files:
        last_file = files[-1]
        df = pd.read_csv(last_file)
        return df
    return pd.DataFrame()


def load_dev_intransit(output_dir):
    """加载Dev版本的in-transit数据"""
    orch_dir = output_dir / "orchestrator"
    
    # 查找最后一天的文件
    files = sorted(orch_dir.glob("planning_intransit_*.csv"))
    if files:
        last_file = files[-1]
        df = pd.read_csv(last_file)
        return df
    return pd.DataFrame()


def compare_dataframes(df1, df2, name, key_cols=None):
    """比较两个DataFrame"""
    result = {
        "name": name,
        "df1_rows": len(df1),
        "df2_rows": len(df2),
        "match": False,
        "details": []
    }
    
    # 行数比较
    if len(df1) != len(df2):
        result["details"].append(f"行数不一致: Dev={len(df1)}, DB={len(df2)}")
    else:
        result["details"].append(f"行数一致: {len(df1)}")
    
    # 如果都为空，认为匹配
    if df1.empty and df2.empty:
        result["match"] = True
        result["details"].append("两者均为空")
        return result
    
    # 数值列求和比较
    numeric_cols_1 = df1.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols_2 = df2.select_dtypes(include=[np.number]).columns.tolist()
    
    common_numeric = set(numeric_cols_1) & set(numeric_cols_2)
    
    # 排除一些不需要比较的列
    exclude_cols = {'file_date', 'sim_date', 'run_id', 'index', 'Unnamed: 0'}
    common_numeric = common_numeric - exclude_cols
    
    all_match = True
    for col in sorted(common_numeric):
        try:
            sum1 = df1[col].fillna(0).sum()
            sum2 = df2[col].fillna(0).sum()
            
            # 使用相对误差比较
            if abs(sum1) < 1e-10 and abs(sum2) < 1e-10:
                match = True
            elif abs(sum1) < 1e-10 or abs(sum2) < 1e-10:
                match = abs(sum1 - sum2) < 1e-6
            else:
                rel_diff = abs(sum1 - sum2) / max(abs(sum1), abs(sum2))
                match = rel_diff < 1e-6
            
            if match:
                result["details"].append(f"  {col}: {sum1:.2f} ✓")
            else:
                result["details"].append(f"  {col}: Dev={sum1:.2f}, DB={sum2:.2f} ✗")
                all_match = False
        except Exception as e:
            result["details"].append(f"  {col}: 比较失败 - {e}")
            all_match = False
    
    if len(df1) == len(df2) and all_match:
        result["match"] = True
    
    return result


def main():
    print("=" * 70)
    print("数据库版本 vs Dev版本 输出对比验证")
    print("=" * 70)
    print(f"Dev输出目录: {DEV_OUTPUT_DIR}")
    print(f"数据库: {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
    print(f"DB Run ID: {DB_RUN_ID}")
    print("=" * 70)
    
    # 连接数据库
    db = get_db_connection()
    conn_result = db.test_connection()
    if not conn_result["success"]:
        print(f"❌ 数据库连接失败: {conn_result['message']}")
        return 1
    
    print("✅ 数据库连接成功")
    print()
    
    results = []
    
    # ==================== 1. 最终库存对比 ====================
    print("=" * 50)
    print("1. 最终库存对比 (Unrestricted Inventory)")
    print("=" * 50)
    
    # Dev版本
    dev_inv = load_dev_inventory(DEV_OUTPUT_DIR)
    print(f"Dev版本: {len(dev_inv)} 条记录")
    
    # 数据库版本 - 查询最后一天的数据
    db_inv = query_db_table(db, "orchestrator_unrestricted_inventory")
    if not db_inv.empty:
        # 获取最后一天的数据
        if 'file_date' in db_inv.columns:
            last_date = db_inv['file_date'].max()
            db_inv = db_inv[db_inv['file_date'] == last_date]
    print(f"DB版本: {len(db_inv)} 条记录")
    
    if not dev_inv.empty and not db_inv.empty:
        # 比较关键指标
        dev_qty = dev_inv['quantity'].sum() if 'quantity' in dev_inv.columns else 0
        db_qty = db_inv['quantity'].sum() if 'quantity' in db_inv.columns else 0
        
        print(f"\n库存数量总和:")
        print(f"  Dev版本: {dev_qty:,.0f}")
        print(f"  DB版本:  {db_qty:,.0f}")
        
        if abs(dev_qty - db_qty) < 1:
            print(f"  ✅ 匹配!")
            results.append(("最终库存数量", "✅ 匹配", dev_qty, db_qty))
        else:
            print(f"  ❌ 不匹配! 差异: {dev_qty - db_qty:,.0f}")
            results.append(("最终库存数量", "❌ 不匹配", dev_qty, db_qty))
        
        print(f"\n库存条目数:")
        print(f"  Dev版本: {len(dev_inv)}")
        print(f"  DB版本:  {len(db_inv)}")
        
        if len(dev_inv) == len(db_inv):
            print(f"  ✅ 匹配!")
            results.append(("库存条目数", "✅ 匹配", len(dev_inv), len(db_inv)))
        else:
            print(f"  ❌ 不匹配!")
            results.append(("库存条目数", "❌ 不匹配", len(dev_inv), len(db_inv)))
    
    # ==================== 2. Open Deployment对比 ====================
    print("\n" + "=" * 50)
    print("2. Open Deployment对比")
    print("=" * 50)
    
    dev_od = load_dev_open_deployment(DEV_OUTPUT_DIR)
    print(f"Dev版本: {len(dev_od)} 条记录")
    
    db_od = query_db_table(db, "orchestrator_open_deployment")
    if not db_od.empty and 'file_date' in db_od.columns:
        last_date = db_od['file_date'].max()
        db_od = db_od[db_od['file_date'] == last_date]
    print(f"DB版本: {len(db_od)} 条记录")
    
    if len(dev_od) == len(db_od):
        print(f"  ✅ 条目数匹配!")
        results.append(("Open Deployment数量", "✅ 匹配", len(dev_od), len(db_od)))
    else:
        print(f"  ❌ 条目数不匹配!")
        results.append(("Open Deployment数量", "❌ 不匹配", len(dev_od), len(db_od)))
    
    # ==================== 3. In-Transit对比 ====================
    print("\n" + "=" * 50)
    print("3. In-Transit对比")
    print("=" * 50)
    
    dev_it = load_dev_intransit(DEV_OUTPUT_DIR)
    print(f"Dev版本: {len(dev_it)} 条记录")
    
    db_it = query_db_table(db, "orchestrator_planning_intransit")
    if not db_it.empty and 'file_date' in db_it.columns:
        last_date = db_it['file_date'].max()
        db_it = db_it[db_it['file_date'] == last_date]
    print(f"DB版本: {len(db_it)} 条记录")
    
    if len(dev_it) == len(db_it):
        print(f"  ✅ 条目数匹配!")
        results.append(("In-Transit数量", "✅ 匹配", len(dev_it), len(db_it)))
    else:
        print(f"  ❌ 条目数不匹配!")
        results.append(("In-Transit数量", "❌ 不匹配", len(dev_it), len(db_it)))
    
    # ==================== 4. Summary对比 ====================
    print("\n" + "=" * 50)
    print("4. Summary报告对比")
    print("=" * 50)
    
    # Dev版本Summary
    dev_summary_dir = DEV_OUTPUT_DIR / "summary"
    summary_files = [
        ("full_deployment_plan_report.xlsx", "summary_output_fulldeploymentplan"),
        ("full_delivery_plan_report.xlsx", "summary_output_fulldeliveryplan"),
        ("full_production_plan_report.xlsx", "summary_output_fullproductionplan"),
        ("historical_inventory_record.csv", "summary_historical_inventory_record"),
    ]
    
    for dev_file, db_table in summary_files:
        dev_path = dev_summary_dir / dev_file
        if dev_path.exists():
            if dev_file.endswith('.xlsx'):
                dev_df = pd.read_excel(dev_path)
            else:
                dev_df = pd.read_csv(dev_path)
            
            db_df = query_db_table(db, db_table)
            
            print(f"\n{dev_file}:")
            print(f"  Dev版本: {len(dev_df)} 条记录")
            print(f"  DB版本:  {len(db_df)} 条记录")
            
            if len(dev_df) == len(db_df):
                print(f"  ✅ 条目数匹配!")
                results.append((dev_file, "✅ 匹配", len(dev_df), len(db_df)))
            else:
                print(f"  ❌ 条目数不匹配!")
                results.append((dev_file, "❌ 不匹配", len(dev_df), len(db_df)))
    
    # ==================== 汇总报告 ====================
    print("\n" + "=" * 70)
    print("验证结果汇总")
    print("=" * 70)
    print(f"{'指标':<40} {'状态':<12} {'Dev':<15} {'DB':<15}")
    print("-" * 70)
    
    all_match = True
    for name, status, dev_val, db_val in results:
        if isinstance(dev_val, float):
            print(f"{name:<40} {status:<12} {dev_val:>15,.0f} {db_val:>15,.0f}")
        else:
            print(f"{name:<40} {status:<12} {dev_val:>15} {db_val:>15}")
        if "❌" in status:
            all_match = False
    
    print("-" * 70)
    if all_match:
        print("✅ 所有指标匹配! 数据库版本与Dev版本输出一致!")
    else:
        print("❌ 存在不匹配项，请检查!")
    print("=" * 70)
    
    db.close()
    return 0 if all_match else 1


if __name__ == "__main__":
    sys.exit(main())
