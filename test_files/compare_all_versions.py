# -*- coding: utf-8 -*-
"""
Full comparison of all 3 versions:
1. ChainSight_Dev source output (golden reference)
2. Refactored local mode output  
3. Refactored DB mode output (from PostgreSQL)
"""

import pandas as pd
import psycopg
from pathlib import Path

# Paths
DEV_OUTPUT = Path(r"C:\Users\25936\Desktop\Code\chainsight\ChainSight_Dev\BC_S5\run_20260129_204512")
LOCAL_OUTPUT = Path(r"C:\Users\25936\Desktop\Code\chainsight\outputs\BC_S5\run_20260130_125705")

# DB config
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'dbname': 'test_db',
    'user': 'postgres',
    'password': '123456'
}

# Latest DB run
DB_RUN_ID = "BC_S5_20260130_130233"

DATES = ['2025-10-06', '2025-10-07', '2025-10-08', '2025-10-09', '2025-10-10']

def connect_db():
    conn_str = f"host={DB_CONFIG['host']} port={DB_CONFIG['port']} dbname={DB_CONFIG['dbname']} user={DB_CONFIG['user']} password={DB_CONFIG['password']}"
    return psycopg.connect(conn_str)

def get_db_module5_unfulfilled_counts():
    """Get UnfulfilledLog counts from DB by date"""
    counts = {}
    with connect_db() as conn:
        for d in DATES:
            query = """
                SELECT COUNT(*) as cnt 
                FROM module5_output_unfulfilledlog 
                WHERE run_id = %s AND sim_date::date = %s::date
            """
            try:
                df = pd.read_sql(query, conn, params=[DB_RUN_ID, d])
                counts[d] = df['cnt'].iloc[0] if not df.empty else 0
            except Exception as e:
                print(f"DB Error for {d}: {e}")
                counts[d] = 'N/A'
    return counts

def get_db_module4_production_counts():
    """Get ProductionPlan counts from DB by date"""
    counts = {}
    with connect_db() as conn:
        for d in DATES:
            query = """
                SELECT COUNT(*) as cnt 
                FROM module4_output_productionplan 
                WHERE run_id = %s AND sim_date::date = %s::date
            """
            try:
                df = pd.read_sql(query, conn, params=[DB_RUN_ID, d])
                counts[d] = df['cnt'].iloc[0] if not df.empty else 0
            except Exception as e:
                print(f"DB Error for {d}: {e}")
                counts[d] = 'N/A'
    return counts

def get_db_module6_delivery_counts():
    """Get DeliveryPlan counts from DB by date"""
    counts = {}
    with connect_db() as conn:
        for d in DATES:
            query = """
                SELECT COUNT(*) as cnt 
                FROM module6_output_deliveryplan 
                WHERE run_id = %s AND sim_date::date = %s::date
            """
            try:
                df = pd.read_sql(query, conn, params=[DB_RUN_ID, d])
                counts[d] = df['cnt'].iloc[0] if not df.empty else 0
            except Exception as e:
                print(f"DB Error for {d}: {e}")
                counts[d] = 'N/A'
    return counts

def compare_module5_unfulfilled():
    """Compare UnfulfilledLog counts across all 3 versions"""
    print("\n" + "="*80)
    print("Module5 UnfulfilledLog Comparison (Row Counts)")
    print("="*80)
    print(f"{'Date':<12} | {'Dev':>8} | {'Local':>8} | {'DB':>8} | {'Dev==Local':>10} | {'Dev==DB':>10}")
    print("-"*80)
    
    # Get DB counts
    db_counts = get_db_module5_unfulfilled_counts()
    
    for d in DATES:
        d_str = d.replace('-', '')
        
        # Dev version
        dev_file = DEV_OUTPUT / 'module5' / f'Module5Output_{d_str}.xlsx'
        if dev_file.exists():
            dev_df = pd.read_excel(dev_file, sheet_name='UnfulfilledLog')
            dev_cnt = len(dev_df)
        else:
            dev_cnt = 'N/A'
        
        # Local version  
        local_file = LOCAL_OUTPUT / 'module5' / f'Module5Output_{d_str}.xlsx'
        if local_file.exists():
            local_df = pd.read_excel(local_file, sheet_name='UnfulfilledLog')
            local_cnt = len(local_df)
        else:
            local_cnt = 'N/A'
        
        # DB version
        db_cnt = db_counts.get(d, 'N/A')
        
        match_local = '✅ YES' if dev_cnt == local_cnt else '❌ NO'
        match_db = '✅ YES' if dev_cnt == db_cnt else '❌ NO'
        
        print(f"{d:<12} | {str(dev_cnt):>8} | {str(local_cnt):>8} | {str(db_cnt):>8} | {match_local:>10} | {match_db:>10}")

def compare_module4_production():
    """Compare Module4 ProductionPlan counts across all 3 versions"""
    print("\n" + "="*80)
    print("Module4 ProductionPlan Comparison (Row Counts)")
    print("="*80)
    print(f"{'Date':<12} | {'Dev':>8} | {'Local':>8} | {'DB':>8} | {'Dev==Local':>10} | {'Dev==DB':>10}")
    print("-"*80)
    
    # Get DB counts
    db_counts = get_db_module4_production_counts()
    
    for d in DATES:
        d_str = d.replace('-', '')
        
        # Dev version
        dev_file = DEV_OUTPUT / 'module4' / f'Module4Output_{d_str}.xlsx'
        if dev_file.exists():
            xl = pd.ExcelFile(dev_file)
            if 'ProductionPlan' in xl.sheet_names:
                dev_df = pd.read_excel(dev_file, sheet_name='ProductionPlan')
                dev_cnt = len(dev_df)
            else:
                dev_cnt = 0
        else:
            dev_cnt = 'N/A'
        
        # Local version  
        local_file = LOCAL_OUTPUT / 'module4' / f'Module4Output_{d_str}.xlsx'
        if local_file.exists():
            xl = pd.ExcelFile(local_file)
            if 'ProductionPlan' in xl.sheet_names:
                local_df = pd.read_excel(local_file, sheet_name='ProductionPlan')
                local_cnt = len(local_df)
            else:
                local_cnt = 0
        else:
            local_cnt = 'N/A'
        
        # DB version
        db_cnt = db_counts.get(d, 'N/A')
        
        match_local = '✅ YES' if dev_cnt == local_cnt else '❌ NO'
        match_db = '✅ YES' if dev_cnt == db_cnt else '❌ NO'
        
        print(f"{d:<12} | {str(dev_cnt):>8} | {str(local_cnt):>8} | {str(db_cnt):>8} | {match_local:>10} | {match_db:>10}")

def compare_module6_delivery():
    """Compare Module6 DeliveryPlan counts across all 3 versions"""
    print("\n" + "="*80)
    print("Module6 DeliveryPlan Comparison (Row Counts)")
    print("="*80)
    print(f"{'Date':<12} | {'Dev':>8} | {'Local':>8} | {'DB':>8} | {'Dev==Local':>10} | {'Dev==DB':>10}")
    print("-"*80)
    
    # Get DB counts
    db_counts = get_db_module6_delivery_counts()
    
    for d in DATES:
        d_str = d.replace('-', '')
        
        # Dev version
        dev_file = DEV_OUTPUT / 'module6' / f'Module6Output_{d_str}.xlsx'
        if dev_file.exists():
            xl = pd.ExcelFile(dev_file)
            if 'DeliveryPlan' in xl.sheet_names:
                dev_df = pd.read_excel(dev_file, sheet_name='DeliveryPlan')
                dev_cnt = len(dev_df)
            else:
                dev_cnt = 0
        else:
            dev_cnt = 'N/A'
        
        # Local version  
        local_file = LOCAL_OUTPUT / 'module6' / f'Module6Output_{d_str}.xlsx'
        if local_file.exists():
            xl = pd.ExcelFile(local_file)
            if 'DeliveryPlan' in xl.sheet_names:
                local_df = pd.read_excel(local_file, sheet_name='DeliveryPlan')
                local_cnt = len(local_df)
            else:
                local_cnt = 0
        else:
            local_cnt = 'N/A'
        
        # DB version
        db_cnt = db_counts.get(d, 'N/A')
        
        match_local = '✅ YES' if dev_cnt == local_cnt else '❌ NO'
        match_db = '✅ YES' if dev_cnt == db_cnt else '❌ NO'
        
        print(f"{d:<12} | {str(dev_cnt):>8} | {str(local_cnt):>8} | {str(db_cnt):>8} | {match_local:>10} | {match_db:>10}")

def main():
    print("="*80)
    print("ChainSight Full Version Comparison")
    print("="*80)
    print(f"Dev output:   {DEV_OUTPUT}")
    print(f"Local output: {LOCAL_OUTPUT}")
    print(f"DB run_id:    {DB_RUN_ID}")
    
    compare_module4_production()
    compare_module5_unfulfilled()
    compare_module6_delivery()
    
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    print("Expected values from ChainSight_Dev (golden reference):")
    print("  Module4 ProductionPlan: [0, 64, 0, 0, 0]")
    print("  Module5 UnfulfilledLog: [9128, 8300, 9716, 10039, 10202]")
    print("  Module6 DeliveryPlan:   [0, 103, 52, 391, 91]")

if __name__ == '__main__':
    main()
