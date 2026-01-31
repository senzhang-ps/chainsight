# -*- coding: utf-8 -*-
"""
Full comparison of 3 versions:
1. ChainSight_Dev source output (golden reference)
2. Refactored local mode output
3. Refactored DB mode output
"""

import pandas as pd
from pathlib import Path
import sys

# Paths
DEV_OUTPUT = Path(r"C:\Users\25936\Desktop\Code\chainsight\ChainSight_Dev\BC_S5\run_20260129_204512")
LOCAL_OUTPUT = Path(r"C:\Users\25936\Desktop\Code\chainsight\outputs\BC_S5\run_20260130_125705")

DATES = ['2025-10-06', '2025-10-07', '2025-10-08', '2025-10-09', '2025-10-10']

def compare_module5_unfulfilled():
    """Compare UnfulfilledLog counts across versions"""
    print("\n" + "="*60)
    print("Module5 UnfulfilledLog Comparison")
    print("="*60)
    print(f"{'Date':<12} | {'Dev':>8} | {'Local':>8} | {'Dev==Local':>10}")
    print("-"*60)
    
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
        
        match = 'YES' if dev_cnt == local_cnt else 'NO'
        print(f"{d:<12} | {str(dev_cnt):>8} | {str(local_cnt):>8} | {match:>10}")

def compare_module4_production():
    """Compare Module4 ProductionPlan across versions"""
    print("\n" + "="*60)
    print("Module4 ProductionPlan Comparison")
    print("="*60)
    print(f"{'Date':<12} | {'Dev':>8} | {'Local':>8} | {'Dev==Local':>10}")
    print("-"*60)
    
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
        
        match = 'YES' if dev_cnt == local_cnt else 'NO'
        print(f"{d:<12} | {str(dev_cnt):>8} | {str(local_cnt):>8} | {match:>10}")

def compare_module1_orders():
    """Compare Module1 Orders across versions"""
    print("\n" + "="*60)
    print("Module1 Orders Comparison")
    print("="*60)
    print(f"{'Date':<12} | {'Dev':>8} | {'Local':>8} | {'Dev==Local':>10}")
    print("-"*60)
    
    for d in DATES:
        d_str = d.replace('-', '')
        
        # Dev version
        dev_file = DEV_OUTPUT / 'module1' / f'module1_output_{d_str}.xlsx'
        if dev_file.exists():
            dev_df = pd.read_excel(dev_file, sheet_name='OrderLog')
            dev_cnt = len(dev_df)
        else:
            dev_cnt = 'N/A'
        
        # Local version  
        local_file = LOCAL_OUTPUT / 'module1' / f'Module1Output_{d_str}.xlsx'
        if local_file.exists():
            local_df = pd.read_excel(local_file, sheet_name='OrderLog')
            local_cnt = len(local_df)
        else:
            local_cnt = 'N/A'
        
        match = 'YES' if dev_cnt == local_cnt else 'NO'
        print(f"{d:<12} | {str(dev_cnt):>8} | {str(local_cnt):>8} | {match:>10}")

def compare_module6_delivery():
    """Compare Module6 DeliveryPlan across versions"""
    print("\n" + "="*60)
    print("Module6 DeliveryPlan Comparison")
    print("="*60)
    print(f"{'Date':<12} | {'Dev':>8} | {'Local':>8} | {'Dev==Local':>10}")
    print("-"*60)
    
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
        
        match = 'YES' if dev_cnt == local_cnt else 'NO'
        print(f"{d:<12} | {str(dev_cnt):>8} | {str(local_cnt):>8} | {match:>10}")

def compare_module4_details():
    """Compare Module4 ProductionPlan details for Day 2"""
    print("\n" + "="*60)
    print("Module4 ProductionPlan Details (Day 2: 2025-10-07)")
    print("="*60)
    
    d_str = '20251007'
    
    # Dev version
    dev_file = DEV_OUTPUT / 'module4' / f'Module4Output_{d_str}.xlsx'
    if dev_file.exists():
        dev_df = pd.read_excel(dev_file, sheet_name='ProductionPlan')
        print(f"\nDev version: {len(dev_df)} records")
        if not dev_df.empty and 'available_date' in dev_df.columns:
            by_date = dev_df.groupby(pd.to_datetime(dev_df['available_date']).dt.date).size()
            print("  By available_date:")
            for dt, cnt in by_date.items():
                print(f"    {dt}: {cnt}")
    
    # Local version  
    local_file = LOCAL_OUTPUT / 'module4' / f'Module4Output_{d_str}.xlsx'
    if local_file.exists():
        local_df = pd.read_excel(local_file, sheet_name='ProductionPlan')
        print(f"\nLocal version: {len(local_df)} records")
        if not local_df.empty and 'available_date' in local_df.columns:
            by_date = local_df.groupby(pd.to_datetime(local_df['available_date']).dt.date).size()
            print("  By available_date:")
            for dt, cnt in by_date.items():
                print(f"    {dt}: {cnt}")

def main():
    print("="*60)
    print("ChainSight Version Comparison")
    print("="*60)
    print(f"Dev output:   {DEV_OUTPUT}")
    print(f"Local output: {LOCAL_OUTPUT}")
    
    compare_module1_orders()
    compare_module4_production()
    compare_module5_unfulfilled()
    compare_module6_delivery()
    compare_module4_details()

if __name__ == '__main__':
    main()
