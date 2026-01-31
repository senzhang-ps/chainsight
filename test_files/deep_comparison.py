# -*- coding: utf-8 -*-
"""
Deep comparison of Dev vs Local outputs to find root cause of differences.
"""

import pandas as pd
from pathlib import Path
import numpy as np

DEV_OUTPUT = Path(r"C:\Users\25936\Desktop\Code\chainsight\ChainSight_Dev\BC_S5\run_20260129_204512")
LOCAL_OUTPUT = Path(r"C:\Users\25936\Desktop\Code\chainsight\outputs\BC_S5\run_20260130_115401")

def compare_module4_production_records():
    """Compare actual production records between Dev and Local for Day 2"""
    print("\n" + "="*70)
    print("Module4 ProductionPlan Deep Comparison (Day 2: 2025-10-07)")
    print("="*70)
    
    d_str = '20251007'
    
    dev_file = DEV_OUTPUT / 'module4' / f'Module4Output_{d_str}.xlsx'
    local_file = LOCAL_OUTPUT / 'module4' / f'Module4Output_{d_str}.xlsx'
    
    dev_df = pd.read_excel(dev_file, sheet_name='ProductionPlan')
    local_df = pd.read_excel(local_file, sheet_name='ProductionPlan')
    
    # Find common columns for comparison
    common_cols = set(dev_df.columns) & set(local_df.columns)
    print(f"\nCommon columns: {sorted(common_cols)}")
    
    # Compare key columns
    key_cols = ['material', 'line', 'produced_qty', 'available_date']
    
    print("\n--- Dev version sample (first 10 rows) ---")
    print(dev_df[key_cols].head(10).to_string())
    
    print("\n--- Local version sample (first 10 rows) ---")
    print(local_df[key_cols].head(10).to_string())
    
    # Compare sorted by same columns
    dev_sorted = dev_df.sort_values(by=['material_code', 'available_date']).reset_index(drop=True)
    local_sorted = local_df.sort_values(by=['material_code', 'available_date']).reset_index(drop=True)
    
    print("\n--- Differences when sorted by material_code + available_date ---")
    for col in key_cols:
        dev_vals = dev_sorted[col].tolist()
        local_vals = local_sorted[col].tolist()
        diff_count = sum(1 for d, l in zip(dev_vals, local_vals) if d != l)
        print(f"  {col}: {diff_count} different values")

def compare_module5_unfulfilled_day1():
    """Compare Module5 UnfulfilledLog for Day 1 - should be most similar"""
    print("\n" + "="*70)
    print("Module5 UnfulfilledLog Deep Comparison (Day 1: 2025-10-06)")
    print("="*70)
    
    d_str = '20251006'
    
    dev_file = DEV_OUTPUT / 'module5' / f'Module5Output_{d_str}.xlsx'
    local_file = LOCAL_OUTPUT / 'module5' / f'Module5Output_{d_str}.xlsx'
    
    dev_df = pd.read_excel(dev_file, sheet_name='UnfulfilledLog')
    local_df = pd.read_excel(local_file, sheet_name='UnfulfilledLog')
    
    print(f"\nDev rows: {len(dev_df)}")
    print(f"Local rows: {len(local_df)}")
    print(f"Difference: {len(dev_df) - len(local_df)} rows")
    
    # Compare by key columns
    key_cols = ['Date', 'Material', 'Plant', 'Sloc'] if 'Date' in dev_df.columns else dev_df.columns[:4].tolist()
    
    # Create composite keys
    dev_df['key'] = dev_df.apply(lambda r: f"{r.get('Date', '')}_{r.get('Material', '')}_{r.get('Plant', '')}_{r.get('Sloc', '')}", axis=1)
    local_df['key'] = local_df.apply(lambda r: f"{r.get('Date', '')}_{r.get('Material', '')}_{r.get('Plant', '')}_{r.get('Sloc', '')}", axis=1)
    
    dev_keys = set(dev_df['key'])
    local_keys = set(local_df['key'])
    
    only_in_dev = dev_keys - local_keys
    only_in_local = local_keys - dev_keys
    
    print(f"\nKeys only in Dev: {len(only_in_dev)}")
    print(f"Keys only in Local: {len(only_in_local)}")
    
    if only_in_dev:
        print("\nSample records only in Dev:")
        sample = dev_df[dev_df['key'].isin(list(only_in_dev)[:5])]
        print(sample[key_cols].to_string())
    
    if only_in_local:
        print("\nSample records only in Local:")
        sample = local_df[local_df['key'].isin(list(only_in_local)[:5])]
        print(sample[key_cols].to_string())

def compare_module5_effective_deployment_day1():
    """Compare Module5 Effective Deployment for Day 1"""
    print("\n" + "="*70)
    print("Module5 DeploymentPlan (Effective) Deep Comparison (Day 1: 2025-10-06)")
    print("="*70)
    
    d_str = '20251006'
    
    dev_file = DEV_OUTPUT / 'module5' / f'Module5Output_{d_str}.xlsx'
    local_file = LOCAL_OUTPUT / 'module5' / f'Module5Output_{d_str}.xlsx'
    
    dev_df = pd.read_excel(dev_file, sheet_name='DeploymentPlan')
    local_df = pd.read_excel(local_file, sheet_name='DeploymentPlan')
    
    print(f"\nDev rows: {len(dev_df)}")
    print(f"Local rows: {len(local_df)}")
    
    # Check the effective deployment rows (status = effective or similar)
    if 'status' in dev_df.columns:
        dev_effective = dev_df[dev_df['status'] == 'effective']
        local_effective = local_df[local_df['status'] == 'effective']
        print(f"\nDev effective: {len(dev_effective)}")
        print(f"Local effective: {len(local_effective)}")

def compare_module1_orders_day1():
    """Compare Module1 Orders for Day 1"""
    print("\n" + "="*70)
    print("Module1 OrderLog Deep Comparison (Day 1: 2025-10-06)")
    print("="*70)
    
    d_str = '20251006'
    
    dev_file = DEV_OUTPUT / 'module1' / f'module1_output_{d_str}.xlsx'
    local_file = LOCAL_OUTPUT / 'module1' / f'module1_output_{d_str}.xlsx'
    
    dev_df = pd.read_excel(dev_file, sheet_name='OrderLog')
    local_df = pd.read_excel(local_file, sheet_name='OrderLog')
    
    print(f"\nDev rows: {len(dev_df)}")
    print(f"Local rows: {len(local_df)}")
    
    # Compare columns
    print(f"\nDev columns: {list(dev_df.columns)}")
    print(f"Local columns: {list(local_df.columns)}")
    
    # Compare data
    if len(dev_df) == len(local_df):
        print("\nSame row count - checking if data matches...")
        for col in dev_df.columns:
            if col in local_df.columns:
                if dev_df[col].dtype == 'float64' or local_df[col].dtype == 'float64':
                    matches = np.allclose(dev_df[col].fillna(0), local_df[col].fillna(0), rtol=1e-5)
                else:
                    matches = dev_df[col].equals(local_df[col])
                if not matches:
                    print(f"  {col}: DIFFERENT")
        print("\nModule1 comparison done")

def compare_orchestrator_inventory_day1():
    """Compare orchestrator inventory for Day 1"""
    print("\n" + "="*70)
    print("Orchestrator UnrestrictedInventory Deep Comparison (Day 1)")
    print("="*70)
    
    d_str = '20251006'
    
    dev_file = DEV_OUTPUT / 'orchestrator' / f'Orchestrator_State_{d_str}.xlsx'
    local_file = LOCAL_OUTPUT / 'orchestrator' / f'Orchestrator_State_{d_str}.xlsx'
    
    if not dev_file.exists() or not local_file.exists():
        print("Files not found")
        return
    
    dev_df = pd.read_excel(dev_file, sheet_name='UnrestrictedInventory')
    local_df = pd.read_excel(local_file, sheet_name='UnrestrictedInventory')
    
    print(f"\nDev rows: {len(dev_df)}")
    print(f"Local rows: {len(local_df)}")
    
    # Check total inventory
    if 'qty' in dev_df.columns:
        print(f"\nDev total qty: {dev_df['qty'].sum()}")
        print(f"Local total qty: {local_df['qty'].sum()}")

def main():
    print("="*70)
    print("ChainSight Deep Comparison Analysis")
    print("="*70)
    print(f"Dev output:   {DEV_OUTPUT}")
    print(f"Local output: {LOCAL_OUTPUT}")
    
    compare_module1_orders_day1()
    compare_orchestrator_inventory_day1()
    compare_module5_unfulfilled_day1()
    compare_module5_effective_deployment_day1()
    compare_module4_production_records()

if __name__ == '__main__':
    main()
