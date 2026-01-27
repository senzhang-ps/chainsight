#!/usr/bin/env python3
"""
Comparison Test: Dev vs Refactored Version

This script runs both the Dev (ChainSight_Dev) and Refactored (src) versions
of the supply chain planning system using the same configuration file (BC_S5.xlsx)
for the date range 2025-10-06 to 2025-10-10, then compares the outputs.

Usage:
    python test_files/compare_dev_refactored.py
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import shutil

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_FILE = PROJECT_ROOT / "config" / "BC_S5.xlsx"
OUTPUT_DIR = PROJECT_ROOT / "test_output_comparison"

# Date range for testing
START_DATE = "2025-10-06"
END_DATE = "2025-10-10"


def run_dev_version():
    """Run the Dev (baseline) version"""
    print("\n" + "="*80)
    print("Running DEV Version (ChainSight_Dev)")
    print("="*80)
    
    dev_dir = PROJECT_ROOT / "ChainSight_Dev"
    dev_output = OUTPUT_DIR / "dev_output"
    
    # Clean previous output
    if dev_output.exists():
        shutil.rmtree(dev_output)
    dev_output.mkdir(parents=True, exist_ok=True)
    
    # Add dev directory to path
    sys.path.insert(0, str(dev_dir))
    
    try:
        # Import dev modules
        from main_integration import run_integrated_simulation, load_configuration
        
        # Run simulation
        result = run_integrated_simulation(
            config_path=str(CONFIG_FILE),
            start_date=START_DATE,
            end_date=END_DATE,
            output_base_dir=str(dev_output),
            force_restart=True
        )
        
        print(f"DEV Version completed: {result.get('dates_processed', 'N/A')} days processed")
        return True, dev_output
        
    except Exception as e:
        print(f"DEV Version failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None
    finally:
        # Remove dev directory from path
        if str(dev_dir) in sys.path:
            sys.path.remove(str(dev_dir))


def run_refactored_version():
    """Run the Refactored version"""
    print("\n" + "="*80)
    print("Running REFACTORED Version (src)")
    print("="*80)
    
    src_dir = PROJECT_ROOT / "src"
    refactored_output = OUTPUT_DIR / "refactored_output"
    
    # Clean previous output
    if refactored_output.exists():
        shutil.rmtree(refactored_output)
    refactored_output.mkdir(parents=True, exist_ok=True)
    
    # Add src directory to path
    sys.path.insert(0, str(PROJECT_ROOT))
    
    try:
        # Import refactored modules
        from src.core.main_integration import run_integrated_simulation, load_configuration
        
        # Run simulation
        result = run_integrated_simulation(
            config_path=str(CONFIG_FILE),
            start_date=START_DATE,
            end_date=END_DATE,
            output_base_dir=str(refactored_output),
            force_restart=True
        )
        
        print(f"REFACTORED Version completed: {result.get('dates_processed', 'N/A')} days processed")
        return True, refactored_output
        
    except Exception as e:
        print(f"REFACTORED Version failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None
    finally:
        # Remove src directory from path
        if str(PROJECT_ROOT) in sys.path:
            sys.path.remove(str(PROJECT_ROOT))


def compare_csv_files(dev_file: Path, refactored_file: Path, tolerance: float = 0.01) -> dict:
    """Compare two CSV files and return differences"""
    result = {
        'match': False,
        'dev_exists': dev_file.exists(),
        'refactored_exists': refactored_file.exists(),
        'differences': []
    }
    
    if not dev_file.exists() or not refactored_file.exists():
        return result
    
    try:
        df_dev = pd.read_csv(dev_file)
        df_ref = pd.read_csv(refactored_file)
        
        # Check column match
        if set(df_dev.columns) != set(df_ref.columns):
            result['differences'].append(f"Column mismatch: DEV={list(df_dev.columns)}, REF={list(df_ref.columns)}")
            return result
        
        # Check row count
        if len(df_dev) != len(df_ref):
            result['differences'].append(f"Row count mismatch: DEV={len(df_dev)}, REF={len(df_ref)}")
        
        # Sort both dataframes for comparison
        sort_cols = [c for c in ['material', 'location', 'date', 'sending', 'receiving'] if c in df_dev.columns]
        if sort_cols:
            df_dev = df_dev.sort_values(sort_cols).reset_index(drop=True)
            df_ref = df_ref.sort_values(sort_cols).reset_index(drop=True)
        
        # Compare values
        numeric_cols = df_dev.select_dtypes(include=[np.number]).columns
        
        # Check numeric columns with tolerance
        for col in numeric_cols:
            if col in df_ref.columns:
                dev_vals = df_dev[col].fillna(0)
                ref_vals = df_ref[col].fillna(0)
                
                if len(dev_vals) == len(ref_vals):
                    diff = np.abs(dev_vals - ref_vals)
                    max_diff = diff.max()
                    if max_diff > tolerance:
                        result['differences'].append(f"Column '{col}': max diff = {max_diff:.4f}")
        
        # Check non-numeric columns
        str_cols = [c for c in df_dev.columns if c not in numeric_cols]
        for col in str_cols:
            if col in df_ref.columns:
                if len(df_dev) == len(df_ref):
                    dev_vals = df_dev[col].astype(str).fillna('')
                    ref_vals = df_ref[col].astype(str).fillna('')
                    mismatches = (dev_vals != ref_vals).sum()
                    if mismatches > 0:
                        result['differences'].append(f"Column '{col}': {mismatches} value mismatches")
        
        result['match'] = len(result['differences']) == 0
        
    except Exception as e:
        result['differences'].append(f"Comparison error: {e}")
    
    return result


def compare_outputs(dev_output: Path, refactored_output: Path):
    """Compare outputs from both versions"""
    print("\n" + "="*80)
    print("COMPARING OUTPUTS")
    print("="*80)
    
    # Key output directories
    dev_orchestrator = dev_output / "orchestrator"
    ref_orchestrator = refactored_output / "orchestrator"
    
    # Generate date range
    dates = pd.date_range(START_DATE, END_DATE, freq='D')
    
    comparison_results = {}
    
    # Files to compare per day
    file_patterns = [
        "unrestricted_inventory_{}.csv",
        "open_deployment_{}.csv",
        "planning_intransit_{}.csv",
        "space_quota_{}.csv",
        "delivery_gr_{}.csv",
        "production_gr_{}.csv",
        "shipment_log_{}.csv",
    ]
    
    for date in dates:
        date_str = date.strftime('%Y%m%d')
        date_key = date.strftime('%Y-%m-%d')
        comparison_results[date_key] = {}
        
        print(f"\n--- {date_key} ---")
        
        for pattern in file_patterns:
            filename = pattern.format(date_str)
            dev_file = dev_orchestrator / filename
            ref_file = ref_orchestrator / filename
            
            result = compare_csv_files(dev_file, ref_file)
            comparison_results[date_key][filename] = result
            
            if result['match']:
                print(f"  [MATCH] {filename}")
            elif not result['dev_exists']:
                print(f"  [MISSING-DEV] {filename}")
            elif not result['refactored_exists']:
                print(f"  [MISSING-REF] {filename}")
            else:
                print(f"  [DIFF] {filename}")
                for diff in result['differences'][:3]:  # Show first 3 differences
                    print(f"         - {diff}")
    
    # Summary
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    total_files = 0
    matched_files = 0
    diff_files = 0
    missing_files = 0
    
    for date_key, files in comparison_results.items():
        for filename, result in files.items():
            total_files += 1
            if result['match']:
                matched_files += 1
            elif not result['dev_exists'] or not result['refactored_exists']:
                missing_files += 1
            else:
                diff_files += 1
    
    print(f"Total files compared: {total_files}")
    print(f"Matching files: {matched_files}")
    print(f"Different files: {diff_files}")
    print(f"Missing files: {missing_files}")
    
    match_rate = (matched_files / total_files * 100) if total_files > 0 else 0
    print(f"\nMatch rate: {match_rate:.1f}%")
    
    if match_rate == 100.0:
        print("\n*** SUCCESS: All outputs match between DEV and REFACTORED versions! ***")
    else:
        print("\n*** WARNING: Some differences detected between versions ***")
    
    return comparison_results


def main():
    """Main entry point"""
    print("="*80)
    print("DEV vs REFACTORED Comparison Test")
    print(f"Config: {CONFIG_FILE}")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print("="*80)
    
    # Check config file exists
    if not CONFIG_FILE.exists():
        print(f"ERROR: Config file not found: {CONFIG_FILE}")
        return 1
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run both versions
    dev_success, dev_output = run_dev_version()
    if not dev_success:
        print("DEV version failed. Aborting comparison.")
        return 1
    
    refactored_success, refactored_output = run_refactored_version()
    if not refactored_success:
        print("REFACTORED version failed. Aborting comparison.")
        return 1
    
    # Compare outputs
    results = compare_outputs(dev_output, refactored_output)
    
    # Save comparison results
    results_file = OUTPUT_DIR / "comparison_results.txt"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write(f"Comparison Test Results\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Config: {CONFIG_FILE}\n")
        f.write(f"Range: {START_DATE} to {END_DATE}\n\n")
        
        for date_key, files in results.items():
            f.write(f"\n{date_key}:\n")
            for filename, result in files.items():
                status = "MATCH" if result['match'] else "DIFF" if result['dev_exists'] and result['refactored_exists'] else "MISSING"
                f.write(f"  {status}: {filename}\n")
                if result['differences']:
                    for diff in result['differences']:
                        f.write(f"    - {diff}\n")
    
    print(f"\nResults saved to: {results_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
