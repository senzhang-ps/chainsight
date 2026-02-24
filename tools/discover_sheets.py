"""Discover sheet names for all modules from Dev and Src (Day 1 sample)"""
import openpyxl
import os

dev_base = r"D:\PG\test\chainsight\ChainSight_Dev\BC_S5\run_20260211_181635"
src_base = r"D:\PG\test\chainsight\outputs\BC_S5\run_20260211_145215"
sample_date = "20251005"

modules = {
    "module1": f"module1_output_{sample_date}.xlsx",
    "module3": f"Module3Output_{sample_date}.xlsx",
    "module4": f"Module4Output_{sample_date}.xlsx",
    "module5": f"Module5Output_{sample_date}.xlsx",
    "module6": f"Module6Output_{sample_date}.xlsx",
}

for mod, fname in modules.items():
    print(f"\n=== {mod} ===")
    dev_path = os.path.join(dev_base, mod, fname)
    src_path = os.path.join(src_base, mod, fname)
    
    for label, path in [("Dev", dev_path), ("Src", src_path)]:
        if os.path.exists(path):
            wb = openpyxl.load_workbook(path, read_only=True)
            sheets = wb.sheetnames
            print(f"  {label}: {sheets}")
            wb.close()
        else:
            print(f"  {label}: FILE NOT FOUND at {path}")
