# -*- coding: utf-8 -*-
"""Compare output data between Dev and Refactored versions."""
import pandas as pd
import os

dev_base = r'C:\Users\25936\Desktop\Code\chainsight\ChainSight_Dev\BC_S5\run_20260125_223924'
new_base = r'C:\Users\25936\Desktop\Code\chainsight\outputs\BC_S5\run_20260125_225404'

print("=" * 60)
print("DATA CONSISTENCY VERIFICATION")
print("=" * 60)

# 1. Compare Module5 deployment
dev_m5 = pd.read_excel(f'{dev_base}/module5/Module5Output_20251006.xlsx', sheet_name='DeploymentPlan')
new_m5 = pd.read_excel(f'{new_base}/module5/Module5Output_20251006.xlsx', sheet_name='DeploymentPlan')

print("\n=== Module5 Deployment Plan ===")
print(f"Dev rows: {len(dev_m5)}")
print(f"New rows: {len(new_m5)}")
print(f"Row count match: {len(dev_m5) == len(new_m5)}")

dev_deployed = dev_m5['deployed_qty'].fillna(0).sum() if 'deployed_qty' in dev_m5.columns else 0
new_deployed = new_m5['deployed_qty'].fillna(0).sum() if 'deployed_qty' in new_m5.columns else 0
print(f"Dev deployed_qty sum: {dev_deployed}")
print(f"New deployed_qty sum: {new_deployed}")
print(f"Deployed qty match: {abs(dev_deployed - new_deployed) < 0.001}")

# 2. Compare Module1 output
dev_m1 = pd.read_excel(f'{dev_base}/module1/module1_output_20251006.xlsx', sheet_name='ShipmentLog')
new_m1 = pd.read_excel(f'{new_base}/module1/module1_output_20251006.xlsx', sheet_name='ShipmentLog')

print("\n=== Module1 Shipment ===")
print(f"Dev rows: {len(dev_m1)}")
print(f"New rows: {len(new_m1)}")
dev_ship_qty = dev_m1['quantity'].sum() if 'quantity' in dev_m1.columns else 0
new_ship_qty = new_m1['quantity'].sum() if 'quantity' in new_m1.columns else 0
print(f"Dev shipment qty: {dev_ship_qty}")
print(f"New shipment qty: {new_ship_qty}")
print(f"Shipment match: {dev_ship_qty == new_ship_qty}")

# 3. Compare Module3 net demand
dev_m3 = pd.read_excel(f'{dev_base}/module3/Module3Output_20251006.xlsx', sheet_name='NetDemand')
new_m3 = pd.read_excel(f'{new_base}/module3/Module3Output_20251006.xlsx', sheet_name='NetDemand')

print("\n=== Module3 Net Demand ===")
print(f"Dev rows: {len(dev_m3)}")
print(f"New rows: {len(new_m3)}")
dev_nd_qty = dev_m3['quantity'].sum() if 'quantity' in dev_m3.columns else 0
new_nd_qty = new_m3['quantity'].sum() if 'quantity' in new_m3.columns else 0
print(f"Dev net demand qty: {dev_nd_qty:.2f}")
print(f"New net demand qty: {new_nd_qty:.2f}")
print(f"Net demand match: {abs(dev_nd_qty - new_nd_qty) < 0.01}")

# 4. Compare inventory from orchestrator
dev_inv = pd.read_csv(f'{dev_base}/orchestrator/unrestricted_inventory_20251006.csv')
new_inv = pd.read_csv(f'{new_base}/orchestrator/unrestricted_inventory_20251006.csv')

print("\n=== Inventory ===")
print(f"Dev rows: {len(dev_inv)}")
print(f"New rows: {len(new_inv)}")
dev_inv_qty = dev_inv['quantity'].sum() if 'quantity' in dev_inv.columns else 0
new_inv_qty = new_inv['quantity'].sum() if 'quantity' in new_inv.columns else 0
print(f"Dev inventory qty: {dev_inv_qty}")
print(f"New inventory qty: {new_inv_qty}")
print(f"Inventory match: {dev_inv_qty == new_inv_qty}")

# 5. Compare open deployment
dev_od = pd.read_csv(f'{dev_base}/orchestrator/open_deployment_20251006.csv')
new_od = pd.read_csv(f'{new_base}/orchestrator/open_deployment_20251006.csv')

print("\n=== Open Deployment ===")
print(f"Dev rows: {len(dev_od)}")
print(f"New rows: {len(new_od)}")
dev_od_qty = dev_od['quantity'].sum() if 'quantity' in dev_od.columns else 0
new_od_qty = new_od['quantity'].sum() if 'quantity' in new_od.columns else 0
print(f"Dev open deployment qty: {dev_od_qty}")
print(f"New open deployment qty: {new_od_qty}")
print(f"Open deployment match: {dev_od_qty == new_od_qty}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
all_match = (
    len(dev_m5) == len(new_m5) and
    abs(dev_deployed - new_deployed) < 0.001 and
    dev_ship_qty == new_ship_qty and
    abs(dev_nd_qty - new_nd_qty) < 0.01 and
    dev_inv_qty == new_inv_qty and
    dev_od_qty == new_od_qty
)
if all_match:
    print("ALL METRICS MATCH - Data consistency verified!")
else:
    print("SOME METRICS DO NOT MATCH - Please investigate")
