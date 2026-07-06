"""
Generate consolidated Excel report and HTML analysis for SDC baseline scenario.
Run with Windows Python.
"""
from pathlib import Path
import json, sys

# Load KPI results
data = json.load(Path(r'\\wsl.localhost\Ubuntu-22.04\home\zhangs37\ai_chainsight\workspace\sdc-space-rccp-simulation-202605\scripts\kpi_results.json').open())

import pandas as pd

# Build DataFrames
df_cfr_dc = pd.DataFrame(data['cfr_by_month_dc']['rows'], columns=data['cfr_by_month_dc']['columns'])
df_cfr_month = pd.DataFrame(data['cfr_by_month']['rows'], columns=data['cfr_by_month']['columns'])
df_rccp_dc = pd.DataFrame(data['rccp_by_month_dc']['rows'], columns=data['rccp_by_month_dc']['columns'])
df_rccp_month = pd.DataFrame(data['rccp_by_month']['rows'], columns=data['rccp_by_month']['columns'])
df_mei_dc = pd.DataFrame(data['month_end_inventory_by_dc']['rows'], columns=data['month_end_inventory_by_dc']['columns'])
df_mei_net = pd.DataFrame(data['month_end_inventory_network']['rows'], columns=data['month_end_inventory_network']['columns'])
df_lane_lt = pd.DataFrame(data['lane_lt']['rows'], columns=data['lane_lt']['columns'])

# Filter out Jan/Feb 2027 (simulation end artifacts)
for df_name in ['df_cfr_dc', 'df_cfr_month']:
    df = locals()[df_name]
    locals()[df_name] = df[df['month'] < '2027-01']
df_cfr_dc = df_cfr_dc[df_cfr_dc['month'] < '2027-01']
df_cfr_month = df_cfr_month[df_cfr_month['month'] < '2027-01']

# Write Excel
out_path = Path(r'\\wsl.localhost\Ubuntu-22.04\home\zhangs37\ai_chainsight\workspace\sdc-space-rccp-simulation-202605\deliverables\sdc-space-rccp-simulation-202605_result_summary_20260513.xlsx')
with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
    df_cfr_month.to_excel(writer, sheet_name='CFR_by_Month', index=False)
    df_cfr_dc.to_excel(writer, sheet_name='CFR_by_Month_DC', index=False)
    df_rccp_month.to_excel(writer, sheet_name='RCCP_by_Month', index=False)
    df_rccp_dc.to_excel(writer, sheet_name='RCCP_by_Month_DC', index=False)
    df_mei_net.to_excel(writer, sheet_name='MonthEnd_Inv_Network', index=False)
    df_mei_dc.to_excel(writer, sheet_name='MonthEnd_Inv_by_DC', index=False)
    df_lane_lt.to_excel(writer, sheet_name='Lane_LeadTime', index=False)

print(f"Excel written: {out_path}", file=sys.stderr)
print("OK")
