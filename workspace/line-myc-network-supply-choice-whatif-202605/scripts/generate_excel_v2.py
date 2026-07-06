"""
Re-generate the consolidated Excel with MEI sheet added.
Run with Windows Python.
"""
from pathlib import Path
import json, sys
try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter
except ImportError:
    print("openpyxl not found, installing...", file=sys.stderr)
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'openpyxl', '-q'])
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

data = json.load(open(r'\\wsl.localhost\Ubuntu-22.04\home\zhangs37\ai_chainsight\workspace\line-myc-network-supply-choice-whatif-202605\scripts\kpi_results.json'))

wb = openpyxl.Workbook()
hdr_font = Font(bold=True, color='FFFFFF', size=10)
hdr_fill = PatternFill('solid', fgColor='1E40AF')
hdr_align = Alignment(horizontal='center', vertical='center')
num_fmt_pct = '0.00%'
num_fmt_2 = '#,##0.00'
num_fmt_0 = '#,##0'

scenarios = {
    'wf1': 'Produce HS / Order 50-50 / Current SS',
    'wf2': 'Produce 50-50 / Order HS / Current SS',
    'wf3': 'Produce HS / Order 50-50 / Reduced SS',
    'wf4': 'Produce 50-50 / Order HS / Reduced SS',
}

def write_sheet(ws, title, columns, rows, num_cols=None, pct_cols=None):
    ws.title = title
    for c, col in enumerate(columns, 1):
        cell = ws.cell(1, c, col)
        cell.font = hdr_font
        cell.fill = hdr_fill
        cell.alignment = hdr_align
    for r, row in enumerate(rows, 2):
        for c, val in enumerate(row, 1):
            cell = ws.cell(r, c, val)
            if pct_cols and c in pct_cols and isinstance(val, (int, float)):
                cell.number_format = num_fmt_pct
            elif num_cols and c in num_cols and isinstance(val, (int, float)):
                cell.number_format = num_fmt_2
    for c in range(1, len(columns)+1):
        ws.column_dimensions[get_column_letter(c)].width = max(14, len(columns[c-1])+4)
    ws.auto_filter.ref = ws.dimensions

# --- metadata ---
ws = wb.active
ws.title = 'metadata'
meta = [
    ['Project', 'line-myc-network-supply-choice-whatif-202605'],
    ['Generated', '2026-05-13'],
    ['Period', '2026-05-04 to 2026-06-19'],
    ['Scenarios', '4 (wf1, wf2, wf3, wf4)'],
    ['KPIs', 'Service, Space RCCP, Month End Inventory, Changeover, Lane LT'],
    ['Note', 'MEI added in this version'],
]
for r, row in enumerate(meta, 1):
    for c, val in enumerate(row, 1):
        ws.cell(r, c, val)

# --- scenario_definitions ---
ws2 = wb.create_sheet('scenario_definitions')
cols = ['Scenario', 'Produce Policy', 'Order Policy', 'Safety Stock', 'Run ID']
run_ids = {
    'wf1': 'db_wf1-produce-hs-order-5050-current-ss_20260512_230651',
    'wf2': 'db_wf2-produce-5050-order-hs-current-ss_20260512_230707',
    'wf3': 'db_wf3-produce-hs-order-5050-reduced-ss_20260512_230717',
    'wf4': 'db_wf4-produce-5050-order-hs-reduced-ss_20260512_230848',
}
sdef_rows = [
    ['wf1', 'High-side', '50/50 forecast', 'Current SS', run_ids['wf1']],
    ['wf2', '50/50', 'High-side', 'Current SS', run_ids['wf2']],
    ['wf3', 'High-side', '50/50 forecast', 'DTC 30d→15d', run_ids['wf3']],
    ['wf4', '50/50', 'High-side', 'DTC 30d→15d', run_ids['wf4']],
]
write_sheet(ws2, 'scenario_definitions', cols, sdef_rows)

# --- service_month ---
ws3 = wb.create_sheet()
sm_cols = ['scenario','month','shipment_qty','order_qty','cfr']
sm_rows = []
for wf in ['wf1','wf2','wf3','wf4']:
    for r in data[f'{wf}_service_month']['rows']:
        sm_rows.append([wf] + r)
write_sheet(ws3, 'service_month', sm_cols, sm_rows, num_cols={3,4}, pct_cols={5})

# --- service_month_loc ---
ws4 = wb.create_sheet()
sml_cols = ['scenario','month','location','shipment_qty','order_qty','cfr']
sml_rows = []
for wf in ['wf1','wf2','wf3','wf4']:
    for r in data[f'{wf}_service_month_loc']['rows']:
        sml_rows.append([wf] + r)
write_sheet(ws4, 'service_month_loc', sml_cols, sml_rows, num_cols={4,5}, pct_cols={6})

# --- space_rccp_month_loc ---
ws5 = wb.create_sheet()
rccp_cols = ['scenario','month','location','rccp_cbm','peak_date']
rccp_rows = []
for wf in ['wf1','wf2','wf3','wf4']:
    for r in data[f'{wf}_rccp_month_loc']['rows']:
        rccp_rows.append([wf] + r)
write_sheet(ws5, 'space_rccp_month_loc', rccp_cols, rccp_rows, num_cols={4})

# --- month_end_inv_loc ---
ws_mei = wb.create_sheet()
mei_cols = ['scenario','month','location','month_end_date','inventory_qty','inventory_cbm']
mei_rows = []
for wf in ['wf1','wf2','wf3','wf4']:
    for r in data[f'{wf}_mei_loc']['rows']:
        mei_rows.append([wf] + r)
write_sheet(ws_mei, 'month_end_inv_loc', mei_cols, mei_rows, num_cols={5,6})

# --- month_end_inv_network ---
ws_mein = wb.create_sheet()
mein_cols = ['scenario','month','month_end_date','inventory_qty','inventory_cbm']
mein_rows = []
for wf in ['wf1','wf2','wf3','wf4']:
    for r in data[f'{wf}_mei_network']['rows']:
        mein_rows.append([wf] + r)
write_sheet(ws_mein, 'month_end_inv_network', mein_cols, mein_rows, num_cols={4,5})

# --- changeover_month_line ---
ws6 = wb.create_sheet()
co_cols = ['scenario','month','location','line','changeover_type','co_count','co_time','pct']
co_rows = []
for wf in ['wf1','wf2','wf3','wf4']:
    for r in data[f'{wf}_changeover']['rows']:
        co_rows.append([wf] + r)
write_sheet(ws6, 'changeover_month_line', co_cols, co_rows, num_cols={6,7}, pct_cols={8})

# --- lane_lt_diag ---
ws7 = wb.create_sheet()
lt_cols = ['scenario'] + data['wf1_lane_lt']['columns']
lt_rows = []
for wf in ['wf1','wf2','wf3','wf4']:
    for r in data[f'{wf}_lane_lt']['rows']:
        lt_rows.append([wf] + r)
write_sheet(ws7, 'lane_lt_diag', lt_cols, lt_rows, num_cols=set(range(5,18)))

# --- scenario_summary ---
ws_sum = wb.create_sheet('scenario_summary')
sum_cols = ['Scenario','May Service','Jun Service','C937 Jun Service','0386 Jun RCCP (CBM)','May CO Count',
            'Network May ME Inv (Cases)','Network Jun ME Inv (Cases)','Network May ME Inv (CBM)','Network Jun ME Inv (CBM)',
            '0386 Jun ME Inv (CBM)','Role']
# Build summary rows
def get_cfr(wf, month):
    for r in data[f'{wf}_service_month']['rows']:
        if r[0] == month: return r[3]
    return None
def get_loc_cfr(wf, month, loc):
    for r in data[f'{wf}_service_month_loc']['rows']:
        if r[0] == month and r[1] == loc: return r[4]
    return None
def get_rccp(wf, month, loc):
    for r in data[f'{wf}_rccp_month_loc']['rows']:
        if r[0] == month and r[1] == loc: return r[2]
    return None
def get_co_count(wf, month):
    return sum(r[4] for r in data[f'{wf}_changeover']['rows'] if r[0] == month)
def get_mei_net(wf, month):
    for r in data[f'{wf}_mei_network']['rows']:
        if r[0] == month: return r[2], r[3]  # qty, cbm
    return None, None
def get_mei_loc(wf, month, loc):
    for r in data[f'{wf}_mei_loc']['rows']:
        if r[0] == month and r[1] == loc: return r[4]  # cbm
    return None

roles = {'wf1': 'dominated by wf2', 'wf2': 'service-first', 'wf3': 'balanced space-relief', 'wf4': 'dominated by wf3'}
sum_rows = []
for wf in ['wf1','wf2','wf3','wf4']:
    may_q, may_c = get_mei_net(wf, '2026-05')
    jun_q, jun_c = get_mei_net(wf, '2026-06')
    sum_rows.append([
        wf,
        get_cfr(wf, '2026-05'),
        get_cfr(wf, '2026-06'),
        get_loc_cfr(wf, '2026-06', 'C937'),
        get_rccp(wf, '2026-06', '0386'),
        get_co_count(wf, '2026-05'),
        may_q, jun_q, may_c, jun_c,
        get_mei_loc(wf, '2026-06', '0386'),
        roles[wf],
    ])
write_sheet(ws_sum, 'scenario_summary', sum_cols, sum_rows, num_cols={5,7,8,9,10,11}, pct_cols={2,3,4})

# Move scenario_summary after scenario_definitions
wb.move_sheet('scenario_summary', offset=-5)

out = Path(r'\\wsl.localhost\Ubuntu-22.04\home\zhangs37\ai_chainsight\workspace\line-myc-network-supply-choice-whatif-202605\deliverables\line-myc-network-supply-choice-whatif-202605_result_summary_20260513.xlsx')
wb.save(str(out))
print(f"Excel written: {out}")
print("OK")
