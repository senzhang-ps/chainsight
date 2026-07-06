"""
Re-generate the consolidated Excel — v3.
Changes vs v2: remove Space RCCP, MEI in case qty, add material-level MEI sheet.
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
    ['KPIs', 'Service, Month End Inventory, Changeover, Lane LT'],
    ['Note', 'v3: RCCP removed, MEI in case qty, material-level MEI added'],
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

# --- month_end_inv_loc (cases only) ---
ws_mei = wb.create_sheet()
mei_cols = ['scenario','month','location','month_end_date','inventory_qty_cases']
mei_rows = []
for wf in ['wf1','wf2','wf3','wf4']:
    for r in data[f'{wf}_mei_loc']['rows']:
        # r = [month, location, month_end_date, inventory_qty, inventory_cbm]
        mei_rows.append([wf, r[0], r[1], r[2], r[3]])  # skip cbm (r[4])
write_sheet(ws_mei, 'month_end_inv_loc', mei_cols, mei_rows, num_cols={5})

# --- month_end_inv_network (cases only) ---
ws_mein = wb.create_sheet()
mein_cols = ['scenario','month','month_end_date','inventory_qty_cases']
mein_rows = []
for wf in ['wf1','wf2','wf3','wf4']:
    for r in data[f'{wf}_mei_network']['rows']:
        # r = [month, month_end_date, inventory_qty, inventory_cbm]
        mein_rows.append([wf, r[0], r[1], r[2]])  # skip cbm (r[3])
write_sheet(ws_mein, 'month_end_inv_network', mein_cols, mein_rows, num_cols={4})

# --- month_end_inv_material_loc (new) ---
ws_meiml = wb.create_sheet()
meiml_cols = ['scenario','month','location','material','month_end_date','inventory_qty_cases']
meiml_rows = []
for wf in ['wf1','wf2','wf3','wf4']:
    key = f'{wf}_mei_material_loc'
    if key in data:
        for r in data[key]['rows']:
            meiml_rows.append([wf] + r)
write_sheet(ws_meiml, 'month_end_inv_material_loc', meiml_cols, meiml_rows, num_cols={6})

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

# --- scenario_summary (no RCCP, MEI in cases) ---
ws_sum = wb.create_sheet('scenario_summary')
sum_cols = ['Scenario','May Service','Jun Service','C937 Jun Service',
            'May CO Count',
            'Network May ME Inv (Cases)','Network Jun ME Inv (Cases)',
            '0386 May ME Inv (Cases)','0386 Jun ME Inv (Cases)',
            'Role']

def get_cfr(wf, month):
    for r in data[f'{wf}_service_month']['rows']:
        if r[0] == month: return r[3]
    return None
def get_loc_cfr(wf, month, loc):
    for r in data[f'{wf}_service_month_loc']['rows']:
        if r[0] == month and r[1] == loc: return r[4]
    return None
def get_co_count(wf, month):
    return sum(r[4] for r in data[f'{wf}_changeover']['rows'] if r[0] == month)
def get_mei_net(wf, month):
    for r in data[f'{wf}_mei_network']['rows']:
        if r[0] == month: return r[2]  # inventory_qty (cases)
    return None
def get_mei_loc_qty(wf, month, loc):
    for r in data[f'{wf}_mei_loc']['rows']:
        if r[0] == month and r[1] == loc: return r[3]  # inventory_qty (cases)
    return None

roles = {'wf1': 'dominated by wf2', 'wf2': 'service-first', 'wf3': 'balanced inventory-relief', 'wf4': 'dominated by wf3'}
sum_rows = []
for wf in ['wf1','wf2','wf3','wf4']:
    sum_rows.append([
        wf,
        get_cfr(wf, '2026-05'),
        get_cfr(wf, '2026-06'),
        get_loc_cfr(wf, '2026-06', 'C937'),
        get_co_count(wf, '2026-05'),
        get_mei_net(wf, '2026-05'),
        get_mei_net(wf, '2026-06'),
        get_mei_loc_qty(wf, '2026-05', '0386'),
        get_mei_loc_qty(wf, '2026-06', '0386'),
        roles[wf],
    ])
write_sheet(ws_sum, 'scenario_summary', sum_cols, sum_rows, num_cols={5,6,7,8,9}, pct_cols={2,3,4})

# Move scenario_summary after scenario_definitions
wb.move_sheet('scenario_summary', offset=-5)

out = Path(r'\\wsl.localhost\Ubuntu-22.04\home\zhangs37\ai_chainsight\workspace\line-myc-network-supply-choice-whatif-202605\deliverables\line-myc-network-supply-choice-whatif-202605_result_summary_20260513.xlsx')
wb.save(str(out))
print(f"Excel written: {out}")
print("OK")
