from pathlib import Path
from datetime import datetime
import pandas as pd

ROOT = Path(r"\\wsl.localhost\Ubuntu-22.04\home\zhangs37\ai_chainsight")
RESULTS = ROOT / 'workspace' / 'south-space-rccp-jason26' / 'scenarios' / 'production-cycle' / 'results'
DELIVERABLES = ROOT / 'workspace' / 'south-space-rccp-jason26' / 'deliverables'
DELIVERABLES.mkdir(parents=True, exist_ok=True)

def read_csv(name):
    return pd.read_csv(RESULTS / name)

rccp = read_csv('rccp_by_month_focus_locations.csv')
cfr = read_csv('cfr_by_month_focus_locations.csv')
dfc = read_csv('dfc_on_rccp_peak_day_by_month_focus_locations.csv')
cat = read_csv('rccp_by_month_category_en_focus_locations.csv')
loc_cat = read_csv('rccp_by_month_location_category_en_focus_locations.csv')

rccp_pivot = rccp.pivot(index='month', columns='location', values='peak_cbm').reset_index()
cfr_pivot = cfr.pivot(index='month', columns='location', values='cfr_pct').reset_index()
dfc_pivot = dfc.pivot(index='month', columns='location', values='dfc_days').reset_index()
cat_pivot = cat.pivot(index='month', columns='category_en', values='peak_cbm').reset_index()

summary = rccp.merge(cfr[['month','location','cfr_pct']], on=['month','location'], how='left') \
              .merge(dfc[['month','location','dfc_days']], on=['month','location'], how='left')

stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
out = DELIVERABLES / f'south-space-rccp-jason26_focus-summary_{stamp}.xlsx'
with pd.ExcelWriter(out, engine='openpyxl') as writer:
    summary.to_excel(writer, sheet_name='focus_summary', index=False)
    rccp.to_excel(writer, sheet_name='rccp_month_location', index=False)
    rccp_pivot.to_excel(writer, sheet_name='rccp_pivot', index=False)
    cfr.to_excel(writer, sheet_name='cfr_month_location', index=False)
    cfr_pivot.to_excel(writer, sheet_name='cfr_pivot', index=False)
    dfc.to_excel(writer, sheet_name='dfc_peakday_month_loc', index=False)
    dfc_pivot.to_excel(writer, sheet_name='dfc_pivot', index=False)
    cat.to_excel(writer, sheet_name='rccp_month_category_en', index=False)
    cat_pivot.to_excel(writer, sheet_name='rccp_cat_pivot', index=False)
    loc_cat.to_excel(writer, sheet_name='rccp_month_loc_cat_en', index=False)
print(out)
