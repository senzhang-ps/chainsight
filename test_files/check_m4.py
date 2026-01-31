# Check local mode M4 output files
import pandas as pd
from pathlib import Path

local_dir = Path('outputs/BC_S5/run_20260130_113019/module4')

for f in sorted(local_dir.glob('Module4Output_*.xlsx')):
    xl = pd.ExcelFile(f)
    if 'ProductionPlan' in xl.sheet_names:
        df = pd.read_excel(f, sheet_name='ProductionPlan')
        print(f'{f.name}: {len(df)} production records')
        if not df.empty and 'available_date' in df.columns:
            avail_dates = df["available_date"].unique().tolist()
            print(f'  available_dates: {avail_dates}')
    else:
        print(f'{f.name}: No ProductionPlan sheet')
