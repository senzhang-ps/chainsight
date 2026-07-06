from pathlib import Path
import json
import pandas as pd
import psycopg

ROOT = Path(r"\\wsl.localhost\Ubuntu-22.04\home\zhangs37\ai_chainsight")
RUN_ID = 'db_PDS2_20260527_181240'
FOCUS_LOCATIONS = ['C816', 'C810', 'D873', 'C866', 'C867']
RESULTS_DIR = ROOT / 'workspace' / 'south-space-rccp-jason26' / 'scenarios' / 'production-cycle' / 'results'
MATERIALS_FILE = RESULTS_DIR / 'focus_run_materials.csv'
DBX_MAP_FILE = RESULTS_DIR / 'material_category_en_databricks.csv'
OUT_LOC_CAT = RESULTS_DIR / 'rccp_by_month_location_category_en_focus_locations.csv'
OUT_CAT = RESULTS_DIR / 'rccp_by_month_category_en_focus_locations.csv'
CREDS_FILE = ROOT / 'db_credentials.yaml'


def load_creds() -> dict:
    creds = {}
    for raw in CREDS_FILE.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        k, v = line.split(':', 1)
        creds[k.strip()] = v.strip()
    creds['database'] = 'south_rccp'
    return creds


def fetch_inventory_detail() -> pd.DataFrame:
    creds = load_creds()
    query = """
    select
        to_char(i.date::date, 'YYYY-MM') as month,
        i.location,
        i.date::date as inventory_date,
        i.material,
        round(sum(i.quantity * coalesce(m.demand_unit_to_volume, 0))::numeric, 4) as material_cbm
    from orchestrator_unrestricted_inventory i
    left join cfg_m6_materialmd m
      on m.config_name = i.config_name
     and m.material = i.material
    where i.run_id = %(run_id)s
      and i.location = any(%(locations)s)
    group by 1, 2, 3, 4
    order by 1, 2, 3, 4
    """
    with psycopg.connect(
        host=creds['host'], port=int(creds['port']), dbname=creds['database'],
        user=creds['user'], password=creds['password'], connect_timeout=10,
    ) as conn:
        return pd.read_sql_query(query, conn, params={'run_id': RUN_ID, 'locations': FOCUS_LOCATIONS})


def build_from_mapping(mapping: pd.DataFrame):
    detail = fetch_inventory_detail()
    detail['material'] = detail['material'].astype(str).str.strip()
    mapping['material'] = mapping['material'].astype(str).str.strip()
    mapping['category_en'] = mapping['category_en'].fillna('(unmapped)').astype(str).str.strip()
    mapping.loc[mapping['category_en'].isin(['', 'nan', 'None']), 'category_en'] = '(unmapped)'
    detail = detail.merge(mapping[['material', 'category_en']].drop_duplicates('material'), on='material', how='left')
    detail['category_en'] = detail['category_en'].fillna('(unmapped)')

    by_loc_cat_daily = (
        detail.groupby(['month', 'location', 'inventory_date', 'category_en'], as_index=False)['material_cbm']
        .sum()
        .rename(columns={'material_cbm': 'daily_cbm'})
    )
    by_loc_cat_peak = (
        by_loc_cat_daily.sort_values(['month', 'location', 'category_en', 'daily_cbm', 'inventory_date'], ascending=[True, True, True, False, True])
        .drop_duplicates(subset=['month', 'location', 'category_en'], keep='first')
        .rename(columns={'inventory_date': 'peak_date', 'daily_cbm': 'peak_cbm'})
        .sort_values(['month', 'location', 'peak_cbm', 'category_en'], ascending=[True, True, False, True])
        .reset_index(drop=True)
    )

    by_cat_daily = (
        detail.groupby(['month', 'inventory_date', 'category_en'], as_index=False)['material_cbm']
        .sum()
        .rename(columns={'material_cbm': 'daily_cbm'})
    )
    by_cat_peak = (
        by_cat_daily.sort_values(['month', 'category_en', 'daily_cbm', 'inventory_date'], ascending=[True, True, False, True])
        .drop_duplicates(subset=['month', 'category_en'], keep='first')
        .rename(columns={'inventory_date': 'peak_date', 'daily_cbm': 'peak_cbm'})
        .sort_values(['month', 'peak_cbm', 'category_en'], ascending=[True, False, True])
        .reset_index(drop=True)
    )

    by_loc_cat_peak.to_csv(OUT_LOC_CAT, index=False, encoding='utf-8-sig')
    by_cat_peak.to_csv(OUT_CAT, index=False, encoding='utf-8-sig')
    print('mapping_rows', len(mapping))
    print('mapped_unique_materials', mapping['material'].nunique())
    print('OUT_LOC_CAT', OUT_LOC_CAT)
    print('OUT_CAT', OUT_CAT)
    print(by_cat_peak.to_string(index=False))


if __name__ == '__main__':
    mapping = pd.read_csv(DBX_MAP_FILE, dtype=str)
    build_from_mapping(mapping)
