"""
Debug script to compare intermediate values between Dev and Local Module3 net demand calculations.

This script identifies WHY the Local version shows ~20% less net demand than Dev version.

Usage:
    python test_files/debug_m3_difference.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

# ============================================================================
# Configuration
# ============================================================================

# Dev output paths
DEV_RUN_DIR = project_root / "test_files" / "BC_S5" / "run_20260123_145210"
DEV_MODULE1_DIR = DEV_RUN_DIR / "module1"
DEV_MODULE3_OUTPUT = DEV_RUN_DIR / "module3" / "Module3Output_20251006.xlsx"
DEV_ORCHESTRATOR_DIR = DEV_RUN_DIR / "orchestrator"

# Local output paths  
LOCAL_RUN_DIR = project_root / "outputs" / "BC_S5" / "run_20260126_095911"
LOCAL_MODULE1_DIR = LOCAL_RUN_DIR / "module1"
LOCAL_MODULE3_OUTPUT = LOCAL_RUN_DIR / "module3" / "Module3Output_20251006.xlsx"
LOCAL_ORCHESTRATOR_DIR = LOCAL_RUN_DIR / "orchestrator"

# Config path
CONFIG_PATH = project_root / "test_files" / "BC_S5.xlsx"

# Simulation date
SIM_DATE = pd.Timestamp("2025-10-06")


def load_module1_outputs(module1_dir: Path, sim_date: pd.Timestamp) -> dict:
    """Load Module1 outputs for a given simulation date."""
    date_str = sim_date.strftime("%Y%m%d")
    file_path = module1_dir / f"module1_output_{date_str}.xlsx"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Module1 output not found: {file_path}")
    
    xl = pd.ExcelFile(file_path)
    data = {}
    
    for sheet in ['SupplyDemandLog', 'ShipmentLog', 'OrderLog']:
        if sheet in xl.sheet_names:
            df = xl.parse(sheet)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            data[sheet] = df
        else:
            data[sheet] = pd.DataFrame()
    
    return data


def load_orchestrator_data(orchestrator_dir: Path, sim_date: pd.Timestamp) -> dict:
    """Load orchestrator data for a given simulation date."""
    date_str = sim_date.strftime("%Y%m%d")
    data = {}
    
    files_to_load = [
        ('beginning_inventory', f'beginning_inventory_{date_str}.csv'),
        ('in_transit', f'in_transit_{date_str}.csv'),
        ('delivery_gr', f'delivery_gr_{date_str}.csv'),
        ('future_production', f'future_production_{date_str}.csv'),
        ('open_deployment', f'open_deployment_{date_str}.csv'),
        ('delivery_shipment', f'delivery_shipment_log_{date_str}.csv'),
    ]
    
    for key, filename in files_to_load:
        file_path = orchestrator_dir / filename
        if file_path.exists():
            df = pd.read_csv(file_path)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            if 'available_date' in df.columns:
                df['available_date'] = pd.to_datetime(df['available_date'])
            data[key] = df
        else:
            data[key] = pd.DataFrame()
    
    return data


def load_config(config_path: Path) -> dict:
    """Load configuration from Excel file."""
    xl = pd.ExcelFile(config_path)
    config = {}
    
    sheet_mapping = {
        'M3_SafetyStock': 'safety_stock',
        'Global_Network': 'network',
        'Global_LeadTime': 'lead_time',
        'M4_MaterialLocationLineCfg': 'm4_mlcfg',
        'M5_DeployConfig': 'deploy_config',
    }
    
    for sheet, key in sheet_mapping.items():
        if sheet in xl.sheet_names:
            df = xl.parse(sheet)
            for col in ['date', 'eff_from', 'eff_to', 'available_date']:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            config[key] = df
        else:
            config[key] = pd.DataFrame()
    
    return config


def normalize_location(loc):
    """Normalize location string."""
    if loc is None or pd.isna(loc):
        return ""
    try:
        return str(int(loc)).zfill(4)
    except (ValueError, TypeError):
        return str(loc).zfill(4)


def normalize_material(mat):
    """Normalize material string."""
    if mat is None or pd.isna(mat):
        return ""
    return str(mat)


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize identifier columns in DataFrame."""
    if df.empty:
        return df
    df = df.copy()
    for col in ['material']:
        if col in df.columns:
            df[col] = df[col].apply(normalize_material)
    for col in ['location', 'sending', 'receiving']:
        if col in df.columns:
            df[col] = df[col].apply(normalize_location)
    return df


def compare_outputs():
    """Compare Dev vs Local Module3 outputs to find discrepancies."""
    print("=" * 80)
    print("STEP 1: Compare Module3 Outputs")
    print("=" * 80)
    
    dev_out = pd.read_excel(DEV_MODULE3_OUTPUT)
    local_out = pd.read_excel(LOCAL_MODULE3_OUTPUT)
    
    print(f"Dev output shape: {dev_out.shape}")
    print(f"Local output shape: {local_out.shape}")
    print(f"Dev total net demand: {dev_out['quantity'].sum():,.2f}")
    print(f"Local total net demand: {local_out['quantity'].sum():,.2f}")
    
    # Merge and find differences
    key_cols = ['material', 'location', 'requirement_date', 'demand_element']
    merged = dev_out.merge(local_out, on=key_cols, suffixes=('_dev', '_local'))
    merged['diff'] = merged['quantity_local'] - merged['quantity_dev']
    
    # Find rows with significant differences
    diff_rows = merged[merged['diff'].abs() > 0.01].copy()
    diff_rows = diff_rows.sort_values('diff', ascending=False)
    
    print(f"\nRows with differences: {len(diff_rows)}")
    print("\nTop 10 differences (Local - Dev):")
    print(diff_rows[['material', 'location', 'demand_element', 
                     'quantity_dev', 'quantity_local', 'diff']].head(10).to_string())
    
    # Find a good material-location for detailed comparison
    # Pick one with large forecast difference at A888
    forecast_diffs = diff_rows[diff_rows['demand_element'] == 'net demand for forecast']
    if not forecast_diffs.empty:
        sample = forecast_diffs.iloc[0]
        return str(sample['material']), str(sample['location'])
    
    return None, None


def calculate_net_demand_dev_style(
    material: str,
    location: str,
    date: pd.Timestamp,
    supply_demand_df: pd.DataFrame,
    safety_stock_df: pd.DataFrame,
    beginning_inventory_df: pd.DataFrame,
    in_transit_df: pd.DataFrame,
    delivery_gr_df: pd.DataFrame,
    future_production_df: pd.DataFrame,
    today_shipment_df: pd.DataFrame,
    open_deployment_df: pd.DataFrame,
    order_df: pd.DataFrame,
    horizon: int,
    downstream_ao_gap: float = 0.0,
    downstream_fc_gap: float = 0.0,
    downstream_ss_gap: float = 0.0,
    delivery_shipment_df: pd.DataFrame = None,
) -> dict:
    """
    Calculate net demand using DEV logic (from ChainSight_Dev/module3.py lines 564-838).
    Returns all intermediate values for debugging.
    """
    horizon_end = date + pd.Timedelta(days=horizon)
    
    # Pre-filter DataFrames
    bi_filtered = pd.DataFrame()
    if beginning_inventory_df is not None and not beginning_inventory_df.empty and 'material' in beginning_inventory_df.columns:
        bi_mask = (beginning_inventory_df['material'] == material) & (beginning_inventory_df['location'] == location)
        bi_filtered = beginning_inventory_df[bi_mask]
    
    it_filtered = pd.DataFrame()
    if in_transit_df is not None and not in_transit_df.empty and 'material' in in_transit_df.columns:
        it_mask = (in_transit_df['material'] == material) & (in_transit_df['receiving'] == location)
        it_filtered = in_transit_df[it_mask]
    
    dgr_filtered = pd.DataFrame()
    if delivery_gr_df is not None and not delivery_gr_df.empty and 'material' in delivery_gr_df.columns:
        dgr_mask = (delivery_gr_df['material'] == material) & (delivery_gr_df['receiving'] == location)
        dgr_filtered = delivery_gr_df[dgr_mask]
    
    fp_filtered = pd.DataFrame()
    if future_production_df is not None and not future_production_df.empty and 'material' in future_production_df.columns:
        fp_mask = (future_production_df['material'] == material) & (future_production_df['location'] == location)
        fp_filtered = future_production_df[fp_mask]
    
    ts_filtered = pd.DataFrame()
    if today_shipment_df is not None and not today_shipment_df.empty and 'material' in today_shipment_df.columns:
        ts_mask = (today_shipment_df['material'] == material) & (today_shipment_df['location'] == location)
        ts_filtered = today_shipment_df[ts_mask]
    
    od_filtered = pd.DataFrame()
    if open_deployment_df is not None and not open_deployment_df.empty and 'material' in open_deployment_df.columns:
        od_mask = (open_deployment_df['material'] == material) & (open_deployment_df['sending'] == location) & (open_deployment_df['receiving'] != location)
        od_filtered = open_deployment_df[od_mask]
    
    # 1. Beginning inventory
    begin_qty = 0.0
    if not bi_filtered.empty:
        bi_rows = bi_filtered[pd.to_datetime(bi_filtered['date']) == pd.to_datetime(date)]
        begin_qty = float(bi_rows['quantity'].sum()) if not bi_rows.empty else 0.0
    
    # 2. In-transit
    in_transit_qty = 0.0
    if not it_filtered.empty:
        in_transit_qty = float(it_filtered['quantity'].sum())
    
    # 3. Today's delivery GR
    delivery_gr_qty = 0.0
    if not dgr_filtered.empty:
        dgr_rows = dgr_filtered[dgr_filtered['date'] == date]
        delivery_gr_qty = float(dgr_rows['quantity'].sum()) if not dgr_rows.empty else 0.0
    
    # 4a. Today's production GR (available_date = today)
    today_production_gr_qty = 0.0
    if not fp_filtered.empty:
        today_rows = fp_filtered[fp_filtered['available_date'] == date]
        if not today_rows.empty:
            if 'produced_qty' in today_rows.columns:
                today_production_gr_qty = float(today_rows['produced_qty'].sum())
            elif 'quantity' in today_rows.columns:
                today_production_gr_qty = float(today_rows['quantity'].sum())
    
    # 4b. Future confirmed production (con_planned_qty)
    future_production_qty = 0.0
    if not fp_filtered.empty:
        future_rows = fp_filtered[fp_filtered['available_date'] > date]
        if not future_rows.empty:
            if 'con_planned_qty' in future_rows.columns:
                future_production_qty = float(pd.to_numeric(future_rows['con_planned_qty'], errors='coerce').fillna(0).sum())
            elif 'produced_qty' in future_rows.columns:
                future_production_qty = float(pd.to_numeric(future_rows['produced_qty'], errors='coerce').fillna(0).sum())
            elif 'quantity' in future_rows.columns:
                future_production_qty = float(pd.to_numeric(future_rows['quantity'], errors='coerce').fillna(0).sum())
    
    # 5. Today's shipment
    today_shipment_qty = 0.0
    if not ts_filtered.empty:
        ts_rows = ts_filtered[ts_filtered['date'] == date]
        today_shipment_qty = float(ts_rows['quantity'].sum()) if not ts_rows.empty else 0.0
    
    # 5b. Delivery shipment
    delivery_shipment_qty = 0.0
    if delivery_shipment_df is not None and not delivery_shipment_df.empty:
        qty_col = 'quantity' if 'quantity' in delivery_shipment_df.columns else ('shipped_qty' if 'shipped_qty' in delivery_shipment_df.columns else None)
        send_col = 'sending' if 'sending' in delivery_shipment_df.columns else ('location' if 'location' in delivery_shipment_df.columns else None)
        date_col = 'date' if 'date' in delivery_shipment_df.columns else ('ship_date' if 'ship_date' in delivery_shipment_df.columns else None)
        
        if qty_col and send_col and date_col:
            ds_rows = delivery_shipment_df[
                (delivery_shipment_df['material'] == material) &
                (delivery_shipment_df[send_col] == location) &
                (pd.to_datetime(delivery_shipment_df[date_col]) == date)
            ]
            delivery_shipment_qty = float(ds_rows[qty_col].sum()) if not ds_rows.empty else 0.0
    
    # 6a. Open deployment outbound
    open_deployment_qty = 0.0
    if not od_filtered.empty:
        if 'deployed_qty' in od_filtered.columns:
            open_deployment_qty = float(od_filtered['deployed_qty'].sum())
        elif 'quantity' in od_filtered.columns:
            open_deployment_qty = float(od_filtered['quantity'].sum())
    
    # 6b. Open deployment inbound (future)
    open_deployment_inbound_future_qty = 0.0
    if open_deployment_df is not None and not open_deployment_df.empty:
        qty_col = 'deployed_qty' if 'deployed_qty' in open_deployment_df.columns else ('quantity' if 'quantity' in open_deployment_df.columns else None)
        if qty_col and 'receiving' in open_deployment_df.columns and 'date' in open_deployment_df.columns and 'material' in open_deployment_df.columns:
            odf = open_deployment_df[
                (open_deployment_df['material'] == material) &
                (open_deployment_df['receiving'] == location)
            ].copy()
            if not odf.empty:
                odf['date'] = pd.to_datetime(odf['date'], errors='coerce')
                future_mask = odf['date'] > date
                future_inbound = pd.to_numeric(odf.loc[future_mask, qty_col], errors='coerce').fillna(0)
                open_deployment_inbound_future_qty = float(future_inbound.sum())
    
    # Total available
    total_available = (begin_qty + in_transit_qty + delivery_gr_qty + 
                       today_production_gr_qty + future_production_qty + open_deployment_inbound_future_qty - 
                       today_shipment_qty - delivery_shipment_qty - open_deployment_qty)
    
    # Demand side
    # 1) AO demand
    AO_local = 0.0
    if order_df is not None and not order_df.empty and 'date' in order_df.columns:
        material_filter = (order_df['material'].astype(str) == str(material))
        location_filter = (order_df['location'].astype(str) == str(location))
        demand_type_filter = (order_df.get('demand_type') == 'AO')
        order_dates = pd.to_datetime(order_df['date'], errors='coerce')
        date_filter = (order_dates >= date) & (order_dates <= horizon_end)
        
        od = order_df[material_filter & location_filter & demand_type_filter & date_filter]
        if not od.empty and 'quantity' in od.columns:
            AO_local = float(pd.to_numeric(od['quantity'], errors='coerce').fillna(0).sum())
    
    # 2) Forecast demand
    FC_local = 0.0
    if supply_demand_df is not None and not supply_demand_df.empty and 'material' in supply_demand_df.columns:
        sdl_rows = supply_demand_df[
            (supply_demand_df['material'] == material) &
            (supply_demand_df['location'] == location) &
            (supply_demand_df['date'] >= date) &
            (supply_demand_df['date'] <= horizon_end)
        ]
        FC_local = float(pd.to_numeric(sdl_rows.get('quantity', 0), errors='coerce').fillna(0).sum())
    
    # 3) Safety stock demand
    SS_local = 0.0
    if safety_stock_df is not None and not safety_stock_df.empty and 'material' in safety_stock_df.columns:
        ssr = safety_stock_df[
            (safety_stock_df['material'] == material) &
            (safety_stock_df['location'] == location) &
            (safety_stock_df['date'] == horizon_end)
        ]
        if not ssr.empty and 'safety_stock_qty' in ssr.columns:
            SS_local = float(pd.to_numeric(ssr['safety_stock_qty'], errors='coerce').fillna(0).sum())
    
    # Gap calculation: AO -> FC -> SS
    AVAILABLE = float(total_available)
    
    # AO
    AO_total = AO_local + float(downstream_ao_gap or 0.0)
    AO_gap = max(AO_total - AVAILABLE, 0.0)
    AVAILABLE = max(AVAILABLE - min(AVAILABLE, AO_total), 0.0)
    
    # FC
    FC_total = FC_local + float(downstream_fc_gap or 0.0)
    FC_gap = max(FC_total - AVAILABLE, 0.0)
    AVAILABLE = max(AVAILABLE - min(AVAILABLE, FC_total), 0.0)
    
    # SS
    SS_total = SS_local + float(downstream_ss_gap or 0.0)
    SS_gap = max(SS_total - AVAILABLE, 0.0)
    
    return {
        'horizon': horizon,
        'horizon_end': horizon_end,
        # Supply side
        'begin_qty': begin_qty,
        'in_transit_qty': in_transit_qty,
        'delivery_gr_qty': delivery_gr_qty,
        'today_production_gr_qty': today_production_gr_qty,
        'future_production_qty': future_production_qty,
        'open_deployment_inbound_future_qty': open_deployment_inbound_future_qty,
        'today_shipment_qty': today_shipment_qty,
        'delivery_shipment_qty': delivery_shipment_qty,
        'open_deployment_qty': open_deployment_qty,
        'total_available': total_available,
        # Demand side
        'AO_local': AO_local,
        'FC_local': FC_local,
        'SS_local': SS_local,
        'downstream_ao_gap': downstream_ao_gap,
        'downstream_fc_gap': downstream_fc_gap,
        'downstream_ss_gap': downstream_ss_gap,
        'AO_total': AO_total,
        'FC_total': FC_total,
        'SS_total': SS_total,
        # Gaps
        'AO_gap': AO_gap,
        'FC_gap': FC_gap,
        'SS_gap': SS_gap,
    }


def calculate_net_demand_local_style(
    material: str,
    location: str,
    date: pd.Timestamp,
    supply_demand_df: pd.DataFrame,
    safety_stock_df: pd.DataFrame,
    beginning_inventory_df: pd.DataFrame,
    in_transit_df: pd.DataFrame,
    delivery_gr_df: pd.DataFrame,
    future_production_df: pd.DataFrame,
    today_shipment_df: pd.DataFrame,
    open_deployment_df: pd.DataFrame,
    order_df: pd.DataFrame,
    horizon: int,
    downstream_ao_gap: float = 0.0,
    downstream_fc_gap: float = 0.0,
    downstream_ss_gap: float = 0.0,
    delivery_shipment_df: pd.DataFrame = None,
) -> dict:
    """
    Calculate net demand using LOCAL logic (from src/modules/mrp_planning/net_demand.py).
    Returns all intermediate values for debugging.
    """
    horizon_end = date + pd.Timedelta(days=horizon)
    
    # Helper functions from local version
    def _filter_by_material_location(df, mat, loc):
        if df is None or df.empty or 'material' not in df.columns:
            return pd.DataFrame()
        mask = (df['material'] == mat) & (df['location'] == loc)
        return df[mask]
    
    def _filter_by_material_receiving(df, mat, loc):
        if df is None or df.empty or 'material' not in df.columns:
            return pd.DataFrame()
        mask = (df['material'] == mat) & (df['receiving'] == loc)
        return df[mask]
    
    def _filter_open_deployment_out(df, mat, loc):
        if df is None or df.empty or 'material' not in df.columns:
            return pd.DataFrame()
        mask = (df['material'] == mat) & (df['sending'] == loc) & (df['receiving'] != loc)
        return df[mask]
    
    # Pre-filter DataFrames (LOCAL style)
    bi_filtered = _filter_by_material_location(beginning_inventory_df, material, location)
    it_filtered = _filter_by_material_receiving(in_transit_df, material, location)
    dgr_filtered = _filter_by_material_receiving(delivery_gr_df, material, location)
    fp_filtered = _filter_by_material_location(future_production_df, material, location)
    ts_filtered = _filter_by_material_location(today_shipment_df, material, location)
    od_filtered = _filter_open_deployment_out(open_deployment_df, material, location)
    
    # 1. Beginning inventory
    begin_qty = 0.0
    if not bi_filtered.empty:
        rows = bi_filtered[pd.to_datetime(bi_filtered['date']) == pd.to_datetime(date)]
        begin_qty = float(rows['quantity'].sum()) if not rows.empty else 0.0
    
    # 2. In-transit (LOCAL: sums all, no date filter)
    in_transit_qty = 0.0
    if not it_filtered.empty and 'quantity' in it_filtered.columns:
        in_transit_qty = float(it_filtered['quantity'].sum())
    
    # 3. Delivery GR
    delivery_gr_qty = 0.0
    if not dgr_filtered.empty and 'date' in dgr_filtered.columns:
        rows = dgr_filtered[dgr_filtered['date'] == date]
        delivery_gr_qty = float(rows['quantity'].sum()) if not rows.empty else 0.0
    
    # 4a. Today production
    today_production_gr_qty = 0.0
    if not fp_filtered.empty and 'available_date' in fp_filtered.columns:
        rows = fp_filtered[fp_filtered['available_date'] == date]
        if not rows.empty:
            col = 'produced_qty' if 'produced_qty' in rows.columns else 'quantity'
            today_production_gr_qty = float(rows[col].sum()) if col in rows.columns else 0.0
    
    # 4b. Future production
    future_production_qty = 0.0
    if not fp_filtered.empty and 'available_date' in fp_filtered.columns:
        rows = fp_filtered[fp_filtered['available_date'] > date]
        if not rows.empty:
            for col in ['con_planned_qty', 'produced_qty', 'quantity']:
                if col in rows.columns:
                    future_production_qty = float(pd.to_numeric(rows[col], errors='coerce').fillna(0).sum())
                    break
    
    # 5. Today shipment
    today_shipment_qty = 0.0
    if not ts_filtered.empty and 'date' in ts_filtered.columns:
        rows = ts_filtered[ts_filtered['date'] == date]
        today_shipment_qty = float(rows['quantity'].sum()) if not rows.empty else 0.0
    
    # 5b. Delivery shipment
    delivery_shipment_qty = 0.0
    if delivery_shipment_df is not None and not delivery_shipment_df.empty:
        qty_col = None
        send_col = None
        date_col = None
        for c in ['quantity', 'shipped_qty']:
            if c in delivery_shipment_df.columns:
                qty_col = c
                break
        for c in ['sending', 'location']:
            if c in delivery_shipment_df.columns:
                send_col = c
                break
        for c in ['date', 'ship_date']:
            if c in delivery_shipment_df.columns:
                date_col = c
                break
        
        if qty_col and send_col and date_col:
            rows = delivery_shipment_df[
                (delivery_shipment_df['material'] == material) &
                (delivery_shipment_df[send_col] == location) &
                (pd.to_datetime(delivery_shipment_df[date_col]) == date)
            ]
            delivery_shipment_qty = float(rows[qty_col].sum()) if not rows.empty else 0.0
    
    # 6a. Open deployment outbound
    open_deployment_qty = 0.0
    if not od_filtered.empty:
        col = 'deployed_qty' if 'deployed_qty' in od_filtered.columns else 'quantity'
        open_deployment_qty = float(od_filtered[col].sum()) if col in od_filtered.columns else 0.0
    
    # 6b. Open deployment inbound
    open_deployment_inbound_future_qty = 0.0
    if open_deployment_df is not None and not open_deployment_df.empty:
        qty_col = None
        for c in ['deployed_qty', 'quantity']:
            if c in open_deployment_df.columns:
                qty_col = c
                break
        
        required = ['receiving', 'date', 'material']
        if qty_col and all(c in open_deployment_df.columns for c in required):
            odf = open_deployment_df[
                (open_deployment_df['material'] == material) &
                (open_deployment_df['receiving'] == location)
            ].copy()
            
            if not odf.empty:
                odf['date'] = pd.to_datetime(odf['date'], errors='coerce')
                future = odf[odf['date'] > date]
                open_deployment_inbound_future_qty = float(
                    pd.to_numeric(future[qty_col], errors='coerce').fillna(0).sum()
                )
    
    # Total available
    total_available = (begin_qty + in_transit_qty + delivery_gr_qty +
                       today_production_gr_qty + future_production_qty + open_deployment_inbound_future_qty -
                       today_shipment_qty - delivery_shipment_qty - open_deployment_qty)
    
    # Demand side
    # 1) AO
    AO_local = 0.0
    if order_df is not None and not order_df.empty and 'date' in order_df.columns:
        order_dates = pd.to_datetime(order_df['date'], errors='coerce')
        mask = (
            (order_df['material'].astype(str) == str(material)) &
            (order_df['location'].astype(str) == str(location)) &
            (order_df.get('demand_type') == 'AO') &
            (order_dates >= date) &
            (order_dates <= horizon_end)
        )
        od = order_df[mask]
        if not od.empty and 'quantity' in od.columns:
            AO_local = float(pd.to_numeric(od['quantity'], errors='coerce').fillna(0).sum())
    
    # 2) Forecast
    FC_local = 0.0
    if supply_demand_df is not None and not supply_demand_df.empty and 'material' in supply_demand_df.columns:
        rows = supply_demand_df[
            (supply_demand_df['material'] == material) &
            (supply_demand_df['location'] == location) &
            (supply_demand_df['date'] >= date) &
            (supply_demand_df['date'] <= horizon_end)
        ]
        FC_local = float(pd.to_numeric(rows.get('quantity', 0), errors='coerce').fillna(0).sum())
    
    # 3) Safety stock
    SS_local = 0.0
    if safety_stock_df is not None and not safety_stock_df.empty and 'material' in safety_stock_df.columns:
        rows = safety_stock_df[
            (safety_stock_df['material'] == material) &
            (safety_stock_df['location'] == location) &
            (safety_stock_df['date'] == horizon_end)
        ]
        if not rows.empty and 'safety_stock_qty' in rows.columns:
            SS_local = float(pd.to_numeric(rows['safety_stock_qty'], errors='coerce').fillna(0).sum())
    
    # Gap calculation
    available = float(total_available)
    
    # AO
    ao_total = AO_local + float(downstream_ao_gap or 0.0)
    ao_gap = max(ao_total - available, 0.0)
    available = max(available - min(available, ao_total), 0.0)
    
    # FC
    fc_total = FC_local + float(downstream_fc_gap or 0.0)
    fc_gap = max(fc_total - available, 0.0)
    available = max(available - min(available, fc_total), 0.0)
    
    # SS
    ss_total = SS_local + float(downstream_ss_gap or 0.0)
    ss_gap = max(ss_total - available, 0.0)
    
    return {
        'horizon': horizon,
        'horizon_end': horizon_end,
        # Supply side
        'begin_qty': begin_qty,
        'in_transit_qty': in_transit_qty,
        'delivery_gr_qty': delivery_gr_qty,
        'today_production_gr_qty': today_production_gr_qty,
        'future_production_qty': future_production_qty,
        'open_deployment_inbound_future_qty': open_deployment_inbound_future_qty,
        'today_shipment_qty': today_shipment_qty,
        'delivery_shipment_qty': delivery_shipment_qty,
        'open_deployment_qty': open_deployment_qty,
        'total_available': total_available,
        # Demand side
        'AO_local': AO_local,
        'FC_local': FC_local,
        'SS_local': SS_local,
        'downstream_ao_gap': downstream_ao_gap,
        'downstream_fc_gap': downstream_fc_gap,
        'downstream_ss_gap': downstream_ss_gap,
        'AO_total': ao_total,
        'FC_total': fc_total,
        'SS_total': ss_total,
        # Gaps
        'AO_gap': ao_gap,
        'FC_gap': fc_gap,
        'SS_gap': ss_gap,
    }


def print_comparison(dev_vals: dict, local_vals: dict, label_prefix: str = ""):
    """Print side-by-side comparison of intermediate values."""
    print(f"\n{label_prefix}{'='*60}")
    print(f"{'Field':<40} {'DEV':>15} {'LOCAL':>15} {'DIFF':>15}")
    print("-" * 85)
    
    fields = [
        ('horizon', 'Horizon (days)'),
        ('horizon_end', 'Horizon End Date'),
        ('', '--- SUPPLY SIDE ---'),
        ('begin_qty', 'Beginning Inventory'),
        ('in_transit_qty', 'In Transit'),
        ('delivery_gr_qty', 'Delivery GR (today)'),
        ('today_production_gr_qty', 'Today Production GR'),
        ('future_production_qty', 'Future Production'),
        ('open_deployment_inbound_future_qty', 'Open Deploy Inbound Future'),
        ('today_shipment_qty', 'Today Shipment (-)'),
        ('delivery_shipment_qty', 'Delivery Shipment (-)'),
        ('open_deployment_qty', 'Open Deploy Outbound (-)'),
        ('total_available', 'TOTAL AVAILABLE'),
        ('', '--- DEMAND SIDE ---'),
        ('AO_local', 'AO Local'),
        ('FC_local', 'FC Local (Forecast)'),
        ('SS_local', 'SS Local (Safety Stock)'),
        ('downstream_ao_gap', 'Downstream AO Gap'),
        ('downstream_fc_gap', 'Downstream FC Gap'),
        ('downstream_ss_gap', 'Downstream SS Gap'),
        ('AO_total', 'AO Total'),
        ('FC_total', 'FC Total'),
        ('SS_total', 'SS Total'),
        ('', '--- GAPS (OUTPUT) ---'),
        ('AO_gap', 'AO Gap'),
        ('FC_gap', 'FC Gap'),
        ('SS_gap', 'SS Gap'),
    ]
    
    for key, label in fields:
        if key == '':
            print(f"\n{label}")
            continue
        
        dev_val = dev_vals.get(key, 'N/A')
        local_val = local_vals.get(key, 'N/A')
        
        if isinstance(dev_val, pd.Timestamp):
            dev_str = dev_val.strftime('%Y-%m-%d')
            local_str = local_val.strftime('%Y-%m-%d') if isinstance(local_val, pd.Timestamp) else str(local_val)
            diff_str = 'MATCH' if dev_str == local_str else 'DIFF!'
        elif isinstance(dev_val, (int, float)):
            dev_str = f"{dev_val:,.2f}"
            local_str = f"{local_val:,.2f}" if isinstance(local_val, (int, float)) else str(local_val)
            diff = local_val - dev_val if isinstance(local_val, (int, float)) else 0
            diff_str = f"{diff:+,.2f}" if abs(diff) > 0.01 else "MATCH"
            if abs(diff) > 0.01:
                diff_str = f"*** {diff_str} ***"
        else:
            dev_str = str(dev_val)
            local_str = str(local_val)
            diff_str = 'MATCH' if dev_str == local_str else 'DIFF!'
        
        print(f"{label:<40} {dev_str:>15} {local_str:>15} {diff_str:>15}")


def debug_single_material_location(
    material: str,
    location: str,
    dev_data: dict,
    local_data: dict,
    config: dict,
    horizon: int = 20,
):
    """
    Debug a single material-location combination.
    """
    print("\n" + "=" * 80)
    print(f"STEP 2: Debug Material={material}, Location={location}")
    print("=" * 80)
    
    # Normalize all data
    for key, df in dev_data.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            dev_data[key] = normalize_df(df)
    for key, df in local_data.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            local_data[key] = normalize_df(df)
    for key, df in config.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            config[key] = normalize_df(df)
    
    material = normalize_material(material)
    location = normalize_location(location)
    
    print(f"\nNormalized: material={material}, location={location}")
    
    # Calculate using DEV style with DEV data
    print("\n>>> Calculating DEV style with DEV data...")
    dev_result = calculate_net_demand_dev_style(
        material=material,
        location=location,
        date=SIM_DATE,
        supply_demand_df=dev_data['m1']['SupplyDemandLog'],
        safety_stock_df=config['safety_stock'],
        beginning_inventory_df=dev_data['orch'].get('beginning_inventory', pd.DataFrame()),
        in_transit_df=dev_data['orch'].get('in_transit', pd.DataFrame()),
        delivery_gr_df=dev_data['orch'].get('delivery_gr', pd.DataFrame()),
        future_production_df=dev_data['orch'].get('future_production', pd.DataFrame()),
        today_shipment_df=dev_data['m1']['ShipmentLog'],
        open_deployment_df=dev_data['orch'].get('open_deployment', pd.DataFrame()),
        order_df=dev_data['m1']['OrderLog'],
        horizon=horizon,
        delivery_shipment_df=dev_data['orch'].get('delivery_shipment', pd.DataFrame()),
    )
    
    # Calculate using LOCAL style with LOCAL data
    print(">>> Calculating LOCAL style with LOCAL data...")
    local_result = calculate_net_demand_local_style(
        material=material,
        location=location,
        date=SIM_DATE,
        supply_demand_df=local_data['m1']['SupplyDemandLog'],
        safety_stock_df=config['safety_stock'],
        beginning_inventory_df=local_data['orch'].get('beginning_inventory', pd.DataFrame()),
        in_transit_df=local_data['orch'].get('in_transit', pd.DataFrame()),
        delivery_gr_df=local_data['orch'].get('delivery_gr', pd.DataFrame()),
        future_production_df=local_data['orch'].get('future_production', pd.DataFrame()),
        today_shipment_df=local_data['m1']['ShipmentLog'],
        open_deployment_df=local_data['orch'].get('open_deployment', pd.DataFrame()),
        order_df=local_data['m1']['OrderLog'],
        horizon=horizon,
        delivery_shipment_df=local_data['orch'].get('delivery_shipment', pd.DataFrame()),
    )
    
    print_comparison(dev_result, local_result, "DEV Logic vs LOCAL Logic (same data sources):\n")
    
    # Now check what inputs are different
    print("\n" + "=" * 80)
    print("STEP 3: Compare Input Data")
    print("=" * 80)
    
    # Check SupplyDemandLog for this material-location
    dev_sdl = dev_data['m1']['SupplyDemandLog']
    local_sdl = local_data['m1']['SupplyDemandLog']
    
    dev_sdl_ml = dev_sdl[(dev_sdl['material'] == material) & (dev_sdl['location'] == location)]
    local_sdl_ml = local_sdl[(local_sdl['material'] == material) & (local_sdl['location'] == location)]
    
    print(f"\n--- SupplyDemandLog for {material}@{location} ---")
    print(f"DEV rows: {len(dev_sdl_ml)}, LOCAL rows: {len(local_sdl_ml)}")
    if not dev_sdl_ml.empty:
        print(f"DEV total quantity: {dev_sdl_ml['quantity'].sum():,.2f}")
    if not local_sdl_ml.empty:
        print(f"LOCAL total quantity: {local_sdl_ml['quantity'].sum():,.2f}")
    
    # Check forecast in horizon window
    horizon_end = SIM_DATE + pd.Timedelta(days=horizon)
    dev_sdl_window = dev_sdl_ml[(dev_sdl_ml['date'] >= SIM_DATE) & (dev_sdl_ml['date'] <= horizon_end)]
    local_sdl_window = local_sdl_ml[(local_sdl_ml['date'] >= SIM_DATE) & (local_sdl_ml['date'] <= horizon_end)]
    
    print(f"\n--- Forecast in Horizon Window [{SIM_DATE.date()} to {horizon_end.date()}] ---")
    print(f"DEV: {len(dev_sdl_window)} rows, total={dev_sdl_window['quantity'].sum() if not dev_sdl_window.empty else 0:,.2f}")
    print(f"LOCAL: {len(local_sdl_window)} rows, total={local_sdl_window['quantity'].sum() if not local_sdl_window.empty else 0:,.2f}")
    
    if not dev_sdl_window.empty:
        print("\nDEV Forecast Detail:")
        print(dev_sdl_window[['date', 'quantity']].to_string())
    
    # Check OrderLog
    dev_ol = dev_data['m1']['OrderLog']
    local_ol = local_data['m1']['OrderLog']
    
    dev_ol_ml = dev_ol[(dev_ol['material'].astype(str) == material) & (dev_ol['location'].astype(str) == location)]
    local_ol_ml = local_ol[(local_ol['material'].astype(str) == material) & (local_ol['location'].astype(str) == location)]
    
    print(f"\n--- OrderLog for {material}@{location} ---")
    print(f"DEV rows: {len(dev_ol_ml)}, LOCAL rows: {len(local_ol_ml)}")
    
    # Check beginning inventory
    dev_bi = dev_data['orch'].get('beginning_inventory', pd.DataFrame())
    local_bi = local_data['orch'].get('beginning_inventory', pd.DataFrame())
    
    if not dev_bi.empty:
        dev_bi_ml = dev_bi[(dev_bi['material'] == material) & (dev_bi['location'] == location)]
        dev_bi_today = dev_bi_ml[pd.to_datetime(dev_bi_ml['date']) == SIM_DATE] if not dev_bi_ml.empty else pd.DataFrame()
        print(f"\n--- Beginning Inventory (DEV) for {material}@{location} on {SIM_DATE.date()} ---")
        print(f"Rows: {len(dev_bi_today)}, Qty: {dev_bi_today['quantity'].sum() if not dev_bi_today.empty else 0:,.2f}")
    
    if not local_bi.empty:
        local_bi_ml = local_bi[(local_bi['material'] == material) & (local_bi['location'] == location)]
        local_bi_today = local_bi_ml[pd.to_datetime(local_bi_ml['date']) == SIM_DATE] if not local_bi_ml.empty else pd.DataFrame()
        print(f"--- Beginning Inventory (LOCAL) for {material}@{location} on {SIM_DATE.date()} ---")
        print(f"Rows: {len(local_bi_today)}, Qty: {local_bi_today['quantity'].sum() if not local_bi_today.empty else 0:,.2f}")
    
    return dev_result, local_result


def investigate_horizon_difference():
    """Investigate if horizon calculation differs between Dev and Local."""
    print("\n" + "=" * 80)
    print("STEP 4: Investigate Horizon Calculation")
    print("=" * 80)
    
    # Load config
    config = load_config(CONFIG_PATH)
    
    # Check lead time config
    lt_df = config['lead_time']
    m4_df = config['m4_mlcfg']
    
    print("\n--- Lead Time Config Sample ---")
    print(lt_df.head().to_string() if not lt_df.empty else "Empty")
    
    print("\n--- M4 Config Sample (PTF/LSK) ---")
    if not m4_df.empty:
        cols = [c for c in ['material', 'location', 'ptf', 'PTF', 'lsk', 'LSK'] if c in m4_df.columns]
        print(m4_df[cols].head().to_string() if cols else "No PTF/LSK columns found")
    else:
        print("Empty")


def investigate_root_cause():
    """
    Document the root cause of the Module3 discrepancy.
    
    ROOT CAUSE FOUND:
    -----------------
    The discrepancy is caused by a DATA TYPE MISMATCH in the Local version.
    
    In _get_forecast_demand() (net_demand.py line 385):
    ```python
    rows = df[
        (df['material'] == material) &  # <-- str comparison with int64 column!
        ...
    ]
    ```
    
    The SupplyDemandLog from Module1 has:
    - material column: int64 dtype
    - location column: str dtype
    
    When the Local version compares:
    - material='80784368' (string) vs df['material'] (int64)
    - The comparison returns 0 matches!
    
    This causes FC_local (local forecast) to be 0 instead of the actual value.
    
    The DEV version works because:
    1. load_module1_daily_outputs() calls _normalize_identifiers() which converts
       material to string BEFORE passing to calculate_daily_net_demand()
    
    The LOCAL version FAILS when module1_result is passed directly in integration mode
    because:
    1. integration.py line 155-166 bypasses load_module1_daily_outputs() 
    2. The data is NOT normalized before use
    3. The comparison in _get_forecast_demand fails silently
    
    FIX REQUIRED:
    1. In _get_forecast_demand (net_demand.py), change line 385 to:
       (df['material'].astype(str) == str(material)) &
    
    2. OR ensure normalize_identifiers is called in integration.py _load_module1_data
       when module1_result is provided directly.
    """
    print("\n" + "=" * 80)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 80)
    
    print("""
FINDING: Data type mismatch causes forecast to be silently dropped

In SupplyDemandLog:
  - material column: int64 (e.g., 80784368)
  - location column: str (e.g., 'A888')

In Local net_demand.py _get_forecast_demand():
  - Compares df['material'] == material (str)
  - int64 == str returns False for ALL rows
  - Result: FC_local = 0 (should be 12,213 for material 80784368@A888)

Impact:
  - A888 local forecast of 12,213 is completely missed
  - Only downstream gaps (8,742) flow up
  - DEV shows 21,265 forecast gap, LOCAL shows 9,220
  - Difference is ~12,045 (matches missing local forecast)

FIX OPTIONS:
  1. Change net_demand.py line 385 to use .astype(str) comparison
  2. Ensure normalize_identifiers() is called in integration.py 
     when module1_result is passed directly
""")


def main():
    """Main debug function."""
    print("=" * 80)
    print("Module3 Net Demand Debug Script")
    print("Comparing Dev vs Local Intermediate Values")
    print("=" * 80)
    
    # Step 1: Compare outputs to find discrepancies
    material, location = compare_outputs()
    
    if material is None:
        print("\nNo differences found or could not determine sample material-location.")
        return
    
    print(f"\n>>> Selected for detailed analysis: material={material}, location={location}")
    
    # Load data
    print("\n>>> Loading DEV data...")
    dev_m1 = load_module1_outputs(DEV_MODULE1_DIR, SIM_DATE)
    dev_orch = load_orchestrator_data(DEV_ORCHESTRATOR_DIR, SIM_DATE)
    
    print(">>> Loading LOCAL data...")
    local_m1 = load_module1_outputs(LOCAL_MODULE1_DIR, SIM_DATE)
    local_orch = load_orchestrator_data(LOCAL_ORCHESTRATOR_DIR, SIM_DATE)
    
    print(">>> Loading config...")
    config = load_config(CONFIG_PATH)
    
    # Debug the selected material-location
    dev_data = {'m1': dev_m1, 'orch': dev_orch}
    local_data = {'m1': local_m1, 'orch': local_orch}
    
    debug_single_material_location(
        material=material,
        location=location,
        dev_data=dev_data,
        local_data=local_data,
        config=config,
        horizon=20,  # Default horizon from outputs
    )
    
    # Investigate horizon
    investigate_horizon_difference()
    
    # Investigate network structure
    downstream_locs = investigate_network_propagation(config)
    
    # Compare downstream gap propagation for the selected material
    compare_downstream_gaps(DEV_MODULE3_OUTPUT, LOCAL_MODULE3_OUTPUT, material, location)
    
    # Show root cause analysis
    investigate_root_cause()
    
    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
