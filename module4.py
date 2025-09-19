#module 4
import pandas as pd
import numpy as np
from datetime import timedelta
import argparse
import os
from typing import Optional


def get_or_init_simulation_start(output_dir: str, provided_start: Optional[pd.Timestamp]) -> pd.Timestamp:
    """Load persistent simulation start date or initialize it.

    During the first Module4 run we persist the provided start date to a
    small state file in ``output_dir``. Subsequent runs reuse the stored
    value so that review-day calculations remain consistent.

    Args:
        output_dir: Directory for Module4 outputs/state files.
        provided_start: Start date supplied by the user (optional after the
            first run).

    Returns:
        pd.Timestamp: The persisted simulation start date.
    """
    state_file = os.path.join(output_dir, "simulation_start.txt")

    # If a state file exists, always use it
    if os.path.exists(state_file):
        try:
            with open(state_file, "r") as f:
                return pd.to_datetime(f.read().strip())
        except Exception as e:
            raise ValueError(f"Failed to read simulation start from {state_file}: {e}")

    # No state file yet – require a provided start date and persist it
    if provided_start is None:
        raise ValueError("Simulation start date not provided and state file not found")

    os.makedirs(output_dir, exist_ok=True)
    with open(state_file, "w") as f:
        f.write(provided_start.strftime("%Y-%m-%d"))

    return provided_start


def save_line_state(output_dir: str, simulation_date: pd.Timestamp, line_states: dict):
    """Save line states (last material and remaining changeover time) for cross-day continuity.
    
    Args:
        output_dir: Directory for Module4 outputs/state files
        simulation_date: Current simulation date
        line_states: Dict with line keys containing {'last_material': str, 'remaining_changeover': float}
    """
    os.makedirs(output_dir, exist_ok=True)
    state_file = os.path.join(output_dir, f"line_states_{simulation_date.strftime('%Y%m%d')}.json")
    
    import json
    with open(state_file, "w") as f:
        json.dump(line_states, f, indent=2)


def save_allocated_capacity(output_dir: str, simulation_date: pd.Timestamp, allocated_capacity: dict):
    """Save allocated capacity for future production dates to track capacity usage across simulation dates.
    
    Args:
        output_dir: Directory for Module4 outputs/state files
        simulation_date: Current simulation date
        allocated_capacity: Dict with capacity allocation info for future production dates
    """
    os.makedirs(output_dir, exist_ok=True)
    capacity_file = os.path.join(output_dir, f"allocated_capacity_{simulation_date.strftime('%Y%m%d')}.json")
    
    import json
    with open(capacity_file, "w") as f:
        json.dump(allocated_capacity, f, indent=2)


def load_allocated_capacity(output_dir: str, simulation_date: pd.Timestamp) -> dict:
    """Load previously allocated capacity for future production dates.
    
    Args:
        output_dir: Directory for Module4 outputs/state files
        simulation_date: Current simulation date
        
    Returns:
        dict: Previously allocated capacity, empty dict if not found
    """
    capacity_file = os.path.join(output_dir, f"allocated_capacity_{simulation_date.strftime('%Y%m%d')}.json")
    
    if not os.path.exists(capacity_file):
        return {}
    
    try:
        import json
        with open(capacity_file, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load allocated capacity from {capacity_file}: {e}")
        return {}


def load_all_previous_capacity(output_dir: str, simulation_date: pd.Timestamp) -> dict:
    """Load all previously allocated capacity from all previous simulation dates.
    
    Args:
        output_dir: Directory for Module4 outputs/state files
        simulation_date: Current simulation date
        
    Returns:
        dict: Consolidated capacity allocation from all previous simulation dates
    """
    consolidated_capacity = {}
    
    # Look for all capacity files from previous simulation dates
    for file_name in os.listdir(output_dir):
        if file_name.startswith("allocated_capacity_") and file_name.endswith(".json"):
            try:
                # Extract date from filename
                date_str = file_name.replace("allocated_capacity_", "").replace(".json", "")
                file_date = pd.to_datetime(date_str, format='%Y%m%d')
                
                # Only load capacity from previous simulation dates
                if file_date < simulation_date:
                    capacity_file = os.path.join(output_dir, file_name)
                    with open(capacity_file, "r") as f:
                        import json
                        daily_capacity = json.load(f)
                        
                        # Merge into consolidated capacity
                        for key, value in daily_capacity.items():
                            if key not in consolidated_capacity:
                                consolidated_capacity[key] = 0
                            consolidated_capacity[key] += value
                            
            except Exception as e:
                print(f"Warning: Failed to load capacity from {file_name}: {e}")
                continue
    
    return consolidated_capacity


def extract_allocated_capacity_from_plan(plan_df: pd.DataFrame, rate_map: dict, changeover_def: dict = None) -> dict:
    """Extract allocated capacity information from production plan for persistence.
    
    Args:
        plan_df: Production plan DataFrame
        rate_map: Production rate mapping (material, line) -> rate
        changeover_def: Changeover definition mapping (changeover_id, line) -> time
        
    Returns:
        dict: Capacity allocation info in HOURS keyed by (location, line, production_plan_date)
        Includes both production time and changeover time
    """
    allocated_capacity = {}
    
    if plan_df.empty:
        return allocated_capacity
    
    # Group by location, line, and production_plan_date to get allocated capacity
    for (location, line, prod_date), group in plan_df.groupby(['location', 'line', 'production_plan_date']):
        # Calculate total allocated capacity in HOURS (production + changeover)
        total_allocated_hours = 0
        
        for _, row in group.iterrows():
            material = row['material']
            quantity = row['con_planned_qty']
            changeover_id = row.get('changeover_id')
            
            # Calculate production time
            rate = rate_map.get((material, line), 1)
            production_hours = quantity / rate if rate else 0
            total_allocated_hours += production_hours
            
            # Add changeover time if changeover_id exists
            if changeover_id and changeover_def and pd.notna(changeover_id):
                changeover_hours = changeover_def.get((changeover_id, line), 0)
                total_allocated_hours += changeover_hours
        
        # Convert to Python float for JSON serialization
        if isinstance(total_allocated_hours, (np.integer, np.int64, np.floating)):
            total_allocated_hours = float(total_allocated_hours)
        
        # Create key for capacity tracking
        key = f"{location}|{line}|{prod_date.strftime('%Y-%m-%d')}"
        allocated_capacity[key] = total_allocated_hours
    
    return allocated_capacity


def validate_capacity_allocation(plan_log: pd.DataFrame, previously_allocated_capacity: dict, 
                                simulation_date: pd.Timestamp, rate_map: dict, changeover_def: dict = None) -> list:
    """Validate that capacity allocation respects previously allocated capacity.
    
    Args:
        plan_log: Current production plan DataFrame
        previously_allocated_capacity: Previously allocated capacity dict (in hours)
        simulation_date: Current simulation date
        rate_map: Production rate mapping (material, line) -> rate
        changeover_def: Changeover definition mapping (changeover_id, line) -> time
        
    Returns:
        list: Validation issues found
    """
    issues = []
    
    if plan_log.empty or not previously_allocated_capacity:
        return issues
    
    # Group current plan by location, line, and production_plan_date
    for (location, line, prod_date), group in plan_log.groupby(['location', 'line', 'production_plan_date']):
        # Calculate current allocated capacity in hours (production + changeover)
        current_allocated_hours = 0
        for _, row in group.iterrows():
            material = row['material']
            quantity = row['con_planned_qty']
            changeover_id = row.get('changeover_id')
            
            # Calculate production time
            rate = rate_map.get((material, line), 1)
            production_hours = quantity / rate if rate else 0
            current_allocated_hours += production_hours
            
            # Add changeover time if changeover_id exists
            if changeover_id and changeover_def and pd.notna(changeover_id):
                changeover_hours = changeover_def.get((changeover_id, line), 0)
                current_allocated_hours += changeover_hours
        
        capacity_key = f"{location}|{line}|{prod_date.strftime('%Y-%m-%d')}"
        previously_allocated_hours = previously_allocated_capacity.get(capacity_key, 0)
        
        if previously_allocated_hours > 0:
            issues.append({
                'type': 'capacity_validation',
                'location': location,
                'line': line,
                'production_plan_date': prod_date.strftime('%Y-%m-%d'),
                'simulation_date': simulation_date.strftime('%Y-%m-%d'),
                'previously_allocated_hours': previously_allocated_hours,
                'currently_allocated_hours': current_allocated_hours,
                'total_allocated_hours': previously_allocated_hours + current_allocated_hours,
                'message': f"Capacity allocation validation: {location}/{line} on {prod_date.strftime('%Y-%m-%d')} - Previously: {previously_allocated_hours:.2f} hours, Currently: {current_allocated_hours:.2f} hours, Total: {previously_allocated_hours + current_allocated_hours:.2f} hours"
            })
    
    return issues


def load_line_state(output_dir: str, simulation_date: pd.Timestamp) -> dict:
    """Load line states from previous day for cross-day changeover continuity.
    
    Args:
        output_dir: Directory for Module4 outputs/state files
        simulation_date: Current simulation date
        
    Returns:
        dict: Line states from previous day, empty dict if not found
    """
    prev_date = simulation_date - pd.Timedelta(days=1)
    state_file = os.path.join(output_dir, f"line_states_{prev_date.strftime('%Y%m%d')}.json")
    
    if not os.path.exists(state_file):
        return {}
    
    try:
        import json
        with open(state_file, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load line state from {state_file}: {e}")
        return {}


def extract_line_states_from_plan(plan_df: pd.DataFrame) -> dict:
    """Extract line states from production plan for state persistence.
    
    Args:
        plan_df: Production plan DataFrame
        
    Returns:
        dict: Line states with last material and remaining changeover info
    """
    line_states = {}
    
    if plan_df.empty:
        return line_states
    
    # Group by line and simulation_date to get the last production for each line
    for (line, sim_date), group in plan_df.groupby(['line', 'simulation_date']):
        # Sort by production_plan_date to get the last production
        last_production = group.sort_values('production_plan_date').iloc[-1]
        
        line_states[line] = {
            'last_material': str(last_production['material']),
            'last_location': str(last_production['location']),
            'last_production_date': last_production['production_plan_date'].strftime('%Y-%m-%d')
        }
    
    return line_states

IDENTIFIER_COLS = [
    'material', 'location', 'line', 'delegate_line', 'from_material', 'to_material'
]


def _normalize_location(location_str: str) -> str:
    """Normalize location string by padding with leading zeros to 4 digits.
    
    Args:
        location_str: Location string (e.g., "386" or "0386")
        
    Returns:
        str: Normalized location with leading zeros (e.g., "0386")
    """
    try:
        # Convert to int and back to string with 4-digit zero padding
        return str(int(location_str)).zfill(4)
    except (ValueError, TypeError):
        # If conversion fails, return original string with zero padding
        return str(location_str).zfill(4)


def _cast_identifiers_to_str(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    """Cast identifier columns to pandas string dtype and normalize locations.

    Args:
        df: DataFrame to process
        cols: Optional list of columns; defaults to IDENTIFIER_COLS

    Returns:
        DataFrame with specified columns cast to string dtype and normalized
    """
    cols = cols or IDENTIFIER_COLS
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype('string')
            # Normalize location column specifically
            if c == 'location':
                df[c] = df[c].apply(_normalize_location)
    return df


def _validate_merge_keys(df1: pd.DataFrame, df2: pd.DataFrame, keys):
    """Validate that merge keys share the same dtype in both DataFrames."""
    for k in keys:
        if k in df1.columns and k in df2.columns:
            if df1[k].dtype != df2[k].dtype:
                raise TypeError(
                    f"Merge key '{k}' has mismatched dtypes: {df1[k].dtype} vs {df2[k].dtype}"
                )

def load_daily_net_demand(module3_output_dir: str, simulation_date: pd.Timestamp) -> pd.DataFrame:
    """
    Load NetDemand from Module3 daily output, filter layer=0 and convert negative to positive
    按照数据流规范，Module4读取前一天Module3产生的NetDemand数据
    
    Args:
        module3_output_dir: Directory containing Module3 daily output files
        simulation_date: The simulation date to load data for
        
    Returns:
        DataFrame: Filtered and processed NetDemand data
    """
    try:
        # 按照设计逻辑：Module4读取前一天的Module3输出
        # Module3的requirement_date = simulation_date + 1
        # Module4读取前一天文件，所以requirement_date = 当前simulation_date
        prev_date = simulation_date - pd.Timedelta(days=1)
        prev_date_str = prev_date.strftime('%Y%m%d')
        net_demand_file = os.path.join(module3_output_dir, f"Module3Output_{prev_date_str}.xlsx")
        
        if not os.path.exists(net_demand_file):
            # 第一天没有前一天的Module3输出，这是正常的
            # 按照设计，第一天应该没有生产计划
            print(f"Info: No previous day Module3 output found for {prev_date_str}. This is expected for the first day.")
            return pd.DataFrame(columns=['material', 'location', 'requirement_date', 'quantity', 'demand_type', 'layer'])
        
        # Load NetDemand sheet
        xl = pd.ExcelFile(net_demand_file)
        if 'NetDemand' not in xl.sheet_names:
            print(f"Warning: NetDemand sheet not found in {net_demand_file}. Using empty DataFrame.")
            return pd.DataFrame(columns=['material', 'location', 'requirement_date', 'quantity', 'demand_type', 'layer'])
        
        net_demand = pd.read_excel(net_demand_file, sheet_name='NetDemand')
        net_demand = _cast_identifiers_to_str(net_demand, ['material', 'location'])

        if net_demand.empty:
            return pd.DataFrame(columns=['material', 'location', 'requirement_date', 'quantity', 'demand_type', 'layer'])
        
        # Filter for layer=0 (most downstream demands)
        if 'layer' in net_demand.columns:
            layer0_demand = net_demand[net_demand['layer'] == 0].copy()
        else:
            print(f"Warning: 'layer' column not found in NetDemand. Using all demands.")
            layer0_demand = net_demand.copy()
        
        # Convert negative quantities to positive
        if 'quantity' in layer0_demand.columns:
            layer0_demand['quantity'] = layer0_demand['quantity'].abs()
        
        # Ensure requirement_date is datetime
        if 'requirement_date' in layer0_demand.columns:
            layer0_demand['requirement_date'] = pd.to_datetime(layer0_demand['requirement_date'])
        
        return layer0_demand
        
    except Exception as e:
        print(f"Error loading daily NetDemand for {simulation_date.strftime('%Y-%m-%d')}: {e}")
        return pd.DataFrame(columns=['material', 'location', 'requirement_date', 'quantity', 'demand_type', 'layer'])


def compute_planning_window(simulation_date: pd.Timestamp, ptf: int, lsk: int) -> tuple:
    """
    Calculate planning window for both NetDemand filtering AND production distribution
    Respects PTF frozen period for all planning decisions
    
    Args:
        simulation_date: Current simulation date (review date)
        ptf: Planning Time Fence (frozen period in days)
        lsk: Lot Size Key (planning horizon in days)
        
    Returns:
        tuple: (window_start, window_end)
    """
    window_start = simulation_date + timedelta(days=ptf)          # After frozen period
    window_end = simulation_date + timedelta(days=ptf + lsk - 1)  # PTF + planning horizon (LSK days)
    return window_start, window_end


def calculate_changeover_metrics(production_plan: pd.DataFrame, changeover_def: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate changeover metrics (count, time, cost, mu_loss) from production plan
    
    Args:
        production_plan: Production plan DataFrame with changeover_id
        changeover_def: Changeover definition with time, cost, mu_loss by changeover_id and line
        
    Returns:
        DataFrame: Changeover log with metrics
    """
    changeover_log = []
    
    if production_plan.empty or changeover_def.empty:
        return pd.DataFrame(columns=['date', 'location', 'line', 'changeover_type', 'count', 'time', 'cost', 'mu_loss'])
    
    # Group by date, location, line, changeover_id to count occurrences
    changeover_summary = production_plan[
        production_plan['changeover_id'].notna()
    ].groupby(['production_plan_date', 'location', 'line', 'changeover_id']).size().reset_index(name='count')
    
    # Merge with changeover definitions to get time, cost, mu_loss
    changeover_def_indexed = changeover_def.set_index(['changeover_id', 'line'])
    
    for _, row in changeover_summary.iterrows():
        date = row['production_plan_date']
        location = row['location']
        line = row['line']
        changeover_id = row['changeover_id']
        count = row['count']
        
        # Get changeover definition
        try:
            definition = changeover_def_indexed.loc[(changeover_id, line)]
            time_per_changeover = float(definition.get('time', 0))
            cost_per_changeover = float(definition.get('cost', 0))
            mu_loss_per_changeover = float(definition.get('mu_loss', 0))
        except KeyError:
            print(f"Warning: Changeover definition not found for changeover_id={changeover_id}, line={line}")
            time_per_changeover = cost_per_changeover = mu_loss_per_changeover = 0
        
        # Calculate totals
        total_time = count * time_per_changeover
        total_cost = count * cost_per_changeover
        total_mu_loss = count * mu_loss_per_changeover
        
        changeover_log.append({
            'date': date,
            'location': location,
            'line': line,
            'changeover_type': changeover_id,
            'count': count,
            'time': total_time,
            'cost': total_cost,
            'mu_loss': total_mu_loss
        })
    
    return pd.DataFrame(changeover_log)



def load_config(filepath):
    xl = pd.ExcelFile(filepath)
    required = [
        'M4_MaterialLocationLineCfg', 'M4_LineCapacity',
        'M4_ChangeoverMatrix', 'M4_ChangeoverDefinition', 'M4_ProductionReliability',
        'Global_DemandPriority'
    ]
    cfg = {}
    for s in required:
        if s not in xl.sheet_names:
            raise KeyError(f"Missing required sheet: {s}")
        # Map to original keys for backward compatibility
        if s == 'M4_MaterialLocationLineCfg':
            cfg['MaterialLocationLineCfg'] = _cast_identifiers_to_str(xl.parse(s))
        elif s == 'M4_LineCapacity':
            cfg['LineCapacity'] = _cast_identifiers_to_str(xl.parse(s))
        elif s == 'M4_ChangeoverMatrix':
            cfg['ChangeoverMatrix'] = _cast_identifiers_to_str(xl.parse(s))
        elif s == 'M4_ChangeoverDefinition':
            cfg['ChangeoverDefinition'] = _cast_identifiers_to_str(xl.parse(s))
        elif s == 'M4_ProductionReliability':
            cfg['ProductionReliability'] = _cast_identifiers_to_str(xl.parse(s))
        elif s == 'Global_DemandPriority':
            cfg['NetDemandTypePriority'] = _cast_identifiers_to_str(xl.parse(s))

    # Optional NetDemand sheet for legacy mode
    if 'NetDemand' in xl.sheet_names:
        cfg['NetDemand'] = _cast_identifiers_to_str(
            xl.parse('NetDemand'), ['material', 'location']
        )

    # Handle optional Global_seed sheet
    if 'Global_seed' in xl.sheet_names:
        seed_df = pd.read_excel(filepath, sheet_name='Global_seed')
        if not seed_df.empty:
            cfg['RandomSeed'] = int(seed_df.iloc[0, 0])
    
    return cfg

def validate_config(cfg):
    issues = []
    
    # Skip NetDemand validation if it doesn't exist (daily execution mode)
    if 'NetDemand' in cfg and not cfg['NetDemand'].empty:
        nd = cfg['NetDemand'][['material', 'location']]
        ml = cfg['MaterialLocationLineCfg'][['material', 'location']]
        _validate_merge_keys(nd, ml, ['material', 'location'])
        merged = pd.merge(nd, ml, on=['material', 'location'], how='left', indicator=True)
        bad = merged[merged['_merge'] == 'left_only']
        for mat, loc in bad[['material', 'location']].drop_duplicates().values:
            issues.append({
                'sheet': 'MaterialLocationLineCfg',
                'row': '',
                'issue': f"No line config for material {mat} at location {loc}"
            })
    
    # Check for multiple eligible lines per material-location
    if 'MaterialLocationLineCfg' in cfg:
        line_counts = cfg['MaterialLocationLineCfg'].groupby(['material', 'location']).size()
        for (mat, loc), cnt in line_counts.items():
            if cnt > 1:
                issues.append({
                    'sheet': 'MaterialLocationLineCfg',
                    'row': '',
                    'issue': f"Multiple eligible lines for material-location: {mat}/{loc}"
                })
    
    return issues

def is_review_day(simulation_date, simulation_start, lsk, day):
    """
    Check if a simulation date is a review day for a material based on integer LSK
    
    Args:
        simulation_date: Current simulation date
        simulation_start: First day of simulation period
        lsk: Review interval in days (integer)
        day: Offset days from simulation_start for first review
        
    Returns:
        bool: True if this is a review day for the material
    """
    days_since_start = (simulation_date - simulation_start).days
    first_review_day = int(day)-1  # offset from start
    return (days_since_start - first_review_day) % int(lsk) == 0 and days_since_start >= first_review_day



def build_unconstrained_plan_for_single_day(net_demand_df, mlcfg, simulation_date, simulation_start, issues):
    """
    Build unconstrained plan for a single simulation date
    Only plans materials that are on review day
        Args:
        net_demand_df: NetDemand DataFrame for this simulation date
        mlcfg: Material location line configuration
        simulation_date: Current simulation date
        simulation_start: Simulation period start date
        issues: List to append validation issues
        
    Returns:
        DataFrame: Unconstrained production plans
    """
    plans = []
    
    if net_demand_df.empty:
        return pd.DataFrame(columns=['material', 'location', 'line', 'planned_date', 'uncon_planned_qty', 'simulation_date', 'original_quantity'])
    
    # 确保MLCFG的标识符也是string类型，与NetDemand保持一致
    mlcfg = _cast_identifiers_to_str(mlcfg.copy(), ['material', 'location'])
    
    for idx, row in mlcfg.iterrows():
        material = row['material']
        location = row['location']
        line = row['delegate_line']
        lsk = int(row['lsk'])  # Now integer
        day = int(row['day'])
        ptf = int(row['ptf'])
        
        # Check if this material is on review day
        if not is_review_day(simulation_date, simulation_start, lsk, day):
            continue  # Skip non-review materials
        
        # Get demands for this material-location (两边都是string类型)
        nd_sub = net_demand_df[
            (net_demand_df['material'] == material) & 
            (net_demand_df['location'] == location)
        ].copy()
        
        if nd_sub.empty:
            continue
        
        # Add configuration data to demands (现在两边都是string类型，可以安全merge)
        cfg_slice = mlcfg[mlcfg['material'].eq(material) & mlcfg['location'].eq(location)]
        _validate_merge_keys(nd_sub, cfg_slice, ['material', 'location'])
        nd_sub = nd_sub.merge(
            cfg_slice,
            on=['material', 'location'],
            how='left'
        )
        
        # Calculate planning window (respects PTF)
        window_start, window_end = compute_planning_window(simulation_date, ptf, lsk)
        # Note: requirement_date from Module3 already includes lead time calculation
        # === 恢复正确的日期匹配逻辑：requirement_date应该等于simulation_date ===
        nd_sub['requirement_date'] = pd.to_datetime(nd_sub['requirement_date']).dt.normalize()
        _sim_d = pd.to_datetime(simulation_date).normalize()
        
        # 严格匹配：Module3的requirement_date = simulation_date + 1，
        # Module4读取前一天文件，所以requirement_date应该等于当前simulation_date
        mask = (nd_sub['requirement_date'] == _sim_d)
        
        # === 报告日期不匹配的需求 ===
        for _, r in nd_sub[~mask].iterrows():
            try:
                issues.append({
                    'sheet': 'NetDemand',
                    'row': '',
                    'issue': f"Demand for material {r['material']} at location {r['location']} with requirement date {pd.to_datetime(r['requirement_date']).date()} does not match simulation date {simulation_date.date()} (excluded from plan)."
                })
            except Exception:
                pass  # 容错，避免 issues 写入异常中断
        
        # Use only demands within planning window
        # === 修复对应：这里改为仅用“当日需求” ===
        nd_sub = nd_sub[mask]
        
        if nd_sub.empty:
            continue
        
        # Aggregate quantity
        agg_qty = nd_sub['quantity'].sum()
        min_batch = int(row['min_batch'])
        rv = int(row['rv'])
        
        # Round up to minimum batch and rounding volume
        def round_up(q, mb, rv):
            base = max(q, mb)
            return base if base % rv == 0 else int(np.ceil(base / rv) * rv)
        
        uncon_planned_qty = round_up(agg_qty, min_batch, rv)
        
        result = pd.DataFrame([{
            'material': material,
            'location': location,
            'line': line,
            'planned_date': simulation_date,  # Plan on review date
            'uncon_planned_qty': uncon_planned_qty,
            'simulation_date': simulation_date,
            'original_quantity': agg_qty  # 保存原始quantity用于排序
        }])
        
        plans.append(result)
    
    if not plans:
        return pd.DataFrame(columns=['material', 'location', 'line', 'planned_date', 'uncon_planned_qty', 'simulation_date', 'original_quantity'])
    
    return pd.concat(plans, ignore_index=True)

def optimal_changeover_sequence(batches, co_mat, co_def, line):
    """
    Enhanced production sequencing with quantity-based prioritization:
    1. First SKU: Select by largest original quantity (not batch size)
    2. Subsequent SKUs: Minimize changeover time, with quantity as tie-breaker
    
    batches: list of dict, each batch at least: 'material', 'uncon_planned_qty', 'original_quantity'
    co_mat: pandas.Series, MultiIndex [from, to] -> changeover_id
    co_def: dict, (changeover_id, line) -> time
    line: str
    Returns: list of batch indices (order)
    """
    batch_idx_list = list(range(len(batches)))
    if not batch_idx_list:
        return []
    left = set(batch_idx_list)
    
    # 1. First SKU: Select by largest original quantity (not batch size)
    first = max(left, key=lambda i: batches[i].get('original_quantity', batches[i]['uncon_planned_qty']))
    seq = [first]
    left.remove(first)
    cur_mat = batches[first]['material']
    
    while left:
        # 2. Pick next with min changeover time from cur_mat
        # If multiple options have same changeover time, use quantity as tie-breaker
        min_cost = None
        candidates = []  # Store all candidates with minimum changeover time
        
        for i in left:
            next_mat = batches[i]['material']
            try:
                # Ensure material IDs are strings for lookup
                cur_mat_str = str(cur_mat)
                next_mat_str = str(next_mat)
                coid = co_mat.loc[(cur_mat_str, next_mat_str)]
                co_time = co_def.get((coid, line), 0)
            except Exception:
                co_time = 0
            
            if min_cost is None or co_time < min_cost:
                min_cost = co_time
                candidates = [i]
            elif co_time == min_cost:
                candidates.append(i)
        
        # Among candidates with minimum changeover time, select by largest quantity
        if len(candidates) == 1:
            min_idx = candidates[0]
        else:
            min_idx = max(candidates, key=lambda i: batches[i].get('original_quantity', batches[i]['uncon_planned_qty']))
        
        seq.append(min_idx)
        left.remove(min_idx)
        cur_mat = batches[min_idx]['material']
    
    return seq

def centralized_capacity_allocation_with_changeover(uncon, cap_df, rate_map, co_mat, co_def, mlcfg, 
                                                   previous_line_states=None, simulation_date=None, 
                                                   previously_allocated_capacity=None):
    """
    Enhanced capacity allocation with cross-day changeover continuity and capacity tracking.
    
    Args:
        uncon: Unconstrained production plan
        cap_df: Capacity DataFrame
        rate_map: Production rate mapping
        co_mat: Changeover matrix
        co_def: Changeover definition
        mlcfg: Material location line configuration
        previous_line_states: Line states from previous day (optional)
        simulation_date: Current simulation date (optional)
        previously_allocated_capacity: Previously allocated capacity from earlier simulation dates (optional)
        
    Returns:
        tuple: (plans_log, exceed_log)
    """
    plans_log = []
    exceed = []
    mct_map = mlcfg.set_index(['material', 'location'])['MCT'].to_dict()
    if 'location' in cap_df.columns:
        cap_df['capacity'] = cap_df['capacity'].astype(float)
        cap_map = cap_df.set_index(['location', 'line', 'date'])['capacity'].to_dict()
    else:
        cap_df['capacity'] = cap_df['capacity'].astype(float)
        cap_map = cap_df.set_index(['line', 'date'])['capacity'].to_dict()

    uncon = uncon.sort_values(['line', 'simulation_date', 'planned_date', 'material']).reset_index(drop=True)
    for (line, sim_date), uncon_grp in uncon.groupby(['line', 'simulation_date']):
        batch_list = uncon_grp.to_dict(orient='records')
        
        # Apply changeover optimization
        if len(batch_list) > 1:
            co_seq = optimal_changeover_sequence(batch_list, co_mat, co_def, line)
            batch_list = [batch_list[i] for i in co_seq]

        # Initialize previous material from previous day's state
        prev_mat = None
        if previous_line_states and line in previous_line_states:
            prev_mat = previous_line_states[line].get('last_material')
            print(f"Line {line}: Starting with previous day's last material: {prev_mat}")

        for cur_plan in batch_list:
            material = cur_plan['material']
            location = cur_plan['location']
            
            # Get configuration for this material-location
            row_cfg = mlcfg[(mlcfg['material'] == material) & (mlcfg['location'] == location)].iloc[0]
            lsk = int(row_cfg['lsk'])
            ptf = int(row_cfg['ptf'])
            
            # Calculate planning window for production distribution
            window_start, window_end = compute_planning_window(cur_plan['simulation_date'], ptf, lsk)
            horizon_days = pd.date_range(window_start, window_end)

            cur_mat = material
            prod_remain = cur_plan['uncon_planned_qty']
            co_remain = 0
            coid_to_log = None
            is_first_co_day = False

            # Calculate changeover time if material changes
            if prev_mat is not None and prev_mat != cur_mat:
                try:
                    # Ensure material IDs are strings for lookup
                    prev_mat_str = str(prev_mat)
                    cur_mat_str = str(cur_mat)
                    coid = co_mat.loc[(prev_mat_str, cur_mat_str)]
                    co_time = co_def.get((coid, line), 0) if coid else 0
                except Exception:
                    coid = None
                    co_time = 0
                co_remain = co_time
                coid_to_log = coid
                is_first_co_day = True
                print(f"Line {line}: Changeover from {prev_mat} to {cur_mat}, time needed: {co_time}")
            else:
                co_remain = 0
                coid_to_log = None
                is_first_co_day = False

            # Allocate production across horizon days
            for day_dt in horizon_days:
                cap_key = (location, line, day_dt) if 'location' in cap_df.columns else (line, day_dt)
                # Get current remaining capacity (already considers previous allocations from cap_map)
                current_remaining_cap = cap_map.get(cap_key, 0)
                today_cap = current_remaining_cap
                
                # Check for previously allocated capacity for this production plan date (from other simulation dates)
                if previously_allocated_capacity:
                    capacity_key = f"{location}|{line}|{day_dt.strftime('%Y-%m-%d')}"
                    previously_used_hours = previously_allocated_capacity.get(capacity_key, 0)
                    
                    # Deduct previously allocated capacity (in hours) from available capacity (in hours)
                    today_cap = max(0, today_cap - previously_used_hours)
                    
                    if previously_used_hours > 0:
                        print(f"Line {line}: Production plan date {day_dt.strftime('%Y-%m-%d')} has {previously_used_hours:.2f} hours already allocated by previous simulation dates. Available capacity: {today_cap:.2f} hours")
                
                # First deduct changeover time from capacity (without creating separate record)
                changeover_completed_this_day = False
                changeover_used = 0
                if co_remain > 0:
                    changeover_used = min(today_cap, co_remain)
                    co_remain -= changeover_used
                    today_cap -= changeover_used
                    # Mark if changeover completed this day
                    if co_remain == 0:
                        changeover_completed_this_day = True
                    # If changeover not completed, update cap_map and continue to next day
                    if co_remain > 0:
                        cap_map[cap_key] = current_remaining_cap - changeover_used
                        continue
                
                # Then allocate production capacity (with changeover_id if applicable)
                rate = float(rate_map.get((cur_mat, line), 1))
                can_produce = min(prod_remain, int(today_cap * rate))
                hours_used = can_produce / rate if rate else 0
                
                # Only create record if there's actual production
                if can_produce > 0:
                    plans_log.append({
                        'material': cur_mat, 'location': location, 'line': line,
                        'simulation_date': sim_date, 'production_plan_date': day_dt,
                        'available_date': day_dt + timedelta(days=int(mct_map.get((cur_mat, location), 0))),
                        'uncon_planned_qty': cur_plan['uncon_planned_qty'],
                        'con_planned_qty': can_produce,
                        'changeover_id': coid_to_log if (changeover_completed_this_day or is_first_co_day) else None
                    })
                    # Reset changeover tracking after first production record
                    if coid_to_log is not None:
                        coid_to_log = None
                        is_first_co_day = False
                
                prod_remain -= can_produce
                today_cap -= hours_used
                # Update cap_map with remaining capacity after all usage (changeover + production)
                cap_map[cap_key] = current_remaining_cap - changeover_used - hours_used
                if prod_remain <= 0:
                    break
                    
            # Record unmet demand
            if prod_remain > 0:
                exceed.append({
                    'material': cur_plan['material'],
                    'location': cur_plan['location'],
                    'line': line,
                    'simulation_date': sim_date,
                    'production_plan_date': window_end,
                    'unmet_uncon_planned_qty': prod_remain
                })
            prev_mat = cur_mat
            
    return pd.DataFrame(plans_log), pd.DataFrame(exceed)

def simulate_production(plan, pr_cfg, seed=None):
    if plan.empty or 'con_planned_qty' not in plan.columns:
        plan['produced_qty'] = []
        return plan
    rng = np.random.RandomState(seed)
    pr_map = pr_cfg.set_index(['location', 'line'])['pr'].to_dict()
    plan['produced_qty'] = plan.apply(
        lambda r: rng.binomial(int(r['con_planned_qty']), pr_map.get((r['location'], r['line']), 1)),
        axis=1
    )
    return plan

def dedup_issues(issues):
    if not issues:
        return issues
    df = pd.DataFrame(issues)
    df = df.drop_duplicates()
    return df.to_dict(orient='records')

def write_output(plan, exc, issues, changeover_log, out_path, simulation_date=None):
    """
    Write outputs to Excel file with ChangeoverLog support and dual output structure
    
    Args:
        plan: Production plan DataFrame
        exc: Capacity exceed DataFrame 
        issues: Validation issues list
        changeover_log: Changeover metrics DataFrame
        out_path: Base output file path
        simulation_date: If provided, creates daily versioned file; otherwise creates consolidated file
    """
    # Determine output file path
    if simulation_date is not None:
        # Daily versioned output
        date_str = simulation_date.strftime('%Y%m%d')
        out_dir = os.path.dirname(out_path)
        base_name = os.path.splitext(os.path.basename(out_path))[0]
        daily_path = os.path.join(out_dir, f"{base_name}_{date_str}.xlsx")
        final_path = daily_path
    else:
        # Consolidated output
        final_path = out_path
    
    with pd.ExcelWriter(final_path, engine='openpyxl') as w:
        # Write all output sheets
        plan.to_excel(w, sheet_name='ProductionPlan', index=False)
        exc.to_excel(w, sheet_name='CapacityExceed', index=False)
        pd.DataFrame(issues).to_excel(w, sheet_name='Validation', index=False)
        changeover_log.to_excel(w, sheet_name='ChangeoverLog', index=False)
    
    return final_path

def run_daily_production_planning(config_file: str, module3_output_dir: str, 
                                 simulation_date: pd.Timestamp, simulation_start: pd.Timestamp,
                                 output_dir: str) -> str:
    """
    Run daily production planning for a single simulation date with cross-day changeover continuity
    
    Args:
        config_file: Path to M4 configuration Excel file
        module3_output_dir: Directory containing Module3 daily outputs
        simulation_date: Current simulation date
        simulation_start: Simulation period start date
        output_dir: Directory to save output files
        
    Returns:
        str: Path to generated output file
    """
    try:
        # Load configuration
        cfg = load_config(config_file)
        issues = validate_config(cfg)
        
        # Load daily NetDemand from Module3
        net_demand_df = load_daily_net_demand(module3_output_dir, simulation_date)
        net_demand_df = _cast_identifiers_to_str(net_demand_df, ['material', 'location'])        
        if net_demand_df.empty:
            print(f"Warning: No NetDemand data for {simulation_date.strftime('%Y-%m-%d')}. Generating empty output.")
        
        # Ensure requirement_date is datetime
        if not net_demand_df.empty and 'requirement_date' in net_demand_df.columns:
            net_demand_df['requirement_date'] = pd.to_datetime(net_demand_df['requirement_date'])
        
        # Build unconstrained plan for this simulation date
        mlcfg = cfg['MaterialLocationLineCfg']
        # Normalize identifiers to ensure proper matching
        from orchestrator import _normalize_identifiers
        mlcfg = _normalize_identifiers(mlcfg)
        # Also normalize NetDemand data to ensure proper matching
        net_demand_df = _normalize_identifiers(net_demand_df)
        uncon_plan = build_unconstrained_plan_for_single_day(
            net_demand_df, mlcfg, simulation_date, simulation_start, issues
        )
        
        # Load previous day's line states for cross-day changeover continuity
        previous_line_states = load_line_state(output_dir, simulation_date)
        if previous_line_states:
            print(f"Loaded previous day's line states: {list(previous_line_states.keys())}")
        else:
            print("No previous day's line states found - starting fresh")
        
        # Load previously allocated capacity from all previous simulation dates
        previously_allocated_capacity = load_all_previous_capacity(output_dir, simulation_date)
        if previously_allocated_capacity:
            print(f"Loaded previously allocated capacity: {len(previously_allocated_capacity)} capacity allocations from previous simulation dates")
        else:
            print("No previously allocated capacity found - starting fresh")
        
        # Set up capacity allocation parameters
        # Also normalize changeover matrix to ensure proper matching
        co_mat_df = cfg['ChangeoverMatrix'].copy()
        co_mat_df['from_material'] = co_mat_df['from_material'].astype(str)
        co_mat_df['to_material'] = co_mat_df['to_material'].astype(str)
        co_mat = co_mat_df.set_index(['from_material', 'to_material'])['changeover_id']
        
        # Enhanced changeover definition with cost and mu_loss
        co_def_df = cfg['ChangeoverDefinition']
        co_def = co_def_df.set_index(['changeover_id', 'line'])['time'].to_dict()
        
        cap_df = cfg['LineCapacity'].copy()
        cap_df['date'] = pd.to_datetime(cap_df['date'])
        
        rate_map = mlcfg.set_index(['material', 'delegate_line'])['prd_rate']
        rate_map.index.set_names(['material', 'line'], inplace=True)
        
        # Allocate capacity with changeover consideration and cross-day continuity
        plan_log, exceed_log = centralized_capacity_allocation_with_changeover(
            uncon_plan, cap_df, rate_map, co_mat, co_def, mlcfg,
            previous_line_states=previous_line_states, simulation_date=simulation_date,
            previously_allocated_capacity=previously_allocated_capacity
        )
        
        # Simulate production reliability
        random_seed = cfg.get('RandomSeed', 42)
        plan_log = simulate_production(plan_log, cfg['ProductionReliability'], seed=random_seed)
        
        # Calculate changeover metrics
        changeover_log = calculate_changeover_metrics(plan_log, co_def_df)
        
        # Extract and save current day's line states for next day
        current_line_states = extract_line_states_from_plan(plan_log)
        if current_line_states:
            save_line_state(output_dir, simulation_date, current_line_states)
            print(f"Saved current day's line states: {list(current_line_states.keys())}")
        
        # Extract and save current day's allocated capacity for future simulation dates
        current_allocated_capacity = extract_allocated_capacity_from_plan(plan_log, rate_map.to_dict(), co_def)
        if current_allocated_capacity:
            save_allocated_capacity(output_dir, simulation_date, current_allocated_capacity)
            print(f"Saved current day's allocated capacity: {len(current_allocated_capacity)} capacity allocations (in hours)")
        
        # Validate capacity allocation to ensure no over-allocation
        capacity_validation_issues = validate_capacity_allocation(plan_log, previously_allocated_capacity, simulation_date, rate_map.to_dict(), co_def)
        if capacity_validation_issues:
            print(f"Capacity allocation validation completed: {len(capacity_validation_issues)} validation records generated")
            # Add validation issues to the main issues list
            issues.extend(capacity_validation_issues)
        
        # Deduplicate issues
        issues = dedup_issues(issues)
        
        # Generate output file path
        base_output_file = os.path.join(output_dir, "Module4Output.xlsx")
        
        # Write daily output
        daily_output_path = write_output(
            plan_log, exceed_log, issues, changeover_log, 
            base_output_file, simulation_date
        )
        
        print(f"Module4 daily output generated: {daily_output_path}")
        
        # Check for critical issues
        critical_issues = [x for x in issues if 'No line config' in x['issue'] or 'Multiple eligible lines' in x['issue']]
        if critical_issues:
            msg = f'Critical validation errors found for {simulation_date.strftime("%Y-%m-%d")}! See Validation sheet.\n' + \
                '\n'.join(x['issue'] for x in critical_issues)
            print(f"WARNING: {msg}")
            # Don't raise exception for daily execution - log and continue
        
        return daily_output_path
        
    except Exception as e:
        print(f'[ERROR] Module4 daily execution failed for {simulation_date.strftime("%Y-%m-%d")}: {str(e)}')
        raise


def generate_consolidated_output(daily_output_files: list, output_path: str):
    """
    Generate consolidated output from multiple daily output files
    
    Args:
        daily_output_files: List of daily output file paths
        output_path: Path for consolidated output file
    """
    if not daily_output_files:
        print("Warning: No daily output files to consolidate.")
        return
    
    all_plans = []
    all_exceeds = []
    all_issues = []
    all_changeovers = []
    
    for file_path in daily_output_files:
        if not os.path.exists(file_path):
            print(f"Warning: Daily output file not found: {file_path}")
            continue
            
        try:
            xl = pd.ExcelFile(file_path)
            
            if 'ProductionPlan' in xl.sheet_names:
                plan_df = xl.parse('ProductionPlan')
                if not plan_df.empty:
                    all_plans.append(plan_df)
            
            if 'CapacityExceed' in xl.sheet_names:
                exceed_df = xl.parse('CapacityExceed')
                if not exceed_df.empty:
                    all_exceeds.append(exceed_df)
            
            if 'Validation' in xl.sheet_names:
                issues_df = xl.parse('Validation')
                if not issues_df.empty:
                    all_issues.append(issues_df)
            
            if 'ChangeoverLog' in xl.sheet_names:
                changeover_df = xl.parse('ChangeoverLog')
                if not changeover_df.empty:
                    all_changeovers.append(changeover_df)
        
        except Exception as e:
            print(f"Error reading daily output file {file_path}: {e}")
            continue
    
    # Consolidate data
    consolidated_plan = pd.concat(all_plans, ignore_index=True) if all_plans else pd.DataFrame()
    consolidated_exceed = pd.concat(all_exceeds, ignore_index=True) if all_exceeds else pd.DataFrame()
    consolidated_issues = pd.concat(all_issues, ignore_index=True).drop_duplicates() if all_issues else pd.DataFrame()
    consolidated_changeover = pd.concat(all_changeovers, ignore_index=True) if all_changeovers else pd.DataFrame()
    
    # Write consolidated output
    write_output(
        consolidated_plan, consolidated_exceed, 
        consolidated_issues.to_dict('records') if not consolidated_issues.empty else [],
        consolidated_changeover, output_path
    )
    
    print(f"Consolidated Module4 output generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Module 4: APS Industrial Production Simulation with Daily Execution Support')
    parser.add_argument('--config', required=True, help='Path to configuration Excel file')
    parser.add_argument('--mode', choices=['daily', 'legacy'], default='daily', help='Execution mode')
    
    # Daily mode arguments
    parser.add_argument('--module3_output_dir', help='Directory containing Module3 daily outputs (daily mode)')
    parser.add_argument('--simulation_date', help='Simulation date YYYY-MM-DD (daily mode)')
    parser.add_argument('--simulation_start', help='Simulation start date YYYY-MM-DD (daily mode, required on first run)')
    parser.add_argument('--output_dir', help='Output directory (daily mode)')
    
    # Legacy mode arguments (backward compatibility)
    parser.add_argument('--input', help='Legacy: input file path')
    parser.add_argument('--sim_start', help='Legacy: simulation start date')
    parser.add_argument('--sim_end', help='Legacy: simulation end date')
    parser.add_argument('--output', help='Legacy: output file path')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'daily':
            # Daily execution mode
            if not all([args.module3_output_dir, args.simulation_date, args.output_dir]):
                raise ValueError("Daily mode requires: --module3_output_dir, --simulation_date, --output_dir")
            
            simulation_date = pd.to_datetime(args.simulation_date)
            simulation_start = get_or_init_simulation_start(
                args.output_dir,
                pd.to_datetime(args.simulation_start) if args.simulation_start else None
            )
            
            output_file = run_daily_production_planning(
                config_file=args.config,
                module3_output_dir=args.module3_output_dir,
                simulation_date=simulation_date,
                simulation_start=simulation_start,
                output_dir=args.output_dir
            )
            
            print(f"Daily production planning completed: {output_file}")
            
        else:
            # Legacy execution mode for backward compatibility
            if not all([args.input, args.sim_start, args.sim_end, args.output]):
                raise ValueError("Legacy mode requires: --input, --sim_start, --sim_end, --output")
            
            # Use legacy logic (full simulation window)
            sim_start = pd.to_datetime(args.sim_start)
            sim_end = pd.to_datetime(args.sim_end)
            cfg = load_config(args.input)
            issues = validate_config(cfg)
            

            # Legacy mode uses embedded NetDemand from config
            nd = cfg.get('NetDemand', pd.DataFrame())
            nd = _cast_identifiers_to_str(nd, ['material', 'location'])
            if nd.empty:
                raise ValueError("Legacy mode requires NetDemand sheet in config file")
            
            nd['requirement_date'] = pd.to_datetime(nd['requirement_date'])
            mlcfg = cfg['MaterialLocationLineCfg']
            co_mat = cfg['ChangeoverMatrix'].set_index(['from_material', 'to_material'])['changeover_id']
            co_def_df = cfg['ChangeoverDefinition']
            co_def = co_def_df.set_index(['changeover_id', 'line'])['time'].to_dict()
            cap_df = cfg['LineCapacity']
            rate_map = mlcfg.set_index(['material', 'delegate_line'])['prd_rate']
            rate_map.index.set_names(['material', 'line'], inplace=True)
            
            # For legacy mode - create empty unconstrained plan for now
            uncon = pd.DataFrame(columns=['material', 'location', 'line', 'planned_date', 'uncon_planned_qty', 'simulation_date', 'original_quantity'])
            
            plan_log, exceed_log = centralized_capacity_allocation_with_changeover(
                uncon, cap_df, rate_map, co_mat, co_def, mlcfg
            )
            plan_log = simulate_production(plan_log, cfg['ProductionReliability'], seed=cfg.get('RandomSeed', 42))
            
            # Calculate changeover metrics
            changeover_log = calculate_changeover_metrics(plan_log, co_def_df)
            
            issues = dedup_issues(issues)
            write_output(plan_log, exceed_log, issues, changeover_log, args.output)
            
            # Check critical issues
            critical_issues = [x for x in issues if 'No line config' in x['issue'] or 'Multiple eligible lines' in x['issue']]
            if critical_issues:
                msg = 'Critical validation errors found! See Validation sheet in output file.\n' + \
                    '\n'.join(x['issue'] for x in critical_issues)
                print(msg)
                raise Exception(msg)
            
            print(f"Legacy production planning completed: {args.output}")
            
    except Exception as e:
        print(f'[ERROR]: {str(e)}')
        raise


if __name__ == '__main__':
    main()
