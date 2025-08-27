#module 4
import pandas as pd
import numpy as np
from datetime import timedelta
import argparse
import os

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
        # 首先尝试读取前一天的NetDemand（符合执行顺序 M1→M4→M5→M6→M3）
        prev_date = simulation_date - pd.Timedelta(days=1)
        prev_date_str = prev_date.strftime('%Y%m%d')
        net_demand_file = os.path.join(module3_output_dir, f"Module3Output_{prev_date_str}.xlsx")
        
        if not os.path.exists(net_demand_file):
            # 如果是第一天，尝试读取当天的（可能是初始需求预测）
            current_date_str = simulation_date.strftime('%Y%m%d')
            net_demand_file = os.path.join(module3_output_dir, f"Module3Output_{current_date_str}.xlsx")
            
            if not os.path.exists(net_demand_file):
                print(f"Warning: NetDemand file not found for {prev_date_str} or {current_date_str}. Using empty DataFrame.")
                return pd.DataFrame(columns=['material', 'location', 'requirement_date', 'quantity', 'demand_type', 'layer'])
        
        # Load NetDemand sheet
        xl = pd.ExcelFile(net_demand_file)
        if 'NetDemand' not in xl.sheet_names:
            print(f"Warning: NetDemand sheet not found in {net_demand_file}. Using empty DataFrame.")
            return pd.DataFrame(columns=['material', 'location', 'requirement_date', 'quantity', 'demand_type', 'layer'])
        
        net_demand = pd.read_excel(net_demand_file, sheet_name='NetDemand')
        
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
            cfg['MaterialLocationLineCfg'] = xl.parse(s)
        elif s == 'M4_LineCapacity':
            cfg['LineCapacity'] = xl.parse(s)
        elif s == 'M4_ChangeoverMatrix':
            cfg['ChangeoverMatrix'] = xl.parse(s)
        elif s == 'M4_ChangeoverDefinition':
            cfg['ChangeoverDefinition'] = xl.parse(s)
        elif s == 'M4_ProductionReliability':
            cfg['ProductionReliability'] = xl.parse(s)
        elif s == 'Global_DemandPriority':
            cfg['NetDemandTypePriority'] = xl.parse(s)
    
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
    first_review_day = int(day)  # offset from start
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
        return pd.DataFrame(columns=['material', 'location', 'line', 'planned_date', 'uncon_planned_qty', 'simulation_date'])
    
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
        
        # Get demands for this material-location
        nd_sub = net_demand_df[
            (net_demand_df['material'] == material) & 
            (net_demand_df['location'] == location)
        ].copy()
        
        if nd_sub.empty:
            continue
        
        # Add configuration data to demands
        nd_sub = nd_sub.merge(
            mlcfg[mlcfg['material'].eq(material) & mlcfg['location'].eq(location)], 
            on=['material', 'location'], 
            how='left'
        )
        
        # Calculate planning window (respects PTF)
        window_start, window_end = compute_planning_window(simulation_date, ptf, lsk)
        
        # Filter demands within planning window using requirement_date directly
        # Note: requirement_date from Module3 already includes lead time calculation
        mask = (nd_sub['requirement_date'] >= window_start) & (nd_sub['requirement_date'] <= window_end)
        
        # Log issues for demands outside planning window
        for _, r in nd_sub[~mask].iterrows():
            issues.append({
                'sheet': 'NetDemand',
                'row': '',
                'issue': f"Demand for material {r['material']} at location {r['location']} with requirement date {r['requirement_date'].date()} is outside planning window [{window_start.date()}, {window_end.date()}]."
            })
        
        # Use only demands within planning window
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
            'simulation_date': simulation_date
        }])
        
        plans.append(result)
    
    if not plans:
        return pd.DataFrame(columns=['material', 'location', 'line', 'planned_date', 'uncon_planned_qty', 'simulation_date'])
    
    return pd.concat(plans, ignore_index=True)

def optimal_changeover_sequence(batches, co_mat, co_def, line):
    """
    batches: list of dict, each batch at least: 'material', 'uncon_planned_qty'
    co_mat: pandas.Series, MultiIndex [from, to] -> changeover_id
    co_def: dict, (changeover_id, line) -> time
    line: str
    Returns: list of batch indices (order)
    """
    # 1. Start with largest batch
    batch_idx_list = list(range(len(batches)))
    if not batch_idx_list:
        return []
    left = set(batch_idx_list)
    # First: max batch
    first = max(left, key=lambda i: batches[i]['uncon_planned_qty'])
    seq = [first]
    left.remove(first)
    cur_mat = batches[first]['material']
    while left:
        # Pick next with min changeover time from cur_mat
        min_cost, min_idx = None, None
        for i in left:
            next_mat = batches[i]['material']
            try:
                coid = co_mat.loc[(cur_mat, next_mat)]
                co_time = co_def.get((coid, line), 0)
            except Exception:
                co_time = 0
            if (min_cost is None) or (co_time < min_cost):
                min_cost, min_idx = co_time, i
        seq.append(min_idx)
        left.remove(min_idx)
        cur_mat = batches[min_idx]['material']
    return seq

def centralized_capacity_allocation_with_changeover(uncon, cap_df, rate_map, co_mat, co_def, mlcfg):
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

        prev_mat = None
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
                    coid = co_mat.loc[(prev_mat, cur_mat)]
                    co_time = co_def.get((coid, line), 0) if coid else 0
                except Exception:
                    coid = None
                    co_time = 0
                co_remain = co_time
                coid_to_log = coid
                is_first_co_day = True
            else:
                co_remain = 0
                coid_to_log = None
                is_first_co_day = False

            # Allocate production across horizon days
            for day_dt in horizon_days:
                cap_key = (location, line, day_dt) if 'location' in cap_df.columns else (line, day_dt)
                today_cap = cap_map.get(cap_key, 0)
                
                # First allocate changeover time
                if co_remain > 0:
                    used = min(today_cap, co_remain)
                    co_remain -= used
                    today_cap -= used
                    plans_log.append({
                        'material': cur_mat, 'location': location, 'line': line,
                        'simulation_date': sim_date, 'production_plan_date': day_dt,
                        'available_date': day_dt + timedelta(days=int(mct_map.get((cur_mat, location), 0))),
                        'uncon_planned_qty': cur_plan['uncon_planned_qty'],
                        'con_planned_qty': 0,
                        'changeover_id': coid_to_log if is_first_co_day else None
                    })
                    cap_map[cap_key] = today_cap
                    if co_remain > 0:
                        is_first_co_day = False
                        continue
                    coid_to_log = None
                    is_first_co_day = False
                
                # Then allocate production capacity
                rate = float(rate_map.get((cur_mat, line), 1))
                can_produce = min(prod_remain, int(today_cap * rate))
                hours_used = can_produce / rate if rate else 0
                plans_log.append({
                    'material': cur_mat, 'location': location, 'line': line,
                    'simulation_date': sim_date, 'production_plan_date': day_dt,
                    'available_date': day_dt + timedelta(days=int(mct_map.get((cur_mat, location), 0))),
                    'uncon_planned_qty': cur_plan['uncon_planned_qty'],
                    'con_planned_qty': can_produce,
                    'changeover_id': None
                })
                prod_remain -= can_produce
                today_cap -= hours_used
                cap_map[cap_key] = today_cap
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
    Run daily production planning for a single simulation date
    
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
        
        if net_demand_df.empty:
            print(f"Warning: No NetDemand data for {simulation_date.strftime('%Y-%m-%d')}. Generating empty output.")
        
        # Ensure requirement_date is datetime
        if not net_demand_df.empty and 'requirement_date' in net_demand_df.columns:
            net_demand_df['requirement_date'] = pd.to_datetime(net_demand_df['requirement_date'])
        
        # Build unconstrained plan for this simulation date
        mlcfg = cfg['MaterialLocationLineCfg']
        uncon_plan = build_unconstrained_plan_for_single_day(
            net_demand_df, mlcfg, simulation_date, simulation_start, issues
        )
        
        # Set up capacity allocation parameters
        co_mat = cfg['ChangeoverMatrix'].set_index(['from_material', 'to_material'])['changeover_id']
        
        # Enhanced changeover definition with cost and mu_loss
        co_def_df = cfg['ChangeoverDefinition']
        co_def = co_def_df.set_index(['changeover_id', 'line'])['time'].to_dict()
        
        cap_df = cfg['LineCapacity'].copy()
        cap_df['date'] = pd.to_datetime(cap_df['date'])
        
        rate_map = mlcfg.set_index(['material', 'delegate_line'])['prd_rate']
        rate_map.index.set_names(['material', 'line'], inplace=True)
        
        # Allocate capacity with changeover consideration
        plan_log, exceed_log = centralized_capacity_allocation_with_changeover(
            uncon_plan, cap_df, rate_map, co_mat, co_def, mlcfg
        )
        
        # Simulate production reliability
        random_seed = cfg.get('RandomSeed', 42)
        plan_log = simulate_production(plan_log, cfg['ProductionReliability'], seed=random_seed)
        
        # Calculate changeover metrics
        changeover_log = calculate_changeover_metrics(plan_log, co_def_df)
        
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
    parser.add_argument('--simulation_start', help='Simulation start date YYYY-MM-DD (daily mode)')
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
            if not all([args.module3_output_dir, args.simulation_date, args.simulation_start, args.output_dir]):
                raise ValueError("Daily mode requires: --module3_output_dir, --simulation_date, --simulation_start, --output_dir")
            
            simulation_date = pd.to_datetime(args.simulation_date)
            simulation_start = pd.to_datetime(args.simulation_start)
            
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
            uncon = pd.DataFrame(columns=['material', 'location', 'line', 'planned_date', 'uncon_planned_qty', 'simulation_date'])
            
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
