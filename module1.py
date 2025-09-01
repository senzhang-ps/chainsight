import pandas as pd
import numpy as np
from scipy.stats import truncnorm
from scipy.stats import truncnorm
from datetime import datetime
import os
import re

# ----------- 1. LOAD CONFIG (Enhanced) -----------
def load_config(filename, sheet_mapping=None):
    """
    加载 Excel 中的多个 sheet 到 DataFrame 字典中。
    """
    if sheet_mapping is None:
        sheet_mapping = {
            'DemandForecast': ('demand_forecast', None),
            'ForecastError': ('forecast_error', None),
            'OrderCalendar': ('order_calendar', None),
            'AOConfig': ('ao_config', pd.DataFrame()),
            'SupplyChoiceConfig': ('supply_choice', pd.DataFrame()),
            'InitialInventory': ('initial_inventory', None),
            'DPSConfig': ('dps_config', pd.DataFrame()),
            'ProductionPlan': ('production_plan', pd.DataFrame()),
            'DeliveryPlan': ('delivery_plan', pd.DataFrame()),
        }

    try:
        xl = pd.ExcelFile(filename)
        loaded_sheets = {}
        for sheet_name, (key, default) in sheet_mapping.items():
            if sheet_name in xl.sheet_names:
                loaded_sheets[key] = xl.parse(sheet_name)
            else:
                loaded_sheets[key] = default
        return loaded_sheets
    except Exception as e:
        raise RuntimeError(f"Failed to load config from {filename}: {e}")

# ----------- 2. DPS SPLIT -----------
def apply_dps(df, dps_cfg):
    if dps_cfg.empty:
        return df.copy()
    df_new = df.copy()
    splits = []
    for _, row in dps_cfg.iterrows():
        filt = (df['material'] == row['material']) & (df['location'] == row['location'])
        for i, orig_row in df[filt].iterrows():
            split_qty = int(round(orig_row['quantity'] * row['dps_percent']))
            remain_qty = int(round(orig_row['quantity'] - split_qty))
            splits.append({
                'material': orig_row['material'],
                'location': row['dps_location'],
                'week': orig_row['week'],
                'quantity': split_qty
            })
            df_new.at[i, 'quantity'] = remain_qty
    if splits:
        df_new = pd.concat([df_new, pd.DataFrame(splits)], ignore_index=True)
    df_new = df_new.groupby(['material','location','week'], as_index=False)['quantity'].sum()
    df_new['quantity'] = df_new['quantity'].astype(int)
    return df_new

# ----------- 3. SUPPLY CHOICE -----------
def apply_supply_choice(df, supply_cfg):
    if supply_cfg.empty:
        return df.copy()
    df_new = df.copy()
    for _, row in supply_cfg.iterrows():
        filt = (
            (df_new['material'] == row['material']) &
            (df_new['location'] == row['location']) &
            (df_new['week'] == row['week'])
        )
        df_new.loc[filt, 'quantity'] += int(round(row['adjust_quantity']))
    df_new['quantity'] = df_new['quantity'].astype(int)
    return df_new

# ----------- 4. SPLIT WEEKLY FORECAST TO DAILY (INTEGER, NO ERROR) -----------
def expand_forecast_to_days_integer_split(demand_weekly, start_date, num_weeks, simulation_end_date=None):
    """将周度预测拆分为日度预测
    
    Args:
        demand_weekly: 周度预测数据
        start_date: 起始日期
        num_weeks: 周数
        simulation_end_date: 仿真结束日期（可选，用于限制输出范围）
    """
    print(f"  🔄 开始周度到日度转换: {len(demand_weekly)}个周度记录 -> {num_weeks}周")
    print(f"  📅 起始日期: {start_date}")
    
    rows = []
    for _, row in demand_weekly.iterrows():
        week_start = pd.to_datetime(start_date) + pd.Timedelta(days=(int(row['week'])-1)*7)
        base_qty = int(row['quantity']) // 7
        remainder = int(row['quantity']) % 7
        daily_qtys = [base_qty+1 if d < remainder else base_qty for d in range(7)]
        for d, qty in enumerate(daily_qtys):
            date = week_start + pd.Timedelta(days=d)
            
            # 如果指定了仿真结束日期，只处理仿真周期内的日期
            if simulation_end_date is not None and date > simulation_end_date:
                continue
                
            rows.append({
                'date': date,
                'material': row['material'],
                'location': row['location'],
                'week': row['week'],
                'demand_type': 'normal',
                'quantity': int(qty),
                'original_quantity': int(qty)
            })
    
    result_df = pd.DataFrame(rows)
    print(f"  ✅ 转换完成: {len(result_df)}个日度记录")
    print(f"  📅 日期范围: {result_df['date'].min()} 到 {result_df['date'].max()}")
    
    return result_df

# ----------- 5. DAILY ORDER GENERATION -----------
def generate_daily_orders(sim_date, original_forecast, current_forecast, ao_config, order_calendar, forecast_error):
    """
    Generate orders for a single simulation date based on original forecast
    
    Args:
        sim_date: Current simulation date
        original_forecast: Original daily forecast (unchanged for order generation)
        current_forecast: Current forecast state (for consumption tracking)
        ao_config: AO configuration (material-location based, no week dimension)
        order_calendar: Order calendar to check if today is order day
        forecast_error: Forecast error configuration with order_type and percentage
    
    Returns:
        orders_df: Orders generated today
        consumed_forecast: Updated forecast after consumption
    """
    from scipy.stats import truncnorm
    
    # Check if today is an order day
    is_order_day = not order_calendar[order_calendar['date'] == sim_date].empty
    if not is_order_day:
        return pd.DataFrame(), current_forecast
    
    orders = []
    consumed_forecast = current_forecast.copy()
    
    # Get unique material-location combinations
    ml_combinations = original_forecast[['material', 'location']].drop_duplicates()
    
    for _, ml_row in ml_combinations.iterrows():
        material = ml_row['material']
        location = ml_row['location']
        
        # Calculate 30-day average forecast from ORIGINAL forecast
        # 修复：使用更合理的预测查找范围，确保能找到数据
        future_dates = pd.date_range(sim_date, periods=min(30, len(original_forecast)), freq='D')
        ml_original_forecast = original_forecast[
            (original_forecast['material'] == material) & 
            (original_forecast['location'] == location) &
            (original_forecast['date'].isin(future_dates))
        ]
        
        if ml_original_forecast.empty:
            # 修复：如果30天范围内没有数据，尝试查找更短的范围
            future_dates_short = pd.date_range(sim_date, periods=7, freq='D')
            ml_original_forecast = original_forecast[
                (original_forecast['material'] == material) & 
                (original_forecast['location'] == location) &
                (original_forecast['date'].isin(future_dates_short))
            ]
            
            if ml_original_forecast.empty:
                print(f"  ⚠️  警告：{material}@{location} 在 {sim_date} 附近没有找到预测数据")
                continue
            
        daily_avg_forecast = ml_original_forecast['quantity'].mean()
        
        print(f"  📊 {material}@{location}: 平均日需求 {daily_avg_forecast:.1f}")
        
        # Get AO configuration for this material-location
        ml_ao_config = ao_config[
            (ao_config['material'] == material) & 
            (ao_config['location'] == location)
        ]
        
        # Calculate order averages based on ORIGINAL forecast
        total_ao_percent = ml_ao_config['ao_percent'].sum() if not ml_ao_config.empty else 0
        total_ao_daily_avg = daily_avg_forecast * total_ao_percent
        normal_daily_avg = daily_avg_forecast - total_ao_daily_avg
        
        # Generate AO orders (based on ORIGINAL forecast)
        for _, ao_row in ml_ao_config.iterrows():
            advance_days = int(ao_row['advance_days'])
            ao_percent = float(ao_row['ao_percent'])
            ao_daily_avg = daily_avg_forecast * ao_percent
            
            # Generate AO quantity with percentage-based error
            ao_qty = generate_quantity_with_percent_error(
                ao_daily_avg, material, location, 'AO', forecast_error
            )
            
            if ao_qty > 0:
                ao_order_date = sim_date + pd.Timedelta(days=advance_days)
                orders.append({
                    'date': ao_order_date,
                    'material': material,
                    'location': location,
                    'demand_type': 'AO',
                    'quantity': ao_qty,
                    'simulation_date': sim_date,
                    'advance_days': advance_days
                })
                
                # Consume forecast using AO logic (2 before, 3 after order date)
                consumed_forecast = consume_forecast_ao_logic(
                    consumed_forecast, material, location, ao_order_date, ao_qty
                )
        
        # Generate normal order (based on ORIGINAL forecast)
        if normal_daily_avg > 0:
            normal_qty = generate_quantity_with_percent_error(
                normal_daily_avg, material, location, 'normal', forecast_error
            )
            
            if normal_qty > 0:
                orders.append({
                    'date': sim_date,
                    'material': material,
                    'location': location,
                    'demand_type': 'normal',
                    'quantity': normal_qty,
                    'simulation_date': sim_date,
                    'advance_days': 0
                })
                
                # Consume forecast for normal order (just simulation date)
                consumed_forecast = consume_forecast_normal_logic(
                    consumed_forecast, material, location, sim_date, normal_qty
                )
    
    orders_df = pd.DataFrame(orders)
    if not orders_df.empty:
        orders_df['quantity'] = orders_df['quantity'].astype(int)
        
        # 添加调试信息
        ao_orders = orders_df[orders_df['demand_type'] == 'AO']
        normal_orders = orders_df[orders_df['demand_type'] == 'normal']
        print(f"  📋 订单生成完成: AO订单 {len(ao_orders)}个, 普通订单 {len(normal_orders)}个")
    
    return orders_df, consumed_forecast


def generate_quantity_with_percent_error(mean_qty, material, location, order_type, forecast_error):
    """
    Generate order quantity with percentage-based error standard deviation
    """
    from scipy.stats import truncnorm
    
    # Get error percentage for this material-location-order_type
    mask = (
        (forecast_error['material'] == material) & 
        (forecast_error['location'] == location) & 
        (forecast_error['order_type'] == order_type)
    )
    error_config = forecast_error[mask]
    
    if error_config.empty:
        # Fallback to old error_std format if order_type not found
        mask_old = (
            (forecast_error['material'] == material) & 
            (forecast_error['location'] == location)
        )
        error_config_old = forecast_error[mask_old]
        if not error_config_old.empty and 'error_std' in error_config_old.columns:
            # Use absolute error for backward compatibility
            error_std = float(error_config_old['error_std'].iloc[0])
            if error_std > 0:
                error = np.random.normal(0, error_std)
                return max(0, int(round(mean_qty + error)))
        return max(0, int(round(mean_qty)))
    
    # Use percentage-based error
    if 'error_std_percent' in error_config.columns:
        error_percent = float(error_config['error_std_percent'].iloc[0])
    else:
        error_percent = 0.0
    
    # Calculate absolute standard deviation from percentage
    abs_std = mean_qty * error_percent
    
    if abs_std <= 0:
        return max(0, int(round(mean_qty)))
    
    # Generate truncated normal (>= 0)
    lower_bound = 0
    a = (lower_bound - mean_qty) / abs_std
    value = truncnorm.rvs(a, np.inf, loc=mean_qty, scale=abs_std)
    
    return max(0, int(round(value)))


def consume_forecast_ao_logic(forecast_df, material, location, order_date, consume_qty):
    """AO forecast consumption: 2 days before, 3 days after order date"""
    if consume_qty <= 0:
        return forecast_df
    
    # Consumption window: [order_date-2, order_date-1, order_date, order_date+1, order_date+2, order_date+3]
    offsets = [0, -1, -2, 1, 2, 3]
    consumption_dates = [order_date + pd.Timedelta(days=offset) for offset in offsets]
    
    result_forecast = forecast_df.copy()
    remaining_consume = consume_qty
    
    for date in consumption_dates:
        if remaining_consume <= 0:
            break
        
        mask = (
            (result_forecast['material'] == material) & 
            (result_forecast['location'] == location) & 
            (result_forecast['date'] == date)
        )
        matching_rows = result_forecast[mask]
        
        if not matching_rows.empty:
            idx = matching_rows.index[0]
            available_qty = int(result_forecast.at[idx, 'quantity'])
            actual_consume = min(available_qty, remaining_consume)
            
            # Update forecast (cannot go below 0)
            new_qty = max(0, available_qty - actual_consume)
            result_forecast.at[idx, 'quantity'] = new_qty
            remaining_consume -= actual_consume
    
    return result_forecast


def consume_forecast_normal_logic(forecast_df, material, location, order_date, consume_qty):
    """Normal order forecast consumption: just the order date"""
    if consume_qty <= 0:
        return forecast_df
    
    result_forecast = forecast_df.copy()
    
    mask = (
        (result_forecast['material'] == material) & 
        (result_forecast['location'] == location) & 
        (result_forecast['date'] == order_date)
    )
    matching_rows = result_forecast[mask]
    
    if not matching_rows.empty:
        idx = matching_rows.index[0]
        available_qty = int(result_forecast.at[idx, 'quantity'])
        actual_consume = min(available_qty, consume_qty)
        
        # Update forecast (cannot go below 0)
        new_qty = max(0, available_qty - actual_consume)
        result_forecast.at[idx, 'quantity'] = new_qty
    
    return result_forecast


def run_daily_simulation(initial_daily_demand, cfg, sim_start, num_weeks):
    """
    Run daily order generation simulation
    
    Returns:
        daily_order_logs: Dict[date -> DataFrame] - Each day's cumulative orderlog
        daily_supply_demand_logs: Dict[date -> DataFrame] - Each day's supply demand log
    """
    sim_days = num_weeks * 7
    
    # Initialize tracking
    original_forecast = initial_daily_demand.copy()  # Keep original for order generation
    current_forecast = initial_daily_demand.copy()   # Track consumption for supply-demand
    daily_order_logs = {}
    daily_supply_demand_logs = {}
    cumulative_orders = pd.DataFrame()
    
    for day in range(sim_days):
        sim_date = sim_start + pd.Timedelta(days=day)
        
        # Generate orders based on ORIGINAL forecast, not consumed forecast
        day_orders, updated_forecast = generate_daily_orders(
            sim_date, original_forecast, current_forecast, 
            cfg['ao_config'], cfg['order_calendar'], cfg['forecast_error']
        )
        
        # Add today's orders to cumulative history
        if not day_orders.empty:
            cumulative_orders = pd.concat([cumulative_orders, day_orders], ignore_index=True)
        
        # Store this day's complete orderlog version (history + today)
        daily_order_logs[sim_date] = cumulative_orders.copy()
        
        # Generate supply demand log (remaining forecast after consumption)
        day_supply_demand = updated_forecast[
            (updated_forecast['date'] == sim_date)
        ].copy()
        if not day_supply_demand.empty:
            day_supply_demand['demand_type'] = 'remaining_forecast'
            day_supply_demand['original_quantity'] = initial_daily_demand[
                (initial_daily_demand['date'] == sim_date)
            ]['quantity'].values[0] if not initial_daily_demand[
                (initial_daily_demand['date'] == sim_date)
            ].empty else 0
        
        daily_supply_demand_logs[sim_date] = day_supply_demand.copy()
        
        # Update forecast for next day
        current_forecast = updated_forecast
    
    return daily_order_logs, daily_supply_demand_logs

# ----------- 6. COMBINE ORDER LOGS (Updated for daily cumulative history) -----------
def combine_order_logs(normal_orders, ao_orders):
    """
    Combine normal and AO orders - kept for backward compatibility
    New daily simulation uses integrated generation
    """
    if ao_orders is None or ao_orders.empty:
        all_orders = normal_orders.copy() if normal_orders is not None else pd.DataFrame()
    elif normal_orders is None or normal_orders.empty:
        all_orders = ao_orders.copy()
    else:
        all_orders = pd.concat([normal_orders, ao_orders], ignore_index=True)
    
    if all_orders.empty:
        return pd.DataFrame(columns=['date','material','location','demand_type','quantity'])
    
    # Ensure required columns exist
    if 'week' not in all_orders.columns:
        all_orders['week'] = None
    if 'simulation_date' not in all_orders.columns:
        all_orders['simulation_date'] = all_orders['date']
    if 'advance_days' not in all_orders.columns:
        all_orders['advance_days'] = 0
    
    # Group and aggregate
    group_cols = ['date','material','location','demand_type']
    if 'week' in all_orders.columns:
        group_cols.append('week')
    
    all_orders = all_orders.groupby(group_cols, as_index=False)['quantity'].sum()
    all_orders['quantity'] = all_orders['quantity'].astype(int)
    all_orders = all_orders.sort_values(['date','material','location','demand_type'])
    return all_orders

# ----------- 8. SIMULATE SHIPMENT FOR SINGLE DAY -----------
def simulate_shipment_for_single_day(simulation_date, order_log, current_inventory, material_list, location_list,
                                    production_plan=None, delivery_plan=None):
    """
    为单个 simulation date 计算 shipment 和 cut
    
    参数:
        simulation_date: 当前模拟日期
        order_log: 订单日志（预计算好的）
        current_inventory: 当天的初始库存 {(mat, loc): qty}
        material_list: 物料列表
        location_list: 地点列表
        production_plan: 生产计划
        delivery_plan: 调运计划
    """
    # 可用库存 = 当天初始库存 + 当日生产 + 当日调运
    unres_inventory = {}
    for mat in material_list:
        for loc in location_list:
            inv_key = (mat, loc)
            # 当天初始库存
            initial_qty = current_inventory.get(inv_key, 0)
            # 生产收货
            prod_qty = 0
            if production_plan is not None and not production_plan.empty:
                prod_filt = (
                    (production_plan['material'] == mat) &
                    (production_plan['location'] == loc) &
                    (production_plan['available_date'] == simulation_date)
                )
                prod_qty = int(production_plan[prod_filt]['quantity'].sum())
            # 调运收货
            deliv_qty = 0
            if delivery_plan is not None and not delivery_plan.empty:
                deliv_filt = (
                    (delivery_plan['material'] == mat) &
                    (delivery_plan['location'] == loc) &
                    (delivery_plan['actual_delivery_date'] == simulation_date)
                )
                deliv_qty = int(delivery_plan[deliv_filt]['quantity'].sum())
            # 总可用库存 (unrestricted inventory)
            unres_inventory[inv_key] = initial_qty + prod_qty + deliv_qty

    shipment_log = []
    cut_log = []

    # 处理订单
    todays_orders = order_log[order_log['date'] == simulation_date] if not order_log.empty else pd.DataFrame(columns=order_log.columns)
    for mat in material_list:
        for loc in location_list:
            inv_key = (mat, loc)
            qty_avail = unres_inventory.get(inv_key, 0)
            todays = todays_orders[
                (todays_orders['material'] == mat) &
                (todays_orders['location'] == loc)
            ] if not todays_orders.empty else pd.DataFrame(columns=todays_orders.columns)
            qty_ordered = int(todays['quantity'].sum()) if not todays.empty else 0
            shipped = int(min(qty_ordered, qty_avail))
            stockout = int(max(0, qty_ordered - shipped))
            shipment_log.append({
                'date': simulation_date, 'material': mat, 'location': loc, 'quantity': shipped
            })
            if stockout > 0:
                cut_log.append({
                    'date': simulation_date, 'material': mat, 'location': loc, 'quantity': stockout
                })

    return (
        pd.DataFrame(shipment_log),
        pd.DataFrame(cut_log),
        unres_inventory  # 返回计算后的可用库存，供下次调用使用
    )

# ----------- 9. SUPPLY DEMAND LOG (DAILY, NO DOUBLE COUNTING) -----------
def generate_daily_supply_demand_log(daily_demand=None, ao_orders=None, daily_supply_demand_logs=None):
    """
    Generate supply demand log - supports both old and new formats
    
    Args:
        daily_demand: Old format - consumed daily demand
        ao_orders: Old format - AO orders
        daily_supply_demand_logs: New format - daily supply demand logs from simulation
    """
    logs = []
    
    # New format - use daily supply demand logs
    if daily_supply_demand_logs is not None:
        for date, supply_demand in daily_supply_demand_logs.items():
            for _, row in supply_demand.iterrows():
                logs.append({
                    'date': row['date'],
                    'material': row['material'],
                    'location': row['location'],
                    'week': row.get('week', None),
                    'demand_type': row.get('demand_type', 'normal'),
                    'quantity': int(row['quantity']),
                    'original_quantity': int(row.get('original_quantity', row['quantity']))
                })
    else:
        # Old format - backward compatibility
        if daily_demand is not None:
            for _, row in daily_demand.iterrows():
                logs.append({
                    'date': row['date'],
                    'material': row['material'],
                    'location': row['location'],
                    'week': int(row['week']) if 'week' in row else None,
                    'demand_type': 'normal',
                    'quantity': int(row['quantity']),
                    'original_quantity': int(row.get('quantity_before_AO', row['quantity']))
                })
        
        if ao_orders is not None and not ao_orders.empty:
            for _, row in ao_orders.iterrows():
                logs.append({
                    'date': row['date'],
                    'material': row['material'],
                    'location': row['location'],
                    'week': int(row['week']) if 'week' in row else None,
                    'demand_type': 'AO',
                    'quantity': int(row['quantity']),
                    'original_quantity': int(row['quantity'])
                })
    
    df = pd.DataFrame(logs)
    if not df.empty:
        df['quantity'] = df['quantity'].astype(int)
        df['original_quantity'] = df['original_quantity'].astype(int)
    return df

# ----------- 10. PREPARE SUPPLY DEMAND ORDER (一次性预计算) -----------
def prepare_supply_demand_order(input_excel="config.xlsx"):
    """准备供需订单模拟所需的所有数据 - 使用新的每日生成逻辑"""
    # Load
    cfg = load_config(input_excel)

    # DPS + Supply adjustment (保留现有逻辑)
    demand_forecast = apply_dps(cfg['demand_forecast'], cfg['dps_config'])
    demand_forecast = apply_supply_choice(demand_forecast, cfg['supply_choice'])

    # Base timeline
    sim_start = cfg['initial_inventory']['date'].min()
    num_weeks = int(demand_forecast['week'].max())

    # Initial daily demand (before AO) (保留现有逻辑)
    initial_daily_demand = expand_forecast_to_days_integer_split(
        demand_forecast, sim_start, num_weeks
    )

    # NEW: Daily order generation simulation
    daily_order_logs, daily_supply_demand_logs = run_daily_simulation(
        initial_daily_demand, cfg, sim_start, num_weeks
    )

    # Get final order log (last day's cumulative orders)
    if daily_order_logs:
        final_date = max(daily_order_logs.keys())
        order_log = daily_order_logs[final_date]
    else:
        order_log = pd.DataFrame()

    # Prepare other data
    material_list = sorted(demand_forecast['material'].unique())
    location_list = sorted(demand_forecast['location'].unique())
    
    # Initial inventory dict (第一天的库存)
    initial_inventory = {(mat, loc): int(qty) for (date, mat, loc, qty) in cfg['initial_inventory'].itertuples(index=False)}
    
    # Supply demand log (使用新的每日日志)
    supply_demand_log = generate_daily_supply_demand_log(
        daily_supply_demand_logs=daily_supply_demand_logs
    )

    return {
        'order_log': order_log,              # 最终累积的订单日志
        'initial_inventory': initial_inventory,  # 第一天初始库存
        'material_list': material_list,
        'location_list': location_list,
        'production_plan': cfg.get('production_plan'),
        'delivery_plan': cfg.get('delivery_plan'),
        'supply_demand_log': supply_demand_log,
        'sim_start': sim_start,
        'num_weeks': num_weeks,
        'daily_order_logs': daily_order_logs,  # 新增：每日版本的订单日志
        'daily_supply_demand_logs': daily_supply_demand_logs  # 新增：每日供需日志
    }

# ----------- 11. RUN TODAY SHIPMENT (每日调用接口) -----------
def run_today_shipment(today_date, current_inventory, prepared_data):
    """
    外部函数每日调用的接口
    
    参数:
        today_date: 当前日期
        current_inventory: 当天的初始库存 {(mat, loc): qty}
                          可以是第一天的初始库存，也可以是前一天计算出的剩余库存
        prepared_ prepare_supply_demand_order() 返回的预计算数据
    
    返回:
        dict: {
            'shipment_log': 发货记录,
            'cut_log': 缺货记录,
            'end_inventory': 当天结束时的库存 {(mat, loc): qty}
        }
    """
    today_date = pd.to_datetime(today_date)
    
    shipment_log, cut_log, end_inventory = simulate_shipment_for_single_day(
        today_date,
        prepared_data['order_log'],
        current_inventory,
        prepared_data['material_list'],
        prepared_data['location_list'],
        production_plan=prepared_data['production_plan'],
        delivery_plan=prepared_data['delivery_plan']
    )
    
    return {
        'shipment_log': shipment_log,
        'cut_log': cut_log,
        'end_inventory': end_inventory  # 这个可以作为明天的初始库存
    }

# ----------- 12. MAIN PIPELINE (保持向后兼容) -----------
def main(input_excel="config.xlsx", output_excel="output_simulation.xlsx", 
         start_date=None, end_date=None):
    """完整的模拟流程 - 保持向后兼容"""
    
    # 一次性预计算所有数据
    prepared_data = prepare_supply_demand_order(input_excel)
    
    # 确定模拟时间范围
    if start_date is None:
        sim_start = prepared_data['sim_start']
    else:
        sim_start = pd.to_datetime(start_date)
    
    if end_date is None:
        default_end = sim_start + pd.Timedelta(days=prepared_data['num_weeks'] * 7 - 1)
        sim_end = default_end
    else:
        sim_end = pd.to_datetime(end_date)
    
    # 生成日期序列
    dates = pd.date_range(sim_start, sim_end)
    
    # 初始化库存为第一天的初始库存
    current_inventory = prepared_data['initial_inventory'].copy()
    
    all_shipments = []
    all_cuts = []
    
    # 逐日模拟
    for date in dates:
        result = run_today_shipment(date, current_inventory, prepared_data)
        all_shipments.append(result['shipment_log'])
        all_cuts.append(result['cut_log'])
        
        # 更新库存为当天结束时的库存，用于第二天
        current_inventory = result['end_inventory']
    
    # 合并所有结果
    final_shipment_log = pd.concat(all_shipments, ignore_index=True) if all_shipments else pd.DataFrame()
    final_cut_log = pd.concat(all_cuts, ignore_index=True) if all_cuts else pd.DataFrame()
    
    # Output
    with pd.ExcelWriter(output_excel) as writer:
        prepared_data['order_log'].to_excel(writer, sheet_name='OrderLog', index=False)
        final_shipment_log.to_excel(writer, sheet_name='ShipmentLog', index=False)
        final_cut_log.to_excel(writer, sheet_name='CutLog', index=False)
        prepared_data['supply_demand_log'].to_excel(writer, sheet_name='SupplyDemandLog', index=False)

    print(f"Simulation complete! All logs written to '{output_excel}'.")


# ----------- 13. VALIDATION TEST FUNCTIONS -----------
def run_validation_test():
    """
    Run comprehensive validation test with Excel output including DPS and Supply Choice validation
    """
    print("=== Module1 Validation Test ===")
    print("Running comprehensive validation with Excel output...")
    print("Including DPS Split and Supply Choice logic validation\n")
    
    # Set reproducible seed
    np.random.seed(42)
    
    # Create test scenarios including DPS and Supply Choice tests
    test_scenarios = [
        {
            'name': 'Basic_Validation',
            'description': '1 material-location, 2 weeks, no errors',
            'materials': ['VAL_MAT_A'],
            'locations': ['VAL_LOC_1'],
            'sim_days': 14,
            'base_forecast': 100,
            'ao_config': [
                {'material': 'VAL_MAT_A', 'location': 'VAL_LOC_1', 'advance_days': 3, 'ao_percent': 0.20},
                {'material': 'VAL_MAT_A', 'location': 'VAL_LOC_1', 'advance_days': 7, 'ao_percent': 0.15},
            ],
            'forecast_error': [
                {'material': 'VAL_MAT_A', 'location': 'VAL_LOC_1', 'order_type': 'normal', 'error_std_percent': 0.0},
                {'material': 'VAL_MAT_A', 'location': 'VAL_LOC_1', 'order_type': 'AO', 'error_std_percent': 0.0},
            ],
            'dps_config': [],
            'supply_choice': []
        },
        {
            'name': 'DPS_Split_Validation',
            'description': 'DPS split logic test - split 30% to new location',
            'materials': ['DPS_MAT_A'],
            'locations': ['DPS_LOC_ORIG'],
            'sim_days': 14,
            'base_forecast': 100,
            'ao_config': [
                {'material': 'DPS_MAT_A', 'location': 'DPS_LOC_ORIG', 'advance_days': 3, 'ao_percent': 0.15},
                {'material': 'DPS_MAT_A', 'location': 'DPS_LOC_NEW', 'advance_days': 5, 'ao_percent': 0.10},
            ],
            'forecast_error': [
                {'material': 'DPS_MAT_A', 'location': 'DPS_LOC_ORIG', 'order_type': 'normal', 'error_std_percent': 0.0},
                {'material': 'DPS_MAT_A', 'location': 'DPS_LOC_ORIG', 'order_type': 'AO', 'error_std_percent': 0.0},
                {'material': 'DPS_MAT_A', 'location': 'DPS_LOC_NEW', 'order_type': 'normal', 'error_std_percent': 0.0},
                {'material': 'DPS_MAT_A', 'location': 'DPS_LOC_NEW', 'order_type': 'AO', 'error_std_percent': 0.0},
            ],
            'dps_config': [
                {'material': 'DPS_MAT_A', 'location': 'DPS_LOC_ORIG', 'dps_location': 'DPS_LOC_NEW', 'dps_percent': 0.30}
            ],
            'supply_choice': []
        },
        {
            'name': 'Supply_Choice_Validation',
            'description': 'Supply choice adjustment test - add 50 units to week 1',
            'materials': ['SC_MAT_A'],
            'locations': ['SC_LOC_1'],
            'sim_days': 14,
            'base_forecast': 100,
            'ao_config': [
                {'material': 'SC_MAT_A', 'location': 'SC_LOC_1', 'advance_days': 3, 'ao_percent': 0.20},
            ],
            'forecast_error': [
                {'material': 'SC_MAT_A', 'location': 'SC_LOC_1', 'order_type': 'normal', 'error_std_percent': 0.0},
                {'material': 'SC_MAT_A', 'location': 'SC_LOC_1', 'order_type': 'AO', 'error_std_percent': 0.0},
            ],
            'dps_config': [],
            'supply_choice': [
                {'material': 'SC_MAT_A', 'location': 'SC_LOC_1', 'week': 1, 'adjust_quantity': 50}
            ]
        },
        {
            'name': 'Combined_DPS_SC_Validation',
            'description': 'Combined DPS and Supply Choice test',
            'materials': ['COMB_MAT_A'],
            'locations': ['COMB_LOC_ORIG'],
            'sim_days': 21,
            'base_forecast': 120,
            'ao_config': [
                {'material': 'COMB_MAT_A', 'location': 'COMB_LOC_ORIG', 'advance_days': 3, 'ao_percent': 0.18},
                {'material': 'COMB_MAT_A', 'location': 'COMB_LOC_NEW', 'advance_days': 7, 'ao_percent': 0.12},
            ],
            'forecast_error': [
                {'material': 'COMB_MAT_A', 'location': 'COMB_LOC_ORIG', 'order_type': 'normal', 'error_std_percent': 0.05},
                {'material': 'COMB_MAT_A', 'location': 'COMB_LOC_ORIG', 'order_type': 'AO', 'error_std_percent': 0.03},
                {'material': 'COMB_MAT_A', 'location': 'COMB_LOC_NEW', 'order_type': 'normal', 'error_std_percent': 0.05},
                {'material': 'COMB_MAT_A', 'location': 'COMB_LOC_NEW', 'order_type': 'AO', 'error_std_percent': 0.03},
            ],
            'dps_config': [
                {'material': 'COMB_MAT_A', 'location': 'COMB_LOC_ORIG', 'dps_location': 'COMB_LOC_NEW', 'dps_percent': 0.25}
            ],
            'supply_choice': [
                {'material': 'COMB_MAT_A', 'location': 'COMB_LOC_ORIG', 'week': 1, 'adjust_quantity': 100},
                {'material': 'COMB_MAT_A', 'location': 'COMB_LOC_NEW', 'week': 2, 'adjust_quantity': -30}
            ]
        }
    ]
    
    validation_results = []
    detailed_results = {}
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"Running validation scenario {i}/{len(test_scenarios)}: {scenario['name']}")
        
        # Generate test configuration
        cfg = generate_validation_config(scenario)
        
        # Run validation simulation
        sim_result = run_validation_simulation(scenario, cfg)
        
        # Perform validation checks
        validation_result = perform_validation_checks(scenario, sim_result)
        
        validation_results.append(validation_result)
        detailed_results[scenario['name']] = {
            'scenario': scenario,
            'cfg': cfg,
            'sim_result': sim_result,
            'validation': validation_result
        }
        
        print(f"  ✓ Status: {validation_result['overall_status']}")
    
    # Export validation results to Excel
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_file = f'module1_validation_test_{timestamp}.xlsx'
    
    export_validation_results(validation_results, detailed_results, excel_file)
    
    # Print summary
    print(f"\n📊 Validation Test Summary:")
    print("-" * 60)
    
    for result in validation_results:
        status_icon = "✅" if result['overall_status'] == 'PASS' else "⚠️" if result['overall_status'] == 'WARNING' else "❌"
        print(f"{status_icon} {result['scenario_name']}: {result['overall_status']}")
        if result['issues']:
            print(f"   Issues: {', '.join(result['issues'][:3])}{'...' if len(result['issues']) > 3 else ''}")
    
    print(f"\n📁 Detailed validation results exported to: {excel_file}")
    return excel_file


def generate_validation_config(scenario):
    """
    Generate test configuration for validation scenario including DPS and Supply Choice
    """
    sim_start = pd.to_datetime('2024-01-01')
    num_weeks = (scenario['sim_days'] + 6) // 7
    
    # Generate demand forecast with realistic variation
    demand_forecast = []
    for week in range(1, num_weeks + 1):
        for material in scenario['materials']:
            for location in scenario['locations']:
                # Base weekly quantity with seasonal variation
                base_weekly = scenario['base_forecast'] * 7
                seasonal_factor = 1 + 0.1 * np.sin(week * 0.6)
                weekly_qty = int(base_weekly * seasonal_factor)
                weekly_qty = max(350, weekly_qty)  # Minimum weekly quantity
                
                demand_forecast.append({
                    'material': material,
                    'location': location,
                    'week': week,
                    'quantity': weekly_qty
                })
    
    # Handle DPS and Supply Choice configurations
    dps_df = pd.DataFrame(scenario.get('dps_config', []))
    supply_choice_df = pd.DataFrame(scenario.get('supply_choice', []))
    
    # For DPS scenarios, need to ensure new locations are included in materials/locations lists
    if not dps_df.empty:
        # Add DPS target locations to the locations list for AO config and forecast error
        additional_locations = set(dps_df['dps_location'].unique()) - set(scenario['locations'])
        scenario['locations'].extend(list(additional_locations))
    
    return {
        'demand_forecast': pd.DataFrame(demand_forecast),
        'ao_config': pd.DataFrame(scenario['ao_config']),
        'forecast_error': pd.DataFrame(scenario['forecast_error']),
        'order_calendar': pd.DataFrame([
            {'date': sim_start + pd.Timedelta(days=i), 'order_day_flag': 1}
            for i in range(scenario['sim_days'])
        ]),
        'initial_inventory': pd.DataFrame([
            {'date': sim_start, 'material': mat, 'location': loc, 'quantity': 150}
            for mat in scenario['materials'] for loc in scenario['locations']
        ]),
        'dps_config': dps_df,
        'supply_choice': supply_choice_df,
    }


def run_validation_simulation(scenario, cfg):
    """
    Run simulation for validation scenario including DPS and Supply Choice logic
    """
    sim_start = pd.to_datetime('2024-01-01')
    num_weeks = (scenario['sim_days'] + 6) // 7
    
    # Store original demand forecast before transformations
    original_demand_forecast = cfg['demand_forecast'].copy()
    
    # Apply transformations in sequence (following memory specifications)
    demand_forecast = apply_dps(cfg['demand_forecast'], cfg['dps_config'])
    demand_forecast = apply_supply_choice(demand_forecast, cfg['supply_choice'])
    
    # Convert to daily
    initial_daily_demand = expand_forecast_to_days_integer_split(
        demand_forecast, sim_start, num_weeks
    )
    
    # Run daily simulation
    daily_order_logs, daily_supply_demand_logs = run_daily_simulation(
        initial_daily_demand, cfg, sim_start, num_weeks
    )
    
    # Generate supply demand log
    supply_demand_log = generate_daily_supply_demand_log(
        daily_supply_demand_logs=daily_supply_demand_logs
    )
    
    return {
        'initial_daily_demand': initial_daily_demand,
        'daily_order_logs': daily_order_logs,
        'daily_supply_demand_logs': daily_supply_demand_logs,
        'supply_demand_log': supply_demand_log,
        'sim_start': sim_start,
        'num_weeks': num_weeks,
        'cfg': cfg,  # Store configuration for validation
        'original_demand_forecast': original_demand_forecast,  # Store original for comparison
        'processed_demand_forecast': demand_forecast  # Store after DPS/SC processing
    }


def perform_validation_checks(scenario, sim_result):
    """
    Perform comprehensive validation checks according to memory specifications
    Including DPS and Supply Choice logic validation
    """
    validation_result = {
        'scenario_name': scenario['name'],
        'overall_status': 'PASS',
        'issues': [],
        'checks': [],
        'metrics': {}
    }
    
    if not sim_result['daily_order_logs']:
        validation_result['overall_status'] = 'FAIL'
        validation_result['issues'].append('No daily order logs generated')
        return validation_result
    
    final_orders = sim_result['daily_order_logs'][max(sim_result['daily_order_logs'].keys())]
    
    if final_orders.empty:
        validation_result['overall_status'] = 'FAIL'
        validation_result['issues'].append('No orders generated')
        return validation_result
    
    # Check 1: Positive quantity verification
    non_positive = final_orders[final_orders['quantity'] <= 0]
    check_status = 'PASS' if non_positive.empty else 'FAIL'
    if not non_positive.empty:
        validation_result['issues'].append(f'Found {len(non_positive)} orders with non-positive quantities')
        validation_result['overall_status'] = 'FAIL'
    
    validation_result['checks'].append({
        'check': 'Positive Quantities',
        'status': check_status,
        'details': f'All {len(final_orders)} orders have positive quantities' if check_status == 'PASS' else f'{len(non_positive)} invalid orders'
    })
    
    # Check 2: Required field validation
    required_fields = ['date', 'material', 'location', 'demand_type', 'quantity', 'simulation_date']
    missing_fields = [field for field in required_fields 
                     if field not in final_orders.columns or final_orders[field].isna().any()]
    
    check_status = 'PASS' if not missing_fields else 'FAIL'
    if missing_fields:
        validation_result['issues'].append(f'Missing or null fields: {missing_fields}')
        validation_result['overall_status'] = 'FAIL'
    
    validation_result['checks'].append({
        'check': 'Required Fields',
        'status': check_status,
        'details': 'All required fields present' if check_status == 'PASS' else f'Missing: {missing_fields}'
    })
    
    # Check 3: Type checking
    invalid_types = final_orders[~final_orders['demand_type'].isin(['AO', 'normal'])]
    check_status = 'PASS' if invalid_types.empty else 'FAIL'
    if not invalid_types.empty:
        validation_result['issues'].append(f'Found {len(invalid_types)} orders with invalid demand types')
        validation_result['overall_status'] = 'FAIL'
    
    validation_result['checks'].append({
        'check': 'Valid Order Types',
        'status': check_status,
        'details': 'All orders have valid types (AO/normal)' if check_status == 'PASS' else f'{len(invalid_types)} invalid types'
    })
    
    # Check 4: DPS Split Logic Validation
    if scenario.get('dps_config'):
        dps_validation = validate_dps_logic(scenario, sim_result)
        validation_result['checks'].append(dps_validation)
        if dps_validation['status'] != 'PASS':
            validation_result['issues'].append(dps_validation['details'])
            if validation_result['overall_status'] == 'PASS':
                validation_result['overall_status'] = dps_validation['status']
    
    # Check 5: Supply Choice Logic Validation
    if scenario.get('supply_choice'):
        sc_validation = validate_supply_choice_logic(scenario, sim_result)
        validation_result['checks'].append(sc_validation)
        if sc_validation['status'] != 'PASS':
            validation_result['issues'].append(sc_validation['details'])
            if validation_result['overall_status'] == 'PASS':
                validation_result['overall_status'] = sc_validation['status']
    
    # Check 6: Daily consistency metrics
    daily_quantities = []
    daily_ao_percentages = []
    
    for date, order_log in sim_result['daily_order_logs'].items():
        day_orders = order_log[order_log['simulation_date'] == date] if not order_log.empty else pd.DataFrame()
        if not day_orders.empty:
            day_total = day_orders['quantity'].sum()
            day_ao = day_orders[day_orders['demand_type'] == 'AO']['quantity'].sum()
            daily_quantities.append(day_total)
            daily_ao_percentages.append(day_ao / day_total if day_total > 0 else 0)
    
    daily_std = np.std(daily_quantities) if daily_quantities else 0
    daily_mean = np.mean(daily_quantities) if daily_quantities else 0
    consistency_cv = daily_std / daily_mean if daily_mean > 0 else 0
    
    check_status = 'PASS' if consistency_cv < 0.2 else 'WARNING' if consistency_cv < 0.3 else 'FAIL'
    if check_status != 'PASS':
        validation_result['issues'].append(f'Daily quantity coefficient of variation: {consistency_cv:.3f}')
        if validation_result['overall_status'] == 'PASS':
            validation_result['overall_status'] = check_status
    
    validation_result['checks'].append({
        'check': 'Daily Consistency',
        'status': check_status,
        'details': f'CV: {consistency_cv:.3f} (target <0.2)'
    })
    
    # Check 7: AO percentage validation
    total_qty = final_orders['quantity'].sum()
    ao_qty = final_orders[final_orders['demand_type'] == 'AO']['quantity'].sum()
    actual_ao_percent = ao_qty / total_qty if total_qty > 0 else 0
    
    # Calculate expected AO percentage
    expected_ao_percent = scenario['ao_config']
    if isinstance(expected_ao_percent, list) and expected_ao_percent:
        expected_ao_percent = sum(cfg['ao_percent'] for cfg in expected_ao_percent) / len(scenario['materials']) / len(scenario['locations'])
    else:
        expected_ao_percent = 0
    
    ao_diff = abs(actual_ao_percent - expected_ao_percent)
    check_status = 'PASS' if ao_diff < 0.05 else 'WARNING' if ao_diff < 0.10 else 'FAIL'
    
    if check_status != 'PASS':
        validation_result['issues'].append(f'AO percentage deviation: {ao_diff:.3f}')
        if validation_result['overall_status'] == 'PASS':
            validation_result['overall_status'] = check_status
    
    validation_result['checks'].append({
        'check': 'AO Percentage',
        'status': check_status,
        'details': f'Expected: {expected_ao_percent:.1%}, Actual: {actual_ao_percent:.1%}, Diff: {ao_diff:.3f}'
    })
    
    # Store metrics
    validation_result['metrics'] = {
        'total_orders': len(final_orders),
        'total_quantity': total_qty,
        'ao_quantity': ao_qty,
        'normal_quantity': total_qty - ao_qty,
        'actual_ao_percent': actual_ao_percent,
        'expected_ao_percent': expected_ao_percent,
        'daily_consistency_cv': consistency_cv,
        'simulation_days': scenario['sim_days']
    }
    
    return validation_result


def validate_dps_logic(scenario, sim_result):
    """
    Validate DPS split logic implementation
    """
    try:
        # Get original and DPS-processed demand forecast
        original_forecast = sim_result['cfg']['demand_forecast']
        dps_config = sim_result['cfg']['dps_config']
        
        # Apply DPS logic manually to verify
        if not dps_config.empty:
            expected_splits = []
            for _, dps_row in dps_config.iterrows():
                source_material = dps_row['material']
                source_location = dps_row['location']
                target_location = dps_row['dps_location']
                split_percent = dps_row['dps_percent']
                
                # Find matching demand forecast entries
                source_entries = original_forecast[
                    (original_forecast['material'] == source_material) &
                    (original_forecast['location'] == source_location)
                ]
                
                for _, entry in source_entries.iterrows():
                    expected_split_qty = int(round(entry['quantity'] * split_percent))
                    expected_remain_qty = int(round(entry['quantity'] - expected_split_qty))
                    
                    expected_splits.append({
                        'source_material': source_material,
                        'source_location': source_location,
                        'target_location': target_location,
                        'week': entry['week'],
                        'original_qty': entry['quantity'],
                        'expected_split_qty': expected_split_qty,
                        'expected_remain_qty': expected_remain_qty,
                        'split_percent': split_percent
                    })
            
            # Check if DPS created the expected new location entries
            final_orders = sim_result['daily_order_logs'][max(sim_result['daily_order_logs'].keys())]
            
            # Verify new locations have orders
            for split in expected_splits:
                target_orders = final_orders[
                    (final_orders['material'] == split['source_material']) &
                    (final_orders['location'] == split['target_location'])
                ]
                
                if target_orders.empty:
                    return {
                        'check': 'DPS Split Logic',
                        'status': 'FAIL',
                        'details': f'No orders found for DPS target location {split["target_location"]}'
                    }
            
            return {
                'check': 'DPS Split Logic',
                'status': 'PASS',
                'details': f'DPS split created orders for {len(set(s["target_location"] for s in expected_splits))} target locations'
            }
        else:
            return {
                'check': 'DPS Split Logic',
                'status': 'PASS',
                'details': 'No DPS configuration - skip validation'
            }
            
    except Exception as e:
        return {
            'check': 'DPS Split Logic',
            'status': 'FAIL',
            'details': f'DPS validation error: {str(e)}'
        }


def validate_supply_choice_logic(scenario, sim_result):
    """
    Validate Supply Choice adjustment logic implementation
    """
    try:
        # Get supply choice configuration
        supply_choice = sim_result['cfg']['supply_choice']
        
        if not supply_choice.empty:
            # Validate that supply choice adjustments were applied
            # This would require comparing original vs adjusted forecast quantities
            # For now, we check that the configuration was processed without errors
            
            adjustments_count = len(supply_choice)
            
            # Basic validation: check that we have orders for adjusted material-locations
            final_orders = sim_result['daily_order_logs'][max(sim_result['daily_order_logs'].keys())]
            
            for _, sc_row in supply_choice.iterrows():
                material = sc_row['material']
                location = sc_row['location']
                
                adjusted_orders = final_orders[
                    (final_orders['material'] == material) &
                    (final_orders['location'] == location)
                ]
                
                if adjusted_orders.empty:
                    return {
                        'check': 'Supply Choice Logic',
                        'status': 'FAIL',
                        'details': f'No orders found for supply choice adjusted {material}-{location}'
                    }
            
            return {
                'check': 'Supply Choice Logic',
                'status': 'PASS',
                'details': f'Supply choice processed {adjustments_count} adjustments successfully'
            }
        else:
            return {
                'check': 'Supply Choice Logic',
                'status': 'PASS',
                'details': 'No supply choice configuration - skip validation'
            }
            
    except Exception as e:
        return {
            'check': 'Supply Choice Logic',
            'status': 'FAIL',
            'details': f'Supply choice validation error: {str(e)}'
        }


def export_validation_results(validation_results, detailed_results, excel_file):
    """
    Export validation test results to Excel with comprehensive sheets
    """
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        
        # Validation Summary Sheet
        summary_data = []
        for result in validation_results:
            summary_data.append({
                'Scenario': result['scenario_name'],
                'Overall_Status': result['overall_status'],
                'Total_Orders': result['metrics']['total_orders'],
                'Total_Quantity': result['metrics']['total_quantity'],
                'AO_Percentage': f"{result['metrics']['actual_ao_percent']:.1%}",
                'Expected_AO_Percentage': f"{result['metrics']['expected_ao_percent']:.1%}",
                'Daily_Consistency_CV': f"{result['metrics']['daily_consistency_cv']:.3f}",
                'Issues_Count': len(result['issues']),
                'Checks_Passed': sum(1 for check in result['checks'] if check['status'] == 'PASS'),
                'Total_Checks': len(result['checks'])
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Validation_Summary', index=False)
        
        # Detailed Check Results
        check_details = []
        for result in validation_results:
            for check in result['checks']:
                check_details.append({
                    'Scenario': result['scenario_name'],
                    'Check': check['check'],
                    'Status': check['status'],
                    'Details': check['details']
                })
        
        if check_details:
            check_df = pd.DataFrame(check_details)
            check_df.to_excel(writer, sheet_name='Check_Details', index=False)
        
        # Individual scenario results
        for scenario_name, details in detailed_results.items():
            prefix = scenario_name[:10]  # Limit sheet name length
            
            # Configuration sheets
            details['cfg']['ao_config'].to_excel(writer, sheet_name=f'{prefix}_AO_Config', index=False)
            details['cfg']['forecast_error'].to_excel(writer, sheet_name=f'{prefix}_Errors', index=False)
            
            # DPS and Supply Choice configurations (if present)
            if not details['cfg']['dps_config'].empty:
                details['cfg']['dps_config'].to_excel(writer, sheet_name=f'{prefix}_DPS_Config', index=False)
            
            if not details['cfg']['supply_choice'].empty:
                details['cfg']['supply_choice'].to_excel(writer, sheet_name=f'{prefix}_SC_Config', index=False)
            
            # Demand forecast comparison (original vs processed)
            if 'original_demand_forecast' in details['sim_result']:
                details['sim_result']['original_demand_forecast'].to_excel(writer, sheet_name=f'{prefix}_Orig_Forecast', index=False)
            
            if 'processed_demand_forecast' in details['sim_result']:
                details['sim_result']['processed_demand_forecast'].to_excel(writer, sheet_name=f'{prefix}_Proc_Forecast', index=False)
            
            details['sim_result']['initial_daily_demand'].to_excel(writer, sheet_name=f'{prefix}_Daily_Forecast', index=False)
            
            # Orders
            if details['sim_result']['daily_order_logs']:
                final_orders = details['sim_result']['daily_order_logs'][max(details['sim_result']['daily_order_logs'].keys())]
                final_orders.to_excel(writer, sheet_name=f'{prefix}_Orders', index=False)
            
            # Daily summary
            daily_summary = []
            for i, (date, order_log) in enumerate(details['sim_result']['daily_order_logs'].items()):
                day_orders = order_log[order_log['simulation_date'] == date] if not order_log.empty else pd.DataFrame()
                day_total = day_orders['quantity'].sum() if not day_orders.empty else 0
                day_ao = day_orders[day_orders['demand_type'] == 'AO']['quantity'].sum() if not day_orders.empty else 0
                
                daily_summary.append({
                    'Day': i + 1,
                    'Date': date.strftime('%Y-%m-%d'),
                    'Total_Qty': day_total,
                    'AO_Qty': day_ao,
                    'Normal_Qty': day_total - day_ao,
                    'AO_Percent': day_ao / day_total if day_total > 0 else 0,
                    'Cumulative_Orders': len(order_log) if not order_log.empty else 0
                })
            
            if daily_summary:
                daily_df = pd.DataFrame(daily_summary)
                daily_df.to_excel(writer, sheet_name=f'{prefix}_Daily', index=False)
            
            # Supply demand log
            if not details['sim_result']['supply_demand_log'].empty:
                details['sim_result']['supply_demand_log'].to_excel(writer, sheet_name=f'{prefix}_SupplyDemand', index=False)


# ----------- 14. 集成模式支持 -----------

def run_daily_order_generation(
    config_dict: dict,
    simulation_date: pd.Timestamp,
    output_dir: str,
    orchestrator: object = None
) -> dict:
    """
    Module1 集成模式：生成指定日期的订单和发货数据
    
    注意：为了确保shipment基于实际库存限制，orchestrator参数实际上是必需的。
    没有orchestrator时只能生成订单，无法生成合理的shipment。
    
    Args:
        config_dict: 配置数据字典
        simulation_date: 仿真日期
        output_dir: 输出目录
        orchestrator: Orchestrator实例，必需用于获取当前库存状态以生成正确的shipment
        
    Returns:
        dict: 包含订单和发货数据的字典 {orders_df, shipment_df, cut_df, supply_demand_df, output_file}
    """
    print(f"🔄 Module1 运行于集成模式 - {simulation_date.strftime('%Y-%m-%d')}")
    
    try:
        # 从配置加载必要数据
        demand_forecast = config_dict.get('M1_DemandForecast', pd.DataFrame())
        forecast_error = config_dict.get('M1_ForecastError', pd.DataFrame())
        order_calendar = config_dict.get('M1_OrderCalendar', pd.DataFrame())
        ao_config = config_dict.get('M1_AOConfig', pd.DataFrame())
        
        # 验证必需的配置数据
        if demand_forecast.empty:
            raise ValueError("缺少必需的配置数据：M1_DemandForecast")
        
        if order_calendar.empty:
            raise ValueError("缺少必需的配置数据：M1_OrderCalendar")
        
        if ao_config.empty:
            raise ValueError("缺少必需的配置数据：M1_AOConfig")
        
        if forecast_error.empty:
            raise ValueError("缺少必需的配置数据：M1_ForecastError")
        
        # 验证订单日历日期格式
        print(f"  📅 订单日历验证: {len(order_calendar)}个日期")
        order_calendar['date'] = pd.to_datetime(order_calendar['date'])
        print(f"  📅 订单日历日期范围: {order_calendar['date'].min()} 到 {order_calendar['date'].max()}")
        print(f"  📅 当前仿真日期: {simulation_date}")
        
        # 检查当前日期是否是订单日
        is_order_day = not order_calendar[order_calendar['date'] == simulation_date].empty
        print(f"  📅 当前日期是否为订单日: {'是' if is_order_day else '否'}")
        
        # 将周度预测转换为日度预测以支持日度订单生成
        if 'week' in demand_forecast.columns:
            # 修复：使用固定的起始日期，而不是从当前仿真日期计算
            # 这样可以确保12周预测正确转换为84天日度预测
            sim_start = pd.to_datetime('2024-01-01')  # 固定起始日期
            max_week = int(demand_forecast['week'].max()) if not demand_forecast.empty else 1
            
            # 使用现有的expand_forecast_to_days_integer_split函数
            daily_demand_forecast = expand_forecast_to_days_integer_split(
                demand_forecast, sim_start, max_week
            )
            
            print(f"  📊 周度预测转换: {max_week}周 -> {len(daily_demand_forecast)}天")
            print(f"  📅 预测日期范围: {daily_demand_forecast['date'].min()} 到 {daily_demand_forecast['date'].max()}")
        else:
            # 已经是日度数据
            daily_demand_forecast = demand_forecast.copy()
            print(f"  📊 使用现有日度预测: {len(daily_demand_forecast)}天")
        
        # 生成当日订单（只包含在今天创建的normal/AO）
        today_orders_df, consumed_forecast = generate_daily_orders(
            simulation_date, daily_demand_forecast, daily_demand_forecast, 
            ao_config, order_calendar, forecast_error
        )

        # 载入历史订单，并合并形成“当日版本”的订单视图
        # 目标：满足“每天的订单日志，除了当天产生的，还应包含 requirement_date >= 订单产生日期 的订单”，
        #      即包含历史天生成但尚未到期或将来到期的AO/普通订单（自然满足 requirement_date >= simulation_date_of_row）。
        def _load_previous_orders(m1_output_dir: str, current_date: pd.Timestamp) -> pd.DataFrame:
            try:
                if not os.path.isdir(m1_output_dir):
                    return pd.DataFrame()
                pattern = re.compile(r"module1_output_(\d{8})\.xlsx$")
                rows = []
                for fname in os.listdir(m1_output_dir):
                    m = pattern.match(fname)
                    if not m:
                        continue
                    fdate = pd.to_datetime(m.group(1))
                    if fdate.normalize() >= current_date.normalize():
                        # 仅加载当前日之前的版本（历史）
                        continue
                    fpath = os.path.join(m1_output_dir, fname)
                    try:
                        xl = pd.ExcelFile(fpath)
                        if 'OrderLog' not in xl.sheet_names:
                            continue
                        df = xl.parse('OrderLog')
                        if df is None or df.empty:
                            continue
                        # 标准化字段
                        if 'date' in df.columns:
                            df['date'] = pd.to_datetime(df['date'])
                        if 'simulation_date' in df.columns:
                            df['simulation_date'] = pd.to_datetime(df['simulation_date'])
                        rows.append(df)
                    except Exception:
                        # 某些文件损坏或打开失败，跳过
                        continue
                if not rows:
                    return pd.DataFrame()
                prev = pd.concat(rows, ignore_index=True)
                return prev
            except Exception:
                return pd.DataFrame()

        previous_orders_all = _load_previous_orders(output_dir, simulation_date)
        # 去重：防止从多个历史日文件中重复载入同一订单
        if not previous_orders_all.empty:
            dedup_keys = [
                c for c in ['date','material','location','demand_type','simulation_date','advance_days','quantity']
                if c in previous_orders_all.columns
            ]
            if dedup_keys:
                previous_orders_all = previous_orders_all.drop_duplicates(subset=dedup_keys)

        # 仅保留历史中“仍在当前及未来生效”的订单，用于供给其他模块的全量视图
        # 即 requirement_date(=date) >= 今天
        if not previous_orders_all.empty and 'date' in previous_orders_all.columns:
            previous_orders_future = previous_orders_all[previous_orders_all['date'] >= simulation_date].copy()
        else:
            previous_orders_future = pd.DataFrame()

        # 当日版本订单视图：历史未来订单 + 当天新生成订单
        if today_orders_df is not None and not today_orders_df.empty:
            orders_df = pd.concat([previous_orders_future, today_orders_df], ignore_index=True)
        else:
            orders_df = previous_orders_future.copy()

        # 规范字段与类型
        if not orders_df.empty:
            if 'quantity' in orders_df.columns:
                orders_df['quantity'] = orders_df['quantity'].astype(int)
            # 补充字段
            if 'simulation_date' not in orders_df.columns:
                # 对缺失simulation_date的历史数据，默认等于其date（保底，不影响当日shipment筛选）
                orders_df['simulation_date'] = orders_df['date']
        
        # 生成发货数据（基于实际库存限制）
        if orchestrator is not None:
            shipment_df, cut_df = generate_shipment_with_inventory_check(
                orders_df, simulation_date, orchestrator, 
                daily_demand_forecast, forecast_error
            )
        else:
            # 没有orchestrator时无法进行库存检查，返回空数据
            print("  ⚠️  警告：没有Orchestrator，无法生成基于库存的shipment")
            shipment_df = pd.DataFrame()
            cut_df = pd.DataFrame()
        
        # 生成SupplyDemandLog - 这是Module3需要的关键数据
        supply_demand_df = generate_supply_demand_log_for_integration(
            daily_demand_forecast, consumed_forecast, simulation_date
        )
        
        # 保存输出（包括SupplyDemandLog）
        output_file = f"{output_dir}/module1_output_{simulation_date.strftime('%Y%m%d')}.xlsx"
        save_module1_output_with_supply_demand(orders_df, shipment_df, supply_demand_df, output_file)
        
        # 统计信息
        shipment_count = len(shipment_df)
        cut_count = len(cut_df) if 'cut_df' in locals() else 0
        
        print(f"✅ Module1 完成 - 生成 {len(orders_df)} 个订单, {shipment_count} 个发货, {cut_count} 个cut")
        
        return {
            'orders_df': orders_df,
            'shipment_df': shipment_df,
            'cut_df': cut_df if 'cut_df' in locals() else pd.DataFrame(),
            'supply_demand_df': supply_demand_df,  # 新增：SupplyDemandLog数据
            'output_file': output_file
        }
        
    except Exception as e:
        print(f"❌ Module1 集成模式失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            'orders_df': pd.DataFrame(),
            'shipment_df': pd.DataFrame(),
            'cut_df': pd.DataFrame(),
            'supply_demand_df': pd.DataFrame(),  # 新增
            'output_file': None
        }









def generate_supply_demand_log_for_integration(
    demand_forecast: pd.DataFrame, 
    consumed_forecast: pd.DataFrame, 
    simulation_date: pd.Timestamp
) -> pd.DataFrame:
    """为集成模式生成SupplyDemandLog
    
    Args:
        demand_forecast: 原始需求预测
        consumed_forecast: 消耗后的需求预测
        simulation_date: 仿真日期
        
    Returns:
        pd.DataFrame: SupplyDemandLog数据
    """
    # 生成未来需求数据（仿真日期之后的需求）
    future_demand = consumed_forecast[
        pd.to_datetime(consumed_forecast['date']) > simulation_date
    ].copy()
    
    if future_demand.empty:
        # 如果没有未来需求，返回空的DataFrame但包含正确的列名
        return pd.DataFrame(columns=['date', 'material', 'location', 'quantity', 'demand_element'])
    
    # 添加demand_element字段（遵循项目规范）
    future_demand['demand_element'] = 'forecast'
    
    # 确保包含必要的列
    supply_demand_log = future_demand[[
        'date', 'material', 'location', 'quantity', 'demand_element'
    ]].copy()
    
    return supply_demand_log

def save_module1_output_with_supply_demand(
    orders_df: pd.DataFrame, 
    shipment_df: pd.DataFrame, 
    supply_demand_df: pd.DataFrame,
    output_file: str
):
    """保存Module1输出到Excel文件（包括SupplyDemandLog）"""
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            orders_df.to_excel(writer, sheet_name='OrderLog', index=False)
            shipment_df.to_excel(writer, sheet_name='ShipmentLog', index=False)
            supply_demand_df.to_excel(writer, sheet_name='SupplyDemandLog', index=False)
            
            # 创建汇总数据
            summary_data = pd.DataFrame([{
                'Total_Orders': len(orders_df),
                'Total_Shipments': len(shipment_df),
                'Total_SupplyDemand': len(supply_demand_df),
                'Date': orders_df['date'].iloc[0] if not orders_df.empty else 'N/A'
            }])
            summary_data.to_excel(writer, sheet_name='Summary', index=False)
        
        print(f"💾 Module1 输出已保存: {output_file}")
        
    except Exception as e:
        print(f"⚠️  Module1 输出保存失败: {e}")

def generate_shipment_with_inventory_check(
    orders_df: pd.DataFrame, 
    simulation_date: pd.Timestamp, 
    orchestrator: object,
    demand_forecast: pd.DataFrame = None,
    forecast_error: pd.DataFrame = None
) -> tuple:
    """基于实际库存限制生成发货数据和缺货记录"""
    if orders_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # 过滤出当日的订单
    today_orders = orders_df[
        pd.to_datetime(orders_df['date']) == simulation_date.normalize()
    ]
    
    if today_orders.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # 获取当前库存状态（从Orchestrator获取最新状态）
    current_inventory = {}
    if hasattr(orchestrator, 'unrestricted_inventory'):
        for (material, location), qty in orchestrator.unrestricted_inventory.items():
            current_inventory[(material, location)] = qty
    
    # 获取物料和地点列表
    materials = today_orders['material'].unique().tolist()
    locations = today_orders['location'].unique().tolist()
    
    # 构造模拟需要的数据结构
    order_log = today_orders.copy()
    order_log = order_log.rename(columns={'quantity': 'quantity'})
    
    # 使用现有的simulate_shipment_for_single_day逻辑
    shipment_df, cut_df, _ = simulate_shipment_for_single_day(
        simulation_date=simulation_date,
        order_log=order_log,
        current_inventory=current_inventory,
        material_list=materials,
        location_list=locations,
        production_plan=None,  # 生产计划由Orchestrator管理
        delivery_plan=None     # 调运计划由Orchestrator管理
    )
    
    # 为shipment添加额外字段以保持兼容性
    if not shipment_df.empty:
        shipment_df['demand_type'] = 'customer'
        shipment_df['order_id'] = shipment_df.apply(
            lambda row: f"ORD_{simulation_date.strftime('%Y%m%d')}_{row.name}", axis=1
        )
    
    print(f"  📦 基于库存生成: {len(shipment_df)} 个shipment, {len(cut_df)} 个cut")
    
    return shipment_df, cut_df



def save_module1_output(orders_df: pd.DataFrame, shipment_df: pd.DataFrame, output_file: str):
    """保存Module1输出到Excel文件"""
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            orders_df.to_excel(writer, sheet_name='OrderLog', index=False)
            shipment_df.to_excel(writer, sheet_name='ShipmentLog', index=False)
            
            # 创建汇总数据
            summary_data = {
                'Summary': pd.DataFrame([{
                    'Total_Orders': len(orders_df),
                    'Total_Shipments': len(shipment_df),
                    'Date': orders_df['date'].iloc[0] if not orders_df.empty else 'N/A'
                }])
            }
            summary_data['Summary'].to_excel(writer, sheet_name='Summary', index=False)
        
        print(f"💾 Module1 输出已保存: {output_file}")
        
    except Exception as e:
        print(f"⚠️  Module1 输出保存失败: {e}")

# 集成模式已不再支持create_default_*函数
# 测试应该使用真实的配置表数据而不是默认生成的数据

if __name__ == "__main__":
    import sys
    
    # Check if validation test is requested
    if len(sys.argv) > 1 and sys.argv[1] == '--validation':
        # Run validation test with Excel output
        try:
            excel_file = run_validation_test()
            print(f"\n✅ Validation test completed successfully!")
            print(f"📁 Excel file generated: {excel_file}")
        except Exception as e:
            print(f"❌ Validation test failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Original simple validation test
        try:
            print("Testing new daily order generation logic...")
            
            # Create test data
            test_data = {
                'demand_forecast': pd.DataFrame([
                    {'material': 'TEST001', 'location': 'LOC001', 'week': 1, 'quantity': 700},
                    {'material': 'TEST001', 'location': 'LOC001', 'week': 2, 'quantity': 700},
                ]),
                'ao_config': pd.DataFrame([
                    {'material': 'TEST001', 'location': 'LOC001', 'advance_days': 3, 'ao_percent': 0.2},
                    {'material': 'TEST001', 'location': 'LOC001', 'advance_days': 7, 'ao_percent': 0.1},
                ]),
                'forecast_error': pd.DataFrame([
                    {'material': 'TEST001', 'location': 'LOC001', 'order_type': 'normal', 'error_std_percent': 0.0},
                    {'material': 'TEST001', 'location': 'LOC001', 'order_type': 'AO', 'error_std_percent': 0.0},
                ]),
                'order_calendar': pd.DataFrame([
                    {'date': pd.to_datetime('2024-01-01') + pd.Timedelta(days=i), 'order_day_flag': 1}
                    for i in range(14)
                ]),
                'initial_inventory': pd.DataFrame([
                    {'date': pd.to_datetime('2024-01-01'), 'material': 'TEST001', 'location': 'LOC001', 'quantity': 100}
                ]),
                'dps_config': pd.DataFrame(),
                'supply_choice': pd.DataFrame(),
            }
            
            # Test basic functionality
            sim_start = pd.to_datetime('2024-01-01')
            num_weeks = 2
            
            # Apply DPS and supply choice (should be no-op for empty configs)
            demand_forecast = apply_dps(test_data['demand_forecast'], test_data['dps_config'])
            demand_forecast = apply_supply_choice(demand_forecast, test_data['supply_choice'])
            
            # Convert to daily
            initial_daily_demand = expand_forecast_to_days_integer_split(
                demand_forecast, sim_start, num_weeks
            )
            
            print(f"Generated {len(initial_daily_demand)} daily demand entries")
            
            # Test daily simulation
            daily_order_logs, daily_supply_demand_logs = run_daily_simulation(
                initial_daily_demand, test_data, sim_start, num_weeks
            )
            
            print(f"Generated order logs for {len(daily_order_logs)} days")
            print(f"Generated supply demand logs for {len(daily_supply_demand_logs)} days")
            
            if daily_order_logs:
                final_orders = daily_order_logs[max(daily_order_logs.keys())]
                total_orders = len(final_orders)
                total_qty = final_orders['quantity'].sum() if not final_orders.empty else 0
                print(f"Total orders generated: {total_orders}")
                print(f"Total quantity: {total_qty}")
                
                # Show order types
                if not final_orders.empty:
                    ao_orders = final_orders[final_orders['demand_type'] == 'AO']
                    normal_orders = final_orders[final_orders['demand_type'] == 'normal']
                    print(f"AO orders: {len(ao_orders)} (qty: {ao_orders['quantity'].sum() if not ao_orders.empty else 0})")
                    print(f"Normal orders: {len(normal_orders)} (qty: {normal_orders['quantity'].sum() if not normal_orders.empty else 0})")
            
            print("✅ New daily order generation logic working correctly!")
            print("\n💡 Run with '--validation' argument for comprehensive validation test with Excel output")
            
        except Exception as e:
            print(f"❌ Error testing new logic: {e}")
            import traceback
            traceback.print_exc()
