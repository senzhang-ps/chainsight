import pandas as pd
import numpy as np
from scipy.stats import truncnorm
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
        # 1) 读取集成配置
        demand_forecast = config_dict.get('M1_DemandForecast', pd.DataFrame())
        forecast_error = config_dict.get('M1_ForecastError', pd.DataFrame())
        order_calendar = config_dict.get('M1_OrderCalendar', pd.DataFrame())
        ao_config = config_dict.get('M1_AOConfig', pd.DataFrame())
        dps_cfg = config_dict.get('M1_DPSConfig', pd.DataFrame())
        supply_choice_cfg = config_dict.get('M1_SupplyChoiceConfig', pd.DataFrame())
        # 2) 基本校验（必须）
        if demand_forecast.empty:
            raise ValueError("缺少必需的配置数据：M1_DemandForecast")
        if order_calendar.empty:
            raise ValueError("缺少必需的配置数据：M1_OrderCalendar")
        if ao_config.empty:
            raise ValueError("缺少必需的配置数据：M1_AOConfig")
        if forecast_error.empty:
            raise ValueError("缺少必需的配置数据：M1_ForecastError")

        # 3) 订单日历规范化
        print(f"  📅 订单日历验证: {len(order_calendar)}个日期")
        order_calendar['date'] = pd.to_datetime(order_calendar['date'])
        print(f"  📅 订单日历日期范围: {order_calendar['date'].min()} 到 {order_calendar['date'].max()}")
        print(f"  📅 当前仿真日期: {simulation_date}")
        is_order_day = not order_calendar[order_calendar['date'] == simulation_date].empty
        print(f"  📅 当前日期是否为订单日: {'是' if is_order_day else '否'}")

        # —— 将周度预测转换为日度预测（先做 DPS → Supply Choice），且起始日期必须与全局一致 —— 
        # 强制要求 orchestrator 存在且提供 start_date
        if orchestrator is None or not hasattr(orchestrator, 'start_date'):
            raise ValueError("orchestrator.start_date 必须提供，且 Module1 的起始日期必须与全局一致")

        # 读取 M1_* 配置（若未提供则用空表）
        dps_config = config_dict.get('M1_DPSConfig', pd.DataFrame())
        supply_choice = config_dict.get('M1_SupplyChoiceConfig', pd.DataFrame())

        if 'week' in demand_forecast.columns:
            # 先做 DPS → Supply Choice
            demand_forecast = apply_dps(demand_forecast, dps_config if dps_config is not None else pd.DataFrame())
            demand_forecast = apply_supply_choice(demand_forecast, supply_choice if supply_choice is not None else pd.DataFrame())

            # 起始日期严格来自 orchestrator（无任何兜底）
            sim_start = pd.to_datetime(orchestrator.start_date).normalize()

            max_week = int(demand_forecast['week'].max()) if not demand_forecast.empty else 1

            daily_demand_forecast = expand_forecast_to_days_integer_split(
                demand_forecast, sim_start, max_week
            )
            print(f"  📊 周度预测转换(已过 DPS/SC): {max_week}周 -> {len(daily_demand_forecast)}天")
            print(f"  📅 预测日期范围: {daily_demand_forecast['date'].min()} 到 {daily_demand_forecast['date'].max()}")
        else:
            # 已经是日度数据：通常不再对日度数据应用 DPS/SC（按你当前定义）
            daily_demand_forecast = demand_forecast.copy()
            print(f"  📊 使用现有日度预测(跳过 DPS/SC): {len(daily_demand_forecast)}天")

        # 6) 生成当日订单（consumption 保持原逻辑）
        today_orders_df, consumed_forecast = generate_daily_orders(
            simulation_date, daily_demand_forecast, daily_demand_forecast, 
            ao_config, order_calendar, forecast_error
        )

        # 7) 合并历史未到期订单 → 当日版本订单视图
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
                        continue
                    fpath = os.path.join(m1_output_dir, fname)
                    try:
                        xl = pd.ExcelFile(fpath)
                        if 'OrderLog' not in xl.sheet_names:
                            continue
                        df = xl.parse('OrderLog')
                        if df is None or df.empty:
                            continue
                        if 'date' in df.columns:
                            df['date'] = pd.to_datetime(df['date'])
                        if 'simulation_date' in df.columns:
                            df['simulation_date'] = pd.to_datetime(df['simulation_date'])
                        rows.append(df)
                    except Exception:
                        continue
                return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
            except Exception:
                return pd.DataFrame()

        previous_orders_all = _load_previous_orders(output_dir, simulation_date)
        if not previous_orders_all.empty:
            dedup_keys = [
                c for c in ['date','material','location','demand_type','simulation_date','advance_days','quantity']
                if c in previous_orders_all.columns
            ]
            if dedup_keys:
                previous_orders_all = previous_orders_all.drop_duplicates(subset=dedup_keys)

        previous_orders_future = (
            previous_orders_all[previous_orders_all['date'] >= simulation_date].copy()
            if (not previous_orders_all.empty and 'date' in previous_orders_all.columns)
            else pd.DataFrame()
        )

        orders_df = (
            pd.concat([previous_orders_future, today_orders_df], ignore_index=True)
            if (today_orders_df is not None and not today_orders_df.empty)
            else previous_orders_future.copy()
        )

        if not orders_df.empty:
            if 'quantity' in orders_df.columns:
                orders_df['quantity'] = orders_df['quantity'].astype(int)
            if 'simulation_date' not in orders_df.columns:
                orders_df['simulation_date'] = orders_df['date']

        # 8) 发货（依赖 orchestrator 库存）
        if orchestrator is not None:
            shipment_df, cut_df = generate_shipment_with_inventory_check(
                orders_df, simulation_date, orchestrator,
                daily_demand_forecast, forecast_error
            )
        else:
            print("  ⚠️  警告：没有Orchestrator，无法生成基于库存的shipment")
            shipment_df, cut_df = pd.DataFrame(), pd.DataFrame()

        # 9) 供需日志（集成规范）
        supply_demand_df = generate_supply_demand_log_for_integration(
            daily_demand_forecast, consumed_forecast, simulation_date
        )

        # 10) 落盘
        output_file = f"{output_dir}/module1_output_{simulation_date.strftime('%Y%m%d')}.xlsx"
        save_module1_output_with_supply_demand(orders_df, shipment_df, supply_demand_df, output_file)

        print(f"✅ Module1 完成 - 生成 {len(orders_df)} 个订单, {len(shipment_df)} 个发货, {len(cut_df)} 个cut")
        return {
            'orders_df': orders_df,
            'shipment_df': shipment_df,
            'cut_df': cut_df,
            'supply_demand_df': supply_demand_df,
            'output_file': output_file
        }

    except Exception as e:
        print(f"❌ Module1 集成模式失败: {e}")
        import traceback; traceback.print_exc()
        return {
            'orders_df': pd.DataFrame(),
            'shipment_df': pd.DataFrame(),
            'cut_df': pd.DataFrame(),
            'supply_demand_df': pd.DataFrame(),
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

def _build_available_inventory_from_orchestrator(orchestrator, simulation_date: pd.Timestamp) -> dict:
    """
    可用库存 = 期初库存 + 当日 Production GR + 当日 Delivery GR
    - 期初库存：orchestrator.get_beginning_inventory_view(date)
    - 生产入库：orchestrator.get_production_gr_view(date)   (location 列)
    - 交付入库：orchestrator.get_delivery_gr_view(date)     (receiving 列)
    """
    date_str = simulation_date.strftime('%Y-%m-%d')

    # 期初
    beg_df = orchestrator.get_beginning_inventory_view(date_str)
    # 当日 GR
    prod_df = orchestrator.get_production_gr_view(date_str)
    delv_df = orchestrator.get_delivery_gr_view(date_str)

    inv = {}

    # 期初库存
    if not beg_df.empty:
        for _, r in beg_df.iterrows():
            key = (str(r['material']), str(r['location']))
            inv[key] = inv.get(key, 0) + int(r['quantity'])

    # 生产 GR（location 为入库地点）
    if not prod_df.empty:
        for _, r in prod_df.iterrows():
            key = (str(r['material']), str(r['location']))
            inv[key] = inv.get(key, 0) + int(r['quantity'])

    # 交付 GR（receiving 为入库地点）
    if not delv_df.empty:
        for _, r in delv_df.iterrows():
            key = (str(r['material']), str(r['receiving']))
            inv[key] = inv.get(key, 0) + int(r['quantity'])

    return inv

def generate_shipment_with_inventory_check(
    orders_df: pd.DataFrame, 
    simulation_date: pd.Timestamp, 
    orchestrator: object,
    demand_forecast: pd.DataFrame = None,
    forecast_error: pd.DataFrame = None
) -> tuple:
    """基于实际库存限制生成发货数据和缺货记录（库存=期初+当日GR）"""
    if orders_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # 当日到期订单
    today_orders = orders_df[
        pd.to_datetime(orders_df['date']) == simulation_date.normalize()
    ]
    if today_orders.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # ✅ 可用库存 = 期初 + 当日 Production GR + 当日 Delivery GR
    current_inventory = _build_available_inventory_from_orchestrator(orchestrator, simulation_date)
    
    materials = today_orders['material'].unique().tolist()
    locations = today_orders['location'].unique().tolist()
    order_log = today_orders.copy()
    
    # 注意：此处不再叠加 production_plan / delivery_plan，避免双计
    shipment_df, cut_df, _ = simulate_shipment_for_single_day(
        simulation_date=simulation_date,
        order_log=order_log,
        current_inventory=current_inventory,
        material_list=materials,
        location_list=locations,
        production_plan=None,
        delivery_plan=None
    )
    
    if not shipment_df.empty:
        shipment_df['demand_type'] = 'customer'
        shipment_df['order_id'] = shipment_df.apply(
            lambda row: f"ORD_{simulation_date.strftime('%Y%m%d')}_{row.name}", axis=1
        )
    
    print(f"  📦 基于[期初+当日GR]生成: {len(shipment_df)} 个shipment, {len(cut_df)} 个cut")
    return shipment_df, cut_df

