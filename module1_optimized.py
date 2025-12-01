import pandas as pd


def expand_forecast_to_days_integer_split(demand_weekly, start_date, num_weeks, simulation_end_date=None):
    """将周度预测拆分为日度预测（向量化优化版本）"""
    if demand_weekly.empty:
        return pd.DataFrame(columns=['date', 'material', 'location', 'week', 'demand_type', 'quantity', 'original_quantity'])
    
    # 向量化计算
    start_date = pd.to_datetime(start_date)
    demand_weekly = demand_weekly.copy()
    
    # 预计算每周的起始日期
    demand_weekly['week_start'] = start_date + pd.to_timedelta((demand_weekly['week'] - 1) * 7, unit='D')
    
    # 计算每日基础数量和余数
    demand_weekly['base_qty'] = (demand_weekly['quantity'] // 7).astype(int)
    demand_weekly['remainder'] = (demand_weekly['quantity'] % 7).astype(int)
    
    # 生成7天的数据
    days = []
    for day_offset in range(7):
        day_df = demand_weekly.copy()
        day_df['date'] = day_df['week_start'] + pd.Timedelta(days=day_offset)
        # 前remainder天多分配1个单位
        day_df['quantity'] = day_df['base_qty'] + (day_offset < day_df['remainder']).astype(int)
        days.append(day_df[['date', 'material', 'location', 'week', 'quantity']])
    
    result_df = pd.concat(days, ignore_index=True)
    
    # 过滤结束日期
    if simulation_end_date is not None:
        result_df = result_df[result_df['date'] <= pd.to_datetime(simulation_end_date)]
    
    result_df['demand_type'] = 'normal'
    result_df['original_quantity'] = result_df['quantity']
    result_df['quantity'] = result_df['quantity'].astype(int)
    
    return _normalize_identifiers(result_df)

def generate_daily_orders(sim_date, original_forecast, current_forecast, ao_config, order_calendar, forecast_error):
    """优化版本：预分组避免重复过滤"""
    
    is_order_day = not order_calendar[order_calendar['date'] == sim_date].empty
    if not is_order_day:
        return pd.DataFrame(), current_forecast
    
    orders = []
    consumed_forecast = current_forecast.copy()
    
    # 预过滤30天窗口的数据
    forecast_window_days = 30
    end_date = sim_date + pd.Timedelta(days=forecast_window_days)
    
    windowed_forecast = original_forecast[
        (original_forecast['date'] >= sim_date) &
        (original_forecast['date'] < end_date)
    ].copy()
    
    # 预分组：按material-location计算平均需求
    if not windowed_forecast.empty:
        ml_avg_demand = windowed_forecast.groupby(['material', 'location'])['quantity'].mean().reset_index()
        ml_avg_demand.columns = ['material', 'location', 'avg_daily_demand']
    else:
        ml_avg_demand = pd.DataFrame(columns=['material', 'location', 'avg_daily_demand'])
    
    # 遍历有需求的物料-地点组合
    for _, row in ml_avg_demand.iterrows():
        material = row['material']
        location = row['location']
        daily_avg_forecast = row['avg_daily_demand']
        
        if daily_avg_forecast <= 0:
            continue
        
        # 获取AO配置
        ml_ao_config = ao_config[
            (ao_config['material'] == material) & 
            (ao_config['location'] == location)
        ]
        
        # 计算订单平均值
        total_ao_percent = ml_ao_config['ao_percent'].sum() if not ml_ao_config.empty else 0
        total_ao_daily_avg = daily_avg_forecast * total_ao_percent
        normal_daily_avg = daily_avg_forecast - total_ao_daily_avg
        
        # 生成AO订单
        for _, ao_row in ml_ao_config.iterrows():
            advance_days = int(ao_row['advance_days'])
            ao_percent = float(ao_row['ao_percent'])
            ao_daily_avg = daily_avg_forecast * ao_percent
            
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
                
                consumed_forecast = consume_forecast_ao_logic(
                    consumed_forecast, material, location, ao_order_date, ao_qty
                )
        
        # 生成普通订单
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
                
                consumed_forecast = consume_forecast_normal_logic(
                    consumed_forecast, material, location, sim_date, normal_qty
                )
    
    orders_df = pd.DataFrame(orders)
    if not orders_df.empty:
        orders_df['quantity'] = orders_df['quantity'].astype(int)
        orders_df = _normalize_identifiers(orders_df)
    
    return orders_df, consumed_forecast

from functools import lru_cache

@lru_cache(maxsize=1024)
def _normalize_location_cached(location_str) -> str:
    """Cached version of location normalization"""
    try:
        return str(int(location_str)).zfill(4)
    except (ValueError, TypeError):
        return str(location_str).zfill(4)

def _normalize_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """优化版本：减少重复标准化"""
    if df.empty:
        return df
    
    identifier_cols = ['material', 'location', 'sending', 'receiving', 'sourcing', 'dps_location']
    
    df = df.copy()
    for col in identifier_cols:
        if col in df.columns:
            df[col] = df[col].astype('string')
            if col in ['location', 'dps_location']:
                # 使用向量化操作
                df[col] = df[col].str.zfill(4)
            elif col == 'material':
                df[col] = df[col].fillna("")
            else:
                df[col] = df[col].fillna("")
    
    return df
