import pandas as pd
import numpy as np
from scipy. stats import truncnorm
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Optional, Tuple, List, Dict
import time
import os
import re

# ----------- 0.  CONSTANTS AND CONFIGURATION -----------

# 性能优化：最大AO提前天数的默认值（从配置中动态获取，此为后备值）
DEFAULT_MAX_ADVANCE_DAYS = 10

# 并行计算的默认开关与并发度（谨慎使用，默认关闭以确保与旧版输出一致）
# - use_parallel_ao_consume：是否启用 AO 消耗的并行分组计算（按物料-地点拆分，进程池）
# - use_parallel_file_load：是否启用历史订单文件的并行读取（线程池，适合I/O）
# - parallel_max_workers：并发工作进程/线程数（None表示自动：CPU核心数）
DEFAULT_USE_PARALLEL_AO_CONSUME = True
DEFAULT_USE_PARALLEL_FILE_LOAD = True
DEFAULT_PARALLEL_MAX_WORKERS: Optional[int] = None
DEFAULT_ERROR_LOG_PATH: Optional[str] = None  # 异常日志输出路径（txt），为空则不写盘
DEFAULT_USE_PARALLEL_NORMAL_CONSUME: Optional[bool] = None  # Normal并行开关（None表示继承AO并行开关）

# 简易异常日志记录工具（中文信息）
def _append_error_log(message: str):
    """
    将异常信息追加写入到txt文件。
    使用说明（中文）：
    - 默认不写盘，需在集成入口里设置 `DEFAULT_ERROR_LOG_PATH` 为某个文件路径。
    - 日志内容为简单文本，方便用户快速定位问题（哪个模块、哪个ML/文件、异常类型与信息）。
    """
    try:
        path = globals().get('DEFAULT_ERROR_LOG_PATH', None)
        if not path:
            return
        # 创建目录（若不存在）
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(str(message).rstrip('\n') + '\n')
    except Exception:
        # 日志写入失败时静默，避免影响主流程
        pass

# ----------- 0. STRING NORMALIZATION FUNCTIONS -----------

def _normalize_location(location_str) -> str:
    """
    规范化地点（location）字符串：
    - 将数值或字符串形式的地点编号统一为4位、左侧补零的字符串（如"7"→"0007"）
    - 对 None/NaN 返回空字符串，避免后续合并键出现非预期类型
    重要：本函数用于保障所有与地点相关的键在数据处理中的一致性，防止因类型或位数不同导致的重复键或匹配失败。
    """
    # Handle None and pandas NA
    if location_str is None or pd.isna(location_str):
        return ""
    try:
        return str(int(location_str)).zfill(4)
    except (ValueError, TypeError):
        return str(location_str).zfill(4)

def _normalize_material(material_str) -> str:
    """
    规范化物料（material）字符串：
    - 将输入统一转为字符串；对 None/NaN 返回空字符串
    用途：确保合并与分组时的键一致，避免类型差异造成的对齐问题。
    """
    # Handle None and pandas NA
    if material_str is None or pd.isna(material_str):
        return ""
    return str(material_str)

def _normalize_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """
    统一规范化标识符列（material/location/sending/receiving/sourcing/dps_location）：
    - 全部转换为字符串类型，缺失值填充为空字符串
    - `location` 与 `dps_location` 使用向量化 `zfill(4)` 保证4位编号
    目的：在整个模块中保持键一致性，减少合并时的重复键与错配。
    特殊处理：采用向量化字符串操作，避免逐行 `apply` 带来的性能损耗。
    """
    if df.empty:
        return df
    
    # Define identifier columns that need string conversion
    identifier_cols = ['material', 'location', 'sending', 'receiving', 'sourcing', 'dps_location']
    
    df = df.copy()
    for col in identifier_cols:
        if col in df.columns:
            # Convert to string and handle NaN values
            df[col] = df[col].astype('string')
            # Apply specific normalization for location (vectorized)
            if col in ['location', 'dps_location']:
                # ✅ 性能优化：使用向量化字符串操作替代apply
                df[col] = df[col].str.zfill(4)
            # Apply specific normalization for material
            elif col == 'material':
                df[col] = df[col].fillna("")
            # For other identifier columns, ensure they are properly formatted strings
            else:
                df[col] = df[col].fillna("")
    
    return df

# ----------- 1. LOAD CONFIG (Enhanced) -----------
def load_config(filename, sheet_mapping=None):
    """
    从 Excel 文件加载各配置页为 DataFrame 字典：
    - 对已存在的 sheet 进行解析并调用 `_normalize_identifiers` 保证键规范
    - 对不存在的 sheet 使用默认空表或 None 填充
    异常处理：读取失败时抛出 RuntimeError，便于上层捕获并提示。
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
                df = xl.parse(sheet_name)
                # 确保标识符字段为字符串格式
                loaded_sheets[key] = _normalize_identifiers(df)
            else:
                loaded_sheets[key] = default
        return loaded_sheets
    except Exception as e:
        raise RuntimeError(f"Failed to load config from {filename}: {e}")

# ----------- 2.  DPS SPLIT -----------
def apply_dps(df, dps_cfg):
    """
    按 DPS 配置进行地点拆分：
    - 输入为周度预测 `df` 与 `dps_cfg`（含 `dps_location` 与 `dps_percent`）
    - 逻辑：先在 MaterialLocationWeek（物料-地点-周）粒度聚合，再按百分比分割为“保留量”和“拆分量”，拆分量的地点改为 `dps_location`
    - 输出重新在 MaterialLocationWeek（物料-地点-周）粒度汇总，数量转为整数
    特殊处理：
    - 缺失 `dps_percent` 视为 0，不拆分
    - 使用向量化运算与合并，避免逐行迭代，提高性能
    - 返回前统一规范化标识符，减少后续键匹配问题
    """
    if dps_cfg.empty:
        return df.copy()
    t0 = time.perf_counter()
    df_g = df.groupby(['material','location','week'], as_index=False)['quantity'].sum()
    cols = ['material','location','dps_location','dps_percent']
    m = df_g.merge(dps_cfg[cols], on=['material','location'], how='left')
    m['dps_percent'] = m['dps_percent'].fillna(0.0)
    m['split_qty'] = np.round(m['quantity'] * m['dps_percent']).astype(int)
    m['remain_qty'] = (m['quantity'] - m['split_qty']).astype(int)
    remain = m[['material','location','week','remain_qty']].rename(columns={'remain_qty':'quantity'})
    split = m[['material','dps_location','week','split_qty']].rename(columns={'dps_location':'location','split_qty':'quantity'})
    out = pd.concat([remain, split], ignore_index=True)
    out = out.groupby(['material','location','week'], as_index=False)['quantity'].sum()
    out['quantity'] = out['quantity'].astype(int)
    print(f"[M1] DPS拆分完成，条目: {len(out)}，耗时: {time.perf_counter()-t0:.3f}s")
    return _normalize_identifiers(out)

# ----------- 3. SUPPLY CHOICE -----------
def apply_supply_choice(df, supply_cfg):
    """
    应用供应选择（Supply Choice）对周度预测进行数量调整：
    - 在 MaterialLocationWeek（物料-地点-周）粒度合并 `adjust_quantity` 并进行向量化加总
    - 缺失调整量按 0 处理
    目的：在周度阶段完成所有数量修正，确保后续日度拆分与订单生成的基线正确。
    """
    if supply_cfg.empty:
        return df.copy()
    t0 = time.perf_counter()
    df_g = df.groupby(['material','location','week'], as_index=False)['quantity'].sum()
    sup_g = supply_cfg.groupby(['material','location','week'], as_index=False)['adjust_quantity'].sum()
    m = df_g.merge(sup_g, on=['material','location','week'], how='left')
    m['quantity'] = (m['quantity'] + m['adjust_quantity'].fillna(0)).astype(int)
    out = m[['material','location','week','quantity']]
    print(f"[M1] SupplyChoice调整完成，条目: {len(out)}，耗时: {time.perf_counter()-t0:.3f}s")
    return _normalize_identifiers(out)

# ----------- 4. SPLIT WEEKLY FORECAST TO DAILY (INTEGER, NO ERROR) -----------
def expand_forecast_to_days_integer_split(demand_weekly, start_date, num_weeks, simulation_end_date=None):
    """
    将周度预测均匀拆分为7天的日度预测（整数分配）：
    - 每周数量按 `base_qty = quantity // 7` 分配，余数 `remainder = quantity % 7` 的前 `remainder` 天各加 1
    - 仅生成至 `simulation_end_date`（如提供）
    - 输出保留 `original_quantity` 便于追溯拆分前的数量
    性能优化：仅进行 7 次复制并向量化计算每日数量，避免对每条记录逐日循环。
    """
    if demand_weekly.empty:
        return pd.DataFrame(columns=['date', 'material', 'location', 'week', 'demand_type', 'quantity', 'original_quantity'])
    
    # ✅ 向量化计算
    start_date = pd.to_datetime(start_date)
    demand_weekly = demand_weekly.copy()
    
    # ✅ 预计算每周的起始日期
    demand_weekly['week_start'] = start_date + pd.to_timedelta((demand_weekly['week'] - 1) * 7, unit='D')
    
    # ✅ 计算每日基础数量和余数
    demand_weekly['base_qty'] = (demand_weekly['quantity'] // 7).astype(int)
    demand_weekly['remainder'] = (demand_weekly['quantity'] % 7).astype(int)
    
    # ✅ 生成7天的数据（只循环7次，而不是N*7次）
    t0 = time.perf_counter()
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
    print(f"[M1] 周度→日度拆分完成，生成天数: {len(result_df)}，耗时: {time.perf_counter()-t0:.3f}s")
    # 确保标识符字段为字符串格式
    return _normalize_identifiers(result_df)

# ----------- 5. DAILY ORDER GENERATION -----------
def generate_daily_orders(sim_date, original_forecast, current_forecast, ao_config, order_calendar, forecast_error):
    """
    生成单日订单（含 AO 与 Normal），并消耗预测：
        - 仅在订单日生成；非订单日直接返回空订单与原预测
        - 在物料-地点（ML）粒度计算 7 天平均需求（默认窗口为7天；若7天内无数据则回退至1天，即当日窗口）
    - AO：按去重后的 AO 配置（仅移除完全重复行，不合并 ML）计算 `ao_daily_avg`，并基于百分比误差生成数量；日期为 `sim_date + advance_days`
    - Normal：同一 ML 汇总 AO 百分比后计算 `normal_daily_avg = avg*(1-ao%)`，误差与当天下单生成
    - 订单汇总后进行“消耗”：
      • AO 优先，固定窗口偏移顺序 [0, -1, -2, 1, 2, 3] 进行贪婪扣减，保证确定性
      • Normal 仅在当日扣减
        采样颗粒度：
        - AO 采样在“每条 AO 配置行”颗粒度（material-location-advance_days）向量化生成数量
        - Normal 采样在“每个 ML 当日”颗粒度（material-location 当日一行）向量化生成数量
        - 通过整列 `np.random.normal(mean_vector, std_vector)` 一次性生成，再裁剪为非负整数
    特殊处理与保障：
    - 预测合并与订单生成均在 ML 粒度，避免周或更细粒度导致的重复键
    - 误差生成采用正态并非截断正态，结果向上取整并裁剪为非负整数
    - 统一规范标识符，确保后续库存与发货环节的键一致
    返回：`orders_df`（当日生成的所有订单）与 `consumed_forecast`（扣减后的预测视图）
    """
    
    # Check if today is an order day
    is_order_day = not order_calendar[order_calendar['date'] == sim_date].empty
    if not is_order_day:
        return pd.DataFrame(), current_forecast
    
    orders = []
    t0 = time.perf_counter()
    # 预测视图按键聚合，保障唯一性
    current_forecast = current_forecast.groupby(['material','location','date'], as_index=False)['quantity'].sum()
    consumed_forecast = current_forecast.copy()
    
    # ✅ 性能优化：预过滤7天窗口的数据（默认窗口改为7天）
    forecast_window_days = 7
    end_date = sim_date + pd.Timedelta(days=forecast_window_days)
    
    windowed_forecast = original_forecast[
        (original_forecast['date'] >= sim_date) &
        (original_forecast['date'] < end_date)
    ].copy()
    
    # ✅ 性能优化：预分组计算平均需求（只计算一次）
    if not windowed_forecast.empty:
        ml_avg_demand = windowed_forecast.groupby(['material','location'], as_index=False)['quantity'].mean()
        ml_avg_demand.columns = ['material', 'location', 'avg_daily_demand']
    else:
        # 如果7天窗口内没有数据，回退至1天窗口（仅当天）
        short_end_date = sim_date + pd.Timedelta(days=1)
        windowed_forecast_short = original_forecast[
            (original_forecast['date'] >= sim_date) &
            (original_forecast['date'] < short_end_date)
        ].copy()
        
        if not windowed_forecast_short. empty:
            ml_avg_demand = windowed_forecast_short.groupby(['material', 'location'], as_index=False)['quantity'].mean()
            ml_avg_demand.columns = ['material', 'location', 'avg_daily_demand']
        else:
            ml_avg_demand = pd.DataFrame(columns=['material', 'location', 'avg_daily_demand'])
    
    if ml_avg_demand.empty:
        return pd.DataFrame(), consumed_forecast
    print(f"[M1] 平均需求计算完成，ML数: {len(ml_avg_demand)}，耗时: {time.perf_counter()-t0:.3f}s")
    
    # ✅ 向量化：在物料-地点粒度生成 AO 与 Normal 订单
    # AO 配置去重：仅移除完全重复的行（不按 ML 折叠）
    ao_cols = ['material','location','advance_days','ao_percent']
    ao_cfg_selected = ao_config[ao_cols].drop_duplicates() if not ao_config.empty else ao_config[ao_cols]
    ao_lines = ml_avg_demand.merge(ao_cfg_selected, on=['material','location'], how='left')
    ao_lines = ao_lines.dropna(subset=['ao_percent'])
    fe = forecast_error.groupby(['material','location','order_type'], as_index=False)['error_std_percent'].max()
    t1 = time.perf_counter()
    if not ao_lines.empty:
        ao_lines['ao_daily_avg'] = ao_lines['avg_daily_demand'] * ao_lines['ao_percent']
        fe_ao = fe[fe['order_type'] == 'AO'][['material','location','error_std_percent']]
        ao_e = ao_lines.merge(fe_ao, on=['material','location'], how='left')
        ao_abs_std = ao_e['ao_daily_avg'] * ao_e['error_std_percent'].fillna(0)
        ao_qty = np.maximum(0, np.round(np.random.normal(ao_e['ao_daily_avg'], ao_abs_std))).astype(int)
        ao_dates = sim_date + pd.to_timedelta(ao_e['advance_days'].astype(int), unit='D')
        ao_orders_df = pd.DataFrame({
            'date': ao_dates,
            'material': ao_e['material'].astype(str),
            'location': ao_e['location'].astype(str),
            'demand_type': 'AO',
            'quantity': ao_qty,
            'simulation_date': sim_date,
            'advance_days': ao_e['advance_days'].astype(int)
        })
    else:
        ao_orders_df = pd.DataFrame(columns=['date','material','location','demand_type','quantity','simulation_date','advance_days'])

    # Normal 订单计算使用去重后的 AO 百分比之和（同一 ML 不同 advance_days 会累加）
    total_ao = ao_cfg_selected.groupby(['material','location'], as_index=False)['ao_percent'].sum()
    normal = ml_avg_demand.merge(total_ao, on=['material','location'], how='left')
    normal['ao_percent'] = normal['ao_percent'].fillna(0).clip(0,1)
    normal['normal_daily_avg'] = normal['avg_daily_demand'] * (1 - normal['ao_percent'])
    normal = normal[normal['normal_daily_avg'] > 0]
    t2 = time.perf_counter()
    if not normal.empty:
        fe_n = fe[fe['order_type'] == 'normal'][['material','location','error_std_percent']]
        n_e = normal.merge(fe_n, on=['material','location'], how='left')
        n_abs_std = n_e['normal_daily_avg'] * n_e['error_std_percent'].fillna(0)
        normal_qty = np.maximum(0, np.round(np.random.normal(n_e['normal_daily_avg'], n_abs_std))).astype(int)
        normal_orders_df = pd.DataFrame({
            'date': pd.Series([sim_date] * len(n_e)),
            'material': n_e['material'].astype(str),
            'location': n_e['location'].astype(str),
            'demand_type': 'normal',
            'quantity': normal_qty,
            'simulation_date': pd.Series([sim_date] * len(n_e)),
            'advance_days': 0
        })
    else:
        normal_orders_df = pd.DataFrame(columns=['date','material','location','demand_type','quantity','simulation_date','advance_days'])

    orders_df = pd.concat([ao_orders_df, normal_orders_df], ignore_index=True)
    if not orders_df.empty:
        orders_df = orders_df.groupby(['date','material','location','demand_type','simulation_date','advance_days'], as_index=False)['quantity'].sum()
        orders_df['quantity'] = orders_df['quantity'].astype(int)
        orders_df = _normalize_identifiers(orders_df)
    print(f"[M1] 订单生成完成 (AO耗时: {time.perf_counter()-t1:.3f}s, Normal耗时: {time.perf_counter()-t2:.3f}s, 总耗时: {time.perf_counter()-t0:.3f}s)，订单数: {len(orders_df)}")

    # 消耗：先 AO 后 normal，贪婪优先顺序
    ao_consume = orders_df[orders_df['demand_type'] == 'AO'].copy() if not orders_df.empty else pd.DataFrame(columns=orders_df.columns)
    t3 = time.perf_counter()
    if not ao_consume.empty:
        ao_consume = ao_consume.sort_values(by=['date','advance_days','quantity','simulation_date'])
        offsets = np.array([0, -1, -2, 1, 2, 3], dtype=int)

        # 并行AO消耗：按(物料,地点)分组以避免共享写入冲突（默认关闭）
        # 配置使用方式：在调用本函数前将 `config_dict['M1_ParallelConfig']` 设置如下键：
        # - use_parallel_ao_consume(bool)：是否启用AO消耗并行，默认False；开启后输出应与串行版本一致（同一ML内仍保持确定性顺序）
        # - parallel_max_workers(int|None)：并行工作进程数；None表示使用os.cpu_count()
        use_parallel_cfg = globals().get('DEFAULT_USE_PARALLEL_AO_CONSUME', False)
        max_workers_cfg = globals().get('DEFAULT_PARALLEL_MAX_WORKERS', None)

        if use_parallel_cfg:
            # 简化中文说明：
            # - 风险提示：如果不同分组之间存在共享日期行，合并补丁时需确保不产生负数；本实现对每个ML独立处理后再安全合并。

            def _consume_ao_for_ml(args: Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, str, str]) -> pd.DataFrame:
                # 子进程纯函数：在单一(物料,地点)下消费AO，返回(date, material, location, new_quantity)补丁
                ml_orders, ml_forecast, offsets_local, mat, loc = args
                if ml_orders.empty or ml_forecast.empty:
                    return pd.DataFrame(columns=['material','location','date','new_quantity'])
                # 局部窗口视图
                ml_forecast = ml_forecast.copy()
                for r in ml_orders.itertuples():
                    if r.quantity <= 0:
                        continue
                    remaining = int(r.quantity)
                    for od in offsets_local:
                        if remaining <= 0:
                            break
                        d = pd.to_datetime(r.date) + pd.to_timedelta(int(od), unit='D')
                        idxs = ml_forecast.index[ml_forecast['date'] == d]
                        if len(idxs) == 0:
                            continue
                        idx = idxs[0]
                        avail = int(ml_forecast.at[idx, 'quantity'])
                        take = min(avail, remaining)
                        ml_forecast.at[idx, 'quantity'] = avail - take
                        remaining -= take
                # 输出补丁
                out = ml_forecast[['date','quantity']].copy()
                out['material'] = mat
                out['location'] = loc
                out = out.rename(columns={'quantity':'new_quantity'})
                return out[['material','location','date','new_quantity']]

            # 组装任务（每个ML一个任务）
            tasks: List[Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, str, str]] = []
            for (mat, loc), grp in ao_consume.groupby(['material','location']):
                ml_mask = (consumed_forecast['material'] == mat) & (consumed_forecast['location'] == loc)
                ml_forecast = consumed_forecast.loc[ml_mask, ['date','quantity']].copy()
                if ml_forecast.empty:
                    continue
                tasks.append((grp[['date','quantity','advance_days','simulation_date']], ml_forecast, offsets, mat, loc))

            patches: List[pd.DataFrame] = []
            with ProcessPoolExecutor(max_workers=max_workers_cfg) as ex:
                futures = [ex.submit(_consume_ao_for_ml, t) for t in tasks]
                for f in as_completed(futures):
                    try:
                        res = f.result()
                        if res is not None and not res.empty:
                            patches.append(res)
                    except Exception:
                        # 并行子任务异常时记录日志并忽略，保持稳健（输出与串行可能不同；建议关闭并行）
                        _append_error_log('[AO并行] 子任务异常：某个物料-地点的AO消耗未应用，建议检查数据或关闭并行')

            if patches:
                patch_df = pd.concat(patches, ignore_index=True)
                # 将补丁安全应用到consumed_forecast（防止负数，优先使用补丁的新值）
                key_cols = ['material','location','date']
                cf = consumed_forecast.merge(
                    patch_df, on=key_cols, how='left'
                )
                cf['quantity'] = np.where(
                    cf['new_quantity'].notna(),
                    np.maximum(0, cf['new_quantity'].astype(int)),
                    cf['quantity'].astype(int)
                )
                consumed_forecast = cf[['material','location','date','quantity']]
            # 并行路径计时打印（与串行一致的可读性）
            print(f"[M1] AO消耗完成，耗时: {time.perf_counter()-t3:.3f}s")
            # 若无补丁或并行失败，保持原consumed_forecast不变
        else:
            # 串行路径（保持旧逻辑，确保输出完全一致）
            for r in ao_consume.itertuples():
                if r.quantity <= 0:
                    continue
                target_dates = pd.to_datetime(r.date) + pd.to_timedelta(offsets, unit='D')
                ml_mask = (consumed_forecast['material'] == r.material) & (consumed_forecast['location'] == r.location)
                window_mask = ml_mask & consumed_forecast['date'].isin(target_dates)
                window = consumed_forecast.loc[window_mask, ['date','quantity']].copy()
                remaining = int(r.quantity)
                for od in offsets:
                    if remaining <= 0:
                        break
                    d = pd.to_datetime(r.date) + pd.to_timedelta(int(od), unit='D')
                    idxs = window.index[window['date'] == d]
                    if len(idxs) == 0:
                        continue
                    idx = idxs[0]
                    avail = int(window.at[idx, 'quantity'])
                    take = min(avail, remaining)
                    window.at[idx, 'quantity'] = avail - take
                    remaining -= take
                for _, w in window.iterrows():
                    consumed_forecast.loc[ml_mask & (consumed_forecast['date'] == w['date']), 'quantity'] = int(w['quantity'])
            print(f"[M1] AO消耗完成，耗时: {time.perf_counter()-t3:.3f}s")

    normal_consume = orders_df[orders_df['demand_type'] == 'normal'].copy() if not orders_df.empty else pd.DataFrame(columns=orders_df.columns)
    t4 = time.perf_counter()
    if not normal_consume.empty:
        # 与 AO 相同的贪婪窗口顺序，确保确定性：[0, -1, -2, 1, 2, 3]
        offsets_n = np.array([0, -1, -2, 1, 2, 3], dtype=int)

        # 使用独立的 Normal 并行配置开关；为空时继承 AO 开关
        inherit_flag = globals().get('DEFAULT_USE_PARALLEL_NORMAL_CONSUME', None)
        use_parallel_normal = (inherit_flag if inherit_flag is not None else globals().get('DEFAULT_USE_PARALLEL_AO_CONSUME', False))
        max_workers_cfg = globals().get('DEFAULT_PARALLEL_MAX_WORKERS', None)

        if use_parallel_normal:
            # 子进程函数：在单一(物料,地点)下消费Normal订单，返回(date, material, location, new_quantity)补丁
            def _consume_normal_for_ml(args: Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, str, str]) -> pd.DataFrame:
                ml_orders, ml_forecast, offsets_local, mat, loc = args
                if ml_orders.empty or ml_forecast.empty:
                    return pd.DataFrame(columns=['material','location','date','new_quantity'])
                ml_forecast = ml_forecast.copy()
                # Normal订单在该窗口内贪婪扣减（与AO一致）
                for r in ml_orders.itertuples():
                    if r.quantity <= 0:
                        continue
                    remaining = int(r.quantity)
                    for od in offsets_local:
                        if remaining <= 0:
                            break
                        d = pd.to_datetime(r.date) + pd.to_timedelta(int(od), unit='D')
                        idxs = ml_forecast.index[ml_forecast['date'] == d]
                        if len(idxs) == 0:
                            continue
                        idx = idxs[0]
                        avail = int(ml_forecast.at[idx, 'quantity'])
                        take = min(avail, remaining)
                        ml_forecast.at[idx, 'quantity'] = avail - take
                        remaining -= take
                out = ml_forecast[['date','quantity']].copy()
                out['material'] = mat
                out['location'] = loc
                out = out.rename(columns={'quantity':'new_quantity'})
                return out[['material','location','date','new_quantity']]

            # 组装任务：每个(物料,地点)为一个任务
            tasks_n: List[Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, str, str]] = []
            for (mat, loc), grp in normal_consume.groupby(['material','location']):
                ml_mask = (consumed_forecast['material'] == mat) & (consumed_forecast['location'] == loc)
                ml_forecast = consumed_forecast.loc[ml_mask, ['date','quantity']].copy()
                if ml_forecast.empty:
                    continue
                tasks_n.append((grp[['date','quantity','simulation_date']], ml_forecast, offsets_n, mat, loc))

            patches_n: List[pd.DataFrame] = []
            with ProcessPoolExecutor(max_workers=max_workers_cfg) as ex:
                futures = [ex.submit(_consume_normal_for_ml, t) for t in tasks_n]
                for f in as_completed(futures):
                    try:
                        res = f.result()
                        if res is not None and not res.empty:
                            patches_n.append(res)
                    except Exception:
                        _append_error_log('[Normal并行] 子任务异常：某个物料-地点的Normal消耗未应用，建议检查数据或关闭并行')

            if patches_n:
                patch_df_n = pd.concat(patches_n, ignore_index=True)
                key_cols = ['material','location','date']
                cf_n = consumed_forecast.merge(patch_df_n, on=key_cols, how='left')
                cf_n['quantity'] = np.where(
                    cf_n['new_quantity'].notna(),
                    np.maximum(0, cf_n['new_quantity'].astype(int)),
                    cf_n['quantity'].astype(int)
                )
                consumed_forecast = cf_n[['material','location','date','quantity']]
            print(f"[M1] Normal消耗完成（并行），耗时: {time.perf_counter()-t4:.3f}s")
        else:
            # 串行路径：与 AO 一致的贪婪窗口顺序
            normal_consume = normal_consume.sort_values(by=['date','quantity','simulation_date'])
            for r in normal_consume.itertuples():
                if r.quantity <= 0:
                    continue
                target_dates = pd.to_datetime(r.date) + pd.to_timedelta(offsets_n, unit='D')
                ml_mask = (consumed_forecast['material'] == r.material) & (consumed_forecast['location'] == r.location)
                window_mask = ml_mask & consumed_forecast['date'].isin(target_dates)
                window = consumed_forecast.loc[window_mask, ['date','quantity']].copy()
                remaining = int(r.quantity)
                for od in offsets_n:
                    if remaining <= 0:
                        break
                    d = pd.to_datetime(r.date) + pd.to_timedelta(int(od), unit='D')
                    idxs = window.index[window['date'] == d]
                    if len(idxs) == 0:
                        continue
                    idx = idxs[0]
                    avail = int(window.at[idx, 'quantity'])
                    take = min(avail, remaining)
                    window.at[idx, 'quantity'] = avail - take
                    remaining -= take
                for _, w in window.iterrows():
                    consumed_forecast.loc[ml_mask & (consumed_forecast['date'] == w['date']), 'quantity'] = int(w['quantity'])
            print(f"[M1] Normal消耗完成（串行），耗时: {time.perf_counter()-t4:.3f}s")
    
    return orders_df, consumed_forecast


def generate_quantity_with_percent_error(mean_qty, material, location, order_type, forecast_error):
    """
    根据百分比误差生成带噪声的订单数量（兼容旧格式）：
    - 优先读取 `forecast_error` 中指定 `order_type` 的 `error_std_percent`，计算绝对标准差
    - 若缺失则回退至旧版 `error_std`（绝对误差）
    - 使用截断正态（下限0）生成值并四舍五入为整数
    注：此函数为逐条调用版本，当前主路径使用向量化正态采样；保留该函数用于兼容与单点生成场景。
    """
    
    # Get error percentage for this material-location-order_type
    mask = (
        (forecast_error['material'] == material) & 
        (forecast_error['location'] == location) & 
        (forecast_error['order_type'] == order_type)
    )
    error_config = forecast_error[mask]
    
    if error_config. empty:
        # Fallback to old error_std format if order_type not found
        mask_old = (
            (forecast_error['material'] == material) & 
            (forecast_error['location'] == location)
        )
        error_config_old = forecast_error[mask_old]
        if not error_config_old.empty and 'error_std' in error_config_old.columns:
            # Use absolute error for backward compatibility
            error_std = float(error_config_old['error_std']. iloc[0])
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
    value = truncnorm. rvs(a, np.inf, loc=mean_qty, scale=abs_std)
    
    return max(0, int(round(value)))


def consume_forecast_ao_logic(forecast_df, material, location, order_date, consume_qty):
    """
    AO 预测消耗（示例/兼容函数）：
    - 固定窗口：订单日当天、前2天、后3天（顺序为 [0, -1, -2, 1, 2, 3]）
    - 贪婪扣减，且不产生负数
    说明：主路径的 AO 消耗在 `generate_daily_orders` 内完成，此函数保留用于兼容或单独调用。
    """
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
    """
    Normal 预测消耗（示例/兼容函数）：
    - 仅订单当日进行扣减，且不产生负数
    说明：主路径的 Normal 消耗在 `generate_daily_orders` 内完成，此函数保留用于兼容或单独调用。
    """
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
def simulate_shipment_for_single_day(
    simulation_date, order_log, current_inventory, material_list, location_list,
    production_plan=None, delivery_plan=None
):
    """
    计算单日的发货（shipment）与缺货（cut）：
    - 输入：订单日志（按日聚合）、当前可用库存（字典形式）、可选的当天生产/调运（当前实现不叠加，避免双计）
    - 逻辑：
      • 当日订单在 ML 粒度聚合为 `qty_ordered`
      • 与库存合并得到 `qty_avail`，发货量为二者最小值，cut 为差值
    - 输出：两个 DataFrame（shipment 与 cut），均规范化标识符
    特别说明：当前库存已由 orchestrator 计算为“期初 + 当日 GR”，此处不再叠加生产/调运，以免重复计入。
    """
    # Pre-filter by date once before loops for better performance
    prod_today = None
    if production_plan is not None and not production_plan.empty:
        prod_today = production_plan[production_plan['available_date'] == simulation_date]
    
    deliv_today = None
    if delivery_plan is not None and not delivery_plan.empty:
        deliv_today = delivery_plan[delivery_plan['actual_delivery_date'] == simulation_date]
    
    # 当前库存已由 orchestrator 计算（期初+当日GR）
    inv_df = pd.DataFrame([
        {'material': k[0], 'location': k[1], 'qty_avail': v}
        for k, v in current_inventory.items()
    ])
    if inv_df.empty:
        inv_df = pd.DataFrame(columns=['material','location','qty_avail'])

    todays_orders = order_log[order_log['date'] == simulation_date] if not order_log.empty else pd.DataFrame(columns=order_log.columns)
    ord_g = todays_orders.groupby(['material','location'], as_index=False)['quantity'].sum().rename(columns={'quantity':'qty_ordered'}) if not todays_orders.empty else pd.DataFrame(columns=['material','location','qty_ordered'])
    merged = ord_g.merge(inv_df, on=['material','location'], how='left')
    merged['qty_avail'] = merged['qty_avail'].fillna(0).astype(int)
    merged['qty_ordered'] = merged['qty_ordered'].fillna(0).astype(int)
    merged['shipped'] = np.minimum(merged['qty_ordered'], merged['qty_avail']).astype(int)
    merged['cut'] = (merged['qty_ordered'] - merged['shipped']).astype(int)
    shipment_df = pd.DataFrame({
        'date': simulation_date,
        'material': merged['material'].astype(str),
        'location': merged['location'].astype(str),
        'quantity': merged['shipped'].astype(int)
    })
    cut_df = pd.DataFrame({
        'date': simulation_date,
        'material': merged['material'].astype(str),
        'location': merged['location'].astype(str),
        'quantity': merged['cut'].astype(int)
    })
    shipment_df = _normalize_identifiers(shipment_df)
    cut_df = _normalize_identifiers(cut_df)
    
    return (
        shipment_df,
        cut_df,
        current_inventory  # 返回可用库存
    )


# ----------- 14. 集成模式支持 -----------

def _load_previous_orders(m1_output_dir: str, current_date: pd.Timestamp, max_advance_days: int = DEFAULT_MAX_ADVANCE_DAYS,
                          use_parallel: Optional[bool] = None,
                          max_workers: Optional[int] = None) -> pd.DataFrame:
    """
    加载近期历史订单（集成模式优化）：
    - 只读取 `current_date - (max_advance_days+1)` 到 `current_date` 之间的 `module1_output_YYYYMMDD.xlsx`
    - 只提取 `OrderLog` 工作表，并统一日期类型；过滤到期在 `current_date` 及之后的订单
    - 目的：控制历史读取范围，避免随着仿真推进导致 I/O 和内存消耗快速增长
    容错：遇到文件/解析错误时跳过该文件，整体返回合并后的结果或空表。
    """
    try:
        if not os.path.isdir(m1_output_dir):
            return pd.DataFrame()
        
        pattern = re.compile(r"module1_output_(\d{8})\.xlsx$")
        
        # 性能优化：计算需要读取的最早日期（当前日期 - max_advance_days - 1）
        # 只读取这个时间窗口内的文件，避免随着仿真推进而读取越来越多的历史文件
        # 加1是为了确保覆盖所有可能还未到期的订单
        earliest_relevant_date = current_date - pd.Timedelta(days=max_advance_days + 1)
        
        # 采集候选文件列表
        candidates = []
        for fname in os.listdir(m1_output_dir):
            m = pattern.match(fname)
            if not m:
                continue
            fdate = pd.to_datetime(m.group(1))
            if fdate.normalize() >= current_date.normalize():
                continue
            if fdate.normalize() < earliest_relevant_date.normalize():
                continue
            candidates.append(os.path.join(m1_output_dir, fname))

        # 并行读取（线程池，适合I/O；默认开启）
        if use_parallel is None:
            use_parallel = globals().get('DEFAULT_USE_PARALLEL_FILE_LOAD', True)
        if max_workers is None:
            max_workers = globals().get('DEFAULT_PARALLEL_MAX_WORKERS', None)

        def _read_orderlog(path: str) -> Optional[pd.DataFrame]:
            try:
                xl = pd.ExcelFile(path)
                if 'OrderLog' not in xl.sheet_names:
                    return None
                df = xl.parse('OrderLog')
                if df is None or df.empty:
                    return None
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                if 'simulation_date' in df.columns:
                    df['simulation_date'] = pd.to_datetime(df['simulation_date'])
                return df
            except Exception:
                _append_error_log(f"[历史文件并行] 读取失败：{path}")
                return None

        rows = []
        t0 = time.perf_counter()
        if use_parallel and candidates:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(_read_orderlog, p) for p in candidates]
                for f in as_completed(futures):
                    df = f.result()
                    if df is not None and not df.empty:
                        rows.append(df)
        else:
            for p in candidates:
                df = _read_orderlog(p)
                if df is not None and not df.empty:
                    rows.append(df)
        print(f"[M1] 历史订单读取完成，文件数: {len(candidates)}，合并条目: {sum(len(r) for r in rows) if rows else 0}，耗时: {time.perf_counter()-t0:.3f}s")
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def run_daily_order_generation(
    config_dict: dict,
    simulation_date: pd.Timestamp,
    output_dir: str,
    orchestrator: object = None
) -> dict:
    """
    集成模式主入口：生成指定日期的订单与发货，并输出供需日志。
    核心流程：
    1) 读取并校验 M1_* 配置（预测、误差、订单日历、AO、DPS、Supply Choice）
    2) 若预测为周度：先执行 DPS → Supply Choice，再按 orchestrator 的全局起始日期做整数日拆分
    3) 生成当日订单（AO+Normal，含确定性消耗逻辑）
    4) 读取历史未到期订单（范围受最大 `advance_days` 限制），与当日订单合并
    5) 调用库存校验生成 shipment/cut；构建供需日志并写入 Excel
    关键约束与特殊处理：
    - 起始日期必须来源于 `orchestrator.start_date`，确保全局一致
    - 历史订单仅加载近窗口，且进行去重（包含 `quantity` 在内的全键去重）
    - 输出前统一规范化标识符，保障后续模块的键一致性
    返回：包含订单、发货、缺货、供需日志与输出文件路径的字典。
    """
    # print(f"🔄 Module1 运行于集成模式 - {simulation_date.strftime('%Y-%m-%d')}")
    
    try:
        # 1) 读取集成配置
        demand_forecast = config_dict. get('M1_DemandForecast', pd.DataFrame())
        forecast_error = config_dict.get('M1_ForecastError', pd.DataFrame())
        order_calendar = config_dict.get('M1_OrderCalendar', pd.DataFrame())
        ao_config = config_dict.get('M1_AOConfig', pd.DataFrame())
        dps_cfg = config_dict.get('M1_DPSConfig', pd.DataFrame())
        supply_choice_cfg = config_dict.get('M1_SupplyChoiceConfig', pd.DataFrame())
        # 2) 基本校验（必须）
        if demand_forecast. empty:
            raise ValueError("缺少必需的配置数据：M1_DemandForecast")
        if order_calendar.empty:
            raise ValueError("缺少必需的配置数据：M1_OrderCalendar")
        if ao_config.empty:
            raise ValueError("缺少必需的配置数据：M1_AOConfig")
        if forecast_error.empty:
            raise ValueError("缺少必需的配置数据：M1_ForecastError")

        # 3) 订单日历规范化
        # print(f"  📅 订单日历验证: {len(order_calendar)}个日期")
        order_calendar['date'] = pd.to_datetime(order_calendar['date'])
        # print(f"  📅 订单日历日期范围: {order_calendar['date'].min()} 到 {order_calendar['date']. max()}")
        # print(f"  📅 当前仿真日期: {simulation_date}")
        is_order_day = not order_calendar[order_calendar['date'] == simulation_date].empty
        # print(f"  📅 当前日期是否为订单日: {'是' if is_order_day else '否'}")

        # —— 将周度预测转换为日度预测（先做 DPS → Supply Choice），且起始日期必须与全局一致 —— 
        # 强制要求 orchestrator 存在且提供 start_date
        if orchestrator is None or not hasattr(orchestrator, 'start_date'):
            raise ValueError("orchestrator. start_date 必须提供，且 Module1 的起始日期必须与全局一致")

        # 读取 M1_* 配置（若未提供则用空表）
        dps_config = config_dict.get('M1_DPSConfig', pd.DataFrame())
        supply_choice = config_dict.get('M1_SupplyChoiceConfig', pd.DataFrame())

        if 'week' in demand_forecast.columns:
            # 先做 DPS → Supply Choice
            demand_forecast = apply_dps(demand_forecast, dps_config if dps_config is not None else pd.DataFrame())
            demand_forecast = apply_supply_choice(demand_forecast, supply_choice if supply_choice is not None else pd.DataFrame())

            # 起始日期严格来自 orchestrator（无任何兜底）
            sim_start = pd.to_datetime(orchestrator. start_date). normalize()

            max_week = int(demand_forecast['week'].max()) if not demand_forecast.empty else 1

            daily_demand_forecast = expand_forecast_to_days_integer_split(
                demand_forecast, sim_start, max_week
            )
            # print(f"  📊 周度预测转换(已过 DPS/SC): {max_week}周 -> {len(daily_demand_forecast)}天")
            # print(f"  📅 预测日期范围: {daily_demand_forecast['date'].min()} 到 {daily_demand_forecast['date'].max()}")
        else:
            # 已经是日度数据：通常不再对日度数据应用 DPS/SC（按你当前定义）
            daily_demand_forecast = demand_forecast. copy()
            # print(f"  📊 使用现有日度预测(跳过 DPS/SC): {len(daily_demand_forecast)}天")

        # 6) 生成当日订单（consumption 保持原逻辑）
        # 注意：标识符字段已在main_integration.py中统一标准化，无需重复处理
        t0 = time.perf_counter()
        today_orders_df, consumed_forecast = generate_daily_orders(
            simulation_date, daily_demand_forecast, daily_demand_forecast, 
            ao_config, order_calendar, forecast_error
        )
        print(f"[M1] 当日订单生成完成，订单数: {len(today_orders_df)}，耗时: {time.perf_counter()-t0:.3f}s")

        # 7) 合并历史未到期订单 → 当日版本订单视图
        # 性能优化：从ao_config中获取最大advance_days，用于优化历史订单加载范围
        if not ao_config.empty and 'advance_days' in ao_config. columns:
            max_val = ao_config['advance_days'].max(skipna=True)
            max_advance_days = int(max_val) if pd.notna(max_val) else DEFAULT_MAX_ADVANCE_DAYS
        else:
            max_advance_days = DEFAULT_MAX_ADVANCE_DAYS
        
        t1 = time.perf_counter()
        previous_orders_all = _load_previous_orders(output_dir, simulation_date, max_advance_days)
        print(f"[M1] 历史订单合并前过滤完成，耗时: {time.perf_counter()-t1:.3f}s")
        
        # 性能优化：在去重之前先过滤未来订单，减少处理的数据量
        if not previous_orders_all.empty and 'date' in previous_orders_all.columns:
            previous_orders_all['date'] = pd.to_datetime(previous_orders_all['date'])
            previous_orders_all = previous_orders_all[previous_orders_all['date'] >= simulation_date]. copy()
        
        if not previous_orders_all.empty:
            dedup_keys = [
                c for c in ['date','material','location','demand_type','simulation_date','advance_days','quantity']
                if c in previous_orders_all.columns
            ]
            if dedup_keys:
                previous_orders_all = previous_orders_all.drop_duplicates(subset=dedup_keys)

        previous_orders_future = previous_orders_all.copy() if not previous_orders_all. empty else pd.DataFrame()

        orders_df = (
            pd.concat([previous_orders_future, today_orders_df], ignore_index=True)
            if (today_orders_df is not None and not today_orders_df.empty)
            else previous_orders_future. copy()
        )

        if not orders_df.empty:
            if 'quantity' in orders_df.columns:
                orders_df['quantity'] = orders_df['quantity'].astype(int)
            if 'simulation_date' not in orders_df.columns:
                orders_df['simulation_date'] = orders_df['date']
            # 确保标识符字段为字符串格式
            orders_df = _normalize_identifiers(orders_df)

        # 8) 发货（依赖 orchestrator 库存）
        if orchestrator is not None:
            t2 = time.perf_counter()
            shipment_df, cut_df = generate_shipment_with_inventory_check(
                orders_df, simulation_date, orchestrator,
                daily_demand_forecast, forecast_error
            )
            print(f"[M1] 发货与缺货计算完成，shipment: {len(shipment_df)}，cut: {len(cut_df)}，耗时: {time.perf_counter()-t2:.3f}s")
        else:
            print("  ⚠️  警告：没有Orchestrator，无法生成基于库存的shipment")
            shipment_df, cut_df = pd.DataFrame(), pd.DataFrame()

        # 9) 供需日志（集成规范）
        t3 = time.perf_counter()
        supply_demand_df = generate_supply_demand_log_for_integration(
            daily_demand_forecast, consumed_forecast, simulation_date
        )
        print(f"[M1] 供需日志生成完成，条目: {len(supply_demand_df)}，耗时: {time.perf_counter()-t3:.3f}s")

        # 10) 落盘
        output_file = f"{output_dir}/module1_output_{simulation_date.strftime('%Y%m%d')}.xlsx"
        # 自动设置异常日志保存路径到与Module1输出相同的目录，无需用户额外配置
        try:
            globals()['DEFAULT_ERROR_LOG_PATH'] = os.path.join(
                output_dir,
                f"module1_parallel_errors_{simulation_date.strftime('%Y%m%d')}.txt"
            )
        except Exception:
            # 如果设置失败，忽略，不影响主流程
            pass
        save_module1_output_with_supply_demand(orders_df, shipment_df, supply_demand_df, output_file, cut_df)

        # print(f"✅ Module1 完成 - 生成 {len(orders_df)} 个订单, {len(shipment_df)} 个发货, {len(cut_df)} 个cut")
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
    consumed_forecast: pd. DataFrame, 
    simulation_date: pd.Timestamp
) -> pd.DataFrame:
    """
    生成集成模式的供需日志（SupplyDemandLog）：
    - 仅输出仿真日期之后、未来 90 天内的需求（demand_element="forecast"）
    - 使用 `consumed_forecast` 作为来源，反映订单消耗后的最新需求视图
    - 统一规范标识符，避免后续模块的键不一致
    返回：包含 `date/material/location/quantity/demand_element` 的 DataFrame。
    """
    # 处理空DataFrame
    if consumed_forecast.empty or 'date' not in consumed_forecast.columns:
        return pd.DataFrame(columns=['date', 'material', 'location', 'quantity', 'demand_element'])
    
    # 性能优化：只生成未来90天的需求数据，减少数据量
    # 90天（约3个月），足够满足业务需求
    future_cutoff_date = simulation_date + pd.Timedelta(days=90)
    
    # 生成未来需求数据（仿真日期之后的90天内）
    future_demand = consumed_forecast[
        (pd.to_datetime(consumed_forecast['date']) > simulation_date) &
        (pd.to_datetime(consumed_forecast['date']) <= future_cutoff_date)
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
    
    # 确保标识符字段为字符串格式
    return _normalize_identifiers(supply_demand_log)

def save_module1_output_with_supply_demand(
    orders_df: pd.DataFrame, 
    shipment_df: pd. DataFrame, 
    supply_demand_df: pd.DataFrame,
    output_file: str,
    cut_df: pd. DataFrame = None
):
    """
    将 Module1 的主输出写入 Excel：
    - 工作表：OrderLog、ShipmentLog、CutLog（始终写出）、SupplyDemandLog、Summary
    - 使用 `_ensure_cols` 保证列完整，调用 `_normalize_identifiers` 保持键规范
    容错：整体写入异常时仅打印警告，防止中断主流程。
    """
    # 🆕 统一列头保障函数
    def _ensure_cols(df: pd.DataFrame, cols: list) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=cols)
        # 缺列补列
        for c in cols:
            if c not in df.columns:
                df[c] = pd.Series(dtype='object')
        return df[cols]
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            orders_df = _ensure_cols(orders_df, ['date','material','location','demand_type','quantity','simulation_date','advance_days'])
            shipment_df = _ensure_cols(shipment_df, ['date','material','location','quantity','demand_type','order_id'])
            cut_df = _ensure_cols(cut_df, ['date','material','location','quantity'])
            supply_demand_df = _ensure_cols(supply_demand_df, ['date','material','location','quantity','demand_element'])
            _normalize_identifiers(orders_df).to_excel(writer, sheet_name='OrderLog', index=False)
            _normalize_identifiers(shipment_df).to_excel(writer, sheet_name='ShipmentLog', index=False)
            _normalize_identifiers(cut_df).to_excel(writer, sheet_name='CutLog', index=False)  # 始终写
            _normalize_identifiers(supply_demand_df).to_excel(writer, sheet_name='SupplyDemandLog', index=False)
            summary_data = pd.DataFrame([{
                'Total_Orders': len(orders_df),
                'Total_Shipments': len(shipment_df),
                'Total_Cuts': len(cut_df),
                'Total_SupplyDemand': len(supply_demand_df),
                'Date': orders_df['date'].iloc[0] if not orders_df.empty else 'N/A'
            }])
            summary_data.to_excel(writer, sheet_name='Summary', index=False)
    except Exception as e:
        print(f"⚠️  Module1 输出保存失败: {e}")

def _build_available_inventory_from_orchestrator(orchestrator, simulation_date: pd.Timestamp) -> dict:
    """
    构建当日可用库存（ML 字典）：
    - 可用库存 = 期初库存 + 当日生产入库（location）+ 当日调运入库（receiving）
    - 三视图统一到 ML 粒度并汇总；显式数值转换避免 `fillna` 的 downcasting 预警
    - 返回字典 `{(material, location): qty}` 供发货环节使用
    特别说明：地点列名称在不同视图中不同（production: location；delivery: receiving），此处已统一处理。
    """
    date_str = simulation_date.strftime('%Y-%m-%d')

    # 期初
    beg_df = orchestrator.get_beginning_inventory_view(date_str)
    # 当日 GR
    prod_df = orchestrator.get_production_gr_view(date_str)
    delv_df = orchestrator.get_delivery_gr_view(date_str)

    # 统一并聚合为 ML 粒度
    def _to_ml(df, loc_col):
        if df is None or df.empty:
            return pd.DataFrame(columns=['material','location','quantity'])
        o = df[['material', loc_col, 'quantity']].copy()
        o['material'] = o['material'].astype(str)
        o['location'] = o[loc_col].astype(str).str.zfill(4)
        return o.groupby(['material','location'], as_index=False)['quantity'].sum()

    beg = _to_ml(beg_df, 'location')
    prod = _to_ml(prod_df, 'location')
    delv = _to_ml(delv_df, 'receiving')
    inv_df = beg.merge(prod, on=['material','location'], how='outer', suffixes=('_beg','_prod'))
    inv_df = inv_df.merge(delv, on=['material','location'], how='outer')
    # 显式数值转换以避免 fillna 的未来 downcasting 变更
    for col in ['quantity_beg','quantity_prod','quantity']:
        if col in inv_df.columns:
            inv_df[col] = pd.to_numeric(inv_df[col], errors='coerce')
    inv_df[['quantity_beg','quantity_prod','quantity']] = inv_df[['quantity_beg','quantity_prod','quantity']].fillna(0)
    # 计算总可用库存
    qty = inv_df.get('quantity_beg', 0) + inv_df.get('quantity_prod', 0) + inv_df.get('quantity', 0)
    inv_df['qty'] = pd.to_numeric(qty, errors='coerce').fillna(0).astype(int)
    inv_df = inv_df[['material','location','qty']]
    return {(r.material, r.location): int(r.qty) for r in inv_df.itertuples(index=False)}

def generate_shipment_with_inventory_check(
    orders_df: pd. DataFrame, 
    simulation_date: pd.Timestamp, 
    orchestrator: object,
    demand_forecast: pd.DataFrame = None,
    forecast_error: pd.DataFrame = None
) -> tuple:
    """
    基于真实可用库存（期初+当日 GR）生成当日发货与缺货：
    - 过滤当日到期订单；规范化物料/地点以匹配库存键
    - 通过 `_build_available_inventory_from_orchestrator` 获取当日 ML 库存
    - 调用 `simulate_shipment_for_single_day` 计算 shipment/cut，并为 shipment 生成 `order_id`
    - 返回两个 DataFrame：`shipment_df`（新增 `demand_type='customer'` 与 `order_id`）与 `cut_df`
    说明：不叠加 production_plan/delivery_plan，避免与 orchestrator 的 GR 重复计入。
    """
    if orders_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # 当日到期订单
    today_orders = orders_df[
        pd.to_datetime(orders_df['date']) == simulation_date. normalize()
    ].copy()
    if today_orders.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # 确保与库存键一致的物料数据类型
    today_orders['material'] = today_orders['material']. astype(str)
    
    # ✅ 可用库存 = 期初 + 当日 Production GR + 当日 Delivery GR
    current_inventory = _build_available_inventory_from_orchestrator(orchestrator, simulation_date)
    
    # Normalize material and location lists to match inventory keys
    materials = [_normalize_material(m) for m in today_orders['material'].unique().tolist()]
    locations = [_normalize_location(l) for l in today_orders['location'].unique().tolist()]
    
    # Normalize order_log material and location to match inventory keys
    order_log = today_orders.copy()
    order_log['material'] = order_log['material'].apply(_normalize_material)
    order_log['location'] = order_log['location'].apply(_normalize_location)
    
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
        # Vectorized order_id generation instead of apply
        date_str = simulation_date.strftime('%Y%m%d')
        shipment_df['order_id'] = 'ORD_' + date_str + '_' + shipment_df.index.astype(str)
    
    # print(f"  📦 基于[期初+当日GR]生成: {len(shipment_df)} 个shipment, {len(cut_df)} 个cut")
    return shipment_df, cut_df