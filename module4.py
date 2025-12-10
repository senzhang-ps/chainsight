"""
module4.py

整体目的：
- 模块4负责工业级 APS 生产计划的日度执行逻辑：读取净需求，依据产线配置、产能与换产矩阵进行无约束计划与集中产能分配，跟踪跨天换产连续性，并输出生产与校验日志。

功能点：
- 配置加载与校验：读取 M4 相关配置（LineCfg/Capacity/Changeover）并进行必要的验证与类型标准化。
- 净需求读取：按日从 Module3 输出读取净需求数据，筛选层级与日期，构建当日无约束计划。
- 产能分配与换产：支持换产矩阵与定义，进行最优序列与集中分配，记录超额与未满足。
- 跨天连续性：保存/恢复产线状态与已分配产能，支持未完成换产在次日延续。
- 输出与汇总：每日输出生产计划、超额、校验与换产日志，支持合并输出。

使用方法：
- 日度模式：调用 `run_daily_production_planning(...)` 处理单日并生成每日输出。
- 集成模式：由 `main_integration.py` 直接调用内部函数，如 `build_unconstrained_plan_for_single_day` 与 `centralized_capacity_allocation_with_changeover` 等。
"""
import pandas as pd
import numpy as np
from datetime import timedelta
import argparse
import os
from typing import Optional


def get_or_init_simulation_start(output_dir: str, provided_start: Optional[pd.Timestamp]) -> pd.Timestamp:
    """读取或初始化仿真开始日期（持久化）

    目的：
    - 首次运行时将提供的开始日期写入 `simulation_start.txt`；后续运行读取持久化值，保证审查日计算一致。

    Args:
        output_dir: 模块4输出/状态目录。
        provided_start: 用户提供的开始日期（首次必需，后续可选）。

    Returns:
        pd.Timestamp: 持久化的仿真开始日期。

    逻辑：
        - 若状态文件存在则读取；否则要求提供开始日期并写入文件；返回最终开始日期。
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
    """保存产线状态（最后物料与剩余换产时间）用于跨天连续性

    目的：
    - 在每日结束后持久化产线状态，支持次日延续未完成换产。

    Args:
        output_dir: 输出/状态目录。
        simulation_date: 当前仿真日期。
        line_states: 字典：`line -> {last_material, remaining_changeover, ...}`。

    输出/逻辑：
        - 写入 `line_states_YYYYMMDD.json`，不更改业务逻辑。
    """
    os.makedirs(output_dir, exist_ok=True)
    state_file = os.path.join(output_dir, f"line_states_{simulation_date.strftime('%Y%m%d')}.json")
    
    import json
    with open(state_file, "w") as f:
        json.dump(line_states, f, indent=2)


def save_allocated_capacity(output_dir: str, simulation_date: pd.Timestamp, allocated_capacity: dict):
    """保存已分配产能（小时）用于跨日跟踪

    目的：
    - 将按生产计划日期的已分配产能持久化，避免多个仿真日对同一生产日重复分配。

    Args:
        output_dir: 输出/状态目录。
        simulation_date: 当前仿真日期。
        allocated_capacity: 字典：`location|line|production_date -> hours`。

    输出/逻辑：
        - 写入 `allocated_capacity_YYYYMMDD.json`，供后续仿真日期读取校验。
    """
    os.makedirs(output_dir, exist_ok=True)
    capacity_file = os.path.join(output_dir, f"allocated_capacity_{simulation_date.strftime('%Y%m%d')}.json")
    
    import json
    with open(capacity_file, "w") as f:
        json.dump(allocated_capacity, f, indent=2)


def load_allocated_capacity(output_dir: str, simulation_date: pd.Timestamp) -> dict:
    """加载当前仿真日之前持久化的已分配产能

    目的：
    - 读取当前日期对应的产能文件，以在同一天内避免重复分配。

    Args:
        output_dir: 输出/状态目录。
        simulation_date: 当前仿真日期。

    Returns:
        dict: 已分配产能字典；不存在则返回空字典。
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
    """汇总所有历史仿真日的已分配产能

    目的：
    - 遍历 `output_dir` 下历史 `allocated_capacity_*.json`，合并为统一字典供校验使用。

    Args:
        output_dir: 输出/状态目录。
        simulation_date: 当前仿真日期。

    Returns:
        dict: 之前所有仿真日的合并产能分配（小时）。
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
    """从生产计划提取已分配产能信息（小时）用于持久化

    目的：
    - 依据计划行汇总每个 `location/line/production_plan_date` 的生产与换产耗时。

    Args:
        plan_df: 生产计划 DataFrame。
        rate_map: 产率映射 `(material, line) -> rate`。
        changeover_def: 换产定义 `(changeover_id, line) -> time`（可选）。

    Returns:
        dict: 键为 `location|line|date`，值为小时数（float）。

    逻辑：
        - 分组累加生产时间与换产时间→转为 float 便于 JSON 序列化→返回字典。
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
    """校验产能分配是否尊重历史已分配产能

    目的：
    - 对每个生产计划日检查当前分配与历史分配之和，记录校验信息用于后续分析。

    Args:
        plan_log: 当前生产计划 DataFrame。
        previously_allocated_capacity: 历史已分配产能字典（小时）。
        simulation_date: 当前仿真日期。
        rate_map: 产率映射。
        changeover_def: 换产定义（可选）。

    Returns:
        list: 校验记录列表（非错误，仅信息）。

    逻辑：
        - 按 `location/line/production_date` 分组→计算当前耗时→与历史耗时合并→生成校验记录。
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
    """加载前一日的产线状态（支持跨天换产连续性）

    目的：
    - 读取 `line_states_YYYYMMDD.json`，恢复未完成换产或最后物料状态。

    Args:
        output_dir: 输出/状态目录。
        simulation_date: 当前仿真日期。

    Returns:
        dict: 前一日产线状态；不存在或异常时返回空字典。
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


def analyze_end_of_day_changeover_state(plan_df: pd.DataFrame, cap_df: pd.DataFrame, 
                                        co_def: dict, simulation_date: pd.Timestamp,
                                        rate_map: dict) -> dict:
    """分析日末是否存在未完成换产（基于产能分配重构）

    目的：
    - 通过重建分配逻辑，检测即便未产生生产记录也可能已启动但未完成的换产，并推断剩余时间。

    Args:
        plan_df: 当日生产计划。
        cap_df: 产能数据。
        co_def: 换产定义字典。
        simulation_date: 当前仿真日期。
        rate_map: 产率映射，用于计算生产时间。

    Returns:
        dict: 产线换产状态 `{line: changeover_info 或 None}`。

    逻辑：
        - 逐线逐生产日重构耗时→比对剩余产能→若与典型换产时长接近则推断未完成换产并记录。
    """
    changeover_states = {}
    
    if plan_df.empty:
        return changeover_states
    
    for line, group in plan_df.groupby('line'):
        # Filter for records from the current simulation date
        sim_date_group = group[group['simulation_date'] == simulation_date]
        if sim_date_group.empty:
            continue
        
        # Group by production_plan_date to analyze each production day separately
        for prod_date, prod_group in sim_date_group.groupby('production_plan_date'):
            # Get capacity for the production date (not simulation date)
            day_capacity = cap_df[cap_df['date'] == prod_date]
            if day_capacity.empty:
                continue
                
            # Get total capacity for this line on this production day
            line_cap = day_capacity[day_capacity['line'] == line]['capacity'].sum()
            if line_cap <= 0:
                continue
            
            # Calculate total allocated time from ProductionPlan
            total_allocated = 0
            last_material = None
            
            group_sorted = prod_group.sort_values('production_plan_date')
            for _, row in group_sorted.iterrows():
                material = row['material']
                quantity = row['con_planned_qty']
                changeover_id = row.get('changeover_id')
                
                # Add changeover time if exists
                if changeover_id and pd.notna(changeover_id):
                    changeover_time = co_def.get((changeover_id, line), 0)
                    total_allocated += changeover_time
                
                # Add production time
                rate = rate_map.get((material, line), 1)
                production_time = quantity / rate if rate and quantity > 0 else 0
                total_allocated += production_time
                
                last_material = material
            
            # Check if there's remaining capacity that could be used for a changeover
            remaining_capacity = line_cap - total_allocated
            
            if remaining_capacity > 0.1 and last_material:  # At least 0.1 hour remaining
                # There might be an attempted changeover to the next material
                # We need to infer what the next material would be based on the scheduling sequence
                
                # For now, we'll use a heuristic: if there's exactly ~1 hour remaining
                # and this matches typical changeover times, assume there's an incomplete changeover
                typical_changeover_time = 1.0  # Most changeovers are 1 hour
                
                if abs(remaining_capacity - typical_changeover_time) < 0.1:
                    # Likely that a changeover started but didn't complete
                    # print(f"🔄 Line {line}: Detected likely incomplete changeover on {prod_date.date()} - {remaining_capacity:.2f} hours remaining capacity matches typical changeover time")
                    
                    # Create a generic incomplete changeover record
                    incomplete_changeover = {
                        'changeover_id': 'INFERRED_INCOMPLETE',
                        'from_material': last_material,
                        'to_material': 'UNKNOWN_NEXT',  # Will be determined by next day's schedule
                        'total_time': typical_changeover_time,
                        'completed_time': remaining_capacity,
                        'remaining_time': typical_changeover_time - remaining_capacity
                    }
                    
                    changeover_states[line] = {
                        'last_activity': 'changeover',
                        'changeover_info': incomplete_changeover
                    }
                    continue
            
            # No incomplete changeover detected for this line/date
            if line not in changeover_states:
                changeover_states[line] = None
    
    return changeover_states


def extract_line_states_from_plan(plan_df: pd.DataFrame, cap_df: pd.DataFrame = None, 
                                  co_def: dict = None, simulation_date: pd.Timestamp = None,
                                  rate_map: dict = None) -> dict:
    """从生产计划提取产线状态（包含换产信息）用于持久化

    目的：
    - 提取每条产线的最后生产/换产状态，记录剩余换产以支持跨天连续性。

    Args:
        plan_df: 生产计划。
        cap_df: 产能（可选，用于换产分析）。
        co_def: 换产定义（可选）。
        simulation_date: 当前仿真日期（可选）。
        rate_map: 产率映射（可选）。

    Returns:
        dict: 产线状态字典，包含 `last_material/last_location/last_activity/changeover_info`。

    逻辑：
        - 若提供必要参数则先分析未完成换产→按线与仿真日取最后生产→合并换产状态并返回。
    """
    line_states = {}
    
    if plan_df.empty:
        return line_states
    
    # Analyze end-of-day changeover states if parameters provided
    changeover_states = {}
    if (cap_df is not None and co_def is not None and 
        simulation_date is not None and rate_map is not None):
        changeover_states = analyze_end_of_day_changeover_state(
            plan_df, cap_df, co_def, simulation_date, rate_map)
    
    # Group by line and simulation_date to get the last production for each line
    for (line, sim_date), group in plan_df.groupby(['line', 'simulation_date']):
        # Sort by production_plan_date to get the last production
        last_production = group.sort_values('production_plan_date').iloc[-1]
        
        # Check if we detected incomplete changeover for this line
        changeover_state = changeover_states.get(line)
        if changeover_state and changeover_state.get('last_activity') == 'changeover':
            # Last activity was incomplete changeover
            line_state = {
                'last_material': str(last_production['material']),
                'last_location': str(last_production['location']),
                'last_production_date': last_production['production_plan_date'].strftime('%Y-%m-%d'),
                'last_activity': 'changeover',
                'changeover_info': changeover_state['changeover_info']
            }
        else:
            # Default state: last activity was production
            line_state = {
                'last_material': str(last_production['material']),
                'last_location': str(last_production['location']),
                'last_production_date': last_production['production_plan_date'].strftime('%Y-%m-%d'),
                'last_activity': 'production',
                'changeover_info': None
            }
        
        line_states[line] = line_state
    
    return line_states

IDENTIFIER_COLS = [
    'material', 'location', 'line', 'delegate_line', 'from_material', 'to_material'
]


def _normalize_location(location_str: str) -> str:
    """标准化地点字符串（数字左补零至4位）

    Args:
        location_str: 地点字符串（如 "386"/"0386"/"A888"）。

    Returns:
        str: 数字补零后的地点或原样返回的字母数字地点。
    """
    if pd.isna(location_str) or location_str is None:
        return ""
    
    location_str = str(location_str).strip()
    
    try:
        # 检查是否为纯数字字符串
        if location_str.isdigit():
            return str(int(location_str)).zfill(4)
        else:
            # 非数字location（如A888），直接返回字符串，不做padding
            return location_str
    except (ValueError, TypeError):
        return str(location_str)


def _cast_identifiers_to_str(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    """将标识符列转换为字符串类型并标准化地点

    Args:
        df: 待处理 DataFrame。
        cols: 目标列列表，默认使用 `IDENTIFIER_COLS`。

    Returns:
        DataFrame: 指定列为字符串类型且地点已标准化的副本。
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
    """校验合并键在两个 DataFrame 中的 dtype 一致性"""
    for k in keys:
        if k in df1.columns and k in df2.columns:
            if df1[k].dtype != df2[k].dtype:
                raise TypeError(
                    f"Merge key '{k}' has mismatched dtypes: {df1[k].dtype} vs {df2[k].dtype}"
                )

def load_daily_net_demand(module3_output_dir: str, simulation_date: pd.Timestamp) -> pd.DataFrame:
    """加载前一日 Module3 的净需求，筛选层级并规范化数量

    目的：
    - 按数据流规范读取前一日输出，筛选 `layer=0` 下游需求，数量取绝对值，保证 `requirement_date` 为日期类型。

    Args:
        module3_output_dir: Module3 每日输出目录。
        simulation_date: 当前仿真日期。

    Returns:
        DataFrame: 处理后的净需求数据。

    逻辑：
        - 找到前一日文件→读取 NetDemand→筛选层级→数量绝对值→日期规范化→返回。
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
            # print(f"Info: No previous day Module3 output found for {prev_date_str}. This is expected for the first day.")
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
    """计算计划窗口（同时用于净需求筛选与生产分布）

    目的：
    - 尊重 PTF 冻结期，生成 `[window_start, window_end]` 日期区间供后续分配。

    Args:
        simulation_date: 当前仿真日期（审查日）。
        ptf: 计划冻结期（天）。
        lsk: 批量/周期键（规划视窗天数）。

    Returns:
        tuple: `(window_start, window_end)`。
    """
    window_start = simulation_date + timedelta(days=ptf)          # After frozen period
    window_end = simulation_date + timedelta(days=ptf + lsk - 1)  # PTF + planning horizon (LSK days)
    return window_start, window_end


def calculate_changeover_metrics(production_plan: pd.DataFrame, changeover_def: pd.DataFrame) -> pd.DataFrame:
    """基于生产计划计算换产指标（次数、时间、成本、mu_loss）

    目的：
    - 汇总每个生产日/地点/产线的换产发生次数并乘以定义值，生成换产日志。

    Args:
        production_plan: 包含 `changeover_id` 的生产计划。
        changeover_def: 换产定义（含 `time/cost/mu_loss`）。

    Returns:
        DataFrame: 换产日志（date/location/line/type/count/time/cost/mu_loss）。

    逻辑：
        - 分组计数→与定义合并→计算总指标→返回日志表。
    """
    changeover_log = []
    
    if production_plan.empty or changeover_def.empty:
        return pd.DataFrame(columns=['date', 'location', 'line', 'changeover_type', 'count', 'time', 'cost', 'mu_loss'])
    
    # Group by date, location, line, changeover_id to count occurrences
    changeover_summary = production_plan[
        production_plan['changeover_id'].notna()
    ].groupby(['production_plan_date', 'location', 'line', 'changeover_id']).size().reset_index(name='count')
    
    # 🔧 去重 changeover_def，避免重复键导致返回 Series
    changeover_def_clean = changeover_def.drop_duplicates(subset=['changeover_id', 'line'], keep='first')
    if len(changeover_def_clean) < len(changeover_def):
        print(f"Warning: Removed {len(changeover_def) - len(changeover_def_clean)} duplicate changeover definitions")
    
    # Merge with changeover definitions to get time, cost, mu_loss
    changeover_def_indexed = changeover_def_clean.set_index(['changeover_id', 'line'])
    
    for _, row in changeover_summary.iterrows():
        date = row['production_plan_date']
        location = row['location']
        line = row['line']
        changeover_id = row['changeover_id']
        count = row['count']
        
        # Get changeover definition
        try:
            definition = changeover_def_indexed.loc[(changeover_id, line)]
            
            # 🔧 处理可能返回 Series 的情况（虽然已去重，但保险起见）
            if isinstance(definition, pd.Series):
                time_per_changeover = float(definition.get('time', 0))
                cost_per_changeover = float(definition.get('cost', 0))
                mu_loss_per_changeover = float(definition.get('mu_loss', 0))
            else:
                # DataFrame - 取第一行
                time_per_changeover = float(definition.iloc[0].get('time', 0))
                cost_per_changeover = float(definition.iloc[0].get('cost', 0))
                mu_loss_per_changeover = float(definition.iloc[0].get('mu_loss', 0))
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
    """加载模块4配置（并做向后兼容映射）

    目的：
    - 读取必需工作表，转换标识符类型，映射到旧键名，兼容可选 `NetDemand` 与 `Global_seed`。

    Args:
        filepath: 配置 Excel 路径。

    Returns:
        dict: 配置字典，含标准化后的各表。
    """
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
    """校验配置一致性与潜在问题

    目的：
    - 检查 `NetDemand` 与 `MaterialLocationLineCfg` 的可合并性，提示缺少线配置或一物料地点多线情况。

    Args:
        cfg: 配置字典。

    Returns:
        list: 问题列表（非致命，供报告）。
    """
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
    """判断是否为某物料的审查日（基于整数 LSK）

    Args:
        simulation_date: 当前仿真日期。
        simulation_start: 仿真起始日期。
        lsk: 审查间隔天数（整数）。
        day: 首次审查相对起始的偏移天数。

    Returns:
        bool: 若是审查日返回 True。
    """
    days_since_start = (simulation_date - simulation_start).days
    first_review_day = int(day)-1  # offset from start
    return (days_since_start - first_review_day) % int(lsk) == 0 and days_since_start >= first_review_day



def build_unconstrained_plan_for_single_day(net_demand_df, mlcfg, simulation_date, simulation_start, issues):
    """构建单日无约束生产计划（仅针对审查日物料）

    目的：
    - 读取当日净需求（严格匹配 `requirement_date == simulation_date`），依据 MLCFG 计算最小批与舍入，得到无约束计划。

    Args:
        net_demand_df: 当日净需求。
        mlcfg: 物料地点产线配置。
        simulation_date: 当前仿真日期。
        simulation_start: 仿真起始日期（用于审查日逻辑）。
        issues: 问题收集列表。

    Returns:
        DataFrame: 无约束计划（material/location/line/planned_date/uncon_planned_qty/...）。

    逻辑：
        - 过滤审查日物料→按物料地点合并配置→严格匹配当日需求→聚合数量→按最小批与 RV 上取整→生成结果。
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
    """优化换产序列（数量优先与换产时间联合策略）

    目的：
    - 首件按原始需求量最大选择；后续优先最小换产时间，若并列使用数量打破。

    Args:
        batches: 批次列表，含 `material/uncon_planned_qty/original_quantity`。
        co_mat: MultiIndex 序列 `[(from,to)] -> changeover_id`。
        co_def: 换产定义 `(changeover_id, line) -> time`。
        line: 产线标识。

    Returns:
        list: 批次索引的执行顺序。
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
                                                   previously_allocated_capacity=None, issues=None):
    """集中式产能分配（含跨天换产连续性与产能跟踪）

    目的：
    - 在审查窗内分配产能并考虑换产时间，延续前一日未完成换产，记录未满足与校验信息。

    Args:
        uncon: 无约束计划 DataFrame。
        cap_df: 产能 DataFrame。
        rate_map: 产率映射。
        co_mat: 换产矩阵。
        co_def: 换产定义。
        mlcfg: 物料地点产线配置。
        previous_line_states: 前一日产线状态（可选）。
        simulation_date: 当前仿真日期（可选）。
        previously_allocated_capacity: 历史已分配产能（可选）。
        issues: 问题列表（可选）。

    Returns:
        tuple: `(plans_log, exceed_log)` 产能分配结果与未满足记录。

    逻辑：
        - 按线与仿真日分组→优化序列→处理跨天换产→在视窗内按日分配（先换产后生产）→更新剩余产能→记录未满足。
    """
    plans_log = []
    exceed = []
    if issues is None:
        issues = []
    
    # 默认 changeover 时间（小时）
    DEFAULT_CHANGEOVER_TIME = 24
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

        # Initialize state from previous day with enhanced cross-day changeover handling
        prev_mat = None
        initial_co_remain = 0
        initial_coid = None
        has_incomplete_changeover = False
        
        if previous_line_states and line in previous_line_states:
            line_state = previous_line_states[line]
            prev_mat = line_state.get('last_material')
            last_activity = line_state.get('last_activity', 'production')
            changeover_info = line_state.get('changeover_info')
            
            # print(f"Line {line}: Previous day state - material: {prev_mat}, activity: {last_activity}")
            
            # Handle incomplete changeover from previous day
            if last_activity == 'changeover' and changeover_info:
                remaining_time = changeover_info.get('remaining_time', 0)
                if remaining_time > 0:
                    initial_co_remain = remaining_time
                    initial_coid = changeover_info.get('changeover_id')
                    has_incomplete_changeover = True
                    # Update prev_mat to the target material of the incomplete changeover
                    prev_mat = changeover_info.get('to_material', prev_mat)
                    # print(f"Line {line}: Continuing incomplete changeover {initial_coid}, remaining time: {remaining_time}")
                else:
                    # Changeover was completed, use target material
                    prev_mat = changeover_info.get('to_material', prev_mat)
                    # print(f"Line {line}: Previous changeover completed, starting with material: {prev_mat}")

        for cur_plan_idx, cur_plan in enumerate(batch_list):
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

            # Handle changeover logic with cross-day continuity
            if cur_plan_idx == 0 and has_incomplete_changeover:
                # First material of the day with incomplete changeover from previous day
                co_remain = initial_co_remain
                coid_to_log = initial_coid
                is_first_co_day = True
                # print(f"Line {line}: Using incomplete changeover from previous day: {initial_co_remain} hours")
            elif prev_mat is not None and prev_mat != cur_mat:
                # Normal changeover calculation
                try:
                    # Ensure material IDs are strings for lookup
                    prev_mat_str = str(prev_mat)
                    cur_mat_str = str(cur_mat)
                    
                    # 🔍 调试：显示详细的查找信息
                    # print(f"\n🔍 DEBUG Changeover Lookup:")
                    # print(f"  Line: {line} (type: {type(line)})")
                    # print(f"  From material: '{prev_mat_str}' (type: {type(prev_mat_str)}, len: {len(prev_mat_str)})")
                    # print(f"  To material: '{cur_mat_str}' (type: {type(cur_mat_str)}, len: {len(cur_mat_str)})")
                    # print(f"  Co_mat index type: {co_mat.index.dtypes if hasattr(co_mat.index, 'dtypes') else type(co_mat.index)}")
                    # print(f"  Co_mat has {len(co_mat)} entries")
                    
                    # 检查是否存在于 index 中
                    if (prev_mat_str, cur_mat_str) in co_mat.index:
                        # print(f"  ✅ Key found in co_mat index")
                        coid_result = co_mat.loc[(prev_mat_str, cur_mat_str)]
                        # print(f"  📋 Retrieved changeover_id from co_mat: '{coid_result}' (type: {type(coid_result)})")
                        
                        # 🔧 关键修复：处理可能返回 Series 的情况（重复键）
                        if isinstance(coid_result, pd.Series):
                            # print(f"  ⚠️  WARNING: Multiple changeover definitions found! Using first one.")
                            coid = str(coid_result.iloc[0])
                        else:
                            coid = str(coid_result)
                        
                        # print(f"  📋 Final changeover_id: '{coid}' (type: {type(coid)})")
                        
                        # 检查 co_def 查找
                        # print(f"  🔍 Looking up in co_def with key: ('{coid}', '{line}')")
                        if (coid, line) in co_def:
                            co_time = co_def[(coid, line)]
                            # print(f"  ✅ Found in co_def: time={co_time}")
                        else:
                            co_time = DEFAULT_CHANGEOVER_TIME
                            # print(f"  ⚠️  NOT found in co_def! Using default time={DEFAULT_CHANGEOVER_TIME}")
                            # 记录缺失的 changeover 定义
                            issues.append({
                                'sheet': 'M4_ChangeoverDefinition',
                                'row': '',
                                'issue': f"缺失 changeover 定义: changeover_id={coid}, line={line}。已使用默认时间 {DEFAULT_CHANGEOVER_TIME} 小时。"
                            })
                    else:
                        # print(f"  ❌ Key NOT found in co_mat index")
                        # 显示前5个索引条目
                        # print(f"  Available keys (first 5): {list(co_mat.index[:5])}")
                        # 记录缺失的 changeover matrix 定义
                        coid = f"MISSING_CO_{prev_mat_str}_to_{cur_mat_str}"
                        co_time = DEFAULT_CHANGEOVER_TIME
                        issues.append({
                            'sheet': 'M4_ChangeoverMatrix',
                            'row': '',
                            'issue': f"缺失物料切换定义: {prev_mat_str} → {cur_mat_str}，产线 {line}。已使用默认 changeover_id 和时间 {DEFAULT_CHANGEOVER_TIME} 小时。"
                        })
                    
                    # print(f"  ✅ Final result: changeover_id={coid}, time={co_time}")
                    # print(f"  📌 Will set: co_remain={co_time}, coid_to_log={coid}, is_first_co_day=True")
                except Exception as e:
                    print(f"  ⚠️  Exception: {type(e).__name__}: {e}")
                    print(f"  Available co_mat index keys (sample): {list(co_mat.index[:10]) if len(co_mat) > 0 else 'EMPTY'}")
                    coid = None
                    co_time = 0
                co_remain = co_time
                coid_to_log = coid
                is_first_co_day = True
                # print(f"Line {line}: Changeover from {prev_mat} to {cur_mat}, time needed: {co_time}")
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
                    
                    # if previously_used_hours > 0:
                        # print(f"Line {line}: Production plan date {day_dt.strftime('%Y-%m-%d')} has {previously_used_hours:.2f} hours already allocated by previous simulation dates. Available capacity: {today_cap:.2f} hours")
                
                # First deduct changeover time from capacity (without creating separate record)
                changeover_completed_this_day = False
                changeover_used = 0
                
                # 🔍 调试：显示 changeover 处理逻辑
                # if coid_to_log is not None:
                #     print(f"  🔧 Processing changeover: coid_to_log={coid_to_log}, co_remain={co_remain}, is_first_co_day={is_first_co_day}")
                
                if co_remain > 0:
                    changeover_used = min(today_cap, co_remain)
                    co_remain -= changeover_used
                    today_cap -= changeover_used
                    # Mark if changeover completed this day
                    if co_remain == 0:
                        changeover_completed_this_day = True
                        # print(f"  ✅ Changeover completed this day: {coid_to_log}")
                    # If changeover not completed, update cap_map and continue to next day
                    if co_remain > 0:
                        cap_map[cap_key] = current_remaining_cap - changeover_used
                        # print(f"  ⏳ Changeover not completed, continue to next day")
                        continue
                elif co_remain == 0 and is_first_co_day:
                    # 🔧 关键修复：如果 co_time 为 0（无需换产时间），仍然标记为已完成
                    changeover_completed_this_day = True
                    # print(f"  ✅ Changeover with 0 time (instant changeover): {coid_to_log}")
                
                # Then allocate production capacity (with changeover_id if applicable)
                rate = float(rate_map.get((cur_mat, line), 1))
                can_produce = min(prod_remain, int(today_cap * rate))
                hours_used = can_produce / rate if rate else 0
                
                # Only create record if there's actual production
                if can_produce > 0:
                    # 🔍 调试：显示 changeover_id 记录决策
                    will_record_changeover = coid_to_log if (changeover_completed_this_day or is_first_co_day) else None
                    # if coid_to_log is not None:
                    #     print(f"  📝 Recording production with changeover_id decision:")
                    #     print(f"     coid_to_log={coid_to_log}")
                    #     print(f"     changeover_completed_this_day={changeover_completed_this_day}")
                    #     print(f"     is_first_co_day={is_first_co_day}")
                    #     print(f"     will_record_changeover={will_record_changeover}")
                    
                    plans_log.append({
                        'material': cur_mat, 'location': location, 'line': line,
                        'simulation_date': sim_date, 'production_plan_date': day_dt,
                        'available_date': day_dt + timedelta(days=int(mct_map.get((cur_mat, location), 0))),
                        'uncon_planned_qty': cur_plan['uncon_planned_qty'],
                        'con_planned_qty': can_produce,
                        'changeover_id': will_record_changeover
                    })
                    
                    # if will_record_changeover:
                    #     print(f"  ✅ Recorded changeover_id: {will_record_changeover}")
                    
                    # Reset changeover tracking after first production record
                    if coid_to_log is not None:
                        # print(f"  🔄 Resetting changeover tracking")
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
    """仿真生产可靠性（将计划量转为实际产出）

    Args:
        plan: 生产计划。
        pr_cfg: 生产可靠性配置 `pr`（按地点/线）。
        seed: 随机种子。

    Returns:
        DataFrame: 增加 `produced_qty` 的计划表。
    """
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
    """去重校验问题记录，返回字典列表"""
    if not issues:
        return issues
    df = pd.DataFrame(issues)
    df = df.drop_duplicates()
    return df.to_dict(orient='records')

def write_output(plan, exc, issues, changeover_log, out_path, simulation_date=None):
    """写出每日或汇总输出文件

    目的：
    - 保障列头一致，按 `simulation_date` 写每日版本或写合并版本；包含四张表：ProductionPlan/CapacityExceed/Validation/ChangeoverLog。

    Args:
        plan: 生产计划 DataFrame。
        exc: 超额记录 DataFrame。
        issues: 校验问题列表（或 DataFrame）。
        changeover_log: 换产日志 DataFrame。
        out_path: 基础输出路径。
        simulation_date: 仿真日期（提供则写每日版本）。

    Returns:
        str: 实际写出的文件路径。
    """
    # 🆕 列头保障函数
    def _ensure(df, cols):
        if df is None or df.empty:
            return pd.DataFrame(columns=cols)
        for c in cols:
            if c not in df.columns:
                df[c] = pd.Series(dtype='object')
        return df[cols]
    plan = _ensure(plan, [
        'material','location','line','simulation_date','production_plan_date',
        'available_date','uncon_planned_qty','con_planned_qty','produced_qty',
        'changeover_id','changeover_time','changeover_time_remaining',
        'is_first_changeover_day'
    ])
    exc = _ensure(exc, ['material','location','line','simulation_date','exceed_type','exceed_qty'])
    issues_df = pd.DataFrame(issues)
    issues_df = _ensure(issues_df, [
        'type','location','line','production_plan_date','simulation_date',
        'previously_allocated_hours','currently_allocated_hours',
        'total_allocated_hours','message','issue','sheet','row'
    ])
    changeover_log = _ensure(changeover_log, ['date','location','line','changeover_type','count','time','cost','mu_loss'])
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
        plan.to_excel(w, sheet_name='ProductionPlan', index=False)
        exc.to_excel(w, sheet_name='CapacityExceed', index=False)
        issues_df.to_excel(w, sheet_name='Validation', index=False)
        changeover_log.to_excel(w, sheet_name='ChangeoverLog', index=False)
    return final_path

def run_daily_production_planning(config_file: str, module3_output_dir: str, 
                                 simulation_date: pd.Timestamp, simulation_start: pd.Timestamp,
                                 output_dir: str) -> str:
    """运行单日生产计划（含跨天换产连续性）

    目的：
    - 加载配置与净需求→构建无约束计划→分配产能与仿真生产→保存状态与产能→去重问题并写出当日输出。

    Args:
        config_file: M4 配置 Excel 路径。
        module3_output_dir: Module3 每日输出目录。
        simulation_date: 当前仿真日期。
        simulation_start: 仿真起始日期。
        output_dir: 输出目录。

    Returns:
        str: 生成的每日输出文件路径。

    逻辑：
        - 加载与校验→读取净需求→构建无约束→读取前日状态与历史产能→集中分配→仿真与计算换产指标→保存状态与产能→写输出。
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
        # if previous_line_states:
        #     # print(f"Loaded previous day's line states: {list(previous_line_states.keys())}")
        # else:
        #     print("No previous day's line states found - starting fresh")
        
        # Load previously allocated capacity from all previous simulation dates
        previously_allocated_capacity = load_all_previous_capacity(output_dir, simulation_date)
        # if previously_allocated_capacity:
        #     print(f"Loaded previously allocated capacity: {len(previously_allocated_capacity)} capacity allocations from previous simulation dates")
        # else:
        #     print("No previously allocated capacity found - starting fresh")
        
        # Set up capacity allocation parameters
        # Also normalize changeover matrix to ensure proper matching
        co_mat_df = cfg['ChangeoverMatrix'].copy()
        co_mat_df['from_material'] = co_mat_df['from_material'].astype(str)
        co_mat_df['to_material'] = co_mat_df['to_material'].astype(str)
        co_mat = co_mat_df.set_index(['from_material', 'to_material'])['changeover_id']
        # 对MultiIndex进行排序以避免性能警告
        co_mat = co_mat.sort_index()
        
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
            previously_allocated_capacity=previously_allocated_capacity, issues=issues
        )
        
        # Simulate production reliability
        random_seed = cfg.get('RandomSeed', 42)
        plan_log = simulate_production(plan_log, cfg['ProductionReliability'], seed=random_seed)
        
        # Calculate changeover metrics
        changeover_log = calculate_changeover_metrics(plan_log, co_def_df)
        
        # Extract and save current day's line states for next day with enhanced changeover tracking
        current_line_states = extract_line_states_from_plan(plan_log, cap_df, co_def, simulation_date, rate_map.to_dict())
        if current_line_states:
            save_line_state(output_dir, simulation_date, current_line_states)
            # print(f"Saved current day's line states: {list(current_line_states.keys())}")
        
        # Extract and save current day's allocated capacity for future simulation dates
        current_allocated_capacity = extract_allocated_capacity_from_plan(plan_log, rate_map.to_dict(), co_def)
        if current_allocated_capacity:
            save_allocated_capacity(output_dir, simulation_date, current_allocated_capacity)
            # print(f"Saved current day's allocated capacity: {len(current_allocated_capacity)} capacity allocations (in hours)")
        
        # Validate capacity allocation to ensure no over-allocation
        capacity_validation_issues = validate_capacity_allocation(plan_log, previously_allocated_capacity, simulation_date, rate_map.to_dict(), co_def)
        if capacity_validation_issues:
            # print(f"Capacity allocation validation completed: {len(capacity_validation_issues)} validation records generated")
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
        
        # print(f"Module4 daily output generated: {daily_output_path}")
        
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
    """合并多个每日输出生成汇总文件

    Args:
        daily_output_files: 每日输出文件列表。
        output_path: 汇总输出路径。

    逻辑/输出：
        - 逐文件读取四张表→合并并去重→调用 `write_output` 生成汇总文件。
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
    
    # print(f"Consolidated Module4 output generated: {output_path}")


def main():
    """命令行入口：支持日度与旧版兼容模式

    目的：
    - 解析参数并按模式执行：日度模式读取 Module3 输出；旧版模式走整段窗口逻辑并保持向后兼容。
    """
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
            
            # print(f"Daily production planning completed: {output_file}")
            
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
            # 对MultiIndex进行排序以避免性能警告
            co_mat = co_mat.sort_index()
            co_def_df = cfg['ChangeoverDefinition']
            co_def = co_def_df.set_index(['changeover_id', 'line'])['time'].to_dict()
            cap_df = cfg['LineCapacity']
            rate_map = mlcfg.set_index(['material', 'delegate_line'])['prd_rate']
            rate_map.index.set_names(['material', 'line'], inplace=True)
            
            # For legacy mode - create empty unconstrained plan for now
            uncon = pd.DataFrame(columns=['material', 'location', 'line', 'planned_date', 'uncon_planned_qty', 'simulation_date', 'original_quantity'])
            
            plan_log, exceed_log = centralized_capacity_allocation_with_changeover(
                uncon, cap_df, rate_map, co_mat, co_def, mlcfg, issues=issues
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
            
            # print(f"Legacy production planning completed: {args.output}")
            
    except Exception as e:
        print(f'[ERROR]: {str(e)}')
        raise


if __name__ == '__main__':
    main()
