"""
production_runner.py

生产计划(Module4)集成运行函数模块。
"""

import pandas as pd
import os
from pathlib import Path
from typing import Any, Dict, Optional

from ...modules import module4
from ...utils.defaults import M6_RANDOM_SEED
from ...utils.normalization import normalize_material


def run_module4_integrated(
    config_dict: dict,
    module3_output_dir: str,
    simulation_date: pd.Timestamp,
    simulation_start: pd.Timestamp,
    output_dir: str,
    skip_file_output: bool = False,
    module3_result: Optional[Dict[str, Any]] = None,
    previous_line_states_override: Optional[dict] = None,
    allocated_capacity_override: Optional[dict] = None,
    skip_state_file_output: bool = False,
) -> Dict[str, Any]:
    """集成模式运行 Module4 生产计划（直接用 config_dict）

    目的：
    - 无需临时文件中转，使用内存中的配置与 Module3 输出构建并分配当日生产计划，返回供调用方后续持久化的 M4 结果集合。

    Args:
        config_dict: 配置数据字典（包含 M4 所需表）。
        module3_output_dir: Module3 输出目录，当 `module3_result` 为 None 时用于读取日度净需求。
        simulation_date: 当前仿真日期。
        simulation_start: 仿真开始日期。
        output_dir: 输出目录，用于写每日 M4 输出。
        skip_file_output: 是否跳过写出 Excel 文件（数据库模式下通常启用）。
        module3_result: Module3 运行结果（内存数据），包含 `net_demand_df`；优先使用此参数。
        previous_line_states_override: 如果提供，跳过文件读取，直接使用此产线状态。
        allocated_capacity_override: 如果提供，跳过文件读取，直接使用此已分配产能。
        skip_state_file_output: 是否跳过写入line_states/allocated_capacity JSON文件。

    Returns:
        Dict[str, Any]: 包含 `production_df`、`exceed_log`、`issues_df`、
        `changeover_log`、`current_line_states`、`current_allocated_capacity` 的结果字典。

    输入数据：
        - M4 配置表（LineCfg/Capacity/ChangeoverMatrix/Definition/ProductionReliability）。
        - Module3 的日度净需求（优先从module3_result内存获取，否则从文件读取）。

    输出/副作用：
        - 可按需写出每日 M4 输出文件；返回当日及未来可用的生产记录、校验结果以及跨天状态数据；可能更新产线状态与已分配产能文件。

    逻辑：
        - 校验配置 → 加载净需求 → 构建无约束计划 → 处理换产与产能分配 → 模拟生产可靠性 → 提取并保存跨天状态 → 返回结果字典。
    """
    try:
        # 验证必需的Module4配置数据
        required_m4_configs = [
            'M4_MaterialLocationLineCfg',
            'M4_LineCapacity', 
            'M4_ChangeoverMatrix',
            'M4_ChangeoverDefinition',
            'M4_ProductionReliability'
        ]
        
        for config_name in required_m4_configs:
            if config_name not in config_dict or config_dict[config_name].empty:
                raise ValueError(f"缺少必需的Module4配置数据：{config_name}")
        
        # 直接构建 Module4 所需的配置数据
        # 直接使用config_dict，不再需要子配置字典
        m4_config = config_dict
        
        # 🦆 优先从内存加载Module3净需求数据，否则从文件读取
        if module3_result is not None and 'net_demand_df' in module3_result:
            net_demand_df = module3_result['net_demand_df'].copy()
            # 🦆 内存模式：不按simulation_date筛选，因为previous_day_m3_result包含前一天计算的所有净需求
            # 这些净需求的requirement_date才是实际需求日期，simulation_date只是M3运行的日期
            
            # 🔧 关键修复：应用与load_daily_net_demand相同的处理逻辑
            # 1. 筛选layer=0（下游需求），与文件加载模式保持一致
            if 'layer' in net_demand_df.columns:
                net_demand_df = net_demand_df[net_demand_df['layer'] == 0].copy()
            # 2. 数量取绝对值，与文件加载模式保持一致
            if 'quantity' in net_demand_df.columns:
                net_demand_df['quantity'] = net_demand_df['quantity'].abs()
        else:
            # 从文件加载
            net_demand_df = module4.load_daily_net_demand(module3_output_dir, simulation_date)
        net_demand_df = module4.cast_identifiers_to_str(net_demand_df, ['material', 'location'])
        
        # 🔧 修复Module3→Module4数据流：标准化material字段，移除.0后缀
        if not net_demand_df.empty and 'material' in net_demand_df.columns:
            net_demand_df['material'] = net_demand_df['material'].apply(normalize_material).astype('string')
        
        if net_demand_df.empty:
            pass
        
        # 确保 requirement_date 是 datetime 类型
        if not net_demand_df.empty and 'requirement_date' in net_demand_df.columns:
            net_demand_df['requirement_date'] = pd.to_datetime(net_demand_df['requirement_date'])
        
        # 构建无约束计划
        mlcfg = m4_config['M4_MaterialLocationLineCfg']
        
        # 确保MLCFG也应用类型转换（与NetDemand保持一致）
        mlcfg = module4.cast_identifiers_to_str(mlcfg.copy(), ['material', 'location'])
        
        issues = []
        uncon_plan = module4.build_unconstrained_plan_for_single_day(
            net_demand_df, mlcfg, simulation_date, simulation_start, issues
        )
        
        # 🔧 关键修复：标准化uncon_plan中的material字段，确保与changeover matrix一致
        if not uncon_plan.empty and 'material' in uncon_plan.columns:
            uncon_plan['material'] = uncon_plan['material'].apply(normalize_material).astype('string')
        
        # 设置产能分配参数
        # 🔧 关键修复：标准化 ChangeoverMatrix 中的字段为字符串类型
        co_mat_df = m4_config['M4_ChangeoverMatrix'].copy()
        
        co_mat_df['from_material'] = co_mat_df['from_material'].astype(str)
        co_mat_df['to_material'] = co_mat_df['to_material'].astype(str)
        co_mat_df['changeover_id'] = co_mat_df['changeover_id'].astype(str)
        
        # 注意：Changeover 去重已在 load_configuration 中完成
        
        co_mat = co_mat_df.set_index(['from_material', 'to_material'])['changeover_id']
        # 对MultiIndex进行排序以避免性能警告
        co_mat = co_mat.sort_index()
        
        # 🔧 关键修复：标准化 ChangeoverDefinition 中的 changeover_id 为字符串类型
        co_def_df = m4_config['M4_ChangeoverDefinition'].copy()
        co_def_df['changeover_id'] = co_def_df['changeover_id'].astype(str)
        co_def = co_def_df.set_index(['changeover_id', 'line'])['time'].to_dict()
        
        cap_df = m4_config['M4_LineCapacity'].copy()
        cap_df['date'] = pd.to_datetime(cap_df['date'])
        
        rate_map = mlcfg.set_index(['material', 'delegate_line'])['prd_rate']
        rate_map.index.set_names(['material', 'line'], inplace=True)
        
        # 加载前一天产线状态用于跨天转产连续性
        if previous_line_states_override is not None:
            previous_line_states = previous_line_states_override
        else:
            previous_line_states = module4.load_line_state(output_dir, simulation_date)
        
        # 加载之前所有仿真日期已分配的产能
        if allocated_capacity_override is not None:
            previously_allocated_capacity = allocated_capacity_override
        else:
            previously_allocated_capacity = module4.load_all_previous_capacity(output_dir, simulation_date)
        
        # 分配产能（支持跨天转产连续性和产能跟踪）
        plan_log, exceed_log = module4.centralized_capacity_allocation_with_changeover(
            uncon_plan, cap_df, rate_map, co_mat, co_def, mlcfg,
            previous_line_states=previous_line_states, simulation_date=simulation_date,
            previously_allocated_capacity=previously_allocated_capacity, issues=issues
        )
        
        # 仿真生产可靠性
        random_seed = m4_config.get('RandomSeed', M6_RANDOM_SEED)
        plan_log = module4.simulate_production(plan_log, m4_config['M4_ProductionReliability'], seed=random_seed)
        
        # 计算换产指标
        changeover_log = module4.calculate_changeover_metrics(plan_log, co_def_df)
        
        # 提取并保存当天产线状态供下一天使用（带跨天转产检测）
        current_line_states = module4.extract_line_states_from_plan(plan_log, cap_df, co_def, simulation_date, rate_map.to_dict())
        if current_line_states and not skip_state_file_output:
            module4.save_line_state(output_dir, simulation_date, current_line_states)
        
        # 提取并保存当天分配的产能供后续仿真日期使用
        current_allocated_capacity = module4.extract_allocated_capacity_from_plan(plan_log, rate_map.to_dict(), co_def)
        if current_allocated_capacity and not skip_state_file_output:
            module4.save_allocated_capacity(output_dir, simulation_date, current_allocated_capacity)
        
        # 去重问题
        issues = module4.dedup_issues(issues)
        
        # 转换issues为DataFrame
        issues_df = pd.DataFrame(issues) if issues else pd.DataFrame()
        
        # 生成输出文件（通常仅文件模式需要写出）
        if not skip_file_output:
            base_output_file = os.path.join(output_dir, "Module4Output.xlsx")
            daily_output_path = module4.write_output(
                plan_log, exceed_log, issues, changeover_log, 
                base_output_file, simulation_date
            )
        
        # 返回完整的Module4结果（包含所有输出表）
        # production_df 与 Dev 版本一致：只返回当日及未来可用的生产
        # 🔧 修复：不对 production_df 应用 normalize_identifiers
        # Dev 版本的 M4 输出 Excel 使用原始 location（如 386），不做 zfill(4)
        # normalize_identifiers 仅在传入 orchestrator 时由调用方应用
        production_df = pd.DataFrame()
        if not plan_log.empty and 'available_date' in plan_log.columns:
            plan_log['available_date'] = pd.to_datetime(plan_log['available_date'])
            # 与 Dev 版本一致：只返回 available_date >= simulation_date 的记录
            current_production = plan_log[plan_log['available_date'] >= simulation_date.normalize()]
            if not current_production.empty:
                production_df = current_production.copy()
        
        # 返回完整结构供数据库写入
        return {
            'production_df': production_df,
            'exceed_log': exceed_log if isinstance(exceed_log, pd.DataFrame) else pd.DataFrame(exceed_log) if exceed_log else pd.DataFrame(),
            'issues_df': issues_df,
            'changeover_log': changeover_log if isinstance(changeover_log, pd.DataFrame) else pd.DataFrame(changeover_log) if changeover_log else pd.DataFrame(),
            'current_line_states': current_line_states,
            'current_allocated_capacity': current_allocated_capacity,
        }
        
    except Exception as e:
        import traceback
        # 返回空结构
        return {
            'production_df': pd.DataFrame(),
            'exceed_log': pd.DataFrame(),
            'issues_df': pd.DataFrame(),
            'changeover_log': pd.DataFrame(),
            'current_line_states': {},
            'current_allocated_capacity': {},
        }

# ========== Module4 集成辅助函数 ==========
# 以下辅助函数基于文件读取，供旧路径或回退场景使用


def load_current_date_production_gr(module4_output_dir: str, current_date: pd.Timestamp, start_date: pd.Timestamp) -> pd.DataFrame:
    """加载历史 M4 生产计划并筛选当日入库

    目的：
    - 汇总从仿真开始至今的所有 M4 输出，提取 `available_date == current_date` 的生产记录用于入库。

    Args:
        module4_output_dir: Module4 输出目录。
        current_date: 当前日期。
        start_date: 仿真开始日期。

    Returns:
        pd.DataFrame: 当日应该入库的生产计划。

    输入/输出/逻辑：
        - 遍历日期读取 `Module4Output_YYYYMMDD.xlsx`→合并→按 available_date 过滤当日→返回关键列。
    """
    all_production_plans = []
    
    # 遍历从仿真开始到当前日期的所有M4输出文件
    date_range = pd.date_range(start_date, current_date, freq='D')
    
    for date in date_range:
        m4_file = Path(module4_output_dir) / f"Module4Output_{date.strftime('%Y%m%d')}.xlsx"
        
        if m4_file.exists():
            try:
                xl = pd.ExcelFile(m4_file)
                if 'ProductionPlan' in xl.sheet_names:
                    production_df = xl.parse('ProductionPlan')
                    
                    if not production_df.empty:
                        # 添加数据来源标识
                        production_df['source_file'] = str(m4_file)
                        production_df['source_date'] = date
                        all_production_plans.append(production_df)
                        
            except Exception as e:
                continue
    
    if not all_production_plans:
        return pd.DataFrame()
    
    # 合并所有生产计划
    combined_production = pd.concat(all_production_plans, ignore_index=True)
    
    # 筛选出当日应该入库的生产 (available_date = current_date)
    if 'available_date' in combined_production.columns:
        combined_production['available_date'] = pd.to_datetime(combined_production['available_date'])
        daily_available = combined_production[
            combined_production['available_date'].dt.normalize() == current_date.normalize()
        ]
        
        return daily_available[['material', 'location', 'line', 'simulation_date', 'available_date', 'produced_qty']]

    return pd.DataFrame()
