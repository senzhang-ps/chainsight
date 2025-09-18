# main_integration.py
# 主集成执行脚本 - 协调所有模块通过Orchestrator运行
#
# 执行顺序: M1 → M4 → M5 → M6 → M3
# 每日处理模式，确保模块间数据流环环相扣

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import os

# 导入所有模块
from orchestrator import create_orchestrator
from validation_manager import ValidationManager
from time_manager import SimulationTimeManager, initialize_time_manager
from config_validator import run_pre_simulation_validation
from inventory_balance_checker import InventoryBalanceChecker
from summary_report_generator import SummaryReportGenerator
import module1
import module3
import module4
import module5
import module6


# 标识符字段标准化函数（统一处理所有配置表）
def _normalize_location(location_str) -> str:
    """Normalize location string by padding with leading zeros to 4 digits"""
    try:
        return str(int(location_str)).zfill(4)
    except (ValueError, TypeError):
        return str(location_str).zfill(4)

def _normalize_material(material_str) -> str:
    """Normalize material string"""
    return str(material_str) if material_str is not None else ""

def _normalize_sending(sending_str) -> str:
    """Normalize sending string by padding with leading zeros to 4 digits"""
    try:
        return str(int(sending_str)).zfill(4)
    except (ValueError, TypeError):
        return str(sending_str).zfill(4)

def _normalize_receiving(receiving_str) -> str:
    """Normalize receiving string by padding with leading zeros to 4 digits"""
    try:
        return str(int(receiving_str)).zfill(4)
    except (ValueError, TypeError):
        return str(receiving_str).zfill(4)

def _normalize_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize identifier columns to string format with proper formatting"""
    if df.empty:
        return df
    
    # Define identifier columns that need string conversion
    identifier_cols = ['material', 'location', 'sending', 'receiving', 'sourcing', 'dps_location', 'from_material', 'to_material', 'line', 'delegate_line']
    
    df = df.copy()
    for col in identifier_cols:
        if col in df.columns:
            # Convert to string and handle NaN values
            df[col] = df[col].astype('string')
            # Apply specific normalization for location-type fields
            if col in ['location', 'dps_location']:
                df[col] = df[col].apply(_normalize_location)
            elif col == 'sending':
                df[col] = df[col].apply(_normalize_sending)
            elif col == 'receiving':
                df[col] = df[col].apply(_normalize_receiving)
            # Apply specific normalization for material-type fields
            elif col in ['material', 'from_material', 'to_material']:
                df[col] = df[col].apply(_normalize_material)
            # For other identifier columns (line, delegate_line, etc), ensure they are properly formatted strings
            else:
                df[col] = df[col].apply(lambda x: str(x) if pd.notna(x) else "")
    
    return df

def run_module4_integrated(
    config_dict: dict,
    module3_output_dir: str,
    simulation_date: pd.Timestamp,
    simulation_start: pd.Timestamp,
    output_dir: str
) -> pd.DataFrame:
    """
    集成模式运行 Module4 生产计划，直接使用 config_dict 数据
    
    Args:
        config_dict: 配置数据字典
        module3_output_dir: Module3 输出目录
        simulation_date: 当前仿真日期
        simulation_start: 仿真开始日期
        output_dir: 输出目录
        
    Returns:
        pd.DataFrame: 生产计划数据
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
        
        # 加载 Module3 的日度净需求数据
        net_demand_df = module4.load_daily_net_demand(module3_output_dir, simulation_date)
        net_demand_df = module4._cast_identifiers_to_str(net_demand_df, ['material', 'location'])
        
        if net_demand_df.empty:
            print(f"Warning: No NetDemand data for {simulation_date.strftime('%Y-%m-%d')}. Generating empty output.")
        
        # 确保 requirement_date 是 datetime 类型
        if not net_demand_df.empty and 'requirement_date' in net_demand_df.columns:
            net_demand_df['requirement_date'] = pd.to_datetime(net_demand_df['requirement_date'])
        
        # 构建无约束计划
        mlcfg = m4_config['M4_MaterialLocationLineCfg']
        
        # 确保MLCFG也应用类型转换（与NetDemand保持一致）
        mlcfg = module4._cast_identifiers_to_str(mlcfg.copy(), ['material', 'location'])
        
        issues = []
        uncon_plan = module4.build_unconstrained_plan_for_single_day(
            net_demand_df, mlcfg, simulation_date, simulation_start, issues
        )
        
        # 设置产能分配参数
        co_mat = m4_config['M4_ChangeoverMatrix'].set_index(['from_material', 'to_material'])['changeover_id']
        co_def_df = m4_config['M4_ChangeoverDefinition']
        co_def = co_def_df.set_index(['changeover_id', 'line'])['time'].to_dict()
        
        cap_df = m4_config['M4_LineCapacity'].copy()
        cap_df['date'] = pd.to_datetime(cap_df['date'])
        
        rate_map = mlcfg.set_index(['material', 'delegate_line'])['prd_rate']
        rate_map.index.set_names(['material', 'line'], inplace=True)
        
        # 分配产能
        plan_log, exceed_log = module4.centralized_capacity_allocation_with_changeover(
            uncon_plan, cap_df, rate_map, co_mat, co_def, mlcfg
        )
        
        # 仿真生产可靠性
        random_seed = m4_config.get('RandomSeed', 42)
        plan_log = module4.simulate_production(plan_log, m4_config['M4_ProductionReliability'], seed=random_seed)
        
        # 计算换产指标
        changeover_log = module4.calculate_changeover_metrics(plan_log, co_def_df)
        
        # 去重问题
        issues = module4.dedup_issues(issues)
        
        # 生成输出文件
        base_output_file = os.path.join(output_dir, "Module4Output.xlsx")
        daily_output_path = module4.write_output(
            plan_log, exceed_log, issues, changeover_log, 
            base_output_file, simulation_date
        )
        
        print(f"Module4 daily output generated: {daily_output_path}")
        
        # 返回生产计划数据（只返回当日可用的生产）
        if not plan_log.empty and 'available_date' in plan_log.columns:
            plan_log['available_date'] = pd.to_datetime(plan_log['available_date'])
            current_production = plan_log[plan_log['available_date'] >= simulation_date.normalize()]
            return current_production
        else:
            return pd.DataFrame()
        
    except Exception as e:
        import traceback
        print(f'[ERROR] Module4 integrated execution failed for {simulation_date.strftime("%Y-%m-%d")}: {str(e)}')
        print("Full traceback:")
        traceback.print_exc()
        return pd.DataFrame()

# ========== Module4 集成辅助函数（清理后） ==========

# 以下函数保留作为默认配置的备用，但不再使用临时文件













def load_all_historical_production_plans(module4_output_dir: str, current_date: pd.Timestamp, start_date: pd.Timestamp) -> pd.DataFrame:
    """
    加载所有历史的M4生产计划，筛选出当日应该入库的生产
    
    Args:
        module4_output_dir: Module4 输出目录
        current_date: 当前日期
        start_date: 仿真开始日期
        
    Returns:
        pd.DataFrame: 当日应该入库的生产计划数据
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
                print(f"Warning: Failed to read {m4_file}: {e}")
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
        
        if not daily_available.empty:
            print(f"  📦 发现当日入库的历史生产: {len(daily_available)} 条记录")
            for _, row in daily_available.iterrows():
                print(f"    {row['material']}@{row['location']}: {row['produced_qty']} (生产日期: {row['source_date'].strftime('%Y-%m-%d')})")
        
        return daily_available[['material', 'location', 'line', 'simulation_date', 'available_date', 'produced_qty']]
    
    return pd.DataFrame()

def load_module4_production_output(output_path: str, current_date: pd.Timestamp) -> pd.DataFrame:
    """
    从 Module4 输出文件中加载生产计划数据 (保留用于向后兼容)
    
    Args:
        output_path: Module4 输出文件路径
        current_date: 当前日期
        
    Returns:
        pd.DataFrame: 生产计划数据
    """
    try:
        if not os.path.exists(output_path):
            print(f"Warning: Module4 output file not found: {output_path}")
            return pd.DataFrame()
            
        xl = pd.ExcelFile(output_path)
        if 'ProductionPlan' not in xl.sheet_names:
            print(f"Warning: ProductionPlan sheet not found in {output_path}")
            return pd.DataFrame()
            
        production_df = xl.parse('ProductionPlan')
        
        # 筛选当日的生产计划 (available_date = current_date)
        if not production_df.empty and 'available_date' in production_df.columns:
            production_df['available_date'] = pd.to_datetime(production_df['available_date'])
            # 只返回当日或未来的生产计划
            production_df = production_df[production_df['available_date'] >= current_date.normalize()]
            
        return production_df
        
    except Exception as e:
        print(f"Error loading Module4 production output: {e}")
        return pd.DataFrame()

def load_global_seed(config_dict: dict) -> int:
    """
    统一从 Global_Seed sheet 读取随机种子
    
    Args:
        config_dict: 配置数据字典
        
    Returns:
        int: 随机种子值，默认为 42
    """
    if 'Global_Seed' in config_dict and not config_dict['Global_Seed'].empty:
        seed_df = config_dict['Global_Seed']
        if 'seed' in seed_df.columns:
            seed_value = int(seed_df.iloc[0]['seed'])
            print(f"🌱 从 Global_Seed 读取随机种子: {seed_value}")
            return seed_value
        elif len(seed_df.columns) > 0 and len(seed_df) > 0:
            # 兼容旧格式，读取第一列第一行
            seed_value = int(seed_df.iloc[0, 0])
            print(f"🌱 从 Global_Seed 兼容格式读取随机种子: {seed_value}")
            return seed_value
    
    print(f"⚠️  未找到 Global_Seed 配置，使用默认值: 42")
    return 42

def set_module_seeds(config_dict: dict, global_seed: int = None):
    """
    为所有模块设置统一的随机种子
    
    Args:
        config_dict: 配置数据字典
        global_seed: 全局种子值，如果为 None 则从配置读取
    """
    if global_seed is None:
        global_seed = load_global_seed(config_dict)
    
    # 设置 numpy全局种子
    np.random.seed(global_seed)
    
    # 为各模块设置种子（在配置中覆盖模块特定配置）
    config_dict['M1_RandomSeed'] = global_seed
    config_dict['M3_RandomSeed'] = global_seed  
    config_dict['M4_RandomSeed'] = global_seed
    config_dict['M5_RandomSeed'] = global_seed
    config_dict['M6_RandomSeed'] = global_seed
    
    print(f"✨ 已为所有模块设置统一随机种子: {global_seed}")
    return global_seed

def load_configuration(config_path: str) -> dict:
    """
    加载配置数据
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        dict: 配置数据字典
    """
    print(f"📋 加载配置文件: {config_path}")
    
    try:
        xl = pd.ExcelFile(config_path)
        config_dict = {}
        
        # 加载所有配置表
        for sheet_name in xl.sheet_names:
            config_dict[sheet_name] = xl.parse(sheet_name)
            print(f"  ✅ 加载配置表: {sheet_name} ({len(config_dict[sheet_name])} 行)")
        
        # 确保必要的配置表存在
        required_sheets = [
            'M1_InitialInventory',
            'Global_SpaceCapacity',
            'Global_Network',
            'Global_LeadTime',
            'Global_DemandPriority'
        ]
        
        missing_sheets = [sheet for sheet in required_sheets if sheet not in config_dict]
        if missing_sheets:
            print(f"⚠️  缺少必要配置表: {missing_sheets}")
            # 创建空的配置表
            for sheet in missing_sheets:
                config_dict[sheet] = pd.DataFrame()
        
        # 统一标准化所有配置表的标识符字段
        print(f"🔧 正在标准化标识符字段...")
        standardized_count = 0
        for sheet_name, df in config_dict.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                # 检查是否包含标识符字段
                identifier_cols = ['material', 'location', 'sending', 'receiving', 'sourcing', 'dps_location']
                has_identifiers = any(col in df.columns for col in identifier_cols)
                
                if has_identifiers:
                    original_dtypes = {col: str(df[col].dtype) for col in identifier_cols if col in df.columns}
                    config_dict[sheet_name] = _normalize_identifiers(df)
                    new_dtypes = {col: str(config_dict[sheet_name][col].dtype) for col in identifier_cols if col in config_dict[sheet_name].columns}
                    
                    # 记录标准化的字段
                    normalized_fields = []
                    for col in identifier_cols:
                        if col in df.columns and original_dtypes[col] != new_dtypes[col]:
                            normalized_fields.append(f"{col}({original_dtypes[col]}→{new_dtypes[col]})")
                    
                    if normalized_fields:
                        print(f"  🔧 {sheet_name}: {', '.join(normalized_fields)}")
                        standardized_count += 1
        
        if standardized_count > 0:
            print(f"✅ 已标准化 {standardized_count} 个配置表的标识符字段")
        else:
            print(f"✅ 所有配置表的标识符字段已是标准格式")
        
        return config_dict
        
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        raise

def run_integrated_simulation(
    config_path: str,
    start_date: str,
    end_date: str,
    output_base_dir: str = "./integrated_output"
):
    """
    运行完整的集成仿真
    
    Args:
        config_path: 配置文件路径
        start_date: 仿真开始日期 (YYYY-MM-DD)
        end_date: 仿真结束日期 (YYYY-MM-DD)
        output_base_dir: 输出基础目录
    """
    print(f"🚀 开始集成仿真: {start_date} 到 {end_date}")
    print("=" * 60)
    
    # 1. 预验证配置
    print(f"🔍 正在运行仿真前配置验证...")
    validation_passed, validation_report = run_pre_simulation_validation(config_path, output_base_dir)
    
    print(f"📝 验证报告已生成: {validation_report}")
    
    if not validation_passed:
        print("❌ 配置验证失败，请查看验证报告并修复错误后再运行仿真。")
        return {
            'validation_passed': False,
            'validation_report': validation_report,
            'simulation_completed': False
        }
    
    print("✅ 配置验证通过，开始仿真...")
    
    # 2. 初始化时间管理器
    time_manager = initialize_time_manager(start_date)
    
    # 3. 创建输出目录
    output_dir = Path(output_base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    orchestrator_output_dir = output_dir / "orchestrator"
    module_outputs = {
        'module1': output_dir / "module1",
        'module3': output_dir / "module3", 
        'module4': output_dir / "module4",
        'module5': output_dir / "module5",
        'module6': output_dir / "module6"
    }
    
    for module_dir in module_outputs.values():
        module_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载配置
    config_dict = load_configuration(config_path)
    
    # 设置全局随机种子
    global_seed = set_module_seeds(config_dict)
    
    # 初始化Orchestrator
    print(f"\n🎯 初始化Orchestrator")
    orchestrator = create_orchestrator(
        start_date=start_date,
        output_dir=str(orchestrator_output_dir)
    )
    
    # 设置初始库存
    if 'M1_InitialInventory' in config_dict and not config_dict['M1_InitialInventory'].empty:
        orchestrator.initialize_inventory(config_dict['M1_InitialInventory'])
    else:
        print("⚠️  未找到初始库存配置，使用空库存")
        orchestrator.initialize_inventory(pd.DataFrame(columns=['material', 'location', 'quantity']))
    
    # 设置空间容量
    if 'Global_SpaceCapacity' in config_dict and not config_dict['Global_SpaceCapacity'].empty:
        orchestrator.set_space_capacity(config_dict['Global_SpaceCapacity'])
    else:
        print("⚠️  未找到空间容量配置")
    
    # 生成仿真日期范围
    sim_dates = pd.date_range(start_date, end_date, freq='D')
    print(f"📅 仿真日期范围: {len(sim_dates)} 天")
    
    # 每日循环执行
    all_results = {
        'module1': [],
        'module3': [],
        'module4': [], 
        'module5': [],
        'module6': []
    }
    
    for i, current_date in enumerate(sim_dates, 1):
        print(f"\n{'='*20} 第 {i}/{len(sim_dates)} 天: {current_date.strftime('%Y-%m-%d')} {'='*20}")
        
        # ==================== 每日开始：GR入库处理 ====================
        try:
            print(f"\n🌅 每日开始状态更新")
            # 🔄 第0步：保存期初库存快照（在任何变动之前）
            print(f"  💾 保存期初库存快照...")
            orchestrator.save_beginning_inventory(current_date.strftime('%Y-%m-%d'))

            # 🔄 第1步：处理当日到达的delivery GR (in-transit → inventory)
            print(f"  📦 处理当日delivery GR到达...")
            orchestrator._process_delivery_arrivals(current_date.strftime('%Y-%m-%d'))
            
            # 🔄 第2步：处理历史生产的当日入库 (historical production → inventory)
            print(f"  🏭 处理历史生产当日入库...")
            historical_production = load_all_historical_production_plans(
                module4_output_dir=str(module_outputs['module4']),
                current_date=current_date,
                start_date=pd.to_datetime(start_date)
            )
            
            if not historical_production.empty:
                print(f"    📦 当日需要入库的历史生产: {len(historical_production)} 条记录")
                orchestrator.process_module4_production(historical_production, current_date.strftime('%Y-%m-%d'))
            else:
                print(f"    📦 当日无历史生产入库")
                
        except Exception as e:
            print(f"  ❌ 每日开始处理失败: {e}")
        
        # ==================== 模块运行序列 ====================
        # 初始化每日数据变量
        m1_shipments = pd.DataFrame()
        m4_production = pd.DataFrame()
        m5_deployment_df = pd.DataFrame()
        m6_delivery_df = pd.DataFrame()
        
        try:
            # ========== M1: 订单生成 + 立即库存扣减 ==========
            print(f"\n1️⃣ 运行 Module1 - 订单生成")
            try:
                m1_result = module1.run_daily_order_generation(
                    config_dict=config_dict,
                    simulation_date=current_date,
                    output_dir=str(module_outputs['module1']),
                    orchestrator=orchestrator
                )
                m1_shipments = m1_result.get('shipment_df', pd.DataFrame())
                
                # 🔄 立即处理M1 shipment，扣减库存
                if not m1_shipments.empty:
                    print(f"    🚚 立即处理M1 shipment，扣减库存...")
                    orchestrator.process_module1_shipments(m1_shipments, current_date.strftime('%Y-%m-%d'))
                    print(f"    ✅ 已扣减 {len(m1_shipments)} 个shipment的库存")
                
                print(f"  ✅ Module1 完成 - 生成 {len(m1_result.get('orders_df', []))} 个订单, {len(m1_shipments)} 个发货")
                all_results['module1'].append(m1_result)
            except Exception as e:
                print(f"  ❌ Module1 失败: {e}")
                m1_shipments = pd.DataFrame()  # 失败时使用空数据
                # 不用continue，让后面的模块继续执行
            
            # ========== M4: 生产计划 + 立即当日生产入库 ==========
            print(f"\n2️⃣ 运行 Module4 - 生产计划")
            try:
                # 使用集成模式直接调用 Module4 (改进的解决方案)
                m4_production = run_module4_integrated(
                    config_dict=config_dict,
                    module3_output_dir=str(module_outputs['module3']),
                    simulation_date=current_date,
                    simulation_start=pd.to_datetime(start_date),
                    output_dir=str(module_outputs['module4'])
                )
                
                # 🔄 立即处理M4当日生产入库
                if not m4_production.empty:
                    # 筛选当日可用的生产 (available_date = current_date)
                    daily_available = m4_production[
                        pd.to_datetime(m4_production['available_date']).dt.normalize() == current_date.normalize()
                    ]
                    
                    if not daily_available.empty:
                        print(f"    🏭 立即处理M4当日生产入库...")
                        orchestrator.process_module4_production(daily_available, current_date.strftime('%Y-%m-%d'))
                        print(f"    ✅ 已入库 {len(daily_available)} 条当日生产")
                    else:
                        print(f"    📦 M4当日无可用生产入库")
                
                print(f"  ✅ Module4 完成 - 生成生产计划: {len(m4_production)} 条记录")
                all_results['module4'].append({'production_df': m4_production})
            except Exception as e:
                print(f"  ❌ Module4 失败: {e}")
                m4_production = pd.DataFrame()  # 失败时使用空数据
            
            # ========== M5: 部署计划 ==========
            print(f"\n3️⃣ 运行 Module5 - 部署计划")
            try:
                m5_result = module5.main(
                    # 集成模式参数
                    config_dict=config_dict,
                    module1_output_dir=str(module_outputs['module1']),
                    module4_output_path=str(module_outputs['module4'] / f"Module4Output_{current_date.strftime('%Y%m%d')}.xlsx"),
                    orchestrator=orchestrator,
                    current_date=current_date.strftime('%Y-%m-%d'),
                    # 输出路径
                    output_path=str(module_outputs['module5'] / f"Module5Output_{current_date.strftime('%Y%m%d')}.xlsx")
                )
                
                # 获取部署计划数据
                if m5_result and 'deployment_plan' in m5_result:
                    deployment_plan_df = m5_result['deployment_plan']
                    print(f"    🔍 Module5返回的部署计划: {len(deployment_plan_df)} 条记录")
                    
                    if not deployment_plan_df.empty:
                        print(f"    📊 部署计划示例数据:")
                        print(f"    列名: {list(deployment_plan_df.columns)}")
                        if len(deployment_plan_df) > 0:
                            first_row = deployment_plan_df.iloc[0]
                            print(f"    第一行数据: {dict(first_row)}")
                            if 'deployed_qty_invCon' in deployment_plan_df.columns:
                                qty_stats = deployment_plan_df['deployed_qty_invCon'].describe()
                                print(f"    deployed_qty_invCon统计: {qty_stats}")
                        
                        # 过滤出有实际部署量的计划，排除自循环（sending=receiving）
                        valid_deployment = deployment_plan_df[
                            (deployment_plan_df['deployed_qty_invCon'] > 0) & 
                            (deployment_plan_df['deployed_qty_invCon'].notna()) &
                            (deployment_plan_df['sending'] != deployment_plan_df['receiving'])  # 排除自循环
                        ].copy()
                        
                        print(f"    🎯 有效部署计划: {len(valid_deployment)}/{len(deployment_plan_df)} 条")
                        
                        if not valid_deployment.empty:
                            # 检查是否已有deployed_qty列，避免重复
                            if 'deployed_qty' in valid_deployment.columns:
                                # 如果已有deployed_qty列，直接使用
                                m5_deployment_df = valid_deployment[[
                                    'material', 'sending', 'receiving', 'date', 'deployed_qty', 'demand_element'
                                ]].rename(columns={'date': 'planned_deployment_date'})
                            else:
                                # 重命名列以匹配orchestrator期望的格式
                                m5_deployment_df = valid_deployment.rename(columns={
                                    'date': 'planned_deployment_date',
                                    'deployed_qty_invCon': 'deployed_qty'
                                })[['material', 'sending', 'receiving', 'planned_deployment_date', 'deployed_qty', 'demand_element']]
                            
                            print(f"    ✅ 最终传递给Orchestrator的数据: {len(m5_deployment_df)} 条")
                            if len(m5_deployment_df) > 0:
                                final_qty_stats = m5_deployment_df['deployed_qty'].describe()
                                print(f"    deployed_qty统计: {final_qty_stats}")
                            
                            # 🔄 立即处理M5 deployment，更新open deployment
                            print(f"    📦 立即处理M5 deployment，更新open deployment...")
                            orchestrator.process_module5_deployment(m5_deployment_df, current_date.strftime('%Y-%m-%d'))
                            print(f"    ✅ 已更新 {len(m5_deployment_df)} 条部署计划到open deployment")
                            
                            print(f"  ✅ Module5 完成 - 生成 {len(m5_deployment_df)} 条有效部署计划")
                        else:
                            print(f"  ✅ Module5 完成 - 无有效部署计划")
                    else:
                        print(f"  ✅ Module5 完成 - 部署计划为空")
                else:
                    print(f"  ✅ Module5 完成 - 无返回结果")
                
                all_results['module5'].append(m5_result)
            except Exception as e:
                print(f"  ❌ Module5 失败: {e}")
                # 不用continue，让后面的模块继续执行
                m5_deployment_df = pd.DataFrame()  # 设置默认值
            
            # ========== M6: 物流执行 + 立即多状态更新 ==========
            print(f"\n4️⃣ 运行 Module6 - 物流执行")
            try:
                m6_result = module6.run_daily_physical_flow(
                    config_dict=config_dict,
                    orchestrator=orchestrator,
                    current_date=current_date,
                    output_dir=str(module_outputs['module6']),
                    max_wait_days=7,
                    random_seed=config_dict.get('M6_RandomSeed', 42)  # 使用统一种子
                )
                
                # 获取交付计划数据
                if m6_result and 'delivery_plan' in m6_result:
                    m6_delivery_df = m6_result.get('delivery_plan', pd.DataFrame())
                    
                    # 🔄 立即处理M6 delivery，更新多个状态
                    if not m6_delivery_df.empty:
                        print(f"    🚛 立即处理M6 delivery，更新库存/open deployment/in-transit...")
                        orchestrator.process_module6_delivery(m6_delivery_df, current_date.strftime('%Y-%m-%d'))
                        print(f"    ✅ 已处理 {len(m6_delivery_df)} 条delivery计划，更新相关状态")
                    
                    print(f"  ✅ Module6 完成 - 生成 {len(m6_delivery_df)} 条交付计划")
                else:
                    print(f"  ✅ Module6 完成 - 无交付计划")
                    m6_delivery_df = pd.DataFrame()
                
                all_results['module6'].append(m6_result)
            except Exception as e:
                print(f"  ❌ Module6 失败: {e}")
                # 不用continue，让后面的模块继续执行
                m6_delivery_df = pd.DataFrame()  # 设置默认值
            
            # ========== M3: 净需求计算 ==========
            print(f"\n5️⃣ 运行 Module3 - 净需求计算")
            try:
                m3_result = module3.run_integrated_mode(
                    module1_output_dir=str(module_outputs['module1']),
                    orchestrator=orchestrator,
                    config_dict=config_dict,
                    start_date=current_date.strftime('%Y-%m-%d'),
                    end_date=current_date.strftime('%Y-%m-%d'),
                    output_dir=str(module_outputs['module3'])
                )
                print(f"  ✅ Module3 完成")
                all_results['module3'].append(m3_result)
            except Exception as e:
                print(f"  ❌ Module3 失败: {e}")
                import traceback
                traceback.print_exc()
                # 不用continue，让他继续执行
            
            # ==================== 每日结束：保存状态 ====================
            print(f"\n💾 每日结束状态保存")
            try:
                # 保存期末库存快照（在保存状态之前）
                orchestrator.save_ending_inventory(current_date.strftime('%Y-%m-%d'))
                
                # 输出详细的库存变动记录用于调试
                orchestrator.output_daily_inventory_summary(current_date.strftime('%Y-%m-%d'))
                
                # 直接保存每日状态，状态更新已在各模块运行后实时完成
                orchestrator.save_daily_state(current_date.strftime('%Y-%m-%d'))
                
                # 获取当日统计
                stats = orchestrator.get_summary_statistics(current_date.strftime('%Y-%m-%d'))
                print(f"  📊 当日统计: {stats}")
                print(f"  💾 Orchestrator 状态已保存")
                
            except Exception as e:
                print(f"  ❌ 每日状态保存失败: {e}")
                # 不用continue，让他继续到下一天
            
            print(f"✅ 第 {i} 天处理完成")
            
        except Exception as e:
            print(f"❌ 第 {i} 天处理失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 生成最终报告
    print(f"\n📊 仿真完成报告")
    print("=" * 60)
    print(f"仿真期间: {start_date} 到 {end_date} ({len(sim_dates)} 天)")
    print(f"输出目录: {output_base_dir}")
    
    for module_name, results in all_results.items():
        print(f"{module_name.upper()}: {len(results)} 天成功处理")
    
    # 进行库存平衡检查
    print(f"\n🔎 正在进行库存平衡检查...")
    validation_manager = ValidationManager(str(output_dir))
    inventory_checker = InventoryBalanceChecker(validation_manager, orchestrator)
    balance_passed = inventory_checker.validate_inventory_consistency(start_date, end_date)
    
    if balance_passed:
        print("✅ 库存平衡检查通过")
    else:
        print("⚠️  库存平衡检查发现问题，请查看验证报告")
    
    # 生成汇总报告
    print(f"\n📊 正在生成汇总报告...")
    summary_generator = SummaryReportGenerator(str(output_dir))
    summary_reports = summary_generator.generate_all_reports(start_date, end_date)
    
    # 写入库存平衡检查报告
    balance_report_path = validation_manager.write_report()
    
    # 获取最终Orchestrator统计
    final_date = sim_dates[-1].strftime('%Y-%m-%d')
    final_stats = orchestrator.get_summary_statistics(final_date)
    print(f"\n🎯 最终Orchestrator状态:")
    for key, value in final_stats.items():
        print(f"  {key}: {value}")
    
    print(f"\n🎉 集成仿真完成!")
    
    return {
        'validation_passed': True,
        'simulation_completed': True,
        'dates_processed': len(sim_dates),
        'results': all_results,
        'final_stats': final_stats,
        'output_directory': output_base_dir,
        'validation_report': validation_report,
        'balance_check_passed': balance_passed,
        'balance_report': balance_report_path,
        'summary_reports': summary_reports
    }

def main():
    """主函数 - 独立执行集成仿真"""
    # 配置文件路径（可以通过命令行参数或环境变量指定）
    import argparse
    
    parser = argparse.ArgumentParser(description="运行供应链集成仿真")
    parser.add_argument("--config", "-c", 
                       default="./config/integration_config.json",
                       help="配置文件路径 (默认: ./config/integration_config.json)")
    parser.add_argument("--start-date", "-s", 
                       default="2024-01-01",
                       help="仿真开始日期 (默认: 2024-01-01)")
    parser.add_argument("--end-date", "-e", 
                       default="2024-01-05",
                       help="仿真结束日期 (默认: 2024-01-03)")
    parser.add_argument("--output", "-o", 
                       default=None,
                       help="输出目录 (默认: 根据配置文件名生成)")
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"❌ 配置文件不存在: {args.config}")
        print("请提供有效的配置文件路径，或使用测试脚本生成配置")
        sys.exit(1)
    
    # 如果没有指定输出目录，根据配置文件名生成
    if args.output is None:
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        args.output = f"./{config_name}_output"
        print(f"💫 使用默认输出目录: {args.output}")
    
    try:
        result = run_integrated_simulation(
            config_path=args.config,
            start_date=args.start_date,
            end_date=args.end_date,
            output_base_dir=args.output
        )
        
        print(f"\n✅ 仿真结果:")
        print(f"  处理天数: {result.get('dates_processed', 0)}")
        print(f"  输出目录: {result.get('output_directory', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ 集成仿真失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()