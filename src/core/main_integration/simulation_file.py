"""文件模式集成仿真入口。

职责：
- 在本地文件模式下组织完整的预校验、续跑、按日执行与结果落盘流程。
- 复用 Orchestrator 与各业务模块，保持与主集成链路一致的结果口径。
- 为本地输出、人工排查与回归对比提供标准执行入口。
"""

import pandas as pd
import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from .. import orchestrator
from ..orchestrator import create_orchestrator
from ...utils.validation_manager import ValidationManager
from ...utils.time_manager import initialize_time_manager
from ...utils.config_validator import run_pre_simulation_validation
from ...utils.inventory_balance_checker import InventoryBalanceChecker
from ...services.summary_report_generator import SummaryReportGenerator
from ...services.performance_profiler import PerformanceProfiler
from ...modules import module1, module3, module4, module5, module6

from .normalize import _normalize_identifiers
from .resume import check_resume_capability, restore_orchestrator_state
from .seed import set_module_seeds
from .module4_runner import run_module4_integrated, load_current_date_production_gr
from .config_loader import load_configuration


def run_integrated_simulation(
    config_path: str,
    start_date: str,
    end_date: str,
    output_base_dir: str = "./integrated_output",
    force_restart: bool = False
):
    """运行完整的集成仿真

    目的：
    - 统一编排预验证、续跑检测、Orchestrator 初始化、每日模块执行与状态保存，最终生成汇总与验证报告。

    Args:
        config_path: 配置文件路径（Excel）。
        start_date: 仿真开始日期 (YYYY-MM-DD)。
        end_date: 仿真结束日期 (YYYY-MM-DD)。
        output_base_dir: 输出基础目录。
        force_restart: 强制从头开始（忽略续跑）。

    Returns:
        dict: 包含仿真执行状态、统计与输出路径等的结果字典。

    输入数据：
        - 配置文件、历史输出目录（用于续跑）、各模块的运行所需表。

    输出/副作用：
        - 创建/写入每日输出与日志、保存 Orchestrator 状态、生成最终汇总与验证报告。

    逻辑：
        - 预验证→续跑判断→初始化（新建或恢复）→按日执行模块→每日保存→最终报告与检查→返回结果。
    """
    import time
    from datetime import datetime
    simulation_start_time = time.time()
    simulation_start_datetime = datetime.now()
    
    print("\n" + "=" * 60)
    print("🕐 程序时间信息")
    print("=" * 60)
    print(f"📅 程序开始时间: {simulation_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
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
    
    # 2. 检查续跑能力
    actual_start_date = start_date
    is_resuming = False
    resume_info = None
    
    if force_restart:
        print(f"🔄 强制重启模式：忽略任何现有状态，从头开始")
    else:
        resume_info = check_resume_capability(output_base_dir, start_date, end_date)
        
        if resume_info.get('already_completed', False):
            print(f"🎉 仿真已完成！最后处理日期: {resume_info['last_complete_date']}")
            print(f"   总共处理了 {resume_info['days_completed']} 天")
            return {
                'validation_passed': True,
                'simulation_completed': True,
                'already_completed': True,
                'dates_processed': resume_info['days_completed'],
                'last_complete_date': resume_info['last_complete_date']
            }
        elif resume_info['can_resume']:
            print(f"🔄 检测到未完成的仿真，支持续跑:")
            print(f"   已完成: {resume_info['days_completed']} 天 (到 {resume_info['last_complete_date']})")
            print(f"   剩余: {resume_info['days_remaining']} 天 (从 {resume_info['resume_from_date']} 开始)")
            
            # 提供选择（在实际实现中可以加入用户确认）
            print(f"   ✅ 将从 {resume_info['resume_from_date']} 继续运行")
            actual_start_date = resume_info['resume_from_date'] 
            is_resuming = True
        else:
            print(f"📝 未发现可续跑的状态，将从头开始")
    
    # 3. 初始化时间管理器
    time_manager = initialize_time_manager(actual_start_date)
    
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
    print("🎯 初始化Orchestrator")
    orchestrator = create_orchestrator(
        start_date=start_date,
        output_dir=str(orchestrator_output_dir)
    )
    # 设置 open deployment 的清理天数，3代表保留3天
    orchestrator.set_past_due_cleanup_grace_days(100)
    if is_resuming:
        # 续跑模式：恢复状态
        print("🔄 续跑模式：恢复Orchestrator状态")
        restore_orchestrator_state(orchestrator, resume_info['last_complete_date'], output_base_dir)
        
        # 设置空间容量（续跑时也需要重新设置空间容量配置）
        if 'Global_SpaceCapacity' in config_dict and not config_dict['Global_SpaceCapacity'].empty:
            orchestrator.set_space_capacity(config_dict['Global_SpaceCapacity'])
    else:
        # 全新开始：设置初始状态
        print("🆕全新开始：设置初始状态")
        
        # 设置初始库存
        if 'M1_InitialInventory' in config_dict and not config_dict['M1_InitialInventory'].empty:
            orchestrator.initialize_inventory(config_dict['M1_InitialInventory'])
        else:
            print("⚠️未找到初始库存配置，使用空库存")
            orchestrator.initialize_inventory(pd.DataFrame(columns=['material', 'location', 'quantity']))
        
        # 设置空间容量
        if 'Global_SpaceCapacity' in config_dict and not config_dict['Global_SpaceCapacity'].empty:
            orchestrator.set_space_capacity(config_dict['Global_SpaceCapacity'])
        else:
            print("⚠️未找到空间容量配置")
    
    # 生成仿真日期范围（使用实际开始日期）
    sim_dates = pd.date_range(actual_start_date, end_date, freq='D')
    total_days = len(pd.date_range(start_date, end_date, freq='D'))
    
    if is_resuming:
        print(f"📅 续跑日期范围: {len(sim_dates)} 天 (剩余)")
        print(f"   原始总天数: {total_days}")
        print(f"   已完成: {resume_info['days_completed']} 天")
        print(f"   剩余处理: {len(sim_dates)} 天")
    else:
        print(f"📅 仿真日期范围: {len(sim_dates)} 天")
    
    # 每日循环执行
    all_results = {
        'module1': [],
        'module3': [],
        'module4': [], 
        'module5': [],
        'module6': []
    }
    
    pbar = tqdm(sim_dates, total=len(sim_dates), desc='仿真进度', unit='天', ncols=80, dynamic_ncols=False, leave=True)
    for i, current_date in enumerate(pbar, 1):
        # 计算实际的总进度（考虑续跑情况）
        if is_resuming:
            actual_day_number = resume_info['days_completed'] + i
            total_original_days = total_days
            progress_info = f"第 {actual_day_number}/{total_original_days} 天 (续跑第 {i}/{len(sim_dates)} 天)"
        else:
            progress_info = f"第 {i}/{len(sim_dates)} 天"
            
        print(f"{'='*20} {progress_info}: {current_date.strftime('%Y-%m-%d')} {'='*20}")
        pbar.set_postfix(date=current_date.strftime('%Y-%m-%d'), day=progress_info)
        
        # 🎲 注意：不在每日开始时重置种子，以匹配ChainSight_Dev的随机数行为
        # ChainSight_Dev没有每日种子重置，随机状态自然演变
        # 全局种子只在仿真开始时设置一次 (在set_module_seeds中)
        
        # ==================== 每日开始：GR入库处理 ====================
        try:
            print("🌅 每日开始状态更新")

            # 🔄 第0步：保存期初库存快照（在任何变动之前）
            print("💾保存期初库存快照...")
            orchestrator.save_beginning_inventory(current_date.strftime('%Y-%m-%d'))
            orchestrator.cleanup_past_due_open_deployments(current_date.strftime('%Y-%m-%d'),grace_days=getattr(orchestrator, "cleanup_grace_days", 0),write_audit=True)

            # 🔄 第1步：处理当日到达的delivery GR (in-transit → inventory)
            print("📦处理当日delivery GR到达...")
            orchestrator._process_delivery_arrivals(current_date.strftime('%Y-%m-%d'))
            
            # 🔄 第2步：处理历史生产的当日入库 (historical production → inventory)
            print("🏭处理历史生产当日入库...")
            current_date_production_gr = load_current_date_production_gr(
                module4_output_dir=str(module_outputs['module4']),
                current_date=current_date,
                start_date=pd.to_datetime(start_date)
            )
            
            if not current_date_production_gr.empty:
                print(f"📦当日需要入库的历史生产: {len(current_date_production_gr)} 条记录")
                # 🔧 标准化标识符字段，确保数据类型一致性
                current_date_production_gr_normalized = _normalize_identifiers(current_date_production_gr)
                orchestrator.process_module4_production(current_date_production_gr_normalized, current_date.strftime('%Y-%m-%d'))
            else:
                print("📦当日无历史生产入库")
                
        except Exception as e:
            print(f"❌ 每日开始处理失败: {e}")
        
        # ==================== 模块运行序列 ====================
        # 初始化每日数据变量
        m1_shipments = pd.DataFrame()
        m4_production = pd.DataFrame()
        m5_deployment_df = pd.DataFrame()
        m6_delivery_df = pd.DataFrame()
        
        try:
            # ========== M1: 订单生成 + 立即库存扣减 ==========
            print("1️⃣ 运行 Module1 - 订单生成")
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
                    print("🚚立即处理M1 shipment，扣减库存...")
                    # 🔧 标准化标识符字段，确保数据类型一致性
                    m1_shipments_normalized = _normalize_identifiers(m1_shipments)
                    orchestrator.process_module1_shipments(m1_shipments_normalized, current_date.strftime('%Y-%m-%d'))
                    print(f"✅ 已扣减 {len(m1_shipments_normalized)} 个shipment的库存")
                
                print(f"✅ Module1 完成 - 生成 {len(m1_result.get('orders_df', []))} 个订单, {len(m1_shipments)} 个发货")
                if m1_result is not None:
                    m1_result['simulation_date'] = current_date
                all_results['module1'].append(m1_result)
            except Exception as e:
                print(f"❌ Module1 失败: {e}")
                m1_shipments = pd.DataFrame()  # 失败时使用空数据
                # 不用continue，让后面的模块继续执行
            
            # ========== M4: 生产计划 + 立即当日生产入库 ==========
            print("2️⃣ 运行 Module4 - 生产计划")
            try:
                # 使用集成模式直接调用 Module4 (改进的解决方案)
                m4_result = run_module4_integrated(
                    config_dict=config_dict,
                    module3_output_dir=str(module_outputs['module3']),
                    simulation_date=current_date,
                    simulation_start=pd.to_datetime(start_date),
                    output_dir=str(module_outputs['module4'])
                )
                
                # 从返回结果中获取 production_df
                m4_production = m4_result.get('production_df', pd.DataFrame())
                
                # 🔄 简化调用：仅持久化“未来 available_date”的生产计划，避免重复当日GR
                if not m4_production.empty and 'available_date' in m4_production.columns:
                    m4_production['available_date'] = pd.to_datetime(m4_production['available_date'])
                    future_plans = m4_production[m4_production['available_date'].dt.normalize() > current_date.normalize()]
                    if not future_plans.empty:
                        print("🗂️ 持久化未来生产计划（不触发当日GR）...")
                        future_plans_normalized = _normalize_identifiers(future_plans)
                        orchestrator.process_module4_production(future_plans_normalized, current_date.strftime('%Y-%m-%d'))
                        print(f"✅已写入未来计划回补: {len(future_plans_normalized)} 条")
                    else:
                        print("📦当日无未来 available_date 的计划需要持久化")
                else:
                    print("📦M4当日未生成生产计划或缺少 available_date 列")
                
                print(f"✅ Module4 完成 - 生成生产计划: {len(m4_production)} 条记录")
                # 存储完整的Module4结果（包含所有输出表）
                m4_result['simulation_date'] = current_date
                all_results['module4'].append(m4_result)
            except Exception as e:
                print(f"❌ Module4 失败: {e}")
                m4_production = pd.DataFrame()  # 失败时使用空数据
            
            # ========== M5: 部署计划 ==========
            print("3️⃣ 运行 Module5 - 部署计划")
            try:
                # 启用性能分析
                with PerformanceProfiler("Module5", output_dir=Path(output_base_dir) / "performance", enabled=True):
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
                    # print(f"    🔍 Module5返回的部署计划: {len(deployment_plan_df)} 条记录")
                    
                    if not deployment_plan_df.empty:
                        # print(f"    📊 部署计划示例数据:")
                        # print(f"    列名: {list(deployment_plan_df.columns)}")
                        # if len(deployment_plan_df) > 0:
                        #     first_row = deployment_plan_df.iloc[0]
                        #     print(f"    第一行数据: {dict(first_row)}")
                        #     if 'deployed_qty_invCon' in deployment_plan_df.columns:
                        #         qty_stats = deployment_plan_df['deployed_qty_invCon'].describe()
                        #         print(f"    deployed_qty_invCon统计: {qty_stats}")
                        
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
                            
                            # 🔧 标准化标识符字段，确保数据类型一致性
                            m5_deployment_df = _normalize_identifiers(m5_deployment_df)
                            
                            # print(f"    ✅ 最终传递给Orchestrator的数据: {len(m5_deployment_df)} 条")
                            # if len(m5_deployment_df) > 0:
                            #     final_qty_stats = m5_deployment_df['deployed_qty'].describe()
                            #     print(f"    deployed_qty统计: {final_qty_stats}")
                            #     print(f"    数据类型: material={m5_deployment_df['material'].dtype}, sending={m5_deployment_df['sending'].dtype}, receiving={m5_deployment_df['receiving'].dtype}")
                            
                            # 🔄 立即处理M5 deployment，更新open deployment
                            print(f"\n    📦 立即处理M5 deployment，更新open deployment...")
                            orchestrator.process_module5_deployment(m5_deployment_df, current_date.strftime('%Y-%m-%d'))
                            print(f"    ✅ 已更新 {len(m5_deployment_df)} 条部署计划到open deployment")
                            
                            print(f"\n  ✅ Module5 完成 - 生成 {len(m5_deployment_df)} 条有效部署计划")
                        else:
                            print(f"\n  ✅ Module5 完成 - 无有效部署计划")
                    else:
                        print(f"\n  ✅ Module5 完成 - 部署计划为空")
                else:
                    print(f"\n  ✅ Module5 完成 - 无返回结果")
                
                if m5_result is not None:
                    m5_result['simulation_date'] = current_date
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
                    max_wait_days=30,
                    random_seed=config_dict.get('M6_RandomSeed', 42)  # 使用统一种子
                )
                
                # 获取交付计划数据
                if m6_result and 'delivery_plan' in m6_result:
                    m6_delivery_df = m6_result.get('delivery_plan', pd.DataFrame())
                    
                    # 🔄 立即处理M6 delivery，更新多个状态
                    if not m6_delivery_df.empty:
                        print(f"\n    🚛 立即处理M6 delivery，更新库存/open deployment/in-transit...")
                        # 🔧 标准化标识符字段，确保数据类型一致性
                        m6_delivery_df_normalized = _normalize_identifiers(m6_delivery_df)
                        orchestrator.process_module6_delivery(m6_delivery_df_normalized, current_date.strftime('%Y-%m-%d'))
                        print(f"    ✅ 已处理 {len(m6_delivery_df_normalized)} 条delivery计划，更新相关状态")
                    
                    print(f"\n  ✅ Module6 完成 - 生成 {len(m6_delivery_df)} 条交付计划")
                else:
                    print(f"\n  ✅ Module6 完成 - 无交付计划")
                    m6_delivery_df = pd.DataFrame()
                
                if m6_result is not None:
                    m6_result['simulation_date'] = current_date
                all_results['module6'].append(m6_result)
            except KeyError as ke:
                print(f"  ❌ Module6 失败: 缺少列 '{ke}' - {str(ke)}")
                logging.error(f"Module6 KeyError: {ke}", exc_info=True)
                # 不用continue，让后面的模块继续执行
                m6_delivery_df = pd.DataFrame()  # 设置默认值
            except Exception as e:
                print(f"  ❌ Module6 失败: {e}")
                logging.error(f"Module6 Exception: {e}", exc_info=True)
                # 不用continue，让后面的模块继续执行
                m6_delivery_df = pd.DataFrame()  # 设置默认值
            
            # ========== M3: 净需求计算 ==========
            print(f"\n5️⃣ 运行 Module3 - 净需求计算")
            try:
                # 启用性能分析
                with PerformanceProfiler("Module3", output_dir=Path(output_base_dir) / "performance", enabled=True):
                    m3_result = module3.run_integrated_mode(
                        module1_output_dir=str(module_outputs['module1']),
                        orchestrator=orchestrator,
                        config_dict=config_dict,
                        start_date=current_date.strftime('%Y-%m-%d'),
                        end_date=current_date.strftime('%Y-%m-%d'),
                        output_dir=str(module_outputs['module3'])
                    )
                print(f"  ✅ Module3 完成")
                if m3_result is not None:
                    m3_result['simulation_date'] = current_date
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
                print(f"📊 当日统计: {stats}")
                print("💾 Orchestrator 状态已保存")
                
            except Exception as e:
                print(f"❌ 每日状态保存失败: {e}")
                # 不用continue，让他继续到下一天
            
            print(f"✅ 第 {i} 天处理完成")
            
        except Exception as e:
            print(f"❌ 第 {i} 天处理失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 生成最终报告
    print("📊 仿真完成报告")
    print("=" * 60)
    print(f"仿真期间: {start_date} 到 {end_date} ({len(sim_dates)} 天)")
    print(f"输出目录: {output_base_dir}")
    
    for module_name, results in all_results.items():
        print(f"{module_name.upper()}: {len(results)} 天成功处理")
    
    # 进行库存平衡检查
    print("🔎 正在进行库存平衡检查...")
    validation_manager = ValidationManager(str(output_dir))
    inventory_checker = InventoryBalanceChecker(validation_manager, orchestrator)
    balance_passed = inventory_checker.validate_inventory_consistency(start_date, end_date)
    
    if balance_passed:
        print("✅ 库存平衡检查通过")
    else:
        print("⚠️库存平衡检查发现问题，请查看验证报告")
    
    # 生成汇总报告
    print("📊 正在生成汇总报告...")
    summary_generator = SummaryReportGenerator(str(output_dir), config_dict)
    summary_reports = summary_generator.generate_all_reports(start_date, end_date)
    
    # 写入库存平衡检查报告
    balance_report_path = validation_manager.write_report()
    
    # 获取最终Orchestrator统计
    final_date = sim_dates[-1].strftime('%Y-%m-%d')
    final_stats = orchestrator.get_summary_statistics(final_date)
    print("🎯 最终Orchestrator状态:")
    for key, value in final_stats.items():
        print(f"{key}: {value}")
    
    # 计算总运行时间
    simulation_end_time = time.time()
    total_runtime_seconds = simulation_end_time - simulation_start_time
    
    # 格式化运行时间
    hours, remainder = divmod(total_runtime_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours >= 1:
        runtime_str = f"{int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒"
    elif minutes >= 1:
        runtime_str = f"{int(minutes)}分钟 {seconds:.2f}秒"
    else:
        runtime_str = f"{seconds:.2f}秒"
    
    if is_resuming:
        total_processed = resume_info['days_completed'] + len(sim_dates)
        print("🎉 续跑仿真完成!")
        print(f"本次处理: {len(sim_dates)} 天")
        print(f"总共完成: {total_processed} 天")
    else:
        total_processed = len(sim_dates)
        print("🎉 集成仿真完成!")
        print(f"总共处理: {total_processed} 天")
    
    # 输出运行时间统计
    simulation_end_datetime = datetime.now()
    print("\n" + "=" * 60)
    print("🕐 程序时间统计")
    print("=" * 60)
    print(f"📅 开始时间: {simulation_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📅 结束时间: {simulation_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  总运行时间: {runtime_str}")
    print(f"📊 平均每天耗时: {total_runtime_seconds / len(sim_dates):.2f}秒")
    print("=" * 60)
    
    return {
        'validation_passed': True,
        'simulation_completed': True,
        'is_resuming': is_resuming,
        'dates_processed_this_run': len(sim_dates),
        'total_dates_processed': total_processed if is_resuming else len(sim_dates),
        'resume_info': resume_info if is_resuming else None,
        'results': all_results,
        'final_stats': final_stats,
        'output_directory': output_base_dir,
        'validation_report': validation_report,
        'balance_check_passed': balance_passed,
        'balance_report': balance_report_path,
        'summary_reports': summary_reports
    }

