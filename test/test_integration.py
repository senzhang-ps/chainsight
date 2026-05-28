"""文件模式集成仿真入口（自定义版本，仅 Module1）。

职责：
- 在本地文件模式下组织每日执行与结果落盘流程。
- 复用 Orchestrator 与 Module1，保持与主集成链路一致的结果口径。
"""

import pandas as pd
import time
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from src.core.orchestrator import create_orchestrator, Orch
from src.utils.time_manager import initialize_time_manager
from src.modules import module1
from src.modules.demand_planning.integration_refactor import ModuleOne
from src.utils.normalization import normalize_identifiers
from src.core.main_integration.seed import set_module_seeds
from src.core.main_integration.production_runner import load_current_date_production_gr
from src.core.main_integration.config_loader import load_configuration

logger = logging.getLogger("SupplyChainSimulation")


def run_integrated_simulation(
    config_path: str,
    start_date: str,
    end_date: str,
    output_base_dir: str = "./integrated_output",
    force_restart: bool = False,
):
    actual_start_date = start_date
    is_resuming = False
    resume_info = None

    time_manager = initialize_time_manager(actual_start_date)

    # 创建输出目录
    output_dir = Path(output_base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    orchestrator_output_dir = output_dir / "orchestrator"
    module_outputs = {
        'module1': output_dir / "module1",
        'module4': output_dir / "module4",
    }

    for module_dir in module_outputs.values():
        module_dir.mkdir(parents=True, exist_ok=True)

    # 加载配置
    config_dict = load_configuration(config_path)

    # 设置全局随机种子
    set_module_seeds(config_dict)

    # 初始化Orchestrator
    logger.info("🎯 初始化Orchestrator")
    orchestrator = create_orchestrator(
        start_date=start_date,
        output_dir=str(orchestrator_output_dir)
    )
    orchestrator.set_past_due_cleanup_grace_days(100)

    # 设置初始库存
    if 'M1_InitialInventory' in config_dict and not config_dict['M1_InitialInventory'].empty:
        orchestrator.initialize_inventory(config_dict['M1_InitialInventory'])
    else:
        logger.warning("⚠️ 未找到初始库存配置，使用空库存")
        orchestrator.initialize_inventory(pd.DataFrame(columns=['material', 'location', 'quantity']))

    # 设置空间容量
    if 'Global_SpaceCapacity' in config_dict and not config_dict['Global_SpaceCapacity'].empty:
        orchestrator.set_space_capacity(config_dict['Global_SpaceCapacity'])

    orch = Orch(start_date = start_date, end_date = end_date, config_path = 'None', output_path = output_base_dir, config_dict=config_dict)
    orch.load_datas('M1')

    # 生成仿真日期范围
    sim_dates = pd.date_range(actual_start_date, end_date, freq='D')
    logger.info(f"📅 仿真日期范围: {len(sim_dates)} 天")

    all_results = {'module1': []}

    simulation_start_time = time.time()
    simulation_start_datetime = datetime.now()

    pbar = tqdm(sim_dates, total=len(sim_dates), desc='仿真进度', unit='天', ncols=80, dynamic_ncols=False, leave=True)
    for i, current_date in enumerate(pbar, 1):
        progress_info = f"第 {i}/{len(sim_dates)} 天"

        logger.info(f"{'=' * 20} {progress_info}: {current_date.strftime('%Y-%m-%d')} {'=' * 20}")
        pbar.set_postfix(date=current_date.strftime('%Y-%m-%d'), day=progress_info)

        # ==================== 每日开始：GR入库处理 ====================
        try:
            logger.info("🌅 每日开始状态更新")

            logger.info("💾 保存期初库存快照...")
            orchestrator.save_beginning_inventory(current_date.strftime('%Y-%m-%d'))
            orchestrator.cleanup_past_due_open_deployments(
                current_date.strftime('%Y-%m-%d'),
                grace_days=getattr(orchestrator, "cleanup_grace_days", 0),
                write_audit=True,
            )

            logger.info("📦 处理当日delivery GR到达...")
            orchestrator._process_delivery_arrivals(current_date.strftime('%Y-%m-%d'))

            logger.info("🏭 处理历史生产当日入库...")
            current_date_production_gr = load_current_date_production_gr(
                module4_output_dir=str(module_outputs['module4']),
                current_date=current_date,
                start_date=pd.to_datetime(start_date),
            )

            if not current_date_production_gr.empty:
                logger.info(f"📦 当日需要入库的历史生产: {len(current_date_production_gr)} 条记录")
                current_date_production_gr_normalized = normalize_identifiers(current_date_production_gr)
                orchestrator.process_module4_production(
                    current_date_production_gr_normalized,
                    current_date.strftime('%Y-%m-%d'),
                )
            else:
                logger.info("📦 当日无历史生产入库")

        except Exception as e:
            logger.error(f"❌ 每日开始处理失败: {e}")

        # ==================== M1: 订单生成 + 立即库存扣减 ====================
        m1_shipments = pd.DataFrame()

        logger.info("1️⃣ 运行 Module1 - 订单生成")
        try:
            m1_result = module1.run_daily_order_generation(
                    config_dict=config_dict,
                    simulation_date=current_date,
                    output_dir=str(module_outputs['module1']),
                    orchestrator=orchestrator
                )
                
            m1_shipments = m1_result.get('shipment_df', pd.DataFrame())
            
            if not m1_shipments.empty:
                logger.info("🚚 立即处理M1 shipment，扣减库存...")
                m1_shipments_normalized = normalize_identifiers(m1_shipments)
                orchestrator.process_module1_shipments(
                    m1_shipments_normalized,
                    current_date.strftime('%Y-%m-%d'),
                )
                logger.info(f"✅ 已扣减 {len(m1_shipments_normalized)} 个shipment的库存")

            logger.info(
                f"✅ Module1 完成 - 生成 {len(m1_result.get('orders_df', []))} 个订单, "
                f"{len(m1_shipments)} 个发货"
            )
            m1_result['simulation_date'] = current_date
            all_results['module1'].append(m1_result)

        except Exception as e:
            logger.error(f"❌ Module1 失败: {e}")

        # ==================== 每日结束：保存状态 ====================
        try:
            orchestrator.save_ending_inventory(current_date.strftime('%Y-%m-%d'))
            orchestrator.output_daily_inventory_summary(current_date.strftime('%Y-%m-%d'))
            orchestrator.save_daily_state(current_date.strftime('%Y-%m-%d'))
            stats = orchestrator.get_summary_statistics(current_date.strftime('%Y-%m-%d'))
            logger.info(f"📊 当日统计: {stats}")
        except Exception as e:
            logger.error(f"❌ 每日状态保存失败: {e}")

        logger.info(f"✅ 第 {i} 天处理完成")

    # 仿真完成报告
    simulation_end_time = time.time()
    total_runtime_seconds = simulation_end_time - simulation_start_time
    hours, remainder = divmod(total_runtime_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours >= 1:
        runtime_str = f"{int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒"
    elif minutes >= 1:
        runtime_str = f"{int(minutes)}分钟 {seconds:.2f}秒"
    else:
        runtime_str = f"{seconds:.2f}秒"

    logger.info("🎉 集成仿真完成!")
    logger.info(f"总共处理: {len(sim_dates)} 天")
    logger.info(f"⏱️  总运行时间: {runtime_str}")

    return {
        'simulation_completed': True,
        'dates_processed': len(sim_dates),
        'output_directory': output_base_dir,
        'results': all_results,
    }


def run_integrated_simulation_refactor(
    config_path: str,
    start_date: str,
    end_date: str,
    output_base_dir: str = "./integrated_output",
):
    start_date
    orch = Orch(start_date = start_date, end_date = end_date, config_path = config_path, output_path = output_base_dir)
    orch.load_datas('M1')

    # 初始化Orchestrator
    logger.info("🎯 初始化Orchestrator")
    orchestrator = create_orchestrator(
        start_date=start_date,
        output_dir=str(Path(orch.output_path+"/orchestrator"))
    )
    orchestrator.set_past_due_cleanup_grace_days(100)
    config_dict = orch.all_config

    # 设置初始库存
    if 'M1_InitialInventory' in config_dict and not config_dict['M1_InitialInventory'].empty:
        orchestrator.initialize_inventory(config_dict['M1_InitialInventory'])
    else:
        logger.warning("⚠️ 未找到初始库存配置，使用空库存")
        orchestrator.initialize_inventory(pd.DataFrame(columns=['material', 'location', 'quantity']))

    # 设置空间容量
    if 'Global_SpaceCapacity' in config_dict and not config_dict['Global_SpaceCapacity'].empty:
        orchestrator.set_space_capacity(config_dict['Global_SpaceCapacity'])

    orch.all_results['module1'] = []

    simulation_start_time = time.time()
    m1 = module1.ModuleOne(
        simulation_date=str(start_date),
        output_dir=orch.get_output('module1'),
        orchestrator=orchestrator,
        orch=orch,
        engine='polars',
    )
    m1.prepare()

    for i, current_date in orch.iter_dates():
        progress_info = f"第 {i}/{len(orch.sim_dates)} 天"

        

        # ==================== 每日开始：GR入库处理 ====================
        try:
            logger.info("🌅 每日开始状态更新")

            logger.info("💾 保存期初库存快照...")
            orchestrator.save_beginning_inventory(current_date.strftime('%Y-%m-%d'))
            orchestrator.cleanup_past_due_open_deployments(
                current_date.strftime('%Y-%m-%d'),
                grace_days=getattr(orchestrator, "cleanup_grace_days", 0),
                write_audit=True,
            )

            logger.info("📦 处理当日delivery GR到达...")
            orchestrator._process_delivery_arrivals(current_date.strftime('%Y-%m-%d'))

            logger.info("🏭 处理历史生产当日入库...")
            current_date_production_gr = load_current_date_production_gr(
                module4_output_dir=orch.get_output('module4'),
                current_date=current_date,
                start_date=pd.to_datetime(start_date),
            )

            if not current_date_production_gr.empty:
                logger.info(f"📦 当日需要入库的历史生产: {len(current_date_production_gr)} 条记录")
                current_date_production_gr_normalized = normalize_identifiers(current_date_production_gr)
                orchestrator.process_module4_production(
                    current_date_production_gr_normalized,
                    current_date.strftime('%Y-%m-%d'),
                )
            else:
                logger.info("📦 当日无历史生产入库")

        except Exception as e:
            logger.error(f"❌ 每日开始处理失败: {e}")

        # ==================== M1: 订单生成 + 立即库存扣减 ====================
        m1_shipments = pd.DataFrame()

        logger.info("1️⃣ 运行 Module1 - 订单生成")
        try:
            # m1_result = module1.run_daily_order_generation_refactor(
            #     simulation_date='2025-12-15',
            #     output_dir=orch.get_output('module1'),
            #     orchestrator=orchestrator,
            #     orch=orch,engine = 'polars',verbose=True
            # )
            m1.simulation_date = current_date
            m1.run()
            m1_result = m1.output()
            # orch.save(m1)
            # m1_result = m1.output()

            m1_shipments = m1_result.get('shipment_df', pd.DataFrame())

            if not m1_shipments.empty:
                logger.info("🚚 立即处理M1 shipment，扣减库存...")
                m1_shipments_normalized = normalize_identifiers(m1_shipments)
                orchestrator.process_module1_shipments(
                    m1_shipments_normalized,
                    current_date.strftime('%Y-%m-%d'),
                )
                logger.info(f"✅ 已扣减 {len(m1_shipments_normalized)} 个shipment的库存")

            logger.info(
                f"✅ Module1 完成 - 生成 {len(m1_result.get('orders_df', []))} 个订单, "
                f"{len(m1_shipments)} 个发货"
            )
            m1_result['simulation_date'] = current_date
            orch.all_results['module1'].append(m1_result)

        except Exception as e:
            logger.error(f"❌ Module1 失败: {e}")

        # ==================== 每日结束：保存状态 ====================
        try:
            orchestrator.save_ending_inventory(current_date.strftime('%Y-%m-%d'))
            orchestrator.output_daily_inventory_summary(current_date.strftime('%Y-%m-%d'))
            orchestrator.save_daily_state(current_date.strftime('%Y-%m-%d'))
            stats = orchestrator.get_summary_statistics(current_date.strftime('%Y-%m-%d'))
            logger.info(f"📊 当日统计: {stats}")
        except Exception as e:
            logger.error(f"❌ 每日状态保存失败: {e}")

        logger.info(f"✅ 第 {i} 天处理完成")

    # 仿真完成报告
    simulation_end_time = time.time()
    total_runtime_seconds = simulation_end_time - simulation_start_time
    hours, remainder = divmod(total_runtime_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours >= 1:
        runtime_str = f"{int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒"
    elif minutes >= 1:
        runtime_str = f"{int(minutes)}分钟 {seconds:.2f}秒"
    else:
        runtime_str = f"{seconds:.2f}秒"

    logger.info("🎉 集成仿真完成!")
    logger.info(f"总共处理: {len(orch.sim_dates)} 天")
    logger.info(f"⏱️  总运行时间: {runtime_str}")

    return {
        'simulation_completed': True,
        'dates_processed': len(orch.sim_dates),
        'output_directory': output_base_dir,
        'results': orch.all_results,
    }