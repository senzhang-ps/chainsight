"""
完整优化仿真示例 - 展示如何使用DuckDB优化层

本脚本演示:
1. 初始化优化引擎
2. 集成到仿真流程
3. 性能监控和报告
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

# 导入优化组件
from pgsql_db.optimized_simulation import (
    OptimizedSimulationRunner,
    create_optimized_runner,
    integrate_optimization_to_simulation
)
from pgsql_db.module_optimizers import (
    Module3Optimizer,
    Module5Optimizer,
    Module6Optimizer,
    create_all_optimizers
)
from pgsql_db.performance_dashboard import (
    PerformanceDashboard,
    RealTimeMonitor,
    get_dashboard,
    track_performance
)
from pgsql_db.incremental_processor import (
    IncrementalProcessor,
    create_incremental_processor,
    DeltaCalculator
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_optimized_simulation_example(
    config_path: str,
    start_date: str,
    end_date: str,
    output_dir: str = "./outputs/db_optimized",
    db_connection_string: str = None,
    enable_optimization: bool = True
):
    """
    运行优化仿真示例
    
    Args:
        config_path: 配置文件路径
        start_date: 开始日期
        end_date: 结束日期
        output_dir: 输出目录
        db_connection_string: PostgreSQL连接字符串
        enable_optimization: 是否启用优化
    """
    print("\n" + "=" * 70)
    print("🚀 ChainSight 优化仿真示例")
    print("=" * 70)
    print(f"配置文件: {config_path}")
    print(f"仿真期间: {start_date} 到 {end_date}")
    print(f"优化模式: {'启用' if enable_optimization else '禁用'}")
    print("=" * 70 + "\n")
    
    # ===================== 第1步: 加载配置 =====================
    print("📋 第1步: 加载配置数据...")
    from src.utils.config_loader import load_configuration
    
    start_time = time.time()
    config_dict = load_configuration(config_path)
    load_time = time.time() - start_time
    
    print(f"✅ 配置加载完成，耗时: {load_time:.2f}秒")
    print(f"   加载了 {len(config_dict)} 个配置表")
    
    # ===================== 第2步: 初始化优化引擎 =====================
    print("\n🔧 第2步: 初始化DuckDB优化引擎...")
    
    # 创建优化仿真运行器
    runner = create_optimized_runner(
        config_dict=config_dict,
        db_connection_string=db_connection_string,
        enable_optimization=enable_optimization,
        cache_dir=f"{output_dir}/cache"
    )
    
    # 创建模块优化器
    optimizers = create_all_optimizers()
    
    # 初始化增量处理器
    incremental = create_incremental_processor(
        cache_dir=f"{output_dir}/incremental_cache"
    )
    
    # ===================== 第3步: 初始化性能监控 =====================
    print("\n📊 第3步: 初始化性能监控...")
    
    dashboard = PerformanceDashboard(output_dir=f"{output_dir}/performance")
    dashboard.start_simulation()
    
    # 计算仿真天数
    sim_dates = pd.date_range(start_date, end_date, freq='D')
    total_days = len(sim_dates)
    
    monitor = RealTimeMonitor(total_days=total_days)
    monitor.start()
    
    print(f"✅ 监控初始化完成，总共 {total_days} 天")
    
    # ===================== 第4步: 运行仿真 =====================
    print("\n🏃 第4步: 开始运行仿真...")
    print("-" * 70)
    
    # 导入核心模块
    from src.core.orchestrator import create_orchestrator
    from src.modules import module1, module3, module4, module5, module6
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 创建Orchestrator
    orchestrator = create_orchestrator(
        start_date=start_date,
        output_dir=str(output_path / "orchestrator")
    )
    
    # 设置初始库存
    if 'M1_InitialInventory' in config_dict:
        orchestrator.initialize_inventory(config_dict['M1_InitialInventory'])
    
    # 每日循环
    for i, current_date in enumerate(sim_dates, 1):
        day_start = time.time()
        date_str = current_date.strftime('%Y-%m-%d')
        
        dashboard.start_day(date_str)
        
        print(f"\n{'='*20} 第 {i}/{total_days} 天: {date_str} {'='*20}")
        
        # 准备每日上下文
        with runner.optimized_context(date_str, orchestrator):
            
            # ========== Module1: 订单生成 ==========
            with track_performance('Module1'):
                try:
                    m1_result = module1.run_daily_order_generation(
                        config_dict=config_dict,
                        simulation_date=current_date,
                        output_dir=str(output_path / "module1"),
                        orchestrator=orchestrator
                    )
                    print(f"  ✅ Module1 完成 - 订单: {len(m1_result.get('orders_df', []))}")
                except Exception as e:
                    logger.error(f"Module1 失败: {e}")
                    m1_result = {}
            
            # ========== Module4: 生产计划 ==========
            with track_performance('Module4'):
                try:
                    # 使用优化的计算
                    if enable_optimization and 'Module3' in optimizers:
                        # 向量化净需求计算
                        net_demand = optimizers['Module3'].vectorized_net_demand(
                            m1_result.get('orders_df', pd.DataFrame()),
                            orchestrator.get_current_inventory(),
                            orchestrator.get_in_transit(),
                            pd.DataFrame()
                        )
                    
                    # 运行Module4
                    from src.core.main_integration import run_module4_integrated
                    m4_result = run_module4_integrated(
                        config_dict=config_dict,
                        module3_output_dir=str(output_path / "module3"),
                        simulation_date=current_date,
                        simulation_start=pd.to_datetime(start_date),
                        output_dir=str(output_path / "module4")
                    )
                    print(f"  ✅ Module4 完成 - 生产计划: {len(m4_result) if isinstance(m4_result, pd.DataFrame) else 0}")
                except Exception as e:
                    logger.error(f"Module4 失败: {e}")
            
            # ========== Module5: 部署计划 ==========
            with track_performance('Module5'):
                try:
                    # 使用优化的优先级分配
                    if enable_optimization and 'Module5' in optimizers:
                        demand_df = m1_result.get('orders_df', pd.DataFrame())
                        inventory_df = orchestrator.get_current_inventory()
                        
                        allocation = optimizers['Module5'].vectorized_priority_allocation(
                            demand_df, inventory_df
                        )
                    
                    # 运行Module5
                    m5_result = module5.main(
                        config_dict=config_dict,
                        module1_output_dir=str(output_path / "module1"),
                        module4_output_path=str(output_path / "module4" / f"Module4Output_{current_date.strftime('%Y%m%d')}.xlsx"),
                        orchestrator=orchestrator,
                        current_date=date_str,
                        output_path=str(output_path / "module5" / f"Module5Output_{current_date.strftime('%Y%m%d')}.xlsx")
                    )
                    print(f"  ✅ Module5 完成")
                except Exception as e:
                    logger.error(f"Module5 失败: {e}")
            
            # ========== Module6: 物流执行 ==========
            with track_performance('Module6'):
                try:
                    m6_result = module6.run_daily_physical_flow(
                        config_dict=config_dict,
                        orchestrator=orchestrator,
                        current_date=current_date,
                        output_dir=str(output_path / "module6"),
                        max_wait_days=30
                    )
                    print(f"  ✅ Module6 完成")
                except Exception as e:
                    logger.error(f"Module6 失败: {e}")
            
            # ========== Module3: 净需求计算 ==========
            with track_performance('Module3'):
                try:
                    # 使用增量计算
                    if enable_optimization:
                        orders_changeset = incremental.get_changes(
                            'orders', 
                            m1_result.get('orders_df', pd.DataFrame())
                        )
                        
                        if orders_changeset.has_changes:
                            logger.info(f"增量变化: {orders_changeset.summary()}")
                    
                    # 运行Module3
                    m3_result = module3.run_integrated_mode(
                        module1_output_dir=str(output_path / "module1"),
                        orchestrator=orchestrator,
                        config_dict=config_dict,
                        start_date=date_str,
                        end_date=date_str,
                        output_dir=str(output_path / "module3"),
                        module1_result=m1_result
                    )
                    print(f"  ✅ Module3 完成")
                except Exception as e:
                    logger.error(f"Module3 失败: {e}")
            
            # 保存每日状态
            orchestrator.save_daily_state(date_str)
        
        # 更新进度
        day_time = time.time() - day_start
        monitor.update(i, day_time)
        dashboard.end_day()
        
        # 显示进度条
        monitor.print_progress_bar()
    
    print("\n")
    
    # ===================== 第5步: 生成报告 =====================
    print("\n📈 第5步: 生成性能报告...")
    
    dashboard.end_simulation()
    
    # 打印性能摘要
    dashboard.print_summary()
    runner.print_performance_summary()
    
    # 导出详细报告
    json_report = dashboard.export_report('json')
    html_report = dashboard.export_report('html')
    
    print(f"\n📄 报告已生成:")
    print(f"   JSON报告: {json_report}")
    print(f"   HTML报告: {html_report}")
    
    # 清理资源
    runner.cleanup()
    incremental.close()
    for opt in optimizers.values():
        opt.cleanup()
    
    print("\n" + "=" * 70)
    print("🎉 优化仿真完成!")
    print("=" * 70)


def run_benchmark(
    config_path: str,
    start_date: str,
    end_date: str,
    output_dir: str = "./benchmark_output"
):
    """
    运行性能基准测试
    
    对比优化前后的性能差异
    """
    print("\n" + "=" * 70)
    print("📊 性能基准测试")
    print("=" * 70 + "\n")
    
    results = {}
    
    # 测试1: 标准模式
    print("🔄 测试1: 标准模式 (无优化)...")
    start_time = time.time()
    
    try:
        run_optimized_simulation_example(
            config_path=config_path,
            start_date=start_date,
            end_date=end_date,
            output_dir=f"{output_dir}/standard",
            enable_optimization=False
        )
        results['standard'] = time.time() - start_time
    except Exception as e:
        logger.error(f"标准模式测试失败: {e}")
        results['standard'] = None
    
    # 测试2: 优化模式
    print("\n🚀 测试2: 优化模式 (DuckDB)...")
    start_time = time.time()
    
    try:
        run_optimized_simulation_example(
            config_path=config_path,
            start_date=start_date,
            end_date=end_date,
            output_dir=f"{output_dir}/optimized",
            enable_optimization=True
        )
        results['optimized'] = time.time() - start_time
    except Exception as e:
        logger.error(f"优化模式测试失败: {e}")
        results['optimized'] = None
    
    # 输出对比结果
    print("\n" + "=" * 70)
    print("📊 基准测试结果")
    print("=" * 70)
    
    if results['standard'] and results['optimized']:
        speedup = results['standard'] / results['optimized']
        print(f"标准模式耗时: {results['standard']:.2f}秒")
        print(f"优化模式耗时: {results['optimized']:.2f}秒")
        print(f"性能提升: {speedup:.2f}x ({(1 - 1/speedup) * 100:.1f}% 更快)")
    else:
        print("测试未完全完成，无法进行对比")
    
    print("=" * 70)
    
    return results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ChainSight 优化仿真')
    parser.add_argument('--config', '-c', required=True, help='配置文件路径')
    parser.add_argument('--start', '-s', required=True, help='开始日期 (YYYY-MM-DD)')
    parser.add_argument('--end', '-e', required=True, help='结束日期 (YYYY-MM-DD)')
    parser.add_argument('--output', '-o', default='./optimized_output', help='输出目录')
    parser.add_argument('--db', help='PostgreSQL连接字符串')
    parser.add_argument('--no-optimize', action='store_true', help='禁用优化')
    parser.add_argument('--benchmark', action='store_true', help='运行基准测试')
    
    args = parser.parse_args()
    
    if args.benchmark:
        run_benchmark(
            config_path=args.config,
            start_date=args.start,
            end_date=args.end,
            output_dir=args.output
        )
    else:
        run_optimized_simulation_example(
            config_path=args.config,
            start_date=args.start,
            end_date=args.end,
            output_dir=args.output,
            db_connection_string=args.db,
            enable_optimization=not args.no_optimize
        )


if __name__ == '__main__':
    main()
