# -*- coding: utf-8 -*-
"""
高性能仿真运行脚本
使用OptimizedDataProcessor实现PostgreSQL + DuckDB协同处理
"""

import sys
import argparse
from pathlib import Path
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pgsql_db.optimized_processor import OptimizedDataProcessor
from pgsql_db.module_engine import ModuleCalculationEngine, create_calculation_engine
from pgsql_db.db_connection import DatabaseConnection


def run_optimized_simulation(
    config_name: str,
    start_date: str,
    end_date: str,
    db_host: str = "localhost",
    db_port: int = 5432,
    db_name: str = "test_db",
    db_user: str = "postgres",
    db_password: str = "123456",
    cache_dir: Optional[str] = None,
    memory_limit: str = "4GB",
    threads: int = 4
):
    """
    运行优化后的仿真
    
    性能优化策略：
    1. DuckDB直连PostgreSQL，减少数据传输
    2. 预建索引，避免重复过滤
    3. 向量化计算，替代Python循环
    4. Parquet缓存热点数据
    """
    print("\n" + "=" * 70)
    print("🚀 高性能仿真模式 (PostgreSQL + DuckDB)")
    print("=" * 70)
    print(f"📋 配置: {config_name}")
    print(f"📅 日期: {start_date} 到 {end_date}")
    print(f"🗄️  数据库: {db_host}:{db_port}/{db_name}")
    print(f"💾 内存限制: {memory_limit}")
    print(f"🔧 并行线程: {threads}")
    print("=" * 70)
    
    # 构建连接字符串
    pg_conn_str = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    # 设置缓存目录
    if cache_dir is None:
        cache_dir = str(project_root / "outputs" / "db_optimized_cache")
    
    total_start = time.perf_counter()
    
    # 创建计算引擎
    processor, engine = create_calculation_engine(
        pg_connection_string=pg_conn_str,
        cache_dir=cache_dir,
        memory_limit=memory_limit,
        threads=threads
    )
    
    try:
        # ==================== 阶段1: 连接和配置 ====================
        print("\n📡 阶段1: 数据库连接")
        phase1_start = time.perf_counter()
        
        # 尝试直连PostgreSQL
        try:
            processor.attach_postgres()
            use_direct_connection = True
            print("  ✅ DuckDB直连PostgreSQL成功")
        except Exception as e:
            print(f"  ⚠️ 直连失败，使用传统模式: {e}")
            use_direct_connection = False
        
        # 传统方式加载配置
        db = DatabaseConnection(
            host=db_host, port=db_port, database=db_name,
            user=db_user, password=db_password
        )
        
        config = load_config_from_db(db, config_name)
        if not config:
            print(f"❌ 未找到配置: {config_name}")
            return False
        
        phase1_time = time.perf_counter() - phase1_start
        print(f"  ⏱️ 配置加载完成: {phase1_time:.2f}s, {len(config)}个表")
        
        # ==================== 阶段2: 索引预建 ====================
        print("\n📊 阶段2: 索引预建")
        phase2_start = time.perf_counter()
        
        engine.prepare_config_indexes(config)
        
        phase2_time = time.perf_counter() - phase2_start
        print(f"  ⏱️ 索引预建完成: {phase2_time*1000:.1f}ms")
        
        # ==================== 阶段3: 仿真执行 ====================
        print("\n🔄 阶段3: 仿真执行")
        phase3_start = time.perf_counter()
        
        sim_start = datetime.strptime(start_date, '%Y-%m-%d')
        sim_end = datetime.strptime(end_date, '%Y-%m-%d')
        
        current_date = sim_start
        day_count = 0
        
        while current_date <= sim_end:
            day_start = time.perf_counter()
            
            # 执行单日仿真（使用优化引擎）
            daily_result = run_single_day_optimized(
                engine=engine,
                config=config,
                sim_date=current_date,
                db=db,
                config_name=config_name
            )
            
            day_time = time.perf_counter() - day_start
            print(f"  📅 {current_date.strftime('%Y-%m-%d')}: {day_time*1000:.0f}ms")
            
            current_date += timedelta(days=1)
            day_count += 1
        
        phase3_time = time.perf_counter() - phase3_start
        avg_day_time = phase3_time / day_count if day_count > 0 else 0
        print(f"\n  ⏱️ 仿真完成: {phase3_time:.2f}s ({day_count}天, 平均{avg_day_time*1000:.0f}ms/天)")
        
        # ==================== 统计汇总 ====================
        total_time = time.perf_counter() - total_start
        
        print("\n" + "=" * 70)
        print("📊 性能统计")
        print("=" * 70)
        processor.print_stats()
        
        print(f"\n⏱️ 总耗时: {total_time:.2f}s")
        print(f"📅 处理天数: {day_count}")
        print(f"⚡ 平均每天: {total_time/day_count*1000:.0f}ms")
        print("=" * 70)
        
        return True
        
    finally:
        processor.close()
        db.close()


def load_config_from_db(db: DatabaseConnection, config_name: str) -> Dict[str, Any]:
    """从数据库加载配置"""
    import pandas as pd
    
    all_tables = db.get_all_tables()
    prefix = config_name.lower().replace("-", "_").replace(" ", "_")
    
    config_tables = [t for t in all_tables if t.startswith(prefix + "_")]
    
    if not config_tables:
        return None
    
    config_data = {}
    
    # 表名映射
    sheet_name_map = {
        'm1_demandforecast': 'DemandForecast',
        'm3_safetystock': 'SafetyStock',
        'm5_deployconfig': 'DeployConfig',
        'm5_pushpullmodel': 'PushPullModel',
        'global_network': 'Network',
        'global_leadtime': 'LeadTime',
        'global_demandpriority': 'DemandPriority',
        'm4_materiallocationlinecfg': 'MaterialLocationLineCfg',
    }
    
    for table_name in config_tables:
        raw_sheet = table_name[len(prefix) + 1:]
        sheet_name = sheet_name_map.get(raw_sheet, raw_sheet)
        
        try:
            df = db.read_table(table_name)
            config_data[sheet_name] = df
        except Exception as e:
            print(f"  ⚠️ {table_name}: {e}")
    
    return config_data


def run_single_day_optimized(
    engine: ModuleCalculationEngine,
    config: Dict[str, Any],
    sim_date: datetime,
    db: DatabaseConnection,
    config_name: str
) -> Dict[str, Any]:
    """
    执行单日优化仿真
    
    使用引擎的批量计算能力替代逐节点循环
    """
    result = {
        'date': sim_date,
        'success': True,
        'outputs': {}
    }
    
    try:
        # 1. 预建每日索引
        supply_demand_log = config.get('SupplyDemandLog', pd.DataFrame())
        safety_stock = config.get('SafetyStock', pd.DataFrame())
        order_log = config.get('OrderLog', pd.DataFrame())
        
        engine.prepare_daily_indexes(supply_demand_log, safety_stock, order_log)
        
        # 2. 获取所有需要处理的物料-地点对
        network = config.get('Network', pd.DataFrame())
        if not network.empty:
            material_locations = list(zip(network['material'], network['location']))
        else:
            material_locations = []
        
        # 3. 批量计算净需求（替代逐节点循环）
        if material_locations:
            beginning_inventory = config.get('BeginningInventory', pd.DataFrame())
            intransit = config.get('InTransit', pd.DataFrame())
            open_deployment = config.get('OpenDeployment', pd.DataFrame())
            future_production = config.get('FutureProduction', pd.DataFrame())
            
            net_demands = engine.batch_calculate_net_demand(
                material_locations=material_locations,
                sim_date=sim_date,
                beginning_inventory=beginning_inventory,
                intransit=intransit,
                open_deployment=open_deployment,
                future_production=future_production,
                supply_demand_log=supply_demand_log,
                safety_stock=safety_stock
            )
            
            result['outputs']['net_demand'] = net_demands
        
        # 4. TODO: 集成其他模块计算
        # - Module4: 生产计划
        # - Module5: 部署规划
        # - Module6: 物流执行
        # 这些模块可以类似方式集成到计算引擎
        
    except Exception as e:
        result['success'] = False
        result['error'] = str(e)
        print(f"  ❌ 仿真错误: {e}")
    
    return result


# 需要导入pandas
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description='高性能仿真运行 (PostgreSQL + DuckDB)'
    )
    parser.add_argument('--config', required=True, help='配置名称')
    parser.add_argument('--start-date', required=True, help='开始日期')
    parser.add_argument('--end-date', required=True, help='结束日期')
    parser.add_argument('--db-host', default='localhost', help='数据库主机')
    parser.add_argument('--db-port', type=int, default=5432, help='数据库端口')
    parser.add_argument('--db-name', default='test_db', help='数据库名称')
    parser.add_argument('--db-user', default='postgres', help='数据库用户')
    parser.add_argument('--db-password', default='123456', help='数据库密码')
    parser.add_argument('--memory-limit', default='4GB', help='DuckDB内存限制')
    parser.add_argument('--threads', type=int, default=4, help='并行线程数')
    
    args = parser.parse_args()
    
    success = run_optimized_simulation(
        config_name=args.config,
        start_date=args.start_date,
        end_date=args.end_date,
        db_host=args.db_host,
        db_port=args.db_port,
        db_name=args.db_name,
        db_user=args.db_user,
        db_password=args.db_password,
        memory_limit=args.memory_limit,
        threads=args.threads
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
