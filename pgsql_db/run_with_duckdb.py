"""
DuckDB增强的数据库模式运行脚本
使用DuckDB进行数据处理，结果写入PostgreSQL
"""

import sys
import argparse
from pathlib import Path
import time
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pgsql_db.duckdb_processor import DuckDBProcessor
from pgsql_db.db_connection import DatabaseConnection
from pgsql_db.data_pipeline import DataPipeline


def run_with_duckdb(
    config_name: str,
    start_date: str,
    end_date: str,
    db_host: str = "localhost",
    db_port: int = 5432,
    db_name: str = "test_db",
    db_user: str = "postgres",
    db_password: str = "123456"
):
    """
    使用DuckDB处理数据，结果写入PostgreSQL
    
    Args:
        config_name: 配置名称（如 BC_S5）
        start_date: 开始日期
        end_date: 结束日期
        db_*: 数据库连接参数
    """
    print("\n" + "=" * 70)
    print("🦆 DuckDB增强数据库模式")
    print("=" * 70)
    print(f"📋 配置: {config_name}")
    print(f"📅 日期: {start_date} 到 {end_date}")
    print(f"🗄️  数据库: {db_host}:{db_port}/{db_name}")
    print("=" * 70)
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{config_name}_{ts}"
    
    # 创建日志目录（输出到 outputs）
    log_dir = project_root / "outputs" / f"db_duckdb_{run_id}"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    total_start = time.time()
    
    with DataPipeline(
        pg_host=db_host,
        pg_port=db_port,
        pg_database=db_name,
        pg_user=db_user,
        pg_password=db_password
    ) as pipeline:
        
        # 1. 测试连接
        print("\n🔌 测试数据库连接...")
        result = pipeline.pg.test_connection()
        if not result['success']:
            print(f"❌ 连接失败: {result['message']}")
            return False
        print(f"✅ PostgreSQL连接成功")
        
        # 2. 从数据库加载配置
        print("\n📥 从数据库加载配置...")
        config_data = load_config_from_db(pipeline.pg, config_name)
        if not config_data:
            print(f"❌ 未找到配置: {config_name}")
            return False
        print(f"✅ 已加载 {len(config_data)} 个配置表")
        
        # 3. 使用DuckDB处理配置数据
        print("\n🦆 DuckDB数据处理...")
        process_config_with_duckdb(pipeline, config_data, config_name)
        
        # 4. 运行仿真（使用处理后的数据）
        print("\n🚀 运行仿真...")
        simulation_result = run_simulation(
            pipeline, 
            config_data, 
            start_date, 
            end_date,
            run_id
        )
        
        if simulation_result:
            # 5. 处理并写入输出数据
            print("\n📤 处理输出数据...")
            process_outputs_with_duckdb(pipeline, simulation_result, run_id)
        
        # 统计
        total_time = time.time() - total_start
        pipeline.print_stats()
        
        print(f"\n✅ 完成! 总耗时: {total_time:.2f}s")
        print(f"📁 日志目录: {log_dir}")
        
        # 保存运行日志
        log_file = log_dir / f"run_log_{ts}.txt"
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"运行时间: {datetime.now().isoformat()}\n")
            f.write(f"配置: {config_name}\n")
            f.write(f"日期范围: {start_date} 到 {end_date}\n")
            f.write(f"总耗时: {total_time:.2f}s\n")
            f.write(f"处理统计: {pipeline.get_stats()}\n")
        
        return True


def load_config_from_db(db: DatabaseConnection, config_name: str) -> dict:
    """从数据库加载配置表"""
    all_tables = db.get_all_tables()
    prefix = config_name.lower().replace("-", "_").replace(" ", "_")
    
    config_tables = [t for t in all_tables if t.startswith(prefix + "_")]
    
    if not config_tables:
        return None
    
    config_data = {}
    for table_name in config_tables:
        sheet_name = table_name[len(prefix) + 1:]
        try:
            df = db.read_table(table_name)
            config_data[sheet_name] = df
            print(f"  ✅ {sheet_name}: {len(df)} 行")
        except Exception as e:
            print(f"  ⚠️ {table_name}: {e}")
    
    return config_data


def process_config_with_duckdb(
    pipeline: DataPipeline, 
    config_data: dict, 
    config_name: str
):
    """使用DuckDB处理配置数据"""
    duck = pipeline.duck
    
    # 处理Global_Network - 创建物料位置关联视图
    if 'global_network' in config_data:
        duck.register_dataframe(config_data['global_network'], 'network')
        
        # 创建物料-位置汇总
        material_location_summary = duck.query("""
            SELECT 
                material,
                location,
                sourcing,
                location_type,
                COUNT(*) as record_count
            FROM network
            GROUP BY material, location, sourcing, location_type
        """)
        
        pipeline.pg.create_table_from_df(
            material_location_summary, 
            f'{config_name.lower()}_material_location_summary',
            'replace'
        )
        print(f"  ✅ 物料位置汇总: {len(material_location_summary)} 行")
        duck.unregister_table('network')
    
    # 处理安全库存 - 按物料汇总
    if 'm3_safetystock' in config_data:
        duck.register_dataframe(config_data['m3_safetystock'], 'safety_stock')
        
        ss_summary = duck.query("""
            SELECT 
                material,
                location,
                AVG(safety_stock_qty) as avg_safety_stock,
                MAX(safety_stock_qty) as max_safety_stock,
                MIN(safety_stock_qty) as min_safety_stock
            FROM safety_stock
            GROUP BY material, location
        """)
        
        pipeline.pg.create_table_from_df(
            ss_summary,
            f'{config_name.lower()}_safety_stock_summary',
            'replace'
        )
        print(f"  ✅ 安全库存汇总: {len(ss_summary)} 行")
        duck.unregister_table('safety_stock')
    
    # 处理部署配置 - MOQ/RV分析
    if 'm5_deployconfig' in config_data:
        duck.register_dataframe(config_data['m5_deployconfig'], 'deploy_config')
        
        deploy_summary = duck.query("""
            SELECT 
                material,
                sending,
                receiving,
                moq,
                rv,
                lsk,
                day
            FROM deploy_config
            WHERE moq > 0 OR rv > 0
        """)
        
        pipeline.pg.create_table_from_df(
            deploy_summary,
            f'{config_name.lower()}_deploy_config_with_moq_rv',
            'replace'
        )
        print(f"  ✅ 部署配置(MOQ/RV): {len(deploy_summary)} 行")
        duck.unregister_table('deploy_config')


def run_simulation(
    pipeline: DataPipeline,
    config_data: dict,
    start_date: str,
    end_date: str,
    run_id: str
) -> dict:
    """运行仿真（简化版本，实际应调用main_integration）"""
    # 这里返回模拟的输出数据结构
    # 实际实现中应该调用 run_integrated_simulation
    
    import pandas as pd
    from datetime import datetime, timedelta
    
    # 模拟输出数据
    result = {
        'order_log': pd.DataFrame({
            'material': ['M001', 'M002'],
            'location': ['L1', 'L1'],
            'quantity': [100, 200],
            'order_type': ['normal', 'ao']
        }),
        'shipment_log': pd.DataFrame({
            'material': ['M001'],
            'source_location': ['L1'],
            'destination_location': ['L2'],
            'quantity': [80]
        }),
        'cut_log': pd.DataFrame({
            'material': ['M001', 'M002'],
            'location': ['L1', 'L1'],
            'quantity': [20, 50]
        })
    }
    
    print(f"  ✅ 模拟仿真完成")
    return result


def process_outputs_with_duckdb(
    pipeline: DataPipeline,
    outputs: dict,
    run_id: str
):
    """使用DuckDB处理输出数据"""
    duck = pipeline.duck
    
    # 注册所有输出表
    for name, df in outputs.items():
        if not df.empty:
            duck.register_dataframe(df, name)
    
    # 创建综合分析视图
    if 'order_log' in outputs and not outputs['order_log'].empty:
        # 订单分析
        order_analysis = duck.query("""
            SELECT 
                material,
                location,
                order_type,
                SUM(quantity) as total_qty,
                COUNT(*) as order_count
            FROM order_log
            GROUP BY material, location, order_type
        """)
        order_analysis['run_id'] = run_id
        
        pipeline.pg.create_table_from_df(
            order_analysis,
            'analysis_order_summary',
            'replace'
        )
        print(f"  ✅ 订单分析: {len(order_analysis)} 行")
    
    # 履约分析
    if all(k in outputs for k in ['order_log', 'shipment_log', 'cut_log']):
        fulfillment = duck.query("""
            SELECT 
                o.material,
                SUM(o.quantity) as ordered,
                COALESCE(SUM(s.quantity), 0) as shipped,
                COALESCE(SUM(c.quantity), 0) as cut,
                ROUND(COALESCE(SUM(s.quantity), 0) * 100.0 / NULLIF(SUM(o.quantity), 0), 2) as rate
            FROM order_log o
            LEFT JOIN shipment_log s ON o.material = s.material
            LEFT JOIN cut_log c ON o.material = c.material
            GROUP BY o.material
        """)
        fulfillment['run_id'] = run_id
        
        pipeline.pg.create_table_from_df(
            fulfillment,
            'analysis_fulfillment_rate',
            'replace'
        )
        print(f"  ✅ 履约分析: {len(fulfillment)} 行")
    
    # 清理
    for name in outputs.keys():
        duck.unregister_table(name)


def main():
    parser = argparse.ArgumentParser(description="DuckDB增强数据库模式运行")
    parser.add_argument("--config", required=True, help="配置名称（如 BC_S5）")
    parser.add_argument("--start-date", required=True, help="开始日期")
    parser.add_argument("--end-date", required=True, help="结束日期")
    parser.add_argument("--db-host", default="localhost", help="数据库主机")
    parser.add_argument("--db-port", type=int, default=5432, help="数据库端口")
    parser.add_argument("--db-name", default="test_db", help="数据库名称")
    parser.add_argument("--db-user", default="postgres", help="数据库用户")
    parser.add_argument("--db-password", default="123456", help="数据库密码")
    
    args = parser.parse_args()
    
    success = run_with_duckdb(
        config_name=args.config,
        start_date=args.start_date,
        end_date=args.end_date,
        db_host=args.db_host,
        db_port=args.db_port,
        db_name=args.db_name,
        db_user=args.db_user,
        db_password=args.db_password
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
