"""
数据库集成运行脚本
支持将配置文件和模块输出写入PostgreSQL数据库
"""

import sys
import argparse
from pathlib import Path
import time

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pgsql_db.db_connection import DatabaseConnection, test_database_connection
from pgsql_db.excel_importer import ExcelImporter
from pgsql_db.module_data_writer import ModuleDataWriter


def import_all_config_files(db: DatabaseConnection) -> bool:
    """导入所有配置文件到数据库"""
    
    test_files_dir = project_root / "test_files"
    
    excel_files = [
        test_files_dir / "BC_S5.xlsx",
        test_files_dir / "BC_S9.xlsx",
        test_files_dir / "OC Paste_S1.xlsx"
    ]
    
    # 检查文件存在性
    existing_files = [str(f) for f in excel_files if f.exists()]
    
    if not existing_files:
        print("❌没有找到配置文件")
        return False
    
    print(f"找到 {len(existing_files)} 个配置文件")
    
    importer = ExcelImporter(db)
    results = importer.import_multiple_files(existing_files)
    importer.print_import_summary()
    
    return True


def run_simulation_with_db(
    config_path: str,
    start_date: str,
    end_date: str,
    db: DatabaseConnection,
    write_to_db: bool = True
) -> dict:
    """
    运行仿真并将结果写入数据库
    
    Args:
        config_path: 配置文件路径
        start_date: 开始日期
        end_date: 结束日期
        db: 数据库连接
        write_to_db: 是否写入数据库
    
    Returns:
        dict: 运行结果
    """
    from src.core.main_integration import run_integrated_simulation
    
    print("\n" + "=" * 60)
    print("🚀 开始数据库集成仿真")
    print("=" * 60)
    print(f"配置文件: {config_path}")
    print(f"日期范围: {start_date} 到 {end_date}")
    print(f"写入数据库: {'是' if write_to_db else '否'}")
    print("=" * 60)
    
    start_time = time.time()
    
    # 运行仿真
    result = run_integrated_simulation(
        config_path=config_path,
        start_date=start_date,
        end_date=end_date
    )
    
    simulation_time = time.time() - start_time
    
    if result and result.get('simulation_completed'):
        output_dir = result.get('output_directory')
        print(f"\n✅仿真完成，输出目录: {output_dir}")
        print(f"仿真耗时: {simulation_time:.2f}s")
        
        # 写入数据库
        if write_to_db and output_dir:
            print("\n📝开始写入数据库...")
            db_start = time.time()
            
            writer = ModuleDataWriter(db)
            writer.write_all_modules(output_dir)
            
            # 写入Orchestrator数据
            orch_dir = Path(output_dir) / "orchestrator"
            if orch_dir.exists():
                writer.write_orchestrator_data(str(orch_dir))
            
            writer.print_summary()
            
            db_time = time.time() - db_start
            print(f"数据库写入耗时: {db_time:.2f}s")
        
        return {
            "success": True,
            "output_dir": output_dir,
            "simulation_time": simulation_time
        }
    else:
        print("❌仿真失败")
        return {"success": False}


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="数据库集成仿真运行器")
    
    parser.add_argument("--action", choices=["test", "import", "run", "full"],
                       default="test", help="操作类型")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--start-date", type=str, help="开始日期")
    parser.add_argument("--end-date", type=str, help="结束日期")
    parser.add_argument("--db-host", type=str, default="localhost", help="数据库主机")
    parser.add_argument("--db-port", type=int, default=5432, help="数据库端口")
    parser.add_argument("--db-name", type=str, default="test_db", help="数据库名称")
    parser.add_argument("--db-user", type=str, default="postgres", help="数据库用户")
    parser.add_argument("--db-password", type=str, default="123456", help="数据库密码")
    parser.add_argument("--no-db-write", action="store_true", help="不写入数据库")
    
    args = parser.parse_args()
    
    # 创建数据库连接
    db = DatabaseConnection(
        host=args.db_host,
        port=args.db_port,
        database=args.db_name,
        user=args.db_user,
        password=args.db_password
    )
    
    try:
        if args.action == "test":
            # 测试数据库连接
            print("\n🔍 测试数据库连接...")
            result = db.test_connection()
            if result["success"]:
                print("✅数据库连接成功")
                print(f"版本: {result['version'][:60]}...")
                tables = db.get_all_tables()
                print(f"现有表数量: {len(tables)}")
            else:
                print(f"❌连接失败: {result['message']}")
        
        elif args.action == "import":
            # 导入配置文件
            print("\n📥 导入配置文件...")
            conn_result = db.test_connection()
            if not conn_result["success"]:
                print(f"❌数据库连接失败: {conn_result['message']}")
                return 1
            import_all_config_files(db)
        
        elif args.action == "run":
            # 运行仿真
            if not args.config or not args.start_date or not args.end_date:
                print("❌运行仿真需要指定 --config, --start-date, --end-date")
                return 1
            
            conn_result = db.test_connection()
            if not conn_result["success"]:
                print(f"❌数据库连接失败: {conn_result['message']}")
                return 1
            
            run_simulation_with_db(
                args.config,
                args.start_date,
                args.end_date,
                db,
                write_to_db=not args.no_db_write
            )
        
        elif args.action == "full":
            # 完整流程：导入配置 + 运行仿真
            if not args.config or not args.start_date or not args.end_date:
                print("❌完整流程需要指定 --config, --start-date, --end-date")
                return 1
            
            conn_result = db.test_connection()
            if not conn_result["success"]:
                print(f"❌数据库连接失败: {conn_result['message']}")
                return 1
            
            # 先导入配置
            print("\n📥 步骤1: 导入配置文件...")
            import_all_config_files(db)
            
            # 再运行仿真
            print("\n🚀 步骤2: 运行仿真...")
            run_simulation_with_db(
                args.config,
                args.start_date,
                args.end_date,
                db,
                write_to_db=not args.no_db_write
            )
        
        return 0
        
    except Exception as e:
        print(f"❌执行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
