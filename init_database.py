"""
数据库初始化脚本
检测数据库和表是否存在，如不存在则创建并导入数据
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pgsql_db import DatabaseConnection, ExcelImporter, table_mapping


def init_database(
    config_excel_path: str,
    config_name: str,
    db_host: str = "localhost",
    db_port: int = 5432,
    db_name: str = "test_db",
    db_user: str = "postgres",
    db_password: str = "123456",
    force_recreate: bool = False
) -> bool:
    """
    初始化数据库，检测并创建配置表
    
    Args:
        config_excel_path: 配置Excel文件路径
        config_name: 配置名称（作为表名前缀）
        db_host: 数据库主机
        db_port: 数据库端口
        db_name: 数据库名称
        db_user: 数据库用户名
        db_password: 数据库密码
        force_recreate: 是否强制重新创建表
    
    Returns:
        bool: 是否成功
    """
    print("\n" + "=" * 70)
    print("🔧 数据库初始化")
    print("=" * 70)
    
    # 创建数据库连接
    db = DatabaseConnection(
        host=db_host,
        port=db_port,
        database=db_name,
        user=db_user,
        password=db_password
    )
    
    try:
        # ========== 步骤1: 检测数据库是否存在 ==========
        print("\n📋 步骤1: 检测数据库...")
        
        if not db.database_exists():
            print(f"  ⚠️ 数据库 {db_name} 不存在，正在创建...")
            if not db.create_database_if_not_exists():
                print(f"  ❌ 创建数据库失败")
                return False
        else:
            print(f"  ✅ 数据库 {db_name} 已存在")
        
        # ========== 步骤2: 测试连接 ==========
        print("\n📋 步骤2: 测试数据库连接...")
        test_result = db.test_connection()
        if not test_result["success"]:
            print(f"  ❌ 连接失败: {test_result['message']}")
            return False
        print(f"  ✅ 连接成功 (版本: {test_result['version'][:50]}...)")
        
        # ========== 步骤3: 检测配置表 ==========
        print("\n📋 步骤3: 检测配置表...")
        
        # 获取预期的配置表名
        prefix = config_name.lower().replace("-", "_").replace(" ", "_")
        
        # 检查Excel文件是否存在
        excel_path = Path(config_excel_path)
        if not excel_path.exists():
            print(f"  ❌ 配置文件不存在: {config_excel_path}")
            return False
        
        # 读取Excel文件的所有sheet名
        import pandas as pd
        xl = pd.ExcelFile(config_excel_path)
        sheet_names = xl.sheet_names
        
        # 获取预期的表名列表
        expected_tables = []
        for sheet_name in sheet_names:
            table_name = table_mapping.get_config_table_name(sheet_name, config_name)
            expected_tables.append((sheet_name, table_name))
        
        # 检查哪些表已存在
        existing_tables = set(db.get_all_tables())
        
        tables_to_create = []
        tables_existing = []
        
        for sheet_name, table_name in expected_tables:
            if table_name in existing_tables:
                tables_existing.append((sheet_name, table_name))
            else:
                tables_to_create.append((sheet_name, table_name))
        
        print(f"  📊 预期表数量: {len(expected_tables)}")
        print(f"  ✅ 已存在表数量: {len(tables_existing)}")
        print(f"  📝 需创建表数量: {len(tables_to_create)}")
        
        # 显示详细信息
        if tables_existing and not force_recreate:
            print("\n  已存在的表:")
            for sheet_name, table_name in tables_existing[:10]:
                print(f"    ✅ {sheet_name} -> {table_name}")
            if len(tables_existing) > 10:
                print(f"    ... 还有 {len(tables_existing) - 10} 个表")
        
        if tables_to_create:
            print("\n  需要创建的表:")
            for sheet_name, table_name in tables_to_create[:10]:
                print(f"    📝 {sheet_name} -> {table_name}")
            if len(tables_to_create) > 10:
                print(f"    ... 还有 {len(tables_to_create) - 10} 个表")
        
        # ========== 步骤4: 导入数据 ==========
        if tables_to_create or force_recreate:
            print("\n📋 步骤4: 导入配置数据...")
            
            importer = ExcelImporter(db)
            
            if force_recreate:
                print("  ⚠️ 强制重新创建所有表...")
                if_exists = "replace"
            else:
                if_exists = "fail"  # 只创建不存在的表
            
            # 逐个sheet导入
            results = importer.import_excel_file(
                config_excel_path,
                prefix=prefix,
                if_exists="replace" if force_recreate else "fail"
            )
            
            # 统计结果
            success_count = sum(1 for v in results.values() if v >= 0)
            fail_count = sum(1 for v in results.values() if v < 0)
            
            print(f"\n  📊 导入结果:")
            print(f"    ✅ 成功: {success_count} 个表")
            print(f"    ❌ 失败: {fail_count} 个表")
        else:
            print("\n📋 步骤4: 跳过（所有表已存在）")
        
        # ========== 步骤5: 检测输出表 ==========
        print("\n📋 步骤5: 检测输出表...")
        
        output_tables = table_mapping.get_all_output_tables()
        existing_output = [t for t in output_tables if t in existing_tables]
        
        print(f"  📊 定义的输出表: {len(output_tables)} 个")
        print(f"  ✅ 已存在输出表: {len(existing_output)} 个")
        
        if existing_output:
            print("\n  已存在的输出表:")
            for table in existing_output[:10]:
                print(f"    ✅ {table}")
            if len(existing_output) > 10:
                print(f"    ... 还有 {len(existing_output) - 10} 个表")
        
        # ========== 步骤6: 打印表映射关系 ==========
        print("\n📋 步骤6: 表映射关系...")
        table_mapping.print_table_mapping()
        
        print("\n" + "=" * 70)
        print("✅ 数据库初始化完成")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        db.close()


def check_database_status(
    config_name: str,
    db_host: str = "localhost",
    db_port: int = 5432,
    db_name: str = "test_db",
    db_user: str = "postgres",
    db_password: str = "123456"
) -> dict:
    """
    检查数据库状态
    
    Returns:
        dict: 状态信息
    """
    db = DatabaseConnection(
        host=db_host,
        port=db_port,
        database=db_name,
        user=db_user,
        password=db_password
    )
    
    status = {
        "database_exists": False,
        "connection_ok": False,
        "config_tables": [],
        "output_tables": [],
        "total_tables": 0
    }
    
    try:
        # 检测数据库
        status["database_exists"] = db.database_exists()
        
        if not status["database_exists"]:
            return status
        
        # 测试连接
        test_result = db.test_connection()
        status["connection_ok"] = test_result["success"]
        
        if not status["connection_ok"]:
            return status
        
        # 获取所有表
        all_tables = db.get_all_tables()
        status["total_tables"] = len(all_tables)
        
        # 分类配置表和输出表
        prefix = config_name.lower().replace("-", "_").replace(" ", "_")
        output_table_names = set(table_mapping.get_all_output_tables())
        
        for table in all_tables:
            if table.startswith(prefix + "_"):
                status["config_tables"].append(table)
            elif table in output_table_names:
                status["output_tables"].append(table)
        
        return status
        
    except Exception as e:
        status["error"] = str(e)
        return status
    finally:
        db.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="数据库初始化脚本")
    parser.add_argument("--config-file", "-f", help="配置Excel文件路径")
    parser.add_argument("--config-name", "-n", required=True, help="配置名称（作为表名前缀）")
    parser.add_argument("--host", default="localhost", help="数据库主机")
    parser.add_argument("--port", type=int, default=5432, help="数据库端口")
    parser.add_argument("--database", default="test_db", help="数据库名称")
    parser.add_argument("--user", default="postgres", help="数据库用户名")
    parser.add_argument("--password", default="123456", help="数据库密码")
    parser.add_argument("--force", action="store_true", help="强制重新创建所有表")
    parser.add_argument("--check-only", action="store_true", help="仅检查状态")
    
    args = parser.parse_args()
    
    if args.check_only:
        status = check_database_status(
            config_name=args.config_name,
            db_host=args.host,
            db_port=args.port,
            db_name=args.database,
            db_user=args.user,
            db_password=args.password
        )
        
        print("\n" + "=" * 50)
        print("📋 数据库状态检查")
        print("=" * 50)
        print(f"数据库存在: {'✅' if status['database_exists'] else '❌'}")
        print(f"连接正常: {'✅' if status['connection_ok'] else '❌'}")
        print(f"总表数量: {status['total_tables']}")
        print(f"配置表数量: {len(status['config_tables'])}")
        print(f"输出表数量: {len(status['output_tables'])}")
        
        if "error" in status:
            print(f"错误: {status['error']}")
    else:
        if not args.config_file:
            print("❌ 错误: 初始化模式需要指定 --config-file/-f 参数")
            sys.exit(1)
            
        success = init_database(
            config_excel_path=args.config_file,
            config_name=args.config_name,
            db_host=args.host,
            db_port=args.port,
            db_name=args.database,
            db_user=args.user,
            db_password=args.password,
            force_recreate=args.force
        )
        
        sys.exit(0 if success else 1)
