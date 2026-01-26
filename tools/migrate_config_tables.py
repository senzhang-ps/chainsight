#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置表迁移脚本
将旧格式表（如 bc_s5_config_guide）迁移到新格式统一表（如 cfg_config_guide）
通过 config_name 字段区分不同配置
"""

import sys
from pathlib import Path
from collections import defaultdict

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pgsql_db import DatabaseConnection


def get_config_prefixes(tables: list) -> dict:
    """
    从表名中提取配置前缀
    
    例如：bc_s5_config_guide -> 前缀 bc_s5, 基础名 config_guide
    
    Returns:
        dict: {prefix: [base_table_names]}
    """
    prefixes = defaultdict(list)
    
    for table in tables:
        # 跳过已经是新格式的表
        if table.startswith('cfg_'):
            continue
        # 跳过模块输出表和其他系统表
        if table.startswith('module') or table.startswith('orchestrator_') or table.startswith('summary_'):
            continue
        
        # 尝试提取前缀（假设格式为 prefix_base_name）
        # 常见前缀格式：bc_s5_, bc_s9_, oc_paste_s1_ 等
        parts = table.split('_')
        if len(parts) >= 3:
            # 尝试找到配置前缀的边界
            # 通常配置前缀是 bc_s5 或 bc_s9 这种格式
            for i in range(2, min(4, len(parts))):
                potential_prefix = '_'.join(parts[:i])
                base_name = '_'.join(parts[i:])
                
                # 检查是否是有效的配置前缀（包含字母和数字的组合）
                if any(c.isdigit() for c in potential_prefix) and any(c.isalpha() for c in potential_prefix):
                    prefixes[potential_prefix].append(base_name)
                    break
    
    return prefixes


def migrate_tables(
    db_host: str = "localhost",
    db_port: int = 5432,
    db_name: str = "test_db",
    db_user: str = "postgres",
    db_password: str = "123456",
    dry_run: bool = True,
    delete_old: bool = False
) -> dict:
    """
    迁移配置表到新格式
    
    Args:
        db_host: 数据库主机
        db_port: 数据库端口
        db_name: 数据库名称
        db_user: 数据库用户名
        db_password: 数据库密码
        dry_run: 如果为True，只显示将要执行的操作，不实际执行
        delete_old: 是否删除旧表
    
    Returns:
        dict: 迁移结果统计
    """
    print("\n" + "=" * 70)
    print("🔄 配置表迁移工具")
    print("=" * 70)
    print(f"数据库: {db_host}:{db_port}/{db_name}")
    print(f"模式: {'预览（不执行）' if dry_run else '执行迁移'}")
    print(f"删除旧表: {'是' if delete_old else '否'}")
    print("=" * 70)
    
    db = DatabaseConnection(
        host=db_host,
        port=db_port,
        database=db_name,
        user=db_user,
        password=db_password
    )
    
    stats = {
        "tables_found": 0,
        "tables_migrated": 0,
        "tables_skipped": 0,
        "rows_migrated": 0,
        "old_tables_deleted": 0,
        "errors": []
    }
    
    try:
        # 测试连接
        test_result = db.test_connection()
        if not test_result["success"]:
            print(f"❌ 数据库连接失败: {test_result['message']}")
            return stats
        print(f"✅ 数据库连接成功")
        
        # 获取所有表
        all_tables = db.get_all_tables()
        print(f"\n📊 数据库中共有 {len(all_tables)} 个表")
        
        # 分析配置前缀
        prefixes = get_config_prefixes(all_tables)
        
        if not prefixes:
            print("ℹ️ 未找到需要迁移的旧格式配置表")
            return stats
        
        print(f"\n📋 发现 {len(prefixes)} 个配置前缀:")
        for prefix, tables in sorted(prefixes.items()):
            config_name = prefix.upper().replace('_', '_')  # BC_S5
            print(f"  • {prefix} ({len(tables)} 个表) -> config_name='{config_name}'")
        
        # 按基础表名分组，准备合并
        base_table_groups = defaultdict(list)  # base_name -> [(prefix, old_table_name)]
        
        for prefix, base_names in prefixes.items():
            for base_name in base_names:
                old_table_name = f"{prefix}_{base_name}"
                base_table_groups[base_name].append((prefix, old_table_name))
        
        print(f"\n📦 将合并为 {len(base_table_groups)} 个统一配置表:")
        for base_name, sources in sorted(base_table_groups.items()):
            new_table = f"cfg_{base_name}"
            source_prefixes = [p for p, _ in sources]
            print(f"  • {new_table} <- {source_prefixes}")
        
        stats["tables_found"] = sum(len(tables) for tables in prefixes.values())
        
        if dry_run:
            print("\n⚠️ 预览模式，以下操作不会实际执行")
        
        # 执行迁移
        print("\n" + "-" * 70)
        print("🚀 开始迁移...")
        print("-" * 70)
        
        for base_name, sources in sorted(base_table_groups.items()):
            new_table_name = f"cfg_{base_name}"
            print(f"\n📋 处理: {new_table_name}")
            
            for prefix, old_table_name in sources:
                config_name = prefix.upper()  # BC_S5
                
                try:
                    # 读取旧表数据
                    df = db.read_table(old_table_name)
                    row_count = len(df)
                    
                    if row_count == 0:
                        print(f"  ⏭️ {old_table_name} 为空表，跳过")
                        stats["tables_skipped"] += 1
                        continue
                    
                    print(f"  📥 {old_table_name} ({row_count} 行) -> config_name='{config_name}'")
                    
                    if not dry_run:
                        # 添加 config_name 列（如果不存在）
                        if 'config_name' not in df.columns:
                            df['config_name'] = config_name
                        
                        # 检查新表是否存在
                        new_table_exists = db.table_exists(new_table_name)
                        
                        if new_table_exists:
                            # 追加到现有表
                            db.create_table_from_df(df, new_table_name, if_exists='append', config_name=None)
                        else:
                            # 创建新表
                            db.create_table_from_df(df, new_table_name, if_exists='replace', config_name=None)
                        
                        stats["rows_migrated"] += row_count
                    
                    stats["tables_migrated"] += 1
                    
                except Exception as e:
                    error_msg = f"迁移 {old_table_name} 失败: {e}"
                    print(f"  ❌ {error_msg}")
                    stats["errors"].append(error_msg)
        
        # 删除旧表
        if delete_old and not dry_run:
            print("\n" + "-" * 70)
            print("🗑️ 删除旧表...")
            print("-" * 70)
            
            for prefix, base_names in prefixes.items():
                for base_name in base_names:
                    old_table_name = f"{prefix}_{base_name}"
                    try:
                        db.drop_table(old_table_name)
                        stats["old_tables_deleted"] += 1
                        print(f"  ✅ 已删除: {old_table_name}")
                    except Exception as e:
                        print(f"  ❌ 删除 {old_table_name} 失败: {e}")
        
        # 打印统计
        print("\n" + "=" * 70)
        print("📊 迁移统计")
        print("=" * 70)
        print(f"发现旧表: {stats['tables_found']}")
        print(f"已迁移: {stats['tables_migrated']}")
        print(f"已跳过: {stats['tables_skipped']}")
        print(f"迁移行数: {stats['rows_migrated']}")
        if delete_old:
            print(f"已删除旧表: {stats['old_tables_deleted']}")
        if stats["errors"]:
            print(f"错误: {len(stats['errors'])}")
            for err in stats["errors"]:
                print(f"  - {err}")
        
        if dry_run:
            print("\n💡 提示: 使用 --execute 参数执行实际迁移")
            print("         使用 --delete-old 参数同时删除旧表")
        
        return stats
        
    except Exception as e:
        print(f"\n❌ 迁移失败: {e}")
        import traceback
        traceback.print_exc()
        return stats
    finally:
        db.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="配置表迁移工具 - 将旧格式表迁移到新格式统一表",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 预览迁移（不执行）
  python migrate_config_tables.py
  
  # 执行迁移
  python migrate_config_tables.py --execute
  
  # 执行迁移并删除旧表
  python migrate_config_tables.py --execute --delete-old
        """
    )
    parser.add_argument("--host", default="localhost", help="数据库主机")
    parser.add_argument("--port", type=int, default=5432, help="数据库端口")
    parser.add_argument("--database", default="test_db", help="数据库名称")
    parser.add_argument("--user", default="postgres", help="数据库用户名")
    parser.add_argument("--password", default="123456", help="数据库密码")
    parser.add_argument("--execute", action="store_true", help="执行实际迁移（默认为预览模式）")
    parser.add_argument("--delete-old", action="store_true", help="迁移后删除旧表")
    
    args = parser.parse_args()
    
    migrate_tables(
        db_host=args.host,
        db_port=args.port,
        db_name=args.database,
        db_user=args.user,
        db_password=args.password,
        dry_run=not args.execute,
        delete_old=args.delete_old
    )


if __name__ == "__main__":
    main()
