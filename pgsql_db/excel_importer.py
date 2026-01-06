"""
Excel数据导入模块
将Excel配置文件的各个sheet导入PostgreSQL数据库
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import time

from .db_connection import DatabaseConnection


class ExcelImporter:
    """Excel数据导入器"""
    
    def __init__(self, db: DatabaseConnection):
        """
        初始化Excel导入器
        
        Args:
            db: 数据库连接实例
        """
        self.db = db
        self.imported_tables: Dict[str, Dict] = {}
    
    def import_excel_file(
        self,
        excel_path: str,
        prefix: str = None,
        if_exists: str = "replace"
    ) -> Dict[str, int]:
        """
        导入单个Excel文件的所有sheet到数据库
        
        Args:
            excel_path: Excel文件路径
            prefix: 表名前缀（默认使用文件名）
            if_exists: 如果表存在的处理方式
        
        Returns:
            dict: 每个sheet导入的行数
        """
        path = Path(excel_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {excel_path}")
        
        # 使用文件名作为前缀
        if prefix is None:
            prefix = self._clean_filename(path.stem)
        
        print(f"\n📂 导入Excel文件: {path.name}")
        print("-" * 50)
        
        # 读取所有sheet
        xl = pd.ExcelFile(excel_path)
        results = {}
        
        start_time = time.time()
        
        for sheet_name in xl.sheet_names:
            try:
                df = xl.parse(sheet_name)
                
                # 构建表名: prefix_sheetname
                table_name = f"{prefix}_{self._clean_name(sheet_name)}"
                
                # 空表也要创建（只要有列名）
                if df.empty:
                    # 检查是否有列定义
                    if len(df.columns) > 0:
                        print(f"  📋 Sheet [{sheet_name}] 为空表，创建表结构 ({len(df.columns)} 列)")
                        self.db.create_table_from_df(df, table_name, if_exists)
                        results[sheet_name] = 0
                    else:
                        print(f"  ⚠️ Sheet [{sheet_name}] 无数据且无列定义，跳过")
                        results[sheet_name] = -1
                    continue
                
                # 写入数据库
                self.db.create_table_from_df(df, table_name, if_exists)
                results[sheet_name] = len(df)
                
                # 记录导入信息
                self.imported_tables[table_name] = {
                    "source_file": str(path),
                    "sheet_name": sheet_name,
                    "row_count": len(df),
                    "column_count": len(df.columns)
                }
                
            except Exception as e:
                print(f"  ❌ Sheet [{sheet_name}] 导入失败: {e}")
                results[sheet_name] = -1
        
        elapsed = time.time() - start_time
        total_rows = sum(r for r in results.values() if r > 0)
        print(f"✅ 文件导入完成: {len(results)} 个sheet, {total_rows} 行数据, 耗时 {elapsed:.2f}s")
        
        return results
    
    def import_multiple_files(
        self,
        excel_paths: List[str],
        if_exists: str = "replace"
    ) -> Dict[str, Dict[str, int]]:
        """
        导入多个Excel文件
        
        Args:
            excel_paths: Excel文件路径列表
            if_exists: 如果表存在的处理方式
        
        Returns:
            dict: 每个文件每个sheet的导入行数
        """
        all_results = {}
        
        print("\n" + "=" * 60)
        print("批量导入Excel配置文件")
        print("=" * 60)
        
        total_start = time.time()
        
        for excel_path in excel_paths:
            try:
                results = self.import_excel_file(excel_path, if_exists=if_exists)
                all_results[excel_path] = results
            except Exception as e:
                print(f"❌ 文件导入失败 [{excel_path}]: {e}")
                all_results[excel_path] = {"error": str(e)}
        
        total_elapsed = time.time() - total_start
        
        # 汇总统计
        total_files = len(excel_paths)
        total_sheets = sum(len(r) for r in all_results.values() if isinstance(r, dict))
        total_rows = sum(
            sum(v for v in r.values() if isinstance(v, int) and v > 0)
            for r in all_results.values() if isinstance(r, dict)
        )
        
        print("\n" + "=" * 60)
        print(f"📊 导入汇总:")
        print(f"   文件数: {total_files}")
        print(f"   Sheet数: {total_sheets}")
        print(f"   总行数: {total_rows}")
        print(f"   总耗时: {total_elapsed:.2f}s")
        print("=" * 60)
        
        return all_results
    
    def _clean_filename(self, name: str) -> str:
        """清理文件名作为表名前缀"""
        # 移除空格和特殊字符
        clean = name.replace(" ", "_").replace("-", "_").replace(".", "_")
        clean = "".join(c for c in clean if c.isalnum() or c == "_")
        return clean.lower()
    
    def _clean_name(self, name: str) -> str:
        """清理名称"""
        clean = name.replace(" ", "_").replace("-", "_").replace(".", "_")
        clean = "".join(c for c in clean if c.isalnum() or c == "_")
        if clean and clean[0].isdigit():
            clean = "_" + clean
        return clean.lower()
    
    def get_import_summary(self) -> Dict[str, Dict]:
        """获取导入汇总信息"""
        return self.imported_tables
    
    def print_import_summary(self):
        """打印导入汇总"""
        if not self.imported_tables:
            print("暂无导入记录")
            return
        
        print("\n📋 导入表汇总:")
        print("-" * 80)
        print(f"{'表名':<40} {'行数':>10} {'列数':>10} {'来源Sheet':<20}")
        print("-" * 80)
        
        for table_name, info in self.imported_tables.items():
            print(f"{table_name:<40} {info['row_count']:>10} {info['column_count']:>10} {info['sheet_name']:<20}")
        
        print("-" * 80)
        print(f"共 {len(self.imported_tables)} 个表")


def import_config_files(
    db_host: str = "localhost",
    db_port: int = 5432,
    db_name: str = "test_db",
    db_user: str = "postgres",
    db_password: str = "123456"
) -> bool:
    """
    导入所有配置文件到数据库
    
    Args:
        db_host: 数据库主机
        db_port: 数据库端口
        db_name: 数据库名称
        db_user: 用户名
        db_password: 密码
    
    Returns:
        bool: 是否成功
    """
    # 配置文件列表
    base_path = Path(__file__).parent.parent / "test_files"
    excel_files = [
        str(base_path / "BC_S5.xlsx"),
        str(base_path / "BC_S9.xlsx"),
        str(base_path / "OC Paste_S1.xlsx")
    ]
    
    # 检查文件是否存在
    existing_files = []
    for f in excel_files:
        if Path(f).exists():
            existing_files.append(f)
        else:
            print(f"⚠️ 文件不存在: {f}")
    
    if not existing_files:
        print("❌ 没有找到任何配置文件")
        return False
    
    # 创建数据库连接
    db = DatabaseConnection(
        host=db_host,
        port=db_port,
        database=db_name,
        user=db_user,
        password=db_password
    )
    
    # 测试连接
    conn_result = db.test_connection()
    if not conn_result["success"]:
        print(f"❌ 数据库连接失败: {conn_result['message']}")
        return False
    
    # 创建导入器并执行导入
    importer = ExcelImporter(db)
    results = importer.import_multiple_files(existing_files)
    
    # 打印汇总
    importer.print_import_summary()
    
    # 关闭连接
    db.close()
    
    return True


if __name__ == "__main__":
    import_config_files()
