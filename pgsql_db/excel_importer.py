"""
Excel数据导入模块
将Excel配置文件的各个sheet导入PostgreSQL数据库
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import time

from .db_connection import DatabaseConnection
from . import table_mapping


class ExcelImporter:
    """Excel数据导入器"""
    
    def __init__(self, db: DatabaseConnection):
        """
        初始化Excel导入器
        
        参数：
            db: 数据库连接实例
        """
        self.db = db
        self.imported_tables: Dict[str, Dict] = {}

    @staticmethod
    def _patch_openpyxl_font_family():
        """修补 openpyxl 字体 family 校验，允许超出上限14的值（钳位到2）"""
        import openpyxl.styles.fonts as _fonts
        _orig = _fonts.Font.__init__
        if getattr(_orig, '_patched', False):
            return
        def _patched_init(self, *args, **kwargs):
            if 'family' in kwargs and kwargs['family'] is not None:
                try:
                    if float(kwargs['family']) > 14:
                        kwargs['family'] = 2
                except (ValueError, TypeError):
                    pass
            _orig(self, *args, **kwargs)
        _patched_init._patched = True
        _fonts.Font.__init__ = _patched_init

    @staticmethod
    def _derive_config_type(config_name: str) -> str:
        """根据 config_name 推导配置类型。

        规则：
        - 统一使用配置名本身（去掉路径和扩展名）作为 config_type
          例如 config/BC_S9.xlsx -> BC_S9
        """
        if not config_name:
            return 'UNKNOWN'

        # 去掉可能的路径前缀和扩展名，只保留配置标识
        basename = Path(config_name).stem
        if not basename:
            return 'UNKNOWN'

        return basename

    @staticmethod
    def _scan_csv_overrides(excel_path: str) -> Dict[str, pd.DataFrame]:
        """扫描 Excel 同目录下的 CSV 文件作为配置表覆盖"""
        overrides = {}
        try:
            config_dir = Path(excel_path).parent
            for csv_file in sorted(config_dir.glob('*.csv')):
                sheet_name = csv_file.stem
                try:
                    overrides[sheet_name] = pd.read_csv(str(csv_file))
                except Exception as e:
                    print(f"  ⚠️ CSV 文件读取失败: {csv_file.name} - {e}")
        except Exception as e:
            print(f"  ⚠️ CSV 覆盖扫描失败: {e}")
        return overrides

    def import_excel_file(
        self,
        excel_path: str,
        prefix: str = None,
        if_exists: str = "replace",
        config_name: str = None
    ) -> Dict[str, int]:
        """
        导入单个Excel文件的所有sheet到数据库
        
        参数：
            excel_path: Excel文件路径
            prefix: 表名前缀（默认使用文件名）
            if_exists: 如果表存在的处理方式
            config_name: 配置文件标识（如 BC_S5, BC_S9），将添加到每个表中
        
        返回：
            dict: 每个sheet导入的行数
        """
        path = Path(excel_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {excel_path}")
        
        # 使用文件名作为前缀
        if prefix is None:
            prefix = self._clean_filename(path.stem)
        
        # 如果没有提供config_name，使用文件名作为config_name
        if config_name is None:
            config_name = path.stem  # 例如: BC_S5.xlsx -> BC_S5
        
        # 推导配置类型
        config_type = self._derive_config_type(config_name)

        print(f"\n📂 导入Excel文件: {path.name}")
        print(f"🏷️  配置标识: {config_name}  类型: {config_type} (将通过 config_name/config_type 字段区分)")
        print(f"📋 表名规则: 统一配置表名 (cfg_xxx)")
        print("-" * 50)
        
        # 读取所有sheet
        print(f"  ⏳ 正在读取Excel文件...", end="", flush=True)
        try:
            xl = pd.ExcelFile(excel_path)
        except ValueError:
            # openpyxl 对某些Excel样式不兼容（如字体family值超出上限14），
            # 临时放宽校验后重试
            self._patch_openpyxl_font_family()
            xl = pd.ExcelFile(excel_path)
        print(f" 完成 ({len(xl.sheet_names)} 个sheet)")
        results = {}
        
        start_time = time.time()
        
        for sheet_name in xl.sheet_names:
            try:
                df = xl.parse(sheet_name)
                
                # 构建表名: 使用统一配置表名（同结构同表），通过 config_name 字段区分不同配置
                table_name = table_mapping.get_config_table_name(sheet_name)
                
                # 空表也要创建（只要有列名）
                if df.empty:
                    # 检查是否有列定义
                    if len(df.columns) > 0:
                        print(f"  📋 Sheet [{sheet_name}] 为空表，创建表结构 ({len(df.columns)} 列)")
                        self.db.create_table_from_df(df, table_name, if_exists, config_name=config_name, config_type=config_type)
                        results[sheet_name] = 0
                    else:
                        print(f"  ⚠️ Sheet [{sheet_name}] 无数据且无列定义，跳过")
                        results[sheet_name] = -1
                    continue
                
                # 写入数据库，添加config_name和config_type字段
                self.db.create_table_from_df(df, table_name, if_exists, config_name=config_name, config_type=config_type)
                results[sheet_name] = len(df)
                
                # 记录导入信息
                self.imported_tables[table_name] = {
                    "source_file": str(path),
                    "sheet_name": sheet_name,
                    "row_count": len(df),
                    "column_count": len(df.columns),
                    "config_name": config_name
                }
                
            except Exception as e:
                print(f"  ❌ Sheet [{sheet_name}] 导入失败: {e}")
                results[sheet_name] = -1

        # 扫描并导入 CSV 覆盖文件
        csv_overrides = self._scan_csv_overrides(excel_path)
        if csv_overrides:
            print(f"\n  {'─' * 50}")
            print(f"  📄 发现 {len(csv_overrides)} 个 CSV 覆盖文件:")
        for csv_sheet_name, csv_df in csv_overrides.items():
            try:
                csv_table_name = table_mapping.get_config_table_name(csv_sheet_name)
                is_override = csv_sheet_name in results

                if csv_df.empty and len(csv_df.columns) > 0:
                    if is_override:
                        print(f"  🔄 [CSV 覆盖] {csv_sheet_name} 为空表，创建表结构 ({len(csv_df.columns)} 列)")
                    else:
                        print(f"  ➕ [CSV 新增] {csv_sheet_name} 为空表，创建表结构 ({len(csv_df.columns)} 列)")
                    self.db.create_table_from_df(csv_df, csv_table_name, if_exists, config_name=config_name, config_type=config_type)
                    results[csv_sheet_name] = 0
                elif not csv_df.empty:
                    self.db.create_table_from_df(csv_df, csv_table_name, if_exists, config_name=config_name, config_type=config_type)
                    results[csv_sheet_name] = len(csv_df)
                    self.imported_tables[csv_table_name] = {
                        "source_file": str(Path(excel_path).parent / f"{csv_sheet_name}.csv"),
                        "sheet_name": csv_sheet_name,
                        "row_count": len(csv_df),
                        "column_count": len(csv_df.columns),
                        "config_name": config_name
                    }
                    if is_override:
                        print(f"  🔄 [CSV 覆盖] {csv_sheet_name} → {csv_table_name} ({len(csv_df)} 行) ← 替代Excel版本")
                    else:
                        print(f"  ➕ [CSV 新增] {csv_sheet_name} → {csv_table_name} ({len(csv_df)} 行)")
            except Exception as e:
                print(f"  ❌ CSV [{csv_sheet_name}] 导入失败: {e}")
                results[csv_sheet_name] = -1
        if csv_overrides:
            print(f"  {'─' * 50}")

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
        
        参数：
            excel_paths: Excel文件路径列表
            if_exists: 如果表存在的处理方式
        
        返回：
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
    
    参数：
        db_host: 数据库主机
        db_port: 数据库端口
        db_name: 数据库名称
        db_user: 用户名
        db_password: 密码
    
    返回：
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
