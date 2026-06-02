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
        """[已废弃] 数据库导入已改用 ``load_excel_file_with_csv_priority``。"""
        return {}

    @staticmethod
    def load_excel_file_with_csv_priority(excel_path: str) -> Dict[str, pd.DataFrame]:
        """读取 Excel 配置，并让同目录同名 CSV 无条件优先。

        规则与文件模式 ``load_configuration`` 的原始数据读取阶段保持一致：
        - 以 Excel 的 sheet 名作为匹配权威；
        - 同目录存在同名 CSV（大小写不敏感，含 ``.CSV``）时直接使用 CSV；
        - CSV 没有对应 Excel sheet 时作为扩展表导入；
        - CSV 来源仅限 Excel 同目录一层，由 ``ConfigDir`` 校验大小写不敏感唯一性。
        """
        from src.core.run.config_dir import ConfigDir

        cfg_dir = ConfigDir.from_excel_path(excel_path)
        xl = None
        try:
            try:
                xl = pd.ExcelFile(str(cfg_dir.excel_path))
            except ValueError:
                ExcelImporter._patch_openpyxl_font_family()
                xl = pd.ExcelFile(str(cfg_dir.excel_path))

            sheet_names: tuple[str, ...] = tuple(xl.sheet_names)
            sheet_data: Dict[str, pd.DataFrame] = {}

            for sheet_name in sheet_names:
                csv_path = cfg_dir.csv_for_sheet(sheet_name)
                if csv_path is not None:
                    sheet_data[sheet_name] = pd.read_csv(csv_path)
                else:
                    sheet_data[sheet_name] = xl.parse(sheet_name)

            loaded_lower = {s.lower() for s in sheet_names}
            for stem_lower, csv_path in cfg_dir.csv_map.items():
                if stem_lower not in loaded_lower:
                    sheet_data[csv_path.stem] = pd.read_csv(csv_path)

            return sheet_data
        finally:
            if xl is not None:
                try:
                    xl.close()
                except Exception:
                    pass

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

        
        # 读取所有配置表：同目录同名 CSV 无条件优先于 Excel sheet。
        sheet_data = self.load_excel_file_with_csv_priority(excel_path)
        results = {}
        
        start_time = time.time()
        
        for sheet_name, df in sheet_data.items():
            try:
                # 构建表名: 使用统一配置表名（同结构同表），通过 config_name 字段区分不同配置
                table_name = table_mapping.get_config_table_name(sheet_name)
                
                # 空表也要创建（只要有列名）
                if df.empty:
                    # 检查是否有列定义
                    if len(df.columns) > 0:
                        write_ok = self.db.create_table_from_df(
                            df,
                            table_name,
                            if_exists,
                            config_name=config_name,
                            config_type=config_type,
                        )
                        if not write_ok:
                            results[sheet_name] = -1
                            continue
                        results[sheet_name] = 0
                    else:
                        results[sheet_name] = -1
                    continue
                
                # 写入数据库，添加config_name和config_type字段
                write_ok = self.db.create_table_from_df(
                    df,
                    table_name,
                    if_exists,
                    config_name=config_name,
                    config_type=config_type,
                )
                if not write_ok:
                    results[sheet_name] = -1
                    continue
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
                results[sheet_name] = -1

        elapsed = time.time() - start_time
        total_rows = sum(r for r in results.values() if r > 0)
        
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
        
        
        total_start = time.time()
        
        for excel_path in excel_paths:
            try:
                results = self.import_excel_file(excel_path, if_exists=if_exists)
                all_results[excel_path] = results
            except Exception as e:
                all_results[excel_path] = {"error": str(e)}
        
        total_elapsed = time.time() - total_start
        
        # 汇总统计
        total_files = len(excel_paths)
        total_sheets = sum(len(r) for r in all_results.values() if isinstance(r, dict))
        total_rows = sum(
            sum(v for v in r.values() if isinstance(v, int) and v > 0)
            for r in all_results.values() if isinstance(r, dict)
        )
        
        
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
            return
        
        
        for table_name, info in self.imported_tables.items():
            pass
        


def import_config_files(
    db_host: Optional[str] = None,
    db_port: Optional[int] = None,
    db_name: Optional[str] = None,
    db_user: Optional[str] = None,
    db_password: Optional[str] = None
) -> bool:
    """
    导入所有配置文件到数据库

    未显式传入的字段将从 ``config/defaults.yaml`` 的 ``database:`` 节点读取。

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
            pass
    
    if not existing_files:
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
