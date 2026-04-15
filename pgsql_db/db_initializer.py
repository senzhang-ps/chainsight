"""
数据库初始化模块
提供数据库存在性检测、自动创建数据库和配置表的功能
支持多配置文件管理（BC_S5, BC_S9等），通过config_name字段区分不同配置的数据
"""

from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
import os


class DatabaseInitializer:
    """数据库初始化器 - 检测并自动创建数据库和配置表"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "test_db",
        user: str = "postgres",
        password: str = "123456",
        config_search_paths: Optional[List[Path]] = None
    ):
        """
        初始化数据库初始化器
        
        参数：
            host: 数据库主机
            port: 数据库端口
            database: 数据库名称
            user: 用户名
            password: 密码
            config_search_paths: 配置文件搜索路径列表
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        
        # 默认配置文件搜索路径
        if config_search_paths is None:
            project_root = Path(__file__).parent.parent
            self.config_search_paths = [
                project_root / "config",      # 优先从 config 目录查找
                project_root / "test_files",  # 其次从 test_files 目录查找
                project_root,                  # 最后从项目根目录查找
            ]
        else:
            self.config_search_paths = config_search_paths
        
        self._db = None
        self._importer = None
    
    @property
    def db(self):
        """延迟加载数据库连接"""
        if self._db is None:
            from .db_connection import DatabaseConnection
            self._db = DatabaseConnection(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
        return self._db
    
    def close(self):
        """关闭数据库连接"""
        if self._db is not None:
            self._db.close()
            self._db = None
        self._importer = None
    
    @property
    def importer(self):
        """延迟加载Excel导入器"""
        if self._importer is None:
            from .excel_importer import ExcelImporter
            self._importer = ExcelImporter(self.db)
        return self._importer
    
    def check_database_exists(self) -> bool:
        """
        检测数据库是否存在
        
        返回：
            bool: 数据库是否存在
        """
        return self.db.database_exists()
    
    def create_database_if_not_exists(self) -> Tuple[bool, str]:
        """
        如果数据库不存在则创建
        
        返回：
            Tuple[bool, str]: (是否成功, 消息)
        """
        if self.check_database_exists():
            return True, f"数据库已存在: {self.database}"
        
        success = self.db.create_database_if_not_exists()
        if success:
            return True, f"已创建数据库: {self.database}"
        else:
            return False, f"创建数据库失败: {self.database}"
    
    def check_config_tables_exist(self, config_name: str) -> Tuple[bool, int]:
        """
        检测配置表是否存在
        
        参数：
            config_name: 配置名称（如 BC_S5）
            
        返回：
            Tuple[bool, int]: (是否存在, 表数量)
        """
        # 按“同结构同表”规则：不再以配置前缀分表，改为检查是否存在包含该 config_name 的数据
        exists, rows = self.check_config_data_exists(config_name)
        return exists, rows
    
    def check_config_data_exists(self, config_name: str, table_name: str = None) -> Tuple[bool, int]:
        """
        检测数据库中是否存在指定配置的数据
        
        通过检查表中的 config_name 字段来确定数据归属
        
        参数：
            config_name: 配置名称（如 BC_S5, BC_S9）
            table_name: 可选，指定检查的表名，不指定则检查所有配置表
            
        返回：
            Tuple[bool, int]: (是否存在, 数据行数)
        """
        existing_tables = self.db.get_all_tables()
        
        # 按“同结构同表”规则：配置数据存在于统一表中，通过 config_name 字段过滤
        config_tables = list(existing_tables)
        
        # 如果指定了表名，只检查该表（表名即为统一表名）
        if table_name:
            target_table = table_name.lower()
            if target_table not in config_tables:
                return False, 0
            config_tables = [target_table]
        
        total_rows = 0
        for tbl in config_tables:
            try:
                # 检查表是否有config_name列
                cols = self.db._get_column_types(tbl)
                if 'config_name' in cols:
                    # 有config_name列，检查数据（兼容带/不带路径前缀的config_name）
                    config_basename = config_name.split('/')[-1] if '/' in config_name else config_name
                    result = self.db.execute_query(
                        f'SELECT COUNT(*) FROM "{tbl}" WHERE config_name = %s OR config_name = %s',
                        (config_name, config_basename)
                    )
                    total_rows += result[0][0] if result else 0
                else:
                    # 没有config_name列 -> 旧格式表，不计入统一配置检测
                    # 旧格式表（如 bc_s5_*）无法被 _load_config_from_database 读取，
                    # 计入会导致误判"配置已存在"而跳过自动导入
                    pass
            except Exception:
                continue
        
        return total_rows > 0, total_rows
    
    def get_available_configs(self) -> List[str]:
        """
        获取数据库中所有可用的配置名称
        
        按“同结构同表”规则：配置数据都在统一表中，通过列 `config_name` 区分。
        这里会扫描所有非输出表，收集其中出现过的 config_name。
        
        返回：
            List[str]: 配置名称列表
        """
        existing_tables = self.db.get_all_tables()
        
        configs = set()
        for table in existing_tables:
            # 排除模块输出表和系统表
            if table.startswith(('module', 'orchestrator', 'summary', 'analysis', 'pg_')):
                continue
            
            try:
                cols = self.db._get_column_types(table)
            except Exception:
                continue
            
            if 'config_name' not in cols:
                continue
            
            try:
                rows = self.db.execute_query(
                    f'SELECT DISTINCT config_name FROM "{table}" WHERE config_name IS NOT NULL'
                )
                for r in rows or []:
                    if r and r[0]:
                        configs.add(str(r[0]))
            except Exception:
                continue
        
        return sorted(configs)
    
    def find_config_file(self, config_name: str) -> Optional[Path]:
        """
        查找配置文件
        
        参数：
            config_name: 配置名称
            
        返回：
            Path or None: 配置文件路径，未找到返回None
        """
        # 尝试多种文件名格式
        possible_names = [
            f"{config_name}.xlsx",
            f"{config_name.replace('_', ' ')}.xlsx",
            f"{config_name.replace(' ', '_')}.xlsx",
        ]
        
        for search_path in self.config_search_paths:
            for name in possible_names:
                config_file = search_path / name
                if config_file.exists():
                    return config_file
        
        return None
    
    def import_config_from_excel(
        self, 
        config_name: str, 
        config_file: Optional[Path] = None,
        if_exists: str = 'replace'
    ) -> Tuple[bool, Dict[str, int]]:
        """
        从Excel导入配置到数据库
        
        参数：
            config_name: 配置名称
            config_file: 配置文件路径（可选，不提供则自动查找）
            if_exists: 表存在时的处理方式 ('replace', 'append', 'skip')
            
        返回：
            Tuple[bool, Dict]: (是否成功, 导入结果字典)
        """
        # 查找配置文件
        if config_file is None:
            config_file = self.find_config_file(config_name)
        
        if config_file is None:
            return False, {"error": f"未找到配置文件: {config_name}.xlsx"}
        
        # 导入Excel：按“同结构同表”规则写入统一表，并添加 config_name 字段以区分不同配置
        results = self.importer.import_excel_file(
            str(config_file), 
            prefix=None,
            if_exists=if_exists,
            config_name=config_name
        )
        
        if not results or all(v < 0 for v in results.values()):
            return False, results or {"error": "导入失败"}
        
        return True, results
    
    def initialize(
        self, 
        config_name: Optional[str] = None,
        auto_import_config: bool = True,
        verbose: bool = True
    ) -> Dict[str, any]:
        """
        完整初始化流程：检测数据库、创建数据库、检测配置表、导入配置
        
        参数：
            config_name: 配置名称（如 BC_S5），如果提供则检测并导入配置
            auto_import_config: 是否自动导入缺失的配置
            verbose: 是否输出详细信息
            
        返回：
            Dict: 初始化结果
        """
        result = {
            "success": True,
            "database_created": False,
            "database_exists": False,
            "config_imported": False,
            "config_tables_count": 0,
            "messages": []
        }
        
        def log(msg):
            result["messages"].append(msg)
            if verbose:
                pass
        
        # ========== 步骤1: 检测数据库 ==========
        log(f"\n🔍 检测数据库 '{self.database}'...")
        
        if not self.check_database_exists():
            log(f"⚠️ 数据库 '{self.database}' 不存在，正在创建...")
            success, msg = self.create_database_if_not_exists()
            if not success:
                log(f"❌ {msg}")
                result["success"] = False
                return result
            log(f"✅ {msg}")
            result["database_created"] = True
        else:
            log(f"✅ 数据库已存在: {self.database}")
        
        result["database_exists"] = True
        
        # ========== 步骤2: 测试连接 ==========
        log("🔍 测试数据库连接...")
        conn_result = self.db.test_connection()
        if not conn_result["success"]:
            log(f"❌ 数据库连接失败: {conn_result['message']}")
            result["success"] = False
            return result
        log(f"✅ 数据库连接成功 (版本: {conn_result['version'][:40]}...)")
        
        # ========== 步骤2.5: 初始化 checkpoint 表 ==========
        log("🔍 初始化 checkpoint 表...")
        from pgsql_db.checkpoint import ensure_checkpoint_table
        try:
            ensure_checkpoint_table(self.db)
            log("✅ checkpoint 表已就绪")
        except Exception as e:
            log(f"⚠️ 初始化 checkpoint 表时出现警告: {e}")
        
        # ========== 步骤3: 检测配置表（如果提供配置名） ==========
        if config_name:
            log(f"\n🔍 检测配置表 '{config_name}'...")
            
            exists, count = self.check_config_tables_exist(config_name)
            
            if not exists:
                if auto_import_config:
                    log(f"⚠️ 数据库中未找到配置 '{config_name}' 的表，尝试从Excel导入...")
                    
                    config_file = self.find_config_file(config_name)
                    if config_file is None:
                        log(f"❌ 未找到配置文件 '{config_name}.xlsx'")
                        log(f"   搜索路径: {[str(p) for p in self.config_search_paths]}")
                        result["success"] = False
                        return result
                    
                    log(f"📁 找到配置文件: {config_file}")
                    
                    success, import_results = self.import_config_from_excel(
                        config_name, config_file
                    )
                    
                    if not success:
                        log(f"❌ 导入配置文件失败")
                        result["success"] = False
                        return result
                    
                    imported_count = len([r for r in import_results.values() if r >= 0])
                    log(f"✅ 已导入配置表 {imported_count} 个")
                    result["config_imported"] = True
                    result["config_tables_count"] = imported_count
                else:
                    log(f"⚠️ 数据库中未找到配置 '{config_name}' 的表")
                    result["config_tables_count"] = 0
            else:
                log(f"✅ 找到 {count} 个配置表")
                result["config_tables_count"] = count
        
        return result
    
    def get_status_report(self, config_name: Optional[str] = None) -> str:
        """
        获取数据库状态报告
        
        参数：
            config_name: 配置名称（可选）
            
        返回：
            str: 状态报告
        """
        lines = []
        lines.append("=" * 60)
        lines.append("📊 数据库状态报告")
        lines.append("=" * 60)
        
        # 数据库信息
        lines.append(f"🔌 数据库: {self.host}:{self.port}/{self.database}")
        lines.append(f"👤 用户: {self.user}")
        
        # 检测数据库
        db_exists = self.check_database_exists()
        lines.append(f"📁 数据库存在: {'✅ 是' if db_exists else '❌ 否'}")
        
        if db_exists:
            # 获取表统计
            all_tables = self.db.get_all_tables()
            lines.append(f"📋 总表数: {len(all_tables)}")
            
            # 分类统计
            config_tables = [t for t in all_tables if not t.startswith(('module', 'orchestrator', 'summary', 'analysis'))]
            module_tables = [t for t in all_tables if t.startswith('module')]
            orch_tables = [t for t in all_tables if t.startswith('orchestrator')]
            summary_tables = [t for t in all_tables if t.startswith('summary')]
            
            lines.append(f"   - 配置表: {len(config_tables)}")
            lines.append(f"   - 模块输出表: {len(module_tables)}")
            lines.append(f"   - Orchestrator表: {len(orch_tables)}")
            lines.append(f"   - Summary表: {len(summary_tables)}")
            
            # 检查特定配置
            if config_name:
                exists, count = self.check_config_tables_exist(config_name)
                lines.append(f"\n📋 配置 '{config_name}':")
                lines.append(f"   - 表存在: {'✅ 是' if exists else '❌ 否'}")
                lines.append(f"   - 表数量: {count}")
                
                # 检查配置文件
                config_file = self.find_config_file(config_name)
                if config_file:
                    lines.append(f"   - Excel文件: ✅ {config_file}")
                else:
                    lines.append(f"   - Excel文件: ❌ 未找到")
        
        lines.append("=" * 60)
        return "\n".join(lines)


def initialize_database(
    config_name: Optional[str] = None,
    host: str = "localhost",
    port: int = 5432,
    database: str = "test_db",
    user: str = "postgres",
    password: str = "123456",
    auto_import: bool = True,
    verbose: bool = True
) -> Dict[str, any]:
    """
    便捷函数：初始化数据库
    
    参数：
        config_name: 配置名称（如 BC_S5）
        host: 数据库主机
        port: 数据库端口
        database: 数据库名称
        user: 用户名
        password: 密码
        auto_import: 是否自动导入缺失的配置
        verbose: 是否输出详细信息
        
    返回：
        Dict: 初始化结果
    """
    initializer = DatabaseInitializer(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password
    )
    return initializer.initialize(
        config_name=config_name,
        auto_import_config=auto_import,
        verbose=verbose
    )
