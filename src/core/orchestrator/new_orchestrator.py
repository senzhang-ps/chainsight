import logging
from datetime import date, datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Orchestrator:
    """调度总管：配置加载 + 状态持久化 + 模块调度。

    职责边界：
    - 配置加载：从 DB / yaml 读取系统配置和计算数据，按 module 分发
    - 模块调度：日期迭代、模块实例化与执行顺序
    - 状态持久化：每日状态快照的保存与恢复（通过 Snapshot）
    - 不管：可变状态持有（交给 StateContext）、业务计算（交给 Module）
    """

    def __init__(self, start_date, end_date,
                 config_path=None, output_path=None,
                 config_dict=None, engine='pandas'):
        self.start_date = start_date if isinstance(start_date, date) else pd.Timestamp(start_date).date()
        self.end_date = end_date if isinstance(end_date, date) else pd.Timestamp(end_date).date()
        self.module_idx = [1, 3, 4, 5, 6]
        self.engine = engine
        self.output_path = output_path or './output'

        # ── DB 连接 ──
        self.db = None
        self._config_name = None
        self._run_id = None
        self._init_db()

        # ── 配置加载 ──
        self.sys_config: dict = {}
        self.all_config: dict = {}

        if config_dict is not None:
            # 兼容路径：DB 模式下调用方已从 DB 加载好 config_dict
            self.all_config = config_dict
        elif config_path is not None:
            # 主路径：从 Excel 文件加载配置
            from ..main_integration.config_loader import load_configuration
            logger.info(f"Orch: 从配置文件加载: {config_path}")
            self.all_config = load_configuration(config_path)
            # 从文件名提取 config_name（不含扩展名）
            self._config_name = Path(config_path).stem
        else:
            logger.warning("Orch: 未提供 config_path 或 config_dict，配置为空")

        # ── 随机种子 ──
        from ..main_integration.seed import set_module_seeds
        set_module_seeds(self.all_config)

        # ── 输出目录 ──
        self.build_output_folder()
        self.all_results = {}
        self.sim_dates = []

        # ── 持久化管理 ──
        from .persistence_manager import PersistenceManager
        self.persistence = PersistenceManager(self)

    # ══════════════════════════════════════════
    # config_name / run_id
    # ══════════════════════════════════════════

    @property
    def config_name(self) -> str | None:
        """配置名称，默认取配置文件名（不含扩展名）。

        从 config_path 加载时自动赋值为文件名 stem；
        DB 模式下可由调用方显式赋值（orch.config_name = 'BC_S5'）。
        """
        return self._config_name

    @config_name.setter
    def config_name(self, value: str):
        self._config_name = value

    @property
    def run_id(self) -> str:
        """运行唯一标识，格式: {config_name}_{YYYYMMDD_HHMMSS}。

        首次访问时自动生成，后续访问返回同一值。
        若 config_name 未设置则回退为 'unknown'。
        """
        if self._run_id is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = self.config_name or 'unknown'
            self._run_id = f"{name}_{ts}"
            logger.info(f"run_id 已生成: {self._run_id}")
        return self._run_id

    # ══════════════════════════════════════════
    # 配置加载
    # ══════════════════════════════════════════

    def _load_sys_config(self) -> dict:
        """从 yaml 读取系统配置（DB连接参数、全局运行参数）。

        收敛原 settings.py 职责。
        """
        import yaml
        config_path = Path(__file__).resolve().parents[3] / 'config' / 'defaults.yaml'
        if not config_path.exists():
            logger.warning(f"系统配置文件不存在: {config_path}")
            return {}
        with open(config_path, 'r', encoding='utf-8') as f:
            raw = yaml.safe_load(f)
        # 提取 database 节 + 其他系统参数
        sys_cfg = {}
        if 'database' in raw:
            sys_cfg['database'] = raw['database']
        # 扩展：其他 yaml 中的系统级配置项也可在此提取
        for key in ('engine', 'resource', 'shared'):
            if key in raw:
                sys_cfg[key] = raw[key]
        logger.info(f"系统配置已加载: {list(sys_cfg.keys())}")
        return sys_cfg

    def _init_db(self):
        """从 yaml 配置创建数据库连接。

        读取 config/defaults.yaml 的 database 节，
        创建 src.core.db.pgsql.db.DB 实例并赋值到 self.db。
        """
        sys_cfg = self._load_sys_config()
        if not sys_cfg:
            return

        self.sys_config = sys_cfg
        db_cfg = sys_cfg.get('database')
        if not db_cfg:
            return

        try:
            from ..db.pgsql.db import DB
            self.db = DB(
                host=db_cfg.get('host', 'localhost'),
                port=int(db_cfg.get('port', 5432)),
                database=db_cfg.get('database', 'test_db'),
                user=db_cfg.get('user', 'postgres'),
                password=str(db_cfg.get('password', '')),
            )
            # 验证连接
            self.db.connect()
            logger.info("🗄️ 数据库连接已建立")
        except Exception as e:
            logger.warning(f"🗄️ 数据库连接失败（将使用文件模式）: {e}")
            self.db = None

    def _load_calc_datas(self) -> dict:
        """从 DB 的 cfg_* 表读取计算数据 → 标准化 → 返回 {sheet_name: DataFrame}。

        收敛原 db_config.py + config_loader.py 的重复逻辑。
        映射表引用 table_mapping.py 作为唯一来源。
        """
        if self.db is None:
            raise RuntimeError("_load_calc_datas 需要 db 连接")

        from ...core.db.pgsql.table_mapping import CONFIG_TABLE_MAPPING
        from ...utils.normalization import normalize_identifiers

        # 逆向映射：db_table_name → sheet_name
        reverse_map = {v: k for k, v in CONFIG_TABLE_MAPPING.items()}

        all_tables = self.db.read('_tables') if hasattr(self.db, 'read_tables') else []
        if not all_tables:
            # 回退：直接按映射表逐个读取
            all_tables = [f'cfg_{v}' for v in CONFIG_TABLE_MAPPING.values()]

        config_dict = {}
        for table_name in all_tables:
            if not table_name.startswith('cfg_'):
                continue
            db_key = table_name[4:]  # 去掉 cfg_ 前缀
            sheet_name = reverse_map.get(db_key, db_key)

            try:
                df = self.db.read(table_name, config_name=self.config_name)
            except Exception as e:
                logger.warning(f"读取配置表 {table_name} 失败: {e}")
                continue
            
            config_dict[sheet_name] = df if df is not None else pd.DataFrame()

        # 补齐必要表
        required = ['M1_InitialInventory', 'Global_SpaceCapacity',
                     'Global_Network', 'Global_LeadTime', 'Global_DemandPriority']
        for sheet in required:
            if sheet not in config_dict:
                logger.warning(f"缺少必要配置表: {sheet}")
                config_dict[sheet] = pd.DataFrame()

        logger.info(f"计算数据已加载: {len(config_dict)} 个表")
        return config_dict

    @staticmethod
    def _clean_db_columns(df: pd.DataFrame) -> pd.DataFrame:
        """清理 DB 元数据列和空列。"""
        drop_cols = [c for c in df.columns
                     if c in ('config_name', 'config_type', 'db_write_time')
                     or c.lower().startswith('unnamed')]
        if drop_cols:
            df = df.drop(columns=drop_cols, errors='ignore')
        return df

    # ══════════════════════════════════════════
    # 模块配置分发
    # ══════════════════════════════════════════

    def load_datas(self, module) -> None:
        """按 module.schema 从 self.all_config 加载数据到 module.datas。"""
        from ...modules.module import Module
        if not isinstance(module, Module):
            raise TypeError(f"load_datas 期望 Module 实例，收到 {type(module).__name__}")
        schema = getattr(module, 'schema', {})
        if not schema:
            return
        datas = {}
        for sheet_name, col_schema in schema.items():
            df = self.all_config.get(sheet_name, pd.DataFrame())
            datas[sheet_name] = df
        module.datas = datas

    def get_module_config(self, module_name: str) -> dict:
        """从 self.sys_config 提取模块参数。

        模块在 __init__ 中调用 orchestrator.get_module_config('M1') 获取
        属于自己的系统级参数（如随机种子、引擎选择等）。
        """
        if not self.sys_config:
            return {}
        # 按 module_name 在 sys_config 中查找对应的配置节
        # yaml 结构示例: {M1: {param1: val1}, M4: {...}, shared: {...}}
        result = self.sys_config.get('shared', {})
        result.update(self.sys_config.get(module_name, {}))
        return result


    # ══════════════════════════════════════════
    # 调度
    # ══════════════════════════════════════════

    def iter_dates(self, actual_start_date=None):
        """日期迭代器。"""
        sim_start = actual_start_date or self.start_date
        sim_dates = pd.date_range(sim_start, self.end_date, freq='D')
        self.sim_dates = sim_dates
        logger.info(f"仿真日期范围: {len(sim_dates)} 天")
        pbar = tqdm(enumerate(sim_dates, 1), total=len(sim_dates),
                     desc='仿真进度', unit='天', ncols=80, leave=True)
        for i, current_date in pbar:
            progress_info = f"第 {i}/{len(sim_dates)} 天"
            pbar.write(f"{'=' * 20} {progress_info}: {current_date.strftime('%Y-%m-%d')} {'=' * 20}")
            pbar.set_postfix(date=current_date.strftime('%Y-%m-%d'), day=progress_info)
            yield i, current_date

    def get_output(self, module_name: str) -> Path:
        """返回模块输出目录路径。"""
        if not hasattr(self, '_output_dirs') or module_name not in self._output_dirs:
            raise KeyError(f"未找到模块输出路径: {module_name}")
        return self._output_dirs[module_name]

    def build_output_folder(self):
        """创建输出目录。"""
        self._output_dirs = {}
        for i in self.module_idx:
            path = Path(self.output_path) / f'module{i}'
            path.mkdir(parents=True, exist_ok=True)
            self._output_dirs[f'module{i}'] = path

    # ══════════════════════════════════════════
    # 状态持久化（委托到 PersistenceManager）
    # ══════════════════════════════════════════

    def save_daily_state(self, ctx, date_str):
        """委托到 PersistenceManager 写 DB。"""
        self.persistence.save_daily_state(ctx, date_str)

    def save_module_output(self, module, sim_date: str):
        """委托到 PersistenceManager 写 DB。"""
        self.persistence.save_module_output(module, sim_date)

    def restore_state(self, ctx, run_id: str, sim_date: str):
        """从快照表恢复状态到 StateContext。

        Args:
            ctx: StateContext 实例
            run_id: 运行标识
            sim_date: 要恢复到的日期
        """
        if self.db is None:
            return
        snapshot = self.db.snapshot
        snapshot.restore(run_id, sim_date, ctx)

    def save_checkpoint(self, run_id: str, status: str = 'running'):
        """保存/更新运行元信息。"""
        if self.db is None:
            return
        self.db.snapshot.save_checkpoint(
            run_id=run_id,
            config_name=self.config_name,
            last_batch_end=str(self.start_date),
            status=status,
        )

    def load_checkpoint(self, run_id: str) -> dict | None:
        """加载运行元信息。"""
        if self.db is None:
            return None
        return self.db.snapshot.load_checkpoint(run_id)
