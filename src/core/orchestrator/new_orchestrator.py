import hashlib
import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from ...utils.data_quality import DataQualityError

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
                 config_dict=None, engine='pandas', skip_dq=False):
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
        
        # ── 持久化管理 ──
        from .persistence_manager import PersistenceManager
        self.persistence = PersistenceManager(self)

        # ── 配置加载 + 持久化 ──
        self.sys_config: dict = {}
        self.all_config: dict = {}
        self._last_dq_result: dict | None = None

        try:
            dq_result, needs_write = self._load_config(
                config_path, config_dict, skip_dq=skip_dq,
            )
            if needs_write:
                self._persist_config(dq_result, write_config=True)

        except DataQualityError:
            # DQ 阻断：仍持久化检测结果（不含配置数据），供调试排查
            if self._last_dq_result is not None and self.db is not None:
                self._persist_config(self._last_dq_result, write_config=False)
            raise

        # ── 随机种子 ──
        from ..main_integration.seed import set_module_seeds
        set_module_seeds(self.all_config)

        # ── 输出目录 ──
        self.build_output_folder()
        self.all_results = {}
        self.sim_dates = []


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
        for key in ('engine', 'resource', 'shared', 'data_quality'):
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

    def _load_config(
        self,
        config_path: str | None = None,
        config_dict: dict | None = None,
        *,
        skip_dq: bool = False,
    ) -> tuple[dict | None, bool]:
        """加载配置数据到 ``self.all_config``（纯读取，不写入 DB）。

        三条路径（优先级从高到低）：
        1. ``config_dict`` 直接传入 → 直接使用（跳过 DB 缓存和 DQ 校验）
        2. ``config_path`` → 优先尝试 DB 缓存，失败则从 xlsx 加载 + DQ 校验
        3. 均为空 → 配置为空，打印警告

        DB 表说明：
        - ``cfg_dq_check_result``：存储 config_hash (MD5) + DQ 检测结果
          ``_try_load_config_from_db`` 从此表读 hash，与当前 hash 比对
        - ``cfg_*`` 配置表：存储清洗后的配置 DataFrame（如 cfg_m1_initialinventory）
          ``_try_load_config_from_db`` → ``_load_calc_datas`` 从这些表读回配置数据

        Args:
            config_path: 配置文件路径（xlsx）。
            config_dict: 已加载好的配置表数据 ``{sheet_name: DataFrame}``。
            skip_dq: 是否跳过数据质量检测。数据存在已知问题需先跑通下游时可设为 True。

        Returns:
            ``(dq_result, needs_write)`` 元组：
            - ``dq_result``：DQ 校验结果字典；未执行 DQ 时为 None。
            - ``needs_write``：是否需要将配置数据写入 DB 缓存。
        """
        if config_dict is not None:
            self.all_config = config_dict
            return None, False

        if config_path is None:
            logger.warning("Orch: 未提供 config_path 或 config_dict，配置为空")
            return None, False

        self._config_name = Path(config_path).stem

        # 尝试从 DB 缓存加载（hash 去重）
        if self._try_load_config_from_db():
            logger.info(f"Orch: 从 DB 缓存加载配置: {self._config_name}")
            return None, False

        # DB 无缓存或 hash 不一致 → 从 Excel 加载
        from ..main_integration.config_loader import load_configuration
        logger.info(f"Orch: 从配置文件加载: {config_path}")
        self.all_config = load_configuration(config_path)

        if skip_dq:
            logger.warning("Orch: skip_dq=True，跳过数据质量检测")
            return None, True

        # 加载后执行 DQ 校验
        dq_result = self._run_data_quality_check()

        # 用清洗后的数据替换原始数据
        cleaned = dq_result.get("cleaned_tables", {})
        if cleaned:
            self.all_config = cleaned
            logger.info("Orch: 已用 DQ 清洗后数据替换 all_config")

        return dq_result, True

    # ══════════════════════════════════════════
    # 配置持久化（写入 DB）
    # ══════════════════════════════════════════

    def _persist_config(
        self,
        dq_result: dict[str, Any] | None = None,
        *,
        write_config: bool = True,
    ) -> None:
        """将配置数据和/或 DQ 结果写入 DB 缓存。

        统一的持久化入口，仅在 ``__init__`` 中调用。
        从 ``_load_config`` 和 ``_run_data_quality_check`` 中剥离，
        保证加载方法只负责读取。

        写入目标：
        - ``cfg_dq_check_result``：config_hash + DQ 检测结果（后续 hash 去重依据）
        - ``cfg_*`` 配置表：配置 DataFrame（仅当 ``write_config=True``）

        Args:
            dq_result: DQ 校验结果；为 None 时仅写配置数据（skip_dq 场景）。
            write_config: 是否同时将配置数据写入 ``cfg_*`` 表。
                DQ 阻断时应设为 False，避免问题数据覆盖已有配置。
        """
        if self.db is None or not self.all_config:
            return
        self._write_to_db(dq_result, write_config=write_config)

    # ══════════════════════════════════════════
    # 数据质量检测
    # ══════════════════════════════════════════

    def _run_data_quality_check(self) -> dict[str, Any]:
        """对 all_config 执行数据质量校验。

        在配置文件加载后调用，使用 ``ConfigInputDataQualityChecker`` 执行校验。
        若检测到阻断级问题则抛出 ``DataQualityError``。

        Returns:
            ``validate`` 返回的完整检测结果字典，包含：
            - passed / blocked / issues / cleaned_tables / summary 等
        """
        from ...utils.data_quality import ConfigInputDataQualityChecker

        if not self.all_config:
            logger.info("DQ: all_config 为空，跳过数据质量检查")
            return {
                "passed": True,
                "blocked": False,
                "issues": [],
                "cleaned_tables": {},
                "summary": {"errors": 0, "warnings": 0, "infos": 0},
            }

        dq_cfg = (self.sys_config or {}).get('data_quality') or {}
        if not dq_cfg.get('enabled', True):
            logger.info("DQ: 数据质量检测已禁用 (sys_cfg.data_quality.enabled=false)")
            return {
                "passed": True,
                "blocked": False,
                "issues": [],
                "cleaned_tables": {},
                "summary": {"errors": 0, "warnings": 0, "infos": 0},
            }

        config_tables = dq_cfg.get('config_tables') or {}
        checker = ConfigInputDataQualityChecker(
            mode=dq_cfg.get('mode', 'audit_only'),
            fail_on_error=dq_cfg.get('fail_on_error', False),
            sample_limit=dq_cfg.get('sample_limit', 20),
            report_dir=dq_cfg.get('report_dir'),
            required_import_tables=config_tables.get('required_import') or (),
            optional_import_tables=config_tables.get('optional_import') or (),
            quality_check_enabled=config_tables.get('quality_check_enabled') or {},
        )
        dq_result = checker.validate(
            self.all_config,
            config_name=self.config_name,
            sub_node="input_pre.orchestrator",
        )

        summary = dq_result.get("summary", {})
        errors = summary.get("errors", 0)
        warnings = summary.get("warnings", 0)
        hard_blocks = summary.get("hard_blocks", 0)

        logger.info(
            f"DQ: 校验完成 → errors={errors}, "
            f"warnings={warnings}, hard_blocks={hard_blocks}, "
            f"blocked={dq_result['blocked']}"
        )

        if dq_result["blocked"]:
            # 保存到实例，供 __init__ 在捕获异常后持久化 DQ 结果
            self._last_dq_result = dq_result
            raise DataQualityError(
                "配置数据质量检查发现阻断级问题; "
                f"config={self.config_name}, "
                f"errors={errors}, hard_blocks={hard_blocks}"
            )

        return dq_result

    def _compute_config_hash(self, config_data: dict[str, "pd.DataFrame"]) -> str:
        """计算配置数据的 MD5 指纹。

        对每张表按 sheet 名排序后逐表算 hash，拼接后取整体 MD5。
        """
        from pgsql_db.config_manifest import compute_table_hash

        parts = []
        for sheet_name in sorted(config_data.keys()):
            df = config_data[sheet_name]
            table_hash = compute_table_hash(
                df if isinstance(df, pd.DataFrame) else pd.DataFrame()
            )
            parts.append(f"{sheet_name}:{table_hash}")
        return hashlib.md5("\n".join(parts).encode("utf-8")).hexdigest()

    def _write_to_db(
        self,
        dq_result: dict[str, Any] | None = None,
        *,
        write_config: bool = True,
    ) -> None:
        """将 DQ 检测结果 + 配置数据写入 DB 缓存。

        Args:
            dq_result: DQ 校验返回的完整结果。
            write_config: 是否同时将配置数据写入 cfg_* 表。
                DQ 阻断时应设为 False，避免将问题数据覆盖已有配置表。
        """
        config_hash = self._compute_config_hash(self.all_config)
        self.persistence.save_dq_result(
            config_name=self.config_name or "unknown",
            config_hash=config_hash,
            config_data=self.all_config,
            dq_result=dq_result,
            write_config=write_config,
        )
        logger.info(
            f"DB 缓存: 已写入 hash={config_hash[:12]}... "
            f"(write_config={write_config}, "
            f"{len(self.all_config)} 张表)"
        )

    # ══════════════════════════════════════════
    # 配置缓存（DB 去重）
    # ══════════════════════════════════════════

    def _try_load_config_from_db(self) -> bool:
        """尝试从 DB 缓存加载配置，跳过文件读取和 DQ 校验。

        流程：
        1. 查 ``cfg_dq_check_result`` 表是否有该 config_name 的缓存记录
        2. 有记录 → 从 DB cfg_* 表读回配置 → 计算 hash → 与缓存中 hash 比对
        3. hash 一致 → 填充 ``self.all_config``，返回 True
        4. 无记录或 hash 不一致 → 返回 False，由调用方走文件加载

        Returns:
            True 表示成功从 DB 缓存加载，False 表示需要走文件加载。
        """
        if self.db is None:
            return False

        config_name = self.config_name
        if not config_name:
            return False

        try:
            # 查缓存表中是否有该 config_name 的记录
            table_name = "cfg_dq_check_result"
            rows = self.db.execute_query(
                f"SELECT config_hash FROM {table_name} "
                f"WHERE config_name = %s "
                f"LIMIT 1",
                (config_name,),
            )
            if not rows:
                logger.info(f"DB 缓存: 未找到 {config_name} 的记录，将走文件加载")
                return False

            stored_hash = rows[0][0]

            # 从 cfg_* 表读回配置数据
            restored = self._load_calc_datas()
            if not restored:
                logger.info("DB 缓存: cfg_* 表无数据，将走文件加载")
                return False

            # 计算读回数据的 hash 并比对
            current_hash = self._compute_config_hash(restored)

            if current_hash != stored_hash:
                logger.info(
                    f"DB 缓存: hash 不一致 "
                    f"(存储={stored_hash[:12]}..., 当前={current_hash[:12]}...)，"
                    f"将走文件加载"
                )
                return False

            # hash 一致，使用 DB 数据
            self.all_config = restored
            logger.info(
                f"DB 缓存: hash 一致，已从 DB 加载 {len(restored)} 张配置表 "
                f"(跳过文件加载和 DQ 校验)"
            )
            return True

        except Exception as e:
            logger.info(f"DB 缓存: 查询失败（将走文件加载）: {e}")
            return False

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
