"""配置生命周期管理器 — 拥有 DB 连接、系统/计算配置、DQ 与 run 事件缓存。

从 ``Orchestrator`` 抽出的全部「配置 IO + DQ」职责收敛于此：

- ``bootstrap``：从 yaml 建 DB 连接、设 sys_config、migrate 建表。
- ``load``：Excel/DB/dict 三路径加载 → hash → 缓存命中检查 → 必要时跑 DQ。
- ``persist``：按阻断决策写 ``cfg_*`` + 落定 run 事件。

身份（``config_name``/``run_id``）与状态持久化（``PersistenceManager``）仍属
``Orchestrator``；本类经 ``self._orch`` 反向取用（同 ``PersistenceManager`` 模式）。
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from .new_orchestrator import Orchestrator

logger = logging.getLogger(__name__)


class ConfigManager:
    """配置生命周期总管：DB bootstrap + 配置加载 + DQ + 写回。

    持有 ``_db``/``_sys_config``/``_all_config``/``_last_dq_result``；
    ``Orchestrator`` 通过透传属性对外暴露 ``db``/``sys_config``/``all_config``。
    """

    def __init__(self, orch: "Orchestrator"):
        self._orch = orch
        self._db = None
        self._sys_config: dict = {}
        self._all_config: dict = {}
        self._last_dq_result: dict | None = None

    # ── 对外属性 ──────────────────────────────
    @property
    def db(self):
        return self._db

    @property
    def sys_config(self) -> dict:
        return self._sys_config

    @sys_config.setter
    def sys_config(self, value: dict):
        self._sys_config = value

    @property
    def all_config(self) -> dict:
        return self._all_config

    @all_config.setter
    def all_config(self, value: dict):
        self._all_config = value

    @property
    def last_dq_result(self) -> dict | None:
        return self._last_dq_result

    # ══════════════════════════════════════════
    # DB bootstrap
    # ══════════════════════════════════════════

    def _load_sys_config(self) -> dict:
        """从 yaml 读取系统配置（DB 连接参数、全局运行参数）。"""
        import yaml
        config_path = Path(__file__).resolve().parents[3] / 'config' / 'defaults.yaml'
        if not config_path.exists():
            logger.warning(f"系统配置文件不存在: {config_path}")
            return {}
        with open(config_path, 'r', encoding='utf-8') as f:
            raw = yaml.safe_load(f)
        sys_cfg = {}
        if 'database' in raw:
            sys_cfg['database'] = raw['database']
        for key in ('engine', 'resource', 'shared', 'data_quality'):
            if key in raw:
                sys_cfg[key] = raw[key]
        logger.info(f"系统配置已加载: {list(sys_cfg.keys())}")
        return sys_cfg

    def bootstrap(self) -> None:
        """从 yaml 建 DB 连接 + 设 sys_config + migrate 建表。

        读取 ``config/defaults.yaml`` 的 database 节，创建 DB 实例并 migrate；
        连接失败则 ``_db=None``（文件模式），不抛异常。
        """
        sys_cfg = self._load_sys_config()
        if not sys_cfg:
            return

        self._sys_config = sys_cfg
        db_cfg = sys_cfg.get('database')
        if not db_cfg:
            return

        try:
            from ..db.pgsql.db import DB
            self._db = DB(
                host=db_cfg.get('host', 'localhost'),
                port=int(db_cfg.get('port', 5432)),
                database=db_cfg.get('database', 'test_db'),
                user=db_cfg.get('user', 'postgres'),
                password=str(db_cfg.get('password', '')),
            )
            self._db.connect()
            logger.info("🗄️ 数据库连接已建立")

            # ── migrate: Django 风格统一建表 ──
            from src.models import migrate
            migrate(self._db)
            logger.info("🗄️ 数据库表已迁移")
        except Exception as e:
            logger.warning(f"🗄️ 数据库连接失败（将使用文件模式）: {e}")
            self._db = None

    # ══════════════════════════════════════════
    # 配置加载
    # ══════════════════════════════════════════

    def load(
        self,
        config_path: str | None = None,
        config_dict: dict | None = None,
        *,
        skip_dq: bool = False,
    ) -> tuple[dict | None, bool]:
        """加载配置数据到 ``all_config``，并维护 run 事件生命周期。

        三条路径（优先级从高到低）：
        1. ``config_dict`` 直接传入 → 直接使用（跳过 DB 缓存和 DQ 校验）
        2. ``config_path`` → 从 Excel 加载 → 检查 hash 缓存 → 命中则读 DB，
           否则跑 DQ 校验
        3. 均为空 → 配置为空，打印警告

        Returns:
            ``(dq_result, needs_write)``：未执行 DQ 时 dq_result 为 None。
        """
        if config_dict is not None:
            self._all_config = config_dict
            return None, False

        if config_path is None:
            logger.warning("ConfigManager: 未提供 config_path 或 config_dict，配置为空")
            return None, False

        self._orch.config_name = Path(config_path).stem

        # ── 续跑短路：从 DB cfg_* 读已校验配置（DQ 已通过、配置已写入），跳过 DQ + run 事件 ──
        if getattr(self._orch, '_resuming', False) and self._db is not None:
            logger.info(
                f"ConfigManager: 续跑 {self._orch.config_name}，"
                f"从 DB cfg_* 读已校验配置，跳过 DQ + run 事件"
            )
            self._all_config = self._load_calc_datas()
            return None, False

        # 1. 从 Excel 加载配置（通过 ConfigReader）
        from ...io.reader import ConfigReader
        logger.info(f"ConfigManager: 从配置文件加载: {config_path}")
        reader = ConfigReader(config_path)
        self._all_config = reader.load_all()

        if skip_dq:
            logger.warning("ConfigManager: skip_dq=True，跳过数据质量检测")
            return None, True

        # 无 DB 连接：直接跑 DQ（不做缓存/事件）
        if self._db is None:
            dq_result = self._run_data_quality_check()
            cleaned = dq_result.get("cleaned_tables", {})
            if cleaned:
                self._all_config = cleaned
                logger.info("ConfigManager: 已用 DQ 清洗后数据替换 all_config")
            return dq_result, True

        # 2. 计算 hash + 插入 run 事件（触发生成 run_id）
        config_hash = self._compute_config_hash(self._all_config)
        run_id = self._orch.run_id
        self._orch.persistence.start_run_event(
            run_id, self._orch.config_name, config_hash,
            total_days=self._orch.total_days,
        )

        # 3. 检查 hash 缓存（排除当前 runid，找历史 passed + 同 hash）
        if self._is_dq_cached(self._orch.config_name, config_hash, run_id):
            self._all_config = self._load_calc_datas()
            self._orch.persistence.mark_run_event_cached(run_id)
            logger.info(
                f"ConfigManager: hash 命中缓存，从 DB 加载配置: "
                f"{self._orch.config_name} (跳过 DQ 校验)"
            )
            return None, False

        # 4/5. 缓存未命中 → 跑 DQ
        dq_result = self._run_data_quality_check()
        cleaned = dq_result.get("cleaned_tables", {})
        if cleaned:
            self._all_config = cleaned
            logger.info("ConfigManager: 已用 DQ 清洗后数据替换 all_config")

        return dq_result, True

    def _load_calc_datas(self) -> dict:
        """从 DB 的 cfg_* 表读取计算数据 → 返回 {sheet_name: DataFrame}。"""
        if self._db is None:
            raise RuntimeError("_load_calc_datas 需要 db 连接")

        from src.models.cfg import CONFIG_TABLE_REGISTRY

        # 逆向映射：cfg_* 表名 → sheet_name（从 Model.__tablename__ 派生）
        reverse_map = {}
        for sheet, model_cls in CONFIG_TABLE_REGISTRY.items():
            tablename = model_cls.__tablename__
            db_key = tablename[4:] if tablename.startswith("cfg_") else tablename
            reverse_map[db_key] = sheet

        all_tables = self._db.read('_tables') if hasattr(self._db, 'read_tables') else []
        if not all_tables:
            all_tables = [f'cfg_{k}' for k in reverse_map]

        config_dict = {}
        for table_name in all_tables:
            if not table_name.startswith('cfg_'):
                continue
            db_key = table_name[4:]
            sheet_name = reverse_map.get(db_key, db_key)
            try:
                df = self._db.read(table_name, config_name=self._orch.config_name)
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
    # 配置写回 + run 事件落定
    # ══════════════════════════════════════════

    def persist(
        self,
        dq_result: dict[str, Any] | None = None,
        *,
        write_config: bool = True,
    ) -> None:
        """落定 run 事件 + 按阻断决策写配置到 ``cfg_*``。

        - 未阻断（含 skip_dq、audit_only、阻断模式但无 error）：写配置 + finalize。
        - 阻断（阻断模式且存在 error）：不写配置，仅 finalize blocked。
        """
        if self._db is None or not self._all_config:
            return

        blocked = self._should_block_dq(dq_result)
        if write_config and not blocked:
            self._orch.persistence.save_config(
                self._orch.config_name or "unknown", self._all_config
            )
        if dq_result is None:
            status = "skipped"
        elif blocked:
            status = "blocked"
        elif dq_result.get('passed'):
            status = "passed"
        else:
            status = 'blocked'
        self._orch.persistence.finalize_run_event(
            self._orch.run_id, self._orch.config_name, dq_result, status=status
        )

    def _should_block_dq(self, dq_result: dict[str, Any] | None) -> bool:
        """按 ``sys_config.data_quality`` 模式 + DQ 结果判断是否阻断。"""
        if dq_result is None:
            return False
        dq_cfg = (self._sys_config or {}).get('data_quality') or {}
        if dq_cfg.get('mode') == 'audit_only':
            return False
        if not dq_cfg.get('fail_on_error', False):
            return False
        summary = dq_result.get('summary', {}) or {}
        return summary.get('errors', 0) > 0

    def _is_dq_cached(
        self, config_name: str | None, config_hash: str, current_run_id: str
    ) -> bool:
        """查历史 run 事件：该 config 下是否存在 passed 且 hash 一致的记录。"""
        if self._db is None or not config_name:
            return False
        try:
            rows = self._db.execute_query(
                "SELECT config_hash FROM orch_run_event "
                "WHERE config_name = %s AND dq_status = 'passed' "
                "AND runid <> %s "
                "ORDER BY started_at DESC LIMIT 1",
                (config_name, current_run_id),
            )
            if not rows:
                logger.info(f"DB 缓存: 无 {config_name} 的 passed 记录，将走 DQ")
                return False
            stored_hash = rows[0][0]
            if stored_hash != config_hash:
                logger.info(
                    f"DB 缓存: hash 不一致 "
                    f"(存储={stored_hash[:12]}..., 当前={config_hash[:12]}...)，"
                    f"将走 DQ"
                )
                return False
            return True
        except Exception as e:
            logger.info(f"DB 缓存: 查询失败（将走 DQ）: {e}")
            return False

    # ══════════════════════════════════════════
    # 数据质量检测
    # ══════════════════════════════════════════

    def _run_data_quality_check(self) -> dict[str, Any]:
        """对 all_config 执行数据质量校验，返回检测结果字典。"""
        from ...utils.data_quality import ConfigInputDataQualityChecker

        if not self._all_config:
            logger.info("DQ: all_config 为空，跳过数据质量检查")
            return {
                "passed": True,
                "blocked": False,
                "issues": [],
                "cleaned_tables": {},
                "summary": {"errors": 0, "warnings": 0, "infos": 0},
            }

        dq_cfg = (self._sys_config or {}).get('data_quality') or {}
        if not dq_cfg.get('enabled', True):
            logger.info("DQ: 数据质量检测已禁用 (sys_cfg.data_quality.enabled=false)")
            return {
                "passed": True,
                "blocked": False,
                "issues": [],
                "cleaned_tables": {},
                "summary": {"errors": 0, "warnings": 0, "infos": 0},
            }

        checker = ConfigInputDataQualityChecker()
        dq_result = checker.validate(
            self._all_config,
            config_name=self._orch.config_name,
            report_dir=dq_cfg.get('report_dir'),
        )

        summary = dq_result.get("summary", {})
        errors = summary.get("errors", 0)
        warnings = summary.get("warnings", 0)
        hard_blocks = summary.get("hard_blocks", 0)
        logger.info(
            f"DQ: 校验完成 → errors={errors}, "
            f"warnings={warnings}, hard_blocks={hard_blocks}"
        )

        if errors > 0:
            # 保存到实例，供 Orchestrator.__init__ 捕获异常后持久化 DQ 结果
            self._last_dq_result = dq_result

        return dq_result

    def _compute_config_hash(self, config_data: dict[str, "pd.DataFrame"]) -> str:
        """计算配置数据的 MD5 指纹（按 sheet 名排序逐表算 hash 后拼接取 MD5）。"""
        from pgsql_db.config_manifest import compute_table_hash

        parts = []
        for sheet_name in sorted(config_data.keys()):
            df = config_data[sheet_name]
            table_hash = compute_table_hash(
                df if isinstance(df, pd.DataFrame) else pd.DataFrame()
            )
            parts.append(f"{sheet_name}:{table_hash}")
        return hashlib.md5("\n".join(parts).encode("utf-8")).hexdigest()
