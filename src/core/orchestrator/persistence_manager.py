"""持久化管理器。

职责：将 StateContext 和 Module 的输出数据写入数据库。
表映射统一使用 table_mapping.py 的 OUTPUT_TABLE_MAPPING，不硬编码。
"""

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from ...modules.state_context import StateContext
    from ...modules.module import Module

logger = logging.getLogger(__name__)


class PersistenceManager:
    """持久化总管：只写数据库。

    表映射统一使用 table_mapping.py 的 OUTPUT_TABLE_MAPPING，不硬编码。
    """

    def __init__(self, orch):
        self._orch = orch

    @property
    def db(self):
        return self._orch.db

    @property
    def _run_id(self):
        return self._orch.run_id or 'local'

    @property
    def config_name(self):
        return self._orch.config_name or ''

    # ══════════════════════════════════════════
    # 每日状态保存
    # ══════════════════════════════════════════

    def save_daily_state(self, ctx: 'StateContext', date_str: str):
        """将 StateContext 的每日状态写入数据库。

        Args:
            ctx: StateContext 实例
            date_str: 日期字符串 YYYY-MM-DD
        """
        if self.db is None:
            logger.debug("无 DB 连接，跳过 StateContext 持久化")
            return

        from ...core.db.pgsql.table_mapping import OUTPUT_TABLE_MAPPING
        from ...utils.normalization import normalize_identifiers

        view_table_map = OUTPUT_TABLE_MAPPING.get('viewcontext', {})
        sim_date = pd.Timestamp(date_str).strftime('%Y-%m-%d')
        now = datetime.now()
        run_id = self._run_id

        # 1) 写 views
        for view_name, table_name in view_table_map.items():
            df = ctx.views.get(view_name)
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                continue
            df = normalize_identifiers(df.copy())
            df = self._inject_meta(df, run_id, sim_date, now)
            self.db.write_df(table_name, df)

        # 2) 写 inventory_change_log
        if hasattr(ctx, 'generate_inventory_change_log'):
            inv_log = ctx.generate_inventory_change_log(date_str)
            if inv_log is not None and not inv_log.empty:
                inv_log = normalize_identifiers(inv_log.copy())
                inv_log = self._inject_meta(inv_log, run_id, sim_date, now)
                tbl = view_table_map.get(
                    'inventory_change_log',
                    'viewcontext_inventory_change_log',
                )
                self.db.write_df(tbl, inv_log)

        # 3) 写 daily_logs（写入后清空，避免历史日志重复写入）
        if hasattr(ctx, 'daily_logs') and ctx.daily_logs:
            logs_df = pd.DataFrame(ctx.daily_logs)
            logs_df = self._inject_meta(logs_df, run_id, sim_date, now)
            tbl = view_table_map.get(
                'daily_logs', 'viewcontext_daily_logs',
            )
            self.db.write_df(tbl, logs_df)
            ctx.daily_logs.clear()

        # 4) 写 cleanup_audit
        if hasattr(ctx, 'cleanup_audit_df') and ctx.cleanup_audit_df is not None:
            audit = ctx.cleanup_audit_df
            if isinstance(audit, pd.DataFrame) and not audit.empty:
                audit = normalize_identifiers(audit.copy())
                audit = self._inject_meta(audit, run_id, sim_date, now)
                tbl = view_table_map.get(
                    'open_deployment_pastdue_cleanup',
                    'viewcontext_open_deployment_pastdue_cleanup',
                )
                self.db.write_df(tbl, audit)

        logger.info(f"🗄️ 已写入 {date_str} 每日状态到数据库")

        # 统计信息
        if hasattr(ctx, 'get_summary_statistics'):
            stats = ctx.get_summary_statistics(date_str)
            logger.info(f"📊 当日统计: {stats}")

    # ══════════════════════════════════════════
    # 模块输出保存
    # ══════════════════════════════════════════

    def save_module_output(self, module: 'Module', sim_date: str):
        """从 module.output() 读取数据并写入数据库。

        Args:
            module: Module 实例
            sim_date: 日期字符串 YYYY-MM-DD
        """
        if self.db is None:
            logger.debug("无 DB 连接，跳过模块输出持久化")
            return

        from ...core.db.pgsql.table_mapping import OUTPUT_TABLE_MAPPING
        from ...utils.normalization import normalize_identifiers

        # 推导 module_name: e.g., 'M1' → 'module1'
        module_config = getattr(module, 'module_config', '')
        if module_config.startswith('M') and module_config[1:].isdigit():
            db_key = f"module{module_config[1:]}"
        else:
            db_key = module_config.lower()

        table_map = OUTPUT_TABLE_MAPPING.get(db_key, {})
        if not table_map:
            logger.warning(
                "未找到模块 %s 的输出表映射 (db_key=%s)",
                module_config, db_key,
            )
            return

        results = module.output()
        if not results:
            return

        sim_date_str = pd.Timestamp(sim_date).strftime('%Y-%m-%d')
        now = datetime.now()
        run_id = self._run_id

        written = 0
        for key, df in results.items():
            table_name = table_map.get(key)
            if table_name is None:
                continue
            if df is None:
                continue
            if isinstance(df, pd.DataFrame) and df.empty:
                continue

            # Polars → pandas
            try:
                import polars as pl
                if isinstance(df, pl.DataFrame):
                    df = df.to_pandas()
            except ImportError:
                pass

            if not isinstance(df, pd.DataFrame):
                continue

            df = normalize_identifiers(df.copy())
            df = self._inject_meta(df, run_id, sim_date_str, now)
            self.db.write_df(table_name, df)
            written += 1

        if written:
            logger.info(
                "🗄️ 已写入 %s 输出到数据库 (%d 张表)",
                module_config, written,
            )

    # ══════════════════════════════════════════
    # 数据质量结果持久化
    # ══════════════════════════════════════════

    def save_dq_result(
        self,
        config_name: str,
        config_hash: str,
        config_data: dict[str, pd.DataFrame],
        dq_result: dict[str, Any] | None = None,
        *,
        write_config: bool = True,
    ) -> None:
        """将配置数据 + DQ 检测结果写入 DB 缓存，供后续 hash 去重。

        写入内容：
        1. ``cfg_dq_check_result`` 表：config_name + config_hash + DQ 检测结果
        2. 对应 ``cfg_*`` 配置表：配置数据（仅当 ``write_config=True`` 时写入）

        Args:
            config_name: 配置名称。
            config_hash: 配置数据整体 MD5 指纹。
            config_data: 已加载的配置表数据 ``{sheet_name: DataFrame}``。
            dq_result: ``ConfigInputDataQualityChecker.validate()`` 返回的检测结果。
               为 None 时表示从 DB 缓存加载（无 DQ 结果可写）。
            write_config: 是否同时将配置数据写入 cfg_* 表。
               DQ 阻断时应设为 False，避免将问题数据覆盖已有配置表。
        """
        if self.db is None:
            logger.debug("无 DB 连接，跳过 DQ 缓存持久化")
            return

        now = datetime.now()

        # ── 提取 DQ 检测结果 ──
        issues = (dq_result or {}).get("issues", [])
        status = "passed" if not (dq_result or {}).get("blocked") else "blocked"

        # 按 sheet 分组 issues，避免每行都塞全量 issues
        issues_by_sheet: dict[str, list[dict]] = {}
        for issue in issues:
            sheet = issue.get("sheet", "")
            issues_by_sheet.setdefault(sheet, []).append(issue)

        # ── 写 cfg_dq_check_result 缓存表 ──
        rows_data = []
        for sheet_name in sorted(issues_by_sheet.keys()):
            sheet_issues = issues_by_sheet.get(sheet_name, [])
            # 按 severity 统计当前 sheet 的 errors / warnings / hard_blocks
            sheet_errors = sum(
                1 for i in sheet_issues if i.get("severity") == "error"
            )
            sheet_warnings = sum(
                1 for i in sheet_issues if i.get("severity") == "warning"
            )
            sheet_hard_blocks = sum(
                1 for i in sheet_issues if i.get("severity") == "hard_block"
            )
            rows_data.append({
                "config_name": config_name,
                "config_hash": config_hash,
                "sheet_name": sheet_name,
                "status": status,
                "errors": sheet_errors,
                "warnings": sheet_warnings,
                "hard_blocks": sheet_hard_blocks,
                "issues_json": json.dumps(
                    sheet_issues, ensure_ascii=False, default=str
                ),
                "db_write_time": now,
            })

        # DQ 通过零问题 / skip_dq → 仍写入 __summary__ 行，保证 hash 始终可查
        if not rows_data:
            rows_data.append({
                "config_name": config_name,
                "config_hash": config_hash,
                "sheet_name": "__summary__",
                "status": "passed",
                "errors": 0,
                "warnings": 0,
                "hard_blocks": 0,
                "issues_json": "[]",
                "db_write_time": now,
            })

        if rows_data:
            result_df = pd.DataFrame(rows_data)
            table_name = "cfg_dq_check_result"
            try:
                self.db.write_df(table_name, result_df)
                logger.info(
                    f"DQ: 已写入检测结果到 {table_name} "
                    f"({len(rows_data)} 条记录, status={status})"
                )
            except Exception as e:
                logger.warning(f"DQ: 写入缓存失败: {e}")

        # ── 写配置数据到 cfg_* 表（仅 DQ 通过时） ──
        if write_config:
            written = 0
            for sheet_name, df in config_data.items():
                if not isinstance(df, pd.DataFrame) or df.empty:
                    continue
                cfg_table = self._sheet_to_cfg_table(sheet_name)
                if cfg_table is None:
                    continue
                try:
                    # 删除该 config_name 的旧数据再写入
                    self.db.execute(
                        f"DELETE FROM {cfg_table} WHERE config_name = %s",
                        (config_name,),
                    )
                    write_df = df.copy()
                    write_df = self._inject_config_meta(write_df, config_name, now)
                    self.db.write_df(cfg_table, write_df)
                    written += 1
                except Exception as e:
                    logger.warning(f"DQ: 写回配置表 {cfg_table} 失败: {e}")

            if written:
                logger.info(f"DQ: 已写回 {written} 张配置表")

    def _sheet_to_cfg_table(self, sheet_name: str) -> str | None:
        """将 Sheet 名映射为 cfg_* 物理表名。

        Args:
            sheet_name: 配置表 Sheet 名（如 M1_InitialInventory）。

        Returns:
            cfg_* 物理表名（如 cfg_m1_initialinventory）；无法映射时返回 None。
        """
        try:
            from ...core.db.pgsql.table_mapping import CONFIG_TABLE_MAPPING
            db_key = CONFIG_TABLE_MAPPING.get(sheet_name)
            if db_key:
                return f"cfg_{db_key}"
        except Exception:
            pass
        # 回退：将 Sheet 名转为 snake_case 并加 cfg_ 前缀
        import re
        snake = re.sub(r'(?<=[a-z0-9])([A-Z])', r'_\1', sheet_name).lower()
        return f"cfg_{snake}"

    def _inject_config_meta(
        self, df: pd.DataFrame, config_name: str, now: datetime
    ) -> pd.DataFrame:
        """向配置表 DataFrame 注入 config_name 和 db_write_time 元数据列。

        Args:
            df: 待注入元数据的 DataFrame。
            config_name: 配置名称。
            now: 写入时间。

        Returns:
            注入元数据列后的 DataFrame。
        """
        if 'config_name' not in df.columns:
            df.insert(0, 'config_name', config_name)
        if 'db_write_time' not in df.columns:
            df['db_write_time'] = now
        return df

    # ══════════════════════════════════════════
    # 工具方法
    # ══════════════════════════════════════════

    def _inject_meta(self, df: pd.DataFrame, run_id: str,
                     sim_date: str, now=None) -> pd.DataFrame:
        """向 DataFrame 注入 DB 元数据列。"""
        if now is None:
            now = datetime.now()
        if 'run_id' not in df.columns:
            df.insert(0, 'run_id', run_id)
        if 'sim_date' not in df.columns:
            df.insert(1, 'sim_date', sim_date)
        if 'config_name' not in df.columns:
            df.insert(2, 'config_name', self.config_name)
        if 'db_write_time' not in df.columns:
            df.insert(3, 'db_write_time', now)
        return df