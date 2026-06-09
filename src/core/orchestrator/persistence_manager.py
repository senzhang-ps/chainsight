"""持久化管理器。

职责：将 StateContext 和 Module 的输出数据写入数据库。
表映射统一使用 table_mapping.py 的 OUTPUT_TABLE_MAPPING，不硬编码。
"""

import logging
from datetime import datetime
from typing import TYPE_CHECKING

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
            self.db._write_df(table_name, df)

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
                self.db._write_df(tbl, inv_log)

        # 3) 写 daily_logs（写入后清空，避免历史日志重复写入）
        if hasattr(ctx, 'daily_logs') and ctx.daily_logs:
            logs_df = pd.DataFrame(ctx.daily_logs)
            logs_df = self._inject_meta(logs_df, run_id, sim_date, now)
            tbl = view_table_map.get(
                'daily_logs', 'viewcontext_daily_logs',
            )
            self.db._write_df(tbl, logs_df)
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
                self.db._write_df(tbl, audit)

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
            self.db._write_df(table_name, df)
            written += 1

        if written:
            logger.info(
                "🗄️ 已写入 %s 输出到数据库 (%d 张表)",
                module_config, written,
            )

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