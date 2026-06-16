"""持久化管理器。

职责：将 StateContext 和 Module 的输出数据写入数据库。

迁移版改动：
- __init__ 内创建 DBWriter，所有 self.db.write_df → self._writer.write
- save_config 中 self.db.delete_where → self._writer.delete
- _sheet_to_cfg_table() 改为 Django 风格：从 CONFIG_TABLE_REGISTRY 读 Model.__tablename__
- 建表统一由 Orchestrator._init_db 启动时 migrate() 完成，本类不再 _ensure_run_tables
- OUTPUT_TABLE_MAPPING 从 module/viewcontext 注册表派生（不依赖 table_mapping.py）
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
        # ── Writer ──
        from ...io.writer import DBWriter
        self._writer = DBWriter(self.db)

    @property
    def db(self):
        return self._orch.db

    @property
    def _run_id(self):
        return self._orch.run_id or 'local'

    @property
    def config_name(self):
        return self._orch.config_name or ''

    # ════════════════════════════════════════
    # 每日状态保存
    # ════════════════════════════════════════

    def save_daily_state(self, ctx: 'StateContext', date_str: str):
        """将 StateContext 的每日状态写入数据库。

        Args:
            ctx: StateContext 实例
            date_str: 日期字符串 YYYY-MM-DD
        """
        if self.db is None:
            logger.debug("无 DB 连接，跳过 StateContext 持久化")
            return

        from src.models.viewcontext import VIEWCONTEXT_REGISTRY
        from ...utils.normalization import normalize_identifiers

        view_table_map = VIEWCONTEXT_REGISTRY
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
            self._writer.write(table_name, df)

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
                self._writer.write(tbl, inv_log)

        # 3) 写 daily_logs（写入后清空，避免历史日志重复写入）
        if hasattr(ctx, 'daily_logs') and ctx.daily_logs:
            logs_df = pd.DataFrame(ctx.daily_logs)
            logs_df = self._inject_meta(logs_df, run_id, sim_date, now)
            tbl = view_table_map.get(
                'daily_logs', 'viewcontext_daily_logs',
            )
            self._writer.write(tbl, logs_df)
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
                self._writer.write(tbl, audit)

        logger.info(f"🗄️ 已写入 {date_str} 每日状态到数据库")

        # 统计信息
        if hasattr(ctx, 'get_summary_statistics'):
            stats = ctx.get_summary_statistics(date_str)
            logger.info(f"📊 当日统计: {stats}")

    # ════════════════════════════════════════
    # 模块输出保存
    # ════════════════════════════════════════

    def save_module_output(self, module: 'Module', sim_date: str):
        """从 module.output() 读取数据并写入数据库。

        Args:
            module: Module 实例
            sim_date: 日期字符串 YYYY-MM-DD
        """
        if self.db is None:
            logger.debug("无 DB 连接，跳过模块输出持久化")
            return

        from src.models.module import OUTPUT_REGISTRY
        from ...utils.normalization import normalize_identifiers

        # 推导 module_name: e.g., 'M1' → 'module1'
        module_config = getattr(module, 'module_config', '')
        if module_config.startswith('M') and module_config[1:].isdigit():
            db_key = f"module{module_config[1:]}"
        else:
            db_key = module_config.lower()

        table_map = OUTPUT_REGISTRY.get(db_key, {})
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
            self._writer.write(table_name, df)
            written += 1

        if written:
            logger.info(
                "🗄️ 已写入 %s 输出到数据库 (%d 张表)",
                module_config, written,
            )

    # ════════════════════════════════════════
    # 配置写入（独立于 DQ，仅 DQ 通过时调用）
    # ════════════════════════════════════════

    def save_config(
        self,
        config_name: str,
        config_data: dict[str, pd.DataFrame],
        now: datetime | None = None,
    ) -> None:
        """将配置 DataFrame 写入对应 ``cfg_*`` 表。

        与 DQ 结果写入完全解耦：调用方自行决定何时写（DQ 通过后）。

        Args:
            config_name: 配置名称。
            config_data: 已加载的配置表数据 ``{sheet_name: DataFrame}``。
            now: 写入时间；默认 ``datetime.now()``。
        """
        if self.db is None:
            logger.debug("无 DB 连接，跳过配置写入")
            return
        if now is None:
            now = datetime.now()

        written = 0
        for sheet_name, df in (config_data or {}).items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            cfg_table = self._sheet_to_cfg_table(sheet_name)
            if cfg_table is None:
                continue
            try:
                # 删除该 config_name 的旧数据再写入（表不存在时安全跳过）
                self._writer.delete(cfg_table, {"config_name": config_name})
                write_df = self._inject_config_meta(df.copy(), config_name, now)
                self._writer.write(cfg_table, write_df)
                written += 1
            except Exception as e:
                logger.warning(f"写回配置表 {cfg_table} 失败: {e}")

        if written:
            logger.info(f"已写回 {written} 张配置表")

    # ════════════════════════════════════════
    # 运行事件 + DQ 明细（生命周期）
    # ════════════════════════════════════════
    #
    # run 事件状态机：
    #   start_run_event       → dq_status='running'
    #   mark_run_event_cached → dq_status='cached'   （hash 命中缓存）
    #   finalize_run_event    → dq_status='passed'/'blocked'/'skipped'

    def start_run_event(
        self, run_id: str, config_name: str, config_hash: str
    ) -> None:
        """orch 启动时插入一条 run 事件（``dq_status='running'``）。

        同一 runid 重复进入时通过 ON CONFLICT 重置为 running。
        """
        if self.db is None:
            return
        now = datetime.now()
        from psycopg import sql
        stmt = sql.SQL(
            "INSERT INTO {tbl} "
            "(runid, config_name, config_hash, dq_status, started_at, db_write_time) "
            "VALUES (%s, %s, %s, 'running', %s, %s) "
            "ON CONFLICT (runid) DO UPDATE SET "
            "config_name = EXCLUDED.config_name, "
            "config_hash = EXCLUDED.config_hash, "
            "dq_status = 'running', "
            "started_at = EXCLUDED.started_at, "
            "finished_at = NULL, "
            "errors = NULL, warnings = NULL, hard_blocks = NULL, "
            "db_write_time = EXCLUDED.db_write_time"
        ).format(tbl=sql.Identifier("orch_run_event"))
        try:
            self.db.execute(stmt, (run_id, config_name, config_hash, now, now))
        except Exception as e:
            logger.warning(f"start_run_event 失败: {e}")

    def mark_run_event_cached(self, run_id: str) -> None:
        """hash 命中缓存，将当前 run 事件标记为 ``cached``。"""
        if self.db is None:
            return
        now = datetime.now()
        from psycopg import sql
        stmt = sql.SQL(
            "UPDATE {tbl} SET dq_status = 'cached', finished_at = %s, "
            "db_write_time = %s WHERE runid = %s"
        ).format(tbl=sql.Identifier("orch_run_event"))
        try:
            self.db.execute(stmt, (now, now, run_id))
        except Exception as e:
            logger.warning(f"mark_run_event_cached 失败: {e}")

    def finalize_run_event(
        self,
        run_id: str,
        config_name: str,
        dq_result: dict[str, Any] | None = None,
        *,
        status: str | None = None,
    ) -> None:
        """跑完 DQ 后，**同一事务内**写明细 + 更新 run 事件结论。

        Args:
            run_id: 运行标识。
            config_name: 配置名。
            dq_result: DQ 结果；``None`` 表示 skip_dq（标 ``skipped``）。
            status: 显式状态覆盖；默认按 ``dq_result.passed`` 推导
                passed/blocked（dq_result 为 None → skipped）。
        """
        if self.db is None:
            return
        now = datetime.now()

        if status is None:
            if dq_result is None:
                status = "skipped"
            else:
                status = "passed" if dq_result.get("passed") else "blocked"

        detail_rows = self._build_dq_detail_rows(
            run_id, config_name, dq_result, now, status
        )
        totals = self._summarize_dq(dq_result)

        try:
            with self.db.get_cursor(commit=True) as cur:
                # 先删后写，保证同一 runid 幂等
                cur.execute(
                    "DELETE FROM orch_dq_detail WHERE runid = %s", (run_id,)
                )
                self._copy_dq_detail(cur, detail_rows)
                cur.execute(
                    "UPDATE orch_run_event SET dq_status = %s, errors = %s, "
                    "warnings = %s, hard_blocks = %s, finished_at = %s, "
                    "db_write_time = %s WHERE runid = %s",
                    (
                        status,
                        totals["errors"],
                        totals["warnings"],
                        totals["hard_blocks"],
                        now,
                        now,
                        run_id,
                    ),
                )
            logger.info(
                f"DQ run 事件已落定: runid={run_id} status={status} "
                f"明细 {len(detail_rows)} 行"
            )
        except Exception as e:
            logger.warning(f"finalize_run_event 失败: {e}")

    @staticmethod
    def _build_dq_detail_rows(
        run_id: str,
        config_name: str,
        dq_result: dict[str, Any] | None,
        now: datetime,
        status: str,
    ) -> list[dict]:
        """按 sheet 聚合 DQ issues → orch_dq_detail 行（含 __summary__ 兜底）。"""
        issues = (dq_result or {}).get("issues", []) or []

        issues_by_sheet: dict[str, list[dict]] = {}
        for issue in issues:
            sheet = issue.get("sheet", "") or ""
            issues_by_sheet.setdefault(sheet, []).append(issue)

        rows: list[dict] = []
        for sheet_name in sorted(issues_by_sheet.keys()):
            sheet_issues = issues_by_sheet.get(sheet_name, [])
            rows.append({
                "runid": run_id,
                "config_name": config_name,
                "sheet_name": sheet_name,
                "status": status,
                "errors": sum(
                    1 for i in sheet_issues
                    if i.get("severity") in {"ERROR", "CRITICAL"}
                ),
                "warnings": sum(
                    1 for i in sheet_issues if i.get("severity") == "WARNING"
                ),
                "hard_blocks": sum(
                    1 for i in sheet_issues if i.get("severity") == "hard_block"
                ),
                "issues_json": json.dumps(
                    sheet_issues, ensure_ascii=False, default=str
                ),
                "db_write_time": now,
            })

        # 零问题（passed/skipped）仍写一行 __summary__，保证接口一致
        if not rows:
            rows.append({
                "runid": run_id,
                "config_name": config_name,
                "sheet_name": "__summary__",
                "status": status,
                "errors": 0,
                "warnings": 0,
                "hard_blocks": 0,
                "issues_json": "[]",
                "db_write_time": now,
            })
        return rows

    @staticmethod
    def _summarize_dq(dq_result: dict[str, Any] | None) -> dict[str, int]:
        """从 dq_result.summary 提取总计数。"""
        summary = (dq_result or {}).get("summary", {}) or {}
        return {
            "errors": summary.get("errors", 0),
            "warnings": summary.get("warnings", 0),
            "hard_blocks": summary.get("hard_blocks", 0),
        }

    @staticmethod
    def _copy_dq_detail(cursor, rows: list[dict]) -> None:
        """在给定事务游标内 COPY 明细行（不自行提交）。"""
        if not rows:
            return
        from psycopg import sql
        df = pd.DataFrame(rows)
        columns = list(df.columns)
        col_list = sql.SQL(', ').join(sql.Identifier(c) for c in columns)
        copy_sql = sql.SQL("COPY orch_dq_detail ({}) FROM STDIN").format(col_list)
        with cursor.copy(copy_sql) as copy:
            for row in df.itertuples(index=False, name=None):
                copy.write_row([str(v) if v is not None else None for v in row])

    def _sheet_to_cfg_table(self, sheet_name: str) -> str | None:
        """将 Sheet 名映射为 cfg_* 物理表名（Django 风格）。

        从 CONFIG_TABLE_REGISTRY 读 Model.__tablename__；
        不命中时走正则 snake_case 兜底。

        Args:
            sheet_name: 配置表 Sheet 名（如 M1_InitialInventory）。

        Returns:
            cfg_* 物理表名（如 cfg_m1_initialinventory）；无法映射时返回 None。
        """
        # ── Django 风格：直接从模型注册表读 __tablename__ ──
        try:
            from src.models.cfg import CONFIG_TABLE_REGISTRY
            model_cls = CONFIG_TABLE_REGISTRY.get(sheet_name)
            if model_cls is not None:
                return model_cls.__tablename__
        except Exception:
            pass
        # ── 兜底：正则 snake_case ──
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

    # ════════════════════════════════════════
    # 工具方法
    # ════════════════════════════════════════

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

    # ════════════════════════════════════════
    # 快照恢复 / checkpoint（从 Orchestrator 收敛至此）
    # ════════════════════════════════════════

    def restore_state(self, ctx, run_id: str, sim_date: str):
        """从快照表恢复状态到 StateContext。"""
        if self.db is None:
            return
        self.db.snapshot.restore(run_id, sim_date, ctx)

    def save_checkpoint(self, run_id: str, status: str = 'running'):
        """保存/更新运行元信息（checkpoint）。"""
        if self.db is None:
            return
        self.db.snapshot.save_checkpoint(
            run_id=run_id,
            config_name=self.config_name,
            last_batch_end=str(self._orch.start_date),
            status=status,
        )

    def load_checkpoint(self, run_id: str) -> dict | None:
        """加载运行元信息（checkpoint）。"""
        if self.db is None:
            return None
        return self.db.snapshot.load_checkpoint(run_id)
