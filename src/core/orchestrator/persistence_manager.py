"""持久化管理器。

职责：将 StateContext 和 Module 的输出数据写入数据库。

迁移版改动：
- __init__ 内创建 DBWriter，所有 self.db.write_df → self.writer.write
- save_config 中 self.db.delete_where → self.writer.delete
- _sheet_to_cfg_table() 改为 Django 风格：从 CONFIG_TABLE_REGISTRY 读 Model.__tablename__
- 建表统一由 Orchestrator._init_db 启动时 migrate() 完成，本类不再 _ensure_run_tables
- OUTPUT_TABLE_MAPPING 从 module/viewcontext 注册表派生（不依赖 table_mapping.py）
"""

import json
import logging
from contextlib import contextmanager
from datetime import datetime
from typing import TYPE_CHECKING, Any

import pandas as pd
from psycopg import sql

if TYPE_CHECKING:
    from ...modules.state_context import StateContext
    from ...modules.module import Module

logger = logging.getLogger(__name__)


class PersistenceManager:
    """持久化总管：只写数据库。

    表映射使用 module / viewcontext 注册表派生。
    DBWriter 延迟创建（首次写入时才初始化），确保 db 连接已建立。
    """

    def __init__(self, orch):
        self._orch = orch
        self._writer = None  # 延迟创建，等 db 连接建立后

    @property
    def writer(self):
        """延迟初始化 DBWriter，确保 db 连接已建立。"""
        if self._writer is None and self.db is not None:
            from ...io.writer import DBWriter
            self._writer = DBWriter(self.db)
        return self._writer

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
    # 批量事务（写入优化）
    # ════════════════════════════════════════

    @contextmanager
    def batch_transaction(self):
        """将上下文内所有写入合并为一个事务，减少 fsync 次数。

        psycopg3 的 ``conn.transaction()`` 在 autocommit=False 时创建 SAVEPOINT，
        而非独立事务。因此只需将 autocommit 临时关掉，内部各个 write_df 的
        ``conn.transaction()`` 就会自动退化为 savepoint，最终由外层 ``commit()``
        一次性落盘。

        用法：
            with persistence.batch_transaction():
                persistence.save_daily_state(ctx, date_str)
                persistence.save_module_output(m1, date_str)
                persistence.save_checkpoint(run_id, current_date=date_str)
                # 所有写入在一个事务中，从 N 次 fsync 降为 1 次
        """
        if self.db is None:
            yield
            return

        conn = self.db.connect()
        was_autocommit = conn.autocommit
        conn.autocommit = False
        try:
            yield
            conn.commit()
            logger.debug("batch_transaction: 已提交批量事务")
        except Exception:
            conn.rollback()
            logger.warning("batch_transaction: 已回滚批量事务")
            raise
        finally:
            conn.autocommit = was_autocommit

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
            self._write_idempotent(table_name, df, run_id, sim_date)

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
                self._write_idempotent(tbl, inv_log, run_id, sim_date)

        # 3) 写 daily_logs（写入后清空，避免历史日志重复写入）
        if hasattr(ctx, 'daily_logs') and ctx.daily_logs:
            logs_df = pd.DataFrame(ctx.daily_logs)
            logs_df = self._inject_meta(logs_df, run_id, sim_date, now)
            tbl = view_table_map.get(
                'daily_logs', 'viewcontext_daily_logs',
            )
            self._write_idempotent(tbl, logs_df, run_id, sim_date)
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
                self._write_idempotent(tbl, audit, run_id, sim_date)

        # 5) 写 M4 跨天状态（产线换产连续性 + 已分配产能防重复分配）
        m4_line_states = getattr(ctx, 'm4_line_states', None)
        if m4_line_states and date_str in m4_line_states:
            ls = m4_line_states[date_str]
            if ls:
                rows = []
                for line_name, st in ls.items():
                    ci = st.get('changeover_info') or {}
                    rows.append({
                        'line': line_name,
                        'last_material': st.get('last_material'),
                        'last_location': st.get('last_location'),
                        'last_activity': st.get('last_activity'),
                        'remaining_time': ci.get('remaining_time'),
                        'changeover_id': ci.get('changeover_id'),
                        'to_material': ci.get('to_material'),
                    })
                ls_df = pd.DataFrame(rows)
                ls_df = self._inject_meta(ls_df, run_id, sim_date, now)
                tbl = view_table_map.get('m4_line_states')
                self._write_idempotent(tbl, ls_df, run_id, sim_date)

        m4_allocated = getattr(ctx, 'm4_allocated_capacity', None)
        if m4_allocated and date_str in m4_allocated:
            ac = m4_allocated[date_str]
            if ac:
                ac_df = pd.DataFrame([
                    {'capacity_key': k, 'allocated_hours': v}
                    for k, v in ac.items()
                ])
                ac_df = self._inject_meta(ac_df, run_id, sim_date, now)
                tbl = view_table_map.get('m4_allocated_capacity')
                self._write_idempotent(tbl, ac_df, run_id, sim_date)

        logger.info(f"🗄️ 已写入 {date_str} 每日状态到数据库")

        # 注：progress_date 断点推进已移出本方法，由调用方（主循环）显式调用
        # orch.save_checkpoint(current_date=...) 完成——让「写数据」与「推进断点」
        # 在调用链上分离，save_daily_state 只负责写数据。

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
            self._write_idempotent(table_name, df, run_id, sim_date_str)
            written += 1

        if written:
            logger.info(
                "🗄️ 已写入 %s 输出到数据库 (%d 张表)",
                module_config, written,
            )

    def save_summary_outputs(
        self,
        ctx: 'StateContext',
        start_date: str,
        end_date: str,
    ) -> dict[str, int]:
        """从 StateContext 保存全周期 Summary 到既有 ``summary_*`` 表。

        Summary 只应在全部日度模块输出和状态保存成功后调用。每张表按
        ``run_id`` 整体替换，避免重跑时保留旧周期汇总行；表名完全由
        ``SUMMARY_REGISTRY`` 提供，禁止回退至旧 ``summary_output_*`` 命名。
        """
        if self.db is None:
            logger.debug("无 DB 连接，跳过 StateContext Summary 持久化")
            return {}

        from src.models.viewcontext import SUMMARY_REGISTRY
        from ...utils.normalization import normalize_identifiers

        outputs = ctx.build_summary_outputs(
            start_date=start_date,
            end_date=end_date,
            config_dict=self._orch.all_config,
        )
        run_id = self._run_id
        sim_date = pd.Timestamp(end_date).strftime('%Y-%m-%d')
        now = datetime.now()
        written: dict[str, int] = {}

        for output_name, table_name in SUMMARY_REGISTRY.items():
            # 无行的 Summary 仍是有效计算结果。动态 DataFrame 建表路径不能
            # 从空表推断列，故仅删除当前 run 的旧行并返回 0。
            self.writer.delete(table_name, {'run_id': run_id})
            frame = outputs.get(output_name, pd.DataFrame())
            if not isinstance(frame, pd.DataFrame) or frame.empty:
                written[table_name] = 0
                continue
            frame = normalize_identifiers(frame.copy())
            frame = self._inject_meta(frame, run_id, sim_date, now)
            self._ensure_summary_columns(table_name, frame)
            self.writer.write(table_name, frame)
            written[table_name] = len(frame)

        logger.info(
            "🗄️ 已写入 StateContext Summary: %d 张表, %d 行",
            len(written), sum(written.values()),
        )
        return written

    def _ensure_summary_columns(self, table_name: str, frame: pd.DataFrame) -> None:
        """为迁移预建的 Summary 表补齐 DataFrame 中新增的业务列。

        Summary 没有稳定的统一业务 schema。迁移仅预建四个元数据列；此处在
        首次写入前将其余列以 TEXT 幂等加入，配合 DB.write_df 的表列对齐逻辑
        保留所有业务字段而非静默丢弃。运行元数据的日期/数值类型不受影响。
        """
        if self.db is None or not hasattr(self.db, 'execute'):
            return
        expected = [str(column).strip().lower() for column in frame.columns]
        try:
            with self.db.get_cursor(commit=True) as cursor:
                existing = self.db._get_table_columns(cursor, table_name)
                for column in expected:
                    if column in existing:
                        continue
                    cursor.execute(
                        sql.SQL("ALTER TABLE {} ADD COLUMN IF NOT EXISTS {} TEXT").format(
                            self.db._qualified(table_name), sql.Identifier(column)
                        )
                    )
            self.db._col_type_cache.pop(table_name, None)
        except Exception as exc:
            raise RuntimeError(
                f"Summary 表 {table_name} 扩展业务列失败"
            ) from exc

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
                self.writer.delete(cfg_table, {"config_name": config_name})
                write_df = self._inject_config_meta(df.copy(), config_name, now)
                self.writer.write(cfg_table, write_df)
                written += 1
            except Exception as e:
                logger.warning(f"写回配置表 {cfg_table} 失败: {e}")

        if written:
            logger.info(f"已写回 {written} 张配置表")

    # ════════════════════════════════════════
    # 运行事件 + DQ 明细（生命周期）
    # ════════════════════════════════════════
    #
    # 两套独立状态：
    #   dq_status（DQ 检测）：running → cached / passed / blocked / skipped
    #   status（orch 执行）：running → finished（续跑判断依据）
    #
    # finished_at 只在 status='finished'（mark_finished）时写；DQ 完成（finalize）不写。
    # total_days 首次运行记录（iter_dates 分母），current_date 每日推进（断点）。

    def start_run_event(
        self, run_id: str, config_name: str, config_hash: str,
        total_days: int | None = None,
    ) -> None:
        """orch 启动时插入一条 run 事件（``dq_status='running'``, ``status='running'``）。

        ``total_days`` 首次写入后不再覆盖（ON CONFLICT 时保留旧值，供续跑读分母）。
        """
        if self.db is None:
            return
        now = datetime.now()
        from psycopg import sql
        stmt = sql.SQL(
            "INSERT INTO {tbl} "
            "(runid, config_name, config_hash, dq_status, status, "
            " total_days, started_at, db_write_time) "
            "VALUES (%s, %s, %s, 'running', 'running', %s, %s, %s) "
            "ON CONFLICT (runid) DO UPDATE SET "
            "config_name = EXCLUDED.config_name, "
            "config_hash = EXCLUDED.config_hash, "
            "dq_status = 'running', "
            "status = 'running', "
            "finished_at = NULL, "
            "progress_date = NULL, "
            "total_days = COALESCE(orch_run_event.total_days, EXCLUDED.total_days), "
            "db_write_time = EXCLUDED.db_write_time"
        ).format(tbl=self.db._qualified("orch_run_event"))
        try:
            self.db.execute(stmt, (run_id, config_name, config_hash,
                                   total_days, now, now))
        except Exception as e:
            logger.warning(f"start_run_event 失败: {e}")

    def mark_run_event_cached(self, run_id: str) -> None:
        """hash 命中缓存，将当前 run 事件标记为 ``dq_status='cached'``。

        缓存命中 = 直接复用已校验配置 = 该 run 的 DQ 生命周期结束；
        但 ``status`` 不在此处置 finished（仿真仍待跑），仅 DQ 侧收尾。
        """
        if self.db is None:
            return
        now = datetime.now()
        from psycopg import sql
        stmt = sql.SQL(
            "UPDATE {tbl} SET dq_status = 'cached', "
            "db_write_time = %s WHERE runid = %s"
        ).format(tbl=self.db._qualified("orch_run_event"))
        try:
            self.db.execute(stmt, (now, run_id))
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
        """跑完 DQ 后，**同一事务内**写明细 + 更新 dq_status。

        注意：此处 **不写 finished_at**（DQ 完成 ≠ orch 跑完）；
        finished_at 仅在 ``mark_finished``（orch 真正结束）时写。
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

        try:
            with self.db.get_cursor(commit=True) as cur:
                # 先删后写，保证同一 runid 幂等
                cur.execute(
                    f"DELETE FROM {self.db.qualified_name('orch_dq_detail')} "
                    "WHERE runid = %s", (run_id,)
                )
                self._copy_dq_detail(cur, detail_rows)
                # 仅落 DQ 结论；不碰 finished_at / status（orch 执行态）
                _evt = self.db.qualified_name("orch_run_event")
                cur.execute(
                    f"UPDATE {_evt} SET dq_status = %s, "
                    "db_write_time = %s WHERE runid = %s",
                    (status, now, run_id),
                )
            logger.info(
                f"DQ run 事件已落定: runid={run_id} dq_status={status} "
                f"明细 {len(detail_rows)} 行"
            )
        except Exception as e:
            logger.warning(f"finalize_run_event 失败: {e}")

    # ── orch 执行态（status / current_date / finished_at）──────────────

    def mark_finished(self, run_id: str) -> None:
        """orch 整个流程真正结束后调用：置 status='finished' 并写 finished_at。"""
        if self.db is None:
            return
        now = datetime.now()
        from psycopg import sql
        stmt = sql.SQL(
            "UPDATE {tbl} SET status = 'finished', finished_at = %s, "
            "db_write_time = %s WHERE runid = %s"
        ).format(tbl=self.db._qualified("orch_run_event"))
        try:
            self.db.execute(stmt, (now, now, run_id))
            logger.info(f"orch run 已结束: runid={run_id} status=finished")
        except Exception as e:
            logger.warning(f"mark_finished 失败: {e}")

        # 重跑完成后清 m1 快照表（数据已无用，避免残留）
        self._cleanup_m1_snapshot(run_id)

    def _cleanup_m1_snapshot(self, run_id: str):
        """清空该 run_id 的 m1 快照数据（重跑完成后调用）。"""
        if self.db is None:
            return
        from src.models.resume import M1_SNAPSHOT_REGISTRY
        for table_name in M1_SNAPSHOT_REGISTRY.values():
            try:
                self.db.execute(
                    f"DELETE FROM {self.db.qualified_name(table_name)} WHERE run_id = %s",
                    (run_id,)
                )
            except Exception:
                pass  # 表可能不存在（首次运行无快照）

    def update_orch_status(
        self, run_id: str, current_date=None, status: str | None = None,
    ) -> None:
        """更新 orch 执行态：progress_date（断点日期）和/或 status。

        ``current_date`` 每日推进时调用；``status='running'`` 默认不变。
        DB 列名为 ``progress_date``（避免与 PG 内置 ``current_date`` 冲突）。
        """
        if self.db is None:
            return
        sets, params = [], []
        if current_date is not None:
            sets.append("progress_date = %s")
            params.append(str(pd.Timestamp(current_date).strftime('%Y-%m-%d')))
        if status is not None:
            sets.append("status = %s")
            params.append(status)
        if not sets:
            return
        sets.append("db_write_time = %s")
        params.append(datetime.now())
        params.append(run_id)
        from psycopg import sql
        stmt = sql.SQL(
            "UPDATE {tbl} SET " + ", ".join(sets) + " WHERE runid = %s"
        ).format(tbl=self.db._qualified("orch_run_event"))
        try:
            self.db.execute(stmt, tuple(params))
        except Exception as e:
            logger.warning(f"update_orch_status 失败: {e}")

    def find_unfinished(self, config_name: str | None) -> dict | None:
        """按 config_name 查最近一条可续跑的 run（orch 未完成且确实跑过至少一天）。

        返回 ``{run_id, current_date, total_days}`` 或 None。
        续跑资格仅由 ``orch_run_event.status`` 和 ``progress_date`` 判断，与
        ``dq_status`` 无关：DQ 决定新配置是否写回，运行状态决定是否继续既有 run。
        仅当 progress_date 不为 NULL 时才续跑（progress_date=NULL 说明刚 start 还没跑任何一天，
        是当前正在跑的全新运行，不应误续跑）。
        """
        if self.db is None or not config_name:
            return None
        try:
            _evt = self.db.qualified_name("orch_run_event")
            rows = self.db.execute_query(
                f"SELECT runid, progress_date, total_days FROM {_evt} "
                "WHERE config_name = %s AND status <> 'finished' "
                "AND progress_date IS NOT NULL "
                "ORDER BY started_at DESC NULLS LAST LIMIT 1",
                (config_name,),
            )
            if not rows:
                return None
            runid, cur_date, total_days = rows[0]
            return {
                'run_id': runid,
                'current_date': str(cur_date) if cur_date is not None else None,
                'total_days': int(total_days) if total_days is not None else None,
            }
        except Exception as e:
            logger.warning(f"find_unfinished 失败（视为全新运行）: {e}")
            return None

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

    def _copy_dq_detail(self, cursor, rows: list[dict]) -> None:
        """在给定事务游标内 COPY 明细行（不自行提交）。"""
        if not rows:
            return
        from psycopg import sql
        df = pd.DataFrame(rows)
        columns = list(df.columns)
        col_list = sql.SQL(', ').join(sql.Identifier(c) for c in columns)
        copy_sql = sql.SQL("COPY {} ({}) FROM STDIN").format(
            self.db._qualified("orch_dq_detail"), col_list
        )
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
        # 配置身份由本次运行显式传入，不能沿用 Excel/CSV 中可能存在的旧
        # ``config_name`` 列；否则并行测试或场景副本会把 cfg_* 写到错误身份。
        if 'config_name' in df.columns:
            df['config_name'] = config_name
        else:
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

    def _write_idempotent(self, table_name: str, df: pd.DataFrame,
                          run_id: str, sim_date: str) -> None:
        """先删当天该表旧行（WHERE run_id, sim_date）再写，保证重跑幂等。

        表不存在时 writer.delete 走 db.delete_where(safe=True) 静默返回 0，
        首写由 write_df 自动建表；重跑时先清当天残行再 COPY，不会翻倍。
        与 save_m1_snapshot 的「先 DELETE 再写」一致。
        """
        self.writer.delete(table_name, {'run_id': run_id, 'sim_date': sim_date})
        self.writer.write(table_name, df)

    # ════════════════════════════════════════
    # 快照恢复（从 ViewContext 表读取，不依赖 sim_state_*）
    # ════════════════════════════════════════

    def restore_state_from_views(self, ctx, run_id: str, sim_date: str):
        """从 viewcontext_* 表恢复 ctx 状态（续跑用）。

        读取指定 sim_date（或最近前一天）的 ViewContext views，
        将 DataFrame 还原为 ctx 的内部 dict/list 格式。
        不依赖 sim_state_* 表（那些表已废弃，ctx 状态由 ViewContext 持久化）。
        """
        if self.db is None:
            return
        from src.models.viewcontext import VIEWCONTEXT_REGISTRY
        from ...utils.normalization import normalize_identifiers

        def _read_view(view_key: str) -> "pd.DataFrame | None":
            """按 view_key 读对应 viewcontext 表；表不存在则返回 None（续跑早期阶段表可能尚未建）。"""
            tbl = VIEWCONTEXT_REGISTRY.get(view_key)
            if not tbl:
                return None
            if not self.db.table_exists(tbl):
                logger.debug(f"⏭️ viewcontext 表 {tbl} 尚不存在，跳过恢复 {view_key}")
                return None
            try:
                df = self.db.read(tbl, run_id=run_id, sim_date=sim_date)
            except Exception as e:
                logger.warning(f"读取 viewcontext 表 {tbl} 失败（跳过 {view_key}）: {e}")
                return None
            if df is None or df.empty:
                logger.debug(f"viewcontext {tbl} sim_date={sim_date} 无数据")
                return None
            logger.debug(
                f"viewcontext {tbl} sim_date={sim_date} 读到 {len(df)} 行，列={df.columns.tolist()}"
            )
            return df

        restored = []

        # 注：normalize_identifiers 是 DataFrame 级（入参 df，默认处理 material/
        # location/sending/receiving 等列），不可对单个标量调用。每张表先整体规范化一次。

        # 1) unrestricted_inventory → ctx.unrestricted_inventory {(material, location): qty}
        df = _read_view('unrestricted_inventory')
        if df is not None and not df.empty:
            df = normalize_identifiers(df)
            qty = pd.to_numeric(df['quantity'], errors='coerce').fillna(0).astype(int)
            ctx.unrestricted_inventory = dict(zip(zip(df['material'], df['location']), qty))
            restored.append(f"unrestricted_inventory({len(ctx.unrestricted_inventory)})")

        # 2) open_deployment → ctx.open_deployment {uid: record}
        df = _read_view('open_deployment')
        if df is not None and not df.empty:
            df = normalize_identifiers(df)
            df['deployed_qty'] = pd.to_numeric(df['deployed_qty'], errors='coerce').fillna(0.0)
            recs = df.to_dict('records')
            ctx.open_deployment = {
                str(r.get('ori_deployment_uid', '')): {
                    'material': r.get('material', ''),
                    'sending': r.get('sending', ''),
                    'receiving': r.get('receiving', ''),
                    'planned_deployment_date': str(r.get('planned_deployment_date', '')),
                    'deployed_qty': float(r['deployed_qty']),
                    'demand_element': str(r.get('demand_element', '')),
                }
                for r in recs
            }
            restored.append(f"open_deployment({len(ctx.open_deployment)})")

        # 3) planning_intransit → ctx.in_transit {uid: record}
        df = _read_view('planning_intransit')
        if df is not None and not df.empty:
            df = normalize_identifiers(df)
            df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0.0)
            recs = df.to_dict('records')
            ctx.in_transit = {
                str(r.get('transit_uid', '')): {
                    'material': r.get('material', ''),
                    'sending': r.get('sending', ''),
                    'receiving': r.get('receiving', ''),
                    'actual_ship_date': str(r.get('actual_ship_date', '')),
                    'actual_delivery_date': str(r.get('actual_delivery_date', '')),
                    'quantity': float(r['quantity']),
                    'ori_deployment_uid': str(r.get('ori_deployment_uid', '')),
                    'vehicle_uid': str(r.get('vehicle_uid', '')),
                }
                for r in recs
            }
            restored.append(f"in_transit({len(ctx.in_transit)})")

        # 4) production_plan_backlog → ctx.production_plan_backlog [record...]
        df = _read_view('production_plan_backlog')
        if df is not None and not df.empty:
            df = normalize_identifiers(df)
            if 'available_date' in df.columns:
                df['available_date'] = pd.to_datetime(df['available_date'], errors='coerce')
            if 'quantity' in df.columns:
                df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0)
            cols = ['material', 'location', 'available_date', 'quantity']
            ctx.production_plan_backlog = df[cols].to_dict('records')
            restored.append(f"production_plan_backlog({len(ctx.production_plan_backlog)})")

        # 5) M4 line_states（展平列 → 嵌套 dict）
        df = _read_view('m4_line_states')
        if df is not None and not df.empty:
            for sim_d, group in df.groupby('sim_date'):
                d_str = str(sim_d) if hasattr(sim_d, 'strftime') else str(sim_d)
                ctx.m4_line_states[d_str] = {}
                for row in group.to_dict('records'):
                    remaining = row.get('remaining_time')
                    ci = None
                    if remaining is not None and pd.notna(remaining) and float(remaining) > 0:
                        ci = {
                            'remaining_time': float(remaining),
                            'changeover_id': row.get('changeover_id'),
                            'to_material': row.get('to_material'),
                        }
                    ctx.m4_line_states[d_str][row['line']] = {
                        'last_material': row.get('last_material'),
                        'last_location': row.get('last_location'),
                        'last_activity': row.get('last_activity'),
                        'changeover_info': ci,
                    }
            restored.append(f"m4_line_states({len(ctx.m4_line_states)} days)")

        # 6) M4 allocated_capacity（key-value → dict）
        df = _read_view('m4_allocated_capacity')
        if df is not None and not df.empty:
            for sim_d, group in df.groupby('sim_date'):
                d_str = str(sim_d) if hasattr(sim_d, 'strftime') else str(sim_d)
                ctx.m4_allocated_capacity[d_str] = {
                    row['capacity_key']: float(row['allocated_hours'])
                    for row in group.to_dict('records')
                }
            restored.append(f"m4_allocated_capacity({len(ctx.m4_allocated_capacity)} days)")

        # 7) M3 净需求：它本来就是 module3_output_netdemand 正式输出，
        # 不复制到 viewcontext。恢复时按同一 run 的上一完成日读取，供下一日
        # M4 保持一日 lag 消费。
        m3_table = 'module3_output_netdemand'
        if hasattr(ctx, 'm3_net_demand_by_date') and self.db.table_exists(m3_table):
            try:
                m3_df = self.db.read(m3_table, run_id=run_id, sim_date=sim_date)
                if m3_df is not None:
                    ctx.m3_net_demand_by_date[str(sim_date)[:10]] = m3_df.drop(
                        columns=['run_id', 'sim_date', 'config_name', 'db_write_time'],
                        errors='ignore',
                    ).copy(deep=True)
                    restored.append(f"m3_net_demand({len(m3_df)})")
            except Exception as exc:
                logger.warning("恢复 M3 净需求失败（将由首日空输入保护）: %s", exc)

        if restored:
            logger.info(
                f"🔄 ctx 状态已从 ViewContext 恢复: run_id={run_id}, sim_date={sim_date} "
                f"→ {', '.join(restored)}"
            )
        else:
            logger.info(
                f"🔄 ctx 无可恢复 ViewContext 状态（表尚未落库）: "
                f"run_id={run_id}, sim_date={sim_date}"
            )

    def save_m1_snapshot(self, m1, date_str: str):
        """写 m1.prepare 后 4 属性到各自独立表（由 write_df 动态建表）。"""
        if self.db is None:
            return
        from src.models.resume import M1_SNAPSHOT_REGISTRY
        run_id = self._run_id
        for attr, table_name in M1_SNAPSHOT_REGISTRY.items():
            df = getattr(m1, attr, None)
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                continue
            # polars → pandas
            try:
                import polars as pl
                if isinstance(df, pl.DataFrame):
                    df = df.to_pandas()
            except ImportError:
                pass
            if not isinstance(df, pd.DataFrame):
                continue
            # 先删旧数据再写（幂等）；表不存在则跳过 DELETE，让 write_df 自动建表
            if self.db.table_exists(table_name):
                self.db.execute(
                    f"DELETE FROM {self.db.qualified_name(table_name)} "
                    "WHERE run_id = %s AND sim_date = %s",
                    (run_id, date_str),
                )
            # 注入元数据列 + 写入
            out = df.copy()
            out.insert(0, 'run_id', run_id)
            out.insert(1, 'sim_date', date_str)
            # 日期列转字符串（write_df 全 TEXT 写入）
            for col in out.columns:
                if pd.api.types.is_datetime64_any_dtype(out[col]):
                    out[col] = out[col].astype(str)
            self.db.write_df(table_name, out)

    def load_m1_snapshot(self, run_id: str, sim_date: str) -> dict | None:
        """读 m1 4 属性快照，返回 {attr: DataFrame} 或 None。"""
        if self.db is None:
            return None
        from src.models.resume import M1_SNAPSHOT_REGISTRY
        result = {}
        for attr, table_name in M1_SNAPSHOT_REGISTRY.items():
            # 表可能尚未建（首次运行无快照，或续跑发生在快照写入之前）
            if not self.db.table_exists(table_name):
                continue
            try:
                df = self.db.read(table_name, run_id=run_id, sim_date=sim_date)
            except Exception as e:
                logger.warning(f"读取 m1 快照表 {table_name} 失败（跳过 {attr}）: {e}")
                continue
            if df is None or df.empty:
                continue
            # 剥离元数据列
            meta_cols = {'run_id', 'sim_date', 'config_name', 'db_write_time'}
            keep = [c for c in df.columns if c not in meta_cols]
            df = df[keep].copy()
            # 恢复日期列类型
            for col in ('date', 'available_date', 'simulation_date',
                         'order_date', 'delivery_date', 'ship_date', 'created_date'):
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            # 读边界适配：polars 引擎下转成 polars，喂给 _PolarsBackend
            # （db.read 恒返回 pandas；这 4 属性由 polars 后端消费，pandas 会让
            #   merge_with_history 等处的 is_empty()/pl.col() 崩）。
            if getattr(self._orch, 'engine', 'pandas') == 'polars':
                from ...utils.df_convert import pandas_to_polars
                df = pandas_to_polars(df)
            result[attr] = df
        return result or None

    def save_checkpoint(self, run_id: str, status: str = 'running',
                        current_date=None):
        """更新 orch 执行态（写 orch_run_event）。

        续跑执行态统一落 orch_run_event（status/current_date/finished_at）；
        sim_checkpoint 表不再作为执行态源。
        """
        if self.db is None:
            return
        self.update_orch_status(run_id, status=status, current_date=current_date)

    def load_checkpoint(self, run_id: str) -> dict | None:
        """加载运行元信息（orch_run_event）。"""
        if self.db is None:
            return None
        try:
            _evt = self.db.qualified_name("orch_run_event")
            rows = self.db.execute_query(
                "SELECT runid, status, progress_date, total_days, dq_status "
                f"FROM {_evt} WHERE runid = %s",
                (run_id,),
            )
            if not rows:
                return None
            r = rows[0]
            return {
                'run_id': r[0],
                'status': r[1],
                'current_date': str(r[2]) if r[2] is not None else None,
                'total_days': int(r[3]) if r[3] is not None else None,
                'dq_status': r[4],
            }
        except Exception as e:
            logger.warning(f"load_checkpoint 失败: {e}")
            return None
