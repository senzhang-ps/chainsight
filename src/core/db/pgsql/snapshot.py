"""快照表读写。

替代原 checkpoint.py 的 JSONB 序列化方案。
将状态拆分为结构化的快照表，按 (run_id, sim_date) 组织，
便于增量读写和查询。

职责：
- save(): 将 StateContext 的各状态属性拆分写入对应的快照表
- restore(): 从快照表读取并还原到 StateContext
- save_checkpoint() / load_checkpoint(): 运行元信息管理
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .db import DB

logger = logging.getLogger(__name__)


# ── 快照表名 ──────────────────────────────────

SNAPSHOT_TABLES = {
    'inventory': 'sim_state_inventory',
    'intransit': 'sim_state_intransit',
    'open_deploy': 'sim_state_open_deploy',
    'backlog': 'sim_state_backlog',
    'runtime': 'sim_state_runtime',
}


# ── checkpoint 建表 SQL ──────────────────────

_CHECKPOINT_DDL = """
CREATE TABLE IF NOT EXISTS sim_checkpoint (
    run_id         TEXT PRIMARY KEY,
    config_name    TEXT NOT NULL,
    last_batch_end DATE NOT NULL,
    status         TEXT NOT NULL DEFAULT 'running',
    error_message  TEXT,
    updated_at     TIMESTAMP DEFAULT NOW()
);
"""

_INVENTORY_DDL = """
CREATE TABLE IF NOT EXISTS sim_state_inventory (
    run_id     TEXT NOT NULL,
    sim_date   DATE NOT NULL,
    material   TEXT NOT NULL,
    location   TEXT NOT NULL,
    quantity   BIGINT NOT NULL DEFAULT 0,
    PRIMARY KEY (run_id, sim_date, material, location)
);
"""

_INTRANSIT_DDL = """
CREATE TABLE IF NOT EXISTS sim_state_intransit (
    run_id       TEXT NOT NULL,
    sim_date     DATE NOT NULL,
    transit_uid  TEXT NOT NULL,
    data         JSONB NOT NULL,
    PRIMARY KEY (run_id, sim_date, transit_uid)
);
"""

_OPEN_DEPLOY_DDL = """
CREATE TABLE IF NOT EXISTS sim_state_open_deploy (
    run_id      TEXT NOT NULL,
    sim_date    DATE NOT NULL,
    deploy_uid  TEXT NOT NULL,
    data        JSONB NOT NULL,
    PRIMARY KEY (run_id, sim_date, deploy_uid)
);
"""

_BACKLOG_DDL = """
CREATE TABLE IF NOT EXISTS sim_state_backlog (
    run_id         TEXT NOT NULL,
    sim_date       DATE NOT NULL,
    data           JSONB NOT NULL,
    PRIMARY KEY (run_id, sim_date)
);
"""

_RUNTIME_DDL = """
CREATE TABLE IF NOT EXISTS sim_state_runtime (
    run_id   TEXT NOT NULL,
    sim_date DATE NOT NULL,
    key      TEXT NOT NULL,
    value    JSONB NOT NULL,
    PRIMARY KEY (run_id, sim_date, key)
);
"""


class Snapshot:
    """快照表读写，供 Orch.save_state / restore_state 调用。"""

    def __init__(self, db: DB):
        self.db = db

    def ensure_tables(self):
        """确保所有快照表存在（幂等）。"""
        for ddl in [_CHECKPOINT_DDL, _INVENTORY_DDL, _INTRANSIT_DDL,
                     _OPEN_DEPLOY_DDL, _BACKLOG_DDL, _RUNTIME_DDL]:
            self.db.execute(ddl)
        logger.info("快照表已确保存在")

    # ── 状态保存 ─────────────────────────────

    def save(self, run_id: str, sim_date: str, ctx):
        """将 StateContext 当天状态写入快照表。

        Args:
            run_id: 运行标识
            sim_date: 日期字符串 YYYY-MM-DD
            ctx: StateContext 实例（持有所有可变状态）
        """
        self._save_inventory(run_id, sim_date, ctx)
        self._save_intransit(run_id, sim_date, ctx)
        self._save_open_deploy(run_id, sim_date, ctx)
        self._save_backlog(run_id, sim_date, ctx)
        self._save_runtime(run_id, sim_date, ctx)

    def _save_inventory(self, run_id: str, sim_date: str, ctx):
        """库存快照：{material, location} → quantity。"""
        rows = [
            {'run_id': run_id, 'sim_date': sim_date,
             'material': m, 'location': l, 'quantity': int(q)}
            for (m, l), q in getattr(ctx, 'unrestricted_inventory', {}).items()
        ]
        df = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=['run_id', 'sim_date', 'material', 'location', 'quantity'])
        if not df.empty:
            self.db.write_df(SNAPSHOT_TABLES['inventory'], df)
        logger.debug(f"库存快照: {len(rows)} 条")

    def _save_intransit(self, run_id: str, sim_date: str, ctx):
        """在途快照：逐条写入。"""
        import json
        in_transit = getattr(ctx, 'in_transit', {})
        if not in_transit:
            return
        rows = [
            {'run_id': run_id, 'sim_date': sim_date,
             'transit_uid': uid, 'data': json.dumps(rec, default=str)}
            for uid, rec in in_transit.items()
        ]
        df = pd.DataFrame(rows)
        self.db.write_df(SNAPSHOT_TABLES['intransit'], df)
        logger.debug(f"在途快照: {len(rows)} 条")

    def _save_open_deploy(self, run_id: str, sim_date: str, ctx):
        """在途调拨快照：逐条写入。"""
        import json
        open_deploy = getattr(ctx, 'open_deployment', {})
        if not open_deploy:
            return
        rows = [
            {'run_id': run_id, 'sim_date': sim_date,
             'deploy_uid': uid, 'data': json.dumps(rec, default=str)}
            for uid, rec in open_deploy.items()
        ]
        df = pd.DataFrame(rows)
        self.db.write_df(SNAPSHOT_TABLES['open_deploy'], df)
        logger.debug(f"调拨快照: {len(rows)} 条")

    def _save_backlog(self, run_id: str, sim_date: str, ctx):
        """生产计划积压快照。"""
        import json
        backlog = getattr(ctx, 'production_plan_backlog', [])
        if not backlog:
            return
        df = pd.DataFrame([{
            'run_id': run_id, 'sim_date': sim_date,
            'data': json.dumps(backlog, default=str),
        }])
        self.db.write_df(SNAPSHOT_TABLES['backlog'], df)
        logger.debug(f"积压快照: {len(backlog)} 条")

    def _save_runtime(self, run_id: str, sim_date: str, ctx):
        """运行时杂项状态（uid_sequence, PRNG state 等）。"""
        import json
        items = {}
        items['uid_sequence'] = getattr(ctx, 'uid_sequence', 0)
        # PRNG state
        import numpy as np
        rng_state = np.random.get_state()
        items['numpy_random_state'] = {
            'algorithm': rng_state[0],
            'state': rng_state[1].tolist(),
            'pos': rng_state[2],
        }
        # 日志（截断）
        daily_logs = getattr(ctx, 'daily_logs', [])
        items['daily_logs'] = daily_logs[-1000:] if len(daily_logs) > 1000 else daily_logs

        rows = [
            {'run_id': run_id, 'sim_date': sim_date,
             'key': k, 'value': json.dumps(v, default=str)}
            for k, v in items.items()
        ]
        df = pd.DataFrame(rows)
        self.db.write_df(SNAPSHOT_TABLES['runtime'], df)

    # ── 状态恢复 ─────────────────────────────

    def restore(self, run_id: str, sim_date: str, ctx):
        """从快照表读取并还原到 StateContext。

        Args:
            run_id: 运行标识
            sim_date: 要恢复到的日期
            ctx: StateContext 实例
        """
        self._restore_inventory(run_id, sim_date, ctx)
        self._restore_intransit(run_id, sim_date, ctx)
        self._restore_open_deploy(run_id, sim_date, ctx)
        self._restore_backlog(run_id, sim_date, ctx)
        self._restore_runtime(run_id, sim_date, ctx)
        logger.info(f"状态已从快照恢复: run_id={run_id}, sim_date={sim_date}")

    def _restore_inventory(self, run_id: str, sim_date: str, ctx):
        df = self.db.read(SNAPSHOT_TABLES['inventory'],
                          run_id=run_id, sim_date=sim_date)
        if df.empty:
            return
        ctx.unrestricted_inventory = {
            (row['material'], row['location']): int(row['quantity'])
            for _, row in df.iterrows()
        }

    def _restore_intransit(self, run_id: str, sim_date: str, ctx):
        import json
        df = self.db.read(SNAPSHOT_TABLES['intransit'],
                          run_id=run_id, sim_date=sim_date)
        if df.empty:
            return
        ctx.in_transit = {
            row['transit_uid']: json.loads(row['data'])
            for _, row in df.iterrows()
        }

    def _restore_open_deploy(self, run_id: str, sim_date: str, ctx):
        import json
        df = self.db.read(SNAPSHOT_TABLES['open_deploy'],
                          run_id=run_id, sim_date=sim_date)
        if df.empty:
            return
        ctx.open_deployment = {
            row['deploy_uid']: json.loads(row['data'])
            for _, row in df.iterrows()
        }

    def _restore_backlog(self, run_id: str, sim_date: str, ctx):
        import json
        df = self.db.read(SNAPSHOT_TABLES['backlog'],
                          run_id=run_id, sim_date=sim_date)
        if df.empty:
            return
        ctx.production_plan_backlog = json.loads(df.iloc[0]['data'])

    def _restore_runtime(self, run_id: str, sim_date: str, ctx):
        import json
        import numpy as np
        df = self.db.read(SNAPSHOT_TABLES['runtime'],
                          run_id=run_id, sim_date=sim_date)
        if df.empty:
            return
        runtime = {row['key']: json.loads(row['value']) for _, row in df.iterrows()}

        if 'uid_sequence' in runtime:
            ctx.uid_sequence = runtime['uid_sequence']
        if 'numpy_random_state' in runtime:
            state = runtime['numpy_random_state']
            np.random.set_state((
                state['algorithm'],
                np.array(state['state'], dtype=np.uint32),
                state['pos'],
                0,
                0.0,
            ))
        if 'daily_logs' in runtime:
            ctx.daily_logs = runtime['daily_logs']

    # ── Checkpoint 元信息 ─────────────────────

    def save_checkpoint(self, run_id: str, config_name: str,
                        last_batch_end: str, status: str = 'running',
                        error_message: str = None):
        """写入/更新 sim_checkpoint 元信息。"""
        self.db.execute("""
            INSERT INTO sim_checkpoint (run_id, config_name, last_batch_end, status, error_message, updated_at)
            VALUES (%s, %s, %s, %s, %s, NOW())
            ON CONFLICT (run_id) DO UPDATE SET
                last_batch_end = EXCLUDED.last_batch_end,
                status = EXCLUDED.status,
                error_message = EXCLUDED.error_message,
                updated_at = NOW()
        """, (run_id, config_name, last_batch_end, status, error_message))

    def load_checkpoint(self, run_id: str) -> dict | None:
        """读取 sim_checkpoint。"""
        df = self.db.read('sim_checkpoint', filters={'run_id': run_id})
        if df.empty:
            return None
        row = df.iloc[0]
        return {
            'run_id': row['run_id'],
            'config_name': row['config_name'],
            'last_batch_end': str(row['last_batch_end']),
            'status': row['status'],
            'error_message': row.get('error_message'),
        }
