"""
数据库模式断点续跑：checkpoint 读写与 Orchestrator 状态序列化/反序列化。
"""
from __future__ import annotations
import io
import json
import os
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pgsql_db.db_connection import DatabaseConnection

CREATE_CHECKPOINT_TABLE = """
CREATE TABLE IF NOT EXISTS sim_checkpoint (
    run_key         TEXT NOT NULL,
    run_id          TEXT NOT NULL,
    config_name     TEXT NOT NULL,
    start_date      DATE NOT NULL,
    end_date        DATE NOT NULL,
    last_batch_end  DATE NOT NULL,
    orch_state_json JSONB NOT NULL,
    status          TEXT NOT NULL DEFAULT 'running',
    error_message   TEXT,
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (run_id)
);
"""

# 数据定义语句：为已有表添加 `status` / `error_message` / `created_at` 列（幂等）
ALTER_ADD_STATUS = "ALTER TABLE sim_checkpoint ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'running';"
ALTER_ADD_ERROR  = "ALTER TABLE sim_checkpoint ADD COLUMN IF NOT EXISTS error_message TEXT;"
ALTER_ADD_CREATED = "ALTER TABLE sim_checkpoint ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT NOW();"

# 数据定义语句：主键从 `run_key` 迁移到 `run_id`（幂等）
# 先删旧约束再建新约束；如果旧约束不存在则忽略
MIGRATE_PK_TO_RUN_ID = [
    "ALTER TABLE sim_checkpoint DROP CONSTRAINT IF EXISTS sim_checkpoint_pkey;",
    "ALTER TABLE sim_checkpoint ADD PRIMARY KEY (run_id);",
]

def ensure_checkpoint_table(db: "DatabaseConnection") -> None:
    """确保 sim_checkpoint 表存在并包含所有需要的列（幂等）"""
    db.execute_non_query(CREATE_CHECKPOINT_TABLE)
    # 兼容旧表：追加新列
    for ddl in (ALTER_ADD_STATUS, ALTER_ADD_ERROR, ALTER_ADD_CREATED):
        try:
            db.execute_non_query(ddl)
        except Exception:
            pass  # 列已存在或不支持 IF NOT EXISTS
    # 兼容旧表：主键从 run_key 迁移到 run_id
    for ddl in MIGRATE_PK_TO_RUN_ID:
        try:
            db.execute_non_query(ddl)
        except Exception:
            pass  # 约束已存在或已迁移

def save_checkpoint(
    db: "DatabaseConnection",
    run_key: str,
    run_id: str,
    config_name: str,
    start_date: str,
    end_date: str,
    last_batch_end: str,
    orch_state_dict: dict,
    status: str = "running",
    error_message: str = None,
) -> None:
    """UPSERT 一条 checkpoint 记录（含运行状态）"""
    orch_json = json.dumps(orch_state_dict, default=_json_serializer, ensure_ascii=False)
    db.execute_non_query(
        """
        INSERT INTO sim_checkpoint
            (run_key, run_id, config_name, start_date, end_date,
             last_batch_end, orch_state_json, status, error_message, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, NOW())
        ON CONFLICT (run_id) DO UPDATE SET
            last_batch_end  = EXCLUDED.last_batch_end,
            orch_state_json = EXCLUDED.orch_state_json,
            status          = EXCLUDED.status,
            error_message   = EXCLUDED.error_message,
            updated_at      = NOW()
        """,
        (run_key, run_id, config_name, start_date, end_date,
         last_batch_end, orch_json, status, error_message),
    )

def load_checkpoint(
    db: "DatabaseConnection",
    run_key: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Optional[dict]:
    """按 run_key 查询最近一条可用于断点续跑的 checkpoint（status=running/failed/interrupted），返回 dict 或 None。

    [FIX-#5] 当调用方传入 start_date / end_date 时，仅匹配日期区间完全相同的 checkpoint，
    防止因日期区间变更后错误接续旧 run（会导致断点续跑从错误起点开始）。
    """
    rows = db.execute_query(
        "SELECT run_key, run_id, config_name, start_date::text, end_date::text, "
        "last_batch_end::text, orch_state_json::text, "
        "COALESCE(status, 'running'), error_message "
        "FROM sim_checkpoint WHERE run_key = %s AND status IN ('running', 'failed', 'interrupted') "  # [FIX-#8] interrupted 也可用于断点续跑
        "ORDER BY updated_at DESC LIMIT 1",
        (run_key,),
    )
    if not rows:
        return None
    row = rows[0]
    cp = {
        "run_key":         row[0],
        "run_id":          row[1],
        "config_name":     row[2],
        "start_date":      row[3],
        "end_date":        row[4],
        "last_batch_end":  row[5],
        "orch_state_json": json.loads(row[6]),
        "status":          row[7],
        "error_message":   row[8],
    }
    # [FIX-#5] 校验日期区间一致性
    if start_date is not None and cp["start_date"] != str(start_date):
        print(
            f"  [WARN][FIX-#5] checkpoint start_date 不匹配 "
            f"（DB: {cp['start_date']}，请求: {start_date}）。忽略旧 checkpoint，全新开始。"
        )
        return None
    if end_date is not None and cp["end_date"] != str(end_date):
        print(
            f"  [WARN][FIX-#5] checkpoint end_date 不匹配 "
            f"（DB: {cp['end_date']}，请求: {end_date}）。忽略旧 checkpoint，全新开始。"
        )
        return None
    return cp

def update_checkpoint_status(db: "DatabaseConnection", run_id: str, status: str, error_message: str = None) -> None:
    """按 run_id 更新 checkpoint 的运行状态（running/completed/failed/interrupted）"""
    db.execute_non_query(
        """
        UPDATE sim_checkpoint
        SET status = %s, error_message = %s, updated_at = NOW()
        WHERE run_id = %s
        """,
        (status, error_message, run_id),
    )

def delete_checkpoint(db: "DatabaseConnection", run_id: str) -> None:
    """按 run_id 删除 checkpoint 记录"""
    db.execute_non_query(
        "DELETE FROM sim_checkpoint WHERE run_id = %s", (run_id,)
    )

def serialize_orchestrator_state(orch, max_log_entries: int = 1000) -> dict:
    """将 Orchestrator 内存状态序列化为纯 Python dict（可 JSON 存储）

    [FIX-风险D] max_log_entries 限制历史日志列表的最大条目数（保留最新的 N 条），
    防止长仿真中 orch_state_json 膨胀导致 checkpoint 写入超时。
    核心状态（inventory/in_transit/open_deployment/space_quota）不受限制。
    """
    # 历史日志类状态已按天落库，checkpoint 仅保留断点续跑所需的轻量状态。
    def _inv_key(k):
        # [FIX-风险G] 使用 JSON array 替代 ||| 分隔符，避免 material/location 包含 ||| 导致反序列化错误
        if isinstance(k, tuple):
            return json.dumps([k[0], k[1]], ensure_ascii=False)
        return str(k)

    def _val(v):
        if isinstance(v, pd.Timestamp):
            return v.isoformat()
        if isinstance(v, (date, datetime)):
            return v.isoformat()
        if isinstance(v, float) and pd.isna(v):
            return None
        return v

    def _truncate(lst, name):
        """[FIX-风险D] 截断过长的历史日志，仅保留最近 max_log_entries 条"""
        if len(lst) > max_log_entries:
            print(f"  [WARN] checkpoint: {name} 有 {len(lst)} 条记录，截断为最近 {max_log_entries} 条")
            return lst[-max_log_entries:]
        return lst

    inv = {_inv_key(k): v for k, v in getattr(orch, 'unrestricted_inventory', {}).items()}

    in_transit = {}
    for uid, rec in getattr(orch, 'in_transit', {}).items():
        in_transit[str(uid)] = {k: _val(v) for k, v in rec.items()} if isinstance(rec, dict) else str(rec)

    open_dep = {}
    for uid, rec in getattr(orch, 'open_deployment', {}).items():
        open_dep[str(uid)] = {k: _val(v) for k, v in rec.items()} if isinstance(rec, dict) else str(rec)

    space = {str(k): v for k, v in getattr(orch, 'space_quota', {}).items()}

    # [FIX-#1] 序列化原始 space_capacity DataFrame（M5 收货空间约束的数据源）
    _sc = getattr(orch, 'space_capacity', None)
    if _sc is not None and not _sc.empty:
        space_capacity_records = []
        for _, row in _sc.iterrows():
            space_capacity_records.append({k: _val(v) for k, v in row.items()})
    else:
        space_capacity_records = []

    # Historical logs are already persisted per day. Keep checkpoint JSON lean
    # and rebuild any date indexes from the serialized lightweight state on resume.
    delivery_gr = []
    production_gr = []
    shipment_log = []
    dlv_ship_log = []
    inv_change_log = []

    daily_logs = []

    cur_date = getattr(orch, 'current_date', None)
    cur_date_str = cur_date.isoformat() if isinstance(cur_date, (date, datetime, pd.Timestamp)) else str(cur_date) if cur_date else None

    # [FIX-#2] 序列化 production_plan_backlog（M3 需求净计算的供给数据源）
    _ppb = getattr(orch, 'production_plan_backlog', [])
    production_plan_backlog = []
    if isinstance(_ppb, list):
        for rec in _ppb:
            if isinstance(rec, dict):
                production_plan_backlog.append({k: _val(v) for k, v in rec.items()})
            else:
                production_plan_backlog.append(_val(rec))

    # [FIX-#9] 保存 numpy 全局 PRNG 状态，确保断点续跑时随机序列连续
    # np.random.get_state() 返回 ('MT19937', ndarray(624,), pos, has_gauss, cached_gauss)
    # ndarray 不可直接 JSON 序列化，转为 list
    try:
        _rng = np.random.get_state()
        numpy_random_state = {
            'id': _rng[0],                       # 'MT19937'
            'state': _rng[1].tolist(),            # 624 个 uint32
            'pos': int(_rng[2]),                  # 当前位置
            'has_gauss': int(_rng[3]),            # 是否有缓存高斯值
            'cached_gauss': float(_rng[4]),       # 缓存的高斯值
        }
    except Exception as _e:
        print(f"  [WARN][FIX-#9] 保存 numpy random state 失败: {_e}")
        numpy_random_state = None

    return {
        "unrestricted_inventory": inv,
        "in_transit":             in_transit,
        "open_deployment":        open_dep,
        "space_quota":            space,
        "space_capacity":         space_capacity_records,       # [FIX-#1]
        "production_plan_backlog": production_plan_backlog,     # [FIX-#2]
        "uid_sequence":           getattr(orch, 'uid_sequence', 0),  # [FIX-#6]
        "numpy_random_state":     numpy_random_state,           # [FIX-#9]
        "delivery_gr":            delivery_gr,
        "production_gr":          production_gr,
        "shipment_log":           shipment_log,
        "delivery_shipment_log":  dlv_ship_log,
        "inventory_change_log":   inv_change_log,
        "daily_logs":             daily_logs,
        "current_date":           cur_date_str,
    }

def deserialize_orchestrator_state(orch, state_dict: dict) -> None:
    """将 sim_checkpoint.orch_state_json 反序列化回 Orchestrator 属性

    [FIX-风险7] 对 `_DATE_KEYS` 中声明的日期字段执行 pd.to_datetime 反解析，
    避免后续模块因类型不匹配（str vs datetime）产生逻辑错误。
    """
    def _inv_key(s):
        # [FIX-风险G] 兼容旧格式 "MAT|||LOC" 和新格式 '["MAT", "LOC"]'
        if s.startswith('['):
            try:
                parts = json.loads(s)
                return (parts[0], parts[1]) if len(parts) == 2 else (s, "")
            except (json.JSONDecodeError, IndexError):
                return (s, "")
        # 旧格式兼容
        parts = s.split("|||", 1)
        return (parts[0], parts[1]) if len(parts) == 2 else (s, "")

    # [FIX-风险7] 恢复嵌套字典中的日期字段
    _DATE_KEYS = frozenset({
        'eta', 'start_date', 'end_date', 'date', 'planned_deployment_date',
        'arrival_date', 'ship_date', 'delivery_date', 'available_date',
        'simulation_date', 'created_date',
    })

    def _restore_dates(rec: dict) -> dict:
        """将 ISO 格式日期字符串恢复为 pd.Timestamp"""
        for k in _DATE_KEYS & rec.keys():
            v = rec[k]
            if isinstance(v, str):
                try:
                    rec[k] = pd.to_datetime(v)
                except (ValueError, TypeError):
                    pass  # 非日期字符串，保留原值
        return rec

    orch.unrestricted_inventory = {
        _inv_key(k): float(v) if v is not None else 0.0
        for k, v in state_dict.get("unrestricted_inventory", {}).items()
    }
    orch.in_transit = {
        uid: _restore_dates(rec) if isinstance(rec, dict) else rec
        for uid, rec in state_dict.get("in_transit", {}).items()
    }
    orch.open_deployment = {
        uid: _restore_dates(rec) if isinstance(rec, dict) else rec
        for uid, rec in state_dict.get("open_deployment", {}).items()
    }
    orch.space_quota = {
        k: float(v) if v is not None else 0.0
        for k, v in state_dict.get("space_quota", {}).items()
    }

    # [FIX-#1] 恢复 space_capacity DataFrame（M5 收货空间约束的数据源）
    _sc_records = state_dict.get("space_capacity", [])
    if _sc_records:
        _sc_df = pd.DataFrame(_sc_records)
        # 恢复日期列类型
        for _col in ('eff_from', 'eff_to'):
            if _col in _sc_df.columns:
                _sc_df[_col] = pd.to_datetime(_sc_df[_col], errors='coerce')
        orch.space_capacity = _sc_df
    else:
        orch.space_capacity = pd.DataFrame()

    # [FIX-#2] 恢复 production_plan_backlog（M3 净需求计算的供给数据源）
    _ppb_records = state_dict.get("production_plan_backlog", [])
    if _ppb_records:
        _ppb_df = pd.DataFrame(_ppb_records)
        for _col in ('available_date', 'simulation_date', 'production_plan_date'):
            if _col in _ppb_df.columns:
                _ppb_df[_col] = pd.to_datetime(_ppb_df[_col], errors='coerce')
        orch.production_plan_backlog = _ppb_df.to_dict('records')
    else:
        orch.production_plan_backlog = []

    # [FIX-#6] 恢复 uid_sequence，防止断点续跑后 UID 重置覆盖已有 open_deployment 键
    orch.uid_sequence = int(state_dict.get("uid_sequence", 0))
    # [FIX-风险C] 对日志列表同样应用日期恢复，避免断点续跑后日期字段仍为字符串
    orch.delivery_gr           = [_restore_dates(r) if isinstance(r, dict) else r
                                  for r in state_dict.get("delivery_gr", [])]
    orch.production_gr         = [_restore_dates(r) if isinstance(r, dict) else r
                                  for r in state_dict.get("production_gr", [])]
    orch.shipment_log          = [_restore_dates(r) if isinstance(r, dict) else r
                                  for r in state_dict.get("shipment_log", [])]
    orch.delivery_shipment_log = [_restore_dates(r) if isinstance(r, dict) else r
                                  for r in state_dict.get("delivery_shipment_log", [])]
    orch.inventory_change_log  = [_restore_dates(r) if isinstance(r, dict) else r
                                  for r in state_dict.get("inventory_change_log", [])]
    orch.daily_logs            = state_dict.get("daily_logs", [])

    cur = state_dict.get("current_date")
    if cur:
        orch.current_date = pd.to_datetime(cur)

    # [FIX-#9] 恢复 numpy 全局 PRNG 状态
    _rng_data = state_dict.get("numpy_random_state")
    if _rng_data and isinstance(_rng_data, dict):
        try:
            _state_tuple = (
                _rng_data['id'],                                      # 'MT19937'
                np.array(_rng_data['state'], dtype=np.uint32),        # 624 个 uint32
                int(_rng_data['pos']),                                # 当前位置
                int(_rng_data['has_gauss']),                          # 是否有缓存高斯值
                float(_rng_data['cached_gauss']),                     # 缓存的高斯值
            )
            np.random.set_state(_state_tuple)
            print(f"  [OK][FIX-#9] Restored numpy random state (pos={_rng_data['pos']})")
        except Exception as _e:
            print(f"  [WARN][FIX-#9] 恢复 numpy random state 失败: {_e}，PRNG 状态可能不连续")
    else:
        print(f"  [WARN][FIX-#9] checkpoint 中无 numpy_random_state，PRNG 状态可能不连续（旧版 checkpoint）")

    print(f"  [OK] Restored unrestricted_inventory from DB: {len(orch.unrestricted_inventory)} records")
    print(f"  [OK] Restored in_transit from DB: {len(orch.in_transit)} records")
    print(f"  [OK] Restored open_deployment from DB: {len(orch.open_deployment)} records")
    print(f"  [OK] Restored space_capacity from DB: {len(orch.space_capacity)} rows")          # [FIX-#1]
    print(f"  [OK] Restored production_plan_backlog from DB: {len(orch.production_plan_backlog)} records")  # [FIX-#2]
    print(f"  [OK] Restored uid_sequence from DB: {orch.uid_sequence}")                      # [FIX-#6]

def serialize_m1_previous_orders(df) -> Optional[list]:
    """将 m1_previous_orders DataFrame 序列化为 JSON 兼容的 list[dict]"""
    if df is None:
        return None
    if not hasattr(df, 'to_dict'):
        return None
    if df.empty:
        return []
    # 将 DataFrame 转为 records，处理日期和 NaN
    records = []
    for _, row in df.iterrows():
        rec = {}
        for col in df.columns:
            v = row[col]
            if isinstance(v, pd.Timestamp):
                rec[col] = v.isoformat()
            elif isinstance(v, (date, datetime)):
                rec[col] = v.isoformat()
            elif isinstance(v, float) and pd.isna(v):
                rec[col] = None
            elif hasattr(v, 'item'):  # numpy 标量
                rec[col] = v.item()
            else:
                rec[col] = v
        records.append(rec)
    return records


def deserialize_m1_previous_orders(data) -> Optional["pd.DataFrame"]:
    """将 checkpoint 中的 m1_previous_orders 反序列化为 DataFrame"""
    if data is None:
        return None
    if not isinstance(data, list):
        return None
    if len(data) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    # 恢复常见日期列
    _DATE_COLS = {'simulation_date', 'order_date', 'delivery_date', 'ship_date',
                  'available_date', 'date', 'created_date'}
    for col in _DATE_COLS & set(df.columns):
        df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

def _json_serializer(obj):
    if isinstance(obj, (date, datetime, pd.Timestamp)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

def next_day(date_str: str) -> str:
    """返回 date_str 的下一天，格式 YYYY-MM-DD"""
    d = datetime.strptime(date_str, "%Y-%m-%d").date()
    return (d + timedelta(days=1)).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# [FIX-#3] M4 跨天文件持久化：将 line_states / allocated_capacity / Module4输出
# 存入 DB 表 sim_m4_state，在断点续跑时恢复到临时目录，替代对本地文件的依赖。
# ---------------------------------------------------------------------------

CREATE_M4_STATE_TABLE = """
CREATE TABLE IF NOT EXISTS sim_m4_state (
    run_id      TEXT NOT NULL,
    sim_date    DATE NOT NULL,
    file_type   TEXT NOT NULL,
    file_content BYTEA NOT NULL,
    updated_at  TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (run_id, sim_date, file_type)
);
"""

def ensure_m4_state_table(db: "DatabaseConnection") -> None:
    """确保 sim_m4_state 表存在（幂等）"""
    db.execute_non_query(CREATE_M4_STATE_TABLE)


def save_m4_state_files(
    db: "DatabaseConnection",
    run_id: str,
    sim_date_str: str,
    m4_output_dir: str,
) -> None:
    """将当天 M4 产出的状态文件上传到 DB（幂等 UPSERT）。

    保存的文件类型：
    - line_states      -> line_states_YYYYMMDD.json
    - alloc_capacity   -> allocated_capacity_YYYYMMDD.json
    - m4_output_excel  -> Module4Output_YYYYMMDD.xlsx

    参数：
        db:             DatabaseConnection 实例
        run_id:         当前仿真 run_id
        sim_date_str:   当天日期 YYYY-MM-DD
        m4_output_dir:  Module4 输出目录（临时目录下的 module4 子目录）
    """
    date_tag = sim_date_str.replace('-', '')  # YYYYMMDD
    files_to_save = {
        'line_states':     os.path.join(m4_output_dir, f"line_states_{date_tag}.json"),
        'alloc_capacity':  os.path.join(m4_output_dir, f"allocated_capacity_{date_tag}.json"),
        'm4_output_excel': os.path.join(m4_output_dir, f"Module4Output_{date_tag}.xlsx"),
    }
    for file_type, file_path in files_to_save.items():
        if not os.path.exists(file_path):
            continue
        try:
            with open(file_path, 'rb') as fh:
                content = fh.read()
            db.execute_non_query(
                """
                INSERT INTO sim_m4_state (run_id, sim_date, file_type, file_content, updated_at)
                VALUES (%s, %s::date, %s, %s, NOW())
                ON CONFLICT (run_id, sim_date, file_type) DO UPDATE SET
                    file_content = EXCLUDED.file_content,
                    updated_at   = NOW()
                """,
                (run_id, sim_date_str, file_type, content),
            )
        except Exception as e:
            print(f"  [WARN][FIX-#3] Failed to upload M4 state file {file_type} ({sim_date_str}): {e}")


def restore_m4_state_files(
    db: "DatabaseConnection",
    run_id: str,
    m4_output_dir: str,
) -> int:
    """从 DB 恢复所有历史 M4 状态文件到临时目录。

    断点续跑时在仿真循环开始前调用一次，把所有已存储的 M4 文件写回磁盘，
    使 load_line_state / load_all_previous_capacity / load_current_date_production_gr 能正常读取。

    返回：
        恢复的文件数量
    """
    os.makedirs(m4_output_dir, exist_ok=True)
    rows = db.execute_query(
        "SELECT sim_date::text, file_type, file_content "
        "FROM sim_m4_state WHERE run_id = %s ORDER BY sim_date",
        (run_id,),
    )
    if not rows:
        return 0

    restored = 0
    for sim_date_str, file_type, file_content in rows:
        date_tag = sim_date_str.replace('-', '')
        name_map = {
            'line_states':     f"line_states_{date_tag}.json",
            'alloc_capacity':  f"allocated_capacity_{date_tag}.json",
            'm4_output_excel': f"Module4Output_{date_tag}.xlsx",
        }
        file_name = name_map.get(file_type)
        if not file_name:
            continue
        dest = os.path.join(m4_output_dir, file_name)
        try:
            # psycopg2 返回 memoryview，需转为字节串
            raw = bytes(file_content) if not isinstance(file_content, bytes) else file_content
            with open(dest, 'wb') as fh:
                fh.write(raw)
            restored += 1
        except Exception as e:
            print(f"  [WARN][FIX-#3] Failed to restore M4 state file {file_name}: {e}")

    print(f"  [OK][FIX-#3] 已从 DB 恢复 {restored} 个 M4 状态文件")
    return restored


# ---------------------------------------------------------------------------
# [FIX-#7] 并发保护：PostgreSQL advisory lock。
# 当前 lock key 基于 Python 内置 hash() 折叠，属于最佳努力保护；
# 在未改为稳定哈希前，不应将其表述为严格的跨进程互斥保障。
# ---------------------------------------------------------------------------

def _advisory_lock_key(run_key: str) -> int:
    """将 run_key 字符串映射为 64-bit int，作为 advisory lock key。

    当前实现使用 Python 内置 hash() 的结果再折叠到 int8 范围内，
    主要用于同一进程语义下的锁键生成。
    """
    h = hash(run_key) & 0xFFFFFFFFFFFFFFFF  # 无符号 64-bit
    # 转换为有符号 int8（PostgreSQL bigint）
    if h >= (1 << 63):
        h -= (1 << 64)
    return h


def try_acquire_run_lock(db: "DatabaseConnection", run_key: str) -> bool:
    """尝试获取针对 run_key 的 session-level advisory lock。

    [FIX-#7] 目标是让同一 run_key 的运行尽量串行化；当前实现属于最佳努力保护。
    - 返回 True：获取成功，调用方可继续仿真
    - 返回 False：已有其他进程持有锁，调用方应退出

    锁在 db.close() 或进程退出时自动释放。
    """
    key = _advisory_lock_key(run_key)
    rows = db.execute_query("SELECT pg_try_advisory_lock(%s)", (key,))
    if rows and rows[0][0]:
        return True
    return False


def release_run_lock(db: "DatabaseConnection", run_key: str) -> None:
    """显式释放 advisory lock（正常结束时调用，kill 时 PG 自动释放）。"""
    key = _advisory_lock_key(run_key)
    try:
        db.execute_query("SELECT pg_advisory_unlock(%s)", (key,))
    except Exception:
        pass  # 连接已关闭时忽略
