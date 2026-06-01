"""配置导入清单与稳定表内容哈希。"""
from __future__ import annotations  # 延迟解析类型注解，避免运行期循环导入。

import hashlib  # 用于计算配置表内容的 SHA-256 摘要。
import logging  # 用于记录 manifest 读取/删除失败信息。
from datetime import datetime  # 用于标注来源文件修改时间类型。
from typing import TYPE_CHECKING, Optional  # TYPE_CHECKING 避免运行期导入 DB 类；Optional 用于可选参数。

import pandas as pd  # 配置表以 DataFrame 形式传入并规范化。

if TYPE_CHECKING:  # 仅在类型检查阶段导入，避免运行时循环依赖。
    from .db_connection import DatabaseConnection  # schema-aware 的数据库连接类型。

logger = logging.getLogger("SupplyChainSimulation")  # 复用仿真主日志。

MANIFEST_TABLE = "cfg_import_manifest"  # 配置导入清单表名。

# 这些写库元数据不参与业务内容哈希。
_METADATA_COLS = frozenset({"config_name", "config_type", "db_write_time"})


def ensure_manifest_table(db: "DatabaseConnection") -> None:
    """创建配置导入清单表及查询索引。

    参数：
        db: 带 schema 上下文的数据库连接。
    """
    qualified = db.qualified_name(MANIFEST_TABLE)  # 生成带 schema 的 manifest 表名。
    # 定义 manifest 表结构；主键限定同一 config_name/table_name 只保留一条状态。
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {qualified} (
        config_name   TEXT NOT NULL,
        table_name    TEXT NOT NULL,
        row_count     BIGINT NOT NULL DEFAULT 0,
        content_hash  TEXT NOT NULL,
        source_file   TEXT,
        source_mtime  TIMESTAMP,
        imported_at   TIMESTAMP NOT NULL DEFAULT NOW(),
        PRIMARY KEY (config_name, table_name)
    );
    """
    db.execute_non_query(ddl)  # 幂等创建 manifest 表。

    # config_name 是 manifest 查询的主要过滤条件。
    db.execute_non_query(  # 幂等创建 config_name 查询索引。
        f'CREATE INDEX IF NOT EXISTS "idx_{MANIFEST_TABLE}_config_name" '
        f"ON {qualified} (config_name);"
    )


def load_manifest(db: "DatabaseConnection", config_name: str) -> dict[str, dict]:
    """读取指定配置名对应的 manifest 记录。

    参数：
        db: 带 schema 上下文的数据库连接。
        config_name: 存储在 ``cfg_*`` 表中的配置标识。

    返回：
        以 ``table_name`` 为键的 manifest 元数据字典。
    """
    qualified = db.qualified_name(MANIFEST_TABLE)  # 生成当前 schema 下的 manifest 表名。
    try:  # manifest 表可能尚未创建，因此读取需要容错。
        rows = db.execute_query(  # 按 config_name 在 DB 侧过滤，避免整表读取。
            f"SELECT config_name, table_name, row_count, content_hash, "
            f"source_file, source_mtime::text, imported_at::text "
            f"FROM {qualified} WHERE config_name = %s",
            (config_name,),
        )
    except Exception as e:  # 首次运行或权限异常时进入兜底。
        logger.info(f"[manifest] 读取 {config_name} 失败（视为空 manifest）：{e}")  # 记录原因。
        return {}  # 返回空 manifest，让调用方按首次导入处理。

    result: dict[str, dict] = {}  # 返回结构以 table_name 为键，便于逐表比对 hash。
    for row in rows or []:  # 遍历查询结果；rows 为 None 时按空列表处理。
        result[row[1]] = {  # row[1] 是 table_name。
            "config_name": row[0],  # 配置标识。
            "table_name": row[1],  # 配置表名。
            "row_count": int(row[2]) if row[2] is not None else 0,  # 行数，空值兜底为 0。
            "content_hash": row[3],  # 业务内容稳定哈希。
            "source_file": row[4],  # 来源文件路径。
            "source_mtime": row[5],  # 来源文件修改时间。
            "imported_at": row[6],  # manifest 写入时间。
        }

    return result  # 返回整份 manifest 映射。


def upsert_manifest_entry(
    db: "DatabaseConnection",
    *,
    config_name: str,
    table_name: str,
    row_count: int,
    content_hash: str,
    source_file: Optional[str] = None,
    source_mtime: Optional[datetime] = None,
) -> None:
    """插入或更新一条 manifest 记录。

    参数：
        db: 带 schema 上下文的数据库连接。
        config_name: 配置标识。
        table_name: 目标 ``cfg_*`` 表名。
        row_count: 本次写入目标表的行数。
        content_hash: 目标表业务内容的稳定哈希。
        source_file: 可选的来源文件路径。
        source_mtime: 可选的来源文件修改时间。
    """
    qualified = db.qualified_name(MANIFEST_TABLE)  # 生成当前 schema 下的 manifest 表名。
    db.execute_non_query(  # 使用 UPSERT 保持同一 config/table 只有一条最新状态。
        f"""
        INSERT INTO {qualified} (
            config_name, table_name, row_count, content_hash,
            source_file, source_mtime, imported_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, NOW())
        ON CONFLICT (config_name, table_name) DO UPDATE SET
            row_count    = EXCLUDED.row_count,
            content_hash = EXCLUDED.content_hash,
            source_file  = EXCLUDED.source_file,
            source_mtime = EXCLUDED.source_mtime,
            imported_at  = NOW()
        """,
        (  # 参数化传值，避免 SQL 注入和类型拼接问题。
            config_name,
            table_name,
            int(row_count),
            content_hash,
            source_file,
            source_mtime,
        ),
    )


def delete_manifest_entry(
    db: "DatabaseConnection",
    *,
    config_name: str,
    table_name: str,
) -> int:
    """删除一条 manifest 记录。

    参数：
        db: 带 schema 上下文的数据库连接。
        config_name: 配置标识。
        table_name: 目标 ``cfg_*`` 表名。

    返回：
        删除的行数。
    """
    qualified = db.qualified_name(MANIFEST_TABLE)  # 生成当前 schema 下的 manifest 表名。
    try:  # manifest 表不存在或删除失败时兜底返回 0。
        with db.get_cursor() as cur:  # 获取带事务提交的游标。
            cur.execute(  # 删除指定 config/table 的 manifest 行。
                f"DELETE FROM {qualified} WHERE config_name = %s AND table_name = %s",
                (config_name, table_name),
            )
            return cur.rowcount  # 返回受影响行数。
    except Exception as e:  # 删除失败不影响主流程继续清理其他表。
        logger.info(f"[manifest] 删除 {config_name}/{table_name} 失败：{e}")  # 记录失败原因。
        return 0  # 兜底返回未删除。


def compute_table_hash(df: pd.DataFrame) -> str:
    """计算配置表的稳定 SHA-256 哈希。

    参数：
        df: 配置表内容。

    返回：
        十六进制 SHA-256 摘要。
    """
    if df is None:  # 调用方传 None 时按空表处理。
        df = pd.DataFrame()  # 创建空 DataFrame 作为统一输入。

    # 1. 移除非业务元数据列。
    drop_cols = [c for c in df.columns if str(c) in _METADATA_COLS]  # 找出需要排除的元数据列。
    norm = df.drop(columns=drop_cols, errors="ignore").copy()  # 删除元数据列并复制，避免修改原始 DataFrame。

    # 2. 规范化列名和列顺序。
    norm.columns = [str(c) for c in norm.columns]  # 列名统一转字符串，避免类型差异影响排序。
    norm = norm.reindex(sorted(norm.columns), axis=1)  # 按列名排序，避免输入列顺序影响哈希。

    # 3. 规范化缺失值和标量类型。
    if not norm.empty:  # 非空表才需要替换缺失值。
        norm = norm.where(pd.notna(norm), "")  # NaN/None 统一转为空字符串。
    norm = norm.astype(str)  # 所有值统一转字符串，保证序列化稳定。

    # 4. 对行排序，避免输入行顺序影响哈希。
    if len(norm.columns) > 0 and not norm.empty:  # 有列且有行时可按全部列排序。
        norm = norm.sort_values(by=list(norm.columns), kind="mergesort").reset_index(drop=True)  # 稳定排序并重置索引。
    else:  # 空表或无列表不排序。
        norm = norm.reset_index(drop=True)  # 仅重置索引，保持后续序列化一致。

    # 5. 纳入列签名，保证空表结构变化也会影响哈希。
    col_sig = "|".join(norm.columns)  # 将列名拼成结构签名。
    if norm.empty or len(norm.columns) == 0:  # 空表或无列表走空行 payload。
        payload = f"COLS\x1f{col_sig}\nROWS\x1f"  # 空表仍保留列签名。
    else:  # 非空表序列化每一行。
        rows = [  # 行内字段使用不可见分隔符，降低与业务文本冲突概率。
            "\x1f".join(row) for row in norm.itertuples(index=False, name=None)
        ]
        payload = f"COLS\x1f{col_sig}\nROWS\n" + "\n".join(rows)  # 拼接列签名和行内容。

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()  # 返回 UTF-8 payload 的 SHA-256 摘要。


__all__ = [  # 明确模块公开 API。
    "MANIFEST_TABLE",
    "ensure_manifest_table",
    "load_manifest",
    "upsert_manifest_entry",
    "delete_manifest_entry",
    "compute_table_hash",
]
