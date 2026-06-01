"""PostgreSQL 运行使用的 project 到 schema 解析器。"""
from __future__ import annotations  # 允许类型注解延迟解析，避免运行期导入负担。

import logging  # 用于输出 schema 校验失败 warning。
import re  # 用于校验 schema 字符集合。
from typing import Optional  # 用于声明可选字符串参数。

logger = logging.getLogger("SupplyChainSimulation")  # 复用仿真主日志，保证告警进入运行日志。

_SCHEMA_PATTERN = re.compile(r"^[a-z0-9_-]+$")  # schema 只允许小写字母、数字、下划线、连字符。
_SCHEMA_MAX_LEN = 63  # PostgreSQL identifier 名称长度上限为 63 字节。


def resolve_project_schema(
    project: Optional[str],
    *,
    default_schema: Optional[str] = None,
) -> str:
    """将 project 名解析为通过校验的 PostgreSQL schema 名。

    参数：
        project: 从 ``ConfigDir.project`` 解析出的项目名；为空时回退到 ``default_schema``。
        default_schema: ``project`` 缺失时使用的备用 schema。

    返回：
        仅包含 ``a-z``、``0-9``、``_``、``-`` 的小写 schema 名。

    异常：
        ValueError: 当输入无法生成合法 schema 名时抛出。
    """
    if project is None or not str(project).strip():  # project 缺失或全空白时进入回退逻辑。
        if default_schema is None or not str(default_schema).strip():  # 回退 schema 也缺失时无法继续。
            msg = (  # 构造明确错误消息，告诉调用方如何修正配置。
                "无法解析数据库 schema：未提供 project，且 default_schema 也为空。"
                " 请在 config/defaults.yaml 设置 database.default_schema，"
                " 或通过 --config-dir 提供包含 project/scenario 的路径。"
            )
            logger.warning(msg)  # 按要求先告警。
            raise ValueError(msg)  # 再抛异常阻断后续 DB 初始化和写库。
        return _normalize_and_validate(default_schema, source="default_schema")  # 校验并返回回退 schema。

    return _normalize_and_validate(project, source="project")  # project 存在时按 project 派生 schema。


def _normalize_and_validate(raw: str, *, source: str) -> str:
    """规范化并校验一个 schema 候选值。

    参数：
        raw: 原始 schema 候选值。
        source: 用于告警消息的人类可读来源名称。

    返回：
        通过校验的小写 schema 名。

    异常：
        ValueError: 当候选值为空、过长或格式非法时抛出。
    """
    candidate = str(raw).strip().lower()  # 去首尾空白并统一小写，保留连字符。
    if not candidate:  # 规范化后为空说明输入无有效字符。
        msg = f"数据库 schema 校验失败：{source} 经规范化后为空（原值={raw!r}）。"  # 构造空值错误。
        logger.warning(msg)  # 先记录 warning，便于日志定位。
        raise ValueError(msg)  # 再终止流程，避免写入错误 schema。

    if len(candidate) > _SCHEMA_MAX_LEN:  # schema 名超过 PostgreSQL identifier 上限。
        msg = (  # 构造长度错误消息。
            f"数据库 schema 校验失败：{source}={raw!r} 规范化后长度"
            f" {len(candidate)} 超过 PostgreSQL identifier 上限 {_SCHEMA_MAX_LEN}。"
        )
        logger.warning(msg)  # 先记录 warning。
        raise ValueError(msg)  # 再抛出异常阻断。

    if not _SCHEMA_PATTERN.match(candidate):  # schema 含非法字符时禁止自动替换。
        msg = (  # 构造非法字符错误消息。
            f"数据库 schema 校验失败：{source}={raw!r} 含非法字符。"
            f" 允许字符集合：a-z、0-9、_、-（小写化后），规范化结果={candidate!r}。"
        )
        logger.warning(msg)  # 先记录 warning。
        raise ValueError(msg)  # 再抛出异常，防止数据写错命名空间。

    return candidate  # 所有校验通过后返回合法 schema 名。


__all__ = ["resolve_project_schema"]  # 明确模块公开 API。
