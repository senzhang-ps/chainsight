"""`src.core.run` 子包通用辅助工具。"""
from __future__ import annotations

import gc
import logging
import os
import shutil
import time
from pathlib import Path

logger = logging.getLogger("SupplyChainSimulation")

EXCEL_SUFFIXES = (".xlsx", ".xlsm", ".xls")


def resolve_excel_path(config_arg: str) -> Path | None:
    """[过渡期保留] 将用户传入的 ``--config`` 参数解析为 Excel 配置文件的实际路径。

    .. deprecated::
        新加载路径请使用 ``--config-dir`` + ``ConfigDir.from_path``。本函数会做项目
        内多目录回退与 ``rglob`` 递归查找，违反 ``.claude/IO配置绝对路径读取_实施
        计划.md`` §1.2 路径独立性原则，仅在过渡期内保留供旧 ``--config`` 参数使用。

    支持的输入形式：
      - 仅名字: ``SDC_V1`` / ``SDC_V1.xlsx``
      - 相对路径: ``config/sdc/SDC_V1`` / ``config/sdc/SDC_V1.xlsx``
      - 绝对路径: ``D:/abs/path/SDC_V1.xlsx``

    解析顺序：
      1. 按用户给定的路径直接尝试（带后缀直接匹配；无后缀则补 .xlsx/.xlsm/.xls）
      2. 把输入当相对路径拼到 ``项目根/config``、``项目根/test_files``、``项目根``
      3. 在上述目录下 ``rglob(<basename>.xlsx/.xlsm/.xls)`` 递归查找

    返回：找到时返回绝对 ``Path``；找不到返回 ``None``。
    """
    if not config_arg:
        return None

    raw = Path(config_arg).expanduser()

    def _try_with_suffixes(p: Path) -> Path | None:
        if p.suffix.lower() in EXCEL_SUFFIXES and p.exists():
            return p.resolve()
        if not p.suffix:
            for ext in EXCEL_SUFFIXES:
                candidate = p.with_suffix(ext)
                if candidate.exists():
                    return candidate.resolve()
        return None

    # 1) 按用户给定路径（绝对或相对 cwd）直接尝试
    found = _try_with_suffixes(raw)
    if found is not None:
        return found

    # 用户传了带分隔符的明确路径 → 不做名字回退，避免误命中其他目录的同名文件
    is_explicit_path = raw.is_absolute() or len(raw.parts) > 1

    # 2) 项目内部 search_dirs
    project_root = Path(__file__).resolve().parents[3]
    search_dirs = [
        project_root / "config",
        project_root / "test_files",
        project_root,
    ]
    if is_explicit_path:
        return None

    # 仅当输入是裸名字时，才在 search_dirs 中拼接 + 递归回退
    for d in search_dirs:
        candidate = d / raw
        found = _try_with_suffixes(candidate)
        if found is not None:
            return found

    basename = raw.stem
    if basename:
        for d in search_dirs:
            if not d.is_dir():
                continue
            for ext in EXCEL_SUFFIXES:
                matches = sorted(d.rglob(f"{basename}{ext}"))
                if matches:
                    return matches[0].resolve()

    return None


def _cleanup_data_files(output_dir: str, log_dir: Path):
    """清理数据文件，只保留日志"""
    output_path = Path(output_dir)
    
    # 复制日志文件到log_dir
    for log_file in output_path.glob("**/*.txt"):
        dest = log_dir / log_file.name
        shutil.copy2(log_file, dest)
    
    for log_file in output_path.glob("**/*.log"):
        dest = log_dir / log_file.name
        shutil.copy2(log_file, dest)
    
    # 强制垃圾回收，释放可能被 pandas 持有的文件句柄
    gc.collect()
    
    # 删除整个临时输出目录（带重试机制）
    temp_dir = output_path.parent
    max_retries = 3
    for attempt in range(max_retries):
        try:
            shutil.rmtree(temp_dir)
            break
        except PermissionError as e:
            if attempt < max_retries - 1:
                # 等待一小段时间让文件句柄释放
                time.sleep(0.5)
                gc.collect()
            else:
                # 最后一次尝试失败，尝试逐个删除文件
                try:
                    # 尝试删除可以删除的文件
                    for file in temp_dir.rglob("*"):
                        if file.is_file():
                            try:
                                file.unlink()
                            except:
                                pass
                except:
                    pass
        except Exception as e:
            break


# ============================================================
# workspace_root 解析与 --config-dir 短格式展开
# ============================================================

# 项目根目录（src/core/run/utils.py → src/core/run/ → src/core/ → src/ → 项目根）
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_workspace_root() -> Path:
    """按优先级解析 ``workspace_root``。

    优先级：
      1. 环境变量 ``CHAINSIGHT_WORKSPACE``（最高，给 CI/CD 与容器化用）
      2. 项目根 ``.env`` 文件中的 ``CHAINSIGHT_WORKSPACE=...`` 行（给本地开发用）
      3. ``config/defaults.yaml`` 的 ``workspace_root`` 字段（项目默认）
      4. 兜底：``<项目根>/workspace``

    返回值始终为绝对路径（``Path.resolve()``）。相对路径会以项目根为基拼接。
    """
    return resolve_workspace_root_with_source()[1]


def resolve_workspace_root_with_source() -> tuple[str, Path]:
    """按优先级解析 ``workspace_root``，同时返回命中的来源标签。"""
    # 1) 环境变量
    env_val = os.environ.get("CHAINSIGHT_WORKSPACE")
    if env_val:
        return "env:CHAINSIGHT_WORKSPACE", _to_abs(env_val)

    # 2) 项目根 .env
    env_file = _PROJECT_ROOT / ".env"
    if env_file.is_file():
        try:
            for raw in env_file.read_text(encoding="utf-8-sig").splitlines():
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("CHAINSIGHT_WORKSPACE="):
                    value = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if value:
                        return ".env:CHAINSIGHT_WORKSPACE", _to_abs(value)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"读取 .env 失败，忽略：{e}")

    # 3) config/defaults.yaml
    yaml_path = _PROJECT_ROOT / "config" / "defaults.yaml"
    if yaml_path.is_file():
        try:
            import yaml  # 项目已用 pyyaml
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8-sig")) or {}
            wr = data.get("workspace_root")
            if wr:
                return "config/defaults.yaml:workspace_root", _to_abs(str(wr))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"读取 config/defaults.yaml::workspace_root 失败，忽略：{e}")

    # 4) 兜底
    return "fallback:project_root/workspace", (_PROJECT_ROOT / "workspace").resolve()


def _to_abs(value: str) -> Path:
    """把字符串路径解析成绝对路径——相对值以项目根为基。"""
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    return p.resolve()


def expand_config_dir_arg(arg: str) -> Path:
    """把 ``--config-dir`` 的值展开为 ``config/`` 目录的绝对路径。

    解析规则：

    - **绝对路径**：原样返回（``Path.resolve()`` 规范化）。**首要输入形式。**
    - **相对路径**：依次尝试 ``<workspace_root>/<arg>`` 与 ``<CWD>/<arg>``，
      取首个 ``is_dir()`` 命中的；都不存在则落到 ``<workspace_root>/<arg>``，
      让下游 ``ConfigDir.from_path`` 报准确的"目录不存在"错误。
    - 解析结果若末段不是 ``config``，自动追加 ``/config``——同时兼容：

      * 旧 2 段短格式 ``<project>/<scenario>``
        → ``<workspace_root>/<project>/<scenario>/config``
      * 新 3 段 ``<project>/scenarios/<scenario>``
        → ``<workspace_root>/<project>/scenarios/<scenario>/config``
      * 新 4 段 ``<project>/scenarios/<scenario>/config``（已含 ``config/``）
        → 原样使用

    本函数只做路径展开，不做目录存在性 / Excel 唯一性 / CSV 唯一性校验——
    这些都由 ``ConfigDir.from_path`` 在拿到展开后的路径时执行。
    """
    if not arg:
        raise ValueError("--config-dir 不能为空")

    p = Path(arg).expanduser()
    if p.is_absolute():
        return p.resolve()

    workspace_root = resolve_workspace_root()
    # workspace_root 优先、CWD 兜底
    candidates = [workspace_root / p, Path.cwd() / p]
    base = next((c for c in candidates if c.is_dir()), candidates[0])
    if base.name != "config":
        base = base / "config"
    return base.resolve()
