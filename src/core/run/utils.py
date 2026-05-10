"""`src.core.run` 子包通用辅助工具。"""
from __future__ import annotations

import gc
import shutil
import time
from pathlib import Path

EXCEL_SUFFIXES = (".xlsx", ".xlsm", ".xls")


def resolve_excel_path(config_arg: str) -> Path | None:
    """将用户传入的 ``--config`` 参数解析为 Excel 配置文件的实际路径。

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
