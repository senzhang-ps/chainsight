"""运行输出目录管理工具。

职责：
- 创建标准化的 `outputs/<config>/run_*` 目录结构。
- 枚举历史运行目录并判断是否具备续跑能力。
- 管理配置维度的持久化仿真起始日期。
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..main_integration import check_resume_capability

# 仓库根（绑定到代码物理位置，不依赖 CWD）
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _list_existing_runs(
    root_dir: Path, start_date: str, end_date: str
) -> list[dict]:
    """列出历史运行目录及其续跑状态。

    Args:
        root_dir: 包含 `run_*` 子目录的根目录。
        start_date: 用于校验续跑范围的开始日期。
        end_date: 用于校验续跑范围的结束日期。

    Returns:
        包含运行目录路径、目录名与续跑信息的字典列表。
    """
    
    existing_runs = [
        d for d in root_dir.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    ]
    if not existing_runs:
        return []
    
    # 按名称排序（名称中包含时间戳）
    existing_runs.sort(reverse=True)
    
    run_infos = []
    for run_dir in existing_runs:
        try:
            resume_info = check_resume_capability(
                str(run_dir), start_date, end_date
            )
            run_infos.append({
                'path': run_dir,
                'name': run_dir.name,
                'resume_info': resume_info
            })
        except Exception:
            # 跳过无法分析的目录
            continue
    
    return run_infos


def _prompt_user_run_selection(run_infos: list[dict]) -> Path:
    """交互式选择续跑目录。

    Args:
        run_infos: 可选运行目录及其状态信息列表。

    Returns:
        用户选中的运行目录路径；若用户选择新建目录，则返回 `None`。
    """
    
    for idx, info in enumerate(run_infos, 1):
        resume_info = info['resume_info']
        
        if resume_info.get('already_completed', False):
            pass
        elif resume_info['can_resume']:
            pass
        else:
            pass
    
    
    while True:
        choice = input("\n👉 请选择: ").strip().lower()
        
        if choice in ['q', 'quit']:
            sys.exit(0)
        
        if choice in ['n', 'new']:
            return None  # 用于指示创建新的目录
        
        try:
            idx = int(choice)
            if 1 <= idx <= len(run_infos):
                selected = run_infos[idx - 1]
                return selected['path']
            else:
                pass
        except ValueError:
            pass


def _ensure_output_dir(
    config_source,
    resume_mode: bool = False,
    resume_from: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interactive: bool = True,
    run_suffix: str = ""
) -> Path:
    """创建或选择本次运行使用的输出目录。

    目录结构：
      - 新 ``ConfigDir`` 路径：``outputs/<project>/<scenario>/run_YYYYMMDD_HHMMSS/``
      - 旧 ``Path`` 路径：``outputs/<config_stem>/run_YYYYMMDD_HHMMSS/`` （单层兼容）

    Args:
        config_source: ``ConfigDir``（首选）或旧式 Excel ``Path``。
        resume_mode: 是否启用历史目录续跑能力。
        resume_from: 指定要续跑的运行目录名称。
        start_date: 仿真开始日期，用于续跑校验。
        end_date: 仿真结束日期，用于续跑校验。
        interactive: 是否在存在多个候选目录时启用交互选择。
        run_suffix: 运行目录后缀参数；非空时会作为 ``run_<ts>_<suffix>`` 拼到
            目录名上（如 file 模式传 ``"test"`` → ``run_<ts>_test``）。

    Returns:
        可作为 `output_base_dir` 使用的最终叶子目录路径。
    """
    output_subpath = _output_subpath_from_config_source(config_source)
    project_root = _PROJECT_ROOT

    # 二级输出根：outputs/<project>/<scenario>/
    root_dir = project_root / "outputs" / output_subpath
    # 始终确保顶层目录存在，以便续跑状态与 run_* 子目录位于同一配置维度下
    root_dir.mkdir(parents=True, exist_ok=True)

    # 如果指定了具体运行目录，则校验后返回
    if resume_from:
        target_dir = root_dir / resume_from
        if not target_dir.exists() or not target_dir.is_dir():
            raise ValueError(
                f"Specified run directory does not exist: {resume_from}"
            )
        return target_dir

    # 如果启用了续跑模式，则检查是否存在历史运行目录
    if resume_mode and start_date and end_date:
        run_infos = _list_existing_runs(root_dir, start_date, end_date)
        
        if run_infos:
            # 续跑时过滤掉已完成的运行目录
            resumable_runs = [
                r for r in run_infos
                if r['resume_info']['can_resume'] or
                   not r['resume_info'].get('already_completed', False)
            ]
            
            if resumable_runs:
                if interactive and len(resumable_runs) > 1:
                    # 存在多个可用运行目录：让用户选择
                    selected_dir = _prompt_user_run_selection(resumable_runs)
                    if selected_dir:
                        return selected_dir
                    # 用户选择了"new"：继续往下创建新目录
                elif resumable_runs:
                    # 只有一个可用运行目录或非交互模式：使用最新的目录
                    selected = resumable_runs[0]
                    return selected['path']

    # 在顶层目录下创建唯一的运行文件夹以避免冲突
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = run_suffix.strip()
    if suffix and not suffix.startswith("_"):
        suffix = f"_{suffix}"
    run_dir = root_dir / f"run_{ts}{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)

    return run_dir


def _output_subpath_from_config_source(config_source) -> Path:
    """从 ``ConfigDir`` 或旧式 Excel 路径推导输出二级子路径。

    返回值始终为 ``Path``：
      - ``ConfigDir``：返回 ``Path(<project>) / <scenario>``（二级）。
      - 旧式 Excel ``Path``：返回单段 ``Path(<excel_stem>)``，保持
        ``outputs/<excel_stem>/run_*/`` 旧行为以兼容遗留调用方。
    """
    if hasattr(config_source, "output_subpath"):
        sub = config_source.output_subpath
        if sub:
            return Path(sub)
    if hasattr(config_source, "excel_path"):
        return Path(Path(config_source.excel_path).stem)
    return Path(Path(config_source).stem)


def _write_run_id_file(
    log_dir: Path, run_id: str, filename: str, logger, db_schema: str | None = None
) -> None:
    """把本次运行的 run_id 落盘到 ``log_dir/<filename>``。

    DB 模式 → ``db_run_id.txt``（第一行: ``db_<config>_<ts>``，与 DB 中实际使用的
    ``effective_run_id`` 一致；第二行可写 ``db_schema=<schema>``）。本地模式 →
    ``run_id.txt``（内容: run 目录 basename，与 ``outputs/<...>/<basename>/`` 对得上）。

    无尾换行、UTF-8 无 BOM —— 便于 ``cat <file>`` 直接拿来当字符串使用；续跑
    命中同一目录时覆盖写也是幂等的。失败仅 ``logger.warning``，不抛异常。
    """
    try:
        content = run_id
        if filename == "db_run_id.txt" and db_schema:
            content = f"{run_id}\ndb_schema={db_schema}"
        (log_dir / filename).write_text(content, encoding="utf-8")
    except OSError as e:
        logger.warning(f"[WARN] 写入 {filename} 失败（不影响仿真）: {e}")


def get_or_init_simulation_start(
    output_root: Path, provided_start: Optional[str]
) -> str:
    """返回该配置对应的持久化仿真起始日期。

    ``output_root`` 是包含所有运行目录的根目录。起始日期保存在该目录下的
    ``simulation_start.txt`` 文件中：如果文件已存在则读取并返回其内容；否则
    将 ``provided_start`` 写入文件后返回。若是首次运行且该文件尚不存在，则
    必须提供 ``provided_start``。
    """
    start_file = output_root / "simulation_start.txt"
    if start_file.exists():
        return start_file.read_text(encoding="utf-8-sig").strip()
    if not provided_start:
        raise ValueError("Simulation start date required for first run")
    start_file.write_text(provided_start, encoding="utf-8")
    return provided_start
