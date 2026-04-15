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
    config_path: Path,
    resume_mode: bool = False,
    resume_from: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interactive: bool = True,
    run_suffix: str = ""
) -> Path:
    """创建或选择本次运行使用的输出目录。

    目录结构：
      `outputs/<config_stem>/run_YYYYMMDD_HHMMSS/`

    Args:
        config_path: 配置文件路径。
        resume_mode: 是否启用历史目录续跑能力。
        resume_from: 指定要续跑的运行目录名称。
        start_date: 仿真开始日期，用于续跑校验。
        end_date: 仿真结束日期，用于续跑校验。
        interactive: 是否在存在多个候选目录时启用交互选择。
        run_suffix: 运行目录后缀参数，当前保留兼容接口。

    Returns:
        可作为 `output_base_dir` 使用的最终叶子目录路径。
    """
    cfg_stem = config_path.stem
    project_root = Path.cwd()
    
    # 集中式输出目录结构
    root_dir = project_root / "outputs" / cfg_stem
    # 始终确保顶层目录存在，以便其名称与配置文件名保持一致
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
    run_dir = root_dir / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)

    return run_dir


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
