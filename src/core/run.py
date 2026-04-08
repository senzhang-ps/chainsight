#!/usr/bin/env python3
"""
供应链计划系统的生产集成运行器

用途
- 读取用户提供的配置文件路径（.xlsx）
- 在与配置文件相同的目录下创建输出目录
  其顶层文件夹名称与配置文件的文件名（不含扩展名）一致
- 触发完整的一体化运行，并将结果写入该输出目录

说明
- 不包含测试脚手架、校验占位逻辑或 print 语句
- 默认保持安静；失败信号依赖退出码与异常

命令行示例
  production_integrator.py \
    --config /path/to/your_config.xlsx \
    --start-date 2024-01-01 \
    --end-date 2024-01-31

首次运行必须提供 ``--start-date``。后续运行会复用
保存在 ``<config_dir>/<config_stem>/simulation_start.txt`` 中的起始日期。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Windows UTF-8 编码设置 - 解决emoji和中文输出问题
# 注意：只在 stdout/stderr 尚未被包装时才进行包装，避免重复包装导致 buffer 被关闭
if sys.platform == 'win32':
    import io
    # 检查是否已经被包装（避免重复包装）
    if hasattr(sys.stdout, 'buffer') and not isinstance(sys.stdout, io.TextIOWrapper):
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        except Exception:
            pass  # 忽略包装失败
    if hasattr(sys.stderr, 'buffer') and not isinstance(sys.stderr, io.TextIOWrapper):
        try:
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
        except Exception:
            pass  # 忽略包装失败

# 将父目录添加到路径中以便导入
sys.path.insert(0, str(Path(__file__).parent.parent))

# 外部系统导入（假定在项目环境中可用）
from .main_integration import run_integrated_simulation, load_configuration, check_resume_capability  # type: ignore
from ..utils.logger_config import setup_logging  # type: ignore


def _list_existing_runs(root_dir: Path, start_date: str, end_date: str) -> list[dict]:
    """List all existing run directories with their resume status.
    
    Args:
        root_dir: The root directory containing run_* folders
        start_date: Start date for validation
        end_date: End date for validation
        
    Returns:
        List of dicts with run directory info and resume capability
    """
    
    existing_runs = [d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not existing_runs:
        return []
    
    # 按名称排序（名称中包含时间戳）
    existing_runs.sort(reverse=True)
    
    run_infos = []
    for run_dir in existing_runs:
        try:
            resume_info = check_resume_capability(str(run_dir), start_date, end_date)
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
    """Interactively prompt user to select a run directory.
    
    Args:
        run_infos: List of run directory information
        
    Returns:
        Selected run directory Path
    """
    print("\n" + "="*80)
    print("📂 发现多个运行目录，请选择要续跑的目录：")
    print("="*80)
    
    for idx, info in enumerate(run_infos, 1):
        resume_info = info['resume_info']
        print(f"\n[{idx}] {info['name']}")
        
        if resume_info.get('already_completed', False):
            print(f"    [OK] 状态: 已完成")
            print(f"    📅 最后日期: {resume_info['last_complete_date']}")
            print(f"    [DATA] 完成天数: {resume_info['days_completed']}")
        elif resume_info['can_resume']:
            print(f"    🔄 状态: 可续跑")
            print(f"    📅 已完成: {resume_info['days_completed']} 天 (截至 {resume_info['last_complete_date']})")
            print(f"    📅 剩余: {resume_info['days_remaining']} 天 (从 {resume_info['resume_from_date']} 开始)")
        else:
            print(f"    [LOG] 状态: 无可续跑数据")
            print(f"    [DATA] 需处理: {resume_info['days_remaining']} 天")
    
    print("\n" + "="*80)
    print("请输入选项：")
    print("  - 输入数字 [1-{}] 选择对应目录".format(len(run_infos)))
    print("  - 输入 'n' 或 'new' 创建新的运行目录")
    print("  - 输入 'q' 或 'quit' 退出")
    
    while True:
        choice = input("\n👉 请选择: ").strip().lower()
        
        if choice in ['q', 'quit']:
            print("[ERROR] 用户取消操作")
            sys.exit(0)
        
        if choice in ['n', 'new']:
            return None  # 用于指示创建新的目录
        
        try:
            idx = int(choice)
            if 1 <= idx <= len(run_infos):
                selected = run_infos[idx - 1]
                print(f"\n[OK] 已选择: {selected['name']}")
                return selected['path']
            else:
                print(f"[ERROR] 无效选择，请输入 1-{len(run_infos)} 之间的数字")
        except ValueError:
            print("[ERROR] 无效输入，请输入数字、'n' 或 'q'")


def _ensure_output_dir(config_path: Path, resume_mode: bool = False, 
                      resume_from: Optional[str] = None, 
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      interactive: bool = True,
                      run_suffix: str = "") -> Path:
    """Create the output directory rooted in centralized outputs/ folder.

    Structure:
      outputs/<config_stem>/
        └─ run_YYYYMMDD_HHMMSS/  (actual write target to avoid overwrites)
        └─ run_YYYYMMDD_HHMMSS_suffix/  (if run_suffix is provided)

    Args:
        config_path: Path to the configuration file
        resume_mode: If True, allow resuming from existing run directories
        resume_from: Specific run directory name to resume from (e.g., "run_20241203_120000")
        start_date: Simulation start date (required for resume validation)
        end_date: Simulation end date (required for resume validation)
        interactive: If True, prompt user to select run directory when multiple exist
        run_suffix: Optional suffix to append to run directory name

    Returns the leaf path to be used as `output_base_dir`.
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
            raise ValueError(f"Specified run directory does not exist: {resume_from}")
        print(f"📂 使用指定的运行目录: {resume_from}")
        return target_dir

    # 如果启用了续跑模式，则检查是否存在历史运行目录
    if resume_mode and start_date and end_date:
        run_infos = _list_existing_runs(root_dir, start_date, end_date)
        
        if run_infos:
            # 续跑时过滤掉已完成的运行目录
            resumable_runs = [r for r in run_infos 
                            if r['resume_info']['can_resume'] or 
                               not r['resume_info'].get('already_completed', False)]
            
            if resumable_runs:
                if interactive and len(resumable_runs) > 1:
                    # 存在多个可用运行目录：让用户选择
                    selected_dir = _prompt_user_run_selection(resumable_runs)
                    if selected_dir:
                        return selected_dir
                    # 用户选择了“new”：继续往下创建新目录
                elif resumable_runs:
                    # 只有一个可用运行目录或非交互模式：使用最新的目录
                    selected = resumable_runs[0]
                    print(f"📂 自动选择最新的运行目录: {selected['name']}")
                    return selected['path']

    # 在顶层目录下创建唯一的运行文件夹以避免冲突
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root_dir / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    print(f"📂 创建新的运行目录: run_{ts}")

    return run_dir


def _write_results_to_local(all_results: dict, output_dir: Path, logger) -> None:
    """将内存中的仿真结果写入本地Excel文件（--also-local 模式使用）
    
    Args:
        all_results: 仿真结果字典 {module_name: [day1_result, day2_result, ...]}
        output_dir: 输出目录
        logger: 日志记录器
    """
    import pandas as pd
    
    # 模块名到输出文件名的映射
    module_output_map = {
        'module1': ('order_log', 'Module1_OrderLog.xlsx'),
        'module3': ('net_demand', 'Module3_NetDemand.xlsx'),
        'module4': ('production_plan', 'Module4_ProductionPlan.xlsx'),
        'module5': ('deployment_plan', 'Module5_DeploymentPlan.xlsx'),
        'module6': ('delivery_plan', 'Module6_DeliveryPlan.xlsx'),
    }
    
    for module_name, (data_key, file_name) in module_output_map.items():
        if module_name not in all_results:
            logger.warning(f"  [WARN] {module_name} 不在结果中")
            continue
        
        module_days = all_results[module_name]
        if not module_days:
            logger.warning(f"  [WARN] {module_name} 无数据")
            continue
        
        # 合并所有天的数据
        all_dfs = []
        for day_result in module_days:
            if isinstance(day_result, dict):
                # 尝试多种键名
                df = None
                for key in [data_key, f'{module_name}_{data_key}', 'output', 'result']:
                    if key in day_result and isinstance(day_result[key], pd.DataFrame):
                        df = day_result[key]
                        break
                # 如果找不到特定键，尝试找第一个DataFrame
                if df is None:
                    for v in day_result.values():
                        if isinstance(v, pd.DataFrame) and not v.empty:
                            df = v
                            break
                if df is not None and not df.empty:
                    all_dfs.append(df)
            elif isinstance(day_result, pd.DataFrame) and not day_result.empty:
                all_dfs.append(day_result)
        
        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            excel_path = output_dir / file_name
            combined_df.to_excel(excel_path, index=False)
            logger.info(f"  [OK] {file_name}: {len(combined_df)} 行")
        else:
            logger.warning(f"  [WARN] {module_name} 无有效数据可写入")


def _write_results_to_local_dev_format(
    all_results: dict, 
    output_dir: Path, 
    logger,
    config_dict: Optional[dict] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    orchestrator_output_dir: Optional[str] = None
) -> None:
    """将内存中的仿真结果写入本地文件，使用与Dev版本一致的目录结构和Sheet格式
    
    目录结构 (与ChainSight_Dev一致):
        output_dir/
            module1/
                module1_output_20251215.xlsx  (包含 OrderLog, ShipmentLog, CutLog, SupplyDemandLog, Summary sheets)
                ...
            module3/
                Module3Output_20251215.xlsx  (包含 NetDemand sheet)
                ...
            module4/
                Module4Output_20251215.xlsx  (包含 ProductionPlan, CapacityExceed, Validation, ChangeoverLog sheets)
                allocated_capacity_20251216.json
                line_states_20251216.json
                ...
            module5/
                Module5Output_20251215.xlsx  (包含 DeploymentPlan, UnfulfilledLog, StockOnHandLog, Validation sheets)
                ...
            module6/
                Module6Output_20251215.xlsx  (包含 DeliveryPlan, VehicleLog, TruckUsageLog, etc sheets)
                ...
            orchestrator/
                (CSV files from orchestrator)
            summary/
                full_production_plan_report.xlsx
                full_order_shipment_cut_report.xlsx
                ...
    
    Args:
        all_results: 仿真结果字典 {module_name: [day1_result, day2_result, ...]}
        output_dir: 输出目录
        logger: 日志记录器
        config_dict: 配置字典（用于生成summary报告）
        start_date: 仿真开始日期
        end_date: 仿真结束日期
        orchestrator_output_dir: Orchestrator临时输出目录（用于复制CSV）
    """
    import pandas as pd
    import shutil
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time
    
    write_start = time.time()
    
    # 创建模块输出目录
    module_dirs = {}
    for module_name in ['module1', 'module3', 'module4', 'module5', 'module6']:
        module_dir = output_dir / module_name
        module_dir.mkdir(parents=True, exist_ok=True)
        module_dirs[module_name] = module_dir
    
    # 模块文件名格式
    module_file_formats = {
        'module1': 'module1_output_{date}.xlsx',
        'module3': 'Module3Output_{date}.xlsx',
        'module4': 'Module4Output_{date}.xlsx',
        'module5': 'Module5Output_{date}.xlsx',
        'module6': 'Module6Output_{date}.xlsx',
    }
    
    # 模块的DataFrame键名到Sheet名称的映射 (与Dev版本一致)
    module_sheet_mapping = {
        'module1': {
            'all_orders_for_next_day': 'OrderLog',  # 或 'orders_df'
            'orders_df': 'OrderLog',
            'shipment_df': 'ShipmentLog',
            'cut_df': 'CutLog',
            'supply_demand_df': 'SupplyDemandLog',
            'summary_df': 'Summary',
        },
        'module3': {
            'net_demand_df': 'NetDemand',
        },
        'module4': {
            'production_df': 'ProductionPlan',
            'exceed_log': 'CapacityExceed',
            'issues_df': 'Validation',
            'changeover_log': 'ChangeoverLog',
        },
        'module5': {
            'deployment_plan': 'DeploymentPlan',
            'unfulfilled_log': 'UnfulfilledLog',
            'stock_on_hand_log': 'StockOnHandLog',
            'validation_log': 'Validation',
        },
        'module6': {
            'delivery_plan': 'DeliveryPlan',
            'vehicle_log': 'VehicleLog',
            'truck_usage': 'TruckUsageLog',
            'unsatisfied_log': 'UnsatisfiedMDQLog',
            'validation_log': 'ValidationLog',
            'bypass_log': 'BypassRuleHitLog',
        },
    }
    
    # 收集所有需要写入的任务
    write_tasks = []
    
    for module_name, day_results in all_results.items():
        if not day_results:
            continue
        
        module_dir = module_dirs.get(module_name)
        if not module_dir:
            continue
        
        file_format = module_file_formats.get(module_name)
        if not file_format:
            continue
        
        sheet_mapping = module_sheet_mapping.get(module_name, {})
        
        for day_result in day_results:
            if not isinstance(day_result, dict):
                continue
            
            # 获取仿真日期
            sim_date = day_result.get('simulation_date')
            if sim_date is None:
                continue
            
            if hasattr(sim_date, 'strftime'):
                date_str = sim_date.strftime('%Y%m%d')
            else:
                date_str = str(sim_date).replace('-', '')[:8]
            
            # 收集该天所有需要写入的DataFrame (按Sheet名称)
            sheets_to_write = {}
            for df_key, sheet_name in sheet_mapping.items():
                df = day_result.get(df_key)
                if df is not None and isinstance(df, pd.DataFrame):
                    # 如果同一个sheet已经有数据，跳过（避免重复）
                    if sheet_name not in sheets_to_write:
                        sheets_to_write[sheet_name] = df
            
            # 添加Excel写入任务（包含多个sheets）
            if sheets_to_write:
                file_name = file_format.format(date=date_str)
                file_path = module_dir / file_name
                write_tasks.append(('excel_multi_sheet', file_path, sheets_to_write))
            
            # Module4特殊处理：line_states 和 allocated_capacity (JSON格式)
            if module_name == 'module4':
                line_states = day_result.get('line_states')
                if line_states is not None:
                    json_path = module_dir / f'line_states_{date_str}.json'
                    write_tasks.append(('json', json_path, line_states))
                
                allocated_capacity = day_result.get('allocated_capacity')
                if allocated_capacity is not None:
                    json_path = module_dir / f'allocated_capacity_{date_str}.json'
                    write_tasks.append(('json', json_path, allocated_capacity))
    
    # 使用线程池并行写入文件
    def write_file(task):
        file_type, file_path, data = task
        try:
            if file_type == 'excel_multi_sheet':
                # 写入多个Sheet的Excel文件
                if isinstance(data, dict) and data:
                    total_rows = 0
                    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                        for sheet_name, df in data.items():
                            if isinstance(df, pd.DataFrame):
                                df.to_excel(writer, sheet_name=sheet_name, index=False)
                                total_rows += len(df)
                    return ('success', file_path, total_rows)
                else:
                    return ('skip', file_path, 0)
            elif file_type == 'excel':
                # 单一Sheet（保留向后兼容）
                if isinstance(data, pd.DataFrame) and not data.empty:
                    data.to_excel(file_path, index=False)
                    return ('success', file_path, len(data))
                elif isinstance(data, pd.DataFrame):
                    data.to_excel(file_path, index=False)
                    return ('success', file_path, 0)
            elif file_type == 'json':
                import json
                with open(file_path, 'w', encoding='utf-8') as f:
                    if isinstance(data, dict):
                        json.dump(data, f, indent=2, default=str)
                    elif isinstance(data, pd.DataFrame):
                        data.to_json(f, orient='records', indent=2)
                    else:
                        json.dump(data, f, indent=2, default=str)
                return ('success', file_path, 1)
        except Exception as e:
            return ('error', file_path, str(e))
        return ('skip', file_path, 0)
    
    # 并行写入
    success_count = 0
    error_count = 0
    total_rows = 0
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(write_file, task): task for task in write_tasks}
        for future in as_completed(futures):
            status, path, info = future.result()
            if status == 'success':
                success_count += 1
                if isinstance(info, int):
                    total_rows += info
            elif status == 'error':
                error_count += 1
                logger.warning(f"  [WARN] 写入失败 {path}: {info}")
    
    # 复制orchestrator目录
    if orchestrator_output_dir:
        orch_src = Path(orchestrator_output_dir)
        orch_dst = output_dir / "orchestrator"
        if orch_src.exists():
            orch_dst.mkdir(parents=True, exist_ok=True)
            for csv_file in orch_src.glob("*.csv"):
                shutil.copy2(csv_file, orch_dst / csv_file.name)
            logger.info(f"  [OK] orchestrator: 复制 {len(list(orch_src.glob('*.csv')))} 个CSV文件")
    
    # 生成Summary报告
    if config_dict and start_date and end_date:
        try:
            from .main_integration import SummaryReportGenerator
            summary_dir = output_dir / "summary"
            summary_dir.mkdir(parents=True, exist_ok=True)
            
            report_generator = SummaryReportGenerator(
                output_base_dir=str(output_dir),
                config_dict=config_dict
            )
            summary_reports = report_generator.generate_all_reports(
                start_date=start_date,
                end_date=end_date
            )
            logger.info(f"  [OK] summary: 生成 {len(summary_reports)} 个汇总报告")
        except Exception as e:
            logger.warning(f"  [WARN] Summary报告生成失败: {e}")
    
    write_time = time.time() - write_start
    logger.info(f"  [TIME] 本地文件写入耗时: {write_time:.2f}秒 ({success_count} 文件, {total_rows} 行)")
    if error_count > 0:
        logger.warning(f"  [WARN] {error_count} 个文件写入失败")


def get_or_init_simulation_start(output_root: Path, provided_start: Optional[str]) -> str:
    """返回该配置对应的持久化仿真起始日期。

    ``output_root`` 是包含所有运行目录的根目录。起始日期保存在该目录下的
    ``simulation_start.txt`` 文件中：如果文件已存在则读取并返回其内容；否则
    将 ``provided_start`` 写入文件后返回。若是首次运行且该文件尚不存在，则
    必须提供 ``provided_start``。
    """
    start_file = output_root / "simulation_start.txt"
    if start_file.exists():
        return start_file.read_text(encoding="utf-8").strip()
    if not provided_start:
        raise ValueError("Simulation start date required for first run")
    start_file.write_text(provided_start, encoding="utf-8")
    return provided_start


def _run_with_database(ns: argparse.Namespace) -> int:
    """
    数据库模式运行
    
    - 从数据库读取配置（使用配置名称如 BC_S5）
    - 输出写入数据库
    - 本地只保存运行日志txt文件
    """
    from datetime import datetime
    import time
    
    config_name = ns.config  # 配置名称，如 BC_S5
    start_date = ns.start_date
    end_date = ns.end_date
    
    if not start_date:
        raise ValueError("数据库模式必须提供 --start-date 参数")
    
    print("\n" + "=" * 70)
    print("🗄️  数据库模式运行")
    print("=" * 70)
    print(f"[INFO] 配置名称: {config_name}")
    print(f"📅 日期范围: {start_date} 到 {end_date}")
    print(f"🔌 数据库: {ns.db_host}:{ns.db_port}/{ns.db_name}")
    print("=" * 70)
    
    # 导入数据库模块
    try:
        from pgsql_db.db_connection import DatabaseConnection
        from pgsql_db.excel_importer import ExcelImporter
        from pgsql_db.module_data_writer import ModuleDataWriter
        from pgsql_db.db_initializer import DatabaseInitializer
    except ImportError as e:
        print(f"[ERROR] 无法导入数据库模块: {e}")
        print("   请确保已安装 psycopg: pip install psycopg[binary]")
        return 1
    
    # ========== 使用 DatabaseInitializer 自动检测和初始化 ==========
    initializer = DatabaseInitializer(
        host=ns.db_host,
        port=ns.db_port,
        database=ns.db_name,
        user=ns.db_user,
        password=ns.db_password
    )
    
    # 执行初始化（检测数据库、创建数据库、检测配置表、导入配置）
    init_result = initializer.initialize(
        config_name=config_name,
        auto_import_config=True,
        verbose=True
    )
    
    if not init_result["success"]:
        print("[ERROR] 数据库初始化失败")
        return 1
    
    # 获取数据库连接供后续使用
    db = initializer.db
    
    # 创建本地日志目录（只保存txt日志）
    project_root = Path(__file__).parent.parent.parent
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = project_root / "outputs" / f"db_{config_name}_{ts}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[DIR] 日志目录: {log_dir}")
    
    # 设置日志系统
    from ..utils.logger_config import setup_logging
    logger, redirector = setup_logging(str(log_dir), log_level="INFO", redirect_print=True)
    
    total_start = time.time()
    program_start_datetime = datetime.now()
    
    logger.info("\n" + "=" * 60)
    logger.info("🕐 程序时间信息")
    logger.info("=" * 60)
    logger.info(f"📅 程序开始时间: {program_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # ========== 步骤1: 从数据库获取配置 ==========
        logger.info("\n" + "=" * 60)
        logger.info("📥 步骤1: 从数据库加载配置")
        logger.info("=" * 60)
        
        config_data = _load_config_from_database(db, config_name)
        if not config_data:
            logger.error(f"[ERROR] 未在数据库中找到配置: {config_name}")
            return 1
        
        logger.info(f"[OK] 已加载 {len(config_data)} 个配置表")
        
        # ========== 步骤2: 使用DuckDB处理并运行仿真 ==========
        logger.info("\n" + "=" * 60)
        logger.info("[DUCK] 步骤2: DuckDB处理配置并运行仿真")
        logger.info("=" * 60)
        
        # 创建临时输出目录
        import tempfile
        temp_dir = Path(tempfile.mkdtemp())
        temp_output = temp_dir / "output"
        temp_output.mkdir(exist_ok=True)
        
        # [FIX §3.11] 始终使用标准仿真引擎（与文件模式相同的代码路径）
        # 高性能DuckDB引擎(optimized_simulation)的M3/M4计算结果与标准引擎不一致，
        # 导致NetDemand、ProductionPlan等输出与Dev参考版本存在差异。
        # 标准引擎(run_integrated_simulation_from_dict)已验证100%匹配Dev输出。
        from .main_integration import run_integrated_simulation_from_dict
        logger.info("[RUN] 使用标准仿真引擎运行（与文件模式一致）...")
        result = run_integrated_simulation_from_dict(
            config_data=config_data,
            config_name=config_name,
            start_date=start_date,
            end_date=end_date,
            output_base_dir=str(temp_output),
            skip_validation=True,
            skip_summary_report=True  # DB模式下跳过汇总报告，由run.py在写入DB后单独生成
        )
        
        if not result or not result.get('simulation_completed'):
            logger.error("[ERROR] 仿真运行失败")
            return 1
        
        # ========== 步骤3: 写入输出到数据库 ==========
        logger.info("\n" + "=" * 60)
        logger.info("[OUT] 步骤3: 写入输出到数据库（完整模式）")
        logger.info("=" * 60)
        
        output_dir = result.get('output_directory')
        writer = ModuleDataWriter(db, config_name=config_name)  # 传入config_name参数
        run_id = f"{config_name}_{ts}"
        
        # 【完整模式】写入所有模块的详细输出数据 + Summary + Orchestrator
        # 包含: module1, module3, module4, module5, module6 的每日输出sheet
        db_write_start = time.time()
        
        # [FIX] 关键修复：从内存结果直接写入数据库（而非从文件读取）
        # 仿真使用 skip_file_output=True，因此模块输出仅存在于内存中
        all_results = result.get('results')
        
        # [DEBUG] DEBUG: 详细打印 all_results 结构
        logger.info("=" * 60)
        logger.info("[DEBUG] DEBUG: 检查 all_results 结构")
        logger.info("=" * 60)
        if all_results:
            logger.info(f"all_results 类型: {type(all_results)}")
            logger.info(f"all_results 键: {list(all_results.keys()) if isinstance(all_results, dict) else 'N/A'}")
            for module_name, module_results in all_results.items():
                logger.info(f"  {module_name}: {len(module_results) if isinstance(module_results, list) else 'N/A'} 天的结果")
                if isinstance(module_results, list) and len(module_results) > 0:
                    first_day = module_results[0]
                    if isinstance(first_day, dict):
                        logger.info(f"    第一天的键: {list(first_day.keys())}")
                        for key, value in first_day.items():
                            if hasattr(value, 'shape'):
                                logger.info(f"      {key}: DataFrame shape={value.shape}")
                            elif hasattr(value, '__len__'):
                                logger.info(f"      {key}: type={type(value).__name__}, len={len(value)}")
                            else:
                                logger.info(f"      {key}: type={type(value).__name__}")
        else:
            logger.warning("[WARN] all_results 为空或 None!")
            logger.info(f"result 的键: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
        logger.info("=" * 60)
        
        if all_results:
            logger.info("[OUT] 从内存直接写入模块输出到数据库...")
            writer.write_module_results_from_dict(all_results, run_id=run_id, if_exists='replace')
            
            # ========== 本地文件输出（默认禁用） ==========
            # 数据库模式默认仅写入数据库，不输出本地文件
            # 使用 --local 可以启用本地输出
            if getattr(ns, 'local', False):
                logger.info("\n" + "=" * 60)
                logger.info("[DIR] 输出本地文件（Dev格式）")
                logger.info("=" * 60)
                local_output_dir = project_root / "outputs" / config_name / f"run_{ts}"
                local_output_dir.mkdir(parents=True, exist_ok=True)
                
                # 获取orchestrator输出目录
                orch_output_dir = str(Path(output_dir) / "orchestrator") if output_dir else None
                
                # 使用Dev格式写入（并行写入，高性能）
                _write_results_to_local_dev_format(
                    all_results=all_results,
                    output_dir=local_output_dir,
                    logger=logger,
                    config_dict=result.get('config_dict'),  # 从result获取
                    start_date=start_date,
                    end_date=end_date,
                    orchestrator_output_dir=orch_output_dir
                )
                logger.info(f"[OK] 本地输出目录: {local_output_dir}")
                
        elif output_dir and Path(output_dir).exists():
            # 回退：如果有文件输出，则从文件读取（兼容旧模式）
            logger.info("[OUT] 从文件读取并写入模块输出到数据库...")
            writer.write_all_modules(str(output_dir), run_id=run_id, if_exists='replace')
        else:
            logger.warning("[WARN] 无可用的模块输出数据写入数据库")
        
        # [DIR] 写入 Orchestrator 状态数据（从临时目录读取）
        if output_dir:
            orch_dir = Path(output_dir) / "orchestrator"
            if orch_dir.exists():
                logger.info("[OUT] 写入 Orchestrator 状态数据到数据库...")
                writer.write_orchestrator_data(str(orch_dir), run_id=run_id, if_exists='replace')
            else:
                logger.warning("[WARN] Orchestrator 目录不存在，跳过写入")
        
        # [DATA] 生成 Summary 汇总报告（从数据库已写入的模块输出表直接聚合）
        # 注意：在DB模式下，模块不输出xlsx文件，因此无法使用SummaryReportGenerator
        # 此方法直接从数据库的module1/4/5/6输出表中聚合生成7个Summary报告表
        logger.info("[DATA] 从数据库生成 Summary 汇总报告...")
        writer.generate_summary_reports_from_db(
            run_id=run_id,
            start_date=start_date,
            end_date=end_date,
            if_exists='replace'
        )
        
        db_write_time = time.time() - db_write_start
        logger.info(f"[TIME]  数据库写入耗时: {db_write_time:.2f}秒")
        
        total_time = time.time() - total_start
        program_end_datetime = datetime.now()
        
        # 格式化运行时间
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours >= 1:
            runtime_str = f"{int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒"
        elif minutes >= 1:
            runtime_str = f"{int(minutes)}分钟 {seconds:.2f}秒"
        else:
            runtime_str = f"{seconds:.2f}秒"
        
        logger.info("\n" + "=" * 60)
        logger.info("[OK] 数据库模式运行完成 (DuckDB)")
        logger.info("=" * 60)
        logger.info("🕐 程序时间统计:")
        logger.info(f"   📅 开始时间: {program_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"   📅 结束时间: {program_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"   [TIME]  总运行时间: {runtime_str}")
        logger.info(f"   [TIME]  其中DB写入: {db_write_time:.2f}秒")
        logger.info(f"   [DIR] 日志目录: {log_dir}")
        logger.info("[DATA] 数据存储:")
        logger.info(f"   [SAVE] 所有输出数据已写入 PostgreSQL 数据库")
        logger.info(f"   🔗 连接: postgresql://localhost:5432/test_db")
        logger.info(f"   [INFO] 表前缀: {run_id}")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"[ERROR] 执行出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    finally:
        # 关闭数据库连接
        db.close()
        logger.info("🔌 数据库连接已关闭")
        if redirector:
            redirector.stop_redirect()


def _load_config_from_database(db, config_name: str) -> dict:
    """从数据库加载配置表"""
    import pandas as pd
    
    all_tables = db.get_all_tables()
    config_data = {}
    
    # 计算旧格式的配置前缀（如 bc_s5_）
    old_prefix = config_name.lower().replace("-", "_").replace(" ", "_") + "_"
    
    # 优先尝试新格式（cfg_开头，通过 config_name 字段区分）
    for table_name in all_tables:
        if not table_name.startswith('cfg_'):
            continue
            
        try:
            df = db.read_table(table_name)
        except Exception as e:
            print(f"  [WARN] 加载配置表失败 [{table_name}]: {e}")
            continue
        
        # 只接受包含 config_name 列的表；并按指定配置过滤
        if 'config_name' not in df.columns:
            continue
        
        # 优先精确匹配 config_name，无数据时回退到 basename
        filtered = df[df['config_name'] == config_name]
        if filtered.empty:
            config_basename = config_name.split('/')[-1] if '/' in config_name else config_name
            if config_basename != config_name:
                filtered = df[df['config_name'] == config_basename]
        
        # 清理 DB 元数据列和非标准列（unnamed_*、全 NULL 列等）
        drop_cols = [c for c in filtered.columns if c in ('config_name', 'config_type', 'db_write_time')
                     or c.startswith('unnamed') or c.startswith('Unnamed')]
        if drop_cols:
            filtered = filtered.drop(columns=drop_cols, errors='ignore')
        # 删除全为 NULL 的列（来自其他配置的表结构残留），但保留原有列结构
        cols_before = set(filtered.columns)
        filtered = filtered.dropna(axis=1, how='all')
        for col in cols_before - set(filtered.columns):
            filtered[col] = pd.NA

        # 去掉 cfg_ 前缀，作为配置数据的 key
        clean_table_name = table_name[4:]
        
        # 即使过滤后为空，也保留表结构（对于某些模块配置表是必要的）
        config_data[clean_table_name] = filtered
        print(f"  [OK] 加载配置表: {table_name} -> {clean_table_name} ({len(filtered)} 行)")
    
    # 如果新格式没有数据，回退到旧格式（兼容旧数据）
    if not config_data:
        print(f"  [INFO] 未找到新格式配置表(cfg_*)，尝试旧格式({old_prefix}*)...")
        for table_name in all_tables:
            if not table_name.startswith(old_prefix):
                continue
                
            try:
                df = db.read_table(table_name)
            except Exception as e:
                print(f"  [WARN] 加载配置表失败 [{table_name}]: {e}")
                continue
            
            if df.empty:
                continue
            
            # 去掉旧前缀，作为配置数据的 key
            clean_table_name = table_name[len(old_prefix):]
            config_data[clean_table_name] = df
            print(f"  [OK] 加载配置表(旧格式): {table_name} -> {clean_table_name} ({len(df)} 行)")
    
    return config_data if config_data else None


def _get_column_mapping() -> dict:
    """获取列名映射（数据库小写 -> 原始大小写）"""
    # 定义所有实际使用的列名及其原始大小写形式
    original_columns = [
        # 通用列（小写）
        'material', 'location', 'sourcing', 'location_type', 
        'quantity', 'date', 'week', 'day', 'seed',
        # 带下划线的列
        'eff_from', 'eff_to', 'demand_element', 'priority',
        'order_type', 'error_std_percent', 'order_day_flag',
        'advance_days', 'ao_percent', 'dps_location', 'dps_percent',
        'safety_stock_qty', 'key',
        # Global_LeadTime
        'sending', 'receiving', 'PDT', 'GR', 'MCT', 'OTD',
        # M4相关
        'delegate_line', 'prd_rate', 'min_batch', 'rv', 'ptf', 'lsk',
        'line', 'capacity', 'from_material', 'to_material',
        'changeover_id', 'from line', 'to line',
        'time', 'cost', 'mu_loss', 'pr',
        # M5相关
        'model', 'moq',
        # M6相关
        'truck_type', 'optimal_type', 'WFR', 'VFR', 'MDQ',
        'weight', 'volume', 'demand_unit_to_weight', 'demand_unit_to_volume',
        'delay_days', 'probability', 'condition_logic', 'rule_id',
        'max_weight', 'max_volume', 'capacity_qty_in_weight', 'capacity_qty_in_volume',
    ]
    
    # 创建小写到原始的映射
    mapping = {}
    for col in original_columns:
        mapping[col.lower().replace(' ', '_')] = col
    
    return mapping


def _run_simulation_with_db_config(
    config_data: dict,
    config_name: str,
    start_date: str,
    end_date: str,
    log_dir: Path,
    logger
) -> dict:
    """使用数据库配置运行仿真"""
    import tempfile
    import pandas as pd
    
    # 创建临时Excel文件
    temp_dir = Path(tempfile.mkdtemp())
    temp_config = temp_dir / f"{config_name}.xlsx"
    
    try:
        # 将配置数据写入临时Excel
        with pd.ExcelWriter(temp_config, engine='openpyxl') as writer:
            # 映射表名到原始sheet名
            sheet_mapping = {
                'sit_design': 'SIT Design',
                'global_seed': 'Global_seed',
                'config_guide': 'Config Guide',
                'global_network': 'Global_Network',
                'global_spacecapacity': 'Global_SpaceCapacity',
                'global_leadtime': 'Global_LeadTime',
                'global_demandpriority': 'Global_DemandPriority',
                'm1_initialinventory': 'M1_InitialInventory',
                'm1_initialinventory_30d': 'M1_InitialInventory_30D',
                'sheet1': 'Sheet1',
                'm1_demandforecast': 'M1_DemandForecast',
                'm1_forecasterror': 'M1_ForecastError',
                'm1_ordercalendar': 'M1_OrderCalendar',
                'm1_aoconfig': 'M1_AOConfig',
                'm1_dpsconfig': 'M1_DPSConfig',
                'm1_supplychoiceconfig': 'M1_SupplyChoiceConfig',
                'm3_safetystock': 'M3_SafetyStock',
                'covalidation': 'COValidation',
                'm4_materiallocationlinecfg': 'M4_MaterialLocationLineCfg',
                'm4_linecapacity': 'M4_LineCapacity',
                'm4_changeovermatrix': 'M4_ChangeoverMatrix',
                'm4_changeoverdefinition': 'M4_ChangeoverDefinition',
                'm4_productionreliability': 'M4_ProductionReliability',
                'm5_pushpullmodel': 'M5_PushPullModel',
                'm5_deployconfig': 'M5_DeployConfig',
                'm6_truckreleasecon': 'M6_TruckReleaseCon',
                'm6_materialmd': 'M6_MaterialMD',
                'm6_deliverydelaydistribution': 'M6_DeliveryDelayDistribution',
                'm6_mdqbypassrules': 'M6_MDQBypassRules',
                'm6_trucktypespecs': 'M6_TruckTypeSpecs',
                'm6_truckcapacityplan': 'M6_TruckCapacityPlan',
            }
            
            # 列名映射（数据库小写 -> 原始大小写）
            column_mapping = _get_column_mapping()
            
            for db_name, df in config_data.items():
                # 尝试映射到原始sheet名
                sheet_name = sheet_mapping.get(db_name.lower(), db_name)
                
                # 恢复列名大小写
                df_copy = df.copy()
                df_copy.columns = [column_mapping.get(col.lower(), col) for col in df_copy.columns]
                
                df_copy.to_excel(writer, sheet_name=sheet_name, index=False)
        
        logger.info(f"[LOG] 临时配置文件: {temp_config}")
        
        # 创建临时输出目录
        temp_output = temp_dir / "output"
        temp_output.mkdir(exist_ok=True)
        
        # 运行仿真
        result = run_integrated_simulation(
            config_path=str(temp_config),
            start_date=start_date,
            end_date=end_date,
            output_base_dir=str(temp_output),
            force_restart=True
        )
        
        if result and result.get('simulation_completed'):
            return {
                'success': True,
                'output_dir': str(temp_output)
            }
        else:
            return {'success': False}
            
    except Exception as e:
        logger.error(f"仿真执行出错: {e}")
        return {'success': False, 'error': str(e)}


def _cleanup_data_files(output_dir: str, log_dir: Path):
    """清理数据文件，只保留日志"""
    import shutil
    import gc
    import time
    
    output_path = Path(output_dir)
    
    # 复制日志文件到log_dir
    for log_file in output_path.glob("**/*.txt"):
        dest = log_dir / log_file.name
        shutil.copy2(log_file, dest)
        print(f"  📄 保存日志: {log_file.name}")
    
    for log_file in output_path.glob("**/*.log"):
        dest = log_dir / log_file.name
        shutil.copy2(log_file, dest)
        print(f"  📄 保存日志: {log_file.name}")
    
    # 强制垃圾回收，释放可能被 pandas 持有的文件句柄
    gc.collect()
    
    # 删除整个临时输出目录（带重试机制）
    temp_dir = output_path.parent
    max_retries = 3
    for attempt in range(max_retries):
        try:
            shutil.rmtree(temp_dir)
            print(f"  [DEL] 已清理临时数据目录")
            break
        except PermissionError as e:
            if attempt < max_retries - 1:
                # 等待一小段时间让文件句柄释放
                time.sleep(0.5)
                gc.collect()
            else:
                # 最后一次尝试失败，尝试逐个删除文件
                print(f"  [WARN] 临时目录清理延迟（文件可能被占用）: {temp_dir}")
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
            print(f"  [WARN] 清理临时目录失败: {e}")
            break


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="production_integrator",
        add_help=True,
        description=(
            "Run the integrated planning flow using a given configuration file. "
            "Outputs are written under a folder named after the configuration file. "
            "Supports automatic resume from interruption points with directory selection."
        ),
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the configuration .xlsx file",
    )
    parser.add_argument(
        "--start-date",
        required=False,
        help="Simulation start date in YYYY-MM-DD (required for first run)",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="Simulation end date in YYYY-MM-DD",
    )
    parser.add_argument(
        "--force-restart",
        action="store_true",
        help="Force restart from beginning, ignore resume capability",
    )
    parser.add_argument(
        "--check-resume",
        action="store_true",
        help="Check resume status only, do not execute simulation",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Enable automatic resume from interruption point",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Specific run directory to resume from (e.g., run_20241203_120000)",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List all available run directories and their status",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Disable interactive prompts (auto-select most recent run)",
    )
    parser.add_argument(
        "--use-db",
        action="store_true",
        help="使用数据库模式：从数据库读取配置，输出写入数据库，本地只保存运行日志txt",
    )
    parser.add_argument(
        "--db-host",
        type=str,
        default="localhost",
        help="数据库主机地址 (默认: localhost)",
    )
    parser.add_argument(
        "--db-port",
        type=int,
        default=5432,
        help="数据库端口 (默认: 5432)",
    )
    parser.add_argument(
        "--db-name",
        type=str,
        default="test_db",
        help="数据库名称 (默认: test_db)",
    )
    parser.add_argument(
        "--db-user",
        type=str,
        default="postgres",
        help="数据库用户名 (默认: postgres)",
    )
    parser.add_argument(
        "--db-password",
        type=str,
        default="123456",
        help="数据库密码 (默认: 123456)",
    )
    parser.add_argument(
        "--run-suffix",
        type=str,
        default="",
        help="运行目录后缀，用于区分不同运行 (例如: --run-suffix test 生成 run_YYYYMMDD_HHMMSS_test)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="数据库模式下启用本地文件输出（同时写入数据库和本地Excel文件）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ns = _parse_args(argv or sys.argv[1:])

    # ==================== 数据库模式 ====================
    if ns.use_db:
        return _run_with_database(ns)
    
    # ==================== 本地文件模式 ====================
    cfg_path = Path(ns.config).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
    if cfg_path.suffix.lower() not in {".xlsx", ".xlsm", ".xls"}:
        raise ValueError("Configuration file must be an Excel file (.xlsx/.xlsm/.xls)")

    # 尽早加载配置以便在出现结构/格式问题时快速失败
    #（该调用会返回可供运行函数使用的对象，或用于校验配置文件。）
    _ = load_configuration(str(cfg_path))  # noqa: F841

    # 为 --list-runs 提前获取根目录与日期参数
    cfg_stem = cfg_path.stem
    project_root = Path.cwd()
    root_dir = project_root / "outputs" / cfg_stem
    root_dir.mkdir(parents=True, exist_ok=True)
    
    start_arg = ns["start_date"] if isinstance(ns, dict) else ns.start_date
    simulation_start = get_or_init_simulation_start(root_dir, start_arg)
    end_date = str(ns["end_date"]) if isinstance(ns, dict) else ns.end_date

    # 处理 --list-runs 命令
    if ns.list_runs:
        print("\n" + "="*80)
        print("[INFO] 可用的运行目录列表")
        print("="*80)
        
        run_infos = _list_existing_runs(root_dir, simulation_start, end_date)
        
        if not run_infos:
            print("\n[ERROR] 未找到任何运行目录")
            print(f"   目录: {root_dir}")
            return 0
        
        for idx, info in enumerate(run_infos, 1):
            resume_info = info['resume_info']
            print(f"\n[{idx}] {info['name']}")
            print(f"    📂 路径: {info['path']}")
            
            if resume_info.get('already_completed', False):
                print(f"    [OK] 状态: 已完成")
                print(f"    📅 最后日期: {resume_info['last_complete_date']}")
                print(f"    [DATA] 完成天数: {resume_info['days_completed']}")
            elif resume_info['can_resume']:
                print(f"    🔄 状态: 可续跑")
                print(f"    📅 已完成: {resume_info['days_completed']} 天 (截至 {resume_info['last_complete_date']})")
                print(f"    📅 剩余: {resume_info['days_remaining']} 天 (从 {resume_info['resume_from_date']} 开始)")
            else:
                print(f"    [LOG] 状态: 无可续跑数据")
                print(f"    [DATA] 需处理: {resume_info['days_remaining']} 天")
        
        print("\n" + "="*80)
        print(f"[TIP] 续跑提示:")
        print(f"   python run.py --config {cfg_path} --end-date {end_date} --resume-from <run_dir_name>")
        print("="*80)
        return 0

    # 确定输出目录与续跑模式
    enable_resume = (ns.resume or ns.resume_from) and not ns.force_restart
    output_base_dir = _ensure_output_dir(
        cfg_path, 
        resume_mode=enable_resume,
        resume_from=ns.resume_from,
        start_date=simulation_start,
        end_date=end_date,
        interactive=not ns.non_interactive
    )

    # [NEW] 设置日志系统 - 同时输出到terminal和文件
    logger, redirector = setup_logging(str(output_base_dir), log_level="INFO", redirect_print=True)
    
    import time
    program_start_time = time.time()
    program_start_datetime = datetime.now()
    
    logger.info("\n" + "=" * 60)
    logger.info("🕐 程序时间信息")
    logger.info("=" * 60)
    logger.info(f"📅 程序开始时间: {program_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"[RUN] 供应链仿真系统启动")
    logger.info(f"📂 配置文件: {cfg_path}")
    logger.info(f"[DIR] 输出目录: {output_base_dir}")
    logger.info(f"📅 仿真日期范围: {simulation_start} 到 {end_date}")
    
    try:
        # 处理续跑状态检查
        if ns.check_resume:
            logger.info("[DEBUG] 检查续跑状态...")
            resume_info = check_resume_capability(str(output_base_dir), simulation_start, end_date)
            
            logger.info(f"\n[DATA] 续跑状态报告:")
            logger.info(f"  配置文件: {cfg_path}")
            logger.info(f"  输出目录: {output_base_dir}")
            logger.info(f"  日期范围: {simulation_start} 到 {end_date}")
            
            if resume_info.get('already_completed', False):
                logger.info(f"  [OK] 仿真已完成!")
                logger.info(f"     最后处理日期: {resume_info['last_complete_date']}")
                logger.info(f"     总处理天数: {resume_info['days_completed']}")
            elif resume_info['can_resume']:
                logger.info(f"  🔄 可以续跑!")
                logger.info(f"     已完成: {resume_info['days_completed']} 天 (截至 {resume_info['last_complete_date']})")
                logger.info(f"     剩余: {resume_info['days_remaining']} 天 (从 {resume_info['resume_from_date']} 开始)")
            else:
                logger.info(f"  [LOG] 无续跑能力，将从头开始")
                logger.info(f"     需处理天数: {resume_info['days_remaining']}")
            
            return 0

        # 将执行交由支持续跑能力的一体化仿真流程
        result = run_integrated_simulation(
            config_path=str(cfg_path),
            start_date=simulation_start,
            end_date=end_date,
            output_base_dir=str(output_base_dir),
            force_restart=ns.force_restart,
        )
        
        program_end_time = time.time()
        program_end_datetime = datetime.now()
        total_runtime = program_end_time - program_start_time
        
        # 格式化运行时间
        hours, remainder = divmod(total_runtime, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours >= 1:
            runtime_str = f"{int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒"
        elif minutes >= 1:
            runtime_str = f"{int(minutes)}分钟 {seconds:.2f}秒"
        else:
            runtime_str = f"{seconds:.2f}秒"
        
        logger.info("\n" + "=" * 60)
        if result and result.get('simulation_completed'):
            logger.info("[OK] 仿真成功完成")
        else:
            failure_stage = (
                result.get('failure_stage', 'unknown')
                if isinstance(result, dict) else 'unknown'
            )
            logger.warning(f"[WARN] 仿真未完成，已在阶段 {failure_stage} 停止")
            if isinstance(result, dict) and result.get('validation_report'):
                logger.warning(f"[WARN] 验证报告: {result['validation_report']}")
        logger.info("=" * 60)
        logger.info("🕐 程序时间统计:")
        logger.info(f"   📅 开始时间: {program_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"   📅 结束时间: {program_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"   [TIME]  总运行时间: {runtime_str}")
        logger.info("=" * 60)
        return 0
        
    except Exception as e:
        logger.error(f"[ERROR] 仿真执行出错: {str(e)}")
        raise
    finally:
        # 恢复原始输出
        if redirector:
            redirector.stop_redirect()
            print(f"[LOG] 完整日志已保存到: {output_base_dir}")  # 这条会显示在terminal


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit as e:
        raise e
    except Exception as exc:
        # 不使用 print；通过非零退出码与异常传播来表示失败
        #（调用方如有需要可捕获 stderr/traceback。）
        raise SystemExit(1) from exc

# ================================
# 运行示例 (Run Examples)
# ================================

# ==================== 模式一：本地文件模式 ====================
# 从本地Excel配置文件读取，输出保存到本地文件夹

# 1. 首次运行 (需要提供 --start-date)
# 首次运行需要提供 --start-date
# python run.py \
#   --config test_files/BC_S5.xlsx \
#   --start-date 2025-10-06 \
#   --end-date 2025-10-06

# 2. 强制从头开始
# 强制从头开始
# python run.py \
#   --config test_files/BC_S5.xlsx \
#   --start-date 2025-10-06 \
#   --end-date 2025-10-06 \
#   --force-restart

# 3. 列出所有可用的运行目录及其状态
# python run.py \
#   --config test_files/BC_S5.xlsx \
#   --end-date 2025-10-06 \
#   --list-runs

# 4. 自动续跑 (交互式选择目录)
# python run.py \
#   --config test_files/BC_S5.xlsx \
#   --end-date 2025-10-06 \
#   --resume

# ==================== 模式二：数据库模式 ====================
# 从数据库读取配置，输出写入数据库，本地只保存txt日志

# 1. 数据库模式运行（使用配置名称，非文件路径）
# python run.py \
#   --config BC_S5 \
#   --start-date 2025-10-06 \
#   --end-date 2025-10-06 \
#   --use-db

# 2. 指定数据库连接参数
# python run.py \
#   --config BC_S5 \
#   --start-date 2025-10-06 \
#   --end-date 2025-10-06 \
#   --use-db \
#   --db-host localhost \
#   --db-port 5432 \
#   --db-name test_db \
#   --db-user postgres \
#   --db-password 123456

# ==================== 两种模式对比 ====================
# 
# | 特性           | 本地文件模式                    | 数据库模式                      |
# |---------------|-------------------------------|-------------------------------|
# | 配置来源       | --config test_files/BC_S5.xlsx | --config BC_S5 --use-db       |
# | 输出位置       | 本地文件夹（xlsx/csv）          | PostgreSQL数据库               |
# | 本地保留       | 完整数据表和日志                 | 仅txt日志文件                   |
# | 日志目录       | outputs/BC_S5/run_xxx/        | outputs/db_BC_S5_xxx/         |
# | 适用场景       | 开发调试、本地验证               | 生产环境、数据分析              |
