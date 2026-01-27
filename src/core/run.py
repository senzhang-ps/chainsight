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
            print(f"    ✅ 状态: 已完成")
            print(f"    📅 最后日期: {resume_info['last_complete_date']}")
            print(f"    📊 完成天数: {resume_info['days_completed']}")
        elif resume_info['can_resume']:
            print(f"    🔄 状态: 可续跑")
            print(f"    📅 已完成: {resume_info['days_completed']} 天 (截至 {resume_info['last_complete_date']})")
            print(f"    📅 剩余: {resume_info['days_remaining']} 天 (从 {resume_info['resume_from_date']} 开始)")
        else:
            print(f"    📝 状态: 无可续跑数据")
            print(f"    📊 需处理: {resume_info['days_remaining']} 天")
    
    print("\n" + "="*80)
    print("请输入选项：")
    print("  - 输入数字 [1-{}] 选择对应目录".format(len(run_infos)))
    print("  - 输入 'n' 或 'new' 创建新的运行目录")
    print("  - 输入 'q' 或 'quit' 退出")
    
    while True:
        choice = input("\n👉 请选择: ").strip().lower()
        
        if choice in ['q', 'quit']:
            print("❌ 用户取消操作")
            sys.exit(0)
        
        if choice in ['n', 'new']:
            return None  # 用于指示创建新的目录
        
        try:
            idx = int(choice)
            if 1 <= idx <= len(run_infos):
                selected = run_infos[idx - 1]
                print(f"\n✅ 已选择: {selected['name']}")
                return selected['path']
            else:
                print(f"❌ 无效选择，请输入 1-{len(run_infos)} 之间的数字")
        except ValueError:
            print("❌ 无效输入，请输入数字、'n' 或 'q'")


def _ensure_output_dir(config_path: Path, resume_mode: bool = False, 
                      resume_from: Optional[str] = None, 
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      interactive: bool = True) -> Path:
    """Create the output directory rooted in centralized outputs/ folder.

    Structure:
      outputs/<config_stem>/
        └─ run_YYYYMMDD_HHMMSS/  (actual write target to avoid overwrites)

    Args:
        config_path: Path to the configuration file
        resume_mode: If True, allow resuming from existing run directories
        resume_from: Specific run directory name to resume from (e.g., "run_20241203_120000")
        start_date: Simulation start date (required for resume validation)
        end_date: Simulation end date (required for resume validation)
        interactive: If True, prompt user to select run directory when multiple exist

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
    print(f"📋 配置名称: {config_name}")
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
        print(f"❌ 无法导入数据库模块: {e}")
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
        print("❌ 数据库初始化失败")
        return 1
    
    # 获取数据库连接供后续使用
    db = initializer.db
    
    # 创建本地日志目录（只保存txt日志）
    project_root = Path(__file__).parent.parent.parent
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = project_root / "outputs" / f"db_{config_name}_{ts}"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 日志目录: {log_dir}")
    
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
            logger.error(f"❌ 未在数据库中找到配置: {config_name}")
            return 1
        
        logger.info(f"✅ 已加载 {len(config_data)} 个配置表")
        
        # ========== 步骤2: 使用DuckDB处理并运行仿真 ==========
        logger.info("\n" + "=" * 60)
        logger.info("🦆 步骤2: DuckDB处理配置并运行仿真")
        logger.info("=" * 60)
        
        # 创建临时输出目录
        import tempfile
        temp_dir = Path(tempfile.mkdtemp())
        temp_output = temp_dir / "output"
        temp_output.mkdir(exist_ok=True)
        
        # 🦆 使用优化版本运行仿真（如果可用）
        try:
            from pgsql_db.optimized_simulation import run_optimized_simulation_from_dict
            logger.info("🚀 使用高性能引擎运行仿真...")
            result = run_optimized_simulation_from_dict(
                config_data=config_data,
                config_name=config_name,
                start_date=start_date,
                end_date=end_date,
                output_base_dir=str(temp_output),
                skip_validation=True,
                enable_high_performance=True
            )
        except ImportError:
            logger.info("⚠️ 高性能引擎不可用，使用标准模式...")
            from .main_integration import run_integrated_simulation_from_dict
            result = run_integrated_simulation_from_dict(
                config_data=config_data,
                config_name=config_name,
                start_date=start_date,
                end_date=end_date,
                output_base_dir=str(temp_output),
                skip_validation=True
            )
        
        if not result or not result.get('simulation_completed'):
            logger.error("❌ 仿真运行失败")
            return 1
        
        # ========== 步骤3: 写入输出到数据库 ==========
        logger.info("\n" + "=" * 60)
        logger.info("📤 步骤3: 写入输出到数据库（完整模式）")
        logger.info("=" * 60)
        
        output_dir = result.get('output_directory')
        writer = ModuleDataWriter(db, config_name=config_name)  # 传入config_name参数
        run_id = f"{config_name}_{ts}"
        
        # 【完整模式】写入所有模块的详细输出数据 + Summary + Orchestrator
        # 包含: module1, module3, module4, module5, module6 的每日输出sheet
        db_write_start = time.time()
        if output_dir and Path(output_dir).exists():
            # 写入所有模块数据（包含每个module的每日输出sheet）
            writer.write_all_modules(str(output_dir), run_id=run_id, if_exists='replace')
            
            # 删除本地数据文件，只保留日志
            logger.info("\n🧹 清理本地数据文件（仅保留日志）...")
            _cleanup_data_files(str(output_dir), log_dir)
        
        db_write_time = time.time() - db_write_start
        logger.info(f"⏱️  数据库写入耗时: {db_write_time:.2f}秒")
        
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
        logger.info("✅ 数据库模式运行完成 (DuckDB)")
        logger.info("=" * 60)
        logger.info("🕐 程序时间统计:")
        logger.info(f"   📅 开始时间: {program_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"   📅 结束时间: {program_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"   ⏱️  总运行时间: {runtime_str}")
        logger.info(f"   ⏱️  其中DB写入: {db_write_time:.2f}秒")
        logger.info(f"   📁 日志目录: {log_dir}")
        logger.info("📊 数据存储:")
        logger.info(f"   💾 所有输出数据已写入 PostgreSQL 数据库")
        logger.info(f"   🔗 连接: postgresql://localhost:5432/test_db")
        logger.info(f"   📋 表前缀: {run_id}")
        logger.info("=" * 60)
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ 执行出错: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    finally:
        db.close()
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
            print(f"  ⚠️ 加载配置表失败 [{table_name}]: {e}")
            continue
        
        # 只接受包含 config_name 列的表；并按指定配置过滤
        if 'config_name' not in df.columns:
            continue
        
        filtered = df[df['config_name'] == config_name]
        if filtered.empty:
            continue
        
        # 去掉 cfg_ 前缀，作为配置数据的 key
        clean_table_name = table_name[4:]
        config_data[clean_table_name] = filtered
        print(f"  ✅ 加载配置表: {table_name} -> {clean_table_name} ({len(filtered)} 行)")
    
    # 如果新格式没有数据，回退到旧格式（兼容旧数据）
    if not config_data:
        print(f"  ℹ️ 未找到新格式配置表(cfg_*)，尝试旧格式({old_prefix}*)...")
        for table_name in all_tables:
            if not table_name.startswith(old_prefix):
                continue
                
            try:
                df = db.read_table(table_name)
            except Exception as e:
                print(f"  ⚠️ 加载配置表失败 [{table_name}]: {e}")
                continue
            
            if df.empty:
                continue
            
            # 去掉旧前缀，作为配置数据的 key
            clean_table_name = table_name[len(old_prefix):]
            config_data[clean_table_name] = df
            print(f"  ✅ 加载配置表(旧格式): {table_name} -> {clean_table_name} ({len(df)} 行)")
    
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
        
        logger.info(f"📝 临时配置文件: {temp_config}")
        
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
            print(f"  🗑️ 已清理临时数据目录")
            break
        except PermissionError as e:
            if attempt < max_retries - 1:
                # 等待一小段时间让文件句柄释放
                time.sleep(0.5)
                gc.collect()
            else:
                # 最后一次尝试失败，尝试逐个删除文件
                print(f"  ⚠️ 临时目录清理延迟（文件可能被占用）: {temp_dir}")
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
            print(f"  ⚠️ 清理临时目录失败: {e}")
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
        print("📋 可用的运行目录列表")
        print("="*80)
        
        run_infos = _list_existing_runs(root_dir, simulation_start, end_date)
        
        if not run_infos:
            print("\n❌ 未找到任何运行目录")
            print(f"   目录: {root_dir}")
            return 0
        
        for idx, info in enumerate(run_infos, 1):
            resume_info = info['resume_info']
            print(f"\n[{idx}] {info['name']}")
            print(f"    📂 路径: {info['path']}")
            
            if resume_info.get('already_completed', False):
                print(f"    ✅ 状态: 已完成")
                print(f"    📅 最后日期: {resume_info['last_complete_date']}")
                print(f"    📊 完成天数: {resume_info['days_completed']}")
            elif resume_info['can_resume']:
                print(f"    🔄 状态: 可续跑")
                print(f"    📅 已完成: {resume_info['days_completed']} 天 (截至 {resume_info['last_complete_date']})")
                print(f"    📅 剩余: {resume_info['days_remaining']} 天 (从 {resume_info['resume_from_date']} 开始)")
            else:
                print(f"    📝 状态: 无可续跑数据")
                print(f"    📊 需处理: {resume_info['days_remaining']} 天")
        
        print("\n" + "="*80)
        print(f"💡 续跑提示:")
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

    # 🆕 设置日志系统 - 同时输出到terminal和文件
    logger, redirector = setup_logging(str(output_base_dir), log_level="INFO", redirect_print=True)
    
    import time
    program_start_time = time.time()
    program_start_datetime = datetime.now()
    
    logger.info("\n" + "=" * 60)
    logger.info("🕐 程序时间信息")
    logger.info("=" * 60)
    logger.info(f"📅 程序开始时间: {program_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🚀 供应链仿真系统启动")
    logger.info(f"📂 配置文件: {cfg_path}")
    logger.info(f"📁 输出目录: {output_base_dir}")
    logger.info(f"📅 仿真日期范围: {simulation_start} 到 {end_date}")
    
    try:
        # 处理续跑状态检查
        if ns.check_resume:
            logger.info("🔍 检查续跑状态...")
            resume_info = check_resume_capability(str(output_base_dir), simulation_start, end_date)
            
            logger.info(f"\n📊 续跑状态报告:")
            logger.info(f"  配置文件: {cfg_path}")
            logger.info(f"  输出目录: {output_base_dir}")
            logger.info(f"  日期范围: {simulation_start} 到 {end_date}")
            
            if resume_info.get('already_completed', False):
                logger.info(f"  ✅ 仿真已完成!")
                logger.info(f"     最后处理日期: {resume_info['last_complete_date']}")
                logger.info(f"     总处理天数: {resume_info['days_completed']}")
            elif resume_info['can_resume']:
                logger.info(f"  🔄 可以续跑!")
                logger.info(f"     已完成: {resume_info['days_completed']} 天 (截至 {resume_info['last_complete_date']})")
                logger.info(f"     剩余: {resume_info['days_remaining']} 天 (从 {resume_info['resume_from_date']} 开始)")
            else:
                logger.info(f"  📝 无续跑能力，将从头开始")
                logger.info(f"     需处理天数: {resume_info['days_remaining']}")
            
            return 0

        # 将执行交由支持续跑能力的一体化仿真流程
        _ = run_integrated_simulation(
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
        logger.info("✅ 仿真成功完成")
        logger.info("=" * 60)
        logger.info("🕐 程序时间统计:")
        logger.info(f"   📅 开始时间: {program_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"   📅 结束时间: {program_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"   ⏱️  总运行时间: {runtime_str}")
        logger.info("=" * 60)
        return 0
        
    except Exception as e:
        logger.error(f"❌ 仿真执行出错: {str(e)}")
        raise
    finally:
        # 恢复原始输出
        if redirector:
            redirector.stop_redirect()
            print(f"📝 完整日志已保存到: {output_base_dir}")  # 这条会显示在terminal


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