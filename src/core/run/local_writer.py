"""本地文件写出工具。

职责：
- 将内存中的模块结果写出为本地 Excel、JSON 与 CSV 文件。
- 兼容标准简化输出与 Dev 对齐输出两种目录结构。
- 在需要时补充汇总报告与 Orchestrator 文件复制。
"""
from __future__ import annotations

import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pandas as pd


def _write_results_to_local(
    all_results: dict, output_dir: Path, logger
) -> None:
    """将内存中的仿真结果写入本地Excel文件（--also-local 模式使用）
    
    Args:
        all_results: 仿真结果字典
                     {module_name: [day1_result, day2_result, ...]}
        output_dir: 输出目录
        logger: 日志记录器
    """
    
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
                for key in [
                    data_key, f'{module_name}_{data_key}',
                    'output', 'result'
                ]:
                    if key in day_result and isinstance(
                        day_result[key], pd.DataFrame
                    ):
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
                module1_output_20251215.xlsx  (包含 OrderLog,
                    ShipmentLog, CutLog, SupplyDemandLog, Summary sheets)
                ...
            module3/
                Module3Output_20251215.xlsx  (包含 NetDemand sheet)
                ...
            module4/
                Module4Output_20251215.xlsx  (包含 ProductionPlan,
                    CapacityExceed, Validation, ChangeoverLog sheets)
                allocated_capacity_20251216.json
                line_states_20251216.json
                ...
            module5/
                Module5Output_20251215.xlsx  (包含 DeploymentPlan,
                    UnfulfilledLog, StockOnHandLog, Validation sheets)
                ...
            module6/
                Module6Output_20251215.xlsx  (包含 DeliveryPlan,
                    VehicleLog, TruckUsageLog, etc sheets)
                ...
            orchestrator/
                (CSV files from orchestrator)
            summary/
                full_production_plan_report.xlsx
                full_order_shipment_cut_report.xlsx
                ...
    
    Args:
        all_results: 仿真结果字典
                     {module_name: [day1_result, day2_result, ...]}
        output_dir: 输出目录
        logger: 日志记录器
        config_dict: 配置字典（用于生成summary报告）
        start_date: 仿真开始日期
        end_date: 仿真结束日期
        orchestrator_output_dir: Orchestrator临时输出目录（用于复制CSV）
    """
    
    write_start = time.time()
    
    # 创建模块输出目录
    module_dirs = {}
    for module_name in [
        'module1', 'module3', 'module4', 'module5', 'module6'
    ]:
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
                write_tasks.append((
                    'excel_multi_sheet', file_path, sheets_to_write
                ))
            
            # 模块4特殊处理：line_states 和 allocated_capacity
            # (JSON格式)
            if module_name == 'module4':
                line_states = day_result.get('line_states')
                if line_states is not None:
                    json_path = module_dir / f'line_states_{date_str}.json'
                    write_tasks.append(('json', json_path, line_states))
                
                allocated_capacity = day_result.get('allocated_capacity')
                if allocated_capacity is not None:
                    json_path = (
                        module_dir / f'allocated_capacity_{date_str}.json'
                    )
                    write_tasks.append(('json', json_path, allocated_capacity))
    
    # 使用线程池并行写入文件
    def write_file(task):
        file_type, file_path, data = task
        try:
            if file_type == 'excel_multi_sheet':
                # 写入多个Sheet的Excel文件
                if isinstance(data, dict) and data:
                    total_rows = 0
                    with pd.ExcelWriter(
                        file_path, engine='openpyxl'
                    ) as writer:
                        for sheet_name, df in data.items():
                            if isinstance(df, pd.DataFrame):
                                df.to_excel(
                                    writer, sheet_name=sheet_name, index=False
                                )
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
        futures = {
            executor.submit(write_file, task): task for task in write_tasks
        }
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
            logger.info(
                f"  [OK] orchestrator: 复制 "
                f"{len(list(orch_src.glob('*.csv')))} 个CSV文件"
            )
    
    # 生成Summary报告
    if config_dict and start_date and end_date:
        try:
            from ..main_integration import SummaryReportGenerator
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
            logger.info(
                f"  [OK] summary: 生成 {len(summary_reports)} 个汇总报告"
            )
        except Exception as e:
            logger.warning(f"  [WARN] Summary报告生成失败: {e}")
    
    write_time = time.time() - write_start
    logger.info(
        f"  [TIME] 本地文件写入耗时: {write_time:.2f}秒 "
        f"({success_count} 文件, {total_rows} 行)"
    )
    if error_count > 0:
        logger.warning(f"  [WARN] {error_count} 个文件写入失败")
