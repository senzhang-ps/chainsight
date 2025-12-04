# main_integration.py
# 主集成执行脚本 - 协调所有模块通过Orchestrator运行
#
# 执行顺序: M1 → M4 → M5 → M6 → M3
# 每日处理模式，确保模块间数据流环环相扣

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import os
from pandas.errors import EmptyDataError, ParserError
import logging

import orchestrator

logger = logging.getLogger(__name__)

# 导入所有模块
from orchestrator import create_orchestrator
from validation_manager import ValidationManager
from time_manager import SimulationTimeManager, initialize_time_manager
from config_validator import run_pre_simulation_validation
from inventory_balance_checker import InventoryBalanceChecker
from summary_report_generator import SummaryReportGenerator
from performance_profiler import PerformanceProfiler
import module1
import module3
import module4
import module5
import module6


# ========================= 断点续跑功能 =========================

def detect_last_complete_date(output_base_dir: str, start_date: str, end_date: str) -> str:
    """
    检测最后一个完整处理的日期
    Args:
        output_base_dir: 输出基础目录
        start_date: 原始开始日期
        end_date: 原始结束日期
        
    Returns:
        str: 最后完整处理的日期(YYYY-MM-DD)，如果没有则返回None
    """
    print(f"🔍 检测中断点...")
    
    output_dir = Path(output_base_dir)
    orchestrator_dir = output_dir / "orchestrator"
    
    if not orchestrator_dir.exists():
        print(f"  📁 输出目录不存在，将从头开始: {orchestrator_dir}")
        return None
    
    # 生成日期范围
    date_range = pd.date_range(start_date, end_date, freq='D')
    
    last_complete_date = None
    
    for current_date in date_range:
        date_str = current_date.strftime('%Y%m%d')
        
        # 检查关键状态文件是否都存在
        # 项目约定的完整性视图（见 .github/copilot-instructions.md §6）
        # 新增 daily_logs 作为必须存在的 daily summary 日志
        required_files = [
            f"unrestricted_inventory_{date_str}.csv",
            f"open_deployment_{date_str}.csv",
            f"planning_intransit_{date_str}.csv",
            f"space_quota_{date_str}.csv",
            f"delivery_gr_{date_str}.csv",
            f"production_gr_{date_str}.csv",
            f"shipment_log_{date_str}.csv",
            f"delivery_shipment_log_{date_str}.csv",
            f"inventory_change_log_{date_str}.csv",
            f"daily_logs_{date_str}.csv"
        ]
        
        # 轻量验证：检查文件存在并能被读取（只读表头以避免加载大型文件；表头为空也视为可接受）
        all_files_exist = True
        for file_name in required_files:
            file_path = orchestrator_dir / file_name

            try:
                if not file_path.exists():
                    logger.warning("缺失文件: %s", file_path)
                    all_files_exist = False
                    break

                # 使用 nrows=0 只读表头（即便没有数据也不会尝试读取行）
                # 这样"只有表头"或"无数据"不会中止续跑判断
                pd.read_csv(file_path, nrows=0, encoding="utf-8")

            except EmptyDataError:
                # 有些日期文件只有表头或无数据，这是允许的（记录但不视为致命）
                logger.info("CSV 只有表头或无数据（EmptyDataError，但可接受）: %s", file_path)
                # 继续检查下一个文件
                continue
            except (UnicodeDecodeError, ParserError) as e:
                logger.warning("CSV 解码/解析失败: %s -> %s", file_path, e)
                all_files_exist = False
                break
            except OSError as e:
                logger.error("文件访问错误: %s -> %s", file_path, e)
                all_files_exist = False
                break
            except Exception as e:
                logger.exception("未知错误读取文件 %s", file_path)
                all_files_exist = False
                break
                
        if all_files_exist:
            # 验证文件不为空
            try:
                inventory_file = orchestrator_dir / f"unrestricted_inventory_{date_str}.csv"
                df = pd.read_csv(inventory_file)
                if len(df) >= 0:  # 允许空库存，但文件格式要正确
                    last_complete_date = current_date.strftime('%Y-%m-%d')
                    print(f"  ✅ 发现完整日期: {last_complete_date}")
                else:
                    break
            except Exception as e:
                print(f"  ⚠️  日期 {current_date.strftime('%Y-%m-%d')} 文件损坏: {e}")
                break
        else:
            print(f"  ❌ 日期 {current_date.strftime('%Y-%m-%d')} 文件不完整")
            break
    
    if last_complete_date:
        print(f"  🎯 检测到最后完整日期: {last_complete_date}")
    else:
        print(f"  📝 未发现完整日期，将从头开始")
        
    return last_complete_date

def restore_orchestrator_state(orchestrator, restore_date: str, output_base_dir: str):
    """
    从指定日期的状态文件恢复Orchestrator状态
    
    Args:
        orchestrator: Orchestrator实例
        restore_date: 恢复日期 (YYYY-MM-DD)
        output_base_dir: 输出基础目录
    """
    print(f"🔄 从日期 {restore_date} 恢复Orchestrator状态...")
    
    output_dir = Path(output_base_dir)
    orchestrator_dir = output_dir / "orchestrator"
    date_str = pd.to_datetime(restore_date).strftime('%Y%m%d')
    
    # 可调整的日志回溯天数（默认14天）
    log_lookback_days = 14
    
    try:
        # 1. 恢复无限制库存
        inventory_file = orchestrator_dir / f"unrestricted_inventory_{date_str}.csv"
        if inventory_file.exists():
            try:
                inventory_df = pd.read_csv(inventory_file, dtype=object)
            except EmptyDataError:
                inventory_df = pd.DataFrame()
            inventory_df = _normalize_identifiers(inventory_df) if isinstance(inventory_df, pd.DataFrame) and not inventory_df.empty else pd.DataFrame()
            # 重建库存字典
            orchestrator.unrestricted_inventory = {}
            for _, row in inventory_df.iterrows():
                mat = str(row.get('material', '')).strip()
                loc = str(row.get('location', '')).strip()
                key = (mat, loc)
                try:
                    qty = float(row.get('quantity', 0)) if pd.notna(row.get('quantity', 0)) else 0.0
                except Exception:
                    try:
                        qty = float(str(row.get('quantity', 0)).strip())
                    except Exception:
                        qty = 0.0
                orchestrator.unrestricted_inventory[key] = qty
            print(f"  ✅ 恢复库存记录: {len(inventory_df)} 条")
        else:
            orchestrator.unrestricted_inventory = {}
        
        # 2. 恢复在途库存 (MUST rebuild as in_transit dictionary with UID keys)
        intransit_file = orchestrator_dir / f"planning_intransit_{date_str}.csv"
        if intransit_file.exists():
            try:
                intransit_df = pd.read_csv(intransit_file, dtype=object)
            except EmptyDataError:
                intransit_df = pd.DataFrame()
            if not intransit_df.empty:
                intransit_df = _normalize_identifiers(intransit_df)
                # Rebuild in_transit dictionary: transit_uid -> transit_record
                orchestrator.in_transit = {}
                for _, row in intransit_df.iterrows():
                    transit_uid = row.get('transit_uid')
                    if transit_uid is not None and str(transit_uid).strip() and str(transit_uid) != 'None':
                        uid_str = str(transit_uid)
                        # Safely convert quantity to int
                        try:
                            quantity = int(float(row.get('quantity', 0) or 0))
                        except (ValueError, TypeError):
                            quantity = 0
                        
                        orchestrator.in_transit[uid_str] = {
                            'material': str(row.get('material', '')),
                            'sending': str(row.get('sending', '')),
                            'receiving': str(row.get('receiving', '')),
                            'actual_ship_date': str(row.get('actual_ship_date', '')),
                            'actual_delivery_date': str(row.get('actual_delivery_date', '')),
                            'quantity': quantity,
                            'ori_deployment_uid': str(row.get('ori_deployment_uid', '')),
                            'vehicle_uid': str(row.get('vehicle_uid', ''))
                        }
            else:
                orchestrator.in_transit = {}
            print(f"  ✅ 恢复在途记录: {len(orchestrator.in_transit)} 条")
        else:
            orchestrator.in_transit = {}
        
        # 3. 恢复开放调拨 (MUST be a dict with UID keys, not a list)
        deployment_file = orchestrator_dir / f"open_deployment_{date_str}.csv"
        if deployment_file.exists():
            try:
                deployment_df = pd.read_csv(deployment_file, dtype=object)
            except EmptyDataError:
                deployment_df = pd.DataFrame()
            if not deployment_df.empty:
                deployment_df = _normalize_identifiers(deployment_df)
                # Rebuild as dictionary: uid -> deployment_record
                orchestrator.open_deployment = {}
                for _, row in deployment_df.iterrows():
                    uid = row.get('ori_deployment_uid')
                    if uid is not None and str(uid).strip() and str(uid) != 'None':
                        uid_str = str(uid)
                        # Safely convert deployed_qty to int
                        try:
                            deployed_qty = int(float(row.get('deployed_qty', 0) or 0))
                        except (ValueError, TypeError):
                            deployed_qty = 0
                        
                        orchestrator.open_deployment[uid_str] = {
                            'material': str(row.get('material', '')),
                            'sending': str(row.get('sending', '')),
                            'receiving': str(row.get('receiving', '')),
                            'planned_deployment_date': str(row.get('planned_deployment_date', '')),
                            'deployed_qty': deployed_qty,
                            'demand_element': str(row.get('demand_element', ''))
                        }
            else:
                orchestrator.open_deployment = {}
            print(f"  ✅ 恢复调拨记录: {len(orchestrator.open_deployment)} 条")
        else:
            orchestrator.open_deployment = {}
        
        # 4. 恢复空间配额
        space_file = orchestrator_dir / f"space_quota_{date_str}.csv"
        if space_file.exists():
            try:
                space_df = pd.read_csv(space_file, dtype=object)
            except EmptyDataError:
                space_df = pd.DataFrame()
            if not space_df.empty:
                space_df = _normalize_identifiers(space_df)
                orchestrator.space_quota = {}
                for _, row in space_df.iterrows():
                    key = str(row.get('location', '')).strip()
                    try:
                        used = float(row.get('used_capacity', 0) or 0)
                    except Exception:
                        used = 0.0
                    try:
                        total = float(row.get('total_capacity', 0) or 0)
                    except Exception:
                        total = 0.0
                    orchestrator.space_quota[key] = {'used': used, 'total': total}
            else:
                orchestrator.space_quota = {}
            print(f"  ✅ 恢复空间配额: {len(space_df)} 条")
        else:
            orchestrator.space_quota = {}
        
        # 5. 恢复生产计划backlog (future production)
        production_backlog_file = orchestrator_dir / f"production_plan_backlog_{date_str}.csv"
        if production_backlog_file.exists():
            try:
                backlog_df = pd.read_csv(production_backlog_file, dtype=object)
            except EmptyDataError:
                backlog_df = pd.DataFrame()
            if not backlog_df.empty:
                backlog_df = _normalize_identifiers(backlog_df)
                # Convert quantity to int
                if 'quantity' in backlog_df.columns:
                    backlog_df['quantity'] = pd.to_numeric(backlog_df['quantity'], errors='coerce').fillna(0).astype(int)
                # Convert available_date to datetime to match original structure
                if 'available_date' in backlog_df.columns:
                    backlog_df['available_date'] = pd.to_datetime(backlog_df['available_date']).dt.normalize()
                orchestrator.production_plan_backlog = backlog_df.to_dict('records')
            else:
                orchestrator.production_plan_backlog = []
            print(f"  ✅ 恢复生产计划backlog: {len(orchestrator.production_plan_backlog)} 条")
        else:
            orchestrator.production_plan_backlog = []
        
        # 6. 恢复历史日志（近期的部分） - 可配置回溯天数
        restore_date_obj = pd.to_datetime(restore_date)
        log_start_date = restore_date_obj - pd.Timedelta(days=log_lookback_days)
        
        orchestrator.shipment_log = []
        orchestrator.production_gr = []
        orchestrator.delivery_gr = []
        orchestrator.delivery_shipment_log = []
        orchestrator.inventory_change_log = []
        orchestrator.daily_logs = []
        
        current_scan_date = log_start_date
        while current_scan_date <= restore_date_obj:
            scan_date_str = current_scan_date.strftime('%Y%m%d')
            
            # 恢复发货日志
            shipment_file = orchestrator_dir / f"shipment_log_{scan_date_str}.csv"
            if shipment_file.exists():
                try:
                    shipment_df = pd.read_csv(shipment_file, dtype=object)
                except EmptyDataError:
                    shipment_df = pd.DataFrame()
                if not shipment_df.empty:
                    shipment_df = _normalize_identifiers(shipment_df)
                    orchestrator.shipment_log.extend(shipment_df.to_dict('records'))
            
            # 恢复生产入库日志
            production_file = orchestrator_dir / f"production_gr_{scan_date_str}.csv"
            if production_file.exists():
                try:
                    production_df = pd.read_csv(production_file, dtype=object)
                except EmptyDataError:
                    production_df = pd.DataFrame()
                if not production_df.empty:
                    production_df = _normalize_identifiers(production_df)
                    orchestrator.production_gr.extend(production_df.to_dict('records'))
            
            # 恢复收货日志
            delivery_file = orchestrator_dir / f"delivery_gr_{scan_date_str}.csv"
            if delivery_file.exists():
                try:
                    delivery_df = pd.read_csv(delivery_file, dtype=object)
                except EmptyDataError:
                    delivery_df = pd.DataFrame()
                if not delivery_df.empty:
                    delivery_df = _normalize_identifiers(delivery_df)
                    orchestrator.delivery_gr.extend(delivery_df.to_dict('records'))
            
            # 恢复站点间发运日志 (delivery_shipment_log)
            dship_file = orchestrator_dir / f"delivery_shipment_log_{scan_date_str}.csv"
            if dship_file.exists():
                try:
                    dship_df = pd.read_csv(dship_file, dtype=object)
                except EmptyDataError:
                    dship_df = pd.DataFrame()
                if not dship_df.empty:
                    dship_df = _normalize_identifiers(dship_df)
                    orchestrator.delivery_shipment_log.extend(dship_df.to_dict('records'))
            
            # 恢复库存变动日志 (inventory_change_log)
            invchg_file = orchestrator_dir / f"inventory_change_log_{scan_date_str}.csv"
            if invchg_file.exists():
                try:
                    invchg_df = pd.read_csv(invchg_file, dtype=object)
                except EmptyDataError:
                    invchg_df = pd.DataFrame()
                if not invchg_df.empty:
                    invchg_df = _normalize_identifiers(invchg_df)
                    orchestrator.inventory_change_log.extend(invchg_df.to_dict('records'))
            
            # 恢复 daily_logs（汇总日志）
            daily_file = orchestrator_dir / f"daily_logs_{scan_date_str}.csv"
            if daily_file.exists():
                try:
                    daily_df = pd.read_csv(daily_file, dtype=object)
                except EmptyDataError:
                    daily_df = pd.DataFrame()
                if not daily_df.empty:
                    # daily_logs 可能不含标准标识符列，但调用normalize不会有害
                    daily_df = _normalize_identifiers(daily_df)
                    orchestrator.daily_logs.extend(daily_df.to_dict('records'))
            
            current_scan_date += pd.Timedelta(days=1)
        
        print(f"  ✅ 恢复发货日志: {len(orchestrator.shipment_log)} 条")
        print(f"  ✅ 恢复生产日志: {len(orchestrator.production_gr)} 条")
        print(f"  ✅ 恢复收货日志: {len(orchestrator.delivery_gr)} 条")
        print(f"  ✅ 恢复站点间发运日志: {len(orchestrator.delivery_shipment_log)} 条")
        print(f"  ✅ 恢复库存变动日志: {len(orchestrator.inventory_change_log)} 条")
        print(f"  ✅ 恢复daily_logs: {len(orchestrator.daily_logs)} 条")
        
        # 6. 重建date-indexed dictionaries for Phase 6 optimization
        print(f"  🔧 重建日期索引字典...")
        orchestrator.production_gr_by_date = {}
        orchestrator.delivery_gr_by_date = {}
        orchestrator.shipment_log_by_date = {}
        orchestrator.delivery_shipment_log_by_date = {}
        
        # Index production_gr
        for record in orchestrator.production_gr:
            date_key = record.get('date', '')
            if date_key not in orchestrator.production_gr_by_date:
                orchestrator.production_gr_by_date[date_key] = []
            orchestrator.production_gr_by_date[date_key].append(record)
        
        # Index delivery_gr
        for record in orchestrator.delivery_gr:
            date_key = record.get('date', '')
            if date_key not in orchestrator.delivery_gr_by_date:
                orchestrator.delivery_gr_by_date[date_key] = []
            orchestrator.delivery_gr_by_date[date_key].append(record)
        
        # Index shipment_log
        for record in orchestrator.shipment_log:
            date_key = record.get('date', '')
            if date_key not in orchestrator.shipment_log_by_date:
                orchestrator.shipment_log_by_date[date_key] = []
            orchestrator.shipment_log_by_date[date_key].append(record)
        
        # Index delivery_shipment_log
        for record in orchestrator.delivery_shipment_log:
            date_key = record.get('date', '')
            if date_key not in orchestrator.delivery_shipment_log_by_date:
                orchestrator.delivery_shipment_log_by_date[date_key] = []
            orchestrator.delivery_shipment_log_by_date[date_key].append(record)
        
        print(f"  ✅ 日期索引重建完成: production_gr={len(orchestrator.production_gr_by_date)} 天, "
              f"delivery_gr={len(orchestrator.delivery_gr_by_date)} 天, "
              f"shipment_log={len(orchestrator.shipment_log_by_date)} 天, "
              f"delivery_shipment_log={len(orchestrator.delivery_shipment_log_by_date)} 天")
        
        # 7. 设置当前日期
        orchestrator.current_date = restore_date_obj
        
        print(f"  🎯 Orchestrator状态恢复完成")
        
    except Exception as e:
        print(f"  ❌ 状态恢复失败: {e}")
        raise

def check_resume_capability(output_base_dir: str, start_date: str, end_date: str):
    """
    检查是否可以续跑，返回续跑信息
    
    Returns:
        dict: {
            'can_resume': bool,
            'last_complete_date': str,  
            'resume_from_date': str,
            'days_completed': int,
            'days_remaining': int
        }
    """
    last_complete_date = detect_last_complete_date(output_base_dir, start_date, end_date)
    
    if last_complete_date is None:
        return {
            'can_resume': False,
            'last_complete_date': None,
            'resume_from_date': start_date,
            'days_completed': 0,
            'days_remaining': len(pd.date_range(start_date, end_date, freq='D'))
        }
    
    # 计算续跑信息
    last_date_obj = pd.to_datetime(last_complete_date)
    resume_from_date = (last_date_obj + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    
    total_dates = pd.date_range(start_date, end_date, freq='D')
    completed_dates = pd.date_range(start_date, last_complete_date, freq='D')
    
    # 检查是否已经全部完成
    if last_complete_date >= end_date:
        return {
            'can_resume': False,
            'last_complete_date': last_complete_date,
            'resume_from_date': None,
            'days_completed': len(completed_dates),
            'days_remaining': 0,
            'already_completed': True
        }
    
    remaining_dates = pd.date_range(resume_from_date, end_date, freq='D')
    
    return {
        'can_resume': True,
        'last_complete_date': last_complete_date,
        'resume_from_date': resume_from_date,
        'days_completed': len(completed_dates),
        'days_remaining': len(remaining_dates)
    }

# ========================= 原有函数 =========================

# 标识符字段标准化函数（统一处理所有配置表）
def _normalize_location(location_str) -> str:
    """Normalize location string by padding with leading zeros to 4 digits if numeric"""
    if pd.isna(location_str) or location_str is None:
        return ""
    
    location_str = str(location_str).strip()
    
    try:
        # 检查是否为纯数字字符串
        if location_str.isdigit():
            return str(int(location_str)).zfill(4)
        else:
            # 非数字location（如A888），直接返回字符串，不做padding
            return location_str
    except (ValueError, TypeError):
        return str(location_str)

def _normalize_material(material_str) -> str:
    """Normalize material string to ensure consistent format"""
    if material_str is None or material_str == '' or str(material_str).lower() in ['nan', 'none', '<na>']:
        return ""
    
    try:
        # 如果是数字（int或float），转换为整数字符串以移除多余的.0
        if isinstance(material_str, (int, float)) or str(material_str).replace('.', '').replace('-', '').isdigit():
            return str(int(float(material_str)))
        else:
            # 非数字material，直接返回字符串
            return str(material_str).strip()
    except (ValueError, TypeError):
        # 如果转换失败，直接返回字符串
        return str(material_str).strip()

def _normalize_sending(sending_str) -> str:
    """Normalize sending string by padding with leading zeros to 4 digits if numeric"""
    if pd.isna(sending_str) or sending_str is None:
        return ""
    
    sending_str = str(sending_str).strip()
    
    try:
        # 检查是否为纯数字字符串
        if sending_str.isdigit():
            return str(int(sending_str)).zfill(4)
        else:
            # 非数字sending（如A888），直接返回字符串，不做padding
            return sending_str
    except (ValueError, TypeError):
        return str(sending_str)

def _normalize_receiving(receiving_str) -> str:
    """Normalize receiving string by padding with leading zeros to 4 digits if numeric"""
    if pd.isna(receiving_str) or receiving_str is None:
        return ""
    
    receiving_str = str(receiving_str).strip()
    
    try:
        # 检查是否为纯数字字符串
        if receiving_str.isdigit():
            return str(int(receiving_str)).zfill(4)
        else:
            # 非数字receiving（如A888），直接返回字符串，不做padding
            return receiving_str
    except (ValueError, TypeError):
        return str(receiving_str)

def _normalize_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize identifier columns to string format with proper formatting"""
    if df.empty:
        return df
    
    # Define identifier columns that need string conversion
    identifier_cols = ['material', 'location', 'sending', 'receiving', 'sourcing', 'dps_location', 'from_material', 'to_material', 'line', 'delegate_line', 'changeover_id']
    
    df = df.copy()
    for col in identifier_cols:
        if col in df.columns:
            # 🔧 关键修复：使用 object dtype (Python str) 而不是 pandas StringDtype
            # 这样可以确保与后续 astype(str) 的一致性
            df[col] = df[col].astype(str)
            # Apply specific normalization for location-type fields
            if col in ['location', 'dps_location']:
                df[col] = df[col].apply(_normalize_location)
            elif col == 'sending':
                df[col] = df[col].apply(_normalize_sending)
            elif col == 'receiving':
                df[col] = df[col].apply(_normalize_receiving)
            # Apply specific normalization for material-type fields
            elif col in ['material', 'from_material', 'to_material']:
                df[col] = df[col].apply(_normalize_material)
            # changeover_id 和 line 只需要转换为字符串，不需要特殊格式化
            # (已在 astype('string') 时处理)
            # For other identifier columns (line, delegate_line, etc), ensure they are properly formatted strings
            elif col in ['changeover_id', 'line', 'delegate_line']:
                # 这些字段只需要保持为字符串，不需要额外处理
                pass
            else:
                df[col] = df[col].apply(lambda x: str(x) if pd.notna(x) else "")
    
    return df

def run_module4_integrated(
    config_dict: dict,
    module3_output_dir: str,
    simulation_date: pd.Timestamp,
    simulation_start: pd.Timestamp,
    output_dir: str
) -> pd.DataFrame:
    """
    集成模式运行 Module4 生产计划，直接使用 config_dict 数据
    
    Args:
        config_dict: 配置数据字典
        module3_output_dir: Module3 输出目录
        simulation_date: 当前仿真日期
        simulation_start: 仿真开始日期
        output_dir: 输出目录
        
    Returns:
        pd.DataFrame: 生产计划数据
    """
    try:
        # 验证必需的Module4配置数据
        required_m4_configs = [
            'M4_MaterialLocationLineCfg',
            'M4_LineCapacity', 
            'M4_ChangeoverMatrix',
            'M4_ChangeoverDefinition',
            'M4_ProductionReliability'
        ]
        
        for config_name in required_m4_configs:
            if config_name not in config_dict or config_dict[config_name].empty:
                raise ValueError(f"缺少必需的Module4配置数据：{config_name}")
        
        # 直接构建 Module4 所需的配置数据
        # 直接使用config_dict，不再需要子配置字典
        m4_config = config_dict
        
        # 加载 Module3 的日度净需求数据
        net_demand_df = module4.load_daily_net_demand(module3_output_dir, simulation_date)
        net_demand_df = module4._cast_identifiers_to_str(net_demand_df, ['material', 'location'])
        
        # 🔧 修复Module3→Module4数据流：标准化material字段，移除.0后缀
        if not net_demand_df.empty and 'material' in net_demand_df.columns:
            net_demand_df['material'] = net_demand_df['material'].apply(_normalize_material).astype('string')
        
        if net_demand_df.empty:
            print(f"Warning: No NetDemand data for {simulation_date.strftime('%Y-%m-%d')}. Generating empty output.")
        
        # 确保 requirement_date 是 datetime 类型
        if not net_demand_df.empty and 'requirement_date' in net_demand_df.columns:
            net_demand_df['requirement_date'] = pd.to_datetime(net_demand_df['requirement_date'])
        
        # 构建无约束计划
        mlcfg = m4_config['M4_MaterialLocationLineCfg']
        
        # 确保MLCFG也应用类型转换（与NetDemand保持一致）
        mlcfg = module4._cast_identifiers_to_str(mlcfg.copy(), ['material', 'location'])
        
        issues = []
        uncon_plan = module4.build_unconstrained_plan_for_single_day(
            net_demand_df, mlcfg, simulation_date, simulation_start, issues
        )
        
        # 🔧 关键修复：标准化uncon_plan中的material字段，确保与changeover matrix一致
        if not uncon_plan.empty and 'material' in uncon_plan.columns:
            # print(f"\n🔍 DEBUG uncon_plan 标准化前:")
            # print(f"  material dtype: {uncon_plan['material'].dtype}")
            # print(f"  前5个 material: {list(uncon_plan['material'].head())}")
            
            uncon_plan['material'] = uncon_plan['material'].apply(_normalize_material).astype('string')
            
            # print(f"\n  标准化后:")
            # print(f"  material dtype: {uncon_plan['material'].dtype}")
            # print(f"  前5个 material: {list(uncon_plan['material'].head())}")
            # print(f"  Line 列: {list(uncon_plan['line'].unique())}")
        
        # 设置产能分配参数
        # 🔧 关键修复：标准化 ChangeoverMatrix 中的字段为字符串类型
        co_mat_df = m4_config['M4_ChangeoverMatrix'].copy()
        
        # 🔍 调试：显示原始数据类型
        # print(f"\n🔍 DEBUG M4 ChangeoverMatrix 数据类型:")
        # print(f"  原始 from_material dtype: {co_mat_df['from_material'].dtype}")
        # print(f"  原始 to_material dtype: {co_mat_df['to_material'].dtype}")
        # print(f"  原始 changeover_id dtype: {co_mat_df['changeover_id'].dtype}")
        # print(f"  前5条记录:")
        # print(co_mat_df.head())
        
        co_mat_df['from_material'] = co_mat_df['from_material'].astype(str)
        co_mat_df['to_material'] = co_mat_df['to_material'].astype(str)
        co_mat_df['changeover_id'] = co_mat_df['changeover_id'].astype(str)
        
        # print(f"\n  转换后 from_material dtype: {co_mat_df['from_material'].dtype}")
        # print(f"  转换后 to_material dtype: {co_mat_df['to_material'].dtype}")
        # print(f"  转换后 changeover_id dtype: {co_mat_df['changeover_id'].dtype}")
        # print(f"  转换后前5条记录:")
        # print(co_mat_df.head())
        
        # Note: Changeover 去重已在 load_configuration 中完成
        
        co_mat = co_mat_df.set_index(['from_material', 'to_material'])['changeover_id']
        # 对MultiIndex进行排序以避免性能警告
        co_mat = co_mat.sort_index()
        
        # print(f"\n  Co_mat 索引类型: {co_mat.index.dtypes}")
        # print(f"  Co_mat 总条目数: {len(co_mat)}")
        # print(f"  前5个索引: {list(co_mat.index[:5])}")
        
        # 🔧 关键修复：标准化 ChangeoverDefinition 中的 changeover_id 为字符串类型
        co_def_df = m4_config['M4_ChangeoverDefinition'].copy()
        co_def_df['changeover_id'] = co_def_df['changeover_id'].astype(str)
        co_def = co_def_df.set_index(['changeover_id', 'line'])['time'].to_dict()
        
        cap_df = m4_config['M4_LineCapacity'].copy()
        cap_df['date'] = pd.to_datetime(cap_df['date'])
        
        rate_map = mlcfg.set_index(['material', 'delegate_line'])['prd_rate']
        rate_map.index.set_names(['material', 'line'], inplace=True)
        
        # 加载前一天产线状态用于跨天转产连续性
        previous_line_states = module4.load_line_state(output_dir, simulation_date)
        if previous_line_states:
            print(f"  🔄 加载前一天产线状态: {list(previous_line_states.keys())}")
        else:
            print(f"  🔄 无前一天产线状态 - 全新开始")
        
        # 加载之前所有仿真日期已分配的产能
        previously_allocated_capacity = module4.load_all_previous_capacity(output_dir, simulation_date)
        if previously_allocated_capacity:
            print(f"  🔄 加载之前已分配产能: {len(previously_allocated_capacity)} 个产能分配")
        else:
            print(f"  🔄 无之前已分配产能 - 全新开始")
        
        # 分配产能（支持跨天转产连续性和产能跟踪）
        plan_log, exceed_log = module4.centralized_capacity_allocation_with_changeover(
            uncon_plan, cap_df, rate_map, co_mat, co_def, mlcfg,
            previous_line_states=previous_line_states, simulation_date=simulation_date,
            previously_allocated_capacity=previously_allocated_capacity, issues=issues
        )
        
        # 仿真生产可靠性
        random_seed = m4_config.get('RandomSeed', 42)
        plan_log = module4.simulate_production(plan_log, m4_config['M4_ProductionReliability'], seed=random_seed)
        
        # 计算换产指标
        changeover_log = module4.calculate_changeover_metrics(plan_log, co_def_df)
        
        # 提取并保存当天产线状态供下一天使用（带跨天转产检测）
        current_line_states = module4.extract_line_states_from_plan(plan_log, cap_df, co_def, simulation_date, rate_map.to_dict())
        if current_line_states:
            module4.save_line_state(output_dir, simulation_date, current_line_states)
            print(f"  💾 保存当天产线状态: {list(current_line_states.keys())}")
        
        # 提取并保存当天分配的产能供后续仿真日期使用
        current_allocated_capacity = module4.extract_allocated_capacity_from_plan(plan_log, rate_map.to_dict(), co_def)
        if current_allocated_capacity:
            module4.save_allocated_capacity(output_dir, simulation_date, current_allocated_capacity)
            print(f"  💾 保存当天分配产能: {len(current_allocated_capacity)} 个产能分配 (小时单位)")
        
        # 去重问题
        issues = module4.dedup_issues(issues)
        
        # 生成输出文件
        base_output_file = os.path.join(output_dir, "Module4Output.xlsx")
        daily_output_path = module4.write_output(
            plan_log, exceed_log, issues, changeover_log, 
            base_output_file, simulation_date
        )
        
        print(f"Module4 daily output generated: {daily_output_path}")
        
        # 返回生产计划数据（只返回当日可用的生产）
        if not plan_log.empty and 'available_date' in plan_log.columns:
            plan_log['available_date'] = pd.to_datetime(plan_log['available_date'])
            current_production = plan_log[plan_log['available_date'] >= simulation_date.normalize()]
            
            # 确保返回的数据标识符已标准化，与orchestrator期望格式一致
            if not current_production.empty:
                current_production = _normalize_identifiers(current_production)
                
            return current_production
        else:
            return pd.DataFrame()
        
    except Exception as e:
        import traceback
        print(f'[ERROR] Module4 integrated execution failed for {simulation_date.strftime("%Y-%m-%d")}: {str(e)}')
        print("Full traceback:")
        traceback.print_exc()
        return pd.DataFrame()

# ========== Module4 集成辅助函数（清理后） ==========

# 以下函数保留作为默认配置的备用，但不再使用临时文件













def load_all_historical_production_plans(module4_output_dir: str, current_date: pd.Timestamp, start_date: pd.Timestamp) -> pd.DataFrame:
    """
    加载所有历史的M4生产计划，筛选出当日应该入库的生产
    
    Args:
        module4_output_dir: Module4 输出目录
        current_date: 当前日期
        start_date: 仿真开始日期
        
    Returns:
        pd.DataFrame: 当日应该入库的生产计划数据
    """
    all_production_plans = []
    
    # 遍历从仿真开始到当前日期的所有M4输出文件
    date_range = pd.date_range(start_date, current_date, freq='D')
    
    for date in date_range:
        m4_file = Path(module4_output_dir) / f"Module4Output_{date.strftime('%Y%m%d')}.xlsx"
        
        if m4_file.exists():
            try:
                xl = pd.ExcelFile(m4_file)
                if 'ProductionPlan' in xl.sheet_names:
                    production_df = xl.parse('ProductionPlan')
                    
                    if not production_df.empty:
                        # 添加数据来源标识
                        production_df['source_file'] = str(m4_file)
                        production_df['source_date'] = date
                        all_production_plans.append(production_df)
                        
            except Exception as e:
                print(f"Warning: Failed to read {m4_file}: {e}")
                continue
    
    if not all_production_plans:
        return pd.DataFrame()
    
    # 合并所有生产计划
    combined_production = pd.concat(all_production_plans, ignore_index=True)
    
    # 筛选出当日应该入库的生产 (available_date = current_date)
    if 'available_date' in combined_production.columns:
        combined_production['available_date'] = pd.to_datetime(combined_production['available_date'])
        daily_available = combined_production[
            combined_production['available_date'].dt.normalize() == current_date.normalize()
        ]
        
        # if not daily_available.empty:
        #     print(f"  📦 发现当日入库的历史生产: {len(daily_available)} 条记录")
        #     for _, row in daily_available.iterrows():
        #         print(f"    {row['material']}@{row['location']}: {row['produced_qty']} (生产日期: {row['source_date'].strftime('%Y-%m-%d')})")
        
        return daily_available[['material', 'location', 'line', 'simulation_date', 'available_date', 'produced_qty']]
    
    return pd.DataFrame()

def load_module4_production_output(output_path: str, current_date: pd.Timestamp) -> pd.DataFrame:
    """
    从 Module4 输出文件中加载生产计划数据 (保留用于向后兼容)
    
    Args:
        output_path: Module4 输出文件路径
        current_date: 当前日期
        
    Returns:
        pd.DataFrame: 生产计划数据
    """
    try:
        if not os.path.exists(output_path):
            print(f"Warning: Module4 output file not found: {output_path}")
            return pd.DataFrame()
            
        xl = pd.ExcelFile(output_path)
        if 'ProductionPlan' not in xl.sheet_names:
            print(f"Warning: ProductionPlan sheet not found in {output_path}")
            return pd.DataFrame()
            
        production_df = xl.parse('ProductionPlan')
        
        # 筛选当日的生产计划 (available_date = current_date)
        if not production_df.empty and 'available_date' in production_df.columns:
            production_df['available_date'] = pd.to_datetime(production_df['available_date'])
            # 只返回当日或未来的生产计划
            production_df = production_df[production_df['available_date'] >= current_date.normalize()]
            
        return production_df
        
    except Exception as e:
        print(f"Error loading Module4 production output: {e}")
        return pd.DataFrame()

def load_global_seed(config_dict: dict) -> int:
    """
    统一从 Global_Seed sheet 读取随机种子
    
    Args:
        config_dict: 配置数据字典
        
    Returns:
        int: 随机种子值，默认为 42
    """
    if 'Global_Seed' in config_dict and not config_dict['Global_Seed'].empty:
        seed_df = config_dict['Global_Seed']
        if 'seed' in seed_df.columns:
            seed_value = int(seed_df.iloc[0]['seed'])
            print(f"🌱 从 Global_Seed 读取随机种子: {seed_value}")
            return seed_value
        elif len(seed_df.columns) > 0 and len(seed_df) > 0:
            # 兼容旧格式，读取第一列第一行
            seed_value = int(seed_df.iloc[0, 0])
            print(f"🌱 从 Global_Seed 兼容格式读取随机种子: {seed_value}")
            return seed_value
    
    print(f"⚠️  未找到 Global_Seed 配置，使用默认值: 42")
    return 42

def set_module_seeds(config_dict: dict, global_seed: int = None):
    """
    为所有模块设置统一的随机种子
    
    Args:
        config_dict: 配置数据字典
        global_seed: 全局种子值，如果为 None 则从配置读取
    """
    if global_seed is None:
        global_seed = load_global_seed(config_dict)
    
    # 设置 numpy全局种子
    np.random.seed(global_seed)
    
    # 为各模块设置种子（在配置中覆盖模块特定配置）
    config_dict['M1_RandomSeed'] = global_seed
    config_dict['M3_RandomSeed'] = global_seed  
    config_dict['M4_RandomSeed'] = global_seed
    config_dict['M5_RandomSeed'] = global_seed
    config_dict['M6_RandomSeed'] = global_seed
    
    print(f"✨ 已为所有模块设置统一随机种子: {global_seed}")
    return global_seed

def load_configuration(config_path: str) -> dict:
    """
    加载配置数据
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        dict: 配置数据字典
    """
    print(f"📋 加载配置文件: {config_path}")
    
    try:
        xl = pd.ExcelFile(config_path)
        config_dict = {}
        
        # 加载所有配置表
        for sheet_name in xl.sheet_names:
            config_dict[sheet_name] = xl.parse(sheet_name)
            print(f"  ✅ 加载配置表: {sheet_name} ({len(config_dict[sheet_name])} 行)")
        
        # 确保必要的配置表存在
        required_sheets = [
            'M1_InitialInventory',
            'Global_SpaceCapacity',
            'Global_Network',
            'Global_LeadTime',
            'Global_DemandPriority'
        ]
        
        missing_sheets = [sheet for sheet in required_sheets if sheet not in config_dict]
        if missing_sheets:
            print(f"⚠️  缺少必要配置表: {missing_sheets}")
            # 创建空的配置表
            for sheet in missing_sheets:
                config_dict[sheet] = pd.DataFrame()
        
        # 统一标准化所有配置表的标识符字段
        print(f"🔧 正在标准化标识符字段...")
        standardized_count = 0
        for sheet_name, df in config_dict.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                # 检查是否包含标识符字段
                identifier_cols = ['material', 'location', 'sending', 'receiving', 'sourcing', 'dps_location', 'from_material', 'to_material', 'line', 'delegate_line', 'changeover_id']
                has_identifiers = any(col in df.columns for col in identifier_cols)
                
                if has_identifiers:
                    original_dtypes = {col: str(df[col].dtype) for col in identifier_cols if col in df.columns}
                    config_dict[sheet_name] = _normalize_identifiers(df)
                    new_dtypes = {col: str(config_dict[sheet_name][col].dtype) for col in identifier_cols if col in config_dict[sheet_name].columns}
                    
                    # 记录标准化的字段
                    normalized_fields = []
                    for col in identifier_cols:
                        if col in df.columns and original_dtypes[col] != new_dtypes[col]:
                            normalized_fields.append(f"{col}({original_dtypes[col]}→{new_dtypes[col]})")
                    
                    if normalized_fields:
                        print(f"  🔧 {sheet_name}: {', '.join(normalized_fields)}")
                        standardized_count += 1
        
        if standardized_count > 0:
            print(f"✅ 已标准化 {standardized_count} 个配置表的标识符字段")
        else:
            print(f"✅ 所有配置表的标识符字段已是标准格式")
        
        # 🔧 Changeover 配置校验和去重
        if 'M4_ChangeoverMatrix' in config_dict and not config_dict['M4_ChangeoverMatrix'].empty:
            print(f"\n🔧 校验 Changeover Matrix 配置...")
            co_matrix = config_dict['M4_ChangeoverMatrix']
            
            # 检查重复定义
            duplicates = co_matrix[co_matrix.duplicated(subset=['from_material', 'to_material'], keep=False)]
            if not duplicates.empty:
                print(f"  ⚠️  发现 {len(duplicates)} 条重复的 changeover matrix 定义")
                
                # 详细检查每组重复
                for (from_mat, to_mat), group in duplicates.groupby(['from_material', 'to_material']):
                    unique_coids = group['changeover_id'].unique()
                    if len(unique_coids) > 1:
                        # 不同的 changeover_id - 严重错误
                        print(f"    ❌ ERROR: {from_mat} → {to_mat} 有 {len(unique_coids)} 个不同的 changeover_id: {list(unique_coids)}")
                    else:
                        # 相同的 changeover_id - 只是重复
                        print(f"    ⚠️  {from_mat} → {to_mat} 有 {len(group)} 条重复记录 (changeover_id={unique_coids[0]})")
                
                # 去重（保留第一条）
                original_count = len(co_matrix)
                config_dict['M4_ChangeoverMatrix'] = co_matrix.drop_duplicates(
                    subset=['from_material', 'to_material'], keep='first'
                )
                removed_count = original_count - len(config_dict['M4_ChangeoverMatrix'])
                print(f"  🔧 已去除 {removed_count} 条重复记录")
            else:
                print(f"  ✅ Changeover Matrix 无重复定义")
        
        # 🔧 ChangeoverDefinition 配置校验和去重
        if 'M4_ChangeoverDefinition' in config_dict and not config_dict['M4_ChangeoverDefinition'].empty:
            print(f"\n🔧 校验 Changeover Definition 配置...")
            co_def = config_dict['M4_ChangeoverDefinition']
            
            # 检查重复定义
            duplicates = co_def[co_def.duplicated(subset=['changeover_id', 'line'], keep=False)]
            if not duplicates.empty:
                print(f"  ⚠️  发现 {len(duplicates)} 条重复的 changeover definition 定义")
                
                # 详细检查每组重复
                for (coid, line), group in duplicates.groupby(['changeover_id', 'line']):
                    unique_times = group['time'].unique()
                    if len(unique_times) > 1:
                        # 不同的 time - 严重错误
                        print(f"    ❌ ERROR: changeover_id={coid}, line={line} 有 {len(unique_times)} 个不同的 time 值: {list(unique_times)}")
                    else:
                        # 相同的参数 - 只是重复
                        print(f"    ⚠️  changeover_id={coid}, line={line} 有 {len(group)} 条重复记录 (time={unique_times[0]})")
                
                # 去重（保留第一条）
                original_count = len(co_def)
                config_dict['M4_ChangeoverDefinition'] = co_def.drop_duplicates(
                    subset=['changeover_id', 'line'], keep='first'
                )
                removed_count = original_count - len(config_dict['M4_ChangeoverDefinition'])
                print(f"  🔧 已去除 {removed_count} 条重复记录")
            else:
                print(f"  ✅ Changeover Definition 无重复定义")
        
        # Module4 配置表映射（为了向后兼容）
        print(f"\n🔧 正在映射 Module4 配置表...")
        module4_mappings = {
            'M4_MaterialLocationLineCfg': 'MaterialLocationLineCfg',
            'M4_LineCapacity': 'LineCapacity',
            'M4_ChangeoverMatrix': 'ChangeoverMatrix',
            'M4_ChangeoverDefinition': 'ChangeoverDefinition',
            'M4_ProductionReliability': 'ProductionReliability'
        }

        mapped_count = 0
        for original_key, mapped_key in module4_mappings.items():
            if original_key in config_dict and not config_dict[original_key].empty:
                config_dict[mapped_key] = config_dict[original_key]
                print(f"  🔧 映射 {original_key} → {mapped_key}")
                mapped_count += 1

        if mapped_count > 0:
            print(f"✅ 已映射 {mapped_count} 个 Module4 配置表")
        else:
            print(f"✅ 无需映射 Module4 配置表")
        
        return config_dict
        
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        raise

def run_integrated_simulation(
    config_path: str,
    start_date: str,
    end_date: str,
    output_base_dir: str = "./integrated_output",
    force_restart: bool = False
):
    """
    运行完整的集成仿真
    
    Args:
        config_path: 配置文件路径
        start_date: 仿真开始日期 (YYYY-MM-DD)
        end_date: 仿真结束日期 (YYYY-MM-DD)
        output_base_dir: 输出基础目录
        force_restart: 强制从头开始，忽略续跑能力
    """
    print(f"🚀 开始集成仿真: {start_date} 到 {end_date}")
    print("=" * 60)
    
    # 1. 预验证配置
    print(f"🔍 正在运行仿真前配置验证...")
    validation_passed, validation_report = run_pre_simulation_validation(config_path, output_base_dir)
    
    print(f"📝 验证报告已生成: {validation_report}")
    
    if not validation_passed:
        print("❌ 配置验证失败，请查看验证报告并修复错误后再运行仿真。")
        return {
            'validation_passed': False,
            'validation_report': validation_report,
            'simulation_completed': False
        }
    
    print("✅ 配置验证通过，开始仿真...")
    
    # 2. 检查续跑能力
    actual_start_date = start_date
    is_resuming = False
    resume_info = None
    
    if force_restart:
        print(f"🔄 强制重启模式：忽略任何现有状态，从头开始")
    else:
        resume_info = check_resume_capability(output_base_dir, start_date, end_date)
        
        if resume_info.get('already_completed', False):
            print(f"🎉 仿真已完成！最后处理日期: {resume_info['last_complete_date']}")
            print(f"   总共处理了 {resume_info['days_completed']} 天")
            return {
                'validation_passed': True,
                'simulation_completed': True,
                'already_completed': True,
                'dates_processed': resume_info['days_completed'],
                'last_complete_date': resume_info['last_complete_date']
            }
        elif resume_info['can_resume']:
            print(f"🔄 检测到未完成的仿真，支持续跑:")
            print(f"   已完成: {resume_info['days_completed']} 天 (到 {resume_info['last_complete_date']})")
            print(f"   剩余: {resume_info['days_remaining']} 天 (从 {resume_info['resume_from_date']} 开始)")
            
            # 提供选择（在实际实现中可以加入用户确认）
            print(f"   ✅ 将从 {resume_info['resume_from_date']} 继续运行")
            actual_start_date = resume_info['resume_from_date'] 
            is_resuming = True
        else:
            print(f"📝 未发现可续跑的状态，将从头开始")
    
    # 3. 初始化时间管理器
    time_manager = initialize_time_manager(actual_start_date)
    
    # 3. 创建输出目录
    output_dir = Path(output_base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    orchestrator_output_dir = output_dir / "orchestrator"
    module_outputs = {
        'module1': output_dir / "module1",
        'module3': output_dir / "module3", 
        'module4': output_dir / "module4",
        'module5': output_dir / "module5",
        'module6': output_dir / "module6"
    }
    
    for module_dir in module_outputs.values():
        module_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载配置
    config_dict = load_configuration(config_path)
    
    # 设置全局随机种子
    global_seed = set_module_seeds(config_dict)
    
    # 初始化Orchestrator
    print(f"\n🎯 初始化Orchestrator")
    orchestrator = create_orchestrator(
        start_date=start_date,
        output_dir=str(orchestrator_output_dir)
    )
    # 设置 open deployment 的清理天数，3代表保留3天
    orchestrator.set_past_due_cleanup_grace_days(100)
    if is_resuming:
        # 续跑模式：恢复状态
        print(f"\n🔄 续跑模式：恢复Orchestrator状态")
        restore_orchestrator_state(orchestrator, resume_info['last_complete_date'], output_base_dir)
        
        # 设置空间容量（续跑时也需要重新设置空间容量配置）
        if 'Global_SpaceCapacity' in config_dict and not config_dict['Global_SpaceCapacity'].empty:
            orchestrator.set_space_capacity(config_dict['Global_SpaceCapacity'])
    else:
        # 全新开始：设置初始状态
        print(f"\n🆕 全新开始：设置初始状态")
        
        # 设置初始库存
        if 'M1_InitialInventory' in config_dict and not config_dict['M1_InitialInventory'].empty:
            orchestrator.initialize_inventory(config_dict['M1_InitialInventory'])
        else:
            print("⚠️  未找到初始库存配置，使用空库存")
            orchestrator.initialize_inventory(pd.DataFrame(columns=['material', 'location', 'quantity']))
        
        # 设置空间容量
        if 'Global_SpaceCapacity' in config_dict and not config_dict['Global_SpaceCapacity'].empty:
            orchestrator.set_space_capacity(config_dict['Global_SpaceCapacity'])
        else:
            print("⚠️  未找到空间容量配置")
    
    # 生成仿真日期范围（使用实际开始日期）
    sim_dates = pd.date_range(actual_start_date, end_date, freq='D')
    total_days = len(pd.date_range(start_date, end_date, freq='D'))
    
    if is_resuming:
        print(f"📅 续跑日期范围: {len(sim_dates)} 天 (剩余)")
        print(f"   原始总天数: {total_days}")
        print(f"   已完成: {resume_info['days_completed']} 天")
        print(f"   剩余处理: {len(sim_dates)} 天")
    else:
        print(f"📅 仿真日期范围: {len(sim_dates)} 天")
    
    # 每日循环执行
    all_results = {
        'module1': [],
        'module3': [],
        'module4': [], 
        'module5': [],
        'module6': []
    }
    
    for i, current_date in enumerate(sim_dates, 1):
        # 计算实际的总进度（考虑续跑情况）
        if is_resuming:
            actual_day_number = resume_info['days_completed'] + i
            total_original_days = total_days
            progress_info = f"第 {actual_day_number}/{total_original_days} 天 (续跑第 {i}/{len(sim_dates)} 天)"
        else:
            progress_info = f"第 {i}/{len(sim_dates)} 天"
            
        print(f"\n{'='*20} {progress_info}: {current_date.strftime('%Y-%m-%d')} {'='*20}")
        
        # ==================== 每日开始：GR入库处理 ====================
        try:
            print(f"\n🌅 每日开始状态更新")
            
            # 🔄 第0步：保存期初库存快照（在任何变动之前）
            print(f"  💾 保存期初库存快照...")
            orchestrator.save_beginning_inventory(current_date.strftime('%Y-%m-%d'))
            orchestrator.cleanup_past_due_open_deployments(current_date.strftime('%Y-%m-%d'),grace_days=getattr(orchestrator, "cleanup_grace_days", 0),write_audit=True)

            # 🔄 第1步：处理当日到达的delivery GR (in-transit → inventory)
            print(f"  📦 处理当日delivery GR到达...")
            orchestrator._process_delivery_arrivals(current_date.strftime('%Y-%m-%d'))
            
            # 🔄 第2步：处理历史生产的当日入库 (historical production → inventory)
            print(f"  🏭 处理历史生产当日入库...")
            historical_production = load_all_historical_production_plans(
                module4_output_dir=str(module_outputs['module4']),
                current_date=current_date,
                start_date=pd.to_datetime(start_date)
            )
            
            if not historical_production.empty:
                print(f"    📦 当日需要入库的历史生产: {len(historical_production)} 条记录")
                # 🔧 标准化标识符字段，确保数据类型一致性
                historical_production_normalized = _normalize_identifiers(historical_production)
                orchestrator.process_module4_production(historical_production_normalized, current_date.strftime('%Y-%m-%d'))
            else:
                print(f"    📦 当日无历史生产入库")
                
        except Exception as e:
            print(f"  ❌ 每日开始处理失败: {e}")
        
        # ==================== 模块运行序列 ====================
        # 初始化每日数据变量
        m1_shipments = pd.DataFrame()
        m4_production = pd.DataFrame()
        m5_deployment_df = pd.DataFrame()
        m6_delivery_df = pd.DataFrame()
        
        try:
            # ========== M1: 订单生成 + 立即库存扣减 ==========
            print(f"\n1️⃣ 运行 Module1 - 订单生成")
            try:
                m1_result = module1.run_daily_order_generation(
                    config_dict=config_dict,
                    simulation_date=current_date,
                    output_dir=str(module_outputs['module1']),
                    orchestrator=orchestrator
                )
                m1_shipments = m1_result.get('shipment_df', pd.DataFrame())
                
                # 🔄 立即处理M1 shipment，扣减库存
                if not m1_shipments.empty:
                    print(f"    🚚 立即处理M1 shipment，扣减库存...")
                    # 🔧 标准化标识符字段，确保数据类型一致性
                    m1_shipments_normalized = _normalize_identifiers(m1_shipments)
                    orchestrator.process_module1_shipments(m1_shipments_normalized, current_date.strftime('%Y-%m-%d'))
                    print(f"    ✅ 已扣减 {len(m1_shipments_normalized)} 个shipment的库存")
                
                print(f"  ✅ Module1 完成 - 生成 {len(m1_result.get('orders_df', []))} 个订单, {len(m1_shipments)} 个发货")
                all_results['module1'].append(m1_result)
            except Exception as e:
                print(f"  ❌ Module1 失败: {e}")
                m1_shipments = pd.DataFrame()  # 失败时使用空数据
                # 不用continue，让后面的模块继续执行
            
            # ========== M4: 生产计划 + 立即当日生产入库 ==========
            print(f"\n2️⃣ 运行 Module4 - 生产计划")
            try:
                # 使用集成模式直接调用 Module4 (改进的解决方案)
                m4_production = run_module4_integrated(
                    config_dict=config_dict,
                    module3_output_dir=str(module_outputs['module3']),
                    simulation_date=current_date,
                    simulation_start=pd.to_datetime(start_date),
                    output_dir=str(module_outputs['module4'])
                )
                
                # 🔄 立即处理M4当日生产入库
                if not m4_production.empty:
                    # 筛选当日可用的生产 (available_date = current_date)
                    daily_available = m4_production[
                        pd.to_datetime(m4_production['available_date']).dt.normalize() == current_date.normalize()
                    ]
                    
                    if not daily_available.empty:
                        print(f"    🏭 立即处理M4当日生产入库...")
                        # 🔧 标准化标识符字段，确保数据类型一致性
                        daily_available_normalized = _normalize_identifiers(daily_available)
                        orchestrator.process_module4_production(daily_available_normalized, current_date.strftime('%Y-%m-%d'))
                        print(f"    ✅ 已入库 {len(daily_available_normalized)} 条当日生产")
                    else:
                        print(f"    📦 M4当日无可用生产入库")
                
                print(f"  ✅ Module4 完成 - 生成生产计划: {len(m4_production)} 条记录")
                all_results['module4'].append({'production_df': m4_production})
            except Exception as e:
                print(f"  ❌ Module4 失败: {e}")
                m4_production = pd.DataFrame()  # 失败时使用空数据
            
            # ========== M5: 部署计划 ==========
            print(f"\n3️⃣ 运行 Module5 - 部署计划")
            try:
                # 启用性能分析
                with PerformanceProfiler("Module5", output_dir=Path(output_base_dir) / "performance", enabled=True):
                    m5_result = module5.main(
                        # 集成模式参数
                        config_dict=config_dict,
                        module1_output_dir=str(module_outputs['module1']),
                        module4_output_path=str(module_outputs['module4'] / f"Module4Output_{current_date.strftime('%Y%m%d')}.xlsx"),
                        orchestrator=orchestrator,
                        current_date=current_date.strftime('%Y-%m-%d'),
                        # 输出路径
                        output_path=str(module_outputs['module5'] / f"Module5Output_{current_date.strftime('%Y%m%d')}.xlsx")
                    )
                
                # 获取部署计划数据
                if m5_result and 'deployment_plan' in m5_result:
                    deployment_plan_df = m5_result['deployment_plan']
                    # print(f"    🔍 Module5返回的部署计划: {len(deployment_plan_df)} 条记录")
                    
                    if not deployment_plan_df.empty:
                        # print(f"    📊 部署计划示例数据:")
                        # print(f"    列名: {list(deployment_plan_df.columns)}")
                        # if len(deployment_plan_df) > 0:
                        #     first_row = deployment_plan_df.iloc[0]
                        #     print(f"    第一行数据: {dict(first_row)}")
                        #     if 'deployed_qty_invCon' in deployment_plan_df.columns:
                        #         qty_stats = deployment_plan_df['deployed_qty_invCon'].describe()
                        #         print(f"    deployed_qty_invCon统计: {qty_stats}")
                        
                        # 过滤出有实际部署量的计划，排除自循环（sending=receiving）
                        valid_deployment = deployment_plan_df[
                            (deployment_plan_df['deployed_qty_invCon'] > 0) & 
                            (deployment_plan_df['deployed_qty_invCon'].notna()) &
                            (deployment_plan_df['sending'] != deployment_plan_df['receiving'])  # 排除自循环
                        ].copy()
                        
                        print(f"    🎯 有效部署计划: {len(valid_deployment)}/{len(deployment_plan_df)} 条")
                        
                        if not valid_deployment.empty:
                            # 检查是否已有deployed_qty列，避免重复
                            if 'deployed_qty' in valid_deployment.columns:
                                # 如果已有deployed_qty列，直接使用
                                m5_deployment_df = valid_deployment[[
                                    'material', 'sending', 'receiving', 'date', 'deployed_qty', 'demand_element'
                                ]].rename(columns={'date': 'planned_deployment_date'})
                            else:
                                # 重命名列以匹配orchestrator期望的格式
                                m5_deployment_df = valid_deployment.rename(columns={
                                    'date': 'planned_deployment_date',
                                    'deployed_qty_invCon': 'deployed_qty'
                                })[['material', 'sending', 'receiving', 'planned_deployment_date', 'deployed_qty', 'demand_element']]
                            
                            # 🔧 标准化标识符字段，确保数据类型一致性
                            m5_deployment_df = _normalize_identifiers(m5_deployment_df)
                            
                            # print(f"    ✅ 最终传递给Orchestrator的数据: {len(m5_deployment_df)} 条")
                            # if len(m5_deployment_df) > 0:
                            #     final_qty_stats = m5_deployment_df['deployed_qty'].describe()
                            #     print(f"    deployed_qty统计: {final_qty_stats}")
                            #     print(f"    数据类型: material={m5_deployment_df['material'].dtype}, sending={m5_deployment_df['sending'].dtype}, receiving={m5_deployment_df['receiving'].dtype}")
                            
                            # 🔄 立即处理M5 deployment，更新open deployment
                            print(f"    📦 立即处理M5 deployment，更新open deployment...")
                            orchestrator.process_module5_deployment(m5_deployment_df, current_date.strftime('%Y-%m-%d'))
                            print(f"    ✅ 已更新 {len(m5_deployment_df)} 条部署计划到open deployment")
                            
                            print(f"  ✅ Module5 完成 - 生成 {len(m5_deployment_df)} 条有效部署计划")
                        else:
                            print(f"  ✅ Module5 完成 - 无有效部署计划")
                    else:
                        print(f"  ✅ Module5 完成 - 部署计划为空")
                else:
                    print(f"  ✅ Module5 完成 - 无返回结果")
                
                all_results['module5'].append(m5_result)
            except Exception as e:
                print(f"  ❌ Module5 失败: {e}")
                # 不用continue，让后面的模块继续执行
                m5_deployment_df = pd.DataFrame()  # 设置默认值
            
            # ========== M6: 物流执行 + 立即多状态更新 ==========
            print(f"\n4️⃣ 运行 Module6 - 物流执行")
            try:
                m6_result = module6.run_daily_physical_flow(
                    config_dict=config_dict,
                    orchestrator=orchestrator,
                    current_date=current_date,
                    output_dir=str(module_outputs['module6']),
                    max_wait_days=30,
                    random_seed=config_dict.get('M6_RandomSeed', 42)  # 使用统一种子
                )
                
                # 获取交付计划数据
                if m6_result and 'delivery_plan' in m6_result:
                    m6_delivery_df = m6_result.get('delivery_plan', pd.DataFrame())
                    
                    # 🔄 立即处理M6 delivery，更新多个状态
                    if not m6_delivery_df.empty:
                        print(f"    🚛 立即处理M6 delivery，更新库存/open deployment/in-transit...")
                        # 🔧 标准化标识符字段，确保数据类型一致性
                        m6_delivery_df_normalized = _normalize_identifiers(m6_delivery_df)
                        orchestrator.process_module6_delivery(m6_delivery_df_normalized, current_date.strftime('%Y-%m-%d'))
                        print(f"    ✅ 已处理 {len(m6_delivery_df_normalized)} 条delivery计划，更新相关状态")
                    
                    print(f"  ✅ Module6 完成 - 生成 {len(m6_delivery_df)} 条交付计划")
                else:
                    print(f"  ✅ Module6 完成 - 无交付计划")
                    m6_delivery_df = pd.DataFrame()
                
                all_results['module6'].append(m6_result)
            except Exception as e:
                print(f"  ❌ Module6 失败: {e}")
                # 不用continue，让后面的模块继续执行
                m6_delivery_df = pd.DataFrame()  # 设置默认值
            
            # ========== M3: 净需求计算 ==========
            print(f"\n5️⃣ 运行 Module3 - 净需求计算")
            try:
                # 启用性能分析
                with PerformanceProfiler("Module3", output_dir=Path(output_base_dir) / "performance", enabled=True):
                    m3_result = module3.run_integrated_mode(
                        module1_output_dir=str(module_outputs['module1']),
                        orchestrator=orchestrator,
                        config_dict=config_dict,
                        start_date=current_date.strftime('%Y-%m-%d'),
                        end_date=current_date.strftime('%Y-%m-%d'),
                        output_dir=str(module_outputs['module3'])
                    )
                print(f"  ✅ Module3 完成")
                all_results['module3'].append(m3_result)
            except Exception as e:
                print(f"  ❌ Module3 失败: {e}")
                import traceback
                traceback.print_exc()
                # 不用continue，让他继续执行
            
            # ==================== 每日结束：保存状态 ====================
            print(f"\n💾 每日结束状态保存")
            try:
                # 保存期末库存快照（在保存状态之前）
                orchestrator.save_ending_inventory(current_date.strftime('%Y-%m-%d'))
                
                # 输出详细的库存变动记录用于调试
                orchestrator.output_daily_inventory_summary(current_date.strftime('%Y-%m-%d'))
                
                # 直接保存每日状态，状态更新已在各模块运行后实时完成
                orchestrator.save_daily_state(current_date.strftime('%Y-%m-%d'))
                
                # 获取当日统计
                stats = orchestrator.get_summary_statistics(current_date.strftime('%Y-%m-%d'))
                print(f"  📊 当日统计: {stats}")
                print(f"  💾 Orchestrator 状态已保存")
                
            except Exception as e:
                print(f"  ❌ 每日状态保存失败: {e}")
                # 不用continue，让他继续到下一天
            
            print(f"✅ 第 {i} 天处理完成")
            
        except Exception as e:
            print(f"❌ 第 {i} 天处理失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 生成最终报告
    print(f"\n📊 仿真完成报告")
    print("=" * 60)
    print(f"仿真期间: {start_date} 到 {end_date} ({len(sim_dates)} 天)")
    print(f"输出目录: {output_base_dir}")
    
    for module_name, results in all_results.items():
        print(f"{module_name.upper()}: {len(results)} 天成功处理")
    
    # 进行库存平衡检查
    print(f"\n🔎 正在进行库存平衡检查...")
    validation_manager = ValidationManager(str(output_dir))
    inventory_checker = InventoryBalanceChecker(validation_manager, orchestrator)
    balance_passed = inventory_checker.validate_inventory_consistency(start_date, end_date)
    
    if balance_passed:
        print("✅ 库存平衡检查通过")
    else:
        print("⚠️  库存平衡检查发现问题，请查看验证报告")
    
    # 生成汇总报告
    print(f"\n📊 正在生成汇总报告...")
    summary_generator = SummaryReportGenerator(str(output_dir), config_dict)
    summary_reports = summary_generator.generate_all_reports(start_date, end_date)
    
    # 写入库存平衡检查报告
    balance_report_path = validation_manager.write_report()
    
    # 获取最终Orchestrator统计
    final_date = sim_dates[-1].strftime('%Y-%m-%d')
    final_stats = orchestrator.get_summary_statistics(final_date)
    print(f"\n🎯 最终Orchestrator状态:")
    for key, value in final_stats.items():
        print(f"  {key}: {value}")
    
    if is_resuming:
        total_processed = resume_info['days_completed'] + len(sim_dates)
        print(f"\n🎉 续跑仿真完成!")
        print(f"   本次处理: {len(sim_dates)} 天")
        print(f"   总共完成: {total_processed} 天")
    else:
        total_processed = len(sim_dates)
        print(f"\n🎉 集成仿真完成!")
        print(f"   总共处理: {total_processed} 天")
    
    return {
        'validation_passed': True,
        'simulation_completed': True,
        'is_resuming': is_resuming,
        'dates_processed_this_run': len(sim_dates),
        'total_dates_processed': total_processed if is_resuming else len(sim_dates),
        'resume_info': resume_info if is_resuming else None,
        'results': all_results,
        'final_stats': final_stats,
        'output_directory': output_base_dir,
        'validation_report': validation_report,
        'balance_check_passed': balance_passed,
        'balance_report': balance_report_path,
        'summary_reports': summary_reports
    }

def main():
    """主函数 - 独立执行集成仿真"""
    # 配置文件路径（可以通过命令行参数或环境变量指定）
    import argparse
    
    parser = argparse.ArgumentParser(description="运行供应链集成仿真")
    parser.add_argument("--config", "-c", 
                       default="./config/integration_config.json",
                       help="配置文件路径 (默认: ./config/integration_config.json)")
    parser.add_argument("--start-date", "-s", 
                       default="2024-01-01",
                       help="仿真开始日期 (默认: 2024-01-01)")
    parser.add_argument("--end-date", "-e", 
                       default="2024-01-05",
                       help="仿真结束日期 (默认: 2024-01-03)")
    parser.add_argument("--output", "-o", 
                       default=None,
                       help="输出目录 (默认: 根据配置文件名生成)")
    parser.add_argument("--force-restart", 
                       action="store_true",
                       help="强制从头开始，忽略续跑能力 (默认: False)")
    parser.add_argument("--check-resume", 
                       action="store_true",
                       help="仅检查续跑状态，不执行仿真 (默认: False)")
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"❌ 配置文件不存在: {args.config}")
        print("请提供有效的配置文件路径，或使用测试脚本生成配置")
        sys.exit(1)
    
    # 如果没有指定输出目录，根据配置文件名生成
    if args.output is None:
        config_name = os.path.splitext(os.path.basename(args.config))[0]
        args.output = f"./{config_name}_output"
        print(f"💫 使用默认输出目录: {args.output}")
    
    # 处理续跑检查选项
    if args.check_resume:
        print(f"🔍 检查续跑状态...")
        resume_info = check_resume_capability(args.output, args.start_date, args.end_date)
        
        print(f"\n📊 续跑状态报告:")
        print(f"  输出目录: {args.output}")
        print(f"  原始日期范围: {args.start_date} 到 {args.end_date}")
        
        if resume_info.get('already_completed', False):
            print(f"  ✅ 仿真已完成！")
            print(f"     最后处理日期: {resume_info['last_complete_date']}")
            print(f"     总处理天数: {resume_info['days_completed']}")
        elif resume_info['can_resume']:
            print(f"  🔄 可以续跑！")
            print(f"     已完成: {resume_info['days_completed']} 天 (到 {resume_info['last_complete_date']})")
            print(f"     剩余: {resume_info['days_remaining']} 天 (从 {resume_info['resume_from_date']} 开始)")
        else:
            print(f"  📝 无法续跑，需要从头开始")
            print(f"     需要处理: {resume_info['days_remaining']} 天")
        
        return  # 仅检查，不执行
    
    # 处理强制重启选项
    if args.force_restart:
        print(f"🔄 强制重启模式：将从头开始，忽略任何现有状态")
        # 可以考虑删除现有输出目录，或者修改run_integrated_simulation函数来支持强制重启
        # 这里暂时通过添加标志来实现
    
    try:
        # 添加强制重启参数（需要修改run_integrated_simulation函数签名）
        result = run_integrated_simulation(
            config_path=args.config,
            start_date=args.start_date,
            end_date=args.end_date,
            output_base_dir=args.output,
            force_restart=args.force_restart  # 新增参数
        )
        
        print(f"\n✅ 仿真结果:")
        if result.get('is_resuming', False):
            print(f"  续跑模式: 是")
            print(f"  本次处理天数: {result.get('dates_processed_this_run', 0)}")
            print(f"  总处理天数: {result.get('total_dates_processed', 0)}")
        else:
            print(f"  全新运行: 是")  
            print(f"  处理天数: {result.get('dates_processed_this_run', 0)}")
        print(f"  输出目录: {result.get('output_directory', 'Unknown')}")
        
        if result.get('already_completed', False):
            print(f"  📝 注意: 仿真之前已完成，无需处理")
        
    except Exception as e:
        print(f"❌ 集成仿真失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()