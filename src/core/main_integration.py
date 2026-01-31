"""
main_integration.py

整体目的：
- 作为供应链仿真的主集成执行脚本，统一调度 Module1/3/4/5/6，通过 Orchestrator 在日度粒度上串联生产、部署与物流流程，实现端到端的数据流转与状态维护。

功能点：
- 断点续跑：自动检测最后完整日期、支持状态恢复并继续运行，避免重复计算与中断损失。
- 预验证与加载：在仿真前运行配置校验，统一读取并标准化配置表的标识符字段，同时对 M4 的换产配置进行校验与去重。
- 模块序列执行：按日循环依次执行 M1→M4→M5→M6→M3，各模块间立即更新库存与状态以确保数据一致性。
- 状态管理与输出：在每日开始/结束阶段保存库存快照、输出每日汇总与详细日志，最终生成汇总报告与库存一致性验证。
- 随机性控制：统一读取并设置全局随机种子，保证仿真可复现。

使用方法：
- 通过 `run_integrated_simulation(config_path, start_date, end_date, ...)` 执行完整仿真；或运行 `main()` 支持命令行参数调用。
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import os
from typing import Any, Dict, Optional
from pandas.errors import EmptyDataError, ParserError
import logging
from pathlib import Path

# Windows UTF-8 编码设置 - 解决emoji和中文输出问题
if sys.platform == 'win32':
    import io
    # 设置stdout/stderr为UTF-8编码
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 将父目录添加到路径中以便导入
sys.path.insert(0, str(Path(__file__).parent.parent))

from . import orchestrator
from .orchestrator import create_orchestrator
from ..utils.validation_manager import ValidationManager
from ..utils.time_manager import SimulationTimeManager, initialize_time_manager
from ..utils.config_validator import run_pre_simulation_validation
from ..utils.inventory_balance_checker import InventoryBalanceChecker
from ..services.summary_report_generator import SummaryReportGenerator
from ..services.performance_profiler import PerformanceProfiler
from ..modules import module1
from ..modules import module3
from ..modules import module4
from ..modules import module5
from ..modules import module6

logger = logging.getLogger(__name__)


# ========================= DuckDB内存模式延迟导入 =========================
_memory_store_imported = False
_enable_memory_mode = None
_disable_memory_mode = None
_is_memory_mode_enabled = None
_get_data_store = None


def _ensure_memory_store_imported():
    """延迟导入内存存储模块（避免循环导入和启动延迟）"""
    global _memory_store_imported, _enable_memory_mode, _disable_memory_mode
    global _is_memory_mode_enabled, _get_data_store
    if not _memory_store_imported:
        try:
            from ..utils.memory_data_store import (
                enable_memory_mode,
                disable_memory_mode,
                is_memory_mode_enabled,
                get_data_store,
            )
            _enable_memory_mode = enable_memory_mode
            _disable_memory_mode = disable_memory_mode
            _is_memory_mode_enabled = is_memory_mode_enabled
            _get_data_store = get_data_store
        except ImportError as e:
            logger.warning(f"DuckDB内存模块导入失败: {e}")
            _enable_memory_mode = lambda **kwargs: None
            _disable_memory_mode = lambda: None
            _is_memory_mode_enabled = lambda: False
            _get_data_store = lambda: None
        _memory_store_imported = True


# ========================= 断点续跑功能 =========================

def detect_last_complete_date(output_base_dir: str, start_date: str, end_date: str) -> str:
    """检测最后一个完整处理的日期

    目的：
    - 扫描 `orchestrator/` 目录下的每日关键文件集合，判定最近一次完整输出的仿真日期，用于断点续跑起点。

    Args:
        output_base_dir: 输出基础目录。
        start_date: 原始开始日期（YYYY-MM-DD）。
        end_date: 原始结束日期（YYYY-MM-DD）。

    Returns:
        str: 最后完整处理的日期（YYYY-MM-DD）；若未找到完整日期返回 None。

    输入数据：
        - `orchestrator/` 目录下以日期命名的 CSV 文件（每日库存与日志等）。

    输出/副作用：
        - 仅读取表头验证文件可用，不更改任何状态；打印诊断信息。

    逻辑：
        - 按日期遍历并检查必需文件是否存在且可解析；若全部存在则记录为最后完整日期并继续；遇到缺失或异常则停止并返回当前记录。
    """
    print(f"🔍 检测中断点...")
    
    output_dir = Path(output_base_dir)
    orchestrator_dir = output_dir / "orchestrator"
    
    if not orchestrator_dir.exists():
        print(f"📁 输出目录不存在，将从头开始: {orchestrator_dir}")
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
                    print(f"✅ 发现完整日期: {last_complete_date}")
                else:
                    break
            except Exception as e:
                print(f"⚠️日期 {current_date.strftime('%Y-%m-%d')} 文件损坏: {e}")
                break
        else:
            print(f"❌ 日期 {current_date.strftime('%Y-%m-%d')} 文件不完整")
            break
    
    if last_complete_date:
        print(f"🎯 检测到最后完整日期: {last_complete_date}")
    else:
        print("📝 未发现完整日期，将从头开始")
        
    return last_complete_date

def restore_orchestrator_state(orchestrator, restore_date: str, output_base_dir: str):
    """从指定日期的状态文件恢复 Orchestrator 状态

    目的：
    - 依据已输出的 CSV 状态文件重建 Orchestrator 的内存状态（库存、在途、开放调拨、空间配额、日志与索引），以支持断点续跑。

    Args:
        orchestrator: Orchestrator 实例。
        restore_date: 恢复日期 (YYYY-MM-DD)。
        output_base_dir: 输出基础目录。

    输入数据：
        - `orchestrator/` 目录下以 `restore_date` 为基准的多类 CSV 文件（库存/在途/调拨/空间/日志等）。

    输出/副作用：
        - 回填 Orchestrator 的多个字典/列表属性；重建按日期索引的日志字典；设置 `current_date`。

    逻辑：
        - 逐类文件读取→标准化标识符→重建映射与列表→构建日期索引→设置当前日期，期间对空文件与解析异常采取容错策略。
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
        
        # 2. 恢复在途库存（必须重建为以 UID 为键的 in_transit 字典）
        intransit_file = orchestrator_dir / f"planning_intransit_{date_str}.csv"
        if intransit_file.exists():
            try:
                intransit_df = pd.read_csv(intransit_file, dtype=object)
            except EmptyDataError:
                intransit_df = pd.DataFrame()
            if not intransit_df.empty:
                intransit_df = _normalize_identifiers(intransit_df)
                # 重建 in_transit 字典：transit_uid -> transit_record
                orchestrator.in_transit = {}
                for _, row in intransit_df.iterrows():
                    transit_uid = row.get('transit_uid')
                    if transit_uid is not None and str(transit_uid).strip() and str(transit_uid) != 'None':
                        uid_str = str(transit_uid)
                        # 安全地将 quantity 转为 int
                        try:
                            quantity = int(float(row.get('quantity', 0) or 0))
                        except (ValueError, TypeError):
                            quantity = 0
                        
                        # 将日期字段转换为 datetime（Module6 需要 datetime 进行比较）
                        try:
                            actual_ship_date = pd.to_datetime(row.get('actual_ship_date')).normalize() if pd.notna(row.get('actual_ship_date')) else None
                        except:
                            actual_ship_date = None
                        try:
                            actual_delivery_date = pd.to_datetime(row.get('actual_delivery_date')).normalize() if pd.notna(row.get('actual_delivery_date')) else None
                        except:
                            actual_delivery_date = None
                        
                        orchestrator.in_transit[uid_str] = {
                            'material': str(row.get('material', '')),
                            'sending': str(row.get('sending', '')),
                            'receiving': str(row.get('receiving', '')),
                            'actual_ship_date': actual_ship_date,
                            'actual_delivery_date': actual_delivery_date,
                            'quantity': quantity,
                            'ori_deployment_uid': str(row.get('ori_deployment_uid', '')),
                            'vehicle_uid': str(row.get('vehicle_uid', ''))
                        }
            else:
                orchestrator.in_transit = {}
            print(f"  ✅ 恢复在途记录: {len(orchestrator.in_transit)} 条")
        else:
            orchestrator.in_transit = {}
        
        # 3. 恢复开放调拨（必须是以 UID 为键的字典，而不是列表）
        deployment_file = orchestrator_dir / f"open_deployment_{date_str}.csv"
        if deployment_file.exists():
            try:
                deployment_df = pd.read_csv(deployment_file, dtype=object)
            except EmptyDataError:
                deployment_df = pd.DataFrame()
            if not deployment_df.empty:
                deployment_df = _normalize_identifiers(deployment_df)
                # 重建为字典：uid -> deployment_record
                orchestrator.open_deployment = {}
                for _, row in deployment_df.iterrows():
                    uid = row.get('ori_deployment_uid')
                    if uid is not None and str(uid).strip() and str(uid) != 'None':
                        uid_str = str(uid)
                        # 安全地将 deployed_qty 转为 int
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
        
        # 5. 恢复生产计划 backlog（含未来生产）
        production_backlog_file = orchestrator_dir / f"production_plan_backlog_{date_str}.csv"
        if production_backlog_file.exists():
            try:
                backlog_df = pd.read_csv(production_backlog_file, dtype=object)
            except EmptyDataError:
                backlog_df = pd.DataFrame()
            if not backlog_df.empty:
                backlog_df = _normalize_identifiers(backlog_df)
                # 将 quantity 转为 int
                if 'quantity' in backlog_df.columns:
                    backlog_df['quantity'] = pd.to_numeric(backlog_df['quantity'], errors='coerce').fillna(0).astype(int)
                # 将 available_date 转为 datetime 以匹配原有结构
                if 'available_date' in backlog_df.columns:
                    backlog_df['available_date'] = pd.to_datetime(backlog_df['available_date']).dt.normalize()
                
                # 转为字典列表并保持类型（高效，无需 iterrows）
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
        
        # 6. 重建按日期索引的字典（用于阶段 6 优化）
        print(f"  🔧 重建日期索引字典...")
        orchestrator.production_gr_by_date = {}
        orchestrator.delivery_gr_by_date = {}
        orchestrator.shipment_log_by_date = {}
        orchestrator.delivery_shipment_log_by_date = {}
        
        # 索引 production_gr
        for record in orchestrator.production_gr:
            date_key = record.get('date', '')
            if date_key not in orchestrator.production_gr_by_date:
                orchestrator.production_gr_by_date[date_key] = []
            orchestrator.production_gr_by_date[date_key].append(record)
        
        # 索引 delivery_gr
        for record in orchestrator.delivery_gr:
            date_key = record.get('date', '')
            if date_key not in orchestrator.delivery_gr_by_date:
                orchestrator.delivery_gr_by_date[date_key] = []
            orchestrator.delivery_gr_by_date[date_key].append(record)
        
        # 索引 shipment_log
        for record in orchestrator.shipment_log:
            date_key = record.get('date', '')
            if date_key not in orchestrator.shipment_log_by_date:
                orchestrator.shipment_log_by_date[date_key] = []
            orchestrator.shipment_log_by_date[date_key].append(record)
        
        # 索引 delivery_shipment_log
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
    """检查是否可以续跑，返回续跑信息

    目的：
    - 综合最后完整日期与目标区间，计算续跑起点与剩余天数，指示是否已全部完成或可继续。

    Args:
        output_base_dir: 输出基础目录。
        start_date: 仿真开始日期（YYYY-MM-DD）。
        end_date: 仿真结束日期（YYYY-MM-DD）。

    Returns:
        dict: 包含可续跑标志、最后完整日期、续跑起始日期、已完成和剩余天数等的字典。

    输入/输出/逻辑：
        - 调用 `detect_last_complete_date` 获取完整日期；
        - 若已覆盖到 `end_date`，返回已完成标记；否则计算续跑起点（完整日期+1）与剩余范围。
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
    """标准化地点编号

    目的/逻辑：
    - 若为纯数字字符串则左侧补零至4位；非数字（如 A888）保持原样；空值返回空串。

    输入：`location_str` 任意类型标识。
    输出：规范化的字符串地点标识。
    """
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
    """标准化物料编码

    目的/逻辑：
    - 将数值型/带小数的物料转换为无小数整数字符串；空/None/NAN 返回空串；
      其余去除首尾空格。
    - 与code_v0保持一致的实现。

    输入：`material_str` 任意类型标识。
    输出：规范化的字符串物料编码。
    """
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
    """标准化发送地编号

    目的/逻辑：
    - 纯数字补零至4位；非数字保持；空值返回空串。
    输入：`sending_str`
    输出：规范化字符串。
    """
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
    """标准化接收地编号

    目的/逻辑：同 `_normalize_sending`。
    输入：`receiving_str`
    输出：规范化字符串。
    """
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
    """标准化标识符字段（DataFrame级）

    目的：
    - 统一将多个标识符列转换为字符串并进行必要的格式化（地点补零、物料去小数等），确保跨模块数据类型一致。

    Args:
        df: 输入数据表。

    Returns:
        pd.DataFrame: 标识符标准化后的副本。

    输入数据：
        - 含 `material/location/sending/receiving/sourcing/...` 等列的表。

    输出/副作用：
        - 返回新 DataFrame，不修改原对象。

    逻辑：
        - 针对每个识别列执行 astype(str)+逐列规范化；空表直接返回。
    """
    if df.empty:
        return df
    
    # 定义需要字符串转换的标识符列
    identifier_cols = ['material', 'location', 'sending', 'receiving', 'sourcing', 'dps_location', 'from_material', 'to_material', 'line', 'delegate_line', 'changeover_id']
    
    df = df.copy()
    for col in identifier_cols:
        if col in df.columns:
            # 🔧 关键修复：使用 object dtype (Python str) 而不是 pandas StringDtype
            # 这样可以确保与后续 astype(str) 的一致性
            df[col] = df[col].astype(str)
            # 对地点类字段应用专用规范化
            if col in ['location', 'dps_location']:
                df[col] = df[col].apply(_normalize_location)
            elif col == 'sending':
                df[col] = df[col].apply(_normalize_sending)
            elif col == 'receiving':
                df[col] = df[col].apply(_normalize_receiving)
            # 对物料类字段应用专用规范化
            elif col in ['material', 'from_material', 'to_material']:
                df[col] = df[col].apply(_normalize_material)
            # changeover_id 和 line 只需要转换为字符串，不需要特殊格式化
            # (已在 astype('string') 时处理)
            # 其他标识符列（line、delegate_line 等）确保为正确的字符串格式
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
    output_dir: str,
    skip_file_output: bool = False,
    module3_result: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """集成模式运行 Module4 生产计划（直接用 config_dict）

    目的：
    - 无需临时文件，使用内存中的配置与 Module3 输出，构建并分配当日生产计划，立即入库可用产出。

    Args:
        config_dict: 配置数据字典（包含 M4 所需表）。
        module3_output_dir: Module3 输出目录，用于读取日度净需求（当module3_result为None时使用）。
        simulation_date: 当前仿真日期。
        simulation_start: 仿真开始日期。
        output_dir: 输出目录，用于写每日 M4 输出。
        skip_file_output: 是否跳过写入Excel文件（数据库模式使用）。
        module3_result: Module3运行结果（内存数据），包含net_demand_df。优先使用此参数。

    Returns:
        pd.DataFrame: 生产计划数据（含 available_date 等），用于当日入库处理。

    输入数据：
        - M4 配置表（LineCfg/Capacity/ChangeoverMatrix/Definition/ProductionReliability）。
        - Module3 的日度净需求（优先从module3_result内存获取，否则从文件读取）。

    输出/副作用：
        - 写每日 M4 输出文件（除非 skip_file_output=True）；返回当日及未来的生产记录；可能更新产线状态与已分配产能持久化。

    逻辑：
        - 校验配置→加载净需求→构建无约束计划→处理换产与产能分配→模拟生产可靠性→提取并保存状态→返回可用生产。
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
        
        # 🦆 优先从内存加载Module3净需求数据，否则从文件读取
        if module3_result is not None and 'net_demand_df' in module3_result:
            net_demand_df = module3_result['net_demand_df'].copy()
            # 🦆 内存模式：不按simulation_date筛选，因为previous_day_m3_result包含前一天计算的所有净需求
            # 这些净需求的requirement_date才是实际需求日期，simulation_date只是M3运行的日期
            
            # 🔧 关键修复：应用与load_daily_net_demand相同的处理逻辑
            # 1. 筛选layer=0（下游需求），与文件加载模式保持一致
            if 'layer' in net_demand_df.columns:
                net_demand_df = net_demand_df[net_demand_df['layer'] == 0].copy()
            # 2. 数量取绝对值，与文件加载模式保持一致
            if 'quantity' in net_demand_df.columns:
                net_demand_df['quantity'] = net_demand_df['quantity'].abs()
        else:
            # 从文件加载
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
        
        # 注意：Changeover 去重已在 load_configuration 中完成
        
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
            print(f"🔄加载前一天产线状态: {list(previous_line_states.keys())}")
        else:
            print("🔄无前一天产线状态 - 全新开始")
        
        # 加载之前所有仿真日期已分配的产能
        previously_allocated_capacity = module4.load_all_previous_capacity(output_dir, simulation_date)
        if previously_allocated_capacity:
            print(f"🔄加载之前已分配产能: {len(previously_allocated_capacity)} 个产能分配")
        else:
            print("🔄无之前已分配产能 - 全新开始")
        
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
            print(f"💾保存当天产线状态: {list(current_line_states.keys())}")
        
        # 提取并保存当天分配的产能供后续仿真日期使用
        current_allocated_capacity = module4.extract_allocated_capacity_from_plan(plan_log, rate_map.to_dict(), co_def)
        if current_allocated_capacity:
            module4.save_allocated_capacity(output_dir, simulation_date, current_allocated_capacity)
            print(f"💾保存当天分配产能: {len(current_allocated_capacity)} 个产能分配 (小时单位)")
        
        # 去重问题
        issues = module4.dedup_issues(issues)
        
        # 转换issues为DataFrame
        issues_df = pd.DataFrame(issues) if issues else pd.DataFrame()
        
        # 生成输出文件（仅在非数据库模式下写入）
        if not skip_file_output:
            base_output_file = os.path.join(output_dir, "Module4Output.xlsx")
            daily_output_path = module4.write_output(
                plan_log, exceed_log, issues, changeover_log, 
                base_output_file, simulation_date
            )
            print(f"Module4 daily output generated: {daily_output_path}")
        
        # 返回完整的Module4结果（包含所有输出表）
        # production_df 用于数据库写入，应包含完整的生产计划
        production_df = pd.DataFrame()
        if not plan_log.empty:
            # 确保 available_date 是日期类型
            if 'available_date' in plan_log.columns:
                plan_log['available_date'] = pd.to_datetime(plan_log['available_date'])
            # 标准化标识符后作为完整的生产计划
            production_df = _normalize_identifiers(plan_log.copy())
        
        # 返回完整结构供数据库写入
        return {
            'production_df': production_df,
            'exceed_log': exceed_log if isinstance(exceed_log, pd.DataFrame) else pd.DataFrame(exceed_log) if exceed_log else pd.DataFrame(),
            'issues_df': issues_df,
            'changeover_log': changeover_log if isinstance(changeover_log, pd.DataFrame) else pd.DataFrame(changeover_log) if changeover_log else pd.DataFrame(),
        }
        
    except Exception as e:
        import traceback
        print(f'[ERROR] Module4 integrated execution failed for {simulation_date.strftime("%Y-%m-%d")}: {str(e)}')
        print("Full traceback:")
        traceback.print_exc()
        # 返回空结构
        return {
            'production_df': pd.DataFrame(),
            'exceed_log': pd.DataFrame(),
            'issues_df': pd.DataFrame(),
            'changeover_log': pd.DataFrame(),
        }

# ========== Module4 集成辅助函数（清理后） ==========

# 以下函数保留作为默认配置的备用，但不再使用临时文件













def load_current_date_production_gr(module4_output_dir: str, current_date: pd.Timestamp, start_date: pd.Timestamp) -> pd.DataFrame:
    """加载历史 M4 生产计划并筛选当日入库

    目的：
    - 汇总从仿真开始至今的所有 M4 输出，提取 `available_date == current_date` 的生产记录用于入库。

    Args:
        module4_output_dir: Module4 输出目录。
        current_date: 当前日期。
        start_date: 仿真开始日期。

    Returns:
        pd.DataFrame: 当日应该入库的生产计划。

    输入/输出/逻辑：
        - 遍历日期读取 `Module4Output_YYYYMMDD.xlsx`→合并→按 available_date 过滤当日→返回关键列。
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
    """从 Module4 输出文件加载生产计划（向后兼容）

    目的：
    - 兼容旧流程，从单个输出文件读取生产计划，并筛选当日及未来的可用生产。

    Args:
        output_path: Module4 输出文件路径。
        current_date: 当前日期。

    Returns:
        pd.DataFrame: 可用生产计划数据。

    逻辑：
        - 读取 Excel→解析 `ProductionPlan`→按 `available_date >= current_date` 过滤。
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
    """统一从 Global_Seed 读取随机种子

    目的：
    - 提供稳定的随机性来源，优先读取标准列 `seed`，兼容旧格式（第一列第一行）。

    Args:
        config_dict: 配置数据字典。

    Returns:
        int: 随机种子值，默认 42。

    逻辑：
        - 按优先级读取→打印提示→返回默认或实际种子。
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
    
    print("⚠️未找到 Global_Seed 配置，使用默认值: 42")
    return 42

def set_module_seeds(config_dict: dict, global_seed: int = None):
    """为所有模块设置统一随机种子

    目的：
    - 将全局种子应用于 numpy 及各模块配置，确保仿真可复现。

    Args:
        config_dict: 配置数据字典。
        global_seed: 指定全局种子；为 None 时从配置读取。

    Returns:
        int: 实际使用的全局种子。

    逻辑：
        - 若未提供则读取→设置 numpy 种子→写入各模块种子键→打印确认。
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
    
    print(f"✨已为所有模块设置统一随机种子: {global_seed}")
    return global_seed


def load_configuration_from_dict(config_data: dict, config_name: str = "DB_Config") -> dict:
    """从DataFrame字典加载与标准化配置数据（用于数据库模式）

    目的：
    - 直接接收DataFrame字典，无需创建临时Excel文件
    - 执行与load_configuration相同的标准化处理

    Args:
        config_data: 配置数据字典 {sheet_name: DataFrame}
        config_name: 配置名称（用于日志）

    Returns:
        dict: 标准化后的配置数据字典
    """
    print(f"📋 处理配置数据: {config_name} (共 {len(config_data)} 个表)")
    
    # Sheet名称映射（数据库小写 -> 原始大小写）
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
    column_mapping = {
        'material': 'material', 'location': 'location', 'sourcing': 'sourcing',
        'location_type': 'location_type', 'quantity': 'quantity', 'date': 'date',
        'week': 'week', 'day': 'day', 'seed': 'seed', 'eff_from': 'eff_from',
        'eff_to': 'eff_to', 'demand_element': 'demand_element', 'priority': 'priority',
        'order_type': 'order_type', 'error_std_percent': 'error_std_percent',
        'order_day_flag': 'order_day_flag', 'advance_days': 'advance_days',
        'ao_percent': 'ao_percent', 'dps_location': 'dps_location',
        'dps_percent': 'dps_percent', 'safety_stock_qty': 'safety_stock_qty',
        'key': 'key', 'sending': 'sending', 'receiving': 'receiving',
        'pdt': 'PDT', 'gr': 'GR', 'mct': 'MCT', 'otd': 'OTD',
        'delegate_line': 'delegate_line', 'prd_rate': 'prd_rate',
        'min_batch': 'min_batch', 'rv': 'rv', 'ptf': 'ptf', 'lsk': 'lsk',
        'line': 'line', 'capacity': 'capacity', 'from_material': 'from_material',
        'to_material': 'to_material', 'changeover_id': 'changeover_id',
        'from_line': 'from line', 'to_line': 'to line', 'time': 'time',
        'cost': 'cost', 'mu_loss': 'mu_loss', 'pr': 'pr', 'model': 'model',
        'moq': 'moq', 'truck_type': 'truck_type', 'optimal_type': 'optimal_type',
        'wfr': 'WFR', 'vfr': 'VFR', 'mdq': 'MDQ', 'weight': 'weight',
        'volume': 'volume', 'demand_unit_to_weight': 'demand_unit_to_weight',
        'demand_unit_to_volume': 'demand_unit_to_volume', 'delay_days': 'delay_days',
        'probability': 'probability', 'condition_logic': 'condition_logic',
        'rule_id': 'rule_id', 'max_weight': 'max_weight', 'max_volume': 'max_volume',
        'capacity_qty_in_weight': 'capacity_qty_in_weight',
        'capacity_qty_in_volume': 'capacity_qty_in_volume',
    }
    
    config_dict = {}
    
    # 转换配置数据
    for db_name, df in config_data.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        
        # 映射sheet名称
        sheet_name = sheet_mapping.get(db_name.lower(), db_name)
        
        # 恢复列名大小写
        df_copy = df.copy()
        df_copy.columns = [column_mapping.get(col.lower(), col) for col in df_copy.columns]
        
        config_dict[sheet_name] = df_copy
        print(f"  ✅ 加载配置表: {sheet_name} ({len(df_copy)} 行)")
    
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
        for sheet in missing_sheets:
            config_dict[sheet] = pd.DataFrame()
    
    # 统一标准化所有配置表的标识符字段
    print(f"🔧 正在标准化标识符字段...")
    standardized_count = 0
    for sheet_name, df in config_dict.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            identifier_cols = ['material', 'location', 'sending', 'receiving', 'sourcing', 
                             'dps_location', 'from_material', 'to_material', 'line', 
                             'delegate_line', 'changeover_id']
            has_identifiers = any(col in df.columns for col in identifier_cols)
            
            if has_identifiers:
                original_dtypes = {col: str(df[col].dtype) for col in identifier_cols if col in df.columns}
                config_dict[sheet_name] = _normalize_identifiers(df)
                new_dtypes = {col: str(config_dict[sheet_name][col].dtype) for col in identifier_cols if col in config_dict[sheet_name].columns}
                
                normalized_fields = []
                for col in identifier_cols:
                    if col in df.columns and original_dtypes[col] != new_dtypes[col]:
                        normalized_fields.append(f"{col}({original_dtypes[col]}→{new_dtypes[col]})")
                
                if normalized_fields:
                    print(f"  🔧 {sheet_name}: {', '.join(normalized_fields)}")
                    standardized_count += 1
    
    if standardized_count > 0:
        print(f"✅ 已标准化 {standardized_count} 个配置表的标识符字段")
    
    # Changeover 配置校验和去重
    if 'M4_ChangeoverMatrix' in config_dict and not config_dict['M4_ChangeoverMatrix'].empty:
        print(f"\n🔧 校验 Changeover Matrix 配置...")
        co_matrix = config_dict['M4_ChangeoverMatrix']
        duplicates = co_matrix[co_matrix.duplicated(subset=['from_material', 'to_material'], keep=False)]
        if not duplicates.empty:
            original_count = len(co_matrix)
            config_dict['M4_ChangeoverMatrix'] = co_matrix.drop_duplicates(
                subset=['from_material', 'to_material'], keep='first'
            )
            print(f"  🔧 已去除 {original_count - len(config_dict['M4_ChangeoverMatrix'])} 条重复记录")
        else:
            print(f"  ✅ Changeover Matrix 无重复定义")
    
    # ChangeoverDefinition 配置校验和去重
    if 'M4_ChangeoverDefinition' in config_dict and not config_dict['M4_ChangeoverDefinition'].empty:
        print(f"\n🔧 校验 Changeover Definition 配置...")
        co_def = config_dict['M4_ChangeoverDefinition']
        duplicates = co_def[co_def.duplicated(subset=['changeover_id', 'line'], keep=False)]
        if not duplicates.empty:
            original_count = len(co_def)
            config_dict['M4_ChangeoverDefinition'] = co_def.drop_duplicates(
                subset=['changeover_id', 'line'], keep='first'
            )
            print(f"  🔧 已去除 {original_count - len(config_dict['M4_ChangeoverDefinition'])} 条重复记录")
        else:
            print(f"  ✅ Changeover Definition 无重复定义")
    
    # Module4 配置表映射
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
    
    return config_dict


def load_configuration(config_path: str) -> dict:
    """加载与标准化配置数据

    目的：
    - 从 Excel 读取所有工作表，补齐缺失的必要表，统一标准化标识符字段，并对 M4 换产配置执行重复性检查与去重映射。

    Args:
        config_path: 配置文件路径（Excel）。

    Returns:
        dict: 标准化后的配置数据字典。

    输入数据：
        - Excel 工作簿；可能存在缺失表或非标准类型的标识符列。

    输出/副作用：
        - 打印加载与标准化日志；对 M4 的配置进行去重与键映射以向后兼容。

    逻辑：
        - 加载→补齐必要表→标准化标识符→检验并去重 Changeover 配置→映射关键表→返回字典。
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
    """运行完整的集成仿真

    目的：
    - 统一编排预验证、续跑检测、Orchestrator 初始化、每日模块执行与状态保存，最终生成汇总与验证报告。

    Args:
        config_path: 配置文件路径（Excel）。
        start_date: 仿真开始日期 (YYYY-MM-DD)。
        end_date: 仿真结束日期 (YYYY-MM-DD)。
        output_base_dir: 输出基础目录。
        force_restart: 强制从头开始（忽略续跑）。

    Returns:
        dict: 包含仿真执行状态、统计与输出路径等的结果字典。

    输入数据：
        - 配置文件、历史输出目录（用于续跑）、各模块的运行所需表。

    输出/副作用：
        - 创建/写入每日输出与日志、保存 Orchestrator 状态、生成最终汇总与验证报告。

    逻辑：
        - 预验证→续跑判断→初始化（新建或恢复）→按日执行模块→每日保存→最终报告与检查→返回结果。
    """
    import time
    from datetime import datetime
    simulation_start_time = time.time()
    simulation_start_datetime = datetime.now()
    
    print("\n" + "=" * 60)
    print("🕐 程序时间信息")
    print("=" * 60)
    print(f"📅 程序开始时间: {simulation_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
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
    print("🎯 初始化Orchestrator")
    orchestrator = create_orchestrator(
        start_date=start_date,
        output_dir=str(orchestrator_output_dir)
    )
    # 设置 open deployment 的清理天数，3代表保留3天
    orchestrator.set_past_due_cleanup_grace_days(100)
    if is_resuming:
        # 续跑模式：恢复状态
        print("🔄 续跑模式：恢复Orchestrator状态")
        restore_orchestrator_state(orchestrator, resume_info['last_complete_date'], output_base_dir)
        
        # 设置空间容量（续跑时也需要重新设置空间容量配置）
        if 'Global_SpaceCapacity' in config_dict and not config_dict['Global_SpaceCapacity'].empty:
            orchestrator.set_space_capacity(config_dict['Global_SpaceCapacity'])
    else:
        # 全新开始：设置初始状态
        print("🆕全新开始：设置初始状态")
        
        # 设置初始库存
        if 'M1_InitialInventory' in config_dict and not config_dict['M1_InitialInventory'].empty:
            orchestrator.initialize_inventory(config_dict['M1_InitialInventory'])
        else:
            print("⚠️未找到初始库存配置，使用空库存")
            orchestrator.initialize_inventory(pd.DataFrame(columns=['material', 'location', 'quantity']))
        
        # 设置空间容量
        if 'Global_SpaceCapacity' in config_dict and not config_dict['Global_SpaceCapacity'].empty:
            orchestrator.set_space_capacity(config_dict['Global_SpaceCapacity'])
        else:
            print("⚠️未找到空间容量配置")
    
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
            
        print(f"{'='*20} {progress_info}: {current_date.strftime('%Y-%m-%d')} {'='*20}")
        
        # 🎲 注意：不在每日开始时重置种子，以匹配ChainSight_Dev的随机数行为
        # ChainSight_Dev没有每日种子重置，随机状态自然演变
        # 全局种子只在仿真开始时设置一次 (在set_module_seeds中)
        
        # ==================== 每日开始：GR入库处理 ====================
        try:
            print("🌅 每日开始状态更新")

            # 🔄 第0步：保存期初库存快照（在任何变动之前）
            print("💾保存期初库存快照...")
            orchestrator.save_beginning_inventory(current_date.strftime('%Y-%m-%d'))
            orchestrator.cleanup_past_due_open_deployments(current_date.strftime('%Y-%m-%d'),grace_days=getattr(orchestrator, "cleanup_grace_days", 0),write_audit=True)

            # 🔄 第1步：处理当日到达的delivery GR (in-transit → inventory)
            print("📦处理当日delivery GR到达...")
            orchestrator._process_delivery_arrivals(current_date.strftime('%Y-%m-%d'))
            
            # 🔄 第2步：处理历史生产的当日入库 (historical production → inventory)
            print("🏭处理历史生产当日入库...")
            current_date_production_gr = load_current_date_production_gr(
                module4_output_dir=str(module_outputs['module4']),
                current_date=current_date,
                start_date=pd.to_datetime(start_date)
            )
            
            if not current_date_production_gr.empty:
                print(f"📦当日需要入库的历史生产: {len(current_date_production_gr)} 条记录")
                # 🔧 标准化标识符字段，确保数据类型一致性
                current_date_production_gr_normalized = _normalize_identifiers(current_date_production_gr)
                orchestrator.process_module4_production(current_date_production_gr_normalized, current_date.strftime('%Y-%m-%d'))
            else:
                print("📦当日无历史生产入库")
                
        except Exception as e:
            print(f"❌ 每日开始处理失败: {e}")
        
        # ==================== 模块运行序列 ====================
        # 初始化每日数据变量
        m1_shipments = pd.DataFrame()
        m4_production = pd.DataFrame()
        m5_deployment_df = pd.DataFrame()
        m6_delivery_df = pd.DataFrame()
        
        try:
            # ========== M1: 订单生成 + 立即库存扣减 ==========
            print("1️⃣ 运行 Module1 - 订单生成")
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
                    print("🚚立即处理M1 shipment，扣减库存...")
                    # 🔧 标准化标识符字段，确保数据类型一致性
                    m1_shipments_normalized = _normalize_identifiers(m1_shipments)
                    orchestrator.process_module1_shipments(m1_shipments_normalized, current_date.strftime('%Y-%m-%d'))
                    print(f"✅ 已扣减 {len(m1_shipments_normalized)} 个shipment的库存")
                
                print(f"✅ Module1 完成 - 生成 {len(m1_result.get('orders_df', []))} 个订单, {len(m1_shipments)} 个发货")
                if m1_result is not None:
                    m1_result['simulation_date'] = current_date
                all_results['module1'].append(m1_result)
            except Exception as e:
                print(f"❌ Module1 失败: {e}")
                m1_shipments = pd.DataFrame()  # 失败时使用空数据
                # 不用continue，让后面的模块继续执行
            
            # ========== M4: 生产计划 + 立即当日生产入库 ==========
            print("2️⃣ 运行 Module4 - 生产计划")
            try:
                # 使用集成模式直接调用 Module4 (改进的解决方案)
                m4_result = run_module4_integrated(
                    config_dict=config_dict,
                    module3_output_dir=str(module_outputs['module3']),
                    simulation_date=current_date,
                    simulation_start=pd.to_datetime(start_date),
                    output_dir=str(module_outputs['module4'])
                )
                
                # 从返回结果中获取 production_df
                m4_production = m4_result.get('production_df', pd.DataFrame())
                
                # 🔄 简化调用：仅持久化“未来 available_date”的生产计划，避免重复当日GR
                if not m4_production.empty and 'available_date' in m4_production.columns:
                    m4_production['available_date'] = pd.to_datetime(m4_production['available_date'])
                    future_plans = m4_production[m4_production['available_date'].dt.normalize() > current_date.normalize()]
                    if not future_plans.empty:
                        print("🗂️ 持久化未来生产计划（不触发当日GR）...")
                        future_plans_normalized = _normalize_identifiers(future_plans)
                        orchestrator.process_module4_production(future_plans_normalized, current_date.strftime('%Y-%m-%d'))
                        print(f"✅已写入未来计划回补: {len(future_plans_normalized)} 条")
                    else:
                        print("📦当日无未来 available_date 的计划需要持久化")
                else:
                    print("📦M4当日未生成生产计划或缺少 available_date 列")
                
                print(f"✅ Module4 完成 - 生成生产计划: {len(m4_production)} 条记录")
                # 存储完整的Module4结果（包含所有输出表）
                m4_result['simulation_date'] = current_date
                all_results['module4'].append(m4_result)
            except Exception as e:
                print(f"❌ Module4 失败: {e}")
                m4_production = pd.DataFrame()  # 失败时使用空数据
            
            # ========== M5: 部署计划 ==========
            print("3️⃣ 运行 Module5 - 部署计划")
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
                            print(f"\n    📦 立即处理M5 deployment，更新open deployment...")
                            orchestrator.process_module5_deployment(m5_deployment_df, current_date.strftime('%Y-%m-%d'))
                            print(f"    ✅ 已更新 {len(m5_deployment_df)} 条部署计划到open deployment")
                            
                            print(f"\n  ✅ Module5 完成 - 生成 {len(m5_deployment_df)} 条有效部署计划")
                        else:
                            print(f"\n  ✅ Module5 完成 - 无有效部署计划")
                    else:
                        print(f"\n  ✅ Module5 完成 - 部署计划为空")
                else:
                    print(f"\n  ✅ Module5 完成 - 无返回结果")
                
                if m5_result is not None:
                    m5_result['simulation_date'] = current_date
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
                        print(f"\n    🚛 立即处理M6 delivery，更新库存/open deployment/in-transit...")
                        # 🔧 标准化标识符字段，确保数据类型一致性
                        m6_delivery_df_normalized = _normalize_identifiers(m6_delivery_df)
                        orchestrator.process_module6_delivery(m6_delivery_df_normalized, current_date.strftime('%Y-%m-%d'))
                        print(f"    ✅ 已处理 {len(m6_delivery_df_normalized)} 条delivery计划，更新相关状态")
                    
                    print(f"\n  ✅ Module6 完成 - 生成 {len(m6_delivery_df)} 条交付计划")
                else:
                    print(f"\n  ✅ Module6 完成 - 无交付计划")
                    m6_delivery_df = pd.DataFrame()
                
                if m6_result is not None:
                    m6_result['simulation_date'] = current_date
                all_results['module6'].append(m6_result)
            except KeyError as ke:
                print(f"  ❌ Module6 失败: 缺少列 '{ke}' - {str(ke)}")
                logging.error(f"Module6 KeyError: {ke}", exc_info=True)
                # 不用continue，让后面的模块继续执行
                m6_delivery_df = pd.DataFrame()  # 设置默认值
            except Exception as e:
                print(f"  ❌ Module6 失败: {e}")
                logging.error(f"Module6 Exception: {e}", exc_info=True)
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
                        output_dir=str(module_outputs['module3']),
                        skip_file_output=False,
                        module1_result=m1_result  # 直接从内存传递Module1输出
                    )
                print(f"  ✅ Module3 完成")
                if m3_result is not None:
                    m3_result['simulation_date'] = current_date
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
                print(f"📊 当日统计: {stats}")
                print("💾 Orchestrator 状态已保存")
                
            except Exception as e:
                print(f"❌ 每日状态保存失败: {e}")
                # 不用continue，让他继续到下一天
            
            print(f"✅ 第 {i} 天处理完成")
            
        except Exception as e:
            print(f"❌ 第 {i} 天处理失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 生成最终报告
    print("📊 仿真完成报告")
    print("=" * 60)
    print(f"仿真期间: {start_date} 到 {end_date} ({len(sim_dates)} 天)")
    print(f"输出目录: {output_base_dir}")
    
    for module_name, results in all_results.items():
        print(f"{module_name.upper()}: {len(results)} 天成功处理")
    
    # 进行库存平衡检查
    print("🔎 正在进行库存平衡检查...")
    validation_manager = ValidationManager(str(output_dir))
    inventory_checker = InventoryBalanceChecker(validation_manager, orchestrator)
    balance_passed = inventory_checker.validate_inventory_consistency(start_date, end_date)
    
    if balance_passed:
        print("✅ 库存平衡检查通过")
    else:
        print("⚠️库存平衡检查发现问题，请查看验证报告")
    
    # 生成汇总报告
    print("📊 正在生成汇总报告...")
    summary_generator = SummaryReportGenerator(str(output_dir), config_dict)
    summary_reports = summary_generator.generate_all_reports(start_date, end_date)
    
    # 写入库存平衡检查报告
    balance_report_path = validation_manager.write_report()
    
    # 获取最终Orchestrator统计
    final_date = sim_dates[-1].strftime('%Y-%m-%d')
    final_stats = orchestrator.get_summary_statistics(final_date)
    print("🎯 最终Orchestrator状态:")
    for key, value in final_stats.items():
        print(f"{key}: {value}")
    
    # 计算总运行时间
    simulation_end_time = time.time()
    total_runtime_seconds = simulation_end_time - simulation_start_time
    
    # 格式化运行时间
    hours, remainder = divmod(total_runtime_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours >= 1:
        runtime_str = f"{int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒"
    elif minutes >= 1:
        runtime_str = f"{int(minutes)}分钟 {seconds:.2f}秒"
    else:
        runtime_str = f"{seconds:.2f}秒"
    
    if is_resuming:
        total_processed = resume_info['days_completed'] + len(sim_dates)
        print("🎉 续跑仿真完成!")
        print(f"本次处理: {len(sim_dates)} 天")
        print(f"总共完成: {total_processed} 天")
    else:
        total_processed = len(sim_dates)
        print("🎉 集成仿真完成!")
        print(f"总共处理: {total_processed} 天")
    
    # 输出运行时间统计
    simulation_end_datetime = datetime.now()
    print("\n" + "=" * 60)
    print("🕐 程序时间统计")
    print("=" * 60)
    print(f"📅 开始时间: {simulation_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📅 结束时间: {simulation_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  总运行时间: {runtime_str}")
    print(f"📊 平均每天耗时: {total_runtime_seconds / len(sim_dates):.2f}秒")
    print("=" * 60)
    
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


def run_integrated_simulation_from_dict(
    config_data: dict,
    config_name: str,
    start_date: str,
    end_date: str,
    output_base_dir: str = "./integrated_output",
    skip_validation: bool = True
) -> dict:
    """从DataFrame字典运行集成仿真（数据库模式专用）
    
    目的：
    - 直接接收DataFrame字典，无需创建临时Excel文件
    - 使用DuckDB内存处理配置数据
    - 用于数据库模式运行
    
    Args:
        config_data: 配置数据字典 {sheet_name: DataFrame}
        config_name: 配置名称（用于日志）
        start_date: 仿真开始日期 (YYYY-MM-DD)
        end_date: 仿真结束日期 (YYYY-MM-DD)
        output_base_dir: 输出基础目录
        skip_validation: 是否跳过预验证（数据库数据已验证）
    
    Returns:
        dict: 仿真结果字典
    """
    import time
    simulation_start_time = time.time()
    simulation_start_datetime = datetime.now()
    
    print("\n" + "=" * 60)
    print("🕐 程序时间信息")
    print("=" * 60)
    print(f"📅 程序开始时间: {simulation_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🚀 开始集成仿真 (数据库模式): {start_date} 到 {end_date}")
    print(f"📋 配置: {config_name}")
    print("=" * 60)
    
    # 🦆 启用DuckDB内存模式（加速模块间数据传递）
    _ensure_memory_store_imported()
    if _enable_memory_mode:
        _enable_memory_mode(memory_limit="4GB")
        print("🦆 DuckDB内存模式已启用（4GB限制）")
    
    # 跳过预验证（数据库数据已经过验证）
    if skip_validation:
        print("✅ 跳过预验证（数据库模式）")
    
    # 创建输出目录
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
    
    # 🦆 使用DuckDB处理配置数据（无需临时Excel文件）
    print("🦆 DuckDB处理配置数据...")
    config_dict = load_configuration_from_dict(config_data, config_name)
    
    # 设置全局随机种子
    global_seed = set_module_seeds(config_dict)
    
    # 初始化时间管理器
    time_manager = initialize_time_manager(start_date)
    
    # 初始化Orchestrator
    print("🎯 初始化Orchestrator")
    orch = create_orchestrator(
        start_date=start_date,
        output_dir=str(orchestrator_output_dir)
    )
    orch.set_past_due_cleanup_grace_days(100)
    
    # 全新开始：设置初始状态
    print("🆕全新开始：设置初始状态")
    
    # 设置初始库存
    if 'M1_InitialInventory' in config_dict and not config_dict['M1_InitialInventory'].empty:
        orch.initialize_inventory(config_dict['M1_InitialInventory'])
    else:
        print("⚠️未找到初始库存配置，使用空库存")
        orch.initialize_inventory(pd.DataFrame(columns=['material', 'location', 'quantity']))
    
    # 设置空间容量
    if 'Global_SpaceCapacity' in config_dict and not config_dict['Global_SpaceCapacity'].empty:
        orch.set_space_capacity(config_dict['Global_SpaceCapacity'])
    else:
        print("⚠️未找到空间容量配置")
    
    # 生成仿真日期范围
    sim_dates = pd.date_range(start_date, end_date, freq='D')
    print(f"📅 仿真日期范围: {len(sim_dates)} 天")
    
    # 🦆 完全内存模式：存储前一天的Module3结果供Module4使用
    previous_day_m3_result: Optional[Dict[str, Any]] = None
    
    # 🦆 完全内存模式：存储历史M1订单数据供累积使用（与local模式保持一致）
    accumulated_m1_orders: pd.DataFrame = pd.DataFrame()
    
    # 🦆 完全内存模式：存储历史M4生产计划数据供累积使用（用于历史生产入库）
    accumulated_m4_production: pd.DataFrame = pd.DataFrame()
    
    # 每日循环执行
    all_results = {
        'module1': [],
        'module3': [],
        'module4': [], 
        'module5': [],
        'module6': []
    }
    
    for i, current_date in enumerate(sim_dates, 1):
        print(f"{'='*20} 第 {i}/{len(sim_dates)} 天: {current_date.strftime('%Y-%m-%d')} {'='*20}")
        
        # 🎲 注意：不在每日开始时重置种子，以匹配本地模式和ChainSight_Dev的随机数行为
        # ChainSight_Dev没有每日种子重置，随机状态自然演变
        # 全局种子只在仿真开始时设置一次 (在set_module_seeds中)
        
        # ==================== 每日开始：GR入库处理 ====================
        try:
            print("🌅 每日开始状态更新")
            print("💾保存期初库存快照...")
            orch.save_beginning_inventory(current_date.strftime('%Y-%m-%d'))
            orch.cleanup_past_due_open_deployments(current_date.strftime('%Y-%m-%d'), grace_days=getattr(orch, "cleanup_grace_days", 0), write_audit=True)
            
            print("📦处理当日delivery GR到达...")
            orch._process_delivery_arrivals(current_date.strftime('%Y-%m-%d'))
            
            print("🏭处理历史生产当日入库...")
            # 🦆 使用内存中累积的M4生产计划数据，而非读取文件（DB模式skip_file_output=True不生成文件）
            if not accumulated_m4_production.empty and 'available_date' in accumulated_m4_production.columns:
                accumulated_m4_production['available_date'] = pd.to_datetime(accumulated_m4_production['available_date'])
                current_date_production_gr = accumulated_m4_production[
                    accumulated_m4_production['available_date'].dt.normalize() == current_date.normalize()
                ].copy()
                
                # 确保返回所需列（与load_current_date_production_gr函数一致）
                required_cols = ['material', 'location', 'line', 'simulation_date', 'available_date', 'produced_qty']
                available_cols = [c for c in required_cols if c in current_date_production_gr.columns]
                current_date_production_gr = current_date_production_gr[available_cols]
            else:
                current_date_production_gr = pd.DataFrame()
            
            if not current_date_production_gr.empty:
                print(f"📦当日需要入库的历史生产: {len(current_date_production_gr)} 条记录")
                current_date_production_gr_normalized = _normalize_identifiers(current_date_production_gr)
                orch.process_module4_production(current_date_production_gr_normalized, current_date.strftime('%Y-%m-%d'))
            else:
                print("📦当日无历史生产入库")
                
        except Exception as e:
            print(f"❌ 每日开始处理失败: {e}")
        
        # ==================== 模块运行序列 ====================
        m1_shipments = pd.DataFrame()
        m4_production = pd.DataFrame()
        m5_deployment_df = pd.DataFrame()
        m6_delivery_df = pd.DataFrame()
        
        try:
            # ========== M1: 订单生成 ==========
            print("1️⃣ 运行 Module1 - 订单生成")
            try:
                m1_result = module1.run_daily_order_generation(
                    config_dict=config_dict,
                    simulation_date=current_date,
                    output_dir=str(module_outputs['module1']),
                    orchestrator=orch,
                    skip_file_output=True,  # 🦆 完全内存模式：跳过文件输出
                    previous_orders_df=accumulated_m1_orders if not accumulated_m1_orders.empty else None  # 🦆 传递累积的历史订单
                )
                m1_shipments = m1_result.get('shipment_df', pd.DataFrame())
                
                if not m1_shipments.empty:
                    print("🚚立即处理M1 shipment，扣减库存...")
                    m1_shipments_normalized = _normalize_identifiers(m1_shipments)
                    orch.process_module1_shipments(m1_shipments_normalized, current_date.strftime('%Y-%m-%d'))
                    print(f"✅ 已扣减 {len(m1_shipments_normalized)} 个shipment的库存")
                
                # 🦆 更新累积的历史订单数据（用于下一天的M1）
                # 🔧 修复：使用 all_orders_for_next_day（累积订单）而不是 orders_df（仅当日订单）
                # orders_df 只包含当日新生成的订单，会导致历史订单丢失
                # all_orders_for_next_day 包含所有未来到期的订单（与Excel模式一致）
                all_orders = m1_result.get('all_orders_for_next_day', pd.DataFrame())
                if not all_orders.empty:
                    accumulated_m1_orders = all_orders.copy()
                    print(f"🦆 累积历史订单更新完成，当前总条目: {len(accumulated_m1_orders)}")
                
                print(f"✅ Module1 完成 - 生成 {len(m1_result.get('orders_df', []))} 个订单, {len(m1_shipments)} 个发货")
                if m1_result is not None:
                    m1_result['simulation_date'] = current_date
                all_results['module1'].append(m1_result)
            except Exception as e:
                print(f"❌ Module1 失败: {e}")
                m1_shipments = pd.DataFrame()
            
            # ========== M4: 生产计划 ==========
            print("2️⃣ 运行 Module4 - 生产计划")
            try:
                m4_result = run_module4_integrated(
                    config_dict=config_dict,
                    module3_output_dir=str(module_outputs['module3']),
                    simulation_date=current_date,
                    simulation_start=pd.to_datetime(start_date),
                    output_dir=str(module_outputs['module4']),
                    skip_file_output=True,  # 🦆 完全内存模式：跳过文件输出
                    module3_result=previous_day_m3_result  # 🦆 使用前一天的Module3内存数据
                )
                
                # 从返回结果中获取 production_df
                m4_production = m4_result.get('production_df', pd.DataFrame())
                
                if not m4_production.empty and 'available_date' in m4_production.columns:
                    m4_production['available_date'] = pd.to_datetime(m4_production['available_date'])
                    future_plans = m4_production[m4_production['available_date'].dt.normalize() > current_date.normalize()]
                    if not future_plans.empty:
                        print("🗂️ 持久化未来生产计划...")
                        future_plans_normalized = _normalize_identifiers(future_plans)
                        orch.process_module4_production(future_plans_normalized, current_date.strftime('%Y-%m-%d'))
                        print(f"✅已写入未来计划: {len(future_plans_normalized)} 条")
                else:
                    print(f"📦M4当日未生成生产计划或缺少 available_date 列")
                
                print(f"✅ Module4 完成 - 生成生产计划: {len(m4_production)} 条记录")
                # 存储完整的Module4结果（包含所有输出表）
                m4_result['simulation_date'] = current_date
                all_results['module4'].append(m4_result)
                
                # 🦆 累积历史M4生产计划数据（用于历史生产入库）
                if not m4_production.empty:
                    m4_for_accumulation = m4_production.copy()
                    m4_for_accumulation['source_date'] = current_date
                    accumulated_m4_production = pd.concat([accumulated_m4_production, m4_for_accumulation], ignore_index=True)
                    print(f"🦆 累积历史M4生产计划更新完成，当前总条目: {len(accumulated_m4_production)}")
            except Exception as e:
                print(f"❌ Module4 失败: {e}")
                m4_production = pd.DataFrame()
            
            # ========== M5: 部署计划 ==========
            print("3️⃣ 运行 Module5 - 部署计划")
            try:
                m5_result = module5.main(
                    config_dict=config_dict,
                    module1_output_dir=str(module_outputs['module1']),
                    module4_output_path=str(module_outputs['module4'] / f"Module4Output_{current_date.strftime('%Y%m%d')}.xlsx"),  # fallback路径
                    orchestrator=orch,
                    current_date=current_date.strftime('%Y-%m-%d'),
                    output_path=str(module_outputs['module5'] / f"Module5Output_{current_date.strftime('%Y%m%d')}.xlsx"),
                    skip_file_output=True,  # 🦆 完全内存模式：跳过文件输出
                    module1_result=m1_result,  # 🦆 直接传递Module1内存数据
                    module4_result=m4_result   # 🦆 直接传递Module4内存数据
                )
                
                if m5_result and 'deployment_plan' in m5_result:
                    deployment_plan_df = m5_result['deployment_plan']
                    if not deployment_plan_df.empty:
                        valid_deployment = deployment_plan_df[
                            (deployment_plan_df['deployed_qty_invCon'] > 0) & 
                            (deployment_plan_df['deployed_qty_invCon'].notna()) &
                            (deployment_plan_df['sending'] != deployment_plan_df['receiving'])
                        ].copy()
                        
                        print(f"    🎯 有效部署计划: {len(valid_deployment)}/{len(deployment_plan_df)} 条")
                        
                        if not valid_deployment.empty:
                            if 'deployed_qty' in valid_deployment.columns:
                                m5_deployment_df = valid_deployment[[
                                    'material', 'sending', 'receiving', 'date', 'deployed_qty', 'demand_element'
                                ]].rename(columns={'date': 'planned_deployment_date'})
                            else:
                                m5_deployment_df = valid_deployment.rename(columns={
                                    'date': 'planned_deployment_date',
                                    'deployed_qty_invCon': 'deployed_qty'
                                })[['material', 'sending', 'receiving', 'planned_deployment_date', 'deployed_qty', 'demand_element']]
                            
                            m5_deployment_df = _normalize_identifiers(m5_deployment_df)
                            print("\n    📦 立即处理M5 deployment，更新open deployment...")
                            orch.process_module5_deployment(m5_deployment_df, current_date.strftime('%Y-%m-%d'))
                            print(f"    ✅ 已更新 {len(m5_deployment_df)} 条部署计划到open deployment")
                
                print(f"\n  ✅ Module5 完成 - 生成 {len(valid_deployment) if 'valid_deployment' in dir() else 0} 条有效部署计划")
                if m5_result is not None:
                    m5_result['simulation_date'] = current_date
                all_results['module5'].append(m5_result)
            except Exception as e:
                print(f"❌ Module5 失败: {e}")
                m5_deployment_df = pd.DataFrame()
            
            # ========== M6: 物流执行 ==========
            print("4️⃣ 运行 Module6 - 物流执行")
            try:
                m6_result = module6.run_daily_physical_flow(
                    config_dict=config_dict,
                    orchestrator=orch,
                    current_date=current_date,
                    output_dir=str(module_outputs['module6']),
                    max_wait_days=30,
                    random_seed=config_dict.get('M6_RandomSeed', 42),
                    skip_file_output=True  # 🦆 完全内存模式
                )
                
                if m6_result and 'delivery_plan' in m6_result:
                    m6_delivery_df = m6_result.get('delivery_plan', pd.DataFrame())
                    if not m6_delivery_df.empty:
                        m6_delivery_normalized = _normalize_identifiers(m6_delivery_df)
                        orch.process_module6_delivery(m6_delivery_normalized, current_date.strftime('%Y-%m-%d'))
                        print(f"    ✅ 已处理 {len(m6_delivery_normalized)} 条delivery计划")
                
                print(f"\n  ✅ Module6 完成 - 生成 {len(m6_delivery_df) if 'm6_delivery_df' in dir() else 0} 条交付计划")
                if m6_result is not None:
                    m6_result['simulation_date'] = current_date
                all_results['module6'].append(m6_result)
            except Exception as e:
                print(f"❌ Module6 失败: {e}")
                m6_delivery_df = pd.DataFrame()
            
            # ========== M3: 净需求计算 ==========
            print("5️⃣ 运行 Module3 - 净需求计算")
            try:
                m3_result = module3.run_integrated_mode(
                    module1_output_dir=str(module_outputs['module1']),
                    orchestrator=orch,
                    config_dict=config_dict,
                    start_date=current_date.strftime('%Y-%m-%d'),
                    end_date=current_date.strftime('%Y-%m-%d'),
                    output_dir=str(module_outputs['module3']),
                    skip_file_output=True,  # 🦆 完全内存模式 - M3结果通过previous_day_m3_result传递给下一天M4
                    module1_result=m1_result  # 🦆 直接从内存传递Module1输出（与local模式一致）
                )
                print(f"  ✅ Module3 完成")
                if m3_result is not None:
                    m3_result['simulation_date'] = current_date
                all_results['module3'].append(m3_result)
                
                # 🦆 存储M3结果，供下一天M4使用（完全内存模式）
                previous_day_m3_result = m3_result
            except Exception as e:
                print(f"❌ Module3 失败: {e}")
            
        except Exception as e:
            print(f"❌ 当日模块执行失败: {e}")
            import traceback
            traceback.print_exc()
        
        # ==================== 每日结束：状态保存 ====================
        try:
            print("💾 每日结束状态保存")
            # 保存期末库存快照
            orch.save_ending_inventory(current_date.strftime('%Y-%m-%d'))
            # 输出每日库存汇总
            orch.output_daily_inventory_summary(current_date.strftime('%Y-%m-%d'))
            # 保存每日状态
            orch.save_daily_state(current_date.strftime('%Y-%m-%d'))
            # 获取当日统计
            stats = orch.get_summary_statistics(current_date.strftime('%Y-%m-%d'))
            print(f"📊 当日统计: {stats}")
            print(f"✅ 第 {i} 天处理完成")
        except Exception as e:
            print(f"❌ 每日状态保存失败: {e}")
    
    # 仿真结束统计
    total_runtime_seconds = time.time() - simulation_start_time
    if total_runtime_seconds >= 60:
        minutes = int(total_runtime_seconds // 60)
        seconds = total_runtime_seconds % 60
        runtime_str = f"{minutes}分钟 {seconds:.2f}秒"
    else:
        runtime_str = f"{total_runtime_seconds:.2f}秒"
    
    # 生成汇总报告
    print("📊 正在生成汇总报告...")
    try:
        report_generator = SummaryReportGenerator(
            output_base_dir=str(output_dir),
            config_dict=config_dict
        )
        # start_date 和 end_date 在本函数中已是字符串格式
        summary_reports = report_generator.generate_all_reports(
            start_date=start_date,
            end_date=end_date
        )
        print(f"✅ 汇总报告生成完成，输出目录: {output_dir / 'summary'}")
    except Exception as e:
        print(f"⚠️ 汇总报告生成失败: {e}")
        import traceback
        traceback.print_exc()
        summary_reports = {}
    
    # 最终统计
    try:
        final_stats = orch.get_summary_statistics(end_date)
        print(f"🎯 最终Orchestrator状态:")
        for key, value in final_stats.items():
            print(f"{key}: {value}")
    except Exception as e:
        print(f"⚠️ 获取最终统计失败: {e}")
        final_stats = {}
    
    print("🎉 集成仿真完成!")
    print(f"总共处理: {len(sim_dates)} 天")
    
    # 🦆 禁用DuckDB内存模式并打印统计
    _ensure_memory_store_imported()
    if _is_memory_mode_enabled and _is_memory_mode_enabled():
        if _get_data_store:
            store = _get_data_store()
            if store:
                print("\n" + "=" * 60)
                print("🦆 DuckDB内存模式统计")
                print("=" * 60)
                store.print_stats()
        if _disable_memory_mode:
            _disable_memory_mode()
            print("🦆 DuckDB内存模式已禁用")
    
    # 输出运行时间统计
    simulation_end_datetime = datetime.now()
    print("\n" + "=" * 60)
    print("🕐 程序时间统计")
    print("=" * 60)
    print(f"📅 开始时间: {simulation_start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📅 结束时间: {simulation_end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  总运行时间: {runtime_str}")
    print(f"📊 平均每天耗时: {total_runtime_seconds / len(sim_dates):.2f}秒")
    print("=" * 60)
    
    return {
        'validation_passed': True,
        'simulation_completed': True,
        'dates_processed_this_run': len(sim_dates),
        'results': all_results,
        'final_stats': final_stats,
        'output_directory': str(output_dir),
        'summary_reports': summary_reports
    }


def main():
    """主函数 - 命令行入口执行集成仿真

    目的：
    - 解析命令行参数，进行存在性检查与默认值处理，支持仅检查续跑或执行完整仿真。

    输入/输出/逻辑：
    - 解析参数→检查配置文件→构造默认输出目录→可选续跑检查→调用 `run_integrated_simulation` 并打印结果或错误。
    """
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