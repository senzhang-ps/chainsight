"""
resume.py

断点续跑与 checkpoint 检测函数模块。
"""

import pandas as pd
from pathlib import Path
from pandas.errors import EmptyDataError, ParserError
import logging

from .normalize import _normalize_identifiers

logger = logging.getLogger(__name__)


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
        - 先通过只读表头做轻量可读性校验，再对关键库存文件做一次完整读取校验；不更改任何状态，仅打印诊断信息。

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
        
        # 轻量验证：检查文件存在且表头可读（只读表头以避免加载大型文件；仅有表头也视为可接受）
        all_files_exist = True
        for file_name in required_files:
            file_path = orchestrator_dir / file_name

            try:
                if not file_path.exists():
                    logger.warning("缺失文件: %s", file_path)
                    all_files_exist = False
                    break

                # 使用 nrows=0 只读表头（即便没有数据也不会尝试读取行）
                # 这样“只有表头”或“无数据”不会中止断点续跑判断
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
            # 再次验证关键文件可完整读取（允许空表，但文件结构必须正确）
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
    """检查是否可以断点续跑，返回断点续跑信息

    目的：
        - 综合最后完整日期与目标区间，计算断点续跑起点与剩余天数，指示是否已全部完成或可继续。

    Args:
        output_base_dir: 输出基础目录。
        start_date: 仿真开始日期（YYYY-MM-DD）。
        end_date: 仿真结束日期（YYYY-MM-DD）。

    Returns:
        dict: 包含是否可断点续跑、最后完整日期、断点续跑起始日期、已完成和剩余天数等的字典。

    输入/输出/逻辑：
        - 调用 `detect_last_complete_date` 获取完整日期；
        - 若已覆盖到 `end_date`，返回已完成标记；否则计算断点续跑起点（完整日期+1）与剩余范围。
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
    
    # 计算断点续跑信息
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
