"""
simulation_db.py

数据库模式仿真入口点模块。
"""

import pandas as pd
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .. import orchestrator
from ..orchestrator import create_orchestrator
from ...utils.time_manager import initialize_time_manager
from ...services.summary_report_generator import SummaryReportGenerator
from ...modules import module1, module3, module4, module5, module6
from ...utils.defaults import M6_MAX_WAIT_DAYS, M6_RANDOM_SEED

from ...utils.normalization import normalize_identifiers
from .config_loader import load_configuration_from_dict
from .seed import set_module_seeds
from .production_runner import run_module4_integrated
from .runtime_state import DbRuntimeState
from .memory_store import (_ensure_memory_store_imported, _enable_memory_mode,
                          _disable_memory_mode, _is_memory_mode_enabled, _get_data_store)
from .db_helpers import _flush_batch_to_db


def run_integrated_simulation_from_dict(
    config_data: dict,
    config_name: str,
    start_date: str,
    end_date: str,
    output_base_dir: str = "./integrated_output",
    skip_validation: bool = True,
    resume: bool = False,
    db=None,
    run_key: str = None,
    batch_size: int = 10,
    run_id: str = None,
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
    
    
    # 🦆 启用DuckDB内存模式（加速模块间数据传递）
    _ensure_memory_store_imported()
    if _enable_memory_mode:
        _enable_memory_mode()  # 使用动态90%系统内存
        # 获取实际内存限制用于日志显示
        try:
            from src.utils.resource_config import get_optimal_memory
            memory_limit = get_optimal_memory()
        except ImportError:
            memory_limit = "4GB"
    
    # 跳过预验证（数据库数据已经过验证）
    if skip_validation:
        pass
    
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
    config_dict = load_configuration_from_dict(config_data, config_name)
    
    # 设置全局随机种子
    global_seed = set_module_seeds(config_dict)
    
    # 初始化时间管理器
    time_manager = initialize_time_manager(start_date)
    
    # 初始化Orchestrator
    orch = create_orchestrator(
        start_date=start_date,
        output_dir=str(orchestrator_output_dir)
    )
    # 设置 open deployment 的清理天数，与文件模式保持一致（100天）
    orch.set_past_due_cleanup_grace_days(100)
    # ===== 断点续跑 / 全新运行 分支 =====
    checkpoint = None
    actual_start_date = start_date   # 实际开始日（断点续跑时向后推进）
    run_id_override = run_id           # 优先使用调用方传入的 run_id；断点续跑时会被 checkpoint 覆盖

    # 确保 checkpoint 表存在并包含所有列（幂等）
    if db is not None:
        from pgsql_db.checkpoint import ensure_checkpoint_table, ensure_m4_state_table
        ensure_checkpoint_table(db)
        ensure_m4_state_table(db)

    if resume and db is not None:
        from pgsql_db.checkpoint import load_checkpoint, next_day, deserialize_orchestrator_state
        _rk = run_key or config_name
        checkpoint = load_checkpoint(db, _rk, start_date=start_date, end_date=end_date)

    # DbRuntimeState：内存跨天状态（替代 M4 产线状态、已分配产能及 M3→M4 数据传递中的临时文件中转）。
    runtime_state = DbRuntimeState()

    # m1_previous_orders 初始化：全新运行默认 None；断点续跑时由 checkpoint 恢复覆盖。
    # 必须在 if checkpoint 分支之前声明，防止后续无条件赋值覆盖已恢复的值。
    m1_previous_orders = None

    if checkpoint:
        run_id_override = checkpoint['run_id']
        actual_start_date = next_day(checkpoint['last_batch_end'])
        if pd.to_datetime(actual_start_date) > pd.to_datetime(end_date):
            return {
                'validation_passed': True,
                'simulation_completed': True,
                'dates_processed_this_run': 0,
                'results': {'module1': [], 'module3': [], 'module4': [], 'module5': [], 'module6': []},
                'final_stats': {},
                'output_directory': str(output_dir),
                'summary_reports': {},
                'config_dict': config_dict,
                'run_id': run_id_override,
            }
        deserialize_orchestrator_state(orch, checkpoint['orch_state_json'])
        # 从恢复后的 flat list 重建 *_by_date 索引字典
        # deserialize_orchestrator_state 恢复了 delivery_gr / production_gr / shipment_log /
        # delivery_shipment_log 四个 flat list，但对应的 *_by_date 索引未序列化。
        # 这些索引是 views / inventory_change_log / DB 写入 / summary 的主要查询入口，
        # 若不重建，resume 后第一天的 orchestrator DB 数据将缺失。
        _by_date_lists = [
            ('production_gr', 'production_gr_by_date'),
            ('delivery_gr', 'delivery_gr_by_date'),
            ('shipment_log', 'shipment_log_by_date'),
            ('delivery_shipment_log', 'delivery_shipment_log_by_date'),
        ]
        for _flat_attr, _idx_attr in _by_date_lists:
            _flat = getattr(orch, _flat_attr, [])
            _idx = {}
            for _rec in _flat:
                _dk = _rec.get('date', '')
                if isinstance(_dk, pd.Timestamp):
                    _dk = _dk.strftime('%Y-%m-%d')
                elif hasattr(_dk, 'strftime'):
                    _dk = _dk.strftime('%Y-%m-%d')
                else:
                    _dk = str(_dk)[:10]  # 'YYYY-MM-DD' 截取
                if _dk not in _idx:
                    _idx[_dk] = []
                _idx[_dk].append(_rec)
            setattr(orch, _idx_attr, _idx)
        # 🔧 从 checkpoint 恢复 m1_previous_orders（断点续跑可靠性关键）
        from pgsql_db.checkpoint import deserialize_m1_previous_orders
        m1_previous_orders = deserialize_m1_previous_orders(
            checkpoint['orch_state_json'].get('m1_previous_orders')
        )
        if m1_previous_orders is not None:
            pass
        else:
            # 兼容兜底：旧版 checkpoint 中无 m1_previous_orders 字段（序列化前中断）
            # 从 module1_output_orderlog 按 run_id + sim_date < actual_start_date 读取历史订单
            if db is not None:
                try:
                    _prev_date = checkpoint['last_batch_end']  # 上一批次结束日（e.g. "2025-12-15"）
                    _fallback_sql = (
                        "SELECT * FROM module1_output_orderlog "
                        "WHERE run_id = %s AND sim_date <= %s"
                    )
                    _fallback_rows = db.execute_query(_fallback_sql, (run_id_override, _prev_date))
                    if _fallback_rows:
                        m1_previous_orders = pd.DataFrame(_fallback_rows)
                        # 规范化日期列
                        from pgsql_db.checkpoint import deserialize_m1_previous_orders as _deserialize
                        _DATE_COLS = {'simulation_date', 'order_date', 'delivery_date',
                                      'ship_date', 'available_date', 'date', 'created_date'}
                        for _col in _DATE_COLS & set(m1_previous_orders.columns):
                            m1_previous_orders[_col] = pd.to_datetime(
                                m1_previous_orders[_col], errors='coerce'
                            )
                    else:
                        pass
                except Exception as _e:
                    pass
            # 安全防护：续跑模式下若历史订单仍无法恢复，中止运行以避免静默错算
            if m1_previous_orders is None:
                raise RuntimeError(
                    f"[续跑安全中止] 断点续跑模式下无法恢复 m1_previous_orders。\n"
                    f"  - checkpoint run_id: {run_id_override}\n"
                    f"  - last_batch_end: {checkpoint.get('last_batch_end')}\n"
                    f"请检查 checkpoint 数据完整性或 module1_output_orderlog 表内容后重试。"
                )
        # 从 checkpoint 恢复 DbRuntimeState（M4 line states, capacity, M3 result）
        _db_rt_data = checkpoint['orch_state_json'].get('db_runtime_state')
        if _db_rt_data:
            runtime_state = DbRuntimeState.from_dict(_db_rt_data)
        else:
            # 向后兼容：旧版 checkpoint 在 sim_m4_state 表中使用基于文件的 M4 状态。
            # 回退到从 DB 恢复文件（旧版路径）。
            if db is not None:
                from pgsql_db.checkpoint import restore_m4_state_files
                _restored = restore_m4_state_files(
                    db=db,
                    run_id=run_id_override,
                    m4_output_dir=str(module_outputs['module4']),
                )
                if _restored > 0:
                    pass
    else:
        # 设置初始库存
        if 'M1_InitialInventory' in config_dict and not config_dict['M1_InitialInventory'].empty:
            orch.initialize_inventory(config_dict['M1_InitialInventory'])
        else:
            orch.initialize_inventory(pd.DataFrame(columns=['material', 'location', 'quantity']))
        # 设置空间容量
        if 'Global_SpaceCapacity' in config_dict and not config_dict['Global_SpaceCapacity'].empty:
            orch.set_space_capacity(config_dict['Global_SpaceCapacity'])
        else:
            pass

    # 生成仿真日期范围
    sim_dates = pd.date_range(start_date, end_date, freq='D')
    sim_dates_to_run = pd.date_range(actual_start_date, end_date, freq='D')
    
    
    # 数据库模式保持与文件模式一致的核心计算顺序；模块间状态优先通过内存对象与 Orchestrator 视图传递，不再依赖临时文件中转。
    
    # 每日循环执行
    all_results = {
        'module1': [],
        'module3': [],
        'module4': [], 
        'module5': [],
        'module6': []
    }

    # 新增：数据库批量写入所需变量
    if db is not None:
        from pgsql_db.module_data_writer import ModuleDataWriter
        writer = ModuleDataWriter(db, config_name=config_name)
    else:
        writer = None
    batch_results = {mod: [] for mod in all_results}
    batch_start_date = actual_start_date
    # 追踪每天每模块的失败情况，防止批次内数据静默缺失
    batch_failed_modules = []  # list of (date_str, module_name, error_msg)
    # NOTE: m1_previous_orders 已在 checkpoint 分支之前初始化（=None），
    # 续跑时由 checkpoint 恢复覆盖，不在此重复赋值。


    for i, current_date in enumerate(sim_dates_to_run, 1):
        
        # 🎲 注意：不在每日开始时重置种子，以匹配本地模式和ChainSight_Dev的随机数行为
        # ChainSight_Dev没有每日种子重置，随机状态自然演变
        # 全局种子只在仿真开始时设置一次 (在set_module_seeds中)
        
        # ==================== 每日开始：GR入库处理 ====================
        try:
            orch.save_beginning_inventory(current_date.strftime('%Y-%m-%d'))
            runtime_state.cleanup_audit_df = orch.cleanup_past_due_open_deployments(current_date.strftime('%Y-%m-%d'), grace_days=getattr(orch, "cleanup_grace_days", 0), write_audit=True)
            
            orch._process_delivery_arrivals(current_date.strftime('%Y-%m-%d'))
            
            # [DB-MEM] 从 orchestrator production_plan_backlog 查询当日GR，替代 xlsx 扫描
            _backlog = getattr(orch, 'production_plan_backlog', [])
            if _backlog:
                _backlog_list = _backlog if isinstance(_backlog, list) else (
                    _backlog.to_dict('records') if hasattr(_backlog, 'to_dict') else list(_backlog)
                )
                _backlog_df = pd.DataFrame(_backlog_list)
                if not _backlog_df.empty and 'available_date' in _backlog_df.columns:
                    _backlog_df['available_date'] = pd.to_datetime(_backlog_df['available_date']).dt.normalize()
                    current_date_production_gr = _backlog_df[
                        _backlog_df['available_date'] == current_date.normalize()
                    ].copy()
                    # 对齐列名，与 load_current_date_production_gr 的返回列一致
                    _needed = ['material', 'location', 'available_date', 'produced_qty']
                    # production_plan_backlog 可能用 'quantity' 列而非 'produced_qty'
                    if 'produced_qty' not in current_date_production_gr.columns and 'quantity' in current_date_production_gr.columns:
                        current_date_production_gr = current_date_production_gr.rename(columns={'quantity': 'produced_qty'})
                    _present = [c for c in _needed if c in current_date_production_gr.columns]
                    current_date_production_gr = current_date_production_gr[_present]
                else:
                    current_date_production_gr = pd.DataFrame()
            else:
                current_date_production_gr = pd.DataFrame()
            
            if not current_date_production_gr.empty:
                current_date_production_gr_normalized = normalize_identifiers(current_date_production_gr)
                orch.process_module4_production(current_date_production_gr_normalized, current_date.strftime('%Y-%m-%d'))
            else:
                pass

        except Exception as e:
            # GR入库是核心状态变更，失败后库存不正确，
            # 后续所有模块在错误状态上运算，必须中止仿真而非静默继续。
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"每日GR入库处理失败，中止仿真以防止数据错误: {e}") from e

        # ==================== 模块运行序列 ====================
        m1_shipments = pd.DataFrame()
        m4_production = pd.DataFrame()
        m4_result = None  # 由 M4 设置，通过内存传递给 M5
        m5_deployment_df = pd.DataFrame()
        m6_delivery_df = pd.DataFrame()
        
        try:
            # ========== M1: 订单生成 ==========
            try:
                m1_result = module1.run_daily_order_generation(
                    config_dict=config_dict,
                    simulation_date=current_date,
                    output_dir=str(module_outputs['module1']),
                    orchestrator=orch,
                    previous_orders_df=m1_previous_orders  # 🔧 修复：传递历史订单
                )
                m1_shipments = m1_result.get('shipment_df', pd.DataFrame())
                
                if not m1_shipments.empty:
                    m1_shipments_normalized = normalize_identifiers(m1_shipments)
                    orch.process_module1_shipments(m1_shipments_normalized, current_date.strftime('%Y-%m-%d'))
                
                if m1_result is not None:
                    m1_result['simulation_date'] = current_date
                all_results['module1'].append(m1_result)
                batch_results['module1'].append(m1_result)
                # 🔧 修复：保存累积订单供下一天使用（DB模式不依赖文件读取历史订单）
                m1_previous_orders = m1_result.get('all_orders_for_next_day', None)
            except Exception as e:
                batch_failed_modules.append((current_date.strftime('%Y-%m-%d'), 'module1', str(e)))
                m1_shipments = pd.DataFrame()
            
            # ========== M4: 生产计划 ==========
            try:
                m4_result = run_module4_integrated(
                    config_dict=config_dict,
                    module3_output_dir=str(module_outputs['module3']),
                    simulation_date=current_date,
                    simulation_start=pd.to_datetime(start_date),
                    output_dir=str(module_outputs['module4']),
                    skip_file_output=True,
                    module3_result=runtime_state.previous_m3_result,
                    previous_line_states_override=runtime_state.load_line_state(current_date),
                    allocated_capacity_override=runtime_state.load_all_previous_capacity(current_date),
                    skip_state_file_output=True,
                )
                
                # 从返回结果中获取 production_df
                m4_production = m4_result.get('production_df', pd.DataFrame())
                
                # [DB-MEM] 保存 M4 line states / capacity 到 runtime_state（替代文件写入）
                _m4_ls = m4_result.get('current_line_states', {})
                if _m4_ls:
                    runtime_state.save_line_state(current_date, _m4_ls)
                _m4_cap = m4_result.get('current_allocated_capacity', {})
                if _m4_cap:
                    runtime_state.save_allocated_capacity(current_date, _m4_cap)
                
                # 🔄 简化调用：仅持久化"未来 available_date"的生产计划，避免重复当日GR
                if not m4_production.empty and 'available_date' in m4_production.columns:
                    m4_production['available_date'] = pd.to_datetime(m4_production['available_date'])
                    future_plans = m4_production[m4_production['available_date'].dt.normalize() > current_date.normalize()]
                    if not future_plans.empty:
                        future_plans_normalized = normalize_identifiers(future_plans)
                        orch.process_module4_production(future_plans_normalized, current_date.strftime('%Y-%m-%d'))
                    else:
                        pass
                else:
                    pass
                
                # 存储完整的Module4结果（包含所有输出表）
                m4_result['simulation_date'] = current_date
                all_results['module4'].append(m4_result)
                batch_results['module4'].append(m4_result)
            except Exception as e:
                batch_failed_modules.append((current_date.strftime('%Y-%m-%d'), 'module4', str(e)))
                m4_production = pd.DataFrame()

            # ========== M5: 部署计划 ==========
            try:
                # [DB-MEM] 传递 module4_result 内存数据，无需 M4 xlsx 文件
                m5_result = module5.run_deployment_planning(
                    config_dict=config_dict,
                    module1_output_dir=str(module_outputs['module1']),
                    module4_output_path=None,  # No file — using in-memory module4_result
                    orchestrator=orch,
                    current_date=current_date.strftime('%Y-%m-%d'),
                    output_path=str(module_outputs['module5'] / f"Module5Output_{current_date.strftime('%Y%m%d')}.xlsx"),
                    skip_file_output=True,
                    module4_result=m4_result,
                )
                
                if m5_result and 'deployment_plan' in m5_result:
                    deployment_plan_df = m5_result['deployment_plan']
                    if not deployment_plan_df.empty:
                        valid_deployment = deployment_plan_df[
                            (deployment_plan_df['deployed_qty_invCon'] > 0) & 
                            (deployment_plan_df['deployed_qty_invCon'].notna()) &
                            (deployment_plan_df['sending'] != deployment_plan_df['receiving'])
                        ].copy()
                        
                        
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
                            
                            m5_deployment_df = normalize_identifiers(m5_deployment_df)
                            orch.process_module5_deployment(m5_deployment_df, current_date.strftime('%Y-%m-%d'))
                
                if m5_result is not None:
                    m5_result['simulation_date'] = current_date
                all_results['module5'].append(m5_result)
                batch_results['module5'].append(m5_result)
            except Exception as e:
                batch_failed_modules.append((current_date.strftime('%Y-%m-%d'), 'module5', str(e)))
                m5_deployment_df = pd.DataFrame()

            # ========== M6: 物流执行 ==========
            try:
                m6_result = module6.run_daily_physical_flow(
                    config_dict=config_dict,
                    orchestrator=orch,
                    current_date=current_date,
                    output_dir=str(module_outputs['module6']),
                    max_wait_days=M6_MAX_WAIT_DAYS,
                    random_seed=config_dict.get('M6_RandomSeed', M6_RANDOM_SEED)
                )
                
                if m6_result and 'delivery_plan' in m6_result:
                    m6_delivery_df = m6_result.get('delivery_plan', pd.DataFrame())
                    if not m6_delivery_df.empty:
                        m6_delivery_normalized = normalize_identifiers(m6_delivery_df)
                        orch.process_module6_delivery(m6_delivery_normalized, current_date.strftime('%Y-%m-%d'))
                
                if m6_result is not None:
                    m6_result['simulation_date'] = current_date
                all_results['module6'].append(m6_result)
                batch_results['module6'].append(m6_result)
            except Exception as e:
                batch_failed_modules.append((current_date.strftime('%Y-%m-%d'), 'module6', str(e)))
                m6_delivery_df = pd.DataFrame()

            # ========== M3: 净需求计算 ==========
            try:
                m3_result = module3.run_integrated_mode(
                    module1_output_dir=str(module_outputs['module1']),
                    orchestrator=orch,
                    config_dict=config_dict,
                    start_date=current_date.strftime('%Y-%m-%d'),
                    end_date=current_date.strftime('%Y-%m-%d'),
                    output_dir=str(module_outputs['module3'])
                )
                if m3_result is not None:
                    m3_result['simulation_date'] = current_date
                    # [DB-MEM] 保存 M3 结果供下一天 M4 使用（替代基于文件的 M3→M4 数据流）
                    runtime_state.previous_m3_result = m3_result
                all_results['module3'].append(m3_result)
                batch_results['module3'].append(m3_result)
            except Exception as e:
                batch_failed_modules.append((current_date.strftime('%Y-%m-%d'), 'module3', str(e)))

        except Exception as e:
            import traceback
            traceback.print_exc()
        
        # ==================== 每日结束：状态保存 ====================
        try:
            # 保存期末库存快照
            orch.save_ending_inventory(current_date.strftime('%Y-%m-%d'))
            # 输出每日库存汇总
            orch.output_daily_inventory_summary(current_date.strftime('%Y-%m-%d'))
            # 保存每日状态
            orch.save_daily_state(current_date.strftime('%Y-%m-%d'))
            # 获取当日统计
            stats = orch.get_summary_statistics(current_date.strftime('%Y-%m-%d'))
        except Exception as e:
            pass

        # ===== checkpoint 由 _flush_batch_to_db 在事务内原子写入，不再提前单独写入 =====
        # 早期版本在此处调用独立 checkpoint 写入辅助函数，会产生“checkpoint 超前、模块数据滞后”的不一致窗口：
        # 用户在下一天运行时查询数据库，checkpoint 已显示当天完成，但当天模块数据尚未写入数据库。
        # 若在此窗口内崩溃，断点续跑会跳过该天，永久丢失模块数据。
        # 现在 checkpoint 仅在 _flush_batch_to_db 的原子事务内更新，保证数据与 checkpoint 严格一致。

        # ===== 每 batch_size 天或最后一天：原子批量写入数据库 =====
        is_last_day = (current_date == sim_dates_to_run[-1])
        if writer is not None and (i % batch_size == 0 or is_last_day):
            batch_end_str = current_date.strftime('%Y-%m-%d')
            # 批次内有模块失败时，拒绝推进 checkpoint，避免缺口天被标为已完成
            if batch_failed_modules:
                for fail_date, fail_mod, fail_err in batch_failed_modules:
                    pass
                # 不清空 batch_results / batch_start_date，让下一批次重新尝试这些天
                # （注意：batch_size=1 时每天独立，失败天的数据不进入下一天的 batch_results）
                batch_results = {mod: [] for mod in all_results}  # 丢弃本批次不完整数据
                batch_failed_modules = []
                # 不更新 batch_start_date → 断点续跑将从当前失败日重新开始
            else:
                try:
                    _flush_batch_to_db(
                        writer=writer,
                        batch_results=batch_results,
                        run_id=run_id_override or f"db_{config_name}",
                        db=db,
                        run_key=run_key or config_name,
                        config_name=config_name,
                        start_date=start_date,
                        end_date=end_date,
                        batch_start_date=batch_start_date,
                        batch_end_date=batch_end_str,
                        orch=orch,
                        m1_previous_orders=m1_previous_orders,
                        runtime_state=runtime_state,
                    )
                    # 仅在写入数据库成功后才清空 batch_results
                    batch_results = {mod: [] for mod in all_results}
                    batch_start_date = (current_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                    batch_failed_modules = []  # 清空失败追踪
                    # 数据库模式下释放已写入数据库的历史数据，防止 all_results 无限增长导致 OOM
                    # 将已写入数据库的结果替换为轻量级存根（保留 simulation_date 用于计数和日志）
                    if db is not None:
                        for mod in all_results:
                            all_results[mod] = [
                                {'simulation_date': r.get('simulation_date'), '_flushed_to_db': True}
                                if isinstance(r, dict) else r
                                for r in all_results[mod]
                            ]
                except Exception as flush_err:
                    import traceback
                    traceback.print_exc()
                    raise  # 中止仿真，避免后续批次覆盖丢失数据
    
    # 仿真结束统计
    total_runtime_seconds = time.time() - simulation_start_time
    if total_runtime_seconds >= 60:
        minutes = int(total_runtime_seconds // 60)
        seconds = total_runtime_seconds % 60
        runtime_str = f"{minutes}分钟 {seconds:.2f}秒"
    else:
        runtime_str = f"{total_runtime_seconds:.2f}秒"
    
    # 生成汇总报告
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
    except Exception as e:
        import traceback
        traceback.print_exc()
        summary_reports = {}
    
    # 最终统计
    try:
        final_stats = orch.get_summary_statistics(end_date)
        for key, value in final_stats.items():
            pass
    except Exception as e:
        final_stats = {}
    

    # 仿真循环完成后，不在此处设置 completed 状态。
    # 原因：完整的数据流为 仿真循环完成 -> Summary表生成 -> Orchestrator写入 -> 标记completed。
    # Summary 和 Orchestrator 写入在 db_runner.py 的步骤3中完成，
    # 只有全部成功后才应该标记为 completed。
    # 在此处提前标记会导致：如果 Summary 生成失败，checkpoint 已经是 completed，
    # 断点续跑时将无法检测到未完成状态，Summary 数据会永久缺失。
    if db is not None:
        pass
    
    # 🦆 禁用DuckDB内存模式并打印统计
    _ensure_memory_store_imported()
    if _is_memory_mode_enabled and _is_memory_mode_enabled():
        if _get_data_store:
            store = _get_data_store()
            if store:
                store.print_stats()
        if _disable_memory_mode:
            _disable_memory_mode()
    
    # 输出运行时间统计
    simulation_end_datetime = datetime.now()
    
    return {
        'validation_passed': True,
        'simulation_completed': True,
        'dates_processed_this_run': len(sim_dates_to_run),
        'results': all_results,
        'final_stats': final_stats,
        'output_directory': str(output_dir),
        'summary_reports': summary_reports,
        'config_dict': config_dict,  # 返回config_dict供本地文件输出使用
        'run_id': run_id_override or f"db_{config_name}",
    }
