"""文件模式集成仿真入口（使用 StateContext 管理状态）。

职责：
- 在本地文件模式下组织每日执行与结果落盘流程。
- 使用 StateContext（状态总管）替代旧 Orchestrator 的状态管理职责。
- Orch 负责调度、配置加载和持久化。
- StateContext 负责状态持有、view 计算、processor 写入和每日操作。
"""

import logging
import time
from pathlib import Path

import pandas as pd

from src.core.orchestrator import Orch
from src.modules import module1
from src.modules.production_planning.integration_refactor import ModuleFour
from src.modules.mrp_planning.integration_refactor import ModuleThree
from src.modules.state_context import StateContext

logger = logging.getLogger("SupplyChainSimulation")


def run_integrated_simulation(
    config_path: str,
    start_date: str,
    end_date: str,
    output_base_dir: str = "./integrated_output",
):
    """使用 StateContext + Orch 的集成仿真入口。

    流程：
    1. Orch 从 config_path 加载 Excel 配置
    2. StateContext 管理所有可变状态（库存、调拨、在途等）
    3. ModuleOne 通过 orchestrator=ctx 获取状态视图
    4. 仿真循环：ctx.day_start → M1.run → ctx.apply_shipments → ctx.day_end → orch.save
    """
    # ── 日志配置 ──
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        handlers=[logging.StreamHandler()],
    )

    # ── 初始化 ──
    orch = Orch(
        start_date=start_date,
        end_date=end_date,
        config_path=config_path,
        output_path=output_base_dir,
        engine='polars',skip_dq=False
    )
    ctx = StateContext(simulation_date=start_date, orch=orch)
    ctx.initialize(orch.all_config)

    m1 = module1.ModuleOne(
        simulation_date=str(start_date),
        orchestrator=ctx,
        orch=orch,
    )
    m1.prepare()

    # ── ModuleFour（生产计划）──
    m4 = ModuleFour(
        simulation_date=str(start_date),
        simulation_start_date=start_date,
        orchestrator=ctx,
        orch=orch
    )
    m4.prepare()  # 一次性：load_datas + 标识符归一 + 分配器静态 maps

    # ── ModuleThree（历史回放占位：从 DB 读 module3_output_netdemand）──
    M3_HISTORICAL_RUNID = 'db_OC_Paste_S1_20251224_repare_20260623_102205'  # 后续接真实 MRP 后可废弃
    m3 = ModuleThree(
        simulation_date=str(start_date),
        simulation_start_date=start_date,
        orchestrator=ctx,
        orch=orch,
        output_dir=str(Path(output_base_dir) / 'module3'),
        skip_file_output=True,
        m3_run_id=M3_HISTORICAL_RUNID,
    )
    m3.prepare()


    # ── 仿真循环 ──
    all_results = {'module1': [], 'module4': []}
    simulation_start_time = time.time()

    for i, current_date in orch.iter_dates():
        date_str = current_date.strftime('%Y-%m-%d')

        ctx.day_start(date_str)

        m1.simulation_date = current_date
        m1.run()
        m1_result = m1.output()

        m1_shipments = m1_result.get('shipment_df', pd.DataFrame())
        if not m1_shipments.empty:
            ctx.apply_shipments(m1_shipments, date_str)

        m1_result['simulation_date'] = current_date
        all_results['module1'].append(m1_result)

        # ---- M3 → M4（1 天 lag：M4 先消费上一轮 M3，M3 再产出当日并存回）----
        m3.simulation_date = current_date
        m4.simulation_date = current_date
        # 跨天状态注入：前一日产线状态 + 历史已分配产能
        m4.previous_line_states_override = ctx.get_previous_line_state(date_str)
        m4.allocated_capacity_override = ctx.get_all_previous_allocated_capacity(date_str)
        m4.run()                          # 消费上一轮 module3_result（首日 None → 空产）
        m3.run()                          # 从 DB 读当日净需求
        m4.module3_result = m3.output()   # 存回供下一轮
        m4_result = m4.output()

        # 跨天结转：当日产线状态 + 已分配产能写回 ctx
        ctx.apply_line_state(m4_result.get('current_line_states', {}), date_str)
        ctx.apply_allocated_capacity(m4_result.get('current_allocated_capacity', {}), date_str)

        all_results['module4'].append({
            'date': date_str,
            'n_production': len(m4_result.get('production_df', pd.DataFrame())),
            'm4_result': m4_result,
        })
        orch.save_module_output(m4, date_str)

        ctx.day_end(date_str)
        orch.save_module_output(m1, date_str)
        orch.save_daily_state(ctx, date_str)
        # 显式推进当日断点 progress_date（原埋在 save_daily_state 末尾，
        # 现提到调用链上，让「写数据」与「推进断点」分离）
        orch.save_checkpoint(orch.run_id, current_date=date_str)

    # ── 标记完成 ──
    orch.finish()

    # ── 完成报告 ──
    total_seconds = time.time() - simulation_start_time
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours >= 1:
        runtime_str = f"{int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒"
    elif minutes >= 1:
        runtime_str = f"{int(minutes)}分钟 {seconds:.2f}秒"
    else:
        runtime_str = f"{seconds:.2f}秒"

    n_m4_prod = sum(1 for r in all_results['module4'] if r['n_production'] > 0)
    logger.info(
        "🎉 集成仿真完成! 共 %d 天 (M4 %d 天产出), 耗时 %s",
        len(orch.sim_dates), n_m4_prod, runtime_str,
    )

    return {
        'simulation_completed': True,
        'dates_processed': len(orch.sim_dates),
        'output_directory': output_base_dir,
        'results': all_results,
    }
