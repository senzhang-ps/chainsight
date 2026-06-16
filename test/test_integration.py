"""文件模式集成仿真入口（使用 StateContext 管理状态）。

职责：
- 在本地文件模式下组织每日执行与结果落盘流程。
- 使用 StateContext（状态总管）替代旧 Orchestrator 的状态管理职责。
- Orch 负责调度、配置加载和持久化。
- StateContext 负责状态持有、view 计算、processor 写入和每日操作。
"""

import logging
import time

import pandas as pd

from src.core.orchestrator import Orch
from src.modules import module1
from src.modules.state_context import StateContext
from src.utils.normalization import normalize_identifiers

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

    # ── 仿真循环 ──
    all_results = {'module1': []}
    simulation_start_time = time.time()

    for i, current_date in orch.iter_dates():
        date_str = current_date.strftime('%Y-%m-%d')

        ctx.day_start(date_str)

        m1.simulation_date = current_date
        m1.run()
        m1_result = m1.output()

        m1_shipments = m1_result.get('shipment_df', pd.DataFrame())
        if not m1_shipments.empty:
            ctx.apply_shipments(normalize_identifiers(m1_shipments), date_str)

        m1_result['simulation_date'] = current_date
        all_results['module1'].append(m1_result)

        ctx.day_end(date_str)
        orch.save_module_output(m1, date_str)
        orch.save_daily_state(ctx, date_str)

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

    logger.info("🎉 集成仿真完成! 共 %d 天, 耗时 %s", len(orch.sim_dates), runtime_str)

    return {
        'simulation_completed': True,
        'dates_processed': len(orch.sim_dates),
        'output_directory': output_base_dir,
        'results': all_results,
    }
