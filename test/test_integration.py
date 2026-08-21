"""重构模块的集成调度测试入口（使用 StateContext 管理状态）。

职责：
- 在本地文件模式下组织每日执行与结果落盘流程。
- 使用 StateContext（状态总管）替代旧 Orchestrator 的状态管理职责。
- Orch 负责配置加载；本入口只做调度与内存结果收集，不写正式数据库。
- StateContext 负责状态持有、view 计算、processor 写入和每日操作。
"""

import logging
import os
import time

import pandas as pd

from src.core.orchestrator import Orch
from src.core.orchestrator.models import (
    MODULE_EXECUTION_ORDER,
    validate_module_result,
)
from src.modules import module1
from src.modules.deployment_planning.integration_refactor import ModuleFive
from src.modules.logistics_execution.integration_refactor import ModuleSix
from src.modules.mrp_planning.integration_refactor import ModuleThree
from src.modules.production_planning.integration_refactor import ModuleFour
from src.modules.state_context import StateContext
from src.utils.defaults import M6_MAX_WAIT_DAYS, M6_RANDOM_SEED
from src.utils.performance_telemetry import PerformanceTelemetry


_PRIMARY_OUTPUTS = {
    "module1": "orders_df",
    "module4": "production_df",
    "module5": "deployment_plan",
    "module6": "delivery_plan",
    "module3": "net_demand_df",
}


def _result_row_count(module_id: str, result: dict) -> int | None:
    """返回模块主结果行数；没有主输出时保留为空。"""
    primary = result.get(_PRIMARY_OUTPUTS[module_id])
    return len(primary) if primary is not None and hasattr(primary, "__len__") else None

def run_integrated_simulation(
    config_path: str,
    start_date: str,
    end_date: str,
    output_base_dir: str = "./integrated_output",
    engine: str = "polars",
    test_mode: bool = False,
    test_schema: str | None = None,
    enable_persistence: bool = True,
    interrupt_after: tuple[str, str] | None = None,
    config_name: str | None = None,
    skip_dq: bool = False,
    verbose: bool = False,
    performance_report: str | None = None,
    run_mode: str | None = None,
):
    """使用 StateContext + Orch 的集成仿真入口。

    流程：
    1. Orch 从 config_path 加载 Excel 配置
    2. StateContext 管理所有可变状态（库存、调拨、在途等）
    3. ModuleOne 通过 orchestrator=ctx 获取状态视图
    4. 仿真循环：ctx.day_start → M1 → M4 → M5 → M6 → M3 → ctx.day_end

    ``test_mode`` 使用隔离数据库 schema（默认 ``test``，可由 ``test_schema`` 覆盖），
    不决定是否写库；
    默认持久化配置、日度状态与模块输出，可通过 ``enable_persistence=False`` 关闭。
    ``interrupt_after`` 仅供集成测试使用，格式为 ``(YYYY-MM-DD, module_id)``；
    模块完成后、当日日末状态提交前抛出 ``RuntimeError``。

    ``skip_dq`` 仅用于性能/集成测试时跳过配置数据质量检测；``verbose`` 默认关闭，
    避免模块内部逐步骤耗时日志干扰性能基线。设置 ``performance_report``（或环境变量
    ``CHAINSIGHT_PERFORMANCE_REPORT``）后，按日写出结构化性能 JSON。
    """
    logger = logging.getLogger("SupplyChainSimulation")
    performance_path = performance_report or os.environ.get("CHAINSIGHT_PERFORMANCE_REPORT")
    telemetry = PerformanceTelemetry(
        "refactor",
        run_mode or os.environ.get("CHAINSIGHT_RUN_MODE", "continuous"),
        "test" if test_mode else "public",
    ) if performance_path else None
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
        engine=engine,
        skip_dq=skip_dq,
        enable_persistence=enable_persistence,
        test_mode=test_mode,
        test_schema=test_schema,
        config_name=config_name,
    )
    ctx = StateContext(simulation_date=start_date, orch=orch)
    ctx.initialize(orch.all_config)

    m1 = module1.ModuleOne(
        simulation_date=str(start_date),
        orchestrator=ctx,
        orch=orch,
        verbose=verbose,
    )

    # ── ModuleFour（生产计划）──
    m4 = ModuleFour(
        simulation_date=str(start_date),
        simulation_start_date=start_date,
        orchestrator=ctx,
        orch=orch,
        verbose=verbose,
    )

    # ── ModuleThree（消费 M5 当日 PlanningFacts，产出供下一日 M4 使用的净需求）──
    m3 = ModuleThree(
        simulation_date=str(start_date),
        simulation_start_date=start_date,
        orchestrator=ctx,
        orch=orch,
        state_context=ctx,
        verbose=verbose,
    )
    m5 = ModuleFive(
        simulation_date=str(start_date),
        simulation_start_date=start_date,
        state_context=ctx,
        orch=orch,
        verbose=verbose,
    )
    m6 = ModuleSix(
        simulation_date=str(start_date),
        state_context=ctx,
        orch=orch,
        max_wait_days=M6_MAX_WAIT_DAYS,
        random_seed=M6_RANDOM_SEED,
        verbose=verbose,
    )

    modules = {
        "module1": m1,
        "module4": m4,
        "module5": m5,
        "module6": m6,
        "module3": m3,
    }
    for module in modules.values():
        module.prepare()

    all_results = {module_id: [] for module_id in MODULE_EXECUTION_ORDER}
    context_snapshots = []
    simulation_start_time = time.time()

    # progress_date 表示最后完整完成日；续跑须从下一天 M1 开始。
    actual_start_date = pd.Timestamp(start_date)
    if orch._resuming and orch._resume_date is not None:
        actual_start_date = pd.Timestamp(orch._resume_date) + pd.Timedelta(days=1)
        logger.info("🔁 测试集成入口从 %s 续跑", actual_start_date.strftime('%Y-%m-%d'))
    sim_dates = pd.date_range(actual_start_date, end_date, freq='D')
    for i, current_date in enumerate(sim_dates, 1):
        date_str = current_date.strftime('%Y-%m-%d')

        day_start_timer = telemetry.start() if telemetry else 0.0
        ctx.day_start(date_str)
        if telemetry:
            telemetry.record("day_start", day_start_timer, simulation_date=date_str)

        for module in modules.values():
            module.simulation_date = current_date
        for module_id in MODULE_EXECUTION_ORDER:
            if module_id == "module4":
                # M4 严格消费前一个自然日 M3 的输出，保持一日 lag。
                m4.module3_result = ctx.get_previous_m3_result(date_str)
                m4.previous_line_states_override = ctx.get_previous_line_state(date_str)
                m4.allocated_capacity_override = (
                    ctx.get_all_previous_allocated_capacity(date_str)
                )

            module_timer = telemetry.start() if telemetry else 0.0
            modules[module_id].run()
            result = modules[module_id].output()
            if telemetry:
                telemetry.record(
                    "module_run",
                    module_timer,
                    simulation_date=date_str,
                    module=module_id,
                    rows=_result_row_count(module_id, result),
                )
            validate_module_result(module_id, result, date_str)
            result["simulation_date"] = current_date
            ctx.apply_module_result(module_id, result, date_str)
            ctx.record_summary_module_result(module_id, result, date_str)
            all_results[module_id].append(result)

            if interrupt_after == (date_str, module_id):
                raise RuntimeError(
                    f"受控中断：{date_str} 的 {module_id} 完成后，"
                    "当日日末数据尚未提交"
                )

            if module_id == "module3":
                # 保留显式局部变量，确保 M3 的结果先经过统一校验和记录，
                # 再写入 M4 属性，供下一轮 M4 读取。
                m3_result = result
                m4.module3_result = m3_result

        day_end_timer = telemetry.start() if telemetry else 0.0
        ctx.day_end(date_str)
        if telemetry:
            telemetry.record("day_end", day_end_timer, simulation_date=date_str)
        if enable_persistence:
            persistence_timer = telemetry.start() if telemetry else 0.0
            # 按日原子落库；测试模式下 Orchestrator 固定使用 ``test`` schema。
            with orch.persistence.batch_transaction():
                for module in modules.values():
                    orch.save_module_output(module, date_str)
                orch.save_daily_state(ctx, date_str)
                orch.save_checkpoint(orch.run_id, current_date=date_str)
            if telemetry:
                telemetry.record(
                    "persistence_and_checkpoint",
                    persistence_timer,
                    simulation_date=date_str,
                )
                telemetry.run_id = orch.run_id
                telemetry.write(performance_path)
        context_snapshots.append({
            "simulation_date": current_date,
            "views": ctx.snapshot_integration_views(date_str),
        })
        logger.info("✅ 第 %d/%d 天完整调度与结果校验完成: %s", i, len(sim_dates), date_str)

    # ── 完成报告 ──
    summary_outputs = ctx.build_summary_outputs(
        start_date=start_date,
        end_date=end_date,
        config_dict=orch.all_config,
    )
    final_stats = ctx.get_summary_statistics(end_date)
    if enable_persistence:
        finalise_timer = telemetry.start() if telemetry else 0.0
        orch.finalize_simulation(ctx)
        if telemetry:
            telemetry.record("finalise", finalise_timer)
    total_seconds = time.time() - simulation_start_time
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours >= 1:
        runtime_str = f"{int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒"
    elif minutes >= 1:
        runtime_str = f"{int(minutes)}分钟 {seconds:.2f}秒"
    else:
        runtime_str = f"{seconds:.2f}秒"

    # n_m4_prod = sum(
    #     1 for result in all_results['module4']
    #     if not result['production_df'].empty
    # )
    # logger.info(
    #     "🎉 集成仿真完成! 共 %d 天 (M4 %d 天产出), 耗时 %s",
    #     len(sim_dates), n_m4_prod, runtime_str,
    # )

    if telemetry:
        telemetry.run_id = orch.run_id
        written_report = telemetry.write(performance_path)
        logger.info("📈 性能报告: %s", written_report)

    return {
        'simulation_completed': True,
        'run_id': orch.run_id,
        'dates_processed': len(sim_dates),
        'output_directory': output_base_dir,
        'results': all_results,
        'context_snapshots': context_snapshots,
        'summary_outputs': summary_outputs,
        'final_stats': final_stats,
        'performance_report': str(performance_path) if performance_path else None,
    }
