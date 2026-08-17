"""重构模块的集成调度测试入口（使用 StateContext 管理状态）。

职责：
- 在本地文件模式下组织每日执行与结果落盘流程。
- 使用 StateContext（状态总管）替代旧 Orchestrator 的状态管理职责。
- Orch 负责配置加载；本入口只做调度与内存结果收集，不写正式数据库。
- StateContext 负责状态持有、view 计算、processor 写入和每日操作。
"""

import logging
import time

import pandas as pd

from src.core.orchestrator import Orch
from src.modules import module1
from src.modules.deployment_planning.integration_refactor import ModuleFive
from src.modules.logistics_execution.integration_refactor import ModuleSix
from src.modules.mrp_planning.integration_refactor import ModuleThree
from src.modules.production_planning.integration_refactor import ModuleFour
from src.modules.state_context import StateContext
from src.utils.defaults import M6_MAX_WAIT_DAYS, M6_RANDOM_SEED

logger = logging.getLogger("SupplyChainSimulation")


_MODULE_ORDER = ("module1", "module4", "module5", "module6", "module3")
_RESULT_DATAFRAMES = {
    "module1": (
        "orders_df", "shipment_df", "cut_df", "supply_demand_df", "summary_df",
    ),
    "module3": ("net_demand_df",),
    "module4": (
        "production_df", "exceed_log", "issues_df", "changeover_log",
        "unconstrained_plan",
    ),
    "module5": (
        "deployment_plan", "unfulfilled_log", "stock_on_hand_log", "validation_log",
    ),
    "module6": (
        "delivery_plan", "vehicle_log", "truck_usage", "unsatisfied_log",
        "validation_log", "bypass_log",
    ),
}
_CONTEXT_VIEW_GETTERS = {
    "unrestricted_inventory": "get_unrestricted_inventory_view",
    "open_deployment": "get_open_deployment_view",
    "planning_intransit": "get_planning_intransit_view",
    "space_quota": "get_space_quota_view",
    "delivery_gr": "get_delivery_gr_view",
    "production_gr": "get_production_gr_view",
    "production_plan_backlog": "get_production_plan_backlog_view",
    "shipment_log": "get_shipment_log_view",
    "delivery_shipment_log": "get_delivery_shipment_log_view",
    "inventory_change_log": "generate_inventory_change_log",
}


def _bind_simulation_date(modules: dict[str, object], simulation_date: pd.Timestamp) -> None:
    """将同一个仿真日期注入所有模块，避免逐模块重复赋值。"""
    for module in modules.values():
        module.simulation_date = simulation_date


def _validate_module_result(module_id: str, result: dict, date_str: str) -> None:
    """校验每日模块输出的统一结果契约，避免状态写回时静默丢失数据。"""
    if not isinstance(result, dict):
        raise TypeError(f"{date_str} {module_id} 输出必须为 dict，实际为 {type(result).__name__}")

    missing = [key for key in _RESULT_DATAFRAMES[module_id] if key not in result]
    if missing:
        raise ValueError(f"{date_str} {module_id} 输出缺少结果字段: {missing}")

    invalid = [
        key for key in _RESULT_DATAFRAMES[module_id]
        if not isinstance(result[key], pd.DataFrame)
    ]
    if invalid:
        raise TypeError(f"{date_str} {module_id} 结果字段不是 DataFrame: {invalid}")


def _apply_module_result(
    module_id: str,
    result: dict,
    context: StateContext,
    date_str: str,
) -> None:
    """按模块标识统一将每日结果写回 StateContext。

    Context 仍保留细粒度 ``apply_*`` 业务接口；调度器只负责把模块结果
    映射到正确的状态写回动作，从而消除循环中零散的 ``result.get()`` 调用。
    M3 在 ``ModuleThree.run()`` 内部已经调用 ``apply_m3_net_demand()``，此处不重复写回。
    """
    if module_id == "module1":
        context.apply_shipments(result["shipment_df"], date_str)
        context.apply_deployment_demand_inputs(
            result["supply_demand_df"],
            result.get("all_orders_for_next_day", result["orders_df"]),
            date_str,
        )
    elif module_id == "module4":
        context.apply_line_state(result.get("current_line_states", {}), date_str)
        context.apply_allocated_capacity(
            result.get("current_allocated_capacity", {}), date_str
        )
        context.apply_production(result["production_df"], date_str)
    elif module_id == "module5":
        context.apply_deployment(result["deployment_plan"], date_str)
    elif module_id == "module6":
        context.apply_delivery(result["delivery_plan"], date_str)
    elif module_id != "module3":
        raise ValueError(f"未知模块结果: {module_id}")


def _snapshot_context_views(context: StateContext, date_str: str) -> dict[str, pd.DataFrame]:
    """深拷贝日末 Context 视图，供外部集成对比而不依赖数据库持久化。"""
    return {
        view_name: getattr(context, getter_name)(date_str).copy(deep=True)
        for view_name, getter_name in _CONTEXT_VIEW_GETTERS.items()
    }


def run_integrated_simulation(
    config_path: str,
    start_date: str,
    end_date: str,
    output_base_dir: str = "./integrated_output",
    engine: str = "polars",
):
    """使用 StateContext + Orch 的集成仿真入口。

    流程：
    1. Orch 从 config_path 加载 Excel 配置
    2. StateContext 管理所有可变状态（库存、调拨、在途等）
    3. ModuleOne 通过 orchestrator=ctx 获取状态视图
    4. 仿真循环：ctx.day_start → M1 → M4 → M5 → M6 → M3 → ctx.day_end

    注意：这是正式入库前的调度验证入口。所有模块结果均收集到返回值，
    但不调用任何 ``Orch.save_*``、``save_checkpoint()`` 或 ``finish()`` 方法，
    以免写入正式数据库。
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
        engine=engine,
        skip_dq=True,
        enable_persistence=False,
    )
    ctx = StateContext(simulation_date=start_date, orch=orch)
    ctx.initialize(orch.all_config)

    m1 = module1.ModuleOne(
        simulation_date=str(start_date),
        orchestrator=ctx,
        orch=orch,
    )

    # ── ModuleFour（生产计划）──
    m4 = ModuleFour(
        simulation_date=str(start_date),
        simulation_start_date=start_date,
        orchestrator=ctx,
        orch=orch
    )

    # ── ModuleThree（消费 M5 当日 PlanningFacts，产出供下一日 M4 使用的净需求）──
    m3 = ModuleThree(
        simulation_date=str(start_date),
        simulation_start_date=start_date,
        orchestrator=ctx,
        orch=orch,
        state_context=ctx,
    )
    m5 = ModuleFive(
        simulation_date=str(start_date),
        simulation_start_date=start_date,
        state_context=ctx,
        orch=orch,
    )
    m6 = ModuleSix(
        simulation_date=str(start_date),
        state_context=ctx,
        orch=orch,
        max_wait_days=M6_MAX_WAIT_DAYS,
        random_seed=M6_RANDOM_SEED,
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

    all_results = {module_id: [] for module_id in _MODULE_ORDER}
    context_snapshots = []
    simulation_start_time = time.time()

    # 不使用 orch.iter_dates()：该方法会在每个仿真日开始时写入 checkpoint。
    sim_dates = pd.date_range(start_date, end_date, freq='D')
    for i, current_date in enumerate(sim_dates, 1):
        date_str = current_date.strftime('%Y-%m-%d')

        ctx.day_start(date_str)

        _bind_simulation_date(modules, current_date)
        for module_id in _MODULE_ORDER:
            if module_id == "module4":
                # M4 严格消费前一个自然日 M3 的输出，保持一日 lag。
                m4.module3_result = ctx.get_previous_m3_result(date_str)
                m4.previous_line_states_override = ctx.get_previous_line_state(date_str)
                m4.allocated_capacity_override = (
                    ctx.get_all_previous_allocated_capacity(date_str)
                )

            modules[module_id].run()
            result = modules[module_id].output()
            _validate_module_result(module_id, result, date_str)
            result["simulation_date"] = current_date
            _apply_module_result(module_id, result, ctx, date_str)
            all_results[module_id].append(result)

            if module_id == "module3":
                # 保留显式局部变量，确保 M3 的结果先经过统一校验和记录，
                # 再写入 M4 属性，供下一轮 M4 读取。
                m3_result = result
                m4.module3_result = m3_result

        ctx.day_end(date_str)
        context_snapshots.append({
            "simulation_date": current_date,
            "views": _snapshot_context_views(ctx, date_str),
        })
        logger.info("✅ 第 %d/%d 天完整调度与结果校验完成: %s", i, len(sim_dates), date_str)

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

    # n_m4_prod = sum(
    #     1 for result in all_results['module4']
    #     if not result['production_df'].empty
    # )
    # logger.info(
    #     "🎉 集成仿真完成! 共 %d 天 (M4 %d 天产出), 耗时 %s",
    #     len(sim_dates), n_m4_prod, runtime_str,
    # )

    return {
        'simulation_completed': True,
        'dates_processed': len(sim_dates),
        'output_directory': output_base_dir,
        'results': all_results,
        'context_snapshots': context_snapshots,
    }
