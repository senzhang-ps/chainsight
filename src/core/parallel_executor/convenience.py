"""
convenience.py

并行执行便捷函数

提供：
- run_parallel_modules: 同层并行执行（M1 ∥ M4 ∥ M5）
- 简化并行执行器的使用
"""

from typing import Callable, Any, Optional, Tuple

from .parallel_executor_main import ParallelExecutor


def run_parallel_modules(
    m1_fn: Callable[[], Any],
    m4_fn: Callable[[], Any],
    m5_fn: Callable[[], Any],
    max_workers: int = 3,
    enable_parallel: Optional[bool] = None
) -> Tuple[Any, Any, Any, bool]:
    """
    并行运行三个模块 (M1 ∥ M4 ∥ M5)
    
    功能：
    - 并行提交M1/M4/M5任务
    - 等待全部完成
    - 返回结果及执行状态
    
    Args:
        m1_fn: Module1 可调用函数
        m4_fn: Module4 可调用函数
        m5_fn: Module5 可调用函数
        max_workers: 最大线程数
        enable_parallel: 是否启用并行
    
    Returns:
        (m1_result, m4_result, m5_result, all_success)
    
    示例：
        m1_result, m4_result, m5_result, success = \
            run_parallel_modules(
                m1_fn=lambda: module1.run_daily_order_generation(
                    ...),
                m4_fn=lambda: module4.run_daily_production_planning_integrated(...),
                m5_fn=lambda: module5.run_daily_deployment_planning(...),
            )
    """
    executor = ParallelExecutor(max_workers=max_workers,
                               enable_parallel=enable_parallel)
    
    tasks = [
        ('Module1', m1_fn),
        ('Module4', m4_fn),
        ('Module5', m5_fn),
    ]
    
    results, all_success = executor.run_parallel_stage(tasks)
    results_dict = executor.get_results_dict(results)
    
    executor.close()
    
    return (
        results_dict.get('Module1'),
        results_dict.get('Module4'),
        results_dict.get('Module5'),
        all_success
    )
