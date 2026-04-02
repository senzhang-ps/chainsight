"""
parallel_executor_main.py

并行执行框架 - Phase 5 性能优化

目的：
- 实现同层模块并行执行（M1 ∥ M4 ∥ M5）
- 提供线程池管理和结果收集
- 处理错误传播和状态同步

设计：
- ThreadPoolExecutor 用于 I/O 密集型任务
- 同步点用于状态一致性保证
- 保持功能兼容性（可通过环境变量切换到串行模式）
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Callable, Any, Optional
from datetime import datetime
import os

from .models import ParallelTaskResult

logger = logging.getLogger(__name__)


class ParallelExecutor:
    """
    并行执行管理器
    
    提供线程池管理和并行/串行执行支持
    
    Args:
        max_workers: 最大线程数（默认3）
        enable_parallel: 是否启用并行（默认True，
            可通过CHAINSIGHT_PARALLEL环境变量覆盖）
    """
    
    def __init__(self, max_workers: int = 3,
                 enable_parallel: Optional[bool] = None):
        """初始化并行执行器"""
        # 允许通过环境变量覆盖
        if enable_parallel is None:
            env_val = os.getenv('CHAINSIGHT_PARALLEL', 'true')
            enable_parallel = env_val.lower() == 'true'
        
        self.max_workers = max_workers
        self.enable_parallel = enable_parallel
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        logger.info(f"🔄 并行执行器已初始化 - "
                   f"max_workers={max_workers}, "
                   f"parallel_enabled={enable_parallel}")
    
    def run_parallel_stage(
        self,
        tasks: List[Tuple[str, Callable[[], Any]]]
    ) -> Tuple[Dict[str, ParallelTaskResult], bool]:
        """
        并行执行一组任务
        
        功能：
        - 并行提交任务到线程池
        - 收集结果和错误
        - 返回执行统计
        
        Args:
            tasks: 任务列表，每项为 (任务名称, 可调用函数)
        
        Returns:
            (结果字典, 是否全部成功)
            - 结果字典：{任务名 → ParallelTaskResult}
            - 是否全部成功：bool
        
        示例：
            results, all_success = executor.run_parallel_stage(
                tasks=[
                    ('Module1', lambda: module1.run(...)),
                    ('Module4', lambda: module4.run(...)),
                    ('Module5', lambda: module5.run(...)),
                ])
        """
        results = {}
        
        # 串行模式：依次执行每个任务
        if not self.enable_parallel:
            logger.info("📊 运行在串行模式")
            for task_name, task_fn in tasks:
                result = self._run_single_task(task_name, task_fn)
                results[task_name] = result
            
            all_success = all(r.status == 'success'
                            for r in results.values())
            self._print_results(results, parallel=False)
            return results, all_success
        
        # 并行模式：提交所有任务
        logger.info(f"⚙️ 并行执行 {len(tasks)} 个任务")
        future_to_task = {}
        
        for task_name, task_fn in tasks:
            future = self.executor.submit(self._run_single_task,
                                         task_name, task_fn)
            future_to_task[future] = task_name
        
        # 收集结果（按完成顺序）
        for future in as_completed(future_to_task):
            task_name = future_to_task[future]
            try:
                result = future.result()
                results[task_name] = result
            except Exception as exc:
                logger.error(f"❌ 任务 {task_name} 异常: {exc}")
                results[task_name] = ParallelTaskResult(
                    task_name=task_name,
                    status='error',
                    error=exc,
                    start_time=datetime.now(),
                    end_time=datetime.now()
                )
        
        # 统计结果
        all_success = all(r.status == 'success'
                         for r in results.values())
        self._print_results(results, parallel=True)
        
        return results, all_success
    
    def _run_single_task(
        self,
        task_name: str,
        task_fn: Callable[[], Any]
    ) -> ParallelTaskResult:
        """
        运行单个任务
        
        功能：
        - 包装任务执行
        - 捕获异常
        - 记录耗时
        
        Args:
            task_name: 任务名称
            task_fn: 可调用函数
        
        Returns:
            ParallelTaskResult：执行结果
        """
        start_time = datetime.now()
        
        try:
            logger.debug(f"🚀 开始执行任务: {task_name}")
            result = task_fn()
            end_time = datetime.now()
            
            elapsed = (end_time - start_time).total_seconds()
            logger.debug(f"✅ 任务完成: {task_name} ({elapsed:.2f}s)")
            
            return ParallelTaskResult(
                task_name=task_name,
                status='success',
                result=result,
                start_time=start_time,
                end_time=end_time
            )
        except Exception as exc:
            end_time = datetime.now()
            logger.error(f"❌ 任务失败: {task_name} - {str(exc)}")
            
            return ParallelTaskResult(
                task_name=task_name,
                status='error',
                error=exc,
                start_time=start_time,
                end_time=end_time
            )
    
    def _print_results(
        self,
        results: Dict[str, ParallelTaskResult],
        parallel: bool = False
    ) -> None:
        """
        打印执行结果统计
        
        Args:
            results: 结果字典
            parallel: 是否为并行模式
        """
        mode_str = "并行" if parallel else "串行"
        print(f"\n{'='*60}")
        print(f"📊 任务执行结果 ({mode_str}模式)")
        print(f"{'='*60}")
        
        total_time = 0.0
        success_count = 0
        error_count = 0
        
        for task_name, result in results.items():
            print(f"  {result}")
            total_time += result.elapsed_time
            
            if result.status == 'success':
                success_count += 1
            else:
                error_count += 1
        
        # 摘要统计
        print(f"{'-'*60}")
        print(f"  ✅ 成功: {success_count}/{len(results)}")
        if error_count > 0:
            print(f"  ❌ 失败: {error_count}/{len(results)}")
        
        if parallel:
            # 并行模式：总耗时 = max(各任务耗时)
            max_time = max((r.elapsed_time for r in results.values()),
                          default=0)
            print(f"  ⏱️  总耗时: {max_time:.2f}s (并行优势)")
        else:
            # 串行模式：总耗时 = sum(各任务耗时)
            print(f"  ⏱️  总耗时: {total_time:.2f}s (串行模式)")
        
        print(f"{'='*60}\n")
    
    def get_results_dict(
        self,
        results: Dict[str, ParallelTaskResult]
    ) -> Dict[str, Any]:
        """
        从结果中提取实际数据
        
        功能：
        - 过滤成功结果
        - 提取 result 字段
        - 用于后续处理
        
        Args:
            results: 并行执行结果
        
        Returns:
            {任务名 → 任务结果数据}
        
        示例：
            results_dict = executor.get_results_dict(results)
            m1_result = results_dict['Module1']
        """
        return {
            task_name: result.result
            for task_name, result in results.items()
            if result.status == 'success'
        }
    
    def close(self) -> None:
        """关闭线程池"""
        self.executor.shutdown(wait=True)
        logger.info("🔒 并行执行器已关闭")
