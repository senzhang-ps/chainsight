# performance_profiler.py（性能分析器）
# 性能分析工具 - 用于识别代码瓶颈

import cProfile
import pstats
import io
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """性能分析器上下文管理器"""
    
    def __init__(self, module_name: str, output_dir: Path = None, enabled: bool = True):
        """
        初始化性能分析器
        
        Args:
            module_name: 模块名称（如 "Module3", "Module5"）
            output_dir: 输出目录，如果为None则使用当前目录
            enabled: 是否启用分析（可用于生产环境关闭）
        """
        self.module_name = module_name
        self.output_dir = output_dir or Path(".")
        self.enabled = enabled
        self.profiler = None
        
    def __enter__(self):
        """启动分析器"""
        if self.enabled:
            self.profiler = cProfile.Profile()
            self.profiler.enable()
            logger.info(f"🔍 启动性能分析: {self.module_name}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """停止分析器并保存报告"""
        if self.enabled and self.profiler:
            self.profiler.disable()
            
            # 生成报告
            try:
                self._save_report()
                logger.info(f"✅ 性能分析完成: {self.module_name}")
            except Exception as e:
                logger.error(f"❌ 保存性能报告失败: {e}")
    
    def _save_report(self):
        """保存性能分析报告"""
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"performance_profile_{self.module_name}_{timestamp}.txt"
        filepath = self.output_dir / filename
        
        # 生成报告
        stream = io.StringIO()
        stats = pstats.Stats(self.profiler, stream=stream)
        
        # 按累积时间排序，显示前30个函数
        stream.write("=" * 100 + "\n")
        stream.write(f"性能分析报告: {self.module_name}\n")
        stream.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        stream.write("=" * 100 + "\n\n")
        
        stream.write("【按累积时间排序 - 前30个最耗时函数】\n")
        stream.write("-" * 100 + "\n")
        stats.sort_stats('cumulative')
        stats.print_stats(30)
        
        stream.write("\n\n")
        stream.write("【按调用次数排序 - 前30个最频繁调用函数】\n")
        stream.write("-" * 100 + "\n")
        stats.sort_stats('calls')
        stats.print_stats(30)
        
        stream.write("\n\n")
        stream.write("【按总时间排序 - 前30个纯执行时间最长函数】\n")
        stream.write("-" * 100 + "\n")
        stats.sort_stats('time')
        stats.print_stats(30)
        
        # 保存到文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(stream.getvalue())
        
        # 同时在日志中输出关键信息
        logger.info(f"📊 性能报告已保存: {filepath}")
        
        # 提取并记录前5个最耗时的函数
        stream_summary = io.StringIO()
        stats_summary = pstats.Stats(self.profiler, stream=stream_summary)
        stats_summary.sort_stats('cumulative')
        stats_summary.print_stats(5)
        
        logger.info(f"📈 Top 5 耗时函数 ({self.module_name}):")
        for line in stream_summary.getvalue().split('\n')[6:11]:  # 跳过表头
            if line.strip():
                logger.info(f"   {line}")


def profile_function(func):
    """
    装饰器：对单个函数进行性能分析
    
    用法:
        @profile_function
        def my_slow_function():
            ...
    """
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        
        result = func(*args, **kwargs)
        
        profiler.disable()
        
        # 打印简要统计
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream)
        stats.sort_stats('cumulative')
        stats.print_stats(10)
        
        logger.info(f"⏱️  函数性能分析: {func.__name__}")
        for line in stream.getvalue().split('\n')[:15]:
            if line.strip():
                logger.info(f"   {line}")
        
        return result
    
    return wrapper
