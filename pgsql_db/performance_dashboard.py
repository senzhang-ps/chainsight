"""
性能监控仪表盘 - 实时监控仿真性能

本模块提供:
1. PerformanceDashboard - 性能数据收集和展示
2. RealTimeMonitor - 实时监控器
3. 性能报告生成
"""

import time
import json
import logging
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from contextlib import contextmanager
import threading

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ModuleMetrics:
    """模块性能指标"""
    module_name: str
    call_count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    last_call_time: float = 0.0
    errors: int = 0
    warnings: int = 0
    
    def update(self, elapsed: float, success: bool = True):
        """更新指标"""
        self.call_count += 1
        self.total_time += elapsed
        self.min_time = min(self.min_time, elapsed)
        self.max_time = max(self.max_time, elapsed)
        self.avg_time = self.total_time / self.call_count
        self.last_call_time = elapsed
        if not success:
            self.errors += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'module_name': self.module_name,
            'call_count': self.call_count,
            'total_time_s': round(self.total_time, 3),
            'min_time_ms': round(self.min_time * 1000, 2),
            'max_time_ms': round(self.max_time * 1000, 2),
            'avg_time_ms': round(self.avg_time * 1000, 2),
            'last_call_ms': round(self.last_call_time * 1000, 2),
            'errors': self.errors,
            'success_rate': f"{(self.call_count - self.errors) / max(self.call_count, 1) * 100:.1f}%"
        }


@dataclass  
class DailyMetrics:
    """每日性能指标"""
    date: str
    start_time: float = 0.0
    end_time: float = 0.0
    total_time: float = 0.0
    modules: Dict[str, float] = field(default_factory=dict)
    records_processed: int = 0
    memory_peak_mb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'date': self.date,
            'total_time_s': round(self.total_time, 2),
            'modules': {k: round(v, 3) for k, v in self.modules.items()},
            'records_processed': self.records_processed,
            'throughput': round(self.records_processed / max(self.total_time, 0.001), 1)
        }


class PerformanceDashboard:
    """
    性能监控仪表盘
    
    收集和展示仿真各阶段的性能数据
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("./performance")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 模块级指标
        self.module_metrics: Dict[str, ModuleMetrics] = {}
        
        # 每日指标
        self.daily_metrics: List[DailyMetrics] = []
        self._current_day: Optional[DailyMetrics] = None
        
        # 全局统计
        self.global_stats = {
            'simulation_start': None,
            'simulation_end': None,
            'total_days': 0,
            'total_records': 0,
            'optimization_enabled': False,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # 阈值告警
        self.thresholds = {
            'module_time_warning_ms': 1000,  # 1秒
            'module_time_critical_ms': 5000,  # 5秒
            'daily_time_warning_s': 60,      # 1分钟
            'memory_warning_mb': 1024        # 1GB
        }
        
        # 告警记录
        self.alerts: List[Dict[str, Any]] = []
    
    def start_simulation(self):
        """开始仿真"""
        self.global_stats['simulation_start'] = datetime.now().isoformat()
    
    def end_simulation(self):
        """结束仿真"""
        self.global_stats['simulation_end'] = datetime.now().isoformat()
    
    def start_day(self, date: str):
        """开始新的一天"""
        self._current_day = DailyMetrics(date=date, start_time=time.time())
    
    def end_day(self):
        """结束当天"""
        if self._current_day:
            self._current_day.end_time = time.time()
            self._current_day.total_time = self._current_day.end_time - self._current_day.start_time
            self.daily_metrics.append(self._current_day)
            self.global_stats['total_days'] += 1
            
            # 检查告警
            if self._current_day.total_time > self.thresholds['daily_time_warning_s']:
                self._add_alert('warning', f"Day {self._current_day.date} took {self._current_day.total_time:.1f}s")
            
            self._current_day = None
    
    @contextmanager
    def track_module(self, module_name: str):
        """
        追踪模块执行时间
        
        用法:
            with dashboard.track_module('Module3'):
                module3.run(...)
        """
        if module_name not in self.module_metrics:
            self.module_metrics[module_name] = ModuleMetrics(module_name=module_name)
        
        start = time.time()
        success = True
        try:
            yield
        except Exception:
            success = False
            raise
        finally:
            elapsed = time.time() - start
            self.module_metrics[module_name].update(elapsed, success)
            
            # 更新当日模块时间
            if self._current_day:
                self._current_day.modules[module_name] = elapsed
            
            # 检查告警
            elapsed_ms = elapsed * 1000
            if elapsed_ms > self.thresholds['module_time_critical_ms']:
                self._add_alert('critical', f"{module_name} took {elapsed_ms:.0f}ms")
            elif elapsed_ms > self.thresholds['module_time_warning_ms']:
                self._add_alert('warning', f"{module_name} took {elapsed_ms:.0f}ms")
    
    def record_operation(self, operation_name: str, elapsed: float, records: int = 0):
        """记录单次操作"""
        if operation_name not in self.module_metrics:
            self.module_metrics[operation_name] = ModuleMetrics(module_name=operation_name)
        
        self.module_metrics[operation_name].update(elapsed)
        self.global_stats['total_records'] += records
        
        if self._current_day:
            self._current_day.records_processed += records
    
    def record_cache_hit(self):
        """记录缓存命中"""
        self.global_stats['cache_hits'] += 1
    
    def record_cache_miss(self):
        """记录缓存未命中"""
        self.global_stats['cache_misses'] += 1
    
    def _add_alert(self, level: str, message: str):
        """添加告警"""
        self.alerts.append({
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        cache_total = self.global_stats['cache_hits'] + self.global_stats['cache_misses']
        cache_hit_rate = self.global_stats['cache_hits'] / max(cache_total, 1)
        
        # 计算模块时间分布
        module_times = {
            name: metrics.total_time 
            for name, metrics in self.module_metrics.items()
        }
        total_module_time = sum(module_times.values())
        
        module_distribution = {
            name: f"{(t / max(total_module_time, 0.001)) * 100:.1f}%"
            for name, t in module_times.items()
        }
        
        # 每日平均时间
        daily_times = [d.total_time for d in self.daily_metrics]
        avg_daily_time = np.mean(daily_times) if daily_times else 0
        
        return {
            'global': {
                **self.global_stats,
                'cache_hit_rate': f"{cache_hit_rate * 100:.1f}%",
                'avg_daily_time_s': round(avg_daily_time, 2)
            },
            'modules': {
                name: metrics.to_dict() 
                for name, metrics in self.module_metrics.items()
            },
            'module_distribution': module_distribution,
            'alerts_count': len(self.alerts),
            'critical_alerts': len([a for a in self.alerts if a['level'] == 'critical'])
        }
    
    def print_summary(self):
        """打印性能摘要"""
        summary = self.get_summary()
        
        
        # 全局统计
        
        # 模块性能
        
        for name, metrics in sorted(
            summary['modules'].items(), 
            key=lambda x: x[1]['total_time_s'], 
            reverse=True
        ):
            pass
        
        # 时间分布
        for name, pct in summary['module_distribution'].items():
            bar_len = int(float(pct.replace('%', '')) / 5)
            bar = '█' * bar_len
        
        # 告警
        if summary['alerts_count'] > 0:
            for alert in self.alerts[-5:]:  # 显示最近5条
                icon = '🔴' if alert['level'] == 'critical' else '🟡'
        
    
    def export_report(self, format: str = 'json') -> str:
        """
        导出性能报告
        
        参数：
            format: 导出格式 (json, csv, html)
        
        返回：
            报告文件路径
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'json':
            return self._export_json(timestamp)
        elif format == 'csv':
            return self._export_csv(timestamp)
        elif format == 'html':
            return self._export_html(timestamp)
        else:
            raise ValueError(f"不支持的格式: {format}")
    
    def _export_json(self, timestamp: str) -> str:
        """导出JSON报告"""
        filepath = self.output_dir / f"performance_report_{timestamp}.json"
        
        report = {
            'summary': self.get_summary(),
            'daily_metrics': [d.to_dict() for d in self.daily_metrics],
            'alerts': self.alerts
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        return str(filepath)
    
    def _export_csv(self, timestamp: str) -> str:
        """导出CSV报告"""
        filepath = self.output_dir / f"performance_report_{timestamp}.csv"
        
        # 模块指标
        module_data = [m.to_dict() for m in self.module_metrics.values()]
        pd.DataFrame(module_data).to_csv(filepath, index=False)
        
        # 每日指标
        daily_filepath = self.output_dir / f"daily_metrics_{timestamp}.csv"
        daily_data = [d.to_dict() for d in self.daily_metrics]
        pd.DataFrame(daily_data).to_csv(daily_filepath, index=False)
        
        return str(filepath)
    
    def _export_html(self, timestamp: str) -> str:
        """导出HTML报告"""
        filepath = self.output_dir / f"performance_report_{timestamp}.html"
        
        summary = self.get_summary()
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ChainSight Performance Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 5px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #3498db; color: white; }}
        .bar {{ background: #3498db; height: 20px; border-radius: 3px; }}
        .alert-critical {{ color: #e74c3c; }}
        .alert-warning {{ color: #f39c12; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 ChainSight 性能报告</h1>
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h2>🌍 全局统计</h2>
        <p><strong>仿真期间:</strong> {summary['global']['simulation_start']} - {summary['global']['simulation_end']}</p>
        <p><strong>总天数:</strong> {summary['global']['total_days']}</p>
        <p><strong>总记录数:</strong> {summary['global']['total_records']:,}</p>
        <p><strong>缓存命中率:</strong> {summary['global']['cache_hit_rate']}</p>
        <p><strong>平均每天耗时:</strong> {summary['global']['avg_daily_time_s']}秒</p>
    </div>
    
    <div class="section">
        <h2>📦 模块性能</h2>
        <table>
            <tr>
                <th>模块</th>
                <th>调用次数</th>
                <th>总时间(s)</th>
                <th>平均(ms)</th>
                <th>最大(ms)</th>
                <th>成功率</th>
            </tr>
            {''.join(f'''
            <tr>
                <td>{name}</td>
                <td>{metrics['call_count']}</td>
                <td>{metrics['total_time_s']}</td>
                <td>{metrics['avg_time_ms']}</td>
                <td>{metrics['max_time_ms']}</td>
                <td>{metrics['success_rate']}</td>
            </tr>
            ''' for name, metrics in summary['modules'].items())}
        </table>
    </div>
    
    <div class="section">
        <h2>⏱️ 时间分布</h2>
        {''.join(f'''
        <p><strong>{name}:</strong></p>
        <div class="bar" style="width: {pct}"></div>
        <p>{pct}</p>
        ''' for name, pct in summary['module_distribution'].items())}
    </div>
    
    <div class="section">
        <h2>⚠️ 告警记录 ({len(self.alerts)})</h2>
        {''.join(f'''
        <p class="alert-{alert['level']}">
            [{alert['timestamp']}] {alert['message']}
        </p>
        ''' for alert in self.alerts)}
    </div>
</body>
</html>
"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return str(filepath)


class RealTimeMonitor:
    """
    实时性能监控器
    
    提供实时进度显示和性能指标
    """
    
    def __init__(self, total_days: int, update_interval: float = 1.0):
        self.total_days = total_days
        self.update_interval = update_interval
        self.current_day = 0
        self.start_time = None
        self.day_times: List[float] = []
        self._lock = threading.Lock()
        self._running = False
    
    def start(self):
        """开始监控"""
        self.start_time = time.time()
        self._running = True
    
    def stop(self):
        """停止监控"""
        self._running = False
    
    def update(self, day_number: int, day_time: float):
        """更新进度"""
        with self._lock:
            self.current_day = day_number
            self.day_times.append(day_time)
    
    def get_progress(self) -> Dict[str, Any]:
        """获取当前进度"""
        with self._lock:
            if not self.start_time:
                return {}
            
            elapsed = time.time() - self.start_time
            progress = self.current_day / max(self.total_days, 1)
            
            # 估算剩余时间
            if self.day_times:
                avg_day_time = np.mean(self.day_times)
                remaining_days = self.total_days - self.current_day
                eta_seconds = remaining_days * avg_day_time
            else:
                eta_seconds = 0
            
            return {
                'progress': f"{progress * 100:.1f}%",
                'current_day': self.current_day,
                'total_days': self.total_days,
                'elapsed_time': self._format_time(elapsed),
                'eta': self._format_time(eta_seconds),
                'avg_day_time': f"{np.mean(self.day_times) if self.day_times else 0:.2f}s",
                'throughput': f"{self.current_day / max(elapsed, 0.001):.2f} days/s"
            }
    
    def _format_time(self, seconds: float) -> str:
        """格式化时间"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds // 60:.0f}m {seconds % 60:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h {minutes:.0f}m"
    
    def print_progress_bar(self):
        """打印进度条"""
        progress = self.get_progress()
        if not progress:
            return
        
        bar_width = 40
        filled = int(bar_width * self.current_day / max(self.total_days, 1))
        bar = '█' * filled + '░' * (bar_width - filled)
        


# ===================== 全局仪表盘实例 =====================

_global_dashboard: Optional[PerformanceDashboard] = None


def get_dashboard() -> PerformanceDashboard:
    """获取全局仪表盘实例"""
    global _global_dashboard
    if _global_dashboard is None:
        _global_dashboard = PerformanceDashboard()
    return _global_dashboard


def reset_dashboard():
    """重置全局仪表盘"""
    global _global_dashboard
    _global_dashboard = PerformanceDashboard()


@contextmanager
def track_performance(module_name: str):
    """
    便捷的性能追踪装饰器
    
    用法:
        with track_performance('Module3'):
            module3.run(...)
    """
    dashboard = get_dashboard()
    with dashboard.track_module(module_name):
        yield dashboard
