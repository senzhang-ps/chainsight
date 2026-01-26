#!/usr/bin/env python3
"""
ChainSight 性能基准测试脚本

对比 ChainSight_Dev (原始版本) 与 src (重构版本) 的性能差异
生成详细的性能报告数据用于算法优化测试报告

使用方法:
    python tools/performance_benchmark.py --config test_files/BC_S5.xlsx --days 3

输出:
    - 详细的性能对比数据 (JSON)
    - 模块级别耗时分析
    - 可用于报告生成的结构化数据
"""

import sys
import os
import time
import json
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import argparse

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "ChainSight_Dev"))


class PerformanceBenchmark:
    """性能基准测试类"""
    
    def __init__(self, config_path: str, start_date: str, end_date: str):
        self.config_path = Path(config_path).resolve()
        self.start_date = start_date
        self.end_date = end_date
        self.results = {
            "test_info": {
                "config_file": str(self.config_path),
                "start_date": start_date,
                "end_date": end_date,
                "test_timestamp": datetime.now().isoformat(),
                "python_version": sys.version,
            },
            "dev_version": {},
            "refactored_version": {},
            "comparison": {}
        }
        
    def _calculate_days(self) -> int:
        """计算仿真天数"""
        from datetime import datetime
        start = datetime.strptime(self.start_date, "%Y-%m-%d")
        end = datetime.strptime(self.end_date, "%Y-%m-%d")
        return (end - start).days + 1
        
    def run_dev_version(self) -> Dict[str, Any]:
        """运行 ChainSight_Dev 原始版本"""
        print("\n" + "=" * 70)
        print("🔬 测试 ChainSight_Dev 原始版本")
        print("=" * 70)
        
        result = {
            "version": "ChainSight_Dev",
            "success": False,
            "total_time": 0,
            "module_times": {},
            "daily_times": [],
            "error": None
        }
        
        # 切换到 ChainSight_Dev 目录
        dev_dir = PROJECT_ROOT / "ChainSight_Dev"
        original_dir = os.getcwd()
        
        try:
            os.chdir(dev_dir)
            sys.path.insert(0, str(dev_dir))
            
            # 动态导入 ChainSight_Dev 的模块
            from main_integration import run_integrated_simulation, load_configuration
            
            # 创建临时输出目录
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = dev_dir / f"benchmark_dev_{ts}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"📁 输出目录: {output_dir}")
            print(f"📅 仿真日期: {self.start_date} 到 {self.end_date}")
            
            # 记录开始时间
            start_time = time.time()
            
            # 运行仿真
            sim_result = run_integrated_simulation(
                config_path=str(self.config_path),
                start_date=self.start_date,
                end_date=self.end_date,
                output_base_dir=str(output_dir),
                force_restart=True
            )
            
            end_time = time.time()
            total_time = end_time - start_time
            
            result["success"] = sim_result is not None and sim_result.get('simulation_completed', False)
            result["total_time"] = total_time
            result["output_dir"] = str(output_dir)
            
            # 尝试提取模块级别耗时（从日志或返回结果）
            if sim_result:
                result["module_times"] = sim_result.get('module_times', {})
                result["daily_times"] = sim_result.get('daily_times', [])
            
            print(f"\n✅ Dev版本完成: {total_time:.2f}秒")
            
        except Exception as e:
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
            print(f"\n❌ Dev版本出错: {e}")
            
        finally:
            os.chdir(original_dir)
            # 清理导入
            modules_to_remove = [k for k in sys.modules if 'main_integration' in k or 'module' in k.lower()]
            for mod in modules_to_remove:
                if mod in sys.modules:
                    del sys.modules[mod]
        
        return result
    
    def run_refactored_version(self) -> Dict[str, Any]:
        """运行重构后的 src 版本"""
        print("\n" + "=" * 70)
        print("🚀 测试重构后的 src 版本")
        print("=" * 70)
        
        result = {
            "version": "src_refactored",
            "success": False,
            "total_time": 0,
            "module_times": {},
            "daily_times": [],
            "error": None
        }
        
        original_dir = os.getcwd()
        
        try:
            os.chdir(PROJECT_ROOT)
            
            # 导入重构版本
            from src.core.main_integration import run_integrated_simulation, load_configuration
            
            # 创建输出目录
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = PROJECT_ROOT / "outputs" / f"benchmark_refactored_{ts}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"📁 输出目录: {output_dir}")
            print(f"📅 仿真日期: {self.start_date} 到 {self.end_date}")
            
            # 记录开始时间
            start_time = time.time()
            
            # 运行仿真
            sim_result = run_integrated_simulation(
                config_path=str(self.config_path),
                start_date=self.start_date,
                end_date=self.end_date,
                output_base_dir=str(output_dir),
                force_restart=True
            )
            
            end_time = time.time()
            total_time = end_time - start_time
            
            result["success"] = sim_result is not None and sim_result.get('simulation_completed', False)
            result["total_time"] = total_time
            result["output_dir"] = str(output_dir)
            
            # 提取模块级别耗时
            if sim_result:
                result["module_times"] = sim_result.get('module_times', {})
                result["daily_times"] = sim_result.get('daily_times', [])
            
            print(f"\n✅ 重构版本完成: {total_time:.2f}秒")
            
        except Exception as e:
            result["error"] = str(e)
            result["traceback"] = traceback.format_exc()
            print(f"\n❌ 重构版本出错: {e}")
            
        finally:
            os.chdir(original_dir)
        
        return result
    
    def compare_results(self):
        """对比两个版本的结果"""
        dev = self.results["dev_version"]
        ref = self.results["refactored_version"]
        
        comparison = {
            "both_successful": dev.get("success", False) and ref.get("success", False),
            "total_time_comparison": {},
            "module_comparison": {},
            "speedup_ratio": 0,
            "time_saved": 0,
            "time_saved_percent": 0
        }
        
        if dev.get("total_time") and ref.get("total_time"):
            dev_time = dev["total_time"]
            ref_time = ref["total_time"]
            
            comparison["total_time_comparison"] = {
                "dev_version": dev_time,
                "refactored_version": ref_time,
                "difference": dev_time - ref_time,
                "speedup_ratio": dev_time / ref_time if ref_time > 0 else 0
            }
            
            comparison["speedup_ratio"] = dev_time / ref_time if ref_time > 0 else 0
            comparison["time_saved"] = dev_time - ref_time
            comparison["time_saved_percent"] = ((dev_time - ref_time) / dev_time * 100) if dev_time > 0 else 0
        
        # 模块级别对比
        dev_modules = dev.get("module_times", {})
        ref_modules = ref.get("module_times", {})
        
        all_modules = set(dev_modules.keys()) | set(ref_modules.keys())
        for module in all_modules:
            dev_time = dev_modules.get(module, 0)
            ref_time = ref_modules.get(module, 0)
            comparison["module_comparison"][module] = {
                "dev_time": dev_time,
                "ref_time": ref_time,
                "speedup": dev_time / ref_time if ref_time > 0 else 0
            }
        
        self.results["comparison"] = comparison
        return comparison
    
    def run_benchmark(self) -> Dict[str, Any]:
        """运行完整的基准测试"""
        print("\n" + "=" * 70)
        print("🎯 ChainSight 性能基准测试")
        print("=" * 70)
        print(f"📋 配置文件: {self.config_path}")
        print(f"📅 测试日期范围: {self.start_date} 到 {self.end_date}")
        print(f"📊 仿真天数: {self._calculate_days()} 天")
        print("=" * 70)
        
        # 运行 Dev 版本
        self.results["dev_version"] = self.run_dev_version()
        
        # 运行重构版本
        self.results["refactored_version"] = self.run_refactored_version()
        
        # 对比结果
        self.compare_results()
        
        return self.results
    
    def print_summary(self):
        """打印测试摘要"""
        print("\n" + "=" * 70)
        print("📊 性能测试摘要")
        print("=" * 70)
        
        dev = self.results["dev_version"]
        ref = self.results["refactored_version"]
        comp = self.results["comparison"]
        
        print(f"\n🔬 ChainSight_Dev (原始版本):")
        print(f"   状态: {'✅ 成功' if dev.get('success') else '❌ 失败'}")
        print(f"   总耗时: {dev.get('total_time', 0):.2f} 秒")
        
        print(f"\n🚀 src (重构版本):")
        print(f"   状态: {'✅ 成功' if ref.get('success') else '❌ 失败'}")
        print(f"   总耗时: {ref.get('total_time', 0):.2f} 秒")
        
        if comp.get("both_successful"):
            print(f"\n📈 性能对比:")
            print(f"   加速比: {comp.get('speedup_ratio', 0):.2f}x")
            print(f"   节省时间: {comp.get('time_saved', 0):.2f} 秒 ({comp.get('time_saved_percent', 0):.1f}%)")
        
        print("\n" + "=" * 70)
    
    def save_results(self, output_path: Optional[str] = None):
        """保存测试结果到JSON文件"""
        if output_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = PROJECT_ROOT / "outputs" / f"benchmark_results_{ts}.json"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 结果已保存到: {output_path}")
        return output_path


def run_single_version_test(version: str, config_path: str, start_date: str, end_date: str) -> Dict[str, Any]:
    """
    单独运行一个版本的测试，用于分开测试
    
    Args:
        version: 'dev' 或 'refactored'
        config_path: 配置文件路径
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        测试结果字典
    """
    benchmark = PerformanceBenchmark(config_path, start_date, end_date)
    
    if version == 'dev':
        result = benchmark.run_dev_version()
    else:
        result = benchmark.run_refactored_version()
    
    return result


def main():
    parser = argparse.ArgumentParser(description="ChainSight 性能基准测试")
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--start-date", required=True, help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="结束日期 (YYYY-MM-DD)")
    parser.add_argument("--version", choices=['dev', 'refactored', 'both'], default='both',
                       help="测试版本: dev, refactored, 或 both")
    parser.add_argument("--output", help="输出结果文件路径")
    
    args = parser.parse_args()
    
    benchmark = PerformanceBenchmark(args.config, args.start_date, args.end_date)
    
    if args.version == 'dev':
        benchmark.results["dev_version"] = benchmark.run_dev_version()
    elif args.version == 'refactored':
        benchmark.results["refactored_version"] = benchmark.run_refactored_version()
    else:
        benchmark.run_benchmark()
    
    benchmark.print_summary()
    benchmark.save_results(args.output)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
