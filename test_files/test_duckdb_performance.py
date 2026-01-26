# -*- coding: utf-8 -*-
"""
DuckDB 性能对比测试脚本

测试 DuckDB 向量化计算 vs Pandas 循环计算的性能差异。

Usage:
    python test_duckdb_performance.py

测试内容:
1. Module3 净需求计算性能对比
2. Module5 MOQ/RV 处理性能对比
3. Module5 优先级分配性能对比
"""

import os
import sys
import time
import statistics
from datetime import datetime
from typing import Dict, List, Any, Tuple

import pandas as pd
import numpy as np

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'pgsql_db'))

# 导入测试模块
try:
    from pgsql_db.duckdb_integration import (
        DuckDBConfig,
        get_duckdb_calculator,
        close_duckdb_calculator,
        calculate_net_demand_batch_duckdb,
        _calculate_net_demand_pandas,
        apply_moq_rv_batch_duckdb,
        _apply_moq_rv_pandas,
        priority_allocation_batch_duckdb,
        _priority_allocation_pandas,
        run_ab_comparison,
    )
    DUCKDB_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 无法导入 DuckDB 集成模块: {e}")
    DUCKDB_AVAILABLE = False


def generate_test_data(
    n_nodes: int = 1000,
    n_demand_rows: int = 5000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    生成测试数据
    
    Args:
        n_nodes: 节点数量（用于净需求计算）
        n_demand_rows: 需求行数量（用于MOQ/RV和优先级分配）
        seed: 随机种子
    
    Returns:
        测试数据字典
    """
    np.random.seed(seed)
    
    # 生成物料和位置
    materials = [f"MAT{i:05d}" for i in range(100)]
    locations = [f"{i:04d}" for i in range(50)]
    demand_elements = ['AO', 'Forecast', 'SafetyStock', 'Transfer']
    
    # 净需求计算测试数据
    nodes_df = pd.DataFrame({
        'material': np.random.choice(materials, n_nodes),
        'location': np.random.choice(locations, n_nodes),
        'beginning_inventory': np.random.randint(0, 1000, n_nodes),
        'in_transit': np.random.randint(0, 500, n_nodes),
        'delivery_gr': np.random.randint(0, 200, n_nodes),
        'production': np.random.randint(0, 300, n_nodes),
        'shipment': np.random.randint(0, 400, n_nodes),
        'open_deployment_out': np.random.randint(0, 200, n_nodes),
        'open_deployment_in': np.random.randint(0, 200, n_nodes),
        'ao_demand': np.random.randint(0, 500, n_nodes),
        'forecast_demand': np.random.randint(0, 800, n_nodes),
        'safety_stock': np.random.randint(0, 300, n_nodes),
        'downstream_ao_gap': np.random.randint(0, 100, n_nodes).astype(float),
        'downstream_fc_gap': np.random.randint(0, 150, n_nodes).astype(float),
        'downstream_ss_gap': np.random.randint(0, 80, n_nodes).astype(float),
    })
    
    # MOQ/RV 测试数据
    moq_rv_demand = pd.DataFrame({
        'material': np.random.choice(materials, n_demand_rows),
        'sending': np.random.choice(locations, n_demand_rows),
        'receiving': np.random.choice(locations, n_demand_rows),
        'quantity': np.random.randint(1, 500, n_demand_rows),
    })
    
    moq_rv_config = pd.DataFrame({
        'material': materials,
        'sending': np.random.choice(locations, len(materials)),
        'moq': np.random.choice([1, 10, 50, 100], len(materials)),
        'rv': np.random.choice([1, 5, 10, 20], len(materials)),
    })
    
    # 优先级分配测试数据
    priority_demand = pd.DataFrame({
        'material': np.random.choice(materials, n_demand_rows),
        'sending': np.random.choice(locations, n_demand_rows),
        'receiving': np.random.choice(locations, n_demand_rows),
        'demand_element': np.random.choice(demand_elements, n_demand_rows),
        'demand_qty': np.random.randint(1, 300, n_demand_rows),
    })
    
    # 生成库存数据 - 每个material-location组合
    inv_combos = [(m, l) for m in materials[:20] for l in locations[:10]]
    priority_inventory = pd.DataFrame({
        'material': [c[0] for c in inv_combos],
        'location': [c[1] for c in inv_combos],
        'qty': np.random.randint(0, 2000, len(inv_combos)),
    })
    
    priority_config = pd.DataFrame({
        'demand_element': demand_elements,
        'priority': [1, 2, 3, 4],
    })
    
    return {
        'nodes_df': nodes_df,
        'moq_rv_demand': moq_rv_demand,
        'moq_rv_config': moq_rv_config,
        'priority_demand': priority_demand,
        'priority_inventory': priority_inventory,
        'priority_config': priority_config,
    }


def run_benchmark(
    func_duckdb,
    func_pandas,
    test_data: Any,
    iterations: int = 5,
    warmup: int = 2,
    name: str = "Unknown"
) -> Dict[str, Any]:
    """
    运行基准测试
    
    Args:
        func_duckdb: DuckDB 版本函数
        func_pandas: Pandas 版本函数
        test_data: 测试数据
        iterations: 迭代次数
        warmup: 预热次数
        name: 测试名称
    
    Returns:
        测试结果
    """
    results = {'duckdb': [], 'pandas': []}
    
    print(f"\n{'='*60}")
    print(f"📊 测试: {name}")
    print(f"{'='*60}")
    
    # 预热
    print(f"  🔄 预热 ({warmup} 次)...")
    for _ in range(warmup):
        try:
            func_duckdb(test_data)
        except Exception as e:
            print(f"    ⚠️ DuckDB 预热失败: {e}")
        try:
            func_pandas(test_data)
        except Exception as e:
            print(f"    ⚠️ Pandas 预热失败: {e}")
    
    # 正式测试
    print(f"  🏃 正式测试 ({iterations} 次)...")
    
    for i in range(iterations):
        # DuckDB
        try:
            t0 = time.perf_counter()
            result_duck = func_duckdb(test_data)
            elapsed_duck = (time.perf_counter() - t0) * 1000
            results['duckdb'].append(elapsed_duck)
        except Exception as e:
            print(f"    ⚠️ DuckDB 迭代 {i+1} 失败: {e}")
            results['duckdb'].append(float('inf'))
        
        # Pandas
        try:
            t0 = time.perf_counter()
            result_pandas = func_pandas(test_data)
            elapsed_pandas = (time.perf_counter() - t0) * 1000
            results['pandas'].append(elapsed_pandas)
        except Exception as e:
            print(f"    ⚠️ Pandas 迭代 {i+1} 失败: {e}")
            results['pandas'].append(float('inf'))
    
    # 计算统计
    def calc_stats(times: List[float]) -> Dict[str, float]:
        valid = [t for t in times if t != float('inf')]
        if not valid:
            return {'mean': float('inf'), 'std': 0, 'min': float('inf'), 'max': float('inf')}
        return {
            'mean': statistics.mean(valid),
            'std': statistics.stdev(valid) if len(valid) > 1 else 0,
            'min': min(valid),
            'max': max(valid),
        }
    
    duck_stats = calc_stats(results['duckdb'])
    pandas_stats = calc_stats(results['pandas'])
    
    speedup = pandas_stats['mean'] / max(duck_stats['mean'], 0.001)
    
    # 打印结果
    print(f"\n  📈 结果:")
    print(f"    DuckDB: {duck_stats['mean']:.2f}ms (±{duck_stats['std']:.2f})")
    print(f"    Pandas: {pandas_stats['mean']:.2f}ms (±{pandas_stats['std']:.2f})")
    print(f"    加速比: {speedup:.2f}x {'✅' if speedup > 1 else '❌'}")
    
    return {
        'name': name,
        'duckdb': duck_stats,
        'pandas': pandas_stats,
        'speedup': speedup,
        'iterations': iterations,
    }


def test_net_demand_performance(test_data: Dict[str, Any]) -> Dict[str, Any]:
    """测试净需求计算性能"""
    nodes_df = test_data['nodes_df']
    
    def func_duckdb(df):
        return calculate_net_demand_batch_duckdb(df)
    
    def func_pandas(df):
        return _calculate_net_demand_pandas(df)
    
    return run_benchmark(
        func_duckdb, func_pandas, nodes_df,
        name=f"净需求计算 ({len(nodes_df)} 节点)"
    )


def test_moq_rv_performance(test_data: Dict[str, Any]) -> Dict[str, Any]:
    """测试 MOQ/RV 处理性能"""
    demand_df = test_data['moq_rv_demand']
    config_df = test_data['moq_rv_config']
    
    def func_duckdb(data):
        return apply_moq_rv_batch_duckdb(data[0], data[1])
    
    def func_pandas(data):
        return _apply_moq_rv_pandas(data[0], data[1])
    
    return run_benchmark(
        func_duckdb, func_pandas, (demand_df, config_df),
        name=f"MOQ/RV 处理 ({len(demand_df)} 行)"
    )


def test_priority_allocation_performance(test_data: Dict[str, Any]) -> Dict[str, Any]:
    """测试优先级分配性能"""
    demand_df = test_data['priority_demand']
    inventory_df = test_data['priority_inventory']
    priority_df = test_data['priority_config']
    
    def func_duckdb(data):
        return priority_allocation_batch_duckdb(data[0], data[1], data[2])
    
    def func_pandas(data):
        return _priority_allocation_pandas(data[0], data[1], data[2])
    
    return run_benchmark(
        func_duckdb, func_pandas, (demand_df, inventory_df, priority_df),
        name=f"优先级分配 ({len(demand_df)} 行)"
    )


def run_scalability_test(sizes: List[int] = [1000, 5000, 10000, 50000, 100000]) -> List[Dict]:
    """
    运行可扩展性测试
    
    测试不同数据规模下的性能表现。
    """
    print("\n" + "="*70)
    print("📊 可扩展性测试")
    print("="*70)
    
    results = []
    
    for size in sizes:
        print(f"\n🔢 数据规模: {size}")
        test_data = generate_test_data(n_nodes=size, n_demand_rows=size*2)
        
        result = test_net_demand_performance(test_data)
        result['size'] = size
        results.append(result)
    
    # 打印汇总
    print("\n" + "="*70)
    print("📋 可扩展性测试汇总")
    print("="*70)
    print(f"{'数据量':<10} {'DuckDB(ms)':<15} {'Pandas(ms)':<15} {'加速比':<10}")
    print("-"*50)
    
    for r in results:
        print(f"{r['size']:<10} {r['duckdb']['mean']:<15.2f} {r['pandas']['mean']:<15.2f} {r['speedup']:<10.2f}x")
    
    return results


def main():
    """主测试函数"""
    print("="*70)
    print("🚀 DuckDB 性能对比测试")
    print(f"   测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    if not DUCKDB_AVAILABLE:
        print("\n❌ DuckDB 集成不可用，无法运行测试")
        return
    
    # 配置 DuckDB
    DuckDBConfig.enabled = True
    DuckDBConfig.memory_limit = "4GB"
    DuckDBConfig.threads = 4
    DuckDBConfig.min_rows_threshold = 10  # 测试时降低阈值
    
    print(f"\n📝 DuckDB 配置:")
    print(f"   内存限制: {DuckDBConfig.memory_limit}")
    print(f"   线程数: {DuckDBConfig.threads}")
    
    # 生成测试数据
    print("\n📦 生成测试数据...")
    test_data = generate_test_data(n_nodes=2000, n_demand_rows=5000)
    print(f"   节点数: {len(test_data['nodes_df'])}")
    print(f"   需求行数: {len(test_data['moq_rv_demand'])}")
    
    # 运行各项测试
    all_results = []
    
    # 1. 净需求计算
    try:
        result = test_net_demand_performance(test_data)
        all_results.append(result)
    except Exception as e:
        print(f"❌ 净需求计算测试失败: {e}")
    
    # 2. MOQ/RV 处理
    try:
        result = test_moq_rv_performance(test_data)
        all_results.append(result)
    except Exception as e:
        print(f"❌ MOQ/RV 测试失败: {e}")
    
    # 3. 优先级分配
    try:
        result = test_priority_allocation_performance(test_data)
        all_results.append(result)
    except Exception as e:
        print(f"❌ 优先级分配测试失败: {e}")
    
    # 4. 可扩展性测试
    try:
        scalability_results = run_scalability_test([100, 500, 1000, 2000, 5000])
    except Exception as e:
        print(f"❌ 可扩展性测试失败: {e}")
    
    # 打印总结
    print("\n" + "="*70)
    print("📋 测试总结")
    print("="*70)
    
    total_speedup = []
    for r in all_results:
        if r['speedup'] != float('inf'):
            total_speedup.append(r['speedup'])
            status = "✅" if r['speedup'] > 1 else "❌"
            print(f"  {status} {r['name']}: {r['speedup']:.2f}x 加速")
    
    if total_speedup:
        avg_speedup = statistics.mean(total_speedup)
        print(f"\n  📊 平均加速比: {avg_speedup:.2f}x")
    
    # 清理
    close_duckdb_calculator()
    print("\n✅ 测试完成")


if __name__ == '__main__':
    main()
