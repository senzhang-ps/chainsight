#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DuckDB vs Pandas 性能对比测试

测试不同数据规模下 DuckDB 和 Pandas 的性能差异，
帮助确定最优的切换阈值。

运行方法:
    python tools/benchmark_duckdb_vs_pandas.py
"""

import time
import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    print("⚠️ DuckDB 未安装，只测试 Pandas")


def generate_test_data(n_rows: int, seed: int = 42) -> pd.DataFrame:
    """生成测试数据"""
    np.random.seed(seed)
    
    materials = [f'MAT_{i:04d}' for i in range(min(1000, n_rows // 10))]
    locations = [f'LOC_{i:02d}' for i in range(min(50, n_rows // 100))]
    
    return pd.DataFrame({
        'material': np.random.choice(materials, n_rows),
        'location': np.random.choice(locations, n_rows),
        'quantity': np.random.randint(1, 1000, n_rows),
        'date': pd.date_range('2025-01-01', periods=n_rows, freq='h')[:n_rows],
        'value': np.random.random(n_rows) * 1000,
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_rows)
    })


def benchmark_merge(df1: pd.DataFrame, df2: pd.DataFrame, iterations: int = 3):
    """测试 merge 性能"""
    results = {'pandas': [], 'duckdb': []}
    
    # Pandas
    for _ in range(iterations):
        start = time.perf_counter()
        _ = pd.merge(df1, df2, on=['material', 'location'], how='left')
        results['pandas'].append(time.perf_counter() - start)
    
    # DuckDB
    if DUCKDB_AVAILABLE:
        conn = duckdb.connect(':memory:')
        for _ in range(iterations):
            conn.register('df1', df1)
            conn.register('df2', df2)
            start = time.perf_counter()
            _ = conn.execute("""
                SELECT df1.*, df2.value as value_right, df2.category as category_right
                FROM df1
                LEFT JOIN df2 ON df1.material = df2.material AND df1.location = df2.location
            """).fetchdf()
            results['duckdb'].append(time.perf_counter() - start)
            conn.unregister('df1')
            conn.unregister('df2')
        conn.close()
    
    return results


def benchmark_groupby(df: pd.DataFrame, iterations: int = 3):
    """测试 groupby 聚合性能"""
    results = {'pandas': [], 'duckdb': []}
    
    # Pandas
    for _ in range(iterations):
        start = time.perf_counter()
        _ = df.groupby(['material', 'location'], as_index=False).agg({
            'quantity': 'sum',
            'value': 'mean'
        })
        results['pandas'].append(time.perf_counter() - start)
    
    # DuckDB
    if DUCKDB_AVAILABLE:
        conn = duckdb.connect(':memory:')
        for _ in range(iterations):
            conn.register('df', df)
            start = time.perf_counter()
            _ = conn.execute("""
                SELECT material, location, SUM(quantity) as quantity, AVG(value) as value
                FROM df
                GROUP BY material, location
            """).fetchdf()
            results['duckdb'].append(time.perf_counter() - start)
            conn.unregister('df')
        conn.close()
    
    return results


def benchmark_filter(df: pd.DataFrame, iterations: int = 3):
    """测试 filter 性能"""
    results = {'pandas': [], 'duckdb': []}
    
    # Pandas
    for _ in range(iterations):
        start = time.perf_counter()
        _ = df[(df['quantity'] > 500) & (df['category'].isin(['A', 'B']))]
        results['pandas'].append(time.perf_counter() - start)
    
    # DuckDB
    if DUCKDB_AVAILABLE:
        conn = duckdb.connect(':memory:')
        for _ in range(iterations):
            conn.register('df', df)
            start = time.perf_counter()
            _ = conn.execute("""
                SELECT * FROM df
                WHERE quantity > 500 AND category IN ('A', 'B')
            """).fetchdf()
            results['duckdb'].append(time.perf_counter() - start)
            conn.unregister('df')
        conn.close()
    
    return results


def benchmark_sort(df: pd.DataFrame, iterations: int = 3):
    """测试排序性能"""
    results = {'pandas': [], 'duckdb': []}
    
    # Pandas
    for _ in range(iterations):
        start = time.perf_counter()
        _ = df.sort_values(by=['material', 'location', 'quantity'], ascending=[True, True, False])
        results['pandas'].append(time.perf_counter() - start)
    
    # DuckDB
    if DUCKDB_AVAILABLE:
        conn = duckdb.connect(':memory:')
        for _ in range(iterations):
            conn.register('df', df)
            start = time.perf_counter()
            _ = conn.execute("""
                SELECT * FROM df
                ORDER BY material ASC, location ASC, quantity DESC
            """).fetchdf()
            results['duckdb'].append(time.perf_counter() - start)
            conn.unregister('df')
        conn.close()
    
    return results


def format_time(seconds: float) -> str:
    """格式化时间"""
    if seconds < 0.001:
        return f"{seconds*1000000:.1f}μs"
    elif seconds < 1:
        return f"{seconds*1000:.2f}ms"
    else:
        return f"{seconds:.3f}s"


def print_results(operation: str, row_counts: list, all_results: dict):
    """打印测试结果"""
    print(f"\n{'='*70}")
    print(f"操作: {operation}")
    print(f"{'='*70}")
    print(f"{'行数':>12} | {'Pandas':>12} | {'DuckDB':>12} | {'加速比':>10} | 推荐")
    print(f"{'-'*70}")
    
    for n_rows in row_counts:
        pandas_time = np.mean(all_results[n_rows]['pandas'])
        duckdb_time = np.mean(all_results[n_rows]['duckdb']) if all_results[n_rows]['duckdb'] else float('inf')
        
        speedup = pandas_time / duckdb_time if duckdb_time > 0 else 0
        
        if speedup > 1.2:
            recommend = "🚀 DuckDB"
        elif speedup < 0.8:
            recommend = "📊 Pandas"
        else:
            recommend = "≈ 相当"
        
        print(f"{n_rows:>12,} | {format_time(pandas_time):>12} | {format_time(duckdb_time):>12} | {speedup:>10.2f}x | {recommend}")


def main():
    print("=" * 70)
    print("DuckDB vs Pandas 性能对比测试")
    print("=" * 70)
    
    # 测试不同数据规模
    row_counts = [1000, 5000, 10000, 50000, 100000, 500000]
    
    # 预生成数据
    print("\n📊 生成测试数据...")
    test_data = {}
    for n_rows in row_counts:
        test_data[n_rows] = generate_test_data(n_rows)
        print(f"  - {n_rows:>10,} 行: OK")
    
    # 测试 merge
    print("\n🔗 测试 merge 操作...")
    merge_results = {}
    for n_rows in row_counts:
        df1 = test_data[n_rows]
        # 使用相同数据的一半作为右表
        df2 = df1.iloc[:len(df1)//2].copy()
        merge_results[n_rows] = benchmark_merge(df1, df2)
        print(f"  - {n_rows:>10,} 行: 完成")
    print_results("MERGE (LEFT JOIN)", row_counts, merge_results)
    
    # 测试 groupby
    print("\n📊 测试 groupby 操作...")
    groupby_results = {}
    for n_rows in row_counts:
        groupby_results[n_rows] = benchmark_groupby(test_data[n_rows])
        print(f"  - {n_rows:>10,} 行: 完成")
    print_results("GROUPBY (SUM, AVG)", row_counts, groupby_results)
    
    # 测试 filter
    print("\n🔍 测试 filter 操作...")
    filter_results = {}
    for n_rows in row_counts:
        filter_results[n_rows] = benchmark_filter(test_data[n_rows])
        print(f"  - {n_rows:>10,} 行: 完成")
    print_results("FILTER (多条件)", row_counts, filter_results)
    
    # 测试 sort
    print("\n📋 测试 sort 操作...")
    sort_results = {}
    for n_rows in row_counts:
        sort_results[n_rows] = benchmark_sort(test_data[n_rows])
        print(f"  - {n_rows:>10,} 行: 完成")
    print_results("SORT (多列)", row_counts, sort_results)
    
    # 总结推荐阈值
    print("\n" + "=" * 70)
    print("📋 测试结论与推荐配置")
    print("=" * 70)
    print("""
⚠️  重要发现：在内存数据处理场景下，Pandas 通常比 DuckDB 更快！

原因分析:
    1. DuckDB 每次查询有启动开销（连接、注册表、编译SQL）
    2. Pandas + NumPy 的向量化操作已经非常高效
    3. 数据已在内存中时，DuckDB 的磁盘优化优势不明显

DuckDB 适用场景:
    ✅ 超大数据集 (500K+ 行)
    ✅ 复杂多表 JOIN
    ✅ 需要 SQL 语法便利性
    ✅ 数据在磁盘上（直接查询 Parquet/CSV）

推荐配置 (src/utils/optimization_config.py):
    
    USE_DUCKDB = False              # 默认关闭
    DUCKDB_MIN_ROWS = 100000        # 高阈值
    DUCKDB_LARGE_TABLE_ROWS = 500000
    
环境变量控制:
    # Windows PowerShell
    $env:CHAINSIGHT_USE_DUCKDB = "true"   # 启用 DuckDB
    $env:CHAINSIGHT_USE_DUCKDB = "false"  # 禁用 DuckDB
    
    # Linux/macOS
    export CHAINSIGHT_USE_DUCKDB=true
    export CHAINSIGHT_USE_DUCKDB=false

结论: 保持当前 Pandas 实现，DuckDB 作为可选项保留。
""")
    
    print("=" * 70)


if __name__ == '__main__':
    main()
