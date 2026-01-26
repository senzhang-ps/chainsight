# -*- coding: utf-8 -*-
"""测试高性能引擎"""

import sys
sys.path.insert(0, '.')

# 测试 DuckDB 是否可用
try:
    import duckdb
    print(f'✅ DuckDB 版本: {duckdb.__version__}')
except ImportError as e:
    print(f'❌ DuckDB 不可用: {e}')
    sys.exit(1)

# 测试高性能引擎模块
try:
    from pgsql_db.high_performance_engine import (
        DuckDBCalculator,
        FastChangeDetector,
        HighPerformanceEngine,
        create_high_performance_engine
    )
    print('✅ high_performance_engine 模块加载成功')
    
    # 创建引擎实例
    engine = create_high_performance_engine()
    print('✅ HighPerformanceEngine 实例化成功')
    
    # 测试 DuckDB 计算
    import pandas as pd
    test_df = pd.DataFrame({
        'material': ['A', 'B', 'C'],
        'location': ['L1', 'L2', 'L3'],
        'quantity': [100, 200, 300]
    })
    
    # 测试注册和查询
    engine.calculator.conn.register('test', test_df)
    result = engine.calculator.conn.execute('SELECT SUM(quantity) as total FROM test').fetchdf()
    total = result['total'].iloc[0]
    print(f'✅ DuckDB 查询测试: 总量 = {total}')
    
    # 测试净需求计算
    print('\n测试净需求计算...')
    demand_df = pd.DataFrame({
        'material': ['A', 'A', 'B'],
        'location': ['L1', 'L1', 'L2'],
        'date': ['2025-10-06', '2025-10-06', '2025-10-06'],
        'quantity': [100, 50, 200]
    })
    
    supply_df = pd.DataFrame({
        'material': ['A', 'B'],
        'location': ['L1', 'L2'],
        'qty': [80, 150]
    })
    
    ss_df = pd.DataFrame({
        'material': ['A', 'B'],
        'location': ['L1', 'L2'],
        'date': ['2025-10-06', '2025-10-06'],
        'safety_stock_qty': [20, 30]
    })
    
    result = engine.calculator.calculate_net_demand_batch(
        demand_df, supply_df, ss_df, '2025-10-06'
    )
    print('净需求计算结果:')
    print(result)
    
    # 测试 MOQ/RV
    print('\n测试 MOQ/RV 应用...')
    deploy_df = pd.DataFrame({
        'material': ['A', 'B', 'C'],
        'sending': ['S1', 'S2', 'S3'],
        'receiving': ['R1', 'R2', 'R3'],
        'quantity': [45, 75, 120]
    })
    
    config_df = pd.DataFrame({
        'material': ['A', 'B', 'C'],
        'sending': ['S1', 'S2', 'S3'],
        'moq': [50, 100, 50],
        'rv': [10, 25, 25]
    })
    
    result = engine.calculator.apply_moq_rv_batch(deploy_df, config_df)
    print('MOQ/RV 应用结果:')
    print(result[['material', 'quantity', 'moq', 'rv', 'adjusted_qty']])
    
    # 测试优先级分配
    print('\n测试优先级分配...')
    demand_df = pd.DataFrame({
        'material': ['A', 'A', 'A'],
        'sending': ['S1', 'S1', 'S1'],
        'receiving': ['R1', 'R2', 'R3'],
        'demand_element': ['DE1', 'DE2', 'DE3'],
        'demand_qty': [100, 150, 200]
    })
    
    inv_df = pd.DataFrame({
        'material': ['A'],
        'location': ['S1'],
        'qty': [250]
    })
    
    priority_df = pd.DataFrame({
        'demand_element': ['DE1', 'DE2', 'DE3'],
        'priority': [1, 3, 2]
    })
    
    result = engine.calculator.priority_allocation_batch(
        demand_df, inv_df, priority_df
    )
    print('优先级分配结果:')
    print(result[['material', 'receiving', 'demand_element', 'demand_qty', 'priority_rank', 'allocated_qty', 'unmet_qty']])
    
    # 打印统计
    engine.print_stats()
    
    engine.close()
    print('\n✅ 所有测试通过!')
    
except Exception as e:
    import traceback
    print(f'❌ 测试失败: {e}')
    traceback.print_exc()
    sys.exit(1)
