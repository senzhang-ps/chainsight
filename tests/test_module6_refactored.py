# -*- coding: utf-8 -*-
"""
Module6 重构后的单元测试

测试重构后的各子模块功能是否正常
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_expression_evaluator():
    """测试安全表达式解析器"""
    print("\n测试 SafeExpressionEvaluator...")
    
    from src.modules.logistics_execution.expression_evaluator import SafeExpressionEvaluator
    
    evaluator = SafeExpressionEvaluator(['x', 'y', 'waiting_days'])
    
    # 测试基本比较
    assert evaluator.eval('x > 5', {'x': 10, 'y': 3}) == True
    assert evaluator.eval('x > 5', {'x': 3, 'y': 3}) == False
    
    # 测试逻辑运算
    assert evaluator.eval('x > 5 and y < 5', {'x': 10, 'y': 3}) == True
    assert evaluator.eval('x > 5 or y > 5', {'x': 3, 'y': 10}) == True
    
    # 测试SQL风格操作符转换
    assert evaluator.eval('x > 5 AND y < 5', {'x': 10, 'y': 3}) == True
    assert evaluator.eval('waiting_days >= 3', {'x': 1, 'y': 2, 'waiting_days': 5}) == True
    
    print("  ✅ SafeExpressionEvaluator 测试通过")


def test_validators():
    """测试验证器模块"""
    print("\n测试 validators...")
    
    from src.modules.logistics_execution.validators import (
        check_and_deduplicate,
        validate_deployment_plan,
        validate_truck_config
    )
    
    # 测试去重
    df = pd.DataFrame({
        'material': ['A', 'A', 'B', 'C'],
        'quantity': [10, 20, 30, 40]
    })
    validation_log = []
    result = check_and_deduplicate(df, 'material', 'TestSheet', validation_log)
    
    assert len(result) == 3  # 应该只有3条记录
    assert len(validation_log) == 1  # 应该有一条警告
    
    # 测试部署计划验证
    dp = pd.DataFrame({
        'material': ['A'],
        'sending': ['WH1'],
        'receiving': ['WH2'],
        'deployed_qty': [100]
    })
    validation_log = []
    result = validate_deployment_plan(dp, validation_log)
    
    # 应该添加了缺失的列
    assert 'demand_element' in result.columns
    assert 'planned_deployment_date' in result.columns
    
    print("  ✅ validators 测试通过")


def test_capacity_manager():
    """测试容量管理器"""
    print("\n测试 capacity_manager...")
    
    from src.modules.logistics_execution.capacity_manager import (
        normalize_capacity_plan,
        build_capacity_map,
        get_optimal_truck_sequence
    )
    
    # 测试日粒度容量标准化
    cap_df = pd.DataFrame({
        'date': pd.to_datetime(['2024-01-01', '2024-01-02']),
        'sending': ['WH1', 'WH1'],
        'receiving': ['WH2', 'WH2'],
        'truck_type': ['Type1', 'Type1'],
        'truck_number': [5, 3]
    })
    
    sim_start = pd.Timestamp('2024-01-01')
    sim_end = pd.Timestamp('2024-01-02')
    
    result = normalize_capacity_plan(cap_df, sim_start, sim_end)
    assert len(result) == 2
    
    # 测试容量映射
    cap_map = build_capacity_map(result)
    assert len(cap_map) == 2
    
    # 测试车型序列
    truck_cfgs = pd.DataFrame({
        'truck_type': ['Type1', 'Type2', 'Type3'],
        'optimal_type': ['N', 'Y', 'N']
    })
    seq = get_optimal_truck_sequence(truck_cfgs)
    assert seq[0] == 'Type2'  # 最优车型应该在前面
    
    print("  ✅ capacity_manager 测试通过")


def test_vehicle_packer():
    """测试车辆装载器"""
    print("\n测试 vehicle_packer...")
    
    from src.modules.logistics_execution.vehicle_packer import (
        VehiclePacker,
        calculate_load_ratios,
        determine_trigger_cause
    )
    
    # 创建装载器
    packer = VehiclePacker(cap_weight=1000, cap_volume=100)
    
    assert packer.current_weight == 0
    assert packer.current_volume == 0
    assert not packer.has_load()
    assert not packer.is_full()
    
    # 测试装载比例计算
    wfr, vfr = calculate_load_ratios(500, 50, 1000, 100)
    assert wfr == 0.5
    assert vfr == 0.5
    
    # 测试触发原因判断
    cause = determine_trigger_cause(
        has_load=True, wfr=0.8, vfr=0.5,
        wfr_threshold=0.7, vfr_threshold=0.7,
        bypass=False, max_wait_in_load=5, max_wait_days=30
    )
    assert cause == 'threshold'
    
    cause = determine_trigger_cause(
        has_load=True, wfr=0.5, vfr=0.5,
        wfr_threshold=0.7, vfr_threshold=0.7,
        bypass=True, max_wait_in_load=5, max_wait_days=30
    )
    assert cause == 'bypass'
    
    cause = determine_trigger_cause(
        has_load=True, wfr=0.5, vfr=0.5,
        wfr_threshold=0.7, vfr_threshold=0.7,
        bypass=False, max_wait_in_load=30, max_wait_days=30
    )
    assert cause == 'force_wait_timeout'
    
    print("  ✅ vehicle_packer 测试通过")


def test_delivery_processor():
    """测试发货处理器"""
    print("\n测试 delivery_processor...")
    
    from src.modules.logistics_execution.delivery_processor import (
        sample_delivery_delay,
        calculate_actual_delivery_date
    )
    
    # 测试空延迟分布
    delay = sample_delivery_delay('WH1', 'WH2', pd.DataFrame())
    assert delay == 0
    
    # 测试有效延迟分布
    delay_dist = pd.DataFrame({
        'sending': ['WH1', 'ALL'],
        'receiving': ['WH2', 'ALL'],
        'delay_days': [2, 0],
        'probability': [1.0, 1.0]
    })
    
    np.random.seed(42)
    delay = sample_delivery_delay('WH1', 'WH2', delay_dist)
    assert delay == 2  # 精确匹配应该返回2天
    
    # 测试全局兜底
    delay = sample_delivery_delay('WH3', 'WH4', delay_dist)
    assert delay == 0  # 使用ALL匹配
    
    # 测试交货日期计算
    ship_date = pd.Timestamp('2024-01-01')
    eta = calculate_actual_delivery_date(ship_date, otd=2, gr=1, delay=1)
    assert eta == pd.Timestamp('2024-01-05')
    
    print("  ✅ delivery_processor 测试通过")


def test_inventory_manager():
    """测试库存管理器"""
    print("\n测试 inventory_manager...")
    
    from src.modules.logistics_execution.inventory_manager import (
        update_inventory_after_load,
        get_available_inventory,
        calculate_inventory_limit,
        has_sufficient_inventory
    )
    
    inventory = {
        ('MAT_A', 'WH1'): 100.0,
        ('MAT_B', 'WH1'): 50.0
    }
    
    # 测试获取可用库存
    assert get_available_inventory(inventory, 'MAT_A', 'WH1') == 100.0
    assert get_available_inventory(inventory, 'MAT_C', 'WH1') == 0  # 不存在
    
    # 测试库存限制计算
    limit = calculate_inventory_limit(inventory, 'MAT_A', 'WH1', 30)
    assert limit == 70  # 100 - 30 已装载
    
    # 测试库存充足性检查
    assert has_sufficient_inventory(inventory, 'MAT_A', 'WH1', 50) == True
    assert has_sufficient_inventory(inventory, 'MAT_A', 'WH1', 150) == False
    
    # 测试装载后更新
    inventory = update_inventory_after_load(inventory, 'MAT_A', 'WH1', 30)
    assert inventory[('MAT_A', 'WH1')] == 70.0
    
    print("  ✅ inventory_manager 测试通过")


def test_module6_import():
    """测试主模块导入"""
    print("\n测试主模块 module6 导入...")
    
    from src.modules.module6 import (
        run_physical_flow_module,
        run_daily_physical_flow,
        main
    )
    
    assert callable(run_physical_flow_module)
    assert callable(run_daily_physical_flow)
    assert main == run_physical_flow_module
    
    print("  ✅ module6 主模块导入测试通过")


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("Module6 重构后单元测试")
    print("=" * 60)
    
    tests = [
        test_expression_evaluator,
        test_validators,
        test_capacity_manager,
        test_vehicle_packer,
        test_delivery_processor,
        test_inventory_manager,
        test_module6_import,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ❌ {test.__name__} 失败: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
