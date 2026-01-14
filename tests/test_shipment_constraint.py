#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证订单量与出货量约束的单元测试。

确保：
1. 出货量 (delivery_qty) <= 订单量 (shipment_qty)
2. 部署量 (deployed_qty) 不会因 MOQ/RV 调整而超过订单量
"""
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from src.modules.deployment_planning.allocation import (
    apply_moq_rv,
    apply_grouped_moq_rv
)


class TestApplyMoqRv:
    """测试 apply_moq_rv 函数"""
    
    def test_basic_moq_rv_without_limit(self):
        """测试基本 MOQ/RV 应用（无上限）"""
        # qty < moq 时应返回 moq
        assert apply_moq_rv(30, 50, 20) == 50
        
        # qty >= moq 时向上取整到 rv 倍数
        assert apply_moq_rv(100, 50, 20) == 100
        assert apply_moq_rv(105, 50, 20) == 120
        
    def test_moq_rv_with_max_qty_limit(self):
        """测试带上限的 MOQ/RV 应用"""
        # 当 MOQ 调整会超出 max_qty 时，应限制为 max_qty
        assert apply_moq_rv(30, 50, 20, max_qty=40) == 40
        
        # RV 调整也不应超过 max_qty
        assert apply_moq_rv(105, 50, 20, max_qty=100) == 100
        
    def test_moq_rv_limit_no_effect_when_not_exceeded(self):
        """测试当不超出上限时，限制不起作用"""
        assert apply_moq_rv(30, 50, 20, max_qty=100) == 50
        assert apply_moq_rv(100, 50, 20, max_qty=200) == 100
        
    def test_self_loop_no_moq_rv(self):
        """测试自循环不应用 MOQ/RV"""
        # 自循环时返回原值
        assert apply_moq_rv(30, 50, 20, is_cross_node=False) == 30
        
        # 自循环但有上限时仍应限制
        assert apply_moq_rv(100, 50, 20, is_cross_node=False, max_qty=50) == 50
        
    def test_zero_qty(self):
        """测试零数量"""
        assert apply_moq_rv(0, 50, 20) == 0
        assert apply_moq_rv(0, 50, 20, max_qty=100) == 0


class TestApplyGroupedMoqRv:
    """测试 apply_grouped_moq_rv 函数"""
    
    def test_basic_grouped_without_limit(self):
        """测试基本分组 MOQ/RV（无上限）"""
        demand_rows = [
            {'material': 'M1', 'demand_qty': 30, 'moq': 50, 'rv': 20, 
             'from_location': 'B', 'receiving': 'B'},
        ]
        result = apply_grouped_moq_rv(demand_rows, 'A')  # A -> B 跨节点
        assert result[0] == 50  # 应用 MOQ
        
    def test_grouped_with_shipment_limit(self):
        """测试带订单量上限的分组 MOQ/RV"""
        demand_rows = [
            {'material': 'M1', 'demand_qty': 30, 'moq': 50, 'rv': 20,
             'from_location': 'B', 'receiving': 'B'},
        ]
        result = apply_grouped_moq_rv(demand_rows, 'A', shipment_qty_limit=40)
        assert result[0] == 40  # 被订单量上限限制
        
    def test_grouped_multiple_items(self):
        """测试多项需求的分组处理"""
        demand_rows = [
            {'material': 'M1', 'demand_qty': 20, 'moq': 50, 'rv': 10,
             'from_location': 'B', 'receiving': 'B'},
            {'material': 'M1', 'demand_qty': 15, 'moq': 50, 'rv': 10,
             'from_location': 'B', 'receiving': 'B'},
        ]
        # 总需求 35 < MOQ 50，应调整为 50
        result = apply_grouped_moq_rv(demand_rows, 'A')
        total = sum(result.values())
        assert total == 50
        
        # 加上限为 35，不应超过
        result_limited = apply_grouped_moq_rv(demand_rows, 'A', shipment_qty_limit=35)
        total_limited = sum(result_limited.values())
        assert total_limited == 35


class TestConstraintEnforcement:
    """测试约束强制执行"""
    
    def test_delivery_never_exceeds_shipment(self):
        """验证出货量永远不超过订单量的原则"""
        # 模拟场景：订单量 100，MOQ 要求 120
        shipment_qty = 100
        demanded_qty = 80
        moq = 120
        rv = 20
        
        # 应用约束后的部署量
        deployed = apply_moq_rv(demanded_qty, moq, rv, max_qty=shipment_qty)
        
        # 验证：部署量不超过订单量
        assert deployed <= shipment_qty
        assert deployed == 100  # 被限制为 shipment_qty


def run_tests():
    """运行所有测试"""
    print("=" * 60)
    print("🧪 订单量与出货量约束验证测试")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    # 测试 1: 基本 MOQ/RV 应用（无上限）
    print("\n📝 测试 1: 基本 MOQ/RV 应用（无上限）")
    try:
        assert apply_moq_rv(30, 50, 20) == 50, "qty < moq 时应返回 moq"
        assert apply_moq_rv(100, 50, 20) == 100, "qty >= moq 时保持原值"
        assert apply_moq_rv(105, 50, 20) == 120, "应向上取整到 rv 倍数"
        print("   ✅ 通过")
        passed += 1
    except AssertionError as e:
        print(f"   ❌ 失败: {e}")
        failed += 1
    
    # 测试 2: 带上限的 MOQ/RV 应用
    print("\n📝 测试 2: 带上限的 MOQ/RV 应用（关键修复）")
    try:
        result1 = apply_moq_rv(30, 50, 20, max_qty=40)
        assert result1 == 40, f"MOQ 调整超出 max_qty 时应限制，得到 {result1}"
        
        result2 = apply_moq_rv(105, 50, 20, max_qty=100)
        assert result2 == 100, f"RV 调整超出 max_qty 时应限制，得到 {result2}"
        print("   ✅ 通过 - MOQ/RV 调整被正确限制在订单量上限内")
        passed += 1
    except AssertionError as e:
        print(f"   ❌ 失败: {e}")
        failed += 1
    
    # 测试 3: 上限不影响正常情况
    print("\n📝 测试 3: 上限不影响正常情况")
    try:
        assert apply_moq_rv(30, 50, 20, max_qty=100) == 50
        assert apply_moq_rv(100, 50, 20, max_qty=200) == 100
        print("   ✅ 通过")
        passed += 1
    except AssertionError as e:
        print(f"   ❌ 失败: {e}")
        failed += 1
    
    # 测试 4: 自循环不应用 MOQ/RV
    print("\n📝 测试 4: 自循环不应用 MOQ/RV")
    try:
        assert apply_moq_rv(30, 50, 20, is_cross_node=False) == 30
        assert apply_moq_rv(100, 50, 20, is_cross_node=False, max_qty=50) == 50
        print("   ✅ 通过")
        passed += 1
    except AssertionError as e:
        print(f"   ❌ 失败: {e}")
        failed += 1
    
    # 测试 5: 零数量
    print("\n📝 测试 5: 零数量处理")
    try:
        assert apply_moq_rv(0, 50, 20) == 0
        assert apply_moq_rv(0, 50, 20, max_qty=100) == 0
        print("   ✅ 通过")
        passed += 1
    except AssertionError as e:
        print(f"   ❌ 失败: {e}")
        failed += 1
    
    # 测试 6: 分组 MOQ/RV 基本功能
    print("\n📝 测试 6: 分组 MOQ/RV 基本功能")
    try:
        demand_rows = [
            {'material': 'M1', 'demand_qty': 30, 'moq': 50, 'rv': 20, 
             'from_location': 'B', 'receiving': 'B'},
        ]
        result = apply_grouped_moq_rv(demand_rows, 'A')
        assert result[0] == 50, f"应用 MOQ 应得到 50，实际得到 {result[0]}"
        print("   ✅ 通过")
        passed += 1
    except AssertionError as e:
        print(f"   ❌ 失败: {e}")
        failed += 1
    
    # 测试 7: 分组 MOQ/RV 带订单量上限（关键修复）
    print("\n📝 测试 7: 分组 MOQ/RV 带订单量上限（关键修复）")
    try:
        demand_rows = [
            {'material': 'M1', 'demand_qty': 30, 'moq': 50, 'rv': 20,
             'from_location': 'B', 'receiving': 'B'},
        ]
        result = apply_grouped_moq_rv(demand_rows, 'A', shipment_qty_limit=40)
        assert result[0] == 40, f"应被订单量上限限制为 40，实际得到 {result[0]}"
        print("   ✅ 通过 - 分组 MOQ/RV 被正确限制在订单量上限内")
        passed += 1
    except AssertionError as e:
        print(f"   ❌ 失败: {e}")
        failed += 1
    
    # 测试 8: 验证约束原则
    print("\n📝 测试 8: 验证约束原则 - 出货量永远不超过订单量")
    try:
        shipment_qty = 100
        demanded_qty = 80
        moq = 120
        rv = 20
        
        deployed = apply_moq_rv(demanded_qty, moq, rv, max_qty=shipment_qty)
        
        assert deployed <= shipment_qty, f"部署量 {deployed} 超过订单量 {shipment_qty}"
        assert deployed == 100, f"应被限制为订单量 100，实际得到 {deployed}"
        print("   ✅ 通过 - 即使 MOQ 要求 120，也被限制在订单量 100 内")
        passed += 1
    except AssertionError as e:
        print(f"   ❌ 失败: {e}")
        failed += 1
    
    # 汇总
    print("\n" + "=" * 60)
    print(f"🏁 测试结果汇总: 通过 {passed}/{passed + failed}")
    print("=" * 60)
    
    if failed > 0:
        print(f"❌ 有 {failed} 个测试失败")
        return 1
    else:
        print("✅ 所有测试通过！")
        print("\n📋 修复验证:")
        print("   1. apply_moq_rv 函数已添加 max_qty 参数")
        print("   2. apply_grouped_moq_rv 函数已添加 shipment_qty_limit 参数")
        print("   3. MOQ/RV 调整不会使部署量超过订单量")
        return 0


if __name__ == '__main__':
    sys.exit(run_tests())
