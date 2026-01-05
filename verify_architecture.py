#!/usr/bin/env python3
"""
验证新架构中所有主要模块是否能正确导入
该脚本可在项目根目录直接运行，会自动加载src下的所有模块。
"""

import sys
from pathlib import Path

# 确保可以导入src包
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

def test_imports():
    """测试所有关键模块的导入"""
    
    print("=" * 60)
    print("ChainSight 架构导入验证")
    print("=" * 60)
    print()
    
    tests = []
    
    # 测试Core层
    try:
        from src.core import main_integration, orchestrator, run
        tests.append(("Core Layer", True, "main_integration, orchestrator, run"))
    except Exception as e:
        tests.append(("Core Layer", False, str(e)))
    
    # 测试Modules层
    try:
        from src.modules import module1, module3, module4, module5, module6
        tests.append(("Modules Layer", True, "module1-6"))
    except Exception as e:
        tests.append(("Modules Layer", False, str(e)))
    
    # 测试Utils层
    try:
        from src.utils import (
            ValidationManager, 
            InventoryBalanceChecker,
            SimulationTimeManager,
            run_pre_simulation_validation,
            setup_logging
        )
        tests.append(("Utils Layer", True, "All validators & managers"))
    except Exception as e:
        tests.append(("Utils Layer", False, str(e)))
    
    # 测试Services层
    try:
        from src.services import SummaryReportGenerator, PerformanceProfiler
        tests.append(("Services Layer", True, "Report & Profiler"))
    except Exception as e:
        tests.append(("Services Layer", False, str(e)))
    
    # 测试主入口
    try:
        from src.core.run import main
        tests.append(("CLI Entry (run.py)", True, "main() function"))
    except Exception as e:
        tests.append(("CLI Entry (run.py)", False, str(e)))
    
    # 打印结果
    print("导入测试结果:")
    print("-" * 60)
    
    all_passed = True
    for layer, passed, details in tests:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:8} | {layer:20} | {details}")
        if not passed:
            all_passed = False
    
    print("-" * 60)
    print()
    
    if all_passed:
        print("✅ 所有导入测试通过！")
        print("✅ 架构已成功迁移到分层结构！")
        return 0
    else:
        print("❌ 某些导入失败！")
        print("请检查上述错误信息。")
        return 1

if __name__ == "__main__":
    sys.exit(test_imports())
