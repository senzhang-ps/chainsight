#!/usr/bin/env python3
"""
日志系统测试脚本
验证日志记录功能是否正常工作
"""
import sys
from pathlib import Path
from logger_config import setup_logging, create_simple_file_logger, DualLogger


def test_dual_logger():
    """测试双输出日志器"""
    print("\n" + "="*60)
    print("测试1: DualLogger - 同时输出到terminal和文件")
    print("="*60)
    
    logger, _ = setup_logging("./test_logs", log_level="DEBUG", redirect_print=False)
    
    logger.debug("🔍 这是DEBUG级别日志（只在文件中）")
    logger.info("📝 这是INFO级别日志")
    logger.warning("⚠️  这是WARNING级别日志")
    logger.error("❌ 这是ERROR级别日志")
    logger.critical("🚨 这是CRITICAL级别日志")
    
    print("✅ 测试1完成 - 检查 ./test_logs/ 目录下的日志文件")


def test_print_redirect():
    """测试print重定向功能"""
    print("\n" + "="*60)
    print("测试2: Print重定向 - 所有print自动记录到文件")
    print("="*60)
    
    logger, redirector = setup_logging("./test_logs", log_level="INFO", redirect_print=True)
    
    print("🎉 这条print消息会被重定向到日志文件")
    print("🔢 支持数字: 12345")
    print("🌏 支持中文和emoji")
    print(f"📊 支持格式化字符串: {1 + 1} = 2")
    
    # 恢复原始输出
    if redirector:
        redirector.stop_redirect()
    
    print("✅ 测试2完成 - print已被记录到日志文件")


def test_simple_logger():
    """测试简单文件日志器"""
    print("\n" + "="*60)
    print("测试3: 简单文件日志器")
    print("="*60)
    
    logger = create_simple_file_logger("./test_logs", filename="simple_test.log")
    
    logger.info("这是使用简单日志器记录的信息")
    logger.warning("这是一个警告信息")
    
    print("✅ 测试3完成 - 检查 ./test_logs/simple_test.log")


def test_module_simulation():
    """模拟实际模块使用场景"""
    print("\n" + "="*60)
    print("测试4: 模拟实际仿真场景")
    print("="*60)
    
    logger, redirector = setup_logging("./test_logs", log_level="INFO", redirect_print=True)
    
    # 模拟仿真流程
    print("🚀 供应链仿真系统启动")
    print("📂 配置文件: test_config.xlsx")
    print("📁 输出目录: ./test_output")
    print("📅 仿真日期范围: 2024-01-01 到 2024-01-31")
    print("")
    
    # 模拟模块运行
    modules = ["Module1", "Module3", "Module4", "Module5", "Module6"]
    for i, module in enumerate(modules, 1):
        print(f"🔄 [{i}/{len(modules)}] 正在运行 {module}...")
        logger.debug(f"  └─ {module} 配置已加载")
        logger.debug(f"  └─ {module} 数据处理中...")
        print(f"  ✅ {module} 运行完成")
    
    print("")
    print("✅ 仿真成功完成")
    print(f"📊 总计处理: {len(modules)} 个模块")
    
    # 恢复原始输出
    if redirector:
        redirector.stop_redirect()
    
    print("✅ 测试4完成 - 查看日志文件了解完整流程")


def test_error_logging():
    """测试错误日志记录"""
    print("\n" + "="*60)
    print("测试5: 错误日志记录")
    print("="*60)
    
    logger, redirector = setup_logging("./test_logs", log_level="DEBUG", redirect_print=True)
    
    try:
        print("尝试执行可能出错的操作...")
        # 模拟错误
        result = 10 / 0
    except Exception as e:
        logger.error(f"操作失败: {str(e)}")
        logger.debug("错误详情", exc_info=True)  # 记录完整堆栈
        print("❌ 已捕获异常并记录到日志")
    
    # 恢复原始输出
    if redirector:
        redirector.stop_redirect()
    
    print("✅ 测试5完成 - 日志中包含完整错误信息")


def cleanup_test_logs():
    """清理测试日志"""
    import shutil
    test_dir = Path("./test_logs")
    if test_dir.exists():
        response = input("\n是否删除测试日志目录? (y/n): ")
        if response.lower() == 'y':
            shutil.rmtree(test_dir)
            print("✅ 测试日志已清理")
        else:
            print(f"📁 测试日志保留在: {test_dir.absolute()}")


def main():
    """运行所有测试"""
    print("\n" + "🧪 日志系统功能测试".center(60, "="))
    
    # 运行所有测试
    test_dual_logger()
    test_print_redirect()
    test_simple_logger()
    test_module_simulation()
    test_error_logging()
    
    # 总结
    print("\n" + "="*60)
    print("📋 测试总结")
    print("="*60)
    print("✅ 所有测试完成!")
    print(f"📁 日志文件位置: {Path('./test_logs').absolute()}")
    print("")
    print("请检查 ./test_logs/ 目录下的日志文件:")
    test_dir = Path("./test_logs")
    if test_dir.exists():
        log_files = list(test_dir.glob("*.txt")) + list(test_dir.glob("*.log"))
        for i, log_file in enumerate(log_files, 1):
            size_kb = log_file.stat().st_size / 1024
            print(f"  {i}. {log_file.name} ({size_kb:.2f} KB)")
    
    # 询问是否清理
    cleanup_test_logs()


if __name__ == "__main__":
    main()
